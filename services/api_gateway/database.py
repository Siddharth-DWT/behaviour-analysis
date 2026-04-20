# services/api_gateway/database.py
"""
NEXUS API Gateway - Database Module
Async PostgreSQL operations for sessions, signals, alerts, and reports.

Uses asyncpg for high-performance async queries against the schema
defined in infrastructure/postgres/init/01-schema.sql.
"""
import os
import json
import uuid as _uuid
import logging
from typing import Optional
from datetime import datetime, timezone

import asyncpg

logger = logging.getLogger("nexus.gateway.db")

# Default dev org — matches seed data in 01-schema.sql
DEV_ORG_ID = "00000000-0000-0000-0000-000000000001"

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    logger.error("DATABASE_URL not set! Database operations will fail.")

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
        )
        logger.info("PostgreSQL connection pool created.")
    return _pool


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed.")


# ─────────────────────────────────────────────────────────
# SESSIONS
# ─────────────────────────────────────────────────────────

async def _ensure_upload_config_column(pool: asyncpg.Pool):
    """Add upload_config column to sessions if it doesn't exist yet (idempotent)."""
    try:
        await pool.execute(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS upload_config JSONB DEFAULT '{}'"
        )
    except Exception:
        pass  # Column already exists or DB doesn't support IF NOT EXISTS — safe to ignore


async def create_session(
    title: str,
    session_type: str = "recording",
    meeting_type: str = "sales_call",
    media_url: Optional[str] = None,
    org_id: str = DEV_ORG_ID,
    user_id: Optional[str] = None,
    upload_config: Optional[dict] = None,
) -> dict:
    """Create a new session record. Returns the created session."""
    pool = await get_pool()
    await _ensure_upload_config_column(pool)
    row = await pool.fetchrow(
        """
        INSERT INTO sessions (org_id, title, session_type, meeting_type, media_url, status, user_id, upload_config)
        VALUES ($1, $2, $3, $4, $5, 'created', $6, $7::jsonb)
        RETURNING id, org_id, title, session_type, meeting_type, status,
                  media_url, duration_ms, speaker_count, user_id,
                  created_at, started_at, completed_at, upload_config
        """,
        _uuid.UUID(org_id) if isinstance(org_id, str) else org_id,
        title, session_type, meeting_type, media_url,
        _uuid.UUID(user_id) if isinstance(user_id, str) and user_id else user_id,
        json.dumps(upload_config or {}),
    )
    return _row_to_dict(row)


async def get_session(session_id: str, org_id: str = DEV_ORG_ID) -> Optional[dict]:
    """Get a single session by ID, scoped to org."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT id, org_id, title, session_type, meeting_type, status,
               media_url, duration_ms, speaker_count, user_id,
               created_at, started_at, completed_at
        FROM sessions
        WHERE id = $1 AND org_id = $2
        """,
        session_id, org_id,
    )
    if not row:
        return None
    return _row_to_dict(row)


async def list_sessions(
    org_id: str = DEV_ORG_ID,
    limit: int = 25,
    offset: int = 0,
    status: Optional[str] = None,
    meeting_type: Optional[str] = None,
    user_id: Optional[str] = None,
    session_type: Optional[str] = None,
) -> tuple[list[dict], int]:
    """List sessions with pagination. Returns (sessions, total_count).
    If user_id is provided, only returns sessions owned by that user.
    By default excludes lightweight sessions (transcript/diarize only).
    Pass session_type='lightweight' to get only lightweight sessions."""
    pool = await get_pool()

    # Build WHERE clause
    conditions = ["org_id = $1"]
    params: list = [org_id]
    idx = 2

    if user_id:
        conditions.append(f"user_id = ${idx}")
        params.append(_uuid.UUID(user_id) if isinstance(user_id, str) else user_id)
        idx += 1
    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1
    if meeting_type:
        conditions.append(f"meeting_type = ${idx}")
        params.append(meeting_type)
        idx += 1
    if session_type == "lightweight":
        conditions.append(f"session_type = ${idx}")
        params.append("lightweight")
        idx += 1
    else:
        # Default: exclude lightweight sessions from the main sessions list
        conditions.append(f"session_type != ${idx}")
        params.append("lightweight")
        idx += 1

    where = " AND ".join(conditions)

    # Count
    count_row = await pool.fetchrow(
        f"SELECT COUNT(*) as total FROM sessions WHERE {where}",
        *params,
    )
    total = count_row["total"]

    # Fetch page
    params.extend([limit, offset])
    rows = await pool.fetch(
        f"""
        SELECT id, org_id, title, session_type, meeting_type, status,
               media_url, duration_ms, speaker_count, user_id,
               created_at, started_at, completed_at
        FROM sessions
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )
    return [_row_to_dict(r) for r in rows], total


async def update_session_status(
    session_id: str,
    status: str,
    org_id: str = DEV_ORG_ID,
    duration_ms: Optional[int] = None,
    speaker_count: Optional[int] = None,
):
    """Update session status and optional fields."""
    pool = await get_pool()

    sets = ["status = $3"]
    params: list = [session_id, org_id, status]
    idx = 4

    if status == "processing":
        sets.append(f"started_at = ${idx}")
        params.append(datetime.now(timezone.utc))
        idx += 1
    elif status in ("completed", "failed"):
        sets.append(f"completed_at = ${idx}")
        params.append(datetime.now(timezone.utc))
        idx += 1

    if duration_ms is not None:
        sets.append(f"duration_ms = ${idx}")
        params.append(duration_ms)
        idx += 1

    if speaker_count is not None:
        sets.append(f"speaker_count = ${idx}")
        params.append(speaker_count)
        idx += 1

    set_clause = ", ".join(sets)
    await pool.execute(
        f"""
        UPDATE sessions SET {set_clause}
        WHERE id = $1 AND org_id = $2
        """,
        *params,
    )


# ─────────────────────────────────────────────────────────
# SPEAKERS
# ─────────────────────────────────────────────────────────

async def upsert_speakers(
    session_id: str,
    speakers: list[dict],
) -> dict[str, str]:
    """
    Create or update speaker records for a session.
    Returns mapping of {speaker_label: speaker_uuid}.
    """
    pool = await get_pool()
    label_to_id = {}

    for speaker in speakers:
        label = speaker.get("speaker_id") or speaker.get("speaker_label", "Unknown")
        baseline_data = speaker.get("baseline")
        talk_time_ms = int(speaker.get("talk_time_ms") or 0)
        talk_time_pct = float(speaker.get("talk_time_pct") or 0.0)
        total_words = int(speaker.get("total_words") or 0)
        cal_conf = float(speaker.get("calibration_confidence") or 0.0)

        # Check if already exists
        existing = await pool.fetchrow(
            """
            SELECT id FROM speakers
            WHERE session_id = $1 AND speaker_label = $2
            """,
            session_id, label,
        )

        if existing:
            speaker_uuid = str(existing["id"])
            await pool.execute(
                """
                UPDATE speakers
                SET baseline_data          = COALESCE($1, baseline_data),
                    total_talk_time_ms     = $2,
                    talk_time_pct          = $3,
                    total_words            = $4,
                    calibration_confidence = $5
                WHERE id = $6
                """,
                json.dumps(baseline_data) if baseline_data else None,
                talk_time_ms, talk_time_pct, total_words, cal_conf,
                existing["id"],
            )
        else:
            row = await pool.fetchrow(
                """
                INSERT INTO speakers (
                    session_id, speaker_label, baseline_data,
                    total_talk_time_ms, talk_time_pct, total_words, calibration_confidence
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                session_id, label,
                json.dumps(baseline_data) if baseline_data else None,
                talk_time_ms, talk_time_pct, total_words, cal_conf,
            )
            speaker_uuid = str(row["id"])

        label_to_id[label] = speaker_uuid

    return label_to_id


# ─────────────────────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────────────────────

async def insert_signals(session_id: str, signals: list[dict], speaker_map: dict[str, str] = None):
    """Bulk insert signals into the signals table."""
    if not signals:
        return 0

    pool = await get_pool()
    speaker_map = speaker_map or {}

    unmapped_speakers = set()
    records = []
    for s in signals:
        speaker_label = s.get("speaker_id", "unknown")
        speaker_uuid = speaker_map.get(speaker_label)
        # Conversation Agent emits per-pair signals (e.g. "Speaker_0__Speaker_1"
        # for rapport/latency) and session-level signals (speaker_id="session"
        # for turn-taking, balance, conflict). These intentionally don't map to
        # a single speaker — store with speaker_id=NULL and don't warn.
        if (
            speaker_uuid is None
            and speaker_label not in ("unknown", "all", "multiple", "session", "")
            and "__" not in speaker_label
        ):
            unmapped_speakers.add(speaker_label)

        metadata = s.get("metadata")
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        elif metadata is None:
            metadata = "{}"

        records.append((
            session_id,
            speaker_uuid,
            s.get("agent", "unknown"),
            s.get("signal_type", ""),
            _safe_float(s.get("value")),
            s.get("value_text", ""),
            _safe_float(s.get("confidence", 0.5)),
            _safe_int(s.get("window_start_ms", 0)),
            _safe_int(s.get("window_end_ms", 0)),
            metadata,
        ))

    if unmapped_speakers:
        logger.warning(f"Signals reference {len(unmapped_speakers)} unmapped speakers: {unmapped_speakers}")

    await pool.executemany(
        """
        INSERT INTO signals
            (session_id, speaker_id, agent, signal_type, value, value_text,
             confidence, window_start_ms, window_end_ms, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
        """,
        records,
    )
    return len(records)


async def get_signals(
    session_id: str,
    org_id: str = DEV_ORG_ID,
    agent: Optional[str] = None,
    signal_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Get signals for a session, with optional filters."""
    pool = await get_pool()

    conditions = ["s.session_id = $1", "sess.org_id = $2"]
    params: list = [session_id, org_id]
    idx = 3

    if agent:
        conditions.append(f"s.agent = ${idx}")
        params.append(agent)
        idx += 1
    if signal_type:
        conditions.append(f"s.signal_type = ${idx}")
        params.append(signal_type)
        idx += 1

    where = " AND ".join(conditions)
    params.extend([limit, offset])

    rows = await pool.fetch(
        f"""
        SELECT s.id, s.session_id, s.speaker_id, s.agent, s.signal_type,
               s.value, s.value_text, s.confidence,
               s.window_start_ms, s.window_end_ms, s.metadata, s.created_at,
               sp.speaker_label
        FROM signals s
        JOIN sessions sess ON sess.id = s.session_id
        LEFT JOIN speakers sp ON sp.id = s.speaker_id
        WHERE {where}
        ORDER BY s.window_start_ms ASC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )
    return [_row_to_dict(r) for r in rows]


# ─────────────────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────────────────

async def insert_alerts(session_id: str, alerts: list[dict], speaker_map: dict[str, str] = None):
    """Bulk insert alerts."""
    if not alerts:
        return 0

    pool = await get_pool()
    speaker_map = speaker_map or {}

    records = []
    for a in alerts:
        speaker_label = a.get("speaker_id", "unknown")
        speaker_uuid = speaker_map.get(speaker_label)

        evidence = a.get("evidence")
        if isinstance(evidence, dict):
            evidence = json.dumps(evidence)
        elif evidence is None:
            evidence = "{}"

        records.append((
            session_id,
            speaker_uuid,
            a.get("alert_type", ""),
            a.get("severity", "yellow"),
            a.get("title", ""),
            a.get("description", ""),
            evidence,
            _safe_int(a.get("timestamp_ms", 0)) or int(datetime.now(timezone.utc).timestamp() * 1000),
        ))

    await pool.executemany(
        """
        INSERT INTO alerts
            (session_id, speaker_id, alert_type, severity, title,
             description, evidence, timestamp_ms)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
        """,
        records,
    )
    return len(records)


async def get_alerts(
    session_id: str,
    org_id: str = DEV_ORG_ID,
) -> list[dict]:
    """Get alerts for a session."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT a.id, a.session_id, a.speaker_id, a.alert_type, a.severity,
               a.title, a.description, a.evidence, a.timestamp_ms,
               a.acknowledged, a.created_at,
               sp.speaker_label
        FROM alerts a
        JOIN sessions sess ON sess.id = a.session_id
        LEFT JOIN speakers sp ON sp.id = a.speaker_id
        WHERE a.session_id = $1 AND sess.org_id = $2
        ORDER BY a.timestamp_ms ASC
        """,
        session_id, org_id,
    )
    return [_row_to_dict(r) for r in rows]


# ─────────────────────────────────────────────────────────
# REPORTS
# ─────────────────────────────────────────────────────────

async def save_report(
    session_id: str,
    content: dict,
    narrative: Optional[str] = None,
    report_type: str = "post_session",
) -> dict:
    """Save a session report."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO session_reports (session_id, report_type, content, narrative)
        VALUES ($1, $2, $3::jsonb, $4)
        RETURNING id, session_id, report_type, content, narrative, generated_at
        """,
        session_id, report_type, json.dumps(content), narrative,
    )
    return _row_to_dict(row)


async def get_report(
    session_id: str,
    org_id: str = DEV_ORG_ID,
) -> Optional[dict]:
    """Get the latest report for a session."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT r.id, r.session_id, r.report_type, r.content, r.narrative,
               r.generated_at
        FROM session_reports r
        JOIN sessions sess ON sess.id = r.session_id
        WHERE r.session_id = $1 AND sess.org_id = $2
        ORDER BY r.generated_at DESC
        LIMIT 1
        """,
        session_id, org_id,
    )
    if not row:
        return None
    return _row_to_dict(row)


# ─────────────────────────────────────────────────────────
# TRANSCRIPT SEGMENTS
# ─────────────────────────────────────────────────────────

async def insert_transcript_segments(
    session_id: str,
    segments: list[dict],
    speaker_map: dict[str, str] = None,
):
    """Bulk insert transcript segments."""
    if not segments:
        return 0

    pool = await get_pool()
    speaker_map = speaker_map or {}

    records = []
    for i, seg in enumerate(segments):
        speaker_label = seg.get("speaker", "unknown")
        speaker_uuid = speaker_map.get(speaker_label)
        text = seg.get("text", "")

        records.append((
            session_id,
            speaker_uuid,
            i,
            _safe_int(seg.get("start_ms", 0)),
            _safe_int(seg.get("end_ms", 0)),
            text,
            len(text.split()),
        ))

    await pool.executemany(
        """
        INSERT INTO transcript_segments
            (session_id, speaker_id, segment_index, start_ms, end_ms, text, word_count)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        records,
    )
    return len(records)


async def get_transcript(
    session_id: str,
    org_id: str = DEV_ORG_ID,
) -> list[dict]:
    """Get transcript segments for a session."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT t.id, t.session_id, t.speaker_id, t.segment_index,
               t.start_ms, t.end_ms, t.text, t.word_count,
               t.sentiment, t.sentiment_score,
               sp.speaker_label
        FROM transcript_segments t
        JOIN sessions sess ON sess.id = t.session_id
        LEFT JOIN speakers sp ON sp.id = t.speaker_id
        WHERE t.session_id = $1 AND sess.org_id = $2
        ORDER BY t.segment_index ASC
        """,
        session_id, org_id,
    )
    return [_row_to_dict(r) for r in rows]


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _row_to_dict(row: asyncpg.Record) -> dict:
    """Convert an asyncpg Record to a dict with JSON-serializable values."""
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, _uuid.UUID):
            d[k] = str(v)
        elif isinstance(v, memoryview):
            d[k] = bytes(v).decode("utf-8", errors="replace")
    return d


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_int(v) -> int:
    if v is None or v == "":
        return 0
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0
