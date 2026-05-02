# services/api_gateway/speaker_registry.py
"""
NEXUS Speaker Registry — cross-session speaker identity matching.

Two public functions called by _run_pipeline in main.py:

  match_or_create_speakers  — called immediately after voice signals are
      persisted; matches each speaker's voice embedding against the registry
      (pgvector cosine similarity) and creates a new entry when no match
      exceeds the threshold.

  update_appearance_stats   — called after ALL agent signals are persisted;
      back-fills speaker_appearances.avg_stress / avg_engagement / filler_rate
      from the signals table.
"""
import logging
import uuid as _uuid
from typing import Optional

logger = logging.getLogger("nexus.gateway.speaker_registry")

# Cosine-similarity threshold above which a voice embedding is considered
# a match to an existing registry entry.
DEFAULT_SIMILARITY_THRESHOLD = 0.75


async def match_or_create_speakers(
    pool,
    session_id: str,
    speaker_embeddings: dict[str, list[float]],
    voice_speakers: list[dict],
    speaker_map: dict[str, str],
    org_id: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, dict]:
    """
    For each speaker in the session, try to match their voice embedding
    against existing registry entries using pgvector cosine similarity.
    Create a new registry entry for any unmatched speaker.

    Also inserts a row into speaker_appearances linking the registry entry
    to this session's speaker record (back-filled with stats later by
    update_appearance_stats).

    Args:
        pool:                asyncpg connection pool.
        session_id:          UUID of the current session (string).
        speaker_embeddings:  {speaker_label: [float, ...]} L2-normalised
                             vectors from the voice agent.
        voice_speakers:      List of speaker dicts from voice agent response
                             — each has at least "speaker_id".
        speaker_map:         {speaker_label: speaker_db_uuid} built by
                             upsert_speakers().
        org_id:              Organisation UUID (string) — scopes the search.
        similarity_threshold: Minimum cosine similarity (0–1) to count as a
                             match. Default 0.75.

    Returns:
        {speaker_label: {"registry_id": str, "display_name": str,
                         "match_method": str, "confidence": float}}
    """
    result: dict[str, dict] = {}

    for speaker_label, embedding in speaker_embeddings.items():
        # pgvector expects a literal vector string: '[0.1,0.2,...]'
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        # ── Attempt cosine-similarity match against registry ──────────────
        match = await pool.fetchrow(
            """
            SELECT id, display_name, role,
                   1 - (voice_embedding <=> $1::vector) AS similarity
            FROM   speakers_registry
            WHERE  org_id = $2
              AND  voice_embedding IS NOT NULL
            ORDER  BY voice_embedding <=> $1::vector
            LIMIT  1
            """,
            embedding_str,
            _coerce_uuid(org_id),
        )

        if match and float(match["similarity"]) >= similarity_threshold:
            # ── Matched existing speaker ───────────────────────────────────
            registry_id  = str(match["id"])
            display_name = match["display_name"]
            confidence   = float(match["similarity"])
            match_method = "voice_embedding"

            await pool.execute(
                """
                UPDATE speakers_registry
                SET    session_count      = session_count + 1,
                       voice_sample_count = voice_sample_count + 1,
                       last_seen_at       = NOW(),
                       updated_at         = NOW()
                WHERE  id = $1
                """,
                match["id"],
            )
            logger.info(
                "Registry match: %s → %s (similarity=%.3f)",
                speaker_label, display_name, confidence,
            )
        else:
            # ── New speaker — create registry entry ────────────────────────
            # voice_speakers uses "speaker_id" as the label key
            speaker_info = next(
                (s for s in voice_speakers if s.get("speaker_id") == speaker_label),
                {},
            )
            display_name = speaker_info.get("name") or speaker_label

            row = await pool.fetchrow(
                """
                INSERT INTO speakers_registry
                    (org_id, display_name, role,
                     voice_embedding, voice_sample_count, session_count)
                VALUES ($1, $2, $3, $4::vector, 1, 1)
                RETURNING id
                """,
                _coerce_uuid(org_id),
                display_name,
                speaker_info.get("role", ""),
                embedding_str,
            )
            registry_id  = str(row["id"])
            confidence   = 1.0
            match_method = "new_registration"
            logger.info("Registry new: %s registered as %r", speaker_label, display_name)

        # ── Insert / update appearance record ─────────────────────────────
        # Per-session stats (avg_stress etc.) are left at defaults here and
        # filled in by update_appearance_stats() once all signals are saved.
        speaker_db_id = speaker_map.get(speaker_label)

        await pool.execute(
            """
            INSERT INTO speaker_appearances
                (registry_id, session_id, speaker_id,
                 speaker_label, match_method, match_confidence)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (registry_id, session_id, speaker_label) DO UPDATE
                SET match_method    = EXCLUDED.match_method,
                    match_confidence = EXCLUDED.match_confidence
            """,
            _coerce_uuid(registry_id),
            _coerce_uuid(session_id),
            _coerce_uuid(speaker_db_id) if speaker_db_id else None,
            speaker_label,
            match_method,
            confidence,
        )

        result[speaker_label] = {
            "registry_id":   registry_id,
            "display_name":  display_name,
            "match_method":  match_method,
            "confidence":    confidence,
        }

    return result


async def update_appearance_stats(pool, session_id: str) -> None:
    """
    Back-fill speaker_appearances with aggregate stats computed from the
    signals table for this session.

    Call this after ALL agent signals (voice, language, conversation,
    video, fusion) have been persisted so the aggregates are complete.

    Stats updated:
      avg_stress       — mean vocal_stress_score value
      avg_engagement   — mean conversation_engagement value
      filler_rate      — mean filler_detection value
    """
    await pool.execute(
        """
        UPDATE speaker_appearances sa
        SET    avg_stress     = sub.avg_stress,
               avg_engagement = sub.avg_engagement,
               filler_rate    = sub.filler_rate
        FROM (
            SELECT
                sp.speaker_label,
                AVG(CASE WHEN s.signal_type = 'vocal_stress_score'      THEN s.value END) AS avg_stress,
                AVG(CASE WHEN s.signal_type = 'conversation_engagement'  THEN s.value END) AS avg_engagement,
                AVG(CASE WHEN s.signal_type = 'filler_detection'         THEN s.value END) AS filler_rate
            FROM   signals  s
            JOIN   speakers sp ON sp.id = s.speaker_id
            WHERE  s.session_id = $1
            GROUP  BY sp.speaker_label
        ) sub
        WHERE  sa.session_id   = $1
          AND  sa.speaker_label = sub.speaker_label
        """,
        _coerce_uuid(session_id),
    )
    logger.debug("Appearance stats updated for session %s", session_id)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _coerce_uuid(value: Optional[str]):
    """Convert a string UUID to asyncpg-compatible uuid.UUID, or return None."""
    if value is None:
        return None
    if isinstance(value, _uuid.UUID):
        return value
    try:
        return _uuid.UUID(str(value))
    except (ValueError, AttributeError):
        return None
