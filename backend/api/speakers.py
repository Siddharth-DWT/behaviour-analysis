# backend/api/speakers.py
"""
NEXUS Backend — Speaker registry routes (/speakers/*)
Ported from services/api_gateway/main.py lines ~3687–4311.
"""
from __future__ import annotations

import logging
import re
import uuid as _uuid_module
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from starlette.requests import Request as StarletteRequest

from core.auth import get_current_user, verify_access_token
from core.database import DEV_ORG_ID
from dependencies import get_db_pool

logger = logging.getLogger("nexus.backend.speakers")

router = APIRouter(tags=["speakers"])

_SPEAKER_SORT_COLS = frozenset({"last_seen_at", "first_seen_at", "session_count", "display_name"})
_GENERIC_SPEAKER_LABEL_RE = re.compile(r'^(Speaker|Face)_\d+$')


# ── GET /speakers ──────────────────────────────────────────────────────────────

@router.get("/")
async def list_speakers(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="last_seen_at"),
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """List all registered speakers for this org with aggregate stats."""
    if sort_by not in _SPEAKER_SORT_COLS:
        sort_by = "last_seen_at"

    org_id = current_user.get("org_id", DEV_ORG_ID)

    rows = await pool.fetch(
        f"""
        SELECT sr.id, sr.display_name, sr.role, sr.company,
               sr.session_count, sr.first_seen_at, sr.last_seen_at,
               COALESCE(AVG(sa.avg_stress),      0) AS avg_stress,
               COALESCE(AVG(sa.avg_engagement),  0) AS avg_engagement,
               COALESCE(AVG(sa.filler_rate),     0) AS avg_filler_rate
        FROM   speakers_registry sr
        LEFT JOIN speaker_appearances sa ON sa.registry_id = sr.id
        WHERE  sr.org_id = $1
        GROUP  BY sr.id
        ORDER  BY sr.{sort_by} DESC
        LIMIT  $2 OFFSET $3
        """,
        _uuid_module.UUID(str(org_id)),
        limit,
        offset,
    )

    total = await pool.fetchval(
        "SELECT COUNT(*) FROM speakers_registry WHERE org_id = $1",
        _uuid_module.UUID(str(org_id)),
    )

    return {
        "speakers": [
            {
                **dict(r),
                "display_name": r["display_name"]
                if (r["display_name"] and not _GENERIC_SPEAKER_LABEL_RE.match(r["display_name"]))
                else "",
            }
            for r in rows
        ],
        "total":  total,
        "limit":  limit,
        "offset": offset,
    }


# ── GET /speakers/{id} ─────────────────────────────────────────────────────────

@router.get("/{registry_id}")
async def get_speaker_profile(
    registry_id: str,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Get speaker profile with cross-session performance trends."""
    org_id = current_user.get("org_id", DEV_ORG_ID)

    try:
        reg_uuid = _uuid_module.UUID(registry_id)
    except ValueError:
        raise HTTPException(400, "Invalid registry_id")

    speaker = await pool.fetchrow(
        "SELECT * FROM speakers_registry WHERE id = $1 AND org_id = $2",
        reg_uuid, _uuid_module.UUID(str(org_id)),
    )
    if not speaker:
        raise HTTPException(404, "Speaker not found")

    appearances = await pool.fetch(
        """
        SELECT sa.*, s.title, s.meeting_type, s.created_at AS session_date,
               s.duration_ms, s.status
        FROM   speaker_appearances sa
        JOIN   sessions s ON s.id = sa.session_id
        WHERE  sa.registry_id = $1
        ORDER  BY s.created_at DESC
        """,
        reg_uuid,
    )

    trends = await pool.fetch(
        """
        SELECT
            s.id           AS session_id,
            s.created_at   AS session_date,
            s.title,
            s.meeting_type,
            AVG(CASE WHEN sig.signal_type = 'vocal_stress_score'      THEN sig.value END) AS avg_stress,
            AVG(CASE WHEN sig.signal_type = 'conversation_engagement'  THEN sig.value END) AS avg_engagement,
            AVG(CASE WHEN sig.signal_type = 'rapport_indicator'        THEN sig.value END) AS avg_rapport,
            AVG(CASE WHEN sig.signal_type = 'filler_detection'         THEN sig.value END) AS filler_rate,
            AVG(CASE WHEN sig.signal_type = 'power_language_score'     THEN sig.value END) AS avg_power,
            COUNT(CASE WHEN sig.signal_type = 'facial_stress'          THEN 1 END)         AS facial_stress_count,
            COUNT(CASE WHEN sig.signal_type = 'head_nod'               THEN 1 END)         AS nod_count,
            COUNT(CASE WHEN sig.signal_type = 'head_shake'             THEN 1 END)         AS shake_count,
            AVG(CASE WHEN sig.signal_type = 'attention_level'          THEN sig.value END) AS avg_attention
        FROM   speaker_appearances sa
        JOIN   sessions s  ON s.id  = sa.session_id
        JOIN   speakers sp ON sp.id = sa.speaker_id
        LEFT JOIN signals sig ON sig.session_id = s.id AND sig.speaker_id = sp.id
        WHERE  sa.registry_id = $1
        GROUP  BY s.id, s.created_at, s.title, s.meeting_type
        ORDER  BY s.created_at ASC
        """,
        reg_uuid,
    )

    return {
        "speaker":       dict(speaker),
        "appearances":   [dict(a) for a in appearances],
        "trends":        [dict(t) for t in trends],
        "session_count": len(appearances),
    }


# ── PUT /speakers/{id} ─────────────────────────────────────────────────────────

@router.put("/{registry_id}")
async def update_speaker(
    registry_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Update speaker display_name, role, company, email, or notes."""
    org_id = current_user.get("org_id", DEV_ORG_ID)

    try:
        reg_uuid = _uuid_module.UUID(registry_id)
    except ValueError:
        raise HTTPException(400, "Invalid registry_id")

    sets:   list[str] = []
    params: list[Any] = []
    idx = 1

    for field in ("display_name", "role", "company", "email", "notes"):
        if field in body and body[field] is not None:
            sets.append(f"{field} = ${idx}")
            params.append(body[field])
            idx += 1

    if not sets:
        raise HTTPException(422, "No updatable fields provided")

    sets.append(f"updated_at = ${idx}")
    params.append(datetime.now(timezone.utc))
    idx += 1
    params.append(reg_uuid)
    params.append(_uuid_module.UUID(str(org_id)))

    await pool.execute(
        f"UPDATE speakers_registry SET {', '.join(sets)} WHERE id = ${idx} AND org_id = ${idx + 1}",
        *params,
    )
    return {"success": True}


# ── POST /speakers/{id}/merge ──────────────────────────────────────────────────

@router.post("/{registry_id}/merge")
async def merge_speakers(
    registry_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Merge another speaker into this one. All appearances from source are re-pointed to target."""
    org_id   = current_user.get("org_id", DEV_ORG_ID)
    org_uuid = _uuid_module.UUID(str(org_id))

    try:
        target_uuid = _uuid_module.UUID(registry_id)
    except ValueError:
        raise HTTPException(400, "Invalid registry_id")

    source_id = body.get("merge_from_id")
    if not source_id:
        raise HTTPException(400, "merge_from_id required")

    try:
        source_uuid = _uuid_module.UUID(str(source_id))
    except ValueError:
        raise HTTPException(400, "Invalid merge_from_id")

    target_check = await pool.fetchval(
        "SELECT id FROM speakers_registry WHERE id = $1 AND org_id = $2", target_uuid, org_uuid
    )
    if not target_check:
        raise HTTPException(404, "Target speaker not found")

    source_check = await pool.fetchval(
        "SELECT id FROM speakers_registry WHERE id = $1 AND org_id = $2", source_uuid, org_uuid
    )
    if not source_check:
        raise HTTPException(404, "Source speaker not found")

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE speaker_appearances SET registry_id = $1 WHERE registry_id = $2",
                target_uuid, source_uuid,
            )
            source_row = await conn.fetchrow(
                "SELECT session_count, total_talk_time_ms FROM speakers_registry WHERE id = $1",
                source_uuid,
            )
            if source_row:
                await conn.execute(
                    """
                    UPDATE speakers_registry
                    SET    session_count      = session_count      + $2,
                           total_talk_time_ms = total_talk_time_ms + $3,
                           updated_at         = NOW()
                    WHERE  id = $1
                    """,
                    target_uuid,
                    source_row["session_count"],
                    source_row["total_talk_time_ms"],
                )
            await conn.execute("DELETE FROM speakers_registry WHERE id = $1", source_uuid)

    return {"success": True, "merged_from": str(source_uuid), "merged_into": str(target_uuid)}


# ── POST /speakers/search-by-face ─────────────────────────────────────────────

@router.post("/search-by-face")
async def search_speakers_by_face(
    face_image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """
    Upload a face photo → find all sessions where this person appeared.
    Phase 3: delegates to VideoAgentService.embed_face(); Phase 1 stub returns 501.
    """
    raise HTTPException(501, "Face search via VideoAgentService: Phase 3 pending")


# ── GET /speakers/{id}/thumbnail ──────────────────────────────────────────────

@router.get("/{registry_id}/thumbnail")
async def get_speaker_thumbnail(
    request: StarletteRequest,
    registry_id: str,
    token: Optional[str] = Query(default=None),
    pool=Depends(get_db_pool),
):
    """Return the primary face thumbnail for a speaker as JPEG."""
    auth_token = token
    if not auth_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]
    if not auth_token:
        raise HTTPException(401, "Unauthorized")

    if not verify_access_token(auth_token):
        raise HTTPException(401, "Invalid or expired token")

    try:
        reg_uuid = _uuid_module.UUID(registry_id)
    except ValueError:
        raise HTTPException(400, "Invalid registry_id")

    row = await pool.fetchrow(
        """
        SELECT thumbnail FROM face_thumbnails
        WHERE registry_id = $1
        ORDER BY is_primary DESC, quality_score DESC
        LIMIT 1
        """,
        reg_uuid,
    )
    if not row:
        raise HTTPException(404, "No thumbnail available")

    return Response(content=row["thumbnail"], media_type="image/jpeg")
