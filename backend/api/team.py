# backend/api/team.py
"""
NEXUS Backend — Team dashboard routes (/team, /team/compare)
Ported from services/api_gateway/main.py lines ~3981–4082.
"""
from __future__ import annotations

import asyncio
import logging
import uuid as _uuid_module

from fastapi import APIRouter, Depends, HTTPException, Query

from core.auth import get_current_user
from core.database import DEV_ORG_ID
from dependencies import get_db_pool

logger = logging.getLogger("nexus.backend.team")

router = APIRouter(tags=["team"])


@router.get("/")
async def get_team_dashboard(
    days: int = Query(default=30, ge=7, le=365),
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Team performance dashboard — all speakers with aggregate metrics for the last N days."""
    org_id = current_user.get("org_id", DEV_ORG_ID)

    rows = await pool.fetch(
        """
        WITH recent_appearances AS (
            SELECT sa.registry_id, sa.session_id, sa.speaker_label, sa.speaker_id
            FROM   speaker_appearances sa
            JOIN   sessions s ON s.id = sa.session_id
            WHERE  s.created_at > NOW() - make_interval(days => $2)
        ),
        speaker_metrics AS (
            SELECT
                ra.registry_id,
                COUNT(DISTINCT ra.session_id)                                                  AS session_count,
                AVG(CASE WHEN sig.signal_type = 'vocal_stress_score'      THEN sig.value END)  AS avg_stress,
                AVG(CASE WHEN sig.signal_type = 'conversation_engagement'  THEN sig.value END) AS avg_engagement,
                AVG(CASE WHEN sig.signal_type = 'rapport_indicator'        THEN sig.value END) AS avg_rapport,
                AVG(CASE WHEN sig.signal_type = 'filler_detection'         THEN sig.value END) AS avg_filler_rate,
                AVG(CASE WHEN sig.signal_type = 'power_language_score'     THEN sig.value END) AS avg_power,
                AVG(CASE WHEN sig.signal_type = 'attention_level'          THEN sig.value END) AS avg_attention
            FROM   recent_appearances ra
            JOIN   signals sig ON sig.session_id = ra.session_id AND sig.speaker_id = ra.speaker_id
            GROUP  BY ra.registry_id
        )
        SELECT sr.id, sr.display_name, sr.role, sr.company,
               sr.first_seen_at, sr.last_seen_at,
               sm.*
        FROM   speakers_registry sr
        JOIN   speaker_metrics sm ON sm.registry_id = sr.id
        WHERE  sr.org_id = $1
        ORDER  BY sm.session_count DESC
        """,
        _uuid_module.UUID(str(org_id)),
        days,
    )

    return {"team": [dict(r) for r in rows], "period_days": days}


@router.get("/compare")
async def compare_speakers(
    speaker_a: str = Query(...),
    speaker_b: str = Query(...),
    days: int = Query(default=90, ge=7, le=365),
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Compare two speakers' metrics side by side."""
    org_id   = current_user.get("org_id", DEV_ORG_ID)
    org_uuid = _uuid_module.UUID(str(org_id))

    for sid in (speaker_a, speaker_b):
        try:
            _uuid_module.UUID(sid)
        except ValueError:
            raise HTTPException(400, f"Invalid speaker id: {sid}")

    _metrics_sql = """
        SELECT
            COUNT(DISTINCT sa.session_id)                                                  AS session_count,
            AVG(CASE WHEN sig.signal_type = 'vocal_stress_score'      THEN sig.value END)  AS avg_stress,
            AVG(CASE WHEN sig.signal_type = 'conversation_engagement'  THEN sig.value END) AS avg_engagement,
            AVG(CASE WHEN sig.signal_type = 'rapport_indicator'        THEN sig.value END) AS avg_rapport,
            AVG(CASE WHEN sig.signal_type = 'filler_detection'         THEN sig.value END) AS avg_filler_rate,
            AVG(CASE WHEN sig.signal_type = 'power_language_score'     THEN sig.value END) AS avg_power,
            COUNT(CASE WHEN sig.signal_type = 'head_nod'               THEN 1 END)         AS total_nods,
            COUNT(CASE WHEN sig.signal_type = 'head_shake'             THEN 1 END)         AS total_shakes,
            AVG(CASE WHEN sig.signal_type = 'attention_level'          THEN sig.value END) AS avg_attention
        FROM   speaker_appearances sa
        JOIN   sessions s  ON s.id = sa.session_id
        JOIN   signals sig ON sig.session_id = s.id AND sig.speaker_id = sa.speaker_id
        WHERE  sa.registry_id = $1
          AND  s.created_at > NOW() - make_interval(days => $2)
    """

    _info_sql = "SELECT display_name, role FROM speakers_registry WHERE id = $1 AND org_id = $2"

    metrics_a_row, metrics_b_row, info_a, info_b = await asyncio.gather(
        pool.fetchrow(_metrics_sql, _uuid_module.UUID(speaker_a), days),
        pool.fetchrow(_metrics_sql, _uuid_module.UUID(speaker_b), days),
        pool.fetchrow(_info_sql, _uuid_module.UUID(speaker_a), org_uuid),
        pool.fetchrow(_info_sql, _uuid_module.UUID(speaker_b), org_uuid),
    )

    return {
        "speaker_a": {
            "info":    dict(info_a)        if info_a        else {},
            "metrics": dict(metrics_a_row) if metrics_a_row else {},
        },
        "speaker_b": {
            "info":    dict(info_b)        if info_b        else {},
            "metrics": dict(metrics_b_row) if metrics_b_row else {},
        },
        "period_days": days,
    }
