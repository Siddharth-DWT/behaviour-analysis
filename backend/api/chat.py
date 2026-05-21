# backend/api/chat.py
"""
NEXUS Backend — Global chat route (/chat/global)
Ported from services/api_gateway/main.py lines ~4085–4186.

Session-scoped chat (/sessions/{id}/chat) lives in api/sessions.py
because it uses the /sessions prefix (no prefix on the sessions router).
"""
from __future__ import annotations

import logging
import re
import uuid as _uuid_module

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.auth import get_current_user
from core.database import DEV_ORG_ID
from dependencies import get_db_pool

logger = logging.getLogger("nexus.backend.chat")

router = APIRouter(tags=["chat"])

_GENERIC_SPEAKER_LABEL_RE = re.compile(r'^(Speaker|Face)_\d+$')


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


@router.post("/global")
async def global_chat(
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """
    Ask questions across ALL sessions — team comparisons, trends,
    speaker performance, and coaching insights.
    """
    org_id = current_user.get("org_id") or DEV_ORG_ID

    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    speakers = await pool.fetch(
        """
        SELECT sr.display_name, sr.role, sr.session_count, sr.last_seen_at,
               AVG(sa.avg_stress)      AS avg_stress,
               AVG(sa.avg_engagement)  AS avg_engagement,
               AVG(sa.filler_rate)     AS avg_filler_rate
        FROM   speakers_registry sr
        LEFT JOIN speaker_appearances sa ON sa.registry_id = sr.id
        WHERE  sr.org_id = $1
        GROUP  BY sr.id
        ORDER  BY sr.session_count DESC
        LIMIT  20
        """,
        _uuid_module.UUID(str(org_id)),
    )

    recent_sessions = await pool.fetch(
        """
        SELECT s.id, s.title, s.meeting_type, s.created_at, s.duration_ms,
               array_agg(DISTINCT sp.speaker_label) AS speakers
        FROM   sessions s
        LEFT JOIN speakers sp ON sp.session_id = s.id
        WHERE  s.status = 'completed'
          AND  s.org_id = $1
        GROUP  BY s.id
        ORDER  BY s.created_at DESC
        LIMIT  15
        """,
        _uuid_module.UUID(str(org_id)),
    )

    speaker_context = "Registered speakers:\n"
    for s in speakers:
        name = (
            s["display_name"]
            if (s["display_name"] and not _GENERIC_SPEAKER_LABEL_RE.match(s["display_name"]))
            else "unnamed speaker"
        )
        speaker_context += (
            f"- {name} ({s['role'] or 'unknown role'}): "
            f"{s['session_count']} sessions, "
            f"avg stress {round(float(s['avg_stress'] or 0) * 100)}%, "
            f"avg engagement {round(float(s['avg_engagement'] or 0) * 100)}%\n"
        )

    session_context = "\nRecent sessions:\n"
    for s in recent_sessions:
        session_context += (
            f"- {s['title']} ({s['meeting_type']}, "
            f"{s['created_at'].strftime('%b %d')}): "
            f"speakers {list(s['speakers'])}\n"
        )

    system_prompt = (
        "You are NEXUS, a behavioural analysis assistant. "
        "Answer questions about team performance and speaker trends "
        "using the provided cross-session data.\n\n"
        "Rules:\n"
        "- Reference specific speakers by name and cite their metrics.\n"
        "- For trend questions, describe direction (improving/declining/stable) and magnitude.\n"
        "- For comparison questions, highlight specific metric differences.\n"
        "- Frame observations as data-driven insights, not judgments.\n"
        "- If asked about a specific session, suggest opening that session for details.\n"
        "- Never claim certainty about emotions — describe behavioral indicators."
    )

    user_prompt = f"{speaker_context}\n{session_context}\n\nQuestion: {question}"
    if body.history:
        history_text = "\n".join(
            f"{m.get('role', 'user').title()}: {m.get('content', '')}"
            for m in body.history[-4:]
        )
        user_prompt = f"Previous conversation:\n{history_text}\n\n{user_prompt}"

    from shared.utils.llm_client import acomplete

    try:
        answer = await acomplete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=600,
            model="gpt-4o",
        )
    except Exception as exc:
        logger.error("Global chat LLM call failed: %s", exc)
        raise HTTPException(502, f"LLM generation failed: {exc}")

    return {"answer": answer, "speakers_in_context": len(speakers)}
