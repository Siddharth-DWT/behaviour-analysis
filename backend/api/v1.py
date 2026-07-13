# backend/api/v1.py
"""
NEXUS Backend — Public programmatic API (/v1/*)

Two auth modes, both via get_current_user (which accepts JWT or 'Bearer nxs_…' keys):
  - Key management (POST/GET/DELETE /v1/api-keys) — intended for JWT callers.
  - Analysis (POST /v1/analyze, GET /v1/sessions/*) — intended for API-key callers.

Sessions created here are owned by the token's user (per-user PAT model), and the
read endpoints enforce that ownership (404 on mismatch — no existence leak).
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile,
)
from pydantic import BaseModel, Field

from core.auth import get_current_user, require_role
from core.api_keys import (
    create_api_key, list_api_keys, revoke_api_key, record_usage,
)
from core.webhook_endpoints import (
    create_webhook_endpoint, list_webhook_endpoints, update_webhook_endpoint,
    delete_webhook_endpoint, rotate_secret,
)
from core.database import DEV_ORG_ID, get_report, get_session, get_signals
from api._session_launch import create_session_and_dispatch
from services.webhook_service import validate_callback_url
from dependencies import get_db_pool, get_pipeline

logger = logging.getLogger("nexus.backend.v1")

router = APIRouter(tags=["v1"])


# ── Key management (JWT auth) ───────────────────────────────────────────────

class CreateKeyIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    rate_limit_per_min: Optional[int] = Field(default=None, ge=1, le=6000)
    # Optional expiry. None = never expires (default). Max 10 years.
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


@router.post("/api-keys")
async def create_key(
    body: CreateKeyIn,
    current_user: dict = Depends(require_role("member")),
):
    """Create an API key. The plaintext key is returned EXACTLY ONCE."""
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)
        if body.expires_in_days
        else None
    )
    row, raw = await create_api_key(
        user_id=current_user["id"],
        org_id=current_user.get("org_id"),
        name=body.name,
        rate_limit_per_min=body.rate_limit_per_min or 60,
        expires_at=expires_at,
    )
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "key": raw,  # plaintext — shown once, never retrievable again
        "key_prefix": row["key_prefix"],
        "last4": row["key_last4"],
        "rate_limit_per_min": row["rate_limit_per_min"],
        "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


@router.get("/api-keys")
async def list_keys(current_user: dict = Depends(require_role("member"))):
    """List the caller's API keys (masked — no plaintext, no hashes)."""
    return {"api_keys": await list_api_keys(current_user["id"])}


@router.delete("/api-keys/{key_id}")
async def delete_key(
    key_id: str,
    current_user: dict = Depends(require_role("member")),
):
    """Revoke an API key. Ownership-checked to the caller."""
    ok = await revoke_api_key(key_id, current_user["id"])
    if not ok:
        raise HTTPException(404, "API key not found")
    return {"id": key_id, "revoked": True}


# ── Analysis (API-key or JWT auth) ──────────────────────────────────────────

@router.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(default=""),
    meeting_type: str = Form(default="sales_call"),
    config: str = Form(default="{}"),
    callback_url: Optional[str] = Form(default=None),
    retain_media: bool = Form(default=False),
    current_user: dict = Depends(require_role("member")),
    pipeline=Depends(get_pipeline),
    pool=Depends(get_db_pool),
):
    """
    Submit an audio/video file for analysis. Returns immediately with a session_id;
    poll GET /v1/sessions/{id} or register a callback_url for a signed webhook.
    """
    if callback_url:
        validate_callback_url(callback_url)  # SSRF guard at submit time

    try:
        config_dict = json.loads(config) if config and config.strip() else {}
    except json.JSONDecodeError:
        config_dict = {}

    result = await create_session_and_dispatch(
        file=file,
        title=title,
        meeting_type=meeting_type,
        config_dict=config_dict,
        current_user=current_user,
        pipeline=pipeline,
        pool=pool,
        background_tasks=background_tasks,
        callback_url=callback_url,
        retain_media=retain_media,
    )

    await record_usage(
        current_user.get("_api_key_id"), result.get("session_id"), "/v1/analyze", 202,
    )
    return result


# ── Ownership-checked reads ─────────────────────────────────────────────────

async def _owned_session(session_id: str, current_user: dict) -> dict:
    """Fetch a session and assert it belongs to the caller. 404 on any mismatch."""
    try:
        import uuid as _uuid
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id, org_id=current_user.get("org_id") or DEV_ORG_ID)
    if not session or str(session.get("user_id")) != str(current_user["id"]):
        raise HTTPException(404, "Session not found")
    return session


@router.get("/sessions/{session_id}")
async def v1_get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Session status + result summary, scoped to the token's owner."""
    session = await _owned_session(session_id, current_user)
    signals = await get_signals(session_id, limit=50000)
    report = await get_report(session_id)

    signals_by_agent: dict[str, int] = {}
    for s in signals:
        agent = s.get("agent", "unknown")
        signals_by_agent[agent] = signals_by_agent.get(agent, 0) + 1

    return {
        "session_id": session_id,
        "status": session.get("status"),
        "title": session.get("title"),
        "meeting_type": session.get("meeting_type"),
        "duration_ms": session.get("duration_ms"),
        "speaker_count": session.get("speaker_count"),
        "signal_count": len(signals),
        "signals_by_agent": signals_by_agent,
        "has_report": report is not None,
        "created_at": session.get("created_at"),
        "completed_at": session.get("completed_at"),
    }


@router.get("/sessions/{session_id}/signals")
async def v1_get_signals(
    session_id: str,
    agent: Optional[str] = Query(default=None),
    signal_type: Optional[str] = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=50000),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """All signals for a session, scoped to the token's owner."""
    await _owned_session(session_id, current_user)
    signals = await get_signals(
        session_id, agent=agent, signal_type=signal_type, limit=limit, offset=offset,
    )
    return {"session_id": session_id, "signals": signals, "count": len(signals)}


@router.get("/sessions/{session_id}/report")
async def v1_get_report(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Narrative report for a session, scoped to the token's owner."""
    await _owned_session(session_id, current_user)
    report = await get_report(session_id)
    if not report:
        raise HTTPException(404, "No report found for this session")
    return {"session_id": session_id, "report": report}


# ── Managed webhook endpoints (JWT auth) ────────────────────────────────────

class CreateWebhookIn(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)
    description: Optional[str] = Field(default=None, max_length=200)


class UpdateWebhookIn(BaseModel):
    url: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    active: Optional[bool] = None
    description: Optional[str] = Field(default=None, max_length=200)


@router.post("/webhooks")
async def create_webhook(
    body: CreateWebhookIn,
    current_user: dict = Depends(require_role("member")),
):
    """Register a webhook endpoint. The signing secret is returned EXACTLY ONCE."""
    validate_callback_url(body.url)  # SSRF guard
    row, secret = await create_webhook_endpoint(
        user_id=current_user["id"],
        org_id=current_user.get("org_id"),
        url=body.url,
        description=body.description,
    )
    return {**row, "secret": secret}  # secret shown once


@router.get("/webhooks")
async def get_webhooks(current_user: dict = Depends(require_role("member"))):
    """List the caller's webhook endpoints (masked — no secret)."""
    return {"webhooks": await list_webhook_endpoints(current_user["id"])}


@router.patch("/webhooks/{endpoint_id}")
async def patch_webhook(
    endpoint_id: str,
    body: UpdateWebhookIn,
    current_user: dict = Depends(require_role("member")),
):
    """Update url / active / description. Ownership-checked."""
    if body.url is not None:
        validate_callback_url(body.url)  # SSRF guard on new url
    row = await update_webhook_endpoint(
        endpoint_id, current_user["id"],
        url=body.url, active=body.active, description=body.description,
    )
    if row is None:
        raise HTTPException(404, "Webhook endpoint not found")
    return row


@router.delete("/webhooks/{endpoint_id}")
async def remove_webhook(
    endpoint_id: str,
    current_user: dict = Depends(require_role("member")),
):
    """Delete a webhook endpoint. Ownership-checked."""
    ok = await delete_webhook_endpoint(endpoint_id, current_user["id"])
    if not ok:
        raise HTTPException(404, "Webhook endpoint not found")
    return {"id": endpoint_id, "deleted": True}


@router.post("/webhooks/{endpoint_id}/rotate-secret")
async def rotate_webhook_secret(
    endpoint_id: str,
    current_user: dict = Depends(require_role("member")),
):
    """Rotate the signing secret. The new secret is returned EXACTLY ONCE."""
    secret = await rotate_secret(endpoint_id, current_user["id"])
    if secret is None:
        raise HTTPException(404, "Webhook endpoint not found")
    return {"id": endpoint_id, "secret": secret}


@router.get("/webhook-deliveries")
async def get_webhook_deliveries(
    limit: int = Query(default=50, ge=1, le=500),
    current_user: dict = Depends(require_role("member")),
    pool=Depends(get_db_pool),
):
    """Recent webhook delivery attempts for the caller's sessions."""
    import uuid as _uuid
    rows = await pool.fetch(
        """
        SELECT wd.id, wd.session_id, wd.event, wd.callback_url, wd.attempts,
               wd.delivered, wd.last_status, wd.last_error, wd.created_at, wd.delivered_at
        FROM webhook_deliveries wd
        JOIN sessions s ON s.id = wd.session_id
        WHERE s.user_id = $1
        ORDER BY wd.created_at DESC
        LIMIT $2
        """,
        _uuid.UUID(current_user["id"]),
        limit,
    )
    deliveries = []
    for r in rows:
        d = dict(r)
        d["id"] = str(d["id"])
        d["session_id"] = str(d["session_id"]) if d["session_id"] else None
        for ts in ("created_at", "delivered_at"):
            d[ts] = d[ts].isoformat() if d.get(ts) else None
        deliveries.append(d)
    return {"deliveries": deliveries, "count": len(deliveries)}
