# backend/core/webhook_endpoints.py
"""
NEXUS Backend - Managed Webhook Endpoints

A user registers a callback URL once; it is auto-used for their sessions when a
request does not pass its own callback_url (see analysis_pipeline._fire_webhook).
The server signs outgoing payloads with the endpoint's `secret`, so it is stored
plaintext (same model as Stripe endpoint secrets). The secret is shown to the user
once on create / rotate; list responses never include it.
"""
import secrets
import uuid as _uuid
from typing import Optional


SECRET_PREFIX = "whsec_"


def generate_webhook_secret() -> str:
    """Return a signing secret: 'whsec_' + 43 urlsafe chars (256-bit)."""
    return f"{SECRET_PREFIX}{secrets.token_urlsafe(32)}"


async def create_webhook_endpoint(
    user_id: str,
    org_id: Optional[str],
    url: str,
    description: Optional[str] = None,
) -> tuple[dict, str]:
    """Insert a webhook endpoint. Returns (masked_row_dict, secret_shown_once)."""
    from core.database import get_pool

    secret = generate_webhook_secret()
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO webhook_endpoints (user_id, org_id, url, secret, description)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id, url, description, active, created_at, last_delivery_at
        """,
        _uuid.UUID(user_id),
        _uuid.UUID(org_id) if org_id else None,
        url,
        secret,
        description,
    )
    return _mask(row), secret


async def list_webhook_endpoints(user_id: str) -> list[dict]:
    """Return the caller's endpoints, masked (never the full secret)."""
    from core.database import get_pool

    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, url, description, active, created_at, last_delivery_at
        FROM webhook_endpoints WHERE user_id = $1
        ORDER BY created_at DESC
        """,
        _uuid.UUID(user_id),
    )
    return [_mask(r) for r in rows]


async def update_webhook_endpoint(
    endpoint_id: str,
    user_id: str,
    *,
    url: Optional[str] = None,
    active: Optional[bool] = None,
    description: Optional[str] = None,
) -> Optional[dict]:
    """
    Partial update, ownership-checked. Only provided fields change.
    Returns the masked row, or None if not found / not owned.
    """
    from core.database import get_pool

    sets: list[str] = []
    params: list = []
    idx = 1
    if url is not None:
        sets.append(f"url = ${idx}"); params.append(url); idx += 1
    if active is not None:
        sets.append(f"active = ${idx}"); params.append(active); idx += 1
    if description is not None:
        sets.append(f"description = ${idx}"); params.append(description); idx += 1
    if not sets:
        # Nothing to change — just return current row (still ownership-checked).
        return await _get_owned(endpoint_id, user_id)

    sets.append("updated_at = NOW()")
    params.extend([_uuid.UUID(endpoint_id), _uuid.UUID(user_id)])
    pool = await get_pool()
    row = await pool.fetchrow(
        f"""
        UPDATE webhook_endpoints SET {', '.join(sets)}
        WHERE id = ${idx} AND user_id = ${idx + 1}
        RETURNING id, url, description, active, created_at, last_delivery_at
        """,
        *params,
    )
    return _mask(row) if row else None


async def delete_webhook_endpoint(endpoint_id: str, user_id: str) -> bool:
    """Delete an endpoint, ownership-checked. Returns True if a row was removed."""
    from core.database import get_pool

    pool = await get_pool()
    row = await pool.fetchrow(
        "DELETE FROM webhook_endpoints WHERE id = $1 AND user_id = $2 RETURNING id",
        _uuid.UUID(endpoint_id),
        _uuid.UUID(user_id),
    )
    return row is not None


async def rotate_secret(endpoint_id: str, user_id: str) -> Optional[str]:
    """Generate a new signing secret, ownership-checked. Returns it once, or None."""
    from core.database import get_pool

    secret = generate_webhook_secret()
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        UPDATE webhook_endpoints SET secret = $1, updated_at = NOW()
        WHERE id = $2 AND user_id = $3
        RETURNING id
        """,
        secret,
        _uuid.UUID(endpoint_id),
        _uuid.UUID(user_id),
    )
    return secret if row else None


async def get_active_endpoints_for_user(user_id: str) -> list[dict]:
    """
    Return active endpoints as [{id, url, secret}] for the pipeline fallback.
    The ONLY reader of the plaintext secret.
    """
    from core.database import get_pool

    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, url, secret FROM webhook_endpoints WHERE user_id = $1 AND active",
        _uuid.UUID(user_id) if isinstance(user_id, str) else user_id,
    )
    return [{"id": str(r["id"]), "url": r["url"], "secret": r["secret"]} for r in rows]


async def mark_delivered(endpoint_id: str) -> None:
    """Stamp last_delivery_at after a delivery attempt to this endpoint."""
    from core.database import get_pool

    pool = await get_pool()
    await pool.execute(
        "UPDATE webhook_endpoints SET last_delivery_at = NOW() WHERE id = $1",
        _uuid.UUID(endpoint_id) if isinstance(endpoint_id, str) else endpoint_id,
    )


# ── helpers ──

def _mask(row) -> dict:
    d = dict(row)
    d["id"] = str(d["id"])
    for ts in ("created_at", "last_delivery_at"):
        d[ts] = d[ts].isoformat() if d.get(ts) else None
    return d


async def _get_owned(endpoint_id: str, user_id: str) -> Optional[dict]:
    from core.database import get_pool

    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT id, url, description, active, created_at, last_delivery_at
        FROM webhook_endpoints WHERE id = $1 AND user_id = $2
        """,
        _uuid.UUID(endpoint_id),
        _uuid.UUID(user_id),
    )
    return _mask(row) if row else None
