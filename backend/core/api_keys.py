# backend/core/api_keys.py
"""
NEXUS Backend - API Key (Personal Access Token) Module

Programmatic auth for external integrations. A key acts as its creator user:
sessions it creates inherit that user's id / role / org_id (per-user PAT model).

Lookup strategy — SHA-256 in a UNIQUE indexed column (O(1) probe), NOT bcrypt scan:
  API keys are 256-bit random (high-entropy), unlike passwords, so the slow-hash
  rationale does not apply. bcrypt's per-row random salt makes it un-indexable
  (would force an O(n) scan + bcrypt verify per row). We store `key_lookup =
  sha256(raw)` for the lookup and keep a bcrypt `key_hash` as optional
  defense-in-depth, verified against the single located row (gated by env).

The dict returned by resolve_api_key() is shape-identical to core.auth.get_current_user()
so every existing Depends(get_current_user) / require_role() route works unchanged.
"""
import hashlib
import os
import secrets
import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional

import bcrypt

KEY_PREFIX = "nxs_live_"
DEFAULT_RATE_LIMIT = int(os.getenv("API_KEY_RATE_LIMIT_PER_MIN", "60"))
# bcrypt verify is optional — SHA-256 lookup alone is cryptographically sufficient
# for high-entropy keys. Disable via API_KEY_VERIFY_BCRYPT=false for max throughput.
VERIFY_BCRYPT = os.getenv("API_KEY_VERIFY_BCRYPT", "true").lower() == "true"


# ── Generation & hashing ──

def generate_api_key() -> str:
    """Return a full raw key: 'nxs_live_' + 43 urlsafe chars (256-bit entropy)."""
    return f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"


def lookup_hash(raw_key: str) -> str:
    """Deterministic SHA-256 hex of the full key — the indexed lookup value."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def hash_api_key(raw_key: str) -> str:
    """bcrypt hash for defense-in-depth (verified against the single located row)."""
    return bcrypt.hashpw(raw_key.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# ── CRUD ──

async def create_api_key(
    user_id: str,
    org_id: Optional[str],
    name: str,
    rate_limit_per_min: int = DEFAULT_RATE_LIMIT,
    role: Optional[str] = None,
    expires_at: Optional[datetime] = None,
) -> tuple[dict, str]:
    """
    Insert a new key row. Returns (masked_row_dict, raw_key).
    The raw_key is shown to the caller EXACTLY ONCE — it is never stored in plaintext.
    """
    from core.database import get_pool

    raw = generate_api_key()
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO api_keys
            (user_id, org_id, name, key_prefix, key_last4, key_lookup, key_hash,
             role, rate_limit_per_min, expires_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id, name, key_prefix, key_last4, rate_limit_per_min,
                  role, expires_at, created_at
        """,
        _uuid.UUID(user_id),
        _uuid.UUID(org_id) if org_id else None,
        name,
        KEY_PREFIX,
        raw[-4:],
        lookup_hash(raw),
        hash_api_key(raw),
        role,
        rate_limit_per_min,
        expires_at,
    )
    return dict(row), raw


async def resolve_api_key(raw_key: str) -> Optional[dict]:
    """
    Resolve a raw key to a user dict IDENTICAL in shape to get_current_user()'s
    return (plus additive _api_key_id / _rate_limit_per_min). Returns None if the
    key is unknown, revoked, expired, fails bcrypt, or the user is inactive.

    Side effect: stamps last_used_at and increments request_count.
    """
    if not raw_key.startswith(KEY_PREFIX):
        return None

    from core.database import get_pool

    pool = await get_pool()
    key = await pool.fetchrow(
        """
        SELECT id, user_id, org_id, role, key_hash, rate_limit_per_min,
               expires_at, revoked_at
        FROM api_keys WHERE key_lookup = $1
        """,
        lookup_hash(raw_key),
    )
    if not key or key["revoked_at"] is not None:
        return None
    if key["expires_at"] and key["expires_at"] < datetime.now(timezone.utc):
        return None
    if VERIFY_BCRYPT and not bcrypt.checkpw(
        raw_key.encode("utf-8"), key["key_hash"].encode("utf-8")
    ):
        return None

    user = await pool.fetchrow(
        """
        SELECT id, email, full_name, role, company, avatar_url, org_id,
               is_active, created_at, last_login_at
        FROM users WHERE id = $1
        """,
        key["user_id"],
    )
    if not user or not user["is_active"]:
        return None

    # Usage stamp — cheap single-row update.
    await pool.execute(
        "UPDATE api_keys SET last_used_at = NOW(), request_count = request_count + 1 WHERE id = $1",
        key["id"],
    )

    effective_role = key["role"] or user["role"]  # key override, else inherit owner's role
    return {
        "id": str(user["id"]),
        "email": user["email"],
        "full_name": user["full_name"],
        "role": effective_role,
        "company": user["company"],
        "avatar_url": user["avatar_url"],
        "org_id": str(user["org_id"]) if user["org_id"] else None,
        "created_at": user["created_at"].isoformat() if user["created_at"] else None,
        "last_login_at": user["last_login_at"].isoformat() if user["last_login_at"] else None,
        # Additive fields consumed only by the /v1 layer — inert for JWT routes.
        "_api_key_id": str(key["id"]),
        "_rate_limit_per_min": key["rate_limit_per_min"],
    }


async def list_api_keys(user_id: str) -> list[dict]:
    """Return the caller's keys, masked (no plaintext, no hashes)."""
    from core.database import get_pool

    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, name, key_prefix, key_last4, rate_limit_per_min, request_count,
               role, last_used_at, expires_at, revoked_at, created_at
        FROM api_keys WHERE user_id = $1
        ORDER BY created_at DESC
        """,
        _uuid.UUID(user_id),
    )
    result: list[dict] = []
    for r in rows:
        d = dict(r)
        d["id"] = str(d["id"])
        d["masked_key"] = f"{d['key_prefix']}…{d['key_last4']}"
        d["active"] = d["revoked_at"] is None
        for ts in ("last_used_at", "expires_at", "revoked_at", "created_at"):
            d[ts] = d[ts].isoformat() if d[ts] else None
        result.append(d)
    return result


async def revoke_api_key(key_id: str, user_id: str) -> bool:
    """
    Revoke a key (sets revoked_at). Ownership-checked to user_id.
    Returns True if a key was revoked, False if not found / not owned / already revoked.
    """
    from core.database import get_pool

    pool = await get_pool()
    row = await pool.fetchrow(
        """
        UPDATE api_keys SET revoked_at = NOW()
        WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL
        RETURNING id
        """,
        _uuid.UUID(key_id),
        _uuid.UUID(user_id),
    )
    return row is not None


async def record_usage(
    key_id: Optional[str],
    session_id: Optional[str],
    endpoint: str,
    status_code: Optional[int] = None,
) -> None:
    """Insert an api_key_usage audit row. No-op if key_id is None (JWT caller)."""
    if not key_id:
        return
    from core.database import get_pool

    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO api_key_usage (key_id, session_id, endpoint, status_code)
        VALUES ($1, $2, $3, $4)
        """,
        _uuid.UUID(key_id),
        _uuid.UUID(session_id) if session_id else None,
        endpoint,
        status_code,
    )
