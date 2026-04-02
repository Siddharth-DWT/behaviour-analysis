"""
NEXUS API Gateway - Authentication Module
Password hashing (bcrypt), JWT tokens, and FastAPI auth dependencies.
"""
import os
import re
import uuid as _uuid
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Request, HTTPException, Depends
from jose import jwt, JWTError, ExpiredSignatureError
import bcrypt

logger = logging.getLogger("nexus.gateway.auth")


def generate_verification_token() -> str:
    return secrets.token_urlsafe(48)


# ── JWT Configuration ──

JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET:
    JWT_SECRET = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET not set — using auto-generated secret (will change on restart!)"
    )
else:
    logger.info("Using JWT_SECRET from environment")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))

# ── Password hashing (bcrypt directly — passlib is incompatible with bcrypt>=4.1) ──


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── Email / Password validation ──

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
PASSWORD_RE_UPPER = re.compile(r"[A-Z]")
PASSWORD_RE_DIGIT = re.compile(r"\d")


def validate_email(email: str) -> str:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(422, "Invalid email format")
    return email


def validate_password(password: str) -> None:
    if len(password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if not PASSWORD_RE_UPPER.search(password):
        raise HTTPException(422, "Password must contain at least 1 uppercase letter")
    if not PASSWORD_RE_DIGIT.search(password):
        raise HTTPException(422, "Password must contain at least 1 number")


# ── JWT Token creation ──


def create_access_token(user_id: str, email: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token_value() -> tuple[str, datetime]:
    """Generate a secure random refresh token string and its expiry."""
    token = secrets.token_urlsafe(64)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return token, expires_at


def verify_access_token(token: str) -> dict:
    """Verify and decode an access JWT. Returns payload dict or raises 401."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except ExpiredSignatureError:
        raise HTTPException(401, "Token has expired")
    except JWTError:
        raise HTTPException(401, "Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(401, "Invalid token type")

    return payload


# ── FastAPI Dependencies ──


async def get_current_user(request: Request) -> dict:
    """
    Extract and verify the Bearer token from the Authorization header.
    Returns a dict with {id, email, role, full_name} or raises 401.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    token = auth_header[7:]
    payload = verify_access_token(token)

    # Fetch user from DB to ensure they still exist and are active
    from database import get_pool

    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT id, email, full_name, role, company, avatar_url, org_id,
               is_active, created_at, last_login_at
        FROM users WHERE id = $1
        """,
        _uuid.UUID(payload["sub"]),
    )
    if not row:
        raise HTTPException(401, "User not found")
    if not row["is_active"]:
        raise HTTPException(403, "Account is deactivated")

    return {
        "id": str(row["id"]),
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
        "company": row["company"],
        "avatar_url": row["avatar_url"],
        "org_id": str(row["org_id"]) if row["org_id"] else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "last_login_at": row["last_login_at"].isoformat() if row["last_login_at"] else None,
    }


# Role hierarchy: admin > member > viewer
ROLE_HIERARCHY = {"viewer": 0, "member": 1, "admin": 2}


def require_role(required_role: str):
    """
    Returns a FastAPI dependency that checks the user has at least the required role.
    Usage: current_user: dict = Depends(require_role("member"))
    """
    async def _dependency(request: Request) -> dict:
        user = await get_current_user(request)
        user_level = ROLE_HIERARCHY.get(user["role"], 0)
        required_level = ROLE_HIERARCHY.get(required_role, 0)
        if user_level < required_level:
            raise HTTPException(403, f"Requires {required_role} role or higher")
        return user

    return _dependency


# ── DB helpers for auth tokens ──


async def store_refresh_token(
    user_id: str,
    refresh_token: str,
    expires_at: datetime,
    device_info: Optional[str] = None,
) -> None:
    from database import get_pool

    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO auth_tokens (user_id, refresh_token, device_info, expires_at)
        VALUES ($1, $2, $3, $4)
        """,
        _uuid.UUID(user_id),
        refresh_token,
        device_info,
        expires_at,
    )


async def verify_and_consume_refresh_token(refresh_token: str) -> Optional[dict]:
    """
    Look up a refresh token, verify it hasn't expired, delete it (single-use),
    and return the associated user. Returns None if invalid.
    """
    from database import get_pool

    pool = await get_pool()

    row = await pool.fetchrow(
        """
        DELETE FROM auth_tokens
        WHERE refresh_token = $1 AND expires_at > NOW()
        RETURNING user_id
        """,
        refresh_token,
    )
    if not row:
        return None

    user_id = row["user_id"]  # already a UUID from the DB
    user = await pool.fetchrow(
        """
        SELECT id, email, full_name, role, company, avatar_url, org_id,
               is_active, created_at, last_login_at
        FROM users WHERE id = $1 AND is_active = true
        """,
        user_id,
    )
    if not user:
        return None

    return {
        "id": str(user["id"]),
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
        "company": user["company"],
        "avatar_url": user["avatar_url"],
        "org_id": str(user["org_id"]) if user["org_id"] else None,
        "created_at": user["created_at"].isoformat() if user["created_at"] else None,
        "last_login_at": user["last_login_at"].isoformat() if user["last_login_at"] else None,
    }


async def delete_refresh_token(refresh_token: str) -> None:
    from database import get_pool

    pool = await get_pool()
    await pool.execute(
        "DELETE FROM auth_tokens WHERE refresh_token = $1",
        refresh_token,
    )


async def cleanup_expired_tokens() -> int:
    """Delete expired refresh tokens. Returns count deleted."""
    from database import get_pool

    pool = await get_pool()
    result = await pool.execute(
        "DELETE FROM auth_tokens WHERE expires_at < NOW()"
    )
    # asyncpg returns "DELETE N"
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
