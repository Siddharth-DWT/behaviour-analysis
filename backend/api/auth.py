# backend/api/auth.py
"""
NEXUS Backend — Auth routes (/auth/*)
Ported from services/api_gateway/main.py lines ~415–892.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.auth import (
    create_access_token,
    create_refresh_token_value,
    delete_refresh_token,
    generate_verification_token,
    get_current_user,
    hash_password,
    store_refresh_token,
    validate_email,
    validate_password,
    verify_and_consume_refresh_token,
    verify_password,
)
from core.database import DEV_ORG_ID
from core.email_service import (
    is_email_configured,
    send_password_reset_email,
    send_verification_email,
)
from dependencies import get_db_pool

logger = logging.getLogger("nexus.backend.auth")

router = APIRouter(tags=["auth"])


# ── Request models ─────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str
    company: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    company: Optional[str] = None
    avatar_url: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ResendVerificationRequest(BaseModel):
    email: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


# ── Public endpoints ───────────────────────────────────────────────────────────

@router.post("/signup")
async def signup(body: SignupRequest, pool=Depends(get_db_pool)):
    """Register a new user account."""
    email = validate_email(body.email)
    validate_password(body.password)

    existing = await pool.fetchrow(
        "SELECT id, is_verified FROM users WHERE email = $1", email
    )
    if existing:
        if existing["is_verified"]:
            raise HTTPException(409, "Email already registered")
        await pool.execute("DELETE FROM users WHERE id = $1", existing["id"])
        logger.info("Removed unverified orphan user for %s", email)

    password_hash = hash_password(body.password)

    if is_email_configured():
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO users (org_id, email, name, full_name, password_hash, company, role)
                    VALUES ($1, $2, $3, $3, $4, $5, 'member')
                    RETURNING id, email, full_name, role, company
                    """,
                    DEV_ORG_ID, email, body.full_name, password_hash, body.company,
                )
                token = generate_verification_token()
                await conn.execute(
                    """
                    INSERT INTO email_verifications (user_id, token, expires_at)
                    VALUES ($1, $2, NOW() + INTERVAL '24 hours')
                    """,
                    row["id"], token,
                )
                email_sent = await send_verification_email(email, body.full_name, token)
                if not email_sent:
                    raise Exception("Verification email could not be sent")

        return {
            "user": {
                "id": str(row["id"]),
                "email": row["email"],
                "full_name": row["full_name"],
                "role": row["role"],
                "company": row["company"],
            },
            "requires_verification": True,
            "message": "Verification email sent. Please check your inbox to verify your email address.",
        }

    logger.warning("Email verification disabled — auto-verifying user")
    row = await pool.fetchrow(
        """
        INSERT INTO users (org_id, email, name, full_name, password_hash, company, role, is_verified)
        VALUES ($1, $2, $3, $3, $4, $5, 'member', true)
        RETURNING id, email, full_name, role, company
        """,
        DEV_ORG_ID, email, body.full_name, password_hash, body.company,
    )

    user_id = str(row["id"])
    access_token = create_access_token(user_id, email, row["role"])
    refresh_token, expires_at = create_refresh_token_value()
    await store_refresh_token(user_id, refresh_token, expires_at)

    return {
        "user": {
            "id": user_id,
            "email": row["email"],
            "full_name": row["full_name"],
            "role": row["role"],
            "company": row["company"],
        },
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


@router.post("/login")
async def login(body: LoginRequest, pool=Depends(get_db_pool)):
    """Authenticate with email and password."""
    email = body.email.strip().lower()

    row = await pool.fetchrow(
        """
        SELECT id, email, full_name, role, company, avatar_url, org_id,
               password_hash, is_active, is_verified, created_at, last_login_at
        FROM users WHERE email = $1
        """,
        email,
    )

    if not row or not row["password_hash"]:
        raise HTTPException(401, "Invalid email or password")

    if not row["is_active"]:
        raise HTTPException(403, "Account is deactivated")

    if not verify_password(body.password, row["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    if not row["is_verified"]:
        raise HTTPException(
            403,
            "Email not verified. Please check your inbox for the verification link.",
        )

    user_id = str(row["id"])

    await pool.execute(
        "UPDATE users SET last_login_at = $1 WHERE id = $2",
        datetime.now(timezone.utc), row["id"],
    )

    access_token = create_access_token(user_id, row["email"], row["role"])
    refresh_token, expires_at = create_refresh_token_value()
    await store_refresh_token(user_id, refresh_token, expires_at)

    return {
        "user": {
            "id": user_id,
            "email": row["email"],
            "full_name": row["full_name"],
            "role": row["role"],
            "company": row["company"],
            "avatar_url": row["avatar_url"],
        },
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


@router.get("/verify-email")
async def verify_email(token: str = Query(...), pool=Depends(get_db_pool)):
    """Verify a user's email address using the token from the verification email."""
    row = await pool.fetchrow(
        """
        SELECT ev.id AS verification_id, ev.user_id, ev.expires_at, ev.used_at,
               u.email, u.full_name, u.is_verified
        FROM email_verifications ev
        JOIN users u ON u.id = ev.user_id
        WHERE ev.token = $1
        """,
        token,
    )

    if not row:
        raise HTTPException(400, "Invalid verification token.")

    if row["used_at"] is not None:
        raise HTTPException(400, "This verification link has already been used.")

    if row["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(400, "Verification link has expired. Please request a new one.")

    await pool.execute("UPDATE users SET is_verified = true WHERE id = $1", row["user_id"])
    await pool.execute(
        "UPDATE email_verifications SET used_at = $1 WHERE id = $2",
        datetime.now(timezone.utc), row["verification_id"],
    )

    return {"success": True, "message": "Email verified successfully. You can now log in."}


@router.post("/resend-verification")
async def resend_verification(body: ResendVerificationRequest, pool=Depends(get_db_pool)):
    """Resend the verification email for an unverified account."""
    email = body.email.strip().lower()

    row = await pool.fetchrow(
        "SELECT id, full_name, is_verified FROM users WHERE email = $1", email
    )

    if not row or row["is_verified"]:
        return {"message": "If that email is registered, a verification email has been sent."}

    recent_count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM email_verifications
        WHERE user_id = $1 AND created_at > NOW() - INTERVAL '1 hour'
        """,
        row["id"],
    )
    if recent_count >= 3:
        raise HTTPException(429, "Too many verification emails requested. Please try again later.")

    if not is_email_configured():
        raise HTTPException(503, "Email service is not configured. Please contact support.")

    token = generate_verification_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    await pool.execute(
        "INSERT INTO email_verifications (user_id, token, expires_at) VALUES ($1, $2, $3)",
        row["id"], token, expires_at,
    )
    await send_verification_email(email, row["full_name"], token)

    return {"message": "Verification email resent."}


@router.post("/refresh")
async def refresh(body: RefreshRequest):
    """Exchange a valid refresh token for new access + refresh tokens."""
    user = await verify_and_consume_refresh_token(body.refresh_token)
    if not user:
        raise HTTPException(401, "Invalid or expired refresh token")

    access_token = create_access_token(user["id"], user["email"], user["role"])
    new_refresh, expires_at = create_refresh_token_value()
    await store_refresh_token(user["id"], new_refresh, expires_at)

    return {
        "user": {
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
            "company": user.get("company"),
            "avatar_url": user.get("avatar_url"),
        },
        "access_token": access_token,
        "refresh_token": new_refresh,
    }


@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest, pool=Depends(get_db_pool)):
    """
    Send a password reset email.
    Always returns 200 to prevent email enumeration.
    """
    email = body.email.strip().lower()

    row = await pool.fetchrow(
        "SELECT id, full_name FROM users WHERE email = $1 AND is_active = true", email
    )

    if row:
        token = generate_verification_token()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        await pool.execute(
            "DELETE FROM password_reset_tokens WHERE user_id = $1 AND used_at IS NULL",
            row["id"],
        )
        await pool.execute(
            "INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES ($1, $2, $3)",
            row["id"], token, expires_at,
        )

        asyncio.create_task(
            send_password_reset_email(email, row["full_name"] or "there", token)
        )

    return {"message": "If that email is registered you will receive a reset link shortly."}


@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest, pool=Depends(get_db_pool)):
    """Reset password using a valid (unexpired, unused) reset token."""
    validate_password(body.new_password)

    row = await pool.fetchrow(
        """
        SELECT prt.id, prt.user_id, prt.used_at
        FROM password_reset_tokens prt
        WHERE prt.token = $1 AND prt.expires_at > NOW()
        """,
        body.token,
    )

    if not row:
        raise HTTPException(400, "Invalid or expired reset link.")
    if row["used_at"] is not None:
        raise HTTPException(400, "This reset link has already been used.")

    new_hash = hash_password(body.new_password)
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE users SET password_hash = $1, updated_at = $2 WHERE id = $3",
                new_hash, now, row["user_id"],
            )
            await conn.execute(
                "UPDATE password_reset_tokens SET used_at = $1 WHERE id = $2",
                now, row["id"],
            )

    return {"success": True, "message": "Password updated successfully."}


# ── Authenticated endpoints ────────────────────────────────────────────────────

@router.post("/logout")
async def logout(body: LogoutRequest, current_user: dict = Depends(get_current_user)):
    """Invalidate a refresh token."""
    await delete_refresh_token(body.refresh_token)
    return {"success": True}


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user profile."""
    return current_user


@router.put("/me")
async def update_me(
    body: UpdateProfileRequest,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Update current user profile fields (not email or password)."""
    sets: list[str] = []
    params: list = []
    idx = 1

    if body.full_name is not None:
        sets.append(f"full_name = ${idx}")
        params.append(body.full_name)
        idx += 1
    if body.company is not None:
        sets.append(f"company = ${idx}")
        params.append(body.company)
        idx += 1
    if body.avatar_url is not None:
        sets.append(f"avatar_url = ${idx}")
        params.append(body.avatar_url)
        idx += 1

    if not sets:
        raise HTTPException(422, "No fields to update")

    sets.append(f"updated_at = ${idx}")
    params.append(datetime.now(timezone.utc))
    idx += 1

    params.append(current_user["id"])
    set_clause = ", ".join(sets)

    row = await pool.fetchrow(
        f"""
        UPDATE users SET {set_clause}
        WHERE id = ${idx}
        RETURNING id, email, full_name, role, company, avatar_url, created_at, last_login_at
        """,
        *params,
    )

    return {
        "id": str(row["id"]),
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
        "company": row["company"],
        "avatar_url": row["avatar_url"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "last_login_at": row["last_login_at"].isoformat() if row["last_login_at"] else None,
    }


@router.put("/change-password")
async def change_password(
    body: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Change the current user's password."""
    validate_password(body.new_password)

    row = await pool.fetchrow(
        "SELECT password_hash FROM users WHERE id = $1", current_user["id"]
    )

    if not row or not verify_password(body.current_password, row["password_hash"]):
        raise HTTPException(401, "Current password is incorrect")

    new_hash = hash_password(body.new_password)
    await pool.execute(
        "UPDATE users SET password_hash = $1, updated_at = $2 WHERE id = $3",
        new_hash, datetime.now(timezone.utc), current_user["id"],
    )

    return {"success": True}
