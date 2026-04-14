"""
NEXUS API Gateway
Central entry point for the NEXUS system. Accepts audio uploads, orchestrates
the Voice → Language → Fusion pipeline, persists results to PostgreSQL,
and serves session data to the React dashboard.

Endpoints:
  POST /auth/signup             → Register new user
  POST /auth/login              → Authenticate user
  POST /auth/refresh            → Refresh access token
  POST /auth/logout             → Invalidate refresh token
  GET  /auth/me                 → Current user profile
  PUT  /auth/me                 → Update user profile
  PUT  /auth/change-password    → Change password
  POST /sessions                → Upload audio, trigger full analysis pipeline (auth)
  GET  /sessions                → List sessions (auth, user-scoped)
  GET  /sessions/{id}           → Session detail with signals + alerts (auth)
  GET  /sessions/{id}/signals   → Signals for a session (auth)
  GET  /sessions/{id}/report    → Get or generate narrative report (auth)
  GET  /sessions/{id}/transcript→ Get transcript segments (auth)
  GET  /health                  → Health check (public)
"""
import os
import sys
import uuid
import time
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from pydantic import BaseModel
import httpx

# isort: split
# Project root must be on sys.path before shared.* imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# isort: split
from shared.models.requests import SessionListResponse

try:
    from database import (
        get_pool, close_pool,
        create_session, get_session, list_sessions, update_session_status,
        upsert_speakers, insert_signals, get_signals,
        insert_alerts, get_alerts,
        save_report, get_report,
        insert_transcript_segments, get_transcript,
        DEV_ORG_ID,
    )
    from auth import (
        hash_password, verify_password,
        validate_email, validate_password,
        create_access_token, create_refresh_token_value,
        verify_access_token,
        get_current_user, require_role,
        store_refresh_token, verify_and_consume_refresh_token,
        delete_refresh_token, cleanup_expired_tokens,
        generate_verification_token,
    )
    from email_service import is_email_configured, send_verification_email
except ImportError:
    from services.api_gateway.database import (
        get_pool, close_pool,
        create_session, get_session, list_sessions, update_session_status,
        upsert_speakers, insert_signals, get_signals,
        insert_alerts, get_alerts,
        save_report, get_report,
        insert_transcript_segments, get_transcript,
        DEV_ORG_ID,
    )
    from services.api_gateway.auth import (
        hash_password, verify_password,
        validate_email, validate_password,
        create_access_token, create_refresh_token_value,
        verify_access_token,
        get_current_user, require_role,
        store_refresh_token, verify_and_consume_refresh_token,
        delete_refresh_token, cleanup_expired_tokens,
        generate_verification_token,
    )
    try:
        from email_service import is_email_configured, send_verification_email
    except ImportError:
        from services.api_gateway.email_service import is_email_configured, send_verification_email

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.gateway")

# ── Agent URLs (configurable via environment) ──
VOICE_AGENT_URL = os.getenv("VOICE_AGENT_URL", "http://localhost:8002")
LANGUAGE_AGENT_URL = os.getenv("LANGUAGE_AGENT_URL", "http://localhost:8003")
CONVERSATION_AGENT_URL = os.getenv("CONVERSATION_AGENT_URL", "http://localhost:8011")
FUSION_AGENT_URL = os.getenv("FUSION_AGENT_URL", "http://localhost:8004")

# ── Upload directory ──
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/recordings"))
RECORDING_RETENTION_DAYS = int(os.getenv("RECORDING_RETENTION_DAYS", "3"))

# ── HTTP client timeout (Voice Agent with Whisper can be slow) ──
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "1800"))  # 30 minutes

# ── CORS origins (comma-separated list) ──
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3006,http://localhost:5173").split(",")

# ── In-memory pipeline step tracker ──
# Tracks the active pipeline step per session so the frontend can poll real progress.
# Process-local (no DB write needed — sessions only process once per gateway instance).
_pipeline_progress: dict[str, str] = {}

def _set_step(session_id: str, step: str) -> None:
    _pipeline_progress[session_id] = step

app = FastAPI(
    title="NEXUS API Gateway",
    description="Central API for the NEXUS multi-agent behavioural analysis system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request access logger (replaces missing --access-log) ──
class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        logger.info(f"{request.method} {request.url.path} → {response.status_code}")
        return response

app.add_middleware(AccessLogMiddleware)


def _cleanup_old_recordings():
    """Delete recording files older than RECORDING_RETENTION_DAYS."""
    if not UPLOAD_DIR.exists():
        return
    cutoff = time.time() - (RECORDING_RETENTION_DAYS * 86400)
    removed = 0
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        logger.info(f"Cleaned up {removed} recording(s) older than {RECORDING_RETENTION_DAYS} days")


@app.on_event("startup")
async def startup():
    logger.info("Starting NEXUS API Gateway...")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_recordings()

    try:
        await get_pool()
        logger.info("Connected to PostgreSQL.")
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed (non-fatal): {e}")

    try:
        from neo4j_schema import init_neo4j_schema
        await init_neo4j_schema()
    except Exception as e:
        logger.warning(f"Neo4j schema init failed (non-fatal): {e}")

    # Ensure password_reset_tokens table exists (idempotent for existing installs)
    try:
        _pool = await get_pool()
        await _pool.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token       VARCHAR(128) UNIQUE NOT NULL,
                expires_at  TIMESTAMPTZ NOT NULL,
                used_at     TIMESTAMPTZ,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await _pool.execute("CREATE INDEX IF NOT EXISTS idx_prt_token ON password_reset_tokens(token)")
        await _pool.execute("CREATE INDEX IF NOT EXISTS idx_prt_user_id ON password_reset_tokens(user_id)")
    except Exception as e:
        logger.warning(f"Password reset table init failed (non-fatal): {e}")

    logger.info("API Gateway ready.")


@app.on_event("shutdown")
async def shutdown():
    await close_pool()
    try:
        from neo4j_sync import close_driver as close_neo4j_driver
        await close_neo4j_driver()
    except Exception as e:
        logger.warning(f"Neo4j driver close failed: {e}")


@app.get("/health")
async def health():
    """Health check — also probes downstream agents."""
    agent_status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in [
            ("voice", VOICE_AGENT_URL),
            ("language", LANGUAGE_AGENT_URL),
            ("conversation", CONVERSATION_AGENT_URL),
            ("fusion", FUSION_AGENT_URL),
        ]:
            try:
                resp = await client.get(f"{url}/health")
                agent_status[name] = "ok" if resp.status_code == 200 else "error"
            except Exception:
                agent_status[name] = "unreachable"

    db_ok = False
    try:
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        db_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "service": "api-gateway",
        "version": "0.1.0",
        "database": "ok" if db_ok else "unreachable",
        "agents": agent_status,
    }


# ─────────────────────────────────────────────────────────
# AUTH — Request / Response models
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# AUTH — Endpoints (public)
# ─────────────────────────────────────────────────────────

@app.post("/auth/signup")
async def signup(body: SignupRequest):
    """Register a new user account."""
    email = validate_email(body.email)
    validate_password(body.password)

    pool = await get_pool()

    # Check duplicate email — allow re-signup if previous user is unverified
    existing = await pool.fetchrow(
        "SELECT id, is_verified FROM users WHERE email = $1", email
    )
    if existing:
        if existing["is_verified"]:
            raise HTTPException(409, "Email already registered")
        # Unverified orphan — delete so user can re-register cleanly
        await pool.execute("DELETE FROM users WHERE id = $1", existing["id"])
        logger.info(f"Removed unverified orphan user for {email}")

    password_hash = hash_password(body.password)

    # If email service is configured, require email verification
    if is_email_configured():
        # Use a transaction — if email sending fails, rollback the user creation
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
                    row["id"],
                    token,
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

    # No email configured — auto-verify and return tokens (dev mode)
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


@app.post("/auth/login")
async def login(body: LoginRequest):
    """Authenticate with email and password."""
    email = body.email.strip().lower()

    pool = await get_pool()
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

    # Update last_login_at
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


@app.get("/auth/verify-email")
async def verify_email(token: str = Query(...)):
    """Verify a user's email address using the token from the verification email."""
    pool = await get_pool()

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

    # Mark user as verified and consume the token
    await pool.execute(
        "UPDATE users SET is_verified = true WHERE id = $1",
        row["user_id"],
    )
    await pool.execute(
        "UPDATE email_verifications SET used_at = $1 WHERE id = $2",
        datetime.now(timezone.utc), row["verification_id"],
    )

    return {"success": True, "message": "Email verified successfully. You can now log in."}


@app.post("/auth/resend-verification")
async def resend_verification(body: ResendVerificationRequest):
    """Resend the verification email for an unverified account."""
    email = body.email.strip().lower()
    pool = await get_pool()

    row = await pool.fetchrow(
        "SELECT id, full_name, is_verified FROM users WHERE email = $1",
        email,
    )

    if not row:
        # Don't reveal whether the email exists
        return {"message": "If that email is registered, a verification email has been sent."}

    if row["is_verified"]:
        return {"message": "If that email is registered, a verification email has been sent."}

    # Rate limit: max 3 verification emails in the last hour
    recent_count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM email_verifications
        WHERE user_id = $1 AND created_at > NOW() - INTERVAL '1 hour'
        """,
        row["id"],
    )
    if recent_count >= 3:
        raise HTTPException(
            429,
            "Too many verification emails requested. Please try again later.",
        )

    if not is_email_configured():
        logger.warning("Email not configured — cannot resend verification email")
        raise HTTPException(
            503,
            "Email service is not configured. Please contact support.",
        )

    token = generate_verification_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    await pool.execute(
        """
        INSERT INTO email_verifications (user_id, token, expires_at)
        VALUES ($1, $2, $3)
        """,
        row["id"], token, expires_at,
    )
    await send_verification_email(email, row["full_name"], token)

    return {"message": "Verification email resent."}


@app.post("/auth/refresh")
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


# ─────────────────────────────────────────────────────────
# AUTH — Endpoints (authenticated)
# ─────────────────────────────────────────────────────────

@app.post("/auth/logout")
async def logout(body: LogoutRequest, current_user: dict = Depends(get_current_user)):
    """Invalidate a refresh token."""
    await delete_refresh_token(body.refresh_token)
    return {"success": True}


@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user profile."""
    return current_user


@app.put("/auth/me")
async def update_me(body: UpdateProfileRequest, current_user: dict = Depends(get_current_user)):
    """Update current user profile fields (not email or password)."""
    pool = await get_pool()

    sets = []
    params = []
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


@app.put("/auth/change-password")
async def change_password(body: ChangePasswordRequest, current_user: dict = Depends(get_current_user)):
    """Change the current user's password."""
    validate_password(body.new_password)

    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT password_hash FROM users WHERE id = $1",
        current_user["id"],
    )

    if not row or not verify_password(body.current_password, row["password_hash"]):
        raise HTTPException(401, "Current password is incorrect")

    new_hash = hash_password(body.new_password)
    await pool.execute(
        "UPDATE users SET password_hash = $1, updated_at = $2 WHERE id = $3",
        new_hash, datetime.now(timezone.utc), current_user["id"],
    )

    return {"success": True}


@app.post("/auth/forgot-password")
async def forgot_password(body: ForgotPasswordRequest):
    """
    Send a password reset email.
    Always returns 200 to prevent email enumeration.
    """
    from email_service import send_password_reset_email

    email = body.email.strip().lower()
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, full_name FROM users WHERE email = $1 AND is_active = true",
        email,
    )

    if row:
        token = generate_verification_token()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        # Invalidate any existing unused tokens for this user
        await pool.execute(
            "DELETE FROM password_reset_tokens WHERE user_id = $1 AND used_at IS NULL",
            row["id"],
        )
        await pool.execute(
            """
            INSERT INTO password_reset_tokens (user_id, token, expires_at)
            VALUES ($1, $2, $3)
            """,
            row["id"], token, expires_at,
        )

        asyncio.create_task(
            send_password_reset_email(email, row["full_name"] or "there", token)
        )

    return {"message": "If that email is registered you will receive a reset link shortly."}


@app.post("/auth/reset-password")
async def reset_password(body: ResetPasswordRequest):
    """Reset password using a valid (unexpired, unused) reset token."""
    validate_password(body.new_password)

    pool = await get_pool()
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


# ─────────────────────────────────────────────────────────
# POST /quick-transcribe — in-memory transcription, no session created
# ─────────────────────────────────────────────────────────

@app.post("/quick-transcribe")
async def quick_transcribe_endpoint(
    file: UploadFile = File(...),
    config: str = Form(default="{}"),
    _current_user: dict = Depends(require_role("member")),
):
    """
    Lightweight transcription endpoint — no session, no DB, no report.
    Saves file to a temp path, calls Voice Agent /transcribe, returns
    segments immediately, then deletes the temp file.

    Returns:
        { segments, speakers, duration_seconds, backend, model }
    """
    filename = file.filename or "upload.wav"
    suffix = Path(filename).suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Parse config
    try:
        config_dict = json.loads(config) if config and config.strip() else {}
    except json.JSONDecodeError:
        config_dict = {}

    transcription_config = config_dict.get("transcription") or {}
    analysis_config = config_dict.get("analysis") or {}

    # Save to a temp file (same dir as sessions, but prefixed qt_ for easy cleanup)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOAD_DIR / f"qt_{uuid.uuid4()}{suffix}"

    MAX_FILE_SIZE = 300 * 1024 * 1024  # 300 MB
    file_size = 0
    try:
        with open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large. Maximum 300 MB.")
                f.write(chunk)

        logger.info(
            f"[quick-transcribe] {filename} ({file_size / 1024 / 1024:.1f} MB) "
            f"model={transcription_config.get('model_preference')} "
            f"diarize={analysis_config.get('run_diarization', True)}"
        )

        payload: dict[str, Any] = {
            "file_path": str(temp_path.resolve()),
            "session_id": str(uuid.uuid4()),
            "meeting_type": config_dict.get("meeting_type", "sales_call"),
        }
        if config_dict.get("num_speakers"):
            payload["num_speakers"] = config_dict["num_speakers"]
        if transcription_config:
            payload["transcription_config"] = transcription_config
        if analysis_config:
            payload["analysis_config"] = analysis_config

        async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
            resp = await client.post(f"{VOICE_AGENT_URL}/transcribe", json=payload)
            resp.raise_for_status()
            result = resp.json()

        return {
            "segments": result.get("segments", []),
            "speakers": result.get("speakers", []),
            "duration_seconds": result.get("duration_seconds", 0),
            "backend": result.get("backend", "unknown"),
            "model": result.get("model", "unknown"),
        }

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


# POST /sessions — Upload + full pipeline
# ─────────────────────────────────────────────────────────



@app.post("/sessions")
async def create_session_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(default=""),
    meeting_type: str = Form(default="sales_call"),
    num_speakers: Optional[int] = Form(default=None),
    config: str = Form(default="{}"),
    current_user: dict = Depends(require_role("member")),
):
    """
    Upload an audio file and start the analysis pipeline in the background.
    Returns immediately with session_id + status: "processing".
    The dashboard polls GET /sessions/{id} to check progress.
    """
    # Validate file type
    filename = file.filename or "upload.wav"
    suffix = Path(filename).suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
    if suffix not in allowed:
        raise HTTPException(
            400,
            f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}",
        )

    # ── Step 1: Save file ──
    session_id = str(uuid.uuid4())
    file_name = f"{session_id}{suffix}"
    file_path = UPLOAD_DIR / file_name

    MAX_FILE_SIZE = 300 * 1024 * 1024  # 300 MB

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_size = 0
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                f.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB.")
            f.write(chunk)

    file_size_mb = file_size / (1024 * 1024)
    logger.info(f"[{session_id}] Uploaded {filename} ({file_size_mb:.1f} MB)")

    if not title:
        title = Path(filename).stem

    # Parse the optional JSON config field from the settings panel
    try:
        config_dict = json.loads(config) if config and config.strip() else {}
    except json.JSONDecodeError:
        config_dict = {}
    transcription_config = config_dict.get("transcription", {})
    analysis_config = config_dict.get("analysis", {})

    # Override meeting_type from config if present (settings panel sends it there)
    if not meeting_type or meeting_type == "sales_call":
        meeting_type = config_dict.get("meeting_type", meeting_type)

    # ── Step 2: Create session in DB ──
    try:
        session = await create_session(
            title=title,
            session_type="recording",
            meeting_type=meeting_type,
            media_url=str(file_path.resolve()),
            user_id=current_user["id"],
            upload_config=config_dict,
        )
        session_id = str(session["id"])
        await update_session_status(session_id, "processing")
        logger.info(f"[{session_id}] Session created in DB")
    except Exception as e:
        logger.warning(f"[{session_id}] DB create failed (continuing without DB): {e}")

    # ── Step 3: Launch pipeline in background ──
    background_tasks.add_task(
        _run_pipeline,
        session_id=session_id,
        file_path=file_path,
        title=title,
        meeting_type=meeting_type,
        num_speakers=num_speakers,
        transcription_config=transcription_config,
        analysis_config=analysis_config,
    )

    return {
        "session_id": session_id,
        "status": "processing",
        "title": title,
        "meeting_type": meeting_type,
    }


async def _run_pipeline(
    session_id: str,
    file_path: Path,
    title: str,
    meeting_type: str,
    num_speakers: Optional[int] = None,
    transcription_config: Optional[dict] = None,
    analysis_config: Optional[dict] = None,
):
    """
    Run the full analysis pipeline in the background.
    Voice → Language → Conversation → Fusion → Persist → Knowledge Store.
    Updates session status in DB as it progresses.
    """
    transcription_config = transcription_config or {}
    analysis_config = analysis_config or {}
    logger.info(
        f"[{session_id}] Pipeline starting (background) "
        f"meeting_type={meeting_type} sensitivity={analysis_config.get('sensitivity', 0.5)}"
    )

    run_behavioural = analysis_config.get("run_behavioural", True)
    run_sentiment = analysis_config.get("run_sentiment", False)
    run_entity_extraction = analysis_config.get("run_entity_extraction", True)

    # ── Step 3: Voice Agent (with retry — this is the critical agent) ──
    _set_step(session_id, "transcribing")
    voice_result = None
    try:
        voice_result = await _call_with_retry(
            lambda: _call_voice_agent(
                session_id, str(file_path.resolve()),
                num_speakers=num_speakers,
                meeting_type=meeting_type,
                transcription_config=transcription_config,
                analysis_config=analysis_config,
            )
        )
        _set_step(session_id, "diarization")
        logger.info(
            f"[{session_id}] Transcription: "
            f"{voice_result.get('duration_seconds', 0):.0f}s, "
            f"{len(voice_result.get('speakers', []))} speakers"
        )
    except Exception as e:
        logger.error(f"[{session_id}] Voice Agent transcription failed: {e}")
        await _try_update_status(session_id, "failed")
        return  # Background task — cannot raise HTTPException

    if not isinstance(voice_result, dict):
        logger.error(f"[{session_id}] Voice Agent returned unexpected type: {type(voice_result)}")
        voice_result = {}

    duration_seconds = voice_result.get("duration_seconds", 0)
    voice_signals = voice_result.get("signals", [])
    voice_speakers = voice_result.get("speakers", [])
    voice_summary = voice_result.get("summary", {})
    speaker_count = len(voice_speakers)

    # Persist speakers (blocking — required for foreign key on signals)
    speaker_map = {}
    try:
        speaker_map = await upsert_speakers(session_id, voice_speakers)
    except Exception as e:
        logger.warning(f"[{session_id}] Speaker upsert failed: {e}")

    # Persist voice signals
    try:
        count = await insert_signals(session_id, voice_signals, speaker_map)
        logger.info(f"[{session_id}] Persisted {count} voice signals")
    except Exception as e:
        logger.warning(f"[{session_id}] Voice signal persist failed: {e}")

    # ── Track agent statuses ──
    agent_status = {
        "voice": "completed",
        "language": "skipped",
        "conversation": "skipped",
        "fusion": "skipped",
    }

    # Build transcript segments from voice result (always needed for persist + entity extraction)
    transcript_segments = _extract_transcript_segments(voice_result)

    if transcript_segments:
        try:
            await insert_transcript_segments(session_id, transcript_segments, speaker_map)
        except Exception as e:
            logger.warning(f"[{session_id}] Transcript persist failed: {e}")

    # ── Step 4: Language Agent (behavioural OR standalone sentiment) ──
    _set_step(session_id, "language")
    language_result = None
    language_signals = []
    language_summary = {}

    if run_behavioural or run_sentiment:
        if transcript_segments:
            try:
                language_result = await _call_language_agent(
                    session_id, transcript_segments, meeting_type,
                )
                language_signals = language_result.get("signals", [])
                language_summary = language_result.get("summary", {})
                agent_status["language"] = "completed"
                logger.info(f"[{session_id}] Language Agent: {len(language_signals)} signals")
            except Exception as e:
                agent_status["language"] = "failed"
                logger.warning(f"[{session_id}] Language Agent failed (continuing): {e}")

            if language_signals:
                try:
                    count = await insert_signals(session_id, language_signals, speaker_map)
                    logger.info(f"[{session_id}] Persisted {count} language signals")
                except Exception as e:
                    logger.warning(f"[{session_id}] Language signal persist failed: {e}")
        else:
            logger.warning(f"[{session_id}] No transcript segments — skipping Language Agent")
    elif run_entity_extraction and transcript_segments:
        # Entity-extraction-only path: call EntityExtractor directly, no language agent HTTP call.
        # Extracts people, companies, topics, objections, commitments — nothing else.
        try:
            from entity_extractor import EntityExtractor
            extractor = EntityExtractor()
            entities = await extractor.extract(transcript_segments, meeting_type)
            language_summary = {"entities": entities}
            logger.info(
                f"[{session_id}] Entity extraction only: "
                f"{len(entities.get('topics', []))} topics, "
                f"{len(entities.get('people', []))} people, "
                f"{len(entities.get('objections', []))} objections"
            )
        except Exception as e:
            logger.warning(f"[{session_id}] Standalone entity extraction failed (non-fatal): {e}")
    else:
        logger.info(f"[{session_id}] Language Agent skipped (behavioural/sentiment/entity all off)")

    # ── Step 4b: Conversation Agent (only when behavioural analysis enabled) ──
    _set_step(session_id, "conversation")
    conversation_result = None
    conversation_signals = []
    conversation_summary = {}

    if run_behavioural:
        if transcript_segments and len(set(seg.get("speaker") for seg in transcript_segments)) >= 2:
            try:
                conversation_result = await _call_conversation_agent(
                    session_id, transcript_segments, meeting_type,
                )
                conversation_signals = conversation_result.get("signals", [])
                conversation_summary = conversation_result.get("summary", {})
                agent_status["conversation"] = "completed"
                logger.info(f"[{session_id}] Conversation Agent: {len(conversation_signals)} signals")
            except Exception as e:
                agent_status["conversation"] = "failed"
                logger.warning(f"[{session_id}] Conversation Agent failed (continuing): {e}")

            if conversation_signals:
                try:
                    count = await insert_signals(session_id, conversation_signals, speaker_map)
                    logger.info(f"[{session_id}] Persisted {count} conversation signals")
                except Exception as e:
                    logger.warning(f"[{session_id}] Conversation signal persist failed: {e}")
        else:
            logger.info(f"[{session_id}] < 2 speakers — skipping Conversation Agent")
    else:
        logger.info(f"[{session_id}] Behavioural analysis disabled — skipping Conversation Agent")

    # ── Step 5: Fusion Agent (only when behavioural analysis enabled) ──
    _set_step(session_id, "fusion")
    fusion_result = None
    fusion_signals = []
    alerts = []
    report = None

    if run_behavioural:
        # Enrich voice_summary with conversation dynamics for Fusion Agent
        enriched_voice_summary = dict(voice_summary)
        if conversation_summary:
            enriched_voice_summary["conversation"] = conversation_summary

        try:
            fusion_result = await _call_fusion_agent(
                session_id=session_id,
                voice_signals=voice_signals,
                language_signals=language_signals,
                voice_summary=enriched_voice_summary,
                language_summary=language_summary,
                meeting_type=meeting_type,
            )
            fusion_signals = fusion_result.get("fusion_signals", [])
            alerts = fusion_result.get("alerts", [])
            report = fusion_result.get("report")
            agent_status["fusion"] = "completed"
            logger.info(
                f"[{session_id}] Fusion Agent: "
                f"{len(fusion_signals)} signals, {len(alerts)} alerts"
            )
        except Exception as e:
            agent_status["fusion"] = "failed"
            logger.warning(f"[{session_id}] Fusion Agent failed (continuing): {e}")
    else:
        logger.info(f"[{session_id}] Behavioural analysis disabled — skipping Fusion Agent")

    # Persist fusion signals
    if fusion_signals:
        try:
            count = await insert_signals(session_id, fusion_signals, speaker_map)
            logger.info(f"[{session_id}] Persisted {count} fusion signals")
        except Exception as e:
            logger.warning(f"[{session_id}] Fusion signal persist failed: {e}")

    # Persist alerts
    if alerts:
        try:
            count = await insert_alerts(session_id, alerts, speaker_map)
            logger.info(f"[{session_id}] Persisted {count} alerts")
        except Exception as e:
            logger.warning(f"[{session_id}] Alert persist failed: {e}")

    # Enrich report with entities and signal graph from fusion summary
    entities = language_summary.get("entities", {})
    fusion_summary = fusion_result.get("summary", {}) if fusion_result else {}
    signal_graph = fusion_summary.get("signal_graph", {})
    key_paths = fusion_summary.get("key_paths", [])

    # ── Step 6: Persist report ──
    _set_step(session_id, "report")
    report_generated = False
    report_content = report or {}
    graph_analytics = fusion_summary.get("graph_analytics", {})

    if entities or signal_graph:
        report_content["entities"] = entities
        report_content["signal_graph"] = signal_graph
        report_content["key_paths"] = key_paths
        report_content["graph_analytics"] = graph_analytics
    if report_content:
        try:
            await save_report(
                session_id=session_id,
                content=report_content,
                narrative=report_content.get("executive_summary", ""),
            )
            report_generated = True
        except Exception as e:
            logger.warning(f"[{session_id}] Report persist failed: {e}")

    # ── Step 6: Mark session complete (or partial if agents failed) ──
    any_failed = any(v == "failed" for v in agent_status.values())
    final_status = "partial" if any_failed else "completed"

    await _try_update_status(
        session_id, final_status,
        duration_ms=int(duration_seconds * 1000),
        speaker_count=speaker_count,
    )

    logger.info(f"[{session_id}] Pipeline complete (status={final_status}, agents={agent_status})")

    # Store knowledge chunks for RAG chat (non-blocking, non-fatal, gated on run_entity_extraction)
    _set_step(session_id, "entity_extraction")
    if run_entity_extraction:
        try:
            from knowledge_store import store_session_knowledge
            _pool = await get_pool()
            await store_session_knowledge(_pool, session_id, {
                "transcript_segments": transcript_segments,
                "signals": voice_signals + language_signals + conversation_signals + fusion_signals,
                "entities": entities,
                "report": report_content or {},
                "graph_analytics": graph_analytics or {},
                "conversation_summary": conversation_summary or {},
            })
        except Exception as e:
            logger.warning(f"[{session_id}] Knowledge store failed (non-fatal): {e}")
    else:
        logger.info(f"[{session_id}] Entity extraction disabled — skipping knowledge store")

    _set_step(session_id, "knowledge_graph")
    # Sync session graph to Neo4j (non-blocking, non-fatal).
    # Reads canonical data from PG and projects it as a graph for hybrid chat
    # and exploration. Pipeline still completes if Neo4j is unreachable.
    if analysis_config.get("run_knowledge_graph", True):
        try:
            from neo4j_sync import sync_session as neo4j_sync_session
            _pool = await get_pool()
            await neo4j_sync_session(_pool, session_id)
        except Exception as e:
            logger.warning(f"[{session_id}] Neo4j sync failed (non-fatal): {e}")
    else:
        logger.info(f"[{session_id}] Neo4j sync skipped (run_knowledge_graph=false)")

    # Cleanup old recordings in background
    _cleanup_old_recordings()

    _pipeline_progress.pop(session_id, None)  # stop progress tracking

    logger.info(
        f"[{session_id}] Pipeline finished: status={final_status}, "
        f"voice={len(voice_signals)}, lang={len(language_signals)}, "
        f"convo={len(conversation_signals)}, fusion={len(fusion_signals)}, "
        f"alerts={len(alerts)}, report={'yes' if report_generated else 'no'}"
    )


# ─────────────────────────────────────────────────────────
# GET /sessions — List
# ─────────────────────────────────────────────────────────



@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions_endpoint(
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    meeting_type: Optional[str] = Query(default=None),
    current_user: dict = Depends(get_current_user),
):
    """List sessions with pagination and optional filters. User-scoped (admin sees all)."""
    user_id = None if current_user["role"] == "admin" else current_user["id"]
    try:
        sessions, total = await list_sessions(
            limit=limit,
            offset=offset,
            status=status,
            meeting_type=meeting_type,
            user_id=user_id,
        )
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(500, "Database query failed")

    return SessionListResponse(
        sessions=sessions,
        total=total,
        limit=limit,
        offset=offset,
    )


# ─────────────────────────────────────────────────────────
# GET /sessions/{id} — Detail
# ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}")
async def get_session_detail(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get session detail including signals, alerts, and unified states."""
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Ownership check: user can only see their own sessions (admin sees all)
    if current_user["role"] != "admin" and str(session.get("user_id", "")) != current_user["id"]:
        raise HTTPException(404, "Session not found")

    # Fetch related data in parallel-ish
    signals = await get_signals(session_id, limit=5000)
    session_alerts = await get_alerts(session_id)
    report = await get_report(session_id)
    transcript = await get_transcript(session_id)

    # Group signals by agent
    signals_by_agent = {}
    for s in signals:
        agent = s.get("agent", "unknown")
        signals_by_agent.setdefault(agent, []).append(s)

    return {
        "session": session,
        "signal_count": len(signals),
        "signals_by_agent": {
            agent: len(sigs) for agent, sigs in signals_by_agent.items()
        },
        "alerts": session_alerts,
        "alert_count": len(session_alerts),
        "has_report": report is not None,
        "transcript_segment_count": len(transcript),
    }


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/signals — Signals with filters
# ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/signals")
async def get_session_signals(
    session_id: str,
    agent: Optional[str] = Query(default=None),
    signal_type: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """Get signals for a session with optional filtering by agent/type."""
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if current_user["role"] != "admin" and str(session.get("user_id", "")) != current_user["id"]:
        raise HTTPException(404, "Session not found")

    signals = await get_signals(
        session_id,
        agent=agent,
        signal_type=signal_type,
        limit=limit,
        offset=offset,
    )

    return {
        "session_id": session_id,
        "signals": signals,
        "count": len(signals),
        "filters": {
            "agent": agent,
            "signal_type": signal_type,
        },
    }


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/report — Report
# ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/report")
async def get_session_report(
    session_id: str,
    regenerate: bool = Query(default=False),
    current_user: dict = Depends(get_current_user),
):
    """
    Get the narrative report for a session.
    If no report exists (or regenerate=True), triggers Fusion Agent to generate one.
    """
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if current_user["role"] != "admin" and str(session.get("user_id", "")) != current_user["id"]:
        raise HTTPException(404, "Session not found")

    # Return existing report unless regenerate requested
    if not regenerate:
        existing = await get_report(session_id)
        if existing:
            return {"session_id": session_id, "report": existing}

    # Generate new report via Fusion Agent
    signals = await get_signals(session_id, limit=5000)

    voice_signals = [s for s in signals if s.get("agent") == "voice"]
    language_signals = [s for s in signals if s.get("agent") == "language"]
    fusion_signals = [s for s in signals if s.get("agent") == "fusion"]

    speakers = list(set(
        s.get("speaker_label") or s.get("speaker_id", "unknown")
        for s in signals if s.get("speaker_label") or s.get("speaker_id")
    ))

    duration_seconds = (session.get("duration_ms") or 0) / 1000.0

    # Build summaries from stored signals
    voice_summary = _build_voice_summary(voice_signals)
    language_summary = _build_language_summary(language_signals)

    try:
        async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
            resp = await client.post(
                f"{FUSION_AGENT_URL}/report",
                json={
                    "session_id": session_id,
                    "duration_seconds": duration_seconds,
                    "speakers": speakers,
                    "voice_summary": voice_summary,
                    "language_summary": language_summary,
                    "fusion_signals": _serialise_signals(fusion_signals),
                    "unified_states": [],
                    "meeting_type": session.get("meeting_type", "sales_call"),
                },
            )
            resp.raise_for_status()
            result = resp.json()
    except Exception as e:
        logger.error(f"[{session_id}] Report generation failed: {e}")
        raise HTTPException(502, f"Report generation failed: {e}")

    report_data = result.get("report", {})

    # Persist
    try:
        saved = await save_report(
            session_id=session_id,
            content=report_data,
            narrative=report_data.get("executive_summary", ""),
        )
        return {"session_id": session_id, "report": saved}
    except Exception as e:
        logger.warning(f"[{session_id}] Report persist failed: {e}")
        return {"session_id": session_id, "report": report_data}


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/progress — Pipeline step (polling)
# ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/progress")
async def get_session_progress(session_id: str, current_user: dict = Depends(get_current_user)):
    """
    Return the current pipeline step for a session being processed.
    Returns pipeline_step=null when the session is no longer in-flight.
    """
    return {"pipeline_step": _pipeline_progress.get(session_id)}


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/transcript — Transcript
# ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get transcript segments for a session."""
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if current_user["role"] != "admin" and str(session.get("user_id", "")) != current_user["id"]:
        raise HTTPException(404, "Session not found")

    segments = await get_transcript(session_id)

    return {
        "session_id": session_id,
        "segments": segments,
        "count": len(segments),
    }


# ─────────────────────────────────────────────────────────
# Agent call helpers
# ─────────────────────────────────────────────────────────

async def _call_with_retry(coro_fn, max_retries=2, backoff=2.0):
    """Call an async function with retry and exponential backoff."""
    last_error: Exception = Exception("no attempts made")
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = backoff * (2 ** attempt)
                logger.warning(f"Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                await asyncio.sleep(wait)
    raise last_error


async def _call_voice_agent(
    session_id: str,
    file_path: str,
    num_speakers: Optional[int] = None,
    meeting_type: str = "sales_call",
    transcription_config: Optional[dict] = None,
    analysis_config: Optional[dict] = None,
) -> dict:
    """Call Voice Agent POST /analyse with file path."""
    payload: dict[str, Any] = {
        "file_path": file_path,
        "session_id": session_id,
        "meeting_type": meeting_type,
    }
    if num_speakers is not None:
        payload["num_speakers"] = num_speakers
    if transcription_config:
        payload["transcription_config"] = transcription_config
    if analysis_config:
        payload["analysis_config"] = analysis_config
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            f"{VOICE_AGENT_URL}/analyse",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


async def _call_language_agent(
    session_id: str,
    segments: list[dict],
    meeting_type: str,
) -> dict:
    """Call Language Agent POST /analyse with transcript segments."""
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            f"{LANGUAGE_AGENT_URL}/analyse",
            json={
                "segments": segments,
                "session_id": session_id,
                "meeting_type": meeting_type,
                "run_intent_classification": True,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _call_conversation_agent(
    session_id: str,
    segments: list[dict],
    meeting_type: str,
) -> dict:
    """Call Conversation Agent POST /analyse with transcript segments."""
    speakers = list(set(seg.get("speaker", "unknown") for seg in segments))
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            f"{CONVERSATION_AGENT_URL}/analyse",
            json={
                "segments": segments,
                "speakers": speakers,
                "content_type": meeting_type,
                "session_id": session_id,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _call_fusion_agent(
    session_id: str,
    voice_signals: list[dict],
    language_signals: list[dict],
    voice_summary: dict,
    language_summary: dict,
    meeting_type: str,
) -> dict:
    """Call Fusion Agent POST /analyse with all signals."""
    # Convert signals to FusionSignalInput format
    def _to_fusion_input(signal: dict, agent: str) -> dict:
        return {
            "agent": signal.get("agent", agent),
            "speaker_id": signal.get("speaker_id", "unknown"),
            "signal_type": signal.get("signal_type", ""),
            "value": signal.get("value"),
            "value_text": signal.get("value_text", ""),
            "confidence": signal.get("confidence", 0.5),
            "window_start_ms": signal.get("window_start_ms", 0),
            "window_end_ms": signal.get("window_end_ms", 0),
            "metadata": signal.get("metadata"),
        }

    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            f"{FUSION_AGENT_URL}/analyse",
            json={
                "voice_signals": [_to_fusion_input(s, "voice") for s in voice_signals],
                "language_signals": [_to_fusion_input(s, "language") for s in language_signals],
                "session_id": session_id,
                "meeting_type": meeting_type,
                "generate_report": True,
                "voice_summary": voice_summary,
                "language_summary": language_summary,
            },
        )
        resp.raise_for_status()
        return resp.json()


def _extract_transcript_segments(voice_result: dict) -> list[dict]:
    """
    Extract transcript segments from Voice Agent result.
    The Voice Agent returns segments in its summary or we reconstruct
    from the signals' metadata.
    """
    # Voice Agent stores transcript info in the summary
    summary = voice_result.get("summary", {})

    # Try to get segments from the response — Voice Agent may include them
    segments = voice_result.get("transcript_segments", [])
    if segments:
        return segments

    # Reconstruct from voice signals that have transcript text in metadata
    signals = voice_result.get("signals", [])
    seen_starts = set()
    reconstructed = []

    for s in signals:
        meta = s.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        text = meta.get("transcript_text", "")
        if not text:
            continue

        start_ms = s.get("window_start_ms", 0)
        speaker = s.get("speaker_id", "unknown")
        dedup_key = (start_ms, speaker)
        if dedup_key in seen_starts:
            continue
        seen_starts.add(dedup_key)

        reconstructed.append({
            "speaker": s.get("speaker_id", "unknown"),
            "start_ms": start_ms,
            "end_ms": s.get("window_end_ms", 0),
            "text": text,
        })

    return reconstructed


async def _try_update_status(
    session_id: str,
    status: str,
    duration_ms: int = None,
    speaker_count: int = None,
):
    """Try to update session status, log warning on failure."""
    try:
        await update_session_status(
            session_id, status,
            duration_ms=duration_ms,
            speaker_count=speaker_count,
        )
    except Exception as e:
        logger.warning(f"[{session_id}] Status update failed: {e}")


def _build_voice_summary(voice_signals: list[dict]) -> dict:
    """Build a voice summary from stored signals for report generation."""
    per_speaker = {}
    stress_peaks = []

    for s in voice_signals:
        speaker = s.get("speaker_label") or s.get("speaker_id", "unknown")
        if speaker not in per_speaker:
            per_speaker[speaker] = {
                "baseline_f0_hz": 0,
                "baseline_rate_wpm": 0,
                "avg_stress": 0,
                "max_stress": 0,
                "total_fillers": 0,
                "pitch_elevation_events": 0,
                "tone_distribution": {},
                "_stress_values": [],
            }

        sig_type = s.get("signal_type", "")
        value = s.get("value")

        if sig_type == "vocal_stress_score" and value is not None:
            per_speaker[speaker]["_stress_values"].append(float(value))
            if float(value) > 0.5:
                stress_peaks.append({
                    "speaker": speaker,
                    "time_ms": s.get("window_start_ms", 0),
                    "stress_score": float(value),
                })
        elif sig_type == "filler_detection":
            per_speaker[speaker]["total_fillers"] += 1
        elif sig_type == "pitch_elevation_flag":
            per_speaker[speaker]["pitch_elevation_events"] += 1
        elif sig_type == "tone_classification":
            tone = s.get("value_text", "neutral")
            dist = per_speaker[speaker]["tone_distribution"]
            dist[tone] = dist.get(tone, 0) + 1

    # Compute averages
    for speaker, data in per_speaker.items():
        stress_vals = data.pop("_stress_values", [])
        if stress_vals:
            data["avg_stress"] = sum(stress_vals) / len(stress_vals)
            data["max_stress"] = max(stress_vals)

    stress_peaks.sort(key=lambda x: x["stress_score"], reverse=True)

    return {"per_speaker": per_speaker, "stress_peaks": stress_peaks[:5]}


def _build_language_summary(language_signals: list[dict]) -> dict:
    """Build a language summary from stored signals for report generation."""
    per_speaker = {}

    for s in language_signals:
        speaker = s.get("speaker_label") or s.get("speaker_id", "unknown")
        if speaker not in per_speaker:
            per_speaker[speaker] = {
                "total_segments": 0,
                "avg_sentiment": 0,
                "min_sentiment": 0,
                "max_sentiment": 0,
                "buying_signal_count": 0,
                "objection_count": 0,
                "avg_power_score": 0.5,
                "intent_distribution": {},
                "_sent_values": [],
                "_power_values": [],
            }

        sig_type = s.get("signal_type", "")
        value = s.get("value")

        if sig_type == "sentiment_score" and value is not None:
            per_speaker[speaker]["_sent_values"].append(float(value))
            per_speaker[speaker]["total_segments"] += 1
        elif sig_type == "buying_signal":
            per_speaker[speaker]["buying_signal_count"] += 1
        elif sig_type == "objection_signal":
            per_speaker[speaker]["objection_count"] += 1
        elif sig_type == "power_language_score" and value is not None:
            per_speaker[speaker]["_power_values"].append(float(value))
        elif sig_type == "intent_classification":
            intent = s.get("value_text", "UNKNOWN")
            dist = per_speaker[speaker]["intent_distribution"]
            dist[intent] = dist.get(intent, 0) + 1

    for speaker, data in per_speaker.items():
        sent_vals = data.pop("_sent_values", [])
        if sent_vals:
            data["avg_sentiment"] = sum(sent_vals) / len(sent_vals)
            data["min_sentiment"] = min(sent_vals)
            data["max_sentiment"] = max(sent_vals)

        power_vals = data.pop("_power_values", [])
        if power_vals:
            data["avg_power_score"] = sum(power_vals) / len(power_vals)

    return {"per_speaker": per_speaker}


def _serialise_signals(signals: list[dict]) -> list[dict]:
    """Ensure signals are JSON-serialisable (handle DB types)."""
    clean = []
    for s in signals:
        entry = {}
        for k, v in s.items():
            if hasattr(v, "isoformat"):
                entry[k] = v.isoformat()
            elif isinstance(v, memoryview):
                entry[k] = bytes(v).decode("utf-8", errors="replace")
            else:
                entry[k] = v
        clean.append(entry)
    return clean


# ═══════════════════════════════════════════════════════════════
# SESSION CHAT (RAG)
# ═══════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/sessions/{session_id}/chat")
async def chat_with_session(
    session_id: str,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Ask a question about a session's analysis using RAG.
    Embeds the question, searches pgvector for relevant chunks,
    feeds context to LLM, returns the answer with sources.
    """
    pool = await get_pool()

    # Ownership check
    session = await get_session(session_id, current_user.get("org_id"))
    if not session:
        raise HTTPException(404, "Session not found")
    if current_user["role"] != "admin" and str(session["user_id"]) != current_user["id"]:
        raise HTTPException(403, "Access denied")

    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    # Step 1: Embed the question
    from shared.utils.llm_client import get_embedding
    question_embedding = await get_embedding(question)
    if not question_embedding:
        raise HTTPException(500, "Embedding generation failed — check LLM configuration")

    # Step 2: Search pgvector for relevant chunks.
    # Wrap in try/except: if existing rows were indexed under a different
    # provider (e.g. OpenAI 1536 vs Ollama 768), pgvector raises a dimension
    # mismatch. Degrade to empty rows so the chat still answers from any
    # other context (e.g. Neo4j) instead of 500'ing.
    try:
        rows = await pool.fetch(
            """
            SELECT text, chunk_type, metadata,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM knowledge_chunks
            WHERE session_id = $2
            ORDER BY embedding <=> $1::vector
            LIMIT 12
            """,
            "[" + ",".join(str(v) for v in question_embedding) + "]",
            session_id,
        )
    except Exception as e:
        logger.warning(
            f"[{session_id}] pgvector chat search failed (likely embedding-dim "
            f"mismatch from a previous provider): {e}. Continuing with empty rows."
        )
        rows = []

    # Step 2b: Neo4j hybrid graph query.
    # Path A (gpt-4o): picks 1 of 10 pre-built tools → hardcoded Cypher. ~90% of questions.
    # Path B (gpt-5, fallback): generates Cypher when no tool fits. ~10% of questions.
    # Non-fatal: degrades to pgvector-only if Neo4j is unavailable.
    graph_context = ""
    graph_source = None
    try:
        from neo4j_semantic_layer import select_tool, execute_tool, search_graph_context_fallback
        tool_selection = await select_tool(question, session_id)
        tool_name = tool_selection.get("tool", "none")
        if tool_name and tool_name != "none" and tool_name in [
            "get_causal_chain", "get_topic_stress_correlation", "get_speaker_influence",
            "get_unresolved_objections", "get_conversation_arc", "get_signal_decomposition",
            "get_convergent_moments", "get_speaker_summary", "get_signal_timeline",
            "get_entity_network",
        ]:
            # Path A: pre-built tool (gpt-4o selected it, ~$0.002)
            params = tool_selection.get("params", {})
            graph_context = await execute_tool(tool_name, params, session_id)
            graph_source = f"semantic:{tool_name}"
            logger.info(f"[{session_id}] Semantic layer used tool: {tool_name}")
        else:
            # Path B: gpt-5 Cypher fallback (~$0.02, only when no tool fits)
            graph_context = await search_graph_context_fallback(question, session_id)
            if graph_context:
                graph_source = "gpt5_cypher_fallback"
                logger.info(f"[{session_id}] GPT-5 Cypher fallback activated")
    except Exception as e:
        logger.warning(f"[{session_id}] Neo4j graph query failed (non-fatal): {e}")

    # Step 3: Build context from top relevant chunks
    context_parts = []
    sources = []
    for row in rows:
        sim = float(row["similarity"])
        if sim < 0.40:  # text-embedding-3-small (1536d cosine): 0.40 = loosely related
            continue
        context_parts.append(f"[{row['chunk_type']}] {row['text']}")
        sources.append({
            "type": row["chunk_type"],
            "text": row["text"][:200],
            "similarity": round(sim, 3),
        })

    if not context_parts and not graph_context:
        return {
            "answer": "I couldn't find relevant analysis data for that question. Try rephrasing or ask about specific speakers, signals, or moments.",
            "sources": [],
            "chunks_searched": len(rows),
        }

    text_context = "\n".join(context_parts[:8])
    if graph_context:
        context = f"{text_context}\n\n{graph_context}" if text_context else graph_context
        sources.append({"type": "knowledge_graph", "text": graph_context[:200], "similarity": 1.0})
    else:
        context = text_context

    # Step 4: Generate answer with LLM
    from shared.utils.llm_client import acomplete

    system_prompt = (
        "You are NEXUS, a behavioural analysis assistant. Answer questions about "
        "meeting/call analysis using the provided context.\n\n"
        "Rules:\n"
        "- Use BOTH text context and graph context when available. "
        "Graph context provides causal chains and relationships that text context does not.\n"
        "- Only answer based on the provided context. If neither source contains the answer, say so.\n"
        "- Reference specific timestamps (convert ms to mm:ss), speaker names, and signal values.\n"
        "- Be concise but thorough.\n"
        "- Frame behavioural observations as 'indicators suggest' not 'they were definitely'.\n"
        "- When explaining causal chains from graph context, present as a sequence: 'A led to B which triggered C'.\n"
        "- If graph context shows speaker influence patterns, describe them as correlations, not causation.\n"
        "- Never claim to detect deception — only note incongruence between modalities."
    )

    user_prompt = f"Context from session analysis:\n{context}\n\nQuestion: {question}"

    # Include last few history messages for multi-turn context
    if body.history:
        history_text = "\n".join(
            f"{m.get('role', 'user').title()}: {m.get('content', '')}"
            for m in body.history[-4:]
        )
        user_prompt = f"Previous conversation:\n{history_text}\n\n{user_prompt}"

    try:
        answer = await acomplete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=600,
            model="gpt-4o",
        )
    except Exception as e:
        logger.error(f"Chat LLM call failed: {e}")
        raise HTTPException(502, f"LLM generation failed: {e}")

    # Persist both messages to DB
    user_id = current_user["id"]
    top_sources = sources[:5]
    try:
        await pool.execute(
            "INSERT INTO chat_messages (session_id, user_id, role, content) VALUES ($1, $2, 'user', $3)",
            session_id, user_id, question,
        )
        await pool.execute(
            "INSERT INTO chat_messages (session_id, user_id, role, content, sources) VALUES ($1, $2, 'assistant', $3, $4::jsonb)",
            session_id, user_id, answer, json.dumps(top_sources),
        )
    except Exception as e:
        logger.warning(f"Failed to persist chat messages: {e}")

    return {
        "answer": answer,
        "sources": top_sources,
        "chunks_searched": len(rows),
        "graph_source": graph_source,
    }


@app.get("/sessions/{session_id}/chat")
async def get_chat_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get persisted chat history for a session."""
    pool = await get_pool()

    session = await get_session(session_id, current_user.get("org_id", ""))
    if not session:
        raise HTTPException(404, "Session not found")
    if current_user["role"] != "admin" and str(session["user_id"]) != current_user["id"]:
        raise HTTPException(403, "Access denied")

    rows = await pool.fetch(
        """
        SELECT role, content, sources, created_at
        FROM chat_messages
        WHERE session_id = $1 AND user_id = $2
        ORDER BY created_at ASC
        """,
        session_id, current_user["id"],
    )

    messages = []
    for row in rows:
        msg = {"role": row["role"], "content": row["content"]}
        if row["sources"]:
            src = row["sources"]
            msg["sources"] = json.loads(src) if isinstance(src, str) else src
        messages.append(msg)

    return {"messages": messages}
