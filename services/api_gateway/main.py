# services/api_gateway/main.py
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
  GET  /sessions/{id}/video-signals → Video + fusion signals for playback overlay (auth)
  GET  /sessions/{id}/video    → Stream session media file; accepts ?token= for <video> elements (auth)
  GET  /sessions/{id}/video/annotated → Stream landmark-annotated video; accepts ?token= (auth)
  POST /sessions/{id}/identify-speaker → Manually assign a speaker label to a registry entry (auth)
  GET  /health                  → Health check (public)

  Speaker Registry (cross-session identity):
  GET  /speakers                → List all registered speakers with aggregate stats (auth)
  GET  /speakers/{id}           → Speaker profile with cross-session trend data (auth)
  PUT  /speakers/{id}           → Update speaker display_name, role, company, notes (auth)
  POST /speakers/{id}/merge     → Merge two speaker identities (auth)

  Team Dashboard:
  GET  /team                    → All speakers with aggregate metrics for last N days (auth)
  GET  /team/compare            → Side-by-side metric comparison for two speakers (auth)

  Global Chat:
  POST /chat/global             → Cross-session LLM chat with team + speaker context (auth)
"""
import os
import sys
import uuid
import time
import json
import re
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import redis.asyncio as _aioredis

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import FileResponse
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
from shared.redis_layer import (
    AgentStatusRecord,
    EventRecord,
    RedisEventStore,
    RedisRepository,
    SessionStateRecord,
)

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

_GATEWAY_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Module-level connection pool — reused across all _drain_pending_signals calls.
# redis.asyncio pools are async-safe; no race risk since asyncio is single-threaded
# and there is no await between the None-check and the assignment in _get_redis_pool.
_redis_pool: Optional[_aioredis.Redis] = None


async def _get_redis_pool() -> _aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = _aioredis.from_url(
            _GATEWAY_REDIS_URL,
            decode_responses=True,
            max_connections=10,
        )
    return _redis_pool


async def _drain_pending_signals(session_id: str, agent: str) -> list[dict]:
    """
    Pull all signal batches from Redis that the agent wrote during processing.
    Atomically reads the full list and deletes the key so signals aren't double-consumed.
    Returns [] if Redis is unreachable or the key is empty.
    """
    try:
        r = await _get_redis_pool()
        key = f"nexus:pending:{session_id}:{agent}"
        batches = await r.lrange(key, 0, -1)  # type: ignore[misc]
        if batches:
            await r.delete(key)
            signals: list[dict] = []
            for batch_json in batches:
                signals.extend(json.loads(batch_json))
            return signals
    except Exception as exc:
        logger.warning(f"[{session_id}] Redis drain ({agent}) failed (non-fatal): {exc}")
    return []


redis_repo = RedisRepository()
event_store = RedisEventStore()


async def _read_agent_signals(session_id: str, agent: str, count: int = 10_000) -> list[dict]:
    """Read canonical stream signals with legacy pending-list fallback."""
    try:
        signals = await redis_repo.read_latest_signals(session_id, agent, count=count)
        if signals:
            return signals
        return await redis_repo.drain_legacy_pending_signals(session_id, agent)
    except Exception as exc:
        logger.warning(f"[{session_id}] Redis read ({agent}) failed (non-fatal): {exc}")
        return []


# ── Agent URLs (configurable via environment) ──
VOICE_AGENT_URL = os.getenv("VOICE_AGENT_URL", "http://localhost:8002")
LANGUAGE_AGENT_URL = os.getenv("LANGUAGE_AGENT_URL", "http://localhost:8003")
CONVERSATION_AGENT_URL = os.getenv("CONVERSATION_AGENT_URL", "http://localhost:8011")
FUSION_AGENT_URL = os.getenv("FUSION_AGENT_URL", "http://localhost:8004")
VIDEO_AGENT_URL  = os.getenv("VIDEO_AGENT_URL",  "http://localhost:8012")

# ── Upload directory ──
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/recordings"))
RECORDING_RETENTION_DAYS = int(os.getenv("RECORDING_RETENTION_DAYS", "3"))

# ── Landmark overlay directory (shared with video-agent via volume mount) ──
OVERLAY_DIR = Path(os.getenv("OVERLAY_DIR", "data/overlays"))

# ── HTTP client timeout (Voice Agent with Whisper can be slow) ──
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "1800"))        # 30 minutes
# Video annotation runs frame-by-frame (CPU) and can take 45-90 min for long recordings.
VIDEO_AGENT_TIMEOUT = float(os.getenv("VIDEO_AGENT_TIMEOUT", "10800"))  # 3 hours

# ── CORS origins (comma-separated list) ──
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3006,http://localhost:5173").split(",")

# ── In-memory pipeline step tracker ──
# Tracks the active pipeline step per session so the frontend can poll real progress.
# Process-local (no DB write needed — sessions only process once per gateway instance).
_pipeline_progress: dict[str, str] = {}

def _set_step(session_id: str, step: str) -> None:
    _pipeline_progress[session_id] = step


async def _set_step_state(session_id: str, step: str, status: str = "running", error: str = "") -> None:
    _set_step(session_id, step)
    await redis_repo.set_session_state(
        session_id,
        SessionStateRecord(status=status, current_step=step, error=error or ""),
    )


async def _set_agent_status(
    session_id: str,
    agent: str,
    status: str,
    signal_count: int = 0,
    error: str = "",
) -> None:
    await redis_repo.set_agent_status(
        session_id,
        agent,
        AgentStatusRecord(
            status=status,
            signal_count=signal_count,
            summary_key=f"summary:{agent}",
            error=error or "",
        ),
    )


async def _append_agent_event(session_id: str, agent: str, event_type: str, payload: Optional[dict[str, Any]] = None) -> None:
    await event_store.append(
        session_id,
        EventRecord(session_id=session_id, agent=agent, event_type=event_type, payload=payload or {}),
    )

# ── Chunked upload config ──
CHUNK_UPLOAD_DIR = Path(os.getenv("CHUNK_UPLOAD_DIR", "data/chunks"))
CHUNK_SIZE_BYTES  = 10 * 1024 * 1024          # 10 MB per chunk
MAX_UPLOAD_SIZE   = 2 * 1024 * 1024 * 1024    # 2 GB total file limit
UPLOAD_EXPIRY_HOURS = 24                       # abandon incomplete uploads after 24 h

# upload_id → { user_id, filename, file_size, chunk_size, total_chunks,
#               received_chunks: set, created_at, meeting_type, title, config }
_upload_sessions: dict[str, dict] = {}

app = FastAPI(
    title="NEXUS API Gateway",
    description="Central API for the NEXUS multi-agent behavioural analysis system",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
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
        t0 = time.time()
        response = await call_next(request)
        elapsed_ms = (time.time() - t0) * 1000
        qs = f"?{request.url.query}" if request.url.query else ""
        logger.info(
            f"{request.method} {request.url.path}{qs} → {response.status_code} ({elapsed_ms:.0f}ms)"
        )
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


def _cleanup_expired_uploads():
    """Delete incomplete chunked upload sessions older than UPLOAD_EXPIRY_HOURS."""
    import shutil
    cutoff = time.time() - (UPLOAD_EXPIRY_HOURS * 3600)
    expired = [uid for uid, s in _upload_sessions.items() if s["created_at"] < cutoff]
    for uid in expired:
        _upload_sessions.pop(uid, None)
        chunk_dir = CHUNK_UPLOAD_DIR / uid
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired chunked upload session(s)")




@app.on_event("startup")
async def startup():
    logger.info("Starting NEXUS API Gateway...")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_recordings()
    _cleanup_expired_uploads()

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

    try:
        from database import _ensure_speaker_registry_tables
        _pool = await get_pool()
        await _ensure_speaker_registry_tables(_pool)
    except Exception as e:
        logger.warning(f"Speaker registry table init failed (non-fatal): {e}")

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

class ChunkedUploadInitRequest(BaseModel):
    filename:     str
    file_size:    int
    chunk_size:   int           = CHUNK_SIZE_BYTES
    meeting_type: str           = "sales_call"
    title:        str           = ""
    config:       str           = "{}"


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


# ─────────────────────────────────────────────────────────
# Chunked upload endpoints  (/uploads/*)
# ─────────────────────────────────────────────────────────

@app.post("/uploads/init")
async def init_chunked_upload(
    body: ChunkedUploadInitRequest,
    current_user: dict = Depends(require_role("member")),
):
    """
    Step 1 of 3 — initialise a chunked upload session.
    Returns upload_id + total_chunks so the client knows how many pieces to send.
    """
    import math
    suffix = Path(body.filename).suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}")
    if body.file_size <= 0:
        raise HTTPException(400, "Invalid file size")
    if body.file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Maximum {MAX_UPLOAD_SIZE // (1024**3)} GB.")

    upload_id   = str(uuid.uuid4())
    chunk_size  = max(1, body.chunk_size)
    total_chunks = math.ceil(body.file_size / chunk_size)

    (CHUNK_UPLOAD_DIR / upload_id).mkdir(parents=True, exist_ok=True)

    _upload_sessions[upload_id] = {
        "upload_id":       upload_id,
        "user_id":         current_user["id"],
        "filename":        body.filename,
        "file_size":       body.file_size,
        "chunk_size":      chunk_size,
        "total_chunks":    total_chunks,
        "received_chunks": set(),
        "created_at":      time.time(),
        "meeting_type":    body.meeting_type,
        "title":           body.title or Path(body.filename).stem,
        "config":          body.config,
    }
    logger.info(
        f"[upload:{upload_id}] init: {body.filename} "
        f"({body.file_size / 1024 / 1024:.1f} MB, {total_chunks} chunks)"
    )
    return {"upload_id": upload_id, "chunk_size": chunk_size, "total_chunks": total_chunks}


@app.post("/uploads/{upload_id}/chunk")
async def upload_chunk(
    upload_id:    str,
    chunk_number: int        = Form(...),
    chunk:        UploadFile = File(...),
    current_user: dict       = Depends(require_role("member")),
):
    """
    Step 2 of 3 — upload one chunk. Chunks may arrive out of order.
    Retry a chunk at any time; re-sending an already-received chunk is safe.
    """
    session = _upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(404, "Upload session not found or expired")
    if session["user_id"] != current_user["id"]:
        raise HTTPException(403, "Not your upload session")
    if chunk_number < 0 or chunk_number >= session["total_chunks"]:
        raise HTTPException(400, f"Invalid chunk_number {chunk_number} (total={session['total_chunks']})")

    data = await chunk.read()

    # Every chunk except the last must be exactly chunk_size bytes
    if chunk_number < session["total_chunks"] - 1:
        if len(data) != session["chunk_size"]:
            raise HTTPException(
                400,
                f"Chunk {chunk_number} size mismatch: got {len(data)}, expected {session['chunk_size']}"
            )

    chunk_path = CHUNK_UPLOAD_DIR / upload_id / f"chunk_{chunk_number:06d}"
    with open(chunk_path, "wb") as f:
        f.write(data)

    session["received_chunks"].add(chunk_number)
    received = len(session["received_chunks"])
    logger.info(f"[upload:{upload_id}] chunk {chunk_number}/{session['total_chunks']-1} ({received}/{session['total_chunks']} total)")

    return {
        "chunk_number": chunk_number,
        "received":     received,
        "total":        session["total_chunks"],
        "complete":     received == session["total_chunks"],
    }


@app.post("/uploads/{upload_id}/complete")
async def complete_chunked_upload(
    upload_id:        str,
    background_tasks: BackgroundTasks,
    current_user:     dict = Depends(require_role("member")),
):
    """
    Step 3 of 3 — verify all chunks are present, assemble the file,
    create the DB session, and launch the analysis pipeline.
    Identical pipeline call to POST /sessions.
    """
    import shutil
    session = _upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(404, "Upload session not found or expired")
    if session["user_id"] != current_user["id"]:
        raise HTTPException(403, "Not your upload session")

    missing = set(range(session["total_chunks"])) - session["received_chunks"]
    if missing:
        raise HTTPException(400, f"Missing {len(missing)} chunk(s): {sorted(missing)[:10]}")

    # Assemble chunks → final file
    suffix      = Path(session["filename"]).suffix.lower()
    session_id  = str(uuid.uuid4())
    file_name   = f"{session_id}{suffix}"
    final_path  = UPLOAD_DIR / file_name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    chunk_dir   = CHUNK_UPLOAD_DIR / upload_id

    try:
        with open(final_path, "wb") as out:
            for i in range(session["total_chunks"]):
                chunk_file_path = chunk_dir / f"chunk_{i:06d}"
                with open(chunk_file_path, "rb") as cf:
                    while block := cf.read(1024 * 1024):
                        out.write(block)
        assembled_size = final_path.stat().st_size
        logger.info(
            f"[upload:{upload_id}] assembled {session['total_chunks']} chunks "
            f"→ {file_name} ({assembled_size / 1024 / 1024:.1f} MB)"
        )
    except Exception as exc:
        final_path.unlink(missing_ok=True)
        logger.error(f"[upload:{upload_id}] assembly failed: {exc}")
        raise HTTPException(500, f"File assembly failed: {exc}")
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        _upload_sessions.pop(upload_id, None)

    # Parse config — same logic as POST /sessions
    try:
        config_dict = json.loads(session.get("config", "{}"))
    except json.JSONDecodeError:
        config_dict = {}
    transcription_config = config_dict.get("transcription", {})
    analysis_config      = config_dict.get("analysis", {})
    meeting_type  = session.get("meeting_type") or config_dict.get("meeting_type", "sales_call")
    title         = session.get("title") or Path(session["filename"]).stem
    num_speakers  = config_dict.get("num_speakers") or None

    # Create DB session
    try:
        _is_lightweight = not analysis_config.get("run_behavioural", True)
        db_session = await create_session(
            title=title,
            session_type="lightweight" if _is_lightweight else "recording",
            meeting_type=meeting_type,
            media_url=str(final_path.resolve()),
            user_id=current_user["id"],
            upload_config=config_dict,
        )
        session_id = str(db_session["id"])
        await update_session_status(session_id, "processing")
        logger.info(f"[{session_id}] DB session created (chunked upload)")
    except Exception as exc:
        logger.warning(f"[{session_id}] DB create failed (continuing): {exc}")

    # Launch pipeline
    _video_path = final_path if suffix in {".mp4", ".webm"} else None
    background_tasks.add_task(
        _run_pipeline,
        session_id=session_id,
        file_path=final_path,
        title=title,
        meeting_type=meeting_type,
        transcription_config=transcription_config,
        analysis_config=analysis_config,
        video_path=_video_path,
        user_email=current_user.get("email", ""),
        num_speakers=num_speakers,
        org_id=current_user.get("org_id", DEV_ORG_ID),
    )

    return {
        "session_id":  session_id,
        "status":      "processing",
        "title":       title,
        "meeting_type": meeting_type,
        "file_size":   assembled_size,
    }


@app.get("/uploads/{upload_id}/status")
async def get_upload_status(
    upload_id:    str,
    current_user: dict = Depends(require_role("member")),
):
    """
    Resume support — returns which chunks have been received so far.
    Frontend checks this after a reconnect and uploads only missing chunks.
    """
    session = _upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(404, "Upload session not found or expired")
    if session["user_id"] != current_user["id"]:
        raise HTTPException(403, "Not your upload session")
    received = len(session["received_chunks"])
    return {
        "upload_id":       upload_id,
        "filename":        session["filename"],
        "total_chunks":    session["total_chunks"],
        "received_chunks": sorted(session["received_chunks"]),
        "received_count":  received,
        "progress_pct":    round(received / session["total_chunks"] * 100, 1),
        "complete":        received == session["total_chunks"],
    }


# POST /sessions — Upload + full pipeline
# ─────────────────────────────────────────────────────────

@app.post("/sessions")
async def create_session_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(default=""),
    meeting_type: str = Form(default="sales_call"),
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

    num_speakers = config_dict.get("num_speakers") or None

    # ── Step 2: Create session in DB ──
    try:
        # Lightweight sessions (transcript/diarize/entity only) stored separately
        # so they don't appear in the main Sessions tab
        _is_lightweight = not analysis_config.get("run_behavioural", True)
        session = await create_session(
            title=title,
            session_type="lightweight" if _is_lightweight else "recording",
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
    # Pass the same file as video_path when it contains a video track
    _video_path = file_path if suffix in {".mp4", ".webm"} else None
    background_tasks.add_task(
        _run_pipeline,
        session_id=session_id,
        file_path=file_path,
        title=title,
        meeting_type=meeting_type,
        transcription_config=transcription_config,
        analysis_config=analysis_config,
        video_path=_video_path,
        user_email=current_user.get("email", ""),
        num_speakers=num_speakers,
        org_id=current_user.get("org_id", DEV_ORG_ID),
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
    transcription_config: Optional[dict] = None,
    analysis_config: Optional[dict] = None,
    video_path: Optional[Path] = None,
    user_email: str = "",
    num_speakers: Optional[int] = None,
    org_id: str = DEV_ORG_ID,
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
    await _set_step_state(session_id, "transcribing")
    voice_result = None
    try:
        voice_result = await _call_with_retry(
            lambda: _call_voice_agent(
                session_id, str(file_path.resolve()),
                meeting_type=meeting_type,
                transcription_config=transcription_config,
                analysis_config=analysis_config,
                num_speakers=num_speakers,
            )
        )
        logger.info(
            f"[{session_id}] Transcription: "
            f"{voice_result.get('duration_seconds', 0):.0f}s, "
            f"{len(voice_result.get('speakers', []))} speakers"
        )
    except Exception as e:
        logger.error(f"[{session_id}] Voice Agent transcription failed: {e}")
        await redis_repo.set_session_state(
            session_id,
            SessionStateRecord(status="failed", current_step="transcribing", error=str(e)),
        )
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

    await _persist_agent_signals(session_id, voice_signals, speaker_map, "voice")

    # Speaker registry matching is deferred until after the video agent completes
    # so face embeddings (from video) can be fused with voice embeddings.
    speaker_embeddings = voice_result.get("speaker_embeddings") or {}

    # ── Track agent statuses ──
    agent_status = {
        "voice": "completed",
        "language": "skipped",
        "conversation": "skipped",
        "video": "skipped",
        "fusion": "skipped",
    }

    # Build transcript segments from voice result (always needed for persist + entity extraction)
    transcript_segments = _extract_transcript_segments(voice_result)

    if transcript_segments:
        try:
            await insert_transcript_segments(session_id, transcript_segments, speaker_map)
        except Exception as e:
            logger.warning(f"[{session_id}] Transcript persist failed: {e}")

    # ── Steps 4 / 4b / 4c: Language + Conversation + Video (parallel) ──
    # All three depend only on voice output — no inter-dependency.
    # Video signals are awaited before Fusion so the report is fully multimodal.
    await _set_step_state(session_id, "language")

    video_signals: list[dict] = []
    run_video = video_path is not None and run_behavioural
    diar_segments_for_video: list[dict] = []
    if run_video:
        diar_segments_for_video = [
            {"speaker": seg.get("speaker", "unknown"),
             "start_ms": int(seg.get("start_ms", 0)),
             "end_ms":   int(seg.get("end_ms", 0))}
            for seg in transcript_segments
            if seg.get("start_ms") is not None
        ]

    language_result = None
    language_signals = []
    language_summary = {}
    conversation_result = None
    conversation_signals = []
    conversation_summary = {}

    run_conversation = (
        run_behavioural
        and bool(transcript_segments)
        and len(set(seg.get("speaker") for seg in transcript_segments)) >= 2
    )

    async def _run_language() -> dict:
        try:
            if run_behavioural or run_sentiment:
                if not transcript_segments:
                    logger.warning(f"[{session_id}] No transcript segments — skipping Language Agent")
                    return {}
                result = await _call_language_agent(session_id, transcript_segments, meeting_type)
                agent_status["language"] = "completed"
                return result

            if run_entity_extraction and transcript_segments:
                assemblyai_raw = voice_result.get("assemblyai_entities") if voice_result else None
                if assemblyai_raw is not None:
                    entities = _format_assemblyai_entities(assemblyai_raw)
                    logger.info(
                        f"[{session_id}] Entity extraction (AssemblyAI): "
                        f"{len(entities.get('people', []))} people, "
                        f"{len(entities.get('organizations', []))} orgs, "
                        f"{len(entities.get('topics', []))} topics"
                    )
                    return {"summary": {"entities": entities}}
                try:
                    from entity_extractor import EntityExtractor
                    extractor = EntityExtractor()
                    entities = await extractor.extract(transcript_segments, meeting_type)
                    logger.info(
                        f"[{session_id}] Entity extraction (NEXUS fallback): "
                        f"{len(entities.get('topics', []))} topics, "
                        f"{len(entities.get('people', []))} people"
                    )
                    return {"summary": {"entities": entities}}
                except Exception as e:
                    logger.warning(f"[{session_id}] Standalone entity extraction failed (non-fatal): {e}")
                    return {}

            logger.info(f"[{session_id}] Language Agent skipped (behavioural/sentiment/entity all off)")
            return {}
        except Exception as e:
            agent_status["language"] = "failed"
            logger.warning(f"[{session_id}] Language Agent failed (continuing): {e}")
            return {}

    async def _run_conversation() -> dict:
        try:
            if not run_conversation:
                if not run_behavioural:
                    logger.info(f"[{session_id}] Behavioural analysis disabled — skipping Conversation Agent")
                else:
                    logger.info(f"[{session_id}] < 2 speakers — skipping Conversation Agent")
                return {}
            result = await _call_conversation_agent(session_id, transcript_segments, meeting_type)
            agent_status["conversation"] = "completed"
            return result
        except Exception as e:
            agent_status["conversation"] = "failed"
            logger.warning(f"[{session_id}] Conversation Agent failed (continuing): {e}")
            return {}

    async def _run_video() -> tuple[list[dict], dict, dict, dict, str]:
        """Returns (signals, face_embeddings_dict, face_to_speaker, lip_sync_scores, video_job_id)."""
        if not run_video or video_path is None:
            return [], {}, {}, {}, ""
        try:
            await _set_step_state(session_id, "video")
            result, vid_job_id = await _call_video_agent(
                session_id, str(video_path.resolve()),
                diar_segments_for_video, meeting_type, speaker_count,
            )
            sigs      = result.get("signals", [])
            face_embs = result.get("face_embeddings", {})
            f2s       = {int(k): v for k, v in result.get("face_to_speaker", {}).items()}
            lip_scores = result.get("lip_sync_scores", {})
            agent_status["video"] = "completed"
            logger.info(
                f"[{session_id}] Video Agent: {len(sigs)} signals, "
                f"{len(face_embs)} face embeddings, "
                f"{len(f2s)} face-speaker links, "
                f"{len(lip_scores)} lip-sync scores"
            )
            return sigs, face_embs, f2s, lip_scores, vid_job_id
        except Exception as e:
            agent_status["video"] = "failed"
            logger.warning(f"[{session_id}] Video Agent failed (continuing without video): {e}")
            return [], {}, {}, {}, ""

    # Run Language and Video in parallel; wait for Language result first
    lang_outcome, (vid_signals, face_embeddings_from_video, face_to_speaker, lip_sync_scores, video_job_id) = await asyncio.gather(
        _run_language(), _run_video()
    )

    # ── Unpack language result ──
    if lang_outcome:
        language_result = lang_outcome
        language_signals = language_result.get("signals", [])
        language_summary = language_result.get("summary", {})
        logger.info(f"[{session_id}] Language Agent: {len(language_signals)} signals")
        await _persist_agent_signals(session_id, language_signals, speaker_map, "language")

    # ── Call Conversation Agent (now that language signals are available) ──
    await _set_step_state(session_id, "conversation")
    try:
        # Pass language signals into the Conversation Agent for cross-modal rules
        conv_outcome = await _call_conversation_agent(session_id, transcript_segments, meeting_type, language_signals=language_signals)
        if conv_outcome:
            conversation_result = conv_outcome
            conversation_signals = conversation_result.get("signals", [])
            conversation_summary = conversation_result.get("summary", {})
            logger.info(f"[{session_id}] Conversation Agent: {len(conversation_signals)} signals")
            agent_status["conversation"] = "completed"
    except Exception as e:
        agent_status["conversation"] = "failed"
        logger.warning(f"[{session_id}] Conversation Agent failed (continuing): {e}")
        await _persist_agent_signals(session_id, conversation_signals, speaker_map, "conversation")

    # ── Unpack video result ──
    # Initialized here so fusion, alerts, and face registration can use it
    # regardless of whether the vid_signals block executes. For audio-only
    # sessions this is just a copy of speaker_map; for video sessions the block
    # below extends it with Face_* → UUID entries.
    video_speaker_map = dict(speaker_map)
    if vid_signals:
        video_signals = vid_signals

        # Identity resolution is fully handled by the video agent (ArcFace merge +
        # position fallback). The gateway trusts the face_index values as-is.
        face_ids_received = {
            s.get("speaker_id", "")
            for s in video_signals
            if s.get("speaker_id", "").startswith("Face_")
        }
        logger.info(
            f"[{session_id}] Video signals received: {len(face_ids_received)} "
            f"Face_* IDs: {sorted(face_ids_received)}"
        )

        # ── Single-pass Face_* signal analysis ───────────────────────────────
        # One iteration collects signal counts AND all Face_* IDs seen, then
        # derives weak_faces and unmatched_face_ids via set arithmetic —
        # eliminates a second scan of video_signals (up to 6000 items).
        MIN_FACE_SIGNALS = 10
        face_signal_counts: dict[str, int] = {}
        face_ids_in_signals: set[str] = set()
        for sig in video_signals:
            spk = sig.get("speaker_id", "")
            if spk.startswith("Face_"):
                face_signal_counts[spk] = face_signal_counts.get(spk, 0) + 1
                face_ids_in_signals.add(spk)

        weak_faces: set[str] = {
            fid for fid, cnt in face_signal_counts.items()
            if cnt < MIN_FACE_SIGNALS
        }
        if weak_faces:
            before = len(video_signals)
            video_signals = [
                s for s in video_signals
                if s.get("speaker_id", "") not in weak_faces
            ]
            logger.info(
                f"[{session_id}] Filtered {len(weak_faces)} weak Face_* "
                f"(< {MIN_FACE_SIGNALS} signals): {sorted(weak_faces)} — "
                f"removed {before - len(video_signals)} signals"
            )

        # Unmatched = seen in signals, not weak, not already in speaker_map.
        # Equivalent to iterating filtered video_signals for Face_* not in map,
        # but uses set arithmetic instead of a second list comprehension.
        unmatched_face_ids = face_ids_in_signals - weak_faces - set(video_speaker_map.keys())
        if unmatched_face_ids:
            new_face_speakers = await upsert_speakers(session_id, [
                {"speaker_id": face_id} for face_id in unmatched_face_ids
            ])
            video_speaker_map.update(new_face_speakers)
        # Signals persisted AFTER registry matching below — see end of vid_signals block

    # ── Speaker registry: fused face + voice identity matching ───────────────
    # Runs here (after gather) so both voice embeddings AND face embeddings are
    # available. Falls back to voice-only when no video was processed.

    # Remap face_embeddings keys: Face_N → Speaker_N using lip-sync linkage.
    # Build a session-scoped lock map so only reasonably strong lip-sync links
    # are used as visual evidence for registry matching.
    SESSION_FACE_LOCK_MIN_SCORE = float(os.getenv("SESSION_FACE_LOCK_MIN_SCORE", "0.10"))

    def _build_session_face_locks(
        face_to_speaker: dict,
        lip_sync_scores: dict,
        face_embeddings_from_video: dict,
        session_id: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        locked_face_to_speaker: dict[str, str] = {}
        locked_speaker_to_face: dict[str, str] = {}
        if not face_to_speaker or not lip_sync_scores:
            return locked_face_to_speaker, locked_speaker_to_face

        candidates = []
        for raw_face_idx, speaker_label in face_to_speaker.items():
            if not str(speaker_label).startswith("Speaker_"):
                continue
            try:
                face_label = (
                    raw_face_idx if str(raw_face_idx).startswith("Face_")
                    else f"Face_{int(raw_face_idx)}"
                )
            except Exception:
                face_label = f"Face_{raw_face_idx}"

            if face_label not in face_embeddings_from_video:
                continue

            score = float(lip_sync_scores.get(speaker_label, 0.0) or 0.0)
            candidates.append((score, face_label, speaker_label))

        candidates.sort(reverse=True, key=lambda x: x[0])

        for score, face_label, speaker_label in candidates:
            if score < SESSION_FACE_LOCK_MIN_SCORE:
                logger.warning(
                    f"[{session_id}] Rejecting weak session face lock "
                    f"{face_label} → {speaker_label}, lip_sync_score={score:.4f}, "
                    f"threshold={SESSION_FACE_LOCK_MIN_SCORE:.4f}"
                )
                continue

            if face_label in locked_face_to_speaker:
                continue

            if speaker_label in locked_speaker_to_face:
                logger.warning(
                    f"[{session_id}] Rejecting duplicate speaker lock candidate "
                    f"{face_label} → {speaker_label}; already locked to "
                    f"{locked_speaker_to_face[speaker_label]}"
                )
                continue

            locked_face_to_speaker[face_label] = speaker_label
            locked_speaker_to_face[speaker_label] = face_label

            logger.info(
                f"[{session_id}] Session face lock accepted: "
                f"{face_label} → {speaker_label}, lip_sync_score={score:.4f}"
            )

        return locked_face_to_speaker, locked_speaker_to_face

    locked_face_to_speaker, locked_speaker_to_face = _build_session_face_locks(
        face_to_speaker=face_to_speaker,
        lip_sync_scores=lip_sync_scores,
        face_embeddings_from_video=face_embeddings_from_video,
        session_id=session_id,
    )

    speaker_keyed_face_embs: dict = {}
    for face_label, speaker_label in locked_face_to_speaker.items():
        face_data = face_embeddings_from_video.get(face_label)
        if not face_data:
            continue
        score = float(lip_sync_scores.get(speaker_label, 0.0) or 0.0)
        speaker_keyed_face_embs[speaker_label] = {
            **face_data,
            "source_face_label": face_label,
            "session_face_lock": True,
            "lip_sync_score": score,
        }
        logger.info(
            f"[{session_id}] Locked face embedding for registry: "
            f"{face_label} → {speaker_label}, score={score:.4f}"
        )

    speaker_identity_map: dict = {}
    if speaker_embeddings:
        try:
            from speaker_registry import match_or_create_speakers
            pool = await get_pool()
            speaker_identity_map = await match_or_create_speakers(
                pool=pool,
                session_id=session_id,
                speaker_embeddings=speaker_embeddings,
                voice_speakers=voice_speakers,
                speaker_map=speaker_map,
                org_id=org_id,
                face_embeddings=speaker_keyed_face_embs,
            )
            logger.info(f"[{session_id}] Speaker identity: {speaker_identity_map}")
        except Exception as e:
            logger.warning(f"[{session_id}] Speaker registry match failed (non-fatal): {e}")
    elif face_embeddings_from_video:
        # Video-only session (no voice diarization) — match by face alone
        try:
            from speaker_registry import match_or_create_by_face_only
            pool = await get_pool()
            speaker_identity_map = await match_or_create_by_face_only(
                pool=pool,
                session_id=session_id,
                face_embeddings=face_embeddings_from_video,
                speaker_map=speaker_map,
                org_id=org_id,
            )
            logger.info(f"[{session_id}] Face-only identity: {speaker_identity_map}")
        except Exception as e:
            logger.warning(f"[{session_id}] Face-only registry match failed (non-fatal): {e}")

    # ── Non-speaking face registration (audio+video sessions) ────────────────
    # Speaker_N faces are handled above via fused matching. Face_N entries are
    # faces SpeakerFaceMapper never linked to a diarization speaker because those
    # people never spoke. Register them by face-only so they are recognized across
    # sessions — next time they speak their face match upgrades to fused identity.
    if speaker_embeddings and face_embeddings_from_video:
        # Exclude faces that were session-locked to speakers from face-only registration
        linked_face_labels = set(locked_face_to_speaker.keys())
        non_speaking = {
            label: data
            for label, data in face_embeddings_from_video.items()
            if label.startswith("Face_")
            and label not in speaker_identity_map
            and label not in linked_face_labels
        }
        if non_speaking:
            try:
                from speaker_registry import match_or_create_by_face_only
                pool = await get_pool()
                face_only_matches = await match_or_create_by_face_only(
                    pool=pool,
                    session_id=session_id,
                    face_embeddings=non_speaking,
                    speaker_map=video_speaker_map,
                    org_id=org_id,
                )
                speaker_identity_map.update(face_only_matches)
                logger.info(
                    f"[{session_id}] Non-speaking identities registered: "
                    f"{len(face_only_matches)} (labels: {list(face_only_matches)})"
                )
            except Exception as e:
                logger.warning(
                    f"[{session_id}] Non-speaking face registry failed (non-fatal): {e}"
                )

    # ── Link Face_N → Speaker_N registry entries via lip-sync linkage ────────
    # After fused matching, Speaker_N entries have registry_ids. Face_N tracks
    # that map to the same person (via face_to_speaker) must share that same
    # registry_id so /video-signals and /video-speakers canonical merge groups
    # them as one person in the UI.
    # Create session-locked aliases only from locked_face_to_speaker (strong lip-sync links)
    if locked_face_to_speaker:
        for face_label, speaker_label in locked_face_to_speaker.items():
            if speaker_label not in speaker_identity_map:
                logger.warning(
                    f"[{session_id}] Cannot alias locked face {face_label} → {speaker_label}: "
                    f"{speaker_label} has no registry identity"
                )
                continue

            if face_label not in speaker_identity_map:
                score = float(lip_sync_scores.get(speaker_label, 0.0) or 0.0)

                speaker_identity_map[face_label] = {
                    **speaker_identity_map[speaker_label],
                    "match_method": "session_lip_sync_lock",
                    "match_confidence": min(0.95, max(0.0, score)),
                    "source_speaker_label": speaker_label,
                }

                logger.info(
                    f"[{session_id}] Session-locked alias created: "
                    f"{face_label} → {speaker_label} "
                    f"(registry_id={speaker_identity_map[speaker_label].get('registry_id')}, "
                    f"score={score:.4f})"
                )

    # ── Persist video signals (after registry so unregistered faces are excluded) ──
    # Registry matching (fused + face-only + non-speaking) is now complete.
    # Any Face_* not in speaker_identity_map has no ArcFace embedding — MediaPipe
    # tracked it but couldn't identify it (too small, occluded, background face).
    # Filter those out before writing to DB so they never appear in the UI or reports.
    if vid_signals:
        registered_face_labels: set[str] = {
            label for label in speaker_identity_map
            if label.startswith("Face_")
        }
        all_face_labels_in_signals: set[str] = {
            s.get("speaker_id", "")
            for s in video_signals
            if s.get("speaker_id", "").startswith("Face_")
        }
        unregistered_faces = all_face_labels_in_signals - registered_face_labels
        if unregistered_faces:
            before = len(video_signals)
            video_signals = [
                s for s in video_signals
                if s.get("speaker_id", "") not in unregistered_faces
            ]
            logger.info(
                f"[{session_id}] Excluded {len(unregistered_faces)} unregistered "
                f"Face_* (no registry entry, signals not persisted): "
                f"{sorted(unregistered_faces)} — "
                f"dropped {before - len(video_signals)} signals"
            )
        await _persist_agent_signals(session_id, video_signals, video_speaker_map, "video")

    # ── Post display names to video agent for burned-in overlay ──────────────
    # The burn phase waits 15s after signals_ready before starting, giving us
    # time to complete registry matching above and deliver the display name map.
    if video_job_id and speaker_identity_map:
        display_name_map = {
            label: info["display_name"]
            for label, info in speaker_identity_map.items()
            if info.get("display_name") and info["display_name"] != label
        }
        if display_name_map:
            asyncio.create_task(_post_video_display_names(video_job_id, display_name_map))

    # ── Step 5: Fusion Agent (only when behavioural analysis enabled) ──
    await _set_step_state(session_id, "fusion")
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
                video_signals=video_signals,
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

    await _persist_agent_signals(session_id, fusion_signals, video_speaker_map, "fusion")

    # ── Back-fill speaker_appearances stats now all signals are persisted ──
    if speaker_embeddings:
        try:
            from speaker_registry import update_appearance_stats
            pool = await get_pool()
            await update_appearance_stats(pool, session_id)
            logger.info(f"[{session_id}] Appearance stats updated")
        except Exception as e:
            logger.warning(f"[{session_id}] Appearance stats update failed (non-fatal): {e}")

    if alerts:
        try:
            count = await insert_alerts(session_id, alerts, video_speaker_map)
            logger.info(f"[{session_id}] Persisted {count} alerts")
        except Exception as e:
            logger.warning(f"[{session_id}] Alert persist failed: {e}")

    # Enrich report with entities and signal graph from fusion summary
    entities = language_summary.get("entities", {})
    fusion_summary = fusion_result.get("summary", {}) if fusion_result else {}
    signal_graph = fusion_summary.get("signal_graph", {})
    key_paths = fusion_summary.get("key_paths", [])

    # ── Step 6: Persist report ──
    await _set_step_state(session_id, "report")
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
        participant_count=speaker_count,
    )

    logger.info(f"[{session_id}] Pipeline complete (status={final_status}, agents={agent_status})")

    await _post_process_pipeline(
        session_id=session_id,
        run_behavioural=run_behavioural,
        run_knowledge_graph=analysis_config.get("run_knowledge_graph", True),
        transcript_segments=transcript_segments,
        all_signals=voice_signals + language_signals + conversation_signals + video_signals + fusion_signals,
        entities=entities,
        report_content=report_content,
        graph_analytics=graph_analytics,
        conversation_summary=conversation_summary,
    )

    # ── Completion notification email (behavioural analysis only) ────────────
    # Lightweight sessions (transcription/diarization/entity-only) finish in
    # seconds — user is still on the page, no email needed.
    # Full behavioural analysis takes 5-30+ minutes — user may have closed
    # the browser, so notify them via email that results are ready.
    if run_behavioural and user_email:
        try:
            from email_service import send_processing_complete_email, is_email_configured
            if is_email_configured():
                _user_name = "there"
                try:
                    _pool = await get_pool()
                    _user_row = await _pool.fetchrow(
                        "SELECT full_name FROM users WHERE email = $1", user_email
                    )
                    if _user_row and _user_row["full_name"]:
                        _user_name = _user_row["full_name"]
                except Exception:
                    pass

                await send_processing_complete_email(
                    to_email=user_email,
                    to_name=_user_name,
                    session_id=session_id,
                    session_title=title,
                    meeting_type=meeting_type,
                    status=final_status,
                    signal_counts={
                        "voice":        len(voice_signals),
                        "language":     len(language_signals),
                        "conversation": len(conversation_signals),
                        "video":        len(video_signals),
                        "fusion":       len(fusion_signals),
                    },
                    duration_seconds=duration_seconds,
                )
        except Exception as e:
            logger.warning(f"[{session_id}] Completion email failed (non-fatal): {e}")

    # Cleanup old recordings in background
    _cleanup_old_recordings()

    _pipeline_progress.pop(session_id, None)  # legacy in-memory progress cleanup
    await redis_repo.set_session_state(
        session_id,
        SessionStateRecord(status=final_status, current_step="completed"),
    )

    logger.info(
        f"[{session_id}] Pipeline finished: status={final_status}, "
        f"voice={len(voice_signals)}, lang={len(language_signals)}, "
        f"convo={len(conversation_signals)}, video={len(video_signals)}, "
        f"fusion={len(fusion_signals)}, "
        f"alerts={len(alerts)}, report={'yes' if report_generated else 'no'}"
    )


# ─────────────────────────────────────────────────────────
# GET /sessions — List
# ─────────────────────────────────────────────────────────



@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions_endpoint(
    limit: int = Query(default=25, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    meeting_type: Optional[str] = Query(default=None),
    session_type: Optional[str] = Query(default=None),
    current_user: dict = Depends(get_current_user),
):
    """List sessions with pagination and optional filters."""
    logger.info(
        f"GET /sessions user={current_user.get('email', 'unknown')} "
        f"limit={limit} offset={offset} status={status} meeting_type={meeting_type} session_type={session_type}"
    )
    user_id = None
    try:
        sessions, total = await list_sessions(
            limit=limit,
            offset=offset,
            status=status,
            meeting_type=meeting_type,
            session_type=session_type,
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
async def get_session_detail(session_id: str, _: dict = Depends(get_current_user)):
    """Get session detail including signals, alerts, and unified states."""
    logger.info(f"[{session_id}] session call")
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")


    # Fetch related data in parallel-ish
    signals = [s for s in await get_signals(session_id, limit=50000) if _should_display_signal(s)]
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
# Signal display filter — thresholds matched to backend rule engines
# ─────────────────────────────────────────────────────────

def _should_display_signal(sig: dict) -> bool:
    """
    Return True if this signal is meaningful enough to show in the UI.
    Thresholds are derived from each agent's rule-engine output ranges:

    Voice:
      vocal_stress_score — rule emits moderate(>0.30), elevated(>0.50), high(>0.70)
                           Show elevated+ only (v > 0.50)
      sentiment_score    — rule emits mild(|v|>0.35), pos/neg(|v|>0.55), strong(|v|>0.80)
                           Show pos/neg+ only (|v| > 0.55)
      All others         — backend already gates these; show everything emitted

    Video (all sub-types)              → confidence >= 0.20  (matches MIN_SIGNAL_CONFIDENCE gates)
      VideoSignalPlayer applies its own 0.30 client-side filter for the realtime overlay.
      SessionDetail VISUAL stats need the lower floor to build per-speaker summaries.

    Fusion / language / conversation   → show all (already filtered at source)
    """
    stype = sig.get("signal_type", "")
    value = sig.get("value") or 0.0
    conf  = sig.get("confidence") or 0.0
    agent = sig.get("agent", "")

    if stype == "vocal_stress_score":
        return value > 0.50

    if stype == "sentiment_score":
        return abs(value) > 0.55

    if agent == "video":
        # Video signals: threshold aligned with rule-engine MIN_SIGNAL_CONFIDENCE gates
        # (facial=0.18, body=0.20). VideoSignalPlayer applies its own 0.30 client-side
        # filter for the realtime overlay; we use a lower floor here so that
        # computeVideoStats in SessionDetail can build stats from all stored signals.
        return conf >= 0.20

    if agent == "fusion":
        # Session-level temporal aggregates span the full recording — analytical
        # summaries, not moment signals. Filter windows longer than 2 minutes.
        duration_ms = (sig.get("window_end_ms") or 0) - (sig.get("window_start_ms") or 0)
        if duration_ms > 120_000:
            return False
        # Low floor — fusion temporal signals (trajectory, decay, adaptation) have
        # inherently low confidence by design; the old 0.40 floor was blocking them all.
        return conf >= 0.10

    return True


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/signals — Signals with filters
# ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/signals")
async def get_session_signals(
    session_id: str,
    agent: Optional[str] = Query(default=None),
    signal_type: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=50000),
    offset: int = Query(default=0, ge=0),
    _: dict = Depends(get_current_user),
):
    """Get signals for a session with optional filtering by agent/type."""
    logger.info(
        f"[{session_id}] GET /sessions/{{id}}/signals agent={agent} signal_type={signal_type} "
        f"limit={limit} offset={offset}"
    )
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")


    signals = await get_signals(
        session_id,
        agent=agent,
        signal_type=signal_type,
        limit=limit,
        offset=offset,
    )
    signals = [s for s in signals if _should_display_signal(s)]

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
    _: dict = Depends(get_current_user),
):
    """
    Get the narrative report for a session.
    If no report exists (or regenerate=True), triggers Fusion Agent to generate one.
    """
    logger.info(f"[{session_id}] GET /sessions/{{id}}/report regenerate={regenerate}")
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")


    # Return existing report unless regenerate requested
    if not regenerate:
        existing = await get_report(session_id)
        if existing:
            return {"session_id": session_id, "report": existing}

    # Generate new report via Fusion Agent
    signals = await get_signals(session_id, limit=50000)

    voice_signals = [s for s in signals if s.get("agent") == "voice"]
    language_signals = [s for s in signals if s.get("agent") == "language"]
    fusion_signals = [s for s in signals if s.get("agent") == "fusion"]

    speakers = list(set(
        s.get("speaker_label") or s.get("speaker_id", "unknown")
        for s in signals if s.get("speaker_label") or s.get("speaker_id")
    ))

    duration_seconds = (session.get("duration_ms") or 0) / 1000.0

    # Build summaries from stored signals
    video_signals_stored = [s for s in signals if s.get("agent") == "video"]
    voice_summary = _build_voice_summary(voice_signals)
    language_summary = _build_language_summary(language_signals)
    video_summary = _build_video_summary(video_signals_stored) if video_signals_stored else None

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
                    "video_summary": video_summary,
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
    state = await redis_repo.get_session_state(session_id)
    if state:
        return {"pipeline_step": state.get("current_step"), "status": state.get("status")}
    return {"pipeline_step": _pipeline_progress.get(session_id)}


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/transcript — Transcript
# ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: str, _: dict = Depends(get_current_user)):
    """Get transcript segments for a session."""
    logger.info(f"[{session_id}] GET /sessions/{{id}}/transcript")
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")


    segments = await get_transcript(session_id)

    return {
        "session_id": session_id,
        "segments": segments,
        "count": len(segments),
    }


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/video-signals — Video overlay signals
# ─────────────────────────────────────────────────────────

_VIDEO_OVERLAY_TYPES = [
    # Facial
    "facial_emotion", "facial_stress", "facial_engagement",
    "smile_type", "valence_arousal",
    # Body
    "head_nod", "head_shake", "posture", "body_lean",
    "body_fidgeting", "self_touch", "shoulder_tension",
    "head_body_incongruence", "gesture_animation", "body_mirroring",
    "face_region_touch", "arms_crossed", "finger_steepling",
    "head_supported", "hands_clasped", "cross_speaker_interaction",
    "posture_transition", "body_language_cluster",
    # Facial (extended)
    "lip_pursing",
    # Gaze
    "gaze_direction_shift", "screen_contact", "sustained_distraction",
    "attention_level", "blink_rate_anomaly", "gaze_synchrony",
    # Body (extended)
    "evaluation_cluster", "hidden_disagreement", "frustration_cluster",
    "hand_gesture", "arm_posture",
    # Facial (extended)
    "laughter",
    # Fusion pairwise
    "tone_face_masking", "stress_suppression", "rapport_confirmation",
    "voice_face_alignment",
    # Compound patterns (C-01 through C-12)
    "genuine_engagement", "active_disengagement", "emotional_suppression",
    "decision_engagement", "cognitive_overload", "conflict_escalation",
    "verbal_nonverbal_discordance", "peak_performance", "rapport_building",
    "dominance_display", "submission_signal", "deception_cluster",
    # Temporal patterns (T-03 through T-08, excluding session-arc signals)
    # stress_trajectory, engagement_decay, escalation_ladder are session-level
    # conclusions stored in DB and shown in reports — not moment badges.
    "rapport_evolution",
    "behavioral_shift", "adaptation_pattern", "fatigue_detection",
    "stress_recovery",
    # Graph-based
    "tension_cluster",
    # Voice (shown alongside face signals for Speaker_* identities)
    "vocal_stress_score", "tone_classification", "filler_detection",
    "speech_rate_anomaly", "energy_level", "pitch_elevation_flag",
    "monotone_flag", "interruption_event",
    # Language
    "sentiment_score",
]


# Matches auto-generated speaker labels that should not be used as display names.
_GENERIC_SPEAKER_LABEL_RE = re.compile(r'^(Speaker|Face)_\d+$')


def _speaker_grid_position(face_centre_x: float, face_centre_y: float) -> str:
    """Convert normalised face centre coordinates to a human-readable grid label."""
    if face_centre_x < 0.01 and face_centre_y < 0.01:
        return ""
    col = "Left" if face_centre_x < 0.33 else ("Center" if face_centre_x < 0.66 else "Right")
    row = "Top" if face_centre_y < 0.5 else "Bottom"
    return f"{row}-{col}"


@app.get("/sessions/{session_id}/video-signals")
async def get_video_signals(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return video + fusion signals for playback overlay, ordered by window start."""
    logger.info(f"[{session_id}] GET /sessions/{{id}}/video-signals")
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")
    session = await get_session(session_id, current_user.get("org_id") or DEV_ORG_ID)
    if not session:
        raise HTTPException(404, "Session not found")

    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT s.signal_type, s.value, s.value_text, s.confidence,
               s.window_start_ms, s.window_end_ms, s.agent,
               s.metadata, sp.speaker_label,
               sr.display_name AS registry_name,
               sr.id           AS registry_id
        FROM signals s
        LEFT JOIN speakers sp ON sp.id = s.speaker_id
        LEFT JOIN speaker_appearances sa
               ON sa.session_id    = s.session_id
              AND sa.speaker_label = sp.speaker_label
        LEFT JOIN speakers_registry sr ON sr.id = sa.registry_id
        WHERE s.session_id = $1
          AND s.agent IN ('video', 'fusion')
          AND s.signal_type = ANY($2::text[])
        ORDER BY s.window_start_ms ASC
        """,
        session_id,
        _VIDEO_OVERLAY_TYPES,
    )

    signals = []
    for r in rows:
        meta: dict = {}
        if r["metadata"]:
            try:
                meta = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else dict(r["metadata"])
            except Exception:
                pass
        cx = float(meta.get("face_centre_x", 0))
        cy = float(meta.get("face_centre_y", 0))
        if cx > 0 or cy > 0:
            meta["grid_position"] = _speaker_grid_position(cx, cy)
        reg_name  = r["registry_name"] or ""
        spk_label = r["speaker_label"] or ""
        speaker_name = reg_name if (reg_name and not _GENERIC_SPEAKER_LABEL_RE.match(reg_name)) else spk_label
        signals.append({
            "signal_type":  r["signal_type"],
            "value":        float(r["value"]) if r["value"] is not None else 0.0,
            "value_text":   r["value_text"] or "",
            "confidence":   float(r["confidence"]) if r["confidence"] is not None else 0.0,
            "speaker_id":   spk_label,
            "speaker_name": speaker_name,
            "registry_id":  str(r["registry_id"]) if r["registry_id"] else "",
            "start_ms":     r["window_start_ms"],
            "end_ms":       r["window_end_ms"],
            "agent":        r["agent"] or "",
            "metadata":     meta,
        })

    # Filter session-spanning temporal/fusion signals before building the overlay.
    # Temporal patterns (adaptation, fatigue, behavioral_shift) emit analytical
    # summaries whose window spans the full session — they break computeActive by
    # appearing active at every playback position. Cap at 120 s, same gate as
    # _should_display_signal applies to /sessions/{id}/signals.
    signals = [
        s for s in signals
        if (s["end_ms"] - s["start_ms"]) <= 120_000
    ]

    # Merge duplicate speaker labels that share the same registry identity.
    # The video agent may emit the same person under different labels across a
    # session (e.g. Face_2 for 0–636 s then Speaker_1 when promoted to the
    # active-speaker tile). Rewrite all signals to one canonical label per
    # registry_id so the frontend groups them as one person.
    # Canonical preference: Speaker_* before Face_*, then alphabetical.
    registry_groups: dict[str, list[str]] = {}
    for sig in signals:
        reg = sig.get("registry_id")
        spk = sig.get("speaker_id")
        if reg and spk:
            registry_groups.setdefault(reg, []).append(spk)

    canonical_by_registry: dict[str, str] = {
        reg: sorted(
            set(labels),
            key=lambda x: (0 if x.startswith("Speaker_") else 1, x),
        )[0]
        for reg, labels in registry_groups.items()
    }

    for sig in signals:
        reg = sig.get("registry_id")
        if reg and reg in canonical_by_registry:
            canonical = canonical_by_registry[reg]
            if sig["speaker_id"] != canonical:
                sig["raw_speaker_id"] = sig["speaker_id"]
                sig["speaker_id"]     = canonical
                sig["speaker_name"]   = canonical

    # Build speaker → grid_position map from video signals (face coords present)
    # Used to enrich temporal/fusion signals that have no face_centre_x/y.
    speaker_positions: dict[str, str] = {}
    for sig in signals:
        spk = sig["speaker_id"]
        pos = sig["metadata"].get("grid_position", "")
        if spk and pos and spk not in speaker_positions:
            speaker_positions[spk] = pos

    # Apply grid_position fallback to any signal that is missing one
    for sig in signals:
        if not sig["metadata"].get("grid_position"):
            spk = sig["speaker_id"]
            if spk and spk in speaker_positions:
                sig["metadata"]["grid_position"] = speaker_positions[spk]

    return {"session_id": session_id, "signals": signals}


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/video-speakers — Speaker roster for video UI
# ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/video-speakers")
async def get_video_speakers(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Return all speakers/faces in a video session with identity info,
    grid position inferred from face coordinates, and thumbnail URL.

    Called once by VideoSignalPlayer on mount to build the speaker roster.
    """
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    session = await get_session(session_id, current_user.get("org_id") or DEV_ORG_ID)
    if not session:
        raise HTTPException(404, "Session not found")

    pool = await get_pool()

    rows = await pool.fetch("""
        SELECT DISTINCT ON (sp.speaker_label)
               sp.speaker_label,
               sa.registry_id, sr.display_name, sr.role, sr.company,
               sa.match_method, sa.match_confidence
        FROM   speakers sp
        LEFT JOIN speaker_appearances sa
               ON sa.session_id    = sp.session_id
              AND sa.speaker_label = sp.speaker_label
        LEFT JOIN speakers_registry sr ON sr.id = sa.registry_id
        WHERE  sp.session_id = $1
        ORDER  BY sp.speaker_label,
                  sa.match_confidence DESC NULLS LAST,
                  sa.registry_id      NULLS LAST
    """, uuid.UUID(session_id))

    # Build speaker_label → grid_position from face_centre coords in signal metadata
    pos_rows = await pool.fetch("""
        SELECT DISTINCT ON (sp.speaker_label) sp.speaker_label, s.metadata
        FROM   signals s
        JOIN   speakers sp ON sp.id = s.speaker_id
        WHERE  s.session_id = $1
          AND  s.agent = 'video'
          AND  s.metadata IS NOT NULL
        ORDER  BY sp.speaker_label, s.window_start_ms
    """, uuid.UUID(session_id))

    label_positions: dict[str, str] = {}
    for pr in pos_rows:
        label = pr["speaker_label"]
        try:
            meta = json.loads(pr["metadata"]) if isinstance(pr["metadata"], str) else dict(pr["metadata"])
            cx = float(meta.get("face_centre_x", 0))
            cy = float(meta.get("face_centre_y", 0))
            if cx > 0 or cy > 0:
                label_positions[label] = _speaker_grid_position(cx, cy)
        except Exception:
            pass

    # Build canonical label per registry_id: Speaker_* preferred over Face_*,
    # then alphabetical. Mirrors the same preference used in /video-signals.
    reg_label_groups: dict[str, list[str]] = {}
    for r in rows:
        reg = str(r["registry_id"]) if r["registry_id"] else ""
        lbl = r["speaker_label"] or ""
        if reg and lbl:
            reg_label_groups.setdefault(reg, []).append(lbl)

    canonical_label_for_reg: dict[str, str] = {
        reg: sorted(
            set(labels),
            key=lambda x: (0 if x.startswith("Speaker_") else 1, x),
        )[0]
        for reg, labels in reg_label_groups.items()
    }

    speakers = []
    seen_registries: set[str] = set()
    for r in rows:
        registry_id = str(r["registry_id"]) if r["registry_id"] else ""
        slabel = r["speaker_label"] or ""

        # Skip non-canonical rows — same person already added under canonical label
        if registry_id and registry_id in seen_registries:
            continue
        if registry_id:
            seen_registries.add(registry_id)

        # Use canonical label (Speaker_* preferred) as the displayed speaker_label
        canonical = canonical_label_for_reg.get(registry_id, slabel)

        rname = r["display_name"] or ""
        # Only use registry display_name if it's a real person name, not a generated label
        display_name = rname if (rname and not _GENERIC_SPEAKER_LABEL_RE.match(rname)) else canonical

        speakers.append({
            "speaker_label":    canonical,
            "display_name":     display_name,
            "role":             r["role"] or "",
            "company":          r["company"] or "",
            "grid_position":    label_positions.get(canonical, "") or label_positions.get(slabel, ""),
            "registry_id":      registry_id,
            "match_method":     r["match_method"] or "",
            "match_confidence": float(r["match_confidence"]) if r["match_confidence"] else 0.0,
            "thumbnail_url":    f"/speakers/{registry_id}/thumbnail" if registry_id else "",
        })

    return {"session_id": session_id, "speakers": speakers}


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/video — Stream session video file
# ─────────────────────────────────────────────────────────

_VIDEO_MIME_TYPES: dict[str, str] = {
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".avi":  "video/x-msvideo",
    ".mkv":  "video/x-matroska",
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".m4a":  "audio/mp4",
    ".flac": "audio/flac",
}


@app.get("/sessions/{session_id}/video")
async def get_session_video(
    request: StarletteRequest,
    session_id: str,
    token: Optional[str] = Query(default=None),
):
    """
    Stream the session media file.
    Accepts JWT via Authorization: Bearer header OR ?token= query param.
    The query param form is required for <video> elements which cannot set headers.
    """
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    # Auth: prefer Authorization header, fall back to ?token= for video elements
    auth_token = token
    if not auth_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]
    if not auth_token:
        raise HTTPException(401, "Unauthorized")

    payload = verify_access_token(auth_token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")

    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")


    media_url = session.get("media_url")
    if not media_url:
        raise HTTPException(404, "No media file for this session")

    video_path = Path(media_url)
    if not video_path.exists():
        raise HTTPException(404, "Media file not found on disk")

    media_type = _VIDEO_MIME_TYPES.get(video_path.suffix.lower(), "application/octet-stream")

    return FileResponse(
        str(video_path),
        media_type=media_type,
        headers={"Accept-Ranges": "bytes"},
    )


# ─────────────────────────────────────────────────────────
# GET /sessions/{id}/video/annotated — Landmark-overlay video
# ─────────────────────────────────────────────────────────

@app.api_route("/sessions/{session_id}/video/annotated", methods=["GET", "HEAD"])
async def get_annotated_video(
    request: StarletteRequest,
    session_id: str,
    token: Optional[str] = Query(default=None),
):
    """
    Stream the landmark-annotated video produced by the video agent.
    Accepts JWT via Authorization: Bearer header OR ?token= query param.
    Returns 404 when no annotated video exists for this session yet.
    """
    import uuid as _uuid
    try:
        _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    auth_token = token
    if not auth_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]
    if not auth_token:
        raise HTTPException(401, "Unauthorized")

    payload = verify_access_token(auth_token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")

    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")


    # Support both webm (new) and mp4 (legacy) annotated videos
    overlay_path = OVERLAY_DIR / f"{session_id}_annotated.webm"
    media_type = "video/webm"
    if not overlay_path.exists():
        overlay_path = OVERLAY_DIR / f"{session_id}_annotated.mp4"
        media_type = "video/mp4"
    if not overlay_path.exists():
        raise HTTPException(404, "Annotated video not available for this session")

    file_size = overlay_path.stat().st_size
    logger.info(f"[{session_id}] serving annotated video: {overlay_path.name} ({file_size:,} bytes, {media_type})")

    if request.method == "HEAD":
        from starlette.responses import Response as StarletteResponse
        return StarletteResponse(
            status_code=200,
            headers={
                "Content-Type": media_type,
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            },
        )

    return FileResponse(
        str(overlay_path),
        media_type=media_type,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


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


def _format_assemblyai_entities(raw: list[dict]) -> dict:
    """Convert AssemblyAI entity_detection results to NEXUS entity format."""
    _TYPE_MAP = {
        "person_name":   "people",
        "organization":  "organizations",
        "location":      "locations",
        "product_name":  "products",
    }
    result: dict[str, list] = {
        "people": [], "organizations": [], "locations": [],
        "products": [], "topics": [], "objections": [], "commitments": [],
    }
    seen: set[tuple] = set()
    for ent in raw:
        bucket = _TYPE_MAP.get(ent.get("entity_type", ""))
        if not bucket:
            continue
        text = (ent.get("text") or "").strip()
        key = (bucket, text.lower())
        if text and key not in seen:
            seen.add(key)
            result[bucket].append({"text": text, "start_ms": ent.get("start"), "end_ms": ent.get("end")})
    return result


async def _persist_agent_signals(
    session_id: str,
    signals: list[dict],
    speaker_map: dict,
    label: str,
) -> None:
    """Persist a batch of signals and log the count. Replaces any existing
    signals for this session+agent so re-runs don't accumulate stale rows."""
    if not signals:
        return
    try:
        pool = await get_pool()
        await pool.execute(
            "DELETE FROM signals WHERE session_id = $1 AND agent = $2",
            uuid.UUID(session_id), label,
        )
        count = await insert_signals(session_id, signals, speaker_map)
        logger.info(f"[{session_id}] Persisted {count} {label} signals (replaced)")
    except Exception as e:
        logger.warning(f"[{session_id}] {label} signal persist failed: {e}")



async def _post_process_pipeline(
    session_id: str,
    run_behavioural: bool,
    run_knowledge_graph: bool,
    transcript_segments: list[dict],
    all_signals: list[dict],
    entities: dict,
    report_content: dict,
    graph_analytics: dict,
    conversation_summary: dict,
) -> None:
    """Knowledge store embedding + Neo4j sync — both non-fatal."""
    await _set_step_state(session_id, "entity_extraction")
    if run_behavioural:
        try:
            from knowledge_store import store_session_knowledge
            _pool = await get_pool()
            await store_session_knowledge(_pool, session_id, {
                "transcript_segments": transcript_segments,
                "signals": all_signals,
                "entities": entities,
                "report": report_content or {},
                "graph_analytics": graph_analytics or {},
                "conversation_summary": conversation_summary or {},
            })
        except Exception as e:
            logger.warning(f"[{session_id}] Knowledge store failed (non-fatal): {e}")
    else:
        logger.info(f"[{session_id}] Knowledge store skipped (behavioural analysis off)")

    await _set_step_state(session_id, "knowledge_graph")
    if run_knowledge_graph:
        try:
            from neo4j_sync import sync_session as neo4j_sync_session
            _pool = await get_pool()
            await neo4j_sync_session(_pool, session_id)
        except Exception as e:
            logger.warning(f"[{session_id}] Neo4j sync failed (non-fatal): {e}")

        # Phase 3B — sync PersistentSpeaker nodes for every registry entry that
        # appeared in this session (runs after sync_session so Speaker nodes exist)
        try:
            from neo4j_sync import sync_speaker_registry_to_neo4j
            _pool = await get_pool()
            reg_rows = await _pool.fetch(
                "SELECT DISTINCT registry_id FROM speaker_appearances WHERE session_id = $1",
                uuid.UUID(session_id),
            )
            for row in reg_rows:
                await sync_speaker_registry_to_neo4j(_pool, str(row["registry_id"]))
        except Exception as e:
            logger.warning(f"[{session_id}] Speaker registry Neo4j sync failed (non-fatal): {e}")
    else:
        logger.info(f"[{session_id}] Neo4j sync skipped (run_knowledge_graph=false)")


async def _call_voice_agent(
    session_id: str,
    file_path: str,
    meeting_type: str = "sales_call",
    transcription_config: Optional[dict] = None,
    analysis_config: Optional[dict] = None,
    num_speakers: Optional[int] = None,
) -> dict:
    """Call Voice Agent POST /analyse with file path."""
    t0 = time.time()
    logger.info(f"[{session_id}] → Voice Agent /analyse ({Path(file_path).name}, meeting_type={meeting_type})")
    payload: dict[str, Any] = {
        "file_path": file_path,
        "session_id": session_id,
        "meeting_type": meeting_type,
    }
    if num_speakers:
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
        result = resp.json()
    logger.info(
        f"[{session_id}] ← Voice Agent: {len(result.get('signals', []))} signals, "
        f"{len(result.get('speakers', []))} speakers in {time.time()-t0:.1f}s"
    )
    return result


async def _call_language_agent(
    session_id: str,
    segments: list[dict],
    meeting_type: str,
) -> dict:
    """Call Language Agent POST /analyse with transcript segments."""
    t0 = time.time()
    logger.info(f"[{session_id}] → Language Agent /analyse ({len(segments)} segments, meeting_type={meeting_type})")
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
        result = resp.json()
    logger.info(f"[{session_id}] ← Language Agent: {len(result.get('signals', []))} signals in {time.time()-t0:.1f}s")
    return result


async def _call_conversation_agent(
    session_id: str,
    segments: list[dict],
    meeting_type: str,
    language_signals: list[dict] | None = None,
) -> dict:
    """Call Conversation Agent POST /analyse with transcript segments.

    Optionally include `language_signals` for cross-modal conflict detection.
    """
    t0 = time.time()
    speakers = list(set(seg.get("speaker", "unknown") for seg in segments))
    logger.info(f"[{session_id}] → Conversation Agent /analyse ({len(segments)} segments, {len(speakers)} speakers)")
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            f"{CONVERSATION_AGENT_URL}/analyse",
            json={
                "segments": segments,
                "speakers": speakers,
                "content_type": meeting_type,
                "session_id": session_id,
                "language_signals": language_signals,
            },
        )
        resp.raise_for_status()
        result = resp.json()
    logger.info(f"[{session_id}] ← Conversation Agent: {len(result.get('signals', []))} signals in {time.time()-t0:.1f}s")
    return result


async def _post_video_display_names(job_id: str, display_names: dict) -> None:
    """Fire-and-forget: send registry display names to video agent for burn overlay."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{VIDEO_AGENT_URL}/jobs/{job_id}/display-names",
                json={"names": display_names},
            )
            resp.raise_for_status()
            logger.info(f"Display names posted to video job {job_id}: {list(display_names)}")
    except Exception as exc:
        logger.debug(f"Display names POST to job {job_id} failed (non-fatal): {exc}")


async def _call_video_agent(
    session_id: str,
    video_path: str,
    diar_segments: list[dict],
    meeting_type: str,
    num_speakers: int = 2,
) -> dict:
    """
    Submit video to the Video Agent async job queue, then poll until done.

    Protocol:
      POST /analyse  → HTTP 202 {job_id, status="queued"}
      GET  /jobs/{job_id} every POLL_INTERVAL seconds until status in {done, failed}
    """
    import json as _json

    POLL_INTERVAL = 10   # seconds between status checks
    SUBMIT_TIMEOUT = 120 # seconds for the initial upload POST

    t0 = time.time()
    video_file = Path(video_path)
    logger.info(
        f"[{session_id}] → Video Agent /analyse (async) "
        f"({video_file.name}, {len(diar_segments)} diar segments, num_speakers={num_speakers})"
    )

    with open(video_file, "rb") as f:
        video_bytes = f.read()

    # ── Step 1: Submit job ────────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=SUBMIT_TIMEOUT) as client:
        resp = await client.post(
            f"{VIDEO_AGENT_URL}/analyse",
            files={"video": (video_file.name, video_bytes, "video/mp4")},
            data={
                "session_id":         session_id,
                "meeting_type":       meeting_type,
                "diar_segments_json": _json.dumps(diar_segments),
                "num_speakers":       str(num_speakers),
            },
        )
        resp.raise_for_status()
        job_info = resp.json()

    job_id = job_info["job_id"]
    logger.info(f"[{session_id}] Video Agent job {job_id} queued — polling every {POLL_INTERVAL}s")

    # ── Step 2: Poll until done or deadline ───────────────────────────────────
    deadline = t0 + VIDEO_AGENT_TIMEOUT
    async with httpx.AsyncClient(timeout=30) as poll_client:
        while time.time() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            poll_resp = await poll_client.get(f"{VIDEO_AGENT_URL}/jobs/{job_id}")
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = poll_data["status"]

            # Return as soon as signals are available — don't wait for overlay burn
            if status in ("signals_ready", "annotating", "done"):
                result = poll_data["result"]
                # Prefer Redis-backed signals (written per rule-engine during processing)
                # over the HTTP response body — they survive a gateway crash mid-run.
                redis_signals = await _read_agent_signals(session_id, "video")
                if redis_signals:
                    result["signals"] = redis_signals
                    logger.info(
                        f"[{session_id}] ← Video Agent job {job_id} {status}: "
                        f"{len(redis_signals)} signals from Redis in {time.time()-t0:.1f}s"
                    )
                else:
                    logger.info(
                        f"[{session_id}] ← Video Agent job {job_id} {status}: "
                        f"{len(result.get('signals', []))} signals "
                        f"in {time.time()-t0:.1f}s"
                    )
                return result, job_id

            if status == "failed":
                error = poll_data.get("error", "unknown error")
                logger.error(f"[{session_id}] Video Agent job {job_id} failed: {error}")
                raise RuntimeError(f"Video Agent job failed: {error}")

            elapsed = time.time() - t0
            logger.info(f"[{session_id}] Video Agent job {job_id} status={status} ({elapsed:.0f}s)")

    raise TimeoutError(f"Video Agent job {job_id} timed out after {VIDEO_AGENT_TIMEOUT}s")


def _build_video_summary(video_signals: list[dict]) -> dict:
    """
    Aggregate raw video signals into a per-speaker summary dict for the LLM context.
    Groups facial emotion, gaze, body, and gesture signals by speaker.
    """
    from collections import defaultdict

    per_speaker: dict[str, dict] = defaultdict(lambda: {
        "emotions": [],          # (emotion, confidence) tuples
        "facial_stress": [],
        "facial_engagement": [],
        "gaze_on_screen": [],
        "blink_anomalies": 0,
        "head_nods": 0,
        "head_shakes": 0,
        "body_movement": [],
        "posture_changes": 0,
        "valence_arousal": [],
        "hand_near_face": 0,
        "gaze_breaks": 0,
    })

    for s in video_signals:
        spk = s.get("speaker_id") or s.get("speaker_label") or "unknown"
        st  = s.get("signal_type", "")
        val = s.get("value")
        vt  = s.get("value_text", "")
        conf = s.get("confidence", 0.5)

        sp = per_speaker[spk]
        if st == "facial_emotion" and vt:
            sp["emotions"].append((vt, conf))
        elif st == "facial_stress" and val is not None:
            sp["facial_stress"].append(val)
        elif st == "facial_engagement" and val is not None:
            sp["facial_engagement"].append(val)
        elif st == "screen_contact" and val is not None:
            sp["gaze_on_screen"].append(val)
        elif st == "blink_rate_anomaly":
            sp["blink_anomalies"] += 1
        elif st == "head_nod":
            sp["head_nods"] += 1
        elif st == "head_shake":
            sp["head_shakes"] += 1
        elif st == "body_fidgeting" and val is not None:
            sp["body_movement"].append(val)
        elif st == "posture_transition":
            sp["posture_changes"] += 1
        elif st == "valence_arousal" and vt:
            sp["valence_arousal"].append((vt, val or 0))
        elif st == "self_touch":
            sp["hand_near_face"] += 1
        elif st in ("gaze_direction_shift", "sustained_distraction"):
            sp["gaze_breaks"] += 1

    # Compress into readable summary per speaker
    summary: dict[str, dict] = {}
    for spk, data in per_speaker.items():
        # Dominant emotion — most frequent by count then highest confidence
        from collections import Counter
        emotion_counts = Counter(e for e, _ in data["emotions"])
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else None
        dominant_emotion_conf = (
            max((c for e, c in data["emotions"] if e == dominant_emotion), default=0.0)
            if dominant_emotion else 0.0
        )

        # Dominant valence_arousal
        va_counts = Counter(vt for vt, _ in data["valence_arousal"])
        dominant_valence = va_counts.most_common(1)[0][0] if va_counts else None

        def _avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else None

        summary[spk] = {
            "dominant_emotion": dominant_emotion,
            "dominant_emotion_confidence": round(dominant_emotion_conf, 3) if dominant_emotion else None,
            "dominant_valence": dominant_valence,
            "avg_facial_stress": _avg(data["facial_stress"]),
            "avg_facial_engagement": _avg(data["facial_engagement"]),
            "avg_gaze_on_screen_pct": _avg(data["gaze_on_screen"]),
            "gaze_breaks": data["gaze_breaks"],
            "blink_anomalies": data["blink_anomalies"],
            "head_nods": data["head_nods"],
            "head_shakes": data["head_shakes"],
            "avg_body_movement": _avg(data["body_movement"]),
            "posture_changes": data["posture_changes"],
            "hand_near_face_events": data["hand_near_face"],
        }

    return {"per_speaker": summary}


async def _call_fusion_agent(
    session_id: str,
    voice_signals: list[dict],
    language_signals: list[dict],
    voice_summary: dict,
    language_summary: dict,
    meeting_type: str,
    video_signals: Optional[list[dict]] = None,
) -> dict:
    """Call Fusion Agent POST /analyse with all signals."""
    t0 = time.time()
    logger.info(
        f"[{session_id}] → Fusion Agent /analyse "
        f"({len(voice_signals)} voice + {len(language_signals)} language"
        f"{f' + {len(video_signals)} video' if video_signals else ''} signals)"
    )
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

    # Merge video signals into the voice-side pool so fusion sees all modalities
    all_voice_side = [_to_fusion_input(s, "voice") for s in voice_signals]
    video_summary: Optional[dict] = None
    if video_signals:
        all_voice_side += [_to_fusion_input(s, "video") for s in video_signals]
        video_summary = _build_video_summary(video_signals)

    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            f"{FUSION_AGENT_URL}/analyse",
            json={
                "voice_signals": all_voice_side,
                "language_signals": [_to_fusion_input(s, "language") for s in language_signals],
                "session_id": session_id,
                "meeting_type": meeting_type,
                "generate_report": True,
                "voice_summary": voice_summary,
                "language_summary": language_summary,
                "video_summary": video_summary,
            },
        )
        resp.raise_for_status()
        result = resp.json()
    logger.info(
        f"[{session_id}] ← Fusion Agent: {len(result.get('fusion_signals', []))} fusion signals, "
        f"{len(result.get('alerts', []))} alerts in {time.time()-t0:.1f}s"
    )
    return result


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
    participant_count: int = None,
):
    """Try to update session status, log warning on failure."""
    try:
        await update_session_status(
            session_id, status,
            duration_ms=duration_ms,
            speaker_count=speaker_count,
            participant_count=participant_count,
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

    # Ownership check — fall back to DEV_ORG_ID if user has no org assigned
    session = await get_session(session_id, current_user.get("org_id") or DEV_ORG_ID)
    if not session:
        raise HTTPException(404, "Session not found")

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

    # Step 2b: GraphRAG — extract matched chunks as graph entry points,
    # then run Neo4j tool selection + graph traversal enrichment in parallel.
    # All Neo4j paths are non-fatal: degrade to pgvector-only if unavailable.

    # Pull above-threshold chunks; these become graph entry points
    matched_chunks = []
    sources = []
    for row in rows:
        sim = float(row["similarity"])
        if sim < 0.35:
            continue
        matched_chunks.append({
            "type": row["chunk_type"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "similarity": sim,
        })
        sources.append({
            "type": row["chunk_type"],
            "text": row["text"][:200],
            "similarity": round(sim, 3),
        })

    _SEMANTIC_TOOLS = {
        "get_causal_chain", "get_topic_stress_correlation", "get_speaker_influence",
        "get_unresolved_objections", "get_conversation_arc", "get_signal_decomposition",
        "get_convergent_moments", "get_speaker_summary", "get_signal_timeline",
        "get_entity_network",
        "get_speaker_trend", "get_session_comparison",
        "get_video_behavioral_summary", "get_incongruence_moments",
    }

    async def _tool_query():
        """Path A/B: semantic tool selection → Cypher fallback."""
        try:
            from neo4j_semantic_layer import select_tool, execute_tool, search_graph_context_fallback
            tool_selection = await select_tool(question, session_id, history=body.history)
            tool_name = tool_selection.get("tool", "none")
            if tool_name and tool_name != "none" and tool_name in _SEMANTIC_TOOLS:
                params = tool_selection.get("params") or {}
                result = await execute_tool(tool_name, params, session_id)
                logger.info(f"[{session_id}] GraphRAG tool: {tool_name}")
                return result, f"semantic:{tool_name}"
            else:
                result = await search_graph_context_fallback(question, session_id)
                if result:
                    logger.info(f"[{session_id}] GraphRAG GPT-5 Cypher fallback activated")
                    return result, "gpt5_cypher_fallback"
        except Exception as e:
            logger.warning(f"[{session_id}] Neo4j tool query failed (non-fatal): {e}")
        return "", None

    async def _graph_enrichment():
        """GraphRAG traversal: walk graph edges from matched chunk entry points."""
        if not matched_chunks:
            return ""
        try:
            from neo4j_semantic_layer import enrich_chunks_with_graph
            return await enrich_chunks_with_graph(matched_chunks, session_id)
        except Exception as e:
            logger.warning(f"[{session_id}] Graph enrichment failed (non-fatal): {e}")
        return ""

    # Run tool selection and graph enrichment concurrently
    (graph_context, graph_source), enrichment_context = await asyncio.gather(
        _tool_query(), _graph_enrichment()
    )

    # Step 3: Build structured three-section context
    if not matched_chunks and not graph_context and not enrichment_context and not body.history:
        return {
            "answer": "I couldn't find relevant analysis data for that question. Try rephrasing or ask about specific speakers, signals, or moments.",
            "sources": [],
            "chunks_searched": len(rows),
        }

    context_sections = []

    if matched_chunks:
        vector_parts = [f"[{c['type']}] {c['text']}" for c in matched_chunks[:8]]
        context_sections.append(
            "## Vector Context (semantic similarity matches)\n" + "\n".join(vector_parts)
        )

    if enrichment_context:
        context_sections.append(
            "## Graph Enrichment (relationship traversals from matched chunks)\n" + enrichment_context
        )

    if graph_context:
        context_sections.append(
            "## Tool Query (targeted graph analysis)\n" + graph_context
        )
        sources.append({"type": "knowledge_graph", "text": graph_context[:200], "similarity": 1.0})

    context = "\n\n".join(context_sections)

    # Step 4: Generate answer with LLM
    from shared.utils.llm_client import acomplete

    system_prompt = (
        "You are NEXUS, a behavioural analysis assistant. Answer questions about "
        "meeting/call analysis using three types of context:\n\n"
        "1. **Vector Context** — semantically matched text chunks (transcripts, signal summaries, "
        "speaker profiles, time windows). Use for direct factual answers.\n"
        "2. **Graph Enrichment** — relationship traversals from matched chunks into the knowledge "
        "graph (causal chains, what signals appeared near a topic, what followed an event). "
        "Use for 'why' and 'what led to' questions.\n"
        "3. **Tool Query** — targeted graph analysis (speaker influence, unresolved objections, "
        "conversation arc, signal timelines). Use for high-level pattern questions.\n\n"
        "Rules:\n"
        "- Synthesise all three sources — don't repeat the same fact from multiple sections.\n"
        "- Reference specific timestamps (mm:ss format), speaker names, and signal values.\n"
        "- Frame behavioural observations as 'indicators suggest' not 'they were definitely'.\n"
        "- Present causal chains as sequences: 'A led to B which triggered C'.\n"
        "- Describe speaker influence as correlations, not causation.\n"
        "- Never claim to detect deception — only note incongruence between modalities.\n"
        "- If none of the context contains the answer, say so clearly."
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

    session = await get_session(session_id, current_user.get("org_id") or DEV_ORG_ID)
    if not session:
        raise HTTPException(404, "Session not found")

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


# ─────────────────────────────────────────────────────────
# SPEAKER REGISTRY — CRUD + Team Dashboard + Global Chat
# ─────────────────────────────────────────────────────────

_SPEAKER_SORT_COLS = frozenset(
    {"last_seen_at", "first_seen_at", "session_count", "display_name"}
)


@app.get("/speakers")
async def list_speakers(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="last_seen_at"),
    current_user: dict = Depends(get_current_user),
):
    """List all registered speakers for this org with aggregate stats."""
    if sort_by not in _SPEAKER_SORT_COLS:
        sort_by = "last_seen_at"

    pool = await get_pool()
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
        uuid.UUID(str(org_id)),
        limit,
        offset,
    )

    total = await pool.fetchval(
        "SELECT COUNT(*) FROM speakers_registry WHERE org_id = $1",
        uuid.UUID(str(org_id)),
    )

    return {
        "speakers": [
            {**dict(r),
             "display_name": r["display_name"] if (r["display_name"] and not _GENERIC_SPEAKER_LABEL_RE.match(r["display_name"])) else ""}
            for r in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/speakers/{registry_id}")
async def get_speaker_profile(
    registry_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get speaker profile with cross-session performance trends."""
    pool = await get_pool()
    org_id = current_user.get("org_id", DEV_ORG_ID)

    try:
        reg_uuid = uuid.UUID(registry_id)
    except ValueError:
        raise HTTPException(400, "Invalid registry_id")

    speaker = await pool.fetchrow(
        "SELECT * FROM speakers_registry WHERE id = $1 AND org_id = $2",
        reg_uuid, uuid.UUID(str(org_id)),
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
            COUNT(CASE WHEN sig.signal_type = 'facial_stress'          THEN 1 END) AS facial_stress_count,
            COUNT(CASE WHEN sig.signal_type = 'head_nod'               THEN 1 END) AS nod_count,
            COUNT(CASE WHEN sig.signal_type = 'head_shake'             THEN 1 END) AS shake_count,
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


@app.put("/speakers/{registry_id}")
async def update_speaker(
    registry_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """Update speaker display_name, role, company, email, or notes."""
    pool = await get_pool()
    org_id = current_user.get("org_id", DEV_ORG_ID)

    try:
        reg_uuid = uuid.UUID(registry_id)
    except ValueError:
        raise HTTPException(400, "Invalid registry_id")

    sets: list[str] = []
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
    params.append(uuid.UUID(str(org_id)))

    await pool.execute(
        f"UPDATE speakers_registry SET {', '.join(sets)} WHERE id = ${idx} AND org_id = ${idx + 1}",
        *params,
    )
    return {"success": True}


@app.post("/speakers/{registry_id}/merge")
async def merge_speakers(
    registry_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """
    Merge another speaker into this one.  All appearances from source are
    re-pointed to target; source registry entry is deleted.
    """
    pool = await get_pool()
    org_id = current_user.get("org_id", DEV_ORG_ID)
    org_uuid = uuid.UUID(str(org_id))

    try:
        target_uuid = uuid.UUID(registry_id)
    except ValueError:
        raise HTTPException(400, "Invalid registry_id")

    source_id = body.get("merge_from_id")
    if not source_id:
        raise HTTPException(400, "merge_from_id required")

    try:
        source_uuid = uuid.UUID(str(source_id))
    except ValueError:
        raise HTTPException(400, "Invalid merge_from_id")

    # Verify both speakers belong to the caller's org before merging
    target_check = await pool.fetchval(
        "SELECT id FROM speakers_registry WHERE id = $1 AND org_id = $2",
        target_uuid, org_uuid,
    )
    if not target_check:
        raise HTTPException(404, "Target speaker not found")

    source_check = await pool.fetchval(
        "SELECT id FROM speakers_registry WHERE id = $1 AND org_id = $2",
        source_uuid, org_uuid,
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


@app.post("/sessions/{session_id}/identify-speaker")
async def identify_speaker(
    session_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """
    Manually link a session speaker label to a registry entry.
    Body: { speaker_label, registry_id }  — OR —
          { speaker_label, display_name, role }  to create a new entry.
    """
    pool = await get_pool()
    org_id = current_user.get("org_id", DEV_ORG_ID)

    speaker_label = body.get("speaker_label")
    if not speaker_label:
        raise HTTPException(400, "speaker_label required")

    try:
        sess_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session_id")

    registry_id = body.get("registry_id")
    if registry_id:
        try:
            reg_uuid = uuid.UUID(str(registry_id))
        except ValueError:
            raise HTTPException(400, "Invalid registry_id")
    else:
        display_name = body.get("display_name", speaker_label)
        row = await pool.fetchrow(
            """
            INSERT INTO speakers_registry (org_id, display_name, role, session_count)
            VALUES ($1, $2, $3, 1)
            RETURNING id
            """,
            uuid.UUID(str(org_id)),
            display_name,
            body.get("role", ""),
        )
        reg_uuid = row["id"]

    speaker_row = await pool.fetchrow(
        "SELECT id FROM speakers WHERE session_id = $1 AND speaker_label = $2",
        sess_uuid, speaker_label,
    )
    speaker_db_id = speaker_row["id"] if speaker_row else None

    await pool.execute(
        """
        INSERT INTO speaker_appearances
            (registry_id, session_id, speaker_id, speaker_label, match_method, match_confidence)
        VALUES ($1, $2, $3, $4, 'manual', 1.0)
        ON CONFLICT (registry_id, session_id, speaker_label) DO UPDATE
            SET match_method    = 'manual',
                match_confidence = 1.0
        """,
        reg_uuid, sess_uuid, speaker_db_id, speaker_label,
    )

    return {"success": True, "registry_id": str(reg_uuid)}


@app.get("/team")
async def get_team_dashboard(
    days: int = Query(default=30, ge=7, le=365),
    current_user: dict = Depends(get_current_user),
):
    """Team performance dashboard — all speakers with aggregate metrics for the last N days."""
    pool = await get_pool()
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
                COUNT(DISTINCT ra.session_id)                                                      AS session_count,
                AVG(CASE WHEN sig.signal_type = 'vocal_stress_score'      THEN sig.value END)      AS avg_stress,
                AVG(CASE WHEN sig.signal_type = 'conversation_engagement'  THEN sig.value END)     AS avg_engagement,
                AVG(CASE WHEN sig.signal_type = 'rapport_indicator'        THEN sig.value END)     AS avg_rapport,
                AVG(CASE WHEN sig.signal_type = 'filler_detection'         THEN sig.value END)     AS avg_filler_rate,
                AVG(CASE WHEN sig.signal_type = 'power_language_score'     THEN sig.value END)     AS avg_power,
                AVG(CASE WHEN sig.signal_type = 'attention_level'          THEN sig.value END)     AS avg_attention
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
        uuid.UUID(str(org_id)),
        days,
    )

    return {"team": [dict(r) for r in rows], "period_days": days}


@app.get("/team/compare")
async def compare_speakers(
    speaker_a: str = Query(...),
    speaker_b: str = Query(...),
    days: int = Query(default=90, ge=7, le=365),
    current_user: dict = Depends(get_current_user),
):
    """Compare two speakers' metrics side by side."""
    pool = await get_pool()
    org_id = current_user.get("org_id", DEV_ORG_ID)
    org_uuid = uuid.UUID(str(org_id))

    for sid in (speaker_a, speaker_b):
        try:
            uuid.UUID(sid)
        except ValueError:
            raise HTTPException(400, f"Invalid speaker id: {sid}")

    _metrics_sql = """
        SELECT
            COUNT(DISTINCT sa.session_id)                                                      AS session_count,
            AVG(CASE WHEN sig.signal_type = 'vocal_stress_score'      THEN sig.value END)      AS avg_stress,
            AVG(CASE WHEN sig.signal_type = 'conversation_engagement'  THEN sig.value END)     AS avg_engagement,
            AVG(CASE WHEN sig.signal_type = 'rapport_indicator'        THEN sig.value END)     AS avg_rapport,
            AVG(CASE WHEN sig.signal_type = 'filler_detection'         THEN sig.value END)     AS avg_filler_rate,
            AVG(CASE WHEN sig.signal_type = 'power_language_score'     THEN sig.value END)     AS avg_power,
            COUNT(CASE WHEN sig.signal_type = 'head_nod'               THEN 1 END)             AS total_nods,
            COUNT(CASE WHEN sig.signal_type = 'head_shake'             THEN 1 END)             AS total_shakes,
            AVG(CASE WHEN sig.signal_type = 'attention_level'          THEN sig.value END)     AS avg_attention
        FROM   speaker_appearances sa
        JOIN   sessions s  ON s.id = sa.session_id
        JOIN   signals sig ON sig.session_id = s.id AND sig.speaker_id = sa.speaker_id
        WHERE  sa.registry_id = $1
          AND  s.created_at > NOW() - make_interval(days => $2)
    """

    _info_sql = "SELECT display_name, role FROM speakers_registry WHERE id = $1 AND org_id = $2"

    metrics_a_row, metrics_b_row, info_a, info_b = await asyncio.gather(
        pool.fetchrow(_metrics_sql, uuid.UUID(speaker_a), days),
        pool.fetchrow(_metrics_sql, uuid.UUID(speaker_b), days),
        pool.fetchrow(_info_sql, uuid.UUID(speaker_a), org_uuid),
        pool.fetchrow(_info_sql, uuid.UUID(speaker_b), org_uuid),
    )

    return {
        "speaker_a": {
            "info":    dict(info_a)    if info_a    else {},
            "metrics": dict(metrics_a_row) if metrics_a_row else {},
        },
        "speaker_b": {
            "info":    dict(info_b)    if info_b    else {},
            "metrics": dict(metrics_b_row) if metrics_b_row else {},
        },
        "period_days": days,
    }


@app.post("/chat/global")
async def global_chat(
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Ask questions across ALL sessions — team comparisons, trends,
    speaker performance, and coaching insights.
    """
    pool = await get_pool()
    org_id = current_user.get("org_id") or DEV_ORG_ID

    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    # Cross-session speaker aggregate context (org_id falls back to DEV_ORG_ID if user has no org)
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
        uuid.UUID(str(org_id)),
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
        uuid.UUID(str(org_id)),
    )

    speaker_context = "Registered speakers:\n"
    for s in speakers:
        name = s['display_name'] if (s['display_name'] and not _GENERIC_SPEAKER_LABEL_RE.match(s['display_name'])) else "unnamed speaker"
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
    except Exception as e:
        logger.error(f"Global chat LLM call failed: {e}")
        raise HTTPException(502, f"LLM generation failed: {e}")

    return {"answer": answer, "speakers_in_context": len(speakers)}


@app.post("/speakers/search-by-face")
async def search_speakers_by_face(
    face_image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a face photo → find all sessions where this person appeared.
    Delegates embedding extraction to the video agent's /embed-face endpoint
    (keeps InsightFace/ArcFace out of the gateway's dependency tree).
    """
    import httpx

    org_id = current_user.get("org_id", DEV_ORG_ID)

    img_bytes = await face_image.read()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{VIDEO_AGENT_URL}/embed-face",
                files={"image": (face_image.filename or "face.jpg", img_bytes, face_image.content_type or "image/jpeg")},
            )
        if resp.status_code == 503:
            raise HTTPException(503, "Face recognition model not available in video agent")
        if resp.status_code == 400:
            raise HTTPException(400, resp.json().get("detail", "No face detected in image"))
        resp.raise_for_status()
        payload   = resp.json()
        embedding = payload.get("embedding", [])
        if not embedding or not payload.get("detected"):
            raise HTTPException(400, "No face detected in image")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Video agent face embedding call failed: {exc}")

    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
    pool = await get_pool()

    matches = await pool.fetch("""
        SELECT sr.id, sr.display_name, sr.role, sr.company,
               sr.session_count, sr.first_seen_at, sr.last_seen_at,
               1 - (sr.face_embedding <=> $1::vector) AS similarity
        FROM   speakers_registry sr
        WHERE  sr.org_id = $2
          AND  sr.face_embedding IS NOT NULL
          AND  1 - (sr.face_embedding <=> $1::vector) > 0.45
        ORDER  BY sr.face_embedding <=> $1::vector
        LIMIT  10
    """, emb_str, uuid.UUID(str(org_id)))

    output = []
    for m in matches:
        appearances = await pool.fetch("""
            SELECT sa.session_id, s.title, s.created_at,
                   sa.match_method, sa.match_confidence
            FROM   speaker_appearances sa
            JOIN   sessions s ON s.id = sa.session_id
            WHERE  sa.registry_id = $1
            ORDER  BY s.created_at DESC
            LIMIT  20
        """, m["id"])

        thumb_row = await pool.fetchrow("""
            SELECT thumbnail FROM face_thumbnails
            WHERE registry_id = $1
            ORDER BY is_primary DESC, quality_score DESC
            LIMIT 1
        """, m["id"])

        import base64 as _b64
        output.append({
            "registry_id":  str(m["id"]),
            "display_name": m["display_name"] if (m["display_name"] and not _GENERIC_SPEAKER_LABEL_RE.match(m["display_name"])) else "",
            "role":         m["role"] or "",
            "company":      m["company"] or "",
            "similarity":   round(float(m["similarity"]), 3),
            "session_count": m["session_count"],
            "first_seen":   m["first_seen_at"].isoformat() if m["first_seen_at"] else None,
            "last_seen":    m["last_seen_at"].isoformat()  if m["last_seen_at"]  else None,
            "thumbnail_b64": _b64.b64encode(thumb_row["thumbnail"]).decode() if thumb_row else None,
            "sessions": [{
                "session_id":   str(a["session_id"]),
                "title":        a["title"],
                "date":         a["created_at"].isoformat() if a["created_at"] else None,
                "match_method": a["match_method"],
            } for a in appearances],
        })

    return {"matches": output, "query_faces_detected": 1}


@app.get("/speakers/{registry_id}/thumbnail")
async def get_speaker_thumbnail(
    request: StarletteRequest,
    registry_id: str,
    token: Optional[str] = Query(default=None),
):
    """
    Return the primary face thumbnail for a speaker as JPEG.
    Accepts JWT via Authorization: Bearer header OR ?token= query param.
    The query-param form is required for <img> elements which cannot set headers.
    """
    from fastapi.responses import Response as _Response
    # Auth: prefer Authorization header, fall back to ?token= for img elements
    auth_token = token
    if not auth_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]
    if not auth_token:
        raise HTTPException(401, "Unauthorized")
    verify_access_token(auth_token)
    pool = await get_pool()
    row = await pool.fetchrow("""
        SELECT thumbnail FROM face_thumbnails
        WHERE registry_id = $1
        ORDER BY is_primary DESC, quality_score DESC
        LIMIT 1
    """, uuid.UUID(registry_id))
    if not row:
        raise HTTPException(404, "No thumbnail available")
    return _Response(content=row["thumbnail"], media_type="image/jpeg")
