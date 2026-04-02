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
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# isort: split
# Project root must be on sys.path before shared.* imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# isort: split
from shared.models.requests import SessionCreateResponse, SessionListResponse

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
CONVERSATION_AGENT_URL = os.getenv("CONVERSATION_AGENT_URL", "http://localhost:8006")
FUSION_AGENT_URL = os.getenv("FUSION_AGENT_URL", "http://localhost:8004")

# ── Upload directory ──
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/recordings"))
RECORDING_RETENTION_DAYS = int(os.getenv("RECORDING_RETENTION_DAYS", "3"))

# ── HTTP client timeout (Voice Agent with Whisper can be slow) ──
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "1800"))  # 30 minutes

# ── CORS origins (comma-separated list) ──
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3006,http://localhost:5173").split(",")

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

    logger.info("API Gateway ready.")


@app.on_event("shutdown")
async def shutdown():
    await close_pool()


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


# ─────────────────────────────────────────────────────────
# AUTH — Endpoints (public)
# ─────────────────────────────────────────────────────────

@app.post("/auth/signup")
async def signup(body: SignupRequest):
    """Register a new user account."""
    email = validate_email(body.email)
    validate_password(body.password)

    pool = await get_pool()

    # Check duplicate email
    existing = await pool.fetchrow(
        "SELECT id FROM users WHERE email = $1", email
    )
    if existing:
        raise HTTPException(409, "Email already registered")

    password_hash = hash_password(body.password)

    row = await pool.fetchrow(
        """
        INSERT INTO users (org_id, email, name, full_name, password_hash, company, role)
        VALUES ($1, $2, $3, $3, $4, $5, 'member')
        RETURNING id, email, full_name, role, company, avatar_url, org_id, created_at
        """,
        DEV_ORG_ID, email, body.full_name, password_hash, body.company,
    )

    user_id = str(row["id"])

    # Email verification flow
    if is_email_configured():
        token = generate_verification_token()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        await pool.execute(
            """
            INSERT INTO email_verifications (user_id, token, expires_at)
            VALUES ($1, $2, $3)
            """,
            row["id"], token, expires_at,
        )
        await send_verification_email(email, body.full_name, token)

        return {
            "user": {
                "id": user_id,
                "email": row["email"],
                "full_name": row["full_name"],
                "role": row["role"],
                "company": row["company"],
            },
            "requires_verification": True,
            "message": "Verification email sent. Please check your inbox.",
        }
    else:
        # No email provider — auto-verify and issue tokens immediately
        logger.warning("Email not configured — auto-verifying new user")
        await pool.execute(
            "UPDATE users SET is_verified = true WHERE id = $1", row["id"],
        )
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


# ─────────────────────────────────────────────────────────
# POST /sessions — Upload + full pipeline
# ─────────────────────────────────────────────────────────



@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session_endpoint(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    meeting_type: str = Form(default="sales_call"),
    num_speakers: Optional[int] = Form(default=None),
    current_user: dict = Depends(require_role("member")),
):
    """
    Upload an audio file and run the full analysis pipeline:
    1. Save file to disk
    2. Create session in PostgreSQL
    3. Call Voice Agent → get signals + transcript
    4. Call Language Agent → get language signals
    5. Call Fusion Agent → get fusion signals, unified states, alerts, report
    6. Persist everything to PostgreSQL
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

    # Stream to disk with size check — avoid reading entire file into memory
    file_size = 0
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
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

    # ── Step 2: Create session in DB ──
    try:
        session = await create_session(
            title=title,
            session_type="recording",
            meeting_type=meeting_type,
            media_url=str(file_path.resolve()),
            user_id=current_user["id"],
        )
        session_id = str(session["id"])
        await update_session_status(session_id, "processing")
        logger.info(f"[{session_id}] Session created in DB")
    except Exception as e:
        logger.warning(f"[{session_id}] DB create failed (continuing without DB): {e}")

    # ── Step 3: Voice Agent (with retry — this is the critical agent) ──
    voice_result = None
    try:
        voice_result = await _call_with_retry(
            lambda: _call_voice_agent(session_id, str(file_path.resolve()), num_speakers=num_speakers, meeting_type=meeting_type)
        )
        logger.info(
            f"[{session_id}] Voice Agent: "
            f"{voice_result.get('duration_seconds', 0):.0f}s, "
            f"{len(voice_result.get('signals', []))} signals"
        )
    except Exception as e:
        logger.error(f"[{session_id}] Voice Agent failed: {e}")
        await _try_update_status(session_id, "failed")
        raise HTTPException(502, f"Voice Agent failed: {e}")

    if not isinstance(voice_result, dict):
        logger.error(f"[{session_id}] Voice Agent returned unexpected type: {type(voice_result)}")
        voice_result = {}

    duration_seconds = voice_result.get("duration_seconds", 0)
    voice_signals = voice_result.get("signals", [])
    voice_speakers = voice_result.get("speakers", [])
    voice_summary = voice_result.get("summary", {})
    speaker_count = len(voice_speakers)

    # Persist speakers
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

    # ── Step 4: Language Agent ──
    language_result = None
    language_signals = []
    language_summary = {}

    # Build transcript segments from voice result
    transcript_segments = _extract_transcript_segments(voice_result)

    if transcript_segments:
        # Persist transcript
        try:
            await insert_transcript_segments(session_id, transcript_segments, speaker_map)
        except Exception as e:
            logger.warning(f"[{session_id}] Transcript persist failed: {e}")

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

        # Persist language signals
        if language_signals:
            try:
                count = await insert_signals(session_id, language_signals, speaker_map)
                logger.info(f"[{session_id}] Persisted {count} language signals")
            except Exception as e:
                logger.warning(f"[{session_id}] Language signal persist failed: {e}")
    else:
        logger.warning(f"[{session_id}] No transcript segments — skipping Language Agent")

    # ── Step 4b: Conversation Agent ──
    conversation_result = None
    conversation_signals = []
    conversation_summary = {}

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

        # Persist conversation signals
        if conversation_signals:
            try:
                count = await insert_signals(session_id, conversation_signals, speaker_map)
                logger.info(f"[{session_id}] Persisted {count} conversation signals")
            except Exception as e:
                logger.warning(f"[{session_id}] Conversation signal persist failed: {e}")
    else:
        logger.info(f"[{session_id}] < 2 speakers — skipping Conversation Agent")

    # ── Step 5: Fusion Agent ──
    fusion_result = None
    fusion_signals = []
    alerts = []
    report = None

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

    # Persist report
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

    # Cleanup old recordings in background
    _cleanup_old_recordings()

    return SessionCreateResponse(
        session_id=session_id,
        status=final_status,
        title=title,
        meeting_type=meeting_type,
        duration_seconds=duration_seconds,
        speaker_count=speaker_count,
        voice_signal_count=len(voice_signals),
        language_signal_count=len(language_signals),
        conversation_signal_count=len(conversation_signals),
        fusion_signal_count=len(fusion_signals),
        alert_count=len(alerts),
        report_generated=report_generated,
        agent_status=agent_status,
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
# GET /sessions/{id}/transcript — Transcript
# ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get transcript segments for a session."""
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


async def _call_voice_agent(session_id: str, file_path: str, num_speakers: Optional[int] = None, meeting_type: str = "sales_call") -> dict:
    """Call Voice Agent POST /analyse with file path."""
    payload = {
        "file_path": file_path,
        "session_id": session_id,
        "meeting_type": meeting_type,
    }
    if num_speakers is not None:
        payload["num_speakers"] = num_speakers
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
