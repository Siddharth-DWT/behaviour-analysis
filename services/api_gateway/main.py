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
import logging
from pathlib import Path
from datetime import datetime, timezone
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
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.gateway")

# ── Agent URLs (configurable via environment) ──
VOICE_AGENT_URL = os.getenv("VOICE_AGENT_URL", "http://localhost:8002")
LANGUAGE_AGENT_URL = os.getenv("LANGUAGE_AGENT_URL", "http://localhost:8003")
FUSION_AGENT_URL = os.getenv("FUSION_AGENT_URL", "http://localhost:8004")

# ── Upload directory ──
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/recordings"))

# ── HTTP client timeout (Voice Agent with Whisper can be slow) ──
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "1800"))  # 30 minutes

app = FastAPI(
    title="NEXUS API Gateway",
    description="Central API for the NEXUS multi-agent behavioural analysis system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info("Starting NEXUS API Gateway...")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
               password_hash, is_active, created_at, last_login_at
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

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    file_size_mb = len(content) / (1024 * 1024)
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

    # ── Step 3: Voice Agent ──
    voice_result = None
    try:
        voice_result = await _call_voice_agent(session_id, str(file_path.resolve()), num_speakers=num_speakers, meeting_type=meeting_type)
        logger.info(
            f"[{session_id}] Voice Agent: "
            f"{voice_result.get('duration_seconds', 0):.0f}s, "
            f"{len(voice_result.get('signals', []))} signals"
        )
    except Exception as e:
        logger.error(f"[{session_id}] Voice Agent failed: {e}")
        await _try_update_status(session_id, "failed")
        raise HTTPException(502, f"Voice Agent failed: {e}")

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
            logger.info(f"[{session_id}] Language Agent: {len(language_signals)} signals")
        except Exception as e:
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

    # ── Step 5: Fusion Agent ──
    fusion_result = None
    fusion_signals = []
    alerts = []
    report = None

    try:
        fusion_result = await _call_fusion_agent(
            session_id=session_id,
            voice_signals=voice_signals,
            language_signals=language_signals,
            voice_summary=voice_summary,
            language_summary=language_summary,
            meeting_type=meeting_type,
        )
        fusion_signals = fusion_result.get("fusion_signals", [])
        alerts = fusion_result.get("alerts", [])
        report = fusion_result.get("report")
        logger.info(
            f"[{session_id}] Fusion Agent: "
            f"{len(fusion_signals)} signals, {len(alerts)} alerts"
        )
    except Exception as e:
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

    # ── Step 6: Mark session complete ──
    await _try_update_status(
        session_id, "completed",
        duration_ms=int(duration_seconds * 1000),
        speaker_count=speaker_count,
    )

    logger.info(f"[{session_id}] Pipeline complete")

    return SessionCreateResponse(
        session_id=session_id,
        status="completed",
        title=title,
        meeting_type=meeting_type,
        duration_seconds=duration_seconds,
        speaker_count=speaker_count,
        voice_signal_count=len(voice_signals),
        language_signal_count=len(language_signals),
        fusion_signal_count=len(fusion_signals),
        alert_count=len(alerts),
        report_generated=report_generated,
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
        if start_ms in seen_starts:
            continue
        seen_starts.add(start_ms)

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
