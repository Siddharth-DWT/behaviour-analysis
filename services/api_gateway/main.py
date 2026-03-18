"""
NEXUS API Gateway
Central entry point for the NEXUS system. Accepts audio uploads, orchestrates
the Voice → Language → Fusion pipeline, persists results to PostgreSQL,
and serves session data to the React dashboard.

Endpoints:
  POST /sessions                → Upload audio, trigger full analysis pipeline
  GET  /sessions                → List sessions (paginated)
  GET  /sessions/{id}           → Session detail with signals + alerts
  GET  /sessions/{id}/signals   → Signals for a session (filterable)
  GET  /sessions/{id}/report    → Get or generate narrative report
  GET  /sessions/{id}/transcript→ Get transcript segments
  GET  /health                  → Health check
"""
import os
import sys
import uuid
import time
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.gateway")

# ── Agent URLs (configurable via environment) ──
VOICE_AGENT_URL = os.getenv("VOICE_AGENT_URL", "http://localhost:8001")
LANGUAGE_AGENT_URL = os.getenv("LANGUAGE_AGENT_URL", "http://localhost:8002")
FUSION_AGENT_URL = os.getenv("FUSION_AGENT_URL", "http://localhost:8007")

# ── Upload directory ──
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/recordings"))

# ── HTTP client timeout (Voice Agent with Whisper can be slow) ──
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "300"))  # 5 minutes

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
# POST /sessions — Upload + full pipeline
# ─────────────────────────────────────────────────────────

class SessionCreateResponse(BaseModel):
    session_id: str
    status: str
    title: str
    meeting_type: str
    duration_seconds: Optional[float] = None
    speaker_count: Optional[int] = None
    voice_signal_count: int = 0
    language_signal_count: int = 0
    fusion_signal_count: int = 0
    alert_count: int = 0
    report_generated: bool = False


@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session_endpoint(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    meeting_type: str = Form(default="sales_call"),
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
        )
        session_id = str(session["id"])
        await update_session_status(session_id, "processing")
        logger.info(f"[{session_id}] Session created in DB")
    except Exception as e:
        logger.warning(f"[{session_id}] DB create failed (continuing without DB): {e}")

    # ── Step 3: Voice Agent ──
    voice_result = None
    try:
        voice_result = await _call_voice_agent(session_id, str(file_path.resolve()))
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

    # Persist report
    report_generated = False
    if report:
        try:
            await save_report(
                session_id=session_id,
                content=report,
                narrative=report.get("executive_summary", ""),
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

class SessionListResponse(BaseModel):
    sessions: list[dict]
    total: int
    limit: int
    offset: int


@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions_endpoint(
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    meeting_type: Optional[str] = Query(default=None),
):
    """List sessions with pagination and optional filters."""
    try:
        sessions, total = await list_sessions(
            limit=limit,
            offset=offset,
            status=status,
            meeting_type=meeting_type,
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
async def get_session_detail(session_id: str):
    """Get session detail including signals, alerts, and unified states."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Fetch related data in parallel-ish
    signals = await get_signals(session_id, limit=500)
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
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Get signals for a session with optional filtering by agent/type."""
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
):
    """
    Get the narrative report for a session.
    If no report exists (or regenerate=True), triggers Fusion Agent to generate one.
    """
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Return existing report unless regenerate requested
    if not regenerate:
        existing = await get_report(session_id)
        if existing:
            return {"session_id": session_id, "report": existing}

    # Generate new report via Fusion Agent
    signals = await get_signals(session_id, limit=500)

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
async def get_session_transcript(session_id: str):
    """Get transcript segments for a session."""
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
# Agent call helpers
# ─────────────────────────────────────────────────────────

async def _call_voice_agent(session_id: str, file_path: str) -> dict:
    """Call Voice Agent POST /analyse with file path."""
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            f"{VOICE_AGENT_URL}/analyse",
            json={
                "file_path": file_path,
                "session_id": session_id,
            },
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
