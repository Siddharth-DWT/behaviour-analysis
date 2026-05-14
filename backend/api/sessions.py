# backend/api/sessions.py
"""
NEXUS Backend — Session routes (/sessions/*, /quick-transcribe)
Ported from services/api_gateway/main.py.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid as _uuid_module
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form,
    HTTPException, Query, UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from core.auth import get_current_user, require_role, verify_access_token
from core.database import (
    DEV_ORG_ID,
    create_session,
    get_alerts,
    get_pool,
    get_report,
    get_session,
    get_signals,
    get_transcript,
    insert_signals,
    insert_transcript_segments,
    list_sessions,
    save_report,
    update_session_status,
    upsert_speakers,
)
from dependencies import get_db_pool, get_pipeline, get_redis_repo
from shared.models.requests import SessionListResponse

logger = logging.getLogger("nexus.backend.sessions")

router = APIRouter(tags=["sessions"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/recordings"))
OVERLAY_DIR = Path(os.getenv("OVERLAY_DIR", "data/overlays"))

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

_VIDEO_OVERLAY_TYPES = [
    "facial_emotion", "facial_stress", "facial_engagement",
    "smile_type", "valence_arousal",
    "head_nod", "head_shake", "posture", "body_lean",
    "body_fidgeting", "self_touch", "shoulder_tension",
    "head_body_incongruence", "gesture_animation", "body_mirroring",
    "face_region_touch", "arms_crossed", "finger_steepling",
    "head_supported", "hands_clasped", "cross_speaker_interaction",
    "posture_transition", "body_language_cluster",
    "lip_pursing",
    "gaze_direction_shift", "screen_contact", "sustained_distraction",
    "attention_level", "blink_rate_anomaly", "gaze_synchrony",
    "evaluation_cluster", "hidden_disagreement", "frustration_cluster",
    "hand_gesture", "arm_posture",
    "laughter",
    "tone_face_masking", "stress_suppression", "rapport_confirmation",
    "voice_face_alignment",
    "genuine_engagement", "active_disengagement", "emotional_suppression",
    "decision_engagement", "cognitive_overload", "conflict_escalation",
    "verbal_nonverbal_discordance", "peak_performance", "rapport_building",
    "dominance_display", "submission_signal", "deception_cluster",
    "rapport_evolution",
    "behavioral_shift", "adaptation_pattern", "fatigue_detection",
    "stress_recovery",
    "tension_cluster",
    "vocal_stress_score", "tone_classification", "filler_detection",
    "speech_rate_anomaly", "energy_level", "pitch_elevation_flag",
    "monotone_flag", "interruption_event",
    "sentiment_score",
]

_GENERIC_SPEAKER_LABEL_RE = re.compile(r'^(Speaker|Face)_\d+$')


def _speaker_grid_position(face_centre_x: float, face_centre_y: float) -> str:
    if face_centre_x < 0.01 and face_centre_y < 0.01:
        return ""
    col = "Left" if face_centre_x < 0.33 else ("Center" if face_centre_x < 0.66 else "Right")
    row = "Top" if face_centre_y < 0.5 else "Bottom"
    return f"{row}-{col}"


def _should_display_signal(sig: dict) -> bool:
    stype = sig.get("signal_type", "")
    value = sig.get("value") or 0.0
    conf  = sig.get("confidence") or 0.0
    agent = sig.get("agent", "")

    if stype == "vocal_stress_score":
        return value > 0.50
    if stype == "sentiment_score":
        return abs(value) > 0.55
    if agent == "video":
        return conf >= 0.20
    if agent == "fusion":
        duration_ms = (sig.get("window_end_ms") or 0) - (sig.get("window_start_ms") or 0)
        if duration_ms > 120_000:
            return False
        return conf >= 0.10
    return True


def _serialise_signals(signals: list[dict]) -> list[dict]:
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


# ── POST /sessions ─────────────────────────────────────────────────────────────

@router.post("/sessions")
async def create_session_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(default=""),
    meeting_type: str = Form(default="sales_call"),
    config: str = Form(default="{}"),
    current_user: dict = Depends(require_role("member")),
    pipeline=Depends(get_pipeline),
    pool=Depends(get_db_pool),
):
    """Upload an audio/video file and start the analysis pipeline in the background."""
    filename = file.filename or "upload.wav"
    suffix = Path(filename).suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}")

    session_id = str(_uuid_module.uuid4())
    file_name  = f"{session_id}{suffix}"
    file_path  = UPLOAD_DIR / file_name
    MAX_FILE_SIZE = 300 * 1024 * 1024

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_size = 0
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                f.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(413, "File too large. Maximum size is 300 MB.")
            f.write(chunk)

    if not title:
        title = Path(filename).stem

    try:
        config_dict = json.loads(config) if config and config.strip() else {}
    except json.JSONDecodeError:
        config_dict = {}

    transcription_config = config_dict.get("transcription", {})
    analysis_config      = config_dict.get("analysis", {})
    if not meeting_type or meeting_type == "sales_call":
        meeting_type = config_dict.get("meeting_type", meeting_type)
    num_speakers = config_dict.get("num_speakers") or None

    try:
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
    except Exception as exc:
        logger.warning("[%s] DB create failed (continuing): %s", session_id, exc)

    _video_path = str(file_path.resolve()) if suffix in {".mp4", ".webm"} else None
    background_tasks.add_task(
        pipeline.run,
        session_id=session_id,
        file_path=str(file_path.resolve()),
        video_path=_video_path,
        meeting_type=meeting_type,
        num_speakers=num_speakers,
        pool=pool,
        org_id=current_user.get("org_id", DEV_ORG_ID),
        user_id=current_user["id"],
        run_behavioural=analysis_config.get("run_behavioural", True),
    )

    return {
        "session_id":   session_id,
        "status":       "processing",
        "title":        title,
        "meeting_type": meeting_type,
    }


# ── POST /quick-transcribe ─────────────────────────────────────────────────────

@router.post("/quick-transcribe")
async def quick_transcribe_endpoint(
    file: UploadFile = File(...),
    config: str = Form(default="{}"),
    _current_user: dict = Depends(require_role("member")),
    pipeline=Depends(get_pipeline),
):
    """Lightweight transcription — no session, no DB, no report."""
    filename = file.filename or "upload.wav"
    suffix = Path(filename).suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    try:
        config_dict = json.loads(config) if config and config.strip() else {}
    except json.JSONDecodeError:
        config_dict = {}

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOAD_DIR / f"qt_{_uuid_module.uuid4()}{suffix}"
    MAX_FILE_SIZE = 300 * 1024 * 1024

    file_size = 0
    try:
        with open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large. Maximum 300 MB.")
                f.write(chunk)

        try:
            result = await pipeline.run_quick_transcribe(
                file_path=str(temp_path.resolve()),
                session_id=str(_uuid_module.uuid4()),
                config=config_dict,
            )
            return result
        except NotImplementedError:
            raise HTTPException(501, "Quick transcribe pipeline: Phase 3 pending")

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


# ── GET /sessions ──────────────────────────────────────────────────────────────

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions_endpoint(
    limit: int = Query(default=25, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    meeting_type: Optional[str] = Query(default=None),
    session_type: Optional[str] = Query(default=None),
    current_user: dict = Depends(get_current_user),
):
    """List sessions with pagination and optional filters."""
    try:
        sessions, total = await list_sessions(
            limit=limit, offset=offset,
            status=status, meeting_type=meeting_type,
            session_type=session_type, user_id=None,
        )
    except Exception as exc:
        logger.error("Failed to list sessions: %s", exc)
        raise HTTPException(500, "Database query failed")

    return SessionListResponse(sessions=sessions, total=total, limit=limit, offset=offset)


# ── GET /sessions/{id} ─────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    _: dict = Depends(get_current_user),
):
    """Get session detail including signals, alerts, and unified states."""
    try:
        _uuid_module.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    signals = [s for s in await get_signals(session_id, limit=50000) if _should_display_signal(s)]
    session_alerts = await get_alerts(session_id)
    report     = await get_report(session_id)
    transcript = await get_transcript(session_id)

    signals_by_agent: dict[str, int] = {}
    for s in signals:
        agent = s.get("agent", "unknown")
        signals_by_agent[agent] = signals_by_agent.get(agent, 0) + 1

    return {
        "session":              session,
        "signal_count":         len(signals),
        "signals_by_agent":     signals_by_agent,
        "alerts":               session_alerts,
        "alert_count":          len(session_alerts),
        "has_report":           report is not None,
        "transcript_segment_count": len(transcript),
    }


# ── GET /sessions/{id}/signals ─────────────────────────────────────────────────

@router.get("/sessions/{session_id}/signals")
async def get_session_signals(
    session_id: str,
    agent: Optional[str] = Query(default=None),
    signal_type: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=50000),
    offset: int = Query(default=0, ge=0),
    _: dict = Depends(get_current_user),
):
    """Get signals for a session with optional filtering."""
    try:
        _uuid_module.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    signals = await get_signals(
        session_id, agent=agent, signal_type=signal_type, limit=limit, offset=offset
    )
    signals = [s for s in signals if _should_display_signal(s)]

    return {
        "session_id": session_id,
        "signals":    signals,
        "count":      len(signals),
        "filters":    {"agent": agent, "signal_type": signal_type},
    }


# ── GET /sessions/{id}/report ──────────────────────────────────────────────────

@router.get("/sessions/{session_id}/report")
async def get_session_report(
    session_id: str,
    regenerate: bool = Query(default=False),
    _: dict = Depends(get_current_user),
):
    """Get the narrative report for a session."""
    try:
        _uuid_module.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if not regenerate:
        existing = await get_report(session_id)
        if existing:
            return {"session_id": session_id, "report": existing}
        raise HTTPException(404, "No report found for this session")

    # Regenerate via Fusion Agent — Phase 3
    raise HTTPException(501, "Report regeneration via pipeline: Phase 3 pending")


# ── GET /sessions/{id}/progress ────────────────────────────────────────────────

@router.get("/sessions/{session_id}/progress")
async def get_session_progress(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    redis_repo=Depends(get_redis_repo),
):
    """Return the current pipeline step for a session being processed."""
    state = await redis_repo.get_session_state(session_id)
    if state:
        return {"pipeline_step": state.get("current_step"), "status": state.get("status")}
    return {"pipeline_step": None}


# ── GET /sessions/{id}/transcript ─────────────────────────────────────────────

@router.get("/sessions/{session_id}/transcript")
async def get_session_transcript(
    session_id: str,
    _: dict = Depends(get_current_user),
):
    """Get transcript segments for a session."""
    try:
        _uuid_module.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    segments = await get_transcript(session_id)
    return {"session_id": session_id, "segments": segments, "count": len(segments)}


# ── GET /sessions/{id}/video-signals ──────────────────────────────────────────

@router.get("/sessions/{session_id}/video-signals")
async def get_video_signals(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Return video + fusion signals for playback overlay, ordered by window start."""
    try:
        _uuid_module.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    session = await get_session(session_id, current_user.get("org_id") or DEV_ORG_ID)
    if not session:
        raise HTTPException(404, "Session not found")

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
        _uuid_module.UUID(session_id),
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

    # Filter session-spanning temporal signals (>120 s)
    signals = [s for s in signals if (s["end_ms"] - s["start_ms"]) <= 120_000]

    # Merge duplicate speaker labels that share the same registry identity.
    # Canonical preference: Speaker_* before Face_*, then alphabetical.
    registry_groups: dict[str, list[str]] = {}
    for sig in signals:
        reg = sig.get("registry_id")
        spk = sig.get("speaker_id")
        if reg and spk:
            registry_groups.setdefault(reg, []).append(spk)

    canonical_by_registry: dict[str, str] = {
        reg: sorted(set(labels), key=lambda x: (0 if x.startswith("Speaker_") else 1, x))[0]
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

    # Build speaker → grid_position fallback map from face-coord signals
    speaker_positions: dict[str, str] = {}
    for sig in signals:
        spk = sig["speaker_id"]
        pos = sig["metadata"].get("grid_position", "")
        if spk and pos and spk not in speaker_positions:
            speaker_positions[spk] = pos

    for sig in signals:
        if not sig["metadata"].get("grid_position"):
            spk = sig["speaker_id"]
            if spk and spk in speaker_positions:
                sig["metadata"]["grid_position"] = speaker_positions[spk]

    return {"session_id": session_id, "signals": signals}


# ── GET /sessions/{id}/video-speakers ─────────────────────────────────────────

@router.get("/sessions/{session_id}/video-speakers")
async def get_video_speakers(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Return all speakers/faces in a video session with identity info."""
    try:
        sess_uuid = _uuid_module.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    session = await get_session(session_id, current_user.get("org_id") or DEV_ORG_ID)
    if not session:
        raise HTTPException(404, "Session not found")

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
    """, sess_uuid)

    pos_rows = await pool.fetch("""
        SELECT DISTINCT ON (sp.speaker_label) sp.speaker_label, s.metadata
        FROM   signals s
        JOIN   speakers sp ON sp.id = s.speaker_id
        WHERE  s.session_id = $1
          AND  s.agent = 'video'
          AND  s.metadata IS NOT NULL
        ORDER  BY sp.speaker_label, s.window_start_ms
    """, sess_uuid)

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

    # Canonical label: Speaker_* preferred, then alphabetical
    reg_label_groups: dict[str, list[str]] = {}
    for r in rows:
        reg = str(r["registry_id"]) if r["registry_id"] else ""
        lbl = r["speaker_label"] or ""
        if reg and lbl:
            reg_label_groups.setdefault(reg, []).append(lbl)

    canonical_label_for_reg: dict[str, str] = {
        reg: sorted(set(labels), key=lambda x: (0 if x.startswith("Speaker_") else 1, x))[0]
        for reg, labels in reg_label_groups.items()
    }

    speakers = []
    seen_registries: set[str] = set()
    for r in rows:
        registry_id = str(r["registry_id"]) if r["registry_id"] else ""
        slabel = r["speaker_label"] or ""

        if registry_id and registry_id in seen_registries:
            continue
        if registry_id:
            seen_registries.add(registry_id)

        canonical = canonical_label_for_reg.get(registry_id, slabel)
        rname = r["display_name"] or ""
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


# ── GET /sessions/{id}/video ───────────────────────────────────────────────────

@router.get("/sessions/{session_id}/video")
async def get_session_video(
    request: StarletteRequest,
    session_id: str,
    token: Optional[str] = Query(default=None),
):
    """Stream the session media file. Accepts JWT via header or ?token= for <video> elements."""
    try:
        _uuid_module.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    auth_token = token
    if not auth_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]
    if not auth_token:
        raise HTTPException(401, "Unauthorized")

    if not verify_access_token(auth_token):
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
    return FileResponse(str(video_path), media_type=media_type, headers={"Accept-Ranges": "bytes"})


# ── GET/HEAD /sessions/{id}/video/annotated ────────────────────────────────────

@router.api_route("/sessions/{session_id}/video/annotated", methods=["GET", "HEAD"])
async def get_annotated_video(
    request: StarletteRequest,
    session_id: str,
    token: Optional[str] = Query(default=None),
):
    """Stream the landmark-annotated video produced by the video agent."""
    try:
        _uuid_module.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "Session not found")

    auth_token = token
    if not auth_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]
    if not auth_token:
        raise HTTPException(401, "Unauthorized")

    if not verify_access_token(auth_token):
        raise HTTPException(401, "Invalid or expired token")

    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    overlay_path = OVERLAY_DIR / f"{session_id}_annotated.webm"
    media_type = "video/webm"
    if not overlay_path.exists():
        overlay_path = OVERLAY_DIR / f"{session_id}_annotated.mp4"
        media_type = "video/mp4"
    if not overlay_path.exists():
        raise HTTPException(404, "Annotated video not available for this session")

    file_size = overlay_path.stat().st_size

    if request.method == "HEAD":
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


# ── POST /sessions/{id}/identify-speaker ──────────────────────────────────────

@router.post("/sessions/{session_id}/identify-speaker")
async def identify_speaker(
    session_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Manually link a session speaker label to a registry entry."""
    org_id = current_user.get("org_id", DEV_ORG_ID)

    speaker_label = body.get("speaker_label")
    if not speaker_label:
        raise HTTPException(400, "speaker_label required")

    try:
        sess_uuid = _uuid_module.UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session_id")

    registry_id = body.get("registry_id")
    if registry_id:
        try:
            reg_uuid = _uuid_module.UUID(str(registry_id))
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
            _uuid_module.UUID(str(org_id)),
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
            SET match_method     = 'manual',
                match_confidence = 1.0
        """,
        reg_uuid, sess_uuid, speaker_db_id, speaker_label,
    )

    return {"success": True, "registry_id": str(reg_uuid)}


# ── Session Chat (RAG) ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


@router.post("/sessions/{session_id}/chat")
async def chat_with_session(
    session_id: str,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Ask a question about a session's analysis using RAG + GraphRAG."""
    import asyncio

    session = await get_session(session_id, current_user.get("org_id") or DEV_ORG_ID)
    if not session:
        raise HTTPException(404, "Session not found")

    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    from shared.utils.llm_client import get_embedding, acomplete
    question_embedding = await get_embedding(question)
    if not question_embedding:
        raise HTTPException(500, "Embedding generation failed — check LLM configuration")

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
    except Exception as exc:
        logger.warning("[%s] pgvector chat search failed (non-fatal): %s", session_id, exc)
        rows = []

    _SEMANTIC_TOOLS = {
        "get_causal_chain", "get_topic_stress_correlation", "get_speaker_influence",
        "get_unresolved_objections", "get_conversation_arc", "get_signal_decomposition",
        "get_convergent_moments", "get_speaker_summary", "get_signal_timeline",
        "get_entity_network", "get_speaker_trend", "get_session_comparison",
        "get_video_behavioral_summary", "get_incongruence_moments",
    }

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
        sources.append({"type": row["chunk_type"], "text": row["text"][:200], "similarity": round(sim, 3)})

    async def _tool_query():
        try:
            from core.neo4j_semantic_layer import select_tool, execute_tool, search_graph_context_fallback
            tool_selection = await select_tool(question, session_id, history=body.history)
            tool_name = tool_selection.get("tool", "none")
            if tool_name and tool_name != "none" and tool_name in _SEMANTIC_TOOLS:
                params = tool_selection.get("params") or {}
                result = await execute_tool(tool_name, params, session_id)
                return result, f"semantic:{tool_name}"
            else:
                result = await search_graph_context_fallback(question, session_id)
                if result:
                    return result, "gpt5_cypher_fallback"
        except Exception as exc:
            logger.warning("[%s] Neo4j tool query failed (non-fatal): %s", session_id, exc)
        return "", None

    async def _graph_enrichment():
        if not matched_chunks:
            return ""
        try:
            from core.neo4j_semantic_layer import enrich_chunks_with_graph
            return await enrich_chunks_with_graph(matched_chunks, session_id)
        except Exception as exc:
            logger.warning("[%s] Graph enrichment failed (non-fatal): %s", session_id, exc)
        return ""

    (graph_context, graph_source), enrichment_context = await asyncio.gather(
        _tool_query(), _graph_enrichment()
    )

    if not matched_chunks and not graph_context and not enrichment_context and not body.history:
        return {
            "answer": "I couldn't find relevant analysis data for that question. Try rephrasing or ask about specific speakers, signals, or moments.",
            "sources": [],
            "chunks_searched": len(rows),
        }

    context_sections = []
    if matched_chunks:
        vector_parts = [f"[{c['type']}] {c['text']}" for c in matched_chunks[:8]]
        context_sections.append("## Vector Context (semantic similarity matches)\n" + "\n".join(vector_parts))
    if enrichment_context:
        context_sections.append("## Graph Enrichment (relationship traversals from matched chunks)\n" + enrichment_context)
    if graph_context:
        context_sections.append("## Tool Query (targeted graph analysis)\n" + graph_context)
        sources.append({"type": "knowledge_graph", "text": graph_context[:200], "similarity": 1.0})

    context = "\n\n".join(context_sections)

    system_prompt = (
        "You are NEXUS, a behavioural analysis assistant. Answer questions about "
        "meeting/call analysis using three types of context:\n\n"
        "1. **Vector Context** — semantically matched text chunks.\n"
        "2. **Graph Enrichment** — relationship traversals from matched chunks.\n"
        "3. **Tool Query** — targeted graph analysis.\n\n"
        "Rules:\n"
        "- Synthesise all sources — don't repeat the same fact.\n"
        "- Reference specific timestamps (mm:ss), speaker names, and signal values.\n"
        "- Frame observations as 'indicators suggest', not 'they were definitely'.\n"
        "- Never claim to detect deception — only note incongruence between modalities.\n"
        "- If none of the context contains the answer, say so clearly."
    )

    user_prompt = f"Context from session analysis:\n{context}\n\nQuestion: {question}"
    if body.history:
        history_text = "\n".join(
            f"{m.get('role', 'user').title()}: {m.get('content', '')}"
            for m in body.history[-4:]
        )
        user_prompt = f"Previous conversation:\n{history_text}\n\n{user_prompt}"

    try:
        answer = await acomplete(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=600, model="gpt-4o")
    except Exception as exc:
        logger.error("Chat LLM call failed: %s", exc)
        raise HTTPException(502, f"LLM generation failed: {exc}")

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
    except Exception as exc:
        logger.warning("Failed to persist chat messages: %s", exc)

    return {
        "answer": answer,
        "sources": top_sources,
        "chunks_searched": len(rows),
        "graph_source": graph_source,
    }


@router.get("/sessions/{session_id}/chat")
async def get_chat_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    pool=Depends(get_db_pool),
):
    """Get persisted chat history for a session."""
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
