# backend/api/uploads.py
"""
NEXUS Backend — Chunked upload routes (/uploads/*)
Ported from services/api_gateway/main.py lines ~984–1202.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid as _uuid_module
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi import BackgroundTasks
from pydantic import BaseModel

from core.auth import require_role
from core.database import (
    DEV_ORG_ID,
    create_session,
    update_session_status,
)
from dependencies import get_db_pool, get_pipeline

logger = logging.getLogger("nexus.backend.uploads")

router = APIRouter(tags=["uploads"])

UPLOAD_DIR      = Path(os.getenv("UPLOAD_DIR",       "data/recordings"))
CHUNK_UPLOAD_DIR = Path(os.getenv("CHUNK_UPLOAD_DIR", "data/chunks"))
CHUNK_SIZE_BYTES = 10 * 1024 * 1024          # 10 MB per chunk
MAX_UPLOAD_SIZE  = 2 * 1024 * 1024 * 1024    # 2 GB
UPLOAD_EXPIRY_HOURS = 24

# Process-local session state — upload_id → metadata dict
_upload_sessions: dict[str, dict] = {}


class ChunkedUploadInitRequest(BaseModel):
    filename:     str
    file_size:    int
    chunk_size:   int  = CHUNK_SIZE_BYTES
    meeting_type: str  = "sales_call"
    title:        str  = ""
    config:       str  = "{}"


# ── POST /uploads/init ─────────────────────────────────────────────────────────

@router.post("/init")
async def init_chunked_upload(
    body: ChunkedUploadInitRequest,
    current_user: dict = Depends(require_role("member")),
):
    """Step 1 of 3 — initialise a chunked upload session."""
    suffix = Path(body.filename).suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed))}")
    if body.file_size <= 0:
        raise HTTPException(400, "Invalid file size")
    if body.file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Maximum {MAX_UPLOAD_SIZE // (1024**3)} GB.")

    upload_id    = str(_uuid_module.uuid4())
    chunk_size   = max(1, body.chunk_size)
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
        "[upload:%s] init: %s (%.1f MB, %d chunks)",
        upload_id, body.filename, body.file_size / 1024 / 1024, total_chunks,
    )
    return {"upload_id": upload_id, "chunk_size": chunk_size, "total_chunks": total_chunks}


# ── POST /uploads/{id}/chunk ───────────────────────────────────────────────────

@router.post("/{upload_id}/chunk")
async def upload_chunk(
    upload_id:    str,
    chunk_number: int        = Form(...),
    chunk:        UploadFile = File(...),
    current_user: dict       = Depends(require_role("member")),
):
    """Step 2 of 3 — upload one chunk. Chunks may arrive out of order."""
    session = _upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(404, "Upload session not found or expired")
    if session["user_id"] != current_user["id"]:
        raise HTTPException(403, "Not your upload session")
    if chunk_number < 0 or chunk_number >= session["total_chunks"]:
        raise HTTPException(400, f"Invalid chunk_number {chunk_number} (total={session['total_chunks']})")

    data = await chunk.read()

    if chunk_number < session["total_chunks"] - 1:
        if len(data) != session["chunk_size"]:
            raise HTTPException(
                400,
                f"Chunk {chunk_number} size mismatch: got {len(data)}, expected {session['chunk_size']}",
            )

    chunk_path = CHUNK_UPLOAD_DIR / upload_id / f"chunk_{chunk_number:06d}"
    with open(chunk_path, "wb") as f:
        f.write(data)

    session["received_chunks"].add(chunk_number)
    received = len(session["received_chunks"])
    logger.info(
        "[upload:%s] chunk %d/%d (%d/%d total)",
        upload_id, chunk_number, session["total_chunks"] - 1, received, session["total_chunks"],
    )

    return {
        "chunk_number": chunk_number,
        "received":     received,
        "total":        session["total_chunks"],
        "complete":     received == session["total_chunks"],
    }


# ── POST /uploads/{id}/complete ───────────────────────────────────────────────

@router.post("/{upload_id}/complete")
async def complete_chunked_upload(
    upload_id:        str,
    background_tasks: BackgroundTasks,
    current_user:     dict = Depends(require_role("member")),
    pool=Depends(get_db_pool),
    pipeline=Depends(get_pipeline),
):
    """Step 3 of 3 — verify all chunks, assemble, create DB session, launch pipeline."""
    import shutil

    session = _upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(404, "Upload session not found or expired")
    if session["user_id"] != current_user["id"]:
        raise HTTPException(403, "Not your upload session")

    missing = set(range(session["total_chunks"])) - session["received_chunks"]
    if missing:
        raise HTTPException(400, f"Missing {len(missing)} chunk(s): {sorted(missing)[:10]}")

    suffix     = Path(session["filename"]).suffix.lower()
    session_id = str(_uuid_module.uuid4())
    file_name  = f"{session_id}{suffix}"
    final_path = UPLOAD_DIR / file_name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    chunk_dir  = CHUNK_UPLOAD_DIR / upload_id

    try:
        with open(final_path, "wb") as out:
            for i in range(session["total_chunks"]):
                with open(chunk_dir / f"chunk_{i:06d}", "rb") as cf:
                    while block := cf.read(1024 * 1024):
                        out.write(block)
        assembled_size = final_path.stat().st_size
        logger.info(
            "[upload:%s] assembled %d chunks → %s (%.1f MB)",
            upload_id, session["total_chunks"], file_name, assembled_size / 1024 / 1024,
        )
    except Exception as exc:
        final_path.unlink(missing_ok=True)
        raise HTTPException(500, f"File assembly failed: {exc}")
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        _upload_sessions.pop(upload_id, None)

    try:
        config_dict = json.loads(session.get("config", "{}"))
    except json.JSONDecodeError:
        config_dict = {}

    analysis_config = config_dict.get("analysis", {})
    meeting_type    = session.get("meeting_type") or config_dict.get("meeting_type", "sales_call")
    title           = session.get("title") or Path(session["filename"]).stem
    num_speakers    = config_dict.get("num_speakers") or None

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
    except Exception as exc:
        logger.warning("[%s] DB create failed (continuing): %s", session_id, exc)

    _video_path = str(final_path.resolve()) if suffix in {".mp4", ".webm"} else None
    background_tasks.add_task(
        pipeline.run,
        session_id=session_id,
        file_path=str(final_path.resolve()),
        video_path=_video_path,
        meeting_type=meeting_type,
        num_speakers=num_speakers,
        pool=pool,
        org_id=current_user.get("org_id", DEV_ORG_ID),
        user_id=current_user["id"],
        run_behavioural=analysis_config.get("run_behavioural", True),
    )

    return {
        "session_id":  session_id,
        "status":      "processing",
        "title":       title,
        "meeting_type": meeting_type,
        "file_size":   assembled_size,
    }


# ── GET /uploads/{id}/status ───────────────────────────────────────────────────

@router.get("/{upload_id}/status")
async def get_upload_status(
    upload_id:    str,
    current_user: dict = Depends(require_role("member")),
):
    """Resume support — returns which chunks have been received so far."""
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
