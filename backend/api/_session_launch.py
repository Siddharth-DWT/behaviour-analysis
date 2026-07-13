# backend/api/_session_launch.py
"""
Shared media-submission logic for POST /sessions (dashboard) and POST /v1/analyze
(programmatic API). Keeping this in one place guarantees both paths never diverge
on allowed suffixes, size cap, audio/video detection, or pipeline arguments.
"""
import json
import logging
import os
import uuid as _uuid_module
from pathlib import Path

from fastapi import HTTPException, UploadFile

from core.database import DEV_ORG_ID, create_session, update_session_status

logger = logging.getLogger("nexus.backend.session_launch")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/recordings"))
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
VIDEO_SUFFIXES = {".mp4", ".webm"}
MAX_FILE_SIZE = 300 * 1024 * 1024


async def create_session_and_dispatch(
    *,
    file: UploadFile,
    title: str,
    meeting_type: str,
    config_dict: dict,
    current_user: dict,
    pipeline,
    pool,
    background_tasks,
    callback_url: str | None = None,
    retain_media: bool = False,
) -> dict:
    """
    Validate + persist the upload, create the DB session, and dispatch the
    analysis pipeline as a background task. Returns the response dict.

    retain_media=False: media_url is stored NULL (the pipeline still receives the
    real file_path argument), the overlay burn is skipped, and the raw file is
    deleted when the pipeline finishes — see analysis_pipeline / video_service.
    """
    filename = file.filename or "upload.wav"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400,
            f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    session_id = str(_uuid_module.uuid4())
    file_name = f"{session_id}{suffix}"
    file_path = UPLOAD_DIR / file_name

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

    # Thread webhook + retention through upload_config so the pipeline can read
    # them at fire time. api_key_id lets webhook_deliveries link back to the key.
    if callback_url:
        wh = config_dict.setdefault("webhook", {})
        wh["callback_url"] = callback_url
        if current_user.get("_api_key_id"):
            wh["api_key_id"] = current_user["_api_key_id"]
    config_dict["retain_media"] = retain_media

    transcription_config = config_dict.get("transcription", {})
    analysis_config = config_dict.get("analysis", {})
    if not meeting_type or meeting_type == "sales_call":
        meeting_type = config_dict.get("meeting_type", meeting_type)
    num_speakers = config_dict.get("num_speakers") or None

    resolved_path = str(file_path.resolve())
    try:
        _is_lightweight = not analysis_config.get("run_behavioural", True)
        session = await create_session(
            title=title,
            session_type="lightweight" if _is_lightweight else "recording",
            meeting_type=meeting_type,
            # Ephemeral requests store no media_url so GET /video degrades to 404.
            media_url=resolved_path if retain_media else None,
            user_id=current_user["id"],
            upload_config=config_dict,
        )
        session_id = str(session["id"])
        await update_session_status(session_id, "processing")
    except Exception as exc:
        logger.warning("[%s] DB create failed (continuing): %s", session_id, exc)

    _video_path = resolved_path if suffix in VIDEO_SUFFIXES else None
    background_tasks.add_task(
        pipeline.run,
        session_id=session_id,
        file_path=resolved_path,
        video_path=_video_path,
        meeting_type=meeting_type,
        num_speakers=num_speakers,
        pool=pool,
        org_id=current_user.get("org_id", DEV_ORG_ID),
        user_id=current_user["id"],
        run_behavioural=analysis_config.get("run_behavioural", True),
        title=title,
        transcription_config=transcription_config,
        analysis_config=analysis_config,
        user_email=current_user.get("email", ""),
        retain_media=retain_media,
        callback_url=callback_url,
    )

    return {
        "session_id": session_id,
        "status": "processing",
        "title": title,
        "meeting_type": meeting_type,
        "retain_media": retain_media,
    }
