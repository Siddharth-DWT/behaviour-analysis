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
# _cleanup_old_recordings() (pipeline/analysis_pipeline.py) only scans UPLOAD_DIR's
# direct children — it never recurses into subdirectories — so anything stored
# under here is naturally exempt from the RECORDING_RETENTION_DAYS sweep.
PERMANENT_AUDIO_DIR = UPLOAD_DIR / "permanent"
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
    include_media_in_webhook: bool = False,
    from_api: bool = False,
) -> dict:
    """
    Validate + persist the upload, create the DB session, and dispatch the
    analysis pipeline as a background task. Returns the response dict.

    retain_media=False: media_url is stored NULL (the pipeline still receives the
    real file_path argument), the overlay burn is skipped, and the raw file is
    deleted when the pipeline finishes — see analysis_pipeline / video_service.

    include_media_in_webhook=True: even when retain_media=False, the pipeline
    buffers the raw file on disk for WEBHOOK_MEDIA_BUFFER_HOURS after completion
    (instead of deleting it immediately) and includes a media_url in the
    completion webhook payload — see analysis_pipeline._fire_webhook.

    from_api=True (set only by POST /v1/analyze — never by the dashboard's
    POST /sessions or POST /uploads/complete): combined with retain_media=True
    on an audio-only file, stores it under PERMANENT_AUDIO_DIR instead of
    UPLOAD_DIR, so it's kept indefinitely instead of being swept after
    RECORDING_RETENTION_DAYS. Video files are unaffected — same 3-day sweep as
    always, regardless of caller.
    """
    filename = file.filename or "upload.wav"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400,
            f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    is_permanent_audio = from_api and retain_media and suffix not in VIDEO_SUFFIXES
    target_dir = PERMANENT_AUDIO_DIR if is_permanent_audio else UPLOAD_DIR

    session_id = str(_uuid_module.uuid4())
    file_name = f"{session_id}{suffix}"
    file_path = target_dir / file_name

    target_dir.mkdir(parents=True, exist_ok=True)

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

    # API-only: a transcript-only request (run_behavioural=false) defaults to
    # the Parakeet backend instead of the normal AssemblyAI-first auto cascade
    # (see services/voiceAgent/transcriber.py) — faster/cheaper when full
    # behavioural analysis isn't needed. Never applied from the dashboard, and
    # never overrides an explicit model_preference the caller already set. If
    # Parakeet isn't configured (PARAKEET_URL unset), the transcriber's own
    # fallback logs a warning and uses the normal auto cascade instead.
    if (
        from_api
        and not analysis_config.get("run_behavioural", True)
        and not transcription_config.get("model_preference")
    ):
        transcription_config = {**transcription_config, "model_preference": "parakeet"}
        config_dict["transcription"] = transcription_config

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
        include_media_in_webhook=include_media_in_webhook,
    )

    return {
        "session_id": session_id,
        "status": "processing",
        "title": title,
        "meeting_type": meeting_type,
        "retain_media": retain_media,
    }
