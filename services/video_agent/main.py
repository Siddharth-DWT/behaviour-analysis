# services/video_agent/main.py
"""
NEXUS Video Agent — FastAPI Service
Processes uploaded video files; extracts frame-level visual features
and builds per-speaker baselines.  Rule engines (Phase 2B-D) slot in
after calibration without changing this service's API contract.

Endpoints:
  POST /analyse  → Full video analysis pipeline (feature extraction + calibration)
  GET  /health   → Liveness check

Pipeline:
  1. Save uploaded video to temp file
  2. VideoFeatureExtractor.extract_all()  → list[WindowFeatures]
  3. SpeakerFaceMapper.assign()           → dict[str, list[WindowFeatures]]
  4. VideoCalibrationModule.build_all_baselines() per speaker
  5. Rule engines (Phase 2B-D placeholder — returns empty signals for now)
  6. Return VideoAnalysisResponse
"""
import os
import sys
import time
import uuid
import asyncio
import logging
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from feature_extractor import VideoFeatureExtractor, SpeakerFaceMapper, WindowFeatures
    from calibration import VideoCalibrationModule, FacialBaseline, BodyBaseline, GazeBaseline
    from facial_rules import FacialRuleEngine
    from gaze_rules import GazeRuleEngine
    from body_rules import BodyRuleEngine
except ImportError:
    from services.video_agent.feature_extractor import (
        VideoFeatureExtractor, SpeakerFaceMapper, WindowFeatures,
    )
    from services.video_agent.calibration import (
        VideoCalibrationModule, FacialBaseline, BodyBaseline, GazeBaseline,
    )
    from services.video_agent.facial_rules import FacialRuleEngine
    from services.video_agent.gaze_rules import GazeRuleEngine
    from services.video_agent.body_rules import BodyRuleEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.video")

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="NEXUS Video Agent",
    description="Visual behavioural feature extraction via MediaPipe",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Module-level singletons (created once, reused across requests) ───────────
_extractor:   Optional[VideoFeatureExtractor]   = None
_mapper:      Optional[SpeakerFaceMapper]       = None
_calibrator:  Optional[VideoCalibrationModule]  = None

# ─── Async job state ──────────────────────────────────────────────────────────
# job_id → {status, result, error, created_at}
# Status lifecycle: "queued" → "running" → "done" | "failed"
_video_jobs: dict[str, dict] = {}

# CPU-bound pipeline.run() runs in this thread pool so the event loop stays free
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="video-pipeline")


def _get_extractor() -> VideoFeatureExtractor:
    global _extractor
    if _extractor is None:
        model_dir = os.getenv("MEDIAPIPE_MODEL_DIR", "models/mediapipe")
        _extractor = VideoFeatureExtractor(model_dir=model_dir)
    return _extractor


def _get_mapper() -> SpeakerFaceMapper:
    global _mapper
    if _mapper is None:
        _mapper = SpeakerFaceMapper()
    return _mapper


def _get_calibrator() -> VideoCalibrationModule:
    global _calibrator
    if _calibrator is None:
        _calibrator = VideoCalibrationModule()
    return _calibrator


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response models
# ══════════════════════════════════════════════════════════════════════════════

class DiarSegment(BaseModel):
    """Single diarization segment from the voice agent."""
    speaker:  str
    start_ms: int
    end_ms:   int


class VideoAnalysisRequest(BaseModel):
    """
    Metadata sent alongside the video file.
    diar_segments must come from the voice agent's diarization output
    so that windows can be mapped to the correct speaker.
    """
    session_id:     str
    meeting_type:   str          = "general"
    diar_segments:  list[DiarSegment] = []
    num_speakers:   int          = 2


class SpeakerVideoSummary(BaseModel):
    """Calibration summary for one speaker returned in the response."""
    speaker_id:               str
    window_count:             int
    face_detection_rate:      float
    calibration_confidence:   float

    # Facial baseline snapshot
    head_pitch_baseline:  float
    head_yaw_baseline:    float
    blink_rate_baseline:  float

    # Body baseline snapshot
    spine_angle_baseline:   float
    body_movement_baseline: float

    # Gaze baseline snapshot
    screen_engagement_baseline: float
    gaze_y_offset:              float


class VideoAnalysisResponse(BaseModel):
    """
    Response from POST /analyse.
    signals will be populated by rule engines (Phase 2B-D).
    window_features contains raw aggregated data for debugging.
    """
    session_id:            str
    duration_seconds:      float
    total_windows:         int
    speakers:              list[str]
    speaker_summaries:     list[SpeakerVideoSummary]
    signals:               list[dict]
    processing_time:       float
    participant_count:     int = 0        # max faces detected in any single frame
    backend:               str = "mediapipe"
    annotated_video_path:  Optional[str] = None


class VideoJobResponse(BaseModel):
    """Returned immediately by POST /analyse; polled via GET /jobs/{job_id}."""
    job_id:  str
    status:  str                              # queued | running | done | failed
    result:  Optional[VideoAnalysisResponse] = None
    error:   Optional[str]                   = None


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class VideoPipeline:
    """
    Orchestrates the full video analysis pipeline for one session.

    Encapsulates the stateful progression: extract → map → calibrate → rules.
    Holding this as a class (rather than a bare function) makes it easy to
    inject rule engines in Phase 2B-D without touching the endpoint handler.

    OOP principle: open-closed — new rule engines slot in via register_rules(),
    this class stays closed for modification.
    """

    def __init__(
        self,
        extractor:  VideoFeatureExtractor,
        mapper:     SpeakerFaceMapper,
        calibrator: VideoCalibrationModule,
    ) -> None:
        self._extractor    = extractor
        self._mapper       = mapper
        self._calibrator   = calibrator
        self._facial_rules = FacialRuleEngine()
        self._gaze_rules   = GazeRuleEngine()
        self._body_rules   = BodyRuleEngine()

    def run_analysis(
        self,
        video_path:    str,
        session_id:    str,
        diar_segments: list[dict],
        meeting_type:  str = "general",
    ) -> tuple["VideoAnalysisResponse", str]:
        """
        Phase 1 — MediaPipe extraction + calibration + rule engines.

        Returns (VideoAnalysisResponse with signals, video_path).
        Extraction runs WITHOUT overlay so signals are available as fast
        as possible. Phase 2 (burn_overlay) burns labels onto the original
        video in the background without blocking signal return.
        """
        start = time.time()

        # ── Step 1: Frame extraction (no overlay — fast path) ─────────────────
        logger.info(f"[{session_id}] Extracting video features from {Path(video_path).name}")
        windows: list[WindowFeatures] = self._extractor.extract_all(
            video_path, overlay_output_path=None, meeting_type=meeting_type
        )
        logger.info(f"[{session_id}] {len(windows)} windows extracted")

        if not windows:
            raise ValueError("No frames could be extracted from the video file.")

        duration_sec = (windows[-1].window_end_ms - windows[0].window_start_ms) / 1000.0

        # ── Step 2: Map windows → speakers ────────────────────────────────────
        windows_by_speaker: dict[str, list[WindowFeatures]] = self._mapper.assign(
            windows, diar_segments
        )
        speakers = sorted(windows_by_speaker.keys())
        logger.info(f"[{session_id}] Speakers detected: {speakers}")

        # ── Step 3: Per-speaker baselines ─────────────────────────────────────
        baselines: dict[str, tuple[FacialBaseline, BodyBaseline, GazeBaseline]] = {}
        for speaker_id, spk_windows in windows_by_speaker.items():
            facial, body, gaze = self._calibrator.build_all_baselines(
                speaker_id=speaker_id,
                session_id=session_id,
                windows=spk_windows,
            )
            baselines[speaker_id] = (facial, body, gaze)

        # ── Step 4: Rule engines ──────────────────────────────────────────────
        facial_signals = self._facial_rules.evaluate(
            windows_by_speaker, baselines, session_id, meeting_type
        )
        gaze_signals = self._gaze_rules.evaluate(
            windows_by_speaker, baselines, session_id, meeting_type
        )
        body_signals = self._body_rules.evaluate(
            windows_by_speaker, baselines, session_id, meeting_type
        )
        all_signals: list[dict] = facial_signals + gaze_signals + body_signals

        summaries    = self._build_summaries(windows_by_speaker, baselines)
        participant_count = max(
            (w.face_count for w in windows if hasattr(w, "face_count")),
            default=len(speakers),
        )
        elapsed = time.time() - start

        logger.info(
            f"[{session_id}] Analysis complete: {len(all_signals)} signals in {elapsed:.1f}s"
        )

        return VideoAnalysisResponse(
            session_id=session_id,
            duration_seconds=round(duration_sec, 2),
            total_windows=len(windows),
            speakers=speakers,
            speaker_summaries=summaries,
            signals=all_signals,
            processing_time=round(elapsed, 2),
            participant_count=participant_count,
            annotated_video_path=None,   # Phase 2 fills this in
        ), video_path

    def burn_overlay(
        self,
        session_id:  str,
        video_path:  str,
        all_signals: list[dict],
    ) -> Optional[str]:
        """
        Phase 2 — Burn signal text labels onto the original video.

        Purely cosmetic. Runs after signals are already available in the job store.
        Writes {overlay_dir}/{session_id}_annotated.mp4. Returns that path on
        success, None on any failure (non-fatal — signals are already persisted).
        """
        overlay_dir = Path(os.getenv("OVERLAY_DIR", "data/overlays"))
        overlay_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(overlay_dir / f"{session_id}_annotated.mp4")

        try:
            logger.info(
                f"[{session_id}] Burning landmarks + {len(all_signals)} signal labels onto video "
                f"→ {output_path}"
            )
            self._extractor.burn_landmarks_and_labels(video_path, all_signals, output_path=output_path)
            logger.info(f"[{session_id}] Annotated video ready → {output_path}")
            return output_path
        except Exception as exc:
            logger.warning(f"[{session_id}] Landmark burn failed (non-fatal): {exc}")
            return None

    def _build_summaries(
        self,
        windows_by_speaker: dict[str, list[WindowFeatures]],
        baselines: dict[str, tuple[FacialBaseline, BodyBaseline, GazeBaseline]],
    ) -> list[SpeakerVideoSummary]:
        summaries: list[SpeakerVideoSummary] = []
        for spk, spk_windows in windows_by_speaker.items():
            facial, body, gaze = baselines.get(spk, (
                FacialBaseline(speaker_id=spk),
                BodyBaseline(speaker_id=spk),
                GazeBaseline(speaker_id=spk),
            ))

            avg_face_rate = (
                sum(w.face_detection_rate for w in spk_windows) / len(spk_windows)
                if spk_windows else 0.0
            )

            summaries.append(SpeakerVideoSummary(
                speaker_id=spk,
                window_count=len(spk_windows),
                face_detection_rate=round(avg_face_rate, 3),
                calibration_confidence=round(facial.calibration_confidence, 3),

                head_pitch_baseline=round(facial.head_pitch_mean, 2),
                head_yaw_baseline=round(facial.head_yaw_mean, 2),
                blink_rate_baseline=round(facial.blink_rate_bpm, 1),

                spine_angle_baseline=round(body.spine_angle_mean, 2),
                body_movement_baseline=round(body.body_movement_mean, 5),

                screen_engagement_baseline=round(gaze.screen_engagement_rate, 3),
                gaze_y_offset=round(gaze.gaze_y_offset, 4),
            ))

        return summaries


# ══════════════════════════════════════════════════════════════════════════════
# Background job runner
# ══════════════════════════════════════════════════════════════════════════════

async def _run_video_job(
    job_id: str,
    video_path: str,
    session_id: str,
    diar_segments: list[dict],
    meeting_type: str,
) -> None:
    """
    Coroutine launched as an asyncio background task.
    Runs the CPU-bound pipeline in the thread pool so the event loop stays free
    for health checks and job-status polling during processing.
    """
    loop = asyncio.get_event_loop()
    pipeline = VideoPipeline(
        extractor=_get_extractor(),
        mapper=_get_mapper(),
        calibrator=_get_calibrator(),
    )

    # ── Phase 1: Frame extraction ─────────────────────────────────────────────
    _video_jobs[job_id]["status"] = "extracting"
    try:
        analysis, src_video_path = await loop.run_in_executor(
            _thread_pool,
            lambda: pipeline.run_analysis(
                video_path=video_path,
                session_id=session_id,
                diar_segments=diar_segments,
                meeting_type=meeting_type,
            ),
        )
    except Exception as exc:
        logger.exception(f"[{session_id}] Video job {job_id} analysis failed: {exc}")
        _video_jobs[job_id]["status"] = "failed"
        _video_jobs[job_id]["error"] = str(exc)
        return

    # ── Signals ready — gateway can return results immediately ────────────────
    _video_jobs[job_id]["status"] = "signals_ready"
    _video_jobs[job_id]["result"] = analysis.model_dump()
    logger.info(
        f"[{session_id}] Video job {job_id} signals_ready: {len(analysis.signals)} signals"
    )

    # ── Phase 2: Burn signal labels onto original video (cosmetic) ────────────
    _video_jobs[job_id]["status"] = "annotating"
    try:
        annotated_path = await loop.run_in_executor(
            _thread_pool,
            lambda: pipeline.burn_overlay(session_id, src_video_path, analysis.signals),
        )
        if annotated_path:
            _video_jobs[job_id]["result"]["annotated_video_path"] = annotated_path
    except Exception as exc:
        logger.warning(f"[{session_id}] Overlay burn failed (non-fatal): {exc}")

    _video_jobs[job_id]["status"] = "done"
    logger.info(f"[{session_id}] Video job {job_id} done")


# ══════════════════════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    logger.info("Video Agent starting — singletons initialised on first request.")
    # Model files download lazily on first extract_all() call.
    # Pre-warming here would block startup; lazy is preferred.


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    try:
        import mediapipe  # noqa: F401
        mp_status = "ok"
    except ImportError:
        mp_status = "not_installed"
    try:
        import cv2  # noqa: F401
        cv_status = "ok"
    except ImportError:
        cv_status = "not_installed"

    return {
        "status": "ok",
        "agent": "video",
        "mediapipe": mp_status,
        "opencv": cv_status,
    }


@app.post("/analyse", response_model=VideoJobResponse, status_code=202)
async def analyse(
    video: UploadFile = File(...),
    session_id: str = Form(default=""),
    meeting_type: str = Form(default="general"),
    diar_segments_json: str = Form(default="[]"),
    num_speakers: int = Form(default=2),  # noqa: ARG001 — reserved for future face-budget tuning
):
    """
    Submit a video file for async analysis.

    Returns immediately with a job_id (HTTP 202).
    Poll GET /jobs/{job_id} until status == "done" or "failed".

    Form fields:
      video               — mp4 / webm / avi file
      session_id          — UUID from API Gateway (generated if blank)
      meeting_type        — sales | interview | general | etc.
      diar_segments_json  — JSON array of {speaker,start_ms,end_ms} from voice agent
      num_speakers        — expected speaker count (reserved for Phase 2B face budget)
    """
    import json

    if not session_id:
        session_id = str(uuid.uuid4())

    job_id = str(uuid.uuid4())

    # Parse diarization segments
    try:
        raw_segs: list[dict] = json.loads(diar_segments_json)
    except (json.JSONDecodeError, ValueError):
        raw_segs = []

    # Validate file type
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {suffix}")

    # Write to a persistent staging dir — background task owns cleanup
    staging_dir = Path(tempfile.gettempdir()) / "nexus_video_jobs"
    staging_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(staging_dir / f"{job_id}{suffix}")

    content = await video.read()
    with open(video_path, "wb") as fh:
        fh.write(content)

    # Register job entry before creating task (avoids a race on GET /jobs)
    _video_jobs[job_id] = {
        "status": "queued",
        "session_id": session_id,
        "result": None,
        "error": None,
        "created_at": time.time(),
    }

    asyncio.create_task(
        _run_video_job(job_id, video_path, session_id, raw_segs, meeting_type),
        name=f"video-job-{job_id[:8]}",
    )

    logger.info(f"[{session_id}] Video job {job_id} queued")
    return VideoJobResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=VideoJobResponse)
async def get_job(job_id: str):
    """Poll the status of an async video analysis job."""
    job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result: Optional[VideoAnalysisResponse] = None
    if job["result"] is not None:
        result = VideoAnalysisResponse(**job["result"])

    return VideoJobResponse(
        job_id=job_id,
        status=job["status"],
        result=result,
        error=job.get("error"),
    )
