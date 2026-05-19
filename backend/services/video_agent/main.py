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
2. VideoFeatureExtractor.extract_all()  → (list[WindowFeatures], lip_activity_map)
3. SpeakerFaceMapper.assign()           → (dict[str, list[WindowFeatures]], dict[str, float])
4. VideoCalibrationModule.build_all_baselines() per speaker
5. Rule engines (Phase 2B-D placeholder — returns empty signals for now)
6. Return VideoAnalysisResponse
"""
import os
import json
import time
import uuid
import asyncio
import logging
import tempfile
import threading
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from shared.redis_layer import (
    AgentStatusRecord,
    EventRecord,
    RedisEventStore,
    RedisLockManager,
    SessionStateRecord,
    SignalRecord,
    SyncRedisRepository,
)

from .feature_extractor import (
    VideoFeatureExtractor, SpeakerFaceMapper, WindowFeatures,
    FaceEmbeddingExtractor, LightASDClassifier,
)
from .calibration import VideoCalibrationModule, FacialBaseline, BodyBaseline, GazeBaseline
from .facial_rules import FacialRuleEngine
from .gaze_rules import GazeRuleEngine
from .body_rules import BodyRuleEngine
from .handcuff_detector import HandcuffDetector as _HandcuffDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.video")

_redis_repo = SyncRedisRepository()
_event_store = RedisEventStore()
_lock_manager = RedisLockManager()


def _record_to_signal(session_id: str, signal: dict) -> SignalRecord:
    return SignalRecord(
        session_id=session_id,
        agent="video",
        speaker_id=signal.get("speaker_id", "unknown"),
        registry_id=signal.get("registry_id"),
        signal_type=signal.get("signal_type", ""),
        value=signal.get("value"),
        value_text=signal.get("value_text", ""),
        confidence=signal.get("confidence", 0.5),
        window_start_ms=signal.get("window_start_ms", 0),
        window_end_ms=signal.get("window_end_ms", 0),
        metadata=signal.get("metadata") or {},
    )


def _set_video_status(session_id: str, status: str, signal_count: int = 0, error: str = "") -> None:
    _redis_repo.set_agent_status(
        session_id,
        "video",
        AgentStatusRecord(
            status=status,
            signal_count=signal_count,
            summary_key="summary:video",
            error=error or "",
        ),
    )


def _publish_video_artifacts(session_id: str, analysis: "VideoAnalysisResponse") -> None:
    _redis_repo.write_artifact(
        session_id,
        "summary:video",
        {
            "session_id": session_id,
            "duration_seconds": analysis.duration_seconds,
            "total_windows": analysis.total_windows,
            "speakers": analysis.speakers,
            "speaker_summaries": [summary.model_dump() for summary in analysis.speaker_summaries],
            "participant_count": analysis.participant_count,
            "backend": analysis.backend,
            "face_embeddings": analysis.face_embeddings,
            "lip_sync_scores": analysis.lip_sync_scores,
            "face_to_speaker": analysis.face_to_speaker,
        },
    )


def _static_variance(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((x - mean) ** 2 for x in vals) / len(vals)


def _is_static_face(windows: list) -> bool:
    """
    Detect photo/graphic faces vs real faces using biological motion variance.

    A photo on screen has zero biological motion: no blinks, no breathing-driven
    jaw micro-movement, no head micro-movements. Real faces have measurable
    variance in at least one of these signals across multiple windows.

    Uses three MediaPipe-derived signals from WindowFeatures:
      - blink_rate_bpm variance across windows (blinks change over time)
      - blendshapes_mean["jawOpen"] variance (breathing moves jaw slightly)
      - head_pose_variance mean (within-window orientation spread, 0 for static images)

    DSA: O(W) single pass where W = windows for this track.
    Returns False (not static) when fewer than 5 detected windows to avoid
    false-positives on briefly visible faces.
    """
    active = [w for w in windows if w.face_detection_rate > 0.5]
    if len(active) < 5:
        return False

    blink_var = _static_variance([w.blink_rate_bpm for w in active])
    jaw_var   = _static_variance([w.blendshapes_mean.get("jawOpen", 0.0) for w in active])
    head_avg  = sum(w.head_pose_variance for w in active) / len(active)

    return blink_var < 0.001 and jaw_var < 0.0005 and head_avg < 0.01


def _push_signals_to_redis(session_id: str, signals: list[dict]) -> None:
    """Persist signals to the canonical Redis stream for this session."""
    if not signals:
        return
    try:
        _redis_repo.publish_signal_batch(_record_to_signal(session_id, signal) for signal in signals)
    except Exception as exc:
        logger.warning(f"[{session_id}] Redis signal publish failed (non-fatal): {exc}")


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
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(os.cpu_count() or 4, 8),
    thread_name_prefix="video-pipeline",
)


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
    face_embeddings:       dict = {}      # {Face_N: {embedding, thumbnail_b64, track_id}}
    lip_sync_scores:       dict = {}      # {Speaker_N: correlation_score} — lip-sync assignment quality
    face_to_speaker:       dict = {}      # {face_index_int: "Speaker_N"} — linkage map for gateway


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
        extractor:      VideoFeatureExtractor,
        mapper:         SpeakerFaceMapper,
        calibrator:     VideoCalibrationModule,
        extractor_lock: Optional[threading.Lock] = None,
    ) -> None:
        self._extractor      = extractor
        self._mapper         = mapper
        self._calibrator     = calibrator
        self._facial_rules   = FacialRuleEngine()
        self._gaze_rules     = GazeRuleEngine()
        self._body_rules     = BodyRuleEngine()
        # Protects shared extractor state (_diar_segments, _active_tile_face_to_speaker)
        # from concurrent mutation when multiple sessions run simultaneously.
        self._extractor_lock = extractor_lock or threading.Lock()

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
        # Lock guards shared extractor state: _diar_segments (written before
        # extract_all) and _active_tile_face_to_speaker (read after). Without
        # the lock, concurrent sessions would race on these attributes.
        logger.info(f"[{session_id}] Extracting video features from {Path(video_path).name}")
        with self._extractor_lock:
            self._extractor._diar_segments = diar_segments
            try:
                windows, lip_activity_map = self._extractor.extract_all(
                    video_path, overlay_output_path=None, meeting_type=meeting_type
                )
            finally:
                self._extractor._diar_segments = []
            active_tile_tags = getattr(self._extractor, "_active_tile_face_to_speaker", {})

        logger.info(f"[{session_id}] {len(windows)} windows extracted")

        if not windows:
            raise ValueError("No frames could be extracted from the video file.")

        duration_sec = (windows[-1].window_end_ms - windows[0].window_start_ms) / 1000.0

        _is_interrogation = meeting_type == "interrogation_video"

        # ── Inject active-tile tags into mapper ───────────────────────────────
        # Skipped for interrogation videos — no speaker-face mapping is performed.
        if not _is_interrogation:
            self._mapper.set_active_tile_tags(active_tile_tags)
            if active_tile_tags:
                logger.info(
                    f"[{session_id}] Active-tile tags → mapper: "
                    f"{len(active_tile_tags)} face(s) {{{', '.join(f'Face_{k}: {v}' for k, v in active_tile_tags.items())}}}"
                )
            else:
                logger.warning(
                    f"[{session_id}] No active-tile tags from extractor — "
                    f"mapper will use lip-sync only (may produce weak linkage)"
                )

        # ── Step 1b: Light-ASD active speaker scoring (optional) ─────────────
        # Skipped for interrogation videos — no speaker-face mapping is performed.
        # For other sessions: replaces MediaPipe jawOpen lip-sync correlation with
        # a learned AV model (94.1% precision on AVA-ActiveSpeaker).
        asd_scores: dict | None = None
        if not _is_interrogation:
            _asd = LightASDClassifier.get_instance(
                model_dir=os.path.join(os.path.dirname(__file__), "..", "..", "models")
            )
            if _asd.available:
                _tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                _tmp_audio.close()
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", video_path,
                            "-vn", "-ar", "16000", "-ac", "1",
                            _tmp_audio.name,
                        ],
                        check=True,
                        capture_output=True,
                    )
                    _face_crops = getattr(self._extractor, "_face_crops_sequence", {})
                    _fps = getattr(self._extractor, "_last_video_fps", 5.0)
                    if _face_crops:
                        asd_scores = _asd.score(_face_crops, _tmp_audio.name, fps=_fps)
                        logger.info(
                            "[%s] Light-ASD: scored %d tracks", session_id, len(asd_scores)
                        )
                    else:
                        logger.warning(
                            "[%s] Light-ASD: no face crops buffered — "
                            "falling back to lip-sync", session_id
                        )
                except Exception as exc:
                    logger.warning(
                        "[%s] Light-ASD audio extraction failed (non-fatal): %s",
                        session_id, exc,
                    )
                finally:
                    try:
                        os.unlink(_tmp_audio.name)
                    except OSError:
                        pass

        # ── Step 2: Map windows → speakers ────────────────────────────────────
        # Interrogation videos skip all speaker-face linking — single-camera
        # ceiling/oblique angle makes lip-sync unreliable (suspect looks down,
        # interrogator often off-camera). Face_N tracks carry behavioral signals
        # on their own timeline without being merged to Speaker_N labels.
        windows_by_speaker, lip_sync_scores, face_to_speaker = self._mapper.assign(
            windows, diar_segments, lip_activity_map,
            asd_scores=asd_scores,
            skip_speaker_link=_is_interrogation,
        )
        # ── Step 2b: Remove static faces (photos/graphics on screen) ─────────
        # Photos have zero biological motion — no blink variance, no jaw
        # micro-movement from breathing, no head micro-movements. Applies to
        # both interrogation (Face_N only) and standard sessions equally since
        # _is_static_face operates on whatever tracks are in windows_by_speaker.
        static_tracks: set[str] = {
            spk for spk, wins in windows_by_speaker.items()
            if _is_static_face(wins)
        }
        if static_tracks:
            for tid in static_tracks:
                del windows_by_speaker[tid]
            logger.info(
                "[%s] Static face(s) removed before signal computation: %s",
                session_id, sorted(static_tracks),
            )

        speakers = sorted(windows_by_speaker.keys())
        logger.info(
            f"[{session_id}] Speakers: {speakers} | "
            f"face_to_speaker={face_to_speaker} | lip_sync_scores={lip_sync_scores}"
        )

        # ── Step 3: Per-speaker baselines (parallel — stateless, each speaker independent) ─
        # VideoCalibrationModule has no mutable instance state; each call only reads
        # its own `windows` slice and returns new typed dataclasses. Safe for threads.
        baselines: dict[str, tuple[FacialBaseline, BodyBaseline, GazeBaseline]] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(windows_by_speaker), 4),
        ) as bl_pool:
            future_to_speaker = {
                bl_pool.submit(
                    self._calibrator.build_all_baselines,
                    speaker_id=spk,
                    session_id=session_id,
                    windows=wins,
                ): spk
                for spk, wins in windows_by_speaker.items()
            }
            for fut in concurrent.futures.as_completed(future_to_speaker):
                spk = future_to_speaker[fut]
                facial, body, gaze = fut.result()
                baselines[spk] = (facial, body, gaze)

        # ── Step 4: Rule engines ──────────────────────────────────────────────────
        # Facial and Gaze are fully stateless and mutually independent — run in parallel.
        # Body depends on their outputs (extra_signals) — runs strictly after both finish.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as rule_pool:
            facial_future = rule_pool.submit(
                self._facial_rules.evaluate,
                windows_by_speaker, baselines, session_id, meeting_type,
            )
            gaze_future = rule_pool.submit(
                self._gaze_rules.evaluate,
                windows_by_speaker, baselines, session_id, meeting_type,
            )
            facial_signals = facial_future.result()
            gaze_signals   = gaze_future.result()

        body_signals = self._body_rules.evaluate(
            windows_by_speaker, baselines, session_id, meeting_type,
            extra_signals=facial_signals + gaze_signals,
        )

        all_signals: list[dict] = facial_signals + gaze_signals + body_signals

        # ── Step 4b: Interrogation-specific video rules ───────────────────────
        if meeting_type == "interrogation_video":
            try:
                import cv2 as _cv2
                _cap = _cv2.VideoCapture(video_path)
                _real_fps = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
                _cap.release()

                # Handcuff detection — visual (pose) + contextual (transcript).
                _transcript_text = " ".join(
                    seg.get("text", "") for seg in (diar_segments or [])
                )
                _hc_result  = _HandcuffDetector().detect(
                    windows_by_speaker=windows_by_speaker,
                    transcript=_transcript_text,
                    session_id=session_id,
                )
                _handcuffed = _hc_result["handcuffs_detected"]

                from .interrogation_rules import InterrogationVideoRules
                interrog_signals = InterrogationVideoRules().evaluate(
                    windows_by_speaker=windows_by_speaker,
                    baselines=baselines,
                    diar_segments=diar_segments,
                    session_id=session_id,
                    video_fps=_real_fps,
                    handcuffed=_handcuffed,
                )
                all_signals.extend(interrog_signals)
                if interrog_signals:
                    logger.info(
                        "[%s] Interrogation video rules: %d signals (handcuffed=%s)",
                        session_id, len(interrog_signals), _handcuffed,
                    )
            except Exception as exc:
                logger.warning("[%s] Interrogation video rules failed (non-fatal): %s", session_id, exc)

        # One presence_detected marker per speaker — spans their first to last window.
        # Consumed by the gateway whitelist and frontend left panel to show a thumbnail
        # from first appearance, before calibration produces any behavioral signals.
        for spk, spk_wins in windows_by_speaker.items():
            if not spk_wins:
                continue
            all_signals.append({
                "agent":          "video",
                "signal_type":    "presence_detected",
                "speaker_id":     spk,
                "window_start_ms": min(w.window_start_ms for w in spk_wins),
                "window_end_ms":   max(w.window_end_ms   for w in spk_wins),
                "confidence":     1.0,
                "value":          1.0,
                "value_text":     "present",
                "metadata":       {},
            })

        # Persist all signals in one Redis call instead of three separate connections
        _push_signals_to_redis(session_id, all_signals)

        summaries    = self._build_summaries(windows_by_speaker, baselines)
        participant_count = max(
            (w.face_count for w in windows if hasattr(w, "face_count")),
            default=len(speakers),
        )

        # ── Step 5: ArcFace embeddings for cross-session identity ─────────────
        # Prefer cached embeddings from the identity merge in _extract_frames
        # to avoid running ArcFace a second time on the same crops.
        face_embeddings_data: dict = {}
        try:
            import base64
            cached = getattr(self._extractor, "_cached_embeddings", None)
            if cached:
                emb_results = cached
                source = "cache"
            elif hasattr(self._extractor, "_best_face_crops") and self._extractor._best_face_crops:
                embedder = FaceEmbeddingExtractor.get_instance()
                emb_results = embedder.extract_from_crops(self._extractor._best_face_crops) if embedder.available else {}
                source = "fresh"
            else:
                emb_results = {}
                source = "none"

            for track_id, (embedding, thumbnail) in emb_results.items():
                # Always key by Face_N — SpeakerFaceMapper guarantees wf.speaker_id
                # is Face_N for every window. _build_session_face_locks looks up
                # face_embeddings_data by Face_N key; a Speaker_N key here would
                # make the lookup miss and silently drop the face→speaker link.
                face_embeddings_data[f"Face_{track_id}"] = {
                    "embedding":     embedding,
                    "thumbnail_b64": base64.b64encode(thumbnail).decode(),
                    "track_id":      track_id,
                }
            if emb_results:
                logger.info(
                    f"[{session_id}] Face embeddings: {len(face_embeddings_data)} "
                    f"({source})"
                )
        except Exception as exc:
            logger.warning(f"[{session_id}] Face embedding extraction failed (non-fatal): {exc}")

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
            face_embeddings=face_embeddings_data,
            lip_sync_scores=lip_sync_scores,
            face_to_speaker={str(k): v for k, v in face_to_speaker.items()},
        ), video_path

    def burn_overlay(
        self,
        session_id:    str,
        video_path:    str,
        all_signals:   list[dict],
        diar_segments: list[dict] | None = None,
        display_names: dict | None = None,
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
            self._extractor._diar_segments = diar_segments or []
            self._extractor.burn_landmarks_and_labels(
                video_path, all_signals, output_path=output_path,
                display_names=display_names or {},
            )
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
    lock_token = await _lock_manager.acquire(session_id, "video")
    if not lock_token:
        logger.warning(f"[{session_id}] Video agent lock already held; skipping duplicate execution")
        _video_jobs[job_id]["status"] = "failed"
        _video_jobs[job_id]["error"] = "video agent already running for this session"
        _set_video_status(session_id, "skipped", error="duplicate execution prevented")
        return

    _redis_repo.set_session_state(
        session_id,
        SessionStateRecord(status="running", current_step="video"),
    )
    _set_video_status(session_id, "running")
    await _event_store.append(
        session_id,
        EventRecord(session_id=session_id, agent="video", event_type="agent_started", payload={"job_id": job_id}),
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
        _set_video_status(session_id, "failed", error=str(exc))
        await _event_store.append(
            session_id,
            EventRecord(session_id=session_id, agent="video", event_type="agent_failed", payload={"error": str(exc)}),
        )
        await _lock_manager.release(session_id, "video", lock_token)
        return

    # ── Signals ready — gateway can return results immediately ────────────────
    _video_jobs[job_id]["status"] = "signals_ready"
    _video_jobs[job_id]["result"] = analysis.model_dump()
    _publish_video_artifacts(session_id, analysis)
    _set_video_status(session_id, "completed", signal_count=len(analysis.signals))
    await _event_store.append(
        session_id,
        EventRecord(
            session_id=session_id,
            agent="video",
            event_type="signals_ready",
            payload={"job_id": job_id, "signal_count": len(analysis.signals)},
        ),
    )
    logger.info(
        f"[{session_id}] Video job {job_id} signals_ready: {len(analysis.signals)} signals"
    )

    # Wait 15s so the gateway can complete registry matching and POST display names
    # before Phase 2 burn begins. Non-blocking — event loop handles display-names
    # POST requests from the gateway during this window.
    await asyncio.sleep(15)
    display_names: dict = _video_jobs[job_id].get("display_names") or {}
    if display_names:
        logger.info(f"[{session_id}] Video job {job_id} burn: {len(display_names)} display names received")

    # ── Phase 2: Burn signal labels onto original video (cosmetic) ────────────
    _video_jobs[job_id]["status"] = "annotating"
    try:
        annotated_path = await loop.run_in_executor(
            _thread_pool,
            lambda: pipeline.burn_overlay(
                session_id, src_video_path, analysis.signals, diar_segments,
                display_names=display_names,
            ),
        )
        if annotated_path:
            _video_jobs[job_id]["result"]["annotated_video_path"] = annotated_path
    except Exception as exc:
        logger.warning(f"[{session_id}] Overlay burn failed (non-fatal): {exc}")

    _video_jobs[job_id]["status"] = "done"
    logger.info(f"[{session_id}] Video job {job_id} done")
    await _event_store.append(
        session_id,
        EventRecord(session_id=session_id, agent="video", event_type="agent_completed", payload={"job_id": job_id}),
    )
    await _lock_manager.release(session_id, "video", lock_token)


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


@app.post("/embed-face")
async def embed_face(image: UploadFile = File(...)):
    """
    Extract a 512-dim ArcFace embedding from an uploaded face image.
    Used by the API gateway's /speakers/search-by-face endpoint.

    Returns: { embedding: [float×512], detected: bool }
    """
    import cv2
    import numpy as np

    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    fe = FaceEmbeddingExtractor.get_instance()
    if not fe.available:
        raise HTTPException(status_code=503, detail="Face recognition model not available")

    results = fe.extract_from_crops({0: bgr})
    if not results:
        return {"embedding": [], "detected": False}

    embedding, _ = results[0]
    return {"embedding": embedding, "detected": True}


@app.post("/jobs/{job_id}/display-names")
async def set_display_names(job_id: str, body: dict):
    """
    Set speaker display names for the overlay burn pass.
    Called by the API gateway after registry matching completes,
    during the 15s window between signals_ready and Phase 2 burn.
    """
    job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job["display_names"] = body.get("names", {})
    return {"status": "ok", "names_received": len(job["display_names"])}


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
