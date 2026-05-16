# backend/agents/video_service.py
"""
VideoAgentService — in-process wrapper for the Video Agent.

CPU-bound frame extraction runs in a thread pool via run_in_executor() so the
asyncio event loop stays free for health checks and other requests during
the 10–90 minute processing window.

Fire-and-forget: analyse() returns as soon as signals are ready.
Overlay burn (cosmetic) continues as an asyncio.create_task() background coroutine
and reads display names from Redis after the pipeline finishes registry matching.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from shared.redis_layer import (
    AgentStatusRecord,
    EventRecord,
    RedisEventStore,
    RedisLockManager,
    SessionStateRecord,
    SignalRecord,
    SyncRedisRepository,
)

from .base import BaseAgentService

logger = logging.getLogger("nexus.backend.video")

# Display names are written here by AnalysisPipeline after registry matching.
# The burn task reads this key after its 15s grace period.
_DISPLAY_NAMES_ARTIFACT = "display_names"


class VideoAgentService(BaseAgentService):
    """
    In-process Video Agent.

    MediaPipe (face mesh, hands, pose) + ArcFace identity merge + 3 rule engines:
      FacialRuleEngine, GazeRuleEngine, BodyRuleEngine.

    Models are lazy-initialised on first call to avoid a ~3s startup penalty.
    A ThreadPoolExecutor is allocated at startup() and reused across requests
    (no new pool per job, O(1) dispatch).
    """

    name = "video"

    def __init__(self) -> None:
        self._extractor = None
        self._mapper = None
        self._calibrator = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._redis_repo = SyncRedisRepository()     # thread-safe sync client
        self._event_store = RedisEventStore()
        self._lock_manager = RedisLockManager()

    async def startup(self) -> None:
        cpu_count = os.cpu_count() or 4
        self._thread_pool = ThreadPoolExecutor(
            max_workers=min(cpu_count, 8),
            thread_name_prefix="nexus-video",
        )
        asyncio.create_task(self._prewarm_mediapipe())
        self._log("Video Agent ready (MediaPipe pre-warming in background).")

    async def _prewarm_mediapipe(self) -> None:
        """Pre-load MediaPipe models in a background thread to eliminate first-session delay."""
        loop = asyncio.get_running_loop()
        try:
            # _warmup_sync runs entirely in the thread pool — including _get_pipeline()
            # which imports mediapipe and insightface (C-extensions, 2-5s) — so those
            # heavy imports never block the event loop.
            await loop.run_in_executor(self._thread_pool, self._warmup_sync)
            self._log("MediaPipe pre-warm complete — first-session delay eliminated.")
        except Exception as exc:
            logger.warning("MediaPipe pre-warm failed (non-fatal): %s", exc)

    def _warmup_sync(self) -> None:
        """
        Synchronous warmup body — runs in the thread pool.

        Two steps, both blocking, both safe to run off the event loop:
          1. _get_pipeline() — imports mediapipe + insightface C-extensions (~2-5s).
          2. extractor.warmup() — loads MediaPipe model files AND initialises the
             ArcFace singleton (buffalo_l download + app.prepare()) so neither
             the frame loop nor the post-loop ArcFace pass pays a cold-start cost.
        """
        self._get_pipeline()       # imports C-extensions; sets self._extractor
        self._extractor.warmup()   # MediaPipe page-cache + ArcFace singleton init

    async def shutdown(self) -> None:
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)

    def _get_pipeline(self):
        """Lazy-init VideoPipeline (imports MediaPipe/ArcFace on first call)."""
        from services.video_agent.feature_extractor import (
            VideoFeatureExtractor, SpeakerFaceMapper,
        )
        from services.video_agent.calibration import VideoCalibrationModule
        from services.video_agent.main import VideoPipeline

        if self._extractor is None:
            model_dir = os.getenv("MEDIAPIPE_MODEL_DIR", "models/mediapipe")
            self._extractor = VideoFeatureExtractor(model_dir=model_dir)
        if self._mapper is None:
            self._mapper = SpeakerFaceMapper()
        if self._calibrator is None:
            self._calibrator = VideoCalibrationModule()

        return VideoPipeline(self._extractor, self._mapper, self._calibrator)

    async def analyse(
        self,
        session_id: str,
        video_path: str,
        diar_segments: list[dict],
        meeting_type: str = "general",
        num_speakers: int = 2,   # noqa: ARG002 — reserved for future face budget
    ) -> dict:
        """
        Run Phase 1 video analysis; fire-and-forget Phase 2 overlay burn.

        Returns VideoAnalysisResponse.model_dump() as soon as signals are ready
        (~10 min). Overlay burn continues as an asyncio background task.

        DSA: run_in_executor + existing thread pool = O(1) dispatch cost.
        Fire-and-forget burn = asyncio.create_task() — does not block the pipeline.
        """
        lock_token = await self._lock_manager.acquire(session_id, self.name)
        if not lock_token:
            logger.warning("[%s] Video agent lock already held — skipping duplicate", session_id)
            raise RuntimeError(f"Video agent already processing session {session_id}")

        self._redis_repo.set_session_state(
            session_id, SessionStateRecord(status="running", current_step="video")
        )
        _set_video_status(self._redis_repo, session_id, "running")
        await self._event_store.append(
            session_id,
            EventRecord(session_id=session_id, agent=self.name, event_type="agent_started", payload={}),
        )

        loop = asyncio.get_running_loop()
        pipeline = self._get_pipeline()

        # ── Phase 1: MediaPipe extraction + calibration + rule engines ───────
        try:
            analysis, src_path = await loop.run_in_executor(
                self._thread_pool,
                lambda: pipeline.run_analysis(
                    video_path=video_path,
                    session_id=session_id,
                    diar_segments=diar_segments,
                    meeting_type=meeting_type,
                ),
            )
        except Exception as exc:
            logger.exception("[%s] Video analysis failed: %s", session_id, exc)
            _set_video_status(self._redis_repo, session_id, "failed", error=str(exc))
            await self._event_store.append(
                session_id,
                EventRecord(session_id=session_id, agent=self.name, event_type="agent_failed", payload={"error": str(exc)}),
            )
            await self._lock_manager.release(session_id, self.name, lock_token)
            raise

        # Signals ready — pipeline can proceed immediately
        from services.video_agent.main import _publish_video_artifacts
        _publish_video_artifacts(session_id, analysis)
        _set_video_status(self._redis_repo, session_id, "completed", signal_count=len(analysis.signals))
        await self._event_store.append(
            session_id,
            EventRecord(
                session_id=session_id, agent=self.name,
                event_type="signals_ready",
                payload={"signal_count": len(analysis.signals)},
            ),
        )

        logger.info("[%s] Video signals ready: %d signals", session_id, len(analysis.signals))

        result_dict = analysis.model_dump()

        # ── Phase 2: Overlay burn (fire-and-forget) ──────────────────────────
        # Wait 15s grace period so AnalysisPipeline.run() can finish registry
        # matching and write display names to Redis before burn starts.
        asyncio.create_task(
            self._burn_overlay(
                session_id=session_id,
                src_path=src_path,
                all_signals=analysis.signals,
                diar_segments=diar_segments,
                lock_token=lock_token,
            )
        )

        return result_dict

    async def _burn_overlay(
        self,
        session_id: str,
        src_path: str,
        all_signals: list[dict],
        diar_segments: list[dict],
        lock_token: str,
    ) -> None:
        """
        Background coroutine: burns signal labels onto the source video.
        Reads display names from Redis after a grace period.
        """
        # 15s grace period — pipeline writes display names during this window
        await asyncio.sleep(15)

        display_names: dict = {}
        try:
            raw = self._redis_repo.read_artifact(session_id, _DISPLAY_NAMES_ARTIFACT)
            if raw and isinstance(raw, dict):
                display_names = raw
        except Exception as exc:
            logger.debug("[%s] Could not read display names from Redis: %s", session_id, exc)

        if display_names:
            logger.info("[%s] Burn: %d display names", session_id, len(display_names))

        loop = asyncio.get_running_loop()
        pipeline = self._get_pipeline()
        try:
            annotated_path = await loop.run_in_executor(
                self._thread_pool,
                lambda: pipeline.burn_overlay(
                    session_id, src_path, all_signals, diar_segments,
                    display_names=display_names,
                ),
            )
            if annotated_path:
                logger.info("[%s] Annotated video ready: %s", session_id, annotated_path)
        except Exception as exc:
            logger.warning("[%s] Overlay burn failed (non-fatal): %s", session_id, exc)
        finally:
            await self._event_store.append(
                session_id,
                EventRecord(session_id=session_id, agent=self.name, event_type="agent_completed", payload={}),
            )
            await self._lock_manager.release(session_id, self.name, lock_token)


# ── Module-level helpers ─────────────────────────────────────────────────────

def _set_video_status(
    redis_repo: SyncRedisRepository,
    session_id: str,
    status: str,
    signal_count: int = 0,
    error: str = "",
) -> None:
    try:
        redis_repo.set_agent_status(
            session_id, "video",
            AgentStatusRecord(
                status=status,
                signal_count=signal_count,
                summary_key="summary:video",
                error=error or "",
            ),
        )
    except Exception as exc:
        logger.debug("set_video_status failed (non-fatal): %s", exc)
