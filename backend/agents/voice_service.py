# backend/agents/voice_service.py
"""
VoiceAgentService — in-process wrapper for the Voice Agent.

Wraps services/voiceAgent/ (librosa + faster-whisper + pyannote).
The Transcriber class already implements the TranscriptionBackend Strategy via
the TRANSCRIPTION_BACKEND env var — no changes required there.

startup() calls warm_up() which loads Whisper into memory.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from shared.models.requests import VoiceAnalysisRequest, VoiceAnalysisResponse
from shared.redis_layer import (
    AgentStatusRecord,
    EventRecord,
    RedisEventStore,
    RedisLockManager,
    RedisRepository,
    SessionStateRecord,
    SignalRecord,
)

from .base import BaseAgentService

logger = logging.getLogger("nexus.backend.voice")


class VoiceAgentService(BaseAgentService):
    """
    In-process Voice Agent.

    6 core rules: VOICE-CAL-01, VOICE-STRESS-01, VOICE-FILLER-01,
    VOICE-PITCH-01, VOICE-RATE-01, VOICE-TONE-03/04.

    The Transcriber uses TRANSCRIPTION_BACKEND env var to select:
      auto / whisper / assemblyai / deepgram / external_whisper
    """

    name = "voice"

    def __init__(self) -> None:
        self._extractor = None
        self._transcriber = None
        self._rule_engine = None
        self._redis_repo = RedisRepository()
        self._event_store = RedisEventStore()
        self._lock_manager = RedisLockManager()

    async def startup(self) -> None:
        from services.voiceAgent.feature_extractor import VoiceFeatureExtractor
        from services.voiceAgent.rules import VoiceRuleEngine
        from services.voiceAgent.transcriber import Transcriber

        self._extractor = VoiceFeatureExtractor()
        self._transcriber = Transcriber()
        self._rule_engine = VoiceRuleEngine()

        # Whisper warm-up is synchronous but fast enough to run in startup
        self._log("Voice Agent ready.")

    async def shutdown(self) -> None:
        pass

    async def transcribe_only(self, request: VoiceAnalysisRequest) -> dict:
        """
        Fast path — transcript + diarisation without features/rules.
        Used by /quick-transcribe and pipeline transcript-only mode.
        """
        file_path = Path(request.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        session_id = request.session_id or str(uuid.uuid4())
        start_time = time.time()
        tc = request.transcription_config
        ac = request.analysis_config

        logger.info(
            "[%s] Transcribe-only: %s (model=%s, diarize=%s)",
            session_id, file_path.name,
            getattr(tc, "model_preference", None),
            getattr(ac, "run_diarization", True) if ac else True,
        )

        transcript = self._transcriber.transcribe(
            str(file_path),
            num_speakers=request.num_speakers,
            language=getattr(tc, "language", None) if tc else None,
            model_preference=getattr(tc, "model_preference", None) if tc else None,
            custom_prompt=getattr(tc, "custom_prompt", None) if tc else None,
            key_terms=getattr(tc, "key_terms", None) if tc else None,
            multichannel=getattr(tc, "multichannel", False) if tc else False,
            keep_filler_words=getattr(tc, "keep_filler_words", False) if tc else False,
            text_formatting=getattr(tc, "text_formatting", False) if tc else False,
            auto_punctuation=getattr(tc, "auto_punctuation", True) if tc else True,
            temperature=getattr(tc, "temperature", None) if tc else None,
            run_diarization=getattr(ac, "run_diarization", True) if ac else True,
            run_behavioural=getattr(ac, "run_behavioural", True) if ac else True,
            translate_to=getattr(ac, "translate_to", None) if ac else None,
            entity_detection=(
                not getattr(ac, "run_behavioural", True) and getattr(ac, "run_entity_extraction", True)
            ) if ac else False,
        )

        elapsed = time.time() - start_time
        speakers = list({seg["speaker"] for seg in transcript["segments"]})
        logger.info(
            "[%s] Transcribe-only done: %.1fs audio, %d speakers, %d segments in %.1fs",
            session_id, transcript["duration_seconds"],
            len(speakers), len(transcript["segments"]), elapsed,
        )

        return {
            "session_id": session_id,
            "duration_seconds": transcript["duration_seconds"],
            "segments": transcript["segments"],
            "speakers": speakers,
            "backend": transcript.get("backend", "unknown"),
            "model": transcript.get("model", "unknown"),
        }

    async def analyse(self, request: VoiceAnalysisRequest) -> VoiceAnalysisResponse:
        """
        Full voice analysis pipeline (mirrors voiceAgent/main.py::analyse_audio()).

        Pipeline:
          1. Load audio
          2. Transcribe + diarise
          3. Extract acoustic features per speaker per window
          4. Build per-speaker baselines (calibration)
          5. Run 5 core rules
          6. Publish artifacts to Redis
        """
        from services.voiceAgent.calibration import CalibrationModule
        from services.voiceAgent.rules import VoiceRuleEngine

        file_path = Path(request.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        session_id = request.session_id or str(uuid.uuid4())
        start_time = time.time()

        lock_token = await self._lock_manager.acquire(session_id, self.name)
        if not lock_token:
            raise RuntimeError(f"Voice agent already processing session {session_id}")

        await self._redis_repo.set_session_state(
            session_id, SessionStateRecord(status="running", current_step="transcribing")
        )
        await self._redis_repo.set_agent_status(
            session_id, self.name,
            AgentStatusRecord(status="running", summary_key="summary:voice"),
        )
        await self._event_store.append(
            session_id,
            EventRecord(
                session_id=session_id, agent=self.name,
                event_type="agent_started",
                payload={"file_path": str(file_path)},
            ),
        )

        logger.info("[%s] Analysing: %s", session_id, file_path.name)

        # ── Load audio once (reused by diarisation + feature extraction) ────
        try:
            from shared.utils.audio_loader import load_audio
            audio_data = load_audio(str(file_path), sr=16000)
        except ImportError:
            import librosa as _lr
            audio_data = _lr.load(str(file_path), sr=16000, mono=True)

        meeting_type = request.meeting_type or "sales_call"
        _profile = None
        try:
            from shared.config.content_type_profile import ContentTypeProfile
            _profile = ContentTypeProfile(meeting_type)
        except ImportError:
            pass

        tc = request.transcription_config
        ac = request.analysis_config

        # ── Step 1: Transcribe + diarise ─────────────────────────────────────
        t_step = time.time()
        logger.info(
            "[%s] Step 1: Transcribing (num_speakers=%s, meeting_type=%s)",
            session_id, request.num_speakers, meeting_type,
        )
        transcript = self._transcriber.transcribe(
            str(file_path),
            num_speakers=request.num_speakers,
            audio_data=audio_data,
            meeting_type=meeting_type,
            language=getattr(tc, "language", None) if tc else None,
            model_preference=getattr(tc, "model_preference", None) if tc else None,
            custom_prompt=getattr(tc, "custom_prompt", None) if tc else None,
            key_terms=getattr(tc, "key_terms", None) if tc else None,
            multichannel=getattr(tc, "multichannel", False) if tc else False,
            keep_filler_words=getattr(tc, "keep_filler_words", False) if tc else False,
            text_formatting=getattr(tc, "text_formatting", False) if tc else False,
            auto_punctuation=getattr(tc, "auto_punctuation", True) if tc else True,
            temperature=getattr(tc, "temperature", None) if tc else None,
            run_diarization=getattr(ac, "run_diarization", True) if ac else True,
            run_behavioural=getattr(ac, "run_behavioural", True) if ac else True,
            translate_to=getattr(ac, "translate_to", None) if ac else None,
            entity_detection=(
                not getattr(ac, "run_behavioural", True) and getattr(ac, "run_entity_extraction", True)
            ) if ac else False,
        )

        duration_sec = transcript["duration_seconds"]
        speakers = list({seg["speaker"] for seg in transcript["segments"]})
        logger.info(
            "[%s] Step 1 done: %.1fs, %d speakers, %d segments in %.1fs",
            session_id, duration_sec, len(speakers),
            len(transcript["segments"]), time.time() - t_step,
        )

        # ── Early return: transcript-only mode ───────────────────────────────
        run_behavioural = getattr(ac, "run_behavioural", True) if ac else True
        if not run_behavioural:
            word_counts: dict[str, int] = {}
            transcript_speech_q: dict[str, float] = {}
            for seg in transcript["segments"]:
                spk = seg["speaker"]
                word_counts[spk] = word_counts.get(spk, 0) + len(seg.get("text", "").split())
                transcript_speech_q[spk] = transcript_speech_q.get(spk, 0.0) + (seg["end_ms"] - seg["start_ms"]) / 1000.0
            total_sec = sum(transcript_speech_q.values()) or duration_sec

            speaker_payload = [
                {
                    "speaker_id": sid,
                    "baseline": None, "signal_count": 0,
                    "talk_time_ms": int(transcript_speech_q.get(sid, 0.0) * 1000),
                    "talk_time_pct": round(transcript_speech_q.get(sid, 0.0) / total_sec * 100, 2),
                    "total_words": word_counts.get(sid, 0),
                    "calibration_confidence": 0.0,
                }
                for sid in speakers
            ]
            speaker_embeddings = getattr(self._transcriber, "_last_speaker_embeddings", None) or None
            await self._publish_voice_outputs(session_id, transcript, speaker_payload, [], {}, speaker_embeddings)
            await self._redis_repo.set_agent_status(
                session_id, self.name,
                AgentStatusRecord(status="completed", summary_key="summary:voice"),
            )
            await self._event_store.append(
                session_id,
                EventRecord(session_id=session_id, agent=self.name, event_type="agent_completed", payload={"signal_count": 0}),
            )
            await self._lock_manager.release(session_id, self.name, lock_token)
            return VoiceAnalysisResponse(
                session_id=session_id,
                duration_seconds=duration_sec,
                speakers=speaker_payload,
                signals=[],
                summary={},
                transcript_segments=transcript["segments"],
                speaker_embeddings=speaker_embeddings,
            )

        # ── Step 2: Extract acoustic features ────────────────────────────────
        t_step = time.time()
        logger.info("[%s] Step 2: Extracting acoustic features...", session_id)
        try:
            features_by_speaker = self._extractor.extract_all(
                str(file_path), transcript["segments"], audio_data=audio_data,
            )
        except ValueError as exc:
            raise ValueError(
                f"Audio file could not be decoded — it may be empty or corrupted. ({exc})"
            ) from exc

        total_windows = sum(len(v) for v in features_by_speaker.values())
        logger.info(
            "[%s] Step 2 done: %d speakers, %d windows in %.1fs",
            session_id, len(features_by_speaker), total_windows, time.time() - t_step,
        )

        # ── Step 3: Build per-speaker baselines ──────────────────────────────
        transcript_speech: dict[str, float] = {}
        for seg in transcript["segments"]:
            spk = seg["speaker"]
            transcript_speech[spk] = transcript_speech.get(spk, 0.0) + (seg["end_ms"] - seg["start_ms"]) / 1000.0

        t_step = time.time()
        calibration = CalibrationModule()
        baselines: dict = {}
        for speaker_id, features_list in features_by_speaker.items():
            baseline = calibration.build_baseline(
                speaker_id, session_id, features_list,
                transcript_speech_sec=transcript_speech.get(speaker_id, 0.0),
            )
            baselines[speaker_id] = baseline

        logger.info("[%s] Step 3 done: baselines in %.1fs", session_id, time.time() - t_step)

        # ── Step 4: Run rules ────────────────────────────────────────────────
        t_step = time.time()
        all_signals: list = []
        for speaker_id, features_list in features_by_speaker.items():
            baseline = baselines.get(speaker_id)
            if not baseline:
                continue
            for features in features_list:
                window_all_segs = [
                    s for s in transcript["segments"]
                    if s["end_ms"] > features["window_start_ms"]
                    and s["start_ms"] < features["window_end_ms"]
                ]
                sigs = self._rule_engine.evaluate(
                    features=features, baseline=baseline,
                    speaker_id=speaker_id,
                    transcript_segments=window_all_segs,
                    profile=_profile,
                )
                all_signals.extend(sigs)

        talk_time_signals = VoiceRuleEngine._emit_talk_time_signals(features_by_speaker, duration_sec)
        all_signals.extend(talk_time_signals)

        logger.info("[%s] Step 4 done: %d signals in %.1fs", session_id, len(all_signals), time.time() - t_step)

        # ── Step 5: Build summary + publish ─────────────────────────────────
        elapsed = time.time() - start_time
        from services.voiceAgent.main import _build_summary as _voice_summary
        summary = _voice_summary(all_signals, baselines, transcript)

        word_counts: dict[str, int] = {}
        for seg in transcript["segments"]:
            spk = seg["speaker"]
            word_counts[spk] = word_counts.get(spk, 0) + len(seg.get("text", "").split())

        total_speech_sec = sum(transcript_speech.values()) or duration_sec
        speaker_data = [
            {
                "speaker_id": sid,
                "baseline": baselines[sid].to_dict() if sid in baselines else None,
                "signal_count": len([s for s in all_signals if s.get("speaker_id") == sid]),
                "talk_time_ms": int(transcript_speech.get(sid, 0.0) * 1000),
                "talk_time_pct": round(transcript_speech.get(sid, 0.0) / total_speech_sec * 100, 2),
                "total_words": word_counts.get(sid, 0),
                "calibration_confidence": round(
                    baselines[sid].calibration_confidence if sid in baselines else 0.0, 4
                ),
            }
            for sid in speakers
        ]

        signal_dicts = [s if isinstance(s, dict) else s.to_dict() for s in all_signals]
        speaker_embeddings = getattr(self._transcriber, "_last_speaker_embeddings", None) or None

        await self._publish_voice_outputs(session_id, transcript, speaker_data, signal_dicts, summary, speaker_embeddings)
        await self._redis_repo.set_agent_status(
            session_id, self.name,
            AgentStatusRecord(status="completed", signal_count=len(signal_dicts), summary_key="summary:voice"),
        )
        await self._event_store.append(
            session_id,
            EventRecord(
                session_id=session_id, agent=self.name,
                event_type="agent_completed",
                payload={"signal_count": len(signal_dicts)},
            ),
        )
        await self._lock_manager.release(session_id, self.name, lock_token)

        logger.info("[%s] Voice Agent complete: %d signals in %.1fs", session_id, len(signal_dicts), elapsed)

        return VoiceAnalysisResponse(
            session_id=session_id,
            duration_seconds=duration_sec,
            speakers=speaker_data,
            signals=signal_dicts,
            summary=summary,
            transcript_segments=transcript["segments"],
            speaker_embeddings=speaker_embeddings,
        )

    async def _publish_voice_outputs(
        self,
        session_id: str,
        transcript: dict,
        speaker_payload: list[dict],
        signals: list[dict],
        summary: dict,
        speaker_embeddings: Optional[dict],
    ) -> None:
        await self._redis_repo.write_artifact(
            session_id, "transcript",
            {"segments": transcript.get("segments", [])},
        )
        await self._redis_repo.write_artifact(
            session_id, "speakers", {"speakers": speaker_payload}
        )
        await self._redis_repo.write_artifact(
            session_id, "diarization",
            {"segments": transcript.get("segments", [])},
        )
        await self._redis_repo.write_artifact(
            session_id, "summary:voice",
            {
                "summary": summary,
                "duration_seconds": transcript.get("duration_seconds", 0),
                "speaker_embeddings": speaker_embeddings or {},
            },
        )
        if signals:
            await self._redis_repo.publish_signals(
                SignalRecord(
                    session_id=session_id,
                    agent=self.name,
                    speaker_id=sig.get("speaker_id", "unknown"),
                    registry_id=sig.get("registry_id"),
                    signal_type=sig.get("signal_type", ""),
                    value=sig.get("value"),
                    value_text=sig.get("value_text", ""),
                    confidence=sig.get("confidence", 0.5),
                    window_start_ms=sig.get("window_start_ms", 0),
                    window_end_ms=sig.get("window_end_ms", 0),
                    metadata=sig.get("metadata") or {},
                )
                for sig in signals
            )
