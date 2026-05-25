# backend/pipeline/analysis_pipeline.py
"""
NEXUS Backend — AnalysisPipeline orchestrator.

Orchestrates the 5 in-process agent services for one analysis session.
Stateless with respect to sessions — holds only model references loaded
at startup.

Execution graph (mirrors _run_pipeline in services/api_gateway/main.py):
    Voice                             (sequential — transcript needed by all)
    Language + Video                  (asyncio.gather — parallel)
    Conversation                      (sequential — needs language signals)
    Fusion                            (sequential — needs all prior signals)
    Neo4j + knowledge store sync      (sequential, non-fatal)

DSA: asyncio.gather(Language, Video) reduces total time from
     T_lang + T_video → max(T_lang, T_video), saving 2–8 min per session.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid as _uuid_module
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from shared.models.requests import (
    ConversationAnalysisRequest,
    FusionAnalyseRequest,
    FusionSignalInput,
    LanguageAnalysisRequest,
    VoiceAnalysisRequest,
)
from shared.redis_layer import SessionStateRecord

if TYPE_CHECKING:
    from agents.base import BaseAgentService
    from agents.conversation_service import ConversationAgentService
    from agents.fusion_service import FusionAgentService
    from agents.language_service import LanguageAgentService
    from agents.video_service import VideoAgentService
    from agents.voice_service import VoiceAgentService

logger = logging.getLogger("nexus.backend.pipeline")

_UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/recordings"))
_RECORDING_RETENTION_DAYS = int(os.getenv("RECORDING_RETENTION_DAYS", "3"))
_SESSION_FACE_LOCK_MIN_SCORE = float(os.getenv("SESSION_FACE_LOCK_MIN_SCORE", "0.06"))


class AnalysisPipeline:
    """
    Orchestrates 5 agent services for one analysis session.

    Stateless with respect to sessions — holds only model references loaded
    at startup. Thread-safe: async methods only; no shared mutable state.
    """

    def __init__(
        self,
        voice: "VoiceAgentService",
        language: "LanguageAgentService",
        conversation: "ConversationAgentService",
        video: "VideoAgentService",
        fusion: "FusionAgentService",
        redis_repo,
    ) -> None:
        self._voice = voice
        self._language = language
        self._conversation = conversation
        self._video = video
        self._fusion = fusion
        self._redis_repo = redis_repo

    @property
    def services(self) -> list["BaseAgentService"]:
        return [self._voice, self._language, self._conversation, self._video, self._fusion]

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        session_id: str,
        file_path: str,
        video_path: Optional[str],
        meeting_type: str,
        num_speakers: Optional[int],
        pool,
        org_id: str,
        user_id: str,
        run_behavioural: bool = True,
        title: str = "",
        transcription_config: Optional[dict] = None,
        analysis_config: Optional[dict] = None,
        user_email: str = "",
    ) -> None:
        """
        Background task: full analysis pipeline.

        All exceptions caught and logged — never propagated to caller.
        Mirrors services/api_gateway/main.py::_run_pipeline() with HTTP calls
        replaced by direct in-process service method calls.
        """
        transcription_config = transcription_config or {}
        analysis_config = analysis_config or {}
        logger.info(
            "[%s] Pipeline starting meeting_type=%s sensitivity=%s",
            session_id, meeting_type, analysis_config.get("sensitivity", 0.5),
        )

        t_pipeline = time.monotonic()
        _t_voice = _t_lang_video = _t_conv = _t_fusion = 0.0

        run_sentiment = analysis_config.get("run_sentiment", False)
        run_entity_extraction = analysis_config.get("run_entity_extraction", True)

        # ── Step 1: Voice Agent ───────────────────────────────────────────────
        await self._set_step(session_id, "transcribing")
        voice_result: dict = {}
        _t0 = time.monotonic()
        try:
            resp = await self._voice.analyse(
                VoiceAnalysisRequest(
                    file_path=file_path,
                    session_id=session_id,
                    meeting_type=meeting_type,
                    num_speakers=num_speakers,
                    transcription_config=transcription_config or None,
                    analysis_config=analysis_config or None,
                )
            )
            voice_result = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            _t_voice = time.monotonic() - _t0
            logger.info(
                "[%s] Voice: %.0fs audio, %d speakers, stage=%.0fs",
                session_id,
                voice_result.get("duration_seconds", 0),
                len(voice_result.get("speakers", [])),
                _t_voice,
            )
        except Exception as exc:
            logger.error("[%s] Voice Agent failed: %s", session_id, exc)
            await self._redis_repo.set_session_state(
                session_id,
                SessionStateRecord(status="failed", current_step="transcribing", error=str(exc)),
            )
            await self._try_update_status(session_id, "failed", pool=pool)
            return

        duration_seconds = voice_result.get("duration_seconds", 0)
        voice_signals: list[dict] = voice_result.get("signals", [])
        voice_speakers: list[dict] = voice_result.get("speakers", [])
        voice_summary: dict = voice_result.get("summary", {})
        speaker_count = len(voice_speakers)
        speaker_embeddings: dict = voice_result.get("speaker_embeddings") or {}

        # Persist speakers first — FK requirement for signals
        speaker_map: dict = {}
        try:
            from core.database import upsert_speakers
            speaker_map = await upsert_speakers(session_id, voice_speakers)
        except Exception as exc:
            logger.warning("[%s] Speaker upsert failed: %s", session_id, exc)

        await self._persist_signals(session_id, voice_signals, speaker_map, "voice", pool)

        transcript_segments = _extract_transcript_segments(voice_result)
        if transcript_segments:
            try:
                from core.database import insert_transcript_segments
                await insert_transcript_segments(session_id, transcript_segments, speaker_map)
            except Exception as exc:
                logger.warning("[%s] Transcript persist failed: %s", session_id, exc)

        # ── Step 2: Language + Video in parallel ─────────────────────────────
        await self._set_step(session_id, "language")

        agent_status = {
            "voice": "completed",
            "language": "skipped",
            "conversation": "skipped",
            "video": "skipped",
            "fusion": "skipped",
        }

        run_video = video_path is not None and run_behavioural
        diar_segments_for_video: list[dict] = []
        if run_video:
            diar_segments_for_video = [
                {
                    "speaker": seg.get("speaker", "unknown"),
                    "start_ms": int(seg.get("start_ms", 0)),
                    "end_ms": int(seg.get("end_ms", 0)),
                }
                for seg in transcript_segments
                if seg.get("start_ms") is not None
            ]

        run_conversation = (
            run_behavioural
            and bool(transcript_segments)
            and len({seg.get("speaker") for seg in transcript_segments}) >= 2
        )

        async def _run_language() -> dict:
            try:
                if run_behavioural or run_sentiment:
                    if not transcript_segments:
                        logger.warning("[%s] No segments — skipping Language", session_id)
                        return {}
                    resp = await self._language.analyse(
                        LanguageAnalysisRequest(
                            segments=transcript_segments,
                            session_id=session_id,
                            meeting_type=meeting_type,
                            run_intent_classification=True,
                        )
                    )
                    agent_status["language"] = "completed"
                    return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

                if run_entity_extraction and transcript_segments:
                    assemblyai_raw = voice_result.get("assemblyai_entities")
                    if assemblyai_raw is not None:
                        entities = _format_assemblyai_entities(assemblyai_raw)
                        return {"summary": {"entities": entities}}
                    try:
                        ee = getattr(self._language, "_entity_extractor", None)
                        if ee:
                            entities = await ee.extract(transcript_segments, meeting_type)
                            return {"summary": {"entities": entities}}
                    except Exception as exc:
                        logger.warning("[%s] Entity extraction failed (non-fatal): %s", session_id, exc)
                    return {}

                return {}
            except Exception as exc:
                agent_status["language"] = "failed"
                logger.warning("[%s] Language Agent failed (continuing): %s", session_id, exc)
                return {}

        async def _run_video() -> tuple[list[dict], dict, dict, dict]:
            """Returns (signals, face_embeddings, face_to_speaker, lip_sync_scores)."""
            if not run_video or video_path is None:
                return [], {}, {}, {}
            try:
                await self._set_step(session_id, "video")
                result = await self._video.analyse(
                    session_id=session_id,
                    video_path=video_path,
                    diar_segments=diar_segments_for_video,
                    meeting_type=meeting_type,
                    num_speakers=speaker_count or (num_speakers or 2),
                )
                sigs = result.get("signals", [])
                face_embs = result.get("face_embeddings", {})
                f2s = {int(k): v for k, v in result.get("face_to_speaker", {}).items()}
                lip_scores = result.get("lip_sync_scores", {})
                agent_status["video"] = "completed"
                logger.info(
                    "[%s] Video: %d signals, %d face embeddings",
                    session_id, len(sigs), len(face_embs),
                )
                return sigs, face_embs, f2s, lip_scores
            except Exception as exc:
                agent_status["video"] = "failed"
                logger.warning("[%s] Video Agent failed (continuing): %s", session_id, exc)
                return [], {}, {}, {}

        _t0 = time.monotonic()
        lang_outcome, (vid_signals, face_embeddings_from_video, face_to_speaker, lip_sync_scores) = (
            await asyncio.gather(_run_language(), _run_video())
        )
        _t_lang_video = time.monotonic() - _t0

        language_signals: list[dict] = []
        language_summary: dict = {}
        if lang_outcome:
            language_signals = lang_outcome.get("signals", [])
            language_summary = lang_outcome.get("summary", {})
            logger.info("[%s] Language: %d signals", session_id, len(language_signals))
            await self._persist_signals(session_id, language_signals, speaker_map, "language", pool)

        # ── Step 3: Conversation (sequential — needs language signals) ────────
        await self._set_step(session_id, "conversation")
        conversation_signals: list[dict] = []
        conversation_summary: dict = {}
        _t0 = time.monotonic()

        if run_conversation:
            try:
                speakers_list = list({seg.get("speaker", "unknown") for seg in transcript_segments})
                conv_resp = await self._conversation.analyse(
                    ConversationAnalysisRequest(
                        segments=transcript_segments,
                        speakers=speakers_list,
                        content_type=meeting_type,
                        session_id=session_id,
                        language_signals=language_signals or None,
                    )
                )
                conv_result = conv_resp.model_dump() if hasattr(conv_resp, "model_dump") else dict(conv_resp)
                conversation_signals = conv_result.get("signals", [])
                conversation_summary = conv_result.get("summary", {})
                logger.info("[%s] Conversation: %d signals", session_id, len(conversation_signals))
                agent_status["conversation"] = "completed"
            except Exception as exc:
                agent_status["conversation"] = "failed"
                logger.warning("[%s] Conversation Agent failed (continuing): %s", session_id, exc)
            await self._persist_signals(session_id, conversation_signals, speaker_map, "conversation", pool)
        elif not run_behavioural:
            logger.info("[%s] Behavioural off — skipping Conversation", session_id)
        else:
            logger.info("[%s] < 2 speakers — skipping Conversation", session_id)
        _t_conv = time.monotonic() - _t0

        # ── Step 4: Video signal filtering + registry matching ────────────────
        video_signals: list[dict] = []
        video_speaker_map = dict(speaker_map)

        if vid_signals:
            video_signals = vid_signals

            # One-pass: count signals per Face_* and collect IDs for set arithmetic
            MIN_FACE_SIGNALS = 10
            face_signal_counts: dict[str, int] = {}
            face_ids_in_signals: set[str] = set()
            for sig in video_signals:
                spk = sig.get("speaker_id", "")
                if spk.startswith("Face_"):
                    face_ids_in_signals.add(spk)
                    # presence_detected is a marker, not a behavioral signal —
                    # exclude from the weak-face count so a face with only a
                    # presence marker still passes the MIN_FACE_SIGNALS filter.
                    if sig.get("signal_type") != "presence_detected":
                        face_signal_counts[spk] = face_signal_counts.get(spk, 0) + 1

            # Drop Face_* with < MIN_FACE_SIGNALS before anything else
            weak_faces: set[str] = {fid for fid, cnt in face_signal_counts.items() if cnt < MIN_FACE_SIGNALS}
            if weak_faces:
                before = len(video_signals)
                video_signals = [s for s in video_signals if s.get("speaker_id", "") not in weak_faces]
                logger.info(
                    "[%s] Filtered %d weak Face_* (< %d signals) — removed %d signals",
                    session_id, len(weak_faces), MIN_FACE_SIGNALS, before - len(video_signals),
                )

            unmatched_face_ids = face_ids_in_signals - weak_faces - set(video_speaker_map.keys())
            if unmatched_face_ids:
                try:
                    from core.database import upsert_speakers
                    new_face_speakers = await upsert_speakers(
                        session_id,
                        [{"speaker_id": fid} for fid in unmatched_face_ids],
                    )
                    video_speaker_map.update(new_face_speakers)
                except Exception as exc:
                    logger.warning("[%s] Face speaker upsert failed: %s", session_id, exc)

        # ── Registry matching (after gather — both voice + face embeddings ready) ─
        locked_face_to_speaker, locked_speaker_to_face = _build_session_face_locks(
            face_to_speaker=face_to_speaker,
            lip_sync_scores=lip_sync_scores,
            face_embeddings_from_video=face_embeddings_from_video,
            session_id=session_id,
        )

        # Remap face embeddings by locked speaker label for fused registry matching
        speaker_keyed_face_embs: dict = {}
        for face_label, speaker_label in locked_face_to_speaker.items():
            face_data = face_embeddings_from_video.get(face_label)
            if not face_data:
                continue
            score = float(lip_sync_scores.get(speaker_label, 0.0) or 0.0)
            speaker_keyed_face_embs[speaker_label] = {
                **face_data,
                "source_face_label": face_label,
                "session_face_lock": True,
                "lip_sync_score": score,
            }

        speaker_identity_map: dict = {}
        if speaker_embeddings:
            try:
                from core.speaker_registry import match_or_create_speakers
                speaker_identity_map = await match_or_create_speakers(
                    pool=pool,
                    session_id=session_id,
                    speaker_embeddings=speaker_embeddings,
                    voice_speakers=voice_speakers,
                    speaker_map=speaker_map,
                    org_id=org_id,
                    face_embeddings=speaker_keyed_face_embs,
                )
                logger.info("[%s] Speaker identity: %s", session_id, speaker_identity_map)
            except Exception as exc:
                logger.warning("[%s] Speaker registry match failed (non-fatal): %s", session_id, exc)
        elif face_embeddings_from_video:
            try:
                from core.speaker_registry import match_or_create_by_face_only
                speaker_identity_map = await match_or_create_by_face_only(
                    pool=pool,
                    session_id=session_id,
                    face_embeddings=face_embeddings_from_video,
                    speaker_map=speaker_map,
                    org_id=org_id,
                )
                logger.info("[%s] Face-only identity: %s", session_id, speaker_identity_map)
            except Exception as exc:
                logger.warning("[%s] Face-only registry match failed (non-fatal): %s", session_id, exc)

        # Register non-speaking faces (Face_N not linked to any Speaker_N)
        if speaker_embeddings and face_embeddings_from_video:
            linked_face_labels = set(locked_face_to_speaker.keys())
            non_speaking = {
                label: data
                for label, data in face_embeddings_from_video.items()
                if label.startswith("Face_")
                and label not in speaker_identity_map
                and label not in linked_face_labels
            }
            if non_speaking:
                try:
                    from core.speaker_registry import match_or_create_by_face_only
                    face_only_matches = await match_or_create_by_face_only(
                        pool=pool,
                        session_id=session_id,
                        face_embeddings=non_speaking,
                        speaker_map=video_speaker_map,
                        org_id=org_id,
                    )
                    speaker_identity_map.update(face_only_matches)
                    logger.info(
                        "[%s] Non-speaking identities: %d registered",
                        session_id, len(face_only_matches),
                    )
                except Exception as exc:
                    logger.warning("[%s] Non-speaking face registry failed (non-fatal): %s", session_id, exc)

        # Alias Face_N → Speaker_N registry entries via lip-sync lock
        if locked_face_to_speaker:
            for face_label, speaker_label in locked_face_to_speaker.items():
                if speaker_label not in speaker_identity_map:
                    logger.warning(
                        "[%s] Cannot alias locked face %s → %s: no registry identity",
                        session_id, face_label, speaker_label,
                    )
                    continue
                if face_label not in speaker_identity_map:
                    score = float(lip_sync_scores.get(speaker_label, 0.0) or 0.0)
                    speaker_identity_map[face_label] = {
                        **speaker_identity_map[speaker_label],
                        "match_method": "session_lip_sync_lock",
                        "match_confidence": min(0.95, max(0.0, score)),
                        "source_speaker_label": speaker_label,
                    }

        # Persist speaker_appearances DB row for each linked Face_N.
        # The aliasing above writes Face_N into speaker_identity_map in memory,
        # but /video-signals JOINs on speaker_appearances to resolve registry_id.
        # Without this row Face_N signals have registry_id = NULL and the
        # dashboard canonical merge cannot group Face_N with its Speaker_N.
        if locked_face_to_speaker:
            try:
                from core.speaker_registry import _upsert_appearance
                for face_label, speaker_label in locked_face_to_speaker.items():
                    if face_label not in speaker_identity_map:
                        continue
                    reg_id = speaker_identity_map[face_label].get("registry_id")
                    if not reg_id:
                        continue
                    score = float(lip_sync_scores.get(speaker_label, 0.0) or 0.0)
                    await _upsert_appearance(
                        pool, reg_id, session_id,
                        video_speaker_map.get(face_label),
                        face_label,
                        "session_lip_sync_lock",
                        min(0.95, score),
                    )
                    logger.info(
                        "[%s] speaker_appearances persisted: %s → %s (registry_id=%s)",
                        session_id, face_label, speaker_label, reg_id,
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] Linked face appearance persist failed (non-fatal): %s",
                    session_id, exc,
                )

        # Persist video signals — after registry so unregistered faces are excluded
        if vid_signals:
            registered_face_labels: set[str] = {
                label for label in speaker_identity_map if label.startswith("Face_")
            }
            all_face_labels_in_signals: set[str] = {
                s.get("speaker_id", "")
                for s in video_signals
                if s.get("speaker_id", "").startswith("Face_")
            }
            unregistered_faces = all_face_labels_in_signals - registered_face_labels
            if unregistered_faces:
                before = len(video_signals)
                video_signals = [s for s in video_signals if s.get("speaker_id", "") not in unregistered_faces]
                logger.info(
                    "[%s] Excluded %d unregistered Face_* — dropped %d signals",
                    session_id, len(unregistered_faces), before - len(video_signals),
                )
            await self._persist_signals(session_id, video_signals, video_speaker_map, "video", pool)

        # Write display names to Redis for overlay burn (replaces HTTP POST to video agent)
        if speaker_identity_map:
            display_name_map = {
                label: info["display_name"]
                for label, info in speaker_identity_map.items()
                if info.get("display_name") and info["display_name"] != label
            }
            if display_name_map:
                asyncio.create_task(
                    self._write_display_names(session_id, display_name_map)
                )

        # ── Step 5: Fusion ────────────────────────────────────────────────────
        await self._set_step(session_id, "fusion")
        _t0 = time.monotonic()
        fusion_signals: list[dict] = []
        alerts: list[dict] = []
        report: Optional[dict] = None
        fusion_result: Optional[dict] = None

        if run_behavioural:
            enriched_voice_summary = dict(voice_summary)
            if conversation_summary:
                enriched_voice_summary["conversation"] = conversation_summary

            # Convert signals to FusionSignalInput format (mirrors _to_fusion_input in gateway)
            def _to_fusion_input(sig: dict, agent: str) -> dict:
                return {
                    "agent": sig.get("agent", agent),
                    "speaker_id": sig.get("speaker_id", "unknown"),
                    "signal_type": sig.get("signal_type", ""),
                    "value": sig.get("value"),
                    "value_text": sig.get("value_text", ""),
                    "confidence": sig.get("confidence", 0.5),
                    "window_start_ms": sig.get("window_start_ms", 0),
                    "window_end_ms": sig.get("window_end_ms", 0),
                    "metadata": sig.get("metadata"),
                }

            all_voice_side = [_to_fusion_input(s, "voice") for s in voice_signals]
            video_summary_for_fusion: Optional[dict] = None
            if video_signals:
                all_voice_side += [_to_fusion_input(s, "video") for s in video_signals]
                video_summary_for_fusion = _build_video_summary(video_signals)

            try:
                fusion_resp = await self._fusion.analyse(
                    FusionAnalyseRequest(
                        voice_signals=[FusionSignalInput(**s) for s in all_voice_side],
                        language_signals=[
                            FusionSignalInput(**_to_fusion_input(s, "language"))
                            for s in language_signals
                        ],
                        session_id=session_id,
                        meeting_type=meeting_type,
                        generate_report=True,
                        voice_summary=enriched_voice_summary,
                        language_summary=language_summary,
                        video_summary=video_summary_for_fusion,
                    )
                )
                fusion_result = fusion_resp.model_dump() if hasattr(fusion_resp, "model_dump") else dict(fusion_resp)
                fusion_signals = fusion_result.get("fusion_signals", [])
                alerts = fusion_result.get("alerts", [])
                report = fusion_result.get("report")
                agent_status["fusion"] = "completed"
                logger.info(
                    "[%s] Fusion: %d signals, %d alerts", session_id, len(fusion_signals), len(alerts)
                )
            except Exception as exc:
                agent_status["fusion"] = "failed"
                logger.warning("[%s] Fusion Agent failed (continuing): %s", session_id, exc)
        else:
            logger.info("[%s] Behavioural off — skipping Fusion", session_id)
        _t_fusion = time.monotonic() - _t0

        await self._persist_signals(session_id, fusion_signals, video_speaker_map, "fusion", pool)

        # Back-fill speaker_appearances stats now all signals are persisted
        if speaker_embeddings:
            try:
                from core.speaker_registry import update_appearance_stats
                await update_appearance_stats(pool, session_id)
            except Exception as exc:
                logger.warning("[%s] Appearance stats update failed (non-fatal): %s", session_id, exc)

        if alerts:
            try:
                from core.database import insert_alerts
                count = await insert_alerts(session_id, alerts, video_speaker_map)
                logger.info("[%s] Persisted %d alerts", session_id, count)
            except Exception as exc:
                logger.warning("[%s] Alert persist failed: %s", session_id, exc)

        # ── Step 6: Persist report ────────────────────────────────────────────
        await self._set_step(session_id, "report")
        entities = language_summary.get("entities", {})
        fusion_summary = fusion_result.get("summary", {}) if fusion_result else {}
        signal_graph = fusion_summary.get("signal_graph", {})
        key_paths = fusion_summary.get("key_paths", [])
        graph_analytics = fusion_summary.get("graph_analytics", {})

        report_content: dict = report or {}
        if entities or signal_graph:
            report_content["entities"] = entities
            report_content["signal_graph"] = signal_graph
            report_content["key_paths"] = key_paths
            report_content["graph_analytics"] = graph_analytics

        report_generated = False
        if report_content:
            try:
                from core.database import save_report
                await save_report(
                    session_id=session_id,
                    content=report_content,
                    narrative=report_content.get("executive_summary", ""),
                )
                report_generated = True
            except Exception as exc:
                logger.warning("[%s] Report persist failed: %s", session_id, exc)

        # ── Finalise session ──────────────────────────────────────────────────
        any_failed = any(v == "failed" for v in agent_status.values())
        final_status = "partial" if any_failed else "completed"

        await self._try_update_status(
            session_id, final_status, pool=pool,
            duration_ms=int(duration_seconds * 1000),
            speaker_count=speaker_count,
            participant_count=speaker_count,
        )
        logger.info(
            "[%s] Stage timings (s): voice=%.0f lang+video=%.0f convo=%.0f fusion=%.0f total=%.0f",
            session_id, _t_voice, _t_lang_video, _t_conv, _t_fusion,
            time.monotonic() - t_pipeline,
        )
        logger.info("[%s] Pipeline complete status=%s agents=%s", session_id, final_status, agent_status)

        await self._post_process(
            session_id=session_id,
            run_behavioural=run_behavioural,
            run_knowledge_graph=analysis_config.get("run_knowledge_graph", True),
            transcript_segments=transcript_segments,
            all_signals=(
                voice_signals + language_signals + conversation_signals
                + video_signals + fusion_signals
            ),
            entities=entities,
            report_content=report_content,
            graph_analytics=graph_analytics,
            conversation_summary=conversation_summary,
            pool=pool,
        )

        # Completion email — only for full behavioural runs (takes 5-30+ min)
        if run_behavioural and user_email:
            try:
                from core.email_service import is_email_configured, send_processing_complete_email
                if is_email_configured():
                    user_name = "there"
                    try:
                        row = await pool.fetchrow(
                            "SELECT full_name FROM users WHERE email = $1", user_email
                        )
                        if row and row["full_name"]:
                            user_name = row["full_name"]
                    except Exception:
                        pass
                    await send_processing_complete_email(
                        to_email=user_email,
                        to_name=user_name,
                        session_id=session_id,
                        session_title=title or session_id,
                        meeting_type=meeting_type,
                        status=final_status,
                        signal_counts={
                            "voice": len(voice_signals),
                            "language": len(language_signals),
                            "conversation": len(conversation_signals),
                            "video": len(video_signals),
                            "fusion": len(fusion_signals),
                        },
                        duration_seconds=duration_seconds,
                    )
            except Exception as exc:
                logger.warning("[%s] Completion email failed (non-fatal): %s", session_id, exc)

        _cleanup_old_recordings()
        await self._redis_repo.set_session_state(
            session_id,
            SessionStateRecord(status=final_status, current_step="completed"),
        )
        logger.info(
            "[%s] Pipeline finished: status=%s voice=%d lang=%d convo=%d video=%d fusion=%d alerts=%d report=%s",
            session_id, final_status,
            len(voice_signals), len(language_signals), len(conversation_signals),
            len(video_signals), len(fusion_signals),
            len(alerts), "yes" if report_generated else "no",
        )

    async def run_quick_transcribe(
        self,
        file_path: str,
        session_id: str,
        config: dict,
    ) -> dict:
        """
        Lightweight path for /quick-transcribe — no DB, no full pipeline.
        Delegates directly to VoiceAgentService.transcribe_only().
        """
        tc = config.get("transcription_config") if config else None
        return await self._voice.transcribe_only(
            VoiceAnalysisRequest(
                file_path=file_path,
                session_id=session_id,
                transcription_config=tc,
            )
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _set_step(self, session_id: str, step: str, error: str = "") -> None:
        await self._redis_repo.set_session_state(
            session_id,
            SessionStateRecord(status="running", current_step=step, error=error),
        )

    async def _persist_signals(
        self,
        session_id: str,
        signals: list[dict],
        speaker_map: dict,
        label: str,
        pool,
    ) -> None:
        if not signals:
            return
        try:
            from core.database import insert_signals
            await pool.execute(
                "DELETE FROM signals WHERE session_id = $1 AND agent = $2",
                _uuid_module.UUID(session_id), label,
            )
            count = await insert_signals(session_id, signals, speaker_map)
            logger.info("[%s] Persisted %d %s signals", session_id, count, label)
        except Exception as exc:
            logger.warning("[%s] %s signal persist failed: %s", session_id, label, exc)

    async def _try_update_status(
        self,
        session_id: str,
        status: str,
        pool=None,  # unused — update_session_status uses internal get_pool()
        duration_ms: Optional[int] = None,
        speaker_count: Optional[int] = None,
        participant_count: Optional[int] = None,
    ) -> None:
        try:
            from core.database import update_session_status
            await update_session_status(
                session_id, status,
                duration_ms=duration_ms,
                speaker_count=speaker_count,
                participant_count=participant_count,
            )
        except Exception as exc:
            logger.warning("[%s] Status update failed: %s", session_id, exc)

    async def _write_display_names(self, session_id: str, display_name_map: dict) -> None:
        """Write display names to Redis for video overlay burn (replaces HTTP POST to video agent)."""
        try:
            await self._redis_repo.write_artifact(session_id, "display_names", display_name_map)
            logger.info("[%s] Display names written to Redis: %s", session_id, list(display_name_map))
        except Exception as exc:
            logger.debug("[%s] Display names write failed (non-fatal): %s", session_id, exc)

    async def _post_process(
        self,
        session_id: str,
        run_behavioural: bool,
        run_knowledge_graph: bool,
        transcript_segments: list[dict],
        all_signals: list[dict],
        entities: dict,
        report_content: dict,
        graph_analytics: dict,
        conversation_summary: dict,
        pool,
    ) -> None:
        """Knowledge store embedding + Neo4j sync — both non-fatal."""
        await self._set_step(session_id, "entity_extraction")
        if run_behavioural:
            try:
                from core.knowledge_store import store_session_knowledge
                await store_session_knowledge(pool, session_id, {
                    "transcript_segments": transcript_segments,
                    "signals": all_signals,
                    "entities": entities,
                    "report": report_content or {},
                    "graph_analytics": graph_analytics or {},
                    "conversation_summary": conversation_summary or {},
                })
            except Exception as exc:
                logger.warning("[%s] Knowledge store failed (non-fatal): %s", session_id, exc)

        await self._set_step(session_id, "knowledge_graph")
        if run_knowledge_graph:
            try:
                from core.neo4j_sync import sync_session as neo4j_sync_session
                await neo4j_sync_session(pool, session_id)
            except Exception as exc:
                logger.warning("[%s] Neo4j sync failed (non-fatal): %s", session_id, exc)

            try:
                from core.neo4j_sync import sync_speaker_registry_to_neo4j
                reg_rows = await pool.fetch(
                    "SELECT DISTINCT registry_id FROM speaker_appearances WHERE session_id = $1",
                    _uuid_module.UUID(session_id),
                )
                for row in reg_rows:
                    await sync_speaker_registry_to_neo4j(pool, str(row["registry_id"]))
            except Exception as exc:
                logger.warning("[%s] Speaker registry Neo4j sync failed (non-fatal): %s", session_id, exc)


# ── Module-level helpers (stateless — no self reference needed) ───────────────

def _extract_transcript_segments(voice_result: dict) -> list[dict]:
    """Extract transcript segments from Voice Agent result dict."""
    segments = voice_result.get("transcript_segments", [])
    if segments:
        return segments

    # Reconstruct from signal metadata as fallback
    signals = voice_result.get("signals", [])
    seen: set[tuple] = set()
    reconstructed: list[dict] = []
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
        speaker = s.get("speaker_id", "unknown")
        key = (start_ms, speaker)
        if key in seen:
            continue
        seen.add(key)
        reconstructed.append({
            "speaker": speaker,
            "start_ms": start_ms,
            "end_ms": s.get("window_end_ms", 0),
            "text": text,
        })
    return reconstructed


def _format_assemblyai_entities(raw: list[dict]) -> dict:
    """Convert AssemblyAI entity_detection results to NEXUS entity format."""
    _TYPE_MAP = {
        "person_name": "people",
        "organization": "organizations",
        "location": "locations",
        "product_name": "products",
    }
    result: dict[str, list] = {
        "people": [], "organizations": [], "locations": [],
        "products": [], "topics": [], "objections": [], "commitments": [],
    }
    seen: set[tuple] = set()
    for ent in raw:
        bucket = _TYPE_MAP.get(ent.get("entity_type", ""))
        if not bucket:
            continue
        text = (ent.get("text") or "").strip()
        key = (bucket, text.lower())
        if text and key not in seen:
            seen.add(key)
            result[bucket].append({"text": text, "start_ms": ent.get("start"), "end_ms": ent.get("end")})
    return result


def _build_session_face_locks(
    face_to_speaker: dict,
    lip_sync_scores: dict,
    face_embeddings_from_video: dict,
    session_id: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build a 1:1 mapping of face labels → speaker labels based on lip-sync score.

    Only accepts links above SESSION_FACE_LOCK_MIN_SCORE. Greedy assignment:
    sort candidates by score descending so the strongest link wins any conflict.
    Returns (locked_face_to_speaker, locked_speaker_to_face).
    """
    locked_f2s: dict[str, str] = {}
    locked_s2f: dict[str, str] = {}
    if not face_to_speaker or not lip_sync_scores:
        return locked_f2s, locked_s2f

    candidates: list[tuple[float, str, str]] = []
    for raw_face_idx, speaker_label in face_to_speaker.items():
        if not str(speaker_label).startswith("Speaker_"):
            continue
        try:
            face_label = (
                raw_face_idx if str(raw_face_idx).startswith("Face_")
                else f"Face_{int(raw_face_idx)}"
            )
        except Exception:
            face_label = f"Face_{raw_face_idx}"

        if face_label not in face_embeddings_from_video:
            continue

        score = float(lip_sync_scores.get(speaker_label, 0.0) or 0.0)
        candidates.append((score, face_label, speaker_label))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for score, face_label, speaker_label in candidates:
        if score < _SESSION_FACE_LOCK_MIN_SCORE:
            logger.warning(
                "[%s] Rejecting weak face lock %s → %s score=%.4f threshold=%.4f",
                session_id, face_label, speaker_label, score, _SESSION_FACE_LOCK_MIN_SCORE,
            )
            continue
        if face_label in locked_f2s or speaker_label in locked_s2f:
            continue
        locked_f2s[face_label] = speaker_label
        locked_s2f[speaker_label] = face_label
        logger.info(
            "[%s] Session face lock: %s → %s score=%.4f",
            session_id, face_label, speaker_label, score,
        )

    return locked_f2s, locked_s2f


def _build_video_summary(video_signals: list[dict]) -> dict:
    """Aggregate raw video signals into a per-speaker summary dict for Fusion LLM context."""
    per_speaker: dict = defaultdict(lambda: {
        "emotions": [], "facial_stress": [], "facial_engagement": [],
        "gaze_on_screen": [], "blink_anomalies": 0, "head_nods": 0,
        "head_shakes": 0, "body_movement": [], "posture_changes": 0,
        "valence_arousal": [], "hand_near_face": 0, "gaze_breaks": 0,
    })

    for s in video_signals:
        spk = s.get("speaker_id") or s.get("speaker_label") or "unknown"
        st = s.get("signal_type", "")
        val = s.get("value")
        vt = s.get("value_text", "")
        conf = s.get("confidence", 0.5)
        sp = per_speaker[spk]

        if st == "facial_emotion" and vt:
            sp["emotions"].append((vt, conf))
        elif st == "facial_stress" and val is not None:
            sp["facial_stress"].append(val)
        elif st == "facial_engagement" and val is not None:
            sp["facial_engagement"].append(val)
        elif st == "screen_contact" and val is not None:
            sp["gaze_on_screen"].append(val)
        elif st == "blink_rate_anomaly":
            sp["blink_anomalies"] += 1
        elif st == "head_nod":
            sp["head_nods"] += 1
        elif st == "head_shake":
            sp["head_shakes"] += 1
        elif st == "body_fidgeting" and val is not None:
            sp["body_movement"].append(val)
        elif st == "posture_transition":
            sp["posture_changes"] += 1
        elif st == "valence_arousal" and vt:
            sp["valence_arousal"].append((vt, val or 0))
        elif st == "self_touch":
            sp["hand_near_face"] += 1
        elif st in ("gaze_direction_shift", "sustained_distraction"):
            sp["gaze_breaks"] += 1

    def _avg(lst: list) -> Optional[float]:
        return round(sum(lst) / len(lst), 3) if lst else None

    summary: dict = {}
    for spk, data in per_speaker.items():
        emotion_counts = Counter(e for e, _ in data["emotions"])
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else None
        dominant_emotion_conf = (
            max((c for e, c in data["emotions"] if e == dominant_emotion), default=0.0)
            if dominant_emotion else 0.0
        )
        va_counts = Counter(vt for vt, _ in data["valence_arousal"])
        dominant_valence = va_counts.most_common(1)[0][0] if va_counts else None

        summary[spk] = {
            "dominant_emotion": dominant_emotion,
            "dominant_emotion_confidence": round(dominant_emotion_conf, 3) if dominant_emotion else None,
            "dominant_valence": dominant_valence,
            "avg_facial_stress": _avg(data["facial_stress"]),
            "avg_facial_engagement": _avg(data["facial_engagement"]),
            "avg_gaze_on_screen_pct": _avg(data["gaze_on_screen"]),
            "gaze_breaks": data["gaze_breaks"],
            "blink_anomalies": data["blink_anomalies"],
            "head_nods": data["head_nods"],
            "head_shakes": data["head_shakes"],
            "avg_body_movement": _avg(data["body_movement"]),
            "posture_changes": data["posture_changes"],
            "hand_near_face_events": data["hand_near_face"],
        }

    return {"per_speaker": summary}


def _cleanup_old_recordings() -> None:
    """Delete recording files older than RECORDING_RETENTION_DAYS."""
    if not _UPLOAD_DIR.exists():
        return
    cutoff = time.time() - (_RECORDING_RETENTION_DAYS * 86400)
    removed = 0
    for f in _UPLOAD_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        logger.info("Cleaned up %d recording(s) older than %d days", removed, _RECORDING_RETENTION_DAYS)
