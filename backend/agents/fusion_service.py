# backend/agents/fusion_service.py
"""
FusionAgentService — in-process wrapper for the Fusion Agent.

Wraps services/fusion_agent/ (pairwise rules, compound patterns, temporal patterns,
signal graph, analytics, and narrative report generation via Claude/OpenAI API).
No GPU models — purely algorithmic after startup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from typing import Optional

from shared.models.requests import FusionAnalyseRequest, FusionAnalyseResponse
from shared.models.signals import FusionSignalInput
from shared.redis_layer import (
    AgentStatusRecord,
    EventRecord,
    RedisEventStore,
    RedisLockManager,
    RedisRepository,
    SessionStateRecord,
)

from .base import BaseAgentService

try:
    from shared.utils.message_bus import message_bus
    _HAS_BUS = True
except ImportError:
    _HAS_BUS = False

logger = logging.getLogger("nexus.backend.fusion")


class FusionAgentService(BaseAgentService):
    """
    In-process Fusion Agent.

    Orchestrates cross-modal fusion:
      - 3 pairwise rules (FUSION-02, FUSION-07, FUSION-13)
      - 12 compound pattern detectors (Phase 2F)
      - 8 temporal cascade sequences (Phase 2G)
      - Signal graph analytics
      - Narrative report generation via LLM

    Speakers processed in parallel via asyncio.gather() — O(max(speakers)) vs O(sum).
    """

    name = "fusion"

    def __init__(self) -> None:
        self._rule_engine = None
        self._compound_engine = None
        self._temporal_engine = None
        self._redis_repo = RedisRepository()
        self._event_store = RedisEventStore()
        self._lock_manager = RedisLockManager()

    async def startup(self) -> None:
        from services.fusion_agent.rules import FusionRuleEngine
        from services.fusion_agent.compound_patterns import CompoundPatternEngine
        from services.fusion_agent.temporal_patterns import TemporalPatternEngine

        self._rule_engine = FusionRuleEngine()
        self._compound_engine = CompoundPatternEngine()
        self._temporal_engine = TemporalPatternEngine()

        if _HAS_BUS:
            try:
                await message_bus.connect()
            except Exception as exc:
                self._warn(f"Redis message bus connection failed (non-fatal): {exc}")

        self._log("Fusion Agent ready.")

    async def shutdown(self) -> None:
        if _HAS_BUS:
            try:
                await message_bus.disconnect()
            except Exception:
                pass

    async def analyse(
        self,
        request: FusionAnalyseRequest,
    ) -> FusionAnalyseResponse:
        """
        Run fusion analysis on pre-collected voice + language signals.

        Pipeline (mirrors fusion_agent/main.py::analyse_signals()):
          1. Buffer signals; separate video vs. voice
          2. Per-speaker pairwise + compound rules (asyncio.gather)
          3. Session-level temporal patterns
          4. Signal graph + analytics
          5. Narrative report
          6. Publish to Redis Streams
        """
        from services.fusion_agent.fusion_engine import SignalBuffer, compute_unified_state, WINDOW_SHORT_MS
        from services.fusion_agent.narrative import generate_session_narrative
        from services.fusion_agent.signal_graph import SignalGraph
        from services.fusion_agent.graph_analytics import GraphAnalytics

        session_id = request.session_id or str(uuid.uuid4())
        content_type = request.content_type or request.meeting_type or "sales_call"
        start_time = time.time()

        lock_token = await self._lock_manager.acquire(session_id, self.name)
        if not lock_token:
            raise RuntimeError(f"Fusion agent already processing session {session_id}")

        await self._redis_repo.set_session_state(
            session_id, SessionStateRecord(status="running", current_step=self.name)
        )
        await self._redis_repo.set_agent_status(
            session_id, self.name,
            AgentStatusRecord(status="running", summary_key="summary:fusion"),
        )
        await self._event_store.append(
            session_id,
            EventRecord(session_id=session_id, agent=self.name, event_type="agent_started", payload={}),
        )

        _profile = None
        try:
            from shared.config.content_type_profile import ContentTypeProfile
            _profile = ContentTypeProfile(content_type)
        except ImportError:
            pass

        voice_dicts = [s.model_dump() for s in request.voice_signals]
        language_dicts = [s.model_dump() for s in request.language_signals]

        # Gateway bundles video signals in voice_dicts with agent="video"
        video_dicts = [s for s in voice_dicts if s.get("agent") == "video"]
        pure_voice_dicts = [s for s in voice_dicts if s.get("agent") != "video"]

        logger.info(
            "[%s] Fusion: %d voice + %d language + %d video signals",
            session_id, len(pure_voice_dicts), len(language_dicts), len(video_dicts),
        )

        # ── Step 1: Buffer signals ──────────────────────────────────────────
        buffer = SignalBuffer()
        buffer.add_many(pure_voice_dicts)
        buffer.add_many(language_dicts)
        buffer.add_many(video_dicts)

        speakers = buffer.speakers
        ref_time = _max_time(pure_voice_dicts + language_dicts + video_dicts)

        all_fusion_signals: list[dict] = []
        all_unified_states: list[dict] = []
        all_alerts: list[dict] = []

        # ── Step 2: Per-speaker pairwise + compound (parallel) ──────────────
        rule_engine = self._rule_engine
        compound_engine = self._compound_engine

        async def _analyse_speaker(speaker_id: str) -> dict:
            speaker_voice = buffer.get_signals(
                speaker_id, "voice",
                window_ms=WINDOW_SHORT_MS, reference_time_ms=ref_time,
            ) or [s for s in pure_voice_dicts if s.get("speaker_id") == speaker_id]

            speaker_language = buffer.get_signals(
                speaker_id, "language",
                window_ms=WINDOW_SHORT_MS, reference_time_ms=ref_time,
            ) or [s for s in language_dicts if s.get("speaker_id") == speaker_id]

            speaker_video = buffer.get_signals(
                speaker_id, "video",
                window_ms=WINDOW_SHORT_MS, reference_time_ms=ref_time,
            ) or [s for s in video_dicts if s.get("speaker_id") == speaker_id]

            if not speaker_voice and not speaker_language:
                return {"signals": [], "state": None, "alerts": []}

            all_starts = [_to_int(s.get("window_start_ms", 0)) for s in speaker_voice + speaker_language + speaker_video]
            all_ends   = [_to_int(s.get("window_end_ms",   0)) for s in speaker_voice + speaker_language + speaker_video]
            window_start = min(all_starts) if all_starts else 0
            window_end   = max(all_ends)   if all_ends   else 0

            fusion_signals = rule_engine.evaluate(
                speaker_id=speaker_id,
                voice_signals=speaker_voice,
                language_signals=speaker_language,
                video_signals=speaker_video,
                window_start_ms=window_start,
                window_end_ms=window_end,
                content_type=content_type,
                profile=_profile,
            )

            # Compound patterns — per-segment evaluation with ±5s signal window
            compound_signals: list[dict] = []
            if compound_engine is not None:
                all_speaker_sigs = speaker_voice + speaker_language + speaker_video + fusion_signals
                COMPOUND_ALIGN_MS = 5_000
                segment_windows: set[tuple[int, int]] = {
                    (s.get("window_start_ms", 0), s.get("window_end_ms", 0))
                    for s in speaker_voice + speaker_language
                    if s.get("window_start_ms", 0) >= 0
                    and s.get("window_end_ms", 0) > s.get("window_start_ms", 0)
                }
                for seg_start, seg_end in sorted(segment_windows):
                    zone_start = seg_start - COMPOUND_ALIGN_MS
                    zone_end   = seg_end   + COMPOUND_ALIGN_MS
                    seg_sigs = [
                        s for s in all_speaker_sigs
                        if s.get("window_start_ms", 0) < zone_end
                        and s.get("window_end_ms",   0) > zone_start
                    ]
                    if len(seg_sigs) < 3:
                        continue
                    # Timestamps use the earliest/latest contributing signal so compound
                    # badges align with the visual event, not the voice segment boundary.
                    _starts = [s.get("window_start_ms", seg_start) for s in seg_sigs]
                    _ends   = [s.get("window_end_ms",   seg_end)   for s in seg_sigs]
                    compound_signals.extend(compound_engine.evaluate(
                        speaker_id=speaker_id,
                        voice_signals=[s for s in seg_sigs if s.get("agent") == "voice"],
                        language_signals=[s for s in seg_sigs if s.get("agent") == "language"],
                        video_signals=[s for s in seg_sigs if s.get("agent") in ("facial", "body", "gaze", "video")],
                        fusion_signals=[s for s in seg_sigs if s.get("agent") == "fusion"],
                        window_start_ms=min(_starts),
                        window_end_ms=max(_ends),
                    ))

                # Session-wide frequency cap — highest-confidence per signal_type
                _MAX_FIRES = {
                    "peak_performance": 2, "genuine_engagement": 2,
                    "active_disengagement": 2, "conflict_escalation": 2,
                    "deception_cluster": 1, "emotional_suppression": 2,
                }
                if compound_signals:
                    fire_counts: dict[str, int] = defaultdict(int)
                    deduped: list[dict] = []
                    for s in sorted(compound_signals, key=lambda x: x.get("confidence", 0), reverse=True):
                        st = s.get("signal_type", "")
                        cap = _MAX_FIRES.get(st, 3)
                        if fire_counts[st] < cap:
                            deduped.append(s)
                            fire_counts[st] += 1
                    compound_signals = deduped

            fusion_signals = fusion_signals + compound_signals

            state = compute_unified_state(
                speaker_id=speaker_id,
                voice_signals=speaker_voice,
                language_signals=speaker_language,
                fusion_signals=fusion_signals,
            )

            speaker_alerts = [
                alert
                for fs in fusion_signals
                if fs.get("confidence", 0) >= 0.50
                for alert in [_create_alert(session_id, speaker_id, fs, content_type, _profile)]
                if alert
            ]

            return {"signals": fusion_signals, "state": state, "alerts": speaker_alerts}

        speaker_results = await asyncio.gather(*[_analyse_speaker(sid) for sid in speakers])
        for result in speaker_results:
            all_fusion_signals.extend(result["signals"])
            if result["state"] is not None:
                all_unified_states.append(asdict(result["state"]))
            all_alerts.extend(result["alerts"])

        logger.info(
            "[%s] Pairwise+compound: %d fusion signals, %d alerts",
            session_id, len(all_fusion_signals), len(all_alerts),
        )

        # ── Step 2.5: Temporal patterns (session-level) ─────────────────────
        if self._temporal_engine is not None:
            all_source = pure_voice_dicts + language_dicts + video_dicts
            session_start_ms = min(
                (_to_int(s.get("window_start_ms", 0)) for s in all_source), default=0
            )
            for sid in speakers:
                temporal_sigs = self._temporal_engine.evaluate(
                    speaker_id=sid,
                    all_signals=[s for s in all_source if s.get("speaker_id") == sid],
                    session_start_ms=session_start_ms,
                    session_end_ms=ref_time,
                )
                all_fusion_signals.extend(temporal_sigs)

        # ── Step 2.6: Interrogation compound patterns (session-level) ─────────
        if content_type == "interrogation_video":
            try:
                from services.fusion_agent.interrogation_patterns import InterrogationCompoundPatterns
                all_source = pure_voice_dicts + language_dicts + video_dicts
                interrog_compound = InterrogationCompoundPatterns().evaluate(
                    all_signals=all_source + all_fusion_signals,
                    speakers=speakers,
                    session_id=session_id,
                )
                all_fusion_signals.extend(interrog_compound)
                if interrog_compound:
                    logger.info(
                        "[%s] Interrogation compound patterns: %d signal(s)",
                        session_id, len(interrog_compound),
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] Interrogation compound patterns failed (non-fatal): %s",
                    session_id, exc,
                )

        # ── Step 2.7: False confession risk assessment (session-level) ────────
        if content_type == "interrogation_video":
            try:
                from services.fusion_agent.interrogation_patterns import FalseConfessionRiskAssessor
                all_source = pure_voice_dicts + language_dicts + video_dicts
                risk_signals = FalseConfessionRiskAssessor().evaluate(
                    all_signals=all_source + all_fusion_signals,
                    speakers=speakers,
                    session_id=session_id,
                )
                all_fusion_signals.extend(risk_signals)
                if risk_signals:
                    logger.info(
                        "[%s] FalseConfessionRisk: %d assessment(s)",
                        session_id, len(risk_signals),
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] False confession risk assessment failed (non-fatal): %s",
                    session_id, exc,
                )

        # ── Step 3: Publish to Redis Streams ────────────────────────────────
        if _HAS_BUS:
            published = 0
            for signal in all_fusion_signals:
                try:
                    await message_bus.publish_signal(
                        session_id=session_id, agent=self.name,
                        speaker_id=signal.get("speaker_id", "unknown"),
                        signal_type=signal.get("signal_type", ""),
                        value=signal.get("value"),
                        value_text=signal.get("value_text", ""),
                        confidence=signal.get("confidence", 0.5),
                        window_start_ms=signal.get("window_start_ms", 0),
                        window_end_ms=signal.get("window_end_ms", 0),
                        metadata=signal.get("metadata"),
                    )
                    published += 1
                except Exception as exc:
                    self._warn(f"Signal publish failed: {exc}")
            for alert in all_alerts:
                try:
                    await message_bus.publish_alert(
                        session_id=session_id,
                        speaker_id=alert.get("speaker_id", ""),
                        alert_type=alert.get("alert_type", ""),
                        severity=alert.get("severity", "yellow"),
                        title=alert.get("title", ""),
                        description=alert.get("description", ""),
                        evidence=alert.get("evidence"),
                    )
                except Exception:
                    pass
            logger.info("[%s] Published %d fusion signals + %d alerts", session_id, published, len(all_alerts))

        # Extract entities + conversation from upstream summaries
        entities = (request.language_summary or {}).get("entities", {})
        conversation_summary = (request.voice_summary or {}).get("conversation", {})
        video_summary = request.video_summary or {}

        # ── Step 4: Signal graph + analytics ───────────────────────────────
        graph_json: dict = {}
        key_paths: list = []
        graph_insights: dict = {}
        try:
            graph = SignalGraph()
            graph.build_from_session(
                voice_signals=pure_voice_dicts,
                language_signals=language_dicts + conversation_summary.get("signals", []),
                fusion_signals=all_fusion_signals,
                transcript_segments=[],
                entities=entities,
                video_signals=video_dicts,
            )
            graph_json = graph.to_json()
            key_paths = graph.get_key_paths(max_paths=5)
            analytics = GraphAnalytics(graph)
            graph_insights = analytics.compute_all(content_type=content_type)
            graph_signals = self._rule_engine.evaluate_graph_insights(
                graph_insights, speakers, all_fusion_signals,
                content_type=content_type, profile=_profile,
            )
            all_fusion_signals.extend(graph_signals)
            logger.info(
                "[%s] Signal graph: %d nodes, %d edges, %d key paths, %d graph signals",
                session_id,
                graph_json["stats"]["node_count"],
                graph_json["stats"]["edge_count"],
                len(key_paths), len(graph_signals),
            )
        except Exception as exc:
            self._warn(f"Signal graph/analytics failed (non-fatal): {exc}")

        # ── Step 5: Narrative report ────────────────────────────────────────
        report = None
        if request.generate_report:
            t_rep = time.time()
            all_ts = [_to_int(s.get("window_end_ms", 0)) for s in voice_dicts + language_dicts]
            duration_seconds = (max(all_ts) - min(all_ts)) / 1000.0 if all_ts else 0

            report = await generate_session_narrative(
                session_id=session_id,
                duration_seconds=duration_seconds,
                speakers=speakers,
                voice_summary=request.voice_summary or {},
                language_summary=request.language_summary or {},
                fusion_signals=all_fusion_signals,
                unified_states=all_unified_states,
                meeting_type=content_type,
                entities=entities,
                graph_analytics=graph_insights,
                conversation_summary=conversation_summary,
                video_summary=video_summary,
            )
            logger.info("[%s] Narrative report in %.1fs", session_id, time.time() - t_rep)

        elapsed = time.time() - start_time
        logger.info(
            "[%s] Fusion Agent complete: %d signals, %d alerts, %d states in %.1fs",
            session_id, len(all_fusion_signals), len(all_alerts), len(all_unified_states), elapsed,
        )

        summary = _build_fusion_summary(all_fusion_signals, all_unified_states, all_alerts)
        summary["signal_graph"] = graph_json
        summary["key_paths"] = key_paths
        summary["graph_analytics"] = graph_insights

        await self._redis_repo.write_artifact(
            session_id, "summary:fusion",
            {"summary": summary, "alerts": all_alerts},
        )
        await self._redis_repo.set_agent_status(
            session_id, self.name,
            AgentStatusRecord(status="completed", signal_count=len(all_fusion_signals), summary_key="summary:fusion"),
        )
        await self._event_store.append(
            session_id,
            EventRecord(
                session_id=session_id, agent=self.name,
                event_type="agent_completed",
                payload={"signal_count": len(all_fusion_signals)},
            ),
        )
        await self._lock_manager.release(session_id, self.name, lock_token)

        return FusionAnalyseResponse(
            session_id=session_id,
            speakers=speakers,
            fusion_signals=all_fusion_signals,
            unified_states=all_unified_states,
            alerts=all_alerts,
            report=report,
            summary=summary,
        )


# ── Module-level helpers (identical to fusion_agent/main.py) ─────────────────

def _to_int(val) -> int:
    try:
        return int(float(val)) if val is not None else 0
    except (ValueError, TypeError):
        return 0


def _max_time(signals: list[dict]) -> int:
    if not signals:
        return 0
    times = [_to_int(s.get("window_end_ms", 0)) for s in signals]
    return max(times) if times else 0


def _create_alert(
    session_id: str,
    speaker_id: str,
    fusion_signal: dict,
    content_type: str = "sales_call",
    profile=None,
) -> Optional[dict]:
    sig_type   = fusion_signal.get("signal_type", "")
    value_text = fusion_signal.get("value_text", "")
    confidence = fusion_signal.get("confidence", 0)

    SALES_ONLY = {
        ("urgency_authenticity", "manufactured_urgency"),
        ("urgency_authenticity", "ambiguous_urgency"),
        ("credibility_assessment", "credibility_concern"),
    }
    if (sig_type, value_text) in SALES_ONLY and content_type not in ("sales_call", "pitch", "presentation"):
        return None

    if content_type in ("internal", "meeting", "interview", "podcast"):
        min_conf = profile.get_threshold("ALERT", "min_confidence", 0.60) if profile else 0.60
        if confidence < min_conf:
            return None

    alert_map = {
        "credibility_assessment": {
            "credibility_concern": {"severity": "orange", "title": "Content-Voice Incongruence",
                "description": "Positive language detected alongside elevated vocal stress. This may indicate discomfort with the stated position."},
            "mild_incongruence": {"severity": "yellow", "title": "Mild Content-Voice Mismatch",
                "description": "Slight mismatch between verbal content and vocal indicators."},
        },
        "verbal_incongruence": {
            "strong_verbal_incongruence": {"severity": "orange", "title": "Strong Verbal Hedging",
                "description": "Positive sentiment expressed with heavily hedged language. Speaker may be agreeing without genuine conviction."},
            "moderate_verbal_incongruence": {"severity": "yellow", "title": "Hedged Agreement",
                "description": "Agreement expressed with notable hedging language."},
            "incongruence_with_objection": {"severity": "orange", "title": "Hidden Objection Detected",
                "description": "Positive sentiment combined with objection markers and weak power language. Speaker may have unstated concerns."},
        },
        "urgency_authenticity": {
            "manufactured_urgency": {"severity": "yellow", "title": "Potentially Manufactured Urgency",
                "description": "Fast-paced persuasive language with concurrent stress indicators."},
            "authentic_urgency": {"severity": "green", "title": "Authentic Enthusiasm Detected",
                "description": "Fast-paced persuasive language supported by confident vocal patterns."},
        },
    }

    alert_info = alert_map.get(sig_type, {}).get(value_text)
    if not alert_info:
        return None
    if alert_info["severity"] == "green" and confidence < 0.60:
        return None

    return {
        "session_id": session_id,
        "speaker_id": speaker_id,
        "alert_type": sig_type,
        "severity": alert_info["severity"],
        "title": alert_info["title"],
        "description": alert_info["description"],
        "evidence": fusion_signal.get("metadata", {}),
    }


def _build_fusion_summary(
    fusion_signals: list[dict],
    unified_states: list[dict],
    alerts: list[dict],
) -> dict:
    signal_types: dict[str, int] = {}
    for s in fusion_signals:
        st = s.get("signal_type", "unknown")
        signal_types[st] = signal_types.get(st, 0) + 1

    alert_severities: dict[str, int] = {}
    for a in alerts:
        sev = a.get("severity", "unknown")
        alert_severities[sev] = alert_severities.get(sev, 0) + 1

    return {
        "total_fusion_signals": len(fusion_signals),
        "signal_type_counts": signal_types,
        "total_alerts": len(alerts),
        "alert_severity_counts": alert_severities,
        "speakers_analysed": len(unified_states),
    }
