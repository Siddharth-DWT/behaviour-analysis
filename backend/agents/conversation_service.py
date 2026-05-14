# backend/agents/conversation_service.py
"""
ConversationAgentService — in-process wrapper for the Conversation Agent.

Wraps services/conversation_agent/ rule engine and feature extractor.
No heavy models — startup is fast (< 1s).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

from shared.models.requests import (
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
)
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

logger = logging.getLogger("nexus.backend.conversation")


class ConversationAgentService(BaseAgentService):
    """
    In-process Conversation Agent.

    Dialogue dynamics analysis: turn-taking, latency, dominance, rapport,
    engagement, balance (7 rules). No GPU models — runs synchronously inside
    the async event loop without a thread pool.
    """

    name = "conversation"

    def __init__(self) -> None:
        self._extractor = None
        self._rule_engine = None
        self._redis_repo = RedisRepository()
        self._event_store = RedisEventStore()
        self._lock_manager = RedisLockManager()

    async def startup(self) -> None:
        from services.conversation_agent.feature_extractor import ConversationFeatureExtractor
        from services.conversation_agent.rules import ConversationRuleEngine

        self._extractor = ConversationFeatureExtractor()
        self._rule_engine = ConversationRuleEngine()

        if _HAS_BUS:
            try:
                await message_bus.connect()
            except Exception as exc:
                self._warn(f"Redis message bus connection failed (non-fatal): {exc}")

        self._log("Conversation Agent ready.")

    async def shutdown(self) -> None:
        if _HAS_BUS:
            try:
                await message_bus.disconnect()
            except Exception:
                pass

    async def analyse(
        self,
        request: ConversationAnalysisRequest,
    ) -> ConversationAnalysisResponse:
        """
        Analyse transcript segments for conversation dynamics.

        Pipeline (mirrors conversation_agent/main.py::analyse()):
          1. Extract per-speaker / per-pair / session features
          2. Run rule engine (7 rules)
          3. Build summary
          4. Publish signals to Redis Streams (observability)
        """
        t0 = time.time()
        session_id = request.session_id or str(uuid.uuid4())
        segments = request.segments
        content_type = request.content_type or "sales_call"
        speakers = request.speakers or []

        if not segments:
            raise ValueError("No segments provided")

        lock_token = await self._lock_manager.acquire(session_id, self.name)
        if not lock_token:
            raise RuntimeError(f"Conversation agent already processing session {session_id}")

        await self._redis_repo.set_session_state(
            session_id, SessionStateRecord(status="running", current_step=self.name)
        )
        await self._redis_repo.set_agent_status(
            session_id, self.name,
            AgentStatusRecord(status="running", summary_key="summary:conversation"),
        )
        await self._event_store.append(
            session_id,
            EventRecord(
                session_id=session_id, agent=self.name,
                event_type="agent_started",
                payload={"segment_count": len(segments)},
            ),
        )

        _profile = None
        try:
            from shared.config.content_type_profile import ContentTypeProfile
            _profile = ContentTypeProfile(content_type)
        except ImportError:
            pass

        logger.info(
            "[%s] Analysing %d segments, %s speakers, content_type=%s",
            session_id, len(segments),
            len(speakers) if speakers else "auto-detect",
            content_type,
        )

        # ── Step 1: Feature Extraction ──────────────────────────────────────
        features = self._extractor.extract_all(segments, speakers or None)
        per_speaker = features.get("per_speaker", {})
        per_pair = features.get("per_pair", {})
        session_features = features.get("session", {})
        detected_speakers = list(per_speaker.keys())

        logger.info(
            "[%s] Features: %d speakers, %d turns, %.0fms",
            session_id, len(detected_speakers),
            session_features.get("total_turns", 0),
            session_features.get("total_duration_ms", 0),
        )

        # ── Step 2: Rule Engine ─────────────────────────────────────────────
        language_signals = getattr(request, "language_signals", None) or None
        signals = self._rule_engine.evaluate(
            features, content_type,
            language_signals=language_signals,
            profile=_profile,
        )
        logger.info("[%s] Rule engine: %d signals", session_id, len(signals))

        # ── Step 3: Build Summary ───────────────────────────────────────────
        summary = _build_summary(per_speaker, per_pair, session_features, signals, detected_speakers)

        # ── Step 4: Publish to Redis Streams ────────────────────────────────
        if _HAS_BUS and signals:
            try:
                for sig in signals:
                    await message_bus.publish_signal(
                        session_id=session_id,
                        agent=self.name,
                        speaker_id=sig.get("speaker_id", "unknown"),
                        signal_type=sig.get("signal_type", ""),
                        value=sig.get("value"),
                        value_text=sig.get("value_text", ""),
                        confidence=sig.get("confidence", 0.5),
                        window_start_ms=sig.get("window_start_ms", 0),
                        window_end_ms=sig.get("window_end_ms", 0),
                        metadata=sig.get("metadata"),
                    )
            except Exception as exc:
                self._warn(f"Redis publish failed (non-fatal): {exc}")

        elapsed = time.time() - t0
        logger.info("[%s] Conversation analysis complete in %.2fs", session_id, elapsed)

        await self._redis_repo.write_artifact(
            session_id, "summary:conversation", {"summary": summary}
        )
        await self._redis_repo.set_agent_status(
            session_id, self.name,
            AgentStatusRecord(status="completed", signal_count=len(signals), summary_key="summary:conversation"),
        )
        await self._event_store.append(
            session_id,
            EventRecord(
                session_id=session_id, agent=self.name,
                event_type="agent_completed",
                payload={"signal_count": len(signals)},
            ),
        )
        await self._lock_manager.release(session_id, self.name, lock_token)

        return ConversationAnalysisResponse(
            session_id=session_id,
            speaker_count=len(detected_speakers),
            signals=list(signals),
            summary=summary,
        )


# ── Module-level helpers (pure functions — identical to conversation_agent/main.py) ──

def _build_summary(
    per_speaker: dict,
    per_pair: dict,
    session_features: dict,
    signals: list[dict],
    speakers: list[str],
) -> dict:
    signal_index: dict[tuple[str, str], dict] = {}
    for sig in signals:
        key = (sig.get("signal_type", ""), sig.get("speaker_id", ""))
        signal_index[key] = sig

    speaker_summaries: dict[str, dict] = {}
    for spk in speakers:
        spk_data = per_speaker.get(spk, {})

        dom_sig = signal_index.get(("dominance_score", spk), {})
        eng_sig = signal_index.get(("conversation_engagement", spk), {})

        latencies: list[float] = []
        for pair_key, pair_data in per_pair.items():
            if spk in pair_key.split("__"):
                lat = pair_data.get("response_latency_ms_avg", 0)
                if lat > 0:
                    latencies.append(lat)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        speaker_summaries[spk] = {
            "talk_time_pct": spk_data.get("talk_time_pct", 0),
            "dominance_score": round(dom_sig.get("value", 0) or 0, 3),
            "engagement_score": round(eng_sig.get("value", 0) or 0, 3),
            "interruptions_made": spk_data.get("interruption_count", 0),
            "interruptions_received": spk_data.get("was_interrupted_count", 0),
            "avg_response_latency_ms": round(avg_latency, 1),
            "questions_asked": spk_data.get("questions_asked", 0),
            "back_channels": spk_data.get("back_channel_count", 0),
        }

    balance_sig = signal_index.get(("conversation_balance", "session"), {})

    rapport_values = [
        sig.get("value", 0)
        for sig in signals
        if sig.get("signal_type") == "rapport_indicator"
    ]
    avg_rapport = sum(rapport_values) / len(rapport_values) if rapport_values else 0

    total_interruptions = sum(
        per_speaker.get(s, {}).get("interruption_count", 0) for s in speakers
    )

    session_summary = {
        "turn_rate_per_minute": session_features.get("turn_rate_per_minute", 0),
        "dominance_index": session_features.get("dominance_index", 0),
        "conversation_balance": balance_sig.get("value_text", "unknown"),
        "rapport_score": round(avg_rapport, 3),
        "total_interruptions": total_interruptions,
        "longest_monologue_speaker": session_features.get("longest_monologue_speaker", ""),
        "longest_monologue_duration_ms": session_features.get("longest_monologue_ms", 0),
    }

    return {"per_speaker": speaker_summaries, "session": session_summary}
