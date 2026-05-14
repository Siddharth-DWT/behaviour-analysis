# backend/agents/language_service.py
"""
LanguageAgentService — in-process wrapper for the Language Agent.

Wraps services/language_agent/ (DistilBERT sentiment + Claude intent + entity extraction).
startup() warms up DistilBERT so the first request is not slow.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

from shared.models.requests import (
    LanguageAnalysisRequest,
    LanguageAnalysisResponse,
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

logger = logging.getLogger("nexus.backend.language")


class LanguageAgentService(BaseAgentService):
    """
    In-process Language Agent.

    5 core rules: LANG-SENT-01, LANG-BUY-01, LANG-OBJ-01, LANG-PWR-01, LANG-INTENT-01.
    The three LLM tasks (sentiment, intent classification, entity extraction) run in
    parallel via asyncio.gather(), matching the microservice's internal parallelism.
    """

    name = "language"

    def __init__(self) -> None:
        self._extractor = None
        self._rule_engine = None
        self._entity_extractor = None
        self._redis_repo = RedisRepository()
        self._event_store = RedisEventStore()
        self._lock_manager = RedisLockManager()

    async def startup(self) -> None:
        from services.language_agent.feature_extractor import LanguageFeatureExtractor
        from services.language_agent.rules import LanguageRuleEngine
        from services.language_agent.entity_extractor import EntityExtractor

        self._extractor = LanguageFeatureExtractor()
        self._rule_engine = LanguageRuleEngine()
        self._entity_extractor = EntityExtractor()

        # Pre-load DistilBERT so the first analysis request is not penalised
        self._log("Warming up DistilBERT sentiment model...")
        self._extractor.warm_up()

        if _HAS_BUS:
            try:
                await message_bus.connect()
            except Exception as exc:
                self._warn(f"Redis message bus connection failed (non-fatal): {exc}")

        self._log("Language Agent ready.")

    async def shutdown(self) -> None:
        if _HAS_BUS:
            try:
                await message_bus.disconnect()
            except Exception:
                pass

    async def analyse(
        self,
        request: LanguageAnalysisRequest,
    ) -> LanguageAnalysisResponse:
        """
        Analyse transcript segments through the Language Agent pipeline.

        Pipeline (mirrors language_agent/main.py::analyse_transcript()):
          1. Extract non-LLM features (buying, objection, power, lexical) — fast
          2. Run LLM tasks in parallel: sentiment + intent + entities
          3. Merge sentiment into features
          4. Run synchronous rules (SENT, BUY, OBJ, PWR) + intent signals
          5. Publish to Redis Streams (observability)
        """
        if not request.segments:
            raise ValueError("No segments provided")

        session_id = request.session_id or str(uuid.uuid4())
        start_time = time.time()

        lock_token = await self._lock_manager.acquire(session_id, self.name)
        if not lock_token:
            raise RuntimeError(f"Language agent already processing session {session_id}")

        await self._redis_repo.set_session_state(
            session_id, SessionStateRecord(status="running", current_step=self.name)
        )
        await self._redis_repo.set_agent_status(
            session_id, self.name,
            AgentStatusRecord(status="running", summary_key="summary:language"),
        )
        await self._event_store.append(
            session_id,
            EventRecord(
                session_id=session_id, agent=self.name,
                event_type="agent_started",
                payload={"segment_count": len(request.segments)},
            ),
        )

        content_type = request.content_type or request.meeting_type or "sales_call"
        self._rule_engine.set_content_type(content_type)

        _profile = None
        try:
            from shared.config.content_type_profile import ContentTypeProfile
            _profile = ContentTypeProfile(content_type)
        except ImportError:
            pass

        logger.info(
            "[%s] Analysing %d segments (content_type=%s)",
            session_id, len(request.segments), content_type,
        )

        # Normalise to dicts
        segments = [
            seg.model_dump() if hasattr(seg, "model_dump") else dict(seg)
            for seg in request.segments
        ]

        # ── Step 1: Non-LLM features (fast) ────────────────────────────────
        features_list = self._extractor.extract_all_no_llm(segments)
        texts = [f["text"] for f in features_list]
        logger.info("[%s] Non-LLM features for %d segments", session_id, len(features_list))

        # ── Step 2: LLM tasks in parallel ───────────────────────────────────
        t_llm = time.time()

        async def _safe_sentiment():
            try:
                return await self._extractor.batch_sentiment_async(texts)
            except Exception as exc:
                self._warn(f"Async sentiment failed: {exc}")
                return [{"label": "NEUTRAL", "score": 0.0}] * len(texts)

        async def _safe_intent():
            if not request.run_intent_classification:
                return []
            try:
                return await self._rule_engine.evaluate_batch_intent(features_list, profile=_profile)
            except Exception as exc:
                self._warn(f"Intent classification failed: {exc}")
                return []

        async def _safe_entities():
            try:
                return await self._entity_extractor.extract(segments, content_type)
            except Exception as exc:
                self._warn(f"Entity extraction failed (non-fatal): {exc}")
                return {}

        sentiments, intent_signals, entities = await asyncio.gather(
            _safe_sentiment(), _safe_intent(), _safe_entities()
        )
        logger.info(
            "[%s] LLM tasks done: %d sentiments, %d intents, entities=%s in %.1fs",
            session_id, len(sentiments), len(intent_signals), bool(entities),
            time.time() - t_llm,
        )

        # ── Step 3: Merge sentiment into features ───────────────────────────
        for i, features in enumerate(features_list):
            sent = sentiments[i] if i < len(sentiments) else {"label": "NEUTRAL", "score": 0.0}
            features["sentiment_label"] = sent["label"]
            features["sentiment_score"] = sent["score"]
            features["sentiment_value"] = sent["score"]

        # ── Step 4: Synchronous rules (SENT, BUY, OBJ, PWR) ────────────────
        all_signals: list[dict] = []
        for i, features in enumerate(features_list):
            speaker_id = features.get("speaker_id", "unknown")
            signals = self._rule_engine.evaluate(
                features=features,
                speaker_id=speaker_id,
                content_type=content_type,
                all_features_list=features_list,
                current_index=i,
                profile=_profile,
            )
            all_signals.extend(signals)

        all_signals.extend(intent_signals)

        # ── Step 5: Publish to Redis Streams ────────────────────────────────
        if _HAS_BUS:
            published = 0
            for signal in all_signals:
                try:
                    await message_bus.publish_signal(
                        session_id=session_id,
                        agent=self.name,
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
            logger.info("[%s] Published %d signals to Redis", session_id, published)

        elapsed = time.time() - start_time
        speakers = list({f.get("speaker_id", "unknown") for f in features_list})
        logger.info("[%s] Language Agent complete: %d signals in %.1fs", session_id, len(all_signals), elapsed)

        from services.language_agent.main import _build_summary  # reuse existing helper
        summary = _build_summary(all_signals, features_list, speakers, profile=_profile)
        summary["entities"] = entities

        await self._redis_repo.write_artifact(
            session_id, "summary:language",
            {"summary": summary, "entities": entities},
        )
        await self._redis_repo.set_agent_status(
            session_id, self.name,
            AgentStatusRecord(status="completed", signal_count=len(all_signals), summary_key="summary:language"),
        )
        await self._event_store.append(
            session_id,
            EventRecord(
                session_id=session_id, agent=self.name,
                event_type="agent_completed",
                payload={"signal_count": len(all_signals)},
            ),
        )
        await self._lock_manager.release(session_id, self.name, lock_token)

        return LanguageAnalysisResponse(
            session_id=session_id,
            segment_count=len(features_list),
            speakers=speakers,
            signals=all_signals,
            summary=summary,
        )
