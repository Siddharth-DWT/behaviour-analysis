# services/language_agent/main.py
"""
NEXUS Language Agent (Agent 2)
FastAPI service for linguistic analysis of transcript segments.

Implements 5 core rules from the Rule Engine:
  - LANG-SENT-01: Per-sentence sentiment (DistilBERT)
  - LANG-BUY-01:  Buying signal detection (SPIN keyword patterns)
  - LANG-OBJ-01:  Objection signal detection (hedges + resistance)
  - LANG-PWR-01:  Power language score (Lakoff/O'Barr)
  - LANG-INTENT-01: Intent classification (Claude API batch)

Endpoints:
  POST /analyse          → Analyse transcript segments
  GET  /health           → Health check
"""
import os
import sys
import uuid
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# isort: split
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# isort: split
from shared.models.transcript import TranscriptSegment
from shared.models.requests import LanguageAnalysisRequest as AnalysisRequest, LanguageAnalysisResponse as AnalysisResponse

# Import from same directory (works in Docker /app context)
try:
    from feature_extractor import LanguageFeatureExtractor
    from rules import LanguageRuleEngine
    from entity_extractor import EntityExtractor
except ImportError:
    from services.language_agent.feature_extractor import LanguageFeatureExtractor
    from services.language_agent.rules import LanguageRuleEngine
    from services.language_agent.entity_extractor import EntityExtractor

# Shared utilities
try:
    from shared.utils.message_bus import message_bus
    from shared.config.settings import config
    HAS_MESSAGE_BUS = True
except ImportError:
    HAS_MESSAGE_BUS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.language")

app = FastAPI(
    title="NEXUS Language Agent",
    description="Agent 2: Linguistic analysis of transcript segments",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (initialised on startup) ──
feature_extractor: Optional[LanguageFeatureExtractor] = None
rule_engine: Optional[LanguageRuleEngine] = None
entity_extractor: Optional[EntityExtractor] = None


@app.on_event("startup")
async def startup():
    global feature_extractor, rule_engine, entity_extractor
    logger.info("Starting NEXUS Language Agent...")

    feature_extractor = LanguageFeatureExtractor()
    rule_engine = LanguageRuleEngine()
    entity_extractor = EntityExtractor()

    # Pre-load DistilBERT model so first request isn't slow
    logger.info("Warming up sentiment model...")
    feature_extractor.warm_up()

    # Connect message bus if available
    if HAS_MESSAGE_BUS:
        try:
            await message_bus.connect()
            logger.info("Connected to Redis message bus.")
        except Exception as e:
            logger.warning(f"Redis connection failed (non-fatal): {e}")

    logger.info("Language Agent ready.")


@app.on_event("shutdown")
async def shutdown():
    if HAS_MESSAGE_BUS:
        await message_bus.disconnect()


@app.get("/health")
async def health():
    # Get LLM provider info
    llm_info = {"provider": "unknown", "api_key_configured": False}
    try:
        from shared.utils.llm_client import get_provider_info
        llm_info = get_provider_info()
    except ImportError:
        pass

    return {
        "status": "ok",
        "agent": "language",
        "version": "0.2.0",
        "models_loaded": {
            "feature_extractor": feature_extractor is not None,
            "rule_engine": rule_engine is not None,
            "sentiment_model": feature_extractor._sentiment_ready if feature_extractor else False,
        },
        "llm": llm_info,
        "redis_connected": HAS_MESSAGE_BUS,
    }



@app.post("/analyse", response_model=AnalysisResponse)
async def analyse_transcript(request: AnalysisRequest):
    """
    Process transcript segments through the Language Agent pipeline.

    Pipeline:
    1. Extract linguistic features per segment (sentiment, keywords, power)
    2. Run 4 synchronous rules (SENT, BUY, OBJ, PWR) per segment
    3. Run LANG-INTENT-01 via Claude API (batched, optional)
    4. Publish signals to Redis Streams
    5. Return all signals + summary
    """
    if not request.segments:
        raise HTTPException(400, "No segments provided")

    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()

    # Determine content type (affects which rules are active)
    content_type = request.content_type or request.meeting_type or "sales_call"
    rule_engine.set_content_type(content_type)

    # Create content-type profile for gating/renaming/confidence
    _profile = None
    try:
        from shared.config.content_type_profile import ContentTypeProfile
        _profile = ContentTypeProfile(content_type)
    except ImportError:
        pass

    logger.info(
        f"[{session_id}] Analysing {len(request.segments)} segments "
        f"(content_type={content_type})"
    )

    # Convert to dicts (segments may already be dicts or Pydantic models)
    segments = [seg.model_dump() if hasattr(seg, 'model_dump') else dict(seg) for seg in request.segments]

    # ── Step 1: Extract non-LLM features (fast — buying, objection, power, lexical) ──
    logger.info(f"[{session_id}] Step 1: Extracting linguistic features (non-LLM)...")
    features_list = feature_extractor.extract_all_no_llm(segments)
    # Build texts list matching features_list (non-empty segments only)
    texts = [f["text"] for f in features_list]
    logger.info(f"[{session_id}] Extracted features for {len(features_list)} segments")

    # ── Step 2: Run LLM tasks in parallel (sentiment + intent + entities) ──
    logger.info(f"[{session_id}] Step 2: Running LLM tasks in parallel...")

    async def _safe_sentiment():
        try:
            return await feature_extractor.batch_sentiment_async(texts)
        except Exception as e:
            logger.warning(f"[{session_id}] Async sentiment failed: {e}")
            return [{"label": "NEUTRAL", "score": 0.0}] * len(texts)

    async def _safe_intent():
        if not request.run_intent_classification:
            return []
        try:
            return await rule_engine.evaluate_batch_intent(features_list)
        except Exception as e:
            logger.warning(f"[{session_id}] Intent classification failed: {e}")
            return []

    async def _safe_entities():
        try:
            return await entity_extractor.extract(segments, content_type)
        except Exception as e:
            logger.warning(f"[{session_id}] Entity extraction failed (non-fatal): {e}")
            return {}

    sentiments, intent_signals, entities = await asyncio.gather(
        _safe_sentiment(), _safe_intent(), _safe_entities()
    )

    # ── Step 3: Merge sentiment into features ──
    for i, features in enumerate(features_list):
        sent = sentiments[i] if i < len(sentiments) else {"label": "NEUTRAL", "score": 0.0}
        features["sentiment_label"] = sent["label"]
        features["sentiment_score"] = sent["score"]
        features["sentiment_value"] = sent["score"]

    # ── Step 4: Run synchronous rules (SENT, BUY, OBJ, PWR) — now with sentiment populated ──
    logger.info(f"[{session_id}] Step 4: Running rule engine (content_type={content_type})...")
    all_signals = []

    for i, features in enumerate(features_list):
        speaker_id = features.get("speaker_id", "unknown")
        signals = rule_engine.evaluate(
            features=features, speaker_id=speaker_id, content_type=content_type,
            all_features_list=features_list, current_index=i,
            profile=_profile,
        )
        all_signals.extend(signals)

    # Add intent signals from parallel LLM task
    all_signals.extend(intent_signals)
    logger.info(
        f"[{session_id}] Sentiment: {len(sentiments)} scores, "
        f"Intent: {len(intent_signals)} intents, "
        f"Entities: {len(entities.get('people', []))} people, "
        f"{len(entities.get('topics', []))} topics, "
        f"{len(entities.get('commitments', []))} commitments"
    )

    # ── Step 4: Publish to Redis Streams ──
    if HAS_MESSAGE_BUS:
        published = 0
        for signal in all_signals:
            try:
                await message_bus.publish_signal(
                    session_id=session_id,
                    agent="language",
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
            except Exception as e:
                logger.warning(f"Failed to publish signal to Redis: {e}")
        logger.info(f"[{session_id}] Published {published} signals to Redis")

    # ── Step 5: Build summary ──
    elapsed = time.time() - start_time
    speakers = list(set(f.get("speaker_id", "unknown") for f in features_list))
    logger.info(f"[{session_id}] Complete: {len(all_signals)} signals in {elapsed:.1f}s")

    summary = _build_summary(all_signals, features_list, speakers)
    summary["entities"] = entities

    return AnalysisResponse(
        session_id=session_id,
        segment_count=len(features_list),
        speakers=speakers,
        signals=all_signals,
        summary=summary,
    )


def _build_summary(signals: list[dict], features_list: list[dict], speakers: list[str]) -> dict:
    """Build a human-readable summary from all signals."""
    summary = {
        "total_signals": len(signals),
        "per_speaker": {},
        "buying_signal_moments": [],
        "objection_moments": [],
        "objection_resolution": [],
        "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
    }

    for speaker_id in speakers:
        speaker_signals = [s for s in signals if s.get("speaker_id") == speaker_id]
        speaker_features = [f for f in features_list if f.get("speaker_id") == speaker_id]

        # Sentiment stats
        sent_signals = [s for s in speaker_signals if s.get("signal_type") == "sentiment_score"]
        sent_values = [s["value"] for s in sent_signals if s.get("value") is not None]

        # Buying signals
        buy_signals = [s for s in speaker_signals if s.get("signal_type") == "buying_signal"]

        # Objection signals
        obj_signals = [s for s in speaker_signals if s.get("signal_type") == "objection_signal"]

        # Power stats
        pwr_signals = [s for s in speaker_signals if s.get("signal_type") == "power_language_score"]
        pwr_values = [s["value"] for s in pwr_signals if s.get("value") is not None]

        # Intent stats
        intent_signals = [s for s in speaker_signals if s.get("signal_type") == "intent_classification"]
        intent_dist = {}
        for s in intent_signals:
            intent = s.get("value_text", "UNKNOWN")
            intent_dist[intent] = intent_dist.get(intent, 0) + 1

        summary["per_speaker"][speaker_id] = {
            "total_segments": len(speaker_features),
            "avg_sentiment": round(sum(sent_values) / len(sent_values), 3) if sent_values else 0,
            "min_sentiment": round(min(sent_values), 3) if sent_values else 0,
            "max_sentiment": round(max(sent_values), 3) if sent_values else 0,
            "buying_signal_count": len(buy_signals),
            "objection_count": len(obj_signals),
            "avg_power_score": round(sum(pwr_values) / len(pwr_values), 3) if pwr_values else 0.5,
            "intent_distribution": intent_dist,
        }

        # Track notable buying signal moments
        for s in buy_signals:
            if s.get("value", 0) >= 0.50:
                summary["buying_signal_moments"].append({
                    "speaker": speaker_id,
                    "time_ms": s.get("window_start_ms"),
                    "strength": round(s["value"], 3),
                    "categories": s.get("metadata", {}).get("categories", []),
                })

        # Track objection moments
        for s in obj_signals:
            if s.get("value", 0) >= 0.40:
                summary["objection_moments"].append({
                    "speaker": speaker_id,
                    "time_ms": s.get("window_start_ms"),
                    "strength": round(s["value"], 3),
                    "categories": s.get("metadata", {}).get("categories", []),
                })

    # Overall sentiment distribution
    for s in signals:
        if s.get("signal_type") == "sentiment_score":
            val = s.get("value", 0)
            if val > 0.2:
                summary["sentiment_distribution"]["positive"] += 1
            elif val < -0.2:
                summary["sentiment_distribution"]["negative"] += 1
            else:
                summary["sentiment_distribution"]["neutral"] += 1

    # Sort moments by strength
    summary["buying_signal_moments"].sort(key=lambda x: x["strength"], reverse=True)
    summary["buying_signal_moments"] = summary["buying_signal_moments"][:10]
    summary["objection_moments"].sort(key=lambda x: x["strength"], reverse=True)
    summary["objection_moments"] = summary["objection_moments"][:10]

    # ── Objection Resolution: did buying signals follow objections for same speaker? ──
    for speaker_id in speakers:
        speaker_obj = [m for m in summary["objection_moments"] if m["speaker"] == speaker_id]
        speaker_buy = [m for m in summary["buying_signal_moments"] if m["speaker"] == speaker_id]

        if speaker_obj and speaker_buy:
            earliest_obj_ms = min(m["time_ms"] or 0 for m in speaker_obj)
            latest_buy_ms = max(m["time_ms"] or 0 for m in speaker_buy)

            if latest_buy_ms > earliest_obj_ms:
                summary["objection_resolution"].append({
                    "speaker": speaker_id,
                    "status": "handled_successfully",
                    "objection_at_ms": earliest_obj_ms,
                    "buying_signal_at_ms": latest_buy_ms,
                    "detail": f"Objection at {earliest_obj_ms}ms followed by buying signals at {latest_buy_ms}ms",
                })
            else:
                summary["objection_resolution"].append({
                    "speaker": speaker_id,
                    "status": "unresolved",
                    "objection_at_ms": earliest_obj_ms,
                    "detail": "Objection detected but no subsequent buying signals",
                })
        elif speaker_obj:
            summary["objection_resolution"].append({
                "speaker": speaker_id,
                "status": "unresolved",
                "objection_at_ms": min(m["time_ms"] or 0 for m in speaker_obj),
                "detail": "Objection detected but no buying signals from this speaker",
            })

    return summary
