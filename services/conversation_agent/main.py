"""
NEXUS Conversation Agent (Agent 6)
FastAPI service for dialogue dynamics analysis between speakers.

Analyses turn-taking, response latency, interruptions, dominance, rapport,
engagement, and conversation balance from diarised transcript segments.

Implements 7 rules from the NEXUS Rule Engine:
  - CONVO-TURN-01: Turn-taking pattern
  - CONVO-LAT-01:  Response latency pattern
  - CONVO-DOM-01:  Dominance score
  - CONVO-INT-01:  Interruption pattern
  - CONVO-RAP-01:  Rapport indicator
  - CONVO-ENG-01:  Conversation engagement
  - CONVO-BAL-01:  Conversation balance

Endpoints:
  POST /analyse          -> Analyse transcript segments for conversation dynamics
  GET  /health           -> Health check
"""
import os
import sys
import uuid
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# isort: split
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# isort: split
try:
    from shared.models.requests import (
        ConversationAnalysisRequest as AnalysisRequest,
        ConversationAnalysisResponse as AnalysisResponse,
    )
except ImportError:
    # Fallback: define locally if shared models are not available
    class AnalysisRequest(BaseModel):
        segments: list[dict]
        speakers: list[str] = []
        content_type: Optional[str] = "sales_call"
        session_id: Optional[str] = None
        language_signals: Optional[list[dict]] = None

    class AnalysisResponse(BaseModel):
        session_id: str
        speaker_count: int
        signals: list[dict]
        summary: dict

# Import from same directory (works in Docker /app context)
try:
    from feature_extractor import ConversationFeatureExtractor
    from rules import ConversationRuleEngine
except ImportError:
    from services.conversation_agent.feature_extractor import ConversationFeatureExtractor
    from services.conversation_agent.rules import ConversationRuleEngine

# Shared utilities
try:
    from shared.utils.message_bus import message_bus
    from shared.config.settings import config
    HAS_MESSAGE_BUS = True
except ImportError:
    HAS_MESSAGE_BUS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.conversation")

app = FastAPI(
    title="NEXUS Conversation Agent",
    description="Agent 6: Dialogue dynamics analysis between speakers",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Globals (initialised on startup) --
feature_extractor: Optional[ConversationFeatureExtractor] = None
rule_engine: Optional[ConversationRuleEngine] = None


@app.on_event("startup")
async def startup():
    global feature_extractor, rule_engine
    logger.info("Starting NEXUS Conversation Agent...")

    feature_extractor = ConversationFeatureExtractor()
    rule_engine = ConversationRuleEngine()

    # Connect to Redis if available
    if HAS_MESSAGE_BUS:
        try:
            await message_bus.connect()
            logger.info("Connected to Redis message bus")
        except Exception as e:
            logger.warning(f"Redis connection failed (non-fatal): {e}")

    logger.info("Conversation Agent ready.")


@app.on_event("shutdown")
async def shutdown():
    if HAS_MESSAGE_BUS:
        try:
            await message_bus.disconnect()
        except Exception:
            pass


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "conversation-agent",
        "version": "0.1.0",
        "feature_extractor": feature_extractor is not None,
        "rule_engine": rule_engine is not None,
    }


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(request: AnalysisRequest):
    """
    Analyse transcript segments for conversation dynamics.

    Pipeline:
    1. Extract conversation features (per-speaker, per-pair, session)
    2. Run rule engine (7 rules)
    3. Build summary
    4. Publish to Redis (if available)
    5. Return signals + summary
    """
    t0 = time.time()

    session_id = request.session_id or str(uuid.uuid4())
    segments = request.segments
    speakers = request.speakers or []
    content_type = request.content_type or "sales_call"

    if not segments:
        raise HTTPException(400, "No segments provided")

    logger.info(
        f"[{session_id}] Analysing {len(segments)} segments, "
        f"{len(speakers) if speakers else 'auto-detect'} speakers, "
        f"content_type={content_type}"
    )

    # ── Step 1: Feature Extraction ──
    features = feature_extractor.extract_all(segments, speakers or None)

    per_speaker = features.get("per_speaker", {})
    per_pair = features.get("per_pair", {})
    session_features = features.get("session", {})

    detected_speakers = list(per_speaker.keys())
    speaker_count = len(detected_speakers)

    logger.info(
        f"[{session_id}] Features extracted: "
        f"{speaker_count} speakers, "
        f"{session_features.get('total_turns', 0)} turns, "
        f"{session_features.get('total_duration_ms', 0):.0f}ms"
    )

    # ── Step 2: Rule Engine ──
    language_signals = getattr(request, "language_signals", None) or None
    signals = rule_engine.evaluate(features, content_type, language_signals=language_signals)
    logger.info(f"[{session_id}] Rule engine produced {len(signals)} signals")

    # ── Step 3: Build Summary ──
    summary = _build_summary(per_speaker, per_pair, session_features, signals, detected_speakers)

    # ── Step 4: Publish to Redis ──
    if HAS_MESSAGE_BUS and signals:
        try:
            for sig in signals:
                await message_bus.publish(
                    stream=f"nexus:stream:conversation:{session_id}",
                    data=sig,
                )
            logger.info(f"[{session_id}] Published {len(signals)} signals to Redis")
        except Exception as e:
            logger.warning(f"[{session_id}] Redis publish failed (non-fatal): {e}")

    elapsed = time.time() - t0
    logger.info(f"[{session_id}] Conversation analysis complete in {elapsed:.2f}s")

    return AnalysisResponse(
        session_id=session_id,
        speaker_count=speaker_count,
        signals=[s for s in signals],
        summary=summary,
    )


def _build_summary(
    per_speaker: dict,
    per_pair: dict,
    session_features: dict,
    signals: list[dict],
    speakers: list[str],
) -> dict:
    """
    Build a human-readable summary from features and signals.
    Includes per_speaker stats and session-level stats.
    """
    # Index signals by type and speaker for easy lookup
    signal_index = {}
    for sig in signals:
        key = (sig.get("signal_type", ""), sig.get("speaker_id", ""))
        signal_index[key] = sig

    # Per-speaker summary
    speaker_summaries = {}
    for spk in speakers:
        spk_data = per_speaker.get(spk, {})

        # Get dominance score from signals
        dom_sig = signal_index.get(("dominance_score", spk), {})
        dominance_score = dom_sig.get("value", 0)

        # Get engagement score from signals
        eng_sig = signal_index.get(("conversation_engagement", spk), {})
        engagement_score = eng_sig.get("value", 0)

        # Get avg response latency (from pairs involving this speaker)
        latencies = []
        for pair_key, pair_data in per_pair.items():
            if spk in pair_key.split("__"):
                lat = pair_data.get("response_latency_ms_avg", 0)
                if lat > 0:
                    latencies.append(lat)
        avg_response_latency_ms = (
            sum(latencies) / len(latencies) if latencies else 0
        )

        speaker_summaries[spk] = {
            "talk_time_pct": spk_data.get("talk_time_pct", 0),
            "dominance_score": round(dominance_score, 3) if dominance_score else 0,
            "engagement_score": round(engagement_score, 3) if engagement_score else 0,
            "interruptions_made": spk_data.get("interruption_count", 0),
            "interruptions_received": spk_data.get("was_interrupted_count", 0),
            "avg_response_latency_ms": round(avg_response_latency_ms, 1),
            "questions_asked": spk_data.get("questions_asked", 0),
            "back_channels": spk_data.get("back_channel_count", 0),
        }

    # Session summary
    balance_sig = signal_index.get(("conversation_balance", "session"), {})
    turn_sig = signal_index.get(("turn_taking_pattern", "session"), {})

    # Rapport score (average across pairs)
    rapport_scores = []
    for sig in signals:
        if sig.get("signal_type") == "rapport_indicator":
            rapport_scores.append(sig.get("value", 0))
    avg_rapport = sum(rapport_scores) / len(rapport_scores) if rapport_scores else 0

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

    return {
        "per_speaker": speaker_summaries,
        "session": session_summary,
    }
