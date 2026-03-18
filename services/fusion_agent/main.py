"""
NEXUS Fusion Agent (Agent 7)
FastAPI service for cross-modal behavioural analysis.

The Fusion Agent is the orchestrator. It subscribes to Voice and Language
agent Redis Streams, runs pairwise cross-modal rules, computes Unified
Speaker States, and generates narrative session reports via Claude API.

Implements 3 pairwise rules for Phase 1 (audio-only vertical slice):
  - FUSION-02: Content × Stress → Credibility assessment
  - FUSION-07: Hedge × Positive Sentiment → Verbal incongruence
  - FUSION-13: Persuasion × Pace → Urgency authenticity

Endpoints:
  POST /analyse          → Run fusion on pre-collected voice + language signals
  POST /analyse/session  → Run fusion by reading signals from Redis Streams
  POST /report           → Generate narrative report for a completed session
  GET  /health           → Health check
"""
import os
import sys
import uuid
import time
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from fusion_engine import SignalBuffer, compute_unified_state, WINDOW_SHORT_MS
    from rules import FusionRuleEngine
    from narrative import generate_session_narrative
except ImportError:
    from services.fusion_agent.fusion_engine import SignalBuffer, compute_unified_state, WINDOW_SHORT_MS
    from services.fusion_agent.rules import FusionRuleEngine
    from services.fusion_agent.narrative import generate_session_narrative

try:
    from shared.utils.message_bus import message_bus
    from shared.config.settings import config
    HAS_MESSAGE_BUS = True
except ImportError:
    HAS_MESSAGE_BUS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.fusion")

app = FastAPI(
    title="NEXUS Fusion Agent",
    description="Agent 7: Cross-modal behavioural fusion and session reports",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ──
rule_engine: Optional[FusionRuleEngine] = None


@app.on_event("startup")
async def startup():
    global rule_engine
    logger.info("Starting NEXUS Fusion Agent...")
    rule_engine = FusionRuleEngine()

    if HAS_MESSAGE_BUS:
        try:
            await message_bus.connect()
            logger.info("Connected to Redis message bus.")
        except Exception as e:
            logger.warning(f"Redis connection failed (non-fatal): {e}")

    logger.info("Fusion Agent ready.")


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
        "agent": "fusion",
        "version": "0.2.0",
        "models_loaded": {
            "rule_engine": rule_engine is not None,
        },
        "llm": llm_info,
        "redis_connected": HAS_MESSAGE_BUS,
    }


# ── Request / Response Models ──

class FusionSignalInput(BaseModel):
    """A signal from voice or language agent."""
    agent: str
    speaker_id: str = "unknown"
    signal_type: str = ""
    value: Optional[float] = None
    value_text: str = ""
    confidence: float = 0.5
    window_start_ms: int = 0
    window_end_ms: int = 0
    metadata: Optional[dict] = None


class AnalyseRequest(BaseModel):
    """Direct fusion analysis from pre-collected signals."""
    voice_signals: list[FusionSignalInput] = []
    language_signals: list[FusionSignalInput] = []
    session_id: Optional[str] = None
    meeting_type: Optional[str] = "sales_call"
    content_type: Optional[str] = None  # Auto-detected or override
    generate_report: Optional[bool] = True
    voice_summary: Optional[dict] = None
    language_summary: Optional[dict] = None


class SessionAnalyseRequest(BaseModel):
    """Fusion analysis by reading from Redis Streams."""
    session_id: str
    meeting_type: Optional[str] = "sales_call"
    content_type: Optional[str] = None  # Auto-detected or override
    generate_report: Optional[bool] = True
    voice_summary: Optional[dict] = None
    language_summary: Optional[dict] = None


class ReportRequest(BaseModel):
    """Generate a narrative report for a completed session."""
    session_id: str
    duration_seconds: float
    speakers: list[str]
    voice_summary: dict
    language_summary: dict
    fusion_signals: list[dict] = []
    unified_states: list[dict] = []
    meeting_type: Optional[str] = "sales_call"


class AnalyseResponse(BaseModel):
    session_id: str
    speakers: list[str]
    fusion_signals: list[dict]
    unified_states: list[dict]
    alerts: list[dict]
    report: Optional[dict] = None
    summary: dict


@app.post("/analyse", response_model=AnalyseResponse)
async def analyse_signals(request: AnalyseRequest):
    """
    Run fusion analysis on pre-collected voice + language signals.

    Pipeline:
    1. Load signals into temporal buffer
    2. Group by speaker
    3. Run 3 pairwise fusion rules per speaker
    4. Compute Unified Speaker State per speaker
    5. Optionally generate narrative report
    6. Publish fusion signals to Redis Streams
    """
    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()

    voice_dicts = [s.model_dump() for s in request.voice_signals]
    language_dicts = [s.model_dump() for s in request.language_signals]

    logger.info(
        f"[{session_id}] Fusion analysis: "
        f"{len(voice_dicts)} voice + {len(language_dicts)} language signals"
    )

    # ── Step 1: Buffer signals ──
    buffer = SignalBuffer()
    buffer.add_many(voice_dicts)
    buffer.add_many(language_dicts)

    # ── Step 2: Run fusion per speaker ──
    speakers = buffer.speakers
    all_fusion_signals = []
    all_unified_states = []
    all_alerts = []

    for speaker_id in speakers:
        speaker_voice = buffer.get_signals(
            speaker_id, "voice", window_ms=WINDOW_SHORT_MS,
            reference_time_ms=_max_time(voice_dicts + language_dicts),
        )
        speaker_language = buffer.get_signals(
            speaker_id, "language", window_ms=WINDOW_SHORT_MS,
            reference_time_ms=_max_time(voice_dicts + language_dicts),
        )

        # If no short-window signals, use all available signals
        if not speaker_voice:
            speaker_voice = [s for s in voice_dicts if s.get("speaker_id") == speaker_id]
        if not speaker_language:
            speaker_language = [s for s in language_dicts if s.get("speaker_id") == speaker_id]

        if not speaker_voice and not speaker_language:
            continue

        # Compute time window
        all_starts = [
            _to_int(s.get("window_start_ms", 0))
            for s in speaker_voice + speaker_language
        ]
        all_ends = [
            _to_int(s.get("window_end_ms", 0))
            for s in speaker_voice + speaker_language
        ]
        window_start = min(all_starts) if all_starts else 0
        window_end = max(all_ends) if all_ends else 0

        # ── Run pairwise rules ──
        fusion_signals = rule_engine.evaluate(
            speaker_id=speaker_id,
            voice_signals=speaker_voice,
            language_signals=speaker_language,
            window_start_ms=window_start,
            window_end_ms=window_end,
        )
        all_fusion_signals.extend(fusion_signals)

        # ── Compute Unified Speaker State ──
        state = compute_unified_state(
            speaker_id=speaker_id,
            voice_signals=speaker_voice,
            language_signals=speaker_language,
            fusion_signals=fusion_signals,
        )
        state_dict = asdict(state)
        all_unified_states.append(state_dict)

        # ── Generate alerts for significant fusion signals ──
        for fs in fusion_signals:
            confidence = fs.get("confidence", 0)
            if confidence >= 0.50:
                alert = _create_alert(session_id, speaker_id, fs)
                if alert:
                    all_alerts.append(alert)

        logger.info(
            f"[{session_id}] Speaker {speaker_id}: "
            f"{len(fusion_signals)} fusion signals, "
            f"stress={state.stress_level:.2f}, "
            f"confidence={state.confidence_level:.2f}, "
            f"authenticity={state.authenticity_score:.2f}"
        )

    # ── Step 3: Publish to Redis ──
    if HAS_MESSAGE_BUS:
        published = 0
        for signal in all_fusion_signals:
            try:
                await message_bus.publish_signal(
                    session_id=session_id,
                    agent="fusion",
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
                logger.warning(f"Failed to publish fusion signal: {e}")

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
            except Exception as e:
                logger.warning(f"Failed to publish alert: {e}")

        logger.info(f"[{session_id}] Published {published} fusion signals + {len(all_alerts)} alerts")

    # ── Step 4: Generate narrative report ──
    report = None
    if request.generate_report:
        logger.info(f"[{session_id}] Generating narrative report...")

        # Compute duration from signal timestamps
        all_timestamps = [
            _to_int(s.get("window_end_ms", 0))
            for s in voice_dicts + language_dicts
        ]
        duration_seconds = (max(all_timestamps) - min(all_timestamps)) / 1000.0 if all_timestamps else 0

        # Use content_type if provided, falling back to meeting_type
        report_type = request.content_type or request.meeting_type or "sales_call"

        report = generate_session_narrative(
            session_id=session_id,
            duration_seconds=duration_seconds,
            speakers=speakers,
            voice_summary=request.voice_summary or {},
            language_summary=request.language_summary or {},
            fusion_signals=all_fusion_signals,
            unified_states=all_unified_states,
            meeting_type=report_type,
        )

    # ── Build summary ──
    elapsed = time.time() - start_time
    logger.info(
        f"[{session_id}] Fusion complete: "
        f"{len(all_fusion_signals)} fusion signals, "
        f"{len(all_alerts)} alerts, "
        f"{len(all_unified_states)} speaker states in {elapsed:.1f}s"
    )

    summary = _build_summary(all_fusion_signals, all_unified_states, all_alerts)

    return AnalyseResponse(
        session_id=session_id,
        speakers=speakers,
        fusion_signals=all_fusion_signals,
        unified_states=all_unified_states,
        alerts=all_alerts,
        report=report,
        summary=summary,
    )


@app.post("/analyse/session", response_model=AnalyseResponse)
async def analyse_session_from_redis(request: SessionAnalyseRequest):
    """
    Run fusion by reading voice + language signals from Redis Streams.
    This is the intended production flow — the Fusion Agent pulls from
    streams that Voice and Language agents have published to.
    """
    if not HAS_MESSAGE_BUS:
        raise HTTPException(503, "Redis not available — use POST /analyse with direct signals")

    session_id = request.session_id
    logger.info(f"[{session_id}] Reading signals from Redis Streams...")

    # Read all voice and language signals for this session
    voice_signals = await message_bus.get_latest_signals(
        session_id=session_id, agent="voice", count=500
    )
    language_signals = await message_bus.get_latest_signals(
        session_id=session_id, agent="language", count=500
    )

    logger.info(
        f"[{session_id}] Read {len(voice_signals)} voice + "
        f"{len(language_signals)} language signals from Redis"
    )

    if not voice_signals and not language_signals:
        raise HTTPException(
            404,
            f"No signals found in Redis for session {session_id}. "
            "Run Voice and Language agents first."
        )

    # Delegate to the main analyse endpoint
    analyse_req = AnalyseRequest(
        voice_signals=[
            FusionSignalInput(**_normalise_redis_signal(s)) for s in voice_signals
        ],
        language_signals=[
            FusionSignalInput(**_normalise_redis_signal(s)) for s in language_signals
        ],
        session_id=session_id,
        meeting_type=request.meeting_type,
        generate_report=request.generate_report,
        voice_summary=request.voice_summary,
        language_summary=request.language_summary,
    )

    return await analyse_signals(analyse_req)


@app.post("/report")
async def generate_report(request: ReportRequest):
    """Generate a narrative report for a completed session."""
    logger.info(f"[{request.session_id}] Generating narrative report...")

    report = generate_session_narrative(
        session_id=request.session_id,
        duration_seconds=request.duration_seconds,
        speakers=request.speakers,
        voice_summary=request.voice_summary,
        language_summary=request.language_summary,
        fusion_signals=request.fusion_signals,
        unified_states=request.unified_states,
        meeting_type=request.meeting_type,
    )

    if report is None:
        raise HTTPException(500, "Narrative generation failed")

    return {"session_id": request.session_id, "report": report}


# ── Helpers ──

def _to_int(v) -> int:
    if v is None or v == "":
        return 0
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


def _max_time(signals: list[dict]) -> int:
    """Get the latest timestamp from a list of signals."""
    if not signals:
        return int(time.time() * 1000)
    times = [_to_int(s.get("window_end_ms", 0)) for s in signals]
    return max(times) if times else int(time.time() * 1000)


def _normalise_redis_signal(s: dict) -> dict:
    """Normalise a Redis signal dict to match FusionSignalInput fields."""
    metadata = s.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    value = s.get("value", None)
    if value == "" or value is None:
        value = None
    else:
        try:
            value = float(value)
        except (ValueError, TypeError):
            value = None

    return {
        "agent": s.get("agent", "unknown"),
        "speaker_id": s.get("speaker_id", "unknown"),
        "signal_type": s.get("signal_type", ""),
        "value": value,
        "value_text": s.get("value_text", ""),
        "confidence": float(s.get("confidence", 0.5)),
        "window_start_ms": _to_int(s.get("window_start_ms", 0)),
        "window_end_ms": _to_int(s.get("window_end_ms", 0)),
        "metadata": metadata,
    }


def _create_alert(session_id: str, speaker_id: str, fusion_signal: dict) -> Optional[dict]:
    """Create an alert from a significant fusion signal."""
    sig_type = fusion_signal.get("signal_type", "")
    value_text = fusion_signal.get("value_text", "")
    confidence = fusion_signal.get("confidence", 0)

    alert_map = {
        "credibility_assessment": {
            "credibility_concern": {
                "severity": "orange",
                "title": "Content-Voice Incongruence",
                "description": (
                    "Positive language detected alongside elevated vocal stress. "
                    "This may indicate discomfort with the stated position."
                ),
            },
            "mild_incongruence": {
                "severity": "yellow",
                "title": "Mild Content-Voice Mismatch",
                "description": "Slight mismatch between verbal content and vocal indicators.",
            },
        },
        "verbal_incongruence": {
            "strong_verbal_incongruence": {
                "severity": "orange",
                "title": "Strong Verbal Hedging",
                "description": (
                    "Positive sentiment expressed with heavily hedged language. "
                    "Speaker may be agreeing without genuine conviction."
                ),
            },
            "moderate_verbal_incongruence": {
                "severity": "yellow",
                "title": "Hedged Agreement",
                "description": "Agreement expressed with notable hedging language.",
            },
            "incongruence_with_objection": {
                "severity": "orange",
                "title": "Hidden Objection Detected",
                "description": (
                    "Positive sentiment combined with objection markers and weak power language. "
                    "Speaker may have unstated concerns."
                ),
            },
        },
        "urgency_authenticity": {
            "manufactured_urgency": {
                "severity": "yellow",
                "title": "Potentially Manufactured Urgency",
                "description": (
                    "Fast-paced persuasive language with concurrent stress indicators. "
                    "Urgency may be artificially created rather than genuine excitement."
                ),
            },
            "authentic_urgency": {
                "severity": "green",
                "title": "Authentic Enthusiasm Detected",
                "description": "Fast-paced persuasive language supported by confident vocal patterns.",
            },
        },
    }

    type_alerts = alert_map.get(sig_type, {})
    alert_info = type_alerts.get(value_text)

    if not alert_info:
        return None

    # Only emit yellow/orange/red alerts
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


def _build_summary(
    fusion_signals: list[dict],
    unified_states: list[dict],
    alerts: list[dict],
) -> dict:
    """Build a summary from fusion outputs."""
    signal_types = {}
    for s in fusion_signals:
        st = s.get("signal_type", "unknown")
        signal_types[st] = signal_types.get(st, 0) + 1

    alert_severities = {}
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
