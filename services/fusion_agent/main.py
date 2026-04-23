# services/fusion_agent/main.py
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
import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# isort: split
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# isort: split
from shared.models.signals import FusionSignalInput
from shared.models.requests import (
    FusionAnalyseRequest as AnalyseRequest,
    FusionSessionAnalyseRequest as SessionAnalyseRequest,
    ReportRequest,
    FusionAnalyseResponse as AnalyseResponse,
)

try:
    from fusion_engine import SignalBuffer, compute_unified_state, WINDOW_SHORT_MS
    from rules import FusionRuleEngine
    from narrative import generate_session_narrative
    from signal_graph import SignalGraph
    from graph_analytics import GraphAnalytics
    from compound_patterns import CompoundPatternEngine
    from temporal_patterns import TemporalPatternEngine
except ImportError:
    from services.fusion_agent.fusion_engine import SignalBuffer, compute_unified_state, WINDOW_SHORT_MS
    from services.fusion_agent.rules import FusionRuleEngine
    from services.fusion_agent.narrative import generate_session_narrative
    from services.fusion_agent.signal_graph import SignalGraph
    from services.fusion_agent.graph_analytics import GraphAnalytics
    from services.fusion_agent.compound_patterns import CompoundPatternEngine
    from services.fusion_agent.temporal_patterns import TemporalPatternEngine

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
rule_engine:     Optional[FusionRuleEngine]     = None
compound_engine: Optional[CompoundPatternEngine] = None
temporal_engine: Optional[TemporalPatternEngine] = None


@app.on_event("startup")
async def startup():
    global rule_engine, compound_engine, temporal_engine
    logger.info("Starting NEXUS Fusion Agent...")
    rule_engine     = FusionRuleEngine()
    compound_engine = CompoundPatternEngine()
    temporal_engine = TemporalPatternEngine()

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
    content_type = request.content_type or request.meeting_type or "sales_call"
    start_time = time.time()

    # Create content-type profile for all rules
    try:
        from shared.config.content_type_profile import ContentTypeProfile
        profile = ContentTypeProfile(content_type)
    except ImportError:
        profile = None

    voice_dicts = [s.model_dump() for s in request.voice_signals]
    language_dicts = [s.model_dump() for s in request.language_signals]

    # Separate video signals out of voice_dicts (gateway bundles them together)
    video_dicts = [s for s in voice_dicts if s.get("agent") == "video"]
    pure_voice_dicts = [s for s in voice_dicts if s.get("agent") != "video"]

    logger.info(
        f"[{session_id}] Fusion analysis: "
        f"{len(pure_voice_dicts)} voice + {len(language_dicts)} language"
        f" + {len(video_dicts)} video signals"
    )

    # ── Step 1: Buffer signals ──
    t_step = time.time()
    buffer = SignalBuffer()
    buffer.add_many(pure_voice_dicts)
    buffer.add_many(language_dicts)
    buffer.add_many(video_dicts)

    logger.info(
        f"[{session_id}] Step 1 done: buffered {len(pure_voice_dicts)+len(language_dicts)+len(video_dicts)} signals "
        f"for {len(buffer.speakers)} speakers in {time.time()-t_step:.2f}s"
    )

    # ── Step 2: Run fusion per speaker (parallel via asyncio.gather) ──
    t_step = time.time()
    speakers = buffer.speakers
    all_fusion_signals = []
    all_unified_states = []
    all_alerts = []

    ref_time = _max_time(pure_voice_dicts + language_dicts + video_dicts)

    async def _analyse_speaker(speaker_id: str) -> dict:
        speaker_voice = buffer.get_signals(
            speaker_id, "voice", window_ms=WINDOW_SHORT_MS,
            reference_time_ms=ref_time,
        )
        speaker_language = buffer.get_signals(
            speaker_id, "language", window_ms=WINDOW_SHORT_MS,
            reference_time_ms=ref_time,
        )
        speaker_video = buffer.get_signals(
            speaker_id, "video", window_ms=WINDOW_SHORT_MS,
            reference_time_ms=ref_time,
        )

        if not speaker_voice:
            speaker_voice = [s for s in pure_voice_dicts if s.get("speaker_id") == speaker_id]
        if not speaker_language:
            speaker_language = [s for s in language_dicts if s.get("speaker_id") == speaker_id]
        if not speaker_video:
            speaker_video = [s for s in video_dicts if s.get("speaker_id") == speaker_id]

        if not speaker_voice and not speaker_language:
            return {"signals": [], "state": None, "alerts": []}

        all_starts = [
            _to_int(s.get("window_start_ms", 0))
            for s in speaker_voice + speaker_language + speaker_video
        ]
        all_ends = [
            _to_int(s.get("window_end_ms", 0))
            for s in speaker_voice + speaker_language + speaker_video
        ]
        window_start = min(all_starts) if all_starts else 0
        window_end = max(all_ends) if all_ends else 0

        fusion_signals = rule_engine.evaluate(
            speaker_id=speaker_id,
            voice_signals=speaker_voice,
            language_signals=speaker_language,
            video_signals=speaker_video,
            window_start_ms=window_start,
            window_end_ms=window_end,
            content_type=content_type,
            profile=profile,
        )

        # Compound patterns (Phase 2F) — per-segment evaluation with ±5s signal window.
        # Running on the single window_start/window_end span causes compound signals to
        # timestamp across the full session when the buffer fallback is used, making
        # them appear on every transcript segment in the dashboard.
        compound_signals: list[dict] = []
        if compound_engine is not None:
            all_speaker_sigs = speaker_voice + speaker_language + speaker_video + fusion_signals

            COMPOUND_ALIGN_MS = 5000  # look ±5s around each voice/language segment

            # Collect unique time windows from voice + language (the "anchor" agents)
            segment_windows: set[tuple[int, int]] = set()
            for s in speaker_voice + speaker_language:
                ws_ms = s.get("window_start_ms", 0)
                we_ms = s.get("window_end_ms", 0)
                if ws_ms >= 0 and we_ms > ws_ms:
                    segment_windows.add((ws_ms, we_ms))

            for seg_start, seg_end in sorted(segment_windows):
                # Overlap logic: include any signal whose window overlaps the ±5s zone.
                # Containment (<=/>= both ends) excluded long signals like screen_contact
                # (30s block windows) even when they clearly cover the segment.
                zone_start = seg_start - COMPOUND_ALIGN_MS
                zone_end   = seg_end   + COMPOUND_ALIGN_MS
                seg_sigs = [
                    s for s in all_speaker_sigs
                    if s.get("window_start_ms", 0) < zone_end
                    and s.get("window_end_ms", 0)   > zone_start
                ]
                if len(seg_sigs) < 3:
                    continue

                seg_compounds = compound_engine.evaluate(
                    speaker_id=speaker_id,
                    voice_signals=[s for s in seg_sigs if s.get("agent") == "voice"],
                    language_signals=[s for s in seg_sigs if s.get("agent") == "language"],
                    video_signals=[s for s in seg_sigs
                                   if s.get("agent") in ("facial", "body", "gaze", "video")],
                    fusion_signals=[s for s in seg_sigs if s.get("agent") == "fusion"],
                    window_start_ms=seg_start,
                    window_end_ms=seg_end,
                )
                compound_signals.extend(seg_compounds)

            # Session-level frequency cap: keep highest-confidence firing per signal_type.
            # Must be done here (not inside evaluate()) so the cap is session-wide, not
            # per-segment (evaluate() is called once per segment above).
            _MAX_FIRES: dict[str, int] = {
                "peak_performance": 2, "genuine_engagement": 2,
                "active_disengagement": 2, "conflict_escalation": 2,
                "deception_cluster": 1, "emotional_suppression": 2,
            }
            _DEFAULT_MAX = 3
            if compound_signals:
                fire_counts: dict[str, int] = defaultdict(int)
                deduped: list[dict] = []
                for s in sorted(compound_signals,
                                key=lambda x: x.get("confidence", 0), reverse=True):
                    st = s.get("signal_type", "")
                    if fire_counts[st] < _MAX_FIRES.get(st, _DEFAULT_MAX):
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

        speaker_alerts = []
        for fs in fusion_signals:
            if fs.get("confidence", 0) >= 0.50:
                alert = _create_alert(session_id, speaker_id, fs, content_type, profile)
                if alert:
                    speaker_alerts.append(alert)

        logger.info(
            f"[{session_id}] Speaker {speaker_id}: "
            f"{len(fusion_signals)} fusion signals, "
            f"stress={state.stress_level:.2f}, "
            f"confidence={state.confidence_level:.2f}, "
            f"authenticity={state.authenticity_score:.2f}"
        )

        return {"signals": fusion_signals, "state": state, "alerts": speaker_alerts}

    speaker_results = await asyncio.gather(
        *[_analyse_speaker(sid) for sid in speakers]
    )

    for result in speaker_results:
        all_fusion_signals.extend(result["signals"])
        if result["state"] is not None:
            all_unified_states.append(asdict(result["state"]))
        all_alerts.extend(result["alerts"])

    logger.info(
        f"[{session_id}] Step 2 done: pairwise+compound fusion "
        f"{len(all_fusion_signals)} signals, {len(all_alerts)} alerts in {time.time()-t_step:.2f}s"
    )

    # ── Step 2.5: Temporal patterns (Phase 2G) — session-level, full signal set ──
    t_step = time.time()
    if temporal_engine is not None:
        all_source_sigs = pure_voice_dicts + language_dicts + video_dicts
        if all_source_sigs:
            session_start_ms_val = min(
                _to_int(s.get("window_start_ms", 0)) for s in all_source_sigs
            )
        else:
            session_start_ms_val = 0

        for sid in speakers:
            speaker_full = [
                s for s in all_source_sigs
                if s.get("speaker_id") == sid
            ]
            temporal_sigs = temporal_engine.evaluate(
                speaker_id=sid,
                all_signals=speaker_full,
                session_start_ms=session_start_ms_val,
                session_end_ms=ref_time,
            )
            if temporal_sigs:
                logger.info(
                    f"[{session_id}] Speaker {sid}: {len(temporal_sigs)} temporal pattern signals"
                )
            all_fusion_signals.extend(temporal_sigs)

    logger.info(f"[{session_id}] Step 2.5 done: temporal patterns in {time.time()-t_step:.2f}s")

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

    # Extract entities from language summary (used by narrative + graph)
    entities = {}
    if request.language_summary:
        entities = request.language_summary.get("entities", {})

    # Extract conversation summary if present (passed via voice_summary by API Gateway)
    conversation_summary = {}
    if request.voice_summary:
        conversation_summary = request.voice_summary.get("conversation", {})

    # ── Step 4: Build signal graph + analytics ──
    t_step = time.time()
    graph_json = {}
    key_paths = []
    graph_insights = {}
    try:
        graph = SignalGraph()
        # Include conversation signals in graph if available
        conversation_signals_for_graph = []
        if conversation_summary.get("signals"):
            conversation_signals_for_graph = conversation_summary["signals"]

        graph.build_from_session(
            voice_signals=voice_dicts,
            language_signals=language_dicts + conversation_signals_for_graph,
            fusion_signals=all_fusion_signals,
            transcript_segments=[],
            entities=entities,
        )
        graph_json = graph.to_json()
        key_paths = graph.get_key_paths(max_paths=5)

        analytics = GraphAnalytics(graph)
        graph_insights = analytics.compute_all(content_type=content_type)

        graph_signals = rule_engine.evaluate_graph_insights(
            graph_insights, speakers, all_fusion_signals,
            content_type=content_type, profile=profile,
        )
        all_fusion_signals.extend(graph_signals)

        logger.info(
            f"[{session_id}] Signal graph: "
            f"{graph_json['stats']['node_count']} nodes, "
            f"{graph_json['stats']['edge_count']} edges, "
            f"{len(key_paths)} key paths, "
            f"{len(graph_signals)} graph-based signals"
        )
    except Exception as e:
        logger.warning(f"[{session_id}] Signal graph/analytics failed (non-fatal): {e}")

    logger.info(f"[{session_id}] Step 4 done: signal graph built in {time.time()-t_step:.2f}s")

    # ── Step 5: Generate narrative report ──
    report = None
    if request.generate_report:
        t_step = time.time()
        logger.info(f"[{session_id}] Step 5: Generating narrative report...")

        all_timestamps = [
            _to_int(s.get("window_end_ms", 0))
            for s in voice_dicts + language_dicts
        ]
        duration_seconds = (max(all_timestamps) - min(all_timestamps)) / 1000.0 if all_timestamps else 0
        report_type = request.content_type or request.meeting_type or "sales_call"

        report = await generate_session_narrative(
            session_id=session_id,
            duration_seconds=duration_seconds,
            speakers=speakers,
            voice_summary=request.voice_summary or {},
            language_summary=request.language_summary or {},
            fusion_signals=all_fusion_signals,
            unified_states=all_unified_states,
            meeting_type=report_type,
            entities=entities,
            graph_analytics=graph_insights,
            conversation_summary=conversation_summary,
        )

    if request.generate_report:
        logger.info(f"[{session_id}] Step 5 done: narrative report in {time.time()-t_step:.2f}s")

    # ── Build summary ──
    elapsed = time.time() - start_time
    logger.info(
        f"[{session_id}] Fusion Agent complete: "
        f"{len(all_fusion_signals)} fusion signals, "
        f"{len(all_alerts)} alerts, "
        f"{len(all_unified_states)} speaker states in {elapsed:.1f}s"
    )

    summary = _build_summary(all_fusion_signals, all_unified_states, all_alerts)
    summary["signal_graph"] = graph_json
    summary["key_paths"] = key_paths
    summary["graph_analytics"] = graph_insights

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

    report = await generate_session_narrative(
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

try:
    from shared.utils.conversions import to_int as _to_int
except ImportError:
    def _to_int(v) -> int:
        if v is None or v == "":
            return 0
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return 0


def _max_time(signals: list[dict]) -> int:
    """Get the latest timestamp from a list of signals (0 if empty)."""
    if not signals:
        return 0
    times = [_to_int(s.get("window_end_ms", 0)) for s in signals]
    return max(times) if times else 0


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


def _create_alert(session_id: str, speaker_id: str, fusion_signal: dict, content_type: str = "sales_call", profile=None) -> Optional[dict]:
    """Create an alert from a significant fusion signal."""
    sig_type = fusion_signal.get("signal_type", "")
    value_text = fusion_signal.get("value_text", "")
    confidence = fusion_signal.get("confidence", 0)

    # Content-type gating via profile
    SALES_ONLY_ALERTS = {
        ("urgency_authenticity", "manufactured_urgency"),
        ("urgency_authenticity", "ambiguous_urgency"),
        ("credibility_assessment", "credibility_concern"),
    }
    if (sig_type, value_text) in SALES_ONLY_ALERTS and content_type not in ("sales_call", "pitch", "presentation"):
        return None

    # For non-sales content types, raise confidence threshold
    if content_type in ("internal", "meeting", "interview", "podcast"):
        min_alert_conf = 0.60
        if profile:
            min_alert_conf = profile.get_threshold("ALERT", "min_confidence", 0.60)
        if confidence < min_alert_conf:
            return None

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
