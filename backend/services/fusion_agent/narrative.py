# services/fusion_agent/narrative.py
"""
NEXUS Fusion Agent - Narrative Generator
Uses the shared LLM client (Anthropic Claude or OpenAI) to produce
human-readable session analysis reports.

Takes the full set of signals, unified speaker states, and fusion alerts
and generates a structured narrative suitable for post-meeting review.

Report types (from docs/PLAN.md):
  - Sales call: buying signals timeline, objection map, decision readiness moments
  - Client meeting: rapport trajectory, satisfaction indicators, risk flags
  - Internal meeting: engagement distribution, contribution balance, action items

For Phase 1, we implement a general-purpose report structure that covers
sales calls (the primary use case for the audio-only vertical slice).
"""
import os
import json
import logging
from typing import Optional
from dataclasses import asdict

logger = logging.getLogger("nexus.fusion.narrative")


def _get_llm_complete():
    """Get the shared llm_client.complete function, or None if unavailable."""
    try:
        from shared.utils.llm_client import complete, is_configured
        if is_configured():
            return complete
        else:
            logger.warning("LLM client not configured — narrative generation will use fallback")
            return None
    except ImportError:
        logger.warning("shared.utils.llm_client not available — narrative generation will use fallback")
        return None


async def generate_session_narrative(
    session_id: str,
    duration_seconds: float,
    speakers: list[str],
    voice_summary: dict,
    language_summary: dict,
    fusion_signals: list[dict],
    unified_states: list[dict],
    meeting_type: str = "sales_call",
    entities: Optional[dict] = None,
    graph_analytics: Optional[dict] = None,
    conversation_summary: Optional[dict] = None,
    video_summary: Optional[dict] = None,
    transcript_segments: Optional[list[dict]] = None,
) -> Optional[dict]:
    """
    Generate a structured narrative report for the session using the LLM.

    Args:
        session_id: Session identifier
        duration_seconds: Total session duration
        speakers: List of speaker IDs
        voice_summary: Summary from Voice Agent (stress peaks, filler stats, tone)
        language_summary: Summary from Language Agent (sentiment, buying signals, objections)
        fusion_signals: All fusion signals produced during the session
        unified_states: Final unified speaker states
        meeting_type: Type of meeting (sales_call, client_meeting, internal)
        entities: Extracted entities from language analysis
        graph_analytics: Graph-derived analytics
        conversation_summary: Summary from Conversation Agent (turn-taking, rapport, dominance)

    Returns:
        {
            "executive_summary": str,
            "speaker_analyses": {speaker_id: str},
            "key_moments": [{"time_ms": int, "description": str, "significance": str}],
            "cross_modal_insights": [str],
            "recommendations": [str],
            "raw_response": str
        }
    """
    entities = entities or {}
    graph_analytics = graph_analytics or {}
    conversation_summary = conversation_summary or {}
    llm_complete = _get_llm_complete()
    if llm_complete is None:
        return _fallback_narrative(
            speakers, voice_summary, language_summary, fusion_signals, entities,
            transcript_segments=transcript_segments,
            video_summary=video_summary,
            conversation_summary=conversation_summary,
        )

    # ── Build the structured context for the LLM ──
    context = _build_context(
        session_id, duration_seconds, speakers,
        voice_summary, language_summary, fusion_signals, unified_states,
        entities, graph_analytics, conversation_summary,
        video_summary=video_summary,
    )

    system_prompt, user_prompt = _build_prompt(context, meeting_type)

    try:
        from shared.utils.llm_client import acomplete
        raw_text = await acomplete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2048,
        )

        # Parse structured response
        report = _parse_narrative_response(raw_text, speakers)
        report["raw_response"] = raw_text
        return report

    except Exception as e:
        logger.warning(f"LLM narrative generation failed: {e}")
        return _fallback_narrative(
            speakers, voice_summary, language_summary, fusion_signals, entities,
            transcript_segments=transcript_segments,
            video_summary=video_summary,
            conversation_summary=conversation_summary,
        )


def _build_speaker_name_map(entities: dict) -> dict[str, str]:
    """
    Build a mapping from speaker label → display name using extracted people entities.
    e.g. {"Speaker_0": "John (Seller)", "Speaker_1": "Sarah (Prospect)"}
    Falls back to the label itself when no name is known.
    """
    name_map: dict[str, str] = {}
    for p in entities.get("people", []):
        label = p.get("speaker_label", "")
        name = p.get("name", "")
        role = p.get("role", "")
        if label and name:
            display = f"{name} ({role.capitalize()})" if role else name
            name_map[label] = display
    return name_map


def _display(speaker_id: str, name_map: dict[str, str]) -> str:
    """Return display name for a speaker label, e.g. 'John (Seller)' or 'Speaker_0'."""
    return name_map.get(speaker_id, speaker_id)


def _build_context(
    session_id: str,
    duration_seconds: float,
    speakers: list[str],
    voice_summary: dict,
    language_summary: dict,
    fusion_signals: list[dict],
    unified_states: list[dict],
    entities: Optional[dict] = None,
    graph_analytics: Optional[dict] = None,
    conversation_summary: Optional[dict] = None,
    video_summary: Optional[dict] = None,
) -> str:
    """Build a structured text context block for the LLM."""
    entities = entities or {}
    name_map = _build_speaker_name_map(entities)

    lines = []
    lines.append(f"SESSION: {session_id}")
    lines.append(f"DURATION: {duration_seconds:.0f} seconds ({duration_seconds/60:.1f} minutes)")

    # Show speakers with real names if known
    speaker_display = [_display(s, name_map) for s in speakers]
    lines.append(f"SPEAKERS: {', '.join(speaker_display)}")

    # Speaker name legend so LLM always has the mapping
    if name_map:
        lines.append("SPEAKER NAME MAP (use these names in the report, not Speaker_X labels):")
        for label, display in name_map.items():
            lines.append(f"  {label} = {display}")
    lines.append("")

    # Voice summary per speaker
    lines.append("\n=== VOICE ANALYSIS ===")
    for speaker_id, data in voice_summary.get("per_speaker", {}).items():
        lines.append(f"\n[{_display(speaker_id, name_map)}]")
        lines.append(f"  Baseline pitch: {data.get('baseline_f0_hz', 0)} Hz")
        lines.append(f"  Baseline speech rate: {data.get('baseline_rate_wpm', 0)} WPM")
        lines.append(f"  Average stress: {data.get('avg_stress', 0):.3f}")
        lines.append(f"  Maximum stress: {data.get('max_stress', 0):.3f}")
        lines.append(f"  Total fillers: {data.get('total_fillers', 0)}")
        lines.append(f"  Pitch elevations: {data.get('pitch_elevation_events', 0)}")
        tone_dist = data.get("tone_distribution", {})
        if tone_dist:
            lines.append(f"  Tone distribution: {json.dumps(tone_dist)}")

    # Stress peaks
    peaks = voice_summary.get("stress_peaks", [])
    if peaks:
        lines.append("\nSTRESS PEAKS (top moments):")
        for p in peaks[:5]:
            time_s = p.get('time_ms', 0) / 1000
            lines.append(
                f"  {time_s:.0f}s — {_display(p.get('speaker', '?'), name_map)}: "
                f"stress={p.get('stress_score', 0):.3f}"
            )

    # Video analysis per speaker
    if video_summary and video_summary.get("per_speaker"):
        lines.append("\n=== VIDEO ANALYSIS (Facial · Gaze · Body) ===")
        for speaker_id, data in video_summary["per_speaker"].items():
            lines.append(f"\n[{_display(speaker_id, name_map)}]")
            if data.get("dominant_emotion"):
                conf = data.get("dominant_emotion_confidence", 0)
                lines.append(
                    f"  Dominant facial emotion: {data['dominant_emotion']} "
                    f"(confidence={conf:.2f})"
                )
            if data.get("dominant_valence"):
                lines.append(f"  Valence/arousal: {data['dominant_valence']}")
            if data.get("avg_facial_stress") is not None:
                lines.append(f"  Avg facial stress: {data['avg_facial_stress']:.3f}")
            if data.get("avg_facial_engagement") is not None:
                lines.append(f"  Avg facial engagement: {data['avg_facial_engagement']:.3f}")
            if data.get("avg_gaze_on_screen_pct") is not None:
                lines.append(f"  Gaze on screen: {data['avg_gaze_on_screen_pct']:.1%}")
            gaze_breaks = data.get("gaze_breaks", 0)
            if gaze_breaks:
                lines.append(f"  Gaze breaks (distraction events): {gaze_breaks}")
            blink = data.get("blink_anomalies", 0)
            if blink:
                lines.append(f"  Blink rate anomalies: {blink}")
            nods  = data.get("head_nods", 0)
            shakes = data.get("head_shakes", 0)
            if nods or shakes:
                lines.append(f"  Head gestures: {nods} nods, {shakes} shakes")
            if data.get("avg_body_movement") is not None:
                lines.append(f"  Avg body movement: {data['avg_body_movement']:.3f}")
            posture = data.get("posture_changes", 0)
            if posture:
                lines.append(f"  Posture shifts: {posture}")
            hnf = data.get("hand_near_face_events", 0)
            if hnf:
                lines.append(f"  Hand-near-face events (self-soothing): {hnf}")

    # Language summary per speaker
    lines.append("\n=== LANGUAGE ANALYSIS ===")
    for speaker_id, data in language_summary.get("per_speaker", {}).items():
        lines.append(f"\n[{_display(speaker_id, name_map)}]")
        lines.append(f"  Segments analysed: {data.get('total_segments', 0)}")
        lines.append(f"  Average sentiment: {data.get('avg_sentiment', 0):.3f}")
        lines.append(f"  Sentiment range: {data.get('min_sentiment', 0):.3f} to {data.get('max_sentiment', 0):.3f}")
        lines.append(f"  Buying signals: {data.get('buying_signal_count', 0)}")
        lines.append(f"  Objections: {data.get('objection_count', 0)}")
        lines.append(f"  Power score: {data.get('avg_power_score', 0.5):.3f}")
        intent_dist = data.get("intent_distribution", {})
        if intent_dist:
            lines.append(f"  Intents: {json.dumps(intent_dist)}")

    # Buying signal moments
    buy_moments = language_summary.get("buying_signal_moments", [])
    if buy_moments:
        lines.append("\nBUYING SIGNAL MOMENTS:")
        for m in buy_moments[:5]:
            time_s = m.get('time_ms', 0) / 1000
            lines.append(
                f"  {time_s:.0f}s — {_display(m.get('speaker', '?'), name_map)}: "
                f"strength={m.get('strength', 0):.3f}, "
                f"categories={m.get('categories', [])}"
            )

    # Objection moments
    obj_moments = language_summary.get("objection_moments", [])
    if obj_moments:
        lines.append("\nOBJECTION MOMENTS:")
        for m in obj_moments[:5]:
            time_s = m.get('time_ms', 0) / 1000
            lines.append(
                f"  {time_s:.0f}s — {_display(m.get('speaker', '?'), name_map)}: "
                f"strength={m.get('strength', 0):.3f}, "
                f"categories={m.get('categories', [])}"
            )

    # Conversation dynamics
    conversation_summary = conversation_summary or {}
    if conversation_summary:
        lines.append("\n=== CONVERSATION DYNAMICS ===")
        turn_taking = conversation_summary.get("turn_taking", {})
        if turn_taking:
            lines.append(f"  Turn count: {turn_taking.get('total_turns', 0)}")
            lines.append(f"  Avg turn duration: {turn_taking.get('avg_turn_duration_ms', 0):.0f} ms")
            lines.append(f"  Turn rate: {turn_taking.get('turns_per_minute', 0):.1f} turns/min")
        rapport = conversation_summary.get("rapport", {})
        if rapport:
            lines.append(f"  Rapport score: {rapport.get('score', 0):.2f}")
            lines.append(f"  Rapport level: {rapport.get('level', 'unknown')}")
        dominance = conversation_summary.get("dominance", {})
        if dominance:
            lines.append(f"  Dominance index: {dominance.get('index', 0):.2f}")
            for sid, pct in dominance.get("per_speaker", {}).items():
                lines.append(f"    {_display(sid, name_map)}: {pct:.1f}% talk time")
        interruptions = conversation_summary.get("interruptions", {})
        if interruptions:
            lines.append(f"  Total interruptions: {interruptions.get('total', 0)}")
            for sid, cnt in interruptions.get("per_speaker", {}).items():
                lines.append(f"    {_display(sid, name_map)}: {cnt} interruptions")
        response_latency = conversation_summary.get("response_latency", {})
        if response_latency:
            lines.append(f"  Avg response latency: {response_latency.get('avg_ms', 0):.0f} ms")

    # ── Interrogation-specific signal block ──────────────────────────────────
    _INTERROG_TYPES = {
        "false_confession_risk", "denial_weakening", "interrogator_technique",
        "statement_contamination", "capitulation_cascade", "resistance_hardening",
        "freezing_response", "blink_suppression_spike", "motor_inhibition",
        "evidence_response_processing_delay", "narrative_consistency_drift",
    }
    interrog_signals = [s for s in fusion_signals if s.get("signal_type") in _INTERROG_TYPES]
    if interrog_signals:
        lines.append("\n=== INTERROGATION ANALYSIS ===")

        risk_sig = next((s for s in interrog_signals if s["signal_type"] == "false_confession_risk"), None)
        if risk_sig:
            meta = risk_sig.get("metadata") or {}
            lines.append(f"\nFALSE CONFESSION RISK: {risk_sig.get('value_text', '?')} "
                         f"(score={risk_sig.get('value', 0):.2f}, "
                         f"confidence={risk_sig.get('confidence', 0):.2f})")
            factors = {k: meta.get(k) for k in
                       ("contamination", "capitulation", "denial_drop", "duration",
                        "coercive_technique", "processing_delays") if meta.get(k)}
            if factors:
                lines.append(f"  Risk factors present: {', '.join(factors.keys())}")
            if meta.get("duration_minutes"):
                lines.append(f"  Session duration: {meta['duration_minutes']:.0f} min")

        denial_sig = next((s for s in interrog_signals if s["signal_type"] == "denial_weakening"), None)
        if denial_sig:
            meta = denial_sig.get("metadata") or {}
            lines.append(f"\nDENIAL TRAJECTORY: {denial_sig.get('value_text', '?')} "
                         f"(trigger={meta.get('trigger', '?')})")
            if meta.get("first_label") and meta.get("last_label"):
                lines.append(f"  {meta['first_label']} → {meta['last_label']} "
                             f"over {meta.get('denial_count', 0)} denial statements")
            if meta.get("windowed_drop"):
                lines.append(f"  Strength drop: {meta['windowed_drop'] * 100:.0f}%")

        tech_sig = next((s for s in interrog_signals if s["signal_type"] == "interrogator_technique"), None)
        if tech_sig:
            meta = tech_sig.get("metadata") or {}
            lines.append(f"\nINTERROGATOR TECHNIQUE: {tech_sig.get('value_text', '?').upper()} "
                         f"(confidence={tech_sig.get('confidence', 0):.2f})")
            if meta.get("peace_count") or meta.get("reid_count") or meta.get("coercive_count"):
                lines.append(f"  PEACE markers: {meta.get('peace_count', 0)}, "
                             f"Reid markers: {meta.get('reid_count', 0)}, "
                             f"Coercive markers: {meta.get('coercive_count', 0)}")

        contam_sigs = [s for s in interrog_signals if s["signal_type"] == "statement_contamination"]
        if contam_sigs:
            all_terms: list[str] = []
            for s in contam_sigs:
                all_terms.extend((s.get("metadata") or {}).get("contaminated_terms", []))
            unique_terms = list(dict.fromkeys(all_terms))[:10]
            lines.append(f"\nSTATEMENT CONTAMINATION: {len(contam_sigs)} event(s)")
            if unique_terms:
                lines.append(f"  Adopted terms: {', '.join(unique_terms)}")

        cap_sigs = [s for s in interrog_signals if s["signal_type"] == "capitulation_cascade"]
        if cap_sigs:
            lines.append(f"\nCAPITULATION CASCADE: {len(cap_sigs)} event(s)")
            for s in cap_sigs[:3]:
                start_s = s.get("window_start_ms", 0) // 1000
                end_s = s.get("window_end_ms", 0) // 1000
                lines.append(f"  {start_s}s–{end_s}s: {s.get('value_text', '')}")

        behav = [s for s in interrog_signals if s["signal_type"] in
                 ("freezing_response", "blink_suppression_spike", "motor_inhibition",
                  "evidence_response_processing_delay")]
        if behav:
            lines.append(f"\nBEHAVIOURAL INDICATORS ({len(behav)} events):")
            for s in behav[:5]:
                start_s = s.get("window_start_ms", 0) // 1000
                lines.append(f"  {start_s}s — {s['signal_type']} "
                             f"(value={s.get('value', 0):.2f}, "
                             f"conf={s.get('confidence', 0):.2f})")

    # Fusion signals (non-interrogation)
    non_interrog = [s for s in fusion_signals if s.get("signal_type") not in _INTERROG_TYPES] \
        if interrog_signals else fusion_signals
    if non_interrog:
        lines.append("\n=== CROSS-MODAL FUSION INSIGHTS ===")
        for s in non_interrog[:10]:
            lines.append(
                f"  {s.get('signal_type', '?')}: "
                f"{s.get('value_text', '')} "
                f"(confidence={s.get('confidence', 0):.2f})"
            )

    # Unified states
    if unified_states:
        lines.append("\n=== FINAL SPEAKER STATES ===")
        for state in unified_states:
            sid = state.get("speaker_id", "?")
            lines.append(f"\n[{_display(sid, name_map)}]")
            lines.append(f"  Engagement: {state.get('engagement', 0.5):.2f}")
            lines.append(f"  Confidence: {state.get('confidence_level', 0.5):.2f}")
            lines.append(f"  Stress: {state.get('stress_level', 0):.2f}")
            lines.append(f"  Sentiment: {state.get('sentiment_score', 0):.2f}")
            lines.append(f"  Authenticity: {state.get('authenticity_score', 1.0):.2f}")
            alerts = state.get("active_alerts", [])
            if alerts:
                lines.append(f"  Active alerts: {json.dumps(alerts)}")

    # Entities (from Language Agent entity extraction)
    if entities:
        lines.append("\n=== EXTRACTED ENTITIES ===")

        people = entities.get("people", [])
        if people:
            lines.append("PEOPLE:")
            for p in people:
                role = p.get("role", "unknown")
                label = p.get("speaker_label", "")
                lines.append(f"  {p.get('name', '?')} — role: {role}, speaker: {label}")

        topics = entities.get("topics", [])
        if topics:
            lines.append("CONVERSATION PHASES (use these as 'topic' headings in the 'notes' output):")
            for t in topics:
                s_m, s_s = divmod(t.get("start_ms", 0) // 1000, 60)
                e_m, e_s = divmod(t.get("end_ms", 0) // 1000, 60)
                lines.append(f"  {t.get('name', 'Unknown')} ({s_m}:{s_s:02d}–{e_m}:{e_s:02d})")
                if t.get("summary"):
                    lines.append(f"    {t['summary']}")

        objections = entities.get("objections", [])
        if objections:
            lines.append("OBJECTIONS:")
            for o in objections:
                ts = o.get("timestamp_ms", 0) // 1000
                status = "RESOLVED" if o.get("resolved") else "UNRESOLVED"
                spk = _display(o.get("speaker", ""), name_map) if o.get("speaker") else ""
                spk_str = f" by {spk}" if spk else ""
                lines.append(f'  [{ts}s]{spk_str} "{o.get("text", "")}" — {status}')

        commitments = entities.get("commitments", [])
        if commitments:
            lines.append("COMMITMENTS:")
            for c in commitments:
                ts = c.get("timestamp_ms", 0) // 1000
                spk = _display(c.get("speaker", "?"), name_map)
                lines.append(f'  [{ts}s] {spk}: "{c.get("text", "")}"')

        key_terms = entities.get("key_terms", [])
        if key_terms:
            lines.append(f"KEY TERMS: {', '.join(key_terms)}")

    # Graph analytics
    if graph_analytics:
        lines.append("\n=== GRAPH ANALYTICS ===")

        clusters = graph_analytics.get("tension_clusters", [])
        if clusters:
            lines.append(f"\nTENSION CLUSTERS ({len(clusters)} detected):")
            for c in clusters[:5]:
                lines.append(
                    f"  {c['timestamp_ms']//1000}s — {c['signal_count']} signals, "
                    f"severity={c['severity']}, speaker={_display(c['speaker_id'], name_map)}"
                )

        topics_density = graph_analytics.get("topic_signal_density", [])
        if topics_density:
            lines.append("\nTOPIC ANALYSIS:")
            for t in topics_density:
                lines.append(
                    f"  {t['topic_name']}: risk={t['risk_level']}, "
                    f"opportunity={t['opportunity_level']}, "
                    f"{t['total_signals']} signals"
                )

        momentum = graph_analytics.get("momentum", {})
        if momentum:
            lines.append(
                f"\nMOMENTUM: trajectory={momentum.get('overall_trajectory')}, "
                f"score={momentum.get('momentum_score', 0):.2f}"
            )
            if momentum.get("turning_point_ms"):
                lines.append(f"  Turning point at {momentum['turning_point_ms']//1000}s")

        patterns = graph_analytics.get("speaker_patterns", {})
        if patterns:
            lines.append("\nSPEAKER PATTERNS:")
            for sid, p in patterns.items():
                lines.append(
                    f"  {_display(sid, name_map)}: {p.get('response_pattern', 'unknown')} pattern, "
                    f"escalation={p.get('escalation_trend', 'stable')}, "
                    f"contradiction_ratio={p.get('contradiction_ratio', 0):.1%}"
                )

        resolutions = graph_analytics.get("resolution_paths", [])
        if resolutions:
            lines.append("\nOBJECTION RESOLUTION PATHS:")
            for r in resolutions:
                lines.append(
                    f"  Objection at {r['objection_ms']//1000}s → resolved at "
                    f"{r['resolution_ms']//1000}s "
                    f"({r['time_to_resolve_ms']//1000}s to resolve)"
                )

    # ── Synthesized per-speaker cross-modal profile ──────────────────────────
    lines.append("\n=== CROSS-MODAL SPEAKER PROFILES ===")
    lines.append("(Pre-computed synthesis — use these for speaker_analyses in your output)")

    for speaker_id in speakers:
        display = _display(speaker_id, name_map)
        lines.append(f"\n[{display}]")

        vs = voice_summary.get("per_speaker", {}).get(speaker_id, {})
        if vs:
            lines.append(
                f"  VOICE: stress_avg={vs.get('avg_stress', 0):.2f}, "
                f"stress_max={vs.get('max_stress', 0):.2f}, "
                f"fillers={vs.get('total_fillers', 0)}, "
                f"pitch_events={vs.get('pitch_elevation_events', 0)}"
            )

        ls = language_summary.get("per_speaker", {}).get(speaker_id, {})
        if ls:
            lines.append(
                f"  LANGUAGE: sentiment_avg={ls.get('avg_sentiment', 0):.2f}, "
                f"power={ls.get('avg_power_score', 0.5):.2f}, "
                f"buying_signals={ls.get('buying_signal_count', 0)}, "
                f"objections={ls.get('objection_count', 0)}"
            )

        vid = (video_summary or {}).get("per_speaker", {}).get(speaker_id, {})
        if vid:
            lines.append(
                f"  FACE: emotion={vid.get('dominant_emotion', '?')}, "
                f"stress={(vid.get('avg_facial_stress') or 0):.2f}, "
                f"engagement={(vid.get('avg_facial_engagement') or 0):.2f}"
            )
            lines.append(
                f"  GAZE: on_screen={(vid.get('avg_gaze_on_screen_pct') or 0):.0%}, "
                f"breaks={vid.get('gaze_breaks', 0)}, "
                f"blink_anomalies={vid.get('blink_anomalies', 0)}"
            )
            lines.append(
                f"  BODY: movement={(vid.get('avg_body_movement') or 0):.2f}, "
                f"self_touch={vid.get('hand_near_face_events', 0)}, "
                f"nods={vid.get('head_nods', 0)}, "
                f"shakes={vid.get('head_shakes', 0)}"
            )

        cs = conversation_summary or {}
        dom_pct = cs.get("dominance", {}).get("per_speaker", {}).get(speaker_id, 0)
        interrupts = cs.get("interruptions", {}).get("per_speaker", {}).get(speaker_id, 0)
        if dom_pct:
            lines.append(f"  CONVERSATION: talk_time={dom_pct:.1f}%, interruptions={interrupts}")

        # Pre-compute cross-modal incongruence flag
        if vs and vid:
            voice_stress = vs.get("avg_stress", 0)
            face_stress = vid.get("avg_facial_stress") or 0
            delta = abs(voice_stress - face_stress)
            if delta > 0.25:
                higher = "voice" if voice_stress > face_stress else "face"
                lower = "face" if higher == "voice" else "voice"
                lines.append(
                    f"  ⚡ INCONGRUENCE: {higher} stress "
                    f"({max(voice_stress, face_stress):.2f}) vs {lower} "
                    f"({min(voice_stress, face_stress):.2f}) — delta {delta:.2f}"
                )

    return "\n".join(lines)


def _build_prompt(context: str, meeting_type: str) -> tuple[str, str]:
    """Build system + user prompts for narrative generation."""
    type_instructions = {
        # ── Sales-specific ──
        "sales_call": (
            "Focus on: buying signals timeline, objection handling effectiveness, "
            "stress points during pricing/negotiation, decision readiness indicators, "
            "close probability assessment, and whether the prospect showed genuine "
            "or manufactured interest. Identify the seller and buyer roles. "
            "KEY FACTS: List commitments and objections FIRST with speaker, timestamp, and status. "
            "SPEAKER ANALYSIS: Profile the seller's persuasion effectiveness and the prospect's genuine "
            "interest level — distinguish spoken interest from behavioral engagement. "
            "CROSS-MODAL INSIGHTS: Focus on prospect incongruence — verbal agreement with disengaged body "
            "language, nodding with avoidant gaze, positive words with stressed voice. These reveal unspoken objections. "
            "OUTPUT STRUCTURE: Include 'deal_assessment' with close_probability (high/medium/low), "
            "stage_reached, buying_signals count, unresolved_objections count, "
            "prospect_engagement_trend (increasing/stable/declining). "
            "Include 'objection_handling' array (each with 'objection', 'timestamp', "
            "'handling_quality', 'resolved' boolean, 'prospect_reaction'). "
            "If video data is present: note forward lean (engagement), gaze breaks (distraction), "
            "facial stress peaks, head nods (agreement) or shakes (disagreement), "
            "and any incongruence between spoken words and body language."
        ),
        # ── Podcast ──
        "podcast": (
            "Focus on: speaker dynamics and chemistry, topic flow and transitions, "
            "engagement peaks (where energy or interest spikes), most interesting "
            "moments based on vocal/linguistic indicators, host vs guest balance, "
            "and audience engagement proxy from speaker energy patterns. "
            "If video data is present: note facial emotion shifts, posture changes, "
            "and non-verbal rapport signals like head nods and mirroring."
        ),
        # ── Interview ──
        "interview": (
            "Focus on: candidate confidence trajectory over time, question quality "
            "from the interviewer, rapport assessment between participants, "
            "stress indicators during difficult questions, authenticity markers, "
            "and communication style adaptability. "
            "If video data is present: facial stress during tough questions, gaze avoidance, "
            "self-touch (pacifying gestures), posture shifts, and whether body language "
            "is congruent with spoken confidence."
        ),
        # ── Lecture ──
        "lecture": (
            "Focus on: speaker clarity and pacing analysis, energy level changes "
            "over time, engagement indicators (if Q&A present), key emphasis moments, "
            "vocal fatigue patterns, and whether the speaker maintained audience "
            "attention through vocal variety. "
            "If video data is present: facial engagement level, gestural animation, "
            "gaze contact with audience, and posture confidence."
        ),
        # ── Debate ──
        "debate": (
            "Focus on: argument strength per speaker, dominance shifts over time, "
            "emotional escalation patterns, persuasion technique effectiveness, "
            "stress responses to challenges, and which speaker demonstrated "
            "more confidence and control. "
            "If video data is present: facial anger or contempt signals, forward lean "
            "during challenges, head shakes, and postural dominance indicators."
        ),
        # ── Meeting ──
        "meeting": (
            "Focus on: participation balance across all speakers, decision points "
            "and consensus moments, action item indicators, engagement distribution, "
            "stress or tension moments, dominant vs passive speakers, and "
            "overall meeting productivity indicators. "
            "If video data is present: engagement levels per participant (gaze, posture, "
            "head nods), disengagement signals (gaze breaks, backward lean), and "
            "non-verbal agreement or disagreement patterns."
        ),
        # ── Presentation ──
        "presentation": (
            "Focus on: speaker clarity and confidence, persuasion effectiveness, "
            "audience engagement indicators, key message delivery moments, "
            "pacing and energy management, and vocal conviction markers. "
            "If video data is present: facial engagement, animated gestures, "
            "upright power posture, and eye contact with audience."
        ),
        # ── Casual conversation ──
        "casual_conversation": (
            "Focus on: rapport and connection quality, engagement levels, "
            "emotional dynamics, turn-taking balance, and overall "
            "communication chemistry between participants. "
            "If video data is present: shared smiles, mirroring signals, "
            "head nod synchrony, and facial warmth indicators."
        ),
        # ── Interrogation ──
        "interrogation_video": (
            "This is a law enforcement interrogation session. Focus on: "
            "false confession risk level and contributing factors (contamination, denial weakening, "
            "session duration, coercive technique); "
            "denial trajectory — how denial strength changed from session start to end; "
            "interrogation technique classification (PEACE / Reid / Coercive) and its implications; "
            "statement contamination events (case-specific terms the suspect adopted from the interrogator); "
            "capitulation cascade patterns (multi-signal compliance); "
            "behavioural indicators (freezing response, blink suppression, motor inhibition, "
            "evidence-response processing delays). "
            "SPEAKER ANALYSIS: Profile the suspect's behavioral trajectory across the session — "
            "how did their stress, denial strength, body language, and speech patterns evolve? "
            "Profile the interrogator's technique shifts and their effect on the suspect. "
            "CROSS-MODAL INSIGHTS: Focus on moments where voice/face/body signals diverge — "
            "stress peaks without verbal acknowledgment, calm speech during facial distress, "
            "body freezing after specific questions. "
            "OUTPUT STRUCTURE: Include 'risk_assessment' with false_confession_risk level (low/low_moderate/"
            "moderate/elevated/high), risk_score (0.0-1.0), contributing_factors array (each with 'factor', "
            "'present' boolean, 'detail' string) for: contamination, session_duration, coercive_technique, "
            "denial_weakening, processing_delays, capitulation_pattern. Include 'ethical_note' string. "
            "Include 'contamination_timeline' array (each with 'term', 'interrogator_first', "
            "'suspect_adopted', 'context'). Include 'technique_analysis' with 'primary', 'peace_markers', "
            "'reid_markers', 'coercive_markers', 'assessment'. "
            "CRITICAL ETHICAL CONSTRAINTS: Never claim guilt or deception. All findings are "
            "probabilistic indicators only, never binary determinations. "
            "Distinguish between genuine recall difficulty and stress-induced compliance. "
            "Frame every risk indicator with its alternative innocent explanation. "
            "Use 'incongruence', 'stress indicator', 'risk factor' — never 'deception' or 'lying'. "
            "If coercive technique detected, flag elevated false confession risk regardless of other signals. "
            "Research basis: Kassin et al. (2010, 2012), Garrett (2011), Vrij (2005, 2008)."
        ),
        # ── Legacy types ──
        "client_meeting": (
            "Focus on: rapport trajectory, client satisfaction indicators, "
            "risk flags, unspoken concerns detected through cross-modal analysis. "
            "KEY FACTS: List commitments and action items FIRST with speaker, timestamp, and status. "
            "If video data is present: note any body language incongruence with stated satisfaction."
        ),
        "internal": (
            "Focus on: engagement distribution across participants, "
            "contribution balance, stress patterns, and consensus indicators. "
            "If video data is present: note engagement vs disengagement signals per speaker."
        ),
    }

    # Default: general communication analysis
    default_focus = (
        "Focus on: overall communication dynamics, speaker profiles, "
        "key moments, engagement patterns, stress indicators, "
        "and notable cross-modal patterns."
    )

    focus = type_instructions.get(meeting_type, default_focus)

    system_prompt = (
        "You are the NEXUS Behavioural Analysis System producing a post-session analysis report. "
        "You analyse SIX modalities simultaneously: Voice (stress, tone, pace, fillers), "
        "Language (sentiment, buying signals, objections, power), "
        "Facial Expression (emotion, engagement, stress blendshapes), "
        "Body Language (posture, lean, gestures, self-touch, head nods/shakes), "
        "Gaze (screen contact, gaze breaks, blink rate), and "
        "Conversation Dynamics (turn-taking, dominance, rapport). "
        "You produce PROBABILISTIC INDICATORS, never binary determinations. "
        "Never claim to detect deception — only flag incongruence for human review. "
        "Maximum certainty for any claim is 'likely' or 'suggests' — never 'definitely'. "
        "Cross-modal insights are the most valuable findings — especially when voice, language, "
        "face, body, or gaze signals DISAGREE with each other (incongruence reveals hidden state). "
        "When VIDEO ANALYSIS data is present (facial, gaze, body), you MUST incorporate it into "
        "the executive summary, key moments, and cross-modal insights. Do not ignore body language "
        "or facial expression data even if voice/language data is richer. "
        "IMPORTANT: Always use real names (e.g. 'John', 'Sarah') instead of Speaker_X labels "
        "wherever a name is available from the SPEAKER NAME MAP. Only use Speaker_X when no name is known. "
        "STRUCTURE REQUIREMENTS: "
        "1. general_summary: 5-7 bullet points covering the key themes, decisions, and outcomes. "
        "Each bullet is one standalone sentence. No paragraphs. Scannable. "
        "2. notes: Group discussion details by TOPIC or PHASE. Use the CONVERSATION PHASES from the "
        "context data as topic headings. Under each topic, list speaker-attributed details with timestamps. "
        "Each detail can have sub_details for supporting points. This is the detailed record of what was "
        "discussed — the most comprehensive section. "
        "3. action_items: List every commitment, task, or follow-up mentioned. Group by assignee "
        "(the person responsible). Include deadline if mentioned, timestamp of when it was agreed, "
        "and brief context. Extract from both explicit commitments ('I will send the report') and "
        "implicit ones ('We should schedule a follow-up' — assign to whoever said it)."
    )

    # ── Content-type isolation instructions ──────────────────────────────────
    if meeting_type == "interrogation_video":
        extra_fields = (
            "REQUIRED for this interrogation session: include 'risk_assessment', "
            "'contamination_timeline', and 'technique_analysis' fields as described in FOCUS. "
            "Do NOT include 'deal_assessment' or 'objection_handling'."
        )
    elif meeting_type in ("sales_call", "client_meeting"):
        extra_fields = (
            "REQUIRED for this sales/client session: include 'deal_assessment' and "
            "'objection_handling' fields as described in FOCUS. "
            "Do NOT include 'risk_assessment', 'contamination_timeline', or 'technique_analysis'."
        )
    else:
        extra_fields = (
            "Do NOT include 'risk_assessment', 'contamination_timeline', 'technique_analysis', "
            "'deal_assessment', or 'objection_handling'. These are type-specific fields not "
            "applicable to this session type."
        )

    user_prompt = f"""Analyse the following multi-modal signal data from a recorded meeting and produce a structured report.

FOCUS: {focus}

{context}

Respond with a JSON object containing these fields:
{{
  "general_summary": [
    "Key theme or outcome as a single bullet point",
    "Another key theme — one per major topic discussed",
    "Decision or agreement reached",
    "Timeline or deadline established",
    "Technical or operational update if relevant"
  ],

  "notes": [
    {{
      "topic": "Topic or Phase Name",
      "summary": "One sentence describing what was discussed in this topic area",
      "details": [
        {{
          "speaker": "Person Name or Speaker_0",
          "text": "What they said or contributed — paraphrased or key quote",
          "timestamp": "MM:SS",
          "sub_details": [
            "Supporting point or context",
            "Another supporting detail"
          ]
        }}
      ]
    }}
  ],

  "action_items": [
    {{
      "assignee": "Person Name or Speaker_0",
      "task": "Specific task or commitment",
      "deadline": "By Friday afternoon",
      "timestamp": "MM:SS",
      "context": "Brief reason or origin of this action"
    }}
  ],

  "executive_summary": "Comprehensive 8-10 sentence narrative paragraph written as a professional behavioural analyst report. Cover ALL of the following in flowing prose (not bullet points): (1) the session purpose and overall outcome, (2) the most significant behavioural pattern per speaker — stress, hesitation, confidence, or engagement, (3) language dynamics — sentiment arc, persuasion attempts, objections, commitments, (4) cross-modal incongruences or alignments that reveal hidden state — especially where voice, face, body, or gaze DISAGREE, (5) emotional dynamics and rapport between participants, (6) any risks, concerns, or unresolved tensions. Be specific — name speakers, cite signal types, quote timestamps. Do not repeat bullet-point facts from general_summary; synthesise them into analytical prose.",

  "key_facts": [
    {{
      "type": "commitment | objection | decision | admission | disclosure",
      "speaker": "speaker name or label",
      "text": "exact quote or paraphrase",
      "timestamp": "MM:SS",
      "status": "confirmed | unresolved | retracted"
    }}
  ],

  "speaker_analyses": {{
    "Speaker_0": {{
      "role": "role if known",
      "behavioral_profile": "2-3 sentence narrative synthesising voice + language + video",
      "voice_patterns": "one sentence on vocal stress, pace, fillers",
      "body_language": "one sentence on posture, gestures, gaze (omit if no video data)",
      "key_moments": ["timestamp — description of notable moment"]
    }}
  }},

  "key_moments": [
    {{
      "time_description": "timestamp or range",
      "description": "what happened",
      "significance": "why it matters",
      "signals_involved": ["signal_type_1", "signal_type_2"]
    }}
  ],

  "cross_modal_insights": [
    {{
      "insight": "full description of the cross-modal pattern",
      "modalities": ["voice", "face"],
      "type": "incongruence | synchrony | escalation | suppression",
      "significance": "what this reveals about the speaker's state"
    }}
  ],

  "recommendations": [
    {{
      "priority": "high | medium | low",
      "action": "specific actionable recommendation",
      "rationale": "brief reason based on the evidence"
    }}
  ]
}}

{extra_fields}

Return ONLY the JSON object, no other text."""

    return system_prompt, user_prompt


def _parse_narrative_response(raw_text: str, speakers: list[str]) -> dict:
    """Parse the LLM response into structured report."""

    def _extract_json(text: str) -> dict:
        """Try multiple strategies to extract a JSON object from text."""
        text = text.strip()

        # Strategy 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown code fences anywhere in text
        if "```" in text:
            start = text.find("```")
            end = text.rfind("```")
            if start != end:
                inner = text[start:end + 3]
                inner = inner.split("\n", 1)[1] if "\n" in inner else inner[3:]
                inner = inner.rsplit("```", 1)[0].strip()
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass

        # Strategy 3: find outermost { ... } — handles preamble or trailing text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found", text, 0)

    try:
        parsed = _extract_json(raw_text)
        return {
            # Meeting notes fields (all content types)
            "general_summary": parsed.get("general_summary", []),
            "notes": parsed.get("notes", []),
            "action_items": parsed.get("action_items", []),
            # Base fields (all content types)
            "executive_summary": parsed.get("executive_summary", ""),
            "key_facts": parsed.get("key_facts", []),
            "speaker_analyses": parsed.get("speaker_analyses", {}),
            "key_moments": parsed.get("key_moments", []),
            "cross_modal_insights": parsed.get("cross_modal_insights", []),
            "recommendations": parsed.get("recommendations", []),
            # Interrogation-specific (None when absent — frontend guards on contentType)
            "risk_assessment": parsed.get("risk_assessment"),
            "contamination_timeline": parsed.get("contamination_timeline"),
            "technique_analysis": parsed.get("technique_analysis"),
            # Sales-specific
            "deal_assessment": parsed.get("deal_assessment"),
            "objection_handling": parsed.get("objection_handling"),
        }
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM narrative as JSON, returning raw text")
        return {
            "general_summary": [],
            "notes": [],
            "action_items": [],
            "executive_summary": raw_text.strip()[:500],
            "key_facts": [],
            "speaker_analyses": {},
            "key_moments": [],
            "cross_modal_insights": [],
            "recommendations": [],
            "risk_assessment": None,
            "contamination_timeline": None,
            "technique_analysis": None,
            "deal_assessment": None,
            "objection_handling": None,
        }


def _fallback_narrative(
    speakers: list[str],
    voice_summary: dict,
    language_summary: dict,
    fusion_signals: list[dict],
    entities: Optional[dict] = None,
    transcript_segments: Optional[list[dict]] = None,
    video_summary: Optional[dict] = None,
    conversation_summary: Optional[dict] = None,
) -> dict:
    """
    Generate a basic narrative without LLM API.
    Used when API key is not configured or API call fails.
    """
    entities = entities or {}
    insights = []

    # Entity-based insights (richer than signal counts)
    for obj in entities.get("objections", []):
        status = "resolved" if obj.get("resolved") else "unresolved"
        insights.append(f'Objection ({status}): "{obj.get("text", "")}"')
    for com in entities.get("commitments", []):
        insights.append(f'{com.get("speaker", "?")}: "{com.get("text", "")}"')

    # Extract basic insights from summaries
    for speaker_id, data in language_summary.get("per_speaker", {}).items():
        if data.get("buying_signal_count", 0) > 0:
            insights.append(
                f"{speaker_id} showed {data['buying_signal_count']} buying signal(s)"
            )
        if data.get("objection_count", 0) > 0 and not entities.get("objections"):
            insights.append(
                f"{speaker_id} raised {data['objection_count']} objection(s)"
            )
        power = data.get("avg_power_score", 0.5)
        if power < 0.35:
            insights.append(
                f"{speaker_id} used notably powerless/hedging language (score: {power:.2f})"
            )

    for speaker_id, data in voice_summary.get("per_speaker", {}).items():
        stress = data.get("max_stress", 0)
        if stress > 0.60:
            insights.append(
                f"{speaker_id} reached high vocal stress ({stress:.2f}) during the session"
            )

    cross_modal = []
    for s in fusion_signals:
        sig_type = s.get("signal_type", "")
        meta = s.get("metadata") or {}
        if sig_type == "false_confession_risk":
            score = s.get("value", 0)
            factors = [k for k in ("contamination", "capitulation", "denial_drop",
                                   "duration", "coercive_technique") if meta.get(k)]
            label = s.get("value_text", "").replace("_", " ")
            cross_modal.append(
                f"False confession risk: {label} ({score:.0%})"
                + (f" — factors: {', '.join(factors)}" if factors else "")
            )
        elif sig_type == "denial_weakening":
            first = meta.get("first_label", "strong")
            last = meta.get("last_label", "weak")
            cross_modal.append(
                f"Denial trajectory weakening: {first} → {last} "
                f"over {meta.get('denial_count', '?')} denial statements"
            )
        elif sig_type == "interrogator_technique":
            technique = s.get("value_text", "unknown").upper()
            cross_modal.append(f"Interrogator technique classified as: {technique}")
            if technique in ("REID", "COERCIVE"):
                cross_modal.append(
                    f"{technique} technique detected — elevates false confession risk "
                    "per Kassin et al. (2012)"
                )
        elif sig_type == "statement_contamination":
            terms = meta.get("contaminated_terms", [])
            cross_modal.append(
                f"Statement contamination: suspect adopted "
                f"{meta.get('contaminated_count', len(terms))} case-specific term(s) "
                f"from interrogator"
                + (f": {', '.join(terms[:5])}" if terms else "")
            )
        elif sig_type == "capitulation_cascade":
            cross_modal.append(
                f"Capitulation cascade at "
                f"{s.get('window_start_ms', 0) // 1000}s–"
                f"{s.get('window_end_ms', 0) // 1000}s"
            )
        elif sig_type == "freezing_response":
            cross_modal.append(
                f"Freezing response at {s.get('window_start_ms', 0) // 1000}s — "
                "motor inhibition consistent with acute stress"
            )
        elif sig_type == "evidence_response_processing_delay":
            cross_modal.append(
                f"Extended processing delay at {s.get('window_start_ms', 0) // 1000}s "
                "following evidence presentation"
            )
        elif sig_type == "credibility_assessment" and s.get("value_text") == "credibility_concern":
            cross_modal.append(
                "Detected content-voice incongruence: positive words paired with elevated stress"
            )
        elif sig_type == "verbal_incongruence":
            cross_modal.append(
                "Detected verbal incongruence: positive sentiment paired with hedging language"
            )
        elif sig_type == "urgency_authenticity" and s.get("value_text") == "manufactured_urgency":
            cross_modal.append(
                "Detected potentially manufactured urgency: fast pace with stress indicators"
            )

    # ── Build key_facts from entities ────────────────────────────────────────
    key_facts: list[dict] = []
    for obj in entities.get("objections", []):
        ts_s = obj.get("timestamp_ms", 0) // 1000
        m, s = divmod(ts_s, 60)
        key_facts.append({
            "type": "objection",
            "speaker": obj.get("speaker", ""),
            "text": obj.get("text", ""),
            "timestamp": f"{m}:{s:02d}",
            "status": "confirmed" if obj.get("resolved") else "unresolved",
        })
    for com in entities.get("commitments", []):
        ts_s = com.get("timestamp_ms", 0) // 1000
        m, s = divmod(ts_s, 60)
        key_facts.append({
            "type": "commitment",
            "speaker": com.get("speaker", ""),
            "text": com.get("text", ""),
            "timestamp": f"{m}:{s:02d}",
            "status": "confirmed",
        })

    # ── Build speaker_analyses from per-speaker summaries ────────────────────
    speaker_analyses: dict[str, dict] = {}
    for speaker_id in speakers:
        vs = voice_summary.get("per_speaker", {}).get(speaker_id, {})
        ls = language_summary.get("per_speaker", {}).get(speaker_id, {})
        profile_parts = []
        if vs.get("max_stress", 0) > 0.60:
            profile_parts.append(
                f"Vocal stress peaked at {vs['max_stress']:.2f} during the session"
            )
        if ls.get("buying_signal_count", 0) > 0:
            profile_parts.append(
                f"showed {ls['buying_signal_count']} buying signal(s)"
            )
        if ls.get("avg_power_score", 0.5) < 0.35:
            profile_parts.append("used hedging/powerless language")
        speaker_analyses[speaker_id] = {
            "role": "",
            "behavioral_profile": ". ".join(profile_parts) if profile_parts
                else "No notable behavioral patterns detected.",
            "voice_patterns": (
                f"Avg stress {vs.get('avg_stress', 0):.2f}, "
                f"{vs.get('total_fillers', 0)} fillers, "
                f"{vs.get('pitch_elevation_events', 0)} pitch elevation events."
            ) if vs else "",
            "body_language": "",
            "key_moments": [],
        }

    # ── Build general_summary ─────────────────────────────────────────────────
    general_summary: list[str] = []
    if entities.get("commitments"):
        general_summary.append(
            f"{len(entities['commitments'])} commitment(s) made during the session."
        )
    if entities.get("objections"):
        resolved = sum(1 for o in entities["objections"] if o.get("resolved"))
        total = len(entities["objections"])
        general_summary.append(
            f"{total} objection(s) raised, {resolved} resolved."
        )
    for spk, data in voice_summary.get("per_speaker", {}).items():
        if data.get("max_stress", 0) > 0.60:
            general_summary.append(
                f"{spk} showed elevated stress during the session."
            )
    for spk, data in language_summary.get("per_speaker", {}).items():
        if data.get("buying_signal_count", 0) > 2:
            general_summary.append(
                f"{spk} showed {data['buying_signal_count']} buying signals."
            )
    for spk, data in (video_summary or {}).get("per_speaker", {}).items():
        emotion = data.get("dominant_emotion")
        if emotion and emotion not in ("neutral", ""):
            conf = data.get("dominant_emotion_confidence", 0)
            if conf >= 0.50:
                general_summary.append(
                    f"{spk} displayed predominantly {emotion} facial affect (confidence {conf:.0%})."
                )
    conv_rapport = (conversation_summary or {}).get("rapport_score")
    if conv_rapport is not None:
        label = "strong" if conv_rapport >= 0.65 else "moderate" if conv_rapport >= 0.40 else "low"
        general_summary.append(f"Overall conversational rapport was {label} ({conv_rapport:.2f}).")
    if not general_summary:
        general_summary.append(f"Session with {len(speakers)} speaker(s) completed.")

    # ── Build notes from topics + transcript segments ─────────────────────────
    notes: list[dict] = []
    segs = transcript_segments or []
    for topic in entities.get("topics", []):
        topic_start = topic.get("start_ms", 0)
        topic_end = topic.get("end_ms", 0)
        topic_details = []
        for seg in segs:
            if seg.get("start_ms", 0) >= topic_start and seg.get("end_ms", 0) <= topic_end:
                if len(seg.get("text", "")) > 30:
                    ts_s = seg["start_ms"] // 1000
                    m, s = divmod(ts_s, 60)
                    topic_details.append({
                        "speaker": seg.get("speaker", ""),
                        "text": seg["text"][:200],
                        "timestamp": f"{m}:{s:02d}",
                        "sub_details": [],
                    })
        if topic_details:
            notes.append({
                "topic": topic.get("name", "Discussion"),
                "summary": topic.get("summary", ""),
                "details": topic_details[:10],
            })

    # ── Build action_items from commitments and unresolved objections ─────────
    action_items: list[dict] = []
    for com in entities.get("commitments", []):
        ts_s = com.get("timestamp_ms", com.get("start_ms", 0)) // 1000
        m, s = divmod(ts_s, 60)
        action_items.append({
            "assignee": com.get("speaker", "Unknown"),
            "task": com.get("text", ""),
            "deadline": None,
            "timestamp": f"{m}:{s:02d}",
            "context": "",
        })
    for obj in entities.get("objections", []):
        if not obj.get("resolved"):
            ts_s = obj.get("timestamp_ms", obj.get("start_ms", 0)) // 1000
            m, s = divmod(ts_s, 60)
            action_items.append({
                "assignee": "",
                "task": f"Follow up on unresolved objection: {obj.get('text', '')}",
                "deadline": None,
                "timestamp": f"{m}:{s:02d}",
                "context": f"Raised by {obj.get('speaker', 'unknown')}",
            })

    # ── Build risk_assessment from false_confession_risk signal ──────────────
    risk_assessment = None
    risk_sig = next(
        (s for s in fusion_signals if s.get("signal_type") == "false_confession_risk"), None
    )
    if risk_sig:
        meta = risk_sig.get("metadata") or {}
        rf = meta.get("risk_factors", {})
        contributing_factors = [
            {"factor": "contamination",       "present": bool(rf.get("contamination", {}).get("signal_count", 0) > 0),       "detail": f"{rf.get('contamination', {}).get('signal_count', 0)} contamination event(s)"},
            {"factor": "session_duration",    "present": bool((rf.get("duration_risk", {}).get("contribution", 0)) > 0),       "detail": f"{meta.get('duration_minutes', 0):.0f} min session"},
            {"factor": "coercive_technique",  "present": bool(rf.get("coercive_technique", {}).get("signal_count", 0) > 0),   "detail": "coercive markers detected" if rf.get("coercive_technique", {}).get("signal_count", 0) > 0 else "no coercive markers"},
            {"factor": "denial_weakening",    "present": bool(rf.get("denial_evolution", {}).get("weakening_count", 0) > 0), "detail": f"{rf.get('denial_evolution', {}).get('weakening_count', 0)} weakening event(s)"},
            {"factor": "processing_delays",   "present": bool(rf.get("processing_delays", {}).get("signal_count", 0) > 0),   "detail": f"{rf.get('processing_delays', {}).get('signal_count', 0)} delay event(s)"},
            {"factor": "capitulation_pattern","present": bool(rf.get("capitulation_cascade", {}).get("signal_count", 0) > 0),"detail": f"{rf.get('capitulation_cascade', {}).get('signal_count', 0)} cascade event(s)"},
        ]
        risk_assessment = {
            "false_confession_risk": risk_sig.get("value_text", "unknown").replace("_", " "),
            "risk_score": round(risk_sig.get("value", 0), 3),
            "contributing_factors": contributing_factors,
            "ethical_note": (
                "Risk score indicates probability of coerced compliance, NOT guilt or deception. "
                "All factors have alternative innocent explanations."
            ),
        }

    # ── Build executive_summary as a flowing analyst paragraph ──────────────
    _exec: list[str] = []

    # Opening: participant count + tone
    _sentiments = [d.get("avg_sentiment", 0) for d in language_summary.get("per_speaker", {}).values()]
    _avg_sent = sum(_sentiments) / len(_sentiments) if _sentiments else 0
    _tone = "generally positive" if _avg_sent > 0.2 else "mixed" if _avg_sent > -0.1 else "predominantly negative"
    _exec.append(f"This session involved {len(speakers)} participant(s) and carried a {_tone} overall tone.")

    # Vocal stress
    _stressed = [spk for spk, d in voice_summary.get("per_speaker", {}).items() if d.get("max_stress", 0) > 0.60]
    _high_filler = [spk for spk, d in voice_summary.get("per_speaker", {}).items() if d.get("total_fillers", 0) > 5]
    if _stressed:
        _exec.append(
            f"Elevated vocal stress was detected in {', '.join(_stressed)}, "
            "suggesting cognitive load or emotional pressure at key moments."
        )
    if _high_filler:
        _exec.append(
            f"{', '.join(_high_filler)} exhibited high filler-word usage, "
            "indicating real-time reasoning or uncertainty during portions of the discussion."
        )

    # Language engagement signals
    _buyers = [
        (spk, d["buying_signal_count"])
        for spk, d in language_summary.get("per_speaker", {}).items()
        if d.get("buying_signal_count", 0) > 0
    ]
    if _buyers:
        _exec.append(
            f"Positive engagement signals were detected from "
            f"{', '.join(f'{s} ({c} signal(s))' for s, c in _buyers)}."
        )

    # Commitments and objections
    _n_commits = len((entities or {}).get("commitments", []))
    _n_objs = len((entities or {}).get("objections", []))
    _resolved = sum(1 for o in (entities or {}).get("objections", []) if o.get("resolved"))
    if _n_commits or _n_objs:
        _exec.append(
            f"{_n_commits} commitment(s) and {_n_objs} objection(s) "
            f"({_resolved} resolved) were recorded across the session."
        )

    # Video / facial affect
    _emotions = [
        (spk, d.get("dominant_emotion"))
        for spk, d in (video_summary or {}).get("per_speaker", {}).items()
        if d.get("dominant_emotion") and d.get("dominant_emotion") not in ("neutral", "")
        and d.get("dominant_emotion_confidence", 0) >= 0.50
    ]
    if _emotions:
        _exec.append(
            f"Facial expression analysis identified "
            f"{', '.join(f'{s} ({e})' for s, e in _emotions)} as dominant emotional states."
        )

    # Rapport
    _rapport = (conversation_summary or {}).get("rapport_score")
    if _rapport is not None:
        _rlabel = "strong" if _rapport >= 0.65 else "moderate" if _rapport >= 0.40 else "low"
        _exec.append(
            f"Conversational dynamics showed {_rlabel} rapport between participants "
            f"(score: {_rapport:.2f})."
        )

    # Cross-modal fusion
    _n_fusion = len(fusion_signals)
    if _n_fusion > 0 and cross_modal:
        _exec.append(
            f"{_n_fusion} cross-modal fusion event(s) were flagged; "
            f"most notably, {cross_modal[0].lower()}."
        )
        if len(cross_modal) > 1:
            _exec.append(cross_modal[1].rstrip(".") + ".")
    elif _n_fusion > 0:
        _exec.append(
            f"{_n_fusion} cross-modal signal(s) were detected spanning voice, language, "
            "and visual modalities."
        )
    elif not _exec:
        _exec.append("No notable cross-modal patterns were detected in this session.")

    executive_summary = " ".join(_exec)

    return {
        "general_summary": general_summary,
        "notes": notes,
        "action_items": action_items,
        "executive_summary": executive_summary,
        "key_facts": key_facts,
        "speaker_analyses": speaker_analyses,
        "key_moments": [],
        "cross_modal_insights": cross_modal if cross_modal else [
            {"insight": "No cross-modal patterns detected in this window",
             "modalities": [], "type": "none", "significance": ""}
        ],
        "recommendations": [],
        "risk_assessment": risk_assessment,
        "contamination_timeline": None,
        "technique_analysis": None,
        "deal_assessment": None,
        "objection_handling": None,
    }
