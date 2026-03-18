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
import sys
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import asdict

logger = logging.getLogger("nexus.fusion.narrative")

# Ensure shared modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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


def generate_session_narrative(
    session_id: str,
    duration_seconds: float,
    speakers: list[str],
    voice_summary: dict,
    language_summary: dict,
    fusion_signals: list[dict],
    unified_states: list[dict],
    meeting_type: str = "sales_call",
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
    llm_complete = _get_llm_complete()
    if llm_complete is None:
        return _fallback_narrative(
            speakers, voice_summary, language_summary, fusion_signals
        )

    # ── Build the structured context for the LLM ──
    context = _build_context(
        session_id, duration_seconds, speakers,
        voice_summary, language_summary, fusion_signals, unified_states,
    )

    system_prompt, user_prompt = _build_prompt(context, meeting_type)

    try:
        raw_text = llm_complete(
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
            speakers, voice_summary, language_summary, fusion_signals
        )


def _build_context(
    session_id: str,
    duration_seconds: float,
    speakers: list[str],
    voice_summary: dict,
    language_summary: dict,
    fusion_signals: list[dict],
    unified_states: list[dict],
) -> str:
    """Build a structured text context block for the LLM."""
    lines = []
    lines.append(f"SESSION: {session_id}")
    lines.append(f"DURATION: {duration_seconds:.0f} seconds ({duration_seconds/60:.1f} minutes)")
    lines.append(f"SPEAKERS: {', '.join(speakers)}")
    lines.append("")

    # Voice summary per speaker
    lines.append("=== VOICE ANALYSIS ===")
    for speaker_id, data in voice_summary.get("per_speaker", {}).items():
        lines.append(f"\n[{speaker_id}]")
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
                f"  {time_s:.0f}s — {p.get('speaker', '?')}: "
                f"stress={p.get('stress_score', 0):.3f}"
            )

    # Language summary per speaker
    lines.append("\n=== LANGUAGE ANALYSIS ===")
    for speaker_id, data in language_summary.get("per_speaker", {}).items():
        lines.append(f"\n[{speaker_id}]")
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
                f"  {time_s:.0f}s — {m.get('speaker', '?')}: "
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
                f"  {time_s:.0f}s — {m.get('speaker', '?')}: "
                f"strength={m.get('strength', 0):.3f}, "
                f"categories={m.get('categories', [])}"
            )

    # Fusion signals
    if fusion_signals:
        lines.append("\n=== CROSS-MODAL FUSION INSIGHTS ===")
        for s in fusion_signals[:10]:
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
            lines.append(f"\n[{sid}]")
            lines.append(f"  Engagement: {state.get('engagement', 0.5):.2f}")
            lines.append(f"  Confidence: {state.get('confidence_level', 0.5):.2f}")
            lines.append(f"  Stress: {state.get('stress_level', 0):.2f}")
            lines.append(f"  Sentiment: {state.get('sentiment_score', 0):.2f}")
            lines.append(f"  Authenticity: {state.get('authenticity_score', 1.0):.2f}")
            alerts = state.get("active_alerts", [])
            if alerts:
                lines.append(f"  Active alerts: {json.dumps(alerts)}")

    return "\n".join(lines)


def _build_prompt(context: str, meeting_type: str) -> tuple[str, str]:
    """Build system + user prompts for narrative generation."""
    type_instructions = {
        # ── Sales-specific ──
        "sales_call": (
            "Focus on: buying signals timeline, objection handling effectiveness, "
            "stress points during pricing/negotiation, decision readiness indicators, "
            "close probability assessment, and whether the prospect showed genuine "
            "or manufactured interest. Identify the seller and buyer roles."
        ),
        # ── Podcast ──
        "podcast": (
            "Focus on: speaker dynamics and chemistry, topic flow and transitions, "
            "engagement peaks (where energy or interest spikes), most interesting "
            "moments based on vocal/linguistic indicators, host vs guest balance, "
            "and audience engagement proxy from speaker energy patterns."
        ),
        # ── Interview ──
        "interview": (
            "Focus on: candidate confidence trajectory over time, question quality "
            "from the interviewer, rapport assessment between participants, "
            "stress indicators during difficult questions, authenticity markers, "
            "and communication style adaptability."
        ),
        # ── Lecture ──
        "lecture": (
            "Focus on: speaker clarity and pacing analysis, energy level changes "
            "over time, engagement indicators (if Q&A present), key emphasis moments, "
            "vocal fatigue patterns, and whether the speaker maintained audience "
            "attention through vocal variety."
        ),
        # ── Debate ──
        "debate": (
            "Focus on: argument strength per speaker, dominance shifts over time, "
            "emotional escalation patterns, persuasion technique effectiveness, "
            "stress responses to challenges, and which speaker demonstrated "
            "more confidence and control."
        ),
        # ── Meeting ──
        "meeting": (
            "Focus on: participation balance across all speakers, decision points "
            "and consensus moments, action item indicators, engagement distribution, "
            "stress or tension moments, dominant vs passive speakers, and "
            "overall meeting productivity indicators."
        ),
        # ── Presentation ──
        "presentation": (
            "Focus on: speaker clarity and confidence, persuasion effectiveness, "
            "audience engagement indicators, key message delivery moments, "
            "pacing and energy management, and vocal conviction markers."
        ),
        # ── Casual conversation ──
        "casual_conversation": (
            "Focus on: rapport and connection quality, engagement levels, "
            "emotional dynamics, turn-taking balance, and overall "
            "communication chemistry between participants."
        ),
        # ── Legacy types ──
        "client_meeting": (
            "Focus on: rapport trajectory, client satisfaction indicators, "
            "risk flags, unspoken concerns detected through cross-modal analysis."
        ),
        "internal": (
            "Focus on: engagement distribution across participants, "
            "contribution balance, stress patterns, and consensus indicators."
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
        "You produce PROBABILISTIC INDICATORS, never binary determinations. "
        "Never claim to detect deception — only flag incongruence for human review. "
        "Maximum certainty for any claim is 'likely' or 'suggests' — never 'definitely'. "
        "Cross-modal insights (where voice and language disagree) are the most valuable findings."
    )

    user_prompt = f"""Analyse the following multi-modal signal data from a recorded meeting and produce a structured report.

FOCUS: {focus}

{context}

Respond with a JSON object containing EXACTLY these fields:
{{
  "executive_summary": "2-3 sentence overview of the session's key dynamics",
  "key_moments": [
    {{"time_description": "timestamp or time range", "description": "what happened", "significance": "why it matters"}}
  ],
  "cross_modal_insights": [
    "Each insight describes a pattern that only cross-modal analysis reveals"
  ],
  "recommendations": [
    "Actionable recommendations based on the analysis"
  ]
}}

Return ONLY the JSON object, no other text."""

    return system_prompt, user_prompt


def _parse_narrative_response(raw_text: str, speakers: list[str]) -> dict:
    """Parse the LLM response into structured report."""
    # Strip markdown wrapping if present
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(text)
        return {
            "executive_summary": parsed.get("executive_summary", ""),
            "key_moments": parsed.get("key_moments", []),
            "cross_modal_insights": parsed.get("cross_modal_insights", []),
            "recommendations": parsed.get("recommendations", []),
        }
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM narrative as JSON, returning raw text")
        return {
            "executive_summary": text[:500],
            "key_moments": [],
            "cross_modal_insights": [],
            "recommendations": [],
        }


def _fallback_narrative(
    speakers: list[str],
    voice_summary: dict,
    language_summary: dict,
    fusion_signals: list[dict],
) -> dict:
    """
    Generate a basic narrative without LLM API.
    Used when API key is not configured or API call fails.
    """
    insights = []

    # Extract basic insights from summaries
    for speaker_id, data in language_summary.get("per_speaker", {}).items():
        if data.get("buying_signal_count", 0) > 0:
            insights.append(
                f"{speaker_id} showed {data['buying_signal_count']} buying signal(s)"
            )
        if data.get("objection_count", 0) > 0:
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
        if sig_type == "credibility_assessment" and s.get("value_text") == "credibility_concern":
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

    return {
        "executive_summary": (
            f"Session with {len(speakers)} speaker(s). "
            + (insights[0] + ". " if insights else "No notable patterns detected. ")
            + f"{len(fusion_signals)} cross-modal fusion signal(s) generated."
        ),
        "key_moments": [],
        "cross_modal_insights": cross_modal if cross_modal else ["No cross-modal patterns detected in this window"],
        "recommendations": [],
    }
