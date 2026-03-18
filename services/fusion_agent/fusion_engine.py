"""
NEXUS Fusion Agent - Fusion Engine
Temporal alignment, signal buffering, and Unified Speaker State computation.

The engine maintains a sliding buffer of signals per speaker per agent.
On each fusion cycle it:
  1. Collects signals from all agents within the temporal window
  2. Runs pairwise fusion rules on overlapping signals
  3. Computes the Unified Speaker State
  4. Emits fusion signals and alerts

Architecture reference: docs/ARCHITECTURE.md — Fusion Temporal Alignment
  Immediate: 0-2s   (Voice × Face — not used in audio-only mode)
  Short:     2-10s   (Most pairwise rules)
  Medium:    10-60s  (Compound patterns)
  Long:      1-15min (Temporal sequences)
"""
import json
import time
import logging
from collections import defaultdict
from typing import Optional
from dataclasses import asdict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.models.signals import UnifiedSpeakerState

logger = logging.getLogger("nexus.fusion.engine")

# Temporal window sizes in milliseconds
WINDOW_SHORT_MS = 10_000       # 10 seconds — pairwise rules
WINDOW_MEDIUM_MS = 60_000      # 60 seconds — compound patterns
WINDOW_LONG_MS = 300_000       # 5 minutes — temporal sequences
BUFFER_MAX_AGE_MS = 600_000    # 10 minutes — max signal retention


class SignalBuffer:
    """
    Sliding buffer of signals per speaker, organised by agent and signal type.
    Efficiently queries signals within temporal windows.
    """

    def __init__(self):
        # {speaker_id: {agent: [signal_dict, ...]}}
        self._buffer: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add(self, signal: dict):
        """Add a signal to the buffer."""
        speaker = signal.get("speaker_id", "unknown")
        agent = signal.get("agent", "unknown")
        self._buffer[speaker][agent].append(signal)

    def add_many(self, signals: list[dict]):
        """Add multiple signals."""
        for s in signals:
            self.add(s)

    def get_signals(
        self,
        speaker_id: str,
        agent: str,
        signal_type: str = None,
        window_ms: int = WINDOW_SHORT_MS,
        reference_time_ms: int = None,
    ) -> list[dict]:
        """
        Get signals for a speaker from a specific agent within a temporal window.

        Args:
            speaker_id: Speaker to query
            agent: Agent name (voice, language, etc.)
            signal_type: Optional filter by signal_type
            window_ms: How far back to look
            reference_time_ms: Reference point (default: now)

        Returns:
            List of matching signals, oldest first
        """
        if reference_time_ms is None:
            reference_time_ms = int(time.time() * 1000)

        cutoff = reference_time_ms - window_ms
        results = []

        for s in self._buffer[speaker_id].get(agent, []):
            start_ms = _to_int(s.get("window_start_ms", 0))
            if start_ms < cutoff:
                continue
            if signal_type and s.get("signal_type") != signal_type:
                continue
            results.append(s)

        return results

    def get_all_for_speaker(
        self,
        speaker_id: str,
        window_ms: int = WINDOW_SHORT_MS,
        reference_time_ms: int = None,
    ) -> dict[str, list[dict]]:
        """Get all signals for a speaker within window, grouped by agent."""
        if reference_time_ms is None:
            reference_time_ms = int(time.time() * 1000)

        cutoff = reference_time_ms - window_ms
        result = {}

        for agent, signals in self._buffer[speaker_id].items():
            filtered = [
                s for s in signals
                if _to_int(s.get("window_start_ms", 0)) >= cutoff
            ]
            if filtered:
                result[agent] = filtered

        return result

    @property
    def speakers(self) -> list[str]:
        """List all speakers with buffered signals."""
        return list(self._buffer.keys())

    def prune(self, max_age_ms: int = BUFFER_MAX_AGE_MS):
        """Remove signals older than max_age_ms."""
        cutoff = int(time.time() * 1000) - max_age_ms
        for speaker_id in list(self._buffer.keys()):
            for agent in list(self._buffer[speaker_id].keys()):
                self._buffer[speaker_id][agent] = [
                    s for s in self._buffer[speaker_id][agent]
                    if _to_int(s.get("window_start_ms", 0)) >= cutoff
                ]
                if not self._buffer[speaker_id][agent]:
                    del self._buffer[speaker_id][agent]
            if not self._buffer[speaker_id]:
                del self._buffer[speaker_id]

    def signal_count(self) -> int:
        """Total signals in buffer."""
        total = 0
        for speaker in self._buffer.values():
            for signals in speaker.values():
                total += len(signals)
        return total


def compute_unified_state(
    speaker_id: str,
    voice_signals: list[dict],
    language_signals: list[dict],
    fusion_signals: list[dict],
) -> UnifiedSpeakerState:
    """
    Compute the Unified Speaker State from all available signals.
    This is the primary output the dashboard displays.

    Aggregation strategy:
      - stress_level: latest vocal_stress_score value
      - confidence_level: derived from tone + power language
      - sentiment_score: average of recent sentiment values
      - engagement: composite of speech rate + sentiment activity
      - authenticity_score: inverse of fusion credibility flags
    """
    now_ms = int(time.time() * 1000)

    state = UnifiedSpeakerState(
        speaker_id=speaker_id,
        speaker_name=speaker_id,  # Real name comes from session metadata
        timestamp_ms=now_ms,
    )

    # ── Stress: latest vocal stress score ──
    stress_signals = [
        s for s in voice_signals
        if s.get("signal_type") == "vocal_stress_score"
    ]
    if stress_signals:
        latest_stress = max(stress_signals, key=lambda s: _to_int(s.get("window_start_ms", 0)))
        state.stress_level = _to_float(latest_stress.get("value", 0))

    # ── Confidence: from tone + power language ──
    tone_signals = [
        s for s in voice_signals
        if s.get("signal_type") == "tone_classification"
    ]
    power_signals = [
        s for s in language_signals
        if s.get("signal_type") == "power_language_score"
    ]

    confidence_inputs = []
    if tone_signals:
        latest_tone = max(tone_signals, key=lambda s: _to_int(s.get("window_start_ms", 0)))
        tone_text = latest_tone.get("value_text", "neutral")
        if tone_text == "confident":
            confidence_inputs.append(0.80)
        elif tone_text == "nervous":
            confidence_inputs.append(0.25)
        else:
            confidence_inputs.append(0.50)

    if power_signals:
        recent_power = [_to_float(s.get("value", 0.5)) for s in power_signals[-5:]]
        confidence_inputs.append(sum(recent_power) / len(recent_power))

    if confidence_inputs:
        state.confidence_level = round(sum(confidence_inputs) / len(confidence_inputs), 3)

    # ── Sentiment: average of recent values ──
    sentiment_signals = [
        s for s in language_signals
        if s.get("signal_type") == "sentiment_score"
    ]
    if sentiment_signals:
        recent_sent = [_to_float(s.get("value", 0)) for s in sentiment_signals[-10:]]
        state.sentiment_score = round(sum(recent_sent) / len(recent_sent), 3)

    # ── Engagement: composite of activity indicators ──
    engagement_factors = []

    # Speech rate activity (any signal = they're talking = engaged)
    if voice_signals:
        engagement_factors.append(0.60)
    if language_signals:
        engagement_factors.append(0.60)

    # Buying signals boost engagement
    buy_signals = [
        s for s in language_signals
        if s.get("signal_type") == "buying_signal"
    ]
    if buy_signals:
        engagement_factors.append(0.80)

    # High stress slightly reduces engagement score
    if state.stress_level > 0.60:
        engagement_factors.append(0.30)

    if engagement_factors:
        state.engagement = round(sum(engagement_factors) / len(engagement_factors), 3)

    # ── Authenticity: reduced by fusion credibility flags ──
    credibility_signals = [
        s for s in fusion_signals
        if s.get("signal_type") == "credibility_assessment"
    ]
    if credibility_signals:
        # Lower credibility → lower authenticity
        cred_values = [_to_float(s.get("value", 1.0)) for s in credibility_signals[-3:]]
        avg_cred = sum(cred_values) / len(cred_values)
        # credibility_assessment value is 0-1 where lower = less credible
        state.authenticity_score = round(min(avg_cred, 0.85), 3)

    # ── Dominant emotion from sentiment ──
    if state.sentiment_score > 0.3:
        state.dominant_emotion = "positive"
    elif state.sentiment_score < -0.3:
        state.dominant_emotion = "negative"
    else:
        state.dominant_emotion = "neutral"

    # ── Active alerts from fusion signals ──
    alert_signals = [
        s for s in fusion_signals
        if _to_float(s.get("confidence", 0)) >= 0.50
    ]
    state.active_alerts = [
        {
            "type": s.get("signal_type", ""),
            "value_text": s.get("value_text", ""),
            "confidence": _to_float(s.get("confidence", 0)),
        }
        for s in alert_signals[-5:]  # Cap at 5 most recent
    ]

    # ── Uncertainty flags ──
    if state.stress_level > 0.50 and state.sentiment_score > 0.30:
        state.uncertainty_flags.append("stress_sentiment_incongruence")

    return state


def _to_float(v) -> float:
    """Safely convert a value (possibly string from Redis) to float."""
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _to_int(v) -> int:
    """Safely convert a value (possibly string from Redis) to int."""
    if v is None or v == "":
        return 0
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0
