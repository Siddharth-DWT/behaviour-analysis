"""
NEXUS Fusion Agent - Pairwise Cross-Modal Rules
Implements 3 fusion rules for the Phase 1 audio-only vertical slice.

Each rule takes signals from two different agents within a temporal window
and detects patterns that neither agent can detect alone.

Rules implemented:
  FUSION-02: Speech Content × Voice Stress → Credibility assessment
  FUSION-07: Hedge Language × Positive Sentiment → Incongruence detection
             (Audio-only simplification of Head Shake × Affirmative Language)
  FUSION-13: Persuasion Language × Speech Pace → Urgency authenticity

Research references:
  - Bond & DePaulo 2006 (deception detection accuracy ceiling)
  - Levine 2014 (truth-default theory)
  - Rackham 1988 (persuasion + pace in sales)
  - Lakoff 1975 (powerless language as incongruence marker)

Confidence caps from RULES.md:
  FUSION-02 max: 0.55 (deception-adjacent — hard cap)
  FUSION-07 max: 0.70
  FUSION-13 max: 0.60
"""
import logging
from typing import Optional

try:
    from shared.utils.conversions import to_float as _to_float, to_int as _to_int
except ImportError:
    def _to_float(v) -> float:
        if v is None or v == "":
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    def _to_int(v) -> int:
        if v is None or v == "":
            return 0
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return 0

logger = logging.getLogger("nexus.fusion.rules")


class FusionRuleEngine:
    """
    Evaluates cross-modal pairwise rules by correlating signals
    from different agents within temporal windows.
    """

    def evaluate(
        self,
        speaker_id: str,
        voice_signals: list[dict],
        language_signals: list[dict],
        window_start_ms: int = 0,
        window_end_ms: int = 0,
    ) -> list[dict]:
        """
        Run all fusion rules for a speaker given their recent signals.

        Args:
            speaker_id: Speaker identifier
            voice_signals: Recent voice agent signals for this speaker
            language_signals: Recent language agent signals for this speaker
            window_start_ms: Fusion window start
            window_end_ms: Fusion window end

        Returns:
            List of fusion signal dicts
        """
        signals = []

        # ── FUSION-02: Content × Stress → Credibility ──
        cred = self._rule_fusion_02(voice_signals, language_signals)
        if cred is not None:
            signals.append({
                "agent": "fusion",
                "speaker_id": speaker_id,
                "signal_type": "credibility_assessment",
                "value": round(cred["score"], 4),
                "value_text": cred["level"],
                "confidence": round(cred["confidence"], 4),
                "window_start_ms": window_start_ms,
                "window_end_ms": window_end_ms,
                "metadata": cred["evidence"],
            })

        # ── FUSION-07: Hedge × Positive Sentiment → Incongruence ──
        incong = self._rule_fusion_07(language_signals, voice_signals)
        if incong is not None:
            signals.append({
                "agent": "fusion",
                "speaker_id": speaker_id,
                "signal_type": "verbal_incongruence",
                "value": round(incong["score"], 4),
                "value_text": incong["level"],
                "confidence": round(incong["confidence"], 4),
                "window_start_ms": window_start_ms,
                "window_end_ms": window_end_ms,
                "metadata": incong["evidence"],
            })

        # ── FUSION-13: Persuasion × Pace → Urgency Authenticity ──
        urg = self._rule_fusion_13(voice_signals, language_signals)
        if urg is not None:
            signals.append({
                "agent": "fusion",
                "speaker_id": speaker_id,
                "signal_type": "urgency_authenticity",
                "value": round(urg["score"], 4),
                "value_text": urg["level"],
                "confidence": round(urg["confidence"], 4),
                "window_start_ms": window_start_ms,
                "window_end_ms": window_end_ms,
                "metadata": urg["evidence"],
            })

        return signals

    # ════════════════════════════════════════════════════════
    # FUSION-02: Speech Content × Voice Stress → Credibility
    # Research: Bond & DePaulo 2006, Levine 2014
    # Max confidence: 0.55 (deception-adjacent, hard cap)
    # ════════════════════════════════════════════════════════

    def _rule_fusion_02(
        self,
        voice_signals: list[dict],
        language_signals: list[dict],
    ) -> Optional[dict]:
        """
        Cross-modal credibility check:
        When the CONTENT says something positive/certain but the VOICE
        shows elevated stress, the gap suggests reduced credibility.

        Logic:
          IF sentiment is positive (value > 0.3)
             AND vocal_stress is elevated (value > 0.40)
          THEN credibility_concern

          The wider the gap between positive content and vocal stress,
          the stronger the signal.

        Does NOT claim deception — only flags incongruence for human review.
        """
        # Get most recent stress signals
        stress_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "vocal_stress_score"
        ]
        sentiment_signals = [
            s for s in language_signals
            if s.get("signal_type") == "sentiment_score"
        ]

        if not stress_signals or not sentiment_signals:
            return None

        # Use the most recent of each
        latest_stress = max(
            stress_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
        )
        latest_sentiment = max(
            sentiment_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
        )

        stress_value = _to_float(latest_stress.get("value", 0))
        sentiment_value = _to_float(latest_sentiment.get("value", 0))

        # Only fire when content is positive but voice is stressed
        if sentiment_value <= 0.30 or stress_value <= 0.40:
            return None

        # Credibility gap: how far apart are content and voice?
        # sentiment_value is 0.3-1.0 (positive), stress is 0.4-1.0 (elevated)
        gap = (sentiment_value - 0.30) * (stress_value - 0.30)
        # Normalise to 0-1 range (max gap when both are extreme)
        credibility_score = min(gap / 0.49, 1.0)  # 0.7*0.7 = 0.49

        # Invert: lower score = less credible
        credibility_score = 1.0 - credibility_score

        # Classify
        if credibility_score < 0.40:
            level = "credibility_concern"
        elif credibility_score < 0.60:
            level = "mild_incongruence"
        else:
            level = "mostly_congruent"

        # Only emit when there's actual concern
        if credibility_score >= 0.70:
            return None

        # Confidence calculation (hard cap at 0.55)
        raw_confidence = min(0.45 + (1.0 - credibility_score) * 0.20, 0.55)

        # Check for reinforcing signals (fillers, pitch elevation)
        filler_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "filler_detection"
            and s.get("value_text") in ("filler_spike", "filler_elevated")
        ]
        pitch_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "pitch_elevation_flag"
        ]

        reinforcing_count = 0
        if filler_signals:
            reinforcing_count += 1
        if pitch_signals:
            reinforcing_count += 1

        if reinforcing_count > 0:
            raw_confidence = min(raw_confidence + 0.05 * reinforcing_count, 0.55)

        return {
            "score": credibility_score,
            "level": level,
            "confidence": raw_confidence,
            "evidence": {
                "sentiment_value": round(sentiment_value, 3),
                "stress_value": round(stress_value, 3),
                "gap_magnitude": round(1.0 - credibility_score, 3),
                "reinforcing_signals": reinforcing_count,
                "filler_elevated": len(filler_signals) > 0,
                "pitch_elevated": len(pitch_signals) > 0,
            },
        }

    # ════════════════════════════════════════════════════════
    # FUSION-07: Hedge Language × Positive Sentiment → Incongruence
    # Audio-only simplification of Head Shake × Affirmative Language
    # Research: Lakoff 1975, Navarro 2008
    # Max confidence: 0.70
    # ════════════════════════════════════════════════════════

    def _rule_fusion_07(
        self,
        language_signals: list[dict],
        voice_signals: list[dict] = None,
    ) -> Optional[dict]:
        """
        Audio-only adaptation of FUSION-07 (Head Shake × Affirmative Language).

        True incongruence requires ALL of:
          a) Sentiment > +0.4 (clearly positive)
          b) At least ONE of: objection (conf > 0.5), power < 0.25, filler_rate > 3%
          c) Voice stress > 0.35 for same speaker/window
          d) Combined confidence > 0.45

        If only (a)+(b) met but NOT (c): downgrade to "hedged_agreement"
        with confidence capped at 0.35 — logged but not alerted.
        """
        voice_signals = voice_signals or []

        sentiment_signals = [
            s for s in language_signals
            if s.get("signal_type") == "sentiment_score"
        ]
        power_signals = [
            s for s in language_signals
            if s.get("signal_type") == "power_language_score"
        ]

        if not sentiment_signals:
            return None

        latest_sentiment = max(
            sentiment_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
        )
        sentiment_value = _to_float(latest_sentiment.get("value", 0))

        # (a) Sentiment must be clearly positive (> 0.4, not just > 0.3)
        if sentiment_value <= 0.40:
            return None

        # Get power value
        power_value = 0.5
        if power_signals:
            latest_power = max(
                power_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
            )
            power_value = _to_float(latest_power.get("value", 0.5))

        # (b) At least ONE linguistic marker:
        #   - objection with confidence > 0.5
        #   - power < 0.25 (very weak)
        #   - filler elevated
        objection_signals = [
            s for s in language_signals
            if s.get("signal_type") == "objection_signal"
            and _to_float(s.get("confidence", 0)) > 0.50
        ]
        filler_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "filler_detection"
            and s.get("value_text") in ("filler_spike", "filler_elevated", "elevated", "high")
        ]

        has_objection = len(objection_signals) > 0
        has_weak_power = power_value < 0.25
        has_filler_spike = len(filler_signals) > 0

        if not (has_objection or has_weak_power or has_filler_spike):
            return None

        # (c) Check voice stress > 0.35
        stress_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "vocal_stress_score"
        ]
        stress_value = 0.0
        if stress_signals:
            latest_stress = max(
                stress_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
            )
            stress_value = _to_float(latest_stress.get("value", 0))

        has_voice_stress = stress_value > 0.35

        # Build evidence
        evidence = {
            "sentiment_value": round(sentiment_value, 3),
            "power_value": round(power_value, 3),
            "stress_value": round(stress_value, 3),
            "objection_present": has_objection,
            "weak_power": has_weak_power,
            "filler_spike": has_filler_spike,
            "voice_stress_aligned": has_voice_stress,
        }

        # Score based on how many markers fire
        marker_count = sum([has_objection, has_weak_power, has_filler_spike])
        incongruence = (sentiment_value - 0.40) * (marker_count * 0.20)
        score = min(incongruence / 0.36, 1.0)

        if not has_voice_stress:
            # Downgrade: polite hedging, not true incongruence
            return {
                "score": min(score, 0.30),
                "level": "hedged_agreement",
                "confidence": min(0.35, score * 0.50),
                "evidence": evidence,
            }

        # True incongruence: sentiment + markers + voice stress
        if score > 0.60:
            level = "strong_verbal_incongruence"
        elif score > 0.35:
            level = "moderate_verbal_incongruence"
        else:
            level = "mild_verbal_incongruence"

        raw_confidence = min(0.40 + score * 0.30, 0.70)

        # (d) Minimum confidence threshold
        if raw_confidence < 0.45:
            return None

        return {
            "score": score,
            "level": level,
            "confidence": raw_confidence,
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════
    # FUSION-13: Persuasion Language × Speech Pace → Urgency
    # Research: Rackham 1988, Apple et al. 1979
    # Max confidence: 0.60
    # ════════════════════════════════════════════════════════

    def _rule_fusion_13(
        self,
        voice_signals: list[dict],
        language_signals: list[dict],
    ) -> Optional[dict]:
        """
        Urgency authenticity check:
        When someone uses buying/persuasion language while also speaking
        faster than baseline, are they genuinely excited or artificially
        creating urgency?

        Logic:
          IF buying_signal or positive_intent detected
             AND speech_rate_anomaly = rate_elevated
          THEN assess urgency authenticity

        Authentic urgency:
          - Rate elevated + tone confident + sentiment positive
          → genuine excitement about the topic

        Manufactured urgency:
          - Rate elevated + stress elevated + filler elevated
          → artificial pressure or anxiety-driven rushing

        This is commercially valuable: it helps distinguish a salesperson
        genuinely enthusiastic about their product from one artificially
        creating time pressure.
        """
        # Need rate anomaly signal
        rate_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "speech_rate_anomaly"
            and s.get("value_text") == "rate_elevated"
        ]
        # Need buying/persuasion language
        buying_signals = [
            s for s in language_signals
            if s.get("signal_type") == "buying_signal"
        ]
        intent_signals = [
            s for s in language_signals
            if s.get("signal_type") == "intent_classification"
            and s.get("value_text") in ("PROPOSE", "CLOSE", "NEGOTIATE", "COMMIT")
        ]

        persuasion_present = len(buying_signals) > 0 or len(intent_signals) > 0

        if not rate_signals or not persuasion_present:
            return None

        latest_rate = max(
            rate_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
        )

        # Check what's driving the acceleration
        sub_class = ""
        if isinstance(latest_rate.get("metadata"), dict):
            sub_class = latest_rate["metadata"].get("sub_classification", "")
        elif isinstance(latest_rate.get("metadata"), str):
            try:
                import json
                meta = json.loads(latest_rate["metadata"])
                sub_class = meta.get("sub_classification", "")
            except (json.JSONDecodeError, TypeError):
                pass

        # Check tone
        tone_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "tone_classification"
        ]
        latest_tone = None
        if tone_signals:
            latest_tone = max(
                tone_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
            )

        tone_text = latest_tone.get("value_text", "neutral") if latest_tone else "neutral"

        # Check stress
        stress_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "vocal_stress_score"
        ]
        latest_stress_val = 0.0
        if stress_signals:
            latest_stress = max(
                stress_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
            )
            latest_stress_val = _to_float(latest_stress.get("value", 0))

        # Check fillers
        filler_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "filler_detection"
            and s.get("value_text") in ("filler_spike", "filler_elevated")
        ]

        # ── Classify authenticity ──
        authenticity_score = 0.50  # Start neutral

        # Authentic indicators
        if tone_text == "confident":
            authenticity_score += 0.20
        if sub_class == "enthusiasm_driven_acceleration":
            authenticity_score += 0.20
        if latest_stress_val < 0.30:
            authenticity_score += 0.10

        # Manufactured indicators
        if tone_text == "nervous":
            authenticity_score -= 0.20
        if sub_class == "anxiety_driven_acceleration":
            authenticity_score -= 0.25
        if latest_stress_val > 0.50:
            authenticity_score -= 0.15
        if filler_signals:
            authenticity_score -= 0.10

        authenticity_score = max(0.0, min(1.0, authenticity_score))

        # Classify
        if authenticity_score >= 0.65:
            level = "authentic_urgency"
        elif authenticity_score >= 0.40:
            level = "ambiguous_urgency"
        else:
            level = "manufactured_urgency"

        # Confidence (cap at 0.60)
        raw_confidence = min(0.40 + abs(authenticity_score - 0.50) * 0.40, 0.60)

        # Rate magnitude boosts confidence
        rate_delta = abs(_to_float(latest_rate.get("value", 0)))
        if rate_delta > 40.0:  # >40% rate change is very significant
            raw_confidence = min(raw_confidence + 0.05, 0.60)

        evidence = {
            "rate_delta_pct": round(rate_delta, 1),
            "rate_sub_classification": sub_class,
            "tone": tone_text,
            "stress_level": round(latest_stress_val, 3),
            "filler_elevated": len(filler_signals) > 0,
            "buying_signals_present": len(buying_signals) > 0,
            "persuasion_intents": [s.get("value_text", "") for s in intent_signals[:3]],
        }

        return {
            "score": authenticity_score,
            "level": level,
            "confidence": raw_confidence,
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════
    # GRAPH-BASED FUSION RULES
    # ════════════════════════════════════════════════════════

    def evaluate_graph_insights(
        self,
        graph_insights: dict,
        speakers: list[str],
        existing_fusion_signals: list[dict],
    ) -> list[dict]:
        """Generate fusion signals from graph analytics."""
        signals = []

        # FUSION-GRAPH-01: Tension Cluster Detection
        for cluster in graph_insights.get("tension_clusters", []):
            if cluster["signal_count"] >= 3:
                conf = min(0.50 + (cluster["signal_count"] - 3) * 0.05, 0.75)
                signals.append({
                    "agent": "fusion",
                    "speaker_id": cluster["speaker_id"],
                    "signal_type": "tension_cluster",
                    "value": round(cluster["signal_count"] / 10.0, 3),
                    "value_text": "high_tension" if cluster["signal_count"] >= 5 else "moderate_tension",
                    "confidence": round(conf, 3),
                    "window_start_ms": cluster["timestamp_ms"],
                    "window_end_ms": cluster["timestamp_ms"] + cluster["duration_ms"],
                    "metadata": cluster,
                })

        # FUSION-GRAPH-02: Momentum Shift Detection
        momentum = graph_insights.get("momentum", {})
        if (
            momentum.get("turning_point_ms")
            and momentum.get("overall_trajectory") in ("positive", "negative")
        ):
            signals.append({
                "agent": "fusion",
                "speaker_id": "all",
                "signal_type": "momentum_shift",
                "value": momentum.get("momentum_score", 0),
                "value_text": f"{momentum['overall_trajectory']}_trajectory",
                "confidence": 0.55,
                "window_start_ms": momentum["turning_point_ms"],
                "window_end_ms": momentum["turning_point_ms"],
                "metadata": momentum,
            })

        # FUSION-GRAPH-03: Persistent Incongruence
        for speaker_id, pattern in graph_insights.get("incongruence_patterns", {}).items():
            if pattern.get("consistency") == "persistent":
                signals.append({
                    "agent": "fusion",
                    "speaker_id": speaker_id,
                    "signal_type": "persistent_incongruence",
                    "value": round(pattern["total_contradicts_edges"] / 10.0, 3),
                    "value_text": "persistent_incongruence",
                    "confidence": 0.60,
                    "window_start_ms": pattern.get("worst_incongruence_ms", 0),
                    "window_end_ms": pattern.get("worst_incongruence_ms", 0),
                    "metadata": pattern,
                })

        return signals
