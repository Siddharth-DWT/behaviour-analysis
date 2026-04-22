"""
Compound Pattern Engine — Phase 2F
Implements 12 multi-signal behavioral states (C-01 through C-12).

Each pattern requires 3+ co-occurring signals from different modalities.
The "cluster rule" (Pease & Pease 2004): reliable behavioural interpretation
requires N congruent signals — never a single indicator in isolation.

Research anchors:
  Pease & Pease 2004            — 3+ congruent signals for reliable interpretation
  Navarro 2008                  — Multi-channel body language (deception clusters)
  Tickle-Degnen & Rosenthal 1990 — Rapport: attention + positivity + coordination
  Glasl 1982                    — Conflict escalation stages (feeds into C-06)
  Mehrabian 1972                — 7% words / 38% voice / 55% body split (motivation for compound approach)
"""
import logging
from typing import Optional

logger = logging.getLogger("nexus.fusion.compound")

# Max confidence caps per pattern (NEXUS: never exceed 0.85; deception-related ≤ 0.50)
_CAPS: dict[str, float] = {
    "C-01": 0.80, "C-02": 0.75, "C-03": 0.70, "C-04": 0.65,
    "C-05": 0.75, "C-06": 0.80, "C-07": 0.65, "C-08": 0.75,
    "C-09": 0.75, "C-10": 0.70, "C-11": 0.65, "C-12": 0.50,
}


def _sig(
    signals: list[dict],
    signal_type: str,
    value_text: Optional[str] = None,
    min_value: Optional[float] = None,
) -> Optional[dict]:
    """Return first matching signal or None. O(N) scan."""
    for s in signals:
        if s.get("signal_type") != signal_type:
            continue
        if value_text is not None and s.get("value_text") != value_text:
            continue
        if min_value is not None and (s.get("value") or 0.0) < min_value:
            continue
        return s
    return None


def _agent_diversity(hits: dict[str, Optional[dict]]) -> int:
    """
    Count distinct modality sources among matched signals.
    For video signals, rule_id prefix discriminates facial/gaze/body sub-channels
    so that compound patterns can fire from video-only signal combinations.
    Enforces the cluster rule (Pease 2004): congruent signals must cross modalities.
    """
    modalities: set[str] = set()
    for sig in hits.values():
        if sig is None:
            continue
        agent = sig.get("agent")
        if not agent:
            continue
        if agent == "video":
            rule_id = (sig.get("metadata") or {}).get("rule_id", "")
            if rule_id.startswith("FACE"):
                modalities.add("video_facial")
            elif rule_id.startswith("GAZE"):
                modalities.add("video_gaze")
            elif rule_id.startswith("BODY"):
                modalities.add("video_body")
            else:
                modalities.add(agent)
        else:
            modalities.add(agent)
    return len(modalities)


class CompoundPatternEngine:
    """
    Detects complex behavioral states from 3+ co-occurring signals.
    Operates on the merged signal set for a speaker within a time window.

    DSA: each of the 12 patterns is O(N). Total: O(12N) → O(N).
    """

    AGENT_NAME = "fusion"

    def evaluate(
        self,
        speaker_id: str,
        voice_signals: list[dict],
        language_signals: list[dict],
        video_signals: list[dict],
        fusion_signals: list[dict],
        window_start_ms: int,
        window_end_ms: int,
    ) -> list[dict]:
        all_signals = voice_signals + language_signals + video_signals + fusion_signals

        patterns = [
            self._c01_genuine_engagement,
            self._c02_active_disengagement,
            self._c03_emotional_suppression,
            self._c04_decision_engagement,
            self._c05_cognitive_overload,
            self._c06_conflict_escalation,
            self._c07_verbal_nonverbal_discordance,
            self._c08_peak_performance,
            self._c09_rapport_building,
            self._c10_dominance_display,
            self._c11_submission_signal,
            self._c12_deception_cluster,
        ]

        results: list[dict] = []
        for fn in patterns:
            result = fn(speaker_id, all_signals, window_start_ms, window_end_ms)
            if result:
                results.append(result)

        if results:
            logger.debug(
                f"[{speaker_id}] CompoundPatternEngine: {len(results)} compound patterns fired"
            )
        return results

    def _make_signal(
        self,
        rule_id: str,
        signal_type: str,
        speaker_id: str,
        value: float,
        value_text: str,
        confidence: float,
        window_start_ms: int,
        window_end_ms: int,
        metadata: Optional[dict] = None,
    ) -> dict:
        cap = _CAPS.get(rule_id, 0.75)
        return {
            "agent": self.AGENT_NAME,
            "rule_id": rule_id,
            "speaker_id": speaker_id,
            "signal_type": signal_type,
            "value": round(value, 4),
            "value_text": value_text,
            "confidence": round(min(confidence, cap, 0.85), 4),
            "window_start_ms": window_start_ms,
            "window_end_ms": window_end_ms,
            "metadata": metadata or {},
        }

    # ── C-01: Genuine Engagement ───────────────────────────────────────────────
    def _c01_genuine_engagement(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        Forward lean + eye contact + head nods + warm/confident tone + high attention.
        3+ of 5 components required (cluster rule, Pease 2004).
        """
        components = {
            "forward_lean":   _sig(signals, "body_lean", "forward_lean"),
            "eye_contact":    _sig(signals, "screen_contact", "sustained_eye_contact"),
            "head_nod":       _sig(signals, "head_nod"),
            "warm_tone": (
                _sig(signals, "tone_classification", "confident")
                or _sig(signals, "tone_classification", "enthusiastic")
                or _sig(signals, "tone_classification", "excited")
            ),
            "high_attention": _sig(signals, "attention_level", "high_attention"),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * (len(hits) / 3.0) * 1.1, _CAPS["C-01"])
        return self._make_signal(
            "C-01", "genuine_engagement", speaker_id, score, "genuine_engagement",
            confidence, ws, we, {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-02: Active Disengagement ─────────────────────────────────────────────
    def _c02_active_disengagement(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        Backward lean + gaze breaks + fidgeting + low engagement.
        """
        components = {
            "backward_lean": _sig(signals, "body_lean", "backward_lean"),
            "gaze_break": (
                _sig(signals, "gaze_direction_shift")
                or _sig(signals, "sustained_distraction")
            ),
            "fidgeting":      _sig(signals, "body_fidgeting"),
            "low_engagement": (
                _sig(signals, "conversation_engagement", "passive")
                or _sig(signals, "attention_level", "reduced_attention")
            ),
            "low_contact": _sig(signals, "screen_contact", "low_screen_contact"),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * (len(hits) / 3.0) * 1.1, _CAPS["C-02"])
        return self._make_signal(
            "C-02", "active_disengagement", speaker_id, score, "active_disengagement",
            confidence, ws, we, {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-03: Emotional Suppression ────────────────────────────────────────────
    def _c03_emotional_suppression(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        High voice stress + masking fusion signals + controlled speech.
        Classic suppression: internal arousal masked externally.
        Voice stress is required — without it the pattern is ambiguous.
        """
        voice_stress = _sig(signals, "vocal_stress_score", min_value=0.50)
        if voice_stress is None:
            return None
        # Suppression = high internal arousal (voice stress) but masked externally.
        # tone_face_masking / stress_suppression are the fusion signals that encode this
        # directly. facial_stress is intentionally excluded: its presence means the
        # face IS showing stress → not suppressed.
        components = {
            "voice_stress":    voice_stress,
            "masking": (
                _sig(signals, "tone_face_masking")
                or _sig(signals, "stress_suppression")
            ),
            "controlled_rate": _sig(signals, "speech_rate_anomaly"),
            "neutral_words":   _sig(signals, "sentiment_score"),
            "fillers":         _sig(signals, "filler_detection"),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * 0.9, _CAPS["C-03"])
        return self._make_signal(
            "C-03", "emotional_suppression", speaker_id, score, "emotional_suppression",
            confidence, ws, we, {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-04: Decision Engagement (EXPERIMENTAL) ───────────────────────────────
    def _c04_decision_engagement(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        Buying signals + physical engagement indicators.
        Buying signal is required — this is a commercial sales/pitch pattern.
        EXPERIMENTAL: implies readiness to decide.
        """
        buying = (
            _sig(signals, "buying_signal")
            or _sig(signals, "buying_signal_detected")
        )
        if buying is None:
            return None
        components = {
            "buying_signal":  buying,
            "forward_lean":   _sig(signals, "body_lean", "forward_lean"),
            "high_attention": _sig(signals, "attention_level", "high_attention"),
            "eye_contact":    _sig(signals, "screen_contact", "sustained_eye_contact"),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * (len(hits) / 3.0) * 1.1, _CAPS["C-04"])
        return self._make_signal(
            "C-04", "decision_engagement", speaker_id, score, "decision_ready",
            confidence, ws, we,
            {"components": list(hits.keys()), "hit_count": len(hits), "experimental": True},
        )

    # ── C-05: Cognitive Overload ───────────────────────────────────────────────
    def _c05_cognitive_overload(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        High fillers + gaze breaks + slow rate + self-touch + long latency.
        """
        components = {
            "filler":           _sig(signals, "filler_detection"),
            "gaze_break": (
                _sig(signals, "gaze_direction_shift")
                or _sig(signals, "sustained_distraction")
            ),
            "slow_rate":        _sig(signals, "speech_rate_anomaly", "rate_depressed"),
            "self_touch":       _sig(signals, "self_touch"),
            "response_latency": _sig(signals, "response_latency_pattern"),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * (len(hits) / 3.0) * 1.1, _CAPS["C-05"])
        return self._make_signal(
            "C-05", "cognitive_overload", speaker_id, score, "cognitive_overload",
            confidence, ws, we, {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-06: Conflict Escalation ──────────────────────────────────────────────
    def _c06_conflict_escalation(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        Rising stress + interruptions + aggressive tone + objection signals + forward lean.
        """
        components = {
            "high_stress":   _sig(signals, "vocal_stress_score", min_value=0.55),
            "interruption": (
                _sig(signals, "interruption_event")
                or _sig(signals, "interruption_pattern")
            ),
            "aggressive_tone": (
                _sig(signals, "tone_classification", "confrontational")
                or _sig(signals, "tone_classification", "dominant")
                or _sig(signals, "tone_classification", "aggressive")
            ),
            "objection": (
                _sig(signals, "objection_signal")
                or _sig(signals, "objection_detected")
            ),
            "forward_lean":  _sig(signals, "body_lean", "forward_lean"),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * (len(hits) / 3.0) * 1.1, _CAPS["C-06"])
        return self._make_signal(
            "C-06", "conflict_escalation", speaker_id, score, "conflict_escalation",
            confidence, ws, we, {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-07: Verbal-Nonverbal Discordance ─────────────────────────────────────
    def _c07_verbal_nonverbal_discordance(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        Positive words + negative body/face + stress. Classic incongruence cluster.
        Positive sentiment is required — without it, there's no verbal-nonverbal gap.
        """
        positive_sentiment = _sig(signals, "sentiment_score", min_value=0.60)
        if positive_sentiment is None:
            return None
        components = {
            "positive_sentiment": positive_sentiment,
            "face_stress": (
                _sig(signals, "facial_stress", min_value=0.40)
                or _sig(signals, "tone_face_masking")
            ),
            "voice_stress":   _sig(signals, "vocal_stress_score", min_value=0.40),
            "gaze_avoidance": (
                _sig(signals, "gaze_direction_shift")
                or _sig(signals, "screen_contact", "low_screen_contact")
            ),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * 0.9, _CAPS["C-07"])
        return self._make_signal(
            "C-07", "verbal_nonverbal_discordance", speaker_id, score,
            "words_body_mismatch", confidence, ws, we,
            {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-08: Peak Performance ─────────────────────────────────────────────────
    def _c08_peak_performance(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        Confident tone + power language + upright posture + steady gaze + low stress.
        High stress disqualifies this pattern — it contradicts the peak state.
        """
        # Guard: window too short (single utterances like "Hello", "Okay" produce false hits)
        if (we - ws) < 3000:
            return None
        # Guard: high stress disqualifies
        if _sig(signals, "vocal_stress_score", min_value=0.55) is not None:
            return None
        # Guard: signals must come from at least 2 different agents (multimodal requirement)
        agents_present = {s.get("agent") for s in signals if s.get("agent")}
        if len(agents_present) < 2:
            return None
        components = {
            "confident_tone": (
                _sig(signals, "tone_classification", "confident")
                or _sig(signals, "tone_classification", "enthusiastic")
                or _sig(signals, "tone_classification", "excited")
            ),
            "power_language": _sig(signals, "power_language_score"),
            "upright_posture": _sig(signals, "posture", "upright_power_posture"),
            "steady_gaze": (
                _sig(signals, "attention_level", "high_attention")
                or _sig(signals, "screen_contact", "sustained_eye_contact")
            ),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * (len(hits) / 3.0) * 1.1, _CAPS["C-08"])
        return self._make_signal(
            "C-08", "peak_performance", speaker_id, score, "in_the_zone",
            confidence, ws, we, {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-09: Rapport Building ─────────────────────────────────────────────────
    def _c09_rapport_building(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        Rapport indicators + head nods + empathy language + warm tone + balanced talk.
        Tickle-Degnen & Rosenthal 1990: rapport = attention + positivity + coordination.
        """
        components = {
            "rapport": (
                _sig(signals, "rapport_indicator")
                or _sig(signals, "rapport_confirmation")
            ),
            "head_nod":         _sig(signals, "head_nod"),
            "empathy_language": _sig(signals, "empathy_language"),
            "warm_tone": (
                _sig(signals, "tone_classification", "warm")
                or _sig(signals, "tone_classification", "empathetic")
                or _sig(signals, "tone_classification", "enthusiastic")
                or _sig(signals, "tone_classification", "excited")
            ),
            "balanced_talk": _sig(signals, "conversation_balance", "well_balanced"),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * (len(hits) / 3.0) * 1.1, _CAPS["C-09"])
        return self._make_signal(
            "C-09", "rapport_building", speaker_id, score, "strong_rapport",
            confidence, ws, we, {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-10: Dominance Display ────────────────────────────────────────────────
    def _c10_dominance_display(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        High talk time + interruptions + dominant tone + forward lean + direct gaze.
        """
        components = {
            "dominant_speaker": _sig(signals, "dominance_score"),
            "interruption": (
                _sig(signals, "interruption_event")
                or _sig(signals, "interruption_pattern")
            ),
            "dominant_tone": (
                _sig(signals, "tone_classification", "dominant")
                or _sig(signals, "tone_classification", "assertive")
                or _sig(signals, "tone_classification", "aggressive")
                or _sig(signals, "tone_classification", "confident")
            ),
            "forward_lean":  _sig(signals, "body_lean", "forward_lean"),
            "direct_gaze":   _sig(signals, "screen_contact", "sustained_eye_contact"),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * (len(hits) / 3.0) * 1.1, _CAPS["C-10"])
        return self._make_signal(
            "C-10", "dominance_display", speaker_id, score, "dominance_display",
            confidence, ws, we, {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-11: Submission Signal ────────────────────────────────────────────────
    def _c11_submission_signal(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        Reduced talk + backward lean + gaze avoidance + low engagement.
        """
        # conversation_engagement/passive was previously in both low_talk and
        # low_engagement, letting the same signal satisfy two slots. Separated:
        # low_talk uses only the talk-time signal; low_engagement uses only attention.
        components = {
            "low_talk": _sig(signals, "dominance_score"),
            "backward_lean":  _sig(signals, "body_lean", "backward_lean"),
            "gaze_avoidance": (
                _sig(signals, "screen_contact", "low_screen_contact")
                or _sig(signals, "sustained_distraction")
            ),
            "low_engagement": (
                _sig(signals, "conversation_engagement", "passive")
                or _sig(signals, "attention_level", "reduced_attention")
            ),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 3 or _agent_diversity(hits) < 2:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        confidence = min(score * (len(hits) / 3.0) * 0.95, _CAPS["C-11"])
        return self._make_signal(
            "C-11", "submission_signal", speaker_id, score, "submission_signal",
            confidence, ws, we, {"components": list(hits.keys()), "hit_count": len(hits)},
        )

    # ── C-12: Deception Cluster (EXPERIMENTAL) ─────────────────────────────────
    def _c12_deception_cluster(
        self, speaker_id: str, signals: list[dict], ws: int, we: int
    ) -> Optional[dict]:
        """
        Multiple inconsistency indicators co-occurring.
        EXPERIMENTAL — never claimed as deception, only as "requires review".
        Capped at 0.50 per NEXUS design principle for deception-related signals.
        """
        components = {
            "stress_masking": (
                _sig(signals, "tone_face_masking")
                or _sig(signals, "stress_suppression")
            ),
            "gaze_breaks": (
                _sig(signals, "gaze_direction_shift")
                or _sig(signals, "sustained_distraction")
            ),
            "self_touch":          _sig(signals, "self_touch"),
            "filler_spike":        _sig(signals, "filler_detection"),
            "verbal_incongruence": (
                _sig(signals, "verbal_incongruence")
                or _sig(signals, "smile_sentiment_incongruence")
            ),
        }
        hits = {k: v for k, v in components.items() if v is not None}
        if len(hits) < 4 or _agent_diversity(hits) < 3:
            return None
        score = sum(h.get("confidence", 0.5) for h in hits.values()) / len(hits)
        # Conservative: no boost multiplier; hard cap at 0.50
        confidence = min(score * (len(hits) / 4.0), _CAPS["C-12"])
        return self._make_signal(
            "C-12", "deception_cluster", speaker_id, score,
            "multiple_inconsistency_indicators", confidence, ws, we,
            {
                "components": list(hits.keys()),
                "hit_count": len(hits),
                "experimental": True,
                "note": "Multiple inconsistency indicators detected — review recommended",
            },
        )
