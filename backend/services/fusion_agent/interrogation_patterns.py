# services/fusion_agent/interrogation_patterns.py
"""
Interrogation-specific fusion compound patterns (NEXUS INTERROGATION_UPDATES1.MD §6).

Two multi-phase behavioral cascade detectors that require cross-agent signals —
they cannot fire from a single modality.

Patterns implemented:
  CapitulationCascade    — Stress peak → freeze/silence → weakening denials.
                           Indicates subject breaking under sustained evidence
                           presentation (consistent with SUE framework §4).
                           Confidence: dynamic weighted fusion 0.20–0.60 (§4 spec)

  ResistanceHardening    — Sparse early resistance → dense late-session resistance.
                           Increased barrier, pronoun distancing, tense shifts
                           accumulating over the session indicates adopted strategy.
                           Confidence: dynamic weighted fusion 0.20–0.60 (§4 spec)

DESIGN PRINCIPLES:
  - Both patterns require signals from at least TWO distinct agents (cross-modal gate).
  - Confidence caps are deliberately low — behavioral cascades have many innocent
    explanations (fatigue, confusion, intimidation of the innocent).
  - These signals appear in the fusion output and feed into the narrative report.
    They do NOT generate alerts (below the 0.50 alert threshold in fusion_service.py).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("nexus.fusion.interrogation")

# ── Signal type sets ──────────────────────────────────────────────────────────

# Phase-1 stress/arousal signals that anchor the Capitulation Cascade
_STRESS_PEAK_TYPES: frozenset[str] = frozenset({
    "vocal_stress_score",
    "blink_suppression_spike",
    "vocal_agitation",
    "pitch_elevation",
    "voice_energy_change",
    "tone_shift",
})

# Phase-3 weakening/breakdown signals that complete the cascade
_CAPITULATION_TYPES: frozenset[str] = frozenset({
    "denial_weakening",
    "freezing_response",
    "speech_rate_anomaly",
    "statement_contamination",
})

# Resistance signals tracked across early vs late session for hardening detection
_RESISTANCE_TYPES: frozenset[str] = frozenset({
    "barrier_behavior",
    "pronoun_distancing",
    "tense_inconsistency",
    "motor_inhibition",
    "blink_suppression_spike",
})

# ── Tuning constants ──────────────────────────────────────────────────────────

# Minimum stress signals in a 60s anchor window to trigger cascade detection
_STRESS_MIN_COUNT = 2
# Maximum span from first Phase-1 signal to last Phase-3 signal (10 minutes)
_CASCADE_MAX_SPAN_MS = 600_000
# Maximum lag between Phase-1 anchor end and Phase-3 appearance (5 minutes)
_CASCADE_PHASE3_LAG_MS = 300_000

# ResistanceHardening: minimum late-session resistance signals required
_HARDENING_LATE_MIN = 3
# Factor by which late-session count must exceed early-session count
_HARDENING_RATIO = 2.0


class InterrogationCompoundPatterns:
    """
    Stateless detector for interrogation-specific multi-phase cascades.
    Call evaluate() once per session after all per-speaker signals are collected.
    """

    def evaluate(
        self,
        all_signals: list[dict],
        speakers: list[str],
        session_id: str = "",
    ) -> list[dict]:
        """
        Detect CapitulationCascade and ResistanceHardening for each speaker.

        all_signals — flat list of all voice + language + video + fusion signals.
        speakers    — speaker IDs to iterate; signals not in this list are ignored.

        Returns a flat list of new compound signal dicts tagged agent='fusion'.
        """
        results: list[dict] = []
        for spk in speakers:
            spk_signals = [s for s in all_signals if s.get("speaker_id") == spk]
            if not spk_signals:
                continue
            results.extend(self._capitulation_cascade(spk, spk_signals, session_id))
            results.extend(self._resistance_hardening(spk, spk_signals, session_id))

        if results:
            logger.info(
                "[%s] Interrogation compound patterns: %d signal(s) across %d speakers",
                session_id, len(results), len({s.get("speaker_id") for s in results}),
            )
        return results

    # ── CapitulationCascade ───────────────────────────────────────────────────

    def _capitulation_cascade(
        self,
        spk: str,
        signals: list[dict],
        session_id: str,
    ) -> list[dict]:
        """
        Detect: stress/arousal peak → freeze/pause gap → weakening denials.

        Algorithm:
          1. Collect all stress-peak signals and capitulation signals,
             sorted by window_start_ms.
          2. For each stress-peak cluster (≥2 signals within 60s), check whether
             any capitulation signal appears within _CASCADE_PHASE3_LAG_MS after
             the cluster anchor.
          3. Require cross-modal participation: at least 2 distinct agents in
             the combined set (stress cluster + capitulation signal).
          4. Emit one signal per detected cascade, spanning cluster start → phase3 end.
             Multiple cascades may fire if the pattern repeats.
        """
        stress_sigs = sorted(
            [s for s in signals if s.get("signal_type") in _STRESS_PEAK_TYPES],
            key=lambda s: _ms(s, "window_start_ms"),
        )
        cap_sigs = sorted(
            [s for s in signals if s.get("signal_type") in _CAPITULATION_TYPES],
            key=lambda s: _ms(s, "window_start_ms"),
        )

        if not stress_sigs or not cap_sigs:
            return []

        # Build stress-peak clusters: windows where ≥2 stress signals fall within 60s
        clusters: list[tuple[int, int, list[dict]]] = []  # (anchor_start, anchor_end, members)
        ANCHOR_WINDOW_MS = 60_000
        i = 0
        while i < len(stress_sigs):
            anchor_start = _ms(stress_sigs[i], "window_start_ms")
            cluster = [stress_sigs[i]]
            j = i + 1
            while j < len(stress_sigs):
                if _ms(stress_sigs[j], "window_start_ms") - anchor_start <= ANCHOR_WINDOW_MS:
                    cluster.append(stress_sigs[j])
                    j += 1
                else:
                    break
            if len(cluster) >= _STRESS_MIN_COUNT:
                anchor_end = max(_ms(s, "window_end_ms") for s in cluster)
                clusters.append((anchor_start, anchor_end, cluster))
            i = j if j > i + 1 else i + 1

        if not clusters:
            return []

        results: list[dict] = []
        fired_anchors: set[int] = set()  # de-duplicate on anchor_start

        for anchor_start, anchor_end, cluster in clusters:
            if anchor_start in fired_anchors:
                continue

            # Find capitulation signal after the cluster
            phase3 = None
            for cap in cap_sigs:
                cap_start = _ms(cap, "window_start_ms")
                if cap_start < anchor_end:
                    continue
                lag = cap_start - anchor_end
                if lag > _CASCADE_PHASE3_LAG_MS:
                    break
                phase3 = cap
                break

            if phase3 is None:
                continue

            # Cross-modal gate: combined set must span ≥2 distinct agents
            involved = cluster + [phase3]
            agents = {s.get("agent", "unknown") for s in involved}
            if len(agents) < 2:
                continue

            total_span_ms = _ms(phase3, "window_end_ms") - anchor_start
            if total_span_ms > _CASCADE_MAX_SPAN_MS:
                continue

            fired_anchors.add(anchor_start)
            evidence_types = sorted({s.get("signal_type", "") for s in involved})
            evidence_confs = [s.get("confidence", 0.5) for s in involved]
            fused_conf = _weighted_fuse(evidence_confs, 0.60)

            results.append({
                "agent":           "fusion",
                "speaker_id":      spk,
                "signal_type":     "capitulation_cascade",
                "value":           round(total_span_ms / 60_000, 2),
                "value_text":      "stress_peak_to_weakening",
                "confidence":      fused_conf,
                "window_start_ms": anchor_start,
                "window_end_ms":   _ms(phase3, "window_end_ms"),
                "metadata": {
                    "rule_id":            "INTERROG-FUSION-01",
                    "pattern":            "CapitulationCascade",
                    "anchor_agents":      sorted(agents),
                    "stress_signal_count": len(cluster),
                    "phase3_signal_type": phase3.get("signal_type"),
                    "evidence_types":     evidence_types,
                    "evidence_confidences": [round(c, 3) for c in evidence_confs],
                    "fusion_note":        f"Weighted fusion of {len(evidence_confs)} signals (cap=0.60 per §5)",
                    "span_minutes":       round(total_span_ms / 60_000, 1),
                    "interpretation":     (
                        "Behavioral cascade consistent with sustained interrogation pressure: "
                        "stress/arousal peak followed by behavioral freeze, then weakening resistance. "
                        "Commonly observed in both guilty and innocent subjects under prolonged pressure."
                    ),
                    "context": (
                        "Pattern does NOT indicate deception. Innocent subjects under high-stakes "
                        "accusation exhibit this cascade due to fear and cognitive overload. "
                        "Consistent with SUE framework evidence confrontation response."
                    ),
                    "recommendation": (
                        "Note timing relative to evidence presentation. Compare with baseline "
                        "session stress level. Cross-reference with linguistic denial evolution."
                    ),
                },
            })
            logger.info(
                "[%s] CapitulationCascade: %s anchor=%ds span=%.1fmin agents=%s",
                session_id, spk, anchor_start // 1000,
                total_span_ms / 60_000, sorted(agents),
            )

        return results

    # ── ResistanceHardening ───────────────────────────────────────────────────

    def _resistance_hardening(
        self,
        spk: str,
        signals: list[dict],
        session_id: str,
    ) -> list[dict]:
        """
        Detect: sparse early-session resistance → dense late-session resistance.

        Algorithm:
          1. Collect resistance-type signals sorted chronologically.
          2. Split at the session midpoint (median window_start_ms).
          3. If late_count >= _HARDENING_LATE_MIN AND late_count >= _HARDENING_RATIO
             * early_count, pattern fires.
          4. Cross-modal gate: resistance signals must span ≥2 distinct agents.
          5. Emits one signal per speaker spanning the late-session window.
        """
        res_sigs = sorted(
            [s for s in signals if s.get("signal_type") in _RESISTANCE_TYPES],
            key=lambda s: _ms(s, "window_start_ms"),
        )
        if len(res_sigs) < _HARDENING_LATE_MIN:
            return []

        all_starts = sorted(_ms(s, "window_start_ms") for s in signals)
        if not all_starts:
            return []
        session_start_ms = all_starts[0]
        session_end_ms   = max(_ms(s, "window_end_ms") for s in signals)
        midpoint_ms      = (session_start_ms + session_end_ms) // 2

        early = [s for s in res_sigs if _ms(s, "window_start_ms") < midpoint_ms]
        late  = [s for s in res_sigs if _ms(s, "window_start_ms") >= midpoint_ms]

        if len(late) < _HARDENING_LATE_MIN:
            return []
        if len(late) < _HARDENING_RATIO * max(len(early), 1):
            return []

        # Cross-modal gate
        all_res_agents = {s.get("agent", "unknown") for s in res_sigs}
        if len(all_res_agents) < 2:
            return []

        late_start  = _ms(late[0],  "window_start_ms")
        late_end    = _ms(late[-1], "window_end_ms")
        evidence_types = sorted({s.get("signal_type", "") for s in late})
        late_confs = [s.get("confidence", 0.5) for s in late]
        fused_conf = _weighted_fuse(late_confs, 0.60)

        logger.info(
            "[%s] ResistanceHardening: %s early=%d late=%d fused_conf=%.3f agents=%s",
            session_id, spk, len(early), len(late), fused_conf, sorted(all_res_agents),
        )
        return [{
            "agent":           "fusion",
            "speaker_id":      spk,
            "signal_type":     "resistance_hardening",
            "value":           round(len(late) / max(len(early), 1), 2),
            "value_text":      "resistance_increasing",
            "confidence":      fused_conf,
            "window_start_ms": late_start,
            "window_end_ms":   late_end,
            "metadata": {
                "rule_id":             "INTERROG-FUSION-02",
                "pattern":             "ResistanceHardening",
                "early_count":         len(early),
                "late_count":          len(late),
                "late_to_early_ratio": round(len(late) / max(len(early), 1), 2),
                "late_evidence_types": evidence_types,
                "late_confidences":    [round(c, 3) for c in late_confs],
                "fusion_note":         f"Weighted fusion of {len(late_confs)} late-session signals (cap=0.60 per §5)",
                "agents_involved":     sorted(all_res_agents),
                "interpretation": (
                    "Resistance signals accumulate significantly in the latter half of the session. "
                    "Suggests subject has adopted or strengthened a defensive behavioral strategy "
                    "as the interrogation progressed."
                ),
                "context": (
                    "Pattern may reflect: (a) deliberate hardening of deceptive strategy, "
                    "(b) legitimate frustration and distrust from an innocent subject under "
                    "prolonged accusation, (c) fatigue-driven withdrawal. "
                    "Cannot distinguish between these interpretations behaviorally."
                ),
                "recommendation": (
                    "Compare with linguistic signals (pronoun_distancing, denial_weakening trend). "
                    "Note whether resistance hardened before or after specific evidence was presented."
                ),
            },
        }]


# ── FalseConfessionRiskAssessor ──────────────────────────────────────────────

class FalseConfessionRiskAssessor:
    """
    Session-level false confession risk model (INTERROG-FUSION-03).

    Synthesises behavioral, linguistic, and temporal signals into a single
    risk score per suspect speaker.  Factors follow Kassin (2010) and
    Drizin & Leo (2004) — the five most reliably replicated correlates of
    proven false confessions:

      F1  Session duration          — > 6h is a documented risk factor
      F2  Statement contamination   — Garrett 2011: 97.5% in proven cases
      F3  Capitulation cascade      — behavioural breakdown under pressure
      F4  Denial evolution          — declining denial strength trajectory
      F5  Processing delay density  — repeated long latency after evidence
      F6  Resistance hardening      — ABSENT = higher risk (protective if present)
      F7  Resistance building       — complete absence = risk (normalising accusation)

    Confidence deliberately capped at 0.55 — this is a risk indicator, not a
    determination.  Produced once per session per speaker.
    """

    _CONTAMINATION_TYPES  = frozenset({"statement_contamination"})
    _CAPITULATION_TYPES   = frozenset({"capitulation_cascade"})
    _DENIAL_TYPES         = frozenset({"denial_weakening"})
    _HARDENING_TYPES      = frozenset({"resistance_hardening"})
    _PROCESSING_TYPES     = frozenset({"evidence_response_processing_delay"})
    _RESISTANCE_BUILDING  = frozenset({
        "barrier_behavior", "pronoun_distancing",
        "tense_inconsistency", "motor_inhibition",
    })

    def evaluate(
        self,
        all_signals: list[dict],
        speakers: list[str],
        session_id: str = "",
    ) -> list[dict]:
        """
        Assess false confession risk for each speaker.

        all_signals — flat list of all session signals from all agents/fusion.
        speakers    — speaker IDs to assess (all; interrogator naturally scores near-zero).

        Returns one signal per speaker that has enough data for an assessment.
        """
        if not all_signals or not speakers:
            return []

        all_ends   = [_ms(s, "window_end_ms")   for s in all_signals if _ms(s, "window_end_ms")   > 0]
        all_starts = [_ms(s, "window_start_ms") for s in all_signals if _ms(s, "window_start_ms") >= 0]
        session_duration_ms = (max(all_ends) - min(all_starts)) if all_ends and all_starts else 0

        results: list[dict] = []
        for spk in speakers:
            spk_signals = [s for s in all_signals if s.get("speaker_id") == spk]
            sig = self._assess_speaker(spk, spk_signals, session_duration_ms, session_id)
            if sig is not None:
                results.append(sig)

        if results:
            logger.info(
                "[%s] FalseConfessionRisk: %d assessment(s)",
                session_id, len(results),
            )
        return results

    def _assess_speaker(
        self,
        spk: str,
        signals: list[dict],
        session_duration_ms: int,
        session_id: str,
    ) -> dict | None:
        if len(signals) < 5:
            return None

        factors: dict = {}
        risk_score = 0.0
        max_possible = 0.0

        # F1: session duration
        duration_h = session_duration_ms / 3_600_000
        dur_contrib = 0.15 if duration_h >= 6.0 else (0.08 if duration_h >= 2.0 else 0.0)
        dur_level   = "high" if duration_h >= 6.0 else ("moderate" if duration_h >= 2.0 else "low")
        factors["duration_risk"] = {"value_h": round(duration_h, 2), "level": dur_level, "contribution": dur_contrib}
        risk_score   += dur_contrib
        max_possible += 0.15

        # F2: contamination (highest-weight single factor per Garrett 2011)
        cont_sigs = [s for s in signals if s.get("signal_type") in self._CONTAMINATION_TYPES]
        cont_contrib = min(0.25, len(cont_sigs) * 0.10)
        factors["contamination"] = {
            "signal_count": len(cont_sigs),
            "contribution": round(cont_contrib, 3),
            "research": "Garrett 2011 — present in 97.5% of proven false confessions",
        }
        risk_score   += cont_contrib
        max_possible += 0.25

        # F3: capitulation cascade
        cap_sigs = [s for s in signals if s.get("signal_type") in self._CAPITULATION_TYPES]
        cap_contrib = min(0.20, len(cap_sigs) * 0.10)
        factors["capitulation_cascade"] = {
            "signal_count": len(cap_sigs),
            "contribution": round(cap_contrib, 3),
            "research": "Kassin & Kiechel 1996 — behavioural breakdown predicts false confession",
        }
        risk_score   += cap_contrib
        max_possible += 0.20

        # F4: denial evolution — weakening signals
        denial_sigs = [s for s in signals if s.get("signal_type") in self._DENIAL_TYPES]
        denial_contrib = min(0.15, len(denial_sigs) * 0.05)
        factors["denial_evolution"] = {
            "weakening_count": len(denial_sigs),
            "contribution": round(denial_contrib, 3),
            "research": "Horvath 1973; Porter & Yuille 1996",
        }
        risk_score   += denial_contrib
        max_possible += 0.15

        # F5: repeated processing delays
        delay_sigs = [s for s in signals if s.get("signal_type") in self._PROCESSING_TYPES]
        delay_contrib = min(0.15, len(delay_sigs) * 0.03)
        factors["processing_delays"] = {
            "signal_count": len(delay_sigs),
            "contribution": round(delay_contrib, 3),
        }
        risk_score   += delay_contrib
        max_possible += 0.15

        # F6: resistance hardening (PROTECTIVE — reduces risk if present)
        hard_sigs = [s for s in signals if s.get("signal_type") in self._HARDENING_TYPES]
        if hard_sigs:
            risk_score -= 0.05
        factors["resistance_hardening"] = {
            "present": bool(hard_sigs),
            "signal_count": len(hard_sigs),
            "effect": "protective (−0.05)" if hard_sigs else "absent",
        }

        # F7: complete absence of defensive resistance signals (normalisation risk)
        build_sigs = [s for s in signals if s.get("signal_type") in self._RESISTANCE_BUILDING]
        build_absent_contrib = 0.10 if not build_sigs else 0.0
        factors["resistance_building"] = {
            "signal_count": len(build_sigs),
            "absent_contribution": round(build_absent_contrib, 3),
        }
        risk_score   += build_absent_contrib
        max_possible += 0.10

        # Normalise to 0-1
        risk_score = max(0.0, min(risk_score / max_possible, 1.0)) if max_possible > 0 else 0.0

        # Confidence scales with how many factors have evidence
        bearing = sum(1 for f in factors.values() if isinstance(f, dict) and (
            f.get("signal_count", 0) > 0 or f.get("weakening_count", 0) > 0
        ))
        if bearing == 0 and len(cont_sigs) == 0 and len(cap_sigs) == 0:
            return None

        confidence = round(min(0.55, 0.12 + bearing * 0.07), 4)

        if risk_score >= 0.65:
            value_text = "high_risk"
        elif risk_score >= 0.35:
            value_text = "moderate_risk"
        else:
            value_text = "low_risk"

        session_start = min((_ms(s, "window_start_ms") for s in signals), default=0)
        session_end   = max((_ms(s, "window_end_ms")   for s in signals), default=0)

        logger.info(
            "[%s] FalseConfessionRisk: %s risk=%s score=%.3f conf=%.3f (bearing_factors=%d)",
            session_id, spk, value_text, risk_score, confidence, bearing,
        )
        return {
            "agent":           "fusion",
            "speaker_id":      spk,
            "signal_type":     "false_confession_risk",
            "value":           round(risk_score, 4),
            "value_text":      value_text,
            "confidence":      confidence,
            "window_start_ms": session_start,
            "window_end_ms":   session_end,
            "metadata": {
                "rule_id":            "INTERROG-FUSION-03",
                "pattern":            "FalseConfessionRiskAssessor",
                "session_duration_h": round(session_duration_ms / 3_600_000, 2),
                "risk_factors":       factors,
                "bearing_factor_count": bearing,
                "interpretation": (
                    "Multi-factor risk score synthesising behavioural, linguistic, and temporal signals. "
                    "High-risk score indicates convergence of documented false confession correlates."
                ),
                "context": (
                    "CRITICAL: This is a risk INDICATOR, not a determination that a false confession "
                    "occurred or will occur. Proven correlates include prolonged interrogation, "
                    "statement contamination, and behavioural capitulation cascades. "
                    "Research: Kassin (2010), Drizin & Leo (2004), Garrett (2011)."
                ),
                "recommendation": (
                    "High-risk sessions require independent legal review before acting on any "
                    "resulting confession. Compare contamination signals with confirmed case facts. "
                    "Note whether behavioural capitulation preceded linguistic acquiescence."
                ),
            },
        }


# ── Utility ───────────────────────────────────────────────────────────────────

def _ms(signal: dict, key: str) -> int:
    val = signal.get(key, 0)
    try:
        return int(float(val)) if val is not None else 0
    except (ValueError, TypeError):
        return 0


def _weighted_fuse(confidences: list[float], cap: float) -> float:
    """
    Bayesian fusion per §4 spec.

    Weight tiers (from spec §4 MultiRuleFusion):
      high   conf >= 0.60  → weight 1.0
      medium conf >= 0.30  → weight 0.5
      low    conf <  0.30  → weight 0.2

    Bonus: 3+ low-confidence signals converging adds +0.10 (spec: 'weak signals reinforce').
    Result capped at `cap` (pattern-level maximum).
    """
    if not confidences:
        return 0.0
    total_weight = total_score = 0.0
    low_count = 0
    for c in confidences:
        if c >= 0.60:
            w = 1.0
        elif c >= 0.30:
            w = 0.5
        else:
            w = 0.2
            low_count += 1
        total_weight += w
        total_score += c * w
    fused = total_score / total_weight if total_weight > 0 else 0.0
    if low_count >= 3:
        fused += 0.10
    return round(min(fused, cap), 4)
