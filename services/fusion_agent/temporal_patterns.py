"""
Temporal Pattern Engine — Phase 2G
Implements 8 session-level trajectory patterns (T-01 through T-08).

These patterns detect how behaviour CHANGES over time — not what's happening
at a given moment, but whether signals are rising, decaying, shifting, or
following a known model (Glasl escalation, Gottman stress trajectory, etc.).

Minimum session length: 60 seconds. Shorter sessions skip all patterns.

Research anchors:
  Glasl 1982             — 9-stage conflict escalation model (T-08)
  Gottman 1994           — Stress trajectories in high-stakes conversations (T-01)
  Mehrabian 1972         — Progressive engagement/disengagement cues (T-02)
  Chartrand & Bargh 1999 — Behavioural adaptation over time (T-05)
"""
import logging
import math
from typing import Optional

logger = logging.getLogger("nexus.fusion.temporal")


def _linear_slope(values: list[float]) -> float:
    """
    Simple least-squares linear regression slope.
    Returns slope per unit step (index). DSA: O(N) single pass.
    """
    n = len(values)
    if n < 2:
        return 0.0
    sum_x  = n * (n - 1) / 2
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6
    sum_y  = sum(values)
    sum_xy = sum(i * v for i, v in enumerate(values))
    denom  = n * sum_x2 - sum_x ** 2
    return 0.0 if denom == 0 else (n * sum_xy - sum_x * sum_y) / denom


def _bucket_signals(
    signals: list[dict],
    signal_type: str,
    n_buckets: int,
    session_start: int,
    session_end: int,
) -> list[float]:
    """
    Divide the session into N equal time buckets, return mean signal value per bucket.
    Empty buckets are forward/backward-filled from neighbours.
    DSA: O(S) to fill buckets, O(B) to fill gaps.
    """
    if session_end <= session_start or n_buckets < 2:
        return []

    bucket_ms: float = (session_end - session_start) / n_buckets
    buckets: list[list[float]] = [[] for _ in range(n_buckets)]

    for s in signals:
        if s.get("signal_type") != signal_type:
            continue
        t = (s.get("window_start_ms", 0) + s.get("window_end_ms", 0)) / 2.0
        idx = min(int((t - session_start) / bucket_ms), n_buckets - 1)
        idx = max(idx, 0)
        val = s.get("value")
        if val is not None:
            try:
                buckets[idx].append(float(val))
            except (TypeError, ValueError):
                pass

    result = [sum(b) / len(b) if b else float("nan") for b in buckets]

    # Forward fill
    last = 0.0
    for i, v in enumerate(result):
        if not math.isnan(v):
            last = v
        else:
            result[i] = last
    # Backward fill for leading NaNs
    last = 0.0
    for i in range(len(result) - 1, -1, -1):
        if not math.isnan(result[i]):
            last = result[i]
        else:
            result[i] = last

    return result


class TemporalPatternEngine:
    """
    Session-level trajectory analyser.

    All patterns receive the full signal list for one speaker and the session
    time bounds, then compute trends using bucketed time-series.

    DSA: O(S × B) overall where S = signal count, B = n_buckets (constant 10).
    """

    AGENT_NAME = "fusion"
    N_BUCKETS = 10  # divide session into 10 equal blocks

    def evaluate(
        self,
        speaker_id: str,
        all_signals: list[dict],
        session_start_ms: int,
        session_end_ms: int,
    ) -> list[dict]:
        if session_end_ms - session_start_ms < 60_000:
            return []  # need at least 60s for temporal patterns

        patterns = [
            self._t01_stress_trajectory,
            self._t02_engagement_decay,
            self._t03_rapport_evolution,
            self._t04_behavioral_shift,
            self._t05_adaptation_pattern,
            self._t06_fatigue_detection,
            self._t07_recovery_pattern,
            self._t08_escalation_ladder,
        ]

        results: list[dict] = []
        for fn in patterns:
            result = fn(speaker_id, all_signals, session_start_ms, session_end_ms)
            if result:
                results.append(result)

        if results:
            logger.debug(
                f"[{speaker_id}] TemporalPatternEngine: {len(results)} temporal patterns fired"
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
        cap: float = 0.70,
        metadata: Optional[dict] = None,
    ) -> dict:
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

    # ── T-01: Stress Trajectory ────────────────────────────────────────────────
    def _t01_stress_trajectory(
        self, speaker_id: str, signals: list[dict], ss: int, se: int
    ) -> Optional[dict]:
        """
        Track combined voice + facial stress over session buckets.
        Rising delta → stress building; falling → recovering.
        Gottman 1994: sustained stress trajectory predicts conversation outcome.
        """
        voice_b = _bucket_signals(signals, "vocal_stress_score", self.N_BUCKETS, ss, se)
        face_b  = _bucket_signals(signals, "facial_stress",       self.N_BUCKETS, ss, se)

        n = max(len(voice_b), len(face_b))
        if n == 0:
            return None

        combined = [
            max(
                voice_b[i] if i < len(voice_b) else 0.0,
                face_b[i]  if i < len(face_b)  else 0.0,
            )
            for i in range(n)
        ]

        if not any(v > 0.01 for v in combined):
            return None

        early_avg = sum(combined[:3]) / 3
        late_avg  = sum(combined[-3:]) / 3
        delta     = late_avg - early_avg

        if abs(delta) < 0.08:
            return None

        slope = _linear_slope(combined)
        if delta > 0:
            label      = "stress_trajectory_rising"
            confidence = min(slope * 3.0, 0.70)
        else:
            label      = "stress_trajectory_declining"
            confidence = min(abs(slope) * 3.0, 0.65)

        return self._make_signal(
            "T-01", "stress_trajectory", speaker_id, round(delta, 4), label,
            confidence, ss, se, cap=0.70,
            metadata={
                "early_avg": round(early_avg, 4),
                "late_avg":  round(late_avg, 4),
                "slope":     round(slope, 6),
                "buckets":   [round(v, 3) for v in combined],
            },
        )

    # ── T-02: Engagement Decay ─────────────────────────────────────────────────
    def _t02_engagement_decay(
        self, speaker_id: str, signals: list[dict], ss: int, se: int
    ) -> Optional[dict]:
        """
        Progressive disengagement over the session.
        Counts engagement-related signals per time bucket; declining count = decay.
        """
        bucket_ms = (se - ss) / self.N_BUCKETS
        eng_counts = [0.0] * self.N_BUCKETS
        ENGAGEMENT_TYPES = frozenset({
            "attention_level", "facial_engagement", "conversation_engagement",
        })

        for s in signals:
            if s.get("signal_type") not in ENGAGEMENT_TYPES:
                continue
            t   = (s.get("window_start_ms", 0) + s.get("window_end_ms", 0)) / 2.0
            idx = min(max(int((t - ss) / bucket_ms), 0), self.N_BUCKETS - 1)
            eng_counts[idx] += s.get("confidence", 0.5)

        if max(eng_counts) < 0.1:
            return None

        early_avg = sum(eng_counts[:3]) / 3
        late_avg  = sum(eng_counts[-3:]) / 3
        delta     = late_avg - early_avg

        if delta > -0.20:
            return None  # not enough decay

        slope      = _linear_slope(eng_counts)
        confidence = min(abs(slope) * 2.0, 0.65)

        return self._make_signal(
            "T-02", "engagement_decay", speaker_id, round(delta, 4),
            "engagement_decaying", confidence, ss, se, cap=0.65,
            metadata={
                "early_avg":     round(early_avg, 4),
                "late_avg":      round(late_avg, 4),
                "slope":         round(slope, 6),
                "bucket_counts": [round(v, 3) for v in eng_counts],
            },
        )

    # ── T-03: Rapport Evolution ────────────────────────────────────────────────
    def _t03_rapport_evolution(
        self, speaker_id: str, signals: list[dict], ss: int, se: int
    ) -> Optional[dict]:
        """
        Track rapport signals across the session to detect strengthening or decline.
        """
        rapport_sigs = [
            s for s in signals
            if s.get("signal_type") in ("rapport_indicator", "rapport_confirmation")
        ]
        if len(rapport_sigs) < 3:
            return None

        bucket_ms = (se - ss) / self.N_BUCKETS
        buckets: list[list[float]] = [[] for _ in range(self.N_BUCKETS)]
        for s in rapport_sigs:
            t   = (s.get("window_start_ms", 0) + s.get("window_end_ms", 0)) / 2.0
            idx = min(max(int((t - ss) / bucket_ms), 0), self.N_BUCKETS - 1)
            val = s.get("value") or s.get("confidence") or 0.0
            buckets[idx].append(float(val))

        raw = [sum(b) / len(b) if b else float("nan") for b in buckets]
        non_nan = [v for v in raw if not math.isnan(v)]
        if len(non_nan) < 2:
            return None

        # Forward fill
        filled: list[float] = []
        last = non_nan[0]
        for v in raw:
            if not math.isnan(v):
                last = v
            filled.append(last)

        early_avg = sum(filled[:3]) / 3
        late_avg  = sum(filled[-3:]) / 3
        delta     = late_avg - early_avg

        if abs(delta) < 0.05:
            return None

        slope = _linear_slope(filled)
        if delta > 0:
            label      = "rapport_strengthening"
            confidence = min(slope * 5.0, 0.65)
        else:
            label      = "rapport_declining"
            confidence = min(abs(slope) * 5.0, 0.60)

        return self._make_signal(
            "T-03", "rapport_evolution", speaker_id, round(delta, 4), label,
            confidence, ss, se, cap=0.65,
            metadata={
                "early_avg":           round(early_avg, 4),
                "late_avg":            round(late_avg, 4),
                "slope":               round(slope, 6),
                "rapport_signal_count": len(rapport_sigs),
            },
        )

    # ── T-04: Behavioral Shift Point ───────────────────────────────────────────
    def _t04_behavioral_shift(
        self, speaker_id: str, signals: list[dict], ss: int, se: int
    ) -> Optional[dict]:
        """
        Detect the moment in the session where behavioural signals changed most.
        Uses a 2-minute rolling comparison window on a combined stress-engagement trace.
        """
        MINI_BUCKET_MS   = 10_000   # 10s mini-buckets
        SHIFT_WINDOW_MS  = 120_000  # compare 2-min blocks on each side
        MIN_DELTA        = 0.15

        n_mini = max(1, int((se - ss) / MINI_BUCKET_MS))
        stress_trace = [0.0] * n_mini
        engage_trace = [0.0] * n_mini

        for s in signals:
            t   = (s.get("window_start_ms", 0) + s.get("window_end_ms", 0)) / 2.0
            idx = min(max(int((t - ss) / MINI_BUCKET_MS), 0), n_mini - 1)
            st  = s.get("signal_type", "")
            if st == "vocal_stress_score":
                stress_trace[idx] = max(stress_trace[idx], s.get("value") or 0.0)
            elif st in ("attention_level", "facial_engagement"):
                engage_trace[idx] = max(engage_trace[idx], s.get("confidence") or 0.0)

        combined = [s * 1.2 - e * 0.8 for s, e in zip(stress_trace, engage_trace)]
        shift_buckets = max(1, int(SHIFT_WINDOW_MS / MINI_BUCKET_MS))

        max_delta = 0.0
        shift_idx = 0
        for i in range(shift_buckets, n_mini - shift_buckets):
            before = sum(combined[i - shift_buckets: i]) / shift_buckets
            after  = sum(combined[i: i + shift_buckets]) / shift_buckets
            d = abs(after - before)
            if d > max_delta:
                max_delta = d
                shift_idx = i

        if max_delta < MIN_DELTA:
            return None

        shift_ms   = ss + shift_idx * MINI_BUCKET_MS
        confidence = min(max_delta * 1.5, 0.65)

        return self._make_signal(
            "T-04", "behavioral_shift", speaker_id, round(max_delta, 4),
            "behavioral_shift_detected", confidence, shift_ms, se, cap=0.65,
            metadata={
                "shift_time_ms":      shift_ms,
                "shift_time_seconds": round(shift_ms / 1000),
                "max_delta":          round(max_delta, 4),
            },
        )

    # ── T-05: Adaptation Pattern ───────────────────────────────────────────────
    def _t05_adaptation_pattern(
        self, speaker_id: str, signals: list[dict], ss: int, se: int
    ) -> Optional[dict]:
        """
        Increasing engagement + decreasing fidgeting over time = positive adaptation.
        Speaker is adjusting to the conversation (Chartrand & Bargh 1999).
        """
        eng_b    = _bucket_signals(signals, "attention_level", self.N_BUCKETS, ss, se)
        fidget_b = _bucket_signals(signals, "body_fidgeting",  self.N_BUCKETS, ss, se)

        has_eng    = eng_b    and max(eng_b)    > 0.01
        has_fidget = fidget_b and max(fidget_b) > 0.01

        if not has_eng and not has_fidget:
            return None

        adapt_signals: list[tuple[str, float]] = []
        if has_eng:
            slope_e = _linear_slope(eng_b)
            if slope_e > 0.005:
                adapt_signals.append(("engagement_increase", slope_e))
        if has_fidget:
            slope_f = _linear_slope(fidget_b)
            if slope_f < -0.002:
                adapt_signals.append(("fidget_decrease", abs(slope_f)))

        if not adapt_signals:
            return None

        max_slope  = max(v for _, v in adapt_signals)
        confidence = min(max_slope * 8.0, 0.60)

        return self._make_signal(
            "T-05", "adaptation_pattern", speaker_id, round(max_slope, 6),
            "positive_adaptation", confidence, ss, se, cap=0.60,
            metadata={
                "adaptation_types": [t for t, _ in adapt_signals],
                "max_slope":        round(max_slope, 6),
            },
        )

    # ── T-06: Fatigue Detection ────────────────────────────────────────────────
    def _t06_fatigue_detection(
        self, speaker_id: str, signals: list[dict], ss: int, se: int
    ) -> Optional[dict]:
        """
        Declining energy in the last 30% of the session.
        Compares: gesture activity, speech rate, and attention in early vs late session.
        """
        late_start = ss + int((se - ss) * 0.70)

        early_sigs = [s for s in signals if s.get("window_end_ms",   0) <= late_start]
        late_sigs  = [s for s in signals if s.get("window_start_ms", 0) >= late_start]

        if not early_sigs or not late_sigs:
            return None

        def _energy(sigs: list[dict]) -> float:
            score, n = 0.0, 0
            for s in sigs:
                st = s.get("signal_type", "")
                if st == "gesture_animation":
                    score += s.get("value") or 0.3; n += 1
                elif st == "speech_rate_anomaly":
                    if s.get("value_text") == "rate_elevated":
                        score += 0.6; n += 1
                    elif s.get("value_text") == "rate_suppressed":
                        score -= 0.4; n += 1
                elif st == "attention_level" and s.get("value_text") == "high_attention":
                    score += s.get("value") or 0.7; n += 1
            return score / max(n, 1)

        early_e = _energy(early_sigs)
        late_e  = _energy(late_sigs)
        delta   = late_e - early_e

        if delta > -0.10:
            return None

        confidence = min(abs(delta) * 1.5, 0.60)

        return self._make_signal(
            "T-06", "fatigue_detection", speaker_id, round(delta, 4),
            "energy_declining", confidence, late_start, se, cap=0.60,
            metadata={
                "early_energy": round(early_e, 4),
                "late_energy":  round(late_e,  4),
                "energy_delta": round(delta,   4),
            },
        )

    # ── T-07: Recovery Pattern ─────────────────────────────────────────────────
    def _t07_recovery_pattern(
        self, speaker_id: str, signals: list[dict], ss: int, se: int
    ) -> Optional[dict]:
        """
        Detect return to baseline after stress spikes within 5 minutes.
        Fast recovery (< 90s) = good emotional regulation.
        """
        SPIKE_THRESHOLD   = 0.60
        RECOVERY_WINDOW   = 300_000  # 5 min
        MIN_RECOVERY      = 30_000   # 30s

        stress_spikes = sorted(
            [s for s in signals
             if s.get("signal_type") == "vocal_stress_score"
             and (s.get("value") or 0.0) >= SPIKE_THRESHOLD],
            key=lambda s: s.get("window_start_ms", 0),
        )
        if not stress_spikes:
            return None

        recoveries: list[float] = []
        for spike in stress_spikes:
            spike_end    = spike.get("window_end_ms", 0)
            recovery_end = spike_end + RECOVERY_WINDOW

            post_sigs = [
                s for s in signals
                if s.get("signal_type") == "vocal_stress_score"
                and s.get("window_start_ms", 0) > spike_end
                and s.get("window_start_ms", 0) <= recovery_end
            ]
            if not post_sigs:
                continue

            low_stress = [s for s in post_sigs if (s.get("value") or 1.0) < 0.40]
            if not low_stress:
                continue

            first_low   = min(low_stress, key=lambda s: s.get("window_start_ms", 0))
            recovery_ms = first_low.get("window_start_ms", 0) - spike_end
            if recovery_ms >= MIN_RECOVERY:
                recoveries.append(float(recovery_ms))

        if not recoveries:
            return None

        avg_s = sum(recoveries) / len(recoveries) / 1000.0
        # Faster recovery → higher confidence that regulation occurred
        confidence = min(0.45 + (300.0 - min(avg_s, 300.0)) / 600.0, 0.65)

        first_spike = stress_spikes[0]
        return self._make_signal(
            "T-07", "stress_recovery", speaker_id, round(avg_s, 2),
            "good_stress_recovery", confidence,
            first_spike.get("window_start_ms", ss), se, cap=0.65,
            metadata={
                "recovery_count":        len(recoveries),
                "avg_recovery_seconds":  round(avg_s, 2),
                "spike_count":           len(stress_spikes),
            },
        )

    # ── T-08: Escalation Ladder ────────────────────────────────────────────────
    def _t08_escalation_ladder(
        self, speaker_id: str, signals: list[dict], ss: int, se: int
    ) -> Optional[dict]:
        """
        Progressive conflict escalation using Glasl (1982) 3-stage model:
          Stage 1 (objective tension): objections, head shakes, disagreements
          Stage 2 (personal):          stress spikes, interruptions
          Stage 3 (hostile):           aggressive/contemptuous tone, angry facial expression
        """
        stage1 = sum(
            1 for s in signals
            if s.get("signal_type") in ("objection_signal", "objection_detected", "head_shake")
        )
        stage2 = sum(
            1 for s in signals
            if s.get("signal_type") in ("vocal_stress_score", "interruption_event")
            and (s.get("value") or 0.0) > 0.55
        )
        stage3 = sum(
            1 for s in signals
            if (
                s.get("signal_type") == "tone_classification"
                and s.get("value_text") in ("confrontational", "contemptuous")
            ) or (
                s.get("signal_type") == "facial_emotion"
                and s.get("value_text") in ("angry", "contempt", "disgusted")
            )
        )

        if stage3 >= 2:
            stage_label = "escalation_stage_3_hostile"
            stage_num   = 3
        elif stage2 >= 2 or (stage2 >= 1 and stage1 >= 3):
            stage_label = "escalation_stage_2_personal"
            stage_num   = 2
        elif stage1 >= 3:
            stage_label = "escalation_stage_1_objective"
            stage_num   = 1
        else:
            return None

        total      = stage1 + stage2 + stage3
        confidence = min(total * 0.06, 0.55)

        return self._make_signal(
            "T-08", "escalation_ladder", speaker_id, float(stage_num), stage_label,
            confidence, ss, se, cap=0.55,
            metadata={
                "stage_1_signals": stage1,
                "stage_2_signals": stage2,
                "stage_3_signals": stage3,
                "glasl_stage":     stage_num,
            },
        )
