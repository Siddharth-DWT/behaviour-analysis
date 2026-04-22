"""
Body Rule Engine — Phase 2D
Implements BODY-HEAD-01, BODY-POST-01, BODY-LEAN-01, BODY-GEST-01,
BODY-FIDG-01, BODY-TOUCH-01.
BODY-MIRROR-01 is experimental (requires simultaneous multi-speaker pose data).

Research anchors:
  Mehrabian 1972     — Body lean as approach/avoidance indicator
  Navarro 2008       — Head nod clusters: agreement (≥2Hz), pacifying
  Pease & Pease 2004 — Posture and power: upright spine = confidence
  Ekman 1985         — Self-touch / adaptor gestures as stress indicator
  Soukupova 2016     — EAR-based blink detection (used in feature_extractor.py)
"""
import logging
import math
from typing import Optional

import numpy as np

from base_rule_engine import BaseVideoRuleEngine

try:
    from feature_extractor import WindowFeatures
    from calibration import FacialBaseline, BodyBaseline, GazeBaseline
except ImportError:
    from services.video_agent.feature_extractor import WindowFeatures
    from services.video_agent.calibration import FacialBaseline, BodyBaseline, GazeBaseline

logger = logging.getLogger("nexus.video.body_rules")

# Matches feature_extractor.TARGET_FPS = 10 → 2-second windows = 20 frames
_DEFAULT_FPS = 10.0
_DEFAULT_WINDOW_FRAMES = 20


def _count_zero_crossings(seq: list[float]) -> int:
    """
    Count sign changes in a sequence (used for head nod/shake frequency).
    DSA: single O(N) pass with sign comparison.
    """
    if len(seq) < 2:
        return 0
    crossings = 0
    for i in range(1, len(seq)):
        if seq[i - 1] * seq[i] < 0:
            crossings += 1
    return crossings


class BodyRuleEngine(BaseVideoRuleEngine):
    """
    Runs body language rules across all speakers.

    DSA:
      - Head motion analysis uses np.diff for velocity (O(N)).
      - Fidget detection uses a single sigma comparison (O(1) per window).
      - Touch detection uses a sliding concept encoded as per-window pct.
    All rules are O(S × W) overall.
    """

    AGENT_NAME = "video"

    # Head nod/shake
    NOD_MIN_CROSSINGS    = 2      # ≥2 pitch direction reversals per window
    NOD_VELOCITY_MIN     = 15.0   # degrees/s peak pitch velocity for nod
    SHAKE_MIN_CROSSINGS  = 2
    SHAKE_VELOCITY_MIN   = 20.0   # degrees/s peak yaw velocity for shake

    # Posture
    SPINE_SLUMP_THRESHOLD   = 10.0   # degrees forward lean from baseline = slumped
    SPINE_UPRIGHT_THRESHOLD = -5.0   # degrees more upright than baseline = power posture
    SHOULDER_ASYMM_THRESHOLD = 5.0   # shoulder angle std within window > this = tension

    # Lean
    HEAD_SHOULDER_FORWARD   = 0.05   # normalized distance drop from baseline = forward lean
    HEAD_SHOULDER_BACK      = -0.04  # normalized distance increase = backward lean

    # Gestures
    GESTURE_HIGH_VELOCITY    = 0.15  # normalized velocity above this = animated gestures
    GESTURE_PEAK_THRESHOLD   = 0.25  # gesture_velocity_max for very animated

    # Fidgeting (sigma from baseline)
    FIDGET_SIGMA_MODERATE = 2.0
    FIDGET_SIGMA_HIGH     = 3.5

    # Self-touch
    TOUCH_BRIEF_THRESHOLD     = 0.15  # hand_near_face_pct > 15% = brief touch events
    TOUCH_SUSTAINED_THRESHOLD = 0.30  # > 30% = sustained self-touching (stress/pacifying)

    # Minimum body detection rate to process window
    MIN_BODY_RATE = 0.30

    # BODY-MIRROR-01 thresholds
    MIRROR_ALIGN_MS   = 5000  # 5s temporal alignment window for mirroring
    MIRROR_MIN_EVENTS = 3     # minimum matching lean windows before flagging
    LEAN_THRESHOLD    = 0.03  # normalized head-shoulder distance change = directional lean

    def evaluate(
        self,
        windows_by_speaker: dict,
        baselines: dict,
        session_id: str = "",
        meeting_type: str = "general",
    ) -> list[dict]:
        signals: list[dict] = []
        for speaker_id, windows in windows_by_speaker.items():
            _, body_bl, _ = baselines.get(
                speaker_id,
                (None, BodyBaseline(speaker_id=speaker_id), None),
            )
            cal_conf = body_bl.calibration_confidence

            for w in windows:
                if w.body_detection_rate < self.MIN_BODY_RATE:
                    continue
                conf_mult = cal_conf * w.body_detection_rate

                w_head    = self._rule_head_gesture(w, body_bl, speaker_id, conf_mult)
                w_posture = self._rule_posture(w, body_bl, speaker_id, conf_mult)
                w_lean    = self._rule_lean(w, body_bl, speaker_id, conf_mult)

                signals += w_head
                signals += w_posture
                signals += w_lean
                signals += self._rule_gesture(w, body_bl, speaker_id, conf_mult)
                signals += self._rule_fidget(w, body_bl, speaker_id, conf_mult)
                signals += self._rule_self_touch(w, body_bl, speaker_id, conf_mult)
                signals += self._rule_head_body_incongruence(
                    w, body_bl, speaker_id, conf_mult, w_head, w_posture + w_lean
                )

        # Cross-speaker rule (needs all speakers simultaneously)
        signals += self._rule_mirror(windows_by_speaker, baselines)

        logger.info(f"[{session_id}] BodyRuleEngine: {len(signals)} signals")
        return signals

    # ── BODY-HEAD-01: Head nod / shake ────────────────────────────────────────
    def _rule_head_gesture(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        Head nod (pitch oscillation) and head shake (yaw oscillation) detection.
        np.diff gives frame-to-frame velocity; zero-crossing count gives frequency.
        Navarro 2008: ≥2 direction reversals with velocity > 15°/s = meaningful nod.
        """
        signals: list[dict] = []

        if len(w.head_pitch_seq) < 4:
            return signals

        fps = _DEFAULT_FPS
        pitch_vel = list(np.diff(w.head_pitch_seq) * fps)
        yaw_vel   = list(np.diff(w.head_yaw_seq) * fps)

        pitch_crossings = _count_zero_crossings(pitch_vel)
        yaw_crossings   = _count_zero_crossings(yaw_vel)

        peak_pitch_vel = max((abs(v) for v in pitch_vel), default=0.0)
        peak_yaw_vel   = max((abs(v) for v in yaw_vel), default=0.0)

        if (pitch_crossings >= self.NOD_MIN_CROSSINGS
                and peak_pitch_vel >= self.NOD_VELOCITY_MIN):
            freq = pitch_crossings / ((w.window_end_ms - w.window_start_ms) / 1000.0)
            confidence = min(freq * 0.15 * conf_mult, 0.65)
            signals.append(self._make_signal(
                rule_id="BODY-HEAD-01",
                signal_type="head_nod",
                speaker_id=speaker_id,
                value=round(peak_pitch_vel, 2),
                value_text="head_nod_agreement",
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={
                    "pitch_crossings": pitch_crossings,
                    "peak_pitch_velocity_dps": round(peak_pitch_vel, 2),
                    "frequency_hz": round(freq, 3),
                },
            ))

        if (yaw_crossings >= self.SHAKE_MIN_CROSSINGS
                and peak_yaw_vel >= self.SHAKE_VELOCITY_MIN):
            freq = yaw_crossings / ((w.window_end_ms - w.window_start_ms) / 1000.0)
            confidence = min(freq * 0.15 * conf_mult, 0.60)
            signals.append(self._make_signal(
                rule_id="BODY-HEAD-01",
                signal_type="head_shake",
                speaker_id=speaker_id,
                value=round(peak_yaw_vel, 2),
                value_text="head_shake_disagreement",
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={
                    "yaw_crossings": yaw_crossings,
                    "peak_yaw_velocity_dps": round(peak_yaw_vel, 2),
                    "frequency_hz": round(freq, 3),
                },
            ))

        return signals

    # ── BODY-POST-01: Spine angle / posture ───────────────────────────────────
    def _rule_posture(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        Spine angle deviation from baseline.
        Forward slump > 10° = low confidence / fatigue (Pease 2004).
        More upright > 5° = dominance / power posture (Mehrabian 1972).
        Shoulder asymmetry std within window > 5° = tension.
        """
        signals: list[dict] = []
        spine_delta = w.spine_angle_mean - bl.spine_angle_mean

        if spine_delta > self.SPINE_SLUMP_THRESHOLD:
            confidence = min((spine_delta / 20.0) * conf_mult, 0.60)
            signals.append(self._make_signal(
                rule_id="BODY-POST-01",
                signal_type="posture",
                speaker_id=speaker_id,
                value=round(spine_delta, 2),
                value_text="forward_slump",
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={
                    "spine_delta_deg": round(spine_delta, 2),
                    "spine_angle_mean": round(w.spine_angle_mean, 2),
                    "baseline": round(bl.spine_angle_mean, 2),
                },
            ))
        elif spine_delta < self.SPINE_UPRIGHT_THRESHOLD:
            confidence = min((abs(spine_delta) / 15.0) * conf_mult, 0.55)
            signals.append(self._make_signal(
                rule_id="BODY-POST-01",
                signal_type="posture",
                speaker_id=speaker_id,
                value=round(abs(spine_delta), 2),
                value_text="upright_power_posture",
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={
                    "spine_delta_deg": round(spine_delta, 2),
                    "spine_angle_mean": round(w.spine_angle_mean, 2),
                    "baseline": round(bl.spine_angle_mean, 2),
                },
            ))

        # Shoulder tension check
        if w.shoulder_angle_std > self.SHOULDER_ASYMM_THRESHOLD:
            confidence = min((w.shoulder_angle_std / 10.0) * conf_mult * 0.7, 0.50)
            signals.append(self._make_signal(
                rule_id="BODY-POST-01",
                signal_type="shoulder_tension",
                speaker_id=speaker_id,
                value=round(w.shoulder_angle_std, 2),
                value_text="shoulder_tension",
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={"shoulder_angle_std": round(w.shoulder_angle_std, 2)},
            ))

        return signals

    # ── BODY-LEAN-01: Forward / backward lean ─────────────────────────────────
    def _rule_lean(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        Head-to-shoulder distance delta from baseline as lean proxy.
        Forward lean (engagement/interest): distance decreases.
        Backward lean (discomfort/dominance): distance increases.
        Mehrabian 1972: forward lean = positive approach toward interactant.
        """
        dist_delta = w.head_shoulder_dist_mean - bl.head_shoulder_dist_mean
        dist_sigma = dist_delta / max(bl.head_shoulder_dist_std, 0.005)

        if dist_sigma < -2.0:
            label = "forward_lean"
            confidence = min(abs(dist_sigma) * 0.1 * conf_mult, 0.60)
        elif dist_sigma > 2.0:
            label = "backward_lean"
            confidence = min(dist_sigma * 0.1 * conf_mult, 0.55)
        else:
            return []

        return [self._make_signal(
            rule_id="BODY-LEAN-01",
            signal_type="body_lean",
            speaker_id=speaker_id,
            value=round(abs(dist_delta), 5),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "dist_delta": round(dist_delta, 5),
                "dist_sigma": round(dist_sigma, 2),
                "baseline_dist": round(bl.head_shoulder_dist_mean, 5),
            },
        )]

    # ── BODY-GEST-01: Gesture animation ───────────────────────────────────────
    def _rule_gesture(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        Hand gesture velocity → level of engagement and emphasis.
        High velocity = animated / emphatic communication.
        Absence during speaking may indicate suppression.
        """
        if w.hands_detected_rate < 0.10:
            return []

        mean_vel = w.gesture_velocity_mean
        peak_vel = w.gesture_velocity_max

        if peak_vel >= self.GESTURE_PEAK_THRESHOLD:
            label = "very_animated_gestures"
            confidence = min(peak_vel * 1.5 * conf_mult, 0.65)
        elif mean_vel >= self.GESTURE_HIGH_VELOCITY:
            label = "animated_gestures"
            confidence = min(mean_vel * 2.0 * conf_mult, 0.55)
        else:
            return []

        return [self._make_signal(
            rule_id="BODY-GEST-01",
            signal_type="gesture_animation",
            speaker_id=speaker_id,
            value=round(peak_vel, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "gesture_velocity_mean": round(mean_vel, 4),
                "gesture_velocity_max": round(peak_vel, 4),
                "hands_detected_rate": round(w.hands_detected_rate, 3),
            },
        )]

    # ── BODY-FIDG-01: Fidgeting / restlessness ────────────────────────────────
    def _rule_fidget(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        Body movement sigma above baseline → fidgeting / restlessness.
        Ekman 1985: body adaptors (self-touch, rocking) accompany anxiety.
        DSA: sigma = (current - baseline_mean) / baseline_std → O(1).
        """
        sigma = (w.body_movement_mean - bl.body_movement_mean) / max(bl.body_movement_std, 1e-6)

        if sigma >= self.FIDGET_SIGMA_HIGH:
            label = "high_fidgeting"
            confidence = min(sigma * 0.08 * conf_mult, 0.60)
        elif sigma >= self.FIDGET_SIGMA_MODERATE:
            label = "moderate_fidgeting"
            confidence = min(sigma * 0.06 * conf_mult, 0.50)
        else:
            return []

        return [self._make_signal(
            rule_id="BODY-FIDG-01",
            signal_type="body_fidgeting",
            speaker_id=speaker_id,
            value=round(w.body_movement_mean, 5),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "body_movement_mean": round(w.body_movement_mean, 5),
                "baseline_mean": round(bl.body_movement_mean, 5),
                "sigma": round(sigma, 2),
            },
        )]

    # ── BODY-TOUCH-01: Self-touch / adaptor gestures ──────────────────────────
    def _rule_self_touch(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        Hand-near-face percentage within window → self-soothing / pacifying.
        Brief (>15%): minor stress marker.
        Sustained (>30%): strong autonomic pacifying response (Navarro 2008).
        """
        pct = w.hand_near_face_pct

        if pct >= self.TOUCH_SUSTAINED_THRESHOLD:
            label = "sustained_self_touch"
            confidence = min(pct * conf_mult * 0.9, 0.65)
        elif pct >= self.TOUCH_BRIEF_THRESHOLD:
            label = "brief_self_touch"
            confidence = min(pct * conf_mult * 0.6, 0.50)
        else:
            return []

        return [self._make_signal(
            rule_id="BODY-TOUCH-01",
            signal_type="self_touch",
            speaker_id=speaker_id,
            value=round(pct, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "hand_near_face_pct": round(pct, 4),
                "blink_count_window": w.blink_count,
            },
        )]

    # ── BODY-INCONG-01: Head-body incongruence ────────────────────────────────
    def _rule_head_body_incongruence(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
        head_signals: list[dict],
        body_signals: list[dict],
    ) -> list[dict]:
        """
        BODY-INCONG-01: Head gesture contradicts body posture (Navarro 2008).
        Nod + backward-lean / slump = possible false agreement.
        Shake + forward-lean = disagreement with physical engagement.
        Takes per-window head and body signals already computed in the same iteration.
        """
        has_nod   = any(s.get("signal_type") == "head_nod"   for s in head_signals)
        has_shake = any(s.get("signal_type") == "head_shake"  for s in head_signals)

        has_backward = any(
            s.get("signal_type") == "body_lean" and s.get("value_text") == "backward_lean"
            for s in body_signals
        )
        has_slump = any(
            s.get("signal_type") == "posture" and s.get("value_text") == "forward_slump"
            for s in body_signals
        )
        has_forward = any(
            s.get("signal_type") == "body_lean" and s.get("value_text") == "forward_lean"
            for s in body_signals
        )

        signals: list[dict] = []

        if has_nod and (has_backward or has_slump):
            posture_label = "backward_lean" if has_backward else "forward_slump"
            confidence = min(0.45 * conf_mult, 0.55)
            signals.append(self._make_signal(
                rule_id="BODY-INCONG-01",
                signal_type="head_body_incongruence",
                speaker_id=speaker_id,
                value=0.6,
                value_text="nod_but_withdrawing",
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={
                    "head_gesture":    "nod",
                    "body_posture":    posture_label,
                    "interpretation":  "possible_false_agreement",
                },
            ))

        if has_shake and has_forward:
            confidence = min(0.40 * conf_mult, 0.50)
            signals.append(self._make_signal(
                rule_id="BODY-INCONG-01",
                signal_type="head_body_incongruence",
                speaker_id=speaker_id,
                value=0.5,
                value_text="shake_but_engaged",
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={
                    "head_gesture":   "shake",
                    "body_posture":   "forward_lean",
                    "interpretation": "disagreement_with_interest",
                },
            ))

        return signals

    # ── BODY-MIRROR-01 (EXPERIMENTAL) ────────────────────────────────────────
    def _rule_mirror(self, windows_by_speaker: dict, baselines: dict) -> list[dict]:
        """
        BODY-MIRROR-01 (EXPERIMENTAL): synchronized lean direction between speakers.
        When two speakers lean the same direction within 5s, it suggests rapport
        through postural mirroring (Chartrand & Bargh 1999 — Chameleon Effect).
        Cap 0.25 — experimental, lean estimation from head-shoulder distance is noisy.

        DSA: sorted windows + advancing pointer → O(A + B) per pair.
        """
        speaker_ids = list(windows_by_speaker.keys())
        if len(speaker_ids) < 2:
            return []

        def _lean_dir(w: WindowFeatures, bl: BodyBaseline) -> Optional[str]:
            delta = w.head_shoulder_dist_mean - bl.head_shoulder_dist_mean
            if delta < -self.LEAN_THRESHOLD:
                return "forward"
            if delta > self.LEAN_THRESHOLD:
                return "backward"
            return None

        signals: list[dict] = []

        for i in range(len(speaker_ids)):
            for j in range(i + 1, len(speaker_ids)):
                sp_a = speaker_ids[i]
                sp_b = speaker_ids[j]

                _, bl_a, _ = baselines.get(sp_a, (None, BodyBaseline(speaker_id=sp_a), None))
                _, bl_b, _ = baselines.get(sp_b, (None, BodyBaseline(speaker_id=sp_b), None))

                wins_a = sorted(
                    [w for w in windows_by_speaker[sp_a]
                     if w.body_detection_rate >= self.MIN_BODY_RATE],
                    key=lambda w: w.window_start_ms,
                )
                wins_b = sorted(
                    [w for w in windows_by_speaker[sp_b]
                     if w.body_detection_rate >= self.MIN_BODY_RATE],
                    key=lambda w: w.window_start_ms,
                )
                if not wins_a or not wins_b:
                    continue

                mirror_events: list[tuple[int, int]] = []
                eligible_a = 0
                ptr_b = 0

                for wa in wins_a:
                    dir_a = _lean_dir(wa, bl_a)
                    if dir_a is None:
                        continue
                    eligible_a += 1
                    while (ptr_b < len(wins_b)
                           and wins_b[ptr_b].window_end_ms
                           < wa.window_start_ms - self.MIRROR_ALIGN_MS):
                        ptr_b += 1
                    for wb in wins_b[ptr_b:]:
                        if wb.window_start_ms > wa.window_end_ms + self.MIRROR_ALIGN_MS:
                            break
                        if _lean_dir(wb, bl_b) == dir_a:
                            mirror_events.append((
                                min(wa.window_start_ms, wb.window_start_ms),
                                max(wa.window_end_ms,   wb.window_end_ms),
                            ))
                            break

                if len(mirror_events) < self.MIRROR_MIN_EVENTS:
                    continue

                mirror_rate = len(mirror_events) / max(eligible_a, 1)
                confidence  = min(mirror_rate * 0.8, 0.25)  # hard cap 0.25

                all_wins    = wins_a + wins_b
                session_start = min(w.window_start_ms for w in all_wins)
                session_end   = max(w.window_end_ms   for w in all_wins)

                for sp in (sp_a, sp_b):
                    signals.append(self._make_signal(
                        rule_id="BODY-MIRROR-01",
                        signal_type="body_mirroring",
                        speaker_id=sp,
                        value=round(mirror_rate, 4),
                        value_text="synchronized_lean",
                        confidence=confidence,
                        window_start_ms=session_start,
                        window_end_ms=session_end,
                        metadata={
                            "pair": f"{sp_a}+{sp_b}",
                            "mirror_event_count": len(mirror_events),
                            "mirror_rate": round(mirror_rate, 4),
                            "experimental": True,
                        },
                    ))

        return signals
