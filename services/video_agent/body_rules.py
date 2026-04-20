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

import numpy as np

from base_rule_engine import BaseVideoRuleEngine

try:
    from feature_extractor import WindowFeatures
    from calibration import FacialBaseline, BodyBaseline, GazeBaseline
except ImportError:
    from services.video_agent.feature_extractor import WindowFeatures
    from services.video_agent.calibration import FacialBaseline, BodyBaseline, GazeBaseline

logger = logging.getLogger("nexus.video.body_rules")

# Assume 2-second windows at 15 fps → 30 frames/window
_DEFAULT_FPS = 15.0
_DEFAULT_WINDOW_FRAMES = 30


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

                signals += self._rule_head_gesture(w, body_bl, speaker_id, conf_mult)
                signals += self._rule_posture(w, body_bl, speaker_id, conf_mult)
                signals += self._rule_lean(w, body_bl, speaker_id, conf_mult)
                signals += self._rule_gesture(w, body_bl, speaker_id, conf_mult)
                signals += self._rule_fidget(w, body_bl, speaker_id, conf_mult)
                signals += self._rule_self_touch(w, body_bl, speaker_id, conf_mult)

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
