"""
Body Rule Engine — Phase 2D + 2E
Implements BODY-HEAD-01, BODY-POST-01, BODY-LEAN-01, BODY-GEST-01,
BODY-FIDG-01, BODY-TOUCH-01, BODY-TOUCH-02, BODY-ARMS-01, BODY-STEEPLE-01,
BODY-STATE-01, BODY-CLUSTER-01.
BODY-MIRROR-01 is experimental (requires simultaneous multi-speaker pose data).

Research anchors:
  Mehrabian 1972     — Body lean as approach/avoidance indicator
  Navarro 2008       — Head nod clusters: agreement (≥2Hz), pacifying; steepling = confidence
  Pease & Pease 2004 — Posture and power: upright spine = confidence; 3-cue cluster rule
  Ekman 1985         — Self-touch / adaptor gestures as stress indicator
  Soukupova 2016     — EAR-based blink detection (used in feature_extractor.py)
  Chartrand 1999     — Chameleon Effect (mirroring → rapport)
"""
import logging
import math
from typing import Optional

import numpy as np

from .base_rule_engine import BaseVideoRuleEngine
from .feature_extractor import WindowFeatures
from .calibration import FacialBaseline, BodyBaseline, GazeBaseline

logger = logging.getLogger("nexus.video.body_rules")

# Matches feature_extractor.TARGET_FPS = 5 → 2-second windows = 10 frames
_DEFAULT_FPS = 5.0
_DEFAULT_WINDOW_FRAMES = 10


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


class PostureStateMachine:
    """
    Tracks posture state transitions across windows for a single speaker.

    States: open, closed, engaged, withdrawn, neutral.
    Transitions between non-neutral states emit BODY-STATE-01 signals.
    State scores are derived from arm/lean/gesture features per window.
    """

    def classify_state(self, w: "WindowFeatures", bl: Optional[object] = None) -> str:
        score_open = 0
        score_closed = 0
        score_engaged = 0
        score_withdrawn = 0

        if w.arms_crossed_pct > 0.30:
            score_closed += 2
        elif w.hands_detected_rate > 0.50:
            score_open += 1

        if bl is not None:
            dist_delta = w.head_shoulder_dist_mean - getattr(bl, "head_shoulder_dist_mean", 0.0)
            if dist_delta < -0.03:
                score_engaged += 2
                score_open += 1
            elif dist_delta > 0.03:
                score_withdrawn += 2
                score_closed += 1

        if w.spine_angle_mean > 10:
            score_withdrawn += 1
        elif abs(w.spine_angle_mean) < 5:
            score_engaged += 1

        if w.hand_velocity_mean > 0.5:
            score_engaged += 1
            score_open += 1
        elif w.hand_velocity_mean < 0.1:
            score_withdrawn += 1

        scores = {
            "open": score_open, "closed": score_closed,
            "engaged": score_engaged, "withdrawn": score_withdrawn,
        }
        best = max(scores, key=lambda k: scores[k])
        return "neutral" if scores[best] < 2 else best

    def __init__(self) -> None:
        self._history: list[tuple[int, str]] = []

    def update(self, timestamp_ms: int, state: str) -> Optional[dict]:
        """Record new state; return a transition dict if the state changed meaningfully."""
        self._history.append((timestamp_ms, state))
        if len(self._history) < 2:
            return None
        prev_state = self._history[-2][1]
        if prev_state == state or prev_state == "neutral" or state == "neutral":
            return None
        labels = {
            ("open",     "closed"):    "closing_up",
            ("closed",   "open"):      "opening_up",
            ("engaged",  "withdrawn"): "disengaging",
            ("withdrawn","engaged"):   "re_engaging",
            ("engaged",  "closed"):    "defensive_shift",
            ("open",     "withdrawn"): "losing_interest",
        }
        label = labels.get((prev_state, state), f"{prev_state}_to_{state}")
        return {"from_state": prev_state, "to_state": state, "label": label, "timestamp_ms": timestamp_ms}


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
    NOD_MIN_CROSSINGS    = 3      # ≥3 pitch direction reversals (1.5 nod cycles)
    NOD_VELOCITY_MIN     = 15.0   # degrees/s peak pitch velocity for nod
    SHAKE_MIN_CROSSINGS  = 3
    SHAKE_VELOCITY_MIN   = 20.0   # degrees/s peak yaw velocity for shake

    # Posture
    SPINE_SLUMP_THRESHOLD   = 10.0   # degrees forward lean from baseline = slumped
    SPINE_UPRIGHT_THRESHOLD = -5.0   # degrees more upright than baseline = power posture
    SHOULDER_ASYMM_THRESHOLD = 8.0   # shoulder angle std within window > this = tension (was 5.0)

    # Lean
    HEAD_SHOULDER_FORWARD   = 0.05   # normalized distance drop from baseline = forward lean
    HEAD_SHOULDER_BACK      = -0.04  # normalized distance increase = backward lean
    LEAN_ABS_MIN            = 0.025  # absolute dist delta floor — prevents amplification when baseline std is small

    # Gestures
    GESTURE_HIGH_VELOCITY    = 0.15  # normalized velocity above this = animated gestures
    GESTURE_PEAK_THRESHOLD   = 0.25  # gesture_velocity_max for very animated

    # Fidgeting (sigma from baseline)
    FIDGET_SIGMA_MODERATE = 2.0
    FIDGET_SIGMA_HIGH     = 3.5

    # Self-touch
    TOUCH_BRIEF_THRESHOLD     = 0.30  # hand_near_face_pct > 30% = brief touch events (was 0.22)
    TOUCH_SUSTAINED_THRESHOLD = 0.35  # > 35% = sustained self-touching (stress/pacifying) (was 0.30)

    # Gesture suppression — open palms sweeping past the face zone are gestures, not touch
    GESTURE_SUPPRESS_PALMS    = 0.40  # open_palms_pct >= 40% of window frames → likely gesturing
    GESTURE_SUPPRESS_VELOCITY = 0.12  # gesture_velocity_mean above this → hand in active motion

    # Minimum body detection rate to process window
    MIN_BODY_RATE = 0.30

    # Discard signals below this before storage — prevents weak extrapolations from flooding the dashboard
    MIN_SIGNAL_CONFIDENCE = 0.20

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
        extra_signals: Optional[list[dict]] = None,
    ) -> list[dict]:
        signals: list[dict] = []
        state_machines: dict[str, PostureStateMachine] = {}
        _extra = extra_signals or []

        for speaker_id, windows in windows_by_speaker.items():
            _, body_bl, _ = baselines.get(
                speaker_id,
                (None, BodyBaseline(speaker_id=speaker_id), None),
            )
            cal_conf = body_bl.calibration_confidence
            state_machines[speaker_id] = PostureStateMachine()

            for w in sorted(windows, key=lambda x: x.window_start_ms):
                if w.body_detection_rate < self.MIN_BODY_RATE:
                    continue
                self._current_w = w
                conf_mult = cal_conf * w.body_detection_rate

                # Collect all signals for this window before running cluster rule
                window_signals: list[dict] = []

                w_head    = self._rule_head_gesture(w, body_bl, speaker_id, conf_mult)
                w_posture = self._rule_posture(w, body_bl, speaker_id, conf_mult)
                w_lean    = self._rule_lean(w, body_bl, speaker_id, conf_mult)

                window_signals += w_head + w_posture + w_lean
                window_signals += self._rule_gesture(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_fidget(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_self_touch(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_head_body_incongruence(
                    w, body_bl, speaker_id, conf_mult, w_head, w_posture + w_lean
                )
                window_signals += self._rule_touch_classified(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_head_supported(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_hands_clasped(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_crossed_arms(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_steepling(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_evaluation_cluster(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_gesture_signals(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_arm_expansion(w, body_bl, speaker_id, conf_mult)
                window_signals += self._rule_cross_speaker_interaction(w, speaker_id, conf_mult)

                # Cross-modal context: facial + gaze signals overlapping this window
                window_extra = [
                    s for s in _extra
                    if s is not None
                    and s.get("speaker_id") == speaker_id
                    and s.get("window_start_ms", 0) < w.window_end_ms
                    and s.get("window_end_ms", 0) > w.window_start_ms
                ]

                # Cross-modal cluster rules — need both body signals + facial context
                all_ctx = window_signals + window_extra
                window_signals += self._rule_hidden_disagreement(all_ctx, w, speaker_id, conf_mult)
                window_signals += self._rule_frustration_cluster(all_ctx, w, body_bl, speaker_id, conf_mult)

                # Main cluster rule sees everything including the two new signals above
                window_signals += self._rule_body_language_cluster(
                    window_signals + window_extra, w, speaker_id, conf_mult
                )

                # Posture state machine — emit transition when state changes
                state = state_machines[speaker_id].classify_state(w, body_bl)
                transition = state_machines[speaker_id].update(w.window_start_ms, state)
                if transition:
                    window_signals.append(self._make_signal(
                        rule_id="BODY-STATE-01",
                        signal_type="posture_transition",
                        speaker_id=speaker_id,
                        value=0.5,
                        value_text=transition["label"],
                        confidence=min(0.50 * conf_mult, 0.55),
                        window_start_ms=w.window_start_ms,
                        window_end_ms=w.window_end_ms,
                        metadata={
                            "from_state": transition["from_state"],
                            "to_state":   transition["to_state"],
                        },
                    ))

                signals += [
                    s for s in window_signals
                    if s is not None and s.get("confidence", 0) >= self.MIN_SIGNAL_CONFIDENCE
                ]

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

        nod_fires   = (pitch_crossings >= self.NOD_MIN_CROSSINGS
                       and peak_pitch_vel >= self.NOD_VELOCITY_MIN)
        shake_fires = (yaw_crossings >= self.SHAKE_MIN_CROSSINGS
                       and peak_yaw_vel >= self.SHAKE_VELOCITY_MIN)

        # ── Mutual exclusion: a person cannot agree AND disagree simultaneously ──
        # When both nod and shake qualify, head is moving in all directions (general
        # movement). Keep whichever axis has the higher peak velocity — that is the
        # more deliberate, intentional gesture.
        if nod_fires and shake_fires:
            if peak_pitch_vel >= peak_yaw_vel:
                shake_fires = False  # vertical movement dominates → nod
            else:
                nod_fires = False    # horizontal movement dominates → shake

        if nod_fires:
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

        if shake_fires:
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
        if abs(dist_delta) < self.LEAN_ABS_MIN:
            return []
        dist_sigma = dist_delta / max(bl.head_shoulder_dist_std, 0.005)

        if dist_sigma < -2.0:
            label = "forward_lean"
            # Lean from head-shoulder distance has high variance; cap per RESEARCH.md (0.30-0.40)
            confidence = min(abs(dist_sigma) * 0.1 * conf_mult, 0.38)
        elif dist_sigma > 2.0:
            label = "backward_lean"
            confidence = min(dist_sigma * 0.1 * conf_mult, 0.35)
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
            # Subtle fidgets below webcam resolution threshold — cap per RESEARCH.md (0.35-0.45)
            confidence = min(sigma * 0.08 * conf_mult, 0.42)
        elif sigma >= self.FIDGET_SIGMA_MODERATE:
            label = "moderate_fidgeting"
            confidence = min(sigma * 0.06 * conf_mult, 0.35)
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

        if pct < self.TOUCH_BRIEF_THRESHOLD:
            # Secondary path: zone classification confirmed a specific face region
            # even though hand_near_face_pct is below the threshold. At 5fps a
            # 3-second touch may only land in 1-2 sampled frames per window (10%).
            # The zone classifier's own distance threshold independently validates
            # the touch, so treat a confirmed zone as a brief touch.
            if not w.dominant_touch_zone:
                return []
            pct = self.TOUCH_BRIEF_THRESHOLD  # floor to threshold for confidence calc

        if pct >= self.TOUCH_SUSTAINED_THRESHOLD:
            label = "sustained_self_touch"
            confidence = min(pct * conf_mult * 0.9, 0.65)
        else:
            label = "brief_self_touch"
            confidence = min(pct * conf_mult * 0.6, 0.50)

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

    # ── BODY-TOUCH-02: Classified face-region touch ───────────────────────────
    def _rule_touch_classified(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-TOUCH-02: Zone-aware face-region touch classification.
        Navarro 2008: neck = vulnerability/stress; Pease 2004: chin = evaluation,
        mouth = suppression; Ekman 1985: nose = discomfort/self-soothing.
        """
        zone = w.dominant_touch_zone
        if not zone:
            return []

        zone_pct_map = {
            "chin":     w.touch_zone_chin_pct,
            "mouth":    w.touch_zone_mouth_pct,
            "nose":     w.touch_zone_nose_pct,
            "cheek":    w.touch_zone_cheek_pct,
            "ear":      w.touch_zone_ear_pct,
            "neck":     w.touch_zone_neck_pct,
            "forehead": w.touch_zone_forehead_pct,
            "eye":      w.touch_zone_eye_pct,
        }
        zone_pct = zone_pct_map.get(zone, 0.0)
        baseline_touch = getattr(bl, "self_touch_rate", 0.0)
        min_pct = max(baseline_touch + 0.10, 0.15)
        # Neck touch requires a higher dwell percentage — brief hand-near-collar
        # from gesturing must not fire; require sustained contact.
        if zone == "neck":
            min_pct = max(min_pct, 0.25)
        # Mouth touch (→ "Holding Back") over-fires on incidental hand movement;
        # require sustained contact before calling suppression.
        if zone == "mouth":
            min_pct = max(min_pct, 0.30)
        # Nose touch (→ "Uncomfortable") shares the same over-fire risk.
        if zone == "nose":
            min_pct = max(min_pct, 0.25)
        # Eye touch requires sustained contact; both-hand block needs less dwell
        # because the gesture itself is unambiguous.
        if zone == "eye":
            min_pct = max(min_pct, 0.20)
        # Chin touch over-fires when open-palm gestures pass through the lower face
        # zone; require sustained dwell to distinguish a brief sweep from evaluation.
        if zone == "chin":
            min_pct = max(min_pct, 0.25)
        # Cheek touch over-fires from lateral head turns that bring the hand near
        # the face without contact; require sustained dwell.
        if zone == "cheek":
            min_pct = max(min_pct, 0.25)
        # Forehead touch over-fires on upward gestures; require sustained dwell.
        if zone == "forehead":
            min_pct = max(min_pct, 0.25)
        # Suppress when hand is actively gesturing with open palms — raised open
        # palms sweeping past the face zone are gestures, not self-soothing.
        if w.open_palms_pct >= self.GESTURE_SUPPRESS_PALMS and w.gesture_velocity_mean > self.GESTURE_SUPPRESS_VELOCITY:
            return []
        if zone_pct < min_pct:
            return []

        # Eye zone: both hands simultaneously → disbelief/block; single hand → rub.
        eye_both_pct  = getattr(w, "touch_zone_eye_both_pct", 0.0)
        open_dominant = getattr(w, "open_palms_pct", 0.0) >= 0.20
        fist_dominant = getattr(w, "closed_fist_pct", 0.0) >= 0.20

        # Base label per zone (shape-neutral default)
        zone_labels = {
            "chin":     "chin_touch_evaluation",
            "mouth":    "mouth_cover_suppression",
            "nose":     "nose_touch_discomfort",
            "cheek":    "cheek_touch_listening" if zone_pct < 0.40 else "cheek_rest_fatigue",
            "ear":      "ear_touch_soothing",
            "neck":     "neck_touch_vulnerability",
            "forehead": "forehead_touch_frustration",
            "eye":      "eye_block_disbelief" if eye_both_pct >= 0.15 else "eye_rub_discomfort",
        }
        label = zone_labels.get(zone, f"{zone}_touch")

        # Refine label when hand shape is unambiguous (Navarro 2008: shape disambiguates intent)
        # open palm = relaxed/passive/shock; closed/curled = active tension/suppression
        _SHAPE_OVERRIDES: dict[str, dict[str, str]] = {
            "chin_touch_evaluation":    {"open": "chin_touch_contemplation",     "closed": "chin_touch_evaluation"},
            "mouth_cover_suppression":  {"open": "mouth_cover_shock",            "closed": "mouth_cover_suppression"},
            "forehead_touch_frustration":{"open": "forehead_touch_fatigue",      "closed": "forehead_touch_frustration"},
            "neck_touch_vulnerability": {"open": "neck_touch_vulnerability",     "closed": "neck_touch_tension"},
            "cheek_touch_listening":    {"open": "cheek_touch_listening",        "closed": "cheek_touch_distress"},
            "cheek_rest_fatigue":       {"open": "cheek_rest_fatigue",           "closed": "cheek_touch_distress"},
        }
        if label in _SHAPE_OVERRIDES:
            if open_dominant:
                label = _SHAPE_OVERRIDES[label]["open"]
            elif fist_dominant:
                label = _SHAPE_OVERRIDES[label]["closed"]

        confidence = min(zone_pct * conf_mult, 0.50)

        return [self._make_signal(
            rule_id="BODY-TOUCH-02",
            signal_type="face_region_touch",
            speaker_id=speaker_id,
            value=round(zone_pct, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "zone": zone,
                "zone_pct": round(zone_pct, 4),
                "all_zones": {k: round(v, 4) for k, v in zone_pct_map.items() if v > 0.01},
                "interpretation_caveat": "single_cue_only_meaningful_in_cluster",
            },
        )]

    # ── BODY-SUPPORT-01: Head resting on hand ────────────────────────────────
    def _rule_head_supported(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-SUPPORT-01: Head resting on hand (sustained chin/cheek touch +
        head tilted + low facial activity).
        Navarro 2008: palm-supporting-chin = boredom or deep evaluation.
        Listening context → disengagement; speaking context → contemplation.
        """
        if w.head_supported_pct < 0.30:
            return []
        label = "head_resting_disengagement" if not w.is_speaking else "head_resting_contemplation"
        return [self._make_signal(
            rule_id="BODY-SUPPORT-01",
            signal_type="head_supported",
            speaker_id=speaker_id,
            value=round(w.head_supported_pct, 4),
            value_text=label,
            confidence=min(w.head_supported_pct * 0.7 * conf_mult, 0.50),
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "head_supported_pct": round(w.head_supported_pct, 4),
                "is_speaking": w.is_speaking,
                "dominant_touch_zone": w.dominant_touch_zone,
            },
        )]

    # ── BODY-CLASP-01: Hands clasped / interlaced ─────────────────────────────
    def _rule_hands_clasped(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-CLASP-01: Hands clasped/interlaced (both hands together, fingers
        folded — distinct from steepling). Pease 2004: clasped hands = restraint
        or neutral waiting. Listening → waiting; speaking → self-restraint.
        """
        if w.hands_clasped_pct < 0.25:
            return []
        label = "hands_clasped_waiting" if not w.is_speaking else "hands_clasped_restraint"
        return [self._make_signal(
            rule_id="BODY-CLASP-01",
            signal_type="hands_clasped",
            speaker_id=speaker_id,
            value=round(w.hands_clasped_pct, 4),
            value_text=label,
            confidence=min(w.hands_clasped_pct * 0.6 * conf_mult, 0.40),
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "hands_clasped_pct": round(w.hands_clasped_pct, 4),
                "is_speaking": w.is_speaking,
            },
        )]

    # ── BODY-ARMS-01: Crossed arms ────────────────────────────────────────────
    def _rule_crossed_arms(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-ARMS-01: Crossed-arms posture — flagged only when it's a change from
        baseline. Pease 2004: arms crossed CAN indicate defensiveness but also
        comfort/cold/habitual rest; baseline-delta is required before flagging.
        """
        if w.arms_crossed_pct < 0.30:
            return []
        baseline_crossed = getattr(bl, "arms_crossed_rate", 0.0)
        delta = w.arms_crossed_pct - baseline_crossed
        if delta < 0.15:
            return []
        confidence = min(delta * conf_mult, 0.45)
        return [self._make_signal(
            rule_id="BODY-ARMS-01",
            signal_type="arms_crossed",
            speaker_id=speaker_id,
            value=round(w.arms_crossed_pct, 4),
            value_text="closed_posture",
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "arms_crossed_pct": round(w.arms_crossed_pct, 4),
                "baseline_rate": round(baseline_crossed, 4),
                "delta_from_baseline": round(delta, 4),
                "caveat": "may_indicate_comfort_or_cold_not_only_defensiveness",
            },
        )]

    # ── BODY-STEEPLE-01: Finger steepling ────────────────────────────────────
    def _rule_steepling(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-STEEPLE-01: Finger steepling — confidence / authority indicator.
        Navarro 2008: one of the most reliable confidence gestures; both hands with
        fingertips pressed together ≥20% of window.
        """
        if w.finger_steepling_pct < 0.20:
            return []
        confidence = min(w.finger_steepling_pct * 0.8 * conf_mult, 0.55)
        return [self._make_signal(
            rule_id="BODY-STEEPLE-01",
            signal_type="finger_steepling",
            speaker_id=speaker_id,
            value=round(w.finger_steepling_pct, 4),
            value_text="confidence_posture",
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={"steepling_pct": round(w.finger_steepling_pct, 4)},
        )]

    # ── BODY-EVAL-01: Evaluation / contemplation cluster ─────────────────────
    def _rule_evaluation_cluster(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-EVAL-01: Chin touch anchors the cluster; steepling, arms-crossed, or
        backward-lean provide the second cue required by Pease's cluster rule.
        Pease 2004: chin stroke = evaluation; steepling = high confidence in that stance.
        Navarro 2008: chin + backward-lean = deep contemplation / silent processing.
        """
        baseline_touch = getattr(bl, "self_touch_rate", 0.0)
        chin_threshold = max(baseline_touch + 0.10, 0.15)
        if w.touch_zone_chin_pct < chin_threshold:
            return []

        steepling     = w.finger_steepling_pct >= 0.15
        arms_crossed  = w.arms_crossed_pct >= 0.20
        # Backward lean: head moves further from shoulder midpoint than baseline
        backward_lean = (
            w.head_shoulder_dist_mean - getattr(bl, "head_shoulder_dist_mean", w.head_shoulder_dist_mean)
        ) > 0.04

        if steepling:
            label      = "critical_evaluation"
            confidence = min(w.touch_zone_chin_pct * 0.8 * conf_mult, 0.65)
        elif backward_lean and not w.is_speaking:
            label      = "deep_contemplation"
            confidence = min(w.touch_zone_chin_pct * 0.65 * conf_mult, 0.55)
        elif arms_crossed and not w.is_speaking:
            label      = "skeptical_evaluation"
            confidence = min(w.touch_zone_chin_pct * 0.55 * conf_mult, 0.50)
        else:
            return []

        return [self._make_signal(
            rule_id="BODY-EVAL-01",
            signal_type="evaluation_cluster",
            speaker_id=speaker_id,
            value=round(w.touch_zone_chin_pct, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "chin_touch_pct":   round(w.touch_zone_chin_pct, 4),
                "steepling_pct":    round(w.finger_steepling_pct, 4),
                "arms_crossed_pct": round(w.arms_crossed_pct, 4),
                "backward_lean":    backward_lean,
            },
        )]

    # ── BODY-GESTURE-02: GestureRecognizer-based hand signals ────────────────
    def _rule_gesture_signals(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-GESTURE-02: Emit signals for the 5 GestureRecognizer categories
        (Thumb_Up, Thumb_Down, Pointing_Up, Victory, Closed_Fist).
        open_palms is already covered by BODY-CLUSTER-01 confidence scoring.
        Navarro 2008: thumb gestures are clear approval/disapproval; pointing = authority;
        closed fist = tension or determination; victory = relief/celebration.
        """
        signals = []

        _GESTURE_RULES = [
            ("thumb_up_pct",     0.20, "hand_gesture", "approval",    0.60, "BODY-GESTURE-02"),
            ("thumb_down_pct",   0.20, "hand_gesture", "disapproval", 0.60, "BODY-GESTURE-02"),
            ("pointing_up_pct",  0.15, "hand_gesture", "emphasis",    0.50, "BODY-GESTURE-02"),
            ("victory_sign_pct", 0.15, "hand_gesture", "victory",     0.50, "BODY-GESTURE-02"),
            ("closed_fist_pct",  0.20, "hand_gesture", "tension",     0.55, "BODY-GESTURE-02"),
        ]

        for attr, threshold, sig_type, label, max_conf, rule_id in _GESTURE_RULES:
            pct = getattr(w, attr, 0.0)
            if pct < threshold:
                continue
            confidence = min(pct * conf_mult, max_conf)
            signals.append(self._make_signal(
                rule_id=rule_id,
                signal_type=sig_type,
                speaker_id=speaker_id,
                value=round(pct, 4),
                value_text=label,
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={"pct": round(pct, 4)},
            ))

        return signals

    # ── BODY-ARM-01: Elbow expansion / arm openness ───────────────────────────
    def _rule_arm_expansion(
        self,
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-ARM-01: Elbow spread relative to shoulder width.
        Expansive (elbows > shoulders): open/dominant posture (Carney 2010: power pose).
        Contracted (elbows < shoulders by >20%): closed/defensive posture.
        """
        signals = []
        expansion = getattr(w, "elbow_expansion_mean", 0.0)

        if expansion >= 0.15:
            confidence = min(expansion * 0.6 * conf_mult, 0.50)
            signals.append(self._make_signal(
                rule_id="BODY-ARM-01",
                signal_type="arm_posture",
                speaker_id=speaker_id,
                value=round(expansion, 4),
                value_text="expansive",
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={"elbow_expansion": round(expansion, 4)},
            ))
        elif expansion <= -0.20:
            confidence = min(abs(expansion) * 0.5 * conf_mult, 0.45)
            signals.append(self._make_signal(
                rule_id="BODY-ARM-01",
                signal_type="arm_posture",
                speaker_id=speaker_id,
                value=round(expansion, 4),
                value_text="contracted",
                confidence=confidence,
                window_start_ms=w.window_start_ms,
                window_end_ms=w.window_end_ms,
                metadata={"elbow_expansion": round(expansion, 4)},
            ))

        return signals

    # ── BODY-HDIS-01: Hidden / suppressed disagreement cluster ────────────────
    def _rule_hidden_disagreement(
        self,
        all_signals: list[dict],
        w: WindowFeatures,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-HDIS-01: Lip pursing (FACE-LIPS-01) is the anchor — a person suppressing
        disagreement they won't voice.  One additional body cue elevates it to a cluster;
        two or more produce high-confidence suppressed disagreement.
        Pease 2004: pursed lips + backward lean + low engagement = textbook hidden rejection.
        """
        has_lip_pursing = any(
            s.get("signal_type") == "lip_pursing" and s.get("speaker_id") == speaker_id
            for s in all_signals
        )
        if not has_lip_pursing:
            return []

        backward_lean = any(
            s.get("signal_type") == "lean" and s.get("value_text") == "backward_lean"
            and s.get("speaker_id") == speaker_id
            for s in all_signals
        )
        low_engagement = any(
            s.get("signal_type") == "facial_engagement" and
            s.get("value_text") == "low_engagement" and
            s.get("speaker_id") == speaker_id
            for s in all_signals
        )
        arms_crossed   = w.arms_crossed_pct >= 0.20
        mouth_touching = w.touch_zone_mouth_pct >= 0.20

        cues = sum([backward_lean, low_engagement, arms_crossed, mouth_touching])
        if cues < 1:
            return []

        label      = "suppressed_disagreement_cluster"
        confidence = min((0.30 + cues * 0.08) * conf_mult, 0.60)

        return [self._make_signal(
            rule_id="BODY-HDIS-01",
            signal_type="hidden_disagreement",
            speaker_id=speaker_id,
            value=round(min(cues / 3.0, 1.0), 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "lip_pursing":      True,
                "backward_lean":    backward_lean,
                "low_engagement":   low_engagement,
                "arms_crossed_pct": round(w.arms_crossed_pct, 4),
                "mouth_touch_pct":  round(w.touch_zone_mouth_pct, 4),
                "corroborating_cues": cues,
            },
        )]

    # ── BODY-FRUST-01: Frustration cluster ────────────────────────────────────
    def _rule_frustration_cluster(
        self,
        all_signals: list[dict],
        w: WindowFeatures,
        bl: BodyBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-FRUST-01: Forehead or nose touch anchors the cluster (self-soothing under
        mental strain). Corroborated by facial stress signal, body fidgeting, or elevated
        brow tension in the same window.
        Navarro 2008: forehead rub + jaw clench + body restlessness = active frustration.
        """
        baseline_touch = getattr(bl, "self_touch_rate", 0.0)
        forehead_pct   = w.touch_zone_forehead_pct
        nose_pct       = w.touch_zone_nose_pct

        # Either forehead or nose touch must be elevated above baseline
        touch_threshold = max(baseline_touch + 0.10, 0.15)
        if forehead_pct < touch_threshold and nose_pct < (touch_threshold + 0.10):
            return []

        anchor_pct = max(forehead_pct, nose_pct)

        has_facial_stress = any(
            s.get("signal_type") == "facial_stress" and s.get("speaker_id") == speaker_id
            for s in all_signals
        )
        has_fidgeting = any(
            s.get("signal_type") == "body_fidgeting" and s.get("speaker_id") == speaker_id
            for s in all_signals
        )
        # Direct blendshape check — doesn't need facial engine output
        brow_tension = (
            w.blendshapes_mean.get("browDownLeft",  0.0) +
            w.blendshapes_mean.get("browDownRight", 0.0)
        ) / 2.0
        high_brow      = brow_tension > 0.15
        high_movement  = w.body_movement_mean > 0.12

        cues = sum([has_facial_stress, has_fidgeting, high_brow, high_movement])
        if cues < 1:
            return []

        label      = "active_frustration" if cues >= 2 else "building_frustration"
        confidence = min((anchor_pct * 0.5 + cues * 0.07) * conf_mult, 0.55)

        return [self._make_signal(
            rule_id="BODY-FRUST-01",
            signal_type="frustration_cluster",
            speaker_id=speaker_id,
            value=round(anchor_pct, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "forehead_touch_pct":  round(forehead_pct, 4),
                "nose_touch_pct":      round(nose_pct, 4),
                "facial_stress":       has_facial_stress,
                "body_fidgeting":      has_fidgeting,
                "brow_tension":        round(brow_tension, 4),
                "high_movement":       high_movement,
                "corroborating_cues":  cues,
            },
        )]

    # ── BODY-CLUSTER-01: Multi-cue body language interpretation ───────────────
    def _rule_body_language_cluster(
        self,
        speaker_signals: list[dict],
        w: WindowFeatures,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-CLUSTER-01: Only emit an interpretation when 3+ independent cues
        within the same window align. Pease 2004: "Look for clusters of three or
        more cues before drawing conclusions."
        """
        signals: list[dict] = []
        signal_types = {s.get("signal_type") for s in speaker_signals}
        signal_texts = {s.get("value_text") for s in speaker_signals}

        def _avg_conf() -> float:
            if not speaker_signals:
                return 0.3
            return sum(s.get("confidence", 0.3) for s in speaker_signals) / len(speaker_signals)

        # ── Skepticism / Critical Evaluation ──────────────────────────────────
        skep = 0
        skep_ev: list[str] = []
        if any("chin" in s.get("value_text", "") for s in speaker_signals
               if s.get("signal_type") == "face_region_touch"):
            skep += 1; skep_ev.append("chin_touch")
        if "backward_lean" in signal_texts:
            skep += 1; skep_ev.append("backward_lean")
        if "lip_pursing" in signal_types:
            skep += 1; skep_ev.append("lip_pursing")
        if "low_engagement" in signal_texts or "reduced_attention" in signal_texts:
            skep += 1; skep_ev.append("low_engagement")
        if "head_shake" in signal_types:
            skep += 1; skep_ev.append("head_shake")
        if skep >= 3:
            signals.append(self._make_signal(
                rule_id="BODY-CLUSTER-01", signal_type="body_language_cluster",
                speaker_id=speaker_id, value=round(skep / 5.0, 2),
                value_text="skepticism_cluster",
                confidence=min(_avg_conf() * (skep / 3.0) * conf_mult, 0.60),
                window_start_ms=w.window_start_ms, window_end_ms=w.window_end_ms,
                metadata={"cluster_type": "skepticism", "cue_count": skep, "evidence": skep_ev},
            ))

        # ── Stress / Anxiety ──────────────────────────────────────────────────
        stress = 0
        stress_ev: list[str] = []
        if any("neck" in s.get("value_text", "") for s in speaker_signals
               if s.get("signal_type") == "face_region_touch"):
            stress += 1; stress_ev.append("neck_touch")
        if "body_fidgeting" in signal_types:
            stress += 1; stress_ev.append("fidgeting")
        if "facial_stress" in signal_types:
            stress += 1; stress_ev.append("facial_stress")
        if "elevated_blink_rate" in signal_texts:
            stress += 1; stress_ev.append("elevated_blink")
        if any(z in s.get("value_text", "") for s in speaker_signals
               for z in ("nose", "ear")
               if s.get("signal_type") == "face_region_touch"):
            stress += 1; stress_ev.append("self_soothing_touch")
        if stress >= 3:
            signals.append(self._make_signal(
                rule_id="BODY-CLUSTER-01", signal_type="body_language_cluster",
                speaker_id=speaker_id, value=round(stress / 5.0, 2),
                value_text="stress_anxiety_cluster",
                confidence=min(_avg_conf() * (stress / 3.0) * conf_mult, 0.55),
                window_start_ms=w.window_start_ms, window_end_ms=w.window_end_ms,
                metadata={"cluster_type": "stress_anxiety", "cue_count": stress, "evidence": stress_ev},
            ))

        # ── Confidence / Authority ─────────────────────────────────────────────
        conf = 0
        conf_ev: list[str] = []
        if "finger_steepling" in signal_types:
            conf += 1; conf_ev.append("finger_steepling")
        if "upright_power_posture" in signal_texts:
            conf += 1; conf_ev.append("upright_posture")
        if "high_attention" in signal_texts or "sustained_eye_contact" in signal_texts:
            conf += 1; conf_ev.append("steady_gaze")
        if "forward_lean" in signal_texts:
            conf += 1; conf_ev.append("forward_lean")
        if w.hand_velocity_mean > 0.3:
            conf += 1; conf_ev.append("active_gestures")
        if w.open_palms_pct > 0.40:
            conf += 1; conf_ev.append("open_palms")
        if conf >= 3:
            signals.append(self._make_signal(
                rule_id="BODY-CLUSTER-01", signal_type="body_language_cluster",
                speaker_id=speaker_id, value=round(conf / 5.0, 2),
                value_text="confidence_authority_cluster",
                confidence=min(_avg_conf() * (conf / 3.0) * conf_mult, 0.60),
                window_start_ms=w.window_start_ms, window_end_ms=w.window_end_ms,
                metadata={"cluster_type": "confidence_authority", "cue_count": conf, "evidence": conf_ev},
            ))

        # ── Disengagement / Boredom ────────────────────────────────────────────
        bored = 0
        bored_ev: list[str] = []
        if any("cheek_rest_fatigue" in s.get("value_text", "") for s in speaker_signals):
            bored += 1; bored_ev.append("head_resting_on_hand")
        elif any("head_resting_disengagement" in s.get("value_text", "") for s in speaker_signals):
            # elif: cheek_rest_fatigue and head_resting_disengagement both describe head-on-hand;
            # counting both from the same gesture would inflate bored past the 3-cue threshold
            # on a single cue, violating the cluster rule.
            bored += 1; bored_ev.append("head_supported_disengaged")
        if "backward_lean" in signal_texts:
            bored += 1; bored_ev.append("backward_lean")
        if "low_engagement" in signal_texts:
            bored += 1; bored_ev.append("low_engagement")
        if "sustained_distraction" in signal_types:
            bored += 1; bored_ev.append("distracted")
        if w.hand_velocity_mean < 0.1 and w.hands_detected_rate > 0.3:
            bored += 1; bored_ev.append("no_gestures")
        if "arms_crossed" in signal_types:
            bored += 1; bored_ev.append("arms_crossed")
        if bored >= 3:
            signals.append(self._make_signal(
                rule_id="BODY-CLUSTER-01", signal_type="body_language_cluster",
                speaker_id=speaker_id, value=round(bored / 6.0, 2),
                value_text="disengagement_boredom_cluster",
                confidence=min(_avg_conf() * (bored / 3.0) * conf_mult, 0.55),
                window_start_ms=w.window_start_ms, window_end_ms=w.window_end_ms,
                metadata={"cluster_type": "disengagement_boredom", "cue_count": bored, "evidence": bored_ev},
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

    # ── BODY-INTERACT-01: Cross-speaker interaction detection ─────────────────
    def _rule_cross_speaker_interaction(
        self,
        w: "WindowFeatures",
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        BODY-INTERACT-01: Emit a signal when a listener shows a sustained
        reaction (3+ interaction frames in the window) while another speaker
        is active.  Confidence scales with frame count, capped at 0.55.
        """
        signals: list[dict] = []

        interaction = getattr(w, "dominant_interaction", "")
        count = getattr(w, "interaction_count", 0)

        if not interaction or count < 4:
            return signals

        # dominant_interaction format: "{reactor_id}_{interaction_type}"
        parts = interaction.rsplit("_", 1)
        if len(parts) != 2:
            return signals

        reactor_id, interaction_type = parts

        # reactor_id is "Face_N" (face-index format from frame-level detection).
        # speaker_id is a diarization label like "Speaker_2" — the two namespaces
        # don't match, so compare using the window's face_index instead.
        our_face_id = f"Face_{w.face_index}"
        if our_face_id != reactor_id:
            return signals

        label_map = {
            "agrees":        "agreement_reaction",
            "disagrees":     "disagreement_reaction",
            "uncomfortable": "discomfort_reaction",
            "incongruent":   "incongruent_reaction",
            "disengaged":    "disengagement_reaction",
        }
        label = label_map.get(interaction_type)
        if not label:
            return signals

        confidence = min(count / 15.0 * conf_mult, 0.55)

        signals.append(self._make_signal(
            rule_id="BODY-INTERACT-01",
            signal_type="cross_speaker_interaction",
            speaker_id=speaker_id,
            value=round(min(count / 10.0, 1.0), 2),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "interaction_type": interaction_type,
                "frame_count": count,
                "reactor": reactor_id,
            },
        ))
        return signals
