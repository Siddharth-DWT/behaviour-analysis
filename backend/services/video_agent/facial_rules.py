"""
Facial Rule Engine — Phase 2B
Implements FACE-EMO-01, FACE-SMILE-01, FACE-STRESS-01, FACE-ENG-01, FACE-VA-01.
FACE-MICRO-01 is disabled (requires high-FPS video for reliable sub-200ms detection).

Research anchors:
  Ekman & Friesen 1978  — FACS action units / 6 basic emotions
  Ekman 2009            — Duchenne marker: orbicularis oculi (cheek squint)
  Westphal et al. 2015  — Automatic blendshape-to-AU mapping
"""
import logging
import math
from typing import Optional

from .base_rule_engine import BaseVideoRuleEngine
from .feature_extractor import WindowFeatures
from .calibration import FacialBaseline, BodyBaseline, GazeBaseline

logger = logging.getLogger("nexus.video.facial_rules")

# ── Blendshape → emotion groupings (MediaPipe 52 canonical names) ──────────────
EMOTION_BLENDSHAPES: dict[str, dict[str, float]] = {
    "happy":    {"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6, "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4},
    "sad":      {"mouthFrownLeft": 0.5, "mouthFrownRight": 0.5, "browInnerUp": 0.3, "eyeSquintLeft": 0.2, "eyeSquintRight": 0.2},
    "angry":    {"browDownLeft": 0.6, "browDownRight": 0.6, "eyeWideLeft": 0.25, "eyeWideRight": 0.25, "jawForward": 0.20, "noseSneerLeft": 0.20, "noseSneerRight": 0.20, "mouthPressLeft": 0.10},
    "surprised":{"eyeWideLeft": 0.5, "eyeWideRight": 0.5, "jawOpen": 0.3, "browInnerUp": 0.2},
    "disgusted":{"noseSneerLeft": 0.5, "noseSneerRight": 0.5, "mouthShrugUpper": 0.3, "browDownLeft": 0.2},
    "contempt": {"mouthLeft": 0.6, "cheekSquintLeft": 0.3, "mouthDimpleLeft": 0.1},
    "fearful":  {"eyeWideLeft": 0.4, "eyeWideRight": 0.4, "browInnerUp": 0.4, "mouthStretchLeft": 0.2},
}

# Per-emotion delta overrides — emotions whose blendshapes overlap with normal
# video-call expressions (concentration, squinting, screen glare) need higher
# thresholds to avoid false positives.
EMOTION_DELTA_OVERRIDES: dict[str, float] = {
    "angry":     0.15,   # browDown fires on concentration / squinting
    "disgusted": 0.14,   # noseSneer fires on screen glare
    "contempt":  0.14,   # mouthLeft fires on asymmetric talking
    "sad":       0.10,   # mouthFrown fires on neutral resting face
    "fearful":   0.10,   # eyeWide fires on attentive listening
}

# confidence caps per emotion (deception-adjacent emotions capped lower)
EMOTION_CONF_CAPS: dict[str, float] = {
    "happy":     0.70,
    "sad":       0.55,
    "angry":     0.55,
    "surprised": 0.60,
    "disgusted": 0.40,
    "contempt":  0.35,
    "fearful":   0.55,
}

# Stress blendshapes with weights (Ekman: AU4 = brow lowerer, AU17 = chin raiser)
STRESS_INDICATORS: dict[str, float] = {
    "browDownLeft":   1.0,
    "browDownRight":  1.0,
    "mouthPressLeft": 1.2,
    "mouthPressRight":1.2,
    "jawForward":     0.5,
    "jawLeft":        0.4,   # lateral jaw deviation = clenching/grinding tension
    "jawRight":       0.4,
    "eyeSquintLeft":  0.8,
    "eyeSquintRight": 0.8,
    "noseSneerLeft":  0.4,
    "noseSneerRight": 0.4,
}


def _weighted_blend_score(bs_mean: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted average of blendshape values present in the window."""
    total_w = total_score = 0.0
    for key, w in weights.items():
        v = bs_mean.get(key, 0.0)
        total_score += v * w
        total_w += w
    return total_score / total_w if total_w > 0 else 0.0


def _blendshape_delta(current: dict[str, float], baseline: dict[str, float], key: str) -> float:
    """Deviation of a blendshape from its neutral baseline value."""
    return current.get(key, 0.0) - baseline.get(key, 0.0)


class FacialRuleEngine(BaseVideoRuleEngine):
    """
    Runs facial expression rules across all speakers.

    DSA: processes speakers sequentially, windows sequentially — O(S × W).
    Stateless between calls; all state lives in WindowFeatures + FacialBaseline.
    """

    AGENT_NAME = "video"

    # Thresholds (overridable via rule_config in Phase 3)
    EMOTION_DELTA_THRESHOLD  = 0.12   # min blendshape delta above neutral to fire (was 0.08)
    SMILE_DUCHENNE_THRESHOLD = 0.08   # min mouthSmile delta for Duchenne candidacy (was 0.05)
    CHEEK_DUCHENNE_THRESHOLD = 0.07   # min cheekSquint delta to confirm Duchenne (was 0.05)
    SMILE_SOCIAL_THRESHOLD   = 0.12   # higher bar for social smile (no eye crinkling) (was 0.08)
    STRESS_THRESHOLD         = 0.35   # weighted stress score above which rule fires
    STRESS_HIGH_THRESHOLD    = 0.55
    ENGAGEMENT_HIGH          = 0.70   # combined score → high engagement (was 0.60)
    ENGAGEMENT_LOW           = 0.20   # combined score → low engagement
    MIN_FACE_RATE            = 0.30   # skip window if face detected < 30% of frames
    MIN_SIGNAL_CONFIDENCE    = 0.18   # discard signals below this before storage

    def evaluate(
        self,
        windows_by_speaker: dict,
        baselines: dict,
        session_id: str = "",
        meeting_type: str = "general",
    ) -> list[dict]:
        signals: list[dict] = []
        for speaker_id, windows in windows_by_speaker.items():
            windows = sorted(windows, key=lambda x: x.window_start_ms)
            facial_bl, _, _ = baselines.get(
                speaker_id,
                (FacialBaseline(speaker_id=speaker_id), None, None),
            )
            cal_conf = facial_bl.calibration_confidence

            prev_w: Optional[WindowFeatures] = None
            for w in windows:
                if w.face_detection_rate < self.MIN_FACE_RATE:
                    prev_w = w
                    continue
                if not w.blendshapes_mean:
                    prev_w = w
                    continue

                self._current_w = w
                conf_mult = cal_conf * w.face_detection_rate

                # F-2: Skin tone confidence modifier (Buolamwini & Gebru 2018).
                # Lower face luminance → lower blendshape accuracy → reduce confidence.
                face_lum = getattr(w, "face_luminance", 0.5)
                if face_lum < 0.35:
                    conf_mult *= 0.85

                window_signals: list[dict] = []
                window_signals += self._rule_emotion(w, facial_bl, speaker_id, conf_mult)
                window_signals += self._rule_smile(w, facial_bl, speaker_id, conf_mult, prev_w)
                window_signals += self._rule_laughter(w, facial_bl, speaker_id, conf_mult)
                window_signals += self._rule_stress(w, facial_bl, speaker_id, conf_mult)
                window_signals += self._rule_engagement(w, facial_bl, speaker_id, conf_mult)
                window_signals += self._rule_valence_arousal(w, facial_bl, speaker_id, conf_mult)
                window_signals += self._rule_lip_pursing(w, facial_bl, speaker_id, conf_mult)

                signals += [
                    s for s in window_signals
                    if s is not None and s.get("confidence", 0) >= self.MIN_SIGNAL_CONFIDENCE
                ]
                prev_w = w

        logger.info(f"[{session_id}] FacialRuleEngine: {len(signals)} signals")
        return signals

    # ── FACE-EMO-01 ───────────────────────────────────────────────────────────
    def _rule_emotion(
        self,
        w: WindowFeatures,
        bl: FacialBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        Dominant emotion detection via blendshape-to-AU mapping.
        Fires when any emotion score exceeds EMOTION_DELTA_THRESHOLD above neutral.
        """
        scores: dict[str, float] = {}
        for emotion, weights in EMOTION_BLENDSHAPES.items():
            raw = _weighted_blend_score(w.blendshapes_mean, weights)
            baseline_raw = _weighted_blend_score(bl.blendshapes_neutral, weights)
            delta = raw - baseline_raw
            min_delta = EMOTION_DELTA_OVERRIDES.get(emotion, self.EMOTION_DELTA_THRESHOLD)
            if delta > min_delta:
                scores[emotion] = delta

        if not scores:
            return []

        dominant = max(scores, key=lambda e: scores[e])
        intensity = min(scores[dominant], 1.0)
        cap = EMOTION_CONF_CAPS.get(dominant, 0.55)
        confidence = min(intensity * conf_mult, cap)

        return [self._make_signal(
            rule_id="FACE-EMO-01",
            signal_type="facial_emotion",
            speaker_id=speaker_id,
            value=round(intensity, 4),
            value_text=dominant,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "emotion_scores": {e: round(s, 4) for e, s in scores.items()},
                "face_detection_rate": w.face_detection_rate,
            },
        )]

    # ── FACE-SMILE-01 ─────────────────────────────────────────────────────────
    def _rule_smile(
        self,
        w: WindowFeatures,
        bl: FacialBaseline,
        speaker_id: str,
        conf_mult: float,
        prev_window: Optional[WindowFeatures] = None,
    ) -> list[dict]:
        """
        Duchenne vs social smile with onset speed + symmetry (F-3).
        Ekman 2009: Duchenne = AU6 + AU12; onset, symmetry, and cheek squint all count.
        """
        smile_left  = _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "mouthSmileLeft")
        smile_right = _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "mouthSmileRight")
        smile_delta = (smile_left + smile_right) / 2.0

        cheek_delta = (
            _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "cheekSquintLeft")
            + _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "cheekSquintRight")
        ) / 2.0

        if smile_delta < self.SMILE_DUCHENNE_THRESHOLD:
            return []

        # ── Symmetry: genuine smiles have < 25% left-right asymmetry (Ekman 1982) ──
        asymmetry = abs(smile_left - smile_right) / max(smile_delta, 0.05)
        is_symmetric = asymmetry < 0.25

        # ── Onset speed: requires previous window for frame-to-frame delta ──────
        onset_speed = "unknown"
        if prev_window and prev_window.blendshapes_mean:
            prev_smile = (
                prev_window.blendshapes_mean.get("mouthSmileLeft", 0.0)
                + prev_window.blendshapes_mean.get("mouthSmileRight", 0.0)
            ) / 2.0
            curr_smile = (
                w.blendshapes_mean.get("mouthSmileLeft", 0.0)
                + w.blendshapes_mean.get("mouthSmileRight", 0.0)
            ) / 2.0
            delta_per_window = curr_smile - prev_smile
            # Gradual onset (300-500ms ramp) → likely genuine
            # Abrupt onset (< 200ms snap) → likely social/posed
            if delta_per_window > 0.25:
                onset_speed = "abrupt"
            elif delta_per_window > 0.05:
                onset_speed = "gradual"

        # ── Classify: each positive indicator supports Duchenne ──────────────
        duchenne_count = 0
        if cheek_delta >= self.CHEEK_DUCHENNE_THRESHOLD:
            duchenne_count += 1   # Eye crinkle (AU6 proxy)
        if is_symmetric:
            duchenne_count += 1   # Bilateral symmetry
        if onset_speed == "gradual":
            duchenne_count += 1   # Smooth onset

        if duchenne_count >= 2:
            smile_type = "duchenne"
            confidence = min(smile_delta * 0.8 * conf_mult, 0.65)
        elif smile_delta >= self.SMILE_SOCIAL_THRESHOLD:
            smile_type = "social"
            confidence = min(smile_delta * 0.5 * conf_mult, 0.60)
        else:
            return []

        return [self._make_signal(
            rule_id="FACE-SMILE-01",
            signal_type="smile_type",
            speaker_id=speaker_id,
            value=round(smile_delta, 4),
            value_text=smile_type,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "smile_delta":        round(smile_delta, 4),
                "cheek_delta":        round(cheek_delta, 4),
                "asymmetry":          round(asymmetry, 3),
                "is_symmetric":       is_symmetric,
                "onset_speed":        onset_speed,
                "duchenne_indicators": duchenne_count,
            },
        )]

    # ── FACE-STRESS-01 ────────────────────────────────────────────────────────
    def _rule_stress(
        self,
        w: WindowFeatures,
        bl: FacialBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        Facial stress via weighted brow/jaw/eye tension indicators.
        Delta-from-baseline per indicator, then weighted sum.
        """
        stress_score = 0.0
        weight_sum = 0.0
        indicator_hits: dict[str, float] = {}

        for key, weight in STRESS_INDICATORS.items():
            delta = _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, key)
            if delta > 0:
                stress_score += delta * weight
                weight_sum += weight
                indicator_hits[key] = round(delta, 4)

        if weight_sum == 0:
            return []

        normalised = stress_score / weight_sum
        if normalised < self.STRESS_THRESHOLD:
            return []

        label = "high_facial_stress" if normalised >= self.STRESS_HIGH_THRESHOLD else "moderate_facial_stress"
        # AU23/24 (lip tightening) have F1 ≈ 40-55% in webcam conditions — cap lowered
        # from 0.55 to 0.38 per RESEARCH.md (FACE-STRESS-01 note)
        confidence = min(normalised * conf_mult, 0.38)

        return [self._make_signal(
            rule_id="FACE-STRESS-01",
            signal_type="facial_stress",
            speaker_id=speaker_id,
            value=round(normalised, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={"indicators": indicator_hits},
        )]

    # ── FACE-ENG-01 ───────────────────────────────────────────────────────────
    def _rule_engagement(
        self,
        w: WindowFeatures,
        bl: FacialBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        5-factor engagement composite (F-5, Whitehill 2014):
          F1 head-pose variance (0.30) — proxy for AU activity / animated response
          F2 blendshape expressivity (0.25) — overall facial animation
          F3 smile presence (0.20) — mouthSmile mean as smile-frequency proxy
          F4 brow movement (0.15) — browDown/browInnerUp delta from neutral
          F5 face visible % (0.10) — proxy for screen-facing orientation
        """
        # F1: head-pose variance normalised to 0-1 (max meaningful delta ≈ 30°²)
        pose_var_delta = w.head_pose_variance - bl.head_pose_variance_baseline
        f1_pose = max(0.0, min(abs(pose_var_delta) / 30.0, 1.0))

        # F2: blendshape expressivity (std across all blendshapes)
        f2_express = (
            sum(w.blendshapes_std.values()) / len(w.blendshapes_std)
            if w.blendshapes_std else 0.0
        )

        # F3: smile presence (0.3 mean = 1.0 normalised)
        smile_mean = (
            w.blendshapes_mean.get("mouthSmileLeft", 0.0)
            + w.blendshapes_mean.get("mouthSmileRight", 0.0)
        ) / 2.0
        f3_smile = min(smile_mean / 0.3, 1.0)

        # F4: brow activity delta from neutral (more brow movement = more reactive)
        # browOuterUpLeft/Right captures "interested/curious" raise; browInnerUp captures concern
        brow_activity = (
            abs(_blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "browDownLeft"))
            + abs(_blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "browDownRight"))
            + abs(_blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "browInnerUp"))
            + abs(_blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "browOuterUpLeft"))
            + abs(_blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "browOuterUpRight"))
        ) / 5.0
        f4_brow = min(brow_activity / 0.15, 1.0)

        # F5: face detection rate as screen-facing proxy
        f5_visible = w.face_detection_rate

        combined = (
            0.30 * f1_pose
            + 0.25 * f2_express
            + 0.20 * f3_smile
            + 0.15 * f4_brow
            + 0.10 * f5_visible
        )

        if combined >= self.ENGAGEMENT_HIGH:
            label = "high_engagement"
            confidence = min(combined * conf_mult, 0.65)
        elif combined <= self.ENGAGEMENT_LOW:
            label = "low_engagement"
            confidence = min((1.0 - combined) * conf_mult * 0.7, 0.55)
        else:
            return []

        return [self._make_signal(
            rule_id="FACE-ENG-01",
            signal_type="facial_engagement",
            speaker_id=speaker_id,
            value=round(combined, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "f1_pose_variance":  round(f1_pose, 4),
                "f2_expressivity":   round(f2_express, 4),
                "f3_smile":          round(f3_smile, 4),
                "f4_brow_activity":  round(f4_brow, 4),
                "f5_face_visible":   round(f5_visible, 4),
            },
        )]

    # ── FACE-VA-01 ────────────────────────────────────────────────────────────
    def _rule_valence_arousal(
        self,
        w: WindowFeatures,
        bl: FacialBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        2D valence–arousal space classification.
        Valence:  mouthSmile - mouthFrown - browDown (positive to negative)
        Arousal:  eyeWide + jawOpen (low to high energy)
        """
        def delta(key: str) -> float:
            return _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, key)

        valence = (
            (delta("mouthSmileLeft") + delta("mouthSmileRight")) / 2.0
            - (delta("mouthFrownLeft") + delta("mouthFrownRight")) / 2.0
            - (delta("browDownLeft") + delta("browDownRight")) / 2.0
        )
        arousal = (
            (delta("eyeWideLeft") + delta("eyeWideRight")) / 2.0
            + delta("jawOpen") * 0.5
        )

        valence = max(-1.0, min(valence, 1.0))
        arousal = max(0.0, min(arousal, 1.0))

        if abs(valence) < 0.10 and arousal < 0.08:
            return []

        if valence > 0.05:
            v_label = "positive"
        elif valence < -0.05:
            v_label = "negative"
        else:
            v_label = "neutral"

        if arousal > 0.25:
            a_label = "high"
        elif arousal > 0.10:
            a_label = "moderate"
        else:
            a_label = "low"

        label = f"{v_label}_{a_label}"
        magnitude = math.sqrt(valence ** 2 + arousal ** 2)
        confidence = min(magnitude * 0.6 * conf_mult, 0.65)

        return [self._make_signal(
            rule_id="FACE-VA-01",
            signal_type="valence_arousal",
            speaker_id=speaker_id,
            value=round(valence, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
            },
        )]

    # ── FACE-LIPS-01: Lip pursing ─────────────────────────────────────────────
    def _rule_lip_pursing(
        self,
        w: WindowFeatures,
        bl: FacialBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        FACE-LIPS-01: Elevated mouthPucker while NOT speaking and NOT smiling.
        Only meaningful when lips are pursed during listening — during speech it
        is normal articulation. Pease 2004: pursed lips often signal disagreement
        the speaker is not voicing.
        """
        mouth_pucker = w.blendshapes_mean.get("mouthPucker", 0.0)
        mouth_smile = (
            w.blendshapes_mean.get("mouthSmileLeft", 0.0)
            + w.blendshapes_mean.get("mouthSmileRight", 0.0)
        ) / 2.0

        baseline_pucker = bl.blendshapes_neutral.get("mouthPucker", 0.05)
        delta = mouth_pucker - baseline_pucker

        if delta < 0.18:
            return []
        if mouth_pucker < 0.22:
            return []
        pucker_std = w.blendshapes_std.get("mouthPucker", 0.0)
        if pucker_std > 0.08:
            return []
        if mouth_smile > 0.15:
            return []
        if getattr(w, "is_speaking", True):
            return []

        confidence = min(delta * 2.0 * conf_mult, 0.45)
        return [self._make_signal(
            rule_id="FACE-LIPS-01",
            signal_type="lip_pursing",
            speaker_id=speaker_id,
            value=round(delta, 4),
            value_text="suppressed_disagreement",
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "mouth_pucker": round(mouth_pucker, 4),
                "baseline_pucker": round(baseline_pucker, 4),
                "delta": round(delta, 4),
                "is_listening": not getattr(w, "is_speaking", True),
            },
        )]

    # ── FACE-LAUGH-01: Laughter detection ────────────────────────────────────
    def _rule_laughter(
        self,
        w: WindowFeatures,
        bl: FacialBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        FACE-LAUGH-01: Laughter via rapid jawOpen oscillation + smile + eye crinkle.
        At 5 fps each frame is 200ms — true micro-laughter bursts are unresolvable, but
        sustained laughter within a 2s window leaves high jawOpen std (oscillating jaw)
        combined with elevated smile and cheekSquint (Duchenne eye crinkle = genuine).
        Ekman 2003: genuine amusement requires both AU6 (cheekSquint) and AU12 (mouthSmile).
        """
        if not getattr(w, "is_speaking", True):
            return []

        bs      = w.blendshapes_mean
        bs_std  = w.blendshapes_std
        bl_bs   = bl.blendshapes_neutral

        jaw_std  = bs_std.get("jawOpen", 0.0)
        jaw_mean = bs.get("jawOpen", 0.0)

        # Oscillating jaw (std > 0.06) + open jaw (mean > 0.08) = laugh-like movement
        if jaw_std < 0.06 or jaw_mean < 0.08:
            return []

        smile_mean  = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
        smile_base  = (bl_bs.get("mouthSmileLeft", 0.0) + bl_bs.get("mouthSmileRight", 0.0)) / 2.0
        smile_delta = smile_mean - smile_base
        if smile_delta < 0.10:
            return []

        cheek_delta = (
            (bs.get("cheekSquintLeft",  0.0) - bl_bs.get("cheekSquintLeft",  0.0)) +
            (bs.get("cheekSquintRight", 0.0) - bl_bs.get("cheekSquintRight", 0.0))
        ) / 2.0

        is_genuine = cheek_delta >= 0.06   # Duchenne AU6 — hardest to fake
        label      = "genuine_laughter" if is_genuine else "open_mouth_smile"
        cap        = 0.65 if is_genuine else 0.50
        confidence = min((jaw_std * 0.5 + smile_delta * 0.5) * conf_mult, cap)

        return [self._make_signal(
            rule_id="FACE-LAUGH-01",
            signal_type="laughter",
            speaker_id=speaker_id,
            value=round(jaw_std, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "jaw_open_std":  round(jaw_std, 4),
                "jaw_open_mean": round(jaw_mean, 4),
                "smile_delta":   round(smile_delta, 4),
                "cheek_delta":   round(cheek_delta, 4),
                "is_genuine":    is_genuine,
            },
        )]
