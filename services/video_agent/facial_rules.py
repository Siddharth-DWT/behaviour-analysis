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

from base_rule_engine import BaseVideoRuleEngine

try:
    from feature_extractor import WindowFeatures
    from calibration import FacialBaseline, BodyBaseline, GazeBaseline
except ImportError:
    from services.video_agent.feature_extractor import WindowFeatures
    from services.video_agent.calibration import FacialBaseline, BodyBaseline, GazeBaseline

logger = logging.getLogger("nexus.video.facial_rules")

# ── Blendshape → emotion groupings (MediaPipe 52 canonical names) ──────────────
EMOTION_BLENDSHAPES: dict[str, dict[str, float]] = {
    "happy":    {"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6, "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4},
    "sad":      {"mouthFrownLeft": 0.5, "mouthFrownRight": 0.5, "browInnerUp": 0.3, "eyeSquintLeft": 0.2, "eyeSquintRight": 0.2},
    "angry":    {"browDownLeft": 0.6, "browDownRight": 0.6, "noseSneerLeft": 0.3, "noseSneerRight": 0.3, "mouthPressLeft": 0.1},
    "surprised":{"eyeWideLeft": 0.5, "eyeWideRight": 0.5, "jawOpen": 0.3, "browInnerUp": 0.2},
    "disgusted":{"noseSneerLeft": 0.5, "noseSneerRight": 0.5, "mouthShrugUpper": 0.3, "browDownLeft": 0.2},
    "contempt": {"mouthLeft": 0.6, "cheekSquintLeft": 0.3, "mouthDimpleLeft": 0.1},
    "fearful":  {"eyeWideLeft": 0.4, "eyeWideRight": 0.4, "browInnerUp": 0.4, "mouthStretchLeft": 0.2},
}

# confidence caps per emotion (deception-adjacent emotions capped lower)
EMOTION_CONF_CAPS: dict[str, float] = {
    "happy":     0.70,
    "sad":       0.55,
    "angry":     0.55,
    "surprised": 0.60,
    "disgusted": 0.35,
    "contempt":  0.35,
    "fearful":   0.50,
}

# Stress blendshapes with weights (Ekman: AU4 = brow lowerer, AU17 = chin raiser)
STRESS_INDICATORS: dict[str, float] = {
    "browDownLeft":   1.0,
    "browDownRight":  1.0,
    "mouthPressLeft": 1.2,
    "mouthPressRight":1.2,
    "jawForward":     0.5,
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
    EMOTION_DELTA_THRESHOLD  = 0.08   # min blendshape delta above neutral to fire
    SMILE_DUCHENNE_THRESHOLD = 0.05   # min mouthSmile delta for Duchenne candidacy
    CHEEK_DUCHENNE_THRESHOLD = 0.05   # min cheekSquint delta to confirm Duchenne
    SMILE_SOCIAL_THRESHOLD   = 0.08   # higher bar for social smile (no eye crinkling)
    STRESS_THRESHOLD         = 0.35   # weighted stress score above which rule fires
    STRESS_HIGH_THRESHOLD    = 0.55
    ENGAGEMENT_HIGH          = 0.60   # pose variance delta → high engagement
    ENGAGEMENT_LOW           = 0.20   # pose variance delta → low engagement
    MIN_FACE_RATE            = 0.30   # skip window if face detected < 30% of frames

    def evaluate(
        self,
        windows_by_speaker: dict,
        baselines: dict,
        session_id: str = "",
        meeting_type: str = "general",
    ) -> list[dict]:
        signals: list[dict] = []
        for speaker_id, windows in windows_by_speaker.items():
            facial_bl, _, _ = baselines.get(
                speaker_id,
                (FacialBaseline(speaker_id=speaker_id), None, None),
            )
            cal_conf = facial_bl.calibration_confidence

            for w in windows:
                if w.face_detection_rate < self.MIN_FACE_RATE:
                    continue
                if not w.blendshapes_mean:
                    continue

                conf_mult = cal_conf * w.face_detection_rate

                signals += self._rule_emotion(w, facial_bl, speaker_id, conf_mult)
                signals += self._rule_smile(w, facial_bl, speaker_id, conf_mult)
                signals += self._rule_stress(w, facial_bl, speaker_id, conf_mult)
                signals += self._rule_engagement(w, facial_bl, speaker_id, conf_mult)
                signals += self._rule_valence_arousal(w, facial_bl, speaker_id, conf_mult)

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
            if delta > self.EMOTION_DELTA_THRESHOLD:
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
    ) -> list[dict]:
        """
        Duchenne vs social smile discrimination (Ekman 2009).
        Duchenne: mouthSmile delta AND cheekSquint delta both elevated.
        Social:   mouthSmile delta elevated but cheekSquint at baseline.
        """
        smile_delta = (
            _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "mouthSmileLeft")
            + _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "mouthSmileRight")
        ) / 2.0

        cheek_delta = (
            _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "cheekSquintLeft")
            + _blendshape_delta(w.blendshapes_mean, bl.blendshapes_neutral, "cheekSquintRight")
        ) / 2.0

        if smile_delta < self.SMILE_DUCHENNE_THRESHOLD:
            return []

        if cheek_delta >= self.CHEEK_DUCHENNE_THRESHOLD:
            smile_type = "duchenne"
            confidence = min(smile_delta * 0.8 * conf_mult, 0.55)
        elif smile_delta >= self.SMILE_SOCIAL_THRESHOLD:
            smile_type = "social"
            confidence = min(smile_delta * 0.5 * conf_mult, 0.45)
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
                "smile_delta": round(smile_delta, 4),
                "cheek_delta": round(cheek_delta, 4),
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
        confidence = min(normalised * conf_mult, 0.55)

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
        Engagement from head-pose variance + overall blendshape expressivity.
        High variance = active listening / animated expression.
        Low variance = flat affect / disengagement.
        """
        pose_var_delta = w.head_pose_variance - bl.head_pose_variance_baseline
        bs_expressivity = (
            sum(w.blendshapes_std.values()) / len(w.blendshapes_std)
            if w.blendshapes_std else 0.0
        )

        # Normalise pose_var_delta to 0–1 range (assume max meaningful delta = 30°²)
        norm_var = max(0.0, min(abs(pose_var_delta) / 30.0, 1.0))
        combined = 0.6 * norm_var + 0.4 * bs_expressivity

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
                "pose_variance_delta": round(pose_var_delta, 4),
                "blendshape_expressivity": round(bs_expressivity, 4),
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

        if abs(valence) < 0.05 and arousal < 0.05:
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
