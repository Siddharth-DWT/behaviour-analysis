# services/video_agent/calibration.py
"""
NEXUS Video Agent — Calibration Module
Implements FACE-CAL-01, BODY-CAL-01, GAZE-CAL-01: per-speaker video baselines.

Mirrors the pattern of services/voiceAgent/calibration.py.
All rule engine detections operate on DEVIATIONS from these baselines,
not on absolute values (Ekman 1993; Laukka et al. 2011 — within-subject
comparisons dramatically outperform absolute thresholds).

Classes:
  FacialBaseline        — per-speaker facial / blendshape baseline (FACE-CAL-01)
  BodyBaseline          — per-speaker posture / movement baseline (BODY-CAL-01)
  GazeBaseline          — per-speaker gaze / blink baseline (GAZE-CAL-01)
  VideoCalibrationModule — builds all three baselines from first N windows
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from feature_extractor import WindowFeatures

logger = logging.getLogger("nexus.video.calibration")

# ─── Calibration constants ─────────────────────────────────────────────────
# 2-second windows → 45 windows ≈ 90 seconds (same target as voice agent)
CALIBRATION_MIN_WINDOWS:    int = 5
CALIBRATION_TARGET_WINDOWS: int = 45

# Bentivoglio 1997: conversational blink rate 26 bpm, silent-reading 15 bpm
DEFAULT_BLINK_RATE_BPM: float = 20.0

# Argyle 1972 / Sellen 1992: video-call gaze-to-camera ≈ 60-70% of talk time
DEFAULT_SCREEN_ENGAGEMENT: float = 0.65

# Camera-screen vertical offset on laptops (degrees down from camera to screen)
DEFAULT_CAMERA_SCREEN_OFFSET_Y: float = 0.08   # normalised iris units


# ══════════════════════════════════════════════════════════════════════════════
# Baseline dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FacialBaseline:
    """
    Per-speaker facial baseline computed from the first N video windows.
    Implements FACE-CAL-01.

    Fields mirror the structure of WindowFeatures so rule engines can
    call VideoCalibrationModule.compute_delta / compute_sigma directly.
    """
    speaker_id: str
    session_id: str = ""

    # Neutral blendshape values — dict[name → mean_score]
    blendshapes_neutral: dict[str, float] = field(default_factory=dict)
    blendshapes_std:     dict[str, float] = field(default_factory=dict)

    # Resting head pose
    head_pitch_mean: float = 0.0
    head_pitch_std:  float = 2.0    # small non-zero default avoids div-by-zero
    head_yaw_mean:   float = 0.0
    head_yaw_std:    float = 2.0
    head_roll_mean:  float = 0.0
    head_roll_std:   float = 1.0

    # Natural head-pose variability (sum of pitch/yaw/roll variances)
    head_pose_variance_baseline: float = 0.0

    # Resting blink rate (blinks per minute)
    blink_rate_bpm: float = DEFAULT_BLINK_RATE_BPM

    # Resting eye aspect ratio (open eye, relaxed)
    ear_mean: float = 0.30

    # Calibration metadata
    sample_count:             int   = 0
    calibration_confidence:   float = 0.0

    def update_confidence(self, n_windows: int) -> None:
        self.sample_count = n_windows
        if n_windows < CALIBRATION_MIN_WINDOWS:
            self.calibration_confidence = 0.10
        else:
            # Linear ramp: 5 windows → 0.20, 45 windows → 0.90
            self.calibration_confidence = round(
                min(0.20 + (n_windows / CALIBRATION_TARGET_WINDOWS) * 0.70, 0.90), 3
            )


@dataclass
class BodyBaseline:
    """
    Per-speaker body / posture baseline.
    Implements BODY-CAL-01.
    """
    speaker_id: str
    session_id: str = ""

    # Resting posture geometry
    shoulder_angle_mean: float = 0.0
    shoulder_angle_std:  float = 1.0

    # Neutral lean (0 = upright; positive = forward)
    spine_angle_mean: float = 0.0
    spine_angle_std:  float = 2.0

    # Head-to-shoulder distance (normalised by frame height)
    head_shoulder_dist_mean: float = 0.0
    head_shoulder_dist_std:  float = 0.02

    # Resting fidget level (avg body movement per window)
    body_movement_mean: float = 0.0
    body_movement_std:  float = 0.01   # small non-zero default

    sample_count:           int   = 0
    calibration_confidence: float = 0.0

    def update_confidence(self, n_windows: int) -> None:
        self.sample_count = n_windows
        if n_windows < CALIBRATION_MIN_WINDOWS:
            self.calibration_confidence = 0.10
        else:
            self.calibration_confidence = round(
                min(0.20 + (n_windows / CALIBRATION_TARGET_WINDOWS) * 0.70, 0.90), 3
            )


@dataclass
class GazeBaseline:
    """
    Per-speaker gaze / blink baseline.
    Implements GAZE-CAL-01.

    Camera-screen offset: on most laptops the webcam sits 5-15° above the
    screen centre. People appearing to look at the screen actually look
    slightly downward — this offset is learned from the first N windows
    and used to calibrate GAZE-DIR-01 / GAZE-CONTACT-01.
    """
    speaker_id: str
    session_id: str = ""

    # Camera-to-screen gaze offset (learned from calibration windows)
    # Positive gaze_y_offset → camera is above centre of screen
    gaze_y_offset: float = DEFAULT_CAMERA_SCREEN_OFFSET_Y
    gaze_x_offset: float = 0.0

    # Natural screen-engagement rate (fraction of time looking at screen)
    screen_engagement_rate: float = DEFAULT_SCREEN_ENGAGEMENT

    # Baseline blink rate (blinks per minute)
    blink_rate_bpm: float = DEFAULT_BLINK_RATE_BPM

    # Gaze spread (standard deviation of iris offset within windows)
    gaze_x_std_mean: float = 0.05    # typical small variance when on-screen
    gaze_y_std_mean: float = 0.05

    sample_count:           int   = 0
    calibration_confidence: float = 0.0

    def update_confidence(self, n_windows: int) -> None:
        self.sample_count = n_windows
        if n_windows < CALIBRATION_MIN_WINDOWS:
            self.calibration_confidence = 0.10
        else:
            self.calibration_confidence = round(
                min(0.20 + (n_windows / CALIBRATION_TARGET_WINDOWS) * 0.70, 0.90), 3
            )


# ══════════════════════════════════════════════════════════════════════════════
# VideoCalibrationModule
# ══════════════════════════════════════════════════════════════════════════════

class VideoCalibrationModule:
    """
    Builds per-speaker FacialBaseline, BodyBaseline, and GazeBaseline
    from the first N video windows.

    Mirrors VoiceAgent's CalibrationModule:
      - Uses first CALIBRATION_TARGET_WINDOWS windows (≈ 90 seconds)
      - Computes mean + std for all numeric features
      - Provides compute_delta / compute_sigma static helpers

    OOP principles:
      - Single responsibility: baseline construction only
      - Encapsulation: numpy arrays never escape; callers get typed dataclasses
      - Reuse: static delta/sigma helpers shared with rule engines
    """

    def build_all_baselines(
        self,
        speaker_id: str,
        session_id: str,
        windows: list[WindowFeatures],
        max_windows: int = CALIBRATION_TARGET_WINDOWS,
    ) -> tuple[FacialBaseline, BodyBaseline, GazeBaseline]:
        """
        Build all three baselines from the first max_windows windows.

        Returns:
            (FacialBaseline, BodyBaseline, GazeBaseline) — always returns
            objects even with zero data (confidence will be 0.10).
        """
        cal = windows[:max_windows]
        facial = self._build_facial_baseline(speaker_id, session_id, cal)
        body   = self._build_body_baseline(speaker_id, session_id, cal)
        gaze   = self._build_gaze_baseline(speaker_id, session_id, cal)
        return facial, body, gaze

    # ── Private builders ────────────────────────────────────────────────────────

    def _build_facial_baseline(
        self,
        speaker_id: str,
        session_id: str,
        windows: list[WindowFeatures],
    ) -> FacialBaseline:
        baseline = FacialBaseline(speaker_id=speaker_id, session_id=session_id)

        face_wins = [w for w in windows if w.face_detection_rate > 0.3]
        if not face_wins:
            logger.warning(f"[{speaker_id}] No usable face windows for facial baseline.")
            baseline.update_confidence(0)
            return baseline

        # ── Blendshape neutral values ──────────────────────────────────────────
        all_bs_keys: set[str] = set().union(*(w.blendshapes_mean.keys() for w in face_wins))
        for key in all_bs_keys:
            vals = [w.blendshapes_mean[key] for w in face_wins if key in w.blendshapes_mean]
            if vals:
                arr = np.array(vals, dtype=np.float32)
                baseline.blendshapes_neutral[key] = float(np.mean(arr))
                baseline.blendshapes_std[key]     = float(np.std(arr))

        # ── Head pose ──────────────────────────────────────────────────────────
        pitches = np.array([w.head_pitch_mean for w in face_wins], dtype=np.float32)
        yaws    = np.array([w.head_yaw_mean   for w in face_wins], dtype=np.float32)
        rolls   = np.array([w.head_roll_mean  for w in face_wins], dtype=np.float32)

        baseline.head_pitch_mean = float(np.mean(pitches))
        baseline.head_pitch_std  = max(float(np.std(pitches)), 0.5)
        baseline.head_yaw_mean   = float(np.mean(yaws))
        baseline.head_yaw_std    = max(float(np.std(yaws)), 0.5)
        baseline.head_roll_mean  = float(np.mean(rolls))
        baseline.head_roll_std   = max(float(np.std(rolls)), 0.5)
        baseline.head_pose_variance_baseline = float(
            np.mean([w.head_pose_variance for w in face_wins])
        )

        # ── Blink rate ─────────────────────────────────────────────────────────
        blink_rates = [w.blink_rate_bpm for w in face_wins if w.blink_rate_bpm > 0]
        if blink_rates:
            # Clip extreme outliers (>60 bpm is measurement noise, not real blinking)
            clipped = [b for b in blink_rates if b <= 60.0]
            if clipped:
                baseline.blink_rate_bpm = float(np.mean(clipped))

        # ── EAR (open-eye baseline) ─────────────────────────────────────────────
        ears = [w.ear_mean for w in face_wins if w.ear_mean > 0.15]
        if ears:
            baseline.ear_mean = float(np.mean(ears))

        baseline.update_confidence(len(face_wins))
        logger.info(
            f"[{speaker_id}] Facial baseline: "
            f"pitch={baseline.head_pitch_mean:.1f}°±{baseline.head_pitch_std:.1f}°, "
            f"yaw={baseline.head_yaw_mean:.1f}°±{baseline.head_yaw_std:.1f}°, "
            f"blink={baseline.blink_rate_bpm:.1f}bpm, "
            f"conf={baseline.calibration_confidence:.2f} "
            f"from {len(face_wins)} windows"
        )
        return baseline

    def _build_body_baseline(
        self,
        speaker_id: str,
        session_id: str,
        windows: list[WindowFeatures],
    ) -> BodyBaseline:
        baseline = BodyBaseline(speaker_id=speaker_id, session_id=session_id)

        body_wins = [w for w in windows if w.body_detection_rate > 0.3]
        if not body_wins:
            logger.warning(f"[{speaker_id}] No usable body windows for body baseline.")
            baseline.update_confidence(0)
            return baseline

        sh_angles = np.array([w.shoulder_angle_mean     for w in body_wins], dtype=np.float32)
        sp_angles = np.array([w.spine_angle_mean         for w in body_wins], dtype=np.float32)
        hs_dists  = np.array([w.head_shoulder_dist_mean  for w in body_wins], dtype=np.float32)
        movements = np.array([w.body_movement_mean        for w in body_wins], dtype=np.float32)

        baseline.shoulder_angle_mean     = float(np.mean(sh_angles))
        baseline.shoulder_angle_std      = max(float(np.std(sh_angles)), 0.5)
        baseline.spine_angle_mean        = float(np.mean(sp_angles))
        baseline.spine_angle_std         = max(float(np.std(sp_angles)), 0.5)
        baseline.head_shoulder_dist_mean = float(np.mean(hs_dists))
        baseline.head_shoulder_dist_std  = max(float(np.std(hs_dists)), 0.005)
        baseline.body_movement_mean      = float(np.mean(movements))
        baseline.body_movement_std       = max(float(np.std(movements)), 0.001)

        baseline.update_confidence(len(body_wins))
        logger.info(
            f"[{speaker_id}] Body baseline: "
            f"spine={baseline.spine_angle_mean:.1f}°±{baseline.spine_angle_std:.1f}°, "
            f"movement={baseline.body_movement_mean:.4f}±{baseline.body_movement_std:.4f}, "
            f"conf={baseline.calibration_confidence:.2f}"
        )
        return baseline

    def _build_gaze_baseline(
        self,
        speaker_id: str,
        session_id: str,
        windows: list[WindowFeatures],
    ) -> GazeBaseline:
        baseline = GazeBaseline(speaker_id=speaker_id, session_id=session_id)

        gaze_wins = [w for w in windows if w.face_detection_rate > 0.3]
        if not gaze_wins:
            logger.warning(f"[{speaker_id}] No usable windows for gaze baseline.")
            baseline.update_confidence(0)
            return baseline

        # Camera-screen offset: mean gaze position during calibration is the
        # learned "looking at screen" anchor.  Rule engines correct raw gaze
        # values by subtracting this offset.
        gx_vals = np.array([w.gaze_x_mean for w in gaze_wins], dtype=np.float32)
        gy_vals = np.array([w.gaze_y_mean for w in gaze_wins], dtype=np.float32)
        baseline.gaze_x_offset = float(np.mean(gx_vals))
        baseline.gaze_y_offset = float(np.mean(gy_vals))

        # Natural screen-engagement: mean gaze_on_screen_pct during calibration
        eng_vals = [w.gaze_on_screen_pct for w in gaze_wins]
        if eng_vals:
            baseline.screen_engagement_rate = float(np.mean(eng_vals))

        # Baseline blink rate (mirrors facial baseline)
        blinks = [w.blink_rate_bpm for w in gaze_wins if 0 < w.blink_rate_bpm <= 60]
        if blinks:
            baseline.blink_rate_bpm = float(np.mean(blinks))

        # Gaze stability
        gx_stds = [w.gaze_x_std for w in gaze_wins if w.gaze_x_std > 0]
        gy_stds = [w.gaze_y_std for w in gaze_wins if w.gaze_y_std > 0]
        if gx_stds:
            baseline.gaze_x_std_mean = float(np.mean(gx_stds))
        if gy_stds:
            baseline.gaze_y_std_mean = float(np.mean(gy_stds))

        baseline.update_confidence(len(gaze_wins))
        logger.info(
            f"[{speaker_id}] Gaze baseline: "
            f"offset=({baseline.gaze_x_offset:.3f}, {baseline.gaze_y_offset:.3f}), "
            f"screen_eng={baseline.screen_engagement_rate:.0%}, "
            f"blink={baseline.blink_rate_bpm:.1f}bpm, "
            f"conf={baseline.calibration_confidence:.2f}"
        )
        return baseline

    # ── Static computation helpers (used by rule engines) ──────────────────────

    @staticmethod
    def compute_delta(current: float, baseline: float) -> float:
        """
        Fractional deviation from baseline.
        E.g. 0.15 = 15% above baseline; -0.10 = 10% below.
        Returns 0.0 when baseline is zero.
        """
        if baseline == 0.0:
            return 0.0
        return (current - baseline) / abs(baseline)

    @staticmethod
    def compute_sigma(current: float, mean: float, std: float) -> float:
        """
        How many standard deviations from baseline mean.
        Returns 0.0 when std is zero.
        """
        if std == 0.0:
            return 0.0
        return (current - mean) / std

    @staticmethod
    def normalise_to_01(delta: float, max_delta: float = 1.0) -> float:
        """
        Normalise an absolute delta to [0, 1].
        0 = no change; 1 = max_delta or more deviation.
        """
        return min(abs(delta) / max(max_delta, 1e-6), 1.0)

    @staticmethod
    def blendshape_delta(
        current_bs: dict[str, float],
        baseline: FacialBaseline,
        key: str,
    ) -> float:
        """
        Sigma deviation for a single named blendshape.
        Returns 0.0 when key is not in baseline.
        """
        bsl_mean = baseline.blendshapes_neutral.get(key, 0.0)
        bsl_std  = baseline.blendshapes_std.get(key, 0.1)
        current  = current_bs.get(key, 0.0)
        return VideoCalibrationModule.compute_sigma(current, bsl_mean, max(bsl_std, 0.01))
