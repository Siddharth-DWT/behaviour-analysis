# services/video_agent/feature_extractor.py
"""
NEXUS Video Agent — Feature Extractor
Processes video files frame-by-frame using the MediaPipe Tasks API.

Classes:
  FrameFeatures         — typed dataclass for raw per-frame data
  WindowFeatures        — typed dataclass for 2-second aggregated windows
  BlinkDetector         — EAR-based blink state machine (per window)
  MediaPipeModelManager — downloads and caches .task model files on first use
  WindowAggregator      — collapses FrameFeatures into WindowFeatures
  VideoFeatureExtractor — orchestrates MediaPipe + frame processing
  SpeakerFaceMapper     — maps unassigned windows to speakers via diarization

Pipeline:
  VideoFeatureExtractor.extract_all(video_path)
      → (list[WindowFeatures], lip_activity_map)   (face-detected, unassigned + lip timeseries)
  SpeakerFaceMapper.assign(windows, diar_segments, lip_activity_map)
      → (dict[str, list[WindowFeatures]], dict[str, float])   (windows_by_speaker, lip_sync_scores)

Research basis:
  - Soukupova & Cech 2016: Eye Aspect Ratio for blink detection
  - Bentivoglio 1997: resting blink rate 15-26 bpm (mode ~15 silent, ~26 conversation)
  - Ekman & Friesen 1978: FACS Action Units (blendshapes approximate AU scores)
"""
import logging
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("nexus.video.features")

# ─── Processing constants ──────────────────────────────────────────────────
TARGET_FPS: int = 5
WINDOW_MS: int = 2000

# Blink detection (Soukupova & Cech 2016)
BLINK_EAR_THRESHOLD: float = 0.20
BLINK_CONSEC_FRAMES: int = 2

# Gaze: iris offset from eye centre below this threshold → on-screen
GAZE_ON_SCREEN_THRESHOLD: float = 0.25

# ─── EAR landmark indices (MediaPipe 478-point face landmarker) ──────────────
# EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
_RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)   # P1..P6
_LEFT_EYE_EAR  = (362, 385, 387, 263, 373, 380)

# Iris centres (requires refine_landmarks=True)
_LEFT_IRIS_IDX  = 468
_RIGHT_IRIS_IDX = 473

# Eye corners for iris-offset normalisation
_RIGHT_EYE_OUTER, _RIGHT_EYE_INNER = 33, 133
_LEFT_EYE_OUTER,  _LEFT_EYE_INNER  = 263, 362

# ─── Pose landmark indices ────────────────────────────────────────────────────
_POSE_NOSE           = 0
_POSE_LEFT_SHOULDER  = 11
_POSE_RIGHT_SHOULDER = 12
_POSE_LEFT_ELBOW     = 13
_POSE_RIGHT_ELBOW    = 14
_POSE_LEFT_HIP       = 23
_POSE_RIGHT_HIP      = 24

# ─── MediaPipe model download URLs ───────────────────────────────────────────
_MODEL_URLS: dict[str, str] = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    ),
    "blaze_face_full_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameFeatures:
    """
    Raw features extracted from a single processed frame.
    All coordinate values are normalised 0-1 unless noted.
    """
    timestamp_ms: int
    frame_idx: int
    face_index: int = 0            # which face in the frame (0 = largest/dominant)
    face_centre_x: float = 0.0    # normalised x centre of this face's bbox
    face_centre_y: float = 0.0    # normalised y centre (for position-based tracking)

    # ── Face ──────────────────────────────────────────────────────────────────
    face_detected: bool = False
    face_count: int = 0                   # number of faces detected in this frame
    face_box_area: float = 0.0            # bounding box area (normalised, proxy for proximity)

    blendshapes: dict[str, float] = field(default_factory=dict)   # 52 named scores

    head_pitch: float = 0.0   # degrees, + = looking down
    head_yaw:   float = 0.0   # degrees, + = looking right
    head_roll:  float = 0.0   # degrees, + = tilting right

    ear_left:  float = 0.0    # Eye Aspect Ratio per eye
    ear_right: float = 0.0
    ear_avg:   float = 0.0

    gaze_x: float = 0.0       # iris offset from eye centre, normalised by eye width
    gaze_y: float = 0.0       # positive = looking down

    face_luminance: float = 0.5  # mean face-crop brightness (0-1); lower = darker skin

    # ── Body ──────────────────────────────────────────────────────────────────
    body_detected: bool = False

    shoulder_angle: float = 0.0        # degrees from horizontal (+ = right shoulder up)
    spine_angle: float = 0.0           # degrees from vertical (+ = leaning forward)
    head_shoulder_dist: float = 0.0    # nose-to-shoulder-midpoint (normalised by frame height)
    body_movement: float = 0.0         # sum of landmark velocity vectors vs previous frame
    elbow_expansion: float = 0.0       # (elbow_width - shoulder_width) / shoulder_width; + = expansive, - = contracted

    # ── Hands ─────────────────────────────────────────────────────────────────
    hands_detected: int = 0            # 0, 1, or 2
    hand_near_face: bool = False       # bounding-box overlap with face region
    hand_velocity: float = 0.0        # mean wrist landmark velocity (gesture proxy)

    # Face-region touch classification
    hand_touch_zone: str = ""          # "chin","mouth","nose","cheek","ear","neck","forehead",""
    hand_touch_zone_r: str = ""        # right hand zone when both hands detected

    # Arm/hand posture indicators
    arms_crossed: bool = False         # both wrists near opposite shoulders/torso centre
    finger_steepling: bool = False     # both hands detected, fingertips touching
    open_palms: bool = False           # palms facing forward/camera (GestureRecognizer + fallback)
    head_supported_by_hand: bool = False  # palm supporting chin/cheek, head weight on hand
    hands_clasped: bool = False           # both hands interlaced/stacked (not steepling)

    # GestureRecognizer outputs
    thumb_up: bool = False
    thumb_down: bool = False
    pointing_up: bool = False
    victory_sign: bool = False
    closed_fist: bool = False

    # ── Per-frame behavioral state (computed from blendshapes + pose + gaze) ────
    behavioral_state: str = ""         # "speaking","agreeing","disagreeing","listening","distracted","tense","engaged"
    behavioral_state_detail: str = ""  # "nodding", "arms_crossed", "gaze_away", etc.
    stress_level: str = ""             # "low", "moderate", "high"
    engagement_level: str = ""         # "high", "neutral", "low"


@dataclass
class WindowFeatures:
    """
    Aggregated features over a 2-second window (≈20 frames at 10fps).
    One WindowFeatures instance covers one speaker's video in one 2s window.
    """
    window_start_ms: int
    window_end_ms: int
    speaker_id: str = ""
    frame_count: int = 0
    face_index: int = 0
    face_centre_x: float = 0.0
    face_centre_y: float = 0.0

    # ── Face presence ─────────────────────────────────────────────────────────
    face_detection_rate: float = 0.0    # fraction of frames where face was detected
    face_box_area_mean: float = 0.0     # mean normalised bounding-box area
    face_count: int = 0                 # max faces detected in any frame of this window

    # ── Blendshapes ───────────────────────────────────────────────────────────
    blendshapes_mean: dict[str, float] = field(default_factory=dict)
    blendshapes_std:  dict[str, float] = field(default_factory=dict)

    # ── Head pose ─────────────────────────────────────────────────────────────
    head_pitch_mean: float = 0.0
    head_pitch_std:  float = 0.0
    head_yaw_mean:   float = 0.0
    head_yaw_std:    float = 0.0
    head_roll_mean:  float = 0.0
    head_roll_std:   float = 0.0
    head_pose_variance: float = 0.0    # combined orientation variability

    # Raw sequences for velocity / zero-crossing analysis (BODY-HEAD-01)
    head_pitch_seq: list[float] = field(default_factory=list)
    head_yaw_seq:   list[float] = field(default_factory=list)

    # ── Eyes / blink ──────────────────────────────────────────────────────────
    ear_mean:       float = 0.0
    blink_count:    int   = 0
    blink_rate_bpm: float = 0.0

    # ── Gaze ──────────────────────────────────────────────────────────────────
    gaze_x_mean:        float = 0.0
    gaze_y_mean:        float = 0.0
    gaze_x_std:         float = 0.0
    gaze_y_std:         float = 0.0
    gaze_on_screen_pct: float = 0.0    # estimated % of frames looking at screen

    # ── Skin tone ─────────────────────────────────────────────────────────────
    face_luminance: float = 0.5  # mean face-crop brightness (0-1); bias mitigation (F-2)
    # Default True (safe fallback if mapper doesn't run).
    # SpeakerFaceMapper.assign() overrides this per window: True when the assigned
    # speaker's diarization overlap covers >50% of the window, False otherwise.
    is_speaking: bool = True

    # ── Body ──────────────────────────────────────────────────────────────────
    body_detection_rate:      float = 0.0
    shoulder_angle_mean:      float = 0.0
    shoulder_angle_std:       float = 0.0
    spine_angle_mean:         float = 0.0
    spine_angle_delta:        float = 0.0    # change from previous window
    head_shoulder_dist_mean:  float = 0.0
    body_movement_mean:       float = 0.0
    body_movement_std:        float = 0.0
    body_movement_max:        float = 0.0
    elbow_expansion_mean:     float = 0.0    # + = elbows wider than shoulders (expansive/dominant)

    # ── Hands ─────────────────────────────────────────────────────────────────
    hands_detected_rate:   float = 0.0
    hand_near_face_pct:    float = 0.0
    hand_velocity_mean:    float = 0.0
    gesture_velocity_mean: float = 0.0
    gesture_velocity_max:  float = 0.0

    # Touch zone distribution (fraction of frames touching each zone)
    touch_zone_chin_pct:     float = 0.0
    touch_zone_mouth_pct:    float = 0.0
    touch_zone_nose_pct:     float = 0.0
    touch_zone_cheek_pct:    float = 0.0
    touch_zone_ear_pct:      float = 0.0
    touch_zone_neck_pct:     float = 0.0
    touch_zone_forehead_pct: float = 0.0
    touch_zone_eye_pct:      float = 0.0   # either hand near eye
    touch_zone_eye_both_pct: float = 0.0   # both hands on eye simultaneously
    dominant_touch_zone:     str   = ""    # zone with highest pct (empty if no touch)

    # Arm posture
    arms_crossed_pct:      float = 0.0    # fraction of frames with crossed arms
    finger_steepling_pct:  float = 0.0    # fraction of frames with steepling
    open_palms_pct:        float = 0.0    # fraction of frames with open palms
    head_supported_pct:    float = 0.0    # fraction of frames with head resting on hand
    hands_clasped_pct:     float = 0.0    # fraction of frames with hands interlaced/stacked
    thumb_up_pct:          float = 0.0
    thumb_down_pct:        float = 0.0
    pointing_up_pct:       float = 0.0
    victory_sign_pct:      float = 0.0
    closed_fist_pct:       float = 0.0

    # ── Cross-speaker interaction aggregation ─────────────────────────────────
    dominant_interaction: str = ""   # e.g. "Speaker_2_disagrees"
    interaction_count:    int = 0    # total interaction frames in this window

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# BlinkDetector  — state machine, one instance per window
# ══════════════════════════════════════════════════════════════════════════════

class BlinkDetector:
    """
    EAR-based blink state machine (Soukupova & Cech 2016).
    EAR drops below threshold for ≥ BLINK_CONSEC_FRAMES → one blink event.
    Instantiate fresh per aggregation window so counts are scoped correctly.
    """

    def __init__(
        self,
        threshold: float = BLINK_EAR_THRESHOLD,
        consec_frames: int = BLINK_CONSEC_FRAMES,
    ) -> None:
        self._threshold = threshold
        self._consec = consec_frames
        self._counter: int = 0
        self._total:   int = 0

    def update(self, ear: float) -> bool:
        """Feed one frame's EAR value. Returns True when a blink closes."""
        if ear < self._threshold:
            self._counter += 1
            return False
        # Eye re-opened — decide if a blink just completed
        if self._counter >= self._consec:
            self._total += 1
            self._counter = 0
            return True
        self._counter = 0
        return False

    def flush(self) -> None:
        """Call at window end to capture any open blink at end-of-video."""
        if self._counter >= self._consec:
            self._total += 1
        self._counter = 0

    @property
    def total_blinks(self) -> int:
        return self._total


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipeModelManager  — downloads .task files on first use
# ══════════════════════════════════════════════════════════════════════════════

class MediaPipeModelManager:
    """
    Downloads and caches MediaPipe .task model files to model_dir.
    Files are ~4-30 MB each; downloaded once and reused across runs.
    """

    def __init__(self, model_dir: str = "models/mediapipe") -> None:
        self._dir = Path(model_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get_face_landmarker_path(self) -> str:
        return self._get("face_landmarker.task")

    def get_pose_landmarker_path(self) -> str:
        return self._get("pose_landmarker_heavy.task")

    def get_hand_landmarker_path(self) -> str:
        return self._get("hand_landmarker.task")

    def get_face_detector_path(self) -> str:
        return self._get("blaze_face_full_range.tflite")

    def get_gesture_recognizer_path(self) -> str:
        return self._get("gesture_recognizer.task")

    def _get(self, filename: str) -> str:
        dest = self._dir / filename
        if not dest.exists():
            url = _MODEL_URLS[filename]
            logger.info(f"Downloading MediaPipe model: {filename} ...")
            try:
                urllib.request.urlretrieve(url, str(dest))
                logger.info(f"Saved {filename} → {dest}")
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download {filename} from {url}. "
                    f"Download manually to {dest}. Error: {exc}"
                ) from exc
        return str(dest)


# ══════════════════════════════════════════════════════════════════════════════
# WindowAggregator  — FrameFeatures → WindowFeatures
# ══════════════════════════════════════════════════════════════════════════════

class WindowAggregator:
    """
    Collapses a sequence of FrameFeatures into fixed-size time windows.

    DSA: uses defaultdict(list) for O(n) single-pass bucketing
    keyed by (timestamp_ms // window_ms).  Each bucket is then
    reduced via numpy for vectorised mean/std on blendshapes.
    """

    def __init__(self, window_ms: int = WINDOW_MS, target_fps: int = TARGET_FPS) -> None:
        self._window_ms = window_ms
        self._target_fps = target_fps

    def aggregate(self, frames: list[FrameFeatures]) -> list[WindowFeatures]:
        """
        Collapse FrameFeatures into 2-second windows, grouped by face_index.
        Each face in a multi-person frame gets its own independent window stream.
        """
        if not frames:
            return []

        by_face: dict[int, list[FrameFeatures]] = defaultdict(list)
        for ff in frames:
            by_face[ff.face_index].append(ff)

        all_windows: list[WindowFeatures] = []
        for face_idx in sorted(by_face.keys()):
            face_frames = by_face[face_idx]
            windows = self._aggregate_single_face(face_frames)
            for wf in windows:
                wf.face_index = face_idx
                window_frames = [
                    f for f in face_frames
                    if wf.window_start_ms <= f.timestamp_ms < wf.window_end_ms
                ]
                cx = [f.face_centre_x for f in window_frames if f.face_centre_x > 0]
                cy = [f.face_centre_y for f in window_frames if f.face_centre_y > 0]
                if cx:
                    wf.face_centre_x = sum(cx) / len(cx)
                if cy:
                    wf.face_centre_y = sum(cy) / len(cy)
            all_windows.extend(windows)

        return all_windows

    def _aggregate_single_face(self, frames: list[FrameFeatures]) -> list[WindowFeatures]:
        """Time-window aggregation for a single face's frame stream."""
        if not frames:
            return []

        # ── Bucket frames by time window ──────────────────────────────────────
        buckets: dict[int, list[FrameFeatures]] = defaultdict(list)
        for f in frames:
            bucket_key = f.timestamp_ms // self._window_ms
            buckets[bucket_key].append(f)

        # ── Reduce each bucket to a WindowFeatures ────────────────────────────
        windows: list[WindowFeatures] = []
        prev_spine_angle: float = 0.0

        for key in sorted(buckets):
            bucket = buckets[key]
            wf = self._reduce_bucket(
                bucket,
                window_start_ms=key * self._window_ms,
                window_end_ms=(key + 1) * self._window_ms,
                prev_spine_angle=prev_spine_angle,
            )
            prev_spine_angle = wf.spine_angle_mean
            windows.append(wf)

        return windows

    def _reduce_bucket(
        self,
        frames: list[FrameFeatures],
        window_start_ms: int,
        window_end_ms: int,
        prev_spine_angle: float,
    ) -> WindowFeatures:
        wf = WindowFeatures(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            frame_count=len(frames),
        )
        if not frames:
            return wf

        # ── Face presence ──────────────────────────────────────────────────────
        face_frames = [f for f in frames if f.face_detected]
        wf.face_detection_rate = len(face_frames) / len(frames)
        if face_frames:
            wf.face_box_area_mean = float(np.mean([f.face_box_area for f in face_frames]))
            wf.face_count = max(f.face_count for f in face_frames)

        # ── Blendshapes ────────────────────────────────────────────────────────
        if face_frames:
            all_keys: set[str] = set().union(*(f.blendshapes.keys() for f in face_frames))
            for k in all_keys:
                vals = [f.blendshapes.get(k, 0.0) for f in face_frames]
                arr = np.array(vals, dtype=np.float32)
                wf.blendshapes_mean[k] = float(np.mean(arr))
                wf.blendshapes_std[k]  = float(np.std(arr))

        # ── Head pose ──────────────────────────────────────────────────────────
        if face_frames:
            pitches = np.array([f.head_pitch for f in face_frames], dtype=np.float32)
            yaws    = np.array([f.head_yaw   for f in face_frames], dtype=np.float32)
            rolls   = np.array([f.head_roll  for f in face_frames], dtype=np.float32)

            wf.head_pitch_mean = float(np.mean(pitches))
            wf.head_pitch_std  = float(np.std(pitches))
            wf.head_yaw_mean   = float(np.mean(yaws))
            wf.head_yaw_std    = float(np.std(yaws))
            wf.head_roll_mean  = float(np.mean(rolls))
            wf.head_roll_std   = float(np.std(rolls))
            wf.head_pose_variance = float(np.var(pitches) + np.var(yaws) + np.var(rolls))

            wf.head_pitch_seq = pitches.tolist()
            wf.head_yaw_seq   = yaws.tolist()

        # ── Blink detection (state machine per window) ─────────────────────────
        if face_frames:
            detector = BlinkDetector()
            ear_vals: list[float] = []
            for f in face_frames:
                detector.update(f.ear_avg)
                ear_vals.append(f.ear_avg)
            detector.flush()

            wf.ear_mean    = float(np.mean(ear_vals))
            wf.blink_count = detector.total_blinks
            duration_min   = (window_end_ms - window_start_ms) / 60_000
            wf.blink_rate_bpm = wf.blink_count / max(duration_min, 1e-6)

        # ── Skin tone (luminance) ──────────────────────────────────────────────
        if face_frames:
            wf.face_luminance = float(np.mean([f.face_luminance for f in face_frames]))

        # ── Gaze ───────────────────────────────────────────────────────────────
        if face_frames:
            gx = np.array([f.gaze_x for f in face_frames], dtype=np.float32)
            gy = np.array([f.gaze_y for f in face_frames], dtype=np.float32)
            wf.gaze_x_mean = float(np.mean(gx))
            wf.gaze_y_mean = float(np.mean(gy))
            wf.gaze_x_std  = float(np.std(gx))
            wf.gaze_y_std  = float(np.std(gy))

            on_screen = sum(
                1 for x, y in zip(gx, gy)
                if abs(x) < GAZE_ON_SCREEN_THRESHOLD and abs(y) < GAZE_ON_SCREEN_THRESHOLD
            )
            wf.gaze_on_screen_pct = on_screen / len(face_frames)

        # ── Body ───────────────────────────────────────────────────────────────
        body_frames = [f for f in frames if f.body_detected]
        wf.body_detection_rate = len(body_frames) / len(frames)
        if body_frames:
            sh_angles = np.array([f.shoulder_angle for f in body_frames], dtype=np.float32)
            sp_angles = np.array([f.spine_angle    for f in body_frames], dtype=np.float32)
            hs_dists  = np.array([f.head_shoulder_dist for f in body_frames], dtype=np.float32)
            movements = np.array([f.body_movement  for f in body_frames], dtype=np.float32)

            wf.shoulder_angle_mean     = float(np.mean(sh_angles))
            wf.shoulder_angle_std      = float(np.std(sh_angles))
            wf.spine_angle_mean        = float(np.mean(sp_angles))
            wf.spine_angle_delta       = wf.spine_angle_mean - prev_spine_angle
            wf.head_shoulder_dist_mean = float(np.mean(hs_dists))
            wf.body_movement_mean      = float(np.mean(movements))
            wf.body_movement_std       = float(np.std(movements))
            wf.body_movement_max       = float(np.max(movements))
            expansions = np.array([f.elbow_expansion for f in body_frames], dtype=np.float32)
            wf.elbow_expansion_mean    = float(np.mean(expansions))

        # ── Hands ──────────────────────────────────────────────────────────────
        wf.hands_detected_rate   = float(np.mean([f.hands_detected > 0 for f in frames]))
        wf.hand_near_face_pct    = float(np.mean([f.hand_near_face    for f in frames]))
        hand_vels = [f.hand_velocity for f in frames if f.hand_velocity > 0]
        if hand_vels:
            wf.hand_velocity_mean    = float(np.mean(hand_vels))
            wf.gesture_velocity_mean = float(np.mean(hand_vels))
            wf.gesture_velocity_max  = float(np.max(hand_vels))

        # ── Touch zone percentages ─────────────────────────────────────────────
        total = len(frames)
        if total > 0:
            zone_counts: dict[str, int] = defaultdict(int)
            for f in frames:
                if f.hand_touch_zone:
                    zone_counts[f.hand_touch_zone] += 1
                if f.hand_touch_zone_r:
                    zone_counts[f.hand_touch_zone_r] += 1

            wf.touch_zone_chin_pct     = zone_counts.get("chin",     0) / total
            wf.touch_zone_mouth_pct    = zone_counts.get("mouth",    0) / total
            wf.touch_zone_nose_pct     = zone_counts.get("nose",     0) / total
            wf.touch_zone_cheek_pct    = zone_counts.get("cheek",    0) / total
            wf.touch_zone_ear_pct      = zone_counts.get("ear",      0) / total
            wf.touch_zone_neck_pct     = zone_counts.get("neck",     0) / total
            wf.touch_zone_forehead_pct = zone_counts.get("forehead", 0) / total
            wf.touch_zone_eye_pct      = zone_counts.get("eye",      0) / total
            wf.touch_zone_eye_both_pct = sum(
                1 for f in frames
                if f.hand_touch_zone == "eye" and f.hand_touch_zone_r == "eye"
            ) / total

            if zone_counts:
                wf.dominant_touch_zone = max(zone_counts, key=lambda k: zone_counts[k])

            wf.arms_crossed_pct       = sum(1 for f in frames if f.arms_crossed)           / total
            wf.finger_steepling_pct   = sum(1 for f in frames if f.finger_steepling)       / total
            wf.open_palms_pct         = sum(1 for f in frames if f.open_palms)             / total
            wf.head_supported_pct     = sum(1 for f in frames if f.head_supported_by_hand) / total
            wf.hands_clasped_pct      = sum(1 for f in frames if f.hands_clasped)          / total
            wf.thumb_up_pct           = sum(1 for f in frames if f.thumb_up)               / total
            wf.thumb_down_pct         = sum(1 for f in frames if f.thumb_down)             / total
            wf.pointing_up_pct        = sum(1 for f in frames if f.pointing_up)            / total
            wf.victory_sign_pct       = sum(1 for f in frames if f.victory_sign)           / total
            wf.closed_fist_pct        = sum(1 for f in frames if f.closed_fist)            / total

        # ── Cross-speaker interaction aggregation ─────────────────────────────
        window_interaction_counts: dict[str, int] = defaultdict(int)
        for f in frames:
            for interaction in getattr(f, "_interactions", []):
                key = f"{interaction['reactor']}_{interaction['interaction']}"
                window_interaction_counts[key] += 1
        if window_interaction_counts:
            wf.dominant_interaction = max(window_interaction_counts, key=lambda k: window_interaction_counts[k])
            wf.interaction_count    = sum(window_interaction_counts.values())

        return wf


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# Browser-compatible video encoding helper
# ══════════════════════════════════════════════════════════════════════════════

def _reencode_for_browser(src: str, dst: str) -> None:
    """
    Convert src (mp4v/raw) to a browser-playable H.264 MP4 at dst.

    Strategy (in order):
      1. ffmpeg with libx264 + -movflags +faststart  (best — H.264, seekable)
      2. avc1 via OpenCV VideoWriter                 (works if libx264 linked)
      3. Atomic rename src → dst                     (mp4v fallback, may not play in Chrome)

    The src file is always removed after success; dst is only written when
    fully complete so the gateway HEAD-check never races a partial write.
    """
    import os
    import subprocess

    # ── Strategy 1: ffmpeg ────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", src,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                "-an",
                dst,
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            try:
                os.remove(src)
            except OSError:
                pass
            logger.info(f"[overlay] ffmpeg H.264 → {Path(dst).name}")
            return
        logger.warning(f"[overlay] ffmpeg failed (rc={result.returncode}): {result.stderr[-300:]}")
    except FileNotFoundError:
        logger.info("[overlay] ffmpeg not found, trying OpenCV avc1")
    except subprocess.TimeoutExpired:
        logger.warning("[overlay] ffmpeg timed out")

    # ── Strategy 2: OpenCV with avc1 (H.264) ──────────────────────────────────
    try:
        import cv2
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(dst, fourcc, fps, (fw, fh))
            if writer.isOpened():
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                cap.release()
                writer.release()
                try:
                    os.remove(src)
                except OSError:
                    pass
                logger.info(f"[overlay] OpenCV avc1 → {Path(dst).name}")
                return
            writer.release()
        cap.release()
        logger.warning("[overlay] OpenCV avc1 writer failed to open, using mp4v fallback")
    except Exception as exc:
        logger.warning(f"[overlay] OpenCV avc1 attempt failed: {exc}")

    # ── Strategy 3: rename as-is (mp4v — may not play in Chrome/Firefox) ──────
    try:
        os.rename(src, dst)
        logger.warning(f"[overlay] mp4v fallback (Chrome may not play): {Path(dst).name}")
    except Exception as exc:
        logger.error(f"[overlay] rename failed: {exc}")


def _nms_boxes(boxes: list[tuple], iou_threshold: float = 0.4) -> list[tuple]:
    """
    Non-maximum suppression over (x, y, w, h) face bounding boxes.
    Keeps the largest box when two boxes overlap above iou_threshold.
    Used to deduplicate faces detected in both the full frame and quadrant passes.
    """
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple] = []
    for box in sorted_boxes:
        x1, y1, w1, h1 = box
        dominated = False
        for kx, ky, kw, kh in kept:
            ix = max(0, min(x1 + w1, kx + kw) - max(x1, kx))
            iy = max(0, min(y1 + h1, ky + kh) - max(y1, ky))
            inter = ix * iy
            union = w1 * h1 + kw * kh - inter
            if union > 0 and inter / union > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(box)
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# Tiled detection — result stubs + TiledFrameProcessor
# ══════════════════════════════════════════════════════════════════════════════

class _Pt:
    """Lightweight landmark compatible with MediaPipe NormalizedLandmark (.x .y .z .visibility)."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0) -> None:
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _FaceResult:
    """Drop-in face result for OverlayRenderer and _process_frame_from_results."""
    __slots__ = ("face_landmarks", "face_blendshapes", "facial_transformation_matrixes")

    def __init__(self, face_landmarks, face_blendshapes=None, facial_transformation_matrixes=None):
        self.face_landmarks = face_landmarks or []
        self.face_blendshapes = face_blendshapes or []
        self.facial_transformation_matrixes = facial_transformation_matrixes or []


class _PoseResult:
    __slots__ = ("pose_landmarks",)

    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks or []


# Maps GestureRecognizer category names to FrameFeatures bool fields
_GESTURE_FIELD_MAP: dict[str, str] = {
    "Open_Palm":   "open_palms",
    "Closed_Fist": "closed_fist",
    "Thumb_Up":    "thumb_up",
    "Thumb_Down":  "thumb_down",
    "Pointing_Up": "pointing_up",
    "Victory":     "victory_sign",
}


class _HandResult:
    __slots__ = ("hand_landmarks", "hand_gestures")

    def __init__(self, hand_landmarks, hand_gestures=None):
        self.hand_landmarks = hand_landmarks or []
        self.hand_gestures  = hand_gestures  or [""] * len(self.hand_landmarks)


class TiledFrameProcessor:
    """
    Grid-agnostic multi-person landmark detection via two-pass MediaPipe.

    Works for any video layout (Zoom 2x2, 3x3, spotlight, side-by-side, etc.)
    without knowing the grid dimensions in advance.

    Pass 1 — FaceDetector (VIDEO mode) on the full frame.
              Lightweight blazeface model returns stable bounding boxes for ALL
              visible faces regardless of frame layout or number of participants.

    Pass 2 — FaceLandmarker + PoseLandmarker (IMAGE mode) run independently on
              each face/body crop zoomed to near-portrait resolution. IMAGE mode
              is stateless (no temporal smoothing) but produces far better results
              than full-frame VIDEO mode that misses small faces in dense grids.

    Hands  — HandLandmarker (VIDEO mode) on the full frame because hands
              frequently cross tile boundaries, making per-crop detection unreliable.

    Coordinate remapping:
      A landmark from crop (cx1, cy1)→(cx2, cy2) with normalised coord (lm.x, lm.y)
      maps to full-frame normalised coord:
        full_x = (cx1 + lm.x * crop_w) / frame_w
        full_y = (cy1 + lm.y * crop_h) / frame_h
      Output _Pt objects expose .x .y so OverlayRenderer's
        `pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lm]`
      works without any changes.
    """

    _FACE_PAD   = 0.30   # expand detected bbox by this fraction on all sides
    _BODY_SCALE = 2.5    # extend bbox this many face-heights downward for pose (was 3.0)

    def __init__(self, face_det, face_det_img, face_lm, pose_lm, hand_lm, gesture_rec_img, num_faces: int) -> None:
        self._face_det        = face_det         # FaceDetector      — VIDEO mode, full frame
        self._face_det_img    = face_det_img     # FaceDetector      — IMAGE mode, quadrant passes
        self._face_lm         = face_lm          # FaceLandmarker    — IMAGE mode, per crop
        self._pose_lm         = pose_lm          # PoseLandmarker    — IMAGE mode, per crop
        self._hand_lm         = hand_lm          # HandLandmarker    — VIDEO mode, full frame (fallback)
        self._gesture_rec_img = gesture_rec_img  # GestureRecognizer — IMAGE mode, per wrist crop
        self._num_faces       = num_faces

    @classmethod
    def create(
        cls, mp, model_mgr: "MediaPipeModelManager", num_faces: int
    ) -> "TiledFrameProcessor":
        """Build all six MediaPipe instances. Called once per video."""
        BaseOptions = mp.tasks.BaseOptions
        Mode = mp.tasks.vision.RunningMode
        det_path = model_mgr.get_face_detector_path()

        face_det = mp.tasks.vision.FaceDetector.create_from_options(
            mp.tasks.vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=det_path),
                running_mode=Mode.VIDEO,
                min_detection_confidence=0.15,
            )
        )
        face_det_img = mp.tasks.vision.FaceDetector.create_from_options(
            mp.tasks.vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=det_path),
                running_mode=Mode.IMAGE,
                min_detection_confidence=0.15,
            )
        )
        face_lm = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_mgr.get_face_landmarker_path()),
                running_mode=Mode.IMAGE,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.1,
                min_face_presence_confidence=0.1,
                min_tracking_confidence=0.1,
            )
        )
        pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_mgr.get_pose_landmarker_path()),
                running_mode=Mode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.2,
                min_pose_presence_confidence=0.2,
                min_tracking_confidence=0.2,
            )
        )
        hand_lm = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_mgr.get_hand_landmarker_path()),
                running_mode=Mode.VIDEO,
                num_hands=num_faces * 2,
                min_hand_detection_confidence=0.2,
                min_hand_presence_confidence=0.2,
                min_tracking_confidence=0.3,
            )
        )
        # IMAGE mode GestureRecognizer — replaces per-crop HandLandmarker;
        # returns hand landmarks + gesture category in one call.
        gesture_rec_img = mp.tasks.vision.GestureRecognizer.create_from_options(
            mp.tasks.vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=model_mgr.get_gesture_recognizer_path()),
                running_mode=Mode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.15,
                min_hand_presence_confidence=0.15,
                min_tracking_confidence=0.15,
            )
        )
        return cls(face_det, face_det_img, face_lm, pose_lm, hand_lm, gesture_rec_img, num_faces)

    def _detect_all_faces(self, rgb: np.ndarray, ts_ms: int, mp) -> list[tuple]:
        """
        Two-pass face detection that handles any grid layout.

        Pass 1 — full frame via VIDEO-mode FaceDetector (temporally smooth).
        Pass 2 — 2×2 quadrants via IMAGE-mode FaceDetector with 12.5% overlap.
                  Zooms in 2× per quadrant so small faces that the full-frame
                  pass misses get detected at near-portrait resolution.

        Returns a deduplicated list of (x, y, w, h) pixel boxes sorted
        largest-first via _nms_boxes (IoU threshold 0.35).
        """
        fh, fw = rgb.shape[:2]
        raw_boxes: list[tuple] = []

        # Pass 1: full frame
        try:
            mp_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._face_det.detect_for_video(mp_full, ts_ms)
            for det in (res.detections if res else []):
                bb = det.bounding_box
                raw_boxes.append((int(bb.origin_x), int(bb.origin_y),
                                   int(bb.width), int(bb.height)))
        except Exception as exc:
            logger.debug(f"[tiled] full-frame detect error: {exc}")

        # Pass 2: 2×2 quadrants with 12.5% overlap to catch faces near edges
        half_w, half_h = fw // 2, fh // 2
        overlap_x, overlap_y = fw // 8, fh // 8
        for qy in range(2):
            for qx in range(2):
                cx1 = qx * half_w
                cy1 = qy * half_h
                cx2 = min(fw, cx1 + half_w + overlap_x)
                cy2 = min(fh, cy1 + half_h + overlap_y)
                quad = rgb[cy1:cy2, cx1:cx2]
                if quad.size < 64:
                    continue
                try:
                    mp_quad = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(quad),
                    )
                    res = self._face_det_img.detect(mp_quad)
                    for det in (res.detections if res else []):
                        bb = det.bounding_box
                        raw_boxes.append((
                            cx1 + int(bb.origin_x),
                            cy1 + int(bb.origin_y),
                            int(bb.width),
                            int(bb.height),
                        ))
                except Exception as exc:
                    logger.debug(f"[tiled] quadrant detect error: {exc}")

        return _nms_boxes(raw_boxes, iou_threshold=0.35)[:self._num_faces]

    def detect(
        self, rgb: np.ndarray, ts_ms: int
    ) -> tuple["_FaceResult", "_PoseResult", "_HandResult"]:
        """
        Detect all faces and poses in a full-frame RGB image.
        Returns _FaceResult/_PoseResult/_HandResult with all landmark coordinates
        remapped to full-frame normalised space (drop-in for direct MediaPipe calls).
        Faces are ordered largest-first so face_landmarks[0] = dominant speaker.
        """
        import mediapipe as mp
        fh, fw = rgb.shape[:2]

        # ── Face bounding boxes: full-frame + quadrant passes ─────────────────
        face_boxes = self._detect_all_faces(rgb, ts_ms, mp)

        all_hand_lm:      list = []
        all_hand_gestures: list = []

        if not face_boxes:
            # No faces — still run full-frame hand detection for completeness
            try:
                mp_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hand_raw = self._hand_lm.detect_for_video(mp_full, ts_ms)
                if hand_raw and hand_raw.hand_landmarks:
                    all_hand_lm.extend(hand_raw.hand_landmarks)
                    all_hand_gestures.extend("" for _ in hand_raw.hand_landmarks)
            except Exception:
                pass
            return _FaceResult([]), _PoseResult([]), _HandResult(all_hand_lm, all_hand_gestures)

        all_face_lm:  list = []
        all_face_bs:  list = []
        all_face_mat: list = []
        all_pose_lm:  list = []
        wrist_crops:  list[tuple[int, int, int, int]] = []  # (x1,y1,x2,y2) full-frame px

        for fx, fy, fbw, fbh in face_boxes:
            px = int(fbw * self._FACE_PAD)
            py = int(fbh * self._FACE_PAD)

            # ── Face crop (bbox + padding on all sides) ───────────────────────
            fc_x1 = max(0, fx - px)
            fc_y1 = max(0, fy - py)
            fc_x2 = min(fw, fx + fbw + px)
            fc_y2 = min(fh, fy + fbh + py)
            face_crop = rgb[fc_y1:fc_y2, fc_x1:fc_x2]
            if face_crop.size < 64:
                continue

            try:
                fc_h, fc_w = face_crop.shape[:2]
                fc_mp  = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(face_crop),
                )
                fc_res = self._face_lm.detect(fc_mp)
                if fc_res.face_landmarks:
                    remapped_face = [
                        _Pt(
                            x=(fc_x1 + lm.x * fc_w) / fw,
                            y=(fc_y1 + lm.y * fc_h) / fh,
                            z=lm.z,
                        )
                        for lm in fc_res.face_landmarks[0]
                    ]
                    all_face_lm.append(remapped_face)
                    all_face_bs.append(
                        fc_res.face_blendshapes[0] if fc_res.face_blendshapes else None
                    )
                    all_face_mat.append(
                        fc_res.facial_transformation_matrixes[0]
                        if fc_res.facial_transformation_matrixes else None
                    )
            except Exception as exc:
                logger.debug(f"[tiled] face crop error: {exc}")

            # ── Body crop (face bbox extended downward, clamped to tile) ──────
            bc_x1 = max(0, fx - px)
            bc_y1 = max(0, fy - py)
            bc_x2 = min(fw, fx + fbw + px)

            raw_bc_y2 = fy + int(fbh * self._BODY_SCALE)

            # Clamp to midpoint between this face's bottom and the nearest face
            # directly below in the same column — prevents body crop from bleeding
            # into adjacent tiles in multi-person grid layouts.
            face_bottom = fy + fbh
            nearest_below_top = fh  # default: frame bottom

            for ofx, ofy, ofbw, _ in face_boxes:
                if ofy > face_bottom and abs((ofx + ofbw / 2) - (fx + fbw / 2)) < fw * 0.4:
                    # Stop at the face TOP of the tile below — not the midpoint.
                    # The midpoint was too tight (only ~40 px of body for the middle row),
                    # preventing MediaPipe Pose from detecting shoulders. Using the face top
                    # gives the full inter-tile gap as body crop space while still
                    # stopping before the next person's face begins.
                    nearest_below_top = min(nearest_below_top, ofy)

            bc_y2 = min(fh, raw_bc_y2, nearest_below_top)
            # Ensure we capture at least chin + shoulders (1.5× face height from face top).
            bc_y2 = max(bc_y2, fy + int(fbh * 1.5))

            body_crop = rgb[bc_y1:bc_y2, bc_x1:bc_x2]
            if body_crop.size < 64:
                continue

            try:
                bc_h, bc_w = body_crop.shape[:2]
                bc_mp  = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(body_crop),
                )
                bc_res = self._pose_lm.detect(bc_mp)
                if bc_res.pose_landmarks:
                    remapped_pose = [
                        _Pt(
                            x=(bc_x1 + lm.x * bc_w) / fw,
                            y=(bc_y1 + lm.y * bc_h) / fh,
                            z=lm.z,
                            visibility=getattr(lm, "visibility", 1.0),
                        )
                        for lm in bc_res.pose_landmarks[0]
                    ]
                    all_pose_lm.append(remapped_pose)

                    # ── Collect wrist positions for pose-guided hand crops ─────
                    _WRIST_L, _WRIST_R = 15, 16
                    _INDEX_MCP_L, _PINKY_MCP_L = 17, 19
                    _INDEX_MCP_R, _PINKY_MCP_R = 18, 20
                    plm_raw = bc_res.pose_landmarks[0]
                    for wrist_idx in (_WRIST_L, _WRIST_R):
                        if wrist_idx >= len(plm_raw):
                            continue
                        if getattr(plm_raw[wrist_idx], "visibility", 0) < 0.3:
                            continue
                        wx_px = int(bc_x1 + plm_raw[wrist_idx].x * bc_w)
                        wy_px = int(bc_y1 + plm_raw[wrist_idx].y * bc_h)
                        hand_size = max(int(fbw * 1.5), 80)
                        mcp_pair = (_INDEX_MCP_L, _PINKY_MCP_L) if wrist_idx == _WRIST_L else (_INDEX_MCP_R, _PINKY_MCP_R)
                        for mcp_idx in mcp_pair:
                            if mcp_idx < len(plm_raw) and getattr(plm_raw[mcp_idx], "visibility", 0) > 0.3:
                                mx = int(bc_x1 + plm_raw[mcp_idx].x * bc_w)
                                my = int(bc_y1 + plm_raw[mcp_idx].y * bc_h)
                                wrist_to_mcp = int(((wx_px - mx) ** 2 + (wy_px - my) ** 2) ** 0.5)
                                if wrist_to_mcp > 10:
                                    hand_size = max(int(wrist_to_mcp * 2.7), hand_size)
                                break
                        hx1_w = max(0, wx_px - hand_size)
                        hy1_w = max(0, wy_px - hand_size)
                        hx2_w = min(fw, wx_px + hand_size)
                        hy2_w = min(fh, wy_px + hand_size)
                        if (hx2_w - hx1_w) > 40 and (hy2_w - hy1_w) > 40:
                            wrist_crops.append((hx1_w, hy1_w, hx2_w, hy2_w))
            except Exception as exc:
                logger.debug(f"[tiled] pose crop error: {exc}")

        # ── Pose-guided hand detection (full-res crops around each wrist) ────
        # Crops are tight ROIs from the original frame — hand occupies ~200×200px
        # instead of ~60×60px in body crops, matching HandLandmarker's 256×256 sweet spot.
        for hx1_c, hy1_c, hx2_c, hy2_c in wrist_crops:
            hand_crop = rgb[hy1_c:hy2_c, hx1_c:hx2_c]
            if hand_crop.size < 64:
                continue
            try:
                hc_h, hc_w = hand_crop.shape[:2]
                hc_mp = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(hand_crop),
                )
                gr_raw = self._gesture_rec_img.recognize(hc_mp)
                if gr_raw and gr_raw.hand_landmarks:
                    for idx, hlm in enumerate(gr_raw.hand_landmarks):
                        remapped_hand = [
                            _Pt(
                                x=(hx1_c + lm.x * hc_w) / fw,
                                y=(hy1_c + lm.y * hc_h) / fh,
                                z=lm.z,
                            )
                            for lm in hlm
                        ]
                        gesture_str = ""
                        if gr_raw.gestures and idx < len(gr_raw.gestures) and gr_raw.gestures[idx]:
                            cat = gr_raw.gestures[idx][0].category_name
                            if cat != "None":
                                gesture_str = cat
                        dup = any(
                            ((remapped_hand[0].x - ex[0].x) ** 2 +
                             (remapped_hand[0].y - ex[0].y) ** 2) ** 0.5 < 0.05
                            for ex in all_hand_lm
                        )
                        if not dup:
                            all_hand_lm.append(remapped_hand)
                            all_hand_gestures.append(gesture_str)
            except Exception as exc:
                logger.debug(f"[tiled] pose-guided hand crop error: {exc}")

        # ── Full-frame fallback (catches hands where pose wrist is not visible) ──
        try:
            mp_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_raw_full = self._hand_lm.detect_for_video(mp_full, ts_ms)
            if hand_raw_full and hand_raw_full.hand_landmarks:
                for hlm in hand_raw_full.hand_landmarks:
                    dup = any(
                        ((hlm[0].x - ex[0].x) ** 2 + (hlm[0].y - ex[0].y) ** 2) ** 0.5 < 0.05
                        for ex in all_hand_lm
                    )
                    if not dup:
                        all_hand_lm.append(hlm)
                        all_hand_gestures.append("")
        except Exception:
            pass

        return (
            _FaceResult(
                face_landmarks=all_face_lm,
                face_blendshapes=all_face_bs,       # keep None entries so face_landmarks[i] ↔ face_blendshapes[i]
                facial_transformation_matrixes=all_face_mat,  # same reason
            ),
            _PoseResult(pose_landmarks=all_pose_lm),
            _HandResult(all_hand_lm, all_hand_gestures),
        )

    def close(self) -> None:
        for obj in (self._face_det, self._face_det_img, self._face_lm,
                    self._pose_lm, self._hand_lm, self._gesture_rec_img):
            try:
                obj.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Cross-speaker incongruence helpers (module level — used by InteractionDetector
# and independently by body_rules)
# ══════════════════════════════════════════════════════════════════════════════

def _has_incongruence(ff: FrameFeatures) -> bool:
    """Return True when a person's face and body send conflicting signals."""
    bs = ff.blendshapes or {}
    smile_score  = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
    brow_tension = (bs.get("browDownLeft",   0.0) + bs.get("browDownRight",   0.0)) / 2.0
    if smile_score > 0.20 and brow_tension > 0.20:
        return True
    if ff.behavioral_state == "agreeing" and getattr(ff, "arms_crossed", False):
        return True
    return False


def _describe_incongruence(ff: FrameFeatures) -> list[str]:
    """Return a list of human-readable incongruence descriptors."""
    evidence: list[str] = []
    bs = ff.blendshapes or {}
    smile_score  = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
    brow_tension = (bs.get("browDownLeft",   0.0) + bs.get("browDownRight",   0.0)) / 2.0
    if smile_score > 0.20 and brow_tension > 0.20:
        evidence.append("smiling_but_tense")
    if ff.behavioral_state == "agreeing" and getattr(ff, "arms_crossed", False):
        evidence.append("nodding_but_closed")
    return evidence if evidence else ["mixed_signals"]


# ══════════════════════════════════════════════════════════════════════════════
# InteractionDetector — cross-speaker reaction detection
# ══════════════════════════════════════════════════════════════════════════════

class InteractionDetector:
    """
    Detect cross-speaker interactions from per-frame behavioral states.

    Runs after all faces in a frame have their behavioral_state set.
    Compares the speaking face with every listener face to detect:
      - Agreement: listener nodding/smiling while speaker talks
      - Disagreement: listener shaking/arms_crossed while speaker talks
      - Disengagement: listener looking away/distracted while speaker talks
      - Incongruence: listener's face and body contradict each other
    """

    @staticmethod
    def detect_interactions(
        frame_features_list: list[FrameFeatures],
        diar_segments: list[dict],
        timestamp_ms: int,
    ) -> list[dict]:
        """
        Returns list of interaction dicts:
          { "speaker", "reactor", "interaction", "evidence", "timestamp_ms" }
        """
        interactions: list[dict] = []

        if len(frame_features_list) < 2:
            return interactions

        # Find who is speaking at this timestamp from behavioral state first,
        # then fall back to diarization segments.
        speaking_face = None
        for ff in frame_features_list:
            if ff.behavioral_state == "speaking":
                speaking_face = ff
                break

        if speaking_face is None:
            active_speaker_label: str | None = None
            for seg in diar_segments:
                if seg.get("start_ms", 0) <= timestamp_ms <= seg.get("end_ms", 0):
                    active_speaker_label = seg.get("speaker")
                    break
            if active_speaker_label:
                for ff in frame_features_list:
                    if getattr(ff, "speaker_id", "") == active_speaker_label:
                        speaking_face = ff
                        break

        if speaking_face is None:
            return interactions

        speaker_id = (
            getattr(speaking_face, "speaker_id", "")
            or f"Face_{getattr(speaking_face, 'face_index', 0)}"
        )

        for ff in frame_features_list:
            if ff is speaking_face or not ff.face_detected:
                continue

            reactor_id = (
                getattr(ff, "speaker_id", "")
                or f"Face_{getattr(ff, 'face_index', 0)}"
            )

            if ff.behavioral_state == "agreeing":
                interactions.append({
                    "speaker": speaker_id,
                    "reactor": reactor_id,
                    "interaction": "agrees",
                    "evidence": [ff.behavioral_state_detail],
                    "timestamp_ms": timestamp_ms,
                })

            elif ff.behavioral_state == "disagreeing":
                interactions.append({
                    "speaker": speaker_id,
                    "reactor": reactor_id,
                    "interaction": "disagrees",
                    "evidence": ff.behavioral_state_detail.split("+") if ff.behavioral_state_detail else ["disagreeing"],
                    "timestamp_ms": timestamp_ms,
                })

            elif ff.behavioral_state == "tense":
                interactions.append({
                    "speaker": speaker_id,
                    "reactor": reactor_id,
                    "interaction": "uncomfortable",
                    "evidence": ff.behavioral_state_detail.split("+") if ff.behavioral_state_detail else ["tense"],
                    "timestamp_ms": timestamp_ms,
                })

            elif ff.behavioral_state == "distracted":
                interactions.append({
                    "speaker": speaker_id,
                    "reactor": reactor_id,
                    "interaction": "disengaged",
                    "evidence": [ff.behavioral_state_detail],
                    "timestamp_ms": timestamp_ms,
                })

            if _has_incongruence(ff):
                interactions.append({
                    "speaker": speaker_id,
                    "reactor": reactor_id,
                    "interaction": "incongruent",
                    "evidence": _describe_incongruence(ff),
                    "timestamp_ms": timestamp_ms,
                })

        return interactions


# ══════════════════════════════════════════════════════════════════════════════
# OverlayRenderer  — draws landmarks + signal labels on BGR frames
# ══════════════════════════════════════════════════════════════════════════════

class OverlayRenderer:
    """
    Draws MediaPipe landmark overlays and signal text onto BGR video frames.
    Called from VideoFeatureExtractor._extract_frames when overlay_output_path
    is supplied, and from VideoPipeline.run() for the signal burn-in pass.
    """

    # Face mesh contours (478-point MediaPipe model landmark indices)
    _FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
    ]
    _LEFT_EYE = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398, 362,
    ]
    _RIGHT_EYE = [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246, 33,
    ]
    _LIPS = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
    ]

    # Pose skeleton connections (COCO 33-landmark body model)
    _POSE_SEGS = [
        (11, 12),            # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso sides
        (23, 24),            # hips
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
        (0, 11), (0, 12),    # neck to shoulders
    ]

    # Hand skeleton connections (21-point hand model)
    _HAND_SEGS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    def draw_frame(
        self,
        bgr: np.ndarray,
        face_result,
        pose_result,
        hand_result,
        active_signals: Optional[list] = None,
        ff: Optional["FrameFeatures"] = None,
        frame_features_list: Optional[list["FrameFeatures"]] = None,
        interactions: Optional[list[dict]] = None,
        display_names: Optional[dict] = None,
    ) -> np.ndarray:
        """Return an annotated copy of bgr with all landmark overlays applied."""
        out = bgr.copy()
        h, w = out.shape[:2]

        if face_result and getattr(face_result, "face_landmarks", None):
            self._draw_face_mesh(out, face_result, h, w)
        if pose_result and getattr(pose_result, "pose_landmarks", None):
            self._draw_pose(out, pose_result, h, w)
        if hand_result and getattr(hand_result, "hand_landmarks", None):
            self._draw_hands(out, hand_result, h, w)
        if face_result and getattr(face_result, "face_landmarks", None):
            self._draw_gaze_arrow(out, face_result, h, w)

        # Per-face behavioral state panels + per-face signal badges
        ffl = frame_features_list or ([ff] if ff else [])
        for face_ff in ffl:
            if not face_ff or not face_ff.face_detected:
                continue
            fi = getattr(face_ff, "face_index", 0)
            face_lms = getattr(face_result, "face_landmarks", None) if face_result else None
            if face_lms and fi < len(face_lms):
                self._draw_behavioral_state_panel(
                    out, face_ff, face_lms[fi], h, w, display_names=display_names
                )
                if face_ff.hand_touch_zone:
                    self._draw_touch_zone(out, face_lms[fi], face_ff.hand_touch_zone, h, w)
                if active_signals:
                    spk = getattr(face_ff, "speaker_id", "") or f"Face_{fi}"
                    face_sigs = [s for s in active_signals if s.get("speaker_id") == spk]
                    if face_sigs:
                        self._draw_face_signal_badges(out, face_sigs, face_lms[fi], h, w)

        # Posture indicators for primary face
        primary_ff = ffl[0] if ffl else ff
        if primary_ff:
            self._draw_posture_indicators(out, primary_ff, h, w)

        # Cross-speaker interaction bar
        if interactions:
            self._draw_interaction_bar(out, interactions, h, w)

        # Global bottom-left labels — only session-level / unattributed signals
        if active_signals:
            unattributed = [
                s for s in active_signals
                if not s.get("speaker_id") or s.get("speaker_id") in ("session", "all", "")
            ]
            if unattributed:
                self._draw_labels(out, unattributed, h)
        return out

    def _draw_face_signal_badges(
        self, bgr: np.ndarray, signals: list[dict],
        face_landmarks, h: int, w: int,
    ) -> None:
        """Draw top-2 signal badges below each face's behavioral state panel."""
        import cv2
        face_xs = [lm.x * w for lm in face_landmarks]
        face_ys = [lm.y * h for lm in face_landmarks]
        face_cx     = int(sum(face_xs) / len(face_xs))
        face_bottom = int(max(face_ys))

        top2 = sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True)[:2]
        y = face_bottom + 16
        for sig in top2:
            label = self._get_overlay_label(sig)
            conf  = f"{sig.get('confidence', 0.0):.0%}"
            text  = f"{label} {conf}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            tx = max(5, min(face_cx - tw // 2, w - tw - 5))
            overlay = bgr.copy()
            cv2.rectangle(overlay, (tx - 3, y - th - 2), (tx + tw + 3, y + 3), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, bgr, 0.4, 0, bgr)
            cv2.putText(bgr, text, (tx, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 200), 1, cv2.LINE_AA)
            y += 14

    def _draw_touch_zone(self, bgr: np.ndarray, face_landmarks, touch_zone: str, h: int, w: int) -> None:
        """Highlight the face zone being touched with a coloured circle + label."""
        import cv2
        zone_landmark: dict[str, int] = {
            "chin": 152, "mouth": 13, "nose": 1,
            "cheek": 116, "ear": 234, "neck": 152, "forehead": 10,
        }
        zone_color: dict[str, tuple] = {
            "chin":     (0,  200, 255),
            "mouth":    (0,   0,  255),
            "nose":     (0,  165, 255),
            "cheek":    (255, 200,  0),
            "ear":      (200, 200,  0),
            "neck":     (0,  100, 255),
            "forehead": (255,  0,  200),
        }
        lm_idx = zone_landmark.get(touch_zone, 1)
        if lm_idx >= len(face_landmarks):
            return
        x = int(face_landmarks[lm_idx].x * w)
        y = int(face_landmarks[lm_idx].y * h)
        if touch_zone == "neck":
            y += int(h * 0.05)
        _ZONE_DISPLAY = {
            "chin":     "EVALUATING",
            "mouth":    "HOLDING BACK",
            "nose":     "UNCOMFORTABLE",
            "cheek":    "ATTENTIVE",
            "ear":      "SELF-SOOTHING",
            "neck":     "EXPOSED",
            "forehead": "FRUSTRATED",
        }
        color = zone_color.get(touch_zone, (0, 200, 200))
        cv2.circle(bgr, (x, y), 18, color, 2, cv2.LINE_AA)
        cv2.circle(bgr, (x, y), 8,  color, -1, cv2.LINE_AA)
        cv2.putText(bgr, _ZONE_DISPLAY.get(touch_zone, touch_zone.upper()), (x + 22, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def _draw_posture_indicators(self, bgr: np.ndarray, ff: "FrameFeatures", h: int, w: int) -> None:
        """Draw posture badges in the bottom-right corner."""
        import cv2
        indicators = []
        if ff.arms_crossed:
            indicators.append(("ARMS CROSSED",  (0, 165, 255)))
        if ff.finger_steepling:
            indicators.append(("STEEPLING",     (0, 255, 100)))
        if ff.head_supported_by_hand:
            indicators.append(("HEAD RESTING",  (200, 200, 0)))
        if ff.hands_clasped:
            indicators.append(("HANDS CLASPED", (180, 180, 180)))
        if ff.thumb_up:
            indicators.append(("THUMB UP",      (0, 220, 80)))
        if ff.thumb_down:
            indicators.append(("THUMB DOWN",    (0, 80, 220)))
        if ff.pointing_up:
            indicators.append(("POINTING",      (200, 180, 0)))
        if ff.victory_sign:
            indicators.append(("VICTORY",       (0, 200, 180)))
        if ff.closed_fist:
            indicators.append(("FIST",          (180, 60, 60)))
        if ff.open_palms:
            indicators.append(("OPEN PALMS",    (60, 180, 60)))
        if ff.elbow_expansion >= 0.15:
            indicators.append(("ARMS OPEN",     (80, 200, 80)))
        elif ff.elbow_expansion <= -0.20:
            indicators.append(("ARMS TIGHT",    (180, 80, 80)))
        if not indicators:
            return
        y_offset = h - 30 * len(indicators) - 10
        for label, color in indicators:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x = w - tw - 20
            cv2.rectangle(bgr, (x - 5, y_offset - th - 4), (x + tw + 5, y_offset + 4), (0, 0, 0), -1)
            cv2.putText(bgr, label, (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_offset += 30

    def _draw_behavioral_state_panel(
        self, bgr: np.ndarray, ff: "FrameFeatures",
        face_landmarks, h: int, w: int,
        display_names: Optional[dict] = None,
    ) -> None:
        """Draw a small state panel anchored above each face bbox."""
        import cv2
        if not ff.face_detected or not ff.behavioral_state:
            return

        face_xs = [lm.x * w for lm in face_landmarks]
        face_ys = [lm.y * h for lm in face_landmarks]
        face_cx  = int(sum(face_xs) / len(face_xs))
        face_top = int(min(face_ys))

        state_config = {
            "speaking":    ("SPEAKING",   (0, 200,   0)),
            "agreeing":    ("AGREEING",   (0, 255, 100)),
            "disagreeing": ("DISAGREES",  (0,   0, 255)),
            "tense":       ("TENSE",      (0, 100, 255)),
            "distracted":  ("DISTRACTED", (128, 128, 128)),
            "engaged":     ("ENGAGED",    (200, 200,   0)),
            "listening":   ("LISTENING",  (180, 180, 180)),
        }
        label, color = state_config.get(ff.behavioral_state, ("", (180, 180, 180)))

        raw_label = getattr(ff, "speaker_id", "") or f"Face_{getattr(ff, 'face_index', 0)}"
        speaker_label = (display_names or {}).get(raw_label, raw_label)
        stress_colors = {"low": (0, 200, 0), "moderate": (0, 200, 255), "high": (0, 0, 255)}
        stress_color  = stress_colors.get(ff.stress_level, (180, 180, 180))

        panel_w = 130
        panel_h = 55
        panel_x = max(5, min(face_cx - panel_w // 2, w - panel_w - 5))
        panel_y = max(5, face_top - panel_h - 8)

        overlay = bgr.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, bgr, 0.3, 0, bgr)
        cv2.rectangle(bgr, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), color, 1, cv2.LINE_AA)

        line_y = panel_y + 14
        cv2.putText(bgr, speaker_label, (panel_x + 4, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        line_y += 13
        cv2.putText(bgr, label, (panel_x + 4, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        line_y += 13
        if ff.behavioral_state_detail and ff.behavioral_state_detail != ff.behavioral_state:
            detail_text = ff.behavioral_state_detail.replace("+", " + ").replace("_", " ")
            if len(detail_text) > 18:
                detail_text = detail_text[:17] + "…"
            cv2.putText(bgr, detail_text, (panel_x + 4, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 160, 160), 1, cv2.LINE_AA)
            line_y += 13
        stress_text = f"Stress: {ff.stress_level}"
        cv2.putText(bgr, stress_text, (panel_x + 4, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, stress_color, 1, cv2.LINE_AA)

    def _draw_interaction_bar(
        self, bgr: np.ndarray, interactions: list[dict], h: int, w: int,
    ) -> None:
        """Draw bottom bar showing active cross-speaker interactions (max 2 lines)."""
        import cv2
        if not interactions:
            return

        priority = {"disagrees": 0, "incongruent": 1, "uncomfortable": 2, "agrees": 3, "disengaged": 4}
        sorted_ints = sorted(interactions, key=lambda x: priority.get(x["interaction"], 5))
        display = sorted_ints[:2]

        bar_h = 18 * len(display) + 8
        bar_y = h - bar_h - 5

        overlay = bgr.copy()
        cv2.rectangle(overlay, (5, bar_y), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, bgr, 0.25, 0, bgr)

        interaction_colors = {
            "agrees":        (0, 255, 100),
            "disagrees":     (0,   0, 255),
            "uncomfortable": (0, 100, 255),
            "incongruent":   (0,   0, 200),
            "disengaged":    (128, 128, 128),
        }

        line_y = bar_y + 14
        for interaction in display:
            color       = interaction_colors.get(interaction["interaction"], (180, 180, 180))
            evidence_str = ", ".join(interaction.get("evidence", []))
            text = (
                f"{interaction['speaker']} speaking -> "
                f"{interaction['reactor']} {interaction['interaction'].upper()}"
            )
            if evidence_str:
                text += f" ({evidence_str})"
            if len(text) > 80:
                text = text[:79] + "…"
            cv2.putText(bgr, text, (10, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            line_y += 18

    def _draw_face_mesh(self, bgr: np.ndarray, face_result, h: int, w: int) -> None:
        import cv2
        try:
            import mediapipe as mp
            tesselation = [(c.start, c.end) for c in mp.solutions.face_mesh.FACEMESH_TESSELATION]
            contours    = [(c.start, c.end) for c in mp.solutions.face_mesh.FACEMESH_CONTOURS]
        except Exception:
            tesselation = []
            contours    = [(self._FACE_OVAL[i], self._FACE_OVAL[i+1]) for i in range(len(self._FACE_OVAL)-1)]

        color_tess    = (0, 200, 60)    # green mesh lines
        color_contour = (0, 255, 120)   # brighter green for outer contours
        color_dot     = (50, 255, 50)   # landmark dots

        for face_lm in face_result.face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lm]
            n = len(pts)
            # Full tessellation mesh
            for a, b in tesselation:
                if a < n and b < n:
                    cv2.line(bgr, pts[a], pts[b], color_tess, 1, cv2.LINE_AA)
            # Contour lines on top (brighter)
            for a, b in contours:
                if a < n and b < n:
                    cv2.line(bgr, pts[a], pts[b], color_contour, 1, cv2.LINE_AA)
            # Landmark dots — radius 2 so they're visible at 1080p
            for pt in pts[:468]:
                cv2.circle(bgr, pt, 2, color_dot, -1, cv2.LINE_AA)

    def _draw_pose(self, bgr: np.ndarray, pose_result, h: int, w: int) -> None:
        import cv2
        # MediaPipe extrapolates hips/knees/ankles even for head+shoulders crops.
        # Only draw landmarks the model is confident about to prevent skeleton
        # branches from bleeding into adjacent tiles in multi-person grid views.
        _VIS_MIN = 0.5
        s = max(1.0, min(h, w) / 720.0)
        line_t = max(1, round(s))           # 1 at 720p, 2 at 1440p
        dot_r  = max(1, round(1.5 * s))     # 1-2px at 720p

        color_bone = (255, 100, 30)
        color_dot  = (200, 200, 255)
        for pose_lm in pose_result.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_lm]
            vis = [getattr(lm, "visibility", 1.0) for lm in pose_lm]
            n = len(pts)
            for a, b in self._POSE_SEGS:
                if a < n and b < n and vis[a] >= _VIS_MIN and vis[b] >= _VIS_MIN:
                    cv2.line(bgr, pts[a], pts[b], color_bone, line_t, cv2.LINE_AA)
            for i, pt in enumerate(pts[:33]):
                if vis[i] >= _VIS_MIN:
                    cv2.circle(bgr, pt, dot_r, color_dot, -1, cv2.LINE_AA)

    def _draw_hands(self, bgr: np.ndarray, hand_result, h: int, w: int) -> None:
        import cv2
        # Thin connections + small joints — 21 overlapping circles become a blob at radius 4
        s = max(1.0, min(h, w) / 720.0)
        dot_r = max(1, round(s))    # 1px at 720p, 2px at 1440p

        color_seg = (0, 200, 220)
        color_dot = (0, 255, 255)
        for hand_lm in hand_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            n = len(pts)
            for a, b in self._HAND_SEGS:
                if a < n and b < n:
                    cv2.line(bgr, pts[a], pts[b], color_seg, 1, cv2.LINE_AA)
            # Wrist (0) and knuckles (1,5,9,13,17) slightly larger; fingertips tiny
            for i, pt in enumerate(pts):
                r = dot_r + 1 if i in (0, 1, 5, 9, 13, 17) else dot_r
                cv2.circle(bgr, pt, r, color_dot, -1, cv2.LINE_AA)

    def _draw_gaze_arrow(
        self, bgr: np.ndarray, face_result, h: int, w: int
    ) -> None:
        """Cyan gaze arrow from nose bridge for every detected face."""
        import cv2
        scale = min(w, h) * 0.12
        for face_lm in face_result.face_landmarks:
            try:
                if len(face_lm) <= _RIGHT_IRIS_IDX + 4:
                    continue
                gaze_x, gaze_y = VideoFeatureExtractor._compute_gaze(face_lm)
                nose_x = int(face_lm[1].x * w)
                nose_y = int(face_lm[1].y * h)
                end_x = int(nose_x + gaze_x * scale)
                end_y = int(nose_y + gaze_y * scale)
                cv2.arrowedLine(
                    bgr, (nose_x, nose_y), (end_x, end_y),
                    (255, 230, 0), 2, tipLength=0.35, line_type=cv2.LINE_AA,
                )
            except Exception:
                continue

    _OVERLAY_LABELS: dict = {
        "self_touch": "Self-Touch",
        "face_region_touch": {
            "chin_touch_evaluation":      "Evaluating",
            "mouth_cover_suppression":    "Holding Back",
            "nose_touch_discomfort":      "Uncomfortable",
            "cheek_touch_listening":      "Attentive",
            "cheek_rest_fatigue":         "Low Energy",
            "ear_touch_soothing":         "Self-Soothing",
            "neck_touch_vulnerability":   "Feeling Exposed",
            "forehead_touch_frustration": "Frustrated",
            "eye_rub_discomfort":         "Eye Rub",
            "eye_block_disbelief":        "Disbelief",
        },
        "arms_crossed":      "Guarded",
        "finger_steepling":  "Confident",
        "hands_clasped":     "Restrained",
        "hand_gesture": {
            "approval":    "Approval",
            "disapproval": "Disapproval",
            "emphasis":    "Emphasizing",
            "victory":     "Victory",
            "tension":     "Tense",
            "open":        "Open",
        },
        "head_supported":    "Checked Out",
        "head_nod":          "Agreeing",
        "head_shake":        "Disagreeing",
        "lip_pursing":       "Biting Tongue",
        "facial_stress":     "Under Pressure",
        "sustained_distraction": "Not Paying Attention",
        "body_language_cluster": {
            "skepticism_cluster":           "Skeptical",
            "stress_anxiety_cluster":       "Stressed",
            "confidence_authority_cluster": "In Command",
            "disengagement_boredom_cluster":"Checked Out",
        },
        "cross_speaker_interaction": {
            "agreement_reaction":   "Agrees",
            "disagreement_reaction":"Disagrees",
            "discomfort_reaction":  "Uncomfortable",
            "incongruent_reaction": "Mixed Signals",
        },
        "screen_contact":         "Eye Contact",
        "attention_level":        "Attention",
        "gaze_direction_shift":   "Looked Away",
        "posture":                "Posture",
        "facial_engagement":      "Expression Level",
        "blink_rate_anomaly":     "Blink Rate",
        "body_fidgeting":         "Restless",
        "body_lean":              "Leaning",
        "shoulder_tension":       "Shoulder Tension",
        "gesture_animation":      "Gesturing",
        "cognitive_overload":     "Overwhelmed",
        "emotional_suppression":  "Suppressing",
        "vocal_stress_score":     "Voice Stress",
        "filler_detection":       "Fillers",
        "tone_classification":    "Tone",
        "speech_rate_anomaly":    "Speech Pace",
        "pitch_elevation_flag":   "Pitch Spike",
        "buying_signal":          "Buying Signal",
        "objection_signal":       "Objection",
        "rapport_indicator":      "Rapport",
        "head_body_incongruence": "Mixed Signals",
        "escalation_ladder":      "Conflict Rising",
        "engagement_decay":       "Engagement Fading",
        "stress_trajectory":      "Stress Trend",
        "adaptation_pattern":     "Adapting",
        "fatigue_detection":      "Getting Tired",
    }

    def _get_overlay_label(self, signal: dict) -> str:
        sig_type   = signal.get("signal_type", "")
        value_text = signal.get("value_text", "")
        entry = self._OVERLAY_LABELS.get(sig_type)
        if isinstance(entry, dict):
            return entry.get(value_text, value_text.replace("_", " ").title())
        if isinstance(entry, str):
            return entry
        return sig_type.replace("_", " ").title()

    def _draw_labels(self, bgr: np.ndarray, signals: list, h: int) -> None:
        """Up to 8 signal labels at bottom-left, sorted by confidence desc."""
        import cv2
        top = sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True)[:8]
        y = h - 12
        for sig in top:
            label = self._get_overlay_label(sig)
            conf  = f"{sig.get('confidence', 0.0):.0%}"
            cv2.putText(
                bgr, f"{label}  {conf}", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
            )
            y -= 16

    def burn_signal_labels(
        self,
        video_path: str,
        signals: list,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Burn per-frame signal text labels onto a video.

        If output_path is None the source file is overwritten in-place.
        If output_path is set the source is untouched and output is written there.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"burn_signal_labels: cannot open {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        final_dst = output_path or video_path
        tmp = final_dst + ".burn.tmp.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp, fourcc, fps, (fw, fh))

        frame_idx = 0
        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break
            ts_ms = int(frame_idx / fps * 1000)
            active = [
                s for s in signals
                if s.get("window_start_ms", 0) <= ts_ms < s.get("window_end_ms", ts_ms + 1)
            ]
            if active:
                self._draw_labels(bgr, active, fh)
            writer.write(bgr)
            frame_idx += 1

        cap.release()
        writer.release()

        _reencode_for_browser(tmp, final_dst)


# ══════════════════════════════════════════════════════════════════════════════
# CentroidTracker  — stable face IDs across frames
# ══════════════════════════════════════════════════════════════════════════════

class CentroidTracker:
    """
    Assigns temporally stable face IDs across video frames using centroid proximity.

    Each detected face gets a persistent track_id that survives temporary detection
    drops, size-ranking changes (someone leans forward), and tile re-ordering.
    Without this, face_index is re-sorted by area every frame, so WindowAggregator
    groups frames from multiple physical people into the same "speaker" baseline.

    Algorithm: greedy nearest-neighbour matching with exponential moving average
    position updates.  Centroid threshold 0.08 (normalised) handles small head
    movements while preventing cross-tile matches in video-call grids.
    """

    def __init__(
        self,
        max_disappeared: int = 15,      # frames before a lost track is removed (~3s at 5fps)
        match_threshold: float = 0.08,  # max centroid distance for a match (normalised)
    ) -> None:
        self._next_id: int = 0
        self._tracks: dict[int, tuple[float, float]] = {}
        self._disappeared: dict[int, int] = {}
        self._max_disappeared = max_disappeared
        self._match_threshold = match_threshold

    def update(self, centroids: list[tuple[float, float]]) -> list[int]:
        """
        Update tracker with new frame's face centroids.

        Args:
            centroids: [(cx, cy), ...] one per detected face, same order as face_landmarks.

        Returns:
            List of stable track_ids, one per input centroid, in the same order.
        """
        if not centroids:
            for tid in list(self._disappeared):
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self._max_disappeared:
                    del self._tracks[tid]
                    del self._disappeared[tid]
            return []

        if not self._tracks:
            return [self._register(cx, cy) for cx, cy in centroids]

        track_ids = list(self._tracks.keys())
        track_centres = [self._tracks[tid] for tid in track_ids]

        distances: list[tuple[float, int, int]] = []
        for ti, (tx, ty) in enumerate(track_centres):
            for ci, (cx, cy) in enumerate(centroids):
                d = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                distances.append((d, ti, ci))
        distances.sort(key=lambda x: x[0])

        matched_tracks: set[int] = set()
        matched_centroids: set[int] = set()
        result_ids: dict[int, int] = {}

        for dist, ti, ci in distances:
            if ti in matched_tracks or ci in matched_centroids:
                continue
            if dist > self._match_threshold:
                break
            tid = track_ids[ti]
            result_ids[ci] = tid
            old_cx, old_cy = self._tracks[tid]
            new_cx, new_cy = centroids[ci]
            self._tracks[tid] = (old_cx * 0.7 + new_cx * 0.3, old_cy * 0.7 + new_cy * 0.3)
            self._disappeared[tid] = 0
            matched_tracks.add(ti)
            matched_centroids.add(ci)

        for ti, tid in enumerate(track_ids):
            if ti not in matched_tracks:
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self._max_disappeared:
                    del self._tracks[tid]
                    del self._disappeared[tid]

        for ci, (cx, cy) in enumerate(centroids):
            if ci not in matched_centroids:
                result_ids[ci] = self._register(cx, cy)

        return [result_ids[ci] for ci in range(len(centroids))]

    def _register(self, cx: float, cy: float) -> int:
        tid = self._next_id
        self._tracks[tid] = (cx, cy)
        self._disappeared[tid] = 0
        self._next_id += 1
        return tid


# ══════════════════════════════════════════════════════════════════════════════
# FaceEmbeddingExtractor  — ArcFace embeddings for cross-session identity
# ══════════════════════════════════════════════════════════════════════════════

class FaceEmbeddingExtractor:
    """
    Extract 512-dim ArcFace face embeddings for cross-session identification.

    Uses InsightFace's buffalo_l model (ArcFace, NIST FRVT Rank-1 2021).
    Called once per session after extraction completes — selects the single
    best frame per tracked face and extracts a normalised embedding.

    The embedding is L2-normalised so cosine similarity = dot product.
    Matching threshold: 0.55 face-only, 0.50 combined with voice.

    Singleton pattern: model is ~200 MB — load once per process.
    """

    _instance: Optional["FaceEmbeddingExtractor"] = None

    @classmethod
    def get_instance(cls) -> "FaceEmbeddingExtractor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            self._available = True
            logger.info("InsightFace ArcFace (buffalo_l) loaded for face identification")
        except Exception as exc:
            logger.warning(f"InsightFace not available — face identification disabled: {exc}")
            self._available = False
            self._app = None

    @property
    def available(self) -> bool:
        return self._available

    def extract_from_crops(
        self,
        face_crops: dict[int, np.ndarray],
    ) -> dict[int, tuple[list[float], bytes]]:
        """
        Extract ArcFace embedding + JPEG thumbnail from pre-cropped face images.

        Args:
            face_crops: {track_id: BGR face crop ndarray}
                        Each crop should be the best-quality frame for this face.

        Returns:
            {track_id: (embedding_list_512, thumbnail_jpeg_bytes)}
            Only includes faces where InsightFace detection succeeded.
        """
        if not self._available:
            return {}

        import cv2
        results: dict[int, tuple[list[float], bytes]] = {}

        for track_id, crop_bgr in face_crops.items():
            if crop_bgr is None or crop_bgr.size < 100:
                continue
            try:
                faces = self._app.get(crop_bgr)
                if not faces:
                    # Retry with padding — InsightFace needs context around the face
                    h, w = crop_bgr.shape[:2]
                    padded = cv2.copyMakeBorder(
                        crop_bgr, h // 4, h // 4, w // 4, w // 4,
                        cv2.BORDER_CONSTANT, value=(128, 128, 128),
                    )
                    faces = self._app.get(padded)
                if not faces:
                    logger.debug(f"InsightFace: no face in crop for track_{track_id}")
                    continue
                # Largest detected face = primary subject of this crop
                best_face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                # L2-normalize so np.dot() == cosine similarity in [-1, 1]
                emb_raw = best_face.embedding
                norm = float(np.linalg.norm(emb_raw))
                embedding = (emb_raw / norm) if norm > 0 else emb_raw
                thumbnail = self._make_thumbnail(crop_bgr, best_face.bbox, size=96)
                results[track_id] = (embedding.tolist(), thumbnail)
            except Exception as exc:
                logger.debug(f"InsightFace embedding failed for track_{track_id}: {exc}")

        logger.info(
            f"Face embeddings extracted: {len(results)}/{len(face_crops)} faces"
        )
        return results

    @staticmethod
    def _make_thumbnail(bgr: np.ndarray, bbox: np.ndarray, size: int = 96) -> bytes:
        """Crop face from bbox with 20% padding, resize to size×size, encode as JPEG."""
        import cv2
        h, w = bgr.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        face_crop = bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = bgr
        thumb = cv2.resize(face_crop, (size, size), interpolation=cv2.INTER_AREA)
        _, jpeg_bytes = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg_bytes.tobytes()


# ══════════════════════════════════════════════════════════════════════════════
# VideoFeatureExtractor  — main extractor
# ══════════════════════════════════════════════════════════════════════════════

class VideoFeatureExtractor:
    """
    Orchestrates MediaPipe Face Landmarker + Pose Landmarker + Hands.
    Processes a video file at target_fps and returns per-window features.

    Landmarkers are created fresh per-video (VIDEO mode requires monotonically
    increasing timestamps; fresh instances avoid state carry-over across calls).
    Model files are downloaded lazily on first use via MediaPipeModelManager.

    OOP principles:
    - Single responsibility: extraction only (aggregation delegated to WindowAggregator)
    - Encapsulation: MediaPipe objects never escape the class
    - Dependency injection: model_dir and fps configurable via __init__
    """

    _DEFAULT_FACE_BUDGET = 10

    @classmethod
    def faces_for_meeting(cls, meeting_type: str) -> int:
        """Max faces/poses/hands from shared SPEAKER_DEFAULTS.max for this meeting type."""
        from shared.config.content_types import get_speaker_defaults
        return get_speaker_defaults(meeting_type).get("max", cls._DEFAULT_FACE_BUDGET)

    def __init__(
        self,
        model_dir: str = "models/mediapipe",
        target_fps: int = TARGET_FPS,
        window_ms: int = WINDOW_MS,
        num_faces: int = _DEFAULT_FACE_BUDGET,
    ) -> None:
        self._model_mgr  = MediaPipeModelManager(model_dir)
        self._target_fps = target_fps
        self._window_ms  = window_ms
        self._num_faces  = num_faces
        self._aggregator = WindowAggregator(window_ms, target_fps)
        self._mp_available: Optional[bool] = None    # lazy probe
        self._diar_segments: list[dict] = []         # set by caller before extract_all for interaction detection

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_all(
        self,
        video_path: str,
        overlay_output_path: Optional[str] = None,
        meeting_type: str = "",
    ) -> tuple[list[WindowFeatures], dict[int, list[tuple[int, float]]]]:
        """
        Process video at target_fps and return aggregated windows + lip activity map.

        Args:
            video_path:           Path to mp4/webm/avi file.
            overlay_output_path:  If set, write a landmark-annotated mp4 here.
            meeting_type:         Used to set the face/pose/hand detection budget.

        Returns:
            (windows, lip_activity_map)
            windows:          List of WindowFeatures, one per 2-second window per face.
            lip_activity_map: {face_index: [(timestamp_ms, lip_score), ...]}
                              Empty dict if video cannot be read or MediaPipe unavailable.
        """
        if meeting_type:
            self._num_faces = self.faces_for_meeting(meeting_type)
            logger.info(f"Face budget for '{meeting_type}': {self._num_faces}")
        if not self._check_mediapipe():
            logger.error("MediaPipe not available — returning empty features.")
            return [], {}

        frames = self._extract_frames(video_path, overlay_output_path=overlay_output_path)

        # Build lip activity timeseries BEFORE aggregation discards per-frame blendshapes
        lip_activity_map = self.build_lip_activity_map(frames)

        windows = self._aggregator.aggregate(frames)
        return windows, lip_activity_map

    @staticmethod
    def build_lip_activity_map(
        frames: list[FrameFeatures],
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Build a per-face-index timeseries of lip/mouth activity from raw frames.

        Each entry: (timestamp_ms, lip_activity_score)

        Composite blendshape score (Haider 2021: jawOpen is the strongest single
        predictor of active speech in MediaPipe, r=0.82 with ground-truth VAD):
          jawOpen          × 0.50  — primary mouth-opening during speech
          mouthLowerDown   × 0.20  — lower-lip descent (averaged L+R)
          mouthFunnel      × 0.15  — vowel articulation ("oo", "oh")
          mouthPucker      × 0.10  — lip rounding
          (1 - mouthClose) × 0.05  — inverse: high mouthClose = mouth shut
        """
        lip_map: dict[int, list[tuple[int, float]]] = defaultdict(list)

        for ff in frames:
            if not ff.face_detected or not ff.blendshapes:
                continue

            face_idx = getattr(ff, "face_index", 0)
            bs = ff.blendshapes

            jaw_open       = bs.get("jawOpen", 0.0)
            mouth_funnel   = bs.get("mouthFunnel", 0.0)
            mouth_pucker   = bs.get("mouthPucker", 0.0)
            mouth_close    = bs.get("mouthClose", 0.0)
            mouth_lower_down = (
                bs.get("mouthLowerDownLeft", 0.0) + bs.get("mouthLowerDownRight", 0.0)
            ) / 2.0

            lip_score = (
                jaw_open         * 0.50
                + mouth_lower_down * 0.20
                + mouth_funnel     * 0.15
                + mouth_pucker     * 0.10
                + (1.0 - mouth_close) * 0.05
            )

            lip_map[face_idx].append((ff.timestamp_ms, lip_score))

        return dict(lip_map)

    def burn_landmarks_and_labels(
        self,
        video_path: str,
        signals: list,
        output_path: Optional[str] = None,
        display_names: Optional[dict] = None,
    ) -> None:
        """
        Phase 2 overlay pass: re-run MediaPipe on the original video and draw
        face mesh, pose skeleton, hand landmarks, plus signal text labels on
        every frame. Writes a browser-playable H.264 MP4.

        MediaPipe is sampled every N frames (same skip ratio as extraction) and
        the last results are cached for intermediate frames — same approach as
        _extract_frames — so this runs at roughly the same speed as Phase 1.
        Falls back to text-only burn if MediaPipe is unavailable.
        """
        import cv2

        if not self._check_mediapipe():
            logger.warning("[burn] MediaPipe unavailable — falling back to text-only burn")
            OverlayRenderer().burn_signal_labels(video_path, signals, output_path)
            return

        import mediapipe as mp

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"[burn] Cannot open {video_path}")
            return

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        skip: int = max(1, round(video_fps / self._target_fps))

        final_dst = output_path or video_path
        tmp = final_dst + ".lm.tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp, fourcc, video_fps, (fw, fh))

        # Use full face budget during burn so every visible person gets landmarks.
        saved_nf = self._num_faces
        self._num_faces = self._DEFAULT_FACE_BUDGET
        processor = self._build_tiled_processor(mp)
        self._num_faces = saved_nf
        renderer = OverlayRenderer()

        last_face_result = None
        last_pose_result = None
        last_hand_result = None
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, bgr = cap.read()
                if not ret:
                    break
                ts_ms = int(frame_idx / video_fps * 1000)

                if frame_idx % skip == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    try:
                        last_face_result, last_pose_result, last_hand_result = (
                            processor.detect(rgb, ts_ms)
                        )
                    except Exception:
                        last_face_result = last_pose_result = last_hand_result = None

                active = [
                    s for s in signals
                    if s.get("window_start_ms", 0) <= ts_ms < s.get("window_end_ms", ts_ms + 1)
                ]
                annotated = renderer.draw_frame(
                    bgr, last_face_result, last_pose_result, last_hand_result,
                    active if active else None,
                    display_names=display_names,
                )
                writer.write(annotated)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()
            processor.close()

        _reencode_for_browser(tmp, final_dst)
        logger.info(f"[burn] Landmarks+labels → {Path(final_dst).name} ({frame_idx} frames)")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check_mediapipe(self) -> bool:
        if self._mp_available is not None:
            return self._mp_available
        try:
            import mediapipe  # noqa: F401
            import cv2  # noqa: F401
            self._mp_available = True
        except ImportError:
            logger.warning("mediapipe or opencv not installed.")
            self._mp_available = False
        return self._mp_available

    def _extract_frames(
        self,
        video_path: str,
        overlay_output_path: Optional[str] = None,
    ) -> list[FrameFeatures]:
        import cv2
        import mediapipe as mp

        processor = self._build_tiled_processor(mp)
        renderer = OverlayRenderer() if overlay_output_path else None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip: int = max(1, round(video_fps / self._target_fps))

        writer = None
        overlay_tmp_path: Optional[str] = None
        if overlay_output_path and renderer:
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            Path(overlay_output_path).parent.mkdir(parents=True, exist_ok=True)
            overlay_tmp_path = overlay_output_path + ".tmp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(overlay_tmp_path, fourcc, video_fps, (fw, fh))
            logger.info(f"Overlay output: {overlay_output_path} (writing via tmp)")

        frames: list[FrameFeatures] = []
        frame_idx: int = 0
        prev_pose_lm_data: Optional[list] = None

        centroid_tracker = CentroidTracker(
            max_disappeared=90,   # ~18s at 5fps — faces in calls don't physically leave
            match_threshold=0.20, # wide enough to survive head turns and fast leans
        )

        # Cache last MediaPipe results — reused for overlay on non-sampled frames
        last_face_result = None
        last_pose_result = None
        last_hand_result = None
        last_ff: Optional[FrameFeatures] = None
        last_frame_features_list: list[FrameFeatures] = []
        last_interactions: list[dict] = []

        # Diarization segments (populated by caller via _diar_segments attribute if set)
        diar_segments: list[dict] = getattr(self, "_diar_segments", [])

        try:
            while cap.isOpened():
                ret, bgr = cap.read()
                if not ret:
                    break

                timestamp_ms = int(frame_idx / video_fps * 1000)

                if frame_idx % skip == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                    try:
                        last_face_result, last_pose_result, last_hand_result = (
                            processor.detect(rgb, timestamp_ms)
                        )
                    except Exception as exc:
                        logger.debug(f"Tiled detect error frame {frame_idx}: {exc}")
                        last_face_result = last_pose_result = last_hand_result = None

                    try:
                        frame_features_list = self._process_frame_from_results(
                            bgr, rgb, frame_idx, timestamp_ms,
                            last_face_result, last_pose_result, last_hand_result,
                            prev_pose_lm_data,
                        )

                        # ── Apply temporal face tracking ──────────────────────
                        # Replace area-sorted face_index with stable track_id so
                        # WindowAggregator always groups the same physical person.
                        centroids = [
                            (ff.face_centre_x, ff.face_centre_y)
                            for ff in frame_features_list
                            if ff.face_detected
                        ]
                        if centroids:
                            track_ids = centroid_tracker.update(centroids)
                            ci = 0
                            for ff in frame_features_list:
                                if ff.face_detected and ci < len(track_ids):
                                    ff.face_index = track_ids[ci]
                                    ci += 1

                        # Detect cross-speaker interactions for this frame
                        last_interactions = InteractionDetector.detect_interactions(
                            frame_features_list, diar_segments, timestamp_ms
                        )
                        # Store only the interactions where THIS face is the reactor,
                        # so WindowAggregator never sees another face's reactions.
                        for ffi in frame_features_list:
                            face_key = f"Face_{ffi.face_index}"
                            ffi._interactions = [  # type: ignore[attr-defined]
                                i for i in last_interactions if i["reactor"] == face_key
                            ]

                        frames.extend(frame_features_list)
                        last_frame_features_list = frame_features_list
                        for ff in frame_features_list:
                            if ff.face_index == 0:
                                last_ff = ff
                            if ff.face_index == 0 and ff.body_detected:
                                prev_pose_lm_data = getattr(ff, "_raw_pose_lm", None)
                                break
                    except Exception as exc:
                        logger.warning(f"Frame {frame_idx} processing error (skipping): {exc}")

                # Write overlay on every original-fps frame using cached results
                if writer is not None and renderer is not None:
                    try:
                        annotated = renderer.draw_frame(
                            bgr, last_face_result, last_pose_result, last_hand_result,
                            ff=last_ff,
                            frame_features_list=last_frame_features_list,
                            interactions=last_interactions,
                        )
                        writer.write(annotated)
                    except Exception as exc:
                        logger.debug(f"Overlay draw error frame {frame_idx}: {exc}")

                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
                if overlay_tmp_path and overlay_output_path:
                    _reencode_for_browser(overlay_tmp_path, overlay_output_path)
            processor.close()

        logger.info(
            f"Extracted {len(frames)} frames from {Path(video_path).name} "
            f"({frame_idx} total, every {skip}th frame at video_fps={video_fps:.1f})"
        )

        # ── Select best frame per tracked face for ArcFace embedding extraction ─
        # Quality = frontal (low yaw+pitch) × large bbox × well-lit (luminance).
        # Stored on self so run_analysis can access after extract_all() returns.
        self._best_face_crops: dict[int, np.ndarray] = {}
        best_quality: dict[int, float] = {}
        best_frame_info: dict[int, tuple[float, int, float, float, float]] = {}
        # track_id → (quality, frame_idx, cx, cy, face_box_area)

        for ff in frames:
            if not ff.face_detected or ff.face_box_area < 0.001:
                continue
            tid = ff.face_index
            frontal = 1.0 / (1.0 + abs(ff.head_yaw) * 0.05 + abs(ff.head_pitch) * 0.05)
            quality = frontal * ff.face_box_area * max(ff.face_luminance, 0.2)
            if quality > best_quality.get(tid, -1.0):
                best_quality[tid] = quality
                best_frame_info[tid] = (
                    quality, ff.frame_idx,
                    ff.face_centre_x, ff.face_centre_y, ff.face_box_area,
                )

        if best_frame_info:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(video_path)
            if _cap.isOpened():
                _fw = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                _fh = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))

                # Build frame_idx → list[track_ids] so each frame is read once
                frames_to_read: dict[int, list[int]] = {}
                for tid, (_, fidx, _, _, _) in best_frame_info.items():
                    frames_to_read.setdefault(fidx, []).append(tid)

                target_indices = sorted(frames_to_read.keys())
                target_pos = 0
                read_idx = 0

                while _cap.isOpened() and target_pos < len(target_indices):
                    ret, bgr = _cap.read()
                    if not ret:
                        break
                    if read_idx == target_indices[target_pos]:
                        for tid in frames_to_read[read_idx]:
                            _, _, cx, cy, area = best_frame_info[tid]
                            face_size = max(int((area ** 0.5) * _fw * 1.15), 250)
                            x1 = max(0, int(cx * _fw) - face_size // 2)
                            y1 = max(0, int(cy * _fh) - face_size // 2)
                            x2 = min(_fw, x1 + face_size)
                            y2 = min(_fh, y1 + face_size)
                            self._best_face_crops[tid] = bgr[y1:y2, x1:x2].copy()
                        target_pos += 1
                    read_idx += 1
                _cap.release()
            logger.info(
                f"Best-frame crops selected: {len(self._best_face_crops)} faces "
                f"from {len(best_frame_info)} tracked identities"
            )

        # ── Filter transient face tracks before identity merge ─────────────────
        # Tracks with very few frames are avatar images in shared screen content
        # or momentary false-positive detections — discard before running ArcFace.
        MIN_FRAMES_FOR_IDENTITY = 3  # at 5 fps this is 0.6 seconds of detection
        frame_counts: dict[int, int] = {}
        for ff in frames:
            if ff.face_detected:
                frame_counts[ff.face_index] = frame_counts.get(ff.face_index, 0) + 1

        transient = [tid for tid, cnt in frame_counts.items()
                     if cnt < MIN_FRAMES_FOR_IDENTITY]
        if transient:
            transient_set = set(transient)
            self._best_face_crops = {
                tid: crop for tid, crop in self._best_face_crops.items()
                if tid not in transient_set
            }
            before_count = len(frames)
            frames = [ff for ff in frames if ff.face_index not in transient_set]
            logger.info(
                f"Filtered {len(transient_set)} transient tracks "
                f"(< {MIN_FRAMES_FOR_IDENTITY} frames): {transient} — "
                f"removed {before_count - len(frames)} ghost frames from pipeline"
            )

        # ── Identity-based track deduplication via ArcFace ────────────────────
        # CentroidTracker gives stable IDs within stable layouts but creates
        # duplicate IDs on layout changes (screen share, grid rearrange, spotlight).
        # ArcFace embeddings are layout-invariant — rewrite face_index here so
        # WindowAggregator merges the same physical person into one stream.
        # Results are cached in self._cached_embeddings for reuse in run_analysis
        # Step 5, avoiding a second ArcFace pass.
        self._cached_embeddings: dict[int, tuple[list[float], bytes]] = {}
        if self._best_face_crops and len(self._best_face_crops) > 1:
            try:
                embedder = FaceEmbeddingExtractor.get_instance()
                if embedder.available:
                    emb_results = embedder.extract_from_crops(self._best_face_crops)
                    self._cached_embeddings = emb_results

                    if len(emb_results) > 1:
                        track_embs: dict[int, np.ndarray] = {
                            tid: np.array(emb_list)
                            for tid, (emb_list, _) in emb_results.items()
                        }
                        face_areas = [
                            ff.face_box_area for ff in frames
                            if ff.face_detected and ff.face_box_area > 0.001
                        ]
                        avg_face_h = float(np.mean([a ** 0.5 for a in face_areas])) if face_areas else 0.10
                        duration_s = max((ff.timestamp_ms for ff in frames), default=60000) / 1000.0
                        merge_threshold = self._compute_merge_threshold(duration_s, avg_face_h)
                        logger.info(
                            f"Adaptive merge threshold: {merge_threshold:.3f} "
                            f"(duration={duration_s:.0f}s, avg_face_h={avg_face_h:.3f})"
                        )
                        canonical = self._merge_tracks_by_embedding(
                            track_embs,
                            threshold=merge_threshold,
                            frame_counts=frame_counts,
                        )
                        merges = {k: v for k, v in canonical.items() if k != v}

                        if merges:
                            for ff in frames:
                                if ff.face_index in canonical:
                                    ff.face_index = canonical[ff.face_index]

                            # Extend mapping to transient/unprocessed tracks via
                            # nearest-canonical-centroid assignment.  These tracks
                            # were filtered before ArcFace so they are absent from
                            # `canonical`, but their frames still appear in the
                            # windows that SpeakerFaceMapper processes.  Without
                            # this pass, lip-sync can assign Speaker_N to a
                            # transient track while the canonical track (same
                            # physical person, many more frames) gets Face_N.
                            canonical_ids = set(canonical.values())
                            canon_pts: dict[int, list[tuple[float, float]]] = {
                                cid: [] for cid in canonical_ids
                            }
                            for ff in frames:
                                if ff.face_detected and ff.face_index in canonical_ids:
                                    canon_pts[ff.face_index].append(
                                        (ff.face_centre_x, ff.face_centre_y)
                                    )
                            canonical_centroids: dict[int, tuple[float, float]] = {
                                cid: (
                                    float(np.mean([p[0] for p in pts])),
                                    float(np.mean([p[1] for p in pts])),
                                )
                                for cid, pts in canon_pts.items() if pts
                            }
                            # Max distance for same-person transient reassignment.
                            # In a 7-person grid, adjacent cells are ~0.33 apart.
                            # Same-person transients sit < 0.05 from their canonical;
                            # different people at other grid positions sit > 0.25 away.
                            # 0.15 catches transients while leaving distinct faces alone.
                            _MAX_TRANSIENT_DIST = 0.15
                            if canonical_centroids:
                                for ff in frames:
                                    if ff.face_index not in canonical_ids and ff.face_detected:
                                        cx, cy = ff.face_centre_x, ff.face_centre_y
                                        nearest = min(
                                            canonical_centroids,
                                            key=lambda tid: (
                                                (cx - canonical_centroids[tid][0]) ** 2
                                                + (cy - canonical_centroids[tid][1]) ** 2
                                            ),
                                        )
                                        dist = (
                                            (cx - canonical_centroids[nearest][0]) ** 2
                                            + (cy - canonical_centroids[nearest][1]) ** 2
                                        ) ** 0.5
                                        if dist < _MAX_TRANSIENT_DIST:
                                            ff.face_index = nearest
                                        # else: different person at a different grid
                                        # position — keep original track ID so they
                                        # become their own Face_N canonical.

                            # Keep only canonical crops; prefer the larger crop
                            merged_crops: dict[int, np.ndarray] = {}
                            for tid, crop in self._best_face_crops.items():
                                canon_tid = canonical.get(tid, tid)
                                if (canon_tid not in merged_crops
                                        or crop.size > merged_crops[canon_tid].size):
                                    merged_crops[canon_tid] = crop
                            self._best_face_crops = merged_crops

                            # Rebuild cached embeddings after merge
                            self._cached_embeddings = embedder.extract_from_crops(
                                self._best_face_crops
                            )

                            unique_count = len(set(canonical.values()))
                            logger.info(
                                f"Identity merge: {len(track_embs)} tracks → "
                                f"{unique_count} unique people (merged: {merges})"
                            )
                        else:
                            logger.info(
                                f"Identity merge: {len(emb_results)} tracks — "
                                f"all unique (no merges needed)"
                            )
            except Exception as exc:
                logger.warning(
                    f"Identity-based track merge failed (non-fatal, "
                    f"position-based IDs preserved): {exc}"
                )

        return frames

    def _build_landmarkers(self, mp):
        """Construct MediaPipe landmarker instances for one video session."""
        BaseOptions = mp.tasks.BaseOptions
        RunningMode = mp.tasks.vision.RunningMode

        face_lm = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=self._model_mgr.get_face_landmarker_path()
                ),
                running_mode=RunningMode.VIDEO,
                num_faces=self._num_faces,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.2,
                min_face_presence_confidence=0.2,
                min_tracking_confidence=0.3,
            )
        )

        pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=self._model_mgr.get_pose_landmarker_path()
                ),
                running_mode=RunningMode.VIDEO,
                num_poses=self._num_faces,
                min_pose_detection_confidence=0.2,
                min_pose_presence_confidence=0.2,
                min_tracking_confidence=0.3,
            )
        )

        hand_lm = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=self._model_mgr.get_hand_landmarker_path()
                ),
                running_mode=RunningMode.VIDEO,
                num_hands=self._num_faces * 2,
                min_hand_detection_confidence=0.2,
                min_hand_presence_confidence=0.2,
                min_tracking_confidence=0.3,
            )
        )

        return face_lm, pose_lm, hand_lm

    def _build_tiled_processor(self, mp) -> "TiledFrameProcessor":
        """Create a TiledFrameProcessor for the current face budget."""
        return TiledFrameProcessor.create(mp, self._model_mgr, self._num_faces)

    @staticmethod
    def _compute_frame_behavioral_state(ff: "FrameFeatures") -> None:
        """
        Compute a single behavioral state label for this frame.
        Drives the continuous per-face overlay panel; operates on raw frame data
        (not 2-second window signals) for frame-level granularity.

        State priority (highest wins):
          1. SPEAKING  — jaw open + is_speaking flag
          2. DISAGREEING — head shake OR arms_crossed + tension
          3. AGREEING  — head nod while not speaking
          4. TENSE     — high facial tension without speaking/shaking
          5. DISTRACTED — gaze away
          6. ENGAGED   — active expressions / smile
          7. LISTENING — default
        """
        if not ff.face_detected:
            ff.behavioral_state = ""
            return

        bs = ff.blendshapes or {}

        jaw_open = bs.get("jawOpen", 0.0)
        mouth_movement = (bs.get("mouthLowerDownLeft", 0.0) + bs.get("mouthLowerDownRight", 0.0)) / 2.0
        is_mouth_active = jaw_open > 0.08 or mouth_movement > 0.05

        brow_tension = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
        mouth_press  = (bs.get("mouthPressLeft", 0.0) + bs.get("mouthPressRight", 0.0)) / 2.0
        eye_squint   = (bs.get("eyeSquintLeft", 0.0) + bs.get("eyeSquintRight", 0.0)) / 2.0
        tension_score = brow_tension * 0.4 + mouth_press * 0.35 + eye_squint * 0.25

        smile_score = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0

        if bs:
            active_bs_count  = sum(1 for v in bs.values() if v > 0.10)
            expression_energy = min(active_bs_count / 15.0, 1.0)
        else:
            expression_energy = 0.0

        head_pitch_velocity = getattr(ff, "head_pitch_velocity", 0.0)
        head_yaw_velocity   = getattr(ff, "head_yaw_velocity",   0.0)
        is_nodding  = abs(head_pitch_velocity) > 15.0 if head_pitch_velocity else False
        is_shaking  = abs(head_yaw_velocity)   > 20.0 if head_yaw_velocity   else False

        gaze_magnitude = (ff.gaze_x ** 2 + ff.gaze_y ** 2) ** 0.5 if (ff.gaze_x or ff.gaze_y) else 0.0
        is_looking_away = gaze_magnitude > 0.25

        is_arms_crossed = getattr(ff, "arms_crossed", False)

        # ── State priority ladder ──────────────────────────────────────────────
        # is_speaking lives on WindowFeatures (set post-aggregation), not FrameFeatures.
        # At frame level, jaw activity is the only reliable speaking indicator.
        if is_mouth_active:
            ff.behavioral_state = "speaking"
            if tension_score > 0.25:
                ff.behavioral_state_detail = "speaking_stressed"
            elif smile_score > 0.20:
                ff.behavioral_state_detail = "speaking_positive"
            else:
                ff.behavioral_state_detail = "speaking_neutral"

        elif is_shaking or (is_arms_crossed and tension_score > 0.15):
            ff.behavioral_state = "disagreeing"
            details = []
            if is_shaking:
                details.append("head_shake")
            if is_arms_crossed:
                details.append("arms_crossed")
            if tension_score > 0.20:
                details.append("tense")
            ff.behavioral_state_detail = "+".join(details) if details else "disagreeing"

        elif is_nodding and not is_mouth_active:
            ff.behavioral_state = "agreeing"
            ff.behavioral_state_detail = "nodding+smiling" if smile_score > 0.15 else "nodding"

        elif tension_score > 0.45:
            ff.behavioral_state = "tense"
            details = []
            if brow_tension > 0.20:
                details.append("brow_furrowed")
            if mouth_press > 0.15:
                details.append("lips_pressed")
            if is_arms_crossed:
                details.append("arms_crossed")
            ff.behavioral_state_detail = "+".join(details) if details else "tense"

        elif is_looking_away:
            ff.behavioral_state = "distracted"
            ff.behavioral_state_detail = "gaze_away"

        elif expression_energy > 0.3 or smile_score > 0.15:
            ff.behavioral_state = "engaged"
            ff.behavioral_state_detail = "smiling" if smile_score > 0.15 else "attentive"

        else:
            ff.behavioral_state = "listening"
            ff.behavioral_state_detail = "neutral"

        # ── Stress level ──────────────────────────────────────────────────────
        if tension_score > 0.45:
            ff.stress_level = "high"
        elif tension_score > 0.15:
            ff.stress_level = "moderate"
        else:
            ff.stress_level = "low"

        # ── Engagement level ──────────────────────────────────────────────────
        if expression_energy > 0.3 or is_nodding or smile_score > 0.15:
            ff.engagement_level = "high"
        elif is_looking_away or expression_energy < 0.1:
            ff.engagement_level = "low"
        else:
            ff.engagement_level = "neutral"

    def _process_frame_from_results(
        self,
        bgr: np.ndarray,
        rgb: np.ndarray,
        frame_idx: int,
        timestamp_ms: int,
        face_result,
        pose_result,
        hand_result,
        prev_pose_lm_data: Optional[list],
    ) -> list[FrameFeatures]:
        """
        Build one FrameFeatures per detected face.
        Returns a list (empty-face entry when nothing detected, one per face otherwise).
        Each entry has face_index set so WindowAggregator can group them separately.
        """
        h, w = rgb.shape[:2]

        if not face_result or not face_result.face_landmarks:
            ff = FrameFeatures(timestamp_ms=timestamp_ms, frame_idx=frame_idx)
            if hand_result and hand_result.hand_landmarks:
                ff.hands_detected = len(hand_result.hand_landmarks)
                ff.hand_velocity = self._compute_hand_velocity(hand_result.hand_landmarks)
            return [ff]

        num_faces = len(face_result.face_landmarks)
        num_poses = len(pose_result.pose_landmarks) if pose_result and pose_result.pose_landmarks else 0
        results: list[FrameFeatures] = []

        # Pre-assign each detected hand to its nearest face so that per-face
        # hand proximity checks don't accidentally claim a neighbour's hands.
        hand_assignments: dict[int, list] = {}
        if hand_result and hand_result.hand_landmarks and num_faces > 0:
            hand_assignments = self._assign_hands_to_faces(
                hand_result.hand_landmarks, face_result.face_landmarks
            )

        for fi in range(num_faces):
            ff = FrameFeatures(timestamp_ms=timestamp_ms, frame_idx=frame_idx)
            ff.face_index = fi

            lm = face_result.face_landmarks[fi]
            ff.face_detected = True
            ff.face_count = num_faces
            ff.face_box_area = self._face_box_area(lm)
            ff.face_luminance = self._compute_face_luminance(lm, rgb, h, w)

            xs = [pt.x for pt in lm]
            ys = [pt.y for pt in lm]
            ff.face_centre_x = sum(xs) / len(xs)
            ff.face_centre_y = sum(ys) / len(ys)

            # Blendshapes — indices preserved with possible None entries
            if (face_result.face_blendshapes
                    and fi < len(face_result.face_blendshapes)
                    and face_result.face_blendshapes[fi] is not None):
                ff.blendshapes = {
                    b.category_name: round(float(b.score), 5)
                    for b in face_result.face_blendshapes[fi]
                }

            # Head pose matrix
            if (face_result.facial_transformation_matrixes
                    and fi < len(face_result.facial_transformation_matrixes)
                    and face_result.facial_transformation_matrixes[fi] is not None):
                mat = np.array(
                    face_result.facial_transformation_matrixes[fi].data
                ).reshape(4, 4)
                ff.head_pitch, ff.head_yaw, ff.head_roll = self._matrix_to_euler(mat)

            ff.ear_left, ff.ear_right, ff.ear_avg = self._compute_ear(lm)

            if len(lm) > _RIGHT_IRIS_IDX + 4:
                ff.gaze_x, ff.gaze_y = self._compute_gaze(lm)

            # Pose — proximity-match this face to the nearest detected pose
            if num_poses > 0:
                best_pose_idx = self._match_pose_to_face(lm, pose_result.pose_landmarks)
                if best_pose_idx is not None:
                    plm = pose_result.pose_landmarks[best_pose_idx]
                    ff.body_detected = True
                    ff.shoulder_angle, ff.spine_angle, ff.head_shoulder_dist = (
                        self._compute_body_angles(plm, h)
                    )
                    ff.body_movement = self._compute_body_movement(plm, prev_pose_lm_data)
                    ff._raw_pose_lm = plm  # type: ignore[attr-defined]
                    ff.arms_crossed    = self._detect_crossed_arms(plm)
                    ff.elbow_expansion = self._compute_elbow_expansion(plm)

            # Hands — use only hands pre-assigned to this face
            face_hands = hand_assignments.get(fi, [])
            if face_hands:
                ff.hands_detected = len(face_hands)
                ff.hand_near_face = self._check_hand_near_face_single(face_hands, lm)
                ff.hand_velocity  = self._compute_hand_velocity(face_hands)

                # Face-region touch classification
                if ff.hand_near_face:
                    slot_idx = 0
                    for hand_lm in face_hands:
                        zone = self._classify_hand_touch_zone(hand_lm, lm)
                        if zone:
                            if slot_idx == 0:
                                ff.hand_touch_zone = zone
                            elif slot_idx == 1:
                                ff.hand_touch_zone_r = zone
                            slot_idx += 1

                # Finger steepling
                if len(face_hands) >= 2:
                    ff.finger_steepling = self._detect_finger_steepling(face_hands)
                    ff.hands_clasped    = self._detect_hands_clasped(face_hands)

                # Map GestureRecognizer outputs to FrameFeature bools
                for fh in face_hands:
                    for i, global_h in enumerate(hand_result.hand_landmarks):
                        if (abs(global_h[0].x - fh[0].x) < 0.005 and
                                abs(global_h[0].y - fh[0].y) < 0.005 and
                                i < len(hand_result.hand_gestures)):
                            field = _GESTURE_FIELD_MAP.get(hand_result.hand_gestures[i])
                            if field:
                                setattr(ff, field, True)
                            break

                # Geometric open_palms fallback — fires when GestureRecognizer
                # didn't return Open_Palm but fingers are clearly extended.
                if not ff.open_palms:
                    ff.open_palms = self._detect_open_palms(face_hands)

                # Head supported by hand: chin/cheek touch + head tilted + low expression
                if ff.hand_touch_zone in ("chin", "cheek"):
                    head_tilted = abs(ff.head_roll) > 8.0
                    if ff.blendshapes:
                        active_bs    = sum(1 for v in ff.blendshapes.values() if v > 0.15)
                        low_activity = active_bs < 5
                    else:
                        # Blendshapes absent (face too small for model) — require chin zone
                        # specifically; cheek alone without expression confirmation is too
                        # ambiguous to call head-resting.
                        low_activity = ff.hand_touch_zone == "chin"
                    ff.head_supported_by_hand = head_tilted and low_activity

            # Compute per-frame behavioral state for overlay panels
            self._compute_frame_behavioral_state(ff)

            results.append(ff)

        return results

    # ── Face zone landmark clusters (MediaPipe 478-point model) ───────────────
    _FACE_ZONES: dict[str, list[int]] = {
        "chin":     [152, 200, 428, 175, 396],
        "mouth":    [0, 13, 14, 17, 37, 267, 39, 269],
        "nose":     [1, 2, 4, 5, 6, 94, 168, 197],
        "cheek_l":  [116, 117, 118, 119, 100, 36],
        "cheek_r":  [345, 346, 347, 348, 329, 266],
        "ear_l":    [234, 127, 162],
        "ear_r":    [454, 356, 389],
        "forehead": [10, 338, 297, 332, 251, 21, 54, 103],
        "eye_l":    [33, 159, 158, 133, 153, 145],    # right eye EAR 6-pt (image-left)
        "eye_r":    [362, 380, 374, 263, 386, 385],   # left eye EAR 6-pt (image-right)
    }

    @staticmethod
    def _classify_hand_touch_zone(hand_landmarks, face_landmarks) -> str:
        """
        Determine which face zone a hand is touching/near.
        Uses the closest fingertip to the face centre as the contact point — not the
        wrist-based centroid, which sits at chest level when someone touches their nose.
        Returns zone name or "" when hand is too far from face.
        """
        face_xs = [lm.x for lm in face_landmarks]
        face_ys = [lm.y for lm in face_landmarks]
        face_cx = sum(face_xs) / len(face_xs)
        face_cy = sum(face_ys) / len(face_ys)
        face_w  = max(face_xs) - min(face_xs)
        face_h  = max(face_ys) - min(face_ys)

        # Use the fingertip closest to the face centre as the contact point.
        # Wrist (0) + MCPs (5, 17) centroid sits at chest level for vertical
        # touches (nose, forehead); fingertips are the actual contact points.
        _FINGERTIP_IDX = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky
        hx, hy = hand_landmarks[0].x, hand_landmarks[0].y  # wrist fallback
        best_tip_d = float("inf")
        for tip_idx in _FINGERTIP_IDX:
            if tip_idx < len(hand_landmarks):
                tx, ty = hand_landmarks[tip_idx].x, hand_landmarks[tip_idx].y
                td = ((tx - face_cx) ** 2 + (ty - face_cy) ** 2) ** 0.5
                if td < best_tip_d:
                    best_tip_d = td
                    hx, hy = tx, ty

        dist_to_centre = ((hx - face_cx) ** 2 + (hy - face_cy) ** 2) ** 0.5
        if dist_to_centre > face_h * 1.5:
            return ""

        # Neck: just below chin, tight vertical + narrow horizontal.
        # Reduced vertical from 0.5→0.25 face_h — hands at collar/chest level
        # during gestures must not trigger this.  Narrowed horizontal to exclude
        # hand paths that cross the shoulder/collar area on either side.
        chin_y = face_landmarks[152].y if len(face_landmarks) > 152 else face_cy + face_h * 0.5
        if chin_y < hy < chin_y + face_h * 0.25:
            if min(face_xs) + face_w * 0.1 < hx < max(face_xs) - face_w * 0.1:
                return "neck"

        best_zone = ""
        best_dist = float("inf")
        for zone_name, indices in VideoFeatureExtractor._FACE_ZONES.items():
            valid = [face_landmarks[i] for i in indices if i < len(face_landmarks)]
            if not valid:
                continue
            zx = sum(lm.x for lm in valid) / len(valid)
            zy = sum(lm.y for lm in valid) / len(valid)
            d  = ((hx - zx) ** 2 + (hy - zy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_zone = zone_name

        touch_threshold = max(face_w * 0.45, face_h * 0.35, 0.04)
        if best_dist > touch_threshold:
            return ""

        # Merge left/right variants
        if best_zone.startswith("cheek"):
            best_zone = "cheek"
        if best_zone.startswith("ear"):
            best_zone = "ear"
        if best_zone.startswith("eye"):
            best_zone = "eye"
        return best_zone

    @staticmethod
    def _detect_crossed_arms(pose_landmarks) -> bool:
        """
        Detect crossed arms: both wrists on opposite side of body centre,
        at chest height, within shoulder width of body midline.
        """
        if len(pose_landmarks) < 17:
            return False
        for idx in [11, 12, 15, 16]:
            if pose_landmarks[idx].visibility < 0.5:
                return False

        ls = pose_landmarks[11]
        rs = pose_landmarks[12]
        lw = pose_landmarks[15]
        rw = pose_landmarks[16]

        centre_x      = (ls.x + rs.x) / 2
        shoulder_w    = abs(rs.x - ls.x)
        chest_y       = (ls.y + rs.y) / 2
        hip_y         = (
            (pose_landmarks[23].y + pose_landmarks[24].y) / 2
            if len(pose_landmarks) > 24 else chest_y + shoulder_w
        )

        if len(pose_landmarks) <= _POSE_RIGHT_ELBOW:
            return (
                abs(lw.x - centre_x) < shoulder_w * 0.6 and
                abs(rw.x - centre_x) < shoulder_w * 0.6 and
                lw.x > centre_x and rw.x < centre_x and
                chest_y < lw.y < hip_y and chest_y < rw.y < hip_y
            )

        le = pose_landmarks[_POSE_LEFT_ELBOW]
        re = pose_landmarks[_POSE_RIGHT_ELBOW]
        # Elbows must be visible and at chest height — rules out wrists crossing
        # during typing or hands-in-lap poses which also satisfy the wrist check.
        elbows_visible = (
            getattr(le, "visibility", 1.0) >= 0.4 and
            getattr(re, "visibility", 1.0) >= 0.4
        )
        elbows_at_chest = chest_y < le.y < hip_y and chest_y < re.y < hip_y

        wrists_crossed = (
            abs(lw.x - centre_x) < shoulder_w * 0.6 and
            abs(rw.x - centre_x) < shoulder_w * 0.6 and
            lw.x > centre_x and rw.x < centre_x and
            chest_y < lw.y < hip_y and chest_y < rw.y < hip_y
        )

        if not elbows_visible:
            return wrists_crossed
        return wrists_crossed and elbows_at_chest

    @staticmethod
    def _compute_elbow_expansion(pose_landmarks) -> float:
        """
        Elbow expansion = (elbow_width - shoulder_width) / shoulder_width.
        Positive → elbows wider than shoulders (expansive/dominant posture).
        Negative → elbows tucked in (contracted/defensive posture).
        Returns 0.0 when landmarks are not visible.
        """
        needed = [_POSE_LEFT_SHOULDER, _POSE_RIGHT_SHOULDER,
                  _POSE_LEFT_ELBOW,    _POSE_RIGHT_ELBOW]
        if len(pose_landmarks) <= max(needed):
            return 0.0
        for idx in needed:
            if getattr(pose_landmarks[idx], "visibility", 1.0) < 0.4:
                return 0.0
        shoulder_w = abs(
            pose_landmarks[_POSE_RIGHT_SHOULDER].x -
            pose_landmarks[_POSE_LEFT_SHOULDER].x
        )
        if shoulder_w < 0.05:
            return 0.0
        elbow_w = abs(
            pose_landmarks[_POSE_RIGHT_ELBOW].x -
            pose_landmarks[_POSE_LEFT_ELBOW].x
        )
        return float((elbow_w - shoulder_w) / shoulder_w)

    @staticmethod
    def _detect_finger_steepling(hand_landmarks_list: list) -> bool:
        """
        Detect finger steepling: two hands with 3+ fingertip pairs within
        0.03 normalised distance of each other.
        Fingertip indices: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky).
        """
        if len(hand_landmarks_list) < 2:
            return False
        a, b = hand_landmarks_list[0], hand_landmarks_list[1]
        close = sum(
            1 for idx in [4, 8, 12, 16, 20]
            if ((a[idx].x - b[idx].x) ** 2 + (a[idx].y - b[idx].y) ** 2) ** 0.5 < 0.03
        )
        return close >= 3

    @staticmethod
    def _detect_hands_clasped(hand_landmarks_list: list) -> bool:
        """
        Detect clasped/interlaced hands: wrists and MCPs close together but
        fingertips NOT close (curled in, not spread like steepling).
        Pease 2004: clasped hands = self-restraint / neutral waiting.
        """
        if len(hand_landmarks_list) < 2:
            return False
        a, b = hand_landmarks_list[0], hand_landmarks_list[1]

        wrist_dist = ((a[0].x - b[0].x) ** 2 + (a[0].y - b[0].y) ** 2) ** 0.5
        if wrist_dist > 0.12:
            return False

        close_mcps = sum(
            1 for idx in [5, 9, 13, 17]
            if ((a[idx].x - b[idx].x) ** 2 + (a[idx].y - b[idx].y) ** 2) ** 0.5 < 0.06
        )
        if close_mcps < 2:
            return False

        # Steepling has 3+ close tips; clasped has tips curled away
        close_tips = sum(
            1 for idx in [4, 8, 12, 16, 20]
            if ((a[idx].x - b[idx].x) ** 2 + (a[idx].y - b[idx].y) ** 2) ** 0.5 < 0.03
        )
        return close_tips < 3

    @staticmethod
    def _detect_open_palms(hand_landmarks_list: list) -> bool:
        """
        Geometric fallback for open-palm detection.
        Returns True when at least one hand has 3+ fingers extended
        (fingertip y < PIP y in normalised image coords, i.e. tip is higher).
        Fires when GestureRecognizer didn't classify the hand as Open_Palm.
        """
        for hand_lm in hand_landmarks_list:
            tips = [8, 12, 16, 20]   # index, middle, ring, pinky tips
            pips = [6, 10, 14, 18]   # corresponding PIP joints
            extended = sum(
                1 for tip, pip in zip(tips, pips)
                if tip < len(hand_lm) and pip < len(hand_lm)
                and hand_lm[tip].y < hand_lm[pip].y
            )
            if extended >= 3:
                return True
        return False

    @staticmethod
    def _assign_hands_to_faces(hand_landmarks_list: list, face_landmarks_list: list) -> dict:
        """
        Assign each detected hand to its nearest face.
        Uses the minimum distance from ANY hand landmark to each face centre so that
        a finger touching a face registers even when the wrist is at chest level.
        A hand belongs to exactly one face; multiple hands can belong to the same face.
        Returns {face_index: [hand_landmarks, ...]}.
        """
        assignments: dict[int, list] = {}
        face_centres = [
            (sum(lm.x for lm in face_lm) / len(face_lm),
             sum(lm.y for lm in face_lm) / len(face_lm))
            for face_lm in face_landmarks_list
        ]
        for hand_lm in hand_landmarks_list:
            best_fi   = 0
            best_dist = float("inf")
            for fi, (fx, fy) in enumerate(face_centres):
                for pt in hand_lm:
                    d = ((pt.x - fx) ** 2 + (pt.y - fy) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_fi   = fi
            assignments.setdefault(best_fi, []).append(hand_lm)
        return assignments

    # ── Feature computation helpers ────────────────────────────────────────────

    @staticmethod
    def _face_box_area(landmarks) -> float:
        """Normalised bounding box area of the face landmark cloud."""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    @staticmethod
    def _matrix_to_euler(mat: np.ndarray) -> tuple[float, float, float]:
        """
        Extract pitch, yaw, roll (degrees) from a 4×4 homogeneous rotation matrix.
        Handles gimbal-lock singularity when |r[2,0]| ≈ 1.
        """
        r = mat[:3, :3]
        r20 = float(np.clip(r[2, 0], -1.0, 1.0))
        pitch = float(np.degrees(np.arcsin(-r20)))
        if abs(r20) < 0.99999:
            yaw  = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
            roll = float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))
        else:
            yaw  = float(np.degrees(np.arctan2(-r[0, 1], r[1, 1])))
            roll = 0.0
        return pitch, yaw, roll

    @staticmethod
    def _compute_ear(landmarks) -> tuple[float, float, float]:
        """
        Eye Aspect Ratio for both eyes. EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        Returns (ear_left, ear_right, ear_avg).
        """
        def ear_for(idx: tuple[int, ...]) -> float:
            try:
                pts = np.array([[landmarks[i].x, landmarks[i].y] for i in idx])
                v1 = np.linalg.norm(pts[1] - pts[5])
                v2 = np.linalg.norm(pts[2] - pts[4])
                h  = np.linalg.norm(pts[0] - pts[3])
                return float((v1 + v2) / (2.0 * h + 1e-6))
            except (IndexError, ZeroDivisionError):
                return 0.0

        el = ear_for(_LEFT_EYE_EAR)
        er = ear_for(_RIGHT_EYE_EAR)
        return el, er, (el + er) / 2.0

    @staticmethod
    def _compute_gaze(landmarks) -> tuple[float, float]:
        """
        Estimate gaze direction from iris offset relative to eye centre.
        Returns (gaze_x, gaze_y) normalised by eye horizontal width.
        Positive gaze_x = looking right; positive gaze_y = looking down.
        """
        try:
            # Left iris centre
            l_iris = np.array([landmarks[_LEFT_IRIS_IDX].x, landmarks[_LEFT_IRIS_IDX].y])
            l_outer = np.array([landmarks[_LEFT_EYE_OUTER].x, landmarks[_LEFT_EYE_OUTER].y])
            l_inner = np.array([landmarks[_LEFT_EYE_INNER].x, landmarks[_LEFT_EYE_INNER].y])
            l_eye_centre = (l_outer + l_inner) / 2.0
            l_eye_width  = np.linalg.norm(l_outer - l_inner) + 1e-6

            # Right iris centre
            r_iris  = np.array([landmarks[_RIGHT_IRIS_IDX].x, landmarks[_RIGHT_IRIS_IDX].y])
            r_outer = np.array([landmarks[_RIGHT_EYE_OUTER].x, landmarks[_RIGHT_EYE_OUTER].y])
            r_inner = np.array([landmarks[_RIGHT_EYE_INNER].x, landmarks[_RIGHT_EYE_INNER].y])
            r_eye_centre = (r_outer + r_inner) / 2.0
            r_eye_width  = np.linalg.norm(r_outer - r_inner) + 1e-6

            l_offset = (l_iris - l_eye_centre) / l_eye_width
            r_offset = (r_iris - r_eye_centre) / r_eye_width
            avg = (l_offset + r_offset) / 2.0
            return float(avg[0]), float(avg[1])
        except (IndexError, Exception):
            return 0.0, 0.0

    @staticmethod
    def _compute_body_angles(
        landmarks, frame_height: int
    ) -> tuple[float, float, float]:
        """
        Returns (shoulder_angle°, spine_angle°, head_shoulder_dist_normalised).
        shoulder_angle: tilt of the shoulder line from horizontal.
        spine_angle:    tilt of the torso from vertical (+ = leaning forward).
        head_shoulder_dist: nose-to-shoulder-midpoint / frame_height.
        """
        try:
            ls  = np.array([landmarks[_POSE_LEFT_SHOULDER].x,  landmarks[_POSE_LEFT_SHOULDER].y])
            rs  = np.array([landmarks[_POSE_RIGHT_SHOULDER].x, landmarks[_POSE_RIGHT_SHOULDER].y])
            lh  = np.array([landmarks[_POSE_LEFT_HIP].x,       landmarks[_POSE_LEFT_HIP].y])
            rh  = np.array([landmarks[_POSE_RIGHT_HIP].x,      landmarks[_POSE_RIGHT_HIP].y])
            nos = np.array([landmarks[_POSE_NOSE].x,            landmarks[_POSE_NOSE].y])

            sh_mid  = (ls + rs) / 2.0
            hip_mid = (lh + rh) / 2.0

            sh_vec = rs - ls
            shoulder_angle = float(np.degrees(np.arctan2(sh_vec[1], sh_vec[0])))

            torso_vec = hip_mid - sh_mid
            # spine_angle relative to straight-down (0,1) vector
            spine_angle = float(np.degrees(
                np.arctan2(torso_vec[0], torso_vec[1] + 1e-6)
            ))

            head_shoulder_dist = float(np.linalg.norm(nos - sh_mid))

            return shoulder_angle, spine_angle, head_shoulder_dist
        except (IndexError, Exception):
            return 0.0, 0.0, 0.0

    @staticmethod
    def _compute_body_movement(
        landmarks, prev_landmarks: Optional[list]
    ) -> float:
        """
        Sum of 2D Euclidean distances for all pose landmarks vs previous frame.
        Returns 0.0 when no previous frame is available.
        """
        if prev_landmarks is None:
            return 0.0
        try:
            total = 0.0
            n = min(len(landmarks), len(prev_landmarks))
            for i in range(n):
                dx = landmarks[i].x - prev_landmarks[i].x
                dy = landmarks[i].y - prev_landmarks[i].y
                total += (dx * dx + dy * dy) ** 0.5
            return float(total)
        except Exception:
            return 0.0

    @staticmethod
    def _check_hand_near_face(
        hand_lm_list, ff: FrameFeatures, w: int, h: int
    ) -> bool:
        """
        True if any hand landmark falls inside the face bounding box
        extended by 20% (self-touch proxy).
        """
        if not ff.face_detected or not ff.blendshapes:
            return False
        try:
            # Approximate face box centre from nose bridge (landmark 1)
            face_area = ff.face_box_area
            face_half = face_area ** 0.5 * 0.5 * 1.2   # 20% margin
            # Use a simple circle check around gaze origin (rough head centre)
            for hand_lm in hand_lm_list:
                for lm in hand_lm:
                    # Check against head-size-estimated region around (0.5, 0.5)
                    # Proper check uses face landmark bounding box
                    dx = lm.x - 0.5
                    dy = lm.y - 0.3   # face is typically upper-third of frame
                    if (dx * dx + dy * dy) ** 0.5 < face_half:
                        return True
        except Exception:
            pass
        return False

    @staticmethod
    def _match_pose_to_face(
        face_landmarks, all_pose_landmarks
    ) -> Optional[int]:
        """
        Proximity-match a face to the nearest independently-detected pose.
        Returns the index into all_pose_landmarks, or None if no pose is close enough.
        MediaPipe pose nose (index 0) should align with the face centre in the same tile.
        """
        fx = sum(lm.x for lm in face_landmarks) / len(face_landmarks)
        fy = sum(lm.y for lm in face_landmarks) / len(face_landmarks)

        best_idx: Optional[int] = None
        best_dist = float("inf")
        MATCH_THRESHOLD = 0.15  # normalised; face & pose cropped from same tile → close

        for pi, pose_lm in enumerate(all_pose_landmarks):
            if len(pose_lm) > 0:
                dist = ((fx - pose_lm[0].x) ** 2 + (fy - pose_lm[0].y) ** 2) ** 0.5
                if dist < best_dist and dist < MATCH_THRESHOLD:
                    best_dist = dist
                    best_idx = pi

        return best_idx

    @staticmethod
    def _count_hands_near_face(hand_landmarks_list, face_landmarks) -> int:
        """Count hands whose wrist (landmark 0) is within 0.2 normalised units of this face."""
        fx = sum(lm.x for lm in face_landmarks) / len(face_landmarks)
        fy = sum(lm.y for lm in face_landmarks) / len(face_landmarks)
        count = 0
        for hand_lm in hand_landmarks_list:
            if hand_lm:
                dist = ((fx - hand_lm[0].x) ** 2 + (fy - hand_lm[0].y) ** 2) ** 0.5
                if dist < 0.2:
                    count += 1
        return count

    @staticmethod
    def _check_hand_near_face_single(hand_landmarks_list, face_landmarks) -> bool:
        """
        True if any hand landmark falls within the vertically-extended face bbox.

        Uses face_h-based vertical padding so hands at the forehead (above the
        face top) and chin/neck (below the face bottom) are captured.  Horizontal
        padding is kept modest to avoid stealing hands from adjacent faces in a
        multi-person grid.
        """
        fxs = [lm.x for lm in face_landmarks]
        fys = [lm.y for lm in face_landmarks]
        fx1, fx2 = min(fxs), max(fxs)
        fy1, fy2 = min(fys), max(fys)
        face_w = fx2 - fx1
        face_h = fy2 - fy1
        # Vertical: 70% of face height above/below — catches forehead + chin touches
        v_pad = face_h * 0.70
        # Horizontal: 25% of face width — enough for cheek touches, avoids neighbours
        h_pad = face_w * 0.25
        fx1 -= h_pad
        fx2 += h_pad
        fy1 -= v_pad
        fy2 += v_pad
        for hand_lm in hand_landmarks_list:
            for pt in hand_lm:
                if fx1 <= pt.x <= fx2 and fy1 <= pt.y <= fy2:
                    return True
        return False

    @staticmethod
    def _compute_face_luminance(landmarks, rgb: np.ndarray, h: int, w: int) -> float:
        """
        Mean brightness of the face crop normalised to [0, 1].
        Used as skin-tone confidence modifier (Buolamwini & Gebru 2018 bias mitigation).
        Lower luminance → darker skin → lower blendshape accuracy → reduce confidence.
        """
        try:
            xs = [pt.x * w for pt in landmarks]
            ys = [pt.y * h for pt in landmarks]
            x1 = max(0, int(min(xs)))
            x2 = min(w, int(max(xs)))
            y1 = max(0, int(min(ys)))
            y2 = min(h, int(max(ys)))
            if x2 > x1 and y2 > y1:
                return float(rgb[y1:y2, x1:x2].mean() / 255.0)
        except Exception:
            pass
        return 0.5

    @staticmethod
    def _compute_hand_velocity(hand_lm_list) -> float:
        """
        Mean magnitude of hand landmark positions (proxy for gesture extent).
        Proper velocity needs previous frame; this returns extent from centre.
        """
        try:
            if not hand_lm_list:
                return 0.0
            pts = np.array([
                [lm.x, lm.y]
                for hand_lm in hand_lm_list
                for lm in hand_lm
            ], dtype=np.float32)
            centre = pts.mean(axis=0)
            return float(np.mean(np.linalg.norm(pts - centre, axis=1)))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_merge_threshold(duration_s: float, avg_face_height_ratio: float) -> float:
        """
        Adaptive ArcFace cosine-similarity threshold for track deduplication.

        avg_face_height_ratio = mean sqrt(face_box_area) across detected frames.
          > 0.20  → grid / video-call  (large faces filling boxes)
          ≤ 0.20  → in-person room     (small faces around a table)

        Grid calls:  different-person ceiling ≈ 0.25–0.30, floor = 0.42.
        In-person:   different-person ceiling ≈ 0.40–0.42, floor = 0.48.
        Longer videos accumulate more track fragments so same-person scores drop;
        decay reduces the threshold to catch those lower-scoring pairs.
        """
        duration_min = duration_s / 60.0
        if avg_face_height_ratio > 0.20:
            base  = min(0.65, max(0.50, 0.35 + avg_face_height_ratio * 1.5))
            decay = duration_min * 0.012
            floor = 0.42
        else:
            base  = 0.58
            decay = duration_min * 0.005
            floor = 0.48
        return round(max(floor, base - decay), 3)

    @staticmethod
    def _merge_tracks_by_embedding(
        track_embeddings: dict[int, "np.ndarray"],
        threshold: float = 0.65,
        frame_counts: dict[int, int] | None = None,
    ) -> dict[int, int]:
        """
        Merge CentroidTracker track_ids that belong to the same physical person.

        Embeddings are L2-normalised in extract_from_crops so np.dot() here
        equals cosine similarity in [-1, 1].  Threshold is computed adaptively
        by _compute_merge_threshold based on video duration and face size.

        Tracks are processed dominant-first (most frames → canonical anchor),
        so fragment tracks are always compared against the highest-quality
        embedding rather than an arbitrary earlier track.

        Returns {track_id: canonical_track_id} for ALL input track_ids.
        Tracks that match nothing keep their own ID as canonical.
        """
        import logging as _logging
        _log = _logging.getLogger("nexus.video.features")

        canonical: dict[int, int] = {}
        canon_embs: dict[int, "np.ndarray"] = {}

        counts = frame_counts or {}
        # Dominant tracks (most frames) become canonical anchors first so that
        # fragment tracks are always merged into the best-quality embedding.
        for tid in sorted(track_embeddings.keys(), key=lambda t: counts.get(t, 0), reverse=True):
            emb = track_embeddings[tid]
            matched_canon: int | None = None
            best_sim = 0.0

            # Log all scores so we can see near-threshold matches
            scores = {
                canon_tid: float(np.dot(emb, canon_emb))
                for canon_tid, canon_emb in canon_embs.items()
            }
            if scores:
                top = max(scores.values())
                top_tid = max(scores, key=lambda k: scores[k])
                _log.info(
                    f"Track {tid:>2} vs canonicals: best={top:.3f} (vs track {top_tid}) "
                    f"— {'MERGE' if top > threshold else 'keep separate'}"
                )

            for canon_tid, sim in scores.items():
                if sim > threshold and sim > best_sim:
                    best_sim = sim
                    matched_canon = canon_tid

            if matched_canon is not None:
                canonical[tid] = matched_canon
            else:
                canonical[tid] = tid
                canon_embs[tid] = emb

        return canonical


# ══════════════════════════════════════════════════════════════════════════════
# SpeakerFaceMapper  — assigns windows to speakers
# ══════════════════════════════════════════════════════════════════════════════

class SpeakerFaceMapper:
    """
    Maps detected faces to speakers using lip-sync correlation.

    Strategy:
      1. For each (face_index, speaker) pair compute a correlation score:
             avg_lip_activity during speaker's segments
           - avg_lip_activity during speaker's silence
         A face that moves its mouth specifically when a speaker talks → high score.
         A face that moves equally in speech and silence (chewing, smiling) → ~0.
      2. Greedy best-match-first assignment prevents duplicate face→speaker mapping.
      3. Falls back to pure time-overlap when lip data is unavailable (single face,
         audio-only, or no blendshapes extracted).

    Research: Haider 2021 — jawOpen is the strongest single predictor of active
    speech in MediaPipe blendshapes (r=0.82 with ground-truth voice activity).
    """

    def assign(
        self,
        windows: list[WindowFeatures],
        diar_segments: list[dict],
        lip_activity_map: Optional[dict[int, list[tuple[int, float]]]] = None,
    ) -> tuple[dict[str, list[WindowFeatures]], dict[str, float]]:
        """
        Args:
            windows:          WindowFeatures from VideoFeatureExtractor.extract_all().
            diar_segments:    [{speaker, start_ms, end_ms}, ...] from voice agent.
            lip_activity_map: {face_index: [(timestamp_ms, lip_score), ...]}
                              from VideoFeatureExtractor.build_lip_activity_map().
                              If None or single face, falls back to time-overlap.

        Returns:
            (windows_by_speaker, lip_sync_scores) where lip_sync_scores maps
            speaker_label → correlation score (0.0 when time-overlap fallback used).
        """
        result: dict[str, list[WindowFeatures]] = defaultdict(list)

        if not windows:
            return dict(result), {}

        speakers = sorted(set(seg.get("speaker", "Speaker_0") for seg in diar_segments))
        if not speakers:
            speakers = ["Speaker_0"]

        face_indices = sorted(set(getattr(wf, "face_index", 0) for wf in windows))

        use_lip_sync = (
            lip_activity_map is not None
            and len(face_indices) > 1
            and len(speakers) > 1
        )

        if use_lip_sync:
            face_to_speaker, assignment_scores = self._lip_sync_assignment(
                face_indices, speakers, diar_segments, lip_activity_map  # type: ignore[arg-type]
            )
            method = "lip_sync"
        else:
            face_to_speaker, assignment_scores = self._time_overlap_assignment(
                face_indices, speakers, windows, diar_segments
            )
            method = "time_overlap"

        for wf in windows:
            face_idx = getattr(wf, "face_index", 0)
            speaker = face_to_speaker.get(face_idx, speakers[0])
            wf.speaker_id = speaker
            wf.is_speaking = self._is_speaking_in_window(
                wf.window_start_ms, wf.window_end_ms, speaker, diar_segments
            )
            result[speaker].append(wf)

        # lip_sync_scores: only meaningful for confirmed speaker→face assignments
        lip_sync_scores: dict[str, float] = {}
        if method == "lip_sync":
            for fi, spk in face_to_speaker.items():
                if not spk.startswith("Face_"):
                    lip_sync_scores[spk] = round(assignment_scores.get(fi, 0.0), 4)

        logger.info(
            "SpeakerFaceMapper: %d face(s) → %s (method=%s)",
            len(face_indices),
            {fi: spk for fi, spk in sorted(face_to_speaker.items())},
            method,
        )
        return dict(result), lip_sync_scores

    def _lip_sync_assignment(
        self,
        face_indices: list[int],
        speakers: list[str],
        diar_segments: list[dict],
        lip_activity_map: dict[int, list[tuple[int, float]]],
    ) -> tuple[dict[int, str], dict[int, float]]:
        """
        Correlate each face's jawOpen composite with each speaker's active intervals.

        For each (face, speaker) pair:
            speaking_score = mean lip_activity over frames where speaker is active
            silence_score  = mean lip_activity over frames where speaker is NOT active
            correlation    = speaking_score − silence_score

        Positive  → face moves more when this speaker talks   (good match)
        Near zero → face moves equally regardless             (listener / background)
        Negative  → face moves less when this speaker talks   (wrong face)
        """
        speaker_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for seg in diar_segments:
            spk = seg.get("speaker", "Speaker_0")
            speaker_intervals[spk].append((seg.get("start_ms", 0), seg.get("end_ms", 0)))

        scores: dict[tuple[int, str], float] = {}

        for face_idx in face_indices:
            lip_data = lip_activity_map.get(face_idx, [])
            if not lip_data:
                continue

            for speaker in speakers:
                intervals = speaker_intervals.get(speaker, [])
                speaking_sum = 0.0
                silence_sum  = 0.0
                speaking_n   = 0
                silence_n    = 0

                for ts_ms, lip_score in lip_data:
                    if self._in_intervals(ts_ms, intervals):
                        speaking_sum += lip_score
                        speaking_n   += 1
                    else:
                        silence_sum += lip_score
                        silence_n   += 1

                avg_speaking = speaking_sum / max(speaking_n, 1)
                avg_silence  = silence_sum  / max(silence_n,  1)
                scores[(face_idx, speaker)] = avg_speaking - avg_silence

                logger.debug(
                    "lip_sync score  face=%d × %s: %.4f  (spk_avg=%.4f  sil_avg=%.4f)",
                    face_idx, speaker,
                    avg_speaking - avg_silence, avg_speaking, avg_silence,
                )

        return self._greedy_assign(face_indices, speakers, scores)

    def _time_overlap_assignment(
        self,
        face_indices: list[int],
        speakers: list[str],
        windows: list[WindowFeatures],
        diar_segments: list[dict],
    ) -> tuple[dict[int, str], dict[int, float]]:
        """Fallback: original time-overlap approach used when lip data is unavailable."""
        by_face: dict[int, list[WindowFeatures]] = defaultdict(list)
        for wf in windows:
            by_face[getattr(wf, "face_index", 0)].append(wf)

        scores: dict[tuple[int, str], float] = {}
        for face_idx in face_indices:
            for speaker in speakers:
                total_ov = 0.0
                for wf in by_face.get(face_idx, []):
                    for seg in diar_segments:
                        if seg.get("speaker") != speaker:
                            continue
                        total_ov += max(
                            0.0,
                            min(wf.window_end_ms, seg.get("end_ms", 0))
                            - max(wf.window_start_ms, seg.get("start_ms", 0)),
                        )
                scores[(face_idx, speaker)] = total_ov

        return self._greedy_assign(face_indices, speakers, scores)

    @staticmethod
    def _greedy_assign(
        face_indices: list[int],
        speakers: list[str],
        scores: dict[tuple[int, str], float],
    ) -> tuple[dict[int, str], dict[int, float]]:
        """
        Greedy best-match-first assignment. Prevents duplicate face→speaker mappings.

        Sorts all (face, speaker) pairs by score descending, greedily picks the
        highest unassigned pair, skips any pair where face or speaker already taken.
        Leftover faces (more faces than speakers) receive "Face_{index}" labels.

        Returns (mapping, assignment_scores) — assignment_scores maps face_index to
        the winning correlation/overlap score for auditing and UI display.
        """
        candidates = sorted(
            ((scores.get((fi, spk), 0.0), fi, spk) for fi in face_indices for spk in speakers),
            key=lambda x: x[0],
            reverse=True,
        )

        assigned_faces: set[int] = set()
        assigned_speakers: set[str] = set()
        mapping: dict[int, str] = {}
        assignment_scores: dict[int, float] = {}

        for score, face_idx, speaker in candidates:
            if face_idx in assigned_faces or speaker in assigned_speakers:
                continue
            mapping[face_idx] = speaker
            assignment_scores[face_idx] = score
            assigned_faces.add(face_idx)
            assigned_speakers.add(speaker)

        for fi in face_indices:
            if fi not in mapping:
                mapping[fi] = f"Face_{fi}"
                assignment_scores[fi] = 0.0

        return mapping, assignment_scores

    @staticmethod
    def _in_intervals(ts_ms: int, intervals: list[tuple[int, int]]) -> bool:
        """Return True if ts_ms falls within any (start, end) interval."""
        for start, end in intervals:
            if start <= ts_ms <= end:
                return True
        return False

    @staticmethod
    def _is_speaking_in_window(
        win_start: int,
        win_end: int,
        speaker: str,
        diar_segments: list[dict],
    ) -> bool:
        """
        True when the assigned speaker's diarization covers >50% of this window.
        Used by gaze rule G-1 to split speaking vs. listening thresholds.
        """
        window_dur = max(win_end - win_start, 1)
        overlap = 0.0
        for seg in diar_segments:
            if seg.get("speaker") != speaker:
                continue
            overlap += max(
                0.0,
                min(win_end, seg.get("end_ms", 0)) - max(win_start, seg.get("start_ms", 0)),
            )
        return (overlap / window_dur) > 0.5
