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
      → list[WindowFeatures]                       (face-detected, unassigned)
  SpeakerFaceMapper.assign(windows, diar_segments)
      → dict[str, list[WindowFeatures]]            (per-speaker)

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
TARGET_FPS: int = 10
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
_POSE_LEFT_HIP       = 23
_POSE_RIGHT_HIP      = 24

# ─── MediaPipe model download URLs ───────────────────────────────────────────
_MODEL_URLS: dict[str, str] = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
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

    # ── Hands ─────────────────────────────────────────────────────────────────
    hands_detected: int = 0            # 0, 1, or 2
    hand_near_face: bool = False       # bounding-box overlap with face region
    hand_velocity: float = 0.0        # mean wrist landmark velocity (gesture proxy)


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
    # Note: in the current single-camera architecture, SpeakerFaceMapper assigns each
    # window to the dominant speaker (whoever is talking), so is_speaking is always True.
    # The field exists for future multi-camera / full-session-face-tracking support.
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

    # ── Hands ─────────────────────────────────────────────────────────────────
    hands_detected_rate:   float = 0.0
    hand_near_face_pct:    float = 0.0
    gesture_velocity_mean: float = 0.0
    gesture_velocity_max:  float = 0.0

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
        return self._get("pose_landmarker_full.task")

    def get_hand_landmarker_path(self) -> str:
        return self._get("hand_landmarker.task")

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

        # ── Hands ──────────────────────────────────────────────────────────────
        wf.hands_detected_rate   = float(np.mean([f.hands_detected > 0 for f in frames]))
        wf.hand_near_face_pct    = float(np.mean([f.hand_near_face    for f in frames]))
        hand_vels = [f.hand_velocity for f in frames if f.hand_velocity > 0]
        if hand_vels:
            wf.gesture_velocity_mean = float(np.mean(hand_vels))
            wf.gesture_velocity_max  = float(np.max(hand_vels))

        return wf


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
        ff: Optional["FrameFeatures"],
        active_signals: Optional[list] = None,
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
        if ff and ff.face_detected and face_result and getattr(face_result, "face_landmarks", None):
            self._draw_gaze_arrow(out, face_result, ff, h, w)
        if active_signals:
            self._draw_labels(out, active_signals, h)
        return out

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
        color_bone = (255, 100, 30)
        color_dot  = (200, 200, 255)
        for pose_lm in pose_result.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_lm]
            n = len(pts)
            for a, b in self._POSE_SEGS:
                if a < n and b < n:
                    cv2.line(bgr, pts[a], pts[b], color_bone, 2, cv2.LINE_AA)
            for pt in pts[:33]:
                cv2.circle(bgr, pt, 3, color_dot, -1)

    def _draw_hands(self, bgr: np.ndarray, hand_result, h: int, w: int) -> None:
        import cv2
        color_seg = (0, 200, 220)
        color_dot = (0, 255, 255)
        for hand_lm in hand_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            n = len(pts)
            for a, b in self._HAND_SEGS:
                if a < n and b < n:
                    cv2.line(bgr, pts[a], pts[b], color_seg, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(bgr, pt, 4, color_dot, -1)

    def _draw_gaze_arrow(
        self, bgr: np.ndarray, face_result, ff: "FrameFeatures", h: int, w: int
    ) -> None:
        """Cyan arrow from nose bridge showing gaze direction offset."""
        import cv2
        try:
            lm = face_result.face_landmarks[0]
            nose_x = int(lm[1].x * w)
            nose_y = int(lm[1].y * h)
            scale = min(w, h) * 0.12
            end_x = int(nose_x + ff.gaze_x * scale)
            end_y = int(nose_y + ff.gaze_y * scale)
            cv2.arrowedLine(
                bgr, (nose_x, nose_y), (end_x, end_y),
                (255, 230, 0), 2, tipLength=0.35, line_type=cv2.LINE_AA,
            )
        except Exception:
            pass

    def _draw_labels(self, bgr: np.ndarray, signals: list, h: int) -> None:
        """Up to 5 signal labels at bottom-left, sorted by confidence desc."""
        import cv2
        top = sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True)[:5]
        y = h - 12
        for sig in top:
            label = f"{sig.get('signal_type', '')[:18]}  {sig.get('confidence', 0.0):.2f}"
            cv2.putText(
                bgr, label, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
            )
            y -= 16

    def burn_signal_labels(self, video_path: str, signals: list) -> None:
        """
        Second-pass rewrite: adds per-frame signal text to an already-rendered
        overlay video.  Replaces the file in-place via temp rename.
        """
        import cv2
        import shutil

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"burn_signal_labels: cannot open {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp = video_path + ".burn.mp4"

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

        try:
            shutil.move(tmp, video_path)
        except Exception as exc:
            logger.warning(f"burn_signal_labels rename failed: {exc}")
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass


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

    # Max participants per meeting type — drives num_faces/poses/hands budget
    _MEETING_FACE_BUDGET: dict[str, int] = {
        "sales_call":     4,   # seller + buyer, maybe 1-2 more
        "client_meeting": 6,   # client team can be larger
        "interview":      3,   # interviewer(s) + candidate
        "internal":       8,   # team meetings, larger groups
        "podcast":        4,   # host + guests
    }
    _DEFAULT_FACE_BUDGET = 6

    @classmethod
    def faces_for_meeting(cls, meeting_type: str) -> int:
        return cls._MEETING_FACE_BUDGET.get(meeting_type, cls._DEFAULT_FACE_BUDGET)

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

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_all(
        self,
        video_path: str,
        overlay_output_path: Optional[str] = None,
        meeting_type: str = "",
    ) -> list[WindowFeatures]:
        """
        Process video at target_fps and return aggregated window features.

        Args:
            video_path:           Path to mp4/webm/avi file.
            overlay_output_path:  If set, write a landmark-annotated mp4 here.
            meeting_type:         Used to set the face/pose/hand detection budget.

        Returns:
            List of WindowFeatures, one per 2-second window.
            Empty list if video cannot be read or MediaPipe unavailable.
        """
        if meeting_type:
            self._num_faces = self.faces_for_meeting(meeting_type)
            logger.info(f"Face budget for '{meeting_type}': {self._num_faces}")
        if not self._check_mediapipe():
            logger.error("MediaPipe not available — returning empty features.")
            return []

        frames = self._extract_frames(video_path, overlay_output_path=overlay_output_path)
        return self._aggregator.aggregate(frames)

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

        face_lm, pose_lm, hand_lm = self._build_landmarkers(mp)
        renderer = OverlayRenderer() if overlay_output_path else None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip: int = max(1, round(video_fps / self._target_fps))

        writer = None
        if overlay_output_path and renderer:
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            Path(overlay_output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(overlay_output_path, fourcc, video_fps, (fw, fh))
            logger.info(f"Overlay output: {overlay_output_path}")

        frames: list[FrameFeatures] = []
        frame_idx: int = 0
        prev_pose_lm_data: Optional[list] = None

        # Cache last MediaPipe results — reused for overlay on non-sampled frames
        last_face_result = None
        last_pose_result = None
        last_hand_result = None
        last_ff: Optional[FrameFeatures] = None

        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx / video_fps * 1000)

            if frame_idx % skip == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    last_face_result = face_lm.detect_for_video(mp_img, timestamp_ms)
                except Exception as exc:
                    logger.debug(f"Face detect error frame {frame_idx}: {exc}")
                    last_face_result = None

                try:
                    last_pose_result = pose_lm.detect_for_video(mp_img, timestamp_ms)
                except Exception as exc:
                    logger.debug(f"Pose detect error frame {frame_idx}: {exc}")
                    last_pose_result = None

                try:
                    last_hand_result = hand_lm.detect_for_video(mp_img, timestamp_ms)
                except Exception as exc:
                    logger.debug(f"Hand detect error frame {frame_idx}: {exc}")
                    last_hand_result = None

                ff = self._process_frame_from_results(
                    bgr, rgb, frame_idx, timestamp_ms,
                    last_face_result, last_pose_result, last_hand_result,
                    prev_pose_lm_data,
                )

                if ff.body_detected:
                    prev_pose_lm_data = getattr(ff, "_raw_pose_lm", None)

                last_ff = ff
                frames.append(ff)

            # Write overlay on every original-fps frame using cached results
            if writer is not None and renderer is not None:
                annotated = renderer.draw_frame(
                    bgr, last_face_result, last_pose_result, last_hand_result, last_ff
                )
                writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        face_lm.close()
        pose_lm.close()
        hand_lm.close()

        logger.info(
            f"Extracted {len(frames)} frames from {Path(video_path).name} "
            f"({frame_idx} total, every {skip}th frame at video_fps={video_fps:.1f})"
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
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
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
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
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
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )

        return face_lm, pose_lm, hand_lm

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
    ) -> FrameFeatures:
        """Build FrameFeatures from pre-computed MediaPipe detector results."""
        h, w = rgb.shape[:2]
        ff = FrameFeatures(timestamp_ms=timestamp_ms, frame_idx=frame_idx)

        # ── Face ──────────────────────────────────────────────────────────────
        if face_result and face_result.face_landmarks:
            lm = face_result.face_landmarks[0]
            ff.face_detected = True
            ff.face_count = len(face_result.face_landmarks)
            ff.face_box_area = self._face_box_area(lm)
            ff.face_luminance = self._compute_face_luminance(lm, rgb, h, w)

            if face_result.face_blendshapes:
                ff.blendshapes = {
                    bs.category_name: round(float(bs.score), 5)
                    for bs in face_result.face_blendshapes[0]
                }

            if face_result.facial_transformation_matrixes:
                mat = np.array(
                    face_result.facial_transformation_matrixes[0].data
                ).reshape(4, 4)
                ff.head_pitch, ff.head_yaw, ff.head_roll = self._matrix_to_euler(mat)

            ff.ear_left, ff.ear_right, ff.ear_avg = self._compute_ear(lm)

            if len(lm) > _RIGHT_IRIS_IDX + 4:
                ff.gaze_x, ff.gaze_y = self._compute_gaze(lm)

        # ── Pose ──────────────────────────────────────────────────────────────
        if pose_result and pose_result.pose_landmarks:
            plm = pose_result.pose_landmarks[0]
            ff.body_detected = True
            ff.shoulder_angle, ff.spine_angle, ff.head_shoulder_dist = (
                self._compute_body_angles(plm, h)
            )
            ff.body_movement = self._compute_body_movement(plm, prev_pose_lm_data)
            ff._raw_pose_lm = plm  # type: ignore[attr-defined]

        # ── Hands ─────────────────────────────────────────────────────────────
        if hand_result and hand_result.hand_landmarks:
            ff.hands_detected = len(hand_result.hand_landmarks)
            ff.hand_near_face = self._check_hand_near_face(
                hand_result.hand_landmarks, ff, w, h
            )
            ff.hand_velocity = self._compute_hand_velocity(hand_result.hand_landmarks)

        return ff

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


# ══════════════════════════════════════════════════════════════════════════════
# SpeakerFaceMapper  — assigns windows to speakers
# ══════════════════════════════════════════════════════════════════════════════

class SpeakerFaceMapper:
    """
    Maps unassigned WindowFeatures to speakers using diarization timestamps.

    Strategy: for each window, find the speaker who was active for the
    longest overlap with that window.  Falls back to 'Speaker_0' when no
    segment overlaps (mono-speaker or no diarization provided).

    DSA: O(W × S) overlap scan — acceptable for typical session sizes
    (≤300 windows × ≤500 segments).  Indexed by speaker for O(1) lookup
    when building the per-speaker lists.
    """

    def assign(
        self,
        windows: list[WindowFeatures],
        diar_segments: list[dict],
    ) -> dict[str, list[WindowFeatures]]:
        """
        Args:
            windows:        Output of VideoFeatureExtractor.extract_all().
            diar_segments:  [{speaker, start_ms, end_ms}, ...] from voice agent.

        Returns:
            dict[speaker_id → list[WindowFeatures]] with speaker_id set on each window.
        """
        result: dict[str, list[WindowFeatures]] = defaultdict(list)

        for wf in windows:
            speaker = self._dominant_speaker(wf.window_start_ms, wf.window_end_ms, diar_segments)
            wf.speaker_id = speaker
            result[speaker].append(wf)

        return dict(result)

    @staticmethod
    def _dominant_speaker(
        win_start: int,
        win_end: int,
        segments: list[dict],
    ) -> str:
        """Return the speaker with the greatest overlap with [win_start, win_end]."""
        overlap: dict[str, float] = defaultdict(float)

        for seg in segments:
            seg_start = seg.get("start_ms", 0)
            seg_end   = seg.get("end_ms", 0)
            speaker   = seg.get("speaker", "Speaker_0")

            # Overlap = intersection of [win_start,win_end] and [seg_start,seg_end]
            ov = max(0.0, min(win_end, seg_end) - max(win_start, seg_start))
            if ov > 0:
                overlap[speaker] += ov

        if not overlap:
            return "Speaker_0"
        return max(overlap, key=lambda s: overlap[s])
