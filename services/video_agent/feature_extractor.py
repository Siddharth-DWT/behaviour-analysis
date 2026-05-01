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
        return self._get("pose_landmarker_heavy.task")

    def get_hand_landmarker_path(self) -> str:
        return self._get("hand_landmarker.task")

    def get_face_detector_path(self) -> str:
        return self._get("blaze_face_full_range.tflite")

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

        # ── Hands ──────────────────────────────────────────────────────────────
        wf.hands_detected_rate   = float(np.mean([f.hands_detected > 0 for f in frames]))
        wf.hand_near_face_pct    = float(np.mean([f.hand_near_face    for f in frames]))
        hand_vels = [f.hand_velocity for f in frames if f.hand_velocity > 0]
        if hand_vels:
            wf.gesture_velocity_mean = float(np.mean(hand_vels))
            wf.gesture_velocity_max  = float(np.max(hand_vels))

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
    """Lightweight landmark compatible with MediaPipe NormalizedLandmark (.x .y .z)."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


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


class _HandResult:
    __slots__ = ("hand_landmarks",)

    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks or []


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
    _BODY_SCALE = 3.0    # extend bbox this many face-heights downward for pose

    def __init__(self, face_det, face_det_img, face_lm, pose_lm, hand_lm, num_faces: int) -> None:
        self._face_det     = face_det      # FaceDetector — VIDEO mode, full frame
        self._face_det_img = face_det_img  # FaceDetector — IMAGE mode, quadrant passes
        self._face_lm      = face_lm       # FaceLandmarker — IMAGE mode, per crop
        self._pose_lm      = pose_lm       # PoseLandmarker — IMAGE mode, per crop
        self._hand_lm      = hand_lm       # HandLandmarker — VIDEO mode, full frame
        self._num_faces    = num_faces

    @classmethod
    def create(
        cls, mp, model_mgr: "MediaPipeModelManager", num_faces: int
    ) -> "TiledFrameProcessor":
        """Build all five MediaPipe instances. Called once per video."""
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
        return cls(face_det, face_det_img, face_lm, pose_lm, hand_lm, num_faces)

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

        # ── Hands on full frame (may cross tile boundaries) ───────────────────
        try:
            mp_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_raw = self._hand_lm.detect_for_video(mp_full, ts_ms)
            hand_lms = hand_raw.hand_landmarks if hand_raw else []
        except Exception:
            hand_lms = []

        if not face_boxes:
            return _FaceResult([]), _PoseResult([]), _HandResult(hand_lms)

        all_face_lm:  list = []
        all_face_bs:  list = []
        all_face_mat: list = []
        all_pose_lm:  list = []

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

            # ── Body crop (face bbox extended downward for pose) ──────────────
            bc_x1 = max(0, fx - px)
            bc_y1 = max(0, fy - py)
            bc_x2 = min(fw, fx + fbw + px)
            bc_y2 = min(fh, fy + int(fbh * self._BODY_SCALE))
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
                        )
                        for lm in bc_res.pose_landmarks[0]
                    ]
                    all_pose_lm.append(remapped_pose)
            except Exception as exc:
                logger.debug(f"[tiled] pose crop error: {exc}")

        return (
            _FaceResult(
                face_landmarks=all_face_lm,
                face_blendshapes=all_face_bs,       # keep None entries so face_landmarks[i] ↔ face_blendshapes[i]
                facial_transformation_matrixes=all_face_mat,  # same reason
            ),
            _PoseResult(pose_landmarks=all_pose_lm),
            _HandResult(hand_landmarks=hand_lms),
        )

    def close(self) -> None:
        for obj in (self._face_det, self._face_det_img, self._face_lm, self._pose_lm, self._hand_lm):
            try:
                obj.close()
            except Exception:
                pass


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
        # Scale thickness/radius to frame size so 720p and 1080p look consistent
        s = max(1.0, min(h, w) / 720.0)
        line_t = max(1, round(s))           # 1 at 720p, 2 at 1440p
        dot_r  = max(1, round(1.5 * s))     # 1-2px at 720p

        color_bone = (255, 100, 30)
        color_dot  = (200, 200, 255)
        for pose_lm in pose_result.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_lm]
            n = len(pts)
            for a, b in self._POSE_SEGS:
                if a < n and b < n:
                    cv2.line(bgr, pts[a], pts[b], color_bone, line_t, cv2.LINE_AA)
            for pt in pts[:33]:
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

    def burn_landmarks_and_labels(
        self,
        video_path: str,
        signals: list,
        output_path: Optional[str] = None,
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

        # Cache last MediaPipe results — reused for overlay on non-sampled frames
        last_face_result = None
        last_pose_result = None
        last_hand_result = None

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
                        frames.extend(frame_features_list)
                        for ff in frame_features_list:
                            if ff.face_index == 0 and ff.body_detected:
                                prev_pose_lm_data = getattr(ff, "_raw_pose_lm", None)
                                break
                    except Exception as exc:
                        logger.warning(f"Frame {frame_idx} processing error (skipping): {exc}")

                # Write overlay on every original-fps frame using cached results
                if writer is not None and renderer is not None:
                    try:
                        annotated = renderer.draw_frame(
                            bgr, last_face_result, last_pose_result, last_hand_result
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

            # Hands — associate only hands near this face
            if hand_result and hand_result.hand_landmarks:
                ff.hands_detected = self._count_hands_near_face(
                    hand_result.hand_landmarks, lm
                )
                ff.hand_near_face = self._check_hand_near_face_single(
                    hand_result.hand_landmarks, lm
                )
                ff.hand_velocity = self._compute_hand_velocity(hand_result.hand_landmarks)

            results.append(ff)

        return results

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
        """True if any hand landmark falls inside this face's bounding box (20% expanded)."""
        fxs = [lm.x for lm in face_landmarks]
        fys = [lm.y for lm in face_landmarks]
        fx1, fx2 = min(fxs), max(fxs)
        fy1, fy2 = min(fys), max(fys)
        pad = (fx2 - fx1) * 0.2
        fx1 -= pad
        fx2 += pad
        fy1 -= pad
        fy2 += pad
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
        Map windows to speakers using face position + diarization overlap.

        Each face_index occupies a stable grid position. For each face_index
        we sum diarization overlap across all its windows to find its dominant
        speaker.  Conflict resolution keeps the best-overlap assignment and
        falls back to a positional label for displaced faces.

        Args:
            windows:        Output of VideoFeatureExtractor.extract_all().
            diar_segments:  [{speaker, start_ms, end_ms}, ...] from voice agent.

        Returns:
            dict[speaker_id → list[WindowFeatures]] with speaker_id set on each window.
        """
        result: dict[str, list[WindowFeatures]] = defaultdict(list)

        if not windows:
            return dict(result)

        # Group windows by face_index
        by_face: dict[int, list[WindowFeatures]] = defaultdict(list)
        for wf in windows:
            by_face[wf.face_index].append(wf)

        # Per face_index: accumulate diarization overlap to find dominant speaker
        face_to_speaker: dict[int, str] = {}
        face_overlap_totals: dict[int, float] = {}

        for face_idx, face_windows in by_face.items():
            speaker_overlap: dict[str, float] = defaultdict(float)
            for wf in face_windows:
                for seg in diar_segments:
                    seg_start = seg.get("start_ms", 0)
                    seg_end   = seg.get("end_ms",   0)
                    speaker   = seg.get("speaker",  "Speaker_0")
                    ov = max(0.0, min(wf.window_end_ms, seg_end) - max(wf.window_start_ms, seg_start))
                    if ov > 0:
                        speaker_overlap[speaker] += ov

            if speaker_overlap:
                best = max(speaker_overlap, key=lambda s: speaker_overlap[s])
                face_to_speaker[face_idx]    = best
                face_overlap_totals[face_idx] = speaker_overlap[best]
            else:
                face_to_speaker[face_idx]    = f"Speaker_{face_idx}"
                face_overlap_totals[face_idx] = 0.0

        # Resolve conflicts: two faces mapped to the same speaker
        speaker_to_faces: dict[str, list[int]] = defaultdict(list)
        for fi, spk in face_to_speaker.items():
            speaker_to_faces[spk].append(fi)

        for spk, face_idxs in speaker_to_faces.items():
            if len(face_idxs) > 1:
                # Keep the face with the highest overlap; re-label the rest
                face_idxs_sorted = sorted(
                    face_idxs,
                    key=lambda fi: face_overlap_totals.get(fi, 0.0),
                    reverse=True,
                )
                for fi in face_idxs_sorted[1:]:
                    face_to_speaker[fi] = f"Face_{fi}"

        # Apply mapping
        for wf in windows:
            speaker = face_to_speaker.get(wf.face_index, f"Face_{wf.face_index}")
            wf.speaker_id = speaker
            result[speaker].append(wf)

        logger.info(
            "SpeakerFaceMapper: %d face(s) → %s",
            len(by_face),
            {fi: spk for fi, spk in sorted(face_to_speaker.items())},
        )
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
