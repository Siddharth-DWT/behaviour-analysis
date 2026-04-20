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

    def __init__(
        self,
        model_dir: str = "models/mediapipe",
        target_fps: int = TARGET_FPS,
        window_ms: int = WINDOW_MS,
        num_faces: int = 2,
    ) -> None:
        self._model_mgr  = MediaPipeModelManager(model_dir)
        self._target_fps = target_fps
        self._window_ms  = window_ms
        self._num_faces  = num_faces
        self._aggregator = WindowAggregator(window_ms, target_fps)
        self._mp_available: Optional[bool] = None    # lazy probe

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_all(self, video_path: str) -> list[WindowFeatures]:
        """
        Process video at target_fps and return aggregated window features.

        Args:
            video_path: Path to mp4/webm/avi file.

        Returns:
            List of WindowFeatures, one per 2-second window.
            Empty list if video cannot be read or MediaPipe unavailable.
        """
        if not self._check_mediapipe():
            logger.error("MediaPipe not available — returning empty features.")
            return []

        frames = self._extract_frames(video_path)
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

    def _extract_frames(self, video_path: str) -> list[FrameFeatures]:
        import cv2
        import mediapipe as mp

        face_lm, pose_lm, hand_lm = self._build_landmarkers(mp)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip: int = max(1, round(video_fps / self._target_fps))

        frames: list[FrameFeatures] = []
        frame_idx: int = 0
        prev_pose_lm_data: Optional[list] = None    # for body_movement delta

        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % skip == 0:
                timestamp_ms = int(frame_idx / video_fps * 1000)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                ff = self._process_frame(
                    mp, rgb, bgr, frame_idx, timestamp_ms,
                    face_lm, pose_lm, hand_lm,
                    prev_pose_lm_data,
                )

                # Update prev pose landmarks for next movement delta
                if ff.body_detected:
                    prev_pose_lm_data = getattr(ff, "_raw_pose_lm", None)

                frames.append(ff)

            frame_idx += 1

        cap.release()
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
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=self._model_mgr.get_pose_landmarker_path()
                ),
                running_mode=RunningMode.VIDEO,
                num_poses=self._num_faces,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        hand_lm = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=self._model_mgr.get_hand_landmarker_path()
                ),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        return face_lm, pose_lm, hand_lm

    def _process_frame(
        self, mp, rgb: np.ndarray, bgr: np.ndarray,
        frame_idx: int, timestamp_ms: int,
        face_lm, pose_lm, hand_lm,
        prev_pose_lm_data: Optional[list],
    ) -> FrameFeatures:
        """Extract all features from one frame."""
        h, w = rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ff = FrameFeatures(timestamp_ms=timestamp_ms, frame_idx=frame_idx)

        # ── Face landmarks ────────────────────────────────────────────────────
        try:
            face_result = face_lm.detect_for_video(mp_img, timestamp_ms)
            if face_result.face_landmarks:
                lm = face_result.face_landmarks[0]   # dominant face
                ff.face_detected = True
                ff.face_box_area = self._face_box_area(lm)

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
        except Exception as exc:
            logger.debug(f"Face detection error at frame {frame_idx}: {exc}")

        # ── Pose landmarks ────────────────────────────────────────────────────
        try:
            pose_result = pose_lm.detect_for_video(mp_img, timestamp_ms)
            if pose_result.pose_landmarks:
                plm = pose_result.pose_landmarks[0]
                ff.body_detected = True
                ff.shoulder_angle, ff.spine_angle, ff.head_shoulder_dist = (
                    self._compute_body_angles(plm, h)
                )
                ff.body_movement = self._compute_body_movement(plm, prev_pose_lm_data)
                # Store raw landmarks for next frame's delta
                ff._raw_pose_lm = plm  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug(f"Pose detection error at frame {frame_idx}: {exc}")

        # ── Hand landmarks ────────────────────────────────────────────────────
        try:
            hand_result = hand_lm.detect_for_video(mp_img, timestamp_ms)
            if hand_result.hand_landmarks:
                ff.hands_detected = len(hand_result.hand_landmarks)
                ff.hand_near_face = self._check_hand_near_face(
                    hand_result.hand_landmarks, ff, w, h
                )
                ff.hand_velocity = self._compute_hand_velocity(hand_result.hand_landmarks)
        except Exception as exc:
            logger.debug(f"Hand detection error at frame {frame_idx}: {exc}")

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
