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
      → (dict[str, list[WindowFeatures]], dict[str, float], dict[int, str])   (windows_by_face, lip_sync_scores, face_to_speaker)

Research basis:
  - Soukupova & Cech 2016: Eye Aspect Ratio for blink detection
  - Bentivoglio 1997: resting blink rate 15-26 bpm (mode ~15 silent, ~26 conversation)
  - Ekman & Friesen 1978: FACS Action Units (blendshapes approximate AU scores)
"""
import logging
import os
import urllib.request
import bisect
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("nexus.video.features")

# ─── Processing constants ──────────────────────────────────────────────────
TARGET_FPS: int = 5
WINDOW_MS: int = 2000
MIN_LIP_SYNC_LINK_SCORE: float = 0.02

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
# FrameSample + MeetingProfile  — pre-scan meeting type classification
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FrameSample:
    """Immutable snapshot of one probed frame's face geometry."""
    face_count: int
    face_areas: tuple[float, ...]                       # sorted largest-first, normalised 0-1
    face_centroids: tuple[tuple[float, float], ...]     # (cx, cy) normalised
    timestamp_pct: float                                # position in video 0.0-1.0


@dataclass
class MeetingProfile:
    """
    Configuration bundle derived from pre-scan probe.

    Configures all downstream components with optimal parameters for the
    detected meeting type.  Follows Builder Pattern via class factory methods.
    default() returns current hardcoded values so any failure is zero-regression.
    """
    meeting_type: str                       # "grid" | "active_speaker" | "room" | "unknown"
    expected_faces: int
    tracker_match_threshold: float          # CentroidTracker distance threshold
    verifier_check_interval: int            # IdentityVerifier frames between checks
    active_tile_tagger_enabled: bool
    layout_reset_sensitivity: float         # passed as cooldown_frames to LayoutClassifier
    merge_threshold_offset: float           # added to base ArcFace merge threshold
    static_face_filter_enabled: bool
    body_rules_confidence_cap: float
    tracker_max_disappeared: int = 90       # CentroidTracker expiry; 90 = 18s at 5fps (CLAUDE.md min)
    num_faces_override: Optional[int] = None
    min_detection_confidence: float = 0.15  # Bug 5: 0.30 for room/interrogation (large faces, fewer FPs)

    @classmethod
    def grid(cls, face_count: int) -> "MeetingProfile":
        """Grid/gallery: faces stationary, all similar size, grid-aligned."""
        return cls(
            meeting_type="grid",
            expected_faces=face_count,
            tracker_match_threshold=0.05,
            verifier_check_interval=50,
            active_tile_tagger_enabled=False,
            layout_reset_sensitivity=40,
            merge_threshold_offset=-0.05,
            static_face_filter_enabled=True,
            body_rules_confidence_cap=0.60,
            num_faces_override=face_count + 2,
        )

    @classmethod
    def active_speaker(cls, face_count: int) -> "MeetingProfile":
        """Active-speaker view: one large tile, tile swaps on speaker change."""
        return cls(
            meeting_type="active_speaker",
            expected_faces=face_count,
            tracker_match_threshold=0.10,
            verifier_check_interval=10,
            active_tile_tagger_enabled=True,
            layout_reset_sensitivity=25,
            merge_threshold_offset=0.0,
            static_face_filter_enabled=True,
            body_rules_confidence_cap=1.0,
            num_faces_override=max(face_count + 2, 6),
        )

    @classmethod
    def room(cls, face_count: int) -> "MeetingProfile":
        """Physical room camera: people move, no tiles, no screen share."""
        return cls(
            meeting_type="room",
            expected_faces=face_count,
            tracker_match_threshold=0.15,
            verifier_check_interval=30,
            active_tile_tagger_enabled=False,
            layout_reset_sensitivity=50,
            merge_threshold_offset=0.0,
            static_face_filter_enabled=True,
            body_rules_confidence_cap=1.0,
            num_faces_override=face_count + 4,
            min_detection_confidence=0.30,
        )

    @classmethod
    def default(cls) -> "MeetingProfile":
        """Fallback: identical to current hardcoded values.  Zero regression."""
        return cls(
            meeting_type="unknown",
            expected_faces=3,
            tracker_match_threshold=0.10,
            verifier_check_interval=30,
            active_tile_tagger_enabled=True,
            layout_reset_sensitivity=25,
            merge_threshold_offset=0.0,
            static_face_filter_enabled=True,
            body_rules_confidence_cap=1.0,
            num_faces_override=None,
        )


class MeetingTypeProbe:
    """
    Pre-scan video to classify meeting type before main extraction.

    Samples 10 frames across the video and runs the already-loaded MediaPipe
    FaceDetector on each.  Classifies from face count + size distribution +
    position alignment pattern.

    Total cost: ~500ms (10 × ~50ms detection).  Models are already loaded from
    warmup() — no additional model initialisation.

    Design:
      - Template Method: probe() → _sample_frames() → _classify() → factory
      - O(S × F) where S=10 samples, F=faces per frame
      - HashMap: row clustering for grid detection (quantized Y → face list)
      - Sorted arrays: grid alignment via sorted centroids + gap detection
      - Graceful degradation: any failure → MeetingProfile.default()

    Classification priority: grid > active_speaker > room > unknown.
    """

    SAMPLE_POINTS: tuple[float, ...] = tuple(i / 100.0 for i in range(2, 98, 5))  # 20 points, 2%→97%

    _GRID_MIN_FACES       = 4
    _GRID_MAX_AREA_RATIO  = 2.5
    _GRID_ROW_TOLERANCE   = 0.06
    _GRID_MIN_ROWS        = 2
    _GRID_MIN_COLS        = 2

    _AS_MIN_DOMINANT_RATIO   = 2.0
    _AS_MIN_DOMINANT_SAMPLES = 4

    _ROOM_Y_SPREAD  = 0.25
    _ROOM_MAX_FACES = 8

    def probe(self, video_path: str, face_detector, mp_module=None) -> "MeetingProfile":
        """
        Sample frames and classify meeting type.

        Returns MeetingProfile with optimal parameters for detected type.
        On ANY failure returns MeetingProfile.default() (zero regression).
        """
        try:
            samples = self._sample_frames(video_path, face_detector, mp_module)
            if not samples:
                logger.warning("MeetingTypeProbe: no valid samples — using default profile")
                return MeetingProfile.default()

            profile = self._classify(samples)
            logger.info(
                "MeetingTypeProbe: %s (expected %d faces) from %d samples "
                "[counts=%s, largest_areas=%s]",
                profile.meeting_type,
                profile.expected_faces,
                len(samples),
                [s.face_count for s in samples],
                [round(s.face_areas[0], 3) if s.face_areas else 0 for s in samples],
            )
            return profile

        except Exception as exc:
            logger.warning("MeetingTypeProbe failed (non-fatal): %s — using default", exc)
            return MeetingProfile.default()

    def _sample_frames(self, video_path: str, face_detector, mp_module) -> "list[FrameSample]":
        """Read SAMPLE_POINTS frames and run face detection on each. O(S)."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            cap.release()
            return []

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        samples: list[FrameSample] = []

        for pct in self.SAMPLE_POINTS:
            target = int(pct * total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, bgr = cap.read()
            if not ret or bgr is None:
                continue
            try:
                face_areas: list[float] = []
                face_centroids: list[tuple[float, float]] = []

                if mp_module:
                    mp_img = mp_module.Image(
                        image_format=mp_module.ImageFormat.SRGB,
                        data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    )
                    result = face_detector.detect(mp_img)
                    for det in (result.detections if result else []):
                        bb = det.bounding_box
                        area = (bb.width * bb.height) / (frame_w * frame_h)
                        cx   = (bb.origin_x + bb.width  / 2) / frame_w
                        cy   = (bb.origin_y + bb.height / 2) / frame_h
                        face_areas.append(area)
                        face_centroids.append((cx, cy))

                if face_areas:
                    paired = sorted(zip(face_areas, face_centroids), key=lambda p: -p[0])
                    face_areas     = [p[0] for p in paired]
                    face_centroids = [p[1] for p in paired]

                samples.append(FrameSample(
                    face_count=len(face_areas),
                    face_areas=tuple(face_areas),
                    face_centroids=tuple(face_centroids),
                    timestamp_pct=pct,
                ))
            except Exception as exc:
                logger.debug("Probe frame at %.0f%% failed: %s", pct * 100, exc)

        cap.release()
        return samples

    def _classify(self, samples: "list[FrameSample]") -> "MeetingProfile":
        """Classify meeting type from sampled frame geometry. Priority: grid > active_speaker > room."""
        face_samples = [s for s in samples if s.face_count >= 2]
        if not face_samples:
            solo = [s for s in samples if s.face_count == 1]
            return MeetingProfile.active_speaker(1) if solo else MeetingProfile.default()

        median_count = sorted([s.face_count for s in face_samples])[len(face_samples) // 2]

        if self._is_grid(face_samples):
            return MeetingProfile.grid(median_count)
        if self._is_active_speaker(face_samples):
            return MeetingProfile.active_speaker(median_count)
        if self._is_room(face_samples):
            return MeetingProfile.room(median_count)
        # Fallback: if 2+ faces consistently detected and not grid/active_speaker,
        # treat as physical room. Covers seated 2-person scenes where faces are at
        # the same height (Y-spread < _ROOM_Y_SPREAD) — e.g., interrogation rooms.
        if len(face_samples) >= 3:
            return MeetingProfile.room(median_count)
        return MeetingProfile.default()

    def _is_grid(self, samples: "list[FrameSample]") -> bool:
        """Grid: ≥4 faces, similar sizes, positions form rows.  O(F log F) per sample."""
        grid_samples = [s for s in samples if s.face_count >= self._GRID_MIN_FACES]
        if len(grid_samples) < len(samples) * 0.5:
            return False
        for s in grid_samples:
            if len(s.face_areas) < self._GRID_MIN_FACES:
                continue
            min_area = min((a for a in s.face_areas if a > 0.001), default=0.001)
            if max(s.face_areas) / min_area > self._GRID_MAX_AREA_RATIO:
                return False
        aligned = sum(1 for s in grid_samples if self._check_grid_alignment(s.face_centroids))
        return aligned >= len(grid_samples) * 0.5

    def _check_grid_alignment(self, centroids: "tuple[tuple[float, float], ...]") -> bool:
        """Sort by Y → group into rows → require ≥2 rows with ≥2 faces each.  O(F log F)."""
        if len(centroids) < self._GRID_MIN_FACES:
            return False
        sorted_y = sorted(centroids, key=lambda c: c[1])
        rows: list[list] = []
        cur = [sorted_y[0]]
        for c in sorted_y[1:]:
            if c[1] - cur[-1][1] < self._GRID_ROW_TOLERANCE:
                cur.append(c)
            else:
                rows.append(cur)
                cur = [c]
        rows.append(cur)
        return sum(1 for r in rows if len(r) >= self._GRID_MIN_COLS) >= self._GRID_MIN_ROWS

    def _is_active_speaker(self, samples: "list[FrameSample]") -> bool:
        """Active-speaker: largest face > 2× second-largest in ≥4 of 10 samples."""
        dominant = sum(
            1 for s in samples
            if len(s.face_areas) >= 2
            and s.face_areas[0] / max(s.face_areas[1], 0.001) >= self._AS_MIN_DOMINANT_RATIO
        )
        return dominant >= self._AS_MIN_DOMINANT_SAMPLES

    def _is_room(self, samples: "list[FrameSample]") -> bool:
        """Room: high Y-spread across face positions, no grid alignment."""
        for s in samples:
            if len(s.face_centroids) < 2:
                continue
            ys = [cy for _, cy in s.face_centroids]
            if (
                max(ys) - min(ys) >= self._ROOM_Y_SPREAD
                and s.face_count <= self._ROOM_MAX_FACES
                and not self._check_grid_alignment(s.face_centroids)
            ):
                return True
        return False


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
                areas = [f.face_box_area for f in window_frames if f.face_box_area > 0]
                if areas:
                    wf.face_box_area_mean = float(np.mean(areas))
                if window_frames:
                    detected = sum(1 for f in window_frames if f.face_detected)
                    wf.face_detection_rate = detected / len(window_frames)
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
            # Clamp window to actual detection span so signals and the face
            # highlight box disappear when the face leaves the frame, not at
            # the fixed bucket boundary. Faces detected for only part of a
            # window no longer bleed their signals into frames where they
            # are absent.
            wf.window_start_ms = min(f.timestamp_ms for f in face_frames)
            wf.window_end_ms   = max(f.timestamp_ms for f in face_frames)

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
            # Skip pose for tiny tiles: body crop < ~200 px tall produces noisy
            # or empty MediaPipe Pose output — face landmarks already appended above.
            if fbh < 80:
                continue

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
            # Diarization fallback: diar_segments tells us *which* speaker label is
            # active, but FrameFeatures has no speaker_id field — that attribute is
            # assigned later by SpeakerFaceMapper on WindowFeatures after the full
            # extraction loop.  Matching by speaker_id here always fails silently.
            # Instead, pick the face with the highest jaw activity at this timestamp
            # as the most likely speaker (best proxy available during extraction).
            active_at_ts = any(
                seg.get("start_ms", 0) <= timestamp_ms <= seg.get("end_ms", 0)
                for seg in diar_segments
            )
            if active_at_ts:
                best_jaw = -1.0
                for ff in frame_features_list:
                    jaw = ff.blendshapes.get("jawOpen", 0.0)
                    if ff.face_detected and jaw > best_jaw:
                        best_jaw = jaw
                        speaking_face = ff

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
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
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
# Identity swap detection — IdentitySwapEvent / IdentityVerifier / TrackletSplitter
# ══════════════════════════════════════════════════════════════════════════════

class IdentitySwapEvent:
    """
    Immutable record of a detected identity change within a CentroidTracker track.

    __slots__ used for memory efficiency — IdentityVerifier may accumulate many
    events across a long session (one per detected swap per track).
    """
    __slots__ = ("track_id", "timestamp_ms", "old_embedding", "new_embedding")

    def __init__(
        self,
        track_id: int,
        timestamp_ms: int,
        old_embedding: "np.ndarray",
        new_embedding: "np.ndarray",
    ) -> None:
        self.track_id = track_id
        self.timestamp_ms = timestamp_ms
        self.old_embedding = old_embedding
        self.new_embedding = new_embedding


class IdentityVerifier:
    """
    Periodic ArcFace identity verification during the frame extraction loop.

    Detects identity swaps within CentroidTracker tracks by comparing face
    embeddings at regular intervals.  Catches BOTH speaker-boundary swaps
    (when a tile changes occupant at a diarization boundary) AND silent swaps
    (layout changes, join/leave, screen-share, pin/unpin) that diarization
    cannot detect.

    Design:
      - Single Responsibility: only detects swaps; does not split tracks.
      - Strategy Pattern: configurable check_interval and similarity_threshold.
      - HashMap: dict[int, np.ndarray] — O(1) track_id → embedding lookup.
      - Chronological output: swap_events property returns events sorted by time.

    Cost model (21-min video, 5fps, check_interval=30):
      total_sampled_frames ≈ 6300  →  checks ≈ 210
      210 × ~50ms per InsightFace call = ~10.5s  →  <1% of total loop time.
    """

    def __init__(
        self,
        embedder_app,
        check_interval: int = 30,
        similarity_threshold: float = 0.50,
        min_face_area: float = 0.005,
    ) -> None:
        self._app = embedder_app
        self._check_interval = check_interval
        self._sim_threshold = similarity_threshold
        self._min_face_area = min_face_area

        # HashMap: track_id → last verified L2-normalised ArcFace embedding
        self._track_embeddings: dict[int, "np.ndarray"] = {}
        self._swap_events: list[IdentitySwapEvent] = []
        self._frames_since_check: int = 0

    def check_frame(
        self,
        bgr: "np.ndarray",
        timestamp_ms: int,
        tracker: "CentroidTracker",
        frame_w: int,
        frame_h: int,
        current_layout: str = "",
    ) -> None:
        """
        Called on every sampled frame.  Runs InsightFace every check_interval
        frames; O(0) on intermediate frames.

        Matches each detected face to the nearest CentroidTracker track by
        normalised centroid distance, then compares the embedding against the
        stored reference.  A drop below similarity_threshold records a swap event.

        Layout-aware interval:
          screenshare  → skip (faces are tiny sidebars, embeddings unreliable)
          gallery_*    → half interval (tiles can swap occupants, check more often)
          active_speaker / solo → double interval (one large stable face, low swap risk)
          unknown / ""  → nominal interval
        """
        if current_layout == "screenshare":
            return

        if current_layout in ("gallery_2x2", "gallery_3x3"):
            effective_interval = max(1, self._check_interval // 2)
        elif current_layout in ("active_speaker", "solo"):
            effective_interval = self._check_interval * 2
        else:
            effective_interval = self._check_interval

        self._frames_since_check += 1
        if self._frames_since_check < effective_interval:
            return
        self._frames_since_check = 0

        if not self._app or frame_w == 0 or frame_h == 0:
            return

        try:
            faces = self._app.get(bgr)
        except Exception as exc:
            logger.debug("IdentityVerifier ArcFace error at %dms: %s", timestamp_ms, exc)
            return

        if not faces:
            return

        import numpy as np

        for face in faces:
            embedding = getattr(face, "embedding", None)
            if embedding is None or len(embedding) < 512:
                continue

            # InsightFace returns raw (unnormalized) embeddings; normalize to unit
            # vector so np.dot() == cosine similarity in [-1, 1].
            norm = float(np.linalg.norm(embedding))
            if norm <= 0:
                continue
            embedding = embedding / norm

            bbox = face.bbox  # [x1, y1, x2, y2] pixels
            # Normalised centroid of this detected face
            cx = ((bbox[0] + bbox[2]) / 2.0) / frame_w
            cy = ((bbox[1] + bbox[3]) / 2.0) / frame_h

            # Skip faces too small for reliable embedding (tiny sidebar tiles)
            face_area = ((bbox[2] - bbox[0]) / frame_w) * ((bbox[3] - bbox[1]) / frame_h)
            if face_area < self._min_face_area:
                continue

            # Large faces (main pinned tile, face_area > 4% of frame) produce
            # stable ArcFace embeddings — only a genuine occupant change drives
            # similarity below 0.35.  Normal head movement keeps sim in 0.40-0.60,
            # so using the standard 0.50 threshold triggers false splits every few
            # seconds on the main tile.  Small faces use the full threshold because
            # their noisier embeddings need the wider range to catch real swaps.
            effective_threshold = (
                min(self._sim_threshold, 0.35)
                if face_area > 0.04
                else self._sim_threshold
            )

            # Match to nearest active CentroidTracker track by position.
            # O(T) where T = number of active tracks — typically ≤ 10.
            best_tid: Optional[int] = None
            best_dist = float("inf")
            for tid, (tx, ty) in tracker._tracks.items():
                d = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                if d < best_dist and d < 0.15:
                    best_dist = d
                    best_tid = tid

            if best_tid is None:
                continue

            # Compare against stored embedding; ArcFace embeddings are
            # L2-normalised so dot product == cosine similarity directly.
            if best_tid in self._track_embeddings:
                sim = float(np.dot(self._track_embeddings[best_tid], embedding))
                if abs(sim) < 0.15:
                    # Near-zero cosine similarity means ArcFace produced noise
                    # (tiny face tile — embedding unreliable). Skip check and
                    # do not overwrite stored embedding with a noisy one.
                    continue
                if sim < effective_threshold:
                    self._swap_events.append(IdentitySwapEvent(
                        track_id=best_tid,
                        timestamp_ms=timestamp_ms,
                        old_embedding=self._track_embeddings[best_tid].copy(),
                        new_embedding=embedding.copy(),
                    ))
                    logger.info(
                        "IdentityVerifier: swap on track %d at %dms (sim=%.3f < %.2f, face_area=%.3f)",
                        best_tid, timestamp_ms, sim, effective_threshold, face_area,
                    )

            # Always update stored embedding to track the current occupant
            self._track_embeddings[best_tid] = embedding.copy()

    @property
    def swap_events(self) -> list[IdentitySwapEvent]:
        """Swap events sorted chronologically (O(E log E), called once after loop)."""
        return sorted(self._swap_events, key=lambda e: e.timestamp_ms)

    def force_check_next_frame(self) -> None:
        """Force an identity check on the very next sampled frame.

        Called when face count changes — a new person appearing at a position
        previously occupied by someone else is the earliest signal of an occupant
        swap. Without this, the periodic interval (up to 6s) delays detection and
        contaminates the joining person's first windows with the wrong track ID.

        999_999 exceeds any effective_interval (max = check_interval * 2 = 60)
        so the very next check_frame call always passes the < effective_interval
        guard, including in active_speaker/solo layout where effective_interval=60.
        Previously using self._check_interval (30) meant 30+1=31 < 60 → skipped.
        """
        self._frames_since_check = 999_999

    def clear(self) -> None:
        """Reset all state for re-use."""
        self._track_embeddings.clear()
        self._swap_events.clear()
        self._frames_since_check = 0


class TrackletSplitter:
    """
    Splits contaminated CentroidTracker tracks at identity swap points (in-place).

    Consumes IdentitySwapEvent records from IdentityVerifier and rewrites
    ff.face_index on all FrameFeatures at or after each swap timestamp, giving
    the post-swap frames a fresh track_id.  Must run BEFORE best-crop selection
    so the subsequent ArcFace step receives clean, single-person tracks.

    Design:
      - Single Responsibility: only splits tracks; detection and linking are elsewhere.
      - HashMap: dict[int, list[int]] — O(1) track → swap-point list.
      - Chronological processing: events sorted so multiple swaps on the same
        track produce independent clean segments (tid chain: A → B → C).
      - O(E × F): E swap events (typically 5–20), F frames (≈ 6300) → ≤ 126,000
        integer comparisons → < 0.1 s wall-clock.
    """

    def __init__(self, centroid_tracker: "CentroidTracker") -> None:
        self._tracker = centroid_tracker

    def split_tracks(
        self,
        frames: list["FrameFeatures"],
        swap_events: list[IdentitySwapEvent],
    ) -> None:
        """
        Rewrite face_index on frames in-place.  Each swap creates a new track_id
        allocated from centroid_tracker._next_id.

        Args:
            frames:       All FrameFeatures extracted during the loop (mutated).
            swap_events:  Chronologically sorted events from IdentityVerifier.
        """
        if not swap_events:
            return

        # Build HashMap: track_id → sorted list of swap timestamps
        # Sorting inside the grouping step ensures chronological processing
        # even if caller provides unsorted events.  O(E log E).
        swap_map: dict[int, list[int]] = {}
        for event in swap_events:
            swap_map.setdefault(event.track_id, []).append(event.timestamp_ms)
        for timestamps in swap_map.values():
            timestamps.sort()

        for original_tid, timestamps in swap_map.items():
            current_tid = original_tid
            for swap_ts in timestamps:
                new_tid = self._tracker._next_id
                self._tracker._next_id += 1

                rewritten = 0
                for ff in frames:
                    if ff.face_index == current_tid and ff.timestamp_ms >= swap_ts:
                        ff.face_index = new_tid
                        rewritten += 1

                logger.info(
                    "TrackletSplitter: Face_%d split at %dms → Face_%d (%d frames rewritten)",
                    current_tid, swap_ts, new_tid, rewritten,
                )
                # Future swaps from the same position target the new track
                current_tid = new_tid


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

    def reset(self) -> None:
        """
        Kill all active tracks on a layout change event.

        _next_id is intentionally preserved — post-reset tracks must NOT reuse
        pre-reset IDs, or ArcFace merge would conflate tracks from different
        layout epochs into the same canonical identity.
        """
        self._tracks.clear()
        self._disappeared.clear()


# ══════════════════════════════════════════════════════════════════════════════
# LayoutClassifier  — per-frame meeting layout detection
# ══════════════════════════════════════════════════════════════════════════════

class LayoutClassifier:
    """
    Classifies each frame into a meeting layout type and emits a layout_changed
    event when the stable classification transitions.

    Observer Pattern: layout_changed property is consumed by CentroidTracker.reset()
    and IdentityVerifier interval adjustment in _extract_frames.

    Sliding window of WINDOW_SIZE frames with Counter majority vote prevents
    single-frame misclassifications from triggering unnecessary tracker resets.

    Layout types:
      active_speaker  — one face clearly dominates (area ratio >= 1.5)
      gallery_2x2     — 2-4 equal-sized tiles
      gallery_3x3     — 5+ equal-sized tiles
      screenshare     — all faces tiny (screen content shared, faces in sidebar)
      solo            — single face detected
      unknown         — insufficient data
    """

    ACTIVE_SPEAKER_RATIO: float = 1.5   # largest / second-largest area for dominant tile
    SCREENSHARE_MAX_AREA: float = 0.005 # face_box_area below this → screenshare mode
    WINDOW_SIZE: int = 15               # frames for majority-vote stabilisation (3s at 5fps)
    COOLDOWN_FRAMES: int = 25           # min frames between reset events (~5s at 5fps)

    # O(1) layout → family lookup.  Resets only fire on cross-family transitions:
    #   single  : solo ↔ active_speaker — same person, tile size changed slightly
    #   gallery : gallery_2x2 ↔ gallery_3x3 — tile count changed, same grid epoch
    _FAMILY_MAP: "dict[str, str]" = {
        "active_speaker": "single",
        "solo":           "single",
        "gallery_2x2":    "gallery",
        "gallery_3x3":    "gallery",
        "screenshare":    "screenshare",
        "unknown":        "unknown",
    }

    def __init__(self, cooldown_frames: Optional[int] = None) -> None:
        self._window: "deque[str]" = deque(maxlen=self.WINDOW_SIZE)
        self._stable_layout: str = "unknown"
        self._layout_changed: bool = False
        self._cooldown_remaining: int = 0
        self._cooldown_frames: int = cooldown_frames if cooldown_frames is not None else self.COOLDOWN_FRAMES

    @classmethod
    def _layout_family(cls, layout: str) -> str:
        return cls._FAMILY_MAP.get(layout, "unknown")

    def classify_frame(self, frame_features_list: list) -> str:
        """
        Classify this frame and update the stable layout via majority vote.
        Sets layout_changed=True only on cross-family transitions with cooldown.

        Cooldown + family grouping prevents rapid gallery↔active_speaker
        oscillations (every 200ms) from triggering hundreds of tracker resets,
        which cause track ID fragmentation and inflate post-loop ArcFace work.
        """
        raw = self._classify_raw(frame_features_list)
        self._window.append(raw)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if len(self._window) < self.WINDOW_SIZE:
            self._layout_changed = False
            return raw

        majority, _ = Counter(self._window).most_common(1)[0]
        prev = self._stable_layout
        self._stable_layout = majority  # always update for current_layout accuracy

        if (prev != majority
                and self._cooldown_remaining == 0
                and (prev == "screenshare" or majority == "screenshare")):
            # Reset only on screenshare transitions — faces physically disappear/
            # reappear in different counts and positions. Gallery ↔ active_speaker
            # transitions reposition the same faces; the centroid tracker assigns
            # new IDs naturally (distance > match_threshold) and ArcFace merge
            # re-associates them, so a full wipe is unnecessary and harmful.
            self._layout_changed = True
            self._cooldown_remaining = self._cooldown_frames
        else:
            self._layout_changed = False

        return majority

    def _classify_raw(self, frame_features_list: list) -> str:
        areas = sorted(
            [ff.face_box_area for ff in frame_features_list if ff.face_detected],
            reverse=True,
        )
        if not areas:
            return "screenshare"
        # Screenshare check before solo: a single tiny face (area < 0.5%) is a
        # sidebar thumbnail during screenshare, not a solo speaker tile.
        if all(a < self.SCREENSHARE_MAX_AREA for a in areas):
            return "screenshare"
        if len(areas) == 1:
            return "solo"
        if areas[0] / max(areas[1], 0.001) >= self.ACTIVE_SPEAKER_RATIO:
            return "active_speaker"
        if len(areas) <= 4:
            return "gallery_2x2"
        return "gallery_3x3"

    @property
    def layout_changed(self) -> bool:
        return self._layout_changed

    @property
    def stable_layout(self) -> str:
        return self._stable_layout


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
            self._app.prepare(ctx_id=0, det_size=(320, 320))
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
# LightASDClassifier  — audio-visual active speaker detection
# ══════════════════════════════════════════════════════════════════════════════

class LightASDClassifier:
    """
    Active Speaker Detection via Light-ASD (Liao et al., ICASSP 2023).

    Replaces MediaPipe jawOpen lip-sync correlation with a learned audio-visual
    model.  94.1% precision on AVA-ActiveSpeaker benchmark.

    Pipeline:
      face_crops_sequence + audio  →  per-frame speaking probabilities
      →  Viterbi 2-state HMM smoothing  →  per-(face, speaker) scores
      →  Hungarian assignment  →  face_to_speaker mapping

    Fixes three failure modes that defeat lip-sync correlation:
      • Small tiles (face_area < 0.02) — jawOpen landmarks noisy at low res
      • Large/active-speaker tiles    — jaw saturation (always near open)
      • Side profiles                 — jawOpen not visible at yaw > 60°

    Model loading (priority order):
      1. ONNX        — <model_dir>/light_asd/light_asd.onnx  (onnxruntime)
      2. TorchScript — <model_dir>/light_asd/light_asd.pt   (torch.jit)
      3. Disabled    — falls back to MediaPipe lip-sync correlation

    Audio features: 40-dim log-mel filterbank, AUDIO_FRAMES_PER_VIDEO
    sub-frames per sampled video frame (librosa required).

    Obtain model: https://github.com/Junhua-Liao/Light-ASD
    Export to ONNX and place at models/light_asd/light_asd.onnx.

    Singleton — model loaded once per process, reused across sessions.
    """

    CROP_SIZE: int = 96               # Light-ASD standard face crop (96×96 gray)
    N_MELS: int = 40                  # log-mel filterbank dimensions
    AUDIO_FRAMES_PER_VIDEO: int = 4   # audio temporal resolution per video frame
    VITERBI_SELF_TRANS: float = 0.92  # HMM self-transition — high = smoother labels
    _LOG_EPS: float = 1e-7

    _instance: Optional["LightASDClassifier"] = None

    @classmethod
    def get_instance(cls, model_dir: str = "models") -> "LightASDClassifier":
        if cls._instance is None:
            cls._instance = cls(model_dir=model_dir)
        return cls._instance

    def __init__(self, model_dir: str = "models") -> None:
        self._sess = None           # onnxruntime.InferenceSession
        self._torch_model = None    # torch TorchScript model
        self._available: bool = False
        self._librosa_ok: bool = False
        self._sr: int = 16000

        onnx_path = Path(model_dir) / "light_asd" / "light_asd.onnx"
        pt_path   = Path(model_dir) / "light_asd" / "light_asd.pt"

        if onnx_path.exists():
            try:
                import onnxruntime as ort
                self._sess = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                )
                self._available = True
                logger.info("LightASD: ONNX model loaded — %s", onnx_path)
            except Exception as exc:
                logger.warning("LightASD: ONNX load failed: %s", exc)

        if not self._available and pt_path.exists():
            try:
                import torch
                self._torch_model = torch.jit.load(str(pt_path), map_location="cpu")
                self._torch_model.eval()
                self._available = True
                logger.info("LightASD: TorchScript model loaded — %s", pt_path)
            except Exception as exc:
                logger.warning("LightASD: PyTorch load failed: %s", exc)

        if not self._available:
            logger.info(
                "LightASD model not found at %s — "
                "active speaker detection disabled, using lip-sync fallback",
                Path(model_dir) / "light_asd",
            )

        try:
            import librosa as _lr  # noqa: F401
            self._librosa_ok = True
        except ImportError:
            logger.debug("LightASD: librosa unavailable — ASD disabled")

    @property
    def available(self) -> bool:
        return self._available and self._librosa_ok

    def score(
        self,
        face_crops_sequence: "dict[int, list[tuple[int, np.ndarray]]]",
        audio_path: str,
        fps: float = 5.0,
    ) -> "dict[int, list[tuple[int, float]]]":
        """
        Score all face tracks; return Viterbi-smoothed per-frame speaking labels.

        Args:
            face_crops_sequence: {track_id: [(ts_ms, gray_96x96), ...]} sorted by ts_ms.
                                 Built by _extract_frames and stored on the extractor.
            audio_path:          16kHz mono WAV extracted from the source video.
            fps:                 video sampling rate used during _extract_frames.

        Returns:
            {track_id: [(ts_ms, 0.0_or_1.0), ...]} — Viterbi binary labels.
            Empty dict on model/audio failure (triggers lip-sync fallback in assign()).
        """
        if not self.available or not face_crops_sequence:
            return {}

        try:
            log_mel = self._load_log_mel(audio_path, fps)   # [N_MELS, T_audio]
        except Exception as exc:
            logger.warning("LightASD: audio feature extraction failed: %s", exc)
            return {}

        results: "dict[int, list[tuple[int, float]]]" = {}
        for track_id, crops in face_crops_sequence.items():
            if len(crops) < 3:
                continue
            try:
                smoothed = self._score_track(crops, log_mel, fps)
                if smoothed:
                    results[track_id] = smoothed
            except Exception as exc:
                logger.debug("LightASD: track %d failed: %s", track_id, exc)

        logger.info(
            "LightASD: scored %d / %d tracks", len(results), len(face_crops_sequence)
        )
        return results

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_log_mel(self, audio_path: str, fps: float) -> np.ndarray:
        """
        Compute log-mel spectrogram aligned to video frames.

        hop_length = sr / (fps × AUDIO_FRAMES_PER_VIDEO)
        → exactly AUDIO_FRAMES_PER_VIDEO columns per sampled video frame.
        Returns shape [N_MELS, T_audio].
        """
        import librosa
        y, _ = librosa.load(audio_path, sr=self._sr, mono=True)
        hop = max(1, int(self._sr / (fps * self.AUDIO_FRAMES_PER_VIDEO)))
        mel = librosa.feature.melspectrogram(
            y=y, sr=self._sr,
            n_mels=self.N_MELS,
            n_fft=min(512, len(y)),
            hop_length=hop,
        )
        return np.log(mel + self._LOG_EPS).astype(np.float32)

    def _score_track(
        self,
        crops: "list[tuple[int, np.ndarray]]",
        log_mel: np.ndarray,
        fps: float,
    ) -> "list[tuple[int, float]]":
        """
        Chunk track into 1000-frame blocks, run inference, apply Viterbi,
        return smoothed binary labels sorted by timestamp.
        """
        CHUNK = 1000
        all_probs: list[float] = []
        for start in range(0, len(crops), CHUNK):
            all_probs.extend(
                self._infer_chunk(crops[start : start + CHUNK], log_mel, fps)
            )

        raw = np.array(all_probs, dtype=np.float32)
        smoothed = self._viterbi_smooth(raw)
        return [(crops[i][0], float(smoothed[i])) for i in range(len(crops))]

    def _infer_chunk(
        self,
        crops: "list[tuple[int, np.ndarray]]",
        log_mel: np.ndarray,
        fps: float,
    ) -> list[float]:
        """
        Build audio [1, T×4, N_MELS] and visual [1, T, 1, H, W] tensors,
        run one model forward pass, return per-frame speaking probabilities.
        """
        import cv2 as _cv2

        T = len(crops)
        n_audio = T * self.AUDIO_FRAMES_PER_VIDEO

        # ── Visual ────────────────────────────────────────────────────────
        frames_arr: list[np.ndarray] = []
        for _, crop in crops:
            gray = (
                _cv2.cvtColor(crop, _cv2.COLOR_BGR2GRAY)
                if crop.ndim == 3 else crop
            )
            if gray.shape != (self.CROP_SIZE, self.CROP_SIZE):
                gray = _cv2.resize(
                    gray,
                    (self.CROP_SIZE, self.CROP_SIZE),
                    interpolation=_cv2.INTER_AREA,
                )
            frames_arr.append(gray.astype(np.float32) / 255.0)
        vis_batch = (
            np.stack(frames_arr, axis=0)[:, np.newaxis, :, :][np.newaxis]
        )  # [1, T, 1, H, W]

        # ── Audio — align first sub-frame to crops[0] timestamp ──────────
        first_ts_ms = crops[0][0]
        a_start = int(round(first_ts_ms / 1000.0 * fps * self.AUDIO_FRAMES_PER_VIDEO))
        a_end = a_start + n_audio
        a_slice = log_mel[:, max(0, a_start) : min(log_mel.shape[1], a_end)]
        if a_slice.shape[1] < n_audio:
            a_slice = np.pad(a_slice, ((0, 0), (0, n_audio - a_slice.shape[1])), mode="edge")
        audio_batch = a_slice.T[np.newaxis]  # [1, T*4, N_MELS]

        # ── Inference ─────────────────────────────────────────────────────
        if self._sess is not None:
            out = self._run_onnx(audio_batch, vis_batch)
        else:
            out = self._run_torch(audio_batch, vis_batch)

        if out is None or out.size == 0:
            return [0.5] * T

        # expected: [1, T, 2] logits or [1, T] probs
        o = out[0]   # [T, 2] or [T]
        if o.ndim == 2 and o.shape[-1] == 2:
            exp = np.exp(o - o.max(axis=1, keepdims=True))
            probs = (exp[:, 1] / exp.sum(axis=1)).astype(np.float32)
        elif o.ndim == 1:
            probs = np.clip(o, 0.0, 1.0).astype(np.float32)
        else:
            return [0.5] * T

        probs = np.clip(probs[:T], 0.0, 1.0)
        if len(probs) < T:
            probs = np.pad(probs, (0, T - len(probs)), mode="edge")
        return probs.tolist()

    def _run_onnx(
        self,
        audio_batch: np.ndarray,
        vis_batch: np.ndarray,
    ) -> "np.ndarray | None":
        """Run ONNX session; match inputs to audio/visual by name."""
        try:
            feeds: dict = {}
            for inp in self._sess.get_inputs():
                feeds[inp.name] = (
                    audio_batch if "audio" in inp.name.lower() else vis_batch
                )
            return np.array(self._sess.run(None, feeds)[0], dtype=np.float32)
        except Exception as exc:
            logger.debug("LightASD ONNX error: %s", exc)
            return None

    def _run_torch(
        self,
        audio_batch: np.ndarray,
        vis_batch: np.ndarray,
    ) -> "np.ndarray | None":
        """Run PyTorch TorchScript model."""
        try:
            import torch
            with torch.no_grad():
                out = self._torch_model(
                    torch.from_numpy(audio_batch),
                    torch.from_numpy(vis_batch),
                )
            return out.numpy() if hasattr(out, "numpy") else np.array(out, dtype=np.float32)
        except Exception as exc:
            logger.debug("LightASD PyTorch error: %s", exc)
            return None

    def _viterbi_smooth(self, probs: np.ndarray) -> np.ndarray:
        """
        2-state HMM Viterbi decoder.

        Corrects isolated per-frame mis-classifications by enforcing temporal
        consistency: if 9 of 10 consecutive frames say speaking, the 1 outlier
        that disagrees is corrected.

        States: 0=silent, 1=speaking
        Emission: P(obs | silent) = 1-p,  P(obs | speaking) = p
        Transition: VITERBI_SELF_TRANS on diagonal (0.92 → strong persistence)

        Returns binary float array (0.0 = silent, 1.0 = speaking).  O(T).
        """
        T = len(probs)
        if T == 0:
            return probs.copy()

        eps = self._LOG_EPS
        st = self.VITERBI_SELF_TRANS
        log_trans = np.log(
            np.array([[st, 1.0 - st], [1.0 - st, st]], dtype=np.float64)
        )

        dp = np.full((T, 2), -np.inf, dtype=np.float64)
        bp = np.zeros((T, 2), dtype=np.int8)

        p0 = float(probs[0])
        dp[0, 0] = np.log(0.5) + np.log(max(1.0 - p0, eps))
        dp[0, 1] = np.log(0.5) + np.log(max(p0, eps))

        for t in range(1, T):
            p = float(probs[t])
            log_emit = np.array(
                [np.log(max(1.0 - p, eps)), np.log(max(p, eps))],
                dtype=np.float64,
            )
            for s in range(2):
                candidates = dp[t - 1] + log_trans[:, s]
                bp[t, s] = int(np.argmax(candidates))
                dp[t, s] = float(np.max(candidates)) + log_emit[s]

        # Viterbi traceback
        path = np.zeros(T, dtype=np.float32)
        path[-1] = float(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = float(bp[t + 1, int(path[t + 1])])

        return path


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

    def apply_profile(self, profile: "MeetingProfile") -> None:
        """
        Configure extraction parameters from pre-scan meeting type.

        Called by VideoPipeline.run_analysis() BEFORE extract_all().
        Stores profile settings as instance variables that _extract_frames()
        reads when constructing per-session objects (Strategy Pattern).
        """
        self._profile_meeting_type        = profile.meeting_type
        self._profile_tracker_threshold   = profile.tracker_match_threshold
        self._profile_verifier_interval   = profile.verifier_check_interval
        self._profile_active_tile_enabled = profile.active_tile_tagger_enabled
        self._profile_layout_cooldown     = int(profile.layout_reset_sensitivity)
        self._profile_merge_offset        = profile.merge_threshold_offset
        self._profile_static_filter       = profile.static_face_filter_enabled
        self._profile_body_conf_cap       = profile.body_rules_confidence_cap
        self._profile_tracker_max_disapp  = profile.tracker_max_disappeared
        self._profile_min_det_conf        = profile.min_detection_confidence

        if profile.num_faces_override and profile.num_faces_override > self._num_faces:
            old = self._num_faces
            self._num_faces = profile.num_faces_override
            logger.info(
                "MeetingProfile: num_faces %d → %d (%s)",
                old, self._num_faces, profile.meeting_type,
            )

        logger.info(
            "MeetingProfile applied: type=%s tracker=%.3f verifier=%d "
            "active_tile=%s cooldown=%d merge_offset=%.3f body_cap=%.2f",
            profile.meeting_type,
            profile.tracker_match_threshold,
            profile.verifier_check_interval,
            profile.active_tile_tagger_enabled,
            int(profile.layout_reset_sensitivity),
            profile.merge_threshold_offset,
            profile.body_rules_confidence_cap,
        )

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
                ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

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

        logger.info("Loading MediaPipe models (face / pose / hand)…")
        processor = self._build_tiled_processor(mp)
        logger.info("MediaPipe models ready")
        renderer = OverlayRenderer() if overlay_output_path else None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip: int = max(1, round(video_fps / self._target_fps))
        _total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        _video_duration_s: float = _total_frames / video_fps if _total_frames else 0.0
        if _total_frames:
            logger.info(
                f"Video: {_total_frames} frames, {_video_duration_s:.0f}s "
                f"at {video_fps:.1f}fps — processing every {skip}th frame "
                f"(~{_total_frames // skip} frames to analyse)"
            )

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

        _tracker_threshold = getattr(self, "_profile_tracker_threshold", 0.10)
        _tracker_max_disapp = getattr(self, "_profile_tracker_max_disapp", 90)
        centroid_tracker = CentroidTracker(
            max_disappeared=_tracker_max_disapp,
            match_threshold=_tracker_threshold,
        )

        # Frame dimensions for IdentityVerifier centroid normalisation
        _frame_w: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _frame_h: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Identity verifier — periodic ArcFace swap detection during frame loop.
        # Reuses the singleton FaceEmbeddingExtractor already loaded for post-loop
        # embedding extraction; no additional model load.
        _embedder = FaceEmbeddingExtractor.get_instance()
        _verifier_interval = getattr(self, "_profile_verifier_interval", 30)
        identity_verifier = IdentityVerifier(
            embedder_app=_embedder._app if _embedder.available else None,
            check_interval=_verifier_interval,
            similarity_threshold=0.50,
            min_face_area=0.005,
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

        # Active-tile tagger — collects diarization-based face→speaker tags during
        # the frame loop for faces that dominate screen space (pinned/spotlight tiles).
        # A second rebuild pass after TrackletSplitter rewrites face_index produces the
        # final mapping injected into SpeakerFaceMapper.
        _active_tile_enabled = getattr(self, "_profile_active_tile_enabled", True)
        active_tagger = ActiveTileTagger(
            diar_segments=diar_segments,
            min_face_area=ActiveTileTagger.ACTIVE_TILE_MIN_AREA,
        ) if _active_tile_enabled else None

        # Layout classifier — detects layout transitions (gallery ↔ active_speaker)
        # and triggers CentroidTracker reset + IdentityVerifier interval adjustment.
        _layout_cooldown = getattr(self, "_profile_layout_cooldown", None)
        layout_classifier = LayoutClassifier(cooldown_frames=_layout_cooldown)

        # Tracks face count frame-to-frame so identity_verifier.force_check_next_frame()
        # fires immediately when a new person appears (count increases).
        _prev_face_count: int = 0
        # Scene cut detection — downsampled previous frame for pixel-diff comparison.
        _prev_small: "np.ndarray | None" = None
        _SCENE_CUT_THRESH: int = 40    # per-channel absolute diff to count a pixel as "changed"
        _SCENE_CUT_FRAC: float = 0.65  # fraction of pixels that must change to trigger reset
        _cut_timestamps: list[int] = []      # ms timestamp of each detected scene cut
        _track_first_ms: dict[int, int] = {} # track_id → first frame timestamp_ms
        _track_last_ms:  dict[int, int] = {} # track_id → last frame timestamp_ms
        # Tracks stable layout value to detect any layout change (even within-family).
        # When the large tile changes occupant (solo→active_speaker, gallery→active_speaker),
        # face count stays 1 because the small tile is filtered by area < 0.02 — so the
        # face count check misses the swap. Any layout value change triggers a force check.
        _prev_stable_layout: str = ""

        # Border detector — platform-native active-speaker signal (Zoom/Meet/Teams).
        # Cheap O(P) per frame; complements ActiveTileTagger in gallery mode.
        border_detector = ActiveSpeakerBorderDetector(
            platform=os.environ.get("VIDEO_PLATFORM", "auto")
        )

        # Keyed by (timestamp_ms, cx_int, cy_int) — a physical face position that
        # is invariant through TrackletSplitter and ArcFace merge.  After the loop,
        # remapped to canonical face IDs and stored as self._face_crops_sequence.
        _crop_by_position: "dict[tuple[int, int, int], np.ndarray]" = {}

        try:
            while cap.isOpened():
                ret, bgr = cap.read()
                if not ret:
                    break

                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                if frame_idx % skip == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                    # ── Scene cut detection (Bug 3) ───────────────────────────
                    # Hard cuts (interrogation → news footage) change >80% of pixels.
                    # Reset tracker so dead tracks from the previous scene don't
                    # absorb faces from the new scene.
                    _small = cv2.resize(bgr, (80, 45))
                    if _prev_small is not None:
                        _diff = np.abs(_small.astype(np.int16) - _prev_small.astype(np.int16))
                        _changed_frac = float(np.mean(_diff.max(axis=2) > _SCENE_CUT_THRESH))
                        if _changed_frac >= _SCENE_CUT_FRAC:
                            _cut_timestamps.append(timestamp_ms)
                            centroid_tracker.reset()
                            identity_verifier.clear()
                            if active_tagger is not None:
                                active_tagger.reset()
                            logger.info(
                                "Scene cut at %dms (%.0f%% pixels changed) — tracker reset",
                                timestamp_ms, _changed_frac * 100,
                            )
                    _prev_small = _small

                    try:
                        last_face_result, last_pose_result, last_hand_result = (
                            processor.detect(rgb, timestamp_ms)
                        )
                    except Exception as exc:
                        logger.debug(f"Tiled detect error frame {frame_idx}: {exc}")
                        last_face_result = last_pose_result = last_hand_result = None

                    # ── Profile detection confidence gate (Bug 5) ────────────
                    # Room/interrogation cameras have large faces — raise threshold
                    # from 0.15 to 0.30 to reject false detections on textures.
                    _min_det = getattr(self, "_profile_min_det_conf", 0.15)
                    if last_face_result is not None and _min_det > 0.15:
                        try:
                            last_face_result.detections = [
                                d for d in (last_face_result.detections or [])
                                if d.categories and d.categories[0].score >= _min_det
                            ]
                        except (AttributeError, TypeError):
                            pass

                    try:
                        frame_features_list = self._process_frame_from_results(
                            bgr, rgb, frame_idx, timestamp_ms,
                            last_face_result, last_pose_result, last_hand_result,
                            prev_pose_lm_data,
                        )

                        # ── Layout classification — BEFORE tracker update ──────
                        # Must run first: if the layout just changed, all face
                        # positions shifted simultaneously.  Resetting the tracker
                        # before update() prevents the new faces being matched to
                        # stale tracks from the previous layout epoch.
                        layout_classifier.classify_frame(frame_features_list)
                        current_layout = layout_classifier.stable_layout
                        if layout_classifier.layout_changed:
                            # Room-type meetings (interrogation, conference) use fixed
                            # camera positions — layout changes are camera angle cuts,
                            # not tile rearrangements. Resetting the tracker on every
                            # cut fragments face IDs and causes bleed-over signals.
                            # Cross-shot re-ID (_cross_shot_merge) handles consolidation.
                            if self._profile_meeting_type != "room":
                                centroid_tracker.reset()
                                identity_verifier.clear()
                                if active_tagger is not None:
                                    active_tagger.reset()
                            logger.info(
                                "Layout → %s at %dms — %s",
                                current_layout, timestamp_ms,
                                "tracker reset" if self._profile_meeting_type != "room" else "skipped reset (room profile)",
                            )
                        elif current_layout != _prev_stable_layout and _prev_stable_layout != "":
                            # Layout value changed within the same family (e.g. solo →
                            # active_speaker when Mirko joins). Tiles rearranged — large tile
                            # occupant may have swapped. Force immediate identity check so
                            # the swap is caught within one frame instead of up to 6s later.
                            identity_verifier.force_check_next_frame()
                        _prev_stable_layout = current_layout

                        # ── Apply temporal face tracking ──────────────────────
                        # Replace area-sorted face_index with stable track_id so
                        # WindowAggregator always groups the same physical person.
                        centroids = [
                            (ff.face_centre_x, ff.face_centre_y)
                            for ff in frame_features_list
                            if ff.face_detected and ff.face_box_area > 0.02
                        ]
                        if centroids:
                            track_ids = centroid_tracker.update(centroids)
                            ci = 0
                            for ff in frame_features_list:
                                if ff.face_detected and ci < len(track_ids):
                                    ff.face_index = track_ids[ci]
                                    ci += 1

                            # If face count increased, a new person just appeared.
                            # Force an immediate identity check so their join is
                            # detected within one frame rather than up to 6s later.
                            curr_face_count = len(centroids)
                            if curr_face_count > _prev_face_count:
                                identity_verifier.force_check_next_frame()
                            _prev_face_count = curr_face_count

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

                        # ── Platform border detection ─────────────────────────
                        # Build pixel bounding boxes from normalised coords for all
                        # detected faces; preserves order matching frame_features_list.
                        detected_ffs = [
                            ff for ff in frame_features_list if ff.face_detected
                        ]
                        face_boxes_px: list[tuple[int, int, int, int]] = []
                        for ff in detected_ffs:
                            side = ff.face_box_area ** 0.5
                            cx_px = int(ff.face_centre_x * _frame_w)
                            cy_px = int(ff.face_centre_y * _frame_h)
                            half_w = int(side * _frame_w * 0.5)
                            half_h = int(side * _frame_h * 0.5)
                            face_boxes_px.append((
                                max(0, cx_px - half_w),
                                max(0, cy_px - half_h),
                                min(2 * half_w, _frame_w),
                                min(2 * half_h, _frame_h),
                            ))

                        # ── Buffer face crops for Light-ASD inference ────────
                        # Grayscale 96×96; ~9KB per crop.
                        # Key = physical position (ts_ms, cx_int, cy_int) — stable
                        # through TrackletSplitter; remapped to canonical IDs after loop.
                        import cv2 as _cv2
                        for _ff, (_x, _y, _w, _h) in zip(detected_ffs, face_boxes_px):
                            if _w > 0 and _h > 0 and _ff.face_box_area >= 0.001:
                                _raw = bgr[_y : _y + _h, _x : _x + _w]
                                if _raw.size > 0:
                                    _gray = _cv2.cvtColor(_raw, _cv2.COLOR_BGR2GRAY)
                                    _crop = _cv2.resize(
                                        _gray,
                                        (LightASDClassifier.CROP_SIZE,
                                         LightASDClassifier.CROP_SIZE),
                                        interpolation=_cv2.INTER_AREA,
                                    )
                                    _crop_by_position[(
                                        timestamp_ms,
                                        int(_ff.face_centre_x * 1000),
                                        int(_ff.face_centre_y * 1000),
                                    )] = _crop

                        border_box_idx = border_detector.detect_active_speaker(
                            bgr, face_boxes_px
                        ) if face_boxes_px else None
                        # Map box list index → track_id (ff.face_index)
                        border_face_idx: "int | None" = (
                            detected_ffs[border_box_idx].face_index
                            if border_box_idx is not None else None
                        )

                        # Tag large-tile faces to diarization speakers in-loop.
                        # Uses pre-split face_index; rebuilt post-split after loop.
                        if active_tagger is not None:
                            active_tagger.tag_frame(
                                frame_features_list, timestamp_ms,
                                border_face_idx=border_face_idx,
                            )

                        # Periodic identity verification — every check_interval
                        # sampled frames; O(0) cost on intermediate frames.
                        identity_verifier.check_frame(
                            bgr, timestamp_ms, centroid_tracker, _frame_w, _frame_h,
                            current_layout=current_layout,
                        )

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

                if frame_idx > 0 and frame_idx % 1500 == 0:
                    logger.info(
                        f"Extraction progress: {frame_idx}/{_total_frames} frames "
                        f"({timestamp_ms / 1000:.0f}s / {_video_duration_s:.0f}s of video)"
                    )

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

        # ── Split contaminated tracks at ArcFace-verified swap points ────────
        # Must run BEFORE best-crop selection so crops are always from clean,
        # single-person track segments.
        _swap_events = identity_verifier.swap_events
        if _swap_events:
            logger.info(
                "IdentityVerifier: %d swap(s) on %d track(s) — splitting now",
                len(_swap_events),
                len(set(e.track_id for e in _swap_events)),
            )
            TrackletSplitter(centroid_tracker).split_tracks(frames, _swap_events)
        identity_verifier.clear()

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

        # ── Identity-based track deduplication via ArcFace ────────────────────
        # Run ArcFace on ALL tracks including short 1-2 frame ones.
        # A 1-frame track during a head turn IS the same person as Face_0 for
        # 200 frames — it deserves identity matching, not silent discard.
        # The old MIN_FRAMES filter before ArcFace caused those short tracks to
        # become orphaned Face_N IDs that never merged.
        #
        # Order of operations:
        #   1. ArcFace embed + adaptive threshold merge (all tracks)
        #   2. Position fallback for tracks where ArcFace extraction failed
        #   3. Remove true transients AFTER all merging:
        #      any track with < 10 frames (2s at 5fps), regardless of embedding
        self._cached_embeddings: dict[int, tuple[list[float], bytes]] = {}

        # Frame counts per track — dominant tracks anchor the merge
        frame_counts: dict[int, int] = {}
        for ff in frames:
            if ff.face_detected:
                frame_counts[ff.face_index] = frame_counts.get(ff.face_index, 0) + 1

        # Track time ranges for cross-shot re-identification
        for ff in frames:
            if ff.face_detected:
                tid = ff.face_index
                if tid not in _track_first_ms:
                    _track_first_ms[tid] = ff.timestamp_ms
                _track_last_ms[tid] = ff.timestamp_ms

        # Per-track avg face height, centroid, and pose quality — used for merge.
        # Computed from all frames (not just best-frame crop) for accuracy.
        _track_face_areas: dict[int, list[float]] = {}
        _track_cx: dict[int, list[float]] = {}
        _track_cy: dict[int, list[float]] = {}
        _track_pose_quality: dict[int, list[float]] = {}
        for ff in frames:
            if ff.face_detected and ff.face_box_area > 0.0:
                tid = ff.face_index
                _track_face_areas.setdefault(tid, []).append(ff.face_box_area)
                _track_cx.setdefault(tid, []).append(ff.face_centre_x)
                _track_cy.setdefault(tid, []).append(ff.face_centre_y)
                # Pose quality: 1.0 = frontal, <0.5 = profile.
                # |yaw| > 30° → quality below 0.7; |yaw| > 45° → below 0.5.
                yaw_pen = 1.0 / (1.0 + abs(ff.head_yaw) * 0.03)
                pitch_pen = 1.0 / (1.0 + abs(ff.head_pitch) * 0.02)
                _track_pose_quality.setdefault(tid, []).append(yaw_pen * pitch_pen)
        track_face_heights: dict[int, float] = {
            tid: float(np.mean([a ** 0.5 for a in areas]))
            for tid, areas in _track_face_areas.items()
        }
        track_centroids: dict[int, tuple[float, float]] = {
            tid: (float(np.mean(_track_cx[tid])), float(np.mean(_track_cy[tid])))
            for tid in _track_face_areas
        }
        track_pose_quality: dict[int, float] = {
            tid: float(np.mean(vals))
            for tid, vals in _track_pose_quality.items()
        }

        # Face size + duration: computed once, used by merge threshold AND
        # position fallback threshold selection below.
        face_areas = [
            ff.face_box_area for ff in frames
            if ff.face_detected and ff.face_box_area > 0.001
        ]
        avg_face_h = float(np.mean([a ** 0.5 for a in face_areas])) if face_areas else 0.10
        duration_s = max((ff.timestamp_ms for ff in frames), default=60000) / 1000.0

        if self._best_face_crops and len(self._best_face_crops) > 1:
            logger.info(
                f"ArcFace embedding: {len(self._best_face_crops)} face crops"
            )
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

                        _merge_offset = getattr(self, "_profile_merge_offset", 0.0)
                        merge_threshold = self._compute_merge_threshold(
                            duration_s, avg_face_h
                        ) + _merge_offset
                        logger.info(
                            f"Adaptive merge threshold: {merge_threshold:.3f} "
                            f"(duration={duration_s:.0f}s, avg_face_h={avg_face_h:.3f}, "
                            f"tracks_with_emb={len(track_embs)}, "
                            f"total_tracks={len(self._best_face_crops)})"
                        )

                        canonical = self._merge_tracks_by_embedding(
                            track_embs,
                            threshold=merge_threshold,
                            frame_counts=frame_counts,
                            track_face_heights=track_face_heights,
                            track_centroids=track_centroids,
                            pose_quality=track_pose_quality,
                        )
                        if _cut_timestamps:
                            canonical = self._cross_shot_merge(
                                canonical=canonical,
                                track_embeddings=track_embs,
                                track_first_ms=_track_first_ms,
                                track_last_ms=_track_last_ms,
                                cut_timestamps=_cut_timestamps,
                            )
                        merges = {k: v for k, v in canonical.items() if k != v}

                        if merges:
                            # Rewrite face_index on all frames
                            for ff in frames:
                                if ff.face_index in canonical:
                                    ff.face_index = canonical[ff.face_index]

                            # Keep only canonical crops; prefer the highest quality frame
                            # (frontal × large bbox × well-lit) across all merged tracks.
                            # Using raw crop.size (pixel count) picked the biggest region, not
                            # the clearest face — causing wrong-person thumbnails after merge.
                            merged_crops: dict[int, np.ndarray] = {}
                            merged_crop_quality: dict[int, float] = {}
                            for tid, crop in self._best_face_crops.items():
                                canon_tid = canonical.get(tid, tid)
                                q = best_quality.get(tid, 0.0)
                                if canon_tid not in merged_crop_quality or q > merged_crop_quality[canon_tid]:
                                    merged_crops[canon_tid] = crop
                                    merged_crop_quality[canon_tid] = q
                            self._best_face_crops = merged_crops

                            # Update cached embeddings to canonical only, then regenerate
                            # the thumbnail from the selected best crop so the stored image
                            # matches the face that was actually embedded (avoids mismatch).
                            merged_embs: dict[int, tuple[list[float], bytes]] = {}
                            for tid, data in self._cached_embeddings.items():
                                canon_tid = canonical.get(tid, tid)
                                if canon_tid not in merged_embs:
                                    merged_embs[canon_tid] = data
                            import cv2 as _cv2_t
                            for canon_tid, (emb_list, _) in list(merged_embs.items()):
                                if canon_tid in self._best_face_crops:
                                    crop = self._best_face_crops[canon_tid]
                                    thumb = _cv2_t.resize(crop, (96, 96), interpolation=_cv2_t.INTER_AREA)
                                    _, jpeg = _cv2_t.imencode(".jpg", thumb, [_cv2_t.IMWRITE_JPEG_QUALITY, 85])
                                    merged_embs[canon_tid] = (emb_list, jpeg.tobytes())
                            self._cached_embeddings = merged_embs

                        # NOTE: No remap of _active_tile_face_to_speaker through the
                        # ArcFace canonical dict is needed here. ArcFace merge (above)
                        # already rewrote ff.face_index on every frame to canonical IDs.
                        # The post-split rebuild (below) iterates these rewritten frames,
                        # so its output mapping references final canonical track IDs
                        # directly.

                        unique_count = len(set(canonical.values()))
                        logger.info(
                            f"Identity merge: {len(track_embs)} embedded tracks "
                            f"→ {unique_count} unique people "
                            f"(merged {len(merges)} tracks)"
                        )

                    # Tracks without an ArcFace embedding keep their own Face_N
                    # identity and pass through the transient filter independently.
                    tracks_without_emb = (
                        set(self._best_face_crops.keys())
                        - set(self._cached_embeddings.keys())
                    )
                    if tracks_without_emb:
                        logger.info(
                            f"{len(tracks_without_emb)} tracks had no ArcFace embedding "
                            f"— kept as independent Face_N identities"
                        )

            except Exception as exc:
                logger.warning(
                    f"Identity merge failed (non-fatal): {exc}"
                )

        # ── Remove true transient tracks ──────────────────────────────────────
        # After ArcFace embedding merge, remove any track that has
        # fewer than MIN_FRAMES_AFTER_MERGE frames. A track this brief does not
        # have enough data for reliable behavioural analysis regardless of whether
        # ArcFace extracted a crop — embeddings from 1-2 frames are unreliable
        # (person mid-turn, partially occluded). The old embedding-exemption
        # was the root cause of ghost faces reaching signal emission.
        # At 5 fps, 10 frames = 2 seconds = minimum for one complete window.
        MIN_FRAMES_AFTER_MERGE = 10
        final_counts: dict[int, int] = {}
        for ff in frames:
            if ff.face_detected:
                final_counts[ff.face_index] = final_counts.get(ff.face_index, 0) + 1

        true_transients = {
            tid for tid, cnt in final_counts.items()
            if cnt < MIN_FRAMES_AFTER_MERGE
        }
        if true_transients:
            before = len(frames)
            frames = [ff for ff in frames if ff.face_index not in true_transients]
            self._best_face_crops = {
                tid: crop for tid, crop in self._best_face_crops.items()
                if tid not in true_transients
            }
            self._cached_embeddings = {
                tid: data for tid, data in self._cached_embeddings.items()
                if tid not in true_transients
            }
            logger.info(
                f"Removed {len(true_transients)} transient tracks "
                f"(< {MIN_FRAMES_AFTER_MERGE} frames): "
                f"{sorted(true_transients)} — dropped {before - len(frames)} frames"
            )

        # ── Build face→speaker mapping from active-tile tags (post-split) ───────
        # Rebuild from FINAL frames (post-TrackletSplitter + post-ArcFace-merge
        # canonical IDs) so the mapping uses the track IDs windows actually see.
        # Applies the same gallery guard as the in-loop tagger: only tag frames
        # where one face clearly dominates (ratio >= 1.5 over second face).
        #
        # DSA: HashMap keyed by timestamp_ms → dominant face_index (or None)
        # built in O(F); per-frame lookup during tagging is O(1).
        if active_tagger is not None and active_tagger.tags:
            post_split_tagger = ActiveTileTagger(
                diar_segments=diar_segments,
                min_face_area=ActiveTileTagger.ACTIVE_TILE_MIN_AREA,
            )

            # Step 1: Group detected frames by timestamp, find dominant face per ts.
            frames_by_ts: dict[int, list] = defaultdict(list)
            for ff in frames:
                if ff.face_detected:
                    frames_by_ts[ff.timestamp_ms].append(ff)

            dominant_at_ts: dict[int, int | None] = {}
            for ts, ff_list in frames_by_ts.items():
                areas = sorted(
                    [ff.face_box_area for ff in ff_list], reverse=True
                )
                if not areas or areas[0] < ActiveTileTagger.ACTIVE_TILE_MIN_AREA:
                    dominant_at_ts[ts] = None
                    continue
                if len(areas) >= 2 and areas[0] / max(areas[1], 0.001) < 1.5:
                    dominant_at_ts[ts] = None  # gallery layout — no dominant face
                    continue
                dominant_at_ts[ts] = max(ff_list, key=lambda f: f.face_box_area).face_index

            # Step 2: Tag only the dominant face at each timestamp.
            for ff in frames:
                dom_idx = dominant_at_ts.get(ff.timestamp_ms)
                if dom_idx is not None and ff.face_index == dom_idx:
                    spk = post_split_tagger.find_speaker_at(ff.timestamp_ms)
                    if spk:
                        post_split_tagger.record_tag(ff.face_index, ff.timestamp_ms, spk)

            self._active_tile_face_to_speaker: dict[int, str] = post_split_tagger.build_mapping()
        else:
            self._active_tile_face_to_speaker = {}
        if active_tagger is not None:
            active_tagger.clear()

        # ── Remap face crops to canonical (post-split + post-ArcFace) IDs ──────
        # _crop_by_position uses (ts_ms, cx_int, cy_int) — invariant through splits.
        # Iterating final `frames` gives us each ff's canonical face_index, allowing
        # a clean O(F) rebuild without a second VideoCapture pass.
        _face_crops_seq: "dict[int, list[tuple[int, np.ndarray]]]" = defaultdict(list)
        for ff in frames:
            if not ff.face_detected or ff.face_box_area < 0.001:
                continue
            pos_key = (
                ff.timestamp_ms,
                int(ff.face_centre_x * 1000),
                int(ff.face_centre_y * 1000),
            )
            crop = _crop_by_position.get(pos_key)
            if crop is not None:
                _face_crops_seq[ff.face_index].append((ff.timestamp_ms, crop))
        for seq in _face_crops_seq.values():
            seq.sort(key=lambda x: x[0])
        self._face_crops_sequence: "dict[int, list[tuple[int, np.ndarray]]]" = dict(
            _face_crops_seq
        )
        self._last_video_fps: float = float(self._target_fps)
        del _crop_by_position   # free ~9KB × n_face_frames before returning

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

    def warmup(self) -> None:
        """
        Pre-load all heavy models so the first real session has no cold-start delay.

        Two stages run sequentially in the caller's thread-pool thread:
          1. MediaPipe — builds + discards a TiledFrameProcessor to page the 6
             model files into the OS cache (~2-5s, CPU-bound).
          2. ArcFace (InsightFace buffalo_l) — initialises the singleton
             FaceEmbeddingExtractor, which downloads the model weights if absent
             and calls app.prepare() to load ONNX models into memory (~2-5s on
             first run, ~0.5s on subsequent runs from disk cache).

        Both stages are non-fatal: a failure logs a warning and the first real
        session falls back to cold initialisation.
        """
        try:
            import mediapipe as mp
            proc = self._build_tiled_processor(mp)
            del proc
            logger.info("MediaPipe warmup complete — models page-cached.")
        except Exception as exc:
            logger.warning("MediaPipe warmup failed (non-fatal): %s", exc)

        try:
            embedder = FaceEmbeddingExtractor.get_instance()
            if embedder.available:
                logger.info("ArcFace warmup complete — buffalo_l model loaded.")
            else:
                logger.warning("ArcFace warmup: InsightFace not available — face ID disabled.")
        except Exception as exc:
            logger.warning("ArcFace warmup failed (non-fatal): %s", exc)

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
          > 0.20  → grid / video-call    (large faces filling tiles)
          ≤ 0.20  → conference room      (small faces, many angles, people lean/turn)

        Grid calls:
          - High-quality frontal embeddings → scores 0.70+
          - Base 0.50-0.65, decay 0.012/min, floor 0.42

        Conference rooms:
          - Small faces, profile angles, variable lighting → same-person 0.40-0.65
          - Base 0.50, decay 0.008/min (more variation in longer rooms)
          - Floor 0.40 — same-person pairs at extreme angles score as low as 0.40;
            different-person pairs typically score < 0.35 in conference conditions.
            (floor was 0.48 — too strict, left too many fragments unmerged)
        """
        duration_min = duration_s / 60.0
        if avg_face_height_ratio > 0.20:
            base  = min(0.65, max(0.50, 0.35 + avg_face_height_ratio * 1.5))
            decay = duration_min * 0.012
            floor = 0.42
        else:
            base  = 0.50
            decay = duration_min * 0.008
            floor = 0.40
        return round(max(floor, base - decay), 3)

    @staticmethod
    def _merge_tracks_by_embedding(
        track_embeddings: dict[int, "np.ndarray"],
        threshold: float = 0.65,
        frame_counts: dict[int, int] | None = None,
        track_face_heights: dict[int, float] | None = None,
        track_centroids: dict[int, tuple[float, float]] | None = None,
        pose_quality: dict[int, float] | None = None,
    ) -> dict[int, int]:
        """
        Merge CentroidTracker track_ids that belong to the same physical person.

        Embeddings are L2-normalised in extract_from_crops so np.dot() here
        equals cosine similarity in [-1, 1].  Threshold is computed adaptively
        by _compute_merge_threshold based on video duration and face size.

        Tracks are processed dominant-first (most frames → canonical anchor),
        so fragment tracks are always compared against the highest-quality
        embedding rather than an arbitrary earlier track.

        Tiny-face guard: when either track in a pair has avg face height < 0.07
        (face_area < 0.005, ~70px in 1080p), ArcFace produces noisy embeddings
        that score 0.40-0.49 against unrelated faces. The effective threshold is
        raised (floor 0.55 symmetric, 0.45 asymmetric) for those pairs.

        Pose discount: off-angle tracks (|yaw| > 30°) have degraded embeddings.
        _effective_thresh lowers the floor by up to 0.05 so same-person profile
        pairs can merge when embedding noise is from pose, not identity mismatch.

        Returns {track_id: canonical_track_id} for ALL input track_ids.
        Tracks that match nothing keep their own ID as canonical.
        """
        import logging as _logging
        _log = _logging.getLogger("nexus.video.features")

        canonical: dict[int, int] = {}
        canon_embs: dict[int, "np.ndarray"] = {}

        counts = frame_counts or {}
        face_hs = track_face_heights or {}
        centroids = track_centroids or {}
        pq = pose_quality or {}

        def _effective_thresh(tid_a: int, tid_b: int, sim: float) -> float:
            """Return merge threshold for this track pair, with tiny-face guard,
            pose-quality discount, and position-based fallback."""
            fh_a = face_hs.get(tid_a, 1.0)
            fh_b = face_hs.get(tid_b, 1.0)
            min_fh = min(fh_a, fh_b)
            max_fh = max(fh_a, fh_b)

            # Pose discount: poor pose (profile angle) degrades embedding quality.
            # If either track is off-angle, lower the threshold by up to 0.05 so
            # same-person profile pairs aren't blocked by embedding noise.
            # max discount = (0.7 - 0.0) * 0.10 = 0.07, capped conservatively.
            pq_a = pq.get(tid_a, 1.0)
            pq_b = pq.get(tid_b, 1.0)
            pose_discount = max(0.0, (0.7 - min(pq_a, pq_b)) * 0.10)

            if min_fh >= 0.07:
                # Both normal-sized: standard threshold minus pose discount
                effective = max(threshold - pose_discount, 0.35)
                # Cross-tile guard: if two normal-sized tracks sit at clearly
                # different grid positions (dist > 0.20) with similarity below 0.55,
                # they are almost certainly different people. Raise the bar to 0.55
                # to block false merges (sim 0.40-0.49) without affecting same-tile
                # same-person pairs (dist < 0.05) or high-confidence matches.
                if sim < 0.55 and centroids:
                    ca = centroids.get(tid_a)
                    cb = centroids.get(tid_b)
                    if ca and cb:
                        pos_dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                        if pos_dist > 0.20:
                            effective = max(effective, 0.55)
                return effective
            if max_fh < 0.07:
                # SYMMETRIC: both tiny → strict floor minus small pose discount
                floor = max(threshold, 0.55) - pose_discount
                floor = max(floor, 0.40)
            else:
                # ASYMMETRIC: one tiny (noisy), one large (clean).
                floor = max(threshold, 0.45) - pose_discount
                floor = max(floor, 0.35)
            if sim >= floor:
                return floor
            # Score below floor but above noise floor (0.35): use grid position
            # as a second signal. Same grid tile (dist < 0.05) means same person.
            if sim > 0.35 and centroids:
                ca = centroids.get(tid_a)
                cb = centroids.get(tid_b)
                if ca and cb:
                    pos_dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                    if pos_dist < 0.05:
                        return threshold  # same tile → allow merge
            return floor

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
                fh_tid = face_hs.get(tid, 1.0)
                fh_top = face_hs.get(top_tid, 1.0)
                eff_thresh = _effective_thresh(tid, top_tid, top)
                pos_note = ""
                if min(fh_tid, fh_top) < 0.07 and top < eff_thresh and top > 0.35:
                    ca, cb = centroids.get(tid), centroids.get(top_tid)
                    if ca and cb:
                        d = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                        pos_note = f" pos_dist={d:.3f}{'(grid-match)' if d < 0.05 else ''}"
                _log.info(
                    f"Track {tid:>2} vs canonicals: best={top:.3f} (vs track {top_tid}) "
                    f"thresh={eff_thresh:.3f} (face_h: {fh_tid:.3f}×{fh_top:.3f}){pos_note} "
                    f"— {'MERGE' if top > eff_thresh else 'keep separate'}"
                )

            for canon_tid, sim in scores.items():
                eff = _effective_thresh(tid, canon_tid, sim)
                if sim > eff and sim > best_sim:
                    best_sim = sim
                    matched_canon = canon_tid

            if matched_canon is not None:
                canonical[tid] = matched_canon
            else:
                canonical[tid] = tid
                canon_embs[tid] = emb

        return canonical

    @staticmethod
    def _cross_shot_merge(
        canonical: dict[int, int],
        track_embeddings: "dict[int, np.ndarray]",
        track_first_ms: dict[int, int],
        track_last_ms: dict[int, int],
        cut_timestamps: list[int],
        threshold: float = 0.40,
    ) -> dict[int, int]:
        """
        Link canonical track groups that span scene cut boundaries.

        _merge_tracks_by_embedding() handles same-shot same-person merges.
        This pass handles the cross-shot case: the same person reappears
        in a different camera angle after a cut. Three conditions must hold:
          1. Groups are temporally non-overlapping (temporal exclusivity).
          2. The gap between them straddles at least one cut timestamp.
          3. Cosine similarity of representative embeddings >= threshold.

        threshold=0.40 is lower than the main merge threshold because temporal
        exclusivity removes identity ambiguity and cut-induced pose/lighting
        changes depress ArcFace similarity ~0.05-0.10 below the true value.
        """
        if not cut_timestamps or not track_embeddings:
            return canonical

        from collections import defaultdict

        groups: dict[int, list[int]] = defaultdict(list)
        for tid, cid in canonical.items():
            groups[cid].append(tid)

        group_first: dict[int, int]           = {}
        group_last:  dict[int, int]           = {}
        group_emb:   "dict[int, np.ndarray]"  = {}
        group_size:  dict[int, int]           = {}

        for cid, members in groups.items():
            firsts = [track_first_ms[t] for t in members if t in track_first_ms]
            lasts  = [track_last_ms[t]  for t in members if t in track_last_ms]
            if not firsts:
                continue
            group_first[cid] = min(firsts)
            group_last[cid]  = max(lasts)
            group_size[cid]  = sum(1 for t in members if t in track_first_ms)
            if cid in track_embeddings:
                group_emb[cid] = track_embeddings[cid]
            else:
                for m in members:
                    if m in track_embeddings:
                        group_emb[cid] = track_embeddings[m]
                        break

        valid_cids = [c for c in group_emb if c in group_first]
        if len(valid_cids) < 2:
            return canonical

        parent: dict[int, int] = {c: c for c in valid_cids}

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return
            if group_size.get(ra, 0) >= group_size.get(rb, 0):
                parent[rb] = ra
            else:
                parent[ra] = rb

        valid_cids_sorted = sorted(valid_cids, key=lambda c: group_size.get(c, 0), reverse=True)
        merge_count = 0

        for i, cid_a in enumerate(valid_cids_sorted):
            emb_a = group_emb.get(cid_a)
            if emb_a is None:
                continue
            fa = group_first[cid_a]
            la = group_last[cid_a]
            for cid_b in valid_cids_sorted[i + 1:]:
                if _find(cid_a) == _find(cid_b):
                    continue
                emb_b = group_emb.get(cid_b)
                if emb_b is None:
                    continue
                fb = group_first[cid_b]
                lb = group_last[cid_b]
                if min(la, lb) > max(fa, fb):
                    continue  # tracks overlap in time — cannot be same person
                gap_start = min(la, lb)
                gap_end   = max(fa, fb)
                if not any(gap_start <= ct <= gap_end for ct in cut_timestamps):
                    continue  # gap contains no scene cut
                sim = float(np.dot(emb_a, emb_b))
                if sim >= threshold:
                    _union(cid_a, cid_b)
                    merge_count += 1
                    logger.info(
                        "Cross-shot merge: %d ← %d (sim=%.3f, gap=[%dms, %dms])",
                        _find(cid_a), cid_b, sim, gap_start, gap_end,
                    )

        if merge_count == 0:
            return canonical

        result: dict[int, int] = {}
        for tid, cid in canonical.items():
            result[tid] = _find(cid) if cid in parent else cid
        return result


# ══════════════════════════════════════════════════════════════════════════════
# ActiveSpeakerBorderDetector  — platform-native border color detection
# ══════════════════════════════════════════════════════════════════════════════

class ActiveSpeakerBorderDetector:
    """
    Detects the platform-drawn active-speaker highlight border around face tiles.

    Zoom draws a yellow (#FFD700) or blue (#0E71EB) border ~3-4px wide.
    Google Meet draws a blue (#1A73E8) or white border ~2-3px wide.
    Microsoft Teams uses a thin colored ring.

    Detection: sample pixels along the perimeter of each face bounding box.
    If >30% of perimeter pixels match the highlight color (HSV range),
    this face is the platform-asserted active speaker.

    Strategy Pattern: platform-specific HSV ranges are configurable via
    the platform constructor arg; 'auto' tries all known ranges.

    O(P) per face per frame where P = perimeter pixel count (~200-400).
    Total per session: ~6300 frames × 3 faces × 300 pixels = negligible.
    """

    ZOOM_YELLOW_HSV  = ((20,  150, 150), (35,  255, 255))
    ZOOM_BLUE_HSV    = ((100, 150, 150), (120, 255, 255))
    MEET_BLUE_HSV    = ((100, 100, 150), (125, 255, 255))
    MEET_WHITE_HSV   = ((0,   0,   200), (180, 30,  255))

    def __init__(self, platform: str = "auto") -> None:
        self._platform = platform
        if platform == "zoom":
            self._hsv_ranges = [self.ZOOM_YELLOW_HSV, self.ZOOM_BLUE_HSV]
        elif platform == "meet":
            self._hsv_ranges = [self.MEET_BLUE_HSV, self.MEET_WHITE_HSV]
        elif platform == "teams":
            self._hsv_ranges = [self.MEET_BLUE_HSV]
        else:  # auto — try all
            self._hsv_ranges = [
                self.ZOOM_YELLOW_HSV, self.ZOOM_BLUE_HSV,
                self.MEET_BLUE_HSV,   self.MEET_WHITE_HSV,
            ]

    def detect_active_speaker(
        self,
        bgr: "np.ndarray",
        face_boxes: list[tuple[int, int, int, int]],
        border_width: int = 5,
        match_ratio: float = 0.30,
    ) -> int | None:
        """
        Return the index into face_boxes of the face with an active-speaker border,
        or None if no border detected.

        Two-pass strategy:
          Pass 1 — face-proximity: sample border_width pixels outside each face box.
                   Works for platforms that draw borders close to the face (Teams rings).
          Pass 2 — frame-edge: sample the outermost TILE_EDGE_PX rows/cols of the frame.
                   Works for Zoom/Meet active_speaker layout where the highlighted tile
                   fills most of the frame and its border is at the frame edge, far from
                   the face centre. Assigns the border to whichever face is closest to
                   the frame centre (the dominant tile's face).

        Args:
            bgr:          full frame (BGR)
            face_boxes:   [(x, y, w, h), ...] pixel bounding boxes
            border_width: pixels outside the face box to sample (Pass 1)
            match_ratio:  fraction of perimeter pixels that must match
        """
        import cv2

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        fh, fw = bgr.shape[:2]

        def _color_match_ratio(pixels: "np.ndarray") -> float:
            if pixels.size == 0:
                return 0.0
            best = 0
            for lo, hi in self._hsv_ranges:
                mask = cv2.inRange(
                    pixels.reshape(1, -1, 3),
                    np.array(lo), np.array(hi),
                )
                best = max(best, int(np.count_nonzero(mask)))
            return best / len(pixels)

        # ── Tile-aware border sampling ────────────────────────────────────────
        # Video call platforms draw the active-speaker border around the TILE,
        # not around the face. Face typically occupies ~50% of each tile dimension,
        # so the tile border sits ~face_size * 0.5 pixels outside the face box.
        # Scale the sampling distance proportionally so both large main tiles and
        # small sidebar tiles are covered without needing separate passes.
        best_idx: int | None = None
        best_match: float = 0.0

        for idx, (x, y, w, h) in enumerate(face_boxes):
            # Tile border is roughly half a face-width outside the face bbox.
            # Clamp to [border_width, 200] to avoid absurd values on huge/tiny faces.
            tile_pad = max(border_width, min(200, int(max(w, h) * 0.55)))

            x1 = max(0, x - tile_pad)
            y1 = max(0, y - tile_pad)
            x2 = min(fw, x + w + tile_pad)
            y2 = min(fh, y + h + tile_pad)

            # Sample only the outer ring of the expanded region (8px thick) so
            # interior content doesn't dilute the border color ratio.
            ring = 8
            strips = [
                hsv[y1:y1 + ring,    x1:x2       ].reshape(-1, 3),   # top ring
                hsv[y2 - ring:y2,    x1:x2       ].reshape(-1, 3),   # bottom ring
                hsv[y1:y2,           x1:x1 + ring].reshape(-1, 3),   # left ring
                hsv[y1:y2,    x2 - ring:x2       ].reshape(-1, 3),   # right ring
            ]
            all_border = np.vstack([s for s in strips if s.size > 0])
            ratio = _color_match_ratio(all_border)
            if ratio > match_ratio and ratio > best_match:
                best_match = ratio
                best_idx = idx

        return best_idx


# ══════════════════════════════════════════════════════════════════════════════
# ActiveTileTagger  — diarization-based face→speaker tags for prominent tiles
# ══════════════════════════════════════════════════════════════════════════════

class ActiveTileTagger:
    """
    Tags large-tile faces to diarization speakers using bisect-based O(log N) lookup.

    Strategy:
      1. During the frame loop, tag_frame() calls _find_speaker_at(timestamp_ms) for
         every face whose face_box_area >= ACTIVE_TILE_MIN_AREA (prominent/pinned tile).
         This uses bisect_right on sorted interval start times — O(log N) per frame.
      2. After the loop, build_mapping() applies a Counter majority vote per track ID
         to produce {track_id: "Speaker_N"}.
      3. _extract_frames calls tag_frame() in-loop with pre-split face_index values,
         then rebuilds from the final (post-split) frames so track IDs are canonical.
      4. After ArcFace merge, _active_tile_face_to_speaker is remapped through the
         canonical dict so pre-merge track IDs map to their merged canonical ID.

    Rationale: Large tiles (face_area > 8%) produce jaw saturation in MediaPipe
    blendshapes — the jaw is nearly always open, so lip-sync correlation near zero.
    Direct diarization time-overlap is more reliable for these faces than lip-sync.
    """

    ACTIVE_TILE_MIN_AREA: float = 0.08

    def __init__(
        self,
        diar_segments: list[dict],
        min_face_area: float = ACTIVE_TILE_MIN_AREA,
    ) -> None:
        self._min_face_area = min_face_area
        # Pre-sort intervals once; bisect reuses the sorted list each frame.
        self._intervals: list[tuple[int, int, str]] = sorted(
            (
                (
                    int(s.get("start_ms", 0)),
                    int(s.get("end_ms", 0)),
                    str(s.get("speaker", "")),
                )
                for s in diar_segments
                if s.get("speaker") and s.get("start_ms") is not None
            ),
            key=lambda x: x[0],
        )
        # Parallel list of start times for bisect (avoids attribute access in hot path)
        self._interval_starts: list[int] = [iv[0] for iv in self._intervals]
        # {track_id: [(timestamp_ms, speaker_label), ...]}
        self._tags: dict[int, list[tuple[int, str]]] = defaultdict(list)

    @property
    def tags(self) -> dict[int, list[tuple[int, str]]]:
        return self._tags

    def reset(self) -> None:
        """Clear accumulated frame tags on layout change events."""
        self.clear()

    def record_tag(self, face_index: int, timestamp_ms: int, speaker: str) -> None:
        """Record one face→speaker tag directly — used by the post-split rebuild
        which has already resolved the canonical face_index externally."""
        self._tags[face_index].append((timestamp_ms, speaker))

    def find_speaker_at(self, timestamp_ms: int) -> str:
        """Public accessor for diarization speaker at timestamp_ms."""
        return self._find_speaker_at(timestamp_ms)

    def tag_frame(
        self,
        frame_features_list: list,
        timestamp_ms: int,
        border_face_idx: "int | None" = None,
    ) -> None:
        """
        Tag the active-speaker face in this frame to the diarization speaker.

        border_face_idx: track_id identified by ActiveSpeakerBorderDetector.
          When provided, this is a platform-native signal (more reliable than tile
          size ratio) and bypasses the gallery guard — used in gallery layout where
          all tiles are equal-sized and the ratio check would always skip.
        """
        spk = self._find_speaker_at(timestamp_ms)
        if not spk:
            return

        # Platform-drawn border = definitive active-speaker signal.
        # Bypasses the gallery guard; does not also apply size-based tags
        # for this frame to avoid double-counting.
        if border_face_idx is not None:
            self._tags[border_face_idx].append((timestamp_ms, spk))
            return

        # Size-based dominant-tile detection (existing logic).
        # Gallery/grid views (2×2, 3×3) have all tiles within ~1.1× of each other —
        # no single "active tile" exists so diarization-based tagging would be noise.
        areas = sorted(
            [ff.face_box_area for ff in frame_features_list if ff.face_detected],
            reverse=True,
        )
        if not areas or areas[0] < self._min_face_area:
            return  # all faces too small (screenshare / no speaker tile)
        if len(areas) >= 2 and areas[0] / max(areas[1], 0.001) < 1.5:
            return  # gallery view — tiles roughly equal, no dominant speaker

        # Tag ONLY the single largest face (the active-speaker tile).
        # Tagging every face >= min_face_area was Bug 1: in a 3-person call
        # a sidebar face at area=0.10 would also pass >= 0.08 and accumulate
        # tags from whichever speaker talked longest, corrupting majority vote.
        largest = max(
            (ff for ff in frame_features_list if ff.face_detected),
            key=lambda f: f.face_box_area,
            default=None,
        )
        if largest is not None:
            self._tags[largest.face_index].append((timestamp_ms, spk))

    def _find_speaker_at(self, timestamp_ms: int) -> str:
        """Return the diarization speaker active at timestamp_ms, or '' if none."""
        idx = bisect.bisect_right(self._interval_starts, timestamp_ms) - 1
        if idx < 0:
            return ""
        start, end, speaker = self._intervals[idx]
        return speaker if start <= timestamp_ms < end else ""

    def build_mapping(self) -> dict[int, str]:
        """
        Majority-vote per track ID → {track_id: "Speaker_N"}.
        Only tracks where the majority speaker starts with 'Speaker_' are included.
        """
        mapping: dict[int, str] = {}
        for tid, tag_list in self._tags.items():
            if not tag_list:
                continue
            majority_speaker, _ = Counter(spk for _, spk in tag_list).most_common(1)[0]
            if majority_speaker.startswith("Speaker_"):
                mapping[tid] = majority_speaker
        return mapping

    def clear(self) -> None:
        self._tags.clear()


class IntervalSet:
    """Sorted interval container with O(log n) point membership test.

    Diarization segments for a single speaker are sorted by start time at
    construction. contains() uses bisect_right to find the candidate interval
    in O(log n) instead of the O(n) linear scan that _in_intervals() used.

    DSA: binary search on sorted start times; at most one candidate interval
    to inspect per query (assumes non-overlapping diarization segments).
    """

    __slots__ = ("_intervals", "_starts")

    def __init__(self, intervals: list[tuple[int, int]]) -> None:
        self._intervals: list[tuple[int, int]] = sorted(intervals, key=lambda x: x[0])
        self._starts: list[int] = [s for s, _ in self._intervals]

    def contains(self, ts_ms: int) -> bool:
        """Return True if ts_ms falls within any stored interval. O(log n)."""
        idx = bisect.bisect_right(self._starts, ts_ms) - 1
        if idx >= 0:
            _, end = self._intervals[idx]
            return ts_ms <= end
        return False

    @classmethod
    def from_segments(cls, segments: list[dict], speaker: str) -> "IntervalSet":
        """Build from diarization segment dicts filtered to one speaker."""
        return cls([
            (seg.get("start_ms", 0), seg.get("end_ms", 0))
            for seg in segments
            if seg.get("speaker") == speaker
        ])


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

    def __init__(self) -> None:
        # Pre-computed face→speaker tags from VideoFeatureExtractor._extract_frames.
        # Injected by run_analysis via set_active_tile_tags() before assign() is called.
        self._active_tile_face_to_speaker: dict[int, str] = {}

    def set_active_tile_tags(self, tags: dict[int, str]) -> None:
        """Inject active-tile face→speaker tags from the extractor into the mapper."""
        self._active_tile_face_to_speaker = dict(tags)

    def assign(
        self,
        windows: list[WindowFeatures],
        diar_segments: list[dict],
        lip_activity_map: Optional[dict[int, list[tuple[int, float]]]] = None,
        asd_scores: Optional[dict[int, list[tuple[int, float]]]] = None,
        skip_speaker_link: bool = False,
    ) -> tuple[dict[str, list[WindowFeatures]], dict[str, float], dict[int, str]]:
        """
        Args:
            windows:          WindowFeatures from VideoFeatureExtractor.extract_all().
            diar_segments:    [{speaker, start_ms, end_ms}, ...] from voice agent.
            lip_activity_map: {face_index: [(timestamp_ms, lip_score), ...]}
                              from VideoFeatureExtractor.build_lip_activity_map().
                              Used when asd_scores is None.
            asd_scores:       {face_index: [(timestamp_ms, 0.0_or_1.0), ...]}
                              from LightASDClassifier.score().  When provided, used
                              in preference to lip_activity_map (more accurate).

        Returns:
            (windows_by_face, lip_sync_scores, face_to_speaker) where:
            - windows_by_face:  {Face_N: [WindowFeatures, ...]} — ownership by face track
            - lip_sync_scores:  {Speaker_N: correlation_score} — accepted linkage scores
            - face_to_speaker:  {face_index_int: "Speaker_N"} — conservative linkage map
              Exported only for sufficiently positive scores; time-overlap fallback is
              used for is_speaking state only, not identity linking.

        Assignment priority: asd > lip_sync > time_overlap
        """
        result: dict[str, list[WindowFeatures]] = defaultdict(list)

        if not windows:
            return dict(result), {}, {}

        # Interrogation videos: skip all speaker-face linking. Group windows by
        # Face_N only — is_speaking stays False (no lip-sync or diarization
        # correlation). face_to_speaker returned empty so the gateway performs
        # no remapping.
        if skip_speaker_link:
            for wf in windows:
                face_label = f"Face_{getattr(wf, 'face_index', 0)}"
                wf.speaker_id = face_label
                wf.is_speaking = False
                result[face_label].append(wf)
            logger.info(
                "SpeakerFaceMapper: skip_speaker_link=True — %d face track(s), no speaker linkage",
                len(result),
            )
            return dict(result), {}, {}

        speakers = sorted(set(seg.get("speaker", "Speaker_0") for seg in diar_segments))
        if not speakers:
            speakers = ["Speaker_0"]

        face_indices  = sorted(set(getattr(wf, "face_index", 0) for wf in windows))
        multi_face    = len(face_indices) > 1
        multi_speaker = len(speakers) > 1

        # ASD is reliable with any number of speakers — binary speaking labels
        # work even with one speaker (the speaker's face scores high, others low).
        # Lip-sync correlation needs a silence baseline that only exists when
        # multiple speakers alternate, so it requires multi_speaker as well.
        use_asd      = asd_scores is not None and multi_face
        use_lip_sync = not use_asd and lip_activity_map is not None and multi_face and multi_speaker

        if use_asd:
            face_to_speaker, assignment_scores = self._asd_assignment(
                face_indices, speakers, diar_segments, asd_scores  # type: ignore[arg-type]
            )
            method = "asd"
            confident_face_to_speaker: dict[int, str] = {
                fi: spk
                for fi, spk in face_to_speaker.items()
                if spk.startswith("Speaker_")
                and assignment_scores.get(fi, 0.0) >= MIN_LIP_SYNC_LINK_SCORE
            }
        elif use_lip_sync:
            face_to_speaker, assignment_scores = self._lip_sync_assignment(
                face_indices, speakers, diar_segments, lip_activity_map  # type: ignore[arg-type]
            )
            method = "lip_sync"
            # Only promote assignments that cleared the confidence bar.
            # Low-score assignments exist even when correlation is near zero.
            # Unconfident faces get "" → is_speaking=False (neutral) rather than
            # a wrong diarization-derived value.
            confident_face_to_speaker = {
                fi: spk
                for fi, spk in face_to_speaker.items()
                if spk.startswith("Speaker_")
                and assignment_scores.get(fi, 0.0) >= MIN_LIP_SYNC_LINK_SCORE
            }
        else:
            face_to_speaker, assignment_scores = self._time_overlap_assignment(
                face_indices, speakers, windows, diar_segments
            )
            method = "time_overlap"
            # time_overlap is reliable (single-speaker or no-data sessions) —
            # use its mapping directly for is_speaking.
            confident_face_to_speaker = {
                fi: spk for fi, spk in face_to_speaker.items()
                if spk.startswith("Speaker_")
            }

        # Merge active-tile pre-assignments for any face not already in
        # confident_face_to_speaker.  Active tiles (large faces) get tagged by
        # diarization time-overlap in _extract_frames because lip-sync is unreliable
        # there (jaw saturation).  They take effect only when lip-sync produced no
        # confident result for that face, so a strong lip-sync score still wins.
        for fi, spk in self._active_tile_face_to_speaker.items():
            if fi not in confident_face_to_speaker and spk.startswith("Speaker_"):
                confident_face_to_speaker[fi] = spk
                assignment_scores[fi] = 1.0  # sentinel — pre-assigned; always overwrite stale score

        # Group windows by Face_N — signal ownership stays on face tracks, not
        # voice speakers. face_to_speaker is returned as a linkage map for the
        # gateway to remap face_embeddings keys before registry matching.
        # is_speaking uses confident_face_to_speaker only — unconfident lip-sync
        # assignments are not used to avoid wrong audio-pipeline data on face signals.
        for wf in windows:
            face_idx = getattr(wf, "face_index", 0)
            face_label = f"Face_{face_idx}"
            wf.speaker_id = face_label
            wf.is_speaking = self._is_speaking_in_window(
                wf.window_start_ms, wf.window_end_ms,
                confident_face_to_speaker.get(face_idx, ""),
                diar_segments,
            )
            result[face_label].append(wf)

        # lip_sync_scores: face→speaker linkage exported for gateway registry matching
        # confident_face_to_speaker already has the threshold applied — reuse it.
        lip_sync_scores: dict[str, float] = {}
        confident_mapping: dict[int, str] = {}
        if method in ("asd", "lip_sync"):
            for fi, spk in confident_face_to_speaker.items():
                score = assignment_scores.get(fi, 0.0)
                lip_sync_scores[spk] = round(score, 4)
                confident_mapping[fi] = spk
        elif method == "time_overlap":
            # Export any face with positive speaking-time overlap. Sentinel score 1.0
            # is above SESSION_FACE_LOCK_MIN_SCORE so the gateway treats it as a
            # confirmed face→speaker link for registry matching.
            for fi, spk in confident_face_to_speaker.items():
                score = assignment_scores.get(fi, 0.0)
                if score > 0.0:
                    lip_sync_scores[spk] = 1.0
                    confident_mapping[fi] = spk

        # Active-tile entries (merged above) bypass the asd/lip_sync/time_overlap
        # branches so lip_sync_scores is never set for them. _build_session_face_locks
        # requires lip_sync_scores to be non-empty and rejects entries with score 0.0.
        # Export with sentinel 1.0 so the gateway accepts the face→speaker linkage.
        for fi, spk in self._active_tile_face_to_speaker.items():
            if fi in confident_face_to_speaker and spk not in lip_sync_scores:
                lip_sync_scores[spk] = 1.0
                if fi not in confident_mapping:
                    confident_mapping[fi] = spk

        logger.info(
            "SpeakerFaceMapper: %d face(s), method=%s, exported_linkage=%s",
            len(face_indices), method, confident_mapping,
        )
        # ASD and lip_sync both use the same export structure
        return dict(result), lip_sync_scores, confident_mapping

    def _asd_assignment(
        self,
        face_indices: list[int],
        speakers: list[str],
        diar_segments: list[dict],
        asd_scores: dict[int, list[tuple[int, float]]],
    ) -> tuple[dict[int, str], dict[int, float]]:
        """
        Aggregate Viterbi-smoothed ASD binary labels into per-(face, speaker) scores.

        Same correlation structure as _lip_sync_assignment:
            score = mean(speaking_label) during speaker-active frames
                  - mean(speaking_label) during speaker-silent frames

        ASD labels are binary (0.0 / 1.0) — derived from the Light-ASD model output
        after Viterbi smoothing.  More reliable than MediaPipe jawOpen for:
          • Small tiles   — jawOpen landmarks noisy below face_area 0.02
          • Large tiles   — jaw saturation makes jawOpen near-constant
          • Side profiles — jawOpen not visible above 60° yaw
          • Masked faces  — ASD uses audio-visual fusion, not mouth shape alone

        Delegates to _hungarian_assign for globally-optimal face→speaker pairing.
        """
        # Build IntervalSet per speaker once — O(k log k) sort paid once,
        # then each contains() call is O(log k) instead of O(k) linear scan.
        speaker_isets: dict[str, IntervalSet] = {
            spk: IntervalSet.from_segments(diar_segments, spk)
            for spk in speakers
        }

        scores: dict[tuple[int, str], float] = {}
        for face_idx in face_indices:
            track_data = asd_scores.get(face_idx, [])
            if not track_data:
                continue
            for speaker in speakers:
                iset = speaker_isets[speaker]
                active_sum = 0.0
                silent_sum = 0.0
                active_n   = 0
                silent_n   = 0
                for ts_ms, label in track_data:
                    if iset.contains(ts_ms):
                        active_sum += label
                        active_n   += 1
                    else:
                        silent_sum += label
                        silent_n   += 1
                avg_active = active_sum / max(active_n, 1)
                avg_silent = silent_sum / max(silent_n, 1)
                scores[(face_idx, speaker)] = avg_active - avg_silent
                logger.debug(
                    "ASD score  face=%d × %s: %.4f  (active=%.4f  silent=%.4f)",
                    face_idx, speaker,
                    avg_active - avg_silent, avg_active, avg_silent,
                )

        return self._hungarian_assign(face_indices, speakers, scores)

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
        # Build IntervalSet per speaker once — O(k log k) sort, O(log k) per lookup.
        speaker_isets: dict[str, IntervalSet] = {
            spk: IntervalSet.from_segments(diar_segments, spk)
            for spk in speakers
        }

        scores: dict[tuple[int, str], float] = {}

        for face_idx in face_indices:
            lip_data = lip_activity_map.get(face_idx, [])
            if not lip_data:
                continue

            for speaker in speakers:
                iset = speaker_isets[speaker]
                speaking_sum = 0.0
                silence_sum  = 0.0
                speaking_n   = 0
                silence_n    = 0

                for ts_ms, lip_score in lip_data:
                    if iset.contains(ts_ms):
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

        return self._hungarian_assign(face_indices, speakers, scores)

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

        return self._hungarian_assign(face_indices, speakers, scores)

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
    def _hungarian_assign(
        face_indices: list[int],
        speakers: list[str],
        scores: dict[tuple[int, str], float],
    ) -> tuple[dict[int, str], dict[int, float]]:
        """
        Globally optimal face→speaker assignment via the Hungarian algorithm.

        Builds an n_faces × n_speakers cost matrix (negated scores — scipy minimises),
        then calls scipy.optimize.linear_sum_assignment.  Falls back to _greedy_assign
        if scipy is unavailable.

        Complexity: O(n³) where n = max(faces, speakers); negligible for n ≤ 10.
        Advantage over greedy: avoids locally-optimal but globally-suboptimal pairings
        that occur when the best face for Speaker_0 is also the best face for Speaker_1.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            logger.debug("scipy not available — falling back to greedy assignment")
            return SpeakerFaceMapper._greedy_assign(face_indices, speakers, scores)

        if not face_indices or not speakers:
            return SpeakerFaceMapper._greedy_assign(face_indices, speakers, scores)

        n_f = len(face_indices)
        n_s = len(speakers)
        # Build cost matrix: negate scores so minimisation = maximisation
        cost = np.full((n_f, n_s), fill_value=0.0)
        for fi_idx, fi in enumerate(face_indices):
            for spk_idx, spk in enumerate(speakers):
                cost[fi_idx, spk_idx] = -scores.get((fi, spk), 0.0)

        row_ind, col_ind = linear_sum_assignment(cost)

        mapping: dict[int, str] = {}
        assignment_scores: dict[int, float] = {}
        assigned_faces: set[int] = set()

        for ri, ci in zip(row_ind, col_ind):
            fi = face_indices[ri]
            spk = speakers[ci]
            sc = scores.get((fi, spk), 0.0)
            # Only accept positive-correlation assignments; negatives mean no real link
            if sc > 0.0:
                mapping[fi] = spk
                assignment_scores[fi] = sc
                assigned_faces.add(fi)

        # Leftover faces (more faces than speakers, or score ≤ 0)
        for fi in face_indices:
            if fi not in mapping:
                mapping[fi] = f"Face_{fi}"
                assignment_scores[fi] = 0.0

        return mapping, assignment_scores

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
