# NEXUS — Pre-Scan Meeting Type Classification + Strategy Configuration

## Problem

The video pipeline uses identical parameters for every meeting type. A 3×3 gallery
grid, an active-speaker Zoom call, and a physical room camera all get the same:
- CentroidTracker match_threshold (0.10)
- IdentityVerifier check_interval (30 frames)
- ActiveTileTagger (always ON)
- LayoutClassifier reset sensitivity (same thresholds)
- ArcFace merge threshold (same for tiny grid faces and large room faces)
- MediaPipe face budget (same num_faces)

This causes:
- Grid meetings: tracker too loose (0.10) for faces that barely move → track swaps
- Active-speaker meetings: verifier too slow (6s) for tile swaps happening every 2-3s
- Room meetings: ActiveTileTagger fires on whoever is closest to camera (not active speaker)
- Grid meetings: body rules fire on 80px faces where pose estimation is noise

The LayoutClassifier adapts mid-stream but the initial configuration is wrong for
the first 20-30 seconds (WINDOW_SIZE=15 frames = 3 seconds before first stable
classification). Track contamination in those first seconds persists for the entire session.

## Solution

Sample 10 frames across the video BEFORE the main extraction loop. Run the
already-loaded MediaPipe FaceDetector on each (~50ms per frame = 500ms total).
Classify meeting type from face count + size distribution + position pattern.
Configure all downstream components with optimal parameters for that type.

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**
1. Read `services/video_agent/feature_extractor.py` COMPLETELY — especially:
   - `TiledFrameProcessor` class and `TiledFrameProcessor.create()` method
   - `warmup()` method — how models are pre-loaded
   - `_extract_frames()` method — the frame loop, where CentroidTracker,
     IdentityVerifier, ActiveTileTagger, and LayoutClassifier are initialized
   - `CentroidTracker.__init__()` — current match_threshold parameter
   - `IdentityVerifier.__init__()` — current check_interval parameter
   - `ActiveTileTagger.__init__()` — current ACTIVE_TILE_MIN_AREA
   - `LayoutClassifier.__init__()` — current WINDOW_SIZE, COOLDOWN_FRAMES
   - `_compute_merge_threshold()` — current avg_face_height_ratio classification
   - `FaceEmbeddingExtractor.get_instance()` — singleton pattern
   - `MediaPipeModelManager` — model file paths

2. Read `services/video_agent/main.py` COMPLETELY — especially:
   - `VideoPipeline.__init__()` — how extractor is created
   - `VideoPipeline.run_analysis()` — the flow: extract_all → mapper → rules
   - `_run_video_job()` — async job handler, where to insert probe
   - `startup_event()` — current warmup behavior

3. Think about the problem and the solution I'm giving. Consider:
   - What if the video changes layout mid-session? (probe gives initial type,
     LayoutClassifier still adapts mid-stream)
   - What if the probe samples hit a screenshare segment? (faces absent → 
     wrong classification)
   - What if the video is very short (< 30s)? (fewer samples needed)
   - What if the face detector isn't loaded yet? (graceful fallback to defaults)
   - What about videos with NO faces at all? (podcast/audio-only with video wrapper)

4. Implement using proper OOP concepts and DSA principles:
   - **Strategy Pattern**: MeetingProfile configures downstream components —
     each component reads its config from the profile, not from hardcoded constants
   - **Template Method**: probe() runs detect → classify → build_profile pipeline
   - **Builder Pattern**: MeetingProfile.grid() / .active_speaker() / .room() factory methods
   - **O(S × F)**: S=10 sample frames, F=faces per frame — total ~500ms
   - **HashMap**: face position clustering uses dict keyed by quantized row index
   - **Sorted arrays**: grid alignment check via sorted Y values with gap detection
   - **Graceful degradation**: if probe fails, return MeetingProfile.default() — 
     identical to current behavior (no regression)

---

## Architecture

```
BEFORE main extraction:
  MeetingTypeProbe.probe(video_path, face_detector)
    → reads 10 frames at [0%, 5%, 10%, 25%, 50%, 60%, 75%, 85%, 90%, 95%]
    → runs FaceDetector on each (IMAGE mode, already loaded)
    → classifies: grid | active_speaker | room | screenshare | unknown
    → returns MeetingProfile dataclass

  VideoFeatureExtractor.apply_profile(profile)
    → configures CentroidTracker, IdentityVerifier, ActiveTileTagger,
      LayoutClassifier, merge threshold, num_faces, static face filter

DURING main extraction:
  _extract_frames() uses profile-configured parameters from the start
  LayoutClassifier STILL runs per-frame for mid-session layout changes
  IdentityVerifier STILL runs per-frame with profile-tuned interval
  Everything else unchanged — profile just sets initial parameters
```

---

## Change 1: FrameSample + MeetingProfile Dataclasses

**File:** `services/video_agent/feature_extractor.py`
**Location:** After existing dataclass definitions (FrameFeatures, WindowFeatures)

```python
@dataclass(frozen=True)
class FrameSample:
    """Immutable snapshot of one probed frame's face geometry."""
    face_count: int
    face_areas: tuple[float, ...]          # sorted largest-first, normalised 0-1
    face_centroids: tuple[tuple[float, float], ...]  # (cx, cy) normalised
    timestamp_pct: float                   # position in video 0.0-1.0


@dataclass
class MeetingProfile:
    """
    Configuration bundle derived from pre-scan probe.
    
    Configures all downstream components with optimal parameters
    for the detected meeting type. Follows Builder Pattern via
    class factory methods.
    
    Design:
      - Immutable after construction (set once, read many)
      - Factory methods for each meeting type encode domain knowledge
      - default() returns current hardcoded values (zero regression)
    """
    meeting_type: str                      # "grid" | "active_speaker" | "room" | "unknown"
    expected_faces: int                    # median face count from probe
    tracker_match_threshold: float         # CentroidTracker distance threshold
    verifier_check_interval: int           # IdentityVerifier frames between checks
    active_tile_tagger_enabled: bool       # whether ActiveTileTagger runs
    layout_reset_sensitivity: float        # LayoutClassifier COOLDOWN_FRAMES
    merge_threshold_offset: float          # added to base ArcFace merge threshold
    static_face_filter_enabled: bool       # filter photo/graphic faces
    body_rules_confidence_cap: float       # cap body signal confidence for tiny faces
    num_faces_override: int | None = None  # override MediaPipe num_faces budget

    @classmethod
    def grid(cls, face_count: int) -> "MeetingProfile":
        """
        Grid/gallery meeting (2×2, 3×3, etc.).
        
        Characteristics: faces don't move, all similar size, grid-aligned.
        Strategy: tight tracker (faces stationary), slow verifier (no tile swaps),
        disable active-tile tagger (no dominant tile), looser ArcFace merge
        (tiny faces = noisy embeddings), cap body rules (80px poses are noise).
        """
        return cls(
            meeting_type="grid",
            expected_faces=face_count,
            tracker_match_threshold=0.05,       # tight — faces barely move in grid
            verifier_check_interval=50,          # 10s — positions stable, swaps rare
            active_tile_tagger_enabled=False,     # no dominant tile in gallery
            layout_reset_sensitivity=40,          # high cooldown — grid is stable
            merge_threshold_offset=-0.05,         # looser — tiny face embeddings noisy
            static_face_filter_enabled=True,      # screen share graphics possible
            body_rules_confidence_cap=0.60,       # body unreliable on tiny tiles
            num_faces_override=face_count + 2,    # budget for grid + margin
        )

    @classmethod
    def active_speaker(cls, face_count: int) -> "MeetingProfile":
        """
        Active-speaker view (one large tile + sidebar thumbnails).
        
        Characteristics: one face much larger, tile swaps on speaker change.
        Strategy: medium tracker, fast verifier (catch tile swaps), enable
        active-tile tagger, high reset sensitivity (layout changes frequent).
        """
        return cls(
            meeting_type="active_speaker",
            expected_faces=face_count,
            tracker_match_threshold=0.10,
            verifier_check_interval=10,           # 2s — tile swaps are fast
            active_tile_tagger_enabled=True,
            layout_reset_sensitivity=25,           # standard cooldown
            merge_threshold_offset=0.0,
            static_face_filter_enabled=True,
            body_rules_confidence_cap=1.0,         # large tile has good pose
            num_faces_override=max(face_count + 2, 6),
        )

    @classmethod
    def room(cls, face_count: int) -> "MeetingProfile":
        """
        Physical room camera (conference room, interview room).
        
        Characteristics: faces at different distances/heights, physical movement.
        Strategy: wide tracker (people move), medium verifier, disable active-tile
        tagger (no tiles), no static filter (no screen share in room).
        """
        return cls(
            meeting_type="room",
            expected_faces=face_count,
            tracker_match_threshold=0.15,          # wide — people move physically
            verifier_check_interval=30,             # 6s — gradual movement
            active_tile_tagger_enabled=False,
            layout_reset_sensitivity=50,            # high cooldown — room is stable
            merge_threshold_offset=0.0,
            static_face_filter_enabled=False,       # no screen share in room
            body_rules_confidence_cap=1.0,          # full bodies visible
            num_faces_override=face_count + 4,      # people may enter/leave room
        )

    @classmethod
    def default(cls) -> "MeetingProfile":
        """
        Fallback: identical to current hardcoded values. Zero regression.
        Used when probe fails or video has no faces.
        """
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
```

---

## Change 2: MeetingTypeProbe Class

**File:** `services/video_agent/feature_extractor.py`
**Location:** After MeetingProfile, before CentroidTracker

```python
class MeetingTypeProbe:
    """
    Pre-scan video to classify meeting type before main extraction.

    Samples 10 frames across the video and runs the already-loaded MediaPipe
    FaceDetector on each. Classifies from face count + size distribution +
    position alignment pattern.

    Total cost: ~500ms (10 × ~50ms detection). Runs ONCE before extract_all().
    Models are already loaded from warmup() — no additional model init.

    Design:
      - Template Method: probe() → _sample_frames() → _classify() → factory
      - O(S × F) where S=10 samples, F=faces per frame
      - HashMap: row clustering for grid detection (quantized Y → face list)
      - Sorted arrays: grid alignment via sorted centroids + gap detection
      - Graceful degradation: any failure → MeetingProfile.default()

    Classification rules:
      GRID:            ≥4 faces, sizes within 2.5× of each other, positions form rows
      ACTIVE_SPEAKER:  ≥2 faces, largest > 2× second-largest in most samples
      ROOM:            ≥2 faces, high Y-spread (>0.3), no grid alignment
      UNKNOWN:         everything else → default parameters
    """

    # Sample 10 points spread across the video.
    # Avoid 0% (may be black/title frame) and 100% (may be credits/end).
    # Heavier in early portion — meeting type is usually established by 25%.
    SAMPLE_POINTS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.95)

    # Grid detection thresholds
    _GRID_MIN_FACES = 4
    _GRID_MAX_AREA_RATIO = 2.5    # max(area) / min(area) — grid faces are similar
    _GRID_ROW_TOLERANCE = 0.06    # Y-distance within same row (normalised)
    _GRID_MIN_ROWS = 2
    _GRID_MIN_COLS = 2

    # Active-speaker detection
    _AS_MIN_DOMINANT_RATIO = 2.0  # largest face / second face
    _AS_MIN_DOMINANT_SAMPLES = 4  # must be dominant in ≥4 of 10 samples

    # Room detection
    _ROOM_Y_SPREAD = 0.25         # min Y range across face centroids
    _ROOM_MAX_FACES = 8           # rooms rarely have > 8 visible faces

    def probe(
        self,
        video_path: str,
        face_detector,
        mp_module=None,
    ) -> MeetingProfile:
        """
        Sample frames and classify meeting type.

        Args:
            video_path: path to video file
            face_detector: MediaPipe FaceDetector instance (IMAGE mode)
                          — already loaded in TiledFrameProcessor or warmup
            mp_module: mediapipe module (for Image construction)

        Returns:
            MeetingProfile with optimal parameters for detected meeting type.
            On ANY failure, returns MeetingProfile.default() (zero regression).
        """
        try:
            samples = self._sample_frames(video_path, face_detector, mp_module)
            if not samples:
                logger.warning("MeetingTypeProbe: no valid samples — using default profile")
                return MeetingProfile.default()

            profile = self._classify(samples)
            logger.info(
                "MeetingTypeProbe: %s (expected %d faces) from %d samples "
                "[counts=%s, areas=%s]",
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

    def _sample_frames(
        self,
        video_path: str,
        face_detector,
        mp_module,
    ) -> list[FrameSample]:
        """
        Read 10 frames from the video and run face detection on each.
        O(S) where S = len(SAMPLE_POINTS) = 10.
        """
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
            target_frame = int(pct * total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, bgr = cap.read()
            if not ret or bgr is None:
                continue

            # Run face detection (IMAGE mode — stateless, already loaded)
            try:
                if mp_module:
                    mp_image = mp_module.Image(
                        image_format=mp_module.ImageFormat.SRGB,
                        data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    )
                    result = face_detector.detect(mp_image)
                    detections = result.detections if result else []
                else:
                    # Fallback: use InsightFace if MediaPipe not available
                    detections = []

                face_areas = []
                face_centroids = []

                for det in detections:
                    bbox = det.bounding_box
                    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    area = (w * h) / (frame_w * frame_h)
                    cx = (x + w / 2) / frame_w
                    cy = (y + h / 2) / frame_h
                    face_areas.append(area)
                    face_centroids.append((cx, cy))

                # Sort by area (largest first)
                if face_areas:
                    paired = sorted(
                        zip(face_areas, face_centroids),
                        key=lambda p: -p[0],
                    )
                    face_areas = [p[0] for p in paired]
                    face_centroids = [p[1] for p in paired]

                samples.append(FrameSample(
                    face_count=len(detections),
                    face_areas=tuple(face_areas),
                    face_centroids=tuple(face_centroids),
                    timestamp_pct=pct,
                ))

            except Exception as exc:
                logger.debug("Probe frame at %.0f%% failed: %s", pct * 100, exc)
                continue

        cap.release()
        return samples

    def _classify(self, samples: list[FrameSample]) -> MeetingProfile:
        """
        Classify meeting type from sampled frame geometry.
        Priority: grid > active_speaker > room > unknown.
        """
        # Filter out empty frames (screenshare, black frames)
        face_samples = [s for s in samples if s.face_count >= 2]
        if not face_samples:
            # Possibly solo speaker or no faces
            solo_samples = [s for s in samples if s.face_count == 1]
            if solo_samples:
                return MeetingProfile.active_speaker(1)
            return MeetingProfile.default()

        median_count = sorted([s.face_count for s in face_samples])[len(face_samples) // 2]

        # Check grid pattern first (most specific)
        if self._is_grid(face_samples):
            return MeetingProfile.grid(median_count)

        # Check active-speaker (one dominant face)
        if self._is_active_speaker(face_samples):
            return MeetingProfile.active_speaker(median_count)

        # Check room camera (scattered positions)
        if self._is_room(face_samples):
            return MeetingProfile.room(median_count)

        # Fallback
        return MeetingProfile.default()

    def _is_grid(self, samples: list[FrameSample]) -> bool:
        """
        Grid detection: ≥4 faces, similar sizes, grid-aligned positions.

        Algorithm:
          1. Check face count consistency (most samples have ≥4 faces)
          2. Check area homogeneity (max/min < 2.5×)
          3. Check position grid alignment (faces cluster into rows)

        DSA: Sort centroids by Y. Scan once O(F) to find row gaps.
        Group into rows. Check each row has ≥2 faces at similar X spacing.
        """
        grid_samples = [s for s in samples if s.face_count >= self._GRID_MIN_FACES]
        if len(grid_samples) < len(samples) * 0.5:
            return False  # fewer than half the samples have ≥4 faces

        # Area homogeneity: no single face much larger than others
        for s in grid_samples:
            if not s.face_areas or len(s.face_areas) < self._GRID_MIN_FACES:
                continue
            min_area = min(a for a in s.face_areas if a > 0.001)
            max_area = max(s.face_areas)
            if max_area / max(min_area, 0.001) > self._GRID_MAX_AREA_RATIO:
                return False

        # Position grid alignment
        grid_aligned_count = 0
        for s in grid_samples:
            if self._check_grid_alignment(s.face_centroids):
                grid_aligned_count += 1

        return grid_aligned_count >= len(grid_samples) * 0.5

    def _check_grid_alignment(self, centroids: tuple[tuple[float, float], ...]) -> bool:
        """
        Check if face centroids form a rectangular grid pattern.

        DSA: Sort by Y → O(F log F). Single pass to group into rows → O(F).
        Check each row for ≥2 members → O(R) where R = rows.
        Total: O(F log F).
        """
        if len(centroids) < self._GRID_MIN_FACES:
            return False

        # Sort by Y coordinate
        sorted_by_y = sorted(centroids, key=lambda c: c[1])

        # Group into rows: faces with Y within _GRID_ROW_TOLERANCE are same row
        rows: list[list[tuple[float, float]]] = []
        current_row = [sorted_by_y[0]]

        for centroid in sorted_by_y[1:]:
            if centroid[1] - current_row[-1][1] < self._GRID_ROW_TOLERANCE:
                current_row.append(centroid)
            else:
                rows.append(current_row)
                current_row = [centroid]
        rows.append(current_row)

        # Need ≥2 rows with ≥2 faces each
        valid_rows = [r for r in rows if len(r) >= self._GRID_MIN_COLS]
        return len(valid_rows) >= self._GRID_MIN_ROWS

    def _is_active_speaker(self, samples: list[FrameSample]) -> bool:
        """
        Active-speaker: one face significantly larger than others in most samples.
        """
        dominant_count = 0
        for s in samples:
            if len(s.face_areas) < 2:
                continue
            ratio = s.face_areas[0] / max(s.face_areas[1], 0.001)
            if ratio >= self._AS_MIN_DOMINANT_RATIO:
                dominant_count += 1

        return dominant_count >= self._AS_MIN_DOMINANT_SAMPLES

    def _is_room(self, samples: list[FrameSample]) -> bool:
        """
        Room camera: faces scattered at different Y heights (people at
        different distances), no grid alignment.
        """
        for s in samples:
            if len(s.face_centroids) < 2:
                continue
            ys = [cy for _, cy in s.face_centroids]
            y_spread = max(ys) - min(ys)
            if (y_spread >= self._ROOM_Y_SPREAD
                    and s.face_count <= self._ROOM_MAX_FACES
                    and not self._check_grid_alignment(s.face_centroids)):
                return True
        return False
```

---

## Change 3: VideoFeatureExtractor.apply_profile()

**File:** `services/video_agent/feature_extractor.py`
**Location:** Add method to VideoFeatureExtractor class

This method is called ONCE between probe and extract_all. It configures
instance variables that `_extract_frames()` reads when initializing its
per-session objects (CentroidTracker, IdentityVerifier, etc.).

```python
    def apply_profile(self, profile: MeetingProfile) -> None:
        """
        Configure extraction parameters based on pre-scan meeting type.

        Called by VideoPipeline.run_analysis() BEFORE extract_all().
        Stores profile settings as instance variables that _extract_frames()
        reads when constructing per-session objects.

        Design: Strategy Pattern — each downstream component reads its
        config from self._profile_*, not from module-level constants.
        """
        self._profile_meeting_type = profile.meeting_type
        self._profile_tracker_threshold = profile.tracker_match_threshold
        self._profile_verifier_interval = profile.verifier_check_interval
        self._profile_active_tile_enabled = profile.active_tile_tagger_enabled
        self._profile_layout_cooldown = int(profile.layout_reset_sensitivity)
        self._profile_merge_offset = profile.merge_threshold_offset
        self._profile_static_filter = profile.static_face_filter_enabled
        self._profile_body_conf_cap = profile.body_rules_confidence_cap

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
```

---

## Change 4: _extract_frames() Uses Profile Settings

**File:** `services/video_agent/feature_extractor.py` — `_extract_frames()`
**Location:** Where CentroidTracker, IdentityVerifier, ActiveTileTagger,
LayoutClassifier are initialized (beginning of the method)

Replace hardcoded values with profile-configured values:

```python
        # ── Initialize tracking components with profile-tuned parameters ─────
        tracker_threshold = getattr(self, "_profile_tracker_threshold", 0.10)
        centroid_tracker = CentroidTracker(
            max_disappeared=15,
            match_threshold=tracker_threshold,
        )

        verifier_interval = getattr(self, "_profile_verifier_interval", 30)
        identity_verifier = IdentityVerifier(
            embedder_app=embedder._app if embedder else None,
            check_interval=verifier_interval,
            similarity_threshold=0.50,
        )

        active_tile_enabled = getattr(self, "_profile_active_tile_enabled", True)
        if active_tile_enabled:
            active_tagger = ActiveTileTagger(
                diar_segments=diar_segments,
                min_face_area=ActiveTileTagger.ACTIVE_TILE_MIN_AREA,
            )
        else:
            active_tagger = None  # disabled for grid/room meetings

        layout_cooldown = getattr(self, "_profile_layout_cooldown", 25)
        layout_classifier = LayoutClassifier(
            window_size=LayoutClassifier.WINDOW_SIZE,
            cooldown_frames=layout_cooldown,
        )
```

In the frame loop, guard ActiveTileTagger calls:

```python
                        # ── Active-tile tagging (if enabled for this meeting type) ──
                        if active_tagger is not None:
                            active_tagger.tag_frame(
                                frame_features_list, timestamp_ms,
                                border_active_idx=border_idx,
                            )
```

After ArcFace merge, apply merge_threshold_offset:

```python
        merge_offset = getattr(self, "_profile_merge_offset", 0.0)
        merge_threshold = self._compute_merge_threshold(frames) + merge_offset
```

---

## Change 5: Static Face Filter (Photo/Graphic Detection)

**File:** `services/video_agent/feature_extractor.py`
**Location:** After ArcFace merge, before rule engines

Only runs when `_profile_static_filter` is True (grid + active_speaker meetings
where screen share graphics may appear).

```python
        # ── Filter static faces (photos/graphics) ────────────────────────────
        if getattr(self, "_profile_static_filter", True):
            static_tracks = set()
            for tid, wf_list in windows_by_face.items():
                if self._is_static_face(wf_list):
                    static_tracks.add(tid)
                    logger.info("Static face filtered: %s (photo/graphic)", tid)
            for tid in static_tracks:
                windows_by_face.pop(tid, None)

    @staticmethod
    def _is_static_face(windows: list["WindowFeatures"]) -> bool:
        """
        Detect photo/graphic faces: zero biological motion.
        Real faces blink, breathe (jaw), and micro-move (head).
        Photos have all three at zero variance.

        DSA: O(W) single pass. Returns True only if ALL three are near-zero.
        """
        if len(windows) < 5:
            return False

        blinks = [w.eye_blink_avg for w in windows if w.face_detected]
        jaws = [w.jaw_open_mean for w in windows if w.face_detected]
        yaws = [w.head_yaw for w in windows if w.face_detected]

        if len(blinks) < 5:
            return False

        def _var(vals: list[float]) -> float:
            if not vals:
                return 0.0
            m = sum(vals) / len(vals)
            return sum((v - m) ** 2 for v in vals) / len(vals)

        return (_var(blinks) < 0.001
                and _var(jaws) < 0.0005
                and _var(yaws) < 0.01)
```

---

## Change 6: Body Rules Confidence Cap

**File:** `services/video_agent/main.py` or wherever body rule signals are collected
**Location:** After body rules produce signals, before they're added to all_signals

```python
        # ── Apply body confidence cap from meeting profile ────────────────────
        body_cap = getattr(self._extractor, "_profile_body_conf_cap", 1.0)
        if body_cap < 1.0:
            for sig in body_signals:
                if sig.get("agent") in ("body", "video"):
                    original = sig.get("confidence", 0.5)
                    sig["confidence"] = round(min(original, body_cap), 4)
                    if original != sig["confidence"]:
                        sig.setdefault("metadata", {})["body_cap_applied"] = True
                        sig["metadata"]["original_confidence"] = original
```

---

## Change 7: Integration in VideoPipeline.run_analysis()

**File:** `services/video_agent/main.py` — `VideoPipeline.run_analysis()`
**Location:** BEFORE `extract_all()`, AFTER video file is available

```python
        # ── Pre-scan: classify meeting type (500ms, uses already-loaded models) ──
        import mediapipe as mp
        try:
            probe = MeetingTypeProbe()
            # Use the FaceDetector in IMAGE mode for stateless per-frame detection
            face_det_for_probe = mp.tasks.vision.FaceDetector.create_from_options(
                mp.tasks.vision.FaceDetectorOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=self._extractor._model_mgr.get_face_detector_path()
                    ),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    min_detection_confidence=0.3,
                )
            )
            profile = probe.probe(video_path, face_det_for_probe, mp)
            face_det_for_probe.close()
        except Exception as exc:
            logger.warning(f"[{session_id}] Meeting type probe failed: {exc}")
            profile = MeetingProfile.default()

        self._extractor.apply_profile(profile)
        logger.info(f"[{session_id}] Meeting type: {profile.meeting_type}")

        # ── Extract ──────────────────────────────────────────────────────────
        windows, lip_activity_map = self._extractor.extract_all(...)
```

---

## Edge Cases Handled

| Scenario | What Happens |
|----------|-------------|
| Video changes layout mid-session | Probe gives initial type; LayoutClassifier still adapts mid-stream |
| Probe hits screenshare frames (no faces) | Those samples have face_count=0, filtered out in _classify |
| Very short video (< 30s) | Fewer than 10 frames; probe works with fewer samples or defaults |
| Face detector not loaded | probe() catches exception → MeetingProfile.default() |
| No faces in entire video | median_count=0 → MeetingProfile.default() |
| Mixed meeting (starts grid, becomes room) | Probe classifies by majority of 10 samples; LayoutClassifier handles transitions |
| 2-person meeting | face_count=2, likely active_speaker (one talks, one listens in big tile) |
| Solo presenter | face_count=1, → active_speaker with expected_faces=1 |

---

## Expected Impact

| Meeting Type | Before (wrong params) | After (profiled params) |
|-------------|----------------------|------------------------|
| 3×3 grid | tracker=0.10 (too loose), verifier=30 (wasted checks), body rules fire on 80px faces | tracker=0.05 (tight), verifier=50 (less waste), body capped at 0.60 |
| Active speaker | verifier=30 (misses tile swaps), no pre-config | verifier=10 (catches swaps), active_tile ON, reset sensitivity HIGH |
| Room camera | ActiveTileTagger fires on closest person, static filter wastes time | ActiveTileTagger OFF, tracker=0.15 (accommodates movement), static filter OFF |

## Files Modified:
1. **services/video_agent/feature_extractor.py**:
   - New: `FrameSample` dataclass (~8 lines)
   - New: `MeetingProfile` dataclass with 4 factory methods (~80 lines)
   - New: `MeetingTypeProbe` class (~180 lines)
   - New: `apply_profile()` method on VideoFeatureExtractor (~20 lines)
   - New: `_is_static_face()` static method (~15 lines)
   - Modified: `_extract_frames()` — read profile settings instead of hardcoded values (~15 lines changed)
2. **services/video_agent/main.py**:
   - Modified: `VideoPipeline.run_analysis()` — insert probe + apply_profile (~15 lines)
   - Modified: body signal confidence cap application (~8 lines)