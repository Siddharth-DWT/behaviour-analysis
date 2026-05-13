# NEXUS Phase 2: Video Agents — Sub-Phase Implementation Plan

## Technology Stack

| Component | Tool | Why |
|-----------|------|-----|
| Face landmarks (478 points) | MediaPipe Face Landmarker | Real-time, 52 blendshapes (AU-equivalent), head pose matrix |
| Body landmarks (33 points) | MediaPipe Pose Landmarker | Shoulders, elbows, hips — enough for posture + nod + lean |
| Hand landmarks (21/hand) | MediaPipe Hands | Gesture detection, self-touch via bounding box overlap |
| Iris tracking (10 points) | MediaPipe Face Landmarker (refine_landmarks=True) | Gaze direction, blink detection |
| Video input | OpenCV (cv2.VideoCapture) | Read uploaded video files frame-by-frame |
| Frame rate | Process at 10fps (skip frames) | 30fps is overkill — behavioral signals don't change faster than 10fps. Saves 3x compute |

**Key insight from MediaPipe Face Landmarker:** The 52 blendshapes map loosely to FACS Action Units, giving us AU-like scores (browDownLeft, jawOpen, mouthSmileLeft, eyeSquintLeft, etc.) WITHOUT needing a separate AU detection model. This replaces the need for OpenFace or similar AU detectors.

---

## Architecture — How Video Plugs Into Existing System

```
Video file uploaded alongside audio
    │
    ├─ Audio path (existing): VoiceAgent → LanguageAgent → ConversationAgent
    │
    └─ Video path (NEW):
        │
        ├─ VideoFeatureExtractor (NEW service)
        │   ├─ MediaPipe Face Landmarker → 478 landmarks + 52 blendshapes + head pose
        │   ├─ MediaPipe Pose Landmarker → 33 body landmarks
        │   └─ MediaPipe Hands → 21 landmarks per hand
        │
        ├─ FacialAgent (NEW) → facial signals (7 rules)
        ├─ BodyAgent (NEW) → body signals (8 rules)
        ├─ GazeAgent (NEW) → gaze signals (7 rules)
        │
        └─ All video signals → FusionEngine.SignalBuffer
            └─ 12 pairwise fusion rules fire (voice × face, voice × body, etc.)
            └─ 12 compound patterns fire
            └─ 8 temporal patterns fire
```

**Critical design decision:** Video agents run as SEPARATE microservices (like VoiceAgent, LanguageAgent) OR as modules within the API Gateway. Recommendation: start as modules within a single VideoAgent service, split later if needed.

---

## Sub-Phase Breakdown

### Phase 2A: Video Pipeline Foundation (Week 1-2)

**Goal:** Process video files frame-by-frame, extract raw features, establish baselines. No rules yet — just the infrastructure.

| Task | Description | Effort |
|------|-------------|--------|
| **VideoFeatureExtractor** | New class that wraps MediaPipe Face Landmarker + Pose Landmarker + Hands. Takes a video file path, processes at 10fps, outputs per-frame feature dicts | 2 days |
| **Frame-level features** | For each frame extract: 52 blendshapes, head pose (pitch/yaw/roll), 33 body landmarks, eye aspect ratio (EAR for blink detection), mouth aspect ratio, shoulder angle, head-to-shoulder distance | 1 day |
| **Window aggregation** | Aggregate frame features into 2-second windows (20 frames at 10fps): mean, std, min, max, delta from previous window. Output format matches voice feature windows | 1 day |
| **Facial baseline (FACE-CAL-01)** | Build per-speaker facial baseline from first 90-120 seconds: neutral blendshape values, resting head pose, average blink rate, neutral mouth shape. Uses same CalibrationModule pattern as voice | 1 day |
| **Body baseline (BODY-CAL-01)** | Build per-speaker body baseline: resting posture angles, average fidget rate, neutral position. Same calibration approach | 0.5 day |
| **Gaze baseline (GAZE-CAL-01)** | Build per-speaker gaze baseline: camera position estimation from average head pose, natural screen engagement rate, resting blink rate. Critical: estimate camera-screen offset (5-15° vertical on laptops) | 0.5 day |
| **Speaker-to-face mapping** | Map which face belongs to which speaker. For uploaded video: use diarization timestamps — face visible during Speaker_0's segments = Speaker_0's face. For multi-person video (Recall.ai): separate feeds per speaker | 1 day |
| **API endpoint** | New POST /sessions endpoint accepts video file (mp4/webm). Pipeline: extract audio → existing voice pipeline + extract video → video pipeline → merge signals in FusionEngine | 1 day |
| **Docker setup** | Add MediaPipe dependencies to Dockerfile. Test GPU acceleration (MediaPipe supports GPU on Linux). Fallback to CPU if no GPU | 0.5 day |

**Deliverables:**
- `services/video_agent/feature_extractor.py` — MediaPipe wrapper + window aggregation
- `services/video_agent/calibration.py` — facial + body + gaze baselines
- `services/video_agent/main.py` — FastAPI service with /analyse endpoint
- Updated `docker-compose.yml` with video_agent service
- Updated API Gateway to call video agent after voice agent

**Key code pattern (VideoFeatureExtractor):**
```python
import mediapipe as mp
import cv2

class VideoFeatureExtractor:
    def __init__(self):
        # Face Landmarker with blendshapes enabled
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path='face_landmarker.task'),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=2,  # Support 2 faces for in-person meetings
            )
        )
        # Pose Landmarker
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(...)
    
    def extract_all(self, video_path: str, fps: int = 10) -> list[dict]:
        """Process video at target fps, return per-window feature dicts."""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        skip = max(1, int(video_fps / fps))  # Process every Nth frame
        
        frames = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip != 0:
                frame_idx += 1
                continue
            
            timestamp_ms = int(frame_idx / video_fps * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            features = self._extract_frame_features(face_result, pose_result, timestamp_ms)
            frames.append(features)
            frame_idx += 1
        
        # Aggregate into 2-second windows
        return self._aggregate_windows(frames, window_ms=2000)
```

---

### Phase 2B: Facial Agent — 7 Rules (Week 3-4)

**Goal:** Detect facial expressions, smile quality, visual stress, engagement from blendshapes and landmarks.

| Rule | Signal | What It Detects | MediaPipe Input | Cap | Priority |
|------|--------|----------------|-----------------|-----|----------|
| FACE-EMO-01 | primary_emotion | Happy/Angry/Sad/Surprise/Disgust/Neutral | 52 blendshapes (jawOpen, mouthSmile, browDown, etc.) | 0.55 | High |
| FACE-SMILE-01 | smile_quality | Duchenne vs social smile, temporal onset speed | mouthSmileLeft/Right + cheekSquintLeft/Right + temporal features | 0.45-0.55 | High |
| FACE-MICRO-01 | rapid_expression_change | DISABLED at ≤30fps (requires 100+fps) | N/A | DISABLED | Skip |
| FACE-ENG-01 | engagement_visual | Head pose variability + expression variability | Head pitch/yaw/roll variance + blendshape variance | 0.50 | Medium |
| FACE-STRESS-01 | stress_visual | Lip press (AU23/24) + brow furrow (AU4) + jaw clench | jawOpen, browDownLeft/Right, mouthPressLeft/Right | 0.35-0.40 | High |
| FACE-VA-01 | valence_arousal | Emotional valence (positive/negative) + arousal (calm/activated) | Composite of smile, brow, mouth, eye blendshapes | Valence 0.55, Arousal 0.40 | Medium |
| FACE-CAL-01 | facial_baseline | Built in Phase 2A | — | — | Done |

**Key implementation detail — Blendshape → Emotion mapping:**
```python
# MediaPipe blendshapes that map to each emotion
EMOTION_BLENDSHAPES = {
    "happy": ["mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight"],
    "angry": ["browDownLeft", "browDownRight", "mouthFrownLeft", "mouthFrownRight", "jawForward"],
    "sad": ["mouthFrownLeft", "mouthFrownRight", "browInnerUp", "mouthPucker"],
    "surprise": ["browInnerUp", "browOuterUpLeft", "browOuterUpRight", "jawOpen", "mouthOpen"],
    "disgust": ["noseSneerLeft", "noseSneerRight", "mouthShrugUpper"],
}

# Per-emotion confidence weighting (from research validation)
EMOTION_CONFIDENCE = {
    "happy": 0.70,      # Most reliably detected
    "angry": 0.55,
    "sad": 0.55,
    "surprise": 0.60,
    "disgust": 0.35,    # Least reliable from webcam
    "contempt": 0.35,
}
```

**Files created:**
- `services/video_agent/facial_rules.py` — FacialRuleEngine with 6 active rules
- Tests for each rule

---

### Phase 2C: Gaze Agent — 7 Rules (Week 5-6)

**Goal:** Track gaze direction, blink rate, screen engagement, distraction events from iris landmarks and head pose.

| Rule | Signal | What It Detects | MediaPipe Input | Cap | Priority |
|------|--------|----------------|-----------------|-----|----------|
| GAZE-DIR-01 | gaze_direction | On-screen vs off-screen (binary) | Iris landmarks (468-477) + head pose | 0.45-0.65 | High |
| GAZE-CONTACT-01 | screen_engagement | % time looking at screen (speaking vs listening norms) | gaze_direction over 30s windows | 0.50 | High |
| GAZE-BLINK-01 | blink_rate | Blinks/minute, stress if >30-35 bpm | Eye Aspect Ratio (EAR) from eye landmarks | 0.50 | High |
| GAZE-ATT-01 | attention_score | Composite: gaze % + head stability + blink deviation | Composite of other gaze signals | 0.55 | Medium |
| GAZE-DIST-01 | distraction_event | Sustained gaze break >8-10 seconds | gaze_direction sustained off-screen | 0.45 | Medium |
| GAZE-SYNC-01 | gaze_alignment | Cross-speaker gaze coordination (EXPERIMENTAL) | Both speakers' gaze simultaneously | 0.40 | Low |
| GAZE-CAL-01 | gaze_baseline | Built in Phase 2A | — | — | Done |

**Critical thresholds (research-corrected):**
- BLINK-01: Normal conversational blink rate = 26 bpm (Bentivoglio 1997). Stress threshold at >30-35 bpm (NOT >25 as originally spec'd — that would flag every normal conversation)
- DIST-01: Sustained break >8-10 seconds (NOT >3s — normal cognitive gaze aversion averages ~6s per HAL 2024)
- CONTACT-01: Use baseline-relative, not Argyle norms (Argyle 1972 was face-to-face, not video calls — camera-screen offset makes everyone appear to avoid eye contact)

**Key implementation detail — Eye Aspect Ratio for blink detection:**
```python
def compute_ear(eye_landmarks):
    """Eye Aspect Ratio — drops below 0.2 during blinks."""
    # Vertical distances
    v1 = dist(eye_landmarks[1], eye_landmarks[5])
    v2 = dist(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    h = dist(eye_landmarks[0], eye_landmarks[3])
    return (v1 + v2) / (2.0 * h)

# Blink detection: EAR drops below threshold for 2-4 frames
BLINK_EAR_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2
```

**Files created:**
- `services/video_agent/gaze_rules.py` — GazeRuleEngine with 6 active rules + 1 experimental

---

### Phase 2D: Body Agent — 8 Rules (Week 7-8)

**Goal:** Detect posture, head nods/shakes, leaning, gestures, fidgeting, self-touch from pose landmarks.

| Rule | Signal | What It Detects | MediaPipe Input | Cap | Priority |
|------|--------|----------------|-----------------|-----|----------|
| BODY-HEAD-01 | head_nod_shake | Affirmative nods, negative shakes | Head pitch (nod) and yaw (shake) angular velocity from face landmarks | 0.55-0.75 | **Highest** |
| BODY-POST-01 | posture_score | Upright vs slumped, open vs closed | Shoulder landmarks relative to head | 0.40-0.55 | High |
| BODY-LEAN-01 | leaning_direction | Forward lean (engagement) vs backward (disengagement) | Head size change relative to baseline (proxy for distance) | 0.30-0.40 | Medium |
| BODY-GEST-01 | gesture_type | Beat gestures, deictic (pointing), illustrators | Hand landmark velocity and trajectory from Hands/Pose | 0.45-0.80 | Medium |
| BODY-FIDG-01 | fidget_rate | Elevated movement rate (anxiety/boredom/excitement) | Sum of all landmark position changes per window | 0.35-0.45 | Medium |
| BODY-TOUCH-01 | self_touch | Hand-to-face/neck/hair contact | Bounding box overlap: hand landmarks vs face landmarks | 0.35-0.50 | Medium |
| BODY-MIRROR-01 | mirroring | Cross-speaker posture synchronization (EXPERIMENTAL) | Pose similarity between two speakers | 0.20-0.30 | Low |
| BODY-CAL-01 | body_baseline | Built in Phase 2A | — | — | Done |

**HEAD-01 is the highest-reliability body signal** — a nod while saying "no" or a shake while saying "yes" is extremely diagnostic. Implementation:

```python
def detect_nod_shake(head_pitch_history, head_yaw_history, fps=10):
    """
    Nod: pitch oscillation > 15°/s within 0.5-1.5s period
    Shake: yaw oscillation > 20°/s within 0.5-1.5s period
    """
    # Compute angular velocity
    pitch_velocity = np.diff(head_pitch_history) * fps  # degrees/second
    yaw_velocity = np.diff(head_yaw_history) * fps
    
    # Nod: 2+ direction changes in pitch within 1.5s
    pitch_crossings = count_zero_crossings(pitch_velocity)
    if pitch_crossings >= 2 and max(abs(pitch_velocity)) > 15:
        return "nod", min(0.75, 0.55 + pitch_crossings * 0.05)
    
    # Shake: 2+ direction changes in yaw within 1.5s
    yaw_crossings = count_zero_crossings(yaw_velocity)
    if yaw_crossings >= 2 and max(abs(yaw_velocity)) > 20:
        return "shake", min(0.70, 0.50 + yaw_crossings * 0.05)
    
    return None, 0
```

**Files created:**
- `services/video_agent/body_rules.py` — BodyRuleEngine with 7 active rules + 1 experimental

---

### Phase 2E: Multimodal Fusion Pairwise Rules (Week 9-10)

**Goal:** Fire cross-modal rules that combine audio + video signals. These plug into the EXISTING FusionEngine — the SignalBuffer already accepts signals from any agent.

| Rule | Pattern | Signals Combined | Cap | Commercial Value |
|------|---------|-----------------|-----|------------------|
| FUSION-01 | Tone × Emotion → Masking | tone_classification (voice) + primary_emotion (face) | 0.65 | **Highest** — detects fake positivity |
| FUSION-03 | Stress × Face → Suppression | vocal_stress_score + stress_visual | 0.65 | High — detects hidden stress |
| FUSION-04 | Gaze × Filler → Cognitive Load | filler_detection + gaze_direction (break) | 0.70 | Medium — indicates overwhelm |
| FUSION-05 | Nod × Objection → Disagreement | head_nod_shake + objection_signal | 0.55 | **Highest** — nod while objecting = conflict |
| FUSION-06 | Lean × Engagement → Interest | leaning_direction + conversation_engagement | 0.60 | Medium — physical engagement |
| FUSION-07* | Already built | verbal_incongruence | 0.70 | Already working |
| FUSION-08 | Eye × Hedge → False Confidence | gaze_direction + power_language (hedges) | 0.55 | Medium |
| FUSION-09 | Smile × Sentiment → Sarcasm | smile_quality + sentiment_score (negative) | 0.60 | Medium |
| FUSION-10 | Latency × Face → Processing | response_latency + stress_visual | 0.60 | Medium |
| FUSION-11 | Dominance × Gaze → Anxiety | dominance_score + gaze_direction (avoidance) | 0.65 | Medium |
| FUSION-12 | Interrupt × Body → Intent | interruption_event + leaning_direction | 0.55 | Medium — cooperative vs competitive interrupt |
| FUSION-14 | Empathy × Nod → Rapport | empathy_language + head_nod_shake | **0.70** | **Strongest fusion rule** — Tickle-Degnen 1990 |

**No changes to fusion_engine.py needed.** The SignalBuffer already stores signals by speaker × agent. Video signals feed in with `agent: "facial"`, `agent: "body"`, `agent: "gaze"`. The pairwise rules just need to query the buffer for signals from different agents in the same temporal window.

**Implementation approach:** Add 12 new methods to the FusionRuleEngine class. Each method queries the SignalBuffer for the two signal types, checks temporal alignment (2-10s window), and computes a fusion score.

**Files modified:**
- `services/fusion_agent/rules.py` — add 11 new pairwise methods (FUSION-07 already exists)

---

### Phase 2F: Compound Patterns (Week 11-12)

**Goal:** Detect complex behavioral states from 3+ co-occurring signals. These are the highest-value commercial signals.

| Pattern | Component Signals | Cap | Commercial Use |
|---------|------------------|-----|---------------|
| C-01: Genuine Engagement | Forward lean + eye contact + nodding + warm tone + fast response | 0.80 | "This prospect is genuinely interested" |
| C-02: Active Disengagement | Backward lean + gaze breaks + fidgeting + slow response + monotone | 0.75 | "This person has checked out" |
| C-03: Emotional Suppression | Flat face + high voice stress + controlled rate + neutral words | 0.70 | "They're hiding something" |
| C-04: Decision Engagement (renamed) | Buying signals + forward lean + reduced fidgeting + direct gaze | 0.65 EXPERIMENTAL | "They're ready to decide" |
| C-05: Cognitive Overload | High filler + gaze breaks + slow rate + self-touch + long response latency | 0.75 | "Too much information" |
| C-06: Conflict Escalation | Rising stress + interruptions + aggressive tone + forward lean + contempt | 0.80 | "This is about to blow up" |
| C-07: Verbal-Nonverbal Discordance (renamed) | Positive words + negative face + stress + gaze avoidance | 0.65 | "Words and body don't match" |
| C-08: Peak Performance | Confident tone + powerful language + upright posture + steady gaze + low stress | 0.75 | "This person is in the zone" |
| C-09: Rapport Building | Mirroring + mutual nods + empathy language + warm tone + balanced talk time | 0.75 | "Strong connection forming" |
| C-10: Dominance Display | High talk time + interruptions + loud volume + forward lean + direct gaze | 0.70 | "This person is controlling the room" |
| C-11: Submission Signal | Low talk time + no interruptions + quiet volume + backward lean + gaze avoidance | 0.65 | "This person has given up control" |
| C-12: Deception Cluster | Stress-sentiment incongruence + gaze breaks + self-touch + filler spike + verbal incongruence | 0.50 MAX + EXPERIMENTAL tag | "Multiple inconsistency indicators — review needed" |

**Implementation:** Each compound pattern is a method that checks for N signals co-occurring within a time window. Use the FusionEngine's `get_all_for_speaker()` to get all signals, then check combinations.

**Files modified:**
- `services/fusion_agent/compound_patterns.py` — NEW file, CompoundPatternEngine class

---

### Phase 2G: Temporal Patterns (Week 13-14)

**Goal:** Detect behavioral changes OVER TIME — these require tracking signal trajectories across the session.

| Pattern | What It Tracks | Window | Commercial Use |
|---------|---------------|--------|---------------|
| T-01: Stress Trajectory | Rising/falling/volatile stress over 5+ minutes | 5-15 min | "Stress building throughout pricing discussion" |
| T-02: Engagement Decay | Progressive disengagement over time | Full session | "Lost the room after minute 12" |
| T-03: Rapport Evolution | Rapport strengthening/weakening across session | Full session | "Connection peaked at minute 8, declined after" |
| T-04: Behavioral Shift Point | Sudden behavioral change (stress spike, engagement drop) | Rolling 2-min | "Something changed at 4:30 — all signals shifted" |
| T-05: Adaptation Pattern | Speaker adjusting behavior in response to other (mirroring increase, pace matching) | Full session | "Seller adapted to buyer's style over time" |
| T-06: Fatigue Detection | Increasing monotone, slower rate, fewer gestures, more pauses over time | Last 30% | "Energy declining — wrap up soon" |
| T-07: Recovery Pattern | Return to baseline after stress spike | 2-5 min | "Handled the objection well — recovered in 90 seconds" |
| T-08: Escalation Ladder | Progressive conflict indicators (Glasl model stages) | Full session | "Conflict escalated through 3 stages" |

**Implementation:** Use the graph_analytics.py approach — these are session-level patterns computed after all agents complete. Add to the existing `GraphAnalytics` class or create a new `TemporalPatternEngine`.

**Files created:**
- `services/fusion_agent/temporal_patterns.py` — TemporalPatternEngine class

---

## Full Timeline Summary

| Week | Sub-Phase | Deliverable | Rules Added |
|------|-----------|-------------|-------------|
| 1-2 | **2A: Foundation** | VideoFeatureExtractor, 3 baselines, speaker-face mapping, API endpoint | 3 calibration rules |
| 3-4 | **2B: Facial Agent** | 6 active rules (EMO-01 disabled, MICRO-01 disabled) | +6 rules (total 48) |
| 5-6 | **2C: Gaze Agent** | 6 active rules + 1 experimental | +7 rules (total 55) |
| 7-8 | **2D: Body Agent** | 7 active rules + 1 experimental | +8 rules (total 63) |
| 9-10 | **2E: Fusion Pairwise** | 11 new cross-modal rules | +11 rules (total 74) |
| 11-12 | **2F: Compound Patterns** | 12 multi-signal behavioral states | +12 rules (total 86) |
| 13-14 | **2G: Temporal Patterns** | 8 session-level trajectory patterns | +8 rules (total 94) |

**Total: 14 weeks → 52 new rules → 94 total rules (matching the original spec)**

---

## Files Created (Phase 2 Complete)

```
services/video_agent/
├── main.py                    # FastAPI service + /analyse endpoint
├── feature_extractor.py       # MediaPipe wrapper + window aggregation
├── calibration.py             # Facial + body + gaze baselines
├── facial_rules.py            # FacialRuleEngine (6 rules)
├── gaze_rules.py              # GazeRuleEngine (7 rules)
├── body_rules.py              # BodyRuleEngine (8 rules)
├── Dockerfile                 # With MediaPipe + OpenCV dependencies
└── requirements.txt           # mediapipe, opencv-python, numpy

services/fusion_agent/
├── rules.py                   # MODIFY: add 11 pairwise fusion methods
├── compound_patterns.py       # NEW: 12 compound patterns
└── temporal_patterns.py       # NEW: 8 temporal patterns

services/api_gateway/
└── main.py                    # MODIFY: call video_agent after voice_agent
```

## Files NOT Modified

- services/voiceAgent/* — audio pipeline unchanged
- services/language_agent/* — text pipeline unchanged
- services/conversation_agent/* — conversation pipeline unchanged
- services/fusion_agent/fusion_engine.py — SignalBuffer already modality-agnostic
- services/fusion_agent/signal_graph.py — works with any signal type
- shared/config/content_type_profile.py — video rules are universal (no per-type adaptation needed)

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MediaPipe face detection fails on side profiles | Facial signals drop out during head turns | Use head pose to detect profile angle >45°, reduce confidence for partial faces |
| Webcam quality varies wildly (720p laptop vs 1080p external) | Landmark accuracy drops on low-res feeds | Add resolution check, reduce confidence below 640x480, skip body detection below 480x360 |
| Multiple faces in frame (in-person meetings) | Wrong face mapped to wrong speaker | Use face position consistency + diarization timestamps for mapping. Flag low-confidence mappings |
| GPU not available on deployment server | 10fps processing too slow on CPU | MediaPipe runs at ~15ms/frame on CPU (fine for 10fps). Only concern is if running 3 models simultaneously — stagger them |
| Blendshapes ≠ Action Units | Emotion detection accuracy lower than FACS-trained models | Accept the tradeoff — MediaPipe is free, real-time, and good enough (57-67% for emotions). Don't claim FACS-level accuracy |
| EU AI Act emotion recognition ban | Cannot use FACE-EMO-01 in EU for workplace | Frame as "facial movement analysis" not "emotion recognition". Offer EU-compliant mode that disables EMO-01 and VA-01, keeps only engagement/stress/smile |

---

## Recommended Start Order

**Don't build all sub-phases sequentially.** Some can overlap:

```
Week 1-2:   [2A Foundation] ←── MUST be first
Week 3-4:   [2B Facial] + start [2C Gaze]  ←── Both use face landmarks
Week 5-6:   [2C Gaze complete] + [2D Body]
Week 7-8:   [2D Body complete] + start [2E Fusion Pairwise]
Week 9-10:  [2E Fusion complete]
Week 11-12: [2F Compound] + [2G Temporal] ←── Can parallelize
```

This compresses the timeline from 14 weeks to **~10-12 weeks** with overlap.

**Start with Phase 2A Foundation.** Want me to create a Claude Code implementation prompt for it?
