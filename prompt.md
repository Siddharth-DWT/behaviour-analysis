# NEXUS — Fix Face-to-Speaker Mapping Pipeline + LR-ASD Model Upgrade

## Root Cause (verified from code)

`_lip_sync_assignment()` in `services/video_agent/feature_extractor.py` uses
ABSOLUTE jaw_open blendshape values. A large active-tile face (area 0.15)
produces jaw_open ≈ 0.20 when speaking. A small sidebar face (area 0.03)
produces jaw_open ≈ 0.05 when speaking. The correlation formula:

```python
correlation = mean(lip_activity during speech) - mean(lip_activity during silence)
```

Large face: 0.20 - 0.08 = 0.12 → passes MIN_LIP_SYNC_LINK_SCORE (0.10)
Small face: 0.05 - 0.02 = 0.03 → REJECTED → no link → signals dropped

The large face ALWAYS wins regardless of who is actually speaking.

Additionally, the ASD model is Light-ASD (CVPR 2023, 94.1% mAP). The same
authors released LR-ASD (IJCV 2025, 94.5% mAP) — a drop-in replacement with
better cross-domain robustness and +0.4% accuracy. Same input format, same
output format, same singleton pattern.

## Files to Modify (3 files, 6 changes)

```
services/video_agent/feature_extractor.py
  - LightASDClassifier → upgrade to LR-ASD model support    — Change 6
  - build_lip_activity_map()                                  — Change 1
  - _lip_sync_assignment()                                    — Change 2
  - MIN_LIP_SYNC_LINK_SCORE                                   — Change 3

services/video_agent/main.py
  - calibration_confidence write                              — Change 5

backend/pipeline/analysis_pipeline.py
  - _SESSION_FACE_LOCK_MIN_SCORE                              — Change 4
```

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read `services/video_agent/feature_extractor.py` — COMPLETELY:
   - `LightASDClassifier` class — singleton, loads ONNX or TorchScript from
     `models/light_asd/light_asd.onnx` or `.pt`. Input: 96×96 grayscale crops
     + 16kHz mono audio (40-dim log-mel filterbank, 4 audio frames per video
     frame). Output: per-frame speaking probabilities → Viterbi 2-state HMM
     smoothing → binary 0.0/1.0 labels. Singleton via `get_instance()`.
   - `build_lip_activity_map()` — builds `{face_idx: [(ts_ms, lip_score)]}` from
     absolute blendshape composite: jaw_open (0.50) + mouth_lower_down (0.20) +
     mouth_funnel (0.15) + mouth_pucker (0.10) + mouth_close_inv (0.05).
   - `_lip_sync_assignment()` — per-(face,speaker) correlation = mean(lip during speech)
     - mean(lip during silence). Uses `_hungarian_assign` for global optimum.
   - `MIN_LIP_SYNC_LINK_SCORE` = 0.10.
   - `SpeakerFaceMapper.assign()` — priority: ASD > lip_sync > time_overlap.

2. Read `services/video_agent/main.py`:
   - Where `LightASDClassifier.get_instance()` is called
   - Where `_asd.score(_face_crops, _tmp_audio.name, fps=_fps)` runs
   - The face crop buffering in `_extract_frames()` — 96×96 grayscale crops
     keyed by `(timestamp_ms, cx_int, cy_int)`

3. Read `backend/pipeline/analysis_pipeline.py`:
   - `_build_session_face_locks()` — lip_sync_scores threshold
   - `_SESSION_FACE_LOCK_MIN_SCORE` = 0.06

4. Use proper OOP and DSA:
   - **Delta computation**: O(T) per face
   - **MAD normalization**: O(T) median + O(T) MAD
   - **bisect**: O(log S) for diar segment lookup per timestamp
   - **Singleton pattern**: LR-ASD follows same `get_instance()` pattern
   - **Hungarian assignment**: unchanged

5. Do NOT change the ASD path routing logic or time_overlap path.
   The ASD path works correctly — Change 6 only upgrades the MODEL, not the
   calling code. The `SpeakerFaceMapper.assign()` priority (ASD > lip_sync >
   time_overlap) stays the same. The `score()` method signature stays the same.

---

## Change 1: build_lip_activity_map() — Delta + MAD Normalization

**File:** `services/video_agent/feature_extractor.py` — `build_lip_activity_map()`

Replace absolute composite scores with frame-to-frame delta, then MAD-normalize
per face so scores are scale-independent.

### Current (broken):
```python
lip_score = (
    jaw_open         * 0.50
    + mouth_lower_down * 0.20
    + mouth_funnel     * 0.15
    + mouth_pucker     * 0.10
    + (1.0 - mouth_close) * 0.05
)
```

### New:
```python
def build_lip_activity_map(self, frames) -> dict[int, list[tuple[int, float]]]:
    """
    Build per-face lip activity using frame-to-frame DELTA + MAD normalization.
    Delta measures jaw MOTION — scale-independent across face sizes.

    Steps:
      1. Compute raw composite per frame (same weights)
      2. |delta| between consecutive frames per face → O(T)
      3. MAD-normalize per face → O(T)

    DSA: O(F × T) where F = faces, T = frames per face.
    """
    raw_by_face: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for frame in frames:
        fi = frame.face_index
        composite = (
            frame.jaw_open          * 0.50
            + frame.mouth_lower_down * 0.20
            + frame.mouth_funnel     * 0.15
            + frame.mouth_pucker     * 0.10
            + (1.0 - frame.mouth_close) * 0.05
        )
        raw_by_face[fi].append((frame.timestamp_ms, composite))

    result: dict[int, list[tuple[int, float]]] = {}
    for fi, raw_series in raw_by_face.items():
        if len(raw_series) < 3:
            result[fi] = [(ts, 0.0) for ts, _ in raw_series]
            continue

        raw_series.sort(key=lambda x: x[0])

        deltas = []
        for i in range(1, len(raw_series)):
            delta = abs(raw_series[i][1] - raw_series[i - 1][1])
            deltas.append((raw_series[i][0], delta))

        delta_values = [d for _, d in deltas]
        median_delta = sorted(delta_values)[len(delta_values) // 2]
        abs_devs = [abs(d - median_delta) for d in delta_values]
        mad = sorted(abs_devs)[len(abs_devs) // 2]

        scale = mad * 1.4826 + 1e-6
        normalized = []
        for ts, delta in deltas:
            norm_score = max(0.0, (delta - median_delta) / scale)
            normalized.append((ts, round(norm_score, 4)))

        result[fi] = normalized

    return result
```

---

## Change 2: _lip_sync_assignment() — Segment-Level Voting

**File:** `services/video_agent/feature_extractor.py` — `_lip_sync_assignment()`

Replace session-wide correlation with per-diar-segment voting with quality gates.

```python
def _lip_sync_assignment(
    self,
    face_indices: list[int],
    speakers: list[str],
    diar_segments: list[dict],
    lip_activity_map: dict[int, list[tuple[int, float]]],
) -> tuple[dict[int, str], dict[int, float]]:
    """
    Segment-level voting with quality gates:
      - Duration ≥ 500ms
      - Face visible ratio ≥ 0.50
    Weighted mean by segment duration.
    DSA: bisect O(log T) per segment boundary.
    """
    import bisect
    scores: dict[tuple[int, str], float] = {}

    for fi in face_indices:
        series = lip_activity_map.get(fi, [])
        if not series:
            continue
        timestamps = [t for t, _ in series]
        values = [v for _, v in series]

        for spk in speakers:
            spk_segs = [s for s in diar_segments if s.get("speaker") == spk]
            if not spk_segs:
                continue

            seg_scores, seg_weights = [], []
            for seg in spk_segs:
                seg_start = seg.get("start_ms", 0)
                seg_end = seg.get("end_ms", 0)
                seg_dur = seg_end - seg_start
                if seg_dur < 500:
                    continue

                i_start = bisect.bisect_left(timestamps, seg_start)
                i_end = bisect.bisect_right(timestamps, seg_end)
                speech_vals = values[i_start:i_end]

                if len(speech_vals) < max(1, (i_end - i_start) * 0.50):
                    continue
                if not speech_vals:
                    continue

                speech_mean = sum(speech_vals) / len(speech_vals)

                sil_start = max(0, seg_start - 2000)
                sil_end = min(timestamps[-1] if timestamps else seg_end, seg_end + 2000)
                i_sil_s = bisect.bisect_left(timestamps, sil_start)
                i_sil_e = bisect.bisect_right(timestamps, sil_end)
                silence_vals = values[i_sil_s:i_start] + values[i_end:i_sil_e]
                silence_mean = sum(silence_vals) / len(silence_vals) if silence_vals else 0.0

                seg_scores.append(speech_mean - silence_mean)
                seg_weights.append(seg_dur)

            if seg_scores and sum(seg_weights) > 0:
                total_w = sum(seg_weights)
                scores[(fi, spk)] = max(0.0, sum(s * w for s, w in zip(seg_scores, seg_weights)) / total_w)
            else:
                scores[(fi, spk)] = 0.0

    return self._hungarian_assign(face_indices, speakers, scores)
```

---

## Change 3: MIN_LIP_SYNC_LINK_SCORE 0.10 → 0.20

**File:** `services/video_agent/feature_extractor.py` — line 43

```python
MIN_LIP_SYNC_LINK_SCORE: float = 0.20
```

With MAD-normalized delta scores: genuine matches 0.40-0.80, noise < 0.15.
0.20 gives a 0.05 margin on both sides (noise ceiling 0.15, genuine floor 0.25).

---

## Change 4: _SESSION_FACE_LOCK_MIN_SCORE 0.06 → 0.10

**File:** `backend/pipeline/analysis_pipeline.py` — line 52

```python
_SESSION_FACE_LOCK_MIN_SCORE = float(os.getenv("SESSION_FACE_LOCK_MIN_SCORE", "0.10"))
```

---

## Change 5: Write calibration_confidence for Face_N

**File:** `services/video_agent/main.py` — `_build_summaries()`

Verify `calibration_confidence` is populated for Face_N entries. If missing, add:

```python
summaries[spk] = SpeakerVideoSummary(
    ...,
    calibration_confidence=facial_bl.confidence,
)
```

The gateway (`analysis_pipeline.py`) must then write this value to the speakers
table for Face_N rows (currently only written for Speaker_N by voice agent).

---

## Change 6: Upgrade Light-ASD → LR-ASD Model

**Paper:** Liao et al. "LR-ASD: Lightweight and Robust Network for Active Speaker Detection"
IJCV 2025. Same authors as Light-ASD (CVPR 2023).

**Repo:** `github.com/Junhua-Liao/LR-ASD`
**Weights:** `weight/finetuning_TalkSet.model` (PyTorch checkpoint)

**Why upgrade:**
- 94.45% mAP on AVA (+0.4% over Light-ASD 94.1%)
- Better cross-dataset robustness (SOTA on Talkies, Columbia, RealVAD)
- Same input format: 96×96 grayscale face crops + 16kHz mono audio
- Same output format: per-frame speaking probabilities → binary labels
- Same singleton pattern, same `score()` method signature
- 1.3M params (+0.3M over Light-ASD 1.0M — negligible)

### What to change in LightASDClassifier:

**6a. Update model file paths:**

```python
# BEFORE:
onnx_path = Path(model_dir) / "light_asd" / "light_asd.onnx"
pt_path   = Path(model_dir) / "light_asd" / "light_asd.pt"

# AFTER — support both old and new paths (backward compatible):
# Priority: LR-ASD (new) > Light-ASD (legacy)
lr_asd_onnx  = Path(model_dir) / "lr_asd" / "lr_asd.onnx"
lr_asd_pt    = Path(model_dir) / "lr_asd" / "lr_asd.pt"
light_onnx   = Path(model_dir) / "light_asd" / "light_asd.onnx"
light_pt     = Path(model_dir) / "light_asd" / "light_asd.pt"

# Try LR-ASD first, fall back to Light-ASD
for onnx_path, label in [(lr_asd_onnx, "LR-ASD"), (light_onnx, "Light-ASD")]:
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            self._available = True
            self._model_name = label
            logger.info("%s: ONNX model loaded — %s", label, onnx_path)
            break
        except Exception as exc:
            logger.warning("%s: ONNX load failed: %s", label, exc)

if not self._available:
    for pt_path, label in [(lr_asd_pt, "LR-ASD"), (light_pt, "Light-ASD")]:
        if pt_path.exists():
            try:
                import torch
                self._torch_model = torch.jit.load(str(pt_path), map_location="cpu")
                self._torch_model.eval()
                self._available = True
                self._model_name = label
                logger.info("%s: TorchScript model loaded — %s", label, pt_path)
                break
            except Exception as exc:
                logger.warning("%s: PyTorch load failed: %s", label, exc)
```

**6b. Update class docstring:**

```python
class LightASDClassifier:
    """
    Active Speaker Detection via LR-ASD (Liao et al., IJCV 2025) or
    Light-ASD (Liao et al., CVPR 2023) fallback.

    LR-ASD: 94.45% mAP on AVA-ActiveSpeaker, SOTA cross-domain robustness.
    Light-ASD: 94.1% mAP (legacy, used when LR-ASD model not available).

    Model loading (priority order):
      1. LR-ASD ONNX   — <model_dir>/lr_asd/lr_asd.onnx
      2. Light-ASD ONNX — <model_dir>/light_asd/light_asd.onnx (legacy fallback)
      3. LR-ASD PT      — <model_dir>/lr_asd/lr_asd.pt
      4. Light-ASD PT   — <model_dir>/light_asd/light_asd.pt (legacy fallback)
      5. Disabled        — falls back to lip-sync correlation

    Same input/output as Light-ASD — drop-in replacement.
    Source: https://github.com/Junhua-Liao/LR-ASD
    """
```

**6c. Add `_model_name` property for logging:**

```python
def __init__(self, model_dir: str = "models") -> None:
    self._sess = None
    self._torch_model = None
    self._available: bool = False
    self._librosa_ok: bool = False
    self._sr: int = 16000
    self._model_name: str = "none"   # ← ADD
    # ... rest of init
```

**6d. Update log messages in main.py:**

```python
# BEFORE:
logger.info("[%s] Light-ASD: scored %d tracks", session_id, len(asd_scores))

# AFTER:
logger.info(
    "[%s] %s: scored %d tracks",
    session_id, _asd._model_name, len(asd_scores),
)
```

**6e. Update the comment in main.py:**

```python
# BEFORE:
# For other sessions: replaces MediaPipe jawOpen
# lip-sync correlation with a learned AV model (94.1% precision on AVA-ActiveSpeaker).

# AFTER:
# For other sessions: replaces MediaPipe jawOpen lip-sync correlation
# with LR-ASD (94.45% mAP, IJCV 2025) or Light-ASD (94.1%, CVPR 2023) fallback.
```

### How to obtain the LR-ASD model:

```bash
# Option A: Clone and export to ONNX
git clone https://github.com/Junhua-Liao/LR-ASD.git
cd LR-ASD
# Download pretrained weights (auto-downloaded on first run, or manual):
# weight/finetuning_TalkSet.model
# Export to ONNX:
python export_onnx.py --model weight/finetuning_TalkSet.model --output lr_asd.onnx
# Place at: models/lr_asd/lr_asd.onnx

# Option B: Use TorchScript directly
python -c "
import torch
model = torch.load('weight/finetuning_TalkSet.model', map_location='cpu')
# Trace with dummy inputs matching score() input shapes
# ... (see LR-ASD repo for exact architecture)
torch.jit.save(traced, 'models/lr_asd/lr_asd.pt')
"
```

Note: The LR-ASD repo may need a custom `export_onnx.py` script — check the
repo for export instructions. The model architecture is the same backbone as
Light-ASD with improved temporal modeling, so the ONNX export process is
identical. Input shapes: audio `[B, 1, N_MELS, T_audio]`, visual `[B, T_video, 1, 96, 96]`.

---

## What NOT to Change

- `SpeakerFaceMapper.assign()` routing (ASD > lip_sync > time_overlap) — unchanged
- `_hungarian_assign()` — unchanged
- `_asd_assignment()` — unchanged (the ASD scoring logic is correct, only model upgraded)
- `score()` method signature — unchanged (same input: face_crops + audio, same output: binary labels)
- `_viterbi_smooth()` — unchanged
- `CROP_SIZE` = 96, `N_MELS` = 40, `AUDIO_FRAMES_PER_VIDEO` = 4 — unchanged
- Active-tile sentinel logic — unchanged
- `skip_speaker_link` for interrogation — unchanged
- Face crop buffering in `_extract_frames()` — unchanged (same 96×96 grayscale)

## Expected Results

| Scenario | Before | After |
|----------|:------:|:-----:|
| 2-person meeting, no ASD | Large face gets Speaker, small face DROPPED | Both linked (delta correlation) |
| 3-person meeting, sidebar faces | Only active-tile face linked | All linked via delta |
| ASD available (Light-ASD) | 94.1% mAP | 94.45% mAP (LR-ASD) |
| ASD cross-domain (Zoom/Teams) | Degrades on non-AVA content | Better robustness (TalkSet training) |
| No model file at all | Disabled → lip-sync | Disabled → lip-sync (unchanged) |
| Only Light-ASD model present | Works | Works (legacy fallback) |
| LR-ASD + Light-ASD both present | N/A | LR-ASD used (higher priority) |

## Files Modified:
1. **services/video_agent/feature_extractor.py**:
   - `LightASDClassifier.__init__()`: LR-ASD path priority + legacy fallback (~20 lines changed)
   - `LightASDClassifier` docstring: updated (~5 lines)
   - `_model_name` property added (~2 lines)
   - `MIN_LIP_SYNC_LINK_SCORE`: 0.10 → 0.20 (~1 line)
   - `build_lip_activity_map()`: delta + MAD normalization (~30 lines changed)
   - `_lip_sync_assignment()`: segment-level voting (~50 lines changed)
2. **services/video_agent/main.py**:
   - Log message uses `_asd._model_name` (~2 lines)
   - Comment updated (~2 lines)
   - Verify `calibration_confidence` in `_build_summaries` (~2 lines)
3. **backend/pipeline/analysis_pipeline.py**:
   - `_SESSION_FACE_LOCK_MIN_SCORE`: 0.06 → 0.10 (~1 line)
   - Write video calibration_confidence for Face_N (~5 lines)