# NEXUS — Fix Face-to-Speaker Mapping Pipeline

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

## Files to Modify (3 files, 5 changes)

```
services/video_agent/feature_extractor.py
  - build_lip_activity_map()     — line ~3486
  - _lip_sync_assignment()       — line ~6056
  - MIN_LIP_SYNC_LINK_SCORE      — line 43

services/video_agent/main.py
  - calibration_confidence write  — in _build_summaries or run_analysis

backend/pipeline/analysis_pipeline.py
  - _SESSION_FACE_LOCK_MIN_SCORE  — line 52
```

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read `services/video_agent/feature_extractor.py` — COMPLETELY:
   - `build_lip_activity_map()` — current implementation builds
     `{face_index: [(timestamp_ms, lip_score), ...]}` where `lip_score`
     is a weighted composite of jaw_open (0.50), mouth_lower_down (0.20),
     mouth_funnel (0.15), mouth_pucker (0.10), mouth_close_inv (0.05).
     These are all ABSOLUTE blendshape values (0.0-1.0 range).
   - `_lip_sync_assignment()` — current implementation computes per-(face, speaker)
     correlation = mean(lip during speech) - mean(lip during silence).
     Uses `_hungarian_assign` for globally-optimal pairing.
   - `MIN_LIP_SYNC_LINK_SCORE` = 0.10 — minimum correlation to accept a link.
   - `SpeakerFaceMapper.assign()` — priority: ASD > lip_sync > time_overlap.
     ASD path works correctly (binary labels, scale-independent).
     time_overlap path works correctly (single-speaker sessions).
     ONLY the lip_sync path has the absolute-score bias.

2. Read `services/video_agent/main.py`:
   - `VideoPipeline.run_analysis()` — where lip_activity_map is built and
     passed to mapper.assign()
   - `_build_summaries()` — where speaker summaries (including calibration_confidence)
     are built per speaker. Check if Face_N entries get calibration_confidence.

3. Read `backend/pipeline/analysis_pipeline.py`:
   - `_build_session_face_locks()` — reads lip_sync_scores, rejects entries
     below _SESSION_FACE_LOCK_MIN_SCORE (0.06).
   - `_SESSION_FACE_LOCK_MIN_SCORE` default value and env var.

4. Use proper OOP and DSA:
   - **Delta computation**: O(T) per face where T = timestamps
   - **MAD normalization**: O(T) median + O(T) median of abs deviations
   - **bisect**: O(log S) for diar segment lookup per timestamp
   - **Hungarian assignment**: unchanged — already optimal

5. Do NOT change the ASD path or time_overlap path. ONLY change the lip_sync path.
   ASD works correctly. time_overlap works correctly. The fix is isolated to
   `build_lip_activity_map()` and `_lip_sync_assignment()`.

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
# Returns: {face_idx: [(ts_ms, lip_score), ...]}
```

### New:
```python
def build_lip_activity_map(self, frames) -> dict[int, list[tuple[int, float]]]:
    """
    Build per-face lip activity timeseries using frame-to-frame DELTA
    instead of absolute blendshape values.

    Delta measures jaw MOTION — a small face moving its jaw produces the
    same normalized delta as a large face moving its jaw. This eliminates
    the systematic bias toward larger faces in _lip_sync_assignment.

    Steps:
      1. Compute raw composite per frame (same weights as before)
      2. Compute |delta| between consecutive frames per face → O(T)
      3. MAD-normalize per face so all faces are on the same scale → O(T)

    DSA: O(F × T) where F = faces, T = frames per face.
    """
    # Step 1: raw composites per face (existing code, unchanged)
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

    # Step 2 + 3: delta + MAD normalize per face
    result: dict[int, list[tuple[int, float]]] = {}
    for fi, raw_series in raw_by_face.items():
        if len(raw_series) < 3:
            result[fi] = [(ts, 0.0) for ts, _ in raw_series]
            continue

        # Sort by timestamp (should already be sorted, but guarantee)
        raw_series.sort(key=lambda x: x[0])

        # Compute absolute deltas between consecutive frames
        deltas = []
        for i in range(1, len(raw_series)):
            delta = abs(raw_series[i][1] - raw_series[i - 1][1])
            deltas.append((raw_series[i][0], delta))

        # MAD normalization: median absolute deviation
        delta_values = [d for _, d in deltas]
        median_delta = sorted(delta_values)[len(delta_values) // 2]
        abs_devs = [abs(d - median_delta) for d in delta_values]
        mad = sorted(abs_devs)[len(abs_devs) // 2]

        # Normalize: (delta - median) / (MAD * 1.4826 + epsilon)
        # 1.4826 makes MAD consistent with std for normal distributions
        scale = mad * 1.4826 + 1e-6
        normalized = []
        for ts, delta in deltas:
            norm_score = max(0.0, (delta - median_delta) / scale)
            normalized.append((ts, round(norm_score, 4)))

        result[fi] = normalized

    return result
```

**Why MAD instead of std?** MAD is robust to outliers. A single large jaw movement
(yawn, cough) inflates std but barely affects MAD. This prevents one-off events
from distorting the normalization.

---

## Change 2: _lip_sync_assignment() — Segment-Level Voting

**File:** `services/video_agent/feature_extractor.py` — `_lip_sync_assignment()`

The current correlation (mean during speech - mean during silence) works on
the ENTIRE session. Replace with segment-level voting: for each diar segment,
compute the correlation, then vote. This is more robust because one speaker's
long monologue doesn't dominate the score.

### Replace the correlation computation inside _lip_sync_assignment:

```python
def _lip_sync_assignment(
    self,
    face_indices: list[int],
    speakers: list[str],
    diar_segments: list[dict],
    lip_activity_map: dict[int, list[tuple[int, float]]],
) -> tuple[dict[int, str], dict[int, float]]:
    """
    Assign faces to speakers using delta-normalized lip activity.

    Segment-level voting:
      For each (face, speaker) pair, for each diar segment of that speaker:
        1. Compute mean lip_delta during segment (speech)
        2. Compute mean lip_delta during ±2s around segment (silence)
        3. score_segment = speech_mean - silence_mean
      Aggregate: weighted mean by segment duration (longer = more reliable)

    Quality gates per segment:
      - Duration ≥ 500ms (very short segments are noise)
      - Face visible ratio ≥ 0.50 (face must be detected in ≥50% of segment frames)

    DSA: O(F × S × T/S) ≈ O(F × T) where F=faces, S=segments, T=total timestamps.
    bisect for O(log T) per segment boundary lookup.
    """
    scores: dict[tuple[int, str], float] = {}

    for fi in face_indices:
        series = lip_activity_map.get(fi, [])
        if not series:
            continue

        timestamps = [t for t, _ in series]
        values = [v for _, v in series]

        for spk in speakers:
            spk_segs = [
                seg for seg in diar_segments
                if seg.get("speaker") == spk
            ]
            if not spk_segs:
                continue

            seg_scores = []
            seg_weights = []

            for seg in spk_segs:
                seg_start = seg.get("start_ms", 0)
                seg_end = seg.get("end_ms", 0)
                seg_dur = seg_end - seg_start

                # Quality gate: segment duration
                if seg_dur < 500:
                    continue

                # Find lip values during this segment — bisect O(log T)
                i_start = bisect.bisect_left(timestamps, seg_start)
                i_end = bisect.bisect_right(timestamps, seg_end)
                speech_vals = values[i_start:i_end]

                # Quality gate: face visible ratio
                if len(speech_vals) < max(1, (i_end - i_start) * 0.50):
                    continue

                if not speech_vals:
                    continue

                speech_mean = sum(speech_vals) / len(speech_vals)

                # Silence: 2 seconds before and after the segment
                sil_start = max(0, seg_start - 2000)
                sil_end = min(timestamps[-1] if timestamps else seg_end, seg_end + 2000)
                i_sil_start = bisect.bisect_left(timestamps, sil_start)
                i_sil_end = bisect.bisect_right(timestamps, sil_end)
                silence_vals = (
                    values[i_sil_start:i_start]
                    + values[i_end:i_sil_end]
                )

                silence_mean = (
                    sum(silence_vals) / len(silence_vals)
                    if silence_vals else 0.0
                )

                seg_score = speech_mean - silence_mean
                seg_scores.append(seg_score)
                seg_weights.append(seg_dur)

            # Weighted mean by segment duration
            if seg_scores and sum(seg_weights) > 0:
                total_weight = sum(seg_weights)
                weighted_score = sum(
                    s * w for s, w in zip(seg_scores, seg_weights)
                ) / total_weight
                scores[(fi, spk)] = max(0.0, weighted_score)
            else:
                scores[(fi, spk)] = 0.0

    # Delegate to Hungarian assignment (unchanged)
    return self._hungarian_assign(face_indices, speakers, scores)
```

---

## Change 3: MIN_LIP_SYNC_LINK_SCORE 0.10 → 0.20

**File:** `services/video_agent/feature_extractor.py` — line 43

```python
# BEFORE:
MIN_LIP_SYNC_LINK_SCORE: float = 0.10

# AFTER:
MIN_LIP_SYNC_LINK_SCORE: float = 0.20
```

**Why 0.30:** With MAD-normalized delta scores, genuine speaker-face correlations
produce scores in the 0.40-0.80 range. Random noise correlations produce < 0.15.
0.20 is the midpoint — conservative enough to avoid false rejects, strict enough
to block noise.

The ASD path and active-tile sentinel (1.0) are both well above 0.20.
The time_overlap sentinel (1.0) is also above 0.20.
Only genuine lip-sync correlations need to clear this threshold.

---

## Change 4: _SESSION_FACE_LOCK_MIN_SCORE 0.06 → 0.10

**File:** `backend/pipeline/analysis_pipeline.py` — line 52

```python
# BEFORE:
_SESSION_FACE_LOCK_MIN_SCORE = float(os.getenv("SESSION_FACE_LOCK_MIN_SCORE", "0.06"))

# AFTER:
_SESSION_FACE_LOCK_MIN_SCORE = float(os.getenv("SESSION_FACE_LOCK_MIN_SCORE", "0.10"))
```

**Why 0.10:** The gateway threshold is the final gate before registry linking.
At 0.06, random noise from the old absolute-score system could pass.
With the new delta-based scores, genuine links produce ≥ 0.30 (from the
video agent's threshold). 0.10 is a safety floor — anything that passed the
video agent's 0.30 will easily clear 0.10 at the gateway.

The active-tile sentinel (1.0) and time_overlap sentinel (1.0) are unaffected.

---

## Change 5: Write calibration_confidence for Face_N

**File:** `services/video_agent/main.py` — `_build_summaries()`

The current code builds `SpeakerVideoSummary` per speaker in `windows_by_speaker`.
Since windows are keyed by `Face_N` (not `Speaker_N`), the summaries already
use `Face_N` keys. Check that `calibration_confidence` is populated.

```python
# In _build_summaries, verify this field exists:
for spk, wins in windows_by_speaker.items():
    baseline = baselines.get(spk)
    if baseline:
        facial_bl, body_bl, gaze_bl = baseline
        summaries[spk] = SpeakerVideoSummary(
            ...,
            calibration_confidence=facial_bl.confidence,  # ← must be here
        )
```

If `calibration_confidence` is missing from the summary, add it.

The gateway then needs to write this value to the speakers table for Face_N rows.
Check `analysis_pipeline.py` — the voice agent writes `calibration_confidence`
for Speaker_N during `_publish_voice_outputs`. The video agent's value needs
to be written similarly during Step 4 (registry matching).

```python
# In analysis_pipeline.py, after receiving video_response:
# For each Face_N in video_response.speaker_summaries:
#   UPDATE speakers SET calibration_confidence = summary.calibration_confidence
#   WHERE session_id = $1 AND speaker_label = face_label
```

---

## What NOT to Change

- ASD path (`_asd_assignment`) — works correctly, binary labels are scale-independent
- time_overlap path (`_time_overlap_assignment`) — works correctly for single-speaker
- Active-tile sentinel logic — already sets `lip_sync_scores[spk] = 1.0`
- `_hungarian_assign` — already globally optimal
- `SpeakerFaceMapper.assign()` routing logic — method priority ASD > lip_sync > time_overlap is correct
- `skip_speaker_link` for interrogation — correct by design

## Expected Results

| Scenario | Before | After |
|----------|:------:|:-----:|
| 2-person meeting, no ASD | Large face gets Speaker_0, small face DROPPED | Both faces mapped correctly |
| 3-person meeting, sidebar faces | Only active-tile face linked | All faces linked via delta correlation |
| Single speaker | Works (time_overlap) | Works (unchanged) |
| ASD available | Works | Works (unchanged) |
| Interrogation | Skip (by design) | Skip (unchanged) |
| Noise correlation | Passes at 0.06 gateway | Blocked at 0.30 video + 0.10 gateway |

## Files Modified:
1. **services/video_agent/feature_extractor.py**:
   - `MIN_LIP_SYNC_LINK_SCORE`: 0.10 → 0.30 (~1 line)
   - `build_lip_activity_map()`: delta + MAD normalization (~30 lines changed)
   - `_lip_sync_assignment()`: segment-level voting with quality gates (~50 lines changed)
2. **services/video_agent/main.py**:
   - Verify `calibration_confidence` in `_build_summaries` (~2 lines)
3. **backend/pipeline/analysis_pipeline.py**:
   - `_SESSION_FACE_LOCK_MIN_SCORE`: 0.06 → 0.10 (~1 line)
   - Write video calibration_confidence to speakers table for Face_N (~5 lines)