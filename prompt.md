# NEXUS — Decouple LR-ASD crop frame-rate from the 5fps behavioural sampling (FPS FIX ONLY)

Make ONE change: feed the LR-ASD active-speaker model a higher-frame-rate
crop stream so it runs closer to its 25fps training distribution, WITHOUT
changing the 5fps behavioural sampling. Do nothing else in this task — no
other bug fixes, no refactors, no threshold changes.

## Problem

LR-ASD (Liao et al., IJCV 2025) was trained at ~25fps (one visual frame per
~40ms, 4 MFCC frames per visual frame). NEXUS currently buffers ASD face
crops at the behavioural sampling rate `TARGET_FPS = 5`
(`backend/services/video_agent/feature_extractor.py:41`) and passes `fps=5.0`
into the model:

- `backend/services/video_agent/feature_extractor.py:4350` —
  `self._last_video_fps = float(self._target_fps)`  (= 5.0)
- `backend/services/video_agent/main.py:547-550` —
  `_face_crops = getattr(self._extractor, "_face_crops_sequence", {})`
  `_fps = getattr(self._extractor, "_last_video_fps", 5.0)`
  `asd_scores = _asd.score(_face_crops, _tmp_audio.name, fps=_fps)`

At 5fps each video frame spans 200ms, so a full open→close mouth motion can
fall *between* sampled frames and the model never sees it. The 94.45 mAP
figure does not transfer to 5fps input. The MFCC hop inside the model is
`sr/(fps×4)` (`LightASDClassifier`, ~`feature_extractor.py:3038`,
`_load_mfcc` ~3155, `_infer_chunk` ~3229), so it already adapts to whatever
`fps` it is given — the only problem is that the crops AND the `fps` value
are both stuck at 5.

## The change

1. **Add a separate, configurable ASD crop frame-rate**, independent of
   `TARGET_FPS`. Read it from an env var, e.g.
   `ASD_TARGET_FPS = int(os.getenv("ASD_TARGET_FPS", "25"))`, default **25**
   (LR-ASD's training rate). Place the constant next to `TARGET_FPS`.

2. **Densify ONLY the ASD crop buffer (`_face_crops_sequence`)** to
   `ASD_TARGET_FPS`, leaving every other consumer of the frame loop at 5fps.
   First READ how `_face_crops_sequence` is currently populated in
   `feature_extractor.py` (find where crops are appended during the frame
   loop and where `_face_crops_sequence` / `_last_video_fps` are set), then
   choose the least-invasive way to produce crops at the higher rate:
   - Preferred: decode video frames at `ASD_TARGET_FPS` cadence and crop the
     tracked faces for the ASD buffer only, while the behavioural pipeline
     keeps consuming the 5fps frames exactly as today.
   - Acceptable alternative if full re-detection at 25fps is too costly on
     CPU: keep face detection at 5fps and interpolate each track's crop box
     across the intermediate frames, cropping at `ASD_TARGET_FPS`.
   Pick whichever keeps the change localized; state which you chose in a
   comment.

3. **Set `_last_video_fps` (the value passed to `score()`) to the actual
   cadence of the crops you buffered** — i.e. `ASD_TARGET_FPS`, NOT 5. The
   `fps` argument at `main.py:550` must equal the real crop spacing so the
   model's `sr/(fps×4)` MFCC hop and audio↔visual alignment stay correct.
   `_last_video_fps` is the only fps coupling to fix; do not rename it unless
   trivially safe.

4. **Keep per-chunk audio anchoring on real timestamps.** Crops carry true
   PTS from `CAP_PROP_POS_MSEC`; `_infer_chunk` re-anchors audio from the
   chunk's first timestamp. Densifying to ~25fps makes the spacing more
   uniform (closer to training) — preserve the existing real-timestamp
   anchoring; do not switch to `frame_idx/fps` timing.

## Hard constraints — DO NOT TOUCH

- `TARGET_FPS = 5` and the behavioural frame sampling, windowing, and
  per-face signal attribution — must remain exactly as-is.
- Frame-count-based thresholds that assume 5fps: `CentroidTracker`
  `max_disappeared = 15`, `BLINK_CONSEC_FRAMES = 2`, and any similar
  frame-count constants — leave unchanged.
- `SpeakerFaceMapper`, the lip-sync correlation, the registry, diarization,
  and all rule engines — out of scope.
- No change to `_target_fps` or the value any non-ASD code reads.
- Do not modify Dockerfiles, requirements.txt, or the DB. Do not start or
  restart containers.

## Notes

- This is a **no-op for `interrogation_video`**: ASD is gated behind
  `not _is_interrogation` (`main.py:530`), so interrogation sessions never
  call `score()`. That is expected and correct — the change only affects the
  podcast / video-call / grid path that still does face↔speaker linking.
- CPU cost scales with `ASD_TARGET_FPS`. 25 matches training but is ~5× the
  current crop/detect work on the ASD path; if profiling shows it is too
  heavy on CPU, lowering `ASD_TARGET_FPS` to 15 via the env var is the
  intended tuning knob — keep the default at 25 in code.

## Verification

1. AST-parse the touched files
   (`feature_extractor.py`, `main.py`).
2. Confirm `score()` receives `fps == ASD_TARGET_FPS` and that
   `_face_crops_sequence` now contains ~`ASD_TARGET_FPS` crops per second per
   track (log the count per track at INFO and check against duration).
3. Confirm a podcast/grid session still completes and produces
   `face_to_speaker` links; confirm an `interrogation_video` session is
   unchanged (ASD still skipped).
4. Grep to confirm `TARGET_FPS`, `max_disappeared`, and `BLINK_CONSEC_FRAMES`
   are untouched.
5. Append a dated entry to `task_log.txt` describing only this fps change.
