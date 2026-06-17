# Quality-Adaptive Tiny-Face Tracking — Implementation Plan

> Status: APPROVED (2026-06-16). Default OFF (`FACE_QUALITY_ADAPTIVE=0`) — ships dark, zero
> regression until explicitly enabled. Phased rollout. Owner: video agent.

## Context

Every face gate in the NEXUS video agent is hardcoded on **normalized** `face_box_area`
(a fraction of the frame), which conflates "small fraction of frame" with "low quality."
A face occupying 1.5% of a **4K** frame is ~300px tall and crisp — perfectly trackable and
ArcFace-reliable — yet it is silently rejected by the decisive `face_box_area > 0.02`
CentroidTracker gate, so it gets no `face_index` and is invisible to every downstream
engine (windowing, facial/body/gaze rules, ASD, speaker mapping). Conversely, 0.02 of a
blurry 480p frame is ~68px of mush that *should* stay rejected.

The fix is to make the gates a function of **pixel size and sharpness**, not frame fraction.
The frame's pixel dimensions are already available at the gate (`_frame_w`/`_frame_h`,
`feature_extractor.py:3690-3691`), and the conversion is exact:
`pixel_height = sqrt(face_box_area) * _frame_h` (the codebase already uses `area**0.5` as the
face-height ratio throughout). Reference values for the current `0.02` gate:
**153px @1080p · 204px @1440p · 305px @4K** — i.e. high-resolution faces are penalized purely
for being a small fraction of a large frame.

**Decisions (confirmed with user):**
- **Comprehensive**: full per-video quality descriptor (resolution + variance-of-Laplacian
  sharpness), a face-quality tier, resolution-relative ArcFace guards, and a per-face
  confidence multiplier so admitted small faces are *safe*, not merely *possible*.
- **No regression / opt-in**: master env kill-switch, **default OFF**. At ~1080p ("STANDARD"
  tier) the pixel baselines reproduce today's normalized constants exactly. Relaxation only
  applies to genuinely high-quality video (≥1440p **and** sharp). Phased rollout, one-flag
  rollback.
- **All layouts**: applies to grid/podcast **and** fixed-camera room/interrogation footage.
  Interrogation keeps face↔speaker mapping disabled, but faces are still tracked for
  facial/body/gaze signals, so a 4K room camera will now track distant seated subjects.

The intended outcome: 4K/1440p sharp video tracks faces that occupy a small fraction of the
frame (the headline win); blurry/low-res video stays strict or gets stricter; existing 1080p
output is unchanged until explicitly enabled.

## Core insight & pixel math

Convert each normalized constant `K` into `pixel_K / _frame_h`, with `pixel_K` chosen per
tier so that **STANDARD (~1080p) reproduces today's behavior bit-for-bit**:

| Gate | Today (normalized) | Today's px @1080p | New: `pixel_K / _frame_h` |
|---|---|---|---|
| CentroidTracker input (decisive) | `area > 0.02` → `area**0.5 > 0.141` | **153px** | `min_track_px`: PRISTINE 90 · STANDARD 153 · DEGRADED 153 · POOR 175 |
| ArcFace / IdentityVerifier reliable | `min_face_area = 0.005` | **76px** | `arcface_reliable_px`: PRISTINE 60 · STANDARD 76 · DEGRADED 90 · POOR 110 |
| ASD crop buffer | `area >= 0.001` | **34px** | `asd_min_px`: PRISTINE 28 · STANDARD 34 · DEGRADED 40 · POOR 48 |
| Tiny-face merge guard | `face_h < 0.07` | **76px** | `merge_guard_px`: PRISTINE 60 · STANDARD 76 · DEGRADED 90 · POOR 110 |

Asymmetry is deliberate: **PRISTINE lowers** the pixel floor (admit small-fraction but
large/crisp faces); **DEGRADED/POOR raise** it (soft pixels are unreliable even when
numerous); **STANDARD == today**. Web-grounded anchors: MediaPipe landmark mesh is stable
from ~64–80px face height; ArcFace consumes a 112×112 crop with a reliable floor of ~80–112px;
variance-of-Laplacian (`cv2.Laplacian(gray, CV_64F).var()`) is the standard cheap sharpness
metric and must be computed on a **fixed-size** crop to be comparable.

## Design overview

1. **`VideoQuality` descriptor**, computed once per video inside `_extract_frames`, carrying
   `frame_w/h`, `fps`, a median sharpness score, a `FaceQualityTier`, and the four derived
   **pixel** thresholds above (plus `.default()` = 1080p/neutral that reproduces today's
   constants).
2. **`FaceQualityTier`** enum (PRISTINE / STANDARD / DEGRADED / POOR), resolution-dominant and
   sharpness-modulated (sharpness may demote one band, or promote DEGRADED→… only when
   genuinely ≥1440p sharp — never let an upscaled soft video reach PRISTINE).
3. **All four gates read pixel thresholds** from the descriptor (converted back to the
   normalized comparisons the code already uses, so edits stay minimal).
4. **Per-face confidence multiplier** discounts signals from smaller/softer faces.
5. **Master env flag + phased rollout**; default OFF → `VideoQuality.default()` → today's
   numbers exactly.

## Reuse existing infrastructure (do not rebuild)

- **`MeetingProfile` + `apply_profile()`** (`feature_extractor.py:303-323`, `3393-3430`):
  thread an enable flag and any per-profile policy via the existing `_profile_*` setattr
  pattern, read with defensive `getattr(self, "_profile_*", default)` (as at `3682`, `3697`,
  `3720`).
- **`VideoQualityTier` + `_tier_from_fps()`** (`interrogation_rules.py:54-68`): keep as-is for
  the fps-based interrogation confidence table; add a **parallel** `FaceQualityTier` +
  `tier_from_quality()` combiner so interrogation behavior is untouched.
- **`_compute_merge_threshold` / `_GRID_FACE_H_THRESHOLD`** (`feature_extractor.py:5305-5333`):
  these classify *layout* (grid vs room) from a genuinely normalized signal — **leave
  unchanged**. Only the *quality/noise* guards move to pixels.
- **`conf_mult` pattern** (`facial_rules.py:133`, `gaze_rules.py:88`) and shared
  `base_rule_engine.py`: add the quality multiplier here.
- **`MIN_FACE_SIGNALS = 10`** weak-face filter (`analysis_pipeline.py:~345`): the existing
  backstop that drops marginal admitted faces — no change, relied upon.
- **Env-knob pattern**: mirror `ASD_TARGET_FPS = int(os.getenv(...))` (`feature_extractor.py:44`).

## Implementation steps

All file paths are `backend/services/video_agent/` unless noted. Primary work is in
`feature_extractor.py`.

### Step 1 — Env knobs + `VideoQuality` / `FaceQualityTier`
- Add env constants near line 44: `FACE_QUALITY_ADAPTIVE` (master, default `"0"` = off),
  `FACE_SHARP_VAR_SHARP` (120), `FACE_SHARP_VAR_SOFT` (40), and the per-tier pixel baselines
  (or hold them in a small module table). Mirror the `ASD_TARGET_FPS` pattern.
- Add `FaceQualityTier` enum in `interrogation_rules.py` next to `VideoQualityTier`; add
  `tier_from_quality(fps, frame_h, sharpness_norm)` returning `(VideoQualityTier, FaceQualityTier)`.
- Add `@dataclass VideoQuality` near `MeetingProfile` (`feature_extractor.py:~303`) with the
  fields above and a `default()` classmethod whose derived thresholds reproduce
  `0.02/0.005/0.001/0.07` at 1080p. Defining it here avoids importing `feature_extractor`
  into `interrogation_rules` (preserves the existing one-way dependency).

### Step 2 — Compute the descriptor once per video (sharpness from already-decoded crops)
- In `_extract_frames`, right after `_frame_w/_frame_h` are read (`~3691`) and **before** the
  tracker is built (`~3684`): initialize `VideoQuality` from **resolution + fps with neutral
  sharpness** (resolution is the dominant axis and the only one needed pre-loop).
- Accumulate sharpness **in-loop** on the first ~20 sampled frames using
  `cv2.Laplacian(_crop, CV_64F).var()` on the **96×96 grayscale crop already built for ASD**
  (`~3919-3925`) — zero extra decode/resize; the fixed 96×96 size makes the variance
  comparable across faces. Take the median. Then finalize the tier and **back-fill** the
  pixel thresholds consumed post-loop (ArcFace merge at `~4158-4197`) and on later frames.
- Store `self._video_quality = vq`; default safely to `VideoQuality.default()` on any
  zero-dimension/exception (zero-regression, like `MeetingProfile.default()`).
- When `FACE_QUALITY_ADAPTIVE` is off **or** tier is STANDARD, all derived thresholds equal the
  literal constants → no behavior change.

### Step 3 — Repoint the four gates (the decisive edits)
- **CentroidTracker input** `feature_extractor.py:3857` and `:3864`: replace `> 0.02` with
  `> _min_track_area`, where `_min_track_area = (vq.min_track_px / _frame_h) ** 2` computed
  once before the loop. (4K PRISTINE: `(90/2160)**2 ≈ 0.0017` admits a 90px face; 1080p
  STANDARD: `(153/1080)**2 ≈ 0.0201` ≈ unchanged.)
- **ArcFace IdentityVerifier** ctor at `:3702` (and the skip at `:2519-2521`): switch
  `min_face_area=0.005` to a pixel test (`face.bbox` is already pixel coords, `~2513`) using
  `vq.arcface_reliable_px`.
- **ASD crop gate** `:3916` and `:3937`: replace `>= 0.001` with `>= _asd_min_area`.
- **Tiny-face merge guard** `_merge_tracks_by_embedding` (staticmethod, `:5335`): add a
  `merge_guard_h: float = 0.07` kwarg, pass `vq.merge_guard_px / _frame_h` from the call site
  (`~4188-4199`), and convert the three `0.07` comparisons in `_effective_thresh`
  (`~5403/5421/5425`) to use it (compare in pixels). Update the docstring (`5359-5362`). Use
  `arcface_reliable_px`/112-class values here (this guard is about embedding *reliability*),
  so a 4K face at `face_h=0.09` (~194px) is correctly treated as **clean**, while a true 60px
  face still gets the strict 0.55/0.45 floor.
- **Best-frame selection** `:4049`: rank by **pixel** area, not normalized area
  (`quality = frontal * (area * _frame_h * _frame_h) * max(luminance,0.2)`), and lower the
  hard `250`px crop floor (`~4080`) toward `vq.asd_min_px` so PRISTINE small faces aren't
  force-upscaled.

### Step 4 — Per-face confidence multiplier (makes admission safe)
- Add `_face_quality_mult(w, vq)` to `base_rule_engine.py` (both engines extend it): ramps
  from 1.0 (≥reliable-px, sharp) down to a 0.6 floor (small/soft); never zeros out.
- Multiply into `conf_mult` at `facial_rules.py:133` and `gaze_rules.py:88` (slots in beside
  the existing `face_detection_rate` / luminance factors). Body rules optional/phase-3.
- Pass `vq` via a new `evaluate(..., video_quality=None)` kwarg (defaults to `None` →
  multiplier returns 1.0 → zero regression); wire the two call sites in `main.py:~629-636`
  from `getattr(self._extractor, "_video_quality", VideoQuality.default())`.

### Step 5 — Thread the enable flag via `apply_profile`; optional interrogation fix
- Add `face_quality_adaptive_enabled: bool = False` to `MeetingProfile` and store it in
  `apply_profile` as `_profile_face_quality_enabled` (read with `getattr(..., False)`). All
  factories default to off → no implicit per-type change. "All layouts" = no layout gating
  on the relaxation; room/grid/active_speaker all benefit when enabled.
- **Optional latent-bug fix (in scope for "all layouts"):** `BLINK_FACE_HS_MIN = 0.10`
  (`interrogation_rules.py:133`) is a 480p-derived constant applied at every resolution. Make
  it pixel-based (`BLINK_FACE_PX_MIN = 120`, normalized by `frame_h` at the call site
  `~322-330`), passing `frame_h` into `InterrogationVideoRules.evaluate` (default `0` → falls
  back to `0.10`). Tightens correctly at 1080p, unchanged at 480p.

## Risk guards (only relax in HIGH quality)

- **CentroidTracker ID inflation**: only lower `min_track_px` below STANDARD when tier ==
  PRISTINE, and tighten the tracker `match_threshold` proportionally for small-face videos
  (`effective = tracker_threshold * clamp(min_track_px/153, 0.6, 1.0)`) at construction
  (`~3684`). The post-loop `_merge_tracks_by_embedding` remains a safety net that merges
  spurious duplicate IDs of the same person.
- **ArcFace noise <~76px**: `arcface_reliable_px` only drops to 60 at PRISTINE; the merge
  guard (now pixel-based) still applies the strict 0.55/0.45 cosine floor to sub-guard faces
  even at 4K — relaxation admits a face for *tracking* while keeping *merging* conservative.
- **Unreliable landmarks/blendshapes → weak signals**: discounted by Step 4 and dropped by the
  unchanged `MIN_FACE_SIGNALS=10` filter; nothing extra needed.
- **Global invariant**: every relaxation is wrapped in
  `if _profile_face_quality_enabled and FACE_QUALITY_ADAPTIVE and tier == PRISTINE`.
  DEGRADED/POOR only ever *tighten*; STANDARD is identical to today.
- Do **not** alter the protected `_compute_merge_threshold` layout constants (conference
  base 0.50 / decay 0.008 / floor 0.40, grid floor 0.42, `_GRID_FACE_H_THRESHOLD=0.20`).

## Phased rollout (no-regression / opt-in)

1. **Phase 0 (ship dark)**: `FACE_QUALITY_ADAPTIVE=0`. Descriptor computed for logging only;
   gates fall back to literals → bit-identical output. (Phase 0 is also the "minimal core"
   commit if isolated.)
2. **Phase 1 (observe)**: log per video `frame_h, fps, sharpness, face_tier`, and the count of
   faces that *would* flip admission under the new gate. Validate tier assignment on a corpus;
   no gate change.
3. **Phase 2 (PRISTINE-only)**: enable, but `_face_tier` returns STANDARD for everything except
   `frame_h>=1440 && sharp`. Only 4K-sharp video changes — exactly the target case.
4. **Phase 3 (full)**: enable DEGRADED/POOR tightening + the Step 4 confidence multiplier.
5. **Rollback**: flip the single env var. No code revert.

## Verification

**Unit (pure functions, no video):**
- `VideoQuality.default()` yields `_min_track_area ≈ 0.0201`, `_arcface_min_area ≈ 0.005`,
  `_asd_min_area ≈ 0.001`, `merge_guard_h ≈ 0.0707` (proves zero-regression math at 1080p).
- `_face_tier(2160,"sharp")==PRISTINE`; `_face_tier(2160,"soft")==STANDARD` (demotion);
  `_face_tier(480,"sharp")==POOR` (resolution ceiling); `_face_tier(720,"neutral")==DEGRADED`.
- Round-trip `((px/h)**2)` then `area**0.5*h ≈ px`.

**Integration (headline test):**
- A 4K sharp clip with a person at ~1.5% frame area (~265px tall). With flag **off**: face is
  rejected (no `Face_N`, reproduces the bug). With flag **on**: a stable `Face_N` track
  appears, signals are produced carrying `face_quality_mult < 1.0`, and logs show
  `face_tier=PRISTINE`.
- **Negative control**: down-res that clip to 480p + blur; with flag on it stays rejected
  (POOR raises the bar) and produces no new garbage signals.
- **Regression control**: run the existing 1080p corpus flag-on vs flag-off — signal sets must
  be **identical** (STANDARD = no-op).
- **ID-inflation check**: 4K multi-face grid/room — confirm post-merge canonical track count
  matches a manual headcount (no explosion); confirm the pixel-based merge guard still fires
  for genuinely tiny faces.

## Critical files

- `backend/services/video_agent/feature_extractor.py` — env knobs (~44), `VideoQuality` +
  `MeetingProfile` (~303-323), `apply_profile` (3393-3430), descriptor compute in
  `_extract_frames` (~3684-3938), the four gates (3702, 3857/3864, 3916/3937, 4049/4080,
  5359-5427), merge-call wiring (~4188-4199).
- `backend/services/video_agent/interrogation_rules.py` — `FaceQualityTier` + `tier_from_quality`
  beside `VideoQualityTier`/`_tier_from_fps` (54-68); optional `BLINK_FACE_HS_MIN`→pixel fix
  (129-133, 322-330); `evaluate` signature (~214-238).
- `backend/services/video_agent/main.py` — read `_video_quality` and pass to rule engines
  (~629-636) and interrogation rules (~696-703); profile/fps wiring (425-460).
- `backend/services/video_agent/base_rule_engine.py` — shared `_face_quality_mult`.
- `backend/services/video_agent/facial_rules.py` (133) and `gaze_rules.py` (88) — apply the
  multiplier.
- `backend/pipeline/analysis_pipeline.py` — read-only: confirm `MIN_FACE_SIGNALS` backstop
  (~345) still catches marginal faces.
