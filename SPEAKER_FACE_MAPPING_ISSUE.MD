# NEXUS — Face-Speaker Mapping: Open Issues

---

## Issue 1: Face_N Labels Instead of Speaker_N in Sidebar

**Symptom:** Every session shows Face_2, Face_3, Face_4 in the sidebar. Speaker_0, Speaker_1,
Speaker_2 never appear. Thumbnails are present but faces are not linked to voice speakers.

**Root cause:** `SpeakerFaceMapper.assign()` falls back to lip-sync correlation when
`ActiveTileTagger` produces no mapping. Lip-sync scores on large tiles are near zero because
MediaPipe jaw saturation makes the signal uninformative — the jaw is nearly always open on a
full-screen face, so active-speech and silence intervals look the same. `confident_face_to_speaker`
ends up empty and no Face_N gets promoted to Speaker_N.

**What needs to happen:** `ActiveTileTagger` must produce a non-empty `{face_index: "Speaker_N"}`
mapping. This requires the dominant speaker tile to exist (ratio of largest face area to
second-largest >= 1.5) and diarization segments to be populated when the frame loop runs.
If the session is gallery layout (all tiles equal size), the tagger deliberately skips tagging
and lip-sync becomes the only path — which fails for saturated jaws.

---

## Issue 2: Track Contamination — Same Face_N Contains Two Different People

**Symptom:** Face_2 carries signals from both people who occupied the same tile at different
times. Lip-sync score is diluted (~0.02–0.05) because the track mixes an active speaker's jaw
movement with a different person's silence. Signals show contradictory emotions on the same entry.

**Root cause:** When a video platform promotes a new speaker to the active tile, the new person
occupies the same screen position as the previous speaker. CentroidTracker matches by centroid
proximity — it assigns the new person to the existing track_id because their position is close.
The track now contains frames of two different people mixed together.

`IdentityVerifier` runs ArcFace every 30 sampled frames (6 seconds) and fires a swap event when
cosine similarity drops below threshold. `TrackletSplitter` then rewrites face_index on all
frames after the swap point, giving the new person a fresh track_id. But:

- Swaps shorter than 6 seconds are never detected — the check_interval is too wide
- The frames before each detected swap (up to 6s of contamination) are permanently in the wrong track
- Fragment tracks created by splitting have too few frames for reliable lip-sync

---

## Issue 3: Duplicate Face Entries (e.g. Face_14 alongside Face_2)

**Symptom:** Sidebar shows 4 entries for a 3-person call. Face_14 (99% cross-session match)
is the same person as one of the existing entries but appears as a separate row with its own signals.

**Root cause:** Two failures compound:

**First:** When a person's tile changes position (layout change after speaker promotion),
CentroidTracker creates a new track_id because the centroid moved beyond `match_threshold=0.10`.

**Second:** The ArcFace post-loop merge should unify the two tracks. It fails when one track
is large-tile (clean embedding) and the other was small-tile (noisy embedding, face_area < 0.005).
Cosine similarity between a clean embedding and a noisy one of the same person lands at 0.40–0.49.
The conference-room merge floor is 0.40, so the pair nominally passes — but 0.40–0.49 is also
the range of false-positive matches between completely different people with noisy embeddings.
The merge cannot safely fire at this similarity level.

**What is needed:** Per-track size-aware threshold inside `_merge_tracks_by_embedding()`. When
`min(face_h_A, face_h_B) < 0.07` (tiny face, face_area < 0.005), require similarity >= 0.60
to merge. A flat floor raise is not the answer — same-person tiny-face pairs also score 0.50–0.59,
so a global floor of 0.60 would break legitimate same-person merges. The threshold must be
applied only when at least one track in the pair is below the tiny-face size threshold.

---

## Issue 4: Wrong Signals on Correct Faces

**Symptom:** A smiling person shows "Disagreeing 60%". A person who just joined shows stress
signals that predate their presence.

**Root cause A — head_shake rule ignores smile context:**
The `body_head_shake` rule in `body_rules.py` fires on lateral head movement above threshold
without checking mouth state. A person shaking their head while smiling (laughing, amused
disbelief) is misclassified as disagreeing. Smile suppression is needed: skip the rule when
`(mouthSmileLeft + mouthSmileRight) / 2 > 0.30`.

**Root cause B — contamination window before swap detection:**
IdentityVerifier detects swaps at 6-second resolution. The frames before each detected swap
are permanently grouped into the wrong track's windows. Rule engines compute on those windows
and emit signals attributed to the wrong person's Face_N label. These signals cannot be
corrected retroactively once the windows are built.

---

## Issue 5: Brief Speaker Missed During Active-Tile Tagging

**Symptom:** A person who speaks for 3–5 seconds and briefly occupies the active tile is
not tagged by ActiveTileTagger and remains as Face_N.

**Root cause:** ActiveTileTagger tags based on face_box_area >= 0.08 and diarization speaker
at that timestamp. If the person's tile promotion lasted less than one IdentityVerifier check
interval (6 seconds), TrackletSplitter may not split the track. The frames from that brief
promotion are mixed into the previous speaker's track. ActiveTileTagger sees the previous
speaker's track_id at those timestamps and tags the wrong track with the brief speaker's
diarization label — or the dominant-tile check fails entirely if the brief speaker's tile
never reached the size threshold.
