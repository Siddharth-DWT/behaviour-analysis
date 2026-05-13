# NEXUS — Complete Face Identity Fix

## Three Problems, Three Layers

| Problem | What Happens | Where It Breaks | Fix |
|---------|-------------|----------------|-----|
| A. Tile swap creates duplicate | Ansuya moves from sidebar (0.91,Y) to center (0.35,Y). CentroidTracker distance 0.56 > threshold 0.10 → new Face_10 | feature_extractor.py line 2350: `match_threshold=0.10` | ArcFace merge handles cross-position duplicates (Fix 1) |
| B. ArcFace merge fails on asymmetric pair | Face_2 (small tile, noisy embedding) vs Face_10 (large tile, clean embedding). sim ≈ 0.45, threshold 0.55 → NOT merged | feature_extractor.py line 3593: `_effective_thresh` raises floor to 0.55 for ALL tiny-face pairs | Distinguish symmetric (both tiny) from asymmetric (one tiny, one large) pairs (Fix 1) |
| C. Face_N and Speaker_N have different registry_ids | Mapper disabled (line 3704). Gateway key mismatch: `face_embeddings.get("Speaker_0")` → empty because keys are "Face_N" | speaker_registry.py line 88, main.py line 1557 | Re-enable mapper for linkage only, remap keys in gateway (Fix 2+3) |

**Read these files before making changes:**
- services/video_agent/feature_extractor.py — `CentroidTracker` (line 2350), `_merge_tracks_by_embedding` (line 3550), `_effective_thresh` (line 3590), `SpeakerFaceMapper.assign` (line 3676, disabled at 3704)
- services/api_gateway/main.py — `_run_video` (line 1437), registry matching (line 1545), non-speaking registration (line 1583), canonical merge in `/video-signals` (line 2268) and `/video-speakers` (line 2382)
- services/api_gateway/speaker_registry.py — `match_or_create_speakers` (line 55, key lookup at line 88), `match_or_create_by_face_only` (line 311), thresholds (lines 26-28)

---

## Fix 1: ArcFace Merge — Handle Asymmetric Embedding Quality

**File:** services/video_agent/feature_extractor.py  
**Location:** `_effective_thresh` inside `_merge_tracks_by_embedding` — line 3590

The current code raises the floor to 0.55 when EITHER track has `face_h < 0.07`. This blocks merges where one track is tiny (noisy embedding) and the other is large (clean embedding) — same person viewed from different tile sizes.

The fix: distinguish three cases:

```python
        def _effective_thresh(tid_a: int, tid_b: int, sim: float) -> float:
            fh_a = face_hs.get(tid_a, 1.0)
            fh_b = face_hs.get(tid_b, 1.0)
            min_fh = min(fh_a, fh_b)
            max_fh = max(fh_a, fh_b)

            if min_fh >= 0.07:
                # Both normal-sized faces: use standard threshold
                return threshold

            if max_fh < 0.07:
                # SYMMETRIC: both tiny faces → both embeddings noisy
                # Keep strict floor to prevent false merges between
                # different people's noisy embeddings (both score 0.40-0.49)
                floor = max(threshold, 0.55)
            else:
                # ASYMMETRIC: one tiny (noisy), one large (clean)
                # The large embedding is reliable. The sim score is lower only
                # because the tiny embedding has noise, not because it's a
                # different person. Lower the floor to 0.45.
                # Different people: tiny vs large scores < 0.35 typically.
                # Same person: tiny vs large scores 0.40-0.55.
                # Threshold 0.45 sits in the gap.
                floor = max(threshold, 0.45)

            if sim >= floor:
                return floor

            # Position fallback for scores below floor but above noise (0.35)
            if sim > 0.35 and centroids:
                ca = centroids.get(tid_a)
                cb = centroids.get(tid_b)
                if ca and cb:
                    pos_dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                    if pos_dist < 0.05:
                        return threshold
            return floor
```

**What this changes for the HR session:**

| Pair | Face_h A | Face_h B | Type | Old Floor | New Floor | Sim | Result |
|------|:--------:|:--------:|:----:|:---------:|:---------:|:---:|:------:|
| Face_2 (small Ansuya) × Face_10 (large Ansuya) | 0.032 | 0.20+ | Asymmetric | 0.55 | **0.45** | ~0.48 | **MERGED** ✅ (was blocked) |
| Face_2 (small Ansuya) × Face_3 (small Mirko) | 0.032 | 0.05 | Symmetric | 0.55 | 0.55 | ~0.30 | Not merged ✅ (correct) |
| Sid fragment × Sid fragment | 0.032 | 0.035 | Symmetric | 0.55 | 0.55 | ~0.52 | Not merged ✅ (below 0.55) |
| Sid fragment × Sid large | 0.032 | 0.08+ | Asymmetric | 0.55 | **0.45** | ~0.50 | **MERGED** ✅ (was blocked) |

---

## Fix 2: Re-enable SpeakerFaceMapper for Linkage Only

**File:** services/video_agent/feature_extractor.py  
**Location:** `SpeakerFaceMapper.assign` — line 3676

Delete the early return at line 3704. Replace with linkage-only behavior:

```python
    def assign(
        self,
        windows: list[WindowFeatures],
        diar_segments: list[dict],
        lip_activity_map: Optional[dict[int, list[tuple[int, float]]]] = None,
    ) -> tuple[dict[str, list[WindowFeatures]], dict[str, float], dict[int, str]]:
        """
        Returns 3-tuple:
          windows_by_face:   {"Face_0": [...], "Face_3": [...]}
          lip_sync_scores:   {"Speaker_0": 0.25}
          face_to_speaker:   {2: "Speaker_0", 3: "Speaker_1"}

        Windows keep Face_N speaker_ids. face_to_speaker is used ONLY by the
        gateway for registry linking — NOT for rewriting signal ownership.
        """
        result: dict[str, list[WindowFeatures]] = defaultdict(list)

        if not windows:
            return dict(result), {}, {}

        speakers = sorted(set(seg.get("speaker", "Speaker_0") for seg in diar_segments))
        if not speakers:
            speakers = ["Speaker_0"]

        face_indices = sorted(set(getattr(wf, "face_index", 0) for wf in windows))

        # ── Run lip-sync correlation (existing code, unchanged) ──────────
        use_lip_sync = (
            lip_activity_map is not None
            and len(face_indices) > 1
            and len(speakers) > 1
        )

        if use_lip_sync:
            face_to_speaker, assignment_scores = self._lip_sync_assignment(
                face_indices, speakers, diar_segments, lip_activity_map
            )
            method = "lip_sync"
        else:
            face_to_speaker, assignment_scores = self._time_overlap_assignment(
                face_indices, speakers, windows, diar_segments
            )
            method = "time_overlap"

        # ── Group windows by Face_N — DO NOT rewrite speaker_id ──────────
        for wf in windows:
            face_idx = getattr(wf, "face_index", 0)
            face_label = f"Face_{face_idx}"
            wf.speaker_id = face_label
            wf.is_speaking = self._is_speaking_in_window(
                wf.window_start_ms, wf.window_end_ms,
                face_to_speaker.get(face_idx, ""),
                diar_segments,
            )
            result[face_label].append(wf)

        # ── Build linkage map (Speaker_N assignments only) ───────────────
        lip_sync_scores: dict[str, float] = {}
        confident_mapping: dict[int, str] = {}
        for fi, spk in face_to_speaker.items():
            if spk.startswith("Speaker_"):
                lip_sync_scores[spk] = round(assignment_scores.get(fi, 0.0), 4)
                confident_mapping[fi] = spk

        logger.info(
            "SpeakerFaceMapper: %d face(s), method=%s, linkage=%s",
            len(face_indices), method, confident_mapping,
        )
        return dict(result), lip_sync_scores, confident_mapping
```

All existing methods (`_lip_sync_assignment`, `_time_overlap_assignment`, `_greedy_assign`) remain unchanged.

---

## Fix 3: Gateway — Use face_to_speaker for Registry Key Remapping

**File:** services/api_gateway/main.py

### 3a. Update _run_video to extract face_to_speaker (line 1437)

```python
    async def _run_video() -> tuple[list[dict], dict, dict, str]:
        """Returns (signals, face_embeddings, face_to_speaker, video_job_id)."""
        # ... existing try/except ...
        sigs      = result.get("signals", [])
        face_embs = result.get("face_embeddings", {})
        f2s       = result.get("face_to_speaker", {})
        return sigs, face_embs, f2s, vid_job_id
```

Update gather unpacking (line 1460):
```python
    ..., (vid_signals, face_embeddings_from_video, face_to_speaker, video_job_id) = await asyncio.gather(...)
```

### 3b. Remap face_embeddings keys before registry matching (BEFORE line 1545)

```python
    # ── Remap face_embeddings: Face_N → Speaker_N using lip-sync linkage ─────
    # match_or_create_speakers (speaker_registry.py line 88) does:
    #   face_data = face_embeddings.get(speaker_label)  ← needs Speaker_N keys
    # Without remapping: face_embeddings has Face_N keys → get("Speaker_0") → empty
    # → voice-only registry → no thumbnail → "S" initial in sidebar
    speaker_keyed_face_embs: dict[str, dict] = {}
    if face_to_speaker:
        for face_idx_int, speaker_label in face_to_speaker.items():
            face_label = f"Face_{face_idx_int}"
            if face_label in face_embeddings_from_video:
                speaker_keyed_face_embs[speaker_label] = face_embeddings_from_video[face_label]
                logger.info(
                    f"[{session_id}] Face-voice link: {face_label} → {speaker_label}"
                )
```

### 3c. Pass remapped embeddings to match_or_create_speakers (line 1557)

```python
    face_embeddings=speaker_keyed_face_embs or face_embeddings_from_video,
```

Now `face_embeddings.get("Speaker_0")` → finds Ansuya's ArcFace → fused entry → thumbnail stored.

### 3d. Link Face_N to Speaker_N's registry_id (after line 1607)

```python
    # ── Link Face_N → Speaker_N registry entries ─────────────────────────────
    # Ensures /video-signals canonical merge (line 2268) and /video-speakers
    # canonical merge (line 2382) can group Face_N + Speaker_N as one person.
    if face_to_speaker:
        for face_idx_int, speaker_label in face_to_speaker.items():
            face_label = f"Face_{face_idx_int}"
            if speaker_label in speaker_identity_map and face_label not in speaker_identity_map:
                speaker_identity_map[face_label] = {
                    **speaker_identity_map[speaker_label],
                    "match_method": "lip_sync_link",
                }
                logger.info(
                    f"[{session_id}] Linked {face_label} → {speaker_label} "
                    f"(registry_id={speaker_identity_map[speaker_label]['registry_id']})"
                )
```

### 3e. Non-speaking registration skips linked faces (line 1587 — already correct)

```python
    if label.startswith("Face_") and label not in speaker_identity_map
    # After 3d, linked Face_N IS in speaker_identity_map → excluded ✅
```

---

## Fix 4: Video Agent — Update Response

**File:** services/video_agent/main.py

### 4a. Update SpeakerFaceMapper call (3-tuple)

```python
    windows_by_speaker, lip_sync_scores, face_to_speaker = mapper.assign(
        windows, diar_segments, lip_activity_map
    )
```

### 4b. Include face_to_speaker in response

```python
    return {
        "signals": all_signals,
        "face_embeddings": face_embeddings_data,
        "face_to_speaker": face_to_speaker,
        ...
    }
```

### 4c. Step 5 — key face_embeddings by Face_N always

```python
    face_label = f"Face_{track_id}"
    face_embeddings_data[face_label] = {
        "embedding": embedding,
        "thumbnail_b64": base64.b64encode(thumbnail).decode(),
    }
```

---

## Fix 5: Frontend — Skip Empty Groups

**File:** dashboard/src/components/VideoSignalPlayer.tsx — line 992

```typescript
const visibleSigs = showExpanded ? prioritySigs : prioritySigs.slice(0, 3);
if (visibleSigs.length === 0) return null;
```

---

## How All Three Problems Are Solved

### Problem A: Tile swap creates Face_2 + Face_10

```
Before:
  Face_2 (small tile, 73px) → noisy embedding
  Face_10 (large tile, 300px) → clean embedding
  _effective_thresh: min(0.032, 0.20) < 0.07 → floor = 0.55
  sim ≈ 0.48 < 0.55 → NOT MERGED → two Ansuya entries

After (Fix 1):
  _effective_thresh: min(0.032, 0.20) < 0.07, max(0.032, 0.20) >= 0.07 → ASYMMETRIC
  floor = 0.45
  sim ≈ 0.48 > 0.45 → MERGED → one Face_2 (canonical)
```

### Problem B: Face_N and Speaker_N disconnected

```
Before:
  face_embeddings = {"Face_2": Ansuya_emb}
  match_or_create_speakers: face_embeddings.get("Speaker_0") → {} → voice-only
  Face_2 → non-speaking registration → separate registry_id

After (Fix 2+3):
  face_to_speaker = {2: "Speaker_0"}
  speaker_keyed_face_embs = {"Speaker_0": Ansuya_emb}
  match_or_create_speakers: face_embeddings.get("Speaker_0") → Ansuya_emb → fused registry
  Face_2 linked to Speaker_0's registry_id → canonical merge at line 2268 works
```

### Combined result:

```
Sidebar:
  Speaker_0 / Ansuya (thumbnail): Agreeing, Eye Contact, Tense, Authoritative
      ↑ Face_2 + Face_10 merged (Fix 1)
      ↑ Face_2 + Speaker_0 linked (Fix 2+3)
      ↑ Fusion signals merged via canonical (line 2268, already exists)
  Speaker_1 / Mirko (thumbnail): Eye Contact, Authoritative
  Speaker_2 / Sid (thumbnail): Fully Focused, Deflated
      ↑ Sid fragments merge better (asymmetric threshold 0.45)
```

## Files Modified:
1. **services/video_agent/feature_extractor.py** — `_effective_thresh` asymmetric pair handling (Fix 1), `SpeakerFaceMapper.assign` linkage-only (Fix 2)
2. **services/video_agent/main.py** — 3-tuple from mapper, include face_to_speaker in response, Step 5 Face_N keying (Fix 4)
3. **services/api_gateway/main.py** — extract face_to_speaker, remap keys, link registry_ids (Fix 3)
4. **dashboard/src/components/VideoSignalPlayer.tsx** — skip empty groups (Fix 5)