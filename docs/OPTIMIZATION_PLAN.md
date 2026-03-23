# NEXUS Optimization Plan — Pipeline Parallelization

> How to make the analysis pipeline 2-3x faster with minimal code changes and zero functionality changes.

## Current Pipeline (Sequential)

```
Upload → Voice Agent (30-150s) → Language Agent (15-40s) → Fusion Agent (15-30s) → Done
         ════════════════════     ═══════════════════════    ══════════════════════
         Total: 60-220s (everything waits for the previous step)
```

## Optimized Pipeline (Parallel)

```
Upload ─┬─ Transcription (4-30s)
        │       │
        │       ├─→ Language Agent (starts immediately with transcript)
        │       │     ├─ Sentiment batches (parallel)
        │       │     ├─ Intent batches (parallel)
        │       │     └─ Entity extraction (parallel)
        │       │
        │       └─→ Voice Features + Rules (continues in background)
        │               ├─ Speaker_0 features (parallel)
        │               ├─ Speaker_1 features (parallel)
        │               └─ Speaker_2 features (parallel)
        │
        ├─ DB persistence (background, non-blocking)
        │
        └─→ Fusion Agent
              ├─ Per-speaker fusion rules (parallel)
              ├─ Graph + Analytics (parallel with narrative)
              └─ LLM Narrative (parallel with graph)

Total: 25-80s (2-3x faster)
```

---

## Optimization 1: Start Language Agent After Transcription, Not After Full Voice Analysis

**Priority: CRITICAL — saves 30-50% of total time**

### Problem

```python
# api_gateway/main.py — current flow (lines 199-266)
voice_result = await _call_voice_agent(session_id, file_path)  # 30-150s BLOCKS
# ... Language Agent can only start AFTER voice is completely done
language_result = await _call_language_agent(session_id, segments, meeting_type)
```

The Language Agent only needs **transcript segments** (text + timestamps + speakers). It does NOT need voice features, baselines, or voice signals. But it currently waits for ALL of that to finish.

### Solution

Split Voice Agent into two phases:

**Phase A (fast):** Transcribe + Diarize → return transcript segments immediately
**Phase B (slow):** Feature extraction + Calibration + Rules → return voice signals

```
Current:  [────── Voice Agent (transcribe + features + rules) ──────] → [── Language ──]
Proposed: [── Transcribe ──] → [── Language ──]    (runs in parallel)
                             → [── Voice features + rules ──]
```

### Files to change

#### `services/voiceAgent/main.py` — Add a new endpoint

```python
@app.post("/transcribe")
async def transcribe_only(request: TranscribeRequest):
    """Fast path: return transcript + diarization without features/rules."""
    transcript = transcriber.transcribe(str(file_path), num_speakers=request.num_speakers)
    return {
        "session_id": session_id,
        "duration_seconds": transcript["duration_seconds"],
        "segments": transcript["segments"],
        "speakers": list(set(seg["speaker"] for seg in transcript["segments"])),
    }
```

No change to existing `/analyse` endpoint — it still does the full pipeline.

#### `services/api_gateway/main.py` — Parallel orchestration

```python
# Step 1: Get transcript fast
transcript_result = await _call_voice_transcribe(session_id, file_path, num_speakers)
segments = transcript_result["segments"]

# Step 2: Run Voice analysis + Language analysis IN PARALLEL
voice_task = asyncio.create_task(_call_voice_agent(session_id, file_path, num_speakers))
language_task = asyncio.create_task(_call_language_agent(session_id, segments, meeting_type))

# Step 3: Wait for both
voice_result, language_result = await asyncio.gather(voice_task, language_task)
```

### Time saved

| Audio length | Current | Optimized | Saved |
|-------------|---------|-----------|-------|
| 1 min | 42s | 28s | 33% |
| 5 min | 130s | 85s | 35% |
| 15 min | 328s | 200s | 39% |

### Complexity: Medium (new endpoint + gateway orchestration change)

---

## Optimization 2: Parallelize Voice Feature Extraction Per Speaker

**Priority: HIGH — saves 40-60% of Voice Agent time**

### Problem

```python
# feature_extractor.py — extract_all() (line 57-101)
for speaker_id in speakers:           # Sequential per speaker
    speaker_segments = [...]
    for win_start in range(...):      # Sequential per window
        features = self._extract_features(chunk, sr, ...)  # SLOW: 200-500ms per window
        speaker_features.append(features)
```

For a 2-speaker 5-minute call: ~40 windows per speaker × 2 speakers = 80 sequential calls to `_extract_features`. At 300ms each = **24 seconds**.

Each speaker's features are **completely independent** — no shared state.

### Solution

```python
# feature_extractor.py — extract_all()
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def extract_all(self, audio_path, segments):
    y, sr = self._load_audio(audio_path)  # Load once
    speakers = set(seg["speaker"] for seg in segments)

    # Parallelize across speakers
    with ThreadPoolExecutor(max_workers=len(speakers)) as executor:
        futures = {}
        for speaker_id in speakers:
            speaker_segments = [s for s in segments if s["speaker"] == speaker_id]
            futures[speaker_id] = executor.submit(
                self._extract_speaker_features, y, sr, speaker_segments
            )

        features_by_speaker = {}
        for speaker_id, future in futures.items():
            result = future.result()
            if result:
                features_by_speaker[speaker_id] = result

    return features_by_speaker

def _extract_speaker_features(self, y, sr, speaker_segments):
    """Extract all windows for one speaker. Thread-safe — no shared mutable state."""
    speaker_features = []
    window_samples = int(WINDOW_SIZE_SEC * sr)
    hop_samples = int(HOP_SIZE_SEC * sr)
    for win_start_sample in range(0, len(y) - window_samples + 1, hop_samples):
        # ... existing window extraction logic ...
        features = self._extract_features(speaker_audio, sr, win_segments, ...)
        if features:
            speaker_features.append(features)
    return speaker_features
```

### Files to change

| File | Function | Change |
|------|----------|--------|
| `services/voiceAgent/feature_extractor.py` | `extract_all()` | Wrap speaker loop in ThreadPoolExecutor |

### Time saved

| Speakers | Current | Optimized | Saved |
|----------|---------|-----------|-------|
| 2 | 24s | 12s | 50% |
| 3 | 36s | 12s | 67% |

### Complexity: Medium (threading + ensure no shared mutable state)

---

## Optimization 3: Eliminate Redundant Audio Loading

**Priority: HIGH — saves 3-9 seconds**

### Problem

Audio is loaded from disk **3 separate times** during a single analysis:

| Load # | Location | Function | When |
|--------|----------|----------|------|
| 1 | `feature_extractor.py:49` | `_load_audio()` | Feature extraction |
| 2 | `transcriber.py:454` | `load_audio()` | KMeans diarization |
| 3 | `transcriber.py:639` | `load_audio()` | Gap+pitch diarization fallback |

Each load: decode + resample to 16kHz mono = 1-3s for large files.

### Solution

Load audio ONCE in `analyse_audio()` and pass the numpy array through:

```python
# voiceAgent/main.py — analyse_audio()
# Step 1: Load audio once
from shared.utils.audio_loader import load_audio
y, sr = load_audio(str(file_path), sr=16000)

# Step 2: Transcribe (uses its own loading for Whisper, but diarization reuses y)
transcript = transcriber.transcribe(str(file_path), num_speakers, preloaded_audio=(y, sr))

# Step 3: Extract features (reuses y)
features_by_speaker = feature_extractor.extract_all_from_array(y, sr, transcript["segments"])
```

### Files to change

| File | Function | Change |
|------|----------|--------|
| `services/voiceAgent/main.py` | `analyse_audio()` | Load audio once, pass to both |
| `services/voiceAgent/transcriber.py` | `transcribe()` | Accept optional `preloaded_audio` param |
| `services/voiceAgent/transcriber.py` | `_diarize_simple()` | Accept optional `y, sr` params |
| `services/voiceAgent/transcriber.py` | `_diarize_acoustic_kmeans()` | Accept optional `y, sr` params |
| `services/voiceAgent/transcriber.py` | `_diarize_gap_two_speaker()` | Accept optional `y, sr` params |
| `services/voiceAgent/feature_extractor.py` | `extract_all()` | Add `extract_all_from_array(y, sr, segments)` variant |

### Time saved: 3-9 seconds (eliminates 2 redundant file reads + decodes)

### Complexity: Low (add optional parameters, no logic change)

---

## Optimization 4: Parallelize Fusion Per-Speaker Analysis

**Priority: HIGH — saves 30-40% of Fusion Agent time**

### Problem

```python
# fusion_agent/main.py (line 165-229)
for speaker_id in speakers:        # Sequential
    speaker_voice = buffer.get_signals(speaker_id, "voice", ...)
    speaker_language = buffer.get_signals(speaker_id, "language", ...)
    fusion_signals = rule_engine.evaluate(...)        # 1-3s per speaker
    state = compute_unified_state(...)                # <1s per speaker
    all_fusion_signals.extend(fusion_signals)
```

Each speaker's fusion analysis is independent.

### Solution

```python
# fusion_agent/main.py
import asyncio

async def _analyse_speaker(speaker_id, buffer, voice_dicts, language_dicts, rule_engine):
    speaker_voice = buffer.get_signals(speaker_id, "voice", ...)
    speaker_language = buffer.get_signals(speaker_id, "language", ...)
    fusion_signals = rule_engine.evaluate(...)
    state = compute_unified_state(...)
    alerts = [_create_alert(session_id, speaker_id, fs) for fs in fusion_signals if fs["confidence"] >= 0.50]
    return {"signals": fusion_signals, "state": state, "alerts": alerts}

# Run all speakers in parallel
results = await asyncio.gather(
    *[_analyse_speaker(sid, buffer, ...) for sid in speakers]
)
for r in results:
    all_fusion_signals.extend(r["signals"])
    all_unified_states.append(r["state"])
    all_alerts.extend(r["alerts"])
```

### Files to change

| File | Function | Change |
|------|----------|--------|
| `services/fusion_agent/main.py` | `analyse_signals()` | Wrap speaker loop in asyncio.gather |

### Time saved

| Speakers | Current | Optimized | Saved |
|----------|---------|-----------|-------|
| 2 | 6s | 3s | 50% |
| 3 | 9s | 3s | 67% |

### Complexity: Low (asyncio.gather, no shared state)

---

## Optimization 5: Use Async LLM Calls (`acomplete` instead of `complete`)

**Priority: MEDIUM — saves 10-30% of LLM waiting time**

### Problem

All LLM calls use the **synchronous** `complete()` function, which blocks the event loop:

| Call site | File | Function | Blocking? |
|-----------|------|----------|-----------|
| Sentiment batches | `language_agent/feature_extractor.py` | `llm_complete()` | Yes — sync |
| Intent classification | `language_agent/rules.py` | `llm_complete()` | Yes — sync |
| Entity extraction | `language_agent/entity_extractor.py` | `acomplete()` | No — already async! |
| Narrative | `fusion_agent/narrative.py` | `llm_complete()` | Yes — sync |

The `shared/utils/llm_client.py` already has `acomplete()` — it's just not used in 3 of 4 places.

### Solution

Replace `complete()` with `acomplete()` in the 3 blocking call sites:

```python
# language_agent/feature_extractor.py — _llm_batch_sentiment()
# BEFORE:
raw = llm_complete(system_prompt=..., user_prompt=...)
# AFTER:
raw = await acomplete(system_prompt=..., user_prompt=...)

# language_agent/rules.py — _classify_intent_batch()
# BEFORE:
raw = llm_complete(system_prompt=..., user_prompt=...)
# AFTER:
raw = await acomplete(system_prompt=..., user_prompt=...)

# fusion_agent/narrative.py — generate_session_narrative()
# BEFORE:
raw_text = llm_complete(system_prompt=..., user_prompt=...)
# AFTER:
raw_text = await acomplete(system_prompt=..., user_prompt=...)
```

### Files to change

| File | Function | Change |
|------|----------|--------|
| `services/language_agent/feature_extractor.py` | `_llm_batch_sentiment()` | `complete()` → `await acomplete()` |
| `services/language_agent/rules.py` | `_classify_intent_batch()` | `complete()` → `await acomplete()` |
| `services/fusion_agent/narrative.py` | `generate_session_narrative()` | `complete()` → `await acomplete()` |

### Benefit

With async LLM calls, the event loop can process other work while waiting for API responses. This is especially impactful when multiple LLM batches run:

```
Current:   [batch1 2s] → [batch2 2s] → [batch3 2s] = 6s total
With async: [batch1─────────]
            [batch2─────────] = 2-3s total (concurrent)
            [batch3─────────]
```

### Time saved: 10-30% of LLM time (depends on API concurrency limits)

### Complexity: Low (change 3 function calls + add async/await)

---

## Optimization 6: Parallel Graph Building + Narrative Generation

**Priority: MEDIUM — saves 20-40% of Fusion Agent time**

### Problem

```python
# fusion_agent/main.py (lines 274-336)
# Step 4: Build graph + analytics (2-8s)
graph = SignalGraph()
graph.build_from_session(...)
analytics = GraphAnalytics(graph)
graph_insights = analytics.compute_all()

# Step 5: Generate narrative (5-10s) — waits for Step 4
report = generate_session_narrative(..., graph_analytics=graph_insights)
```

The narrative CAN run without graph analytics (it already has voice + language summaries + entities). Graph analytics just enriches the context.

### Solution

Run graph and narrative in parallel, then enrich:

```python
import asyncio

async def _build_graph_async(voice_dicts, language_dicts, fusion_signals, entities):
    graph = SignalGraph()
    graph.build_from_session(...)
    analytics = GraphAnalytics(graph)
    return graph, analytics.compute_all()

async def _generate_narrative_async(session_id, duration_seconds, speakers, ...):
    # Generate with basic context (no graph analytics)
    return generate_session_narrative(...)

# Run in parallel
graph_task = asyncio.create_task(_build_graph_async(...))
narrative_task = asyncio.create_task(_generate_narrative_async(...))

(graph, graph_insights), report = await asyncio.gather(graph_task, narrative_task)

# Optionally enrich report with graph insights after both complete
if report and graph_insights:
    report["graph_analytics_summary"] = {
        "tension_clusters": len(graph_insights.get("tension_clusters", [])),
        "trajectory": graph_insights.get("momentum", {}).get("overall_trajectory"),
    }
```

### Files to change

| File | Function | Change |
|------|----------|--------|
| `services/fusion_agent/main.py` | `analyse_signals()` | asyncio.gather for graph + narrative |
| `services/fusion_agent/narrative.py` | `generate_session_narrative()` | Make graph_analytics fully optional |

### Time saved

```
Current:  [── graph 5s ──] → [── narrative 8s ──] = 13s
Parallel: [── graph 5s ──────]                     = 8s (38% faster)
          [── narrative 8s ──]
```

### Complexity: Low (asyncio.gather, narrative already handles empty graph_analytics)

---

## Optimization 7: Background Database Persistence

**Priority: LOW — saves 2-5 seconds**

### Problem

```python
# api_gateway/main.py — all DB writes block the critical path
await upsert_speakers(session_id, voice_speakers)           # 0.5s
await insert_signals(session_id, voice_signals, speaker_map)  # 1s
await insert_transcript_segments(session_id, segments, ...)   # 1s
# ... Language Agent call waits for all above to finish
language_result = await _call_language_agent(...)
```

### Solution

Fire-and-forget DB writes for non-critical data:

```python
# Persist in background — don't block the pipeline
asyncio.create_task(insert_signals(session_id, voice_signals, speaker_map))
asyncio.create_task(insert_transcript_segments(session_id, segments, speaker_map))

# Continue immediately to Language Agent
language_result = await _call_language_agent(session_id, segments, meeting_type)
```

### Caveat

Speaker upsert MUST complete before signal insertion (foreign key). So upsert stays blocking, but signal + transcript writes can be background.

### Files to change

| File | Function | Change |
|------|----------|--------|
| `services/api_gateway/main.py` | `create_session_endpoint()` | `asyncio.create_task()` for signal/transcript persistence |

### Time saved: 2-5 seconds

### Complexity: Low (but must handle error logging for background tasks)

---

## Optimization 8: LLM Batch Parallelization in Language Agent

**Priority: MEDIUM — saves 15-25 seconds for long calls**

### Problem

Sentiment and intent batches run **sequentially**:

```python
# language_agent/main.py (lines 157-179)
features_list = feature_extractor.extract_all(segments)   # includes sequential LLM sentiment batches
# ... then ...
intent_signals = rule_engine.evaluate_batch_intent(features_list)  # sequential LLM intent batches
```

For 100 segments: 9 sentiment batches (2-3s each) + 7 intent batches (3-5s each) = **39-62 seconds sequential**.

### Solution

Run sentiment and intent batches concurrently:

```python
# Option A: Run sentiment extraction and intent classification in parallel
# This requires splitting extract_all() to not include sentiment inline

# Step 1: Extract non-LLM features (fast, <1s)
features_list = feature_extractor.extract_features_no_llm(segments)

# Step 2: Run LLM tasks in parallel
sentiment_task = asyncio.create_task(
    feature_extractor.batch_sentiment_async(segments)
)
intent_task = asyncio.create_task(
    rule_engine.evaluate_batch_intent_async(features_list)
)
entity_task = asyncio.create_task(
    entity_extractor.extract(segments, content_type)
)

sentiments, intents, entities = await asyncio.gather(
    sentiment_task, intent_task, entity_task
)

# Step 3: Merge sentiment into features
for i, features in enumerate(features_list):
    features.update(sentiments[i])
```

### Time saved

```
Current:  [─ sentiment 25s ─] → [─ intent 20s ─] → [─ entities 8s ─] = 53s
Parallel: [─ sentiment 25s ────────]                                    = 25s (53% faster)
          [─ intent 20s ──────]
          [─ entities 8s ─]
```

### Files to change

| File | Function | Change |
|------|----------|--------|
| `services/language_agent/feature_extractor.py` | `extract_all()` | Split out LLM sentiment to separate async method |
| `services/language_agent/main.py` | `analyse_transcript()` | Run sentiment + intent + entities via asyncio.gather |
| `services/language_agent/rules.py` | `evaluate_batch_intent()` | Make async |

### Complexity: Medium (requires restructuring extract_all)

---

## Combined Impact Estimate

### For a 5-minute 2-speaker sales call (current: ~130s)

| Optimization | Time saved | New total |
|-------------|-----------|-----------|
| Current baseline | — | 130s |
| 1. Language parallel with Voice | -45s | 85s |
| 2. Voice features parallel per speaker | -10s | 75s |
| 3. Eliminate redundant audio loads | -5s | 70s |
| 4. Fusion parallel per speaker | -3s | 67s |
| 5. Async LLM calls | -8s | 59s |
| 6. Graph + Narrative parallel | -5s | 54s |
| 7. Background DB writes | -3s | 51s |
| 8. LLM batch parallelization | -15s | 36s |
| **Total optimized** | **-94s** | **~36s (72% faster)** |

### For a 15-minute 3-speaker internal meeting (current: ~330s)

| Optimization | Time saved | New total |
|-------------|-----------|-----------|
| Current baseline | — | 330s |
| All optimizations combined | -230s | **~100s (70% faster)** |

---

## Implementation Order

```
Week 1: Optimizations 1 + 3 (biggest impact, medium effort)
         ├─ Add /transcribe endpoint to Voice Agent
         ├─ Parallel gateway orchestration
         └─ Pass preloaded audio through diarization chain

Week 2: Optimizations 2 + 4 (second biggest impact)
         ├─ ThreadPoolExecutor for voice feature extraction
         └─ asyncio.gather for fusion per-speaker

Week 3: Optimizations 5 + 6 + 7 (quick wins)
         ├─ Switch 3 complete() calls to acomplete()
         ├─ Parallel graph + narrative
         └─ Background DB writes

Week 4: Optimization 8 (largest refactor)
         └─ Split Language Agent LLM calls into parallel streams
```

---

## Functions Reference

Every function that would need changes, grouped by file:

### `services/api_gateway/main.py`

| Function | Line | Current | Optimized |
|----------|------|---------|-----------|
| `create_session_endpoint()` | 143 | Sequential: voice → language → fusion | Parallel: transcribe → (voice features ∥ language) → fusion |
| `_call_voice_agent()` | 566 | Single call, waits for full result | Split into transcribe + analyse |
| Signal persistence | 227-264 | Blocking before language call | `asyncio.create_task()` background |

### `services/voiceAgent/main.py`

| Function | Line | Current | Optimized |
|----------|------|---------|-----------|
| `analyse_audio()` | 100 | Loads audio implicitly in each sub-step | Load once, pass array |
| New: `transcribe_only()` | — | Does not exist | Fast transcript endpoint |

### `services/voiceAgent/feature_extractor.py`

| Function | Line | Current | Optimized |
|----------|------|---------|-----------|
| `extract_all()` | 31 | Sequential per speaker | ThreadPoolExecutor per speaker |
| `_extract_features()` | 143 | Called in sequential loop | Called from thread pool |

### `services/voiceAgent/transcriber.py`

| Function | Line | Current | Optimized |
|----------|------|---------|-----------|
| `_diarize_simple()` | 380 | Loads audio from disk | Accept optional `y, sr` params |
| `_diarize_acoustic_kmeans()` | 428 | Loads audio from disk | Accept optional `y, sr` params |
| `_diarize_gap_two_speaker()` | 615 | Loads audio from disk | Accept optional `y, sr` params |

### `services/language_agent/main.py`

| Function | Line | Current | Optimized |
|----------|------|---------|-----------|
| `analyse_transcript()` | 129 | Sequential: features → rules → intent → entities | Parallel: (sentiment ∥ intent ∥ entities) → rules |

### `services/language_agent/feature_extractor.py`

| Function | Line | Current | Optimized |
|----------|------|---------|-----------|
| `extract_all()` | 273 | Includes LLM sentiment inline | Split out sentiment to async method |
| `_llm_batch_sentiment()` | ~413 | Sync `complete()` | Async `acomplete()` |

### `services/language_agent/rules.py`

| Function | Line | Current | Optimized |
|----------|------|---------|-----------|
| `evaluate_batch_intent()` | ~207 | Sync `complete()` in loop | Async `acomplete()` with concurrent batches |

### `services/fusion_agent/main.py`

| Function | Line | Current | Optimized |
|----------|------|---------|-----------|
| `analyse_signals()` | 131 | Sequential per-speaker loop | `asyncio.gather()` per speaker |
| Graph + Narrative | 274-336 | Sequential: graph → narrative | `asyncio.gather()` parallel |

### `services/fusion_agent/narrative.py`

| Function | Line | Current | Optimized |
|----------|------|---------|-----------|
| `generate_session_narrative()` | 45 | Sync `complete()` | Async `acomplete()` |
