# Pipeline Speed-Up Plan — Audit Report

Audit of the proposed speed-up plan against the actual monolithic backend implementation.
Several suggestions are based on incorrect assumptions about the current codebase.

---

## Suggestions Already Implemented

### #3 — MediaPipe Frame Skipping
Already done. `_target_fps = 5` and `skip = max(1, round(video_fps / self._target_fps))` is
live in `backend/services/video_agent/feature_extractor.py`. For a 30fps video that is already
every 6th frame (~83% of frames skipped). The plan describes the current implementation as if
it is missing.

### #6 — Cache the Audio Array
Already done. `backend/agents/voice_service.py` loads audio once and passes the `(y, sr)` tuple
to both the transcriber and the feature extractor. No duplicate disk reads.

### #5 — Skip Behavioural Rules When Not Asked
Fully wired. `run_behavioural`, `run_sentiment`, `run_entity_extraction` flags all exist in
`backend/pipeline/analysis_pipeline.py` and gate the correct agents.

---

## Suggestions That Will Break If Implemented As Written

### #1 — Parallelize Transcription, Diarization, and Feature Extraction ⚠️

The plan proposes:

```python
transcript, diarization, features = await asyncio.gather(
    transcribe_via_assemblyai(audio_path),
    asyncio.to_thread(diarizer.diarize, audio),
    asyncio.to_thread(feature_extractor.extract_all, audio),
)
```

**Three problems:**

1. **Transcription and diarization are a single call, not two separate steps.**
   AssemblyAI, Deepgram, and the GPU `/transcribe-diarize` endpoint return both transcript
   and speaker labels in one response. There is no standalone `diarizer.diarize()` to
   parallelize. The only backend where they are separate is External Whisper + local diarize,
   and that path already parallelizes them via `ThreadPoolExecutor` inside `transcriber.py`.

2. **Feature extraction depends on the transcript output.**
   `VoiceFeatureExtractor.extract_all()` requires `segments` — the speaker-labelled transcript
   segments — to group audio windows per speaker. Without the transcript the call signature is
   incomplete and per-speaker windowing cannot run.

3. **Implementing this as written will crash** with a missing-argument error or produce features
   with no speaker assignment.

### #2 — Wrap Individual Librosa Calls in `asyncio.to_thread` ⚠️

Wrapping `librosa.pyin`, `librosa.feature.rms` etc. individually is counterproductive.
They run inside `VoiceFeatureExtractor._extract_speaker_features()` which is already dispatched
through a `ThreadPoolExecutor` (per-speaker, max 4 workers) in `extract_all()`. Adding
`asyncio.to_thread` inside a thread-pool thread nests executors without any benefit and adds
scheduling overhead.

**Correct fix:** wrap the top-level `voice_service.analyse()` call in `asyncio.to_thread` so the
entire voice stage is offloaded — see the missing opportunity below.

---

## Suggestions That Are Valid and Safe

### #0 — Profile First
Safe. Per-stage timings are already partially logged (`Step 1 done in Xs`). Adding a single
summary log line at the end of `analysis_pipeline.run()` is low-risk and should be done before
any other change.

### #4 — InsightFace `det_size` (640,640) → (320,320)
Valid, but the impact is smaller than implied. ArcFace (`FaceEmbeddingExtractor`) runs **once
per session** on best-frame crops after the full frame loop completes — it is not called
per-frame. The per-frame bottleneck is MediaPipe (`TiledFrameProcessor`), not InsightFace.
This is a safe change but will not visibly move the headline processing time.

**File:** `backend/services/video_agent/feature_extractor.py` — `FaceEmbeddingExtractor.__init__`

### #7 — Pre-Warm MediaPipe at Startup
Valid gap. The video service starts with `"MediaPipe lazy-loads on first request"` (confirmed in
startup logs). The first session pays a ~30s model-load cost for `TiledFrameProcessor.create()`
which loads 6 MediaPipe models. Adding a warmup call during `VideoAgentService.startup()` that
builds a processor on a 1-frame dummy eliminates this first-session tax.

**File:** `backend/agents/video_service.py` — `startup()` method

### #8 — ONNX DistilBERT
Valid. `optimum[onnxruntime]` is already in `backend/requirements.txt`. The language service
already uses `asyncio.gather` across sentiment/intent/entities so switching to
`ORTModelForSequenceClassification` will not break the async structure. Impact is modest since
sentiment inference is already batched at 32 segments per call.

**File:** `backend/agents/language_service.py` — model loading in `startup()`

---

## The Highest-Value Opportunity Missing From the Plan

### Voice Service Blocks the Entire Event Loop

`voice_service.analyse()` is a synchronous blocking call. It waits 60–90 seconds for the
AssemblyAI response, then runs librosa feature extraction for another 4–5 minutes — all on the
asyncio event loop thread. While one session is in the voice stage, no other sessions can start,
health checks slow, and Redis polling stalls.

**Fix — wrap the voice analysis in the thread pool:**

```python
# backend/agents/voice_service.py

async def analyse(self, request: VoiceAnalysisRequest) -> VoiceAnalysisResponse:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,                        # use default thread pool
        self._analyse_sync, request  # move all blocking logic here
    )

def _analyse_sync(self, request: VoiceAnalysisRequest) -> VoiceAnalysisResponse:
    # existing body of analyse() moved here verbatim — no other changes needed
    ...
```

This is a single refactor with no logic changes. The video service already follows this exact
pattern (`run_in_executor` in `video_service.py`). Applying it to voice gives the same benefit:
the event loop stays free while the 5–10 minute voice stage runs in a thread.

---

## Summary Table

| # | Suggestion | Verdict | Notes |
|---|---|---|---|
| 0 | Profile first | ✅ Safe, do it | Low-risk, high information value |
| 1 | Parallelize transcribe / diarize / features | ❌ Breaks | Wrong dependency model — features need transcript output |
| 2 | `asyncio.to_thread` on each librosa call | ❌ Breaks | Wrong level — nests executors inside ThreadPoolExecutor |
| 3 | Lower MediaPipe frame rate | ✅ Already done | 5fps target already live |
| 4 | InsightFace `det_size` 320 | ✅ Safe | Small gain — ArcFace runs once per session, not per-frame |
| 5 | Skip behavioural when not asked | ✅ Already done | All flags fully wired |
| 6 | Cache audio array | ✅ Already done | Single load already passed to both transcriber and extractor |
| 7 | Pre-warm MediaPipe at startup | ✅ Safe, valid gap | First session currently pays ~30s model-load tax |
| 8 | ONNX DistilBERT | ✅ Safe, modest gain | Already batched; async structure unchanged |
| — | Wrap `voice.analyse()` in `run_in_executor` | 🔴 Missing from plan | Highest ROI — event loop currently blocked for 5–10 min per session |

---

## Recommended Implementation Order

1. **Profile** — add stage timing summary to `analysis_pipeline.run()` (5 min)
2. **Wrap voice in run_in_executor** — unblocks event loop, enables concurrent sessions (1–2 hr)
3. **Pre-warm MediaPipe** — eliminates first-session delay (30 min)
4. **InsightFace det_size 320** — safe one-liner (1 min)
5. **ONNX DistilBERT** — only if profiling shows language stage is a bottleneck (2 hr)
