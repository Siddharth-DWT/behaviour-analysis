# CLAUDE.md — NEXUS Project Intelligence File

> **Read this file first.** It contains everything you need to understand, develop, and extend the NEXUS system.

## What Is NEXUS?

NEXUS is a **multi-agent real-time behavioural analysis system** for video calls, meetings, and any audio/video media. It analyses human behaviour across 6 domains simultaneously — Voice, Language, Facial Expression, Body Language, Gaze, and Conversation Dynamics — then **fuses the signals across domains** to detect patterns that no single modality can reveal.

The core insight: **incongruence is the signal**. When someone's words say "yes" but their body says "no", when their face is calm but their voice is stressed — that gap is the most diagnostic information in human communication.

## Architecture Overview

NEXUS runs **7 independent agents** as microservices communicating via Redis Streams:

```
┌─────────┐  ┌──────────┐  ┌─────────┐
│  VOICE  │  │ LANGUAGE │  │ FACIAL  │
│ Agent 1 │  │ Agent 2  │  │ Agent 3 │
└────┬────┘  └────┬─────┘  └────┬────┘
     │            │             │
     └────────────┼─────────────┘
                  │
          ┌───────┴────────┐
          │  REDIS STREAMS  │  ← Message Bus
          └───────┬────────┘
                  │
     ┌────────────┼─────────────┐
     │            │             │
┌────┴────┐  ┌───┴───┐  ┌─────┴─────┐
│  BODY   │  │ GAZE  │  │   CONVO   │
│ Agent 4 │  │ Agent 5│  │  Agent 6  │
└────┬────┘  └───┬───┘  └─────┬─────┘
     │           │            │
     └───────────┼────────────┘
                 │
          ┌──────┴───────┐
          │   FUSION     │  ← The Orchestrator (Agent 7)
          │  15 pairwise │     Cross-modal intelligence
          │  12 compound │
          │  8 temporal  │
          └──────┬───────┘
                 │
          ┌──────┴───────┐
          │  DASHBOARD   │  ← React Admin UI
          └──────────────┘
```

## Current State (What's Built)

### ✅ COMPLETED
- **Rule Engine Documentation**: 94 detection rules across 8 documents (2,158 paragraphs)
  - Voice Agent: 18 rules (tone, stress, fillers, pitch, pace, pauses, interruptions, talk time)
  - Language Agent: 12 rules (sentiment, persuasion, buying signals, objections, power language)
  - Facial Agent: 7 rules (FACS AUs, emotions, Duchenne smile, micro-expressions, engagement)
  - Body Agent: 8 rules (posture, head movement, leaning, gestures, fidgeting, self-touch, mirroring)
  - Gaze Agent: 7 rules (direction, screen engagement, blink rate, attention, distraction, synchrony)
  - Conversation Agent: 7 rules (turn-taking, latency, rapport, conflict, dominance, engagement)
  - Fusion Agent: 15 pairwise cross-modal rules + unified output schema
  - Compound Patterns: 12 multi-domain states + 8 temporal cascade sequences
- **Feasibility Analysis**: Complete feasibility report with gap assessment
- **Research Compendium**: 120+ studies across all 6 domains
- **Infrastructure**: Docker Compose with PostgreSQL+pgvector + Valkey (Redis)
- **Database Schema**: 15 tables with all indexes, including rule_config, fusion_weights, signals, alerts, vector storage
- **Shared Libraries**: Signal models, message bus (Redis Streams wrapper), configuration
- **Voice Agent v0.1**: Complete implementation with 5 core rules
  - Feature extractor (25+ acoustic features via librosa)
  - Calibration module (per-speaker baselines)
  - Rule engine (stress, fillers, pitch, rate, tone)
  - Transcriber (faster-whisper + simple diarization)

- **Authentication**: JWT-based auth with bcrypt password hashing
  - Signup/Login/Logout/Refresh token endpoints
  - Role-based access control (admin, member, viewer)
  - Session ownership (users only see their own sessions, admin sees all)
  - Protected React dashboard with Login/Signup pages
  - Auto-refresh tokens, auth state in memory (XSS safe)

- **Video Agent (Phase 2)**: Complete implementation — MediaPipe + ArcFace + 3 rule engines
  - `feature_extractor.py` — frame extraction, CentroidTracker, ArcFace identity merge
  - `calibration.py` — per-speaker facial/body/gaze baselines (45-window target)
  - `facial_rules.py` / `body_rules.py` / `gaze_rules.py` — 20+ signal types
  - `SpeakerFaceMapper` — lip-sync correlation to assign face tracks → Speaker_N labels
  - Cross-session identity via pgvector (ArcFace 512-dim + voice 256-dim embeddings)
  - Async job pipeline: signals_ready in ~10min, overlay burn continues in background

### 🔲 NOT YET BUILT
- Recall.ai live call integration (Phase 4)
- Neo4j knowledge graph (Phase 3+)
- OAuth 2.0 / SSO integration (Phase 4)
- Per-tenant data isolation / row-level security (Phase 4)

## Tech Stack

| Layer | Technology | Status |
|-------|-----------|--------|
| Database | PostgreSQL 16 + pgvector | ✅ Running in Docker |
| Message Bus | Valkey 8 (Redis fork, BSD license) | ✅ Running in Docker |
| Voice Agent | Python FastAPI + librosa + faster-whisper | ✅ Built |
| Language Agent | Python FastAPI + DistilBERT + Claude API | ✅ Built |
| Fusion Agent | Python FastAPI + Claude API | ✅ Built |
| API Gateway | Python FastAPI + WebSocket | ✅ Built |
| Authentication | JWT (python-jose) + bcrypt | ✅ Built |
| Dashboard | React + Tailwind + Recharts | ✅ Built |
| External Whisper | GPU Whisper API (RTX 5090, optional) | ✅ Integrated |
| External TTS | Coqui XTTS v2 API (RTX 5090, optional) | ✅ Integrated |
| Video Agent | Python + MediaPipe + InsightFace ArcFace | ✅ Built |
| Speaker Registry | pgvector cosine similarity (face+voice fused) | ✅ Built |
| Knowledge Graph | Neo4j Community | 🔲 Phase 3 |
| Live Calls | Recall.ai API | 🔲 Phase 4 |

## How to Run

```bash
# Start all services (infrastructure + agents)
bash scripts/start_all.sh

# Stop all services
bash scripts/stop_all.sh          # agents only
bash scripts/stop_all.sh --all    # agents + docker

# Run end-to-end pipeline test
python3 scripts/test_pipeline.py
python3 scripts/test_pipeline.py --use-external-tts  # GPU TTS for better audio
python3 scripts/test_pipeline.py --skip-audio         # reuse existing audio

# Manual: start individual agents
uvicorn main:app --port 8001 --app-dir services/voice-agent
uvicorn main:app --port 8002 --app-dir services/language-agent
uvicorn main:app --port 8007 --app-dir services/fusion-agent
uvicorn main:app --port 8000 --app-dir services/api-gateway
```

### External GPU APIs (Optional)

NEXUS can optionally use GPU-accelerated Whisper STT, Coqui TTS, and pyannote
diarization services running on a remote server with NVIDIA RTX 5090. Set these env vars to enable:

```bash
export EXTERNAL_WHISPER_URL=http://110.227.200.12:8008   # GPU Whisper STT
export EXTERNAL_TTS_URL=http://110.227.200.12:8009       # Coqui XTTS v2
export EXTERNAL_DIARIZE_URL=http://110.227.200.12:8008   # GPU pyannote diarization
export EXTERNAL_API_KEY=your-api-key-here
export EXTERNAL_WHISPER_MODEL=base                        # or large-v3 for best accuracy
```

- **Whisper STT**: Voice Agent auto-detects and uses the GPU API for transcription
  (falls back to local faster-whisper if unreachable)
- **GPU Diarization**: Voice Agent uses the GPU pyannote API at `/diarize` for speaker
  separation. When both Whisper and diarization are on GPU, they run in parallel via
  ThreadPoolExecutor. Falls back to local KMeans/pyannote if unreachable.
- **Coqui TTS**: Used by `test_pipeline.py --use-external-tts` for natural-sounding
  test audio with voice cloning (distinct speakers)
- **WebSocket STT**: Available at `ws://server:8008/ws/transcribe` for Phase 4 real-time mode

## Key Design Principles

1. **Baseline-relative detection**: Every signal is a DEVIATION from the per-speaker baseline, not an absolute value. A naturally fast speaker at 170 WPM is not flagged — they're flagged when THEY deviate from THEIR normal.

2. **Cluster rule**: No single signal produces high confidence. Pease (2004), Navarro (2008): reliable interpretation requires 3+ congruent signals from different domains.

3. **Configurable thresholds**: All detection thresholds live in the `rule_config` database table, not in code. The feedback loop adjusts them over time.

4. **Graceful degradation**: Audio-only mode uses Voice+Language+Conversation. Video-only uses Facial+Body+Gaze. Domain weights redistribute automatically.

5. **Never claim certainty**: Maximum confidence for any signal is 0.85. Maximum for deception-related signals is 0.55 (deliberately capped). NEXUS produces probabilistic indicators, never binary determinations.

## File Map

```
nexus/
├── CLAUDE.md                    ← YOU ARE HERE
├── .env.example                 ← Environment variable template
├── docs/
│   ├── PLAN.md                  ← Full development plan with phases
│   ├── ARCHITECTURE.md          ← Detailed architecture & data flow
│   ├── RULES.md                 ← Rule engine summary & cross-references
│   ├── UI.md                    ← Dashboard design specification
│   └── rule-engine-docs/        ← The 8 DOCX rule engine documents
├── docker-compose.yml
├── infrastructure/
├── services/
│   ├── voice_agent/             ← ✅ Built (local + external Whisper backends)
│   ├── language_agent/          ← ✅ Built
│   ├── fusion_agent/            ← ✅ Built
│   ├── video_agent/             ← ✅ Built (Phase 2)
│   │   ├── feature_extractor.py ← MediaPipe + CentroidTracker + ArcFace merge
│   │   ├── calibration.py       ← Per-speaker facial/body/gaze baselines
│   │   ├── facial_rules.py      ← Emotion, smile, stress, engagement rules
│   │   ├── body_rules.py        ← Posture, head movement, gesture, touch rules
│   │   ├── gaze_rules.py        ← Screen contact, blink, direction rules
│   │   ├── base_rule_engine.py  ← Shared _make_signal factory (confidence cap 0.85)
│   │   └── main.py              ← FastAPI async job pipeline
│   └── api_gateway/             ← ✅ Built
│       ├── auth.py              ← JWT + bcrypt auth module
│       ├── speaker_registry.py  ← Cross-session face+voice identity matching (pgvector)
│       └── database.py          ← Async PostgreSQL (sessions, signals, users, registry)
├── dashboard/                   ← ✅ React dashboard (with Login/Signup)
├── shared/
│   ├── config/settings.py       ← Central config (DB, Redis, external APIs)
│   ├── models/signals.py
│   └── utils/
│       ├── message_bus.py       ← Redis Streams wrapper
│       └── external_apis.py     ← WhisperClient + TTSClient for GPU server
├── scripts/
│   ├── start_all.sh             ← Start infrastructure + all agents
│   ├── stop_all.sh              ← Stop all agents (+ --all for docker)
│   ├── test_pipeline.py         ← End-to-end pipeline test
│   └── health_check.py
└── data/
    ├── recordings/
    ├── labels/
    └── reports/
```

## Video Agent — Identity Tracking Architecture

Understanding this is critical before touching `feature_extractor.py`.

### Face Track Lifecycle
```
Video frames → MediaPipe detect faces → CentroidTracker assigns track IDs
    → _extract_frames() aggregates per-window features (WindowFeatures)
    → ArcFace _merge_tracks_by_embedding() deduplicates IDs (same person = one ID)
    → Position fallback for tracks where ArcFace extraction failed
    → Transient filter: drop tracks < 10 frames after all merging
    → SpeakerFaceMapper.assign() → lip-sync correlation → Speaker_N or Face_N labels
    → Rule engines (facial/body/gaze) → signals stored under speaker label
```

### Speaker Labels
- **Speaker_N** — face matched to a voice-diarized speaker via lip-sync correlation
- **Face_N** — detected face with no voice match (non-speaking participant, silent listener)
- Both are registered in `speakers_registry` (pgvector) for cross-session identity

### CentroidTracker Settings (feature_extractor.py)
- `max_disappeared = 90` frames (~18s at 5fps) — raised from 15 to survive head turns and detection gaps
- `match_threshold = 0.12` normalised centroid distance — raised from 0.08 for detection jitter tolerance
- **Do NOT lower these** — previous value of 15 caused ID inflation (75+ IDs for a 7-person call)

### ArcFace Identity Merge
`_compute_merge_threshold(duration_s, avg_face_height_ratio)` computes the cosine similarity
threshold used to decide if two face tracks belong to the same person.

```
avg_face_height_ratio = mean(sqrt(face_box_area)) across all detected frames (video-wide average)
  > 0.20 → grid/video-call mode  (large faces filling tiles)
  ≤ 0.20 → conference room mode  (small faces, many angles)
```

**Grid mode**: base=0.50–0.65 (by face size), decay=0.012/min, floor=0.42
**Conference room mode**: base=0.50, decay=0.008/min, floor=0.40

⚠️ **Known Problem — Tiny Face False Merges (UNRESOLVED as of 2026-05-12)**:
The threshold formula uses a video-wide average face size, not per-track size. When a participant
occupies a small grid tile (face_area ≈ 0.001, ~73×73px in 1080p), ArcFace produces noisy
embeddings. Those noisy embeddings generate false cosine similarity scores of 0.40–0.49 against
completely different people's faces — and with floor=0.40, all of them pass the merge check.

**Symptoms**: A small-tile speaker has signals scattered across 4–5 different face positions
(e.g. cx=0.63/0.69/0.34 instead of consistently cx=0.91). Wrong-face merges also delay first
signal appearance because calibration fires on the merged wrong face first.

**Root cause confirmed** on HR session (session `26a50159`): Speaker_2 (Sid, face_area=0.001)
had ~70–80 of 258 signals attributed to wrong faces at positions (0.34,0.96), (0.63,0.65),
(0.69,0.86). All wrong merges had similarity 0.40–0.49.

**Correct fix (not yet applied)**: Per-track face-size-aware threshold inside
`_merge_tracks_by_embedding()`. When `min(face_h_A, face_h_B) < 0.07` (face_area < 0.005),
use `effective_threshold = max(global_threshold, 0.60)`. Requires passing per-track avg face
heights alongside `frame_counts`. Original hardcoded threshold=0.70 (from 2026-05-06 commit)
never had this problem because 0.70 > all false-match scores.

**Do NOT** simply raise the global floor — same-person tiny-face scores can also be 0.50–0.59,
so a flat floor=0.60 would prevent legitimate fragment merges for small faces.

### Signal Pipeline Filters (full chain)
1. **`MIN_FACE_RATE = 0.30`** — rule engines skip a window if face detected < 30% of its frames
2. **`CALIBRATION_MIN_WINDOWS = 5`** — no signals until 5 windows accumulated; full conf at 45
3. **`MIN_SIGNAL_CONFIDENCE`** — facial=0.18, body=0.20; signals below dropped before storage
4. **`MIN_FRAMES_AFTER_MERGE = 10`** — transient tracks (< 2s) pruned after merge
5. **Gateway 120s filter** — `(end_ms - start_ms) > 120,000` signals excluded from `/video-signals`
6. **Gateway `_VIDEO_OVERLAY_TYPES`** — only 60+ whitelisted signal types returned to frontend
7. **Gateway `MIN_FACE_SIGNALS = 10`** — Face_* speakers with < 10 total signals dropped at persist
8. **Frontend `isWeakTrack`** — Face_N hidden if < 30 session signals OR no thumbnail OR ≤ 1 unique signal type
9. **Frontend `FORWARD_GRACE_MS = 250`** — signal stays visible 250ms past its `end_ms`

### Calibration Timing
`CALIBRATION_TARGET_WINDOWS = 45` (90 seconds). First signals for any speaker appear only after
5+ valid windows (10s minimum). For tiny-face participants, detection gaps mean calibration takes
longer — Sid (face_area=0.001) first appeared at 22,000ms vs Face_3 at 16,000ms vs Speaker_1 at 0ms.

---

## Known Issues & Solutions History

This section records recurring problems so future sessions do not re-investigate solved issues.

### 1. ArcFace Merge: Tiny Faces Merged With Wrong People
**File**: `services/video_agent/feature_extractor.py` — `_compute_merge_threshold()`, `_merge_tracks_by_embedding()`
**First seen**: HR session 2026-05-11 (6-person grid call, session `26a50159`)
**Symptom**: Small-tile participant (Speaker_2, Sid) had signals attributed to 4–5 different
face positions instead of consistently cx≈0.91. ~70/258 signals came from wrong faces.
**Root cause**: `avg_face_h` is a video-wide average. Conference room floor=0.40 accepted
false similarity scores (0.40–0.49) from noisy tiny-face ArcFace embeddings.
**Status**: UNRESOLVED — fix designed (per-track size-aware threshold, min 0.60 when face_h < 0.07) but not implemented.
**Do NOT**: Lower the threshold (makes more wrong merges). Do NOT raise global floor to 0.60+ (breaks legitimate tiny-face same-person merges at 0.50–0.59).

### 2. CentroidTracker ID Inflation (SOLVED 2026-05-06)
**File**: `services/video_agent/feature_extractor.py`
**Symptom**: 7-person call generated 75+ unique Face_* IDs. Same person got new ID every 3s.
**Root cause**: `max_disappeared=15` too short. Any detection gap > 3s expired the track.
**Fix**: Raised `max_disappeared` 15→90 (~18s), `match_threshold` 0.08→0.12.
**Do NOT revert** — the old value caused catastrophic ID fragmentation.

### 3. ArcFace Merge Initially Hardcoded at 0.70 (CHANGED — understand before editing)
**File**: `services/video_agent/feature_extractor.py`
**History**: The original merge threshold was flat 0.70 (added 2026-05-06). This was safe but
too strict for conference room sessions (same-person pairs at extreme angles score 0.40–0.65).
Adaptive formula replaced it. Adaptive formula works for normal faces but has the tiny-face
problem described above. Do not blindly revert to flat 0.70 — legitimate same-person conference
room merges will be missed.

### 4. Gateway Position Merge Replaced by ArcFace Merge (SOLVED 2026-05-06)
**File**: `services/api_gateway/main.py`
**Symptom**: Layout changes (screen share, grid rearrange) caused the same person to appear
as multiple speakers because position-based merge (distance < 0.10) relied on fixed tile locations.
**Fix**: Replaced with embedding cosine merge at threshold=0.70 as safety net after video agent merge.

### 5. Video Agent Timeout Killing Signals (SOLVED 2026-04-28)
**File**: `services/api_gateway/main.py`
**Symptom**: Sessions longer than 30 min returned 0 video signals.
**Root cause**: `AGENT_TIMEOUT=1800s`. Video pipeline takes 40–90 min for typical meetings.
**Fix**: Added `VIDEO_AGENT_TIMEOUT=10800` (3 hours). Voice/language/fusion still use 30-min timeout.

### 6. Video Agent Blocking Fusion (SOLVED 2026-04-28)
**File**: `services/api_gateway/main.py`
**Symptom**: Fusion and report blocked for 47+ min waiting for ffmpeg overlay burn.
**Fix**: Made video fire-and-forget. `run_analysis()` returns signals after ~10 min.
`burn_overlay()` runs as Phase 2 background task. Gateway polls for `signals_ready` status.

### 7. Non-Speaking Faces Dropped from Registry (SOLVED 2026-05-05)
**File**: `services/api_gateway/main.py`
**Symptom**: Face_N participants (silent listeners) were unregistered → showed as Unknown in UI.
**Fix**: After fused+voice registry matching, a third block iterates remaining Face_* not in
`speaker_identity_map` and calls `match_or_create_by_face_only()` for each.

### 8. Session-Spanning Temporal Signals Active at Every Playback Position (SOLVED 2026-05-08)
**File**: `services/api_gateway/main.py`
**Symptom**: body_mirroring, adaptation_pattern showed active at every video timestamp.
**Root cause**: Temporal rule engines emitted signals spanning the full session duration.
**Fix**: Gateway `/video-signals` filters out signals where `(end_ms - start_ms) > 120,000`.

### 9. "No Active Signals" After Page Open Before Pipeline Completes
**Not a bug** — if the page is opened while video pipeline is still running, `/video-signals`
returns empty array which the frontend caches. Refreshing the page after pipeline completes
(session status = "completed") returns signals correctly.

### 10. Face_N vs Speaker_N — Why Ansuya Shows as "Face_3 (face match 100%)"
**Face_N** means the lip-sync mapper could not correlate face movements to any diarized voice
speaker. She speaks but her lip movements didn't correlate above the threshold. This is a known
limitation of lip-sync matching with small-face tiles (face_area < 0.02). The "face match 100%"
in the UI means registry match confidence, not lip-sync confidence.

---

## When Writing Code for NEXUS

- **Always use the Signal model** from `shared/models/signals.py` for agent outputs
- **Always publish to Redis Streams** via `shared/utils/message_bus.py`
- **Always load thresholds** from `rule_config` table (fallback to hardcoded defaults)
- **Every agent is a standalone FastAPI service** with its own Dockerfile
- **Every rule traces to specific research** — when creating new rules, cite the study
- **Test against labeled ground truth data** — never deploy a rule without measuring accuracy
- **Use `calibration_confidence` as a multiplier** on all output confidence scores
- **Service folders use underscores** (`video_agent`, `api_gateway`) — hyphens break Python imports
- **Never modify Dockerfiles or requirements.txt** unless explicitly asked
- **Never start Docker containers** without explicit user instruction
- **Never run INSERT/UPDATE/DELETE/DROP** on any DB without explicit user instruction
- **Dashboard + video agent containers have internal source copies** — after code changes, must
  `docker cp` the file into the container and rebuild; just restarting does not pick up changes
- **Check container logs before grepping internal source files** — confirms what code is actually running
- **Always check all containers in one parallel call** — never check logs sequentially

## Video Agent — Critical Rules

- **`_compute_merge_threshold` and `_merge_tracks_by_embedding`** are the most sensitive functions
  in the codebase. Changes here directly affect how many unique people are identified and whether
  signals get attributed to the right person. Read the "Known Issues" section above before touching.
- **`avg_face_h` is a video-wide average** (line ~2543 in feature_extractor.py) — not per-track.
  Any logic that needs per-track face size must compute it separately from `frames`.
- **The adaptive threshold formula has been intentionally reverted** multiple times. The current
  values (conference room: base=0.50, decay=0.008/min, floor=0.40) are the reverted state.
  The pending unresolved fix is per-track size-awareness, not global floor changes.
- **`CAP_PROP_POS_MSEC`** is used for accurate video PTS timestamps — do not replace with
  `frame_idx / fps * 1000` which drifts on variable-frame-rate videos.
- **`_make_signal()` in `base_rule_engine.py`** never returns None — it always returns a dict.
  All rule engine callers depend on this. Do not add None returns or optional guards.
- **Signal confidence is hard-capped at 0.85** globally in `base_rule_engine.py`. Deception-related
  signals are additionally capped at 0.55 in individual rule engines.
