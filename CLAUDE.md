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
- **Redis Job Dispatch Layer** (`shared/redis_layer/`): Full inter-service communication via Redis only
  - `RedisJobConsumer` ABC (Template Method pattern) — each agent subclasses with `process_job()`; base owns the full loop: xgroup_create → xreadgroup → deserialise → write_result → xack
  - `AgentJobDispatcher` — subscribes to `nexus:result-ready:{sid}:{agent}` before pushing the job (no race), wakes the caller via asyncio.Event the instant the result is written (O(1)), falls back to exponential-backoff polling (0.1s → 2s) if pub/sub fails
  - Zero HTTP calls between Gateway and Agents — all job dispatch routes through `nexus:jobs:{agent}` Redis Streams
  - Results stored at `nexus:session:{id}:result:{agent}` and notified via Redis pub/sub channel `nexus:result-ready:{id}:{agent}`
  - `VideoJobConsumer` overrides `_handle()` instead of `process_job()` — ACKs immediately, spawns the 10–90 min background task, result written internally by `_run_video_job()`
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
│   ├── redis_layer/             ← Redis coordination layer (single source of truth)
│   │   ├── client.py            ← RedisClientFactory (async + sync singleton pools)
│   │   ├── keys.py              ← Canonical key builders (all nexus:* key names)
│   │   ├── repository.py        ← RedisRepository (async) + SyncRedisRepository
│   │   ├── job_consumer.py      ← RedisJobConsumer ABC — Template Method pattern
│   │   ├── job_dispatcher.py    ← AgentJobDispatcher — pub/sub notify + backoff polling
│   │   ├── schemas.py           ← Pydantic models (SignalRecord, SessionStateRecord, …)
│   │   ├── events.py            ← RedisEventStore
│   │   └── locks.py             ← RedisLockManager
│   └── utils/
│       ├── message_bus.py       ← Redis Streams wrapper (legacy, kept for fallback drain)
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
  `docker cp` the file into the container and rebuild; just restarting does not pick up changes but only with permission
- **Check container logs before grepping internal source files** — confirms what code is actually running
- **Always check all containers in one parallel call** — never check logs sequentially
- **Apply OOP principles and DSA reasoning** — applies to all code across the system, not just one layer. Encapsulate shared behaviour in base classes rather than duplicating it. Choose data structures for their time-complexity properties (e.g. a set for O(1) membership tests, pub/sub for O(1) event notification vs O(T/interval) polling). Apply design patterns (Template Method, Strategy, Factory) where they eliminate copy-paste. Place shared logic in the appropriate `shared/` module rather than repeating it per-service. Example: `RedisJobConsumer` ABC in `shared/redis_layer/job_consumer.py` owns the full consumer loop; each agent subclasses with one method
- **Inter-service communication is Redis-only between gateway and agents** — all gateway→agent calls use `AgentJobDispatcher.dispatch()` via Redis Streams (`nexus:jobs:{agent}`). Never add direct HTTP calls between the gateway and agents. Dashboard→Gateway HTTPS is the only HTTP boundary in the system

## Video Agent — Critical Rules

- **`_compute_merge_threshold` and `_merge_tracks_by_embedding`** are the most sensitive functions
  in the codebase. Changes here directly affect how many unique people are identified and whether
  signals get attributed to the right person. Read the "Known Issues" section above before touching.
- **`avg_face_h` is a video-wide average** used only by `_compute_merge_threshold()` to select
  grid vs conference-room mode. `_merge_tracks_by_embedding()` uses per-track heights via the
  `track_face_heights` dict — the two are separate; do not confuse them.
- **The adaptive threshold formula has been intentionally reverted** multiple times. The current
  values (conference room: base=0.50, decay=0.008/min, floor=0.40) are the reverted state.
  The per-track tiny-face guard (floor 0.55/0.45 when face_h < 0.07) was then added on top.
  Do not remove the guard. Do not raise the global floor. The still-unresolved problem is
  medium-small faces (face_h ≥ 0.07) that use the global floor ≈0.40, which is too permissive.
- **`CAP_PROP_POS_MSEC`** is used for accurate video PTS timestamps — do not replace with
  `frame_idx / fps * 1000` which drifts on variable-frame-rate videos.
- **`_make_signal()` in `base_rule_engine.py`** never returns None — it always returns a dict.
  All rule engine callers depend on this. Do not add None returns or optional guards.
- **Signal confidence is hard-capped at 0.85** globally in `base_rule_engine.py`. Deception-related
  signals are additionally capped at 0.55 in individual rule engines.
