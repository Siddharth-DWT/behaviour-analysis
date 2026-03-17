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

### 🔲 NOT YET BUILT
- Facial/Body/Gaze agents (Phase 2)
- Recall.ai live call integration (Phase 4)
- Neo4j knowledge graph (Phase 3+)
- Authentication / multi-tenancy

## Tech Stack

| Layer | Technology | Status |
|-------|-----------|--------|
| Database | PostgreSQL 16 + pgvector | ✅ Running in Docker |
| Message Bus | Valkey 8 (Redis fork, BSD license) | ✅ Running in Docker |
| Voice Agent | Python FastAPI + librosa + faster-whisper | ✅ Built |
| Language Agent | Python FastAPI + DistilBERT + Claude API | ✅ Built |
| Fusion Agent | Python FastAPI + Claude API | ✅ Built |
| API Gateway | Python FastAPI + WebSocket | ✅ Built |
| Dashboard | React + Tailwind + Recharts | ✅ Built |
| External Whisper | GPU Whisper API (RTX 5090, optional) | ✅ Integrated |
| External TTS | Coqui XTTS v2 API (RTX 5090, optional) | ✅ Integrated |
| Facial Agent | Python + OpenFace/MediaPipe + DeepFace | 🔲 Phase 2 |
| Body Agent | Python + MediaPipe Holistic | 🔲 Phase 2 |
| Gaze Agent | Python + MediaPipe Face Mesh | 🔲 Phase 2 |
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

NEXUS can optionally use GPU-accelerated Whisper STT and Coqui TTS services
running on a remote server with NVIDIA RTX 5090. Set these env vars to enable:

```bash
export EXTERNAL_WHISPER_URL=http://110.227.200.12:8008   # GPU Whisper STT
export EXTERNAL_TTS_URL=http://110.227.200.12:8009       # Coqui XTTS v2
export EXTERNAL_API_KEY=your-api-key-here
export EXTERNAL_WHISPER_MODEL=base                        # or large-v3 for best accuracy
```

- **Whisper STT**: Voice Agent auto-detects and uses the GPU API for transcription
  (falls back to local faster-whisper if unreachable)
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
│   ├── voice-agent/             ← ✅ Built (local + external Whisper backends)
│   ├── language-agent/          ← ✅ Built
│   ├── fusion-agent/            ← ✅ Built
│   └── api-gateway/             ← ✅ Built
├── dashboard/                   ← ✅ React dashboard
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

## When Writing Code for NEXUS

- **Always use the Signal model** from `shared/models/signals.py` for agent outputs
- **Always publish to Redis Streams** via `shared/utils/message_bus.py`
- **Always load thresholds** from `rule_config` table (fallback to hardcoded defaults)
- **Every agent is a standalone FastAPI service** with its own Dockerfile
- **Every rule traces to specific research** — when creating new rules, cite the study
- **Test against labeled ground truth data** — never deploy a rule without measuring accuracy
- **Use `calibration_confidence` as a multiplier** on all output confidence scores
