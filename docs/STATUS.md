# STATUS.md — NEXUS Build Status Tracker

**Last updated**: March 15, 2026
**Current phase**: Phase 1 Week 6+ — Universal Media Input & Content-Adaptive Analysis

---

## Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| Docker Compose | ✅ Done | PostgreSQL + Valkey orchestration |
| PostgreSQL 16 + pgvector | ✅ Done | 15 tables, auto-created on first start |
| Valkey 8 (Redis) | ✅ Done | Streams-optimised config |
| Database schema | ✅ Done | sessions, signals, alerts, rule_config, fusion_weights, etc. |
| Rule config seeded | ✅ Done | All rule thresholds in rule_config table |
| Shared models | ✅ Done | Signal, SpeakerBaseline, Alert, CompoundPattern, UnifiedSpeakerState |
| Message bus | ✅ Done | Redis Streams wrapper (publish, subscribe, get_latest) |
| Config module | ✅ Done | Central settings from environment variables |
| LLM client (dual provider) | ✅ Done | OpenAI + Anthropic via `shared/utils/llm_client.py` |
| Media ingestion | ✅ Done | Universal audio/video/URL input via `shared/utils/media_ingest.py` |
| Content classifier | ✅ Done | Auto-detect content type via `shared/utils/content_classifier.py` |
| Health check script | ✅ Done | Verifies PostgreSQL, pgvector, Redis, env vars |
| .env.example | ✅ Done | All environment variable templates |
| .gitignore | ✅ Done | Python + Node + Docker + media files |

---

## Recent Changes (March 15, 2026)

### Universal Media Input (`shared/utils/media_ingest.py`)
- **NEW**: Accept any audio format (.wav, .mp3, .flac, .ogg, .m4a, .aac, .wma, .opus)
- **NEW**: Accept video files (.mp4, .mkv, .webm, .mov, .avi, .flv) — auto-extract audio
- **NEW**: Accept URLs (YouTube, podcast RSS, direct audio/video URLs) via yt-dlp
- **NEW**: Auto-convert everything to 16kHz mono WAV for Voice Agent
- **NEW**: `prepare_audio()` (async) and `prepare_audio_sync()` — single entry point for all media

### Multi-Speaker Diarization (`services/voice-agent/transcriber.py`)
- **UPDATED**: Removed 2-speaker assumption in simple diarization
- **NEW**: Support 2-10 speakers with `--speakers` flag
- **NEW**: Multi-speaker heuristic using turn detection, gap analysis, and duration clustering
- Labels: Speaker_0, Speaker_1, Speaker_2, ... Speaker_9

### Content Type Auto-Detection (`shared/utils/content_classifier.py`)
- **NEW**: Auto-classifies content from first ~2 minutes of transcript via LLM
- **NEW**: 9 supported types: sales_call, podcast, interview, lecture, debate, meeting, presentation, casual_conversation, other
- **NEW**: Heuristic fallback when LLM unavailable (keyword matching + speaker count)

### Adaptive Analysis
- **UPDATED**: Language Agent buying signal and objection rules only run for `sales_call` content type
- **UPDATED**: Sentiment, power language, and intent classification run for all content types
- **NEW**: Content-type-specific report templates in Fusion Agent narrative generator
  - sales_call: buying signals, objections, close probability
  - podcast: speaker dynamics, topic flow, engagement peaks
  - interview: confidence trajectory, question quality, rapport
  - lecture: clarity, pacing, energy, audience engagement
  - debate: argument strength, dominance shifts, persuasion
  - meeting: participation balance, decisions, action items
  - presentation: clarity, persuasion, audience engagement
  - casual_conversation: rapport, engagement, dynamics

### Dual LLM Provider Support (`shared/utils/llm_client.py`)
- **UPDATED**: Supports both Anthropic Claude and OpenAI APIs
- **UPDATED**: Configurable via `LLM_PROVIDER` env var (openai or anthropic)
- Default models: gpt-4o-mini (OpenAI), claude-sonnet-4-20250514 (Anthropic)

### Updated Pipeline Script (`scripts/test_pipeline.py`)
- **NEW**: `--url` flag to accept YouTube/podcast/direct URLs
- **NEW**: `--type` flag to override auto-detected content type
- **NEW**: `--speakers` flag for speaker count hint (2-10)
- **UPDATED**: Auto-classifies content type and adapts analysis accordingly
- **UPDATED**: Works with any audio/video file, not just synthetic sales calls

---

## Documentation

| Document | Status | Location |
|----------|--------|----------|
| CLAUDE.md (master brain) | ✅ Done | `CLAUDE.md` |
| Development plan | ✅ Done | `docs/PLAN.md` |
| Architecture reference | ✅ Done | `docs/ARCHITECTURE.md` |
| Rule engine quick reference | ✅ Done | `docs/RULES.md` |
| Dashboard UI specification | ✅ Done | `docs/UI.md` |
| Getting started guide | ✅ Done | `docs/GETTING_STARTED.md` |
| Build status tracker | ✅ Done | `docs/STATUS.md` (this file) |
| Feasibility analysis | ✅ Done | Delivered as DOCX |
| Research compendium | ✅ Done | Delivered as DOCX (120+ studies) |
| Voice rule engine (18 rules) | ✅ Done | Delivered as DOCX |
| Language rule engine (12 rules) | ✅ Done | Delivered as DOCX |
| Facial rule engine (7 rules) | ✅ Done | Delivered as DOCX |
| Body rule engine (8 rules) | ✅ Done | Delivered as DOCX |
| Gaze rule engine (7 rules) | ✅ Done | Delivered as DOCX |
| Conversation rule engine (7 rules) | ✅ Done | Delivered as DOCX |
| Fusion rule engine (15 rules) | ✅ Done | Delivered as DOCX |
| Compound patterns (12+8) | ✅ Done | Delivered as DOCX |

---

## Agent Services

### Voice Agent (Agent 1) — ✅ COMPLETE

| Component | Status | File |
|-----------|--------|------|
| FastAPI application | ✅ Done | `services/voice-agent/main.py` |
| Feature extractor (25+ features) | ✅ Done | `services/voice-agent/feature_extractor.py` |
| Calibration module (VOICE-CAL-01) | ✅ Done | `services/voice-agent/calibration.py` |
| Rule engine (5 core rules) | ✅ Done | `services/voice-agent/rules.py` |
| Transcriber (Whisper + diarisation) | ✅ Done | `services/voice-agent/transcriber.py` |
| Multi-speaker diarization (2-10) | ✅ Done | Simple heuristic + pyannote support |
| Dual transcription backend | ✅ Done | Local CPU + External GPU API |
| Dockerfile | ✅ Done | `services/voice-agent/Dockerfile` |
| Requirements | ✅ Done | `services/voice-agent/requirements.txt` |

### Language Agent (Agent 2) — ✅ COMPLETE (v0.2)

| Component | Status | File |
|-----------|--------|------|
| FastAPI application | ✅ Done | `services/language-agent/main.py` |
| Feature extractor | ✅ Done | `services/language-agent/feature_extractor.py` |
| Rule engine (5 rules, content-adaptive) | ✅ Done | `services/language-agent/rules.py` |
| Content-type gating | ✅ Done | Buying/objection rules only for sales_call |
| Dual LLM support | ✅ Done | OpenAI + Anthropic for intent classification |
| Dockerfile | ✅ Done | `services/language-agent/Dockerfile` |
| Requirements | ✅ Done | `services/language-agent/requirements.txt` |

### Fusion Agent (Agent 7) — ✅ COMPLETE (v0.2)

| Component | Status | File |
|-----------|--------|------|
| FastAPI application | ✅ Done | `services/fusion-agent/main.py` |
| Fusion engine | ✅ Done | `services/fusion-agent/fusion_engine.py` |
| Pairwise rules (3 for Phase 1) | ✅ Done | `services/fusion-agent/rules.py` |
| Narrative report generator | ✅ Done | `services/fusion-agent/narrative.py` |
| Content-specific report templates | ✅ Done | 9 content types supported |
| Dual LLM support | ✅ Done | OpenAI + Anthropic for narrative generation |
| Dockerfile | ✅ Done | `services/fusion-agent/Dockerfile` |
| Requirements | ✅ Done | `services/fusion-agent/requirements.txt` |

### Shared Utilities

| Component | Status | File |
|-----------|--------|------|
| LLM client (dual provider) | ✅ Done | `shared/utils/llm_client.py` |
| Media ingestion (universal) | ✅ Done | `shared/utils/media_ingest.py` |
| Content classifier | ✅ Done | `shared/utils/content_classifier.py` |
| External APIs (Whisper + TTS) | ✅ Done | `shared/utils/external_apis.py` |
| Message bus (Redis Streams) | ✅ Done | `shared/utils/message_bus.py` |
| Signal models | ✅ Done | `shared/models/signals.py` |
| Config | ✅ Done | `shared/config/settings.py` |

### Conversation Agent (Agent 6) — 🔲 NOT STARTED
### Facial Agent (Agent 3) — 🔲 Phase 2
### Body Agent (Agent 4) — 🔲 Phase 2
### Gaze Agent (Agent 5) — 🔲 Phase 2

---

## API Gateway — ✅ COMPLETE (v0.1)

| Component | Status | File |
|-----------|--------|------|
| FastAPI application | ✅ Done | `services/api-gateway/main.py` |
| Database module (asyncpg) | ✅ Done | `services/api-gateway/database.py` |
| POST /sessions (upload + full pipeline) | ✅ Done | Upload → Voice → Language → Fusion → PostgreSQL |
| GET /sessions (list, paginated) | ✅ Done | Filterable by status, meeting_type |
| GET /sessions/:id (detail) | ✅ Done | Includes signal counts, alerts, report status |
| GET /sessions/:id/signals | ✅ Done | Filterable by agent, signal_type |
| GET /sessions/:id/report | ✅ Done | Returns existing or generates via Fusion Agent |
| GET /sessions/:id/transcript | ✅ Done | Ordered transcript segments |
| GET /health | ✅ Done | Probes all downstream agents + DB |

---

## Dashboard — ✅ PHASE 1 COMPLETE (v0.1)

**Stack**: React 18 + TypeScript + Vite + Tailwind CSS + Recharts + React Query + React Router

---

## What To Build Next (Priority Order)

### Immediate
1. ~~Build Language Agent~~ ✅
2. ~~Build Fusion Agent~~ ✅
3. ~~Build API Gateway~~ ✅
4. ~~Build React Dashboard v1~~ ✅
5. ~~Dual LLM provider support~~ ✅
6. ~~Universal media input~~ ✅
7. ~~Content-adaptive analysis~~ ✅
8. Test with real recordings (sales calls, podcasts, interviews)
9. Tune rule_config thresholds from real data

### Next Week
10. Build Conversation Agent
11. Add Speaker Cards to Session Detail
12. Integrate media ingestion into API Gateway (accept URLs via POST)

### Following Weeks
13. Implement remaining compound patterns
14. Add remaining fusion rules as video agents come online
15. WebSocket support for live session updates
16. Multi-track signal timeline (Phase 2 dashboard)

---

## Known Issues & Risks

| Issue | Severity | Mitigation |
|-------|----------|-----------|
| Simple diarisation inaccurate for 3+ speakers | Medium | Enable pyannote (USE_PYANNOTE=true + HF_TOKEN) |
| yt-dlp YouTube download blocked by bot detection | Medium | Use --cookies-from-browser or download manually |
| ffmpeg not installed on dev machine | Low | afconvert (macOS) works for audio; ffmpeg needed for video |
| Whisper "medium" model slow on CPU | Low | Use "small" for dev, "medium" for production |
| No real data testing yet | High | Priority: test with diverse content types |
| No authentication | Low (dev only) | Phase 4 |
