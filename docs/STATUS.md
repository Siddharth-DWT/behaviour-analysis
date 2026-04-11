# STATUS.md — NEXUS Build Status Tracker

**Last updated**: April 10, 2026
**Current phase**: Phase 1 done · Phase 3A in progress (Neo4j single-session knowledge graph + hybrid RAG chat)

---

## Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| Docker Compose | ✅ Done | PostgreSQL + Valkey + Neo4j orchestration |
| PostgreSQL 16 + pgvector | ✅ Done | 15 tables incl. `knowledge_chunks`, `chat_messages`, auto-created on first start |
| Valkey 8 (Redis) | ✅ Done | Streams-optimised config |
| **Neo4j 5.26 (community)** | ✅ Done | Single-session knowledge graph; APOC + GDS plugins enabled |
| Database schema | ✅ Done | sessions, signals, alerts, rule_config, fusion_weights, knowledge_chunks, chat_messages, session_reports, etc. |
| Rule config seeded | ✅ Done | All rule thresholds in rule_config table |
| Shared models | ✅ Done | Signal, SpeakerBaseline, Alert, CompoundPattern, UnifiedSpeakerState |
| Message bus | ✅ Done | Redis Streams wrapper (publish, subscribe, get_latest) |
| Config module | ✅ Done | Central settings from environment variables |
| LLM client (tri-provider) | ✅ Done | OpenAI + Anthropic + **Ollama** via `shared/utils/llm_client.py` (all three with optional auth headers) |
| Media ingestion | ✅ Done | Universal audio/video/URL input via `shared/utils/media_ingest.py` |
| Content classifier | ✅ Done | Auto-detect content type via `shared/utils/content_classifier.py` |
| **ContentTypeProfile** | ✅ Done | `shared/config/content_type_profile.py` — gating, threshold, confidence multiplier, signal renames per content type. **PROFILES dict still sparsely populated** (~10 of 42 rules have explicit entries). |
| **Multi-backend transcription** | ✅ Done | Auto-cascade: Whisper+NeMo combined → Parakeet TDT → AssemblyAI → Deepgram → Whisper-only → local faster-whisper |
| **GPU diarization cascade** | ✅ Done | External GPU `/diarize` → pyannote community-1 → pyannote 3.1 → KMeans → gap+pitch → round-robin |
| Health check script | ✅ Done | Verifies PostgreSQL, pgvector, Redis, env vars |
| .env.example | ✅ Done | All env templates incl. Ollama, Neo4j |
| .gitignore | ✅ Done | Python + Node + Docker + media files |

---

## Recent Changes (April 10, 2026)

### Phase 3A — Single-Session Knowledge Graph (Neo4j)
- **NEW**: `services/api_gateway/neo4j_sync.py` — projects each completed session into Neo4j after the analysis pipeline finishes. PostgreSQL stays the source of truth; Neo4j is a re-buildable projection.
  - Node types: `Session`, `Speaker`, `Segment`, `Topic`, `Signal`, `FusionInsight`, `Entity:Person/Company/Product/Objection/Commitment`, `Alert`
  - Edges built: `PARTICIPATED_IN`, `PART_OF`, `OCCURRED_IN`, `SPOKEN_BY`, `EMITTED_BY`, `NEXT`, `FOLLOWED_BY`, `PRECEDED`, `OCCURRED_DURING`, `DISCUSSES`, `MENTIONED_IN`, `IS_SPEAKER`, `RAISED_BY`, `RESOLVED_IN`, `RAISED_FOR`
  - Causal edges (post-load Cypher): `CONTRADICTS`, `REINFORCES`, `TRIGGERED`
  - Cross-speaker: `INFLUENCED {lag_ms}` within 30 s
  - Sync is non-fatal: pipeline still completes if Neo4j is unreachable; idempotent (`DETACH DELETE` per session before re-write)
- **UPDATED**: `/sessions/:id/chat` is now hybrid — pgvector text similarity **+** Neo4j graph search via LLM-generated Cypher (read-only, mutating queries refused). Either side can fail gracefully; the answer is built from whatever returned.
- **NEW**: docker-compose `neo4j` service (`neo4j:5.26-community`, ports 7474/7687, APOC + GDS plugins, healthcheck), `NEO4J_URI/USER/PASSWORD` env vars, `neo4j_data` + `neo4j_logs` volumes
- **NEW**: `neo4j>=5.20` added to api_gateway requirements
- **NEW**: `services/api_gateway/neo4j_sync.search_graph_context()` — schema-aware Cypher generation with strict guardrails (forbids CREATE/MERGE/DELETE/SET, scopes every query to `$session_id`, caps at 15 rows)

### Ollama LLM Provider Support
- **NEW**: `LLM_PROVIDER=ollama` option in `shared/utils/llm_client.py` — routes `complete()`, `acomplete()`, and `get_embedding()` to a self-hosted or remote Ollama server
- **NEW**: `OLLAMA_API_KEY` env var + `Authorization: Bearer` header support for gated Ollama proxies (e.g. `https://llm.vidoshare.com`)
- **UPDATED**: `OLLAMA_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL` forwarded to `language`, `fusion`, and `api` services in docker-compose (previously only the `api` service)
- **FIX**: Without these env-passthroughs, Language and Fusion agents were silently defaulting to OpenAI and hitting `429 insufficient_quota` despite the global Ollama setting

### Pipeline & Data-Quality Fixes
- **FIX**: AssemblyAI client wiring — collapsed three duplicate `_init_assemblyai` definitions in `services/voiceAgent/transcriber.py` (only the last one ran in Python). The kept definition uses `shared/utils/assemblyai_client.py` (standalone, supports `speakers_expected` hint, has poll timeout)
- **FIX**: AssemblyAI standalone client — added `MAX_POLL_WAIT = 1200` cap in `_poll()` to prevent infinite hang on stuck transcripts
- **FIX**: Parakeet + NeMo `/diarize` cascade — `_diarize_external_gpu` was reading `result["segments"]` but the GPU API returns `result["timeline"]`, so 22 valid speaker turns were always discarded as "0 segments" and the cascade fell through to local pyannote-community-1. Now reads `timeline` first, falls back to `segments` for older builds.
- **FIX**: Removed dead duplicate `_init_deepgram_diarize` definitions in `transcriber.py` (the method was never called from `__init__` and only set a flag that's now confirmed always-False; cleaned up dead code path)
- **FIX**: `entity_extractor.py` lightweight regex fallback now sets `speaker_label` on extracted people, so the dashboard's `speakerNames` map works even when the LLM is unreachable
- **FIX**: `entity_extractor.py` LLM prompt now asks for `resolved_at_ms` per objection so Neo4j can build `(Objection)-[:RESOLVED_IN]->(Segment)` edges
- **FIX**: `database.insert_signals` whitelist now accepts `speaker_id="session"` (Conversation Agent session-level signals) and `"X__Y"` pair labels (rapport/latency between speaker pairs) without warning — these intentionally store with `speaker_id=NULL`
- **FIX**: `/sessions/:id/chat` pgvector query wrapped in try/except — degrades to empty rows on cross-provider embedding-dim mismatch (e.g. OpenAI 1536 → Ollama 768) instead of returning HTTP 500

---

## Recent Changes (March 24, 2026)

### Authentication System (`services/api_gateway/auth.py`)
- **NEW**: User authentication with JWT access tokens (30 min) + refresh tokens (30 days)
- **NEW**: bcrypt password hashing (direct `bcrypt` library, not passlib)
- **NEW**: Auth endpoints: `POST /auth/signup`, `/auth/login`, `/auth/refresh`, `/auth/logout`
- **NEW**: Profile endpoints: `GET /auth/me`, `PUT /auth/me`, `PUT /auth/change-password`
- **NEW**: Role-based access control (admin > member > viewer)
- **NEW**: Session ownership — users only see their own sessions, admin sees all
- **NEW**: All session endpoints now require authentication (Bearer token)
- **NEW**: `GET /health` remains public (no auth required)

### Database Migration (`infrastructure/postgres/init/02-auth.sql`)
- **NEW**: Auth columns on `users` table: `password_hash`, `full_name`, `company`, `avatar_url`, `is_active`, `is_verified`, `last_login_at`
- **NEW**: `auth_tokens` table for refresh token storage (single-use, server-side)
- **NEW**: `user_id` foreign key on `sessions` table for ownership

### Dashboard Authentication
- **NEW**: `AuthContext` — React context for auth state (login, signup, logout, auto-refresh)
- **NEW**: Login page (`/login`) — email/password with show/hide toggle
- **NEW**: Signup page (`/signup`) — with password strength indicator (weak/fair/strong)
- **NEW**: Route protection — `ProtectedRoute` redirects to `/login` if not authenticated
- **NEW**: User menu in navbar — initials avatar + dropdown (Profile, Settings, Sign Out)
- **UPDATED**: API client adds `Authorization: Bearer` header to all requests
- **UPDATED**: 401 interceptor auto-refreshes token, retries request, redirects to login on failure
- **UPDATED**: Access token stored in memory (not localStorage) for XSS protection
- **UPDATED**: Refresh token in localStorage for session persistence across page refreshes

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
| Rule engine (14 rules) | ✅ Done | `services/voice-agent/rules.py` |
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
| Rule engine (12 rules, content-adaptive) | ✅ Done | `services/language-agent/rules.py` |
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

### Conversation Agent (Agent 6) — ✅ COMPLETE (v0.1)

| Component | Status | File |
|-----------|--------|------|
| FastAPI application | ✅ Done | `services/conversation_agent/main.py` |
| Feature extractor (per-speaker, per-pair, session) | ✅ Done | `services/conversation_agent/feature_extractor.py` |
| Rule engine (7 rules) | ✅ Done | `services/conversation_agent/rules.py` |
| CONVO-TURN-01: Turn-taking pattern | ✅ Done | Turn rate classification |
| CONVO-LAT-01: Response latency | ✅ Done | Per-pair latency analysis |
| CONVO-DOM-01: Dominance mapping | ✅ Done | Per-speaker dominance score |
| CONVO-INT-01: Interruption pattern | ✅ Done | Rate-based detection |
| CONVO-RAP-01: Rapport indicator | ✅ Done | 4-component weighted score |
| CONVO-ENG-01: Conversation engagement | ✅ Done | 5-component weighted score |
| CONVO-BAL-01: Conversation balance | ✅ Done | Gini-based dominance index |
| Dockerfile | ✅ Done | `services/conversation_agent/Dockerfile` |
| Requirements | ✅ Done | No ML deps, pure computation |
| Dashboard: Conversation Dynamics panel | ✅ Done | `InsightPanel.tsx` |
| Dashboard: Speaker dominance/engagement chips | ✅ Done | `SessionDetail.tsx` |
| Dashboard: Rapport on speaker graph | ✅ Done | `SpeakerGraph.tsx` |
### Facial Agent (Agent 3) — 🔲 Phase 2
### Body Agent (Agent 4) — 🔲 Phase 2
### Gaze Agent (Agent 5) — 🔲 Phase 2

---

## API Gateway — ✅ COMPLETE (v0.3)

| Component | Status | File |
|-----------|--------|------|
| FastAPI application | ✅ Done | `services/api_gateway/main.py` |
| Database module (asyncpg) | ✅ Done | `services/api_gateway/database.py` |
| Auth module (JWT + bcrypt) | ✅ Done | `services/api_gateway/auth.py` |
| Email service (ZeptoMail) | ✅ Done | `services/api_gateway/email_service.py` |
| **Knowledge store (pgvector RAG)** | ✅ Done | `services/api_gateway/knowledge_store.py` — chunks transcript/signals/entities, embeds, persists to `knowledge_chunks` |
| **Neo4j sync (Phase 3A)** | ✅ Done | `services/api_gateway/neo4j_sync.py` — single-session graph projection + hybrid Cypher chat search |
| POST /auth/signup, /login, /refresh, /logout | ✅ Done | Full JWT lifecycle |
| POST /auth/verify-email, /resend-verification | ✅ Done | Email verification flow |
| GET/PUT /auth/me, /change-password | ✅ Done | Profile management |
| POST /sessions (upload + full pipeline) | ✅ Done | Async background pipeline; emits status updates |
| GET /sessions, /:id, /:id/signals, /:id/report, /:id/transcript | ✅ Done | All auth-gated, ownership-scoped |
| **POST /sessions/:id/chat** | ✅ Done | Hybrid RAG: pgvector + Neo4j graph search |
| **GET /sessions/:id/chat** | ✅ Done | Chat history retrieval |
| GET /health | ✅ Done | Public, probes downstream agents |

---

## Dashboard — ✅ COMPLETE (v0.3)

**Stack**: React 18 + TypeScript + Vite + Tailwind CSS + Recharts + React Query + React Router

| Page / Component | Status | File |
|---|---|---|
| SessionList | ✅ Done | `pages/SessionList.tsx` |
| SessionDetail | ✅ Done | `pages/SessionDetail.tsx` (with content-type-aware speaker name/role inference from `entities.people`) |
| ReportView | ✅ Done | `pages/ReportView.tsx` |
| Login / Signup / VerifyEmail | ✅ Done | `pages/Login.tsx`, `Signup.tsx`, `VerifyEmail.tsx` |
| SignalChainCards, SignalNetwork | ✅ Done | Per-signal explorers |
| SwimlaneTimeline | ✅ Done | Multi-track signal timeline |
| ConversationGraph, SpeakerGraph | ✅ Done | Per-pair rapport / dominance graphs |
| StressTimeline, TopicTimeline | ✅ Done | Time-series visualisations |
| GraphInsightsCard | ✅ Done | Tension clusters / momentum / persistent incongruence |
| **SessionChat** | ✅ Done | RAG chat panel inside Session Detail |
| `/speakers/:id`, `/insights`, `/graph` | 🔲 Phase 3B/3D | Cross-session pages — not started |

---

## What To Build Next (Priority Order)

### Immediate (Phase 3A — in progress)
1. ~~Add Neo4j to docker-compose~~ ✅
2. ~~`neo4j_sync.py` — single-session projection~~ ✅
3. ~~Causal + cross-speaker influence edges~~ ✅
4. ~~Hybrid pgvector + Neo4j chat~~ ✅
5. **Validate on a real session**: upload recording, inspect Neo4j browser at `:7474`, run all 8 prompt.md Cypher use-cases manually
6. Populate `ContentTypeProfile.PROFILES` for all 42 Phase 1 rules × 5 content types (currently ~10 entries; rest fall through to defaults)
7. Test with real recordings (sales calls, podcasts, interviews) and tune `rule_config` thresholds

### Phase 3B Prerequisites (next 1–2 weeks)
8. Add `voice_embedding VECTOR(256)` column to `speakers` table
9. Extract per-speaker pyannote/embedding during diarization (model already loaded for community-1 fallback)
10. `Session.outcome` column (won/lost/unknown) + dashboard tagger — Phase 3C precondition
11. Add `COMPOSED_OF` / `COMBINES` provenance to fusion + conversation composite signals (rapport components, etc.) — unlocks prompt.md use-cases #6 and #7

### Phase 3B–3D (Weeks 19–24)
12. Cross-session speaker identity (cosine match on voice embeddings, manual labelling UI)
13. Entity resolver for cross-session companies/people/topics (`apoc.text.fuzzyMatch`)
14. `/speakers/:id` Speaker Profile page (behavioural DNA radar, trajectory)
15. `/insights` Organization Insights page (team heatmap, topic sensitivity, won-vs-lost)
16. `/graph` Knowledge Graph Explorer (neovis.js)

### Phase 2 (visual agents — deferred)
17. Body Agent first (BODY-HEAD-01 head nod/shake — highest universal value)
18. Facial Agent (FACE-EMO-01, FACE-SMILE-01, FACE-STRESS-01)
19. Gaze Agent (GAZE-DIR-01, GAZE-CONTACT-01, GAZE-BLINK-01)
20. Remaining 12 Fusion pairwise rules (need video signals)
21. WebSocket `/ws/sessions/:id` for live session updates (precondition for Phase 4)

---

## Known Issues & Risks

| Issue | Severity | Mitigation |
|-------|----------|-----------|
| `ContentTypeProfile.PROFILES` dict only ~10 of 42 rules populated | High | Mechanical data-entry from `docs/OPTIMIZATION_PLAN.md` tables; the wiring is in place, just unpopulated |
| No real-data validation of any rule yet | High | Priority: label 3+ recordings, run pipeline, compare outputs to ground truth, tune thresholds |
| `signal_graph.py` and `neo4j_sync.py` derive causal edges with **slightly different** sentiment/stress thresholds | Medium | Cross-check both code paths after first Neo4j validation run |
| `knowledge_chunks` rows may have mixed embedding dims (OpenAI 1536 + Ollama 768) — old sessions break chat | Low | Chat now degrades gracefully (catches dim mismatch). For full re-index, drop and re-run knowledge_store on existing sessions |
| Conversation pair / session signals (`Speaker_0__Speaker_1`, `"session"`) store with `speaker_id=NULL` | Low | Whitelisted to silence the warning. Pair identity preserved in `metadata` JSON, not as a queryable column |
| `STATUS.md` / `PLAN.md` previously said Phase 1 Week 6+ but reality is Phase 3A in progress | ~~High~~ | ✅ Refreshed April 10, 2026 |
| Simple diarisation inaccurate for 3+ speakers | ~~Medium~~ | ✅ External GPU `/diarize` cascade now primary; community-1 + KMeans fallbacks |
| yt-dlp YouTube download blocked by bot detection | Medium | Use --cookies-from-browser or download manually |
| ffmpeg not installed on dev machine | Low | afconvert (macOS) works for audio; ffmpeg needed for video |
| Whisper "medium" model slow on CPU | Low | Use external GPU Whisper instead (now primary) |
| ~~No authentication~~ | ~~Low~~ | ✅ JWT auth implemented (March 24, 2026) |
| ~~Language/Fusion agents hit OpenAI 429 despite LLM_PROVIDER=ollama~~ | ~~High~~ | ✅ Fixed April 10 — env vars now forwarded to all three services |
| ~~Parakeet+NeMo `/diarize` always returns 0 segments~~ | ~~High~~ | ✅ Fixed April 10 — was reading wrong key (`segments` vs `timeline`) |
