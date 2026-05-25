# NEXUS: Microservices → Monolithic Backend Migration Plan

## Context

NEXUS currently runs as 6 separate Docker containers (5 analysis agents + 1 API gateway), each a
FastAPI process communicating via Redis Streams job dispatch (`AgentJobDispatcher` → `nexus:jobs:{agent}`
stream → `RedisJobConsumer` loop → pub/sub result notification). The goal is to collapse all 6 Python
services into **one monolithic FastAPI backend process** while keeping the React dashboard completely
untouched — it continues calling the same REST API on port 8000.

**What changes:** One Python process instead of six. Direct in-process function calls instead of Redis
job dispatch. One Dockerfile instead of six. No HTTP timeouts between agents.

**What stays the same:** All rule engines, feature extractors, calibration modules, database schema,
Redis (for signals/state/locks), all REST API endpoints, all dashboard code, all shared models.

---

## New Directory Structure

```
behaviour-analysis/
├── backend/                         ← NEW: the single monolithic application
│   ├── main.py                      ← FastAPI app factory + lifespan
│   ├── Dockerfile                   ← single container image
│   ├── requirements.txt             ← merged deps from all 6 services
│   ├── dependencies.py              ← FastAPI Depends() factories
│   │
│   ├── agents/                      ← 5 in-process service wrappers (OOP)
│   │   ├── base.py                  ← BaseAgentService ABC + TranscriptionBackend Protocol
│   │   ├── voice_service.py         ← VoiceAgentService
│   │   ├── language_service.py      ← LanguageAgentService
│   │   ├── conversation_service.py  ← ConversationAgentService
│   │   ├── video_service.py         ← VideoAgentService (thread pool + fire-and-forget)
│   │   └── fusion_service.py        ← FusionAgentService
│   │
│   ├── pipeline/
│   │   └── analysis_pipeline.py     ← AnalysisPipeline (orchestrator)
│   │
│   ├── api/                         ← Route modules (split from 4200-line gateway main.py)
│   │   ├── auth.py                  ← /auth/* (15 endpoints)
│   │   ├── sessions.py              ← /sessions/*, /quick-transcribe
│   │   ├── uploads.py               ← /uploads/* chunked protocol
│   │   ├── speakers.py              ← /speakers/*
│   │   ├── team.py                  ← /team, /team/compare
│   │   └── chat.py                  ← /chat/global, /sessions/{id}/chat
│   │
│   └── core/                        ← Moved from services/api_gateway/ verbatim
│       ├── database.py
│       ├── auth.py
│       ├── speaker_registry.py
│       ├── email_service.py
│       ├── knowledge_store.py
│       ├── neo4j_schema.py
│       ├── neo4j_sync.py
│       └── neo4j_semantic_layer.py
│
├── dashboard/                       ← UNCHANGED (separate frontend)
├── shared/                          ← UNCHANGED (models, redis_layer, config, utils)
├── services/                        ← Kept during migration; deleted after cutover
└── docker-compose.yml               ← Simplified: 3 infra + backend + dashboard
```

---

## Key Classes to Create

### `backend/agents/base.py`

```python
class TranscriptionBackend(Protocol):
    """Strategy interface — swappable transcription providers (Whisper/AssemblyAI/Deepgram).
    Existing Transcriber class already implements this internally via env var switching."""
    def transcribe(self, file_path: str, **kwargs) -> dict: ...

class BaseAgentService(ABC):
    """
    Contract for all 5 in-process agent services.
    - startup()  → load models/allocate thread pool (called once at app start)
    - shutdown() → release resources (called at app stop)
    - name       → string identifier used in Redis keys and logging
    Composition: rule engines are constructor parameters, not superclasses.
    """
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def startup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...
```

### `backend/agents/voice_service.py` (representative pattern)

```python
class VoiceAgentService(BaseAgentService):
    name = "voice"

    def __init__(self) -> None:
        self._extractor: Optional[VoiceFeatureExtractor] = None
        self._transcriber: Optional[Transcriber] = None
        self._rule_engine: Optional[VoiceRuleEngine] = None

    async def startup(self) -> None:
        self._extractor  = VoiceFeatureExtractor()
        self._transcriber = Transcriber()          # reads TRANSCRIPTION_BACKEND env var
        self._rule_engine = VoiceRuleEngine()
        self._transcriber.warm_up()               # loads Whisper into memory

    async def shutdown(self) -> None: pass

    async def analyse(self, request: VoiceAnalysisRequest) -> VoiceAnalysisResponse:
        """Direct Python call — body = services/voiceAgent/main.py::analyse_audio() de-FastAPI-ified."""
        ...

    async def transcribe_only(self, request) -> dict:
        """Used by /quick-transcribe."""
        ...
```

Same pattern for Language, Conversation, Fusion. Video differs:

```python
class VideoAgentService(BaseAgentService):
    name = "video"

    async def startup(self) -> None:
        # Thread pool for CPU-bound frame processing (unchanged from existing video_agent)
        self._thread_pool = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() or 4, 8),
            thread_name_prefix="nexus-video",
        )
        # MediaPipe models lazy-loaded on first call (avoids 3s startup delay)

    async def analyse(self, session_id, video_path, diar_segments, meeting_type, num_speakers) -> dict:
        loop = asyncio.get_running_loop()
        pipeline = self._get_pipeline()   # lazy init on first call
        result, _ = await loop.run_in_executor(
            self._thread_pool, pipeline.run_analysis, video_path, session_id, diar_segments, meeting_type
        )
        # Overlay burn remains fire-and-forget — background task, not blocking
        asyncio.create_task(self._burn_overlay(session_id, video_path, diar_segments))
        return result.model_dump()
```

### `backend/pipeline/analysis_pipeline.py`

```python
class AnalysisPipeline:
    """
    Orchestrates 5 agent services for one analysis session.
    Stateless with respect to sessions — holds only model references loaded at startup.

    Execution graph (mirrors existing _run_pipeline in gateway/main.py):
        Voice                                          (sequential — transcript required by all)
        Language + Video                               (asyncio.gather — parallel)
        Conversation                                   (sequential — needs language signals)
        Fusion                                         (sequential — needs all prior signals)
        Neo4j + knowledge store sync                   (sequential, non-fatal)

    DSA: asyncio.gather(Language, Video) reduces total time from
         T_lang + T_video → max(T_lang, T_video), saving 2–8 min per session.
    """

    def __init__(self, voice, language, conversation, video, fusion, redis_repo): ...

    async def run(self, session_id, file_path, video_path, meeting_type, ...) -> None:
        """Background task. All exceptions caught/logged — never propagated."""
        ...

    async def run_quick_transcribe(self, file_path, session_id, config) -> dict:
        """Lightweight path for /quick-transcribe — no DB, no pipeline."""
        return await self._voice.transcribe_only(...)

    @property
    def services(self) -> list[BaseAgentService]:
        return [self._voice, self._language, self._conversation, self._video, self._fusion]
```

### `backend/main.py`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — sequential (Whisper + DistilBERT parallel init causes PyTorch GIL issues)
    services = [VoiceAgentService(), LanguageAgentService(), ConversationAgentService(),
                VideoAgentService(), FusionAgentService()]
    for svc in services:
        await svc.startup()

    redis_repo = RedisRepository()
    db_pool    = await get_pool()
    pipeline   = AnalysisPipeline(*services, redis_repo)
    await init_neo4j_schema()

    app.state.pipeline   = pipeline
    app.state.db_pool    = db_pool
    app.state.redis_repo = redis_repo

    yield

    for svc in services:
        await svc.shutdown()
    await close_pool()

app = FastAPI(title="NEXUS Backend", version="2.0.0", lifespan=lifespan)
app.include_router(auth.router,     prefix="/auth")
app.include_router(sessions.router)
app.include_router(uploads.router,  prefix="/uploads")
app.include_router(speakers.router, prefix="/speakers")
app.include_router(team.router,     prefix="/team")
app.include_router(chat.router,     prefix="/chat")
```

---

## What Changes vs What Stays

### REMOVED
| What | Why |
|---|---|
| `shared/redis_layer/job_consumer.py` (usage) | No more `nexus:jobs:{agent}` streams internally |
| `shared/redis_layer/job_dispatcher.py` (usage) | Direct `await service.analyse()` replaces pub/sub dispatch |
| Per-agent `startup()` / `shutdown()` FastAPI lifecycle hooks | Merged into single lifespan |
| `VOICE_AGENT_URL`, `LANGUAGE_AGENT_URL`, etc. env vars | No inter-service HTTP |
| `AGENT_TIMEOUT`, `VIDEO_AGENT_TIMEOUT` env vars | No HTTP timeouts needed |
| 5 separate Dockerfiles | Single `backend/Dockerfile` |
| Consumer groups on `nexus:jobs:*` streams | Not written; not consumed |

### RETAINED (unchanged code)
| What |
|---|
| All rule engines + feature extractors (voice, language, conversation, fusion, video) |
| `shared/models/signals.py`, `shared/config/settings.py`, `shared/utils/` |
| `RedisRepository`, `RedisKeys`, `RedisLockManager`, `RedisEventStore`, `RedisClientFactory` |
| All Redis key patterns except `nexus:jobs:{agent}` and `nexus:result-ready:*` |
| `services/api_gateway/database.py`, `auth.py`, `speaker_registry.py` (copied, not modified) |
| All 40+ REST API endpoint paths (dashboard requires no changes) |
| `SyncRedisRepository` used inside `VideoAgentService`'s thread pool |
| `dashboard/` — completely untouched |
| `docker-compose.yml` infra services (postgres, redis, neo4j) |

---

## docker-compose.yml Changes

**Remove** these 5 services: `voice`, `language`, `conversation`, `video`, `fusion`, `api`

**Add** one service:
```yaml
backend:
  build: { context: ./backend }
  container_name: nexus-backend
  ports: ["8000:8000"]
  environment:
    - PYTHONPATH=/app
    - DATABASE_URL=postgresql://nexus:${POSTGRES_PASSWORD}@postgres:5432/nexus
    - REDIS_URL=redis://redis:6379/0
    - NEO4J_URI=bolt://neo4j:7687
    # All env vars from all 6 old services combined here
    - TRANSCRIPTION_BACKEND=${TRANSCRIPTION_BACKEND:-auto}
    - LLM_PROVIDER=${LLM_PROVIDER:-openai}
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    - JWT_SECRET=${JWT_SECRET}
    - MEDIAPIPE_MODEL_DIR=/app/models/mediapipe
    - OVERLAY_DIR=/app/data/overlays
    # ... all others
  volumes:
    - ./backend:/app
    - ./shared:/app/shared
    - ./data:/app/data
  depends_on:
    postgres: { condition: service_healthy }
    redis:    { condition: service_healthy }
    neo4j:    { condition: service_healthy }
```

---

## Ordered Migration Steps

### Phase 0 — Scaffold (no behaviour change)

1. **Create `backend/` directory tree** — empty `__init__.py` files, stub `pass` implementations for all 5 agent services with `NotImplementedError` in `analyse()`. The app starts cleanly.
2. **Merge `requirements.txt`** — combine pip deps from all 6 services into `backend/requirements.txt`. Resolve version conflicts (numpy, torch, mediapipe versions must be compatible).
3. **Copy gateway modules** into `backend/core/` — flat copy of `database.py`, `auth.py`, `speaker_registry.py`, `email_service.py`, `neo4j_*.py`, `knowledge_store.py`. Adjust imports: `from database import` → `from core.database import`.
4. **Write `backend/main.py`** with lifespan, middleware, and all routers registered. Verify `uvicorn backend.main:app --port 8000` starts against running infra.
5. **Write `backend/dependencies.py`** — `get_pipeline()`, `get_db_pool()`, `get_redis_repo()` from `request.app.state`.

### Phase 1 — Port Read Endpoints (no pipeline)

6. **`api/auth.py`** — lift all `/auth/*` handlers from gateway `main.py` (lines ~466–894). Replace module-level `db_pool` with `Depends(get_db_pool)`. All auth flows functional.
7. **`api/sessions.py` (reads only)** — port `GET /sessions`, `GET /sessions/{id}`, `/signals`, `/transcript`, `/report`, `/progress`, `/video`, `/video/annotated`, `/video-signals`, `/video-speakers`. Session list and detail views in dashboard work immediately.
8. **`api/uploads.py`** — port chunked upload. `complete` calls `pipeline.run()` stub (session stays in `processing` state until Phase 3).
9. **`api/speakers.py`**, **`api/team.py`**, **`api/chat.py`** — pure DB queries, port verbatim.

### Phase 2 — Implement Agent Services (ascending complexity)

10. **`ConversationAgentService`** — no heavy models; fastest to implement. Import `ConversationFeatureExtractor`, `ConversationRuleEngine` from `services/conversation_agent/`. Body of `analyse()` = `services/conversation_agent/main.py::analyse()` with FastAPI wrapper removed. Unit test with fixture transcripts.
11. **`LanguageAgentService`** — import from `services/language_agent/`. `startup()` warms up DistilBERT. Unit test with fixture segments.
12. **`FusionAgentService`** — import `FusionRuleEngine`, `CompoundPatternEngine`, `TemporalPatternEngine`, `SignalGraph`, `GraphAnalytics`, `generate_session_narrative` from `services/fusion_agent/`. No GPU/heavy models — purely algorithmic. Unit test with fixture signal sets.
13. **`VoiceAgentService`** — import from `services/voiceAgent/`. `Transcriber` class already implements Strategy pattern via `TRANSCRIPTION_BACKEND` env var — no refactoring needed. Integration test: process a real `.wav` end-to-end.
14. **`VideoAgentService`** — import `VideoPipeline` and all rule engines from `services/video_agent/`. Thread pool in `startup()`. Lazy model init on first call. `_burn_overlay()` writes to same Redis key `nexus:artifact:{id}:display_names` as before. Integration test: process a real `.mp4`.

### Phase 3 — Wire Pipeline

15. **`AnalysisPipeline.run()`** — lift `_run_pipeline()` from `services/api_gateway/main.py`. Replace all `await _call_{agent}_agent(...)` with `await self._{agent}.analyse(...)`. Replace `from database import` with `from core.database import`. Keep `asyncio.gather(language, video)` parallel execution intact.
16. **Wire into `api/sessions.py`** — `POST /sessions` and `POST /uploads/{id}/complete` call `background_tasks.add_task(pipeline.run, ...)`. Inject pipeline via `Depends(get_pipeline)`.
17. **`run_quick_transcribe`** — `POST /quick-transcribe` calls `pipeline.run_quick_transcribe()` → `voice_service.transcribe_only()`.

### Phase 4 — Cutover

18. **Full integration test** — run `scripts/test_pipeline.py` against the new backend. Process a `.wav` + `.mp4`. Verify: session completes, signal counts match reference run, report generated, dashboard renders correctly.
19. **Update `docker-compose.yml`** — remove 5 agent definitions, rename `api:` → `backend:`, merge all env vars.
20. **Hard cutover** — `docker compose down voice language conversation video fusion api && docker compose up -d backend`. Dashboard at port 3006 continues without any change.
21. **Cleanup (deferred, 1 week after stable prod)** — delete `services/voiceAgent/`, `services/language_agent/`, `services/conversation_agent/`, `services/video_agent/`, `services/fusion_agent/`, `services/api_gateway/`. Remove `RedisJobConsumer` and `AgentJobDispatcher` from `shared/redis_layer/__init__.py` exports.

---

## Critical Files to Modify

| File | Change |
|---|---|
| `backend/main.py` | **CREATE** — app factory, lifespan, router registration |
| `backend/agents/base.py` | **CREATE** — `BaseAgentService` ABC, `TranscriptionBackend` Protocol |
| `backend/agents/voice_service.py` | **CREATE** — wraps `services/voiceAgent/` rule engines |
| `backend/agents/language_service.py` | **CREATE** — wraps `services/language_agent/` rule engines |
| `backend/agents/conversation_service.py` | **CREATE** — wraps `services/conversation_agent/` |
| `backend/agents/video_service.py` | **CREATE** — wraps `services/video_agent/` + thread pool |
| `backend/agents/fusion_service.py` | **CREATE** — wraps `services/fusion_agent/` rule engines |
| `backend/pipeline/analysis_pipeline.py` | **CREATE** — lifts `_run_pipeline()` from gateway |
| `backend/api/auth.py` | **CREATE** — auth routes from gateway `main.py:466–894` |
| `backend/api/sessions.py` | **CREATE** — session routes from gateway `main.py:901–2765` |
| `backend/api/uploads.py` | **CREATE** — upload routes from gateway `main.py:986–1202` |
| `backend/api/speakers.py` | **CREATE** — speaker routes from gateway `main.py:3580–3840` |
| `backend/api/team.py` | **CREATE** — team routes from gateway `main.py:3840–3976` |
| `backend/api/chat.py` | **CREATE** — chat routes from gateway `main.py:3400–3570` |
| `backend/core/` (8 files) | **COPY** from `services/api_gateway/` with import path adjustments |
| `backend/requirements.txt` | **CREATE** — merged deps from all 6 services |
| `backend/Dockerfile` | **CREATE** — single container image |
| `docker-compose.yml` | **MODIFY** — remove 6 service defs, add 1 `backend:` service |
| `shared/redis_layer/__init__.py` | **MODIFY (Step 21)** — remove job_consumer/dispatcher from `__all__` |

**Not modified:** `dashboard/`, `shared/models/`, `shared/redis_layer/` internals, `shared/config/`, `shared/utils/`, database schema, all rule engines inside `services/*/`

---

## Verification

1. `uvicorn backend.main:app --port 8000` starts with no import errors
2. `GET /health` returns `{"status": "ok", "agents": {"voice": "ok", "language": "ok", ...}}`
3. `POST /auth/signup` + `POST /auth/login` returns JWT tokens
4. `POST /sessions` with a `.wav` file completes with `status=completed` and `signal_count > 0`
5. `GET /sessions/{id}/report` returns a narrative report
6. `GET /sessions/{id}/video-signals` returns signals for `.mp4` sessions
7. Dashboard at `http://localhost:3006` loads, login works, session list shows, video player shows overlay signals
8. No dashboard code changes required — same port 8000, same API paths, same response shapes
