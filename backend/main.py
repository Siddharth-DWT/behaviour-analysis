# backend/main.py
"""
NEXUS Backend — FastAPI application factory.

Single-process monolith: 5 in-process agent services replace the previous
microservice fleet. Direct Python calls replace Redis job dispatch.

Lifespan:
  Startup  — sequential service init (Whisper + DistilBERT CUDA init is not
             safe to parallelize due to PyTorch GIL; one service at a time).
  Shutdown — graceful resource release in reverse startup order.

API surface is identical to the old api_gateway — all REST paths preserved
so the React dashboard requires zero changes.

PYTHONPATH: the Dockerfile sets PYTHONPATH=/app (the backend/ dir) so
imports like `from agents.voice_service` and `from core.database` resolve
correctly. When developing locally, set PYTHONPATH=./backend:. before
running uvicorn.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nexus.backend")

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start all 5 agent services sequentially, connect DB + Redis, build pipeline.

    Sequential init is intentional: parallel Whisper + DistilBERT CUDA warm-up
    triggers a PyTorch GIL deadlock. Each service takes 2–8s at startup.
    Total cold-start budget: ~20s (acceptable; container health check has 60s).
    """
    from agents.voice_service import VoiceAgentService
    from agents.language_service import LanguageAgentService
    from agents.conversation_service import ConversationAgentService
    from agents.video_service import VideoAgentService
    from agents.fusion_service import FusionAgentService
    from pipeline.analysis_pipeline import AnalysisPipeline
    from core.database import get_pool, close_pool
    from core.neo4j_schema import init_neo4j_schema
    from shared.redis_layer import RedisRepository

    services = [
        VoiceAgentService(),
        LanguageAgentService(),
        ConversationAgentService(),
        VideoAgentService(),
        FusionAgentService(),
    ]

    for svc in services:
        logger.info("Starting service: %s", svc.name)
        await svc.startup()

    db_pool    = await get_pool()
    redis_repo = RedisRepository()
    pipeline   = AnalysisPipeline(
        voice=services[0],
        language=services[1],
        conversation=services[2],
        video=services[3],
        fusion=services[4],
        redis_repo=redis_repo,
    )

    # Neo4j schema init is non-fatal — runs IF neo4j package is present.
    try:
        await init_neo4j_schema()
    except Exception as exc:
        logger.warning("Neo4j schema init skipped: %s", exc)

    app.state.pipeline   = pipeline
    app.state.db_pool    = db_pool
    app.state.redis_repo = redis_repo

    logger.info("NEXUS backend ready — all services started.")
    yield

    # ── Shutdown ──
    for svc in reversed(services):
        try:
            await svc.shutdown()
        except Exception as exc:
            logger.warning("Service shutdown error (%s): %s", svc.name, exc)

    await close_pool()
    logger.info("NEXUS backend shut down cleanly.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="NEXUS Backend",
        description="Multi-Agent Behavioural Analysis — monolithic backend",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS — same origins as the old gateway
    allowed_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3006,http://localhost:3000,http://localhost:5173",
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────────
    # Import here (inside factory) to keep module-level imports clean and to
    # allow the factory to be called after PYTHONPATH is fully configured.
    from api.auth import router as auth_router
    from api.sessions import router as sessions_router
    from api.uploads import router as uploads_router
    from api.speakers import router as speakers_router
    from api.team import router as team_router
    from api.chat import router as chat_router

    app.include_router(auth_router,     prefix="/auth")
    app.include_router(sessions_router)
    app.include_router(uploads_router,  prefix="/uploads")
    app.include_router(speakers_router, prefix="/speakers")
    app.include_router(team_router,     prefix="/team")
    app.include_router(chat_router,     prefix="/chat")

    # ── Health check ─────────────────────────────────────────────────────────
    @app.get("/health", tags=["health"])
    async def health():
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is None:
            return {"status": "starting"}
        agents = {svc.name: "ok" for svc in pipeline.services}
        return {"status": "ok", "agents": agents}

    return app


app = create_app()
