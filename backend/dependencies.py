# backend/dependencies.py
"""
FastAPI Depends() factories for the NEXUS monolith.

All shared state is stored on app.state by the lifespan context manager.
Route handlers inject these via Depends(get_pipeline) etc. — no module-level
globals, no circular imports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Request

if TYPE_CHECKING:
    from pipeline.analysis_pipeline import AnalysisPipeline


def get_pipeline(request: Request) -> "AnalysisPipeline":
    """Inject the AnalysisPipeline singleton from app.state."""
    return request.app.state.pipeline


def get_db_pool(request: Request):
    """Inject the asyncpg connection pool from app.state."""
    return request.app.state.db_pool


def get_redis_repo(request: Request):
    """Inject the RedisRepository from app.state."""
    return request.app.state.redis_repo
