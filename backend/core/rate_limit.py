# backend/core/rate_limit.py
"""
NEXUS Backend - API Key Rate Limiting

Redis fixed-window (1-minute bucket) counter. Reuses the already-wired async Redis
client at app.state.redis_repo.client (RedisClientFactory.get_async_client()).

O(1) per request: a single INCR + conditional EXPIRE. No DB write contention.
Fail-open if Redis is unavailable — an infra hiccup must not lock out all API
traffic (keys are still fully authenticated; we log a warning instead).
"""
import logging
import time

from fastapi import HTTPException, Request

logger = logging.getLogger("nexus.backend.ratelimit")


async def enforce_rate_limit(request: Request, key_id: str, limit_per_min: int) -> None:
    """
    Raise HTTPException(429) if this API key has exceeded limit_per_min in the
    current 60-second window. No-op (fail-open) if Redis is unavailable.
    """
    repo = getattr(request.app.state, "redis_repo", None)
    if repo is None:
        return  # fail-open: Redis not wired

    try:
        client = repo.client
        minute = int(time.time() // 60)
        rkey = f"nexus:ratelimit:{key_id}:{minute}"
        count = await client.incr(rkey)
        if count == 1:
            # Bucket lifetime > 60s to cover clock skew; auto-expires so no cleanup.
            await client.expire(rkey, 90)
    except HTTPException:
        raise
    except Exception as exc:  # Redis error → fail-open
        logger.warning("Rate-limit check failed (fail-open): %s", exc)
        return

    if count > limit_per_min:
        retry_after = 60 - int(time.time()) % 60
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
