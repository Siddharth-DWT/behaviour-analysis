"""Redis client factory with shared async and sync pools."""

from __future__ import annotations

from typing import Optional

import redis
import redis.asyncio as aioredis

from shared.config.settings import config


class RedisClientFactory:
    """Lazy-initialized Redis client factory."""

    _async_client: Optional[aioredis.Redis] = None
    _sync_client: Optional[redis.Redis] = None

    @classmethod
    def get_async_client(cls) -> aioredis.Redis:
        if cls._async_client is None:
            cls._async_client = aioredis.from_url(
                config.redis_url,
                decode_responses=True,
                max_connections=20,
            )
        return cls._async_client

    @classmethod
    async def close_async_client(cls) -> None:
        if cls._async_client is not None:
            await cls._async_client.close()
            cls._async_client = None

    @classmethod
    def get_sync_client(cls) -> redis.Redis:
        if cls._sync_client is None:
            cls._sync_client = redis.from_url(
                config.redis_url,
                decode_responses=True,
                socket_timeout=5,
            )
        return cls._sync_client

    @classmethod
    def close_sync_client(cls) -> None:
        if cls._sync_client is not None:
            cls._sync_client.close()
            cls._sync_client = None
