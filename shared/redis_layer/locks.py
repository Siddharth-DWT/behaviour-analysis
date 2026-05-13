"""Session-scoped Redis locks for agent execution."""

from __future__ import annotations

import uuid

from .client import RedisClientFactory
from .keys import RedisKeys


class RedisLockManager:
    """Coordinate per-session agent execution with short-lived Redis locks."""

    def __init__(self, ttl_seconds: int = 30 * 60) -> None:
        self._ttl_seconds = ttl_seconds

    @property
    def client(self):
        return RedisClientFactory.get_async_client()

    async def acquire(self, session_id: str, agent: str, token: str | None = None) -> str | None:
        lock_token = token or str(uuid.uuid4())
        acquired = await self.client.set(
            RedisKeys.lock(session_id, agent),
            lock_token,
            ex=self._ttl_seconds,
            nx=True,
        )
        if acquired:
            return lock_token
        return None

    async def refresh(self, session_id: str, agent: str, token: str) -> bool:
        key = RedisKeys.lock(session_id, agent)
        current = await self.client.get(key)
        if current != token:
            return False
        await self.client.expire(key, self._ttl_seconds)
        return True

    async def release(self, session_id: str, agent: str, token: str) -> bool:
        key = RedisKeys.lock(session_id, agent)
        current = await self.client.get(key)
        if current != token:
            return False
        await self.client.delete(key)
        return True
