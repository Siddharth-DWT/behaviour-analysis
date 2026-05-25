"""Redis-backed orchestration event store."""

from __future__ import annotations

import json
from typing import Any

from .client import RedisClientFactory
from .keys import RedisKeys
from .schemas import EventRecord


class RedisEventStore:
    """Append-only event stream per session."""

    def __init__(self, stream_maxlen: int = 10_000) -> None:
        self._stream_maxlen = stream_maxlen

    @property
    def client(self):
        return RedisClientFactory.get_async_client()

    async def append(self, session_id: str, record: EventRecord) -> str:
        payload = record.model_dump()
        payload["payload"] = json.dumps(payload.get("payload", {}))
        return await self.client.xadd(
            RedisKeys.events(session_id),
            {key: str(value) for key, value in payload.items()},
            maxlen=self._stream_maxlen,
            approximate=True,
        )

    async def read_latest(self, session_id: str, count: int = 100) -> list[dict[str, Any]]:
        messages = await self.client.xrevrange(RedisKeys.events(session_id), count=count)
        events: list[dict[str, Any]] = []
        for event_id, fields in messages:
            payload = dict(fields)
            payload["event_id"] = str(payload.get("event_id") or event_id)
            if payload.get("payload"):
                payload["payload"] = json.loads(payload["payload"])
            events.append(payload)
        return events
