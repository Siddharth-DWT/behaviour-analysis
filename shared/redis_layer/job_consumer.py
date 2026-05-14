"""Abstract base for Redis Streams job consumers — used by all NEXUS agents.

Implements the Template Method pattern:
  RedisJobConsumer.run()       — owns the consumer loop skeleton
  RedisJobConsumer._handle()   — owns per-message lifecycle (parse → process → write → ACK)
  RedisJobConsumer.process_job — abstract; subclasses supply agent-specific logic

The fire-and-forget variant (VideoJobConsumer) overrides _handle() to ACK immediately
and dispatch a background task, because video processing takes 10–90 minutes.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod

from .keys import RedisKeys
from .repository import RedisRepository

logger = logging.getLogger(__name__)


class RedisJobConsumer(ABC):
    """
    Template-Method base class for Redis Streams job consumers.

    Subclasses implement ``process_job`` with their agent-specific logic.
    The base class owns the full consumer lifecycle:
        ensure_group → xreadgroup → deserialise → process_job → write_result → xack
    """

    def __init__(self, agent_name: str, group: str, consumer: str) -> None:
        self._agent = agent_name
        self._stream = RedisKeys.job_stream(agent_name)
        self._group = group
        self._consumer = consumer
        self._repo = RedisRepository()

    @abstractmethod
    async def process_job(self, session_id: str, payload: dict) -> dict:
        """Process one job and return the result dict to be stored."""
        ...

    async def run(self) -> None:
        """Main consumer loop. Run as an asyncio background task."""
        await self._ensure_group()
        logger.info(
            "[%s] Consumer ready (stream=%s, group=%s)",
            self._agent, self._stream, self._group,
        )
        while True:
            try:
                messages = await self._repo.client.xreadgroup(
                    self._group,
                    self._consumer,
                    {self._stream: ">"},
                    count=1,
                    block=5000,
                )
                if not messages:
                    continue
                for _, entries in messages:
                    for msg_id, fields in entries:
                        await self._handle(msg_id, fields)
            except asyncio.CancelledError:
                logger.info("[%s] Consumer shutting down.", self._agent)
                return
            except Exception as exc:
                logger.error("[%s] Consumer loop error: %s", self._agent, exc, exc_info=True)
                await asyncio.sleep(1)

    async def _ensure_group(self) -> None:
        try:
            await self._repo.client.xgroup_create(
                self._stream, self._group, id="0", mkstream=True
            )
            logger.info("[%s] Consumer group '%s' created", self._agent, self._group)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                logger.warning("[%s] xgroup_create: %s", self._agent, exc)

    async def _handle(self, msg_id: str, fields: dict) -> None:
        """Default handler: process synchronously, write result, then ACK."""
        session_id: str = fields.get("session_id", "")
        try:
            payload = json.loads(fields.get("payload", "{}"))
            result = await self.process_job(session_id, payload)
            await self._repo.write_result(session_id, self._agent, result)
            logger.info(
                "[%s] Job %s done (session=%s, signals=%d)",
                self._agent, msg_id, session_id, len(result.get("signals", [])),
            )
        except Exception as exc:
            logger.error(
                "[%s] Job %s failed (session=%s): %s",
                self._agent, msg_id, session_id, exc, exc_info=True,
            )
            try:
                await self._repo.write_result(session_id, self._agent, {"error": str(exc)})
            except Exception:
                pass
        finally:
            await self._ack(msg_id)

    async def _ack(self, msg_id: str) -> None:
        try:
            await self._repo.client.xack(self._stream, self._group, msg_id)
        except Exception as exc:
            logger.warning("[%s] xack failed for %s: %s", self._agent, msg_id, exc)
