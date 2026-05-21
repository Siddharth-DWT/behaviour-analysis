"""AgentJobDispatcher — push jobs to Redis Streams and await results via pub/sub.

DSA rationale: pub/sub over flat polling
  Flat polling at 500 ms → O(T / 0.5) Redis round-trips across job lifetime T.
  For a 30-minute voice analysis that is ~3 600 wasted polls.
  Pub/sub subscription → O(1) Redis calls; the caller wakes the instant the
  result is written.  Exponential-backoff polling is used only as a fallback
  when pub/sub is unavailable.

Notification protocol:
  • write_result() in RedisRepository publishes "1" to nexus:result-ready:{sid}:{agent}.
  • AgentJobDispatcher.dispatch() subscribes before pushing the job (no race condition).
  • asyncio.Event bridges the pub/sub listener task and the dispatch caller.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .repository import RedisRepository

logger = logging.getLogger(__name__)

_RESULT_NOTIFY_PREFIX = "nexus:result-ready"


class AgentJobDispatcher:
    """
    Dispatches jobs to NEXUS agents via Redis Streams and awaits their results.

    Protocol (pub/sub fast path):
        1. Subscribe to nexus:result-ready:{session_id}:{agent} BEFORE pushing the job
           so no notification is missed regardless of agent speed.
        2. Push the job payload to nexus:jobs:{agent} stream.
        3. Poll once for an already-written result (handles retries / very fast agents).
        4. Block on asyncio.Event set by the background pub/sub listener task.
        5. Read the result from Redis and return it.

    Falls back to exponential-backoff polling if pub/sub is unavailable.
    """

    _POLL_START_S: float = 0.1
    _POLL_MAX_S: float = 2.0

    def __init__(self, repo: RedisRepository) -> None:
        self._repo = repo

    async def dispatch(
        self,
        agent: str,
        session_id: str,
        payload: dict,
        timeout: float,
        label: str = "",
    ) -> dict:
        label = label or agent
        t0 = time.monotonic()
        logger.info("[%s] → %s (Redis)", session_id, label)

        channel = f"{_RESULT_NOTIFY_PREFIX}:{session_id}:{agent}"
        result_event: asyncio.Event = asyncio.Event()
        pubsub = self._repo.client.pubsub()
        listener_task: Optional[asyncio.Task] = None

        try:
            await pubsub.subscribe(channel)

            async def _listen() -> None:
                async for msg in pubsub.listen():
                    if isinstance(msg, dict) and msg.get("type") == "message":
                        result_event.set()
                        return

            listener_task = asyncio.create_task(_listen())

            await self._repo.push_job(agent, session_id, payload)

            # Fast path: result may already be written (retry or very quick agent).
            existing = await self._repo.read_result(session_id, agent)
            if existing is not None:
                return self._unwrap(existing, label, t0)

            remaining = timeout - (time.monotonic() - t0)
            try:
                await asyncio.wait_for(result_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError(f"{label} result not available after {timeout:.0f}s")

            result = await self._repo.read_result(session_id, agent)
            if result is None:
                raise RuntimeError(f"{label}: pub/sub notification received but result missing")
            return self._unwrap(result, label, t0)

        except (ConnectionError, OSError) as exc:
            logger.warning(
                "[%s] Pub/sub unavailable (%s) — falling back to polling", session_id, exc
            )
            return await self._poll(agent, session_id, label, timeout, t0)

        finally:
            if listener_task and not listener_task.done():
                listener_task.cancel()
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()
            except Exception:
                pass

    async def _poll(
        self,
        agent: str,
        session_id: str,
        label: str,
        timeout: float,
        t0: float,
    ) -> dict:
        """Exponential-backoff polling fallback.

        Doubles the sleep interval each iteration (0.1 → 0.2 → 0.4 → … → 2.0 s),
        reducing Redis round-trips by ~10× vs flat 500 ms polling for long jobs.
        """
        interval = self._POLL_START_S
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= timeout:
                raise TimeoutError(f"{label} result not available after {timeout:.0f}s")
            result = await self._repo.read_result(session_id, agent)
            if result is not None:
                return self._unwrap(result, label, t0)
            await asyncio.sleep(min(interval, self._POLL_MAX_S, timeout - elapsed))
            interval = min(interval * 2, self._POLL_MAX_S)

    @staticmethod
    def _unwrap(result: dict, label: str, t0: float) -> dict:
        if "error" in result:
            raise RuntimeError(f"{label} failed: {result['error']}")
        logger.info("← %s in %.1fs", label, time.monotonic() - t0)
        return result
