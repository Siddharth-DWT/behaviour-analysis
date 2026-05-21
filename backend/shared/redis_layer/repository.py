"""Repository-style Redis access layer for session coordination."""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from .client import RedisClientFactory
from .keys import RedisKeys
from .schemas import AgentStatusRecord, ArtifactRecord, DlqRecord, SessionStateRecord, SignalRecord

DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_STREAM_MAXLEN = 10_000


class RedisRepository:
    """Async repository for canonical Redis state and streams."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, stream_maxlen: int = DEFAULT_STREAM_MAXLEN) -> None:
        self._ttl_seconds = ttl_seconds
        self._stream_maxlen = stream_maxlen

    @property
    def client(self):
        return RedisClientFactory.get_async_client()

    async def set_session_state(self, session_id: str, record: SessionStateRecord) -> None:
        await self._write_json(RedisKeys.session_state(session_id), record.model_dump())

    async def get_session_state(self, session_id: str) -> Optional[dict[str, Any]]:
        return await self._read_json(RedisKeys.session_state(session_id))

    async def clear_session_state(self, session_id: str) -> None:
        await self.client.delete(RedisKeys.session_state(session_id))

    async def set_agent_status(self, session_id: str, agent: str, record: AgentStatusRecord) -> None:
        await self._write_json(RedisKeys.agent_status(session_id, agent), record.model_dump())

    async def get_agent_status(self, session_id: str, agent: str) -> Optional[dict[str, Any]]:
        return await self._read_json(RedisKeys.agent_status(session_id, agent))

    async def write_artifact(self, session_id: str, artifact_name: str, payload: dict[str, Any]) -> str:
        record = ArtifactRecord(name=artifact_name, session_id=session_id, payload=payload)
        key = RedisKeys.artifact(session_id, artifact_name)
        await self._write_json(key, record.model_dump())
        return key

    async def read_artifact(self, session_id: str, artifact_name: str) -> Optional[dict[str, Any]]:
        record = await self._read_json(RedisKeys.artifact(session_id, artifact_name))
        if record:
            return record.get("payload", {})
        return None

    async def publish_signal(self, signal: SignalRecord) -> str:
        return await self.client.xadd(
            RedisKeys.signal_stream(signal.session_id, signal.agent),
            signal.to_stream_fields(),
            maxlen=self._stream_maxlen,
            approximate=True,
        )

    async def publish_signals(self, signals: Iterable[SignalRecord]) -> int:
        count = 0
        for signal in signals:
            await self.publish_signal(signal)
            count += 1
        return count

    async def read_latest_signals(self, session_id: str, agent: str, count: int = 100) -> list[dict[str, Any]]:
        stream_name = RedisKeys.signal_stream(session_id, agent)
        messages = await self.client.xrevrange(stream_name, count=count)
        parsed = [SignalRecord.from_stream_fields({"id": msg_id, **fields}).model_dump() for msg_id, fields in messages]
        parsed.reverse()
        return parsed

    async def read_signals_since(
        self,
        session_id: str,
        agents: list[str],
        last_ids: Optional[dict[str, str]] = None,
        block_ms: int = 1000,
        count: int = 100,
    ) -> dict[str, list[dict[str, Any]]]:
        streams = {
            RedisKeys.signal_stream(session_id, agent): (last_ids or {}).get(RedisKeys.signal_stream(session_id, agent), "$")
            for agent in agents
        }
        results = await self.client.xread(streams, block=block_ms, count=count)
        parsed: dict[str, list[dict[str, Any]]] = {}
        for stream_name, messages in results:
            parsed[stream_name] = [
                SignalRecord.from_stream_fields({"id": msg_id, **fields}).model_dump()
                for msg_id, fields in messages
            ]
        return parsed

    async def drain_legacy_pending_signals(self, session_id: str, agent: str) -> list[dict[str, Any]]:
        key = RedisKeys.legacy_pending(session_id, agent)
        batches = await self.client.lrange(key, 0, -1)
        if not batches:
            return []
        await self.client.delete(key)
        signals: list[dict[str, Any]] = []
        for batch_json in batches:
            signals.extend(json.loads(batch_json))
        return signals

    # ── Inter-service job dispatch ────────────────────────────────────────────

    async def push_job(self, agent: str, session_id: str, payload: dict[str, Any]) -> str:
        """Push a job onto the agent's job stream. Returns the stream entry ID."""
        return await self.client.xadd(
            RedisKeys.job_stream(agent),
            {"session_id": session_id, "payload": json.dumps(payload)},
            maxlen=1000,
            approximate=True,
        )

    async def write_result(self, session_id: str, agent: str, result: dict[str, Any]) -> str:
        """Store an agent result and notify any waiting dispatcher via pub/sub."""
        key = await self.write_artifact(session_id, f"result:{agent}", result)
        channel = f"nexus:result-ready:{session_id}:{agent}"
        try:
            await self.client.publish(channel, "1")
        except Exception:
            pass
        return key

    async def read_result(self, session_id: str, agent: str) -> Optional[dict[str, Any]]:
        """Read agent result. Returns None if not yet available."""
        return await self.read_artifact(session_id, f"result:{agent}")

    async def publish_dlq(self, agent: str, record: DlqRecord) -> str:
        payload = record.model_dump()
        payload["original_payload"] = json.dumps(payload["original_payload"])
        return await self.client.xadd(RedisKeys.dlq(agent), {k: str(v) for k, v in payload.items()}, maxlen=self._stream_maxlen, approximate=True)

    async def publish_alert(self, session_id: str, payload: dict[str, Any]) -> str:
        stream_payload = {
            "speaker_id": str(payload.get("speaker_id", "")),
            "alert_type": str(payload.get("alert_type", "")),
            "severity": str(payload.get("severity", "yellow")),
            "title": str(payload.get("title", "")),
            "description": str(payload.get("description", "")),
            "evidence": json.dumps(payload.get("evidence", {})),
        }
        return await self.client.xadd(
            RedisKeys.alerts(session_id),
            stream_payload,
            maxlen=self._stream_maxlen,
            approximate=True,
        )

    async def expire_session_scope(self, session_id: str, extra_keys: Optional[list[str]] = None) -> None:
        keys = [
            RedisKeys.session_state(session_id),
            RedisKeys.transcript(session_id),
            RedisKeys.speakers(session_id),
            RedisKeys.diarization(session_id),
            RedisKeys.media(session_id),
            RedisKeys.events(session_id),
        ]
        keys.extend(extra_keys or [])
        for key in keys:
            await self.client.expire(key, self._ttl_seconds)

    async def _write_json(self, key: str, payload: dict[str, Any]) -> None:
        await self.client.set(key, json.dumps(payload))
        await self.client.expire(key, self._ttl_seconds)

    async def _read_json(self, key: str) -> Optional[dict[str, Any]]:
        raw = await self.client.get(key)
        if not raw:
            return None
        return json.loads(raw)


class SyncRedisRepository:
    """Sync repository for CPU-bound or thread-pool agent paths."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, stream_maxlen: int = DEFAULT_STREAM_MAXLEN) -> None:
        self._ttl_seconds = ttl_seconds
        self._stream_maxlen = stream_maxlen

    @property
    def client(self):
        return RedisClientFactory.get_sync_client()

    def publish_signal(self, signal: SignalRecord) -> str:
        return self.client.xadd(
            RedisKeys.signal_stream(signal.session_id, signal.agent),
            signal.to_stream_fields(),
            maxlen=self._stream_maxlen,
            approximate=True,
        )

    def publish_signal_batch(self, signals: Iterable[SignalRecord]) -> int:
        count = 0
        for signal in signals:
            self.publish_signal(signal)
            count += 1
        return count

    def write_artifact(self, session_id: str, artifact_name: str, payload: dict[str, Any]) -> str:
        record = ArtifactRecord(name=artifact_name, session_id=session_id, payload=payload)
        key = RedisKeys.artifact(session_id, artifact_name)
        self.client.set(key, json.dumps(record.model_dump()))
        self.client.expire(key, self._ttl_seconds)
        return key

    def set_agent_status(self, session_id: str, agent: str, record: AgentStatusRecord) -> None:
        key = RedisKeys.agent_status(session_id, agent)
        self.client.set(key, json.dumps(record.model_dump()))
        self.client.expire(key, self._ttl_seconds)

    def set_session_state(self, session_id: str, record: SessionStateRecord) -> None:
        key = RedisKeys.session_state(session_id)
        self.client.set(key, json.dumps(record.model_dump()))
        self.client.expire(key, self._ttl_seconds)

    def publish_dlq(self, agent: str, record: DlqRecord) -> str:
        payload = record.model_dump()
        payload["original_payload"] = json.dumps(payload["original_payload"])
        return self.client.xadd(RedisKeys.dlq(agent), {k: str(v) for k, v in payload.items()}, maxlen=self._stream_maxlen, approximate=True)

    # ── Inter-service job dispatch (sync path — video agent uses thread pool) ──

    def write_result(self, session_id: str, agent: str, result: dict[str, Any]) -> str:
        """Store an agent result and notify any waiting dispatcher via pub/sub (sync)."""
        key = self.write_artifact(session_id, f"result:{agent}", result)
        channel = f"nexus:result-ready:{session_id}:{agent}"
        try:
            self.client.publish(channel, "1")
        except Exception:
            pass
        return key

    def publish_alert(self, session_id: str, payload: dict[str, Any]) -> str:
        stream_payload = {
            "speaker_id": str(payload.get("speaker_id", "")),
            "alert_type": str(payload.get("alert_type", "")),
            "severity": str(payload.get("severity", "yellow")),
            "title": str(payload.get("title", "")),
            "description": str(payload.get("description", "")),
            "evidence": json.dumps(payload.get("evidence", {})),
        }
        return self.client.xadd(
            RedisKeys.alerts(session_id),
            stream_payload,
            maxlen=self._stream_maxlen,
            approximate=True,
        )
