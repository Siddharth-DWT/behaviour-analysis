"""NEXUS message bus backed by the shared Redis coordination layer."""
import json
from typing import Optional
import redis.asyncio as aioredis

from shared.config.settings import config
from shared.redis_layer.client import RedisClientFactory
from shared.redis_layer.keys import RedisKeys
from shared.redis_layer.repository import RedisRepository
from shared.redis_layer.schemas import SignalRecord


class MessageBus:
    """Async Redis Streams wrapper for NEXUS inter-agent messaging."""
    
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._repository = RedisRepository(stream_maxlen=config.max_stream_length)
    
    async def connect(self):
        """Establish connection to Redis/Valkey."""
        if self._redis is None:
            self._redis = RedisClientFactory.get_async_client()
        return self._redis
    
    async def disconnect(self):
        """Close connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    async def publish_signal(
        self,
        session_id: str,
        agent: str,
        speaker_id: str,
        signal_type: str,
        value: float = None,
        value_text: str = "",
        confidence: float = 0.5,
        window_start_ms: int = 0,
        window_end_ms: int = 0,
        metadata: dict = None
    ) -> str:
        signal = SignalRecord(
            session_id=session_id,
            agent=agent,
            speaker_id=speaker_id,
            signal_type=signal_type,
            value=value,
            value_text=value_text,
            confidence=confidence,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            metadata=metadata or {},
        )
        return await self._repository.publish_signal(signal)
    
    async def subscribe(
        self,
        session_id: str,
        agents: list[str],
        last_ids: dict[str, str] = None,
        block_ms: int = 1000,
        count: int = 100
    ) -> dict:
        """
        Read new messages from one or more agent streams.
        
        Args:
            session_id: Which session to listen to
            agents: List of agent names to subscribe to ["voice", "language"]
            last_ids: Dict of {stream_name: last_read_id}. Use "$" for new messages only.
            block_ms: How long to block waiting for new messages (0 = no block)
            count: Max messages to read per stream
            
        Returns:
            Dict of {stream_name: [messages]}
        """
        r = await self.connect()
        
        streams = {}
        for agent in agents:
            stream_name = RedisKeys.signal_stream(session_id, agent)
            last_id = (last_ids or {}).get(stream_name, "$")
            streams[stream_name] = last_id
        
        if not streams:
            return {}
        
        result = await r.xread(streams, count=count, block=block_ms)
        
        # Parse results into a cleaner format
        parsed = {}
        if result:
            for stream_name, messages in result:
                parsed[stream_name] = [
                    SignalRecord.from_stream_fields({"id": msg_id, **fields}).model_dump()
                    for msg_id, fields in messages
                ]
        
        return parsed
    
    async def get_latest_signals(
        self,
        session_id: str,
        agent: str,
        signal_type: str = None,
        count: int = 10
    ) -> list[dict]:
        """
        Get the most recent signals from an agent's stream.
        Useful for FUSION Agent to get current state without subscribing.
        """
        messages = await self._repository.read_latest_signals(session_id, agent, count=count)
        results = []
        for fields in messages:
            if signal_type and fields.get("signal_type") != signal_type:
                continue
            results.append(fields)
        return results
    
    async def publish_alert(
        self,
        session_id: str,
        speaker_id: str,
        alert_type: str,
        severity: str,
        title: str,
        description: str = "",
        evidence: dict = None
    ) -> str:
        """Publish an alert to the alerts stream."""
        return await self._repository.publish_alert(
            session_id,
            {
                "speaker_id": speaker_id,
                "alert_type": alert_type,
                "severity": severity,
                "title": title,
                "description": description,
                "evidence": evidence or {},
            },
        )


# Singleton message bus instance
message_bus = MessageBus()
