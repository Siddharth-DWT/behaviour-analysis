"""
NEXUS Message Bus - Redis Streams wrapper for inter-agent communication.

Every agent publishes signals to its own stream and subscribes to other agents' streams.
The FUSION Agent subscribes to ALL streams.

Stream naming: nexus:stream:{agent}:{session_id}
Message format: {
    "agent": "voice",
    "speaker_id": "uuid",
    "signal_type": "stress_score",
    "value": "0.67",
    "value_text": "",
    "confidence": "0.55",
    "window_start_ms": "1710500420000",
    "window_end_ms": "1710500425000",
    "metadata": "{}"  # JSON string
}
"""
import json
import time
from typing import Optional
import redis.asyncio as aioredis

from shared.config.settings import config


class MessageBus:
    """Async Redis Streams wrapper for NEXUS inter-agent messaging."""
    
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Establish connection to Redis/Valkey."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                config.redis_url,
                decode_responses=True
            )
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
        """
        Publish a signal to the agent's stream.
        Returns the stream message ID.
        """
        r = await self.connect()
        stream_name = config.stream_name(agent, session_id)
        
        message = {
            "agent": agent,
            "speaker_id": speaker_id,
            "signal_type": signal_type,
            "value": str(value) if value is not None else "",
            "value_text": value_text,
            "confidence": str(confidence),
            "window_start_ms": str(window_start_ms),
            "window_end_ms": str(window_end_ms),
            "timestamp": str(int(time.time() * 1000)),
            "metadata": json.dumps(metadata or {})
        }
        
        msg_id = await r.xadd(
            stream_name, 
            message,
            maxlen=config.max_stream_length,
            approximate=True
        )
        
        return msg_id
    
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
            stream_name = config.stream_name(agent, session_id)
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
                    {"id": msg_id, **fields}
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
        r = await self.connect()
        stream_name = config.stream_name(agent, session_id)
        
        # XREVRANGE returns newest first
        messages = await r.xrevrange(stream_name, count=count)
        
        results = []
        for msg_id, fields in messages:
            if signal_type and fields.get("signal_type") != signal_type:
                continue
            fields["id"] = msg_id
            if fields.get("metadata"):
                fields["metadata"] = json.loads(fields["metadata"])
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
        r = await self.connect()
        stream_name = f"nexus:alerts:{session_id}"
        
        message = {
            "speaker_id": speaker_id,
            "alert_type": alert_type,
            "severity": severity,
            "title": title,
            "description": description,
            "evidence": json.dumps(evidence or {}),
            "timestamp": str(int(time.time() * 1000))
        }
        
        return await r.xadd(stream_name, message, maxlen=1000)


# Singleton message bus instance
message_bus = MessageBus()
