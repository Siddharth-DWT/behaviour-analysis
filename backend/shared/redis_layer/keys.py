"""Canonical Redis key builders for session-scoped coordination."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RedisKeys:
    """Single source of truth for Redis key naming."""

    SESSION_PREFIX: str = "nexus:session"
    SIGNAL_PREFIX: str = "nexus:signals"
    EVENT_PREFIX: str = "nexus:events"
    LOCK_PREFIX: str = "nexus:lock"
    DLQ_PREFIX: str = "nexus:dlq"
    LEGACY_PENDING_PREFIX: str = "nexus:pending"
    ALERT_PREFIX: str = "nexus:alerts"

    @classmethod
    def session_state(cls, session_id: str) -> str:
        return f"{cls.SESSION_PREFIX}:{session_id}:state"

    @classmethod
    def agent_status(cls, session_id: str, agent: str) -> str:
        return f"{cls.SESSION_PREFIX}:{session_id}:agent:{agent}"

    @classmethod
    def artifact(cls, session_id: str, artifact_name: str) -> str:
        return f"{cls.SESSION_PREFIX}:{session_id}:{artifact_name}"

    @classmethod
    def transcript(cls, session_id: str) -> str:
        return cls.artifact(session_id, "transcript")

    @classmethod
    def speakers(cls, session_id: str) -> str:
        return cls.artifact(session_id, "speakers")

    @classmethod
    def diarization(cls, session_id: str) -> str:
        return cls.artifact(session_id, "diarization")

    @classmethod
    def summary(cls, session_id: str, agent: str) -> str:
        return cls.artifact(session_id, f"summary:{agent}")

    @classmethod
    def media(cls, session_id: str) -> str:
        return cls.artifact(session_id, "media")

    @classmethod
    def signal_stream(cls, session_id: str, agent: str) -> str:
        return f"{cls.SIGNAL_PREFIX}:{session_id}:{agent}"

    @classmethod
    def events(cls, session_id: str) -> str:
        return f"{cls.EVENT_PREFIX}:{session_id}"

    @classmethod
    def lock(cls, session_id: str, agent: str) -> str:
        return f"{cls.LOCK_PREFIX}:{session_id}:{agent}"

    @classmethod
    def dlq(cls, agent: str) -> str:
        return f"{cls.DLQ_PREFIX}:{agent}"

    @classmethod
    def alerts(cls, session_id: str) -> str:
        return f"{cls.ALERT_PREFIX}:{session_id}"

    @classmethod
    def legacy_pending(cls, session_id: str, agent: str) -> str:
        return f"{cls.LEGACY_PENDING_PREFIX}:{session_id}:{agent}"
