"""Canonical Redis payload schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"
AgentLifecycleStatus = Literal["queued", "running", "completed", "failed", "skipped", "retrying"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RedisRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION


class SessionStateRecord(RedisRecord):
    status: str
    current_step: str
    meeting_type: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    error: str = ""


class AgentStatusRecord(RedisRecord):
    status: AgentLifecycleStatus
    started_at: str = ""
    completed_at: str = ""
    attempt: int = 1
    signal_count: int = 0
    summary_key: str = ""
    error: str = ""


class SignalRecord(RedisRecord):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    agent: str
    speaker_id: str = "unknown"
    registry_id: Optional[str] = None
    signal_type: str
    value: Optional[float] = None
    value_text: str = ""
    confidence: float = 0.5
    window_start_ms: int = 0
    window_end_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)

    def to_stream_fields(self) -> dict[str, str]:
        import json
        payload = self.model_dump()
        stream_fields: dict[str, str] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                stream_fields[key] = json.dumps(value)
            elif value is None:
                stream_fields[key] = ""
            else:
                stream_fields[key] = str(value)
        return stream_fields

    @classmethod
    def from_stream_fields(cls, fields: dict[str, Any]) -> "SignalRecord":
        metadata = fields.get("metadata", {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        value = fields.get("value", None)
        if value in ("", None):
            value = None
        else:
            try:
                value = float(value)
            except Exception:
                value = None
        return cls(
            event_id=str(fields.get("event_id") or uuid4()),
            session_id=str(fields.get("session_id", "")),
            agent=str(fields.get("agent", "unknown")),
            speaker_id=str(fields.get("speaker_id", "unknown")),
            registry_id=str(fields.get("registry_id", "")),
            signal_type=str(fields.get("signal_type", "")),
            value=value,
            value_text=str(fields.get("value_text", "")),
            confidence=float(fields.get("confidence", 0.5)),
            window_start_ms=int(float(fields.get("window_start_ms", 0) or 0)),
            window_end_ms=int(float(fields.get("window_end_ms", 0) or 0)),
            metadata=metadata,
            created_at=str(fields.get("created_at", utc_now_iso())),
            schema_version=str(fields.get("schema_version", SCHEMA_VERSION)),
        )


class EventRecord(RedisRecord):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    event_type: str
    agent: str
    created_at: str = Field(default_factory=utc_now_iso)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_stream_fields(self) -> dict[str, str]:
        import json
        payload = self.model_dump()
        payload["payload"] = json.dumps(payload["payload"])
        return {k: str(v) for k, v in payload.items()}


class DlqRecord(RedisRecord):
    failed_agent: str
    reason: str
    created_at: str = Field(default_factory=utc_now_iso)
    retry_count: int = 0
    original_payload: dict[str, Any] = Field(default_factory=dict)


class ArtifactRecord(RedisRecord):
    name: str
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
