from shared.redis_layer.keys import RedisKeys
from shared.redis_layer.schemas import (
    AgentStatusRecord,
    EventRecord,
    SessionStateRecord,
    SignalRecord,
)


def test_redis_keys_follow_canonical_names() -> None:
    session_id = "session-123"
    assert RedisKeys.session_state(session_id) == "nexus:session:session-123:state"
    assert RedisKeys.agent_status(session_id, "voice") == "nexus:session:session-123:agent:voice"
    assert RedisKeys.transcript(session_id) == "nexus:session:session-123:transcript"
    assert RedisKeys.signal_stream(session_id, "fusion") == "nexus:signals:session-123:fusion"
    assert RedisKeys.events(session_id) == "nexus:events:session-123"
    assert RedisKeys.lock(session_id, "video") == "nexus:lock:session-123:video"
    assert RedisKeys.dlq("language") == "nexus:dlq:language"


def test_signal_record_round_trips_stream_fields() -> None:
    record = SignalRecord(
        session_id="session-123",
        agent="voice",
        speaker_id="Speaker_0",
        registry_id="registry-1",
        signal_type="stress_score",
        value=0.81,
        value_text="high",
        confidence=0.93,
        window_start_ms=1000,
        window_end_ms=2000,
        metadata={"source": "unit-test"},
    )

    stream_fields = record.to_stream_fields()
    parsed = SignalRecord.from_stream_fields({"id": "1-0", **stream_fields})

    assert parsed.session_id == record.session_id
    assert parsed.agent == record.agent
    assert parsed.speaker_id == record.speaker_id
    assert parsed.registry_id == record.registry_id
    assert parsed.signal_type == record.signal_type
    assert parsed.value == record.value
    assert parsed.value_text == record.value_text
    assert parsed.confidence == record.confidence
    assert parsed.window_start_ms == record.window_start_ms
    assert parsed.window_end_ms == record.window_end_ms
    assert parsed.metadata == record.metadata


def test_schema_defaults_are_stable() -> None:
    state = SessionStateRecord(status="running", current_step="language")
    status = AgentStatusRecord(status="completed", signal_count=5, summary_key="summary:language")
    event = EventRecord(session_id="session-123", agent="language", event_type="agent_completed")

    assert state.schema_version == "1"
    assert status.schema_version == "1"
    assert event.schema_version == "1"
