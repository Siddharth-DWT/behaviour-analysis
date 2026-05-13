# Redis Layer Integration Plan

## Goal

Introduce Redis as the shared coordination and temporary data-sharing layer for one processing session, while keeping Postgres as the final source of truth.

Target model:

- Redis = session working memory, orchestration state, transient artifacts, inter-agent signals
- Postgres = persisted outputs, reports, searchable history, registry data
- HTTP = trigger, polling, health, manual inspection

## Current State

Current pipeline behavior is mixed:

- API Gateway keeps some state in process-local memory
- Video Agent pushes pending signal batches to Redis lists
- Language, Conversation, and Fusion use `shared.utils.message_bus`
- Voice Agent does not appear to publish results to Redis directly
- Agents still pass important intermediate data through HTTP payloads and local variables

This causes:

- inconsistent coordination patterns
- restart fragility
- poor observability
- harder multi-container scaling
- duplicate plumbing for transient state

## Desired Architecture

Processing DAG:

```text
Upload
  ->
Voice
  ->
Redis transcript + diarization + voice signals
  ->
Language + Video
  ->
Redis language/video signals
  ->
Conversation
  ->
Redis conversation signals
  ->
Fusion
  ->
Redis fusion signals + alerts
  ->
API Gateway persists final outputs to Postgres
```

Operational principle:

- Redis stores only small/intermediate artifacts and coordination state
- Raw media bytes stay on disk/object storage, not in Redis

## Scope

This plan covers:

- session state storage
- agent status tracking
- shared artifact storage
- unified signal transport
- dependency handoff between agents
- cleanup and TTL strategy
- rollout strategy

This plan does not change:

- Postgres schema as the main persistence target
- registry matching logic
- report generation behavior
- media file storage model

## Redis Data Model

### 1. Session State

Key:

```text
nexus:session:{session_id}:state
```

Recommended type:

- Redis hash

Fields:

- `status`
- `current_step`
- `meeting_type`
- `created_at`
- `updated_at`
- `error`
- `schema_version`

Example:

```json
{
  "status": "processing",
  "current_step": "language",
  "meeting_type": "sales_call",
  "created_at": "2026-05-13T10:15:00Z",
  "updated_at": "2026-05-13T10:16:12Z",
  "error": "",
  "schema_version": "1"
}
```

### 2. Agent Status

Key pattern:

```text
nexus:session:{session_id}:agent:{agent_name}
```

Recommended type:

- Redis hash

Fields:

- `status`
- `started_at`
- `completed_at`
- `attempt`
- `signal_count`
- `summary_key`
- `error`
- `schema_version`

Allowed statuses:

- `queued`
- `running`
- `completed`
- `failed`
- `skipped`
- `retrying`

### 3. Shared Artifacts

Keys:

```text
nexus:session:{sid}:transcript
nexus:session:{sid}:speakers
nexus:session:{sid}:diarization
nexus:session:{sid}:summary:voice
nexus:session:{sid}:summary:language
nexus:session:{sid}:summary:conversation
nexus:session:{sid}:summary:video
nexus:session:{sid}:media
```

Recommended type:

- Redis string containing JSON

Notes:

- Store transcript segments, speaker lists, diarization segments, summaries, and media metadata
- Do not store raw audio or video bytes

Media metadata example:

```json
{
  "media_path": "/app/data/recordings/abc.mp4",
  "duration_seconds": 1234,
  "speaker_count": 4
}
```

### 4. Signals

Key pattern:

```text
nexus:signals:{session_id}:{agent}
```

Recommended type:

- Redis Stream

Streams:

- `nexus:signals:{sid}:voice`
- `nexus:signals:{sid}:language`
- `nexus:signals:{sid}:video`
- `nexus:signals:{sid}:conversation`
- `nexus:signals:{sid}:fusion`

Canonical signal schema:

```json
{
  "event_id": "uuid",
  "session_id": "uuid",
  "agent": "language",
  "speaker_id": "Speaker_0",
  "registry_id": "",
  "signal_type": "sentiment_score",
  "value": 0.72,
  "value_text": "positive",
  "confidence": 0.81,
  "window_start_ms": 12000,
  "window_end_ms": 18000,
  "metadata": {},
  "created_at": "2026-05-13T10:17:00Z",
  "schema_version": "1"
}
```

## Required Building Blocks

### Shared Redis Access Layer

Create a shared Redis integration module in `shared` to avoid each service inventing its own format.

Suggested responsibilities:

- key naming helpers
- JSON serialization helpers
- session state helpers
- agent status helpers
- artifact read/write helpers
- signal stream publish/read helpers
- TTL helpers
- schema version constants

Suggested module area:

- `shared/utils/redis_state.py`
- `shared/utils/redis_keys.py`
- or a small `shared/redis/` package

### Message Bus Unification

Current Redis usage is split between:

- raw Redis list pushes in video
- Redis stream/message bus use in language/conversation/fusion

We should unify on one transport:

- Redis Streams for all signal/event publishing

This likely requires:

- extending or refactoring [message_bus.py](C:\Users\ADMIN\Desktop\behaviour-analysis\shared\utils\message_bus.py)
- migrating video agent writes off list-based buffering
- adding voice agent publishing into the same model

## Service-by-Service Changes

### API Gateway

Files likely involved:

- [main.py](C:\Users\ADMIN\Desktop\behaviour-analysis\services\api_gateway\main.py)
- [database.py](C:\Users\ADMIN\Desktop\behaviour-analysis\services\api_gateway\database.py)

Changes:

- replace process-local session progress with Redis-backed state
- write session-level lifecycle status into Redis
- write agent statuses as each downstream call begins/completes/fails
- stop depending on mixed temporary handoff mechanisms
- drain/read intermediate results from Redis in a consistent way
- persist final outputs from Redis-backed collected artifacts/signals into Postgres

### Voice Agent

Files likely involved:

- `services/voiceAgent/...`

Changes:

- publish transcript to Redis
- publish diarization to Redis
- publish speaker list to Redis
- publish voice signals to Redis Stream
- write its own agent status key

### Language Agent

Files likely involved:

- `services/language_agent/...`

Changes:

- read transcript/speakers/diarization from Redis or continue receiving them from gateway during transition
- publish language signals to Redis Stream using the canonical schema
- write summary to Redis
- update agent status key

### Video Agent

Files likely involved:

- [main.py](C:\Users\ADMIN\Desktop\behaviour-analysis\services\video_agent\main.py)
- [feature_extractor.py](C:\Users\ADMIN\Desktop\behaviour-analysis\services\video_agent\feature_extractor.py)

Changes:

- replace `nexus:pending:{session_id}:video` list writes with Redis Streams
- write video summary/artifacts into Redis
- publish video signals using canonical schema
- write agent status key
- keep face embeddings out of broad Redis usage unless explicitly needed and size-bounded

### Conversation Agent

Files likely involved:

- `services/conversation_agent/...`

Changes:

- consume language outputs from Redis-backed artifacts/signals
- publish conversation signals to Redis Stream
- write summary to Redis
- update agent status key

### Fusion Agent

Files likely involved:

- `services/fusion_agent/...`

Changes:

- read all upstream signals from Redis Streams or from normalized Redis materialization
- publish fusion signals and alerts to Redis
- write summary/status

## Rollout Strategy

### Phase 1: Shared Redis Foundation

Deliverables:

- shared Redis helper module
- canonical key naming
- canonical signal schema
- JSON serialization helpers
- TTL policy helpers

Success criteria:

- all services can import one shared Redis API
- no new ad hoc key formats are introduced

### Phase 2: Session State and Agent Status

Deliverables:

- Redis-backed session state in gateway
- per-agent status keys written by all services

Success criteria:

- pipeline progress survives gateway restart
- debugging can rely on Redis state instead of in-memory state

### Phase 3: Signal Transport Unification

Deliverables:

- all agent signals published to Redis Streams
- video agent migrated off Redis lists
- voice agent added to Redis signal publishing

Success criteria:

- one consistent temporary signal transport across agents
- gateway no longer needs agent-specific signal buffering rules

### Phase 4: Shared Artifact Handoff

Deliverables:

- transcript, speakers, diarization, summaries stored in Redis
- downstream agents consume from Redis-backed artifacts

Success criteria:

- inter-agent dependency flow is explicit and observable
- less brittle HTTP payload chaining

### Phase 5: Gateway Persistence Pass

Deliverables:

- gateway reads final consolidated artifacts/signals
- gateway persists them to Postgres as final truth

Success criteria:

- Redis becomes temporary state only
- Postgres remains system of record

### Phase 6: Cleanup, TTL, and Monitoring

Deliverables:

- TTLs for transient session data
- cleanup jobs or lazy cleanup on completion
- logging/metrics around stream lag and missing dependencies

Success criteria:

- Redis memory use remains bounded
- stale sessions do not accumulate indefinitely

## TTL and Cleanup Policy

Recommended defaults:

- session state: 24 hours after completion
- agent status: 24 hours after completion
- transcript/diarization/summaries: 24 hours after completion
- signal streams: 24 hours after completion or capped by stream length

Possible cleanup approaches:

- set TTL on all session-scoped keys when session finishes
- scheduled cleanup job for abandoned sessions
- stream trimming with `MAXLEN` policy where appropriate

## Observability Requirements

Add enough metadata to debug pipelines without opening Postgres first.

Minimum observability:

- session state key always present during active runs
- agent status keys for every agent
- signal counts per agent
- timestamps for started/completed
- last error message
- summary key references

Useful additions:

- retry counts
- upstream dependency markers
- stream event counts
- final persistence checkpoint markers

## Risks

### 1. Dual-write transition complexity

During migration we may temporarily write to both legacy and new Redis paths.

Mitigation:

- feature flags
- phased rollout by agent
- explicit compatibility window

### 2. Schema drift across agents

If each agent shapes signal payloads differently, Redis unification will break.

Mitigation:

- shared schema helper in `shared`
- normalization at publish time
- strict tests for required fields

### 3. Oversized Redis payloads

Large transcripts or summaries can grow quickly.

Mitigation:

- store only compact JSON
- avoid binary payloads
- compress only if needed later

### 4. Ordering and dependency races

Fusion or conversation may read before upstream data is ready.

Mitigation:

- agent status keys
- dependency checks before consume
- retry/backoff policy

## Testing Strategy

### Unit Tests

Add tests for:

- key generation
- schema normalization
- state/status serialization
- stream publish/read helpers

### Integration Tests

Add pipeline tests for:

- Voice -> Language handoff
- Voice -> Video handoff
- Language -> Conversation dependency
- Fusion read-after-upstream completion
- Gateway persistence from Redis-backed intermediate data

### Failure Scenarios

Test:

- gateway restart mid-session
- one agent failure while others succeed
- missing Redis artifact
- duplicate publish event
- retry after partial completion

## Suggested File-Level Work Plan

### New shared modules

- `shared/utils/redis_keys.py`
- `shared/utils/redis_state.py`
- `shared/utils/redis_signals.py`

### Main files likely to change

- [main.py](C:\Users\ADMIN\Desktop\behaviour-analysis\services\api_gateway\main.py)
- [main.py](C:\Users\ADMIN\Desktop\behaviour-analysis\services\video_agent\main.py)
- [message_bus.py](C:\Users\ADMIN\Desktop\behaviour-analysis\shared\utils\message_bus.py)
- voice agent entrypoints
- language agent entrypoints
- conversation agent entrypoints
- fusion agent entrypoints

## Recommended First Milestone

If we want the safest first implementation, start with:

1. shared Redis key/schema helpers
2. gateway session state in Redis
3. per-agent status keys
4. video signal transport migration from Redis list to Redis Stream

Why this order:

- small enough to land safely
- improves observability quickly
- creates the contract needed for the rest of the migration
- does not require a full pipeline rewrite on day one

## Definition of Done

The Redis layer integration is complete when:

- no critical pipeline state exists only in process-local memory
- all agent statuses are visible in Redis
- all transient signals use one consistent Redis transport
- transcript/speakers/diarization/summaries are available in Redis during processing
- gateway persists final outputs from Redis-backed working state into Postgres
- Redis keys expire or are cleaned up after session completion
