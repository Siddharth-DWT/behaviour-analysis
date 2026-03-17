# ARCHITECTURE.md — NEXUS System Architecture

## System Design Philosophy

NEXUS uses a **polyglot microservices architecture** where 7 independent agents communicate via Redis Streams. Each agent is a standalone FastAPI service that can be deployed, scaled, and restarted independently. The FUSION Agent orchestrates cross-modal analysis by subscribing to all other agents' output streams.

Key design decisions:
- **Agents don't talk to each other directly** — all communication goes through Redis Streams
- **Each agent owns its own models** — no shared GPU memory, no coupled inference
- **Fusion is pull-based** — the Fusion Agent pulls signals on a timer cycle, not push
- **Every signal carries a confidence score** — downstream consumers can filter by threshold
- **Baselines are per-speaker per-session** — no global "normal" assumptions

---

## Data Flow: Complete Pipeline

### Recording Mode (Current Implementation)
```
┌──────────────────────────────────────────────────────────────┐
│                    RECORDING UPLOAD                          │
│  User uploads .wav/.mp4 via POST /sessions                  │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    API GATEWAY                               │
│  - Validates file format                                     │
│  - Creates session record in PostgreSQL                      │
│  - Stores file in data/recordings/                           │
│  - Dispatches to appropriate agents                          │
└─────────────────────┬────────────────────────────────────────┘
                      │
          ┌───────────┼───────────────┐
          ▼           ▼               ▼
   ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │   VOICE    │ │ LANGUAGE │ │   CONVO      │
   │  Agent 1   │ │ Agent 2  │ │  Agent 6     │
   │            │ │          │ │              │
   │ librosa    │ │DistilBERT│ │ Turn-taking  │
   │ pyin F0    │ │ LIWC     │ │ Latency      │
   │ Whisper    │ │ Claude   │ │ Dominance    │
   └─────┬──────┘ └────┬─────┘ └──────┬───────┘
         │              │              │
         ▼              ▼              ▼
   ┌────────────────────────────────────────┐
   │         REDIS STREAMS                  │
   │                                        │
   │  stream:voice:{session_id}             │
   │  stream:language:{session_id}          │
   │  stream:conversation:{session_id}      │
   │  stream:facial:{session_id}            │
   │  stream:body:{session_id}              │
   │  stream:gaze:{session_id}              │
   │  stream:fusion:{session_id}            │
   │  stream:alerts:{session_id}            │
   └───────────────────┬────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  FUSION AGENT  │  Agent 7
              │                │
              │ Reads all 6    │
              │ agent streams  │
              │                │
              │ Runs 15 pair-  │
              │ wise rules     │
              │                │
              │ Detects 12     │
              │ compound       │
              │ patterns       │
              │                │
              │ Tracks 8       │
              │ temporal       │
              │ sequences      │
              │                │
              │ Outputs:       │
              │ - Unified      │
              │   Speaker State│
              │ - Alerts       │
              │ - Narrative    │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  PostgreSQL    │
              │                │
              │ sessions       │
              │ signals        │
              │ alerts         │
              │ speaker_       │
              │   profiles     │
              │ reports        │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  API GATEWAY   │
              │  REST + WS     │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  REACT         │
              │  DASHBOARD     │
              └────────────────┘
```

### Live Mode (Phase 4 — Future)
```
┌──────────────┐
│  Recall.ai   │  Joins Zoom/Meet/Teams via meeting URL
│  Bot         │  Streams audio + video in real-time
└──────┬───────┘
       │
       ├── Audio stream → Voice Agent (5s chunks, 2.5s hop)
       ├── Audio stream → Language Agent (transcript segments)
       ├── Video frames → Facial Agent (3 fps target)
       ├── Video frames → Body Agent (1 fps target)
       ├── Video frames → Gaze Agent (1 fps target)
       └── Transcript → Conversation Agent (turn events)
```

---

## Agent Specifications

### Agent 1: VOICE AGENT ✅ Built
**Port**: 8001
**Input**: Audio file path or audio chunk (WAV/MP3/M4A)
**Output**: Redis stream `stream:voice:{session_id}`

| Feature | Method | Library |
|---------|--------|---------|
| F0 (pitch) | pYIN algorithm | librosa |
| Energy | RMS in dB | librosa |
| Speech rate | WPM from transcript | faster-whisper |
| Jitter | Period perturbation | Custom (autocorrelation) |
| Shimmer | Amplitude perturbation | Custom (RMS per frame) |
| HNR | Harmonics-to-noise | Custom (autocorrelation) |
| Pauses | Inter-segment gaps > 250ms | Goldman-Eisler threshold |
| Fillers | Keyword match in transcript | Clark & Fox Tree taxonomy |
| Spectral centroid | Brightness indicator | librosa |

**Processing**: 5-second windows, 2.5-second hop (50% overlap)
**Calibration**: First ~90 seconds of speech → per-speaker baseline

**Rules implemented**:
- VOICE-STRESS-01: 7-component weighted composite (pitch 0.25, jitter 0.20, rate 0.15, filler 0.15, pause 0.10, HNR 0.10, shimmer 0.05)
- VOICE-FILLER-01/02: Detection + credibility thresholds (1.3% / 2.5% / 4.0%)
- VOICE-PITCH-01: Elevation flag (>8% / >15% / >25%)
- VOICE-RATE-01: Anomaly with sub-classification (anxiety / enthusiasm / rushing / disengagement / cognitive load / deliberation)
- VOICE-TONE-03/04: Nervous vs Confident tone (multi-factor scoring)

---

### Agent 2: LANGUAGE AGENT 🔲 Next to Build
**Port**: 8002
**Input**: Transcript segments from Whisper
**Output**: Redis stream `stream:language:{session_id}`

| Feature | Method | Library/API |
|---------|--------|-------------|
| Sentiment | Per-sentence polarity + magnitude | DistilBERT (HuggingFace) |
| Emotional intensity | LIWC-based word category counting | Custom LIWC dictionary |
| Persuasion | Cialdini 7 principles detection | Claude API (batch) |
| Buying signals | SPIN-derived keyword + intent patterns | Keyword + Claude API |
| Objection signals | Hedge counting + resistance patterns | Custom + Claude API |
| Power language | Lakoff/O'Barr powerless features | Keyword regex |
| Question types | SPIN classification (Situation/Problem/Implication/Need-payoff) | Claude API |
| Intent | Per-utterance classification | Claude API |

**Key architecture decisions**:
- DistilBERT runs locally for sentiment (no API cost, ~50ms per sentence)
- Claude API used in batch mode for complex classifications (persuasion, intent, buying signals)
- Claude calls are batched: send 10-20 utterances per API call to reduce cost
- LIWC dictionary is a custom Python dict (no license needed for core categories)

---

### Agent 3: FACIAL AGENT 🔲 Phase 2
**Port**: 8003
**Input**: Video frames (JPEG/PNG) at 3 fps
**Output**: Redis stream `stream:facial:{session_id}`

| Feature | Method | Library |
|---------|--------|---------|
| 468 face landmarks | Face Mesh | MediaPipe |
| AU proxies | Distance ratios between landmarks | Custom from MediaPipe |
| 7-class emotion | CNN classification | DeepFace (FER model) |
| Valence-arousal | Continuous 2D space | AffectNet model |
| Duchenne smile | AU6 + AU12 co-occurrence | Custom from landmarks |
| Blink detection | Eye Aspect Ratio (EAR) | MediaPipe iris |

**Key limitations documented in rule engine**:
- Webcam angular error: 3-5° — all AU measurements have this margin
- Skin tone bias: DeepFace accuracy varies by ethnicity — documented mitigation
- No reliable micro-expression detection below 10 fps — rule disabled
- Lighting sensitivity: requires face visibility > 60% to produce signals

---

### Agent 4: BODY AGENT 🔲 Phase 2
**Port**: 8004
**Input**: Video frames at 1 fps (upper body sufficient)
**Output**: Redis stream `stream:body:{session_id}`

| Feature | Method | Library |
|---------|--------|---------|
| 33 pose landmarks | Holistic | MediaPipe |
| 21 hand landmarks | Hand tracking | MediaPipe |
| Posture score | Shoulder angle + head tilt | Custom geometry |
| Head movement | Nod/shake detection (pitch/yaw velocity) | Custom from landmarks |
| Body lean | Head size as distance proxy | Custom |
| Hand gestures | Visibility + movement classification | Custom from landmarks |
| Fidgeting | High-frequency low-amplitude movement | FFT on landmark series |
| Self-touch | Hand-face proximity detection | Distance calculation |

**Webcam limitations**: Only shoulders-up visible. No leg crossing, foot tapping, full torso lean. All rules designed for webcam-visible signals only.

---

### Agent 5: GAZE AGENT 🔲 Phase 2
**Port**: 8005
**Input**: Video frames at 1 fps + face landmarks from Facial Agent
**Output**: Redis stream `stream:gaze:{session_id}`

| Feature | Method | Library |
|---------|--------|---------|
| Iris position | Iris landmark tracking | MediaPipe |
| Head pose | 6DOF from landmarks | MediaPipe + solvePnP |
| Gaze direction | Iris + head pose combination | Custom |
| Screen engagement | On-screen vs off-screen classification | Custom (with dead zone) |
| Blink rate | EAR threshold crossing | Shared with Facial Agent |

**Critical design decision**: "Eye contact" is redefined as "screen engagement" because webcam angular error (3-5°) makes true gaze target determination unreliable. We classify: looking at screen (±15° cone) vs looking away.

---

### Agent 6: CONVERSATION AGENT 🔲 Phase 1 (Week 5)
**Port**: 8006
**Input**: All transcript segments + timestamps + speaker labels
**Output**: Redis stream `stream:conversation:{session_id}`

| Feature | Method |
|---------|--------|
| Turn-taking | Sacks et al. 1974 sequential model |
| Turn duration | Per-speaker per-turn timing |
| Response latency | Gap between speaker transitions (Stivers 2009: 200ms universal) |
| Overlap | Simultaneous speech detection |
| Interruption type | Cooperative vs competitive (Tannen classification) |
| Talk time ratio | Per-speaker cumulative |
| Topic changes | Semantic shift detection |

---

### Agent 7: FUSION AGENT 🔲 Phase 1 (Week 5)
**Port**: 8007
**Input**: All 6 agent streams via Redis Streams subscription
**Output**: Redis streams `stream:fusion:{session_id}` + `stream:alerts:{session_id}`

The Fusion Agent is the brain. It subscribes to all other agents' output streams and runs cross-modal analysis every 10-15 seconds.

**15 Pairwise Rules** (agent × agent):
| ID | Cross-Modal Pair | What It Detects |
|----|-----------------|-----------------|
| FUSION-01 | Voice tone × Facial expression | Emotional masking |
| FUSION-02 | Speech content × Voice stress | Credibility assessment |
| FUSION-03 | Body energy × Voice energy | Manufactured enthusiasm |
| FUSION-04 | Gaze break × Filler words | Uncertainty signal |
| FUSION-05 | Buying language × Body lean | Purchase intent validation |
| FUSION-06 | Micro-expression × Stated emotion | Emotional leakage |
| FUSION-07 | Head shake × Affirmative language | Unconscious disagreement |
| FUSION-08 | Gaze maintenance × Hedging language | False confidence |
| FUSION-09 | Smile × Negative sentiment | Sarcasm / social masking |
| FUSION-10 | Response latency × Facial processing | Cognitive load |
| FUSION-11 | Dominance language × Gaze avoidance | Anxiety-driven dominance |
| FUSION-12 | Interruption × Body posture | Interruption intent |
| FUSION-13 | Persuasion language × Speech pace | Urgency authenticity |
| FUSION-14 | Empathy language × Head nodding | Rapport validation |
| FUSION-15 | Filler spike × Gaze break | Sustained uncertainty |

**12 Compound Patterns** (3-6 domains simultaneously):
Genuine Engagement, Active Disengagement, Emotional Suppression, Decision Readiness, Cognitive Overload, Conflict Escalation, Silent Resistance, Rapport Peak, Topic Avoidance, Authentic Confidence, Anxiety Performance, Deception Risk.

**8 Temporal Sequences** (ordered cascades over time):
Stress Cascade (2-15s), Engagement Build (1-3min), Disengagement Cascade (30-120s), Objection Formation (5-30s), Trust Repair, Buying Decision Sequence, Dominance Shift, Authenticity Erosion (15-60min).

---

## Database Architecture

### PostgreSQL Tables (15 tables defined in `infrastructure/postgres/init/01-schema.sql`)

```
┌──────────────────────────────────────────────────────┐
│ CORE TABLES                                          │
│                                                      │
│ sessions           → Meeting metadata, participants  │
│ signals            → All agent outputs (partitioned) │
│ alerts             → Triggered alerts (4 tiers)      │
│ reports            → Generated analysis reports       │
│ transcripts        → Full transcript with timestamps  │
│ participants       → Per-session participant info     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ CONFIGURATION TABLES                                 │
│                                                      │
│ rule_config        → All 94 rule thresholds + weights│
│ fusion_weights     → Domain importance weights       │
│ meeting_types      → Sales/client/internal presets    │
│ display_profiles   → Cultural calibration settings   │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ INTELLIGENCE TABLES                                  │
│                                                      │
│ speaker_profiles   → Cross-session speaker memory    │
│ compound_patterns  → Detected compound pattern log   │
│ temporal_sequences → Detected temporal cascade log   │
│ research_embeddings→ pgvector: research paper chunks │
│ feedback_labels    → Human ground truth for training │
└──────────────────────────────────────────────────────┘
```

### Redis Streams Schema

Each agent publishes to its own stream. Message format:
```json
{
  "signal_id": "uuid",
  "agent": "voice",
  "speaker_id": "Speaker_0",
  "signal_type": "vocal_stress_score",
  "value": 0.67,
  "value_text": "elevated_stress",
  "confidence": 0.52,
  "window_start_ms": 45000,
  "window_end_ms": 50000,
  "metadata": { ... },
  "timestamp": "2025-03-15T10:30:00Z"
}
```

Streams are consumed by Fusion Agent using Redis consumer groups (exactly-once delivery).

### Storage Decision Matrix

| Data Type | Store | Why |
|-----------|-------|-----|
| Detection logic | Python code | Complex conditionals, not data |
| Thresholds & weights | PostgreSQL `rule_config` | Editable without deploy, feedback loop |
| Real-time signals | Redis Streams | Sub-ms pub/sub, ordered, consumer groups |
| Persisted signals | PostgreSQL `signals` | Queryable, indexable, joins |
| Research embeddings | pgvector in PostgreSQL | Similarity search, no extra DB |
| Meeting knowledge | Neo4j (Phase 3) | Entity relationships, graph queries |
| Speaker profiles | PostgreSQL + pgvector | Voice embeddings + metadata |
| Raw media files | Filesystem (MinIO later) | Large binary, no DB overhead |

---

## Communication Patterns

### Agent → Redis → Fusion Flow
```python
# Voice Agent publishes a signal:
await message_bus.publish_signal(
    stream=f"stream:voice:{session_id}",
    signal={
        "agent": "voice",
        "signal_type": "vocal_stress_score",
        "value": 0.67,
        "confidence": 0.52,
        ...
    }
)

# Fusion Agent subscribes and processes:
async for signal in message_bus.subscribe(
    streams=["stream:voice:*", "stream:language:*", ...],
    consumer_group="fusion",
    consumer_name="fusion-0"
):
    await fusion_engine.process_signal(signal)
```

### Fusion Temporal Alignment

The Fusion Agent uses **4 temporal window tiers**:
| Tier | Window | Use Case |
|------|--------|----------|
| Immediate | 0-2 seconds | Voice×Face congruence |
| Short | 2-10 seconds | Most pairwise rules |
| Medium | 10-60 seconds | Compound patterns |
| Long | 1-15 minutes | Temporal sequences |

Signals are aligned by `window_start_ms` / `window_end_ms` timestamps. The Fusion Agent maintains a sliding buffer of the last N minutes of signals per agent per speaker.

### Graceful Degradation

If an agent is unavailable (crashed, not deployed, or no video input):

| Available Agents | Behaviour |
|-----------------|-----------|
| Voice + Language + Conversation | Audio-only mode: 65% of insights. Weights redistribute. |
| Facial + Body + Gaze | Video-only mode: useful for silent observation tasks. |
| All 6 | Full analysis mode. |
| Only Voice | Minimal mode: stress + fillers + tone only. Still useful. |
| Only Language | Transcript-only mode: sentiment + intent + persuasion. |

The Fusion Agent's `graceful_degradation_matrix` automatically adjusts domain weights based on which agents are producing signals.

---

## Deployment Architecture

### Development (Current)
```
Docker Compose on local machine:
  - PostgreSQL 16 + pgvector (port 5432)
  - Valkey 8 (port 6379)
  - Voice Agent runs locally (port 8001)
```

### Staging (Phase 2+)
```
Single VM (8 CPU, 32GB RAM, NVIDIA T4):
  - Docker Compose with all services
  - GPU shared between Facial/Body/Gaze agents
  - All agents containerized
```

### Production (Phase 4+)
```
Kubernetes cluster:
  - Node pool 1: CPU nodes (API Gateway, Voice, Language, Conversation, Fusion)
  - Node pool 2: GPU nodes (Facial, Body, Gaze)
  - Managed PostgreSQL (RDS/CloudSQL)
  - Managed Redis (ElastiCache/MemoryStore)
  - S3/GCS for media storage
  - CloudFront/CDN for dashboard
```

---

## Security Considerations

- **Media files are ephemeral** — deleted after processing unless user opts into storage
- **Signals are de-identified** — speaker IDs are session-scoped UUIDs, not names
- **API authentication** — JWT tokens via OAuth 2.0 (Phase 4)
- **Data isolation** — PostgreSQL row-level security per tenant (Phase 4)
- **Encryption** — TLS in transit, AES-256 at rest for media files
- **GDPR compliance** — right-to-deletion API endpoint, configurable retention policies
- **No biometric storage** — voice/face embeddings are used for within-session matching only; cross-session matching requires explicit user consent

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Audio processing speed | <0.5x real-time | ~0.3x (Whisper medium, CPU) |
| Voice Agent latency | <2s per 5s window | ~1.5s |
| Language Agent latency | <1s per utterance | Not built |
| Facial Agent latency | <100ms per frame @ 3fps | Not built |
| Fusion cycle latency | <500ms per cycle | Not built |
| End-to-end (recorded file) | <3x real-time | ~2x (voice only) |
| End-to-end (live mode) | <15s behind real-time | Not built |
| Concurrent sessions | 10+ on single server | Not tested |
