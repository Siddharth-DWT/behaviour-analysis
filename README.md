# NEXUS — Multi-Agent Behavioural Analysis System

> Real-time call and meeting analysis across 6 behavioural domains.
> Detects what humans miss: the gap between what people say and what they mean.

## What NEXUS Does

NEXUS analyses audio and video from calls and meetings through **7 parallel AI agents**, each specialising in a different behavioural domain. The real power comes from **cross-modal fusion** — detecting incongruence between modalities that no single channel can reveal.

| Agent | Domain | Analyses |
|-------|--------|----------|
| Voice | Acoustic/prosodic | Stress, tone, pitch, pace, fillers, pauses |
| Language | Linguistic content | Sentiment, persuasion, buying signals, objections, power language |
| Facial | Facial expression | Emotions, Duchenne smile, micro-expressions, engagement |
| Body | Body language | Posture, gestures, fidgeting, self-touch, mirroring |
| Gaze | Eye behaviour | Screen engagement, blink rate, attention, distraction |
| Conversation | Dialogue dynamics | Turn-taking, latency, dominance, rapport, interruptions |
| **Fusion** | **Cross-modal** | **15 pairwise rules + 12 compound patterns + 8 temporal sequences** |

**Example insight**: A buyer says "Sounds great" (positive language) while their voice shows stress elevation, body leans backward, and they do a subtle head shake. Single-domain analysis sees agreement. NEXUS sees **Silent Resistance** — a "yes" that means "no."

## Current Status

**Phase 0 complete. Phase 1 in progress.**

- ✅ 94 detection rules designed across all 7 agents (research-backed)
- ✅ Infrastructure running (PostgreSQL + pgvector + Valkey)
- ✅ Voice Agent built and working (5 core rules)
- 🔲 Language Agent (next)
- 🔲 Fusion Agent
- 🔲 Dashboard

See [docs/STATUS.md](docs/STATUS.md) for detailed build tracker.

## Quick Start

```bash
# Start databases
docker compose up -d

# Run Voice Agent
cd services/voice-agent
pip install -r requirements.txt
uvicorn main:app --port 8001 --reload

# Analyse a recording
curl -X POST http://localhost:8001/analyse \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/recording.wav"}'
```

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for full setup instructions.

## Documentation

| Document | What It Covers |
|----------|---------------|
| [CLAUDE.md](CLAUDE.md) | Master brain file — read first when opening project in Claude Code |
| [docs/PLAN.md](docs/PLAN.md) | Full 5-phase development roadmap (40-48 weeks) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, data flow, agent specs, database design |
| [docs/RULES.md](docs/RULES.md) | All 94 detection rules — quick reference with cross-references |
| [docs/UI.md](docs/UI.md) | Dashboard design spec — all views, components, interactions |
| [docs/STATUS.md](docs/STATUS.md) | Current build status with checkboxes |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Development setup guide |

## Tech Stack

All open source, $0 licensing cost.

| Layer | Technology |
|-------|-----------|
| Database | PostgreSQL 16 + pgvector |
| Message Bus | Valkey 8 (Redis Streams) |
| Agent Services | Python 3.11 + FastAPI |
| Audio Processing | librosa + faster-whisper |
| NLP | DistilBERT + Claude API |
| Computer Vision | MediaPipe + DeepFace (Phase 2) |
| Dashboard | React + Tailwind + Recharts |
| Knowledge Graph | Neo4j Community (Phase 3) |
| Live Calls | Recall.ai (Phase 4) |

## Architecture

```
Voice ─────────┐
Language ──────┤
Facial ────────┤──→ Redis Streams ──→ FUSION AGENT ──→ Dashboard
Body ──────────┤                     (cross-modal     (real-time
Gaze ──────────┤                      analysis)        insights)
Conversation ──┘
```

Each agent is an independent microservice. Agents communicate via Redis Streams. The Fusion Agent subscribes to all streams and performs cross-modal congruence checking.

## Key Design Principles

1. **Baseline-relative**: Every signal is a deviation from per-speaker baseline, not absolute
2. **Cluster rule**: Single signals don't produce high confidence — require 3+ congruent signals
3. **Never claim certainty**: Max confidence 0.85. Deception signals capped at 0.55
4. **Configurable**: All thresholds in database, not hardcoded
5. **Graceful degradation**: Audio-only mode works with 65% of full insight value

## License

Proprietary. All rights reserved.
