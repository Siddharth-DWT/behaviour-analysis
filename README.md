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
- ✅ Language Agent (sentiment, buying signals, objections, power language, intent)
- ✅ Fusion Agent (3 pairwise rules, compound patterns, narrative reports)
- ✅ API Gateway (full pipeline orchestration)
- ✅ Dashboard (session list, detail, report, signal explorer)
- ✅ Authentication (JWT + bcrypt, login/signup, role-based access, session ownership)

See [docs/STATUS.md](docs/STATUS.md) for detailed build tracker.

## Quick Start

```bash
# Start databases
docker compose up -d

# Start all services
bash scripts/start_all.sh

# Open dashboard
open http://localhost:3000    # Sign up → upload a recording → view analysis

# Or use the API directly:
# 1. Create account
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"you@company.com","password":"Pass1234","full_name":"Your Name"}'

# 2. Use the returned access_token for authenticated requests
curl http://localhost:8000/sessions -H "Authorization: Bearer {token}"
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
| Authentication | JWT (python-jose) + bcrypt |
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
