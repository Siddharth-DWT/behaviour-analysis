# GETTING STARTED — NEXUS Development Setup

## Prerequisites

| Tool | Version | Required For |
|------|---------|-------------|
| Docker + Docker Compose | 24+ | Infrastructure (PostgreSQL, Valkey) |
| Python | 3.11+ | All agent services |
| Node.js | 20+ | Dashboard (Phase 1 Week 6) |
| ffmpeg | any | Audio processing in Voice Agent |
| Git | any | Version control |

## Quick Start (5 minutes)

```bash
# 1. Clone and enter project
cd nexus

# 2. Create environment file
cp .env.example .env
# Edit .env and add your Anthropic API key:
#   ANTHROPIC_API_KEY=sk-ant-...

# 3. Start infrastructure
docker compose up -d

# 4. Verify infrastructure is running
python scripts/health_check.py
# Should output:
#   ✅ PostgreSQL connected (pgvector enabled)
#   ✅ Redis connected
#   ✅ All environment variables set

# 5. Install Voice Agent dependencies
cd services/voice-agent
pip install -r requirements.txt
# Note: First time downloading Whisper model takes ~1.5GB

# 6. Run Voice Agent
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
# Visit http://localhost:8001/docs for Swagger UI

# 7. Test with a recording
curl -X POST http://localhost:8001/analyse \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/absolute/path/to/your/recording.wav"}'
```

## Project Structure

```
nexus/
├── CLAUDE.md              ← Master brain file (read first)
├── docs/
│   ├── PLAN.md            ← Full development roadmap
│   ├── ARCHITECTURE.md    ← System architecture & data flow
│   ├── RULES.md           ← All 94 rules quick reference
│   ├── UI.md              ← Dashboard design specification
│   ├── GETTING_STARTED.md ← You are here
│   └── STATUS.md          ← Current build status tracker
│
├── docker-compose.yml     ← PostgreSQL + Valkey
├── .env.example           ← Environment template
│
├── infrastructure/
│   ├── postgres/init/01-schema.sql  ← 15 tables auto-created
│   └── redis/valkey.conf            ← Streams-optimised config
│
├── services/
│   └── voice-agent/       ← ✅ Built and working
│       ├── main.py        ← FastAPI app (POST /analyse)
│       ├── feature_extractor.py  ← 25+ acoustic features
│       ├── calibration.py ← Per-speaker baselines
│       ├── rules.py       ← 5 core detection rules
│       ├── transcriber.py ← Whisper + diarisation
│       ├── Dockerfile
│       └── requirements.txt
│
├── shared/
│   ├── config/settings.py     ← Central config from env vars
│   ├── models/signals.py      ← Signal, Baseline, Alert dataclasses
│   └── utils/message_bus.py   ← Redis Streams pub/sub wrapper
│
├── scripts/
│   └── health_check.py   ← Infrastructure health verifier
│
└── data/
    ├── recordings/        ← Put audio/video files here
    ├── labels/            ← Human annotations (ground truth)
    └── reports/           ← Generated analysis reports
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|---------|---------|-------------|
| POSTGRES_HOST | Yes | localhost | PostgreSQL host |
| POSTGRES_PORT | Yes | 5432 | PostgreSQL port |
| POSTGRES_DB | Yes | nexus | Database name |
| POSTGRES_USER | Yes | nexus | Database user |
| POSTGRES_PASSWORD | Yes | nexus_dev_password | Database password |
| REDIS_HOST | Yes | localhost | Valkey/Redis host |
| REDIS_PORT | Yes | 6379 | Valkey/Redis port |
| ANTHROPIC_API_KEY | Yes* | (none) | For Language Agent + Fusion Agent |
| WHISPER_MODEL | No | medium | Whisper model size (tiny/base/small/medium/large-v3) |
| USE_PYANNOTE | No | false | Enable pyannote speaker diarisation |
| HF_TOKEN | No | (none) | HuggingFace token (if USE_PYANNOTE=true) |

*Not needed for Voice Agent alone. Required when Language Agent is built.

## Common Tasks

### Run Voice Agent analysis on a file
```bash
curl -X POST http://localhost:8001/analyse \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/home/user/recordings/sales-call.wav"}'
```

### Upload and analyse
```bash
curl -X POST http://localhost:8001/analyse/upload \
  -F "file=@/home/user/recordings/sales-call.wav"
```

### Check Voice Agent health
```bash
curl http://localhost:8001/health
```

### Connect to PostgreSQL directly
```bash
docker exec -it nexus-postgres psql -U nexus -d nexus
```

### View rule_config thresholds
```sql
SELECT rule_id, agent, threshold_json, weight, enabled
FROM rule_config
ORDER BY agent, rule_id;
```

### Update a threshold without code changes
```sql
UPDATE rule_config
SET threshold_json = jsonb_set(threshold_json, '{high}', '0.65')
WHERE rule_id = 'VOICE-STRESS-01';
```

### View Redis Streams
```bash
docker exec -it nexus-redis valkey-cli
> XLEN stream:voice:some-session-id
> XRANGE stream:voice:some-session-id - + COUNT 5
```

### Reset all data
```bash
docker compose down -v  # Deletes all data volumes
docker compose up -d    # Recreates fresh databases
```

## Supported Audio Formats

The Voice Agent uses ffmpeg + librosa, so it accepts any format ffmpeg can decode:
- WAV (recommended — no transcoding overhead)
- MP3
- M4A / AAC
- FLAC
- OGG
- WebM (audio track)

For best results: 16kHz mono WAV. Higher sample rates work but are downsampled to 16kHz internally.

## Next Steps After Setup

1. **Get real recordings**: 5-10 sales calls, 30-60 minutes each
2. **Run analysis**: Process each through the Voice Agent
3. **Label ground truth**: Manually annotate stress moments, fillers, tone shifts
4. **Compare**: Agent output vs human labels → tune thresholds
5. **Build Language Agent**: See PLAN.md Week 4 tasks
