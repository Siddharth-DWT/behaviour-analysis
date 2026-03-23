# NEXUS Processing Pipeline

> Complete trace of how audio/video files are processed, from upload to dashboard display.

## Architecture

```
User Upload → API Gateway → Voice Agent → Language Agent → Fusion Agent → Dashboard
                  │              │              │               │
                  │         Whisper STT    DistilBERT       Claude API
                  │         + Diarize     + Claude API      + Graph Engine
                  │              │              │               │
                  └──────────── PostgreSQL (signals, transcripts, reports) ──────────┘
```

## Pipeline Overview

| Phase | Service | What Happens | Duration |
|-------|---------|-------------|----------|
| 0 | API Gateway | Upload file, create session in DB | <1s |
| 1 | Voice Agent | Transcribe → Diarize → Extract features → Calibrate → Run rules | 30-150s |
| 2 | Language Agent | Sentiment → Buying/Objections → Power → Intent → Entities | 15-40s |
| 3 | API Gateway | Persist voice + language signals + transcript to DB | 2-5s |
| 4 | Fusion Agent | Cross-modal rules → Graph → Analytics → Narrative | 15-30s |
| 5 | API Gateway | Persist fusion signals + alerts + report to DB | 2-5s |
| **Total** | | | **65-230s** |

---

## Phase 0: Upload & Session Creation

**Endpoint:** `POST /sessions` (API Gateway)

```
Input:  file (audio/video), title, meeting_type, num_speakers
Output: SessionCreateResponse {session_id, status, signal counts}
```

1. Validate file type (WAV, MP3, M4A, FLAC, OGG, WebM, MP4)
2. Save to `data/recordings/{session_id}{extension}`
3. Create session in PostgreSQL: `status = "processing"`
4. Begin pipeline

---

## Phase 1: Voice Agent

**Endpoint:** `POST /analyse` on port 8001

### Step 1.1: Transcription

Two backends, auto-detected:

| Backend | When | Model | Speed |
|---------|------|-------|-------|
| **External GPU** | `EXTERNAL_WHISPER_URL` configured | large-v3 | ~4s for 1min audio |
| **Local CPU** | Fallback | medium | ~60-120s for 1min audio |

Both produce segments with word-level timestamps. A hallucination filter strips
Whisper repetition loops (5+ identical short segments).

### Step 1.2: Speaker Diarization

```
num_speakers provided?
  ├─ Yes → Use that count
  └─ No  → _estimate_speaker_count(segments)
              ├─ ≤3 segments total → 1 speaker
              ├─ ≤8 turn gaps → 2 speakers
              ├─ ≥12 turns + mostly short → 3 speakers
              └─ else → 2 speakers
           (but always ≥2 if segments > 3)

Then try in order:
  1. Acoustic KMeans (if audio available + speakers ≥ 2)
     - 5 features: [mean_pitch, mean_energy, spectral_centroid, zero_crossing_rate, pitch_std]
     - Temporal smoothing: fix micro-fragment speaker islands
     - Balance check: dominant speaker ≤ (1/N + 40%)
  2. Gap+Pitch fallback (2 speakers)
     - Real F0 pitch per segment via librosa.pyin
     - Speaker change on gap > 800ms OR pitch jump > 25Hz
  3. Heuristic multi-speaker (3+ speakers)
     - Gap-based turn detection + round-robin assignment
  4. Pyannote (if USE_PYANNOTE=true + HF_TOKEN)
     - Full neural diarization via pyannote/speaker-diarization-3.1
```

### Step 1.3: Feature Extraction

Per speaker, per 5-second window (2.5s hop = 50% overlap):

| Category | Features |
|----------|----------|
| **Pitch (F0)** | mean, std, variance, max, min, range, voiced_fraction |
| **Energy** | rms_db, std_db, max_db, min_db, dynamic_range_db |
| **Speech Rate** | wpm, word_count, speaking_time_sec, onset_rate |
| **Voice Quality** | jitter_pct, shimmer_pct, hnr_db |
| **Spectral** | spectral_centroid_hz |
| **Pauses** | count, total_ms, avg_ms, max_ms, pause_ratio |
| **Fillers** | count, rate_pct, um_count, uh_count, like_count |

### Step 1.4: Calibration

Uses first ~90 seconds of each speaker's speech to build a baseline:
- Mean pitch, rate, energy, jitter, shimmer, HNR, filler rate
- `calibration_confidence` multiplier (0.1-0.9) based on data available

**All detection is DEVIATION from baseline, not absolute values.**

### Step 1.5: Voice Rules (5)

| Rule | What It Detects | Signal Type |
|------|----------------|-------------|
| VOICE-STRESS-01 | Composite stress from 7 weighted features | `vocal_stress_score` (0-1) |
| VOICE-FILLER-01 | Filler rate spike vs baseline | `filler_detection` |
| VOICE-PITCH-01 | Pitch elevation > 8% from baseline | `pitch_elevation_flag` |
| VOICE-RATE-01 | Speech rate anomaly > 25% from baseline | `speech_rate_anomaly` |
| VOICE-TONE-03/04 | Nervous vs Confident tone classification | `tone_classification` |

**Output:** ~50-200 voice signals per session

---

## Phase 2: Language Agent

**Endpoint:** `POST /analyse` on port 8002

### Step 2.1: Feature Extraction

Per transcript segment:
- **Sentiment**: LLM (primary) → VADER (fallback) → DistilBERT (last resort)
- **Buying signals**: 12 SPIN Selling keyword pattern categories
- **Objections**: Direct, timing, authority, competitor patterns + hedge clusters
- **Power language**: Lakoff/O'Barr powerless features (hedges, tag questions, etc.)

### Step 2.2: Language Rules (5)

| Rule | What It Detects | Signal Type | Content Types |
|------|----------------|-------------|---------------|
| LANG-SENT-01 | Sentiment intensity | `sentiment_score` (-1 to +1) | All |
| LANG-BUY-01 | Buying signals (SPIN) | `buying_signal` | **sales_call only** |
| LANG-OBJ-01 | Objections | `objection_signal` | **sales_call only** |
| LANG-PWR-01 | Power/Powerless speech | `power_language_score` (0-1) | All |
| LANG-INTENT-01 | Intent classification (12 types) | `intent_classification` | All |

### Step 2.3: Entity Extraction (LLM)

Single LLM call with full transcript. Extracts:
- **People**: names, roles, speaker labels
- **Companies**: names, context
- **Topics**: conversation phases with time ranges (3-7 phases)
- **Objections**: exact quotes with resolved/unresolved status
- **Commitments**: action items with speaker and timestamp
- **Key terms**: important words/phrases

**Output:** ~100-500 language signals + entities dict

---

## Phase 3: Fusion Agent

**Endpoint:** `POST /analyse` on port 8007

### Step 3.1: Signal Graph Construction

Builds an in-memory graph from all signals:

**Nodes:** speakers, voice signals, language signals, fusion signals, topics, moments

**Edges:**
| Edge Type | Meaning |
|-----------|---------|
| `speaker_produced` | Speaker → Signal they generated |
| `co_occurred` | Two signals within 10s window |
| `triggered` | Voice + Language signals that caused a fusion insight |
| `contradicts` | High stress + positive sentiment (incongruence) |
| `preceded` | Temporal ordering within 30s |
| `about_topic` | Signal during a conversation topic phase |
| `resolved` | Objection → earliest subsequent buying signal |

### Step 3.2: Graph Analytics (6 analyses)

| Analysis | What It Computes |
|----------|-----------------|
| **Tension Clusters** | Groups of 3+ negative signals in 10s buckets |
| **Speaker Patterns** | Signal density, contradiction ratio, response pattern, escalation trend |
| **Topic Signal Density** | Per-topic risk level and opportunity level |
| **Momentum** | Quartile trajectory (positive/negative/stable/volatile), turning point |
| **Resolution Paths** | Objection → buying signal resolution chains |
| **Incongruence Patterns** | Per-speaker contradicts edge consistency |

### Step 3.3: Pairwise Fusion Rules (3)

| Rule | Cross-Modal Check | Signal Type | Max Confidence |
|------|-------------------|-------------|----------------|
| FUSION-02 | Sentiment + Stress | `credibility_assessment` | **0.55** (deception cap) |
| FUSION-07 | Power + Sentiment | `verbal_incongruence` | 0.70 |
| FUSION-13 | Persuasion + Rate | `urgency_authenticity` | 0.60 |

### Step 3.4: Graph-Based Fusion Rules (3)

| Rule | What It Detects | Signal Type |
|------|----------------|-------------|
| FUSION-GRAPH-01 | Tension clusters (3+ signals) | `tension_cluster` |
| FUSION-GRAPH-02 | Momentum shift detected | `momentum_shift` |
| FUSION-GRAPH-03 | Persistent incongruence (3+ windows) | `persistent_incongruence` |

### Step 3.5: Narrative Report (LLM)

Single LLM call with full context:
- Voice summary (stress peaks, tones, fillers)
- Language summary (sentiment, buying signals, objections)
- Fusion signals (cross-modal detections)
- Entities (people, topics, commitments)
- Graph analytics (tension clusters, momentum, speaker patterns)

**Output:** Executive summary, key moments, cross-modal insights, recommendations

---

## Content-Type Differences

### Sales Call (`sales_call`)

```
Voice Agent:  All 5 rules (same for all types)
Language Agent:
  ├─ LANG-SENT-01: ✅ Sentiment
  ├─ LANG-BUY-01: ✅ Buying signals (ONLY for sales_call)
  ├─ LANG-OBJ-01: ✅ Objection signals (ONLY for sales_call)
  ├─ LANG-PWR-01: ✅ Power language
  └─ LANG-INTENT-01: ✅ Intent classification
Entity Extraction:
  └─ Extracts: objections, buying_signals, commitments, sales_stages
Dashboard (Insights Tab):
  ├─ Deal Progression: Opening → Qualification → Objection → Pitch → Closing
  ├─ Objections list with ✅/❌ resolved status
  ├─ Buying signals count
  ├─ Commitments list
  └─ Speaker roles: Seller / Prospect (auto-detected from transcript)
```

### Internal Meeting (`internal`)

```
Voice Agent:  All 5 rules (same)
Language Agent:
  ├─ LANG-SENT-01: ✅ Sentiment
  ├─ LANG-BUY-01: ❌ SKIPPED (not a sales call)
  ├─ LANG-OBJ-01: ❌ SKIPPED (not a sales call)
  ├─ LANG-PWR-01: ✅ Power language
  └─ LANG-INTENT-01: ✅ Intent classification
Entity Extraction:
  └─ Extracts: action_items, decisions, key_terms, topics
Dashboard (Insights Tab):
  ├─ Participation Balance: stacked bar (% talk time per speaker)
  ├─ Dominance warning if any speaker > 70%
  ├─ Action Items & Commitments list
  └─ Speaker roles: Facilitator / Participant
```

### Interview (`interview`)

```
Voice Agent:  All 5 rules (same)
Language Agent:
  ├─ LANG-SENT-01: ✅ Sentiment
  ├─ LANG-BUY-01: ❌ SKIPPED
  ├─ LANG-OBJ-01: ❌ SKIPPED
  ├─ LANG-PWR-01: ✅ Power language
  └─ LANG-INTENT-01: ✅ Intent classification
Entity Extraction:
  └─ Extracts: questions_asked, candidate_strengths, candidate_concerns
Dashboard (Insights Tab):
  ├─ Candidate Assessment: confidence %, stress level
  ├─ Key Terms chips
  └─ Speaker roles: Interviewer / Candidate
```

### Client Meeting (`client_meeting`)

```
Voice Agent:  All 5 rules (same)
Language Agent:
  ├─ LANG-SENT-01: ✅ Sentiment
  ├─ LANG-BUY-01: ❌ SKIPPED
  ├─ LANG-OBJ-01: ❌ SKIPPED
  ├─ LANG-PWR-01: ✅ Power language
  └─ LANG-INTENT-01: ✅ Intent classification
Entity Extraction:
  └─ Extracts: action_items, decisions, satisfaction_indicators, risk_flags
Dashboard (Insights Tab):
  ├─ Participation Balance
  ├─ Action Items & Commitments
  └─ Speaker roles: Account Manager / Client
```

### What's THE SAME across all types

- Voice Agent runs identically (all 5 rules, same features, same calibration)
- Fusion Agent runs identically (all 3 pairwise rules, all 3 graph rules)
- Signal graph + graph analytics run identically
- Narrative report generation uses same LLM with `meeting_type` in prompt
- Stress, pitch, rate, tone, filler detection — all type-agnostic

### What DIFFERS by type

| Feature | sales_call | internal | interview | client_meeting |
|---------|-----------|----------|-----------|----------------|
| Buying signal detection | ✅ | ❌ | ❌ | ❌ |
| Objection detection | ✅ | ❌ | ❌ | ❌ |
| Deal Progression UI | ✅ | ❌ | ❌ | ❌ |
| Participation Balance UI | ❌ | ✅ | ❌ | ✅ |
| Candidate Assessment UI | ❌ | ❌ | ✅ | ❌ |
| Speaker role labels | Seller/Prospect | Facilitator/Participant | Interviewer/Candidate | Acct Mgr/Client |
| Entity fields | objections, buying, commitments | action_items, decisions | questions, strengths, concerns | action_items, decisions, risks |

---

## LLM API Calls Per Session

| Call | Agent | Batch Size | Calls for 30-segment session | Calls for 300-segment session |
|------|-------|-----------|------------------------------|-------------------------------|
| Sentiment | Language | 12 segments | 3 | 25 |
| Intent | Language | 15 segments | 2 | 20 |
| Entities | Language | Full transcript | 1 | 1 |
| Narrative | Fusion | Full context | 1 | 1 |
| **Total** | | | **7 calls** | **47 calls** |

---

## Database Tables Used

| Table | Written By | Read By |
|-------|-----------|---------|
| `sessions` | API Gateway | Dashboard |
| `speakers` | API Gateway (from Voice result) | All queries (JOIN) |
| `transcript_segments` | API Gateway | Dashboard transcript view |
| `signals` | API Gateway (voice + language + fusion) | Dashboard, Signal Explorer |
| `alerts` | API Gateway (from Fusion result) | Dashboard alerts section |
| `session_reports` | API Gateway (from Fusion narrative) | Dashboard report tab |

---

## Processing Time Breakdown

| Step | Short call (1 min) | Medium call (5 min) | Long call (15 min) |
|------|--------------------|--------------------|--------------------|
| Transcription (GPU) | 4s | 15s | 45s |
| Transcription (CPU) | 30s | 120s | 360s |
| Diarization | 3s | 10s | 25s |
| Voice features | 5s | 20s | 60s |
| Voice rules | 1s | 3s | 8s |
| Language features | 2s | 5s | 10s |
| LLM sentiment | 5s | 15s | 40s |
| LLM intent | 5s | 15s | 35s |
| LLM entities | 5s | 8s | 12s |
| Fusion rules | 1s | 3s | 5s |
| Graph + analytics | 1s | 3s | 8s |
| LLM narrative | 8s | 10s | 15s |
| DB persistence | 2s | 3s | 5s |
| **Total (GPU)** | **42s** | **130s** | **328s** |
| **Total (CPU)** | **68s** | **235s** | **583s** |

---

## Error Handling & Fallbacks

| Component | Failure | Fallback |
|-----------|---------|----------|
| External Whisper | API unreachable | Auto-switch to local faster-whisper |
| KMeans diarization | Too few features | Gap+pitch heuristic |
| LLM sentiment | API error | VADER → DistilBERT |
| LLM entity extraction | API error | Lightweight keyword extraction |
| LLM narrative | API error | Basic template narrative (no LLM) |
| PostgreSQL | Connection failed | Pipeline continues, warns in logs |
| Pyannote | Token/model error | Falls back to KMeans → gap heuristic |

---

## Confidence Caps (from RULES.md)

- **Single-domain signal max:** 0.85
- **Deception-adjacent (FUSION-02):** 0.55 (hard cap by design)
- **Never claim certainty:** All signals are probabilistic indicators
- **Cluster rule:** Reliable interpretation requires 3+ congruent signals from different domains
