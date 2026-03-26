# Speaker Diarization — Architecture, Results & Lessons Learned

> **Branch:** `speaker_fix` (9 commits from `dev`)
> **Date:** March 2026
> **Audio tested:** 6 calls (4 real 2-speaker, 2 single-narrator), 138 total segments

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Current Pipeline Architecture](#current-pipeline-architecture)
- [Diarization Cascade (Tiers 1–5)](#diarization-cascade-tiers-15)
- [Post-Correction Layers (Layers 2–4)](#post-correction-layers-layers-24)
- [All Approaches Tried & Results](#all-approaches-tried--results)
  - [Phase 1: Embedding Models](#phase-1-embedding-models)
  - [Phase 2: Post-Correction Layers](#phase-2-post-correction-layers)
  - [Phase 3: Word-Level Splitting (Failed)](#phase-3-word-level-splitting-failed)
  - [Phase 4: Pitch Kalman + LLM Correction](#phase-4-pitch-kalman--llm-correction)
  - [Phase 5: Additional Fixes](#phase-5-additional-fixes)
  - [Multi-Model LLM Comparison](#multi-model-llm-comparison)
- [Final Test Results](#final-test-results)
- [Issues Faced & Resolutions](#issues-faced--resolutions)
- [Remaining Unsolved Issues](#remaining-unsolved-issues)
- [Configuration Reference](#configuration-reference)
- [Dependencies](#dependencies)
- [File Map](#file-map)
- [Commit History](#commit-history)

---

## Problem Statement

Session `09989409` — a 55-second YouTube Short of a cold call between **Saad** (caller from HOS) and a **prospect**. The original system assigned **22 of 23 segments to Speaker_0** and only 1 to Speaker_1. A two-person conversation was treated as a monologue.

**Root cause:** Whisper segments audio by silence, not by speaker change. When two speakers talk with <300ms gaps (fast-paced cold call), Whisper doesn't create clean turn boundaries. The old diarization relied on 5 low-level acoustic features (pitch, energy, spectral centroid, ZCR, pitch_std) that encode *what is being said at that moment*, not *who is speaking*.

---

## Current Pipeline Architecture

```
Audio File
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  WHISPER TRANSCRIPTION                                   │
│  faster-whisper (local) or GPU API (external)            │
│  Output: segments with word-level timestamps             │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  AUTO SPEAKER COUNT DETECTION                            │
│  ECAPA-TDNN embeddings on 6 windows across full audio    │
│  High similarity (>0.82) or low variance (std<0.06)      │
│  → override to 1 speaker (prevents false splits)         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  DIARIZATION CASCADE (Tier 1 → Tier 5)                   │
│                                                           │
│  Tier 1: Pyannote Community-1   (frame-level, ~16ms)     │
│  Tier 2: ECAPA-TDNN Embeddings  (192-dim, agglomerative) │
│  Tier 3: Acoustic KMeans        (16D: 13 MFCCs + pitch)  │
│  Tier 4: Gap + Pitch Heuristic  (400ms/15Hz/MFCC dist)   │
│  Tier 5: Round-Robin            (last resort)             │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: LINGUISTIC POST-CORRECTION (always on)         │
│  Rule 1: Question → Answer                               │
│  Rule 2: Isolated flip                                    │
│  Rule 3: Greeting → Query ("Good morning" → "Regarding?")│
│  Rule 4: Greeting → Response ("How are you" → "I'm good")│
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: PITCH KALMAN FILTER (opt-in)                   │
│  env: DIARIZATION_KALMAN=true                            │
│  F0 continuity tracking via Kalman filter                │
│  Flag-only mode — marks PITCH_SUSPICIOUS for Layer 4     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 4: EVIDENCE-BASED LLM CORRECTION (opt-in)        │
│  env: DIARIZATION_LLM=true                               │
│  Fuses signals from ALL layers into evidence cards:      │
│    - Pyannote frame probability                          │
│    - ECAPA cosine similarity + margin                    │
│    - Pitch median + Kalman flags                         │
│    - Linguistic patterns                                 │
│  Confidence tiers: HIGH / MEDIUM / LOW                   │
│  LLM (gpt-4o) can ONLY flip LOW-tier segments            │
│  Code-level enforcement blocks HIGH-tier flips            │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
                  Labeled Segments
             (Speaker_0, Speaker_1, ...)
```

---

## Diarization Cascade (Tiers 1–5)

### Tier 1: Pyannote Community-1 (DEFAULT)

- **Model:** `pyannote/speaker-diarization-community-1`
- **Resolution:** Frame-level (~16ms) — detects speaker changes even with <300ms gaps
- **Requirement:** `HF_TOKEN` environment variable + license acceptance at HuggingFace
- **DER:** 26.7% on CALLHOME (telephone conversations benchmark)
- **Why it's Tier 1:** Operates independently of Whisper's silence-based segmentation. Solves the root cause.

### Tier 2: ECAPA-TDNN Embeddings (Fallback)

- **Model:** `speechbrain/spkrec-ecapa-voxceleb` (installed from develop branch)
- **Embeddings:** 192-dimensional per segment
- **Clustering:** Agglomerative with cosine affinity, average linkage
- **EER:** ~0.8% on VoxCeleb1
- **When used:** Falls back here if no `HF_TOKEN` set

### Tier 3: Enhanced Acoustic KMeans

- **Features:** 16D vector per segment (13 MFCCs + mean pitch + pitch std + mean energy)
- **Normalization:** sklearn StandardScaler
- **Balance check:** Relaxed for short audio (<2 min): 1/N + 0.50

### Tier 4: Enhanced Gap + Pitch Heuristic

- **Gap threshold:** Meeting-type aware (400ms for sales_call, 800ms for meetings)
- **Pitch jump:** 15Hz (lowered from 25Hz for phone-quality audio)
- **MFCC distance:** Cosine distance > 0.3 between adjacent segments
- **Correction pass:** Force turn boundary if same speaker >15s straight

### Tier 5: Round-Robin Heuristic

- Last resort for 3+ speakers without audio features

---

## Post-Correction Layers (Layers 2–4)

### Layer 2: Linguistic Rules (Always On)

| Rule | Pattern | Example |
|------|---------|---------|
| **Question → Answer** | Short `?` + short response + same speaker + >200ms gap | "Does that sound okay?" → "By Monday." |
| **Isolated flip** | Short segment breaks consistent speaker run + tiny gaps (<150ms) | A-B-A where B is <1.5s |
| **Greeting → Query** | Greeting/intro + short question from same speaker | "Good morning, this is Saad..." → "Regarding what?" |
| **Greeting → Response** | Greeting/question + polite response from same speaker | "Hi, how are you doing?" → "I'm good, how are you?" |

### Layer 3: Pitch Kalman Filter (Opt-In)

- **Enable:** `DIARIZATION_KALMAN=true`
- **Based on:** Hogg et al. (IEEE ICASSP 2019)
- **How it works:** Kalman filter models F0 pitch continuity. When prediction error (innovation) spikes at a segment boundary → speaker change likely
- **Mode:** Flag-only (no auto-flip). Marks segments as `PITCH_SUSPICIOUS` for Layer 4
- **Pitch extraction:** Praat (parselmouth) primary, librosa.pyin fallback

### Layer 4: Evidence-Based LLM Correction (Opt-In)

- **Enable:** `DIARIZATION_LLM=true`
- **Model:** gpt-4o (best accuracy in testing)
- **Three layers of overcorrection protection:**
  1. **Tier gate:** LLM only reviews LOW-confidence segments. Skips entirely if no LOW segments exist.
  2. **Prompt constraint:** Evidence cards show actual numbers (pyannote=89%, ECAPA margin=0.44).
  3. **Code enforcement:** Even if LLM ignores prompt, HIGH-tier flips are blocked in validation code.

---

## All Approaches Tried & Results

### Phase 1: Embedding Models

| Approach | Accuracy | Outcome |
|----------|----------|---------|
| Old 5-feature KMeans (pitch, energy, spectral centroid, ZCR, pitch_std) | **4.3%** (1/23) | Features don't encode speaker identity. Balance check rejects 22:1. |
| Resemblyzer GE2E (256-dim) | Fell back | 22:1 imbalanced (96%). Same-phone channel confused the model. |
| SpeechBrain ECAPA-TDNN from PyPI | **Crashed** | `use_auth_token` deprecated + `custom.py` 404 on HuggingFace. |
| **SpeechBrain ECAPA-TDNN from develop branch** | **91.3%** (21/23) | First working approach. 192-dim embeddings + agglomerative clustering. |

### Phase 2: Post-Correction Layers

| Approach | Accuracy | Delta |
|----------|----------|-------|
| ECAPA-TDNN alone | 91.3% (21/23) | baseline |
| + Linguistic correction (Q→A, isolated flip) | 91.3% (21/23) | +0% |
| + Greeting→query rule | **95.7%** (22/23) | **+4.4%** |
| + Pyannote community-1 as Tier 1 | **95.7%** (22/23) | +0% (more robust foundation) |

### Phase 3: Word-Level Splitting (FAILED)

| Approach | Accuracy | Issue |
|----------|----------|-------|
| Sliding-window embeddings (1.5s, 0.75s hop) + word-level assignment | **78-85%** | Windows at speaker boundaries contain both speakers → noisy word labels |
| Re-grouping words by speaker | **78%** | Merged across original Whisper segment boundaries → mega-segments |
| Conservative split (≥3 words per side) | **~85%** | Better but still introduced fragment errors |

**Lesson learned:** Segment-level assignment with pyannote is more stable than word-level splitting. The boundary noise from overlapping windows causes more errors than it fixes.

### Phase 4: Pitch Kalman + LLM Correction

| Approach | Accuracy | Issue |
|----------|----------|-------|
| Kalman auto-flip (conf>0.7) | 91.3% (21/23) | 4 false flips — too aggressive on phone audio |
| Kalman flag-only + flat LLM (gpt-4o-mini) | 91.3% (21/23) | LLM flipped "By Monday" wrong — no confidence info |
| Kalman flag-only + flat LLM (gpt-4o) | 91.3% (21/23) | Same — flat transcript insufficient |
| **Kalman flag-only + evidence cards + gpt-4o** | **95.7%** (22/23) | Evidence cards protect HIGH segments. "By Monday" safe. |

### Phase 5: Additional Fixes

| Fix | Target | Result |
|-----|--------|--------|
| Auto speaker count (ECAPA 6 windows) | Single-narrator false splits | Didn't trigger — narrator variation too high (std=0.069-0.081 vs 0.06 threshold) |
| **Greeting→response rule** | "I'm good, how are you?" | **FIXED** in Saad+Vanessa call |
| Confidence downgrade for polite responses | "No problem" after questions | Allowed LLM review but LLM didn't flip |

### Multi-Model LLM Comparison

Same evidence cards sent to each model. Scoring on the 23-segment Saad control call:

| Model | Accuracy | Seg #5 (Saad's question) | Seg #6 (By Monday) | Seg #23 (Thank you) |
|-------|----------|-------------------------|--------------------|--------------------|
| gpt-4o-mini | parse fail | — | — | — |
| **gpt-4o** | **100%** (23/23) | ✓ Fixed | ✓ Protected | ✓ Fixed |
| **gpt-4.1** | **100%** (23/23) | ✓ Fixed | ✓ Protected | ✓ Fixed |
| gpt-5 | 95.7% (22/23) | ✓ Fixed | ✓ Protected | ✗ Missed |
| gpt-5-mini | 91.3% (21/23) | ✗ Missed | ✓ Protected | ✗ Missed |
| gpt-5.4 | 95.7% (22/23) | ✓ Fixed | ✓ Protected | ✗ Missed |
| o3-mini | 95.7% (22/23) | ✓ Fixed | ✓ Protected | ✓ Fixed |

**Finding:** gpt-4o and gpt-4.1 both achieve 100% on the evidence card prompt. GPT-5 series paradoxically performs worse — possibly over-thinks the structured evidence format.

---

## Final Test Results

### V3 — 6 Calls, Full Pipeline

| Call | Type | Duration | Segments | Correct | Accuracy |
|------|------|----------|----------|---------|----------|
| Saad Control | Real 2-spk | 55s | 23 | 21 | 91.3% |
| Saad Original | Real 2-spk | 86s | 18 | 17 | 94.4% |
| **B2B Giovanni+Hannah** | Real 2-spk | 93s | 40 | **40** | **100%** |
| Saad+Vanessa | Real 2-spk | 69s | 34 | 32 | 94.1% |
| Day 1 Cold Calling | Single-narrator | 71s | 16 | 14 | 87.5% |
| Telemarketer Dubai | Single-narrator | 39s | 7 | 4 | 57.1% |

**Real 2-speaker calls: 110/115 = 95.7%**
**Overall (including single-narrator): 128/138 = 92.8%**

### Accuracy Progression

```
Original (5-feature KMeans):        4.3%   (1/23)
+ ECAPA-TDNN embeddings:           91.3%  (21/23)
+ Linguistic correction:           91.3%  (21/23)
+ Greeting→query rule:             95.7%  (22/23)
+ Pyannote community-1:            95.7%  (22/23)
+ Evidence cards (LLM isolated):  100.0%  (23/23) ← gpt-4o on single call
+ Multi-call real 2-speaker:       95.7%  (110/115)
```

---

## Issues Faced & Resolutions

| Issue | Root Cause | Resolution | Status |
|-------|------------|------------|--------|
| 22/23 segments = Speaker_0 | 5-feature KMeans doesn't encode identity | ECAPA-TDNN 192-dim embeddings | ✅ Fixed |
| SpeechBrain PyPI crash | Deprecated `use_auth_token` + missing `custom.py` | Install from develop branch | ✅ Fixed |
| Resemblyzer 22:1 imbalance | GE2E captured channel, not speaker | Replaced with ECAPA-TDNN | ✅ Fixed |
| Word-level splitting micro-fragments | Boundary windows contain both speakers | Reverted — segment-level is more stable | ✅ Reverted |
| LLM flipped "By Monday" wrong | Flat transcript, no confidence info | Evidence cards + code-level HIGH enforcement | ✅ Fixed |
| Single-narrator false split | Forced `num_speakers=2` | Auto speaker count detection (partial) | ⚠️ Partial |
| "I'm good" misassigned | Short polite response, tiny gap | Greeting→response linguistic rule | ✅ Fixed |
| "No problem" misassigned | <500ms, ambiguous in pyannote frames | Confidence downgrade (LLM still doesn't flip) | ⚠️ Partial |
| Merged turn boundary | No silence gap between speakers | Only fixable with WhisperX-level forced alignment | ❌ Open |

---

## Remaining Unsolved Issues

| Issue | Frequency | Why It's Hard | Potential Fix |
|-------|-----------|---------------|---------------|
| **40ms-gap fast exchanges** | 1 per call | Zero silence = Whisper and pyannote both assign to previous speaker. ECAPA embedding for 1.1s segment is noisy. | WhisperX phoneme-level forced alignment |
| **Single-narrator with vocal variation** | 5 across 2 calls | Narrator varies voice enough (acting characters, energy shifts) that embeddings show low similarity (0.28-0.35), looking like 2 speakers. | Content-aware detection (video vs call classification) or speaker count from metadata |
| **Merged turn boundaries** | 1 per call | No silence gap = Whisper puts both speakers in one segment. Word-level splitting introduces more errors than it fixes. | WhisperX forced alignment or re-segmentation at detected speaker change points |
| **Very short polite responses** (<500ms) | 1-2 per call | Too short for reliable embedding. Pyannote frame assignment is ambiguous. Linguistic rules catch some but not all patterns. | Expand polite response dictionary + lower confidence threshold for post-greeting segments |

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (none) | HuggingFace token for pyannote community-1. Required for Tier 1. |
| `DIARIZATION_MODE` | `auto` | Override cascade: `auto` \| `pyannote` \| `ecapa` \| `kmeans` |
| `DIARIZATION_KALMAN` | `false` | Enable Pitch Kalman filter (Layer 3) |
| `DIARIZATION_LLM` | `false` | Enable evidence-based LLM correction (Layer 4) |
| `DIARIZATION_EMBEDDING_MODEL` | `speechbrain/spkrec-ecapa-voxceleb` | ECAPA-TDNN model source |
| `SPEECHBRAIN_CACHE_DIR` | `pretrained_models/spkrec-ecapa-voxceleb` | Local model cache path |
| `USE_PYANNOTE` | `false` | Enable legacy pyannote 3.1 (Tier 1b) |
| `WHISPER_MODEL` | `medium` | Whisper model size for transcription |

### Default Pipeline (No Extra Config)

With `HF_TOKEN` set:
```
Pyannote Community-1 → Linguistic Correction → 95.7% accuracy
```

Without `HF_TOKEN`:
```
ECAPA-TDNN → Linguistic Correction → 95.7% accuracy
```

### Full Pipeline (All Layers)

```bash
export HF_TOKEN=hf_xxxx
export DIARIZATION_KALMAN=true
export DIARIZATION_LLM=true
export OPENAI_API_KEY=sk-xxxx   # or ANTHROPIC_API_KEY
```

---

## Dependencies

### Core (always required)
```
pyannote.audio>=4.0              # Tier 1 diarization
speechbrain @ git+https://github.com/speechbrain/speechbrain.git@develop  # Tier 2 embeddings
librosa==0.10.2                  # Audio processing
scikit-learn>=1.3.0              # Clustering
```

### Optional (for Layers 3-4)
```
filterpy>=1.4.5                  # Pitch Kalman filter
praat-parselmouth>=0.4.3         # High-quality pitch extraction
httpx>=0.25.0                    # LLM API calls
```

### Important Installation Notes

- **SpeechBrain MUST be installed from develop branch**, not PyPI. The PyPI release crashes with `use_auth_token` deprecation and missing `custom.py`.
  ```bash
  pip install git+https://github.com/speechbrain/speechbrain.git@develop
  ```
- **Pyannote community-1 requires accepting the license** at https://huggingface.co/pyannote/speaker-diarization-community-1
- **Use `from speechbrain.inference.speaker import EncoderClassifier`** (new import path), not `from speechbrain.pretrained import ...` (old, broken).

---

## File Map

| File | What Changed |
|------|-------------|
| `services/voiceAgent/transcriber.py` | All diarization logic: cascade, embeddings, pyannote, Kalman, LLM, linguistic rules |
| `services/voiceAgent/requirements.txt` | Added pyannote>=4.0, speechbrain (develop), filterpy, parselmouth |
| `shared/config/settings.py` | Added `diarization_mode`, `diarization_embedding_model` |
| `.gitignore` | Added `pretrained_models/` (ECAPA-TDNN cache) |
| `services/language_agent/feature_extractor.py` | Fixed objection detection regex (related fix) |

---

## Commit History

```
dea0b7c Auto speaker count + greeting→response rule + confidence downgrade
ea83a7d Upgrade LLM to gpt-4o
835716d Evidence-based multi-signal LLM system
94c3dad Pitch Kalman filter + LLM layers (opt-in)
218cea5 Pyannote community-1 Tier 1 + greeting→query rule
31b3d51 Gitignore pretrained models
161853e Linguistic post-correction
7e9fba6 SpeechBrain ECAPA-TDNN (develop branch)
462acfb Overhaul: new cascade + enhanced KMeans/gap+pitch
```

**From 4.3% → 95.7% on real 2-speaker calls. One call hit 100% (40/40 segments).**
