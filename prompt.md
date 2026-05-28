# NEXUS — Fix Notes Transcript Dump + False Contradictions + Flat Stress Fallback

## 3 Bugs Confirmed from Screenshots (21-minute meeting, 3 speakers)

### Bug 1: Notes section dumps nearly the entire transcript
Opening section: 9 segments including "I've been good, thanks" and weather chat.
Discussion: 7+ segments. Main Topic: 6 segments. Follow-up: 4 segments. Closing: 8 segments.
Expected: 2-4 key points per topic. Got: every utterance.

### Bug 2: All 6 statement contradictions are false fires (100% false positive)
- "we would still need more info" → "we can just work with your PDFs" = conversation PROGRESSION, not contradiction
- "it's not relevant for Mirko to see sub-processes" → "when can you share, Mirko" = DIFFERENT topics sharing a name
- "getting precise score is not the goal" → "let's do that earlier" = UNRELATED statements
- "we were getting lesser scores" → "we have it for both of them" = DIFFERENT topics
- "we are NOT JUST giving the prompt" → "getting good results" = statements AGREE ("not just" = "more than")
- "I'll share the videos" → "No, I think we are good" = "No" answers a question, not contradicting

### Bug 3: Voice stress shows 0.28 for almost every segment
`_stress_at()` returns `avg_stress` fallback (0.28) when no peak is within ±5s.
The fallback takes the FIRST speaker's avg (not the correct speaker's).
"Stress decreased from 0.28 to 0.28" is displayed as meaningful data — it's not.

---

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read `services/fusion_agent/narrative.py` COMPLETELY — the file is provided.
   Focus on:
   - `_build_context()` lines 270-300 — the contradiction candidate builder
   - `_stress_at()` lines 153-161 — the stress lookup with broken fallback
   - `_content_words()` lines 168-178 — the stopword filter
   - `_fallback_narrative()` lines 990-1020 — the notes builder
   - `_build_prompt()` lines 720-740 — the LLM instruction for notes

2. Think about the problem:
   - Notes: the LLM ignores "only 2-4 key points" because it sees the full
     transcript in the voice-anomaly and contradiction sections. It has the
     raw text and copies it. The fallback picks by LENGTH not IMPORTANCE.
   - Contradictions: negation flip detection is semantically blind. Shared
     content words ≥ 2 is too low for a meeting where everyone discusses the
     same topic. "not just/only" is an intensifier, not a negation.
   - Stress: `_stress_at()` doesn't know which speaker it's looking up.
     The fallback returns the first speaker's avg, not the correct one.

3. Use proper OOP and DSA:
   - **Regex negative lookahead**: exclude "not just|only|merely|necessarily"
   - **HashMap**: per-speaker stress lookup O(1) instead of iterating all speakers
   - **Content word threshold**: raise from 2 to 4 shared words minimum
   - **Claim detection**: regex for first-person claim patterns before flagging

---

## Fix 1: Notes — Filter Segments Before LLM Sees Them

**File:** `services/fusion_agent/narrative.py`

### Problem
The LLM prompt says "only 2-4 key points per topic" but the LLM receives
the full transcript in the voice-anomaly and contradiction context sections.
It has every utterance available and copies them into notes.

The fallback path picks the 3 LONGEST segments per topic — verbose chit-chat
passes while short decisions are excluded.

### Fix A: Pre-filter segments in _build_context() — voice anomaly section

Only include segments at actual stress peak timestamps, not every segment:

```python
# Line ~240: Voice Anomalies section
# BEFORE: includes all segments near any peak timestamp
# AFTER: only include segments where stress is genuinely elevated
for p in peaks[:10]:
    time_ms = p.get("time_ms", 0)
    speaker = p.get("speaker", "?")
    stress = p.get("stress_score", 0)
    
    # Only include if stress is actually elevated (> 1.5× speaker's baseline)
    spk_avg = voice_summary.get("per_speaker", {}).get(speaker, {}).get("avg_stress", 0.25)
    if stress < spk_avg * 1.5:
        continue  # skip — not a meaningful anomaly
    
    # ... rest of segment lookup unchanged
```

### Fix B: Pre-filter segments in _build_context() — contradiction section

See Fix 2 below — the contradiction section itself is broken. Fixing it
reduces the number of raw transcript segments the LLM sees.

### Fix C: Fix fallback notes builder — filter by IMPORTANCE not LENGTH

```python
# Replace the fallback notes builder (lines 990-1020):

notes: list[dict] = []
segs = transcript_segments or []

# Importance signals: questions, commitments, decisions, numbers, disagreements
_IMPORTANCE_PATTERNS = re.compile(
    r"(\?$"                           # questions
    r"|\b(will|can|should|need to|have to|going to|let's|agreed|decided)\b"  # commitments/decisions
    r"|\b\d{1,2}[:.]\d{2}\b"         # times (8:30, 3:00)
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"  # days
    r"|\b\d+\s*(percent|%|dollars?|hours?|minutes?|days?|weeks?)\b"    # quantities
    r"|\b(problem|issue|concern|blocker|risk|deadline)\b"              # problems
    r"|\b(okay|agree|sounds good|let's do|confirmed)\b"               # agreements
    r")",
    re.IGNORECASE,
)

for topic in entities.get("topics", []):
    topic_start = topic.get("start_ms", 0)
    topic_end = topic.get("end_ms", 0)
    
    # Collect segments in this topic's time range
    topic_segs = [
        seg for seg in segs
        if seg.get("start_ms", 0) >= topic_start
        and seg.get("end_ms", 0) <= topic_end
        and len(seg.get("text", "")) > 30
    ]
    
    # Score each segment by importance (not length)
    scored = []
    for seg in topic_segs:
        text = seg.get("text", "")
        importance = 0
        
        # Has importance pattern (question, commitment, decision, number)
        if _IMPORTANCE_PATTERNS.search(text):
            importance += 3
        
        # Is a commitment from entities
        for com in entities.get("commitments", []):
            if abs(com.get("timestamp_ms", 0) - seg.get("start_ms", 0)) < 3000:
                importance += 5
        
        # Is near an objection
        for obj in entities.get("objections", []):
            if abs(obj.get("timestamp_ms", 0) - seg.get("start_ms", 0)) < 3000:
                importance += 4
        
        # Has a stress peak nearby (genuinely elevated, not baseline)
        spk = seg.get("speaker", "")
        spk_avg = voice_summary.get("per_speaker", {}).get(spk, {}).get("avg_stress", 0.25)
        seg_stress = _stress_at_for_speaker(seg.get("start_ms", 0), voice_summary, spk)
        if seg_stress and seg_stress > spk_avg * 1.5:
            importance += 2
        
        # Minimum length for substance (skip "yeah", "okay", "got it")
        if len(text) < 50:
            importance -= 2
        
        if importance > 0:
            scored.append((importance, seg))
    
    # Take top 3-4 by importance score
    scored.sort(key=lambda x: -x[0])
    topic_details = []
    for _, seg in scored[:4]:
        ts_s = seg["start_ms"] // 1000
        m, s = divmod(ts_s, 60)
        topic_details.append({
            "speaker": seg.get("speaker", ""),
            "text": seg["text"][:200],
            "timestamp": f"{m}:{s:02d}",
            "sub_details": [],
        })
    
    if topic_details:
        notes.append({
            "topic": topic.get("name", "Discussion"),
            "summary": topic.get("summary", ""),
            "details": topic_details,
        })
```

### Fix D: Strengthen LLM instruction

In `_build_prompt()` system prompt, replace the notes instruction:

```python
# BEFORE:
"Under each topic include ONLY the 2-4 most significant contributions"

# AFTER:
"2. notes: Group discussion by TOPIC or PHASE. Under each topic, include "
"ONLY 2-4 entries that represent DECISIONS, COMMITMENTS, PROBLEMS RAISED, "
"or KEY QUESTIONS. Exclude: greetings, pleasantries, filler ('got it', "
"'okay', 'yeah'), repetition, casual chat, and off-topic remarks. "
"If a topic had no decisions or key points, omit the topic entirely. "
"Each entry should be a PARAPHRASED key point, not a verbatim transcript quote. "
"A 20-minute meeting should produce 8-15 total note entries, not 30+."
```

---

## Fix 2: Contradictions — 3 Defects to Fix

**File:** `services/fusion_agent/narrative.py`

All three defects exist in BOTH `_build_context()` (lines 270-300) AND
`_fallback_narrative()` (lines 1070-1110). Fix both locations identically.

### Defect A: Shared content word threshold too low

```python
# BEFORE (line 280 in _build_context, line 1082 in _fallback):
if len(shared) < 2:
    continue

# AFTER:
if len(shared) < 4:
    continue
```

Why 4: In a meeting about "candidate scoring," words like "scoring",
"candidates", "reports", "assessment" appear in every segment. At threshold 2,
any two segments sharing "scoring" + "candidates" are flagged. At threshold 4,
segments must share 4+ distinctive topic words — much more likely to be about
the same specific claim.

### Defect B: "not just/only/merely" parsed as negation

```python
# BEFORE (line 283 in _build_context, line 1085 in _fallback):
neg_a = bool(re.search(
    r"\b(not|never|didn't|don't|wasn't|weren't|no|none)\b",
    seg_a["text"], re.I,
))

# AFTER:
_NEG_PATTERN = re.compile(
    r"\b(not|never|didn't|don't|wasn't|weren't|no(?!t)|none|nobody)\b"
    r"(?!\s+(?:just|only|merely|necessarily|yet|even))",
    re.IGNORECASE,
)
neg_a = bool(_NEG_PATTERN.search(seg_a["text"]))
```

The negative lookahead `(?!\s+(?:just|only|merely|...))` prevents "not just",
"not only", "not merely", "not necessarily", "not yet", "not even" from
being counted as negations. These are intensifiers, not denials.

Also: `no(?!t)` prevents matching "not" via the "no" branch (avoids double-counting).

### Defect C: No semantic direction check

Add a claim-pattern filter. Only flag segments where both sides make a
first-person CLAIM (not questions, backchannels, or scheduling):

```python
# ADD after the shared word check, before the negation check:

# Both segments must contain a first-person claim to be contradictory
_CLAIM = re.compile(r"\b(I |we |it is|it's |that's |there is|there are|they )", re.I)
if not _CLAIM.search(seg_a["text"]) or not _CLAIM.search(seg_b["text"]):
    continue

# Also skip if either segment is primarily a question
if seg_a["text"].rstrip().endswith("?") or seg_b["text"].rstrip().endswith("?"):
    continue
```

### Combined fix for _build_context() contradiction section:

```python
    # Define patterns once at module level or top of function
    _NEG_PATTERN = re.compile(
        r"\b(not|never|didn't|don't|wasn't|weren't|no(?!t)|none|nobody)\b"
        r"(?!\s+(?:just|only|merely|necessarily|yet|even))",
        re.IGNORECASE,
    )
    _CLAIM = re.compile(r"\b(I |we |it is|it's |that's |there is|there are)", re.I)

    # ... inside the loop:
    for seg_b in spk_segs[i + 5:]:
        words_b = _content_words(seg_b["text"])
        shared = words_a & words_b
        if len(shared) < 4:                    # ← raised from 2
            continue
        
        # Both must be claims, not questions or backchannels
        if not _CLAIM.search(seg_a["text"]) or not _CLAIM.search(seg_b["text"]):
            continue
        if seg_a["text"].rstrip().endswith("?") or seg_b["text"].rstrip().endswith("?"):
            continue
        
        neg_a = bool(_NEG_PATTERN.search(seg_a["text"]))
        neg_b = bool(_NEG_PATTERN.search(seg_b["text"]))
        
        if neg_a != neg_b:
            # ... rest of contradiction recording unchanged
```

Apply the SAME 3 fixes to the `_fallback_narrative()` contradiction section
(lines 1070-1110).

---

## Fix 3: `_stress_at()` — Speaker-Aware Lookup, No Fake Fallback

**File:** `services/fusion_agent/narrative.py`

### Current broken code (line 153):

```python
def _stress_at(timestamp_ms: int, voice_summary: dict) -> float:
    peaks = voice_summary.get("stress_peaks", [])
    closest, closest_dist = 0.0, float("inf")
    for p in peaks:
        dist = abs(p.get("time_ms", 0) - timestamp_ms)
        if dist < 5000 and dist < closest_dist:
            closest = p.get("stress_score", 0)
            closest_dist = dist
    if closest == 0.0:
        for spk_data in voice_summary.get("per_speaker", {}).values():
            closest = spk_data.get("avg_stress", 0)   # ← WRONG speaker
            break                                       # ← takes first, not correct
    return closest
```

### Three problems:
1. Does not accept `speaker_id` parameter — can't look up correct speaker
2. Fallback iterates `per_speaker` and breaks on FIRST entry — wrong speaker
3. Returns `avg_stress` when no peak nearby — displayed as real data but it's baseline noise

### Fixed version:

```python
def _stress_at(
    timestamp_ms: int,
    voice_summary: dict,
    speaker_id: str = "",
) -> Optional[float]:
    """
    Find voice stress near a timestamp for a specific speaker.
    
    Returns:
        Actual stress value if a peak or signal exists within ±5s for this speaker.
        None if no real data exists (caller decides how to display).
    """
    # First: check stress peaks within ±5s for THIS speaker
    peaks = voice_summary.get("stress_peaks", [])
    best, best_dist = None, float("inf")
    for p in peaks:
        if speaker_id and p.get("speaker", "") != speaker_id:
            continue  # wrong speaker's peak
        dist = abs(p.get("time_ms", 0) - timestamp_ms)
        if dist < 5000 and dist < best_dist:
            best = p.get("stress_score", 0)
            best_dist = dist
    
    if best is not None:
        return best
    
    # No peak nearby — return None (no data), NOT avg_stress
    # The caller should display "N/A" or omit the stress value
    return None
```

### Update ALL callers of _stress_at():

In `_build_context()` contradiction section:
```python
stress_a = _stress_at(time_a, voice_summary, speaker_id=spk)
stress_b = _stress_at(time_b, voice_summary, speaker_id=spk)

# Only include stress if real data exists
if stress_a is not None:
    lines.append(f"    ... [stress={stress_a:.2f}]")
else:
    lines.append(f"    ... [stress=N/A]")
```

In `_fallback_narrative()` contradiction section:
```python
stress_a = _stress_at(time_a, voice_summary, speaker_id=spk)
stress_b = _stress_at(time_b, voice_summary, speaker_id=spk)

contradiction_analysis.append({
    "speaker": spk,
    "statement_a": {
        "text": seg_a["text"][:200],
        "timestamp": f"{ma}:{sa:02d}",
        "voice_stress": round(stress_a, 3) if stress_a is not None else None,
    },
    "statement_b": {
        "text": seg_b["text"][:200],
        "timestamp": f"{mb}:{sb:02d}",
        "voice_stress": round(stress_b, 3) if stress_b is not None else None,
    },
    "contradiction_type": "negation_flip",
    "voice_delta": (
        f"Stress {'increased' if (stress_b or 0) > (stress_a or 0) else 'decreased'} "
        f"from {stress_a:.2f} to {stress_b:.2f} between statements"
        if stress_a is not None and stress_b is not None
        else "No voice stress data available at these timestamps"
    ),
    ...
})
```

In `_fallback_narrative()` voice-text correlations — already correct (uses
peaks directly, doesn't call `_stress_at`).

### Frontend update — handle null stress:

```tsx
{/* In Statement Contradictions card: */}
<div className="mt-1 text-[10px]">
  voice stress: {c.statement_a.voice_stress != null ? (
    <span className={c.statement_a.voice_stress > 0.5 ? "text-red-400" : "text-green-400"}>
      {c.statement_a.voice_stress.toFixed(2)}
    </span>
  ) : (
    <span className="text-nexus-text-muted">N/A</span>
  )}
</div>

{/* In voice_delta display: */}
{c.voice_delta && !c.voice_delta.includes("No voice stress data") && (
  <div className="text-xs text-nexus-accent-purple italic">{c.voice_delta}</div>
)}
```

---

## Expected Results After Fix

### Notes:
| Before | After |
|:------:|:-----:|
| 30+ entries across 5 topics | 8-15 entries (decisions, commitments, problems, key questions only) |
| "I've been good, thanks" included | Filtered out (no importance pattern) |
| "got it, got it, got it" included | Filtered out (< 50 chars, no importance) |
| Full casual chat in Opening | Opening: 1-2 entries or omitted entirely |

### Contradictions:
| Before | After |
|:------:|:-----:|
| 6 flagged, 0 genuine (100% FP) | 0-1 flagged (only if genuinely incompatible claims exist) |
| "not just" parsed as negation | Excluded by negative lookahead |
| 2 shared words = same topic | 4 shared words required |
| Questions flagged as claims | Excluded by question/claim filter |

### Stress values:
| Before | After |
|:------:|:-----:|
| 0.28 everywhere (first speaker's avg) | Real value when peak exists, `null`/N/A when no data |
| "Stress decreased from 0.28 to 0.28" | "No voice stress data at these timestamps" or omitted |
| Wrong speaker's baseline | Speaker-specific lookup via `speaker_id` parameter |

---

## Files Modified:

### Backend (1):
**services/fusion_agent/narrative.py**:
- `_stress_at()` — add `speaker_id` param, speaker-filtered peak lookup, return `None` not avg (~10 lines changed)
- `_content_words()` — no change needed (threshold raised in callers)
- `_build_context()` contradiction section — raise threshold 2→4, add claim filter, fix negation regex (~8 lines changed)
- `_build_context()` voice anomaly section — skip peaks below 1.5× baseline (~3 lines added)
- `_build_prompt()` — strengthen notes instruction (~5 lines changed)
- `_fallback_narrative()` notes builder — importance-based filtering instead of length (~40 lines replaced)
- `_fallback_narrative()` contradiction section — same 3 fixes as _build_context (~8 lines changed)
- `_fallback_narrative()` all `_stress_at` calls — pass speaker_id, handle None (~6 lines changed)
- Add `_NEG_PATTERN` and `_CLAIM` as module-level compiled regexes (~5 lines added)
- Add `_IMPORTANCE_PATTERNS` compiled regex (~8 lines added)

### Frontend (1):
**SessionDetail.tsx** — handle null voice_stress in contradiction cards (~4 lines changed)