# NEXUS — Cross-Modal Contradiction Analysis in Narrative Report

## Problem

The narrative LLM receives voice stress peaks ("432s — stress=0.78") but NOT
what was being said at that timestamp. It receives language summaries but NOT
the voice signals at the moments of contradictory statements. Two rich data
streams exist side by side but are never cross-referenced at the statement level.

## What to Build

Two cross-referencing analyses that work for ALL content types:

### Analysis 1: Text Contradictions + Voice Evidence

Find moments where the same speaker said contradictory things at different
timestamps. For each contradiction pair, show the voice/video signals at
BOTH timestamps. This reveals whether the speaker was stressed during the
revision (possible deception/discomfort) or calm (likely a genuine update).

```json
"contradiction_analysis": [
  {
    "speaker": "Speaker_0",
    "statement_a": {
      "text": "I was home all night",
      "timestamp": "2:30",
      "voice_stress": 0.22,
      "facial_stress": 0.18,
      "speech_rate_vs_baseline": "+5%"
    },
    "statement_b": {
      "text": "I might have gone out around 10",
      "timestamp": "18:45",
      "voice_stress": 0.71,
      "facial_stress": 0.65,
      "speech_rate_vs_baseline": "-35%"
    },
    "contradiction_type": "factual_reversal",
    "voice_delta": "Voice stress increased 3.2× between statements. Speech rate dropped 35%. Pitch elevated at revision.",
    "significance": "The speaker revised their account of nighttime whereabouts with elevated stress markers at the revision point, suggesting the revision was a significant cognitive event."
  }
]
```

### Analysis 2: Voice Anomalies + Transcript Context

For each voice anomaly (stress peak, pitch spike, speech rate change, hesitation
cluster), show exactly what was being said at that moment. This gives the user
immediate context — they see the signal AND the words together.

```json
"voice_text_correlations": [
  {
    "timestamp": "8:30",
    "speaker": "Speaker_0",
    "anomaly_type": "stress_peak",
    "anomaly_value": "stress=0.78 (baseline=0.25)",
    "transcript_text": "The pricing is... um... competitive for what you're getting",
    "additional_signals": {
      "speech_rate": "-35% from baseline",
      "fillers": "1 (um)",
      "facial_stress": 0.72
    },
    "context": "Stress peak occurred during pricing discussion — speaker showed multi-channel stress (voice + face + speech rate) while discussing price competitiveness"
  }
]
```

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read `services/fusion_agent/narrative.py` — COMPLETELY:
   - `_build_context()` — what it currently receives and builds
   - `generate_session_narrative()` — what parameters are passed
   - `_build_prompt()` — the JSON schema
   - `_fallback_narrative()` — how it uses transcript_segments

2. Read `services/fusion_agent/fusion_service.py`:
   - What data is available when narrative is called
   - `voice_summary` structure — stress_peaks with timestamps
   - `fusion_signals` — all cross-modal signals with timestamps

3. Think about the problem:
   - The LLM needs BOTH transcript text AND voice signals at the same timestamps
   - Full transcript can be 50K+ tokens — can't dump it all into context
   - Need to be SELECTIVE: only include segments at anomaly timestamps + contradiction candidates
   - The fallback path needs to work without an LLM

4. Use proper OOP and DSA:
   - **bisect**: O(log N) lookup of transcript segment at a given timestamp
   - **HashMap**: group segments by speaker for contradiction detection
   - **TF-IDF cosine** or **Jaccard**: find same-topic segment pairs from same speaker
   - **Negation detection**: regex for "not", "never", "didn't", "no" flips between pairs

---

## Change 1: Add Transcript Segments to _build_context()

**File:** `services/fusion_agent/narrative.py` — `_build_context()`

Currently `_build_context()` does NOT receive `transcript_segments`. Add it
as a parameter and build two new context sections:

### 1a. Add `transcript_segments` parameter

```python
def _build_context(
    session_id, duration_seconds, speakers,
    voice_summary, language_summary, fusion_signals,
    unified_states,
    entities=None, graph_analytics=None,
    conversation_summary=None, video_summary=None,
    transcript_segments=None,            # ← ADD
) -> str:
```

Update the call site in `generate_session_narrative()` to pass transcript_segments.

### 1b. Voice Anomaly + Transcript Cross-Reference Section

For each stress peak, pitch elevation, and speech rate anomaly in voice_summary,
find the transcript segment at that timestamp and include the text.

```python
    # ── Voice Anomalies with Transcript Context ────────────────────────
    segs = transcript_segments or []
    if segs and peaks:
        lines.append("\n=== VOICE ANOMALIES WITH TRANSCRIPT CONTEXT ===")
        lines.append("(What was being said at each voice anomaly moment)")

        # Build sorted timestamp index for O(log N) segment lookup
        import bisect
        seg_starts = [s.get("start_ms", 0) for s in segs]

        for p in peaks[:10]:  # top 10 stress peaks
            time_ms = p.get("time_ms", 0)
            speaker = p.get("speaker", "?")
            stress = p.get("stress_score", 0)

            # Find transcript segment at this timestamp
            idx = bisect.bisect_right(seg_starts, time_ms) - 1
            if 0 <= idx < len(segs):
                seg = segs[idx]
                seg_end = seg.get("end_ms", seg.get("start_ms", 0) + 2000)
                # Only use if segment is within ±3 seconds of the anomaly
                if abs(seg.get("start_ms", 0) - time_ms) < 5000:
                    text = seg.get("text", "").strip()
                    if text:
                        time_s = time_ms / 1000
                        m, s = divmod(int(time_s), 60)
                        lines.append(
                            f"\n  {m}:{s:02d} — {_display(speaker, name_map)}: "
                            f"stress={stress:.3f}"
                        )
                        lines.append(f"    Said: \"{text[:200]}\"")
```

### 1c. Statement Pairs for Contradiction Detection

For each speaker, find transcript segments that share topic words but express
different claims. Include both segments in the context so the LLM can assess.

```python
    # ── Potential Contradictions (same speaker, different claims) ────────
    if segs and len(segs) > 10:
        lines.append("\n=== POTENTIAL STATEMENT CONTRADICTIONS ===")
        lines.append("(Same speaker, similar topic, different content — for LLM to evaluate)")

        from collections import defaultdict
        import re

        # Group segments by speaker
        by_speaker: dict[str, list[dict]] = defaultdict(list)
        for seg in segs:
            spk = seg.get("speaker", "")
            text = seg.get("text", "").strip()
            if spk and len(text) > 30:
                by_speaker[spk].append(seg)

        # Content word extraction (reuse _extract_content_words pattern)
        def _content_words(text: str) -> set[str]:
            return {w.lower() for w in re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())}
            # Note: use the validated stopword list if available

        for spk, spk_segs in by_speaker.items():
            if len(spk_segs) < 5:
                continue

            pairs_found = 0
            for i, seg_a in enumerate(spk_segs):
                if pairs_found >= 3:  # max 3 contradiction candidates per speaker
                    break
                words_a = _content_words(seg_a["text"])
                if len(words_a) < 3:
                    continue

                for seg_b in spk_segs[i+5:]:  # at least 5 segments apart (~10s)
                    words_b = _content_words(seg_b["text"])
                    shared = words_a & words_b
                    if len(shared) < 2:
                        continue

                    # Check for negation flip
                    neg_a = bool(re.search(r"\b(not|never|didn't|don't|wasn't|weren't|no|none)\b",
                                           seg_a["text"], re.I))
                    neg_b = bool(re.search(r"\b(not|never|didn't|don't|wasn't|weren't|no|none)\b",
                                           seg_b["text"], re.I))

                    if neg_a != neg_b or len(shared) >= 4:
                        time_a = seg_a.get("start_ms", 0) // 1000
                        time_b = seg_b.get("start_ms", 0) // 1000
                        ma, sa = divmod(time_a, 60)
                        mb, sb = divmod(time_b, 60)

                        # Lookup voice stress at both timestamps
                        stress_a = _stress_at(time_a * 1000, voice_summary)
                        stress_b = _stress_at(time_b * 1000, voice_summary)

                        lines.append(
                            f"\n  {_display(spk, name_map)} — potential contradiction:"
                        )
                        lines.append(
                            f"    A ({ma}:{sa:02d}): \"{seg_a['text'][:150]}\" "
                            f"[stress={stress_a:.2f}]"
                        )
                        lines.append(
                            f"    B ({mb}:{sb:02d}): \"{seg_b['text'][:150]}\" "
                            f"[stress={stress_b:.2f}]"
                        )
                        lines.append(f"    Shared topic words: {', '.join(sorted(shared)[:5])}")
                        if neg_a != neg_b:
                            lines.append(f"    ⚡ Negation flip detected")

                        pairs_found += 1
                        break  # next seg_a
```

Helper function to lookup stress at a timestamp:

```python
def _stress_at(timestamp_ms: int, voice_summary: dict) -> float:
    """Find the closest stress peak within ±5 seconds of a timestamp."""
    peaks = voice_summary.get("stress_peaks", [])
    closest = 0.0
    closest_dist = float("inf")
    for p in peaks:
        dist = abs(p.get("time_ms", 0) - timestamp_ms)
        if dist < 5000 and dist < closest_dist:
            closest = p.get("stress_score", 0)
            closest_dist = dist
    # If no stress peak nearby, check per-speaker average
    if closest == 0.0:
        for spk_data in voice_summary.get("per_speaker", {}).values():
            closest = spk_data.get("avg_stress", 0)
            break
    return closest
```

---

## Change 2: Add to JSON Schema

**File:** `services/fusion_agent/narrative.py` — `_build_prompt()`

Add two new fields to the JSON schema:

```python
    '"contradiction_analysis": ['
    '  {'
    '    "speaker": "Speaker name",'
    '    "statement_a": {"text": "What they said first", "timestamp": "MM:SS", "voice_stress": 0.22},'
    '    "statement_b": {"text": "What they said later that contradicts", "timestamp": "MM:SS", "voice_stress": 0.71},'
    '    "contradiction_type": "factual_reversal | number_change | negation_flip | narrative_shift",'
    '    "voice_delta": "How voice/face signals changed between the two statements",'
    '    "significance": "Why this matters — what does the combination of text change + voice change tell us"'
    '  }'
    '],'
    '"voice_text_correlations": ['
    '  {'
    '    "timestamp": "MM:SS",'
    '    "speaker": "Speaker name",'
    '    "anomaly_type": "stress_peak | pitch_elevation | speech_rate_drop | hesitation_cluster",'
    '    "anomaly_value": "stress=0.78 (baseline=0.25)",'
    '    "transcript_text": "What was being said at this moment",'
    '    "additional_signals": "Other voice/face/body signals active at this timestamp",'
    '    "context": "Why this moment matters — what topic was being discussed"'
    '  }'
    '],'
```

Add to the structure requirements instruction:

```python
    "CONTRADICTION ANALYSIS: Review the POTENTIAL STATEMENT CONTRADICTIONS in the context. "
    "For each genuine contradiction (not just topic revisiting), include both statements, "
    "their timestamps, the voice stress at each moment, and what the voice delta tells us. "
    "Contradiction types: factual_reversal (opposite claims), number_change (different figures), "
    "negation_flip (denied then admitted or vice versa), narrative_shift (story changed). "
    "VOICE-TEXT CORRELATIONS: For each voice anomaly in the context that has transcript text, "
    "include the anomaly details + what was being said. Focus on the top 5 most significant "
    "moments where voice and text together tell a richer story than either alone. "
    "IMPORTANT: Not every statement revision is a contradiction. People update estimates, "
    "add details, and refine positions naturally. Only flag genuinely contradictory statements "
    "where the later version is incompatible with the earlier one. "
```

---

## Change 3: _parse_narrative_response

Add the two new fields:

```python
    "contradiction_analysis": parsed.get("contradiction_analysis", []),
    "voice_text_correlations": parsed.get("voice_text_correlations", []),
```

---

## Change 4: _fallback_narrative (non-LLM path)

Build contradiction_analysis from the pre-computed contradiction candidates:

```python
    # ── Contradiction analysis (from transcript) ─────────────────────────
    contradiction_analysis = []
    segs = transcript_segments or []
    if len(segs) > 10:
        by_speaker = defaultdict(list)
        for seg in segs:
            spk = seg.get("speaker", "")
            text = seg.get("text", "").strip()
            if spk and len(text) > 30:
                by_speaker[spk].append(seg)

        for spk, spk_segs in by_speaker.items():
            for i, seg_a in enumerate(spk_segs):
                words_a = _content_words(seg_a["text"])
                if len(words_a) < 3:
                    continue
                for seg_b in spk_segs[i+5:]:
                    words_b = _content_words(seg_b["text"])
                    shared = words_a & words_b
                    if len(shared) < 2:
                        continue
                    neg_a = bool(re.search(r"\b(not|never|didn't|don't|no)\b", seg_a["text"], re.I))
                    neg_b = bool(re.search(r"\b(not|never|didn't|don't|no)\b", seg_b["text"], re.I))
                    if neg_a != neg_b:
                        time_a = seg_a.get("start_ms", 0)
                        time_b = seg_b.get("start_ms", 0)
                        stress_a = _stress_at(time_a, voice_summary)
                        stress_b = _stress_at(time_b, voice_summary)
                        ma, sa = divmod(time_a // 1000, 60)
                        mb, sb = divmod(time_b // 1000, 60)
                        contradiction_analysis.append({
                            "speaker": spk,
                            "statement_a": {
                                "text": seg_a["text"][:200],
                                "timestamp": f"{ma}:{sa:02d}",
                                "voice_stress": round(stress_a, 3),
                            },
                            "statement_b": {
                                "text": seg_b["text"][:200],
                                "timestamp": f"{mb}:{sb:02d}",
                                "voice_stress": round(stress_b, 3),
                            },
                            "contradiction_type": "negation_flip",
                            "voice_delta": (
                                f"Stress {'increased' if stress_b > stress_a else 'decreased'} "
                                f"from {stress_a:.2f} to {stress_b:.2f} between statements"
                            ),
                            "significance": (
                                "Speaker's position reversed between these statements. "
                                "Cross-reference voice stress change for context."
                            ),
                        })
                        break
                if len(contradiction_analysis) >= 5:
                    break

    # ── Voice-text correlations ──────────────────────────────────────────
    voice_text_correlations = []
    peaks = voice_summary.get("stress_peaks", [])
    seg_starts = [s.get("start_ms", 0) for s in segs]
    for p in peaks[:5]:
        time_ms = p.get("time_ms", 0)
        idx = bisect.bisect_right(seg_starts, time_ms) - 1
        if 0 <= idx < len(segs):
            seg = segs[idx]
            if abs(seg.get("start_ms", 0) - time_ms) < 5000:
                text = seg.get("text", "").strip()
                if text:
                    m, s_val = divmod(time_ms // 1000, 60)
                    voice_text_correlations.append({
                        "timestamp": f"{m}:{s_val:02d}",
                        "speaker": p.get("speaker", ""),
                        "anomaly_type": "stress_peak",
                        "anomaly_value": f"stress={p.get('stress_score', 0):.3f}",
                        "transcript_text": text[:200],
                        "additional_signals": "",
                        "context": f"Voice stress peaked during this statement",
                    })
```

---

## Change 5: Frontend — Report Tab

### Contradiction Analysis Section

```tsx
{content?.contradiction_analysis && content.contradiction_analysis.length > 0 && (
  <section className="rounded-lg border border-amber-500/20 bg-nexus-surface p-5">
    <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-amber-400">
      ⚡ Statement Contradictions
    </h2>
    <div className="space-y-4">
      {content.contradiction_analysis.map((c, i) => (
        <div key={i} className="rounded border border-nexus-border p-3">
          <div className="text-xs font-medium text-nexus-text-muted mb-2">
            {c.speaker} — {c.contradiction_type.replace("_", " ")}
          </div>
          <div className="grid grid-cols-2 gap-3 mb-2">
            <div className="rounded bg-nexus-surface-hover p-2">
              <div className="text-[10px] text-nexus-text-muted">{c.statement_a.timestamp}</div>
              <div className="text-sm">&ldquo;{c.statement_a.text}&rdquo;</div>
              <div className="mt-1 text-[10px]">
                voice stress: <span className={c.statement_a.voice_stress > 0.5 ? "text-red-400" : "text-green-400"}>
                  {c.statement_a.voice_stress?.toFixed(2)}
                </span>
              </div>
            </div>
            <div className="rounded bg-nexus-surface-hover p-2">
              <div className="text-[10px] text-nexus-text-muted">{c.statement_b.timestamp}</div>
              <div className="text-sm">&ldquo;{c.statement_b.text}&rdquo;</div>
              <div className="mt-1 text-[10px]">
                voice stress: <span className={c.statement_b.voice_stress > 0.5 ? "text-red-400" : "text-green-400"}>
                  {c.statement_b.voice_stress?.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
          {c.voice_delta && (
            <div className="text-xs text-nexus-accent-purple italic">{c.voice_delta}</div>
          )}
          {c.significance && (
            <div className="mt-1 text-xs text-nexus-text-secondary">{c.significance}</div>
          )}
        </div>
      ))}
    </div>
  </section>
)}
```

### Voice-Text Correlations Section

```tsx
{content?.voice_text_correlations && content.voice_text_correlations.length > 0 && (
  <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
    <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
      🎙️ Voice Anomalies — What Was Being Said
    </h2>
    <div className="space-y-3">
      {content.voice_text_correlations.map((v, i) => (
        <div key={i} className="flex items-start gap-3 rounded border-l-2 border-nexus-accent-purple p-2 bg-nexus-surface-hover">
          <div className="shrink-0 text-center">
            <div className="text-xs font-mono text-nexus-text-muted">{v.timestamp}</div>
            <div className="text-[10px] text-red-400">{v.anomaly_value}</div>
          </div>
          <div className="flex-1">
            <div className="text-xs font-medium text-nexus-text-muted mb-0.5">
              {v.speaker} — {v.anomaly_type.replace("_", " ")}
            </div>
            <div className="text-sm text-nexus-text-primary">
              &ldquo;{v.transcript_text}&rdquo;
            </div>
            {v.context && (
              <div className="mt-1 text-xs text-nexus-text-secondary italic">{v.context}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  </section>
)}
```

---

## Updated Report Section Order

```
1. General Summary
2. Executive Summary
3. Key Facts & Commitments
4. Notes (topic-grouped)
5. Statement Contradictions (NEW)
6. Voice Anomalies — What Was Being Said (NEW)
7. Risk Assessment (interrogation only)
8. Speaker Analyses
9. Key Moments
10. Cross-Modal Insights
11. Recommendations
12. Action Items
```

Contradictions and voice-text correlations go AFTER notes (the detailed record)
and BEFORE speaker analyses (the synthesis). They bridge raw data and interpretation.

---

## Content-Type Applicability

Both sections work for ALL content types:

| Content Type | Contradiction Example | Voice-Text Example |
|-------------|----------------------|-------------------|
| Sales | "We can deliver in 4 weeks" → "Timeline is 6-8 weeks" | Stress peak during pricing discussion |
| Interview | "I have 5 years of Java experience" → "I mostly did Python" | Hesitation cluster during technical question |
| Interrogation | "I was home" → "I might have gone out" | Stress spike after evidence disclosure |
| Client meeting | "Budget is approved" → "We need to get approval" | Speech rate drop during commitment discussion |
| Internal | "Project is on track" → "We're behind on the deliverable" | Pitch elevation during status update |

---

## Files Modified:

### Backend (1):
1. **services/fusion_agent/narrative.py**:
   - `_build_context()` — add transcript_segments parameter, voice anomaly + transcript section (~40 lines), contradiction candidate section (~50 lines), `_stress_at()` helper (~12 lines)
   - `generate_session_narrative()` — pass transcript_segments to _build_context (~1 line)
   - `_build_prompt()` — add 2 new fields to JSON schema + structure instructions (~20 lines)
   - `_parse_narrative_response()` — parse 2 new fields (~2 lines)
   - `_fallback_narrative()` — build contradiction_analysis + voice_text_correlations (~60 lines)

### Frontend (1):
2. **SessionDetail.tsx** — Statement Contradictions section (~30 lines) + Voice-Text Correlations section (~25 lines)