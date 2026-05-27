# NEXUS — Transcript + Voice Credibility Analysis in Narrative Report

## Available Data (ONLY these — nothing else)

```
transcript_segments: [{speaker, text, start_ms, end_ms}]
  — Who said what, when. Keyed by Speaker_N from diarization.

voice_summary:
  per_speaker: {
    baseline_f0_hz, baseline_rate_wpm,
    avg_stress, max_stress, total_fillers,
    pitch_elevation_events, tone_distribution
  }
  stress_peaks: [{time_ms, speaker, stress_score}]

voice_signals (per 2-second window):
  vocal_stress, pitch_shift, speech_rate, energy_level,
  tone_classification, filler_count, pause_duration
```

**NO face signals. NO body signals. NO gaze signals. NO video data.**

The analysis works entirely from what was SAID and how it SOUNDED.

## What to Build — 4 Analytical Layers

### Layer 1: Position Evolution

Track how each speaker's position on key topics changed across the session.
At each position change, show the voice state — was the revision calm
(likely a genuine update) or stressed (significant cognitive event)?

```
Speaker_0 on "delivery timeline":
  5:00  "We can deliver in 4 weeks"
        voice: calm (stress 0.18), steady rate (142 wpm), no fillers
  12:30 "Realistically 4 to 6 weeks"
        voice: slightly elevated (stress 0.31), rate dropped 15%, 1 filler
  18:00 "Timeline would be 6 to 8 weeks"
        voice: elevated (stress 0.52), rate dropped 30%, 2 fillers

  Pattern: Progressive softening of commitment across 3 revisions over 13 minutes.
  Voice stress escalated 2.9× from first to last statement. Speech rate decelerated
  progressively. Filler density increased. The voice trajectory suggests each
  revision was harder to deliver than the previous one.
```

### Layer 2: Voice-Text Correlation

At each voice anomaly (stress peak, pitch spike, rate change, hesitation
cluster), show exactly what was being said. The voice signal becomes
meaningful only when paired with the words.

```
8:30 — stress=0.78 (3.1× baseline), rate dropped 35%, 1 filler ("um")
  Said: "The pricing is... um... competitive for what you're getting"
  Context: Pricing discussion — multi-signal voice event (stress + rate + filler)
           during the single most sensitive word ("competitive")
```

### Layer 3: Contradiction + Voice Evidence

Same speaker said incompatible things at different timestamps. Show voice
state at BOTH timestamps — was the first statement calm and the second
stressed? Were both calm? That changes the interpretation entirely.

```
Statement A (2:30): "I was home all night"
  voice: calm (0.22), steady rate, zero fillers

Statement B (18:45): "I might have gone out around 10"
  voice: elevated (0.71), rate -40%, 3 fillers in surrounding 10s

Voice delta: stress 3.2×, speech slowed significantly, filler density spiked.
The original statement was delivered with baseline-level voice. The revision
carried multi-signal acoustic stress — suggesting the revision was cognitively
costly, not a casual correction.
```

### Layer 4: Speaker Vocal Consistency Profile

Per speaker: across all their statements, how often did their voice stay
at baseline vs deviate? On which TOPICS did their voice deviate most?
This is a voice-only credibility lens.

```
Speaker_0: 47 statements analysed
  At baseline (stress < 1.5× avg): 31 (66%)
  Moderately elevated (1.5-2.5× avg): 12 (26%)
  Highly elevated (> 2.5× avg): 4 (8%)

  Topics with highest voice deviation:
    "pricing" — avg stress 2.4× baseline across 6 statements
    "competitor" — avg stress 1.8× baseline across 3 statements
    "delivery" — avg stress 1.6× baseline across 4 statements

  Topics with stable voice:
    "product features" — avg stress 0.9× baseline across 12 statements
    "team introduction" — avg stress 0.8× baseline across 5 statements
```

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read `services/fusion_agent/narrative.py` — COMPLETELY
2. This analysis uses ONLY `transcript_segments` + `voice_summary` +
   voice signals from `fusion_signals` (filtered to voice agent signals only).
   Do NOT reference face, body, gaze, video, or any visual data.
3. Use proper OOP and DSA:
   - **bisect**: O(log N) timestamp → transcript segment lookup
   - **HashMap**: per-speaker segment grouping, per-topic grouping
   - **Counter**: topic deviation scoring
   - **Sliding window**: filler density in surrounding 10-second window
4. Pre-compute everything in Python. The LLM interprets patterns, it does
   not discover them. The fallback path uses the same pre-computed data.
5. ALL content types. Not interrogation-specific.

---

## Change 1: _build_credibility_analysis()

**File:** `services/fusion_agent/narrative.py`

New function, called from `generate_session_narrative()` BEFORE `_build_context()`.

```python
import bisect
import re
from collections import defaultdict, Counter


def _build_credibility_analysis(
    transcript_segments: list[dict],
    voice_summary: dict,
    fusion_signals: list[dict],
    speakers: list[str],
    name_map: dict[str, str],
) -> dict:
    """
    Pre-compute credibility analysis from transcript + voice signals ONLY.
    No face, body, gaze, or video data used.

    Returns structured data consumed by both _build_context (LLM path)
    and _fallback_narrative (non-LLM path).

    DSA:
      - bisect: O(log N) segment lookup by timestamp
      - defaultdict: O(1) per-speaker grouping
      - Counter: topic deviation tallying
      - Single-pass content word extraction: O(S × W)
    """
    segs = transcript_segments or []
    if len(segs) < 10:
        return {}

    # ── Index structures ─────────────────────────────────────────────
    seg_starts = [s.get("start_ms", 0) for s in segs]
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for seg in segs:
        spk = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        if spk and len(text) > 20:
            by_speaker[spk].append(seg)

    stress_peaks = voice_summary.get("stress_peaks", [])
    per_spk_voice = voice_summary.get("per_speaker", {})

    # Voice signals filtered to voice agent only
    voice_sigs = [
        s for s in (fusion_signals or [])
        if s.get("agent") == "voice"
    ]
    voice_sig_starts = [s.get("window_start_ms", 0) for s in voice_sigs]

    # ── Layer 1: Position Evolution ──────────────────────────────────
    position_evolutions = {}
    for spk, spk_segs in by_speaker.items():
        topics = _extract_speaker_topics(spk_segs)
        evolutions = []
        for topic_label, topic_segs in topics.items():
            if len(topic_segs) < 2:
                continue
            positions = []
            for seg in topic_segs:
                ts = seg.get("start_ms", 0)
                m, s = divmod(ts // 1000, 60)
                stress = _stress_at(ts, stress_peaks, per_spk_voice.get(spk, {}))
                rate_pct = _rate_change_at(ts, voice_sigs, voice_sig_starts, per_spk_voice.get(spk, {}))
                fillers_nearby = _filler_density_at(ts, voice_sigs, voice_sig_starts)
                positions.append({
                    "timestamp": f"{m}:{s:02d}",
                    "timestamp_ms": ts,
                    "text": seg["text"][:200],
                    "voice_stress": round(stress, 3),
                    "rate_change_pct": rate_pct,
                    "filler_density": fillers_nearby,
                })
            if len(positions) >= 2:
                first_stress = positions[0]["voice_stress"]
                last_stress = positions[-1]["voice_stress"]
                stress_trend = (
                    "escalating" if last_stress > first_stress + 0.15 else
                    "declining" if last_stress < first_stress - 0.15 else
                    "stable"
                )
                has_reversal = _detect_negation_flip(positions)
                evolutions.append({
                    "topic": topic_label,
                    "positions": positions,
                    "count": len(positions),
                    "stress_trend": stress_trend,
                    "stress_first": first_stress,
                    "stress_last": last_stress,
                    "stress_multiplier": round(
                        last_stress / max(first_stress, 0.01), 1
                    ),
                    "has_reversal": has_reversal,
                })
        if evolutions:
            position_evolutions[spk] = sorted(
                evolutions, key=lambda e: -e["count"]
            )[:5]

    # ── Layer 2: Voice-Text Correlation ──────────────────────────────
    voice_text_moments = []
    baseline_stress = {}
    for spk in speakers:
        baseline_stress[spk] = per_spk_voice.get(spk, {}).get("avg_stress", 0.25)

    for p in stress_peaks[:10]:
        ts = p.get("time_ms", 0)
        spk = p.get("speaker", "")
        stress = p.get("stress_score", 0)
        base = baseline_stress.get(spk, 0.25)

        idx = bisect.bisect_right(seg_starts, ts) - 1
        if 0 <= idx < len(segs):
            seg = segs[idx]
            if abs(seg.get("start_ms", 0) - ts) < 5000:
                text = seg.get("text", "").strip()
                if text and seg.get("speaker") == spk:
                    m, s = divmod(ts // 1000, 60)
                    rate_pct = _rate_change_at(ts, voice_sigs, voice_sig_starts, per_spk_voice.get(spk, {}))
                    fillers = _filler_density_at(ts, voice_sigs, voice_sig_starts)
                    pitch = _pitch_state_at(ts, voice_sigs, voice_sig_starts)

                    voice_text_moments.append({
                        "timestamp": f"{m}:{s:02d}",
                        "speaker": spk,
                        "stress": round(stress, 3),
                        "baseline": round(base, 3),
                        "multiplier": round(stress / max(base, 0.01), 1),
                        "rate_change": rate_pct,
                        "filler_density": fillers,
                        "pitch_state": pitch,
                        "text": text[:200],
                    })

    # ── Layer 3: Contradictions + Voice Evidence ─────────────────────
    contradictions = []
    for spk, spk_segs in by_speaker.items():
        if len(spk_segs) < 5:
            continue
        found = 0
        for i, seg_a in enumerate(spk_segs):
            if found >= 3:
                break
            words_a = _content_words(seg_a["text"])
            if len(words_a) < 3:
                continue
            for seg_b in spk_segs[i + 5:]:
                words_b = _content_words(seg_b["text"])
                shared = words_a & words_b
                if len(shared) < 2:
                    continue
                neg_a = _has_negation(seg_a["text"])
                neg_b = _has_negation(seg_b["text"])
                if neg_a == neg_b:
                    continue

                ts_a = seg_a.get("start_ms", 0)
                ts_b = seg_b.get("start_ms", 0)
                stress_a = _stress_at(ts_a, stress_peaks, per_spk_voice.get(spk, {}))
                stress_b = _stress_at(ts_b, stress_peaks, per_spk_voice.get(spk, {}))
                rate_a = _rate_change_at(ts_a, voice_sigs, voice_sig_starts, per_spk_voice.get(spk, {}))
                rate_b = _rate_change_at(ts_b, voice_sigs, voice_sig_starts, per_spk_voice.get(spk, {}))
                fillers_a = _filler_density_at(ts_a, voice_sigs, voice_sig_starts)
                fillers_b = _filler_density_at(ts_b, voice_sigs, voice_sig_starts)

                ma, sa = divmod(ts_a // 1000, 60)
                mb, sb = divmod(ts_b // 1000, 60)

                contradictions.append({
                    "speaker": spk,
                    "shared_topic": ", ".join(sorted(shared)[:4]),
                    "statement_a": {
                        "text": seg_a["text"][:200],
                        "timestamp": f"{ma}:{sa:02d}",
                        "stress": round(stress_a, 3),
                        "rate_change": rate_a,
                        "fillers_nearby": fillers_a,
                    },
                    "statement_b": {
                        "text": seg_b["text"][:200],
                        "timestamp": f"{mb}:{sb:02d}",
                        "stress": round(stress_b, 3),
                        "rate_change": rate_b,
                        "fillers_nearby": fillers_b,
                    },
                    "stress_delta": round(stress_b - stress_a, 3),
                    "stress_multiplier": round(
                        stress_b / max(stress_a, 0.01), 1
                    ),
                })
                found += 1
                break

    # ── Layer 4: Vocal Consistency Profile ───────────────────────────
    vocal_profiles = {}
    for spk, spk_segs in by_speaker.items():
        avg_stress = per_spk_voice.get(spk, {}).get("avg_stress", 0.25)
        if avg_stress < 0.01:
            continue

        at_baseline = 0
        moderate = 0
        elevated = 0
        topic_stress: dict[str, list[float]] = defaultdict(list)

        topics = _extract_speaker_topics(spk_segs)
        for topic_label, topic_segs in topics.items():
            for seg in topic_segs:
                ts = seg.get("start_ms", 0)
                stress = _stress_at(ts, stress_peaks, per_spk_voice.get(spk, {}))
                ratio = stress / max(avg_stress, 0.01)

                if ratio < 1.5:
                    at_baseline += 1
                elif ratio < 2.5:
                    moderate += 1
                else:
                    elevated += 1

                topic_stress[topic_label].append(ratio)

        total = at_baseline + moderate + elevated
        if total < 5:
            continue

        # Rank topics by average stress ratio
        topic_rankings = []
        for t_label, ratios in topic_stress.items():
            avg_ratio = sum(ratios) / len(ratios)
            topic_rankings.append({
                "topic": t_label,
                "avg_stress_ratio": round(avg_ratio, 2),
                "statement_count": len(ratios),
            })
        topic_rankings.sort(key=lambda t: -t["avg_stress_ratio"])

        vocal_profiles[spk] = {
            "total_statements": total,
            "at_baseline_pct": round(at_baseline / total * 100, 1),
            "moderate_pct": round(moderate / total * 100, 1),
            "elevated_pct": round(elevated / total * 100, 1),
            "sensitive_topics": topic_rankings[:3],
            "stable_topics": topic_rankings[-3:] if len(topic_rankings) > 3 else [],
        }

    return {
        "position_evolutions": position_evolutions,
        "voice_text_moments": voice_text_moments,
        "contradictions": contradictions,
        "vocal_profiles": vocal_profiles,
    }
```

### Helper Functions

```python
_NEG_PATTERN = re.compile(
    r"\b(not|never|didn't|don't|wasn't|weren't|no|none|nobody|nothing|nowhere|neither)\b",
    re.IGNORECASE,
)


def _content_words(text: str) -> set[str]:
    """5+ char words, no stopwords. O(W) per call."""
    return {w.lower() for w in re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())}


def _has_negation(text: str) -> bool:
    return bool(_NEG_PATTERN.search(text))


def _detect_negation_flip(positions: list[dict]) -> bool:
    """Check any consecutive pair for negation flip."""
    for i in range(len(positions) - 1):
        if _has_negation(positions[i]["text"]) != _has_negation(positions[i + 1]["text"]):
            return True
    return False


def _extract_speaker_topics(segs: list[dict]) -> dict[str, list[dict]]:
    """
    Group segments by topic using content word overlap.
    Two segments share a topic if they have 2+ common content words.
    O(S²) worst case but S is per-speaker segments (typically < 100).
    """
    topics: dict[str, list[dict]] = {}
    for seg in segs:
        words = _content_words(seg.get("text", ""))
        if len(words) < 2:
            continue
        placed = False
        for key, t_segs in topics.items():
            ref_words = _content_words(t_segs[0].get("text", ""))
            if len(words & ref_words) >= 2:
                topics[key].append(seg)
                placed = True
                break
        if not placed:
            label = ", ".join(sorted(words)[:3])
            topics[label] = [seg]
    return topics


def _stress_at(ts_ms: int, peaks: list, spk_voice: dict) -> float:
    """Closest stress peak within ±5s, else speaker average."""
    best, best_dist = 0.0, float("inf")
    for p in peaks:
        d = abs(p.get("time_ms", 0) - ts_ms)
        if d < 5000 and d < best_dist:
            best, best_dist = p.get("stress_score", 0), d
    return best or spk_voice.get("avg_stress", 0)


def _rate_change_at(
    ts_ms: int, voice_sigs: list, sig_starts: list, spk_voice: dict,
) -> str:
    """Speech rate change vs baseline at a timestamp. Returns string like '+15%' or '-30%'."""
    baseline_wpm = spk_voice.get("baseline_rate_wpm", 0)
    if not baseline_wpm or not sig_starts:
        return "N/A"
    idx = bisect.bisect_right(sig_starts, ts_ms) - 1
    if 0 <= idx < len(voice_sigs):
        sig = voice_sigs[idx]
        if (sig.get("signal_type") == "speech_rate"
                and abs(sig.get("window_start_ms", 0) - ts_ms) < 4000):
            rate = float(sig.get("value", baseline_wpm))
            pct = round((rate - baseline_wpm) / max(baseline_wpm, 1) * 100)
            return f"{'+' if pct >= 0 else ''}{pct}%"
    return "N/A"


def _filler_density_at(ts_ms: int, voice_sigs: list, sig_starts: list) -> int:
    """Count filler signals within ±5 seconds of timestamp."""
    count = 0
    for s in voice_sigs:
        if s.get("signal_type") in ("filler_word", "speech_disfluency"):
            if abs(s.get("window_start_ms", 0) - ts_ms) < 5000:
                count += int(s.get("value", 1))
    return count


def _pitch_state_at(ts_ms: int, voice_sigs: list, sig_starts: list) -> str:
    """Pitch state at a timestamp."""
    idx = bisect.bisect_right(sig_starts, ts_ms) - 1
    if 0 <= idx < len(voice_sigs):
        sig = voice_sigs[idx]
        if (sig.get("signal_type") in ("pitch_shift", "pitch_elevation_flag")
                and abs(sig.get("window_start_ms", 0) - ts_ms) < 4000):
            return sig.get("value_text", "normal")
    return "normal"
```

---

## Change 2: Add to _build_context()

```python
def _build_context(
    ...,
    transcript_segments=None,
    credibility_analysis=None,           # ← ADD
) -> str:
```

```python
    # ── Credibility Analysis (transcript + voice only) ───────────────
    ca = credibility_analysis or {}

    if ca.get("position_evolutions"):
        lines.append("\n=== POSITION EVOLUTION (per speaker, per topic) ===")
        lines.append("(How each speaker's statements on key topics changed, with voice state at each point)")
        for spk, evolutions in ca["position_evolutions"].items():
            for ev in evolutions[:3]:
                lines.append(f"\n  {_display(spk, name_map)} on \"{ev['topic']}\":")
                lines.append(f"    {ev['count']} statements | "
                             f"Voice trend: {ev['stress_trend']} | "
                             f"Stress: {ev['stress_first']:.2f} → {ev['stress_last']:.2f} "
                             f"({ev['stress_multiplier']}×) | "
                             f"Reversal: {'YES' if ev['has_reversal'] else 'no'}")
                for pos in ev["positions"][:5]:
                    lines.append(
                        f"    {pos['timestamp']}: \"{pos['text'][:120]}\" "
                        f"[stress={pos['voice_stress']:.2f}, "
                        f"rate={pos['rate_change_pct']}, "
                        f"fillers={pos['filler_density']}]"
                    )

    if ca.get("contradictions"):
        lines.append("\n=== STATEMENT CONTRADICTIONS (negation flips + voice evidence) ===")
        for c in ca["contradictions"]:
            lines.append(f"\n  {_display(c['speaker'], name_map)} — topic: {c['shared_topic']}")
            a = c["statement_a"]
            b = c["statement_b"]
            lines.append(f"    A ({a['timestamp']}): \"{a['text'][:120]}\" "
                         f"[stress={a['stress']:.2f}, rate={a['rate_change']}, fillers={a['fillers_nearby']}]")
            lines.append(f"    B ({b['timestamp']}): \"{b['text'][:120]}\" "
                         f"[stress={b['stress']:.2f}, rate={b['rate_change']}, fillers={b['fillers_nearby']}]")
            lines.append(f"    Voice delta: stress {c['stress_multiplier']}× between statements")

    if ca.get("voice_text_moments"):
        lines.append("\n=== VOICE ANOMALIES — WHAT WAS BEING SAID ===")
        for vt in ca["voice_text_moments"][:5]:
            lines.append(f"\n  {vt['timestamp']}: {_display(vt['speaker'], name_map)} "
                         f"stress={vt['stress']:.2f} ({vt['multiplier']}× baseline)")
            lines.append(f"    Said: \"{vt['text'][:150]}\"")
            lines.append(f"    Rate: {vt['rate_change']} | Fillers: {vt['filler_density']} | "
                         f"Pitch: {vt['pitch_state']}")

    if ca.get("vocal_profiles"):
        lines.append("\n=== VOCAL CONSISTENCY PROFILES ===")
        for spk, vp in ca["vocal_profiles"].items():
            lines.append(f"\n  {_display(spk, name_map)}: {vp['total_statements']} statements")
            lines.append(f"    Baseline: {vp['at_baseline_pct']}% | "
                         f"Moderate: {vp['moderate_pct']}% | "
                         f"Elevated: {vp['elevated_pct']}%")
            if vp["sensitive_topics"]:
                tops = ", ".join(
                    f"\"{t['topic']}\" ({t['avg_stress_ratio']}×)"
                    for t in vp["sensitive_topics"]
                )
                lines.append(f"    Sensitive topics: {tops}")
            if vp["stable_topics"]:
                stables = ", ".join(
                    f"\"{t['topic']}\" ({t['avg_stress_ratio']}×)"
                    for t in vp["stable_topics"]
                )
                lines.append(f"    Stable topics: {stables}")
```

---

## Change 3: JSON Schema in _build_prompt()

```json
"credibility_analysis": {
  "position_evolutions": [
    {
      "speaker": "Name",
      "topic": "delivery timeline",
      "evolution_summary": "Commitment softened across 3 revisions over 13 minutes. Voice stress escalated 2.9× from first to last statement. Speech rate decelerated progressively. Each revision was vocally harder to deliver than the previous one.",
      "first_statement": {"text": "...", "timestamp": "5:00", "stress": 0.18},
      "last_statement": {"text": "...", "timestamp": "18:00", "stress": 0.52},
      "stress_trajectory": "escalating",
      "revision_count": 3
    }
  ],
  "contradictions": [
    {
      "speaker": "Name",
      "topic": "nighttime location",
      "statement_a": {"text": "...", "timestamp": "2:30", "stress": 0.22, "rate_change": "+5%"},
      "statement_b": {"text": "...", "timestamp": "18:45", "stress": 0.71, "rate_change": "-35%"},
      "voice_assessment": "Voice stress increased 3.2× at the revision. Speech rate dropped 35%. Filler density spiked. The original was delivered at baseline voice; the revision carried significant acoustic load."
    }
  ],
  "voice_text_highlights": [
    {
      "timestamp": "8:30",
      "speaker": "Name",
      "stress_multiplier": "3.1× baseline",
      "transcript": "The pricing is... um... competitive for what you're getting",
      "voice_state": "stress=0.78, rate -35%, pitch elevated, 1 filler",
      "interpretation": "Multi-signal voice convergence during the single most sensitive word in the sentence"
    }
  ],
  "vocal_profiles": [
    {
      "speaker": "Name",
      "baseline_pct": 66,
      "elevated_pct": 8,
      "sensitive_topics": ["pricing (2.4× baseline)", "competitor (1.8×)"],
      "stable_topics": ["product features (0.9×)", "team intro (0.8×)"],
      "assessment": "Voice was stable across most of the session. Deviation concentrated on pricing and competitor discussion — these are the topics where the speaker was working hardest vocally."
    }
  ]
}
```

Structure requirements instruction:

```
"CREDIBILITY ANALYSIS: Using the pre-computed position evolutions, contradictions,
voice-text moments, and vocal profiles from the context data, produce ANALYTICAL
interpretations. Do NOT repeat the raw data. INTERPRET the patterns.

For position_evolutions: describe how the speaker's position EVOLVED. Use narrative:
'Initially committed to X with calm, steady delivery. Over 13 minutes, softened
through two intermediate positions to Y, with voice stress escalating at each revision.'

For contradictions: assess what the VOICE EVIDENCE tells us about each contradiction.
A calm-to-stressed reversal means something different than a stressed-to-stressed one.
A contradiction where both statements were calm is likely a genuine update.
A contradiction where the second statement carried 3× voice stress suggests the
revision was cognitively significant.

For voice_text_highlights: explain WHY this moment matters. 'Stress peaked during
pricing discussion' is description. 'The only multi-signal voice convergence in the
entire session occurred on the word competitive during pricing — this was the moment
the speaker found hardest to deliver' is analysis.

For vocal_profiles: interpret what topic-specific deviation means. '8% elevated
statements, all concentrated on two topics' is more revealing than '8% elevated.'

IMPORTANT: Voice stress indicates cognitive load and arousal, NOT deception.
A speaker may be stressed because they are lying, or because they are anxious,
or because the topic is personally sensitive, or because they fear being
disbelieved despite telling the truth. Always present multiple interpretations."
```

---

## Change 4: _parse_narrative_response()

```python
    "credibility_analysis": parsed.get("credibility_analysis", {}),
```

---

## Change 5: _fallback_narrative()

```python
    ca = credibility_analysis or {}

    credibility = {
        "position_evolutions": [],
        "contradictions": [],
        "voice_text_highlights": [],
        "vocal_profiles": [],
    }

    for spk, evolutions in ca.get("position_evolutions", {}).items():
        for ev in evolutions[:2]:
            first = ev["positions"][0] if ev["positions"] else {}
            last = ev["positions"][-1] if ev["positions"] else {}
            credibility["position_evolutions"].append({
                "speaker": _display(spk, name_map),
                "topic": ev["topic"],
                "evolution_summary": (
                    f"{ev['count']} statements. Voice {ev['stress_trend']} "
                    f"({ev['stress_first']:.2f} → {ev['stress_last']:.2f}, "
                    f"{ev['stress_multiplier']}×). "
                    f"{'Position reversed.' if ev['has_reversal'] else 'Position consistent.'}"
                ),
                "first_statement": {
                    "text": first.get("text", ""), "timestamp": first.get("timestamp", ""),
                    "stress": first.get("voice_stress", 0),
                },
                "last_statement": {
                    "text": last.get("text", ""), "timestamp": last.get("timestamp", ""),
                    "stress": last.get("voice_stress", 0),
                },
                "stress_trajectory": ev["stress_trend"],
                "revision_count": ev["count"],
            })

    for c in ca.get("contradictions", []):
        credibility["contradictions"].append({
            "speaker": _display(c["speaker"], name_map),
            "topic": c["shared_topic"],
            "statement_a": c["statement_a"],
            "statement_b": c["statement_b"],
            "voice_assessment": (
                f"Voice stress {c['stress_multiplier']}× between statements. "
                f"Rate change: {c['statement_a']['rate_change']} → {c['statement_b']['rate_change']}."
            ),
        })

    for vt in ca.get("voice_text_moments", [])[:5]:
        credibility["voice_text_highlights"].append({
            "timestamp": vt["timestamp"],
            "speaker": _display(vt["speaker"], name_map),
            "stress_multiplier": f"{vt['multiplier']}× baseline",
            "transcript": vt["text"],
            "voice_state": (
                f"stress={vt['stress']:.2f}, rate {vt['rate_change']}, "
                f"pitch {vt['pitch_state']}, fillers {vt['filler_density']}"
            ),
            "interpretation": "Multi-signal voice event during this statement.",
        })

    for spk, vp in ca.get("vocal_profiles", {}).items():
        credibility["vocal_profiles"].append({
            "speaker": _display(spk, name_map),
            "baseline_pct": vp["at_baseline_pct"],
            "elevated_pct": vp["elevated_pct"],
            "sensitive_topics": [
                f"{t['topic']} ({t['avg_stress_ratio']}×)"
                for t in vp.get("sensitive_topics", [])
            ],
            "stable_topics": [
                f"{t['topic']} ({t['avg_stress_ratio']}×)"
                for t in vp.get("stable_topics", [])
            ],
            "assessment": (
                f"{vp['at_baseline_pct']}% at baseline, "
                f"{vp['elevated_pct']}% elevated. "
                f"Sensitive topics: "
                f"{', '.join(t['topic'] for t in vp.get('sensitive_topics', []))}"
            ),
        })
```

---

## Change 6: Frontend

### Credibility Analysis Section

Place AFTER Notes, BEFORE Risk Assessment.

4 subsections:
- **Position Evolution** — per-speaker, per-topic, first quote vs last quote with stress trajectory
- **Contradictions** — side-by-side statement cards with voice state at each
- **Voice-Text Highlights** — stress peak moments with transcript text
- **Vocal Profiles** — per-speaker stress distribution bar + sensitive/stable topic lists

Each subsection renders only if its array has entries. Empty arrays = no render.

---

## Updated Report Section Order

```
1. General Summary
2. Executive Summary
3. Key Facts & Commitments
4. Notes (topic-grouped)
5. Credibility Analysis (NEW — transcript + voice only)
   5a. Position Evolution
   5b. Statement Contradictions
   5c. Voice-Text Highlights
   5d. Vocal Consistency Profiles
6. Risk Assessment (interrogation only)
7. Speaker Analyses
8. Key Moments
9. Cross-Modal Insights
10. Recommendations
11. Action Items
```

---

## Files Modified:

### Backend (1):
1. **services/fusion_agent/narrative.py**:
   - New: `_build_credibility_analysis()` (~180 lines)
   - New: 8 helper functions (~80 lines)
   - Modified: `generate_session_narrative()` — call analysis, pass to context + fallback (~3 lines)
   - Modified: `_build_context()` — add credibility_analysis parameter + 4 context sections (~50 lines)
   - Modified: `_build_prompt()` — JSON schema + structure instructions (~25 lines)
   - Modified: `_parse_narrative_response()` — parse new field (~1 line)
   - Modified: `_fallback_narrative()` — build credibility from pre-computed data (~60 lines)

### Frontend (1):
2. **SessionDetail.tsx** — Credibility Analysis section with 4 subsections (~70 lines)