# NEXUS — Signal Display + Quality Fixes + Analysis Layers Implementation

## Context

5,048 raw signals for a 10-minute session. Currently displayed as individual
colored badges in the video sidebar — unreadable wall of icons, colors, and
abbreviations. A new user sees "🟠 Biting Their Tongue" and has no idea what
category of behavior this represents or why it matters.

**New approach:** Replace all colored badges with plain English WORD LABELS.
Each word represents a behavioral CATEGORY that groups multiple signals.
"Stressed" immediately tells anyone what's happening — no legend, no color
chart, no learning curve.

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read ALL these files COMPLETELY — do not skim:
   - `services/fusion_agent/narrative.py` (1591L) — narrative generation
   - `services/fusion_agent/fusion_service.py` (580L) — orchestrator
   - `services/fusion_agent/fusion_engine.py` (359L) — pairwise rules
   - `services/fusion_agent/compound_patterns.py` (563L) — 12 compound patterns
   - `dashboard/src/components/VideoSignalPlayer.tsx` (1611L) — video sidebar
   - `dashboard/src/components/SessionDetail.tsx` (2024L) — report tab
   - `dashboard/src/signalDisplayConfig.ts` (885L) — signal display config

2. Understand the DATA FLOW:
   ```
   Voice Agent → voice_signals[] (Speaker_N)
   Language Agent → language_signals[] (Speaker_N)
   Video Agent → video_signals[] (Face_N)
   Conversation Agent → conversation_signals[] (Speaker_N)
              ↓
   fusion_service.py → compound_patterns → temporal_patterns → all_fusion_signals (~22)
              ↓
   generate_session_narrative() receives:
     voice_summary, language_summary, fusion_signals (only ~22),
     transcript_segments, video_summary, conversation_summary
     *** does NOT receive raw voice/language/conversation signals ***
              ↓
   Frontend receives ALL 5,048 via GET /video-signals
   Frontend receives report via GET /sessions/{id}/report
   ```

3. Use proper OOP and DSA:
   - **HashMap**: O(1) signal_type → category lookup
   - **bisect**: O(log N) segment-to-signal mapping
   - **defaultdict**: grouping by speaker, window, topic
   - **frozenset**: immutable category membership sets
   - **Sliding window**: phase boundary detection

---

## THE CATEGORY SYSTEM

Every signal in the system maps to ONE behavioral category. Each category
has ONE plain English word. This mapping is the foundation of the entire
display system — sidebar, transcript, timeline, and report all use it.

```python
# Backend: services/fusion_agent/narrative.py (module-level constant)
# Frontend: mirror in VideoSignalPlayer.tsx

BEHAVIORAL_CATEGORIES = {
    "Stressed": {
        "signals": frozenset({
            "vocal_stress_score",       # voice: when value > 1.5× baseline
            "facial_stress",            # face: elevated facial muscle tension
            "shoulder_tension",         # body: when elevated above baseline
            "tension_cluster",          # compound: multi-signal stress
            "freezing_response",        # interrogation: acute freeze
            "stress_anxiety_cluster",   # compound: body stress cluster
            "agitated_high_arousal_tone", # voice: high arousal
        }),
        "description": "This person is under pressure right now",
    },
    "Guarded": {
        "signals": frozenset({
            "emotional_suppression",    # compound: masking true state
            "hidden_disagreement",      # body: lip purse + corroborating cues
            "lip_pursing",              # face: suppressed reaction
            "motor_inhibition",         # body: movement below baseline
            "arms_crossed",             # body: barrier posture
            "self_touch",               # body: self-soothing
            "face_region_touch",        # body: specific zone touch
        }),
        "description": "This person is holding something back",
    },
    "Engaged": {
        "signals": frozenset({
            "head_nod",                 # body: agreement/listening
            "smile_type",              # face: genuine or social smile
            "body_lean",               # body: forward lean only
            "buying_signal",            # language: purchase intent
            "facial_engagement",        # face: high engagement composite
            "attention_level",          # gaze: high attention
            "genuine_engagement",       # compound: multi-modal engagement
            "gesture_animation",        # body: animated gestures
            "laughter",                 # face: open laugh
        }),
        "description": "This person is actively interested",
    },
    "Deflecting": {
        "signals": frozenset({
            "gaze_direction_shift",     # gaze: looking away
            "sustained_distraction",    # gaze: prolonged off-screen
            "topic_shift",              # language: changed subject
            "active_disengagement",     # compound: multi-modal withdrawal
            "blink_rate_anomaly",       # gaze: abnormal blink rate
        }),
        "description": "This person is avoiding something",
    },
    "Resistant": {
        "signals": frozenset({
            "head_shake",               # body: disagreement
            "objection_signal",         # language: explicit objection
            "conflict_detection",       # conversation: multi-signal conflict
            "frustration_cluster",      # body: frustration indicators
            "resistance_hardening",     # interrogation: increasing resistance
            "head_body_incongruence",   # body: nod but pulling back
        }),
        "description": "This person is pushing back",
    },
    "Processing": {
        "signals": frozenset({
            "pause_classification",     # voice: extended or strategic pause
            "strategic_pause",          # voice: deliberate thinking pause
            "evaluation_cluster",       # body: chin touch + contemplation
            "cognitive_overload",       # compound: multi-signal overload
            "evidence_response_processing_delay", # interrogation: delayed response
            "decision_engagement",      # compound: approaching decision
        }),
        "description": "This person is thinking through something",
    },
    "Dominant": {
        "signals": frozenset({
            "interruption_event",       # voice: verbal interruption
            "dominance_display",        # compound: power posture + voice
            "dominance_score",          # conversation: high talk dominance
            "arm_posture",             # body: expansive arms
            "finger_steepling",         # body: power gesture
            "peak_performance",         # compound: confident multi-modal
        }),
        "description": "This person is controlling the conversation",
    },
}

# Reverse lookup: signal_type → category name
SIGNAL_TO_CATEGORY: dict[str, str] = {}
for cat_name, cat_data in BEHAVIORAL_CATEGORIES.items():
    for sig_type in cat_data["signals"]:
        SIGNAL_TO_CATEGORY[sig_type] = cat_name
```

### Rules for category assignment per window:
1. A category is ACTIVE for a face/speaker when ANY signal in that category
   is active in the current window with confidence ≥ 0.30
2. Multiple categories can be active simultaneously ("Stressed" + "Guarded")
3. Show NOTHING when no category qualifies — no "Comfortable" or "Neutral" label
4. For voice signals (vocal_stress_score), the category "Stressed" only activates
   when stress > 1.5× speaker's baseline — not on every window
5. For body_lean, only "Engaged" when value_text is forward_lean (not backward)

### Signals that DON'T map to any category (hidden from sidebar):
```
presence_detected     — structural marker, not behavioral
vocal_stress_score    — shown ONLY via "Stressed" category when > 1.5× baseline
tone_classification   — subsumed by Stressed/Engaged depending on tone
sentiment_score       — shown in transcript inline, not sidebar
posture               — subsumed by categories (slump → Guarded, upright → Dominant)
energy_level          — not independently meaningful
speech_rate_anomaly   — not independently meaningful
power_language_score  — shown in report, not sidebar
```

---

## PART 1: Backend Signal Quality Fixes

### Fix 1.1: `tone_classification` — Suppress Neutral Below 0.40

**File:** `services/voiceAgent/rules.py`

Find the `tone_classification` emission. Currently the final else-branch
emits `tone="neutral"` at confidence ~0.30 every window:

```python
# Add gate before emission:
if tone["tone"] == "neutral" and tone["confidence_raw"] < 0.40:
    pass  # suppress — neutral at low confidence is noise
else:
    _add("VOICE-TONE-03", "tone_classification", ...)
```

### Fix 1.2: `shoulder_tension` — Baseline Delta

**File:** `services/video_agent/body_rules.py`

Replace absolute `shoulder_angle_std > 8.0` with baseline comparison:

```python
baseline_std = bl.shoulder_angle_std if bl else 5.0
delta_std = shoulder_angle_std - baseline_std
if delta_std > 3.0:  # elevated above speaker's normal
    confidence = min((delta_std / 8.0) * conf_mult, 0.50)
    # emit signal with display_mode metadata
    signal["metadata"]["display_mode"] = "continuous"
```

### Fix 1.3: `head_nod` — Raise Velocity Threshold

**File:** `services/video_agent/body_rules.py`

```python
NOD_VELOCITY_MIN = 22.0  # was 15.0 — picks up micro-adjustments
```

### Fix 1.4: `motor_inhibition` — Dynamic Confidence + Metadata

**File:** `services/video_agent/interrogation_rules.py`

```python
reduction_pct = 1.0 - (current_velocity / max(baseline_velocity, 0.001))
reduction_pct = max(0.0, min(1.0, reduction_pct))
confidence = min(reduction_pct * 0.50, quality_tier_cap)
signal["metadata"]["display_mode"] = "continuous"
```

Keep generating per-window — fusion engine needs temporal co-occurrence.

### Fix 1.5: `posture` — Add Display Metadata

**File:** `services/video_agent/body_rules.py`

```python
signal["metadata"]["display_mode"] = "continuous"
```

### Fix 1.6: Merge `energy_level` + `volume_shift`

**File:** `services/voiceAgent/rules.py`

Remove the `volume_shift` emission block (same 6dB trigger, same data as
`energy_level`). Keep `energy_level` only.

---

## PART 2: Backend Analysis Layers

### Critical Data Flow Fix

**File:** `services/fusion_agent/fusion_service.py`

`generate_session_narrative()` currently receives ONLY the 22 fusion output
signals via `fusion_signals=all_fusion_signals`. The multi-layer analysis
needs raw voice + language + conversation signals.

```python
# Around line 422, before the generate_session_narrative call:
all_audio_signals = pure_voice_dicts + language_dicts
conv_sigs = conversation_summary.get("signals", []) if conversation_summary else []
all_audio_signals += conv_sigs

report = await generate_session_narrative(
    ...,
    all_audio_signals=all_audio_signals,   # ← NEW parameter
)
```

**File:** `services/fusion_agent/narrative.py`

Add `all_audio_signals` parameter to `generate_session_narrative()`.

### New Function: `_build_signal_analysis()`

**File:** `services/fusion_agent/narrative.py`

Computes 4 analysis layers from raw audio signals. Called BEFORE `_build_context()`.

```python
def _build_signal_analysis(
    transcript_segments: list[dict],
    all_audio_signals: list[dict],
    voice_summary: dict,
    speakers: list[str],
    name_map: dict[str, str],
    entities: dict,
) -> dict:
    """
    Pre-compute multi-layer audio signal analysis.
    Returns structured data for both LLM context and fallback narrative.

    DSA:
      - bisect: O(log N) segment lookup
      - defaultdict: O(1) per-speaker grouping
      - frozenset: O(1) category membership check
    """
```

#### Layer 1: Segment Behavioral Profiles

For each transcript segment, find overlapping audio signals, determine which
CATEGORIES are active, produce a list of active category labels.

```python
# Map signals to segments using bisect
seg_starts = [s.get("start_ms", 0) for s in transcript_segments]
signals_by_seg: dict[tuple[str, int], list[dict]] = defaultdict(list)

for sig in all_audio_signals:
    w_start = sig.get("window_start_ms", 0)
    spk = sig.get("speaker_id", "")
    if not spk:
        continue
    idx = bisect.bisect_right(seg_starts, w_start) - 1
    if 0 <= idx < len(transcript_segments):
        seg = transcript_segments[idx]
        seg_end = seg.get("end_ms", seg.get("start_ms", 0) + 2000)
        if seg.get("start_ms", 0) < sig.get("window_end_ms", 0) and w_start < seg_end:
            signals_by_seg[(spk, idx)].append(sig)

# For each segment, determine active categories
segment_profiles = []
for (spk, seg_idx), sigs in signals_by_seg.items():
    seg = transcript_segments[seg_idx]
    text = seg.get("text", "").strip()
    if not text or len(text) < 10:
        continue

    # Get speaker's stress baseline for the "Stressed" threshold
    spk_avg = voice_summary.get("per_speaker", {}).get(spk, {}).get("avg_stress", 0.25)

    # Determine active categories
    active_categories = []
    for sig in sigs:
        sig_type = sig.get("signal_type", "")
        conf = sig.get("confidence", 0)
        if conf < 0.30:
            continue

        cat = SIGNAL_TO_CATEGORY.get(sig_type)
        if not cat:
            continue

        # Special gate: "Stressed" requires stress > 1.5× baseline
        if cat == "Stressed" and sig_type == "vocal_stress_score":
            stress_val = float(sig.get("value", 0))
            if stress_val < spk_avg * 1.5:
                continue

        # Special gate: "Engaged" body_lean only for forward
        if sig_type == "body_lean" and sig.get("value_text", "") != "forward_lean":
            continue

        if cat not in active_categories:
            active_categories.append(cat)

    # Get stress ratio for display
    stress_sig = next((s for s in sigs if s.get("signal_type") == "vocal_stress_score"), None)
    stress_val = float(stress_sig.get("value", 0)) if stress_sig else 0
    stress_ratio = stress_val / max(spk_avg, 0.01) if stress_val > 0 else 1.0

    # Get language classification
    intent_sig = next((s for s in sigs if s.get("signal_type") == "utterance_intent"), None)
    classification = intent_sig.get("value_text", "") if intent_sig else ""

    # Intensity = sum of confidences (how many channels firing)
    intensity = sum(s.get("confidence", 0) for s in sigs if s.get("confidence", 0) >= 0.30)

    m, s_val = divmod(seg.get("start_ms", 0) // 1000, 60)
    segment_profiles.append({
        "speaker": spk,
        "timestamp": f"{m}:{s_val:02d}",
        "timestamp_ms": seg.get("start_ms", 0),
        "text": text[:200],
        "categories": active_categories,       # ["Stressed", "Guarded"]
        "stress_ratio": round(stress_ratio, 1),
        "classification": classification,
        "intensity": round(intensity, 1),
        "signal_count": len([s for s in sigs if s.get("confidence", 0) >= 0.30]),
    })
```

#### Layer 2: Response Pattern Timeline

Group segment profiles by speaker. Detect phases using consecutive dominant
category runs:

```python
# Phase = run of 2+ consecutive segments where the dominant category stays the same
# Dominant category = the first category in the list (highest priority signal fired first)
# Phase boundary = dominant category changes

# For each speaker:
#   Sort profiles by timestamp_ms
#   Walk through, track current dominant category
#   When it changes for 2+ consecutive segments, start new phase
#   Each phase: name, start, end, dominant_category, avg_stress_ratio, key_segment

# Output per speaker:
# {
#   "phases": [
#     {"name": "Denial", "start": "0:23", "end": "0:46", "dominant": "Resistant",
#      "avg_stress_ratio": 1.1, "key_segment": {...}},
#     {"name": "Evidence Impact", "start": "1:00", "end": "1:22", "dominant": "Stressed",
#      "avg_stress_ratio": 2.4, "key_segment": {...}},
#   ],
#   "trajectory_summary": "Resistant → Stressed → Guarded → Deflecting → Resistant"
# }
```

#### Layer 3: Stimulus-Response Pairs

For each segment from the dominant speaker, capture the responding speaker's
categories in their NEXT segment:

```python
# Determine stimulus speaker (most talk time from voice_summary or conversation)
# For each stimulus segment:
#   Find next segment from a different speaker
#   Record: stimulus text, response text, response categories, stress_ratio
#   Impact: "high" if response has "Stressed" + stress > 2.0×
#           "medium" if response has any non-neutral category
#           "low" if response is empty categories
```

#### Layer 4: Topic Voice Signatures

Group segment profiles by entity extraction topics. Per speaker per topic,
compute which categories appeared most:

```python
# For each topic in entities["topics"]:
#   Collect segment_profiles within topic time range
#   Per speaker: count occurrences of each category, avg stress_ratio
#   dominant_category = most frequent non-empty category
#   sensitivity_rank = ordered by avg_stress_ratio descending
```

### Add to `_build_context()`

Pass `signal_analysis` to `_build_context()`. Add 4 context sections using
the category labels:

```python
sa = signal_analysis or {}

if sa.get("segment_profiles"):
    top = sorted(sa["segment_profiles"], key=lambda p: -p.get("intensity", 0))[:15]
    lines.append("\n=== BEHAVIORAL HOTSPOTS (top moments by signal intensity) ===")
    for p in top:
        cats = ", ".join(p["categories"]) if p["categories"] else "Neutral"
        lines.append(
            f"  [{p['timestamp']}] {_display(p['speaker'], name_map)}: "
            f'"{p["text"][:100]}" — {cats} '
            f"(stress {p['stress_ratio']:.1f}×, {p['signal_count']} signals)"
        )

if sa.get("response_timeline"):
    lines.append("\n=== BEHAVIORAL TRAJECTORY (per speaker) ===")
    for spk, tl in sa["response_timeline"].items():
        lines.append(f"\n  {_display(spk, name_map)}: {tl.get('trajectory_summary', '')}")
        for phase in tl.get("phases", [])[:8]:
            lines.append(
                f"    {phase['start']}-{phase['end']}: {phase['name']} "
                f"— {phase['dominant']}, stress={phase['avg_stress_ratio']:.1f}×"
            )

if sa.get("stimulus_response_pairs"):
    high_impact = [p for p in sa["stimulus_response_pairs"] if p.get("impact") in ("high", "medium")][:8]
    if high_impact:
        lines.append("\n=== STIMULUS → RESPONSE ANALYSIS ===")
        for p in high_impact:
            stim = p["stimulus"]
            resp = p["response"]
            resp_cats = ", ".join(resp["categories"]) if resp["categories"] else "Neutral"
            lines.append(
                f'  [{stim["timestamp"]}] {_display(stim["speaker"], name_map)}: '
                f'"{stim["text"][:80]}"'
            )
            lines.append(
                f'    → [{resp["timestamp"]}] {_display(resp["speaker"], name_map)}: '
                f'"{resp["text"][:80]}" — {resp_cats} (stress {resp["stress_ratio"]:.1f}×)'
            )

if sa.get("topic_voice_signatures"):
    lines.append("\n=== TOPIC SENSITIVITY MAP ===")
    for spk, data in sa["topic_voice_signatures"].items():
        lines.append(f"\n  {_display(spk, name_map)}:")
        for t in data.get("topics", [])[:5]:
            lines.append(
                f"    #{t['sensitivity_rank']} \"{t['topic']}\" — "
                f"dominant={t['dominant_category']}, stress={t['avg_stress_ratio']:.1f}×"
            )
```

### Add to JSON Schema in `_build_main_prompt()`

```json
"behavioral_analysis": {
    "hotspots": [
        {
            "timestamp": "1:14",
            "speaker": "Name",
            "text": "What was said",
            "categories": ["Stressed", "Guarded"],
            "stress_ratio": 2.9,
            "interpretation": "Why this moment matters"
        }
    ],
    "speaker_trajectories": {
        "Speaker_1": {
            "summary": "Narrative of how behavioral state evolved",
            "phases": ["Resistant → Stressed → Guarded → Deflecting → Resistant"],
            "turning_points": ["1:14 — first admission under evidence pressure"]
        }
    },
    "exchange_analysis": [
        {
            "stimulus": "What the interrogator/seller said",
            "response_categories": ["Stressed", "Guarded"],
            "stress_impact": 2.9,
            "interpretation": "What this exchange reveals"
        }
    ],
    "topic_sensitivity": {
        "Speaker_1": {
            "most_sensitive": "walmart/cart",
            "least_sensitive": "general denial",
            "interpretation": "What the topic hierarchy reveals"
        }
    }
}
```

### Add to `_parse_main_response()` and `_fallback_narrative()`

Parse `behavioral_analysis` from LLM response. Build from pre-computed data
in fallback path.

---

## PART 3: Frontend Display Changes

### Change 3.1: Category Label System in Sidebar

**File:** `dashboard/src/components/VideoSignalPlayer.tsx`

Replace the entire badge rendering with category word labels.

#### Step A: Define category mapping (mirror backend)

```typescript
const SIGNAL_TO_CATEGORY: Record<string, string> = {
  // Stressed
  vocal_stress_score: "Stressed",
  facial_stress: "Stressed",
  shoulder_tension: "Stressed",
  freezing_response: "Stressed",
  agitated_high_arousal_tone: "Stressed",
  // Guarded
  emotional_suppression: "Guarded",
  hidden_disagreement: "Guarded",
  lip_pursing: "Guarded",
  motor_inhibition: "Guarded",
  arms_crossed: "Guarded",
  self_touch: "Guarded",
  face_region_touch: "Guarded",
  // Engaged
  head_nod: "Engaged",
  smile_type: "Engaged",
  buying_signal: "Engaged",
  facial_engagement: "Engaged",
  attention_level: "Engaged",
  genuine_engagement: "Engaged",
  gesture_animation: "Engaged",
  laughter: "Engaged",
  // Deflecting
  gaze_direction_shift: "Deflecting",
  sustained_distraction: "Deflecting",
  topic_shift: "Deflecting",
  active_disengagement: "Deflecting",
  blink_rate_anomaly: "Deflecting",
  // Resistant
  head_shake: "Resistant",
  objection_signal: "Resistant",
  conflict_detection: "Resistant",
  frustration_cluster: "Resistant",
  resistance_hardening: "Resistant",
  head_body_incongruence: "Resistant",
  // Processing
  pause_classification: "Processing",
  strategic_pause: "Processing",
  evaluation_cluster: "Processing",
  cognitive_overload: "Processing",
  evidence_response_processing_delay: "Processing",
  decision_engagement: "Processing",
  // Dominant
  interruption_event: "Dominant",
  dominance_display: "Dominant",
  dominance_score: "Dominant",
  arm_posture: "Dominant",
  finger_steepling: "Dominant",
  peak_performance: "Dominant",
};
```

#### Step B: Compute speaker stress baselines (once on load)

```typescript
const speakerBaselines = useMemo(() => {
  const baselines: Record<string, number> = {};
  const stressValues: Record<string, number[]> = {};
  for (const s of signals) {
    if (s.signal_type === "vocal_stress_score" && s.speaker_id) {
      (stressValues[s.speaker_id] ??= []).push(s.value ?? 0);
    }
  }
  for (const [spk, vals] of Object.entries(stressValues)) {
    baselines[spk] = vals.reduce((a, b) => a + b, 0) / vals.length;
  }
  return baselines;
}, [signals]);
```

#### Step C: Compute active categories per face

```typescript
function getActiveCategories(
  sigs: VideoSignal[],
  speakerId: string,
  baselines: Record<string, number>,
): string[] {
  const categories = new Set<string>();
  const baseline = baselines[speakerId] ?? 0.25;

  for (const s of sigs) {
    if ((s.confidence ?? 0) < 0.30) continue;
    if (s.signal_type === "presence_detected") continue;

    const cat = SIGNAL_TO_CATEGORY[s.signal_type];
    if (!cat) continue;

    // Gate: Stressed only when vocal stress > 1.5× baseline
    if (cat === "Stressed" && s.signal_type === "vocal_stress_score") {
      if ((s.value ?? 0) < baseline * 1.5) continue;
    }

    // Gate: Engaged body_lean only for forward
    if (s.signal_type === "body_lean" && s.value_text !== "forward_lean") continue;

    // Gate: Engaged attention_level only for high
    if (s.signal_type === "attention_level" && s.value_text !== "high_attention") continue;

    categories.add(cat);
  }

  return [...categories];
}
```

#### Step D: Replace badge rendering with word labels

Replace the existing badge map (lines ~1375-1400) with:

```tsx
{/* Replace the entire visibleSigs.map(...) block with: */}
{(() => {
  const categories = getActiveCategories(sigs, rawId, speakerBaselines);
  if (categories.length === 0 && hasPresence) {
    return <span className="px-2 text-[9px] text-white/25 italic">in frame</span>;
  }
  return categories.map((cat) => (
    <div
      key={cat}
      className="px-2.5 py-1 text-xs font-medium text-white/80 tracking-wide"
    >
      {cat}
    </div>
  ));
})()}
```

This renders:
```
Face_4
  face match (100%)
  Guarded
  Resistant

Face_0
  face match (100%)
  Stressed

Face_7
  face match (100%)
  Processing
```

No colors. No icons. No badges. Just words.

#### Step E: Remove the "Show more / Show less" toggle

The toggle exists because there were too many badges to show. With category
labels (max 7 categories, typically 1-3 active), everything fits. Remove the
`showExpanded` state and the toggle button.

### Change 3.2: Transcript Inline Category Labels

**File:** `dashboard/src/components/SessionDetail.tsx`

Find where transcript segments render. After the segment text, show active
categories and stress ratio:

```typescript
// Import or define getActiveCategories (same logic as sidebar)
// In transcript segment render:

const segSignals = matchSignalsToSegment(signals, segment.start_ms, segment.end_ms, segment.speaker);
const categories = getActiveCategories(segSignals, segment.speaker, speakerBaselines);
const stressSig = segSignals.find(s => s.signal_type === "vocal_stress_score");
const baseline = speakerBaselines[segment.speaker] ?? 0.25;
const stressRatio = stressSig ? ((stressSig.value ?? 0) / Math.max(baseline, 0.01)) : null;

// Render after segment text:
{categories.length > 0 && (
  <span className="ml-2 text-[9px] font-medium text-white/50 tracking-wide">
    {categories.join(" · ")}
    {stressRatio && stressRatio > 1.3 && (
      <span className="ml-1 text-white/40">{stressRatio.toFixed(1)}×</span>
    )}
  </span>
)}
```

This renders:
```
[1:14] Speaker_1: "That's me. Okay."  Stressed · Guarded 2.9×
[1:22] Speaker_1: "I don't know who that person is."  Resistant
[2:39] Speaker_1: "Trash."  Stressed 2.2×
[4:42] Speaker_1: "we got into a little scuffle"  Guarded
```

### Change 3.3: Behavioral Analysis Report Section

**File:** `dashboard/src/components/SessionDetail.tsx`

Add after Notes section. Uses plain word labels throughout.

```tsx
{report?.behavioral_analysis && (
  <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
    <h2 className="mb-4 text-sm font-semibold text-nexus-text-primary">
      Behavioral Analysis
    </h2>

    {/* Hotspots */}
    {report.behavioral_analysis.hotspots?.length > 0 && (
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-nexus-text-muted uppercase tracking-wide mb-2">
          Key Moments
        </h3>
        {report.behavioral_analysis.hotspots.map((h, i) => (
          <div key={i} className="mb-1.5 flex items-start gap-2 text-sm
               cursor-pointer hover:bg-nexus-surface-hover rounded p-1.5"
               onClick={() => seekTo(parseTimestamp(h.timestamp))}>
            <span className="text-xs font-mono text-nexus-text-muted w-10 shrink-0">
              {h.timestamp}
            </span>
            <span className="font-medium shrink-0">{h.speaker}</span>
            <span className="text-nexus-text-secondary flex-1">
              "{h.text?.slice(0, 80)}"
            </span>
            <span className="text-[10px] font-medium text-nexus-text-muted shrink-0">
              {h.categories?.join(" · ")}
            </span>
          </div>
        ))}
      </div>
    )}

    {/* Speaker Trajectory */}
    {report.behavioral_analysis.speaker_trajectories &&
     Object.keys(report.behavioral_analysis.speaker_trajectories).length > 0 && (
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-nexus-text-muted uppercase tracking-wide mb-2">
          Behavioral Arc
        </h3>
        {Object.entries(report.behavioral_analysis.speaker_trajectories).map(([spk, tl]) => (
          <div key={spk} className="mb-3">
            <div className="text-xs font-medium mb-1">{spk}</div>
            <div className="text-xs text-nexus-text-secondary mb-2 italic">
              {tl.summary}
            </div>
            <div className="flex flex-wrap gap-1">
              {tl.phases?.map((phase, i) => (
                <span key={i} className="rounded px-2 py-0.5 text-[10px] font-medium
                     bg-nexus-surface-hover text-nexus-text-primary">
                  {phase}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    )}

    {/* Exchange Analysis */}
    {report.behavioral_analysis.exchange_analysis?.length > 0 && (
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-nexus-text-muted uppercase tracking-wide mb-2">
          Key Exchanges
        </h3>
        {report.behavioral_analysis.exchange_analysis.map((ex, i) => (
          <div key={i} className="mb-2 rounded border border-nexus-border p-2.5">
            <div className="text-xs text-nexus-text-muted">{ex.stimulus}</div>
            <div className="mt-1 flex items-center gap-2">
              <span className="text-nexus-text-muted">→</span>
              <span className="text-sm font-medium">
                {ex.response_categories?.join(" · ")}
              </span>
              {ex.stress_impact > 1.5 && (
                <span className="text-[10px] text-nexus-text-muted">
                  {ex.stress_impact.toFixed(1)}× baseline
                </span>
              )}
            </div>
            {ex.interpretation && (
              <div className="mt-1 text-xs text-nexus-text-secondary italic">
                {ex.interpretation}
              </div>
            )}
          </div>
        ))}
      </div>
    )}

    {/* Topic Sensitivity */}
    {report.behavioral_analysis.topic_sensitivity &&
     Object.keys(report.behavioral_analysis.topic_sensitivity).length > 0 && (
      <div>
        <h3 className="text-xs font-semibold text-nexus-text-muted uppercase tracking-wide mb-2">
          Topic Sensitivity
        </h3>
        {Object.entries(report.behavioral_analysis.topic_sensitivity).map(([spk, data]) => (
          <div key={spk} className="mb-2">
            <div className="text-xs font-medium mb-1">{spk}</div>
            {data.topics?.map((t, i) => (
              <div key={i} className="flex items-center gap-2 mb-0.5 text-xs">
                <span className="text-nexus-text-muted w-4">#{t.rank}</span>
                <span className="flex-1">{t.topic}</span>
                <span className="text-nexus-text-muted w-16">
                  {t.dominant_category}
                </span>
                <span className="text-nexus-text-muted w-8">
                  {t.stress_ratio.toFixed(1)}×
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    )}
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
5. Behavioral Analysis (NEW)
   5a. Key Moments (hotspots with category labels)
   5b. Behavioral Arc (phase sequence per speaker)
   5c. Key Exchanges (stimulus → response categories)
   5d. Topic Sensitivity (ranked by stress ratio)
6. Credibility Analysis (interrogation only)
7. Risk Assessment (interrogation only)
8. Speaker Analyses
9. Key Moments
10. Cross-Modal Insights
11. Recommendations
12. Action Items
```

---

## What NOT to Change

- Signal generation frequency — keep per-window (fusion needs temporal co-occurrence)
- `presence_detected` — keep as structural marker (confidence 0.0, hidden)
- `vocal_stress_score` — keep firing every window (baseline tracking needs it)
- `sentiment_score` — keep firing every segment (language analysis needs it)
- Fusion engine rules — compound patterns are correctly selective
- `facial_stress` cap at 0.38, `body_lean` cap at 0.38, `deception_cluster` cap at 0.50
- ASD path, time_overlap path, face-speaker mapping logic

---

## Expected Results

### Sidebar:
```
BEFORE:                              AFTER:
8+ colored badges flickering    →    1-3 plain word labels, stable
🔴🟡🔵🟠 unreadable             →    Stressed · Guarded (readable)
New user confused               →    New user instantly understands
```

### Transcript:
```
BEFORE:
[1:14] Speaker_1: "That's me. Okay."   😟 👍

AFTER:
[1:14] Speaker_1: "That's me. Okay."   Stressed · Guarded 2.9×
```

### Report:
```
BEFORE: Executive summary + key moments only

AFTER: + Key Moments (clickable timestamps with category labels)
       + Behavioral Arc (Resistant → Stressed → Guarded → Deflecting)
       + Key Exchanges (evidence → Stressed response)
       + Topic Sensitivity (#1 cart/disposal, #2 hotel room)
```

### Signal quality:
```
tone_classification:  355 → ~175  (neutral suppressed)
shoulder_tension:     520 → ~50   (baseline delta)
head_nod:             242 → ~60   (threshold raised)
volume_shift:          30 →  0    (merged)
Total:              5,048 → ~4,300 (display unchanged, quality improved)
```

---

## Files Modified:

### Backend (4):
1. **services/voiceAgent/rules.py** — suppress neutral tone, remove volume_shift (~18 lines)
2. **services/video_agent/body_rules.py** — shoulder baseline delta, nod threshold, display_mode metadata (~10 lines)
3. **services/fusion_agent/fusion_service.py** — collect + pass all_audio_signals (~5 lines)
4. **services/fusion_agent/narrative.py** — `_build_signal_analysis()` + 4 layers + context + schema + fallback (~350 lines new, ~50 lines modified)

### Frontend (2):
5. **VideoSignalPlayer.tsx** — SIGNAL_TO_CATEGORY map, getActiveCategories, replace badges with word labels, remove show more toggle (~80 lines replaced)
6. **SessionDetail.tsx** — transcript inline categories, BehavioralAnalysis report section (~100 lines new)