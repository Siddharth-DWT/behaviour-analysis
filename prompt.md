# NEXUS — Signal Display + Quality Fixes + Analysis Layers Implementation

## Context

5,048 raw signals for a 10-minute session. Currently displayed as individual
2-second badges in the video sidebar. Continuous states (motor_inhibition,
posture, shoulder_tension) flicker as badges every 2 seconds. The narrative
report receives only 22 fusion signals and summaries, not the raw 5,048.

This prompt covers THREE layers of work:
1. **Backend signal quality** — fix over-firing, confidence, and redundancy
2. **Backend analysis** — new `behavioral_analysis` field in report
3. **Frontend display** — composite labels, continuous bars, timeline, transcript inline

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read ALL these files COMPLETELY — do not skim:
   - `services/fusion_agent/narrative.py` (1591L) — LLM narrative generation
   - `services/fusion_agent/fusion_service.py` (580L) — orchestrates fusion + calls narrative
   - `services/fusion_agent/fusion_engine.py` (359L) — pairwise cross-modal rules
   - `services/fusion_agent/compound_patterns.py` (563L) — 12 compound patterns
   - `dashboard/src/components/VideoSignalPlayer.tsx` (1611L) — video player sidebar
   - `dashboard/src/components/SessionDetail.tsx` (2024L) — report tab rendering
   - `dashboard/src/signalDisplayConfig.ts` (885L) — signal display config with priority

2. Understand the DATA FLOW (this is critical — getting it wrong breaks everything):
   ```
   Voice Agent → voice_signals[] (keyed by Speaker_N)
   Language Agent → language_signals[] (keyed by Speaker_N)
   Video Agent → video_signals[] (keyed by Face_N)
   Conversation Agent → conversation_signals[] (keyed by Speaker_N)
                ↓
   fusion_service.py:
     per-speaker: voice + language + video → fusion_engine → compound_patterns
     session-level: temporal_patterns → interrogation_patterns
     ALL above = all_fusion_signals (only ~22 signals)
                ↓
   generate_session_narrative() receives:
     voice_summary      — aggregated stats (stress peaks, per-speaker avgs)
     language_summary    — aggregated stats (sentiment, buying signals)
     fusion_signals      — ONLY the 22 fusion output signals (NOT the 5,048 raw)
     transcript_segments — raw text
     video_summary       — aggregated per-face stats
     conversation_summary — turn-taking, rapport, dominance
                ↓
   Frontend receives ALL 5,048 via GET /video-signals
   Frontend receives report via GET /sessions/{id}/report
   ```

   **The narrative generator does NOT currently receive raw voice/language/conversation signals.**
   It only gets summaries + fusion output. The multi-layer analysis needs the raw signals.

3. Use proper OOP and DSA:
   - **bisect**: O(log N) segment-to-signal mapping
   - **defaultdict**: O(1) grouping by speaker, window, topic
   - **Counter**: state distribution per phase
   - **Sliding window**: phase boundary detection (2-consecutive-same-category)
   - **Set**: O(1) CONTINUOUS_SIGNALS lookup in frontend

---

## PART 1: Backend Signal Quality Fixes

### Fix 1.1: `tone_classification` — Suppress Neutral Below 0.40

**File:** `services/voiceAgent/rules.py`

Find the `tone_classification` emission block. Currently the final else-branch
emits `tone="neutral"` at confidence ~0.30 on every window. Gate it:

```python
# Find the section that emits tone_classification (around line 186-212)
# Add after computing tone confidence:
if tone["tone"] == "neutral" and tone["confidence_raw"] < 0.40:
    pass  # suppress — neutral at low confidence adds zero information
else:
    _add("VOICE-TONE-03", "tone_classification", ...)
```

**Expected:** 355 → ~175 signals (neutral windows suppressed)

### Fix 1.2: `shoulder_tension` — Baseline Delta Instead of Absolute

**File:** `services/video_agent/body_rules.py`

Find the `shoulder_tension` rule. Currently triggers on `shoulder_angle_std > 8.0`
(absolute). Change to compare against the speaker's baseline std:

```python
# Find: shoulder_angle_std > 8.0 (or similar absolute threshold)
# Replace with:
baseline_std = bl.shoulder_angle_std if bl else 5.0
delta_std = shoulder_angle_std - baseline_std
if delta_std > 3.0:  # 3° above their own baseline
    confidence = min((delta_std / 8.0) * conf_mult, 0.50)
    # ... emit signal
```

**Expected:** 520 → ~30-80 (only fires when tension INCREASES from person's normal)

### Fix 1.3: `head_nod` — Raise Velocity Threshold

**File:** `services/video_agent/body_rules.py`

Find `NOD_VELOCITY_MIN` or the equivalent threshold (currently 15°/s):

```python
# Change:
NOD_VELOCITY_MIN = 15.0  # too low — picks up micro-head-adjustments
# To:
NOD_VELOCITY_MIN = 22.0  # requires intentional nod motion
```

**Expected:** 242 → ~60 (each remaining nod is a real nod)

### Fix 1.4: `motor_inhibition` — Dynamic Confidence

**File:** `services/video_agent/interrogation_rules.py`

Find the motor_inhibition signal emission (around line 414). Currently assigns
constant 0.35 confidence. Scale by severity:

```python
# Find: constant confidence like 0.35 or quality_tier lookup
# Replace with:
reduction_pct = 1.0 - (current_velocity / max(baseline_velocity, 0.001))
reduction_pct = max(0.0, min(1.0, reduction_pct))
confidence = min(reduction_pct * 0.50, quality_tier_cap)

# Add display_mode metadata:
signal["metadata"]["display_mode"] = "continuous"
```

**Keep generating per-window.** Do NOT convert to event-based. The fusion engine
needs per-window signals for temporal co-occurrence. The frontend handles the
display differently for `display_mode: "continuous"`.

### Fix 1.5: `posture` — Add display_mode Metadata

**File:** `services/video_agent/body_rules.py`

Find where `posture` signals are created. Add metadata:

```python
signal["metadata"]["display_mode"] = "continuous"
```

Do NOT change the emission frequency. The signal keeps firing per-window.
The frontend uses `display_mode` to render it as a persistent bar instead of
a flickering badge.

Similarly for `shoulder_tension`:
```python
signal["metadata"]["display_mode"] = "continuous"
```

### Fix 1.6: Merge `energy_level` + `volume_shift`

**File:** `services/voiceAgent/rules.py`

Find both signal emissions (around lines 997-1040). They use the same trigger
(`abs(energy_delta_db) >= 6.0`) and the same data. Keep `energy_level`, remove
`volume_shift`. If any downstream code references `volume_shift`, alias it:

```python
# Remove the volume_shift emission block entirely
# In signalDisplayConfig.ts, add an alias comment for backward compat
```

---

## PART 2: Backend Analysis Layers

### Critical Data Flow Fix

**File:** `services/fusion_agent/fusion_service.py`

Currently `generate_session_narrative()` receives `fusion_signals=all_fusion_signals`
which is ONLY the 22 fusion output signals. The multi-layer analysis needs
the raw voice + language + conversation signals.

Add a new parameter to pass raw audio signals:

```python
# In fusion_service.py, around line 422-436, where generate_session_narrative is called:

# Collect all audio-agent raw signals (voice + language + conversation)
# These are already available as pure_voice_dicts, language_dicts, and
# conversation_summary.get("signals", [])
all_audio_signals = pure_voice_dicts + language_dicts
# NOTE: conversation signals are in conversation_summary["signals"], not a separate list
# Check if conversation_summary has a "signals" key, otherwise use empty list
conv_sigs = conversation_summary.get("signals", []) if conversation_summary else []
all_audio_signals += conv_sigs

report = await generate_session_narrative(
    session_id=session_id,
    duration_seconds=duration_seconds,
    speakers=speakers,
    voice_summary=request.voice_summary or {},
    language_summary=request.language_summary or {},
    fusion_signals=all_fusion_signals,
    unified_states=all_unified_states,
    meeting_type=content_type,
    entities=entities,
    graph_analytics=graph_insights,
    conversation_summary=conversation_summary,
    video_summary=video_summary,
    transcript_segments=request.transcript_segments or [],
    all_audio_signals=all_audio_signals,       # ← NEW
)
```

**File:** `services/fusion_agent/narrative.py`

Add `all_audio_signals` parameter to `generate_session_narrative()`:

```python
async def generate_session_narrative(
    ...,
    all_audio_signals: Optional[list[dict]] = None,   # ← NEW
) -> Optional[dict]:
```

### New Function: `_build_signal_analysis()`

**File:** `services/fusion_agent/narrative.py`

Add a new function that computes all 4 analysis layers. Call it from
`generate_session_narrative()` BEFORE `_build_context()`.

```python
def _build_signal_analysis(
    transcript_segments: list[dict],
    all_audio_signals: list[dict],
    voice_summary: dict,
    speakers: list[str],
    name_map: dict[str, str],
    entities: dict,
) -> dict:
```

#### Layer 1: Segment Behavioral Profiles

For each transcript segment, find overlapping voice + language signals.
Compute a composite state label from the combination.

```python
# Map signals to segments using bisect
seg_starts = [s.get("start_ms", 0) for s in transcript_segments]

# Group signals by (speaker, segment_index)
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

# For each segment, compute composite state
segment_profiles = []
for (spk, seg_idx), sigs in signals_by_seg.items():
    seg = transcript_segments[seg_idx]
    text = seg.get("text", "").strip()
    if not text or len(text) < 10:
        continue
    
    # Extract voice state from overlapping voice signals
    stress_sig = next((s for s in sigs if s.get("signal_type") == "vocal_stress_score"), None)
    tone_sig = next((s for s in sigs if s.get("signal_type") == "tone_classification"), None)
    rate_sig = next((s for s in sigs if s.get("signal_type") == "speech_rate_anomaly"), None)
    filler_sig = next((s for s in sigs if s.get("signal_type") == "filler_detection"), None)
    
    # Extract language state
    intent_sig = next((s for s in sigs if s.get("signal_type") == "utterance_intent"), None)
    sentiment_sig = next((s for s in sigs if s.get("signal_type") == "sentiment_score"), None)
    
    # Compute stress ratio vs baseline
    spk_avg = voice_summary.get("per_speaker", {}).get(spk, {}).get("avg_stress", 0.25)
    stress_val = float(stress_sig.get("value", 0)) if stress_sig else 0
    stress_ratio = stress_val / max(spk_avg, 0.01) if stress_val > 0 else 1.0
    
    # Get language classification
    classification = ""
    if intent_sig:
        classification = intent_sig.get("value_text", "")
    
    # Composite label
    label = _classify_composite_state(stress_ratio, classification)
    
    # Intensity = sum of signal confidences (how many channels firing)
    intensity = sum(s.get("confidence", 0) for s in sigs)
    
    m, s_val = divmod(seg.get("start_ms", 0) // 1000, 60)
    segment_profiles.append({
        "speaker": spk,
        "timestamp": f"{m}:{s_val:02d}",
        "timestamp_ms": seg.get("start_ms", 0),
        "text": text[:200],
        "voice_state": {
            "stress": round(stress_val, 3),
            "stress_vs_baseline": round(stress_ratio, 1),
            "tone": tone_sig.get("value_text", "neutral") if tone_sig else "neutral",
            "rate_change": rate_sig.get("value_text", "normal") if rate_sig else "normal",
            "fillers": int(filler_sig.get("value", 0)) if filler_sig else 0,
        },
        "language_state": {
            "classification": classification,
            "sentiment": round(float(sentiment_sig.get("value", 0)), 2) if sentiment_sig else 0,
        },
        "composite_label": label,
        "intensity": round(intensity, 1),
        "signal_count": len(sigs),
    })
```

Composite state naming function:

```python
_DENIAL_LABELS = frozenset({"DENY", "RESIST", "DEFLECT"})
_ADMISSION_LABELS = frozenset({"ACKNOWLEDGE", "ADMIT", "ANSWER", "COMMIT"})
_NARRATIVE_LABELS = frozenset({"ELABORATE", "EXPLAIN"})
_QUESTION_LABELS = frozenset({"CLARIFY", "QUESTION"})

def _classify_composite_state(stress_ratio: float, classification: str) -> str:
    stressed = stress_ratio >= 1.5
    calm = stress_ratio < 1.2
    cls = classification.upper()
    
    if stressed and cls in _DENIAL_LABELS:
        return "Stressed Denial"
    if stressed and cls in _ADMISSION_LABELS:
        return "Stressed Admission"
    if stressed and cls in _NARRATIVE_LABELS:
        return "Pressured Narrative"
    if stressed and cls in _QUESTION_LABELS:
        return "Anxious Clarification"
    if calm and cls in _DENIAL_LABELS:
        return "Calm Denial"
    if calm and cls in _ADMISSION_LABELS:
        return "Composed Acknowledgment"
    if calm and cls in _NARRATIVE_LABELS:
        return "Fluent Narrative"
    if cls == "DEFLECT":
        return "Active Deflection"
    if cls == "CONTRADICT":
        return "Direct Contradiction"
    if cls in _ADMISSION_LABELS:
        return "Guarded Answer"
    if cls in _DENIAL_LABELS:
        return "Moderate Denial"
    return "Neutral Response"
```

#### Layer 2: Response Pattern Timeline

Group segment profiles by speaker, detect phase boundaries using consecutive
same-category states:

```python
_STATE_CATEGORY = {
    "Stressed Denial": "denial", "Calm Denial": "denial", "Moderate Denial": "denial",
    "Stressed Admission": "admission", "Composed Acknowledgment": "admission",
    "Guarded Answer": "guarded",
    "Fluent Narrative": "narrative", "Pressured Narrative": "narrative",
    "Active Deflection": "deflection",
    "Direct Contradiction": "contradiction",
    "Anxious Clarification": "shock",
    "Neutral Response": "neutral",
}

# Group profiles by speaker, sort by timestamp_ms
# Detect phase = run of consecutive same-category states (min 2 segments)
# For each phase: compute avg stress_ratio, dominant state, key segment (highest intensity)
```

#### Layer 3: Stimulus-Response Pairs

For each segment from the dominant speaker (most talk time), find the next
segment from the responding speaker. Compute voice delta:

```python
# Determine stimulus speaker (highest dominance from conversation_summary)
# For each stimulus segment → find next response segment from different speaker
# Compute: response stress_ratio, delta from responder's previous segment
# Impact: high (>2.0× baseline), medium (1.3-2.0×), low (<1.3×)
```

#### Layer 4: Topic Voice Signatures

Use entity extraction topics. For each topic time range, aggregate voice
metrics per speaker:

```python
# For each topic in entities["topics"]:
#   Collect segment_profiles within topic's start_ms/end_ms
#   Per speaker: mean stress_ratio, sum fillers, dominant composite_label
#   Rank topics by avg stress_ratio → sensitivity_rank
```

### Add to `_build_context()`

Pass `signal_analysis` to `_build_context()`. Add 4 context sections:
- BEHAVIORAL HOTSPOTS (top 15 by intensity)
- BEHAVIORAL TRAJECTORY (per speaker phases)
- STIMULUS → RESPONSE ANALYSIS (high/medium impact pairs)
- TOPIC SENSITIVITY MAP (ranked by stress ratio)

### Add to JSON Schema in `_build_main_prompt()`

Add `behavioral_analysis` field to the JSON schema with 4 sub-fields:
hotspots, speaker_trajectories, exchange_analysis, topic_sensitivity.

### Add to `_parse_main_response()`

```python
"behavioral_analysis": parsed.get("behavioral_analysis", {}),
```

### Add to `_fallback_narrative()`

Build behavioral_analysis from pre-computed signal_analysis data directly
(no LLM needed — the computation already produced structured output).

Pass `all_audio_signals` to `_fallback_narrative()`:
```python
return _fallback_narrative(
    ...,
    all_audio_signals=all_audio_signals,
)
```

---

## PART 3: Frontend Display Changes

### Change 3.1: Continuous State Bars in Sidebar

**File:** `dashboard/src/components/VideoSignalPlayer.tsx`

In the sidebar rendering (around line 1340-1390), split signals into events
and continuous states:

```typescript
// Define continuous signal types
const CONTINUOUS_SIGNALS = new Set([
  "motor_inhibition", "posture", "shoulder_tension", "presence_detected",
]);

// In the render, after computing visibleSigs:
const eventSigs = visibleSigs.filter(s => 
  !CONTINUOUS_SIGNALS.has(s.signal_type) && 
  s.metadata?.display_mode !== "continuous"
);
const stateSigs = visibleSigs.filter(s => 
  CONTINUOUS_SIGNALS.has(s.signal_type) || 
  s.metadata?.display_mode === "continuous"
);

// Render events as badges (existing code)
// Render states as thin persistent bars at the bottom:
{stateSigs.length > 0 && (
  <div className="mt-1 flex flex-col gap-0.5">
    {stateSigs.map((s, i) => {
      const display = getSignalDisplay(s.signal_type, s.value_text ?? "");
      return (
        <div key={`state-${i}`} className="h-1 rounded-full" 
             style={{ backgroundColor: `${display.color}55` }}
             title={display.label} />
      );
    })}
  </div>
)}
```

### Change 3.2: Composite State Label in Sidebar

**File:** `dashboard/src/components/VideoSignalPlayer.tsx`

Add a composite label at the top of each face group. This requires knowing
the speaker's stress baseline. Get it from the signals themselves:

```typescript
// Compute speaker baseline from all signals (one-time on load)
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

// Composite label function
function getCompositeLabel(sigs: VideoSignal[], speakerId: string): string | null {
  const stress = sigs.find(s => s.signal_type === "vocal_stress_score");
  const intent = sigs.find(s => s.signal_type === "utterance_intent");
  const baseline = speakerBaselines[speakerId] ?? 0.25;
  const stressRatio = stress ? (stress.value ?? 0) / Math.max(baseline, 0.01) : 1.0;
  const classification = intent?.value_text ?? "";
  
  const DENIAL = new Set(["DENY", "RESIST", "DEFLECT"]);
  const ADMISSION = new Set(["ACKNOWLEDGE", "ADMIT", "ANSWER"]);
  const NARRATIVE = new Set(["ELABORATE", "EXPLAIN"]);
  
  if (stressRatio >= 1.5 && DENIAL.has(classification)) return "Stressed Denial";
  if (stressRatio >= 1.5 && ADMISSION.has(classification)) return "Stressed Admission";
  if (stressRatio >= 1.5 && NARRATIVE.has(classification)) return "Pressured Narrative";
  if (stressRatio < 1.2 && DENIAL.has(classification)) return "Calm Denial";
  if (stressRatio < 1.2 && ADMISSION.has(classification)) return "Composed Response";
  if (classification === "DEFLECT") return "Active Deflection";
  if (classification === "CONTRADICT") return "Contradiction";
  return null;  // no label for neutral/unknown
}

// In the render, before event badges:
{(() => {
  const label = getCompositeLabel(sigs, rawId);
  if (!label) return null;
  return (
    <div className="rounded px-2 py-0.5 text-[10px] font-semibold tracking-wide"
         style={{ backgroundColor: "rgba(139,92,246,0.15)", color: "#A78BFA" }}>
      {label}
    </div>
  );
})()}
```

### Change 3.3: Transcript Inline Composite Label

**File:** `dashboard/src/components/SessionDetail.tsx`

Find the transcript rendering section (where `matchSignalsToSegment` is called).
Add composite label after the segment text:

```typescript
// After matching signals to segment:
const segSignals = matchSignalsToSegment(signals, segment.start_ms, segment.end_ms, segment.speaker);
const compositeLabel = getCompositeLabel(segSignals, segment.speaker);
const stressSig = segSignals.find(s => s.signal_type === "vocal_stress_score");
const baseline = speakerBaselines[segment.speaker] ?? 0.25;
const stressRatio = stressSig ? ((stressSig.value ?? 0) / Math.max(baseline, 0.01)) : null;

// Render inline after transcript text:
{compositeLabel && (
  <span className="ml-2 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[9px] font-medium"
        style={{ backgroundColor: "rgba(139,92,246,0.12)", color: "#A78BFA" }}>
    {compositeLabel}
    {stressRatio && stressRatio > 1.3 && (
      <span className="text-red-400">{stressRatio.toFixed(1)}×</span>
    )}
  </span>
)}
```

### Change 3.4: Behavioral Analysis in Report Tab

**File:** `dashboard/src/components/SessionDetail.tsx`

Add a new section after Notes. Render only if `report.behavioral_analysis` exists:

```typescript
{report?.behavioral_analysis && (
  <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
    <h2 className="mb-4 text-sm font-semibold text-nexus-text-primary">
      📊 Behavioral Analysis
    </h2>
    
    {/* Hotspots — clickable timestamp list */}
    {report.behavioral_analysis.hotspots?.length > 0 && (
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-nexus-accent-purple uppercase tracking-wide mb-2">
          Behavioral Hotspots
        </h3>
        {report.behavioral_analysis.hotspots.map((h, i) => (
          <div key={i} className="mb-1.5 flex items-start gap-2 text-sm cursor-pointer hover:bg-nexus-surface-hover rounded p-1"
               onClick={() => seekTo(parseTimestamp(h.timestamp))}>
            <span className="text-xs font-mono text-nexus-text-muted w-10 shrink-0">{h.timestamp}</span>
            <span className="font-medium">{h.speaker}</span>
            <span className="text-nexus-text-secondary flex-1">"{h.text?.slice(0, 80)}"</span>
            <span className="rounded px-1.5 py-0.5 text-[9px] font-semibold"
                  style={{ backgroundColor: "rgba(139,92,246,0.15)", color: "#A78BFA" }}>
              {h.state}
            </span>
          </div>
        ))}
      </div>
    )}
    
    {/* Speaker Trajectory — phase bars */}
    {report.behavioral_analysis.speaker_trajectories && 
     Object.keys(report.behavioral_analysis.speaker_trajectories).length > 0 && (
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-nexus-accent-purple uppercase tracking-wide mb-2">
          Behavioral Arc
        </h3>
        {Object.entries(report.behavioral_analysis.speaker_trajectories).map(([spk, tl]) => (
          <div key={spk} className="mb-3">
            <div className="text-xs font-medium mb-1">{spk}</div>
            <div className="text-xs text-nexus-text-secondary mb-2 italic">{tl.summary}</div>
            <div className="flex gap-0.5 h-6 rounded overflow-hidden">
              {tl.phases?.map((phase, i) => (
                <div key={i} className="flex-1 flex items-center justify-center text-[8px] font-medium text-white"
                     style={{ backgroundColor: phaseToColor(phase) }}
                     title={phase}>
                  {phase.length < 15 ? phase : phase.slice(0, 12) + "…"}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    )}
    
    {/* Exchange Analysis — stimulus→response cards */}
    {report.behavioral_analysis.exchange_analysis?.length > 0 && (
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-nexus-accent-purple uppercase tracking-wide mb-2">
          Key Exchanges
        </h3>
        {report.behavioral_analysis.exchange_analysis.map((ex, i) => (
          <div key={i} className="mb-2 rounded border border-nexus-border p-2">
            <div className="text-xs text-nexus-text-muted">{ex.stimulus}</div>
            <div className="mt-1 flex items-center gap-2">
              <span className="text-nexus-accent-purple">→</span>
              <span className="text-sm font-medium">{ex.response_state}</span>
              {ex.stress_impact > 1.5 && (
                <span className="text-[10px] text-red-400">{ex.stress_impact.toFixed(1)}× baseline</span>
              )}
            </div>
          </div>
        ))}
      </div>
    )}
    
    {/* Topic Sensitivity — ranked bars */}
    {report.behavioral_analysis.topic_sensitivity && 
     Object.keys(report.behavioral_analysis.topic_sensitivity).length > 0 && (
      <div>
        <h3 className="text-xs font-semibold text-nexus-accent-purple uppercase tracking-wide mb-2">
          Topic Sensitivity
        </h3>
        {Object.entries(report.behavioral_analysis.topic_sensitivity).map(([spk, data]) => (
          <div key={spk} className="mb-2">
            <div className="text-xs font-medium mb-1">{spk}</div>
            {data.topics?.map((t, i) => (
              <div key={i} className="flex items-center gap-2 mb-0.5">
                <span className="text-[10px] text-nexus-text-muted w-4">#{t.rank}</span>
                <span className="text-xs flex-1">{t.topic}</span>
                <div className="w-20 h-2 rounded-full bg-nexus-surface-hover overflow-hidden">
                  <div className="h-full rounded-full"
                       style={{ width: `${Math.min(t.stress_ratio * 40, 100)}%`,
                                backgroundColor: t.stress_ratio > 2.0 ? "#EF4444" : t.stress_ratio > 1.5 ? "#F59E0B" : "#22C55E" }} />
                </div>
                <span className="text-[10px] text-nexus-text-muted w-8">{t.stress_ratio.toFixed(1)}×</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    )}
  </section>
)}
```

Helper function for phase colors:

```typescript
function phaseToColor(phase: string): string {
  const lower = phase.toLowerCase();
  if (lower.includes("denial") || lower.includes("resist")) return "#EF4444";
  if (lower.includes("stress") || lower.includes("pressur")) return "#F97316";
  if (lower.includes("admission") || lower.includes("acknowledge")) return "#F59E0B";
  if (lower.includes("narrative") || lower.includes("fluent")) return "#22C55E";
  if (lower.includes("deflect")) return "#A855F7";
  if (lower.includes("shock") || lower.includes("clarif")) return "#3B82F6";
  return "#6B7280";
}
```

---

## Updated Report Section Order

```
1. General Summary
2. Executive Summary
3. Key Facts & Commitments
4. Notes (topic-grouped)
5. Behavioral Analysis (NEW — 4 subsections)
   5a. Behavioral Hotspots (clickable timestamps)
   5b. Behavioral Arc (phase bars per speaker)
   5c. Key Exchanges (stimulus→response)
   5d. Topic Sensitivity (ranked bars)
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

- Signal generation frequency — keep per-window for all signals (fusion needs it)
- `presence_detected` — already hidden, working as designed
- Fusion engine rules — compound patterns are correctly selective
- `vocal_stress_score` — fires every window intentionally (voice baseline tracking)
- `sentiment_score` — fires every segment intentionally (language baseline)
- Any threshold on `facial_stress` (capped at 0.38 per RESEARCH.md)
- Any threshold on `body_lean` (capped at 0.38 per RESEARCH.md)
- Any threshold on `deception_cluster` (capped at 0.50 per specification)

---

## Files Modified:

### Backend (4 files):
1. **services/voiceAgent/rules.py**:
   - Suppress neutral `tone_classification` below 0.40 (~3 lines)
   - Remove `volume_shift` emission block (~15 lines removed)

2. **services/video_agent/body_rules.py**:
   - `shoulder_tension` → baseline delta comparison (~5 lines changed)
   - `head_nod` → velocity threshold 15→22 (~1 line)
   - `posture` + `shoulder_tension` → add `display_mode: "continuous"` metadata (~2 lines each)

3. **services/fusion_agent/fusion_service.py**:
   - Collect `all_audio_signals` from pure_voice + language + conversation (~5 lines)
   - Pass to `generate_session_narrative()` (~1 line)

4. **services/fusion_agent/narrative.py**:
   - New: `_build_signal_analysis()` + 4 layer computation functions (~250 lines)
   - New: `_classify_composite_state()` helper (~25 lines)
   - Modified: `generate_session_narrative()` — accept `all_audio_signals`, call analysis (~8 lines)
   - Modified: `_build_context()` — add signal_analysis parameter + 4 sections (~50 lines)
   - Modified: `_build_main_prompt()` — add `behavioral_analysis` to JSON schema (~20 lines)
   - Modified: `_parse_main_response()` — parse `behavioral_analysis` (~1 line)
   - Modified: `_fallback_narrative()` — build behavioral_analysis from pre-computed data (~40 lines)

### Frontend (3 files):
5. **VideoSignalPlayer.tsx**:
   - `CONTINUOUS_SIGNALS` set (~4 lines)
   - Split events/states in render (~10 lines)
   - Continuous state bars rendering (~10 lines)
   - `speakerBaselines` memo (~10 lines)
   - `getCompositeLabel()` function (~25 lines)
   - Composite label rendering in sidebar (~8 lines)

6. **SessionDetail.tsx**:
   - `getCompositeLabel()` function (shared or imported) (~25 lines)
   - Transcript inline composite label (~10 lines)
   - BehavioralAnalysis report section with 4 subsections (~80 lines)
   - `phaseToColor()` helper (~10 lines)

7. **signalDisplayConfig.ts**:
   - Remove `volume_shift` entry (~5 lines removed)
   - No other changes needed