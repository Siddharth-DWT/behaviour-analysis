# NEXUS — Enhanced Narrative Report Generation + Report Panel

## Current State (verified from code)

**narrative.py** (806L) generates reports via Claude/OpenAI API. The LLM receives
400+ lines of rich per-agent data but outputs only 4 generic fields:

```json
{
  "executive_summary": "2-3 sentences",
  "key_moments": [{time_description, description, significance}],
  "cross_modal_insights": ["insight1"],
  "recommendations": ["rec1"]
}
```

Same structure for ALL content types. No per-speaker analysis, no key facts
section, no interrogation-specific sections. The `speaker_analyses` field from
the function docstring was never added to the JSON schema.

**SessionDetail.tsx** has a Report tab (line 1386) that renders these 4 sections
with styled cards. It fetches from `["report", id]` query with `has_report === true`.

**InsightPanel.tsx** shows entities (objections, commitments) and conversation
dynamics but has NO access to the narrative report.

**InterrogationSummaryPanel.tsx** shows risk gauge + denial trajectory on the
Video tab but has NO connection to the Report tab.

## What Top Meeting Analyzers Do (Otter, Gong, Fireflies, Chorus)

All of them structure reports with:
1. Key facts / action items / commitments FIRST (assigned to people)
2. Per-speaker behavioral profile
3. Topic/phase timeline with timestamps
4. Content-type-specific sections
5. Actionable recommendations

NEXUS has 6-modality cross-modal analysis that NONE of them have. The report
should showcase this unique capability — especially incongruence detection
(calm voice + stressed face, nodding during denial, forward lean + avoidant gaze).

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read `services/fusion_agent/narrative.py` COMPLETELY:
   - `_build_context()` — what data is available (voice, language, video,
     conversation, entities, graph analytics, interrogation signals)
   - `_build_prompt()` — current system prompt + per-content-type instructions
   - `_parse_narrative_response()` — JSON parsing
   - `_fallback_narrative()` — non-LLM fallback with interrogation handling
   - The function docstring shows `speaker_analyses` was intended but never implemented

2. Read `frontend/.../SessionDetail.tsx` lines 1386-1490:
   - How the Report tab renders executive_summary, key_moments, cross_modal_insights, recommendations
   - The `report?.narrative` fallback
   - The `content?.` property access pattern

3. Read `frontend/.../InsightPanel.tsx`:
   - SalesPanel, MeetingPanel, InterviewPanel, ConversationDynamicsPanel
   - How entities (objections, commitments) are rendered
   - Content-type routing pattern

4. Read `frontend/.../InterrogationSummaryPanel.tsx`:
   - RiskGauge, DenialTrajectory, TechniqueBadge, ContaminationList
   - How interrogation-specific data is extracted from signals

5. Read `frontend/.../contentTypes.ts`:
   - Content type definitions with roles, entityFields, speakerStats

6. Think about the problem and solution:
   - The report structure should be content-type-aware at BOTH LLM prompt
     level AND frontend rendering level
   - Per-speaker analysis must synthesize voice + language + video data into
     a behavioral narrative, not just repeat numbers
   - Cross-modal insights must describe WHAT disagrees and WHY it matters
   - Key facts (commitments, objections, admissions) go FIRST because that's
     what users look for immediately
   - Interrogation reports need risk assessment, contamination timeline, and
     ethical framing that other content types don't

7. Use proper OOP and DSA:
   - **Strategy Pattern**: content-type-specific prompt builders
   - **Builder Pattern**: report structure assembly
   - **Template Method**: common report skeleton with type-specific sections
   - **HashMap**: O(1) signal lookup by type for per-speaker aggregation

---

## Change 1: Enhanced LLM Output Schema

**File:** `services/fusion_agent/narrative.py` — `_build_prompt()`

Replace the current 4-field JSON schema with a content-type-aware structure:

### Base Schema (ALL content types)

```json
{
  "executive_summary": "3-5 sentence overview emphasizing the most important finding",

  "key_facts": [
    {
      "type": "commitment | objection | decision | admission | disclosure",
      "speaker": "John",
      "text": "I'll send the proposal by Friday",
      "timestamp": "12:30",
      "status": "confirmed | unresolved | retracted"
    }
  ],

  "speaker_analyses": {
    "Speaker_0": {
      "role": "Seller",
      "behavioral_profile": "Confident delivery with controlled pacing. Stress elevated during pricing discussion (8:30-9:15) — facial stress peaked at 0.72 while voice remained steady, suggesting rehearsed composure. Forward lean increased during closing, indicating genuine investment in outcome.",
      "voice_patterns": "Steady baseline with 2 pitch elevation events during value propositions. Speech rate dropped 25% during objection handling — deliberate, not stressed.",
      "body_language": "Open posture maintained throughout. 3 self-touch events during competitor discussion — possible discomfort with the topic.",
      "key_moments": ["8:30 — stress peak during pricing", "15:00 — strongest engagement during demo"]
    }
  },

  "key_moments": [
    {
      "timestamp": "8:30",
      "description": "Pricing discussion opened — prospect's facial engagement dropped while verbal response remained positive",
      "significance": "Classic incongruence — spoken interest may mask concern about budget",
      "signals_involved": ["facial_engagement_drop", "positive_sentiment", "gaze_break"]
    }
  ],

  "cross_modal_insights": [
    {
      "insight": "At 8:30, John's voice remained calm (stress 0.22) while facial stress spiked to 0.72 — a 3× delta indicating emotional suppression during pricing discussion",
      "modalities": ["voice", "face"],
      "type": "incongruence",
      "significance": "Cross-modal mismatch suggests rehearsed composure rather than genuine confidence on pricing"
    }
  ],

  "recommendations": [
    {
      "priority": "high | medium | low",
      "action": "Follow up on the unresolved pricing objection within 48 hours",
      "rationale": "Sarah's facial engagement dropped 40% during pricing but she didn't voice the concern — unspoken objection likely still active"
    }
  ]
}
```

### Interrogation-Specific Extensions

For `interrogation_video` content type, ADD these sections to the schema:

```json
{
  "risk_assessment": {
    "false_confession_risk": "low | low_moderate | moderate | elevated | high",
    "risk_score": 0.45,
    "contributing_factors": [
      {"factor": "contamination", "present": true, "detail": "4 case-specific terms adopted"},
      {"factor": "session_duration", "present": false, "detail": "54 min — below 3-hour risk threshold"},
      {"factor": "coercive_technique", "present": false, "detail": "Mixed PEACE/Reid — no coercion detected"},
      {"factor": "denial_weakening", "present": true, "detail": "Categorical → weak over 54 minutes"},
      {"factor": "processing_delays", "present": true, "detail": "2584ms delay after GPS evidence"},
      {"factor": "capitulation_pattern", "present": true, "detail": "2 cascade events between 7:21-19:25"}
    ],
    "ethical_note": "Risk score indicates probability of coerced compliance, NOT guilt or deception. All factors have alternative innocent explanations."
  },

  "contamination_timeline": [
    {
      "term": "GPS data",
      "interrogator_first": "18:45",
      "suspect_adopted": "19:30",
      "context": "Officer mentioned GPS tracking; suspect referenced GPS 45 seconds later"
    }
  ],

  "technique_analysis": {
    "primary": "mixed",
    "peace_markers": 6,
    "reid_markers": 9,
    "coercive_markers": 0,
    "assessment": "Primarily PEACE framework with Reid accusatory elements. No coercive tactics detected. Reid elements may increase false confession risk in vulnerable suspects."
  }
}
```

### Sales-Specific Extensions

```json
{
  "deal_assessment": {
    "close_probability": "high | medium | low",
    "stage_reached": "Closing",
    "buying_signals": 4,
    "unresolved_objections": 1,
    "prospect_engagement_trend": "increasing | stable | declining"
  },

  "objection_handling": [
    {
      "objection": "Pricing too high",
      "timestamp": "12:30",
      "handling_quality": "partial — addressed ROI but not compared to competitor pricing",
      "resolved": false,
      "prospect_reaction": "Verbal acceptance but facial engagement dropped 30%"
    }
  ]
}
```

---

## Change 2: Content-Type-Specific Prompt Instructions

**File:** `services/fusion_agent/narrative.py` — `_build_prompt()`

Update the `type_instructions` dict to request content-type-specific fields:

### Interrogation Prompt Addition

```python
"interrogation_video": (
    "... existing focus text ... "
    "OUTPUT STRUCTURE: Include 'risk_assessment' with false_confession_risk level, "
    "score, and 6 contributing factors (contamination, session_duration, coercive_technique, "
    "denial_weakening, processing_delays, capitulation_pattern). Each factor has 'present' "
    "(boolean) and 'detail' (string). Include 'contamination_timeline' listing each "
    "case-specific term with interrogator_first and suspect_adopted timestamps. Include "
    "'technique_analysis' with primary technique, marker counts, and assessment. "
    "SPEAKER ANALYSIS: Profile the suspect's behavioral trajectory across the session — "
    "how did their stress, denial strength, body language, and speech patterns change? "
    "Profile the interrogator's technique shifts. "
    "CROSS-MODAL INSIGHTS: Focus on moments where voice/face/body signals diverge — "
    "stress peaks without verbal acknowledgment, calm speech during facial distress, "
    "body freezing after specific questions. "
    "ETHICAL FRAMING: Every finding must include an alternative innocent explanation. "
    "Never use the word 'deception'. Use 'incongruence', 'stress indicator', 'risk factor'. "
),
```

### Sales Prompt Addition

```python
"sales_call": (
    "... existing focus text ... "
    "OUTPUT STRUCTURE: Include 'deal_assessment' with close_probability, stage_reached, "
    "buying_signal count, unresolved_objection count, and prospect_engagement_trend. "
    "Include 'objection_handling' listing each objection with handling_quality assessment "
    "and prospect's true reaction (verbal vs body language). "
    "SPEAKER ANALYSIS: Profile the seller's persuasion effectiveness and the prospect's "
    "genuine interest level (distinguish spoken interest from behavioral engagement). "
    "CROSS-MODAL INSIGHTS: Focus on prospect incongruence — verbal agreement with "
    "disengaged body language, nodding with avoidant gaze, positive words with stressed voice. "
    "These reveal unspoken objections. "
    "KEY FACTS: List commitments and objections FIRST with speaker, timestamp, and status. "
),
```

### Interview, Meeting, Podcast — Similar Additions

Each content type gets a specific `OUTPUT STRUCTURE` instruction telling the LLM
which extra sections to produce and what to focus on in speaker analyses.

---

## Change 3: Per-Speaker Aggregation in _build_context()

**File:** `services/fusion_agent/narrative.py` — `_build_context()`

After the existing per-agent sections, add a SYNTHESIZED per-speaker block
that combines all modalities. This gives the LLM a pre-computed cross-modal
view instead of asking it to cross-reference 4 separate sections:

```python
    # ── Synthesized per-speaker cross-modal profile ──────────────────────────
    lines.append("\n=== CROSS-MODAL SPEAKER PROFILES ===")
    lines.append("(Pre-computed synthesis — use these for speaker_analyses)")

    for speaker_id in speakers:
        display = _display(speaker_id, name_map)
        lines.append(f"\n[{display}]")

        # Voice
        vs = voice_summary.get("per_speaker", {}).get(speaker_id, {})
        if vs:
            lines.append(f"  VOICE: stress_avg={vs.get('avg_stress', 0):.2f}, "
                         f"stress_max={vs.get('max_stress', 0):.2f}, "
                         f"fillers={vs.get('total_fillers', 0)}, "
                         f"pitch_events={vs.get('pitch_elevation_events', 0)}")

        # Language
        ls = language_summary.get("per_speaker", {}).get(speaker_id, {})
        if ls:
            lines.append(f"  LANGUAGE: sentiment_avg={ls.get('avg_sentiment', 0):.2f}, "
                         f"power={ls.get('avg_power_score', 0.5):.2f}, "
                         f"buying_signals={ls.get('buying_signal_count', 0)}, "
                         f"objections={ls.get('objection_count', 0)}")

        # Video
        vid = (video_summary or {}).get("per_speaker", {}).get(speaker_id, {})
        if vid:
            lines.append(f"  FACE: emotion={vid.get('dominant_emotion', '?')}, "
                         f"stress={vid.get('avg_facial_stress', 0):.2f}, "
                         f"engagement={vid.get('avg_facial_engagement', 0):.2f}")
            lines.append(f"  GAZE: on_screen={vid.get('avg_gaze_on_screen_pct', 0):.0%}, "
                         f"breaks={vid.get('gaze_breaks', 0)}, "
                         f"blink_anomalies={vid.get('blink_anomalies', 0)}")
            lines.append(f"  BODY: movement={vid.get('avg_body_movement', 0):.2f}, "
                         f"self_touch={vid.get('hand_near_face_events', 0)}, "
                         f"nods={vid.get('head_nods', 0)}, "
                         f"shakes={vid.get('head_shakes', 0)}")

        # Conversation
        cs = (conversation_summary or {})
        dom = cs.get("dominance", {}).get("per_speaker", {}).get(speaker_id, 0)
        if dom:
            lines.append(f"  CONVERSATION: talk_time={dom:.1f}%, "
                         f"interruptions={cs.get('interruptions', {}).get('per_speaker', {}).get(speaker_id, 0)}")

        # Cross-modal flags
        if vs and vid:
            voice_stress = vs.get("avg_stress", 0)
            face_stress = vid.get("avg_facial_stress", 0)
            delta = abs(voice_stress - face_stress)
            if delta > 0.25:
                higher = "voice" if voice_stress > face_stress else "face"
                lines.append(f"  ⚡ INCONGRUENCE: {higher} stress ({max(voice_stress, face_stress):.2f}) "
                             f"vs {'face' if higher == 'voice' else 'voice'} "
                             f"({min(voice_stress, face_stress):.2f}) — "
                             f"delta {delta:.2f}")
```

---

## Change 4: Enhanced _parse_narrative_response()

**File:** `services/fusion_agent/narrative.py`

Parse the new fields while maintaining backward compatibility:

```python
def _parse_narrative_response(raw_text: str, speakers: list[str]) -> dict:
    # ... existing JSON parsing ...

    return {
        "executive_summary": parsed.get("executive_summary", ""),
        "key_facts": parsed.get("key_facts", []),
        "speaker_analyses": parsed.get("speaker_analyses", {}),
        "key_moments": parsed.get("key_moments", []),
        "cross_modal_insights": parsed.get("cross_modal_insights", []),
        "recommendations": parsed.get("recommendations", []),
        # Content-type-specific sections (may be absent)
        "risk_assessment": parsed.get("risk_assessment"),
        "contamination_timeline": parsed.get("contamination_timeline"),
        "technique_analysis": parsed.get("technique_analysis"),
        "deal_assessment": parsed.get("deal_assessment"),
        "objection_handling": parsed.get("objection_handling"),
    }
```

---

## Change 5: Enhanced _fallback_narrative()

**File:** `services/fusion_agent/narrative.py`

The fallback already handles interrogation signals. Enhance it to produce
the new structure fields:

- Build `key_facts` from entities.commitments + entities.objections
- Build `speaker_analyses` from per-speaker summaries (voice + language + video)
- Build `risk_assessment` from false_confession_risk signal metadata
- Keep existing cross_modal_insights switch/case (already comprehensive)

---

## Change 6: Enhanced Report Tab in SessionDetail.tsx

**File:** `frontend/.../SessionDetail.tsx` — Report tab section (line 1386)

### New Section Order

```
1. Executive Summary (existing — keep)
2. Key Facts & Commitments (NEW — from key_facts array)
3. Risk Assessment (NEW — interrogation only, from risk_assessment)
4. Speaker Analyses (NEW — from speaker_analyses object)
5. Key Moments (existing — enhance with signals_involved)
6. Cross-Modal Insights (existing — enhance with modalities + type)
7. Recommendations (existing — enhance with priority)
```

### Key Facts Section

```tsx
{content?.key_facts && content.key_facts.length > 0 && (
  <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
    <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-nexus-text-primary">
      📋 Key Facts & Commitments
    </h2>
    <div className="space-y-2">
      {content.key_facts.map((fact, i) => (
        <div key={i} className="flex items-start gap-2 rounded border-l-2 bg-nexus-surface-hover p-2 text-xs"
          style={{
            borderLeftColor: fact.type === "commitment" ? "var(--accent-blue)"
              : fact.type === "objection" ? "var(--stress-high)"
              : fact.type === "admission" ? "var(--stress-med)"
              : "var(--text-muted)"
          }}>
          <span className="shrink-0 mt-0.5">
            {fact.type === "commitment" ? "✅" : fact.type === "objection" ? "❌"
             : fact.type === "admission" ? "⚠️" : "📌"}
          </span>
          <div className="flex-1">
            <span className="text-nexus-text-primary">{fact.text}</span>
            <span className="ml-2 text-nexus-text-muted">
              — {fact.speaker} at {fact.timestamp}
              {fact.status && ` (${fact.status})`}
            </span>
          </div>
        </div>
      ))}
    </div>
  </section>
)}
```

### Risk Assessment Section (interrogation only)

```tsx
{content?.risk_assessment && (
  <section className="rounded-lg border border-amber-500/30 bg-nexus-surface p-5">
    <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-amber-400">
      ⚖️ False Confession Risk Assessment
    </h2>
    {/* Risk gauge bar */}
    {/* Contributing factors with ✓/✗ badges */}
    {/* Ethical note in italic muted text */}
  </section>
)}
```

### Speaker Analyses Section

```tsx
{content?.speaker_analyses && Object.keys(content.speaker_analyses).length > 0 && (
  <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
    <h2 className="mb-4 text-sm font-semibold text-nexus-text-primary">
      👤 Speaker Analysis
    </h2>
    {Object.entries(content.speaker_analyses).map(([spk, analysis]) => (
      <div key={spk} className="mb-4 border-l-2 border-accent-purple-40 pl-3">
        <div className="text-xs font-semibold text-nexus-accent-purple mb-1">
          {analysis.role ? `${spk} (${analysis.role})` : spk}
        </div>
        <p className="text-sm text-nexus-text-primary mb-1">{analysis.behavioral_profile}</p>
        {analysis.voice_patterns && (
          <p className="text-xs text-nexus-text-secondary">🎙️ {analysis.voice_patterns}</p>
        )}
        {analysis.body_language && (
          <p className="text-xs text-nexus-text-secondary">🧍 {analysis.body_language}</p>
        )}
      </div>
    ))}
  </section>
)}
```

### Enhanced Cross-Modal Insights

```tsx
{/* Update existing cross-modal section to handle object format */}
{content.cross_modal_insights.map((insight, i) => {
  const text = typeof insight === "string" ? insight : insight.insight;
  const modalities = typeof insight === "object" ? insight.modalities : null;
  return (
    <li key={i} className="flex items-start gap-2 text-sm text-nexus-text-primary">
      <span className="mt-1 text-nexus-accent-purple font-bold">⚡</span>
      <div>
        {text}
        {modalities && (
          <div className="mt-1 flex gap-1">
            {modalities.map((m) => (
              <span key={m} className="rounded-full bg-accent-purple-15 px-2 py-0.5 text-[10px] text-nexus-accent-purple">
                {m}
              </span>
            ))}
          </div>
        )}
      </div>
    </li>
  );
})}
```

### Enhanced Recommendations with Priority

```tsx
{content.recommendations.map((rec, i) => {
  const text = typeof rec === "string" ? rec : rec.action;
  const priority = typeof rec === "object" ? rec.priority : null;
  const rationale = typeof rec === "object" ? rec.rationale : null;
  return (
    <li key={i} className="flex items-start gap-2 text-sm text-nexus-text-primary">
      <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${
        priority === "high" ? "bg-red-400" :
        priority === "medium" ? "bg-amber-400" : "bg-nexus-stress-low"
      }`} />
      <div>
        {text}
        {rationale && <p className="mt-0.5 text-xs text-nexus-text-muted italic">{rationale}</p>}
      </div>
    </li>
  );
})}
```

---

## Change 7: Content-Type Isolation — STRICT

**Content-type-specific sections MUST only appear for their own content type.
They must NOT leak into other types.**

| Section | Shows ONLY for | Never shows for |
|---------|---------------|-----------------|
| `risk_assessment` | `interrogation_video` | sales_call, interview, meeting, podcast, etc |
| `contamination_timeline` | `interrogation_video` | everything else |
| `technique_analysis` | `interrogation_video` | everything else |
| `deal_assessment` | `sales_call` | interrogation_video, interview, meeting, etc |
| `objection_handling` | `sales_call`, `client_meeting` | interrogation_video, podcast, lecture, etc |
| Deal Progression bar | `sales_call` | everything else |
| Candidate Assessment | `interview` | everything else |
| Participation Balance | `internal`, `meeting` | sales_call, interrogation_video |

**Backend enforcement:** The `_build_prompt()` type_instructions for each
content type requests ONLY the fields relevant to that type. The LLM prompt
must explicitly say "Do NOT include risk_assessment for non-interrogation
sessions" and "Do NOT include deal_assessment for non-sales sessions."

```python
# In the user_prompt, after the JSON schema:
if meeting_type == "interrogation_video":
    extra = (
        "Include 'risk_assessment', 'contamination_timeline', and 'technique_analysis' fields. "
        "Do NOT include 'deal_assessment' or 'objection_handling'."
    )
elif meeting_type == "sales_call":
    extra = (
        "Include 'deal_assessment' and 'objection_handling' fields. "
        "Do NOT include 'risk_assessment', 'contamination_timeline', or 'technique_analysis'."
    )
else:
    extra = (
        "Do NOT include 'risk_assessment', 'contamination_timeline', 'technique_analysis', "
        "'deal_assessment', or 'objection_handling'. These are type-specific fields."
    )
```

**Frontend enforcement:** Every content-type-specific section checks BOTH
data availability AND content type before rendering:

```tsx
{/* WRONG — shows for any session that happens to have the data: */}
{content?.risk_assessment && (
  <RiskAssessmentSection ... />
)}

{/* CORRECT — shows ONLY for interrogation AND only when data exists: */}
{contentType === "interrogation_video" && content?.risk_assessment && (
  <RiskAssessmentSection ... />
)}

{/* CORRECT — shows ONLY for sales: */}
{contentType === "sales_call" && content?.deal_assessment && (
  <DealAssessmentSection ... />
)}
```

**The `contentType` prop must be passed to the Report tab.** Currently
`SessionDetail.tsx` has `detail.meeting_type` or `session.meeting_type`
available — pass it into the report rendering block.

**Shared sections** that appear for ALL content types:
- Executive Summary ✅
- Key Facts & Commitments ✅ (commitments in sales, admissions in interrogation, decisions in meetings)
- Speaker Analyses ✅
- Key Moments ✅
- Cross-Modal Insights ✅
- Recommendations ✅

These shared sections use the SAME structure but the LLM populates them
differently based on content type (the type_instructions handle this).

---

## Change 8: Backward Compatibility

ALL changes must be backward compatible:

- Old reports (4-field JSON) still render correctly — the new sections
  check `content?.key_facts && content.key_facts.length > 0` before rendering
- New fields default to empty arrays/objects in `_parse_narrative_response`
- Frontend handles BOTH string format (`"insight text"`) and object format
  (`{insight, modalities, type}`) for cross_modal_insights and recommendations
- `_fallback_narrative` produces the new structure fields when data is available
  but gracefully returns empty when not

---

## Files Modified:

### Backend (1):
1. **services/fusion_agent/narrative.py**:
   - `_build_prompt()` — content-type-specific JSON schema + output structure instructions (~50 lines)
   - `_build_context()` — cross-modal speaker profiles section (~40 lines)
   - `_parse_narrative_response()` — parse new fields (~10 lines)
   - `_fallback_narrative()` — produce new structure fields (~30 lines)

### Frontend (1):
2. **SessionDetail.tsx** — Report tab:
   - Key Facts section (NEW, ~25 lines)
   - Risk Assessment section (NEW, interrogation only, ~20 lines)
   - Speaker Analyses section (NEW, ~20 lines)
   - Enhanced cross-modal insights with modality badges (~10 lines changed)
   - Enhanced recommendations with priority dots (~10 lines changed)
   - Contamination timeline section (NEW, interrogation only, ~15 lines)
   - Technique analysis section (NEW, interrogation only, ~10 lines)