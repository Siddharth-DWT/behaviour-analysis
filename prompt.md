# NEXUS — Add Meeting Notes, Topic-Grouped Details, and Action Items to Narrative Report

## What Fireflies Produces (the target pattern)

```
GENERAL SUMMARY
• 5-7 bullet points — one per key theme/decision/outcome
• No paragraphs — scannable bullets only

NOTES (grouped by topic)
  Topic Heading 1
  Brief description of this topic area.
    • Speaker-attributed detail with timestamp (01:54)
      • Supporting sub-detail
      • Supporting sub-detail
    • Another speaker-attributed detail (08:56)
      • Sub-detail
  
  Topic Heading 2
  Brief description.
    • Detail (14:22)
      • Sub-detail

ACTION ITEMS (grouped by person)
  Person Name
    • Task description (timestamp)
    • Task description (timestamp)
  
  Another Person
    • Task description (timestamp)
```

**Three layers: Summary (what happened) → Notes (what was discussed, by topic) → Action Items (what to do next, by person).**

This structure works for ALL meeting types — not just sales calls. It works for:
- Sales: topics = pricing, demo, objections. Actions = send proposal, schedule follow-up.
- Interviews: topics = intro, technical questions, cultural fit. Actions = send feedback, schedule next round.
- Internal meetings: topics = status updates, blockers, decisions. Actions = assigned tasks.
- Client meetings: topics = project review, concerns, next steps. Actions = deliverables.
- Interrogation: topics = background questions, evidence presentation, denial phase. Actions = review contamination, check procedure compliance.

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read `services/fusion_agent/narrative.py` — COMPLETELY:
   - `_build_context()` — check what's in `entities["topics"]`, `entities["commitments"]`,
     `entities["objections"]`, `entities["people"]`, transcript segments
   - `_build_prompt()` — the JSON schema the LLM is asked to produce
   - `_parse_narrative_response()` — what fields it extracts
   - `_fallback_narrative()` — the non-LLM path

2. Read `services/language_agent/entity_extractor.py`:
   - The `topics` array output — each topic has `name`, `start_ms`, `end_ms`, optional `summary`
   - The `commitments` array — each has `speaker`, `text`, `start_ms`/`timestamp_ms`
   - The `objections` array — each has `speaker`, `text`, `start_ms`, `resolved`

3. Read whatever frontend component renders the report (SessionDetail.tsx or equivalent)
   to understand how the new fields will be displayed.

4. Use proper OOP and DSA:
   - **HashMap**: group transcript segments by topic interval using bisect for O(log T) per segment
   - **Template Method**: topic-grouped notes follow same structure regardless of content type
   - The LLM already receives the full transcript in context — it just needs to be told
     to organize its output by topic

5. Content-type isolation: The general_summary, notes, and action_items sections
   appear for ALL content types. They are NOT type-specific. The content within
   them naturally adapts because the LLM's type_instructions guide the focus.

---

## Change 1: Add Three New Fields to JSON Schema

**File:** `services/fusion_agent/narrative.py` — `_build_prompt()`

Add to the JSON schema between `executive_summary` and `key_facts`:

```json
  "general_summary": [
    "Key theme or outcome as a single bullet point",
    "Another key theme — one per major topic discussed",
    "Decision or agreement reached",
    "Timeline or deadline established",
    "Technical/operational update if relevant"
  ],

  "notes": [
    {
      "topic": "Topic or Phase Name",
      "summary": "One sentence describing what was discussed in this topic area",
      "details": [
        {
          "speaker": "Person Name or Speaker_0",
          "text": "What they said or contributed — paraphrased or key quote",
          "timestamp": "MM:SS",
          "sub_details": [
            "Supporting point or context",
            "Another supporting detail"
          ]
        }
      ]
    }
  ],

  "action_items": [
    {
      "assignee": "Person Name or Speaker_0",
      "task": "Specific task or commitment",
      "deadline": "By Friday afternoon" or null,
      "timestamp": "MM:SS",
      "context": "Brief reason or origin of this action"
    }
  ],
```

**Add to the system prompt instruction:**

```
"STRUCTURE REQUIREMENTS:
1. general_summary: 5-7 bullet points covering the key themes, decisions, and outcomes.
   Each bullet is one standalone sentence. No paragraphs. Scannable.
2. notes: Group discussion details by TOPIC or PHASE. Use the conversation phases/topics
   from the context data. Under each topic, list speaker-attributed details with timestamps.
   Each detail can have sub_details for supporting points. This is the detailed record of
   what was discussed — the most comprehensive section.
3. action_items: List every commitment, task, or follow-up mentioned. Group by assignee
   (the person responsible). Include deadline if mentioned, timestamp of when it was agreed,
   and brief context. Extract from both explicit commitments ('I will send the report')
   and implicit ones ('We should schedule a follow-up' — assign to whoever said it).
"
```

**The notes section should map to the `topics` array from entity extraction.** Add this to the context:

```python
# In _build_context(), after the ENTITIES section:
if topics:
    lines.append("\nCONVERSATION PHASES (use these as 'topic' headings in the 'notes' output):")
    for t in topics:
        s_m, s_s = divmod(t.get("start_ms", 0) // 1000, 60)
        e_m, e_s = divmod(t.get("end_ms", 0) // 1000, 60)
        lines.append(f"  {t.get('name', 'Unknown')} ({s_m}:{s_s:02d}–{e_m}:{e_s:02d})")
        if t.get("summary"):
            lines.append(f"    {t['summary']}")
```

---

## Change 2: Update _parse_narrative_response()

```python
return {
    # ... existing fields ...
    "general_summary": parsed.get("general_summary", []),
    "notes": parsed.get("notes", []),
    "action_items": parsed.get("action_items", []),
    # ... rest of existing fields ...
}
```

And in the JSON fallback block:
```python
"general_summary": [],
"notes": [],
"action_items": [],
```

---

## Change 3: Update _fallback_narrative()

Build `general_summary` from existing insights:
```python
general_summary = []
# From entities
if entities.get("commitments"):
    general_summary.append(
        f"{len(entities['commitments'])} commitment(s) made during the session."
    )
if entities.get("objections"):
    resolved = sum(1 for o in entities["objections"] if o.get("resolved"))
    total = len(entities["objections"])
    general_summary.append(
        f"{total} objection(s) raised, {resolved} resolved."
    )
# From voice summary
for spk, data in voice_summary.get("per_speaker", {}).items():
    if data.get("max_stress", 0) > 0.60:
        general_summary.append(
            f"{spk} showed elevated stress during the session."
        )
# From language summary
for spk, data in language_summary.get("per_speaker", {}).items():
    if data.get("buying_signal_count", 0) > 2:
        general_summary.append(
            f"{spk} showed {data['buying_signal_count']} buying signals."
        )
if not general_summary:
    general_summary.append(f"Session with {len(speakers)} speaker(s) completed.")
```

Build `notes` from topics + transcript segments:
```python
notes = []
for topic in entities.get("topics", []):
    # Find transcript segments that fall within this topic's time range
    topic_start = topic.get("start_ms", 0)
    topic_end = topic.get("end_ms", 0)
    topic_details = []
    for seg in transcript_segments:  # need to pass these to _fallback
        if seg.get("start_ms", 0) >= topic_start and seg.get("end_ms", 0) <= topic_end:
            if len(seg.get("text", "")) > 30:  # skip very short segments
                ts_s = seg["start_ms"] // 1000
                m, s = divmod(ts_s, 60)
                topic_details.append({
                    "speaker": seg.get("speaker", ""),
                    "text": seg["text"][:200],  # truncate for notes
                    "timestamp": f"{m}:{s:02d}",
                    "sub_details": [],
                })
    if topic_details:
        notes.append({
            "topic": topic.get("name", "Discussion"),
            "summary": topic.get("summary", ""),
            "details": topic_details[:10],  # cap at 10 per topic
        })
```

Build `action_items` from commitments:
```python
action_items = []
for com in entities.get("commitments", []):
    ts_s = com.get("timestamp_ms", com.get("start_ms", 0)) // 1000
    m, s = divmod(ts_s, 60)
    action_items.append({
        "assignee": com.get("speaker", "Unknown"),
        "task": com.get("text", ""),
        "deadline": None,
        "timestamp": f"{m}:{s:02d}",
        "context": "",
    })
# Also extract implicit action items from objections marked unresolved
for obj in entities.get("objections", []):
    if not obj.get("resolved"):
        ts_s = obj.get("timestamp_ms", obj.get("start_ms", 0)) // 1000
        m, s = divmod(ts_s, 60)
        action_items.append({
            "assignee": "",  # unassigned — needs follow-up
            "task": f"Follow up on unresolved objection: {obj.get('text', '')}",
            "deadline": None,
            "timestamp": f"{m}:{s:02d}",
            "context": f"Raised by {obj.get('speaker', 'unknown')}",
        })
```

---

## Change 4: Frontend Report Tab

**File:** SessionDetail.tsx (or equivalent report rendering component)

Add three new sections. They render for ALL content types (not type-gated):

### General Summary (before executive_summary or replacing it)

```tsx
{content?.general_summary && content.general_summary.length > 0 && (
  <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
    <h2 className="mb-3 text-sm font-semibold text-nexus-text-primary">
      📋 Summary
    </h2>
    <ul className="space-y-1.5">
      {content.general_summary.map((point, i) => (
        <li key={i} className="flex items-start gap-2 text-sm text-nexus-text-primary">
          <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent-blue" />
          {point}
        </li>
      ))}
    </ul>
  </section>
)}
```

### Notes (after key_facts, before speaker_analyses)

```tsx
{content?.notes && content.notes.length > 0 && (
  <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
    <h2 className="mb-4 text-sm font-semibold text-nexus-text-primary">
      📝 Notes
    </h2>
    <div className="space-y-5">
      {content.notes.map((topic, i) => (
        <div key={i}>
          <h3 className="text-sm font-semibold text-nexus-accent-purple mb-1">
            {topic.topic}
          </h3>
          {topic.summary && (
            <p className="text-xs text-nexus-text-muted mb-2 italic">{topic.summary}</p>
          )}
          <ul className="space-y-2 ml-2 border-l-2 border-nexus-border pl-3">
            {topic.details.map((detail, j) => (
              <li key={j} className="text-sm text-nexus-text-primary">
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 text-nexus-accent-purple">→</span>
                  <div>
                    <span className="font-medium">{detail.speaker}</span>
                    {detail.timestamp && (
                      <span className="ml-1.5 text-[10px] text-nexus-text-muted">
                        ({detail.timestamp})
                      </span>
                    )}
                    <span className="ml-1">{detail.text}</span>
                    {detail.sub_details?.length > 0 && (
                      <ul className="mt-1 ml-3 space-y-0.5">
                        {detail.sub_details.map((sub, k) => (
                          <li key={k} className="text-xs text-nexus-text-secondary flex items-start gap-1.5">
                            <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-nexus-text-muted" />
                            {sub}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  </section>
)}
```

### Action Items (after recommendations, or as the final section)

```tsx
{content?.action_items && content.action_items.length > 0 && (
  <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
    <h2 className="mb-4 text-sm font-semibold text-nexus-text-primary">
      ✅ Action Items
    </h2>
    <div className="space-y-4">
      {/* Group by assignee */}
      {Object.entries(
        content.action_items.reduce((groups, item) => {
          const key = item.assignee || "Unassigned";
          (groups[key] = groups[key] || []).push(item);
          return groups;
        }, {})
      ).map(([assignee, items]) => (
        <div key={assignee}>
          <h3 className="text-xs font-semibold text-nexus-text-primary mb-1.5 uppercase tracking-wide">
            {assignee}
          </h3>
          <ul className="space-y-1.5 ml-2">
            {items.map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-nexus-text-primary">
                <span className="mt-0.5">•</span>
                <div>
                  {item.task}
                  {item.deadline && (
                    <span className="ml-1.5 text-xs text-amber-400 font-medium">
                      — {item.deadline}
                    </span>
                  )}
                  {item.timestamp && (
                    <span className="ml-1.5 text-[10px] text-nexus-text-muted">
                      ({item.timestamp})
                    </span>
                  )}
                  {item.context && (
                    <p className="text-xs text-nexus-text-muted mt-0.5">{item.context}</p>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  </section>
)}
```

---

## Complete Report Section Order

```
1. General Summary (NEW — bullet points, scannable)
2. Executive Summary (existing — 3-5 sentence paragraph for context)
3. Key Facts & Commitments (existing — timestamped facts)
4. Notes (NEW — topic-grouped discussion details)
5. Risk Assessment (interrogation only)
6. Speaker Analyses (existing — per-person behavioral profiles)
7. Key Moments (existing — timestamped events)
8. Cross-Modal Insights (existing — incongruence alerts)
9. Recommendations (existing — prioritized actions)
10. Action Items (NEW — assigned tasks with deadlines)
```

**General Summary goes FIRST** because users scan for "what happened" before reading details.
**Action Items goes LAST** because users check tasks after understanding context.
**Notes goes MIDDLE** because it's the detailed record — only read when drilling in.

---

## Backward Compatibility

- Old reports without `general_summary`, `notes`, `action_items` → fields are `[]` → sections don't render
- Frontend checks `content?.notes && content.notes.length > 0` before rendering
- LLM may produce empty arrays if the meeting is too short for topic grouping — that's fine
- `_fallback_narrative` produces these fields from entity data when LLM is unavailable
- No existing fields are removed or renamed

---

## Files Modified:

### Backend (1):
1. **services/fusion_agent/narrative.py**:
   - `_build_prompt()` — add 3 new fields to JSON schema + structure instructions (~30 lines)
   - `_build_context()` — add topic phase listing for LLM (~10 lines)
   - `_parse_narrative_response()` — parse 3 new fields (~5 lines)
   - `_fallback_narrative()` — build general_summary, notes, action_items from entities (~60 lines)

### Frontend (1):
2. **SessionDetail.tsx** (or equivalent):
   - General Summary section (~15 lines)
   - Notes section with topic grouping (~35 lines)
   - Action Items section with assignee grouping (~30 lines)