# services/api_gateway/neo4j_semantic_layer.py
"""
NEXUS Neo4j Semantic Layer — 10 pre-built Cypher tools.

Architecture:
  User question
    → select_tool()                   : GPT-4o picks 1 of 10 tools + extracts params
    → execute_tool()                  : runs hardcoded, tested Cypher for that tool
    → search_graph_context_fallback() : GPT-5 generates Cypher when no tool fits

Model Tiers:
  Tool selection:  gpt-4o  (MEDIUM — reliable structured output, ~$0.002/call)
  Cypher fallback: gpt-5   (HEAVY  — best Cypher reasoning, ~$0.02/call, ~10% of questions)
"""
import json
import logging

logger = logging.getLogger("nexus.gateway.semantic_layer")

# ─────────────────────────────────────────────────────────
# Speaker label helpers (content-type-aware)
# ─────────────────────────────────────────────────────────
async def fetch_session_speakers(session_id: str) -> list[dict]:
    """
    Fetch real speaker data for a session from Neo4j.
    Returns list of {label, name, role, talk_time_pct} dicts.
    Falls back to empty list if Neo4j is unavailable.
    """
    try:
        from shared.utils.neo4j_client import run_query
        rows = await run_query(
            """
            MATCH (spk:Speaker {session_id: $session_id})
            RETURN spk.label AS label,
                   spk.name  AS name,
                   spk.role  AS role,
                   spk.talk_time_pct AS talk_pct
            ORDER BY spk.label ASC
            """,
            session_id=session_id,
        )
        return [dict(r) for r in rows]
    except Exception:
        return []


_SIGNAL_TYPE_MAP = {
    "stress": "vocal_stress_score",
    "stressed": "vocal_stress_score",
    "anxiety": "vocal_stress_score",
    "pitch": "pitch_elevation_flag",
    "filler": "filler_detection",
    "fillers": "filler_detection",
    "ums": "filler_detection",
    "tone": "tone_classification",
    "pace": "speech_rate_anomaly",
    "speed": "speech_rate_anomaly",
    "rate": "speech_rate_anomaly",
    "pause": "pause_classification",
    "pauses": "pause_classification",
    "monotone": "monotone_flag",
    "interruption": "interruption_event",
    "interrupting": "interruption_event",
    "buying": "buying_signal",
    "buy": "buying_signal",
    "objection": "objection_signal",
    "sentiment": "sentiment_score",
    "emotion": "sentiment_score",
    "rapport": "rapport_indicator",
    "engagement": "conversation_engagement",
    "dominance": "dominance_score",
    "tension": "tension_cluster",
    "conflict": "conflict_score",
    "energy": "energy_level",
    "persuasion": "persuasion_technique",
    "credibility": "credibility_assessment",
    "urgency": "urgency_authenticity",
}

# ─────────────────────────────────────────────────────────
# Tool registry — 10 pre-built tools
# ─────────────────────────────────────────────────────────
NEXUS_TOOLS = [
    {
        "name": "get_causal_chain",
        "description": (
            "Find what signals caused or triggered a specific event. Traces backward through "
            "the signal chain to explain WHY something happened. Use when user asks "
            "'what caused', 'why did', 'what led to', 'what triggered'."
        ),
        "parameters": {
            "signal_type": {
                "description": "Signal type: tension_cluster, momentum_shift, vocal_stress_score, "
                               "stress_sentiment_incongruence, verbal_incongruence",
                "required": True,
                "default": None,
            },
            "timestamp_seconds": {
                "description": "Approximate time in seconds. Use 0 to find all.",
                "required": False,
                "default": 0,
            },
        },
        "cypher": """
MATCH (target:Signal {session_id: $session_id})
WHERE target.signal_type = $signal_type
  AND ($timestamp_seconds = 0
       OR (target.timestamp_ms > ($timestamp_seconds * 1000 - 10000)
           AND target.timestamp_ms < ($timestamp_seconds * 1000 + 10000)))
OPTIONAL MATCH (trigger:Signal)-[:TRIGGERED]->(target)
OPTIONAL MATCH (trigger)-[:OCCURRED_DURING]->(seg:Segment)
RETURN target.signal_type AS event, target.timestamp_ms / 1000.0 AS at_seconds,
       target.value AS event_value, target.speaker_label AS speaker,
       collect(DISTINCT {signal: trigger.signal_type, value: trigger.value,
               time: trigger.timestamp_ms / 1000.0, text: left(seg.text, 80)}) AS triggered_by
ORDER BY target.timestamp_ms LIMIT 10
""",
    },
    {
        "name": "get_topic_stress_correlation",
        "description": (
            "Find which conversation topics cause the most stress or negative reactions. "
            "Use when user asks 'which topics cause stress', 'what was stressful', "
            "'problematic topics'."
        ),
        "parameters": {
            "speaker_label": {
                "description": "Optional: Speaker_0 or Speaker_1",
                "required": False,
                "default": None,
            },
        },
        "cypher": """
MATCH (topic:Topic {session_id: $session_id})<-[:DISCUSSES]-(seg:Segment)
      <-[:OCCURRED_DURING]-(stress:Signal {signal_type: 'vocal_stress_score'})
WHERE stress.value > 0.3
  AND ($speaker_label IS NULL OR stress.speaker_label = $speaker_label)
RETURN topic.name AS topic, round(avg(stress.value) * 100) / 100 AS avg_stress,
       count(stress) AS stress_signals, collect(DISTINCT stress.speaker_label) AS speakers
ORDER BY avg_stress DESC
""",
    },
    {
        "name": "get_speaker_influence",
        "description": (
            "Show how one speaker's behavior affects the other. Use when user asks "
            "'how did X influence Y', 'speaker dynamics', 'reaction', "
            "'dominant speaker', 'who spoke most', 'talk time', "
            "'how did they respond to each other'."
        ),
        "parameters": {
            "speaker_a": {
                "description": "Optional: Speaker_0 or Speaker_1",
                "required": False,
                "default": None,
            },
        },
        "cypher": """
MATCH (a:Signal {session_id: $session_id})-[rel:INFLUENCED]->(b:Signal)
WHERE ($speaker_a IS NULL OR a.speaker_label = $speaker_a)
OPTIONAL MATCH (a)-[:OCCURRED_DURING]->(segA:Segment)
OPTIONAL MATCH (b)-[:OCCURRED_DURING]->(segB:Segment)
RETURN a.speaker_label AS influencer, a.signal_type AS trigger_signal,
       a.value_text AS trigger_detail, b.speaker_label AS responder,
       b.signal_type AS response_signal, b.value AS response_value,
       round(rel.lag_ms / 1000, 1) AS delay_sec,
       left(segA.text, 60) AS trigger_text, left(segB.text, 60) AS response_text
ORDER BY a.timestamp_ms LIMIT 15
""",
    },
    {
        "name": "get_unresolved_objections",
        "description": (
            "Find objections or concerns raised but never addressed. Use when user asks "
            "'unresolved', 'unanswered', 'open concerns'."
        ),
        "parameters": {},
        "cypher": """
MATCH (obj:Entity:Objection {session_id: $session_id})
OPTIONAL MATCH (obj)-[:MENTIONED_IN]->(seg:Segment)
OPTIONAL MATCH (obj)-[:RAISED_BY]->(spk:Speaker)
RETURN obj.text AS objection, obj.timestamp_ms / 1000.0 AS at_seconds,
       obj.resolved AS resolved, obj.resolved_at_ms / 1000.0 AS resolved_at,
       spk.label AS raised_by, left(seg.text, 80) AS context
ORDER BY obj.timestamp_ms
""",
    },
    {
        "name": "get_conversation_arc",
        "description": (
            "Show the complete flow of the conversation as topic phases with key signals. "
            "Use when user asks 'walk me through', 'what was the flow', 'summary', "
            "'what happened', 'conversation arc', 'how did it evolve', 'escalating tension', "
            "'resolution', 'stay flat', 'arc evolve', 'sentiment shift', 'incongruence', "
            "'mood change', 'dynamic change', 'emotional journey'."
        ),
        "parameters": {},
        "cypher": """
MATCH (topic:Topic {session_id: $session_id})
OPTIONAL MATCH (topic)<-[:DISCUSSES]-(seg:Segment)<-[:OCCURRED_DURING]-(sig:Signal)
WHERE sig.confidence > 0.4
  AND sig.signal_type IN ['buying_signal','objection_signal','tension_cluster',
                           'rapport_indicator','momentum_shift','vocal_stress_score',
                           'sentiment_score','tone_classification','energy_level',
                           'persuasion_technique','filler_detection','pitch_elevation_flag',
                           'emotional_intensity','disagreement','power_language_score']
  AND (sig.signal_type <> 'vocal_stress_score' OR sig.value > 0.4)
  AND (sig.signal_type <> 'sentiment_score' OR sig.value < -0.2 OR sig.value > 0.4)
WITH topic, collect(DISTINCT {type: sig.signal_type,
     value: coalesce(sig.value_text, toString(round(sig.value * 100) / 100)),
     speaker: sig.speaker_label}) AS signals
ORDER BY topic.start_ms
RETURN topic.name AS phase, round(topic.start_ms / 1000) AS start_sec,
       round(topic.end_ms / 1000) AS end_sec, signals
""",
    },
    {
        "name": "get_signal_decomposition",
        "description": (
            "Explain why a composite score has a specific value by showing components. "
            "Use when user asks 'why is rapport low', 'explain engagement', "
            "'what drives conflict'."
        ),
        "parameters": {
            "signal_type": {
                "description": "rapport_indicator, conversation_engagement, conflict_score, dominance_score",
                "required": True,
                "default": None,
            },
            "speaker_label": {
                "description": "Optional: Speaker_0 or Speaker_1",
                "required": False,
                "default": None,
            },
        },
        "cypher": """
MATCH (sig:Signal {session_id: $session_id, signal_type: $signal_type})
WHERE ($speaker_label IS NULL OR sig.speaker_label = $speaker_label)
OPTIONAL MATCH (sig)-[:OCCURRED_DURING]->(seg:Segment)
RETURN sig.speaker_label AS speaker, sig.value AS score, sig.value_text AS assessment,
       sig.confidence AS confidence, sig.metadata AS components, left(seg.text, 80) AS context
ORDER BY sig.value DESC LIMIT 5
""",
    },
    {
        "name": "get_convergent_moments",
        "description": (
            "Find critical moments where multiple significant signals co-occur — pivotal points. "
            "Use when user asks 'key moments', 'turning points', 'most important', "
            "'what stood out'."
        ),
        "parameters": {},
        "cypher": """
MATCH (sig:Signal {session_id: $session_id})-[:OCCURRED_DURING]->(seg:Segment)
WHERE sig.confidence > 0.4
  AND sig.signal_type IN ['vocal_stress_score','objection_signal','buying_signal',
                           'tension_cluster','momentum_shift','pitch_elevation_flag',
                           'filler_detection','interruption_event','persuasion_technique',
                           'empathy_language']
  AND (sig.signal_type <> 'vocal_stress_score' OR sig.value > 0.4)
WITH seg, collect(DISTINCT sig.signal_type) AS types,
     collect(DISTINCT {type: sig.signal_type,
                        value: coalesce(sig.value_text, toString(round(sig.value*100)/100))}) AS signals
WHERE size(types) >= 2
RETURN seg.start_ms / 1000.0 AS time_sec, seg.speaker_label AS speaker,
       left(seg.text, 80) AS what_was_said, size(types) AS signal_count, signals
ORDER BY signal_count DESC, seg.start_ms LIMIT 10
""",
    },
    {
        "name": "get_speaker_summary",
        "description": (
            "Get behavioral summary for a speaker — stress, tone, engagement, talk time. "
            "Use when user asks 'how did X do', 'seller performance', "
            "'tell me about the prospect'."
        ),
        "parameters": {
            "speaker_label": {
                "description": "Speaker_0, Speaker_1, or omit for all speakers",
                "required": False,
                "default": None,
            },
        },
        "cypher": """
MATCH (spk:Speaker {session_id: $session_id})
WHERE $speaker_label IS NULL OR spk.label = $speaker_label
OPTIONAL MATCH (stress:Signal {session_id: $session_id, signal_type: 'vocal_stress_score',
                                speaker_label: spk.label})
OPTIONAL MATCH (tone:Signal {session_id: $session_id, signal_type: 'tone_classification',
                               speaker_label: spk.label})
WHERE tone.value_text <> 'neutral'
WITH spk, round(avg(stress.value)*100)/100 AS avg_stress, max(stress.value) AS peak_stress,
     collect(DISTINCT tone.value_text) AS tones
RETURN spk.label AS speaker, spk.name AS name, spk.role AS role,
       spk.talk_time_pct AS talk_pct, spk.word_count AS words,
       avg_stress, peak_stress, tones
ORDER BY spk.label
""",
    },
    {
        "name": "get_signal_timeline",
        "description": (
            "Get signals of a specific type in chronological order. Use when user asks "
            "'stress timeline', 'when did buying signals appear', 'track tone changes'."
        ),
        "parameters": {
            "signal_type": {
                "description": "vocal_stress_score, tone_classification, buying_signal, "
                               "objection_signal, filler_detection, pitch_elevation_flag, "
                               "sentiment_score, speech_rate_anomaly, interruption_event",
                "required": True,
                "default": None,
            },
            "speaker_label": {
                "description": "Optional: Speaker_0 or Speaker_1",
                "required": False,
                "default": None,
            },
        },
        "cypher": """
MATCH (sig:Signal {session_id: $session_id, signal_type: $signal_type})
WHERE ($speaker_label IS NULL OR sig.speaker_label = $speaker_label)
OPTIONAL MATCH (sig)-[:OCCURRED_DURING]->(seg:Segment)
RETURN sig.timestamp_ms / 1000.0 AS time_sec, sig.speaker_label AS speaker,
       sig.value AS value, sig.value_text AS detail, sig.confidence AS confidence,
       left(seg.text, 60) AS what_was_said
ORDER BY sig.timestamp_ms LIMIT 30
""",
    },
    {
        "name": "get_entity_network",
        "description": (
            "Find entities (people, companies, products, commitments) mentioned in the session. "
            "Use when user asks 'who was mentioned', 'what companies', 'any commitments'."
        ),
        "parameters": {
            "entity_type": {
                "description": "Optional: Person, Company, Product, Objection, Commitment",
                "required": False,
                "default": None,
            },
        },
        "cypher": """
MATCH (e:Entity {session_id: $session_id})
WHERE ($entity_type IS NULL OR $entity_type IN labels(e))
OPTIONAL MATCH (e)-[:MENTIONED_IN]->(seg:Segment)
OPTIONAL MATCH (e)-[:RAISED_BY]->(spk:Speaker)
RETURN labels(e) AS types, e.name AS name,
       coalesce(e.text, e.context, '') AS detail,
       e.timestamp_ms / 1000.0 AS at_seconds, spk.label AS raised_by,
       left(seg.text, 60) AS context
ORDER BY e.timestamp_ms LIMIT 20
""",
    },
]

# Build lookup index by name
_TOOL_INDEX = {t["name"]: t for t in NEXUS_TOOLS}

# ─────────────────────────────────────────────────────────
# Tool selection (GPT-4o — MEDIUM tier, ~$0.002/call)
# ─────────────────────────────────────────────────────────

_TOOL_SELECTION_PROMPT = """You are a tool selector for NEXUS conversation analysis. Pick the BEST tool and extract parameters.

Available tools:
{tool_descriptions}

Rules:
1. Pick exactly ONE tool, or "none" if no tool fits.
2. Speaker mapping: "seller"/"rep"/"agent"/"caller" = "Speaker_0", "buyer"/"prospect"/"customer" = "Speaker_1". Interviews: "interviewer" = "Speaker_0", "candidate" = "Speaker_1". Use real speaker labels from the session info below when available.
3. Signal mapping: "stress" = "vocal_stress_score", "buying" = "buying_signal", "objection" = "objection_signal", "tension" = "tension_cluster", "rapport" = "rapport_indicator", "engagement" = "conversation_engagement", "conflict" = "conflict_score", "dominance" = "dominance_score", "pitch" = "pitch_elevation_flag", "filler"/"ums" = "filler_detection", "tone" = "tone_classification", "pace" = "speech_rate_anomaly"
4. Timestamps: "at 2:30" = 150, "minute 5" = 300. Omit if not mentioned.
5. Set params to null if not mentioned in the question.

{speaker_hint}

Return ONLY valid JSON: {{"tool": "name_or_none", "params": {{"param": "value"}}}}"""


async def select_tool(question: str, session_id: str, history: list[dict] | None = None) -> dict:
    """
    GPT-4o picks the best pre-built tool and extracts its parameters.

    Speaker labels are resolved from the real session data (Neo4j) so the LLM
    can map natural-language role references ("the seller", "Holly") to Speaker_N labels.
    history (last 2-4 turns) is appended so follow-up questions ("elaborate more",
    "what about Speaker_0?") resolve to the correct tool.

    Returns {"tool": "tool_name_or_none", "params": {...}}
    """
    try:
        from shared.utils.llm_client import acomplete
    except ImportError:
        return {"tool": "none", "params": {}}

    # Fetch real speaker info for this session
    speakers = await fetch_session_speakers(session_id)
    if speakers:
        speaker_lines = "\n".join(
            f"  - label={s['label']}"
            + (f", name={s['name']}" if s.get("name") else "")
            + (f", role={s['role']}" if s.get("role") else "")
            + (f", talk_pct={s['talk_pct']:.0f}%" if s.get("talk_pct") else "")
            for s in speakers
        )
        speaker_hint = f"Speakers in this session:\n{speaker_lines}"
    else:
        speaker_hint = (
            "Speaker labels not yet available. "
            "Use 'Speaker_0' for the first/primary speaker and 'Speaker_1' for the second."
        )

    tool_descriptions = json.dumps(
        [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    k: v["description"]
                    for k, v in t["parameters"].items()
                },
            }
            for t in NEXUS_TOOLS
        ],
        indent=2,
    )

    history_block = ""
    if history:
        recent = history[-4:]  # last 2 exchanges (4 messages)
        lines = []
        for m in recent:
            role = m.get("role", "user").title()
            content = str(m.get("content", ""))[:300]
            lines.append(f"{role}: {content}")
        history_block = "\n\nPrevious conversation (use for follow-up resolution):\n" + "\n".join(lines)

    prompt = _TOOL_SELECTION_PROMPT.format(
        tool_descriptions=tool_descriptions,
        speaker_hint=speaker_hint,
    ) + history_block + f'\n\nQuestion: "{question}"'

    def _parse(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.split("\n")
                if not line.strip().startswith("```")
            )
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            params = result.get("params", {})
            # Resolve natural-language signal names to canonical types
            for key in ("signal_type",):
                if key in params and isinstance(params[key], str):
                    lower = params[key].lower().strip()
                    params[key] = _SIGNAL_TYPE_MAP.get(lower, params[key])
            return result
        raise ValueError(f"No JSON found in: {text[:80]}")

    try:
        raw = await acomplete(
            system_prompt="You are a precise JSON tool selector. Return only valid JSON, nothing else.",
            user_prompt=prompt,
            max_tokens=200,
            temperature=0.0,
            model="gpt-4o",  # MEDIUM tier
        )
        return _parse(raw)
    except Exception as e:
        logger.warning(f"[{session_id}] Tool selection (gpt-4o) failed: {e}")
        return {"tool": "none", "params": {}}


# ─────────────────────────────────────────────────────────
# Tool execution (hardcoded Cypher, no LLM)
# ─────────────────────────────────────────────────────────

async def execute_tool(tool_name: str, params: dict, session_id: str) -> str:
    """
    Execute a pre-built semantic layer tool and return formatted results.
    session_id is injected separately — never included in the LLM-supplied params.
    Returns "" if tool not found, Neo4j unavailable, or no results.
    """
    tool = _TOOL_INDEX.get(tool_name)
    if not tool:
        logger.warning(f"Unknown tool requested: {tool_name}")
        return ""

    # Build Cypher params: session_id + tool params (with defaults for missing optionals)
    cypher_params: dict = {"session_id": session_id}
    for pname, pdef in tool["parameters"].items():
        cypher_params[pname] = params.get(pname, pdef.get("default"))

    # timestamp_seconds: treat missing/None as 0 (search whole session)
    if "timestamp_seconds" in cypher_params and cypher_params["timestamp_seconds"] is None:
        cypher_params["timestamp_seconds"] = 0

    try:
        from shared.utils.neo4j_client import run_query
        results = await run_query(tool["cypher"], **cypher_params)
    except Exception as e:
        logger.warning(f"Tool execution failed ({tool_name}): {e}")
        return ""

    if not results:
        return f"No results found for {tool_name.replace('_', ' ')}."

    lines = [f"Graph analysis ({tool_name.replace('_', ' ')}, {len(results)} result(s)):"]
    for i, record in enumerate(results[:20]):
        parts = []
        for k, v in record.items():
            if v is None:
                continue
            v_str = str(v)
            if len(v_str) > 300:
                v_str = v_str[:300] + "..."
            parts.append(f"{k}={v_str}")
        if parts:
            lines.append(f"  {i + 1}. " + ", ".join(parts))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# GPT-5 Cypher fallback (HEAVY tier, ~10% of questions)
# ─────────────────────────────────────────────────────────

_CYPHER_GENERATION_PROMPT = """\
You are a Neo4j Cypher expert for the NEXUS behavioural analysis graph.
Generate ONE valid read-only Cypher query to answer the user's question.

SCHEMA:
{schema}

RULES:
1. ALWAYS filter with {{session_id: $session_id}} on every node pattern.
2. NEVER use MERGE, CREATE, SET, DELETE, or any write clause.
3. NEVER use CALL procedures that modify data.
4. $session_id is the ONLY parameter available — do not reference any other parameters.
5. Use OPTIONAL MATCH for fields that might be missing.
6. LIMIT results to 20 rows maximum.
7. Return column names that are human-readable (use AS aliases).
8. If you cannot answer with a safe read query, return exactly: SKIP

Return ONLY the Cypher query or SKIP — no markdown, no explanation.
"""


async def search_graph_context_fallback(question: str, session_id: str) -> str:
    """
    GPT-5 Cypher fallback: activated when no pre-built tool fits the question.

    GPT-5 generates a Cypher query from the schema; we execute it with hardcoded
    $session_id injection. Handles unusual or complex questions. ~$0.02/call.

    Returns formatted graph results as text, or "" on any failure.
    """
    try:
        from shared.utils.llm_client import acomplete
        from neo4j_sync import GRAPH_SCHEMA_HINT
    except ImportError as e:
        logger.warning(f"GPT-5 Cypher fallback import error: {e}")
        return ""

    system_prompt = _CYPHER_GENERATION_PROMPT.format(schema=GRAPH_SCHEMA_HINT)
    # Do NOT include the actual session_id value — the model would hardcode it
    # instead of using $session_id as a parameter. It is injected at execution time.
    user_prompt = (
        f'Question: "{question}"\n\n'
        f'Remember: use $session_id as a Cypher parameter (do NOT hardcode any UUID). '
        f'The query will be executed with $session_id already bound.'
    )

    try:
        raw_cypher = await acomplete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=800,
            temperature=0.0,
            model="gpt-5",  # HEAVY tier — best Cypher reasoning
        )
    except Exception as e:
        logger.warning(f"[{session_id}] GPT-5 Cypher generation failed: {e}")
        return ""

    cypher = raw_cypher.strip()

    # Strip markdown fences if the model wrapped the query anyway
    if cypher.startswith("```"):
        cypher = "\n".join(
            line for line in cypher.split("\n")
            if not line.strip().startswith("```")
        ).strip()

    if not cypher or cypher.upper() == "SKIP":
        logger.info(f"[{session_id}] GPT-5 Cypher fallback: no safe query possible")
        return ""

    # Safety: reject any write operations
    upper = cypher.upper()
    for banned in ("MERGE", "CREATE", "SET ", "DELETE", "DETACH", "REMOVE", "DROP"):
        if banned in upper:
            logger.warning(
                f"[{session_id}] GPT-5 generated write Cypher — blocked: {cypher[:120]}"
            )
            return ""

    # Require $session_id to prevent cross-session data leaks
    if "$session_id" not in cypher:
        logger.warning(f"[{session_id}] GPT-5 Cypher missing $session_id — blocked")
        return ""

    try:
        from shared.utils.neo4j_client import run_query
        results = await run_query(cypher, session_id=session_id)
    except Exception as e:
        logger.warning(f"[{session_id}] GPT-5 Cypher execution failed: {e}")
        return ""

    if not results:
        return ""

    lines = [f"Graph analysis (GPT-5 fallback query, {len(results)} result(s)):"]
    for i, record in enumerate(results[:20]):
        parts = []
        for k, v in record.items():
            if v is None:
                continue
            v_str = str(v)
            if len(v_str) > 300:
                v_str = v_str[:300] + "..."
            parts.append(f"{k}={v_str}")
        if parts:
            lines.append(f"  {i + 1}. " + ", ".join(parts))

    logger.info(f"[{session_id}] GPT-5 Cypher fallback: {len(results)} rows returned")
    return "\n".join(lines)
