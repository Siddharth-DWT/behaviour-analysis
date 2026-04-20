# services/api_gateway/neo4j_sync.py
"""
NEXUS API Gateway — Neo4j Sync (Phase 3A: Single-Session Knowledge Graph)

After the analysis pipeline completes for a session, this module reads the
canonical data back from PostgreSQL and persists it as a property graph in
Neo4j. PostgreSQL remains the source of truth — Neo4j is a re-buildable
projection used for causal queries, RAG-graph hybrid chat, and exploration.

Node types:
  Session, Speaker, Segment, Topic, Signal, FusionInsight, Entity, Alert

Relationship types built today (Phase 3A):
  Attribution / structural:
    (Speaker)-[:PARTICIPATED_IN]->(Session)
    (Segment)-[:PART_OF]->(Session)
    (Topic)-[:OCCURRED_IN]->(Session)
    (Segment)-[:SPOKEN_BY]->(Speaker)
    (Signal)-[:EMITTED_BY]->(Speaker)
    (Alert)-[:RAISED_FOR]->(Session)
  Temporal:
    (Segment)-[:NEXT]->(Segment)
    (Topic)-[:FOLLOWED_BY]->(Topic)
  Semantic:
    (Segment)-[:DISCUSSES]->(Topic)        (temporal containment)
    (Signal)-[:OCCURRED_DURING]->(Segment) (temporal containment)
    (Entity)-[:MENTIONED_IN]->(Segment)
    (Entity)-[:RAISED_BY]->(Speaker)       (people / objections w/ speaker_label)
    (Entity:Objection)-[:RESOLVED_IN]->(Segment)  (when resolved_at_ms set)
  Causal (post-load Cypher MERGE):
    (Signal)-[:CONTRADICTS]->(Signal)      (sentiment + stress same speaker, ≤3s)
    (Signal)-[:REINFORCES]->(Signal)       (confident tone + power language ≤3s)
    (Signal)-[:TRIGGERED]->(Signal)        (stress spike → tension cluster ≤5s)
    (Signal)-[:INFLUENCED]->(Signal)       (cross-speaker temporal causality ≤30s)

Sync is non-fatal: if Neo4j is unreachable or any step fails, the pipeline
still completes. Old session graphs can be rebuilt by re-invoking sync_session.
"""
import os
import json
import logging
import threading
from typing import Optional

logger = logging.getLogger("nexus.gateway.neo4j_sync")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "nexus_graph_2026")

# Lazy-loaded driver (one per process)
_driver = None
_driver_unavailable = False
# Guard driver init against concurrent inits
_driver_lock = threading.Lock()


def _get_driver():
    """Lazy-init the AsyncGraphDatabase driver. Returns None if neo4j pkg missing.

    Thread-safe: guards init with a lock to avoid races. ImportError is treated
    as permanent for the process (package missing). Other runtime errors are
    treated as transient (do not flip the _driver_unavailable flag).
    """
    global _driver, _driver_unavailable
    if _driver_unavailable:
        return None
    if _driver is not None:
        return _driver

    with _driver_lock:
        # Double-check after acquiring lock
        if _driver is not None:
            return _driver
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            logger.warning("neo4j package not installed — Neo4j sync disabled")
            _driver_unavailable = True
            return None
        try:
            _driver = AsyncGraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            logger.info(f"Neo4j driver initialised: {NEO4J_URI}")
            return _driver
        except Exception as e:
            # Transient failure (network / auth). Log but don't permanently
            # disable Neo4j for the process — allow retries later.
            logger.warning(f"Neo4j driver init failed: {e}")
            return None


async def close_driver():
    """Close the driver on app shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


# ─────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────

async def sync_session(pool, session_id: str) -> bool:
    """
    Read session data from PostgreSQL and write a single-session graph to Neo4j.

    Returns True on success, False on any failure (caller treats as non-fatal).
    """
    driver = _get_driver()
    if driver is None:
        return False

    try:
        # Pull everything we need from PG in parallel-friendly chunks
        session_row = await _fetch_session(pool, session_id)
        if not session_row:
            logger.warning(f"[{session_id}] sync_session: session not found in PG")
            return False

        speakers = await _fetch_speakers(pool, session_id)
        segments = await _fetch_segments(pool, session_id)
        signals = await _fetch_signals(pool, session_id)
        alerts = await _fetch_alerts(pool, session_id)
        report = await _fetch_report(pool, session_id)
        entities_json = (report or {}).get("entities") or {}

        topics = entities_json.get("topics") or []
        people = entities_json.get("people") or []
        companies = entities_json.get("companies") or []
        products = entities_json.get("products_services") or []
        objections = entities_json.get("objections") or []
        commitments = entities_json.get("commitments") or []

        async with driver.session() as nsession:
            await _wipe_session(nsession, session_id)
            await _create_session_node(nsession, session_id, session_row)
            await _create_speaker_nodes(nsession, session_id, speakers)
            await _create_segment_nodes(nsession, session_id, segments, speakers)
            await _create_topic_nodes(nsession, session_id, topics)
            await _create_signal_nodes(nsession, session_id, signals, speakers)
            await _create_entity_nodes(
                nsession, session_id, people, companies, products,
                objections, commitments,
            )
            await _create_alert_nodes(nsession, session_id, alerts)
            await _build_temporal_edges(nsession, session_id)
            await _build_containment_edges(nsession, session_id)
            await _build_causal_edges(nsession, session_id)
            await _build_influence_edges(nsession, session_id)
            await _link_fusion_to_signals(nsession, session_id)

        node_counts = {
            "speakers": len(speakers),
            "segments": len(segments),
            "topics": len(topics),
            "signals": len(signals),
            "alerts": len(alerts),
            "people": len(people),
            "objections": len(objections),
            "commitments": len(commitments),
        }

        # Verify edge counts after sync
        async with driver.session() as nsession:
            edge_result = await nsession.run(
                """
                MATCH ()-[r]->()
                WHERE (startNode(r)).session_id = $sid
                   OR (endNode(r)).session_id = $sid
                RETURN type(r) AS rel, count(*) AS cnt
                ORDER BY cnt DESC
                """,
                sid=session_id,
            )
            edge_counts = {rec["rel"]: rec["cnt"] async for rec in edge_result}

        logger.info(
            f"[{session_id}] Neo4j sync complete — nodes: {node_counts} | "
            f"edges: {edge_counts}"
        )
        return True

    except Exception as e:
        logger.warning(f"[{session_id}] Neo4j sync failed (non-fatal): {e}")
        return False


# ─────────────────────────────────────────────────────────
# Postgres reads
# ─────────────────────────────────────────────────────────

async def _fetch_session(pool, session_id: str) -> Optional[dict]:
    row = await pool.fetchrow(
        """
        SELECT id, title, session_type, meeting_type, status,
               duration_ms, speaker_count, created_at, completed_at
        FROM sessions WHERE id = $1
        """,
        session_id,
    )
    return dict(row) if row else None


async def _fetch_speakers(pool, session_id: str) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, speaker_label, speaker_name, role,
               total_talk_time_ms, talk_time_pct, total_words, calibration_confidence
        FROM speakers WHERE session_id = $1
        """,
        session_id,
    )
    return [dict(r) for r in rows]


async def _fetch_segments(pool, session_id: str) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, speaker_id, segment_index, start_ms, end_ms, text,
               word_count, sentiment, sentiment_score
        FROM transcript_segments
        WHERE session_id = $1
        ORDER BY segment_index ASC
        """,
        session_id,
    )
    return [dict(r) for r in rows]


async def _fetch_signals(pool, session_id: str) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, speaker_id, agent, signal_type, value, value_text,
               confidence, window_start_ms, window_end_ms, metadata
        FROM signals
        WHERE session_id = $1
        ORDER BY window_start_ms ASC
        """,
        session_id,
    )
    return [dict(r) for r in rows]


async def _fetch_alerts(pool, session_id: str) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, speaker_id, alert_type, severity, title, description,
               timestamp_ms, evidence
        FROM alerts WHERE session_id = $1
        """,
        session_id,
    )
    return [dict(r) for r in rows]


async def _fetch_report(pool, session_id: str) -> Optional[dict]:
    row = await pool.fetchrow(
        """
        SELECT content FROM session_reports
        WHERE session_id = $1
        ORDER BY generated_at DESC LIMIT 1
        """,
        session_id,
    )
    if not row or not row["content"]:
        return None
    content = row["content"]
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}
    return dict(content)


# ─────────────────────────────────────────────────────────
# Neo4j writes — node creation
# ─────────────────────────────────────────────────────────

async def _wipe_session(nsession, session_id: str):
    """Delete any pre-existing graph for this session before re-syncing.

    Two-step: delete relationships first (avoids DETACH DELETE memory pressure
    on large graphs), then delete orphaned nodes.
    """
    await nsession.run(
        """
        MATCH (n {session_id: $sid})-[r]-()
        DELETE r
        """,
        sid=session_id,
    )
    await nsession.run(
        """
        MATCH (n {session_id: $sid})
        DELETE n
        """,
        sid=session_id,
    )


async def _create_session_node(nsession, session_id: str, row: dict):
    await nsession.run(
        """
        MERGE (s:Session {id: $sid})
        SET s.session_id    = $sid,
            s.title         = $title,
            s.session_type  = $session_type,
            s.meeting_type  = $meeting_type,
            s.status        = $status,
            s.duration_ms   = $duration_ms,
            s.speaker_count = $speaker_count
        """,
        sid=session_id,
        title=row.get("title") or "",
        session_type=row.get("session_type") or "",
        meeting_type=row.get("meeting_type") or "",
        status=row.get("status") or "",
        duration_ms=int(row.get("duration_ms") or 0),
        speaker_count=int(row.get("speaker_count") or 0),
    )


async def _create_speaker_nodes(nsession, session_id: str, speakers: list[dict]):
    for spk in speakers:
        await nsession.run(
            """
            MERGE (s:Speaker {id: $id})
            SET s.session_id        = $sid,
                s.label             = $label,
                s.name              = $name,
                s.role              = $role,
                s.talk_time_ms      = $talk_time_ms,
                s.talk_time_pct     = $talk_time_pct,
                s.word_count        = $word_count
            WITH s
            MATCH (sess:Session {id: $sid})
            MERGE (s)-[:PARTICIPATED_IN]->(sess)
            """,
            id=str(spk["id"]),
            sid=session_id,
            label=spk.get("speaker_label") or "",
            name=spk.get("speaker_name") or "",
            role=spk.get("role") or "",
            talk_time_ms=int(spk.get("total_talk_time_ms") or 0),
            talk_time_pct=float(spk.get("talk_time_pct") or 0.0),
            word_count=int(spk.get("total_words") or 0),
        )


async def _create_segment_nodes(
    nsession, session_id: str, segments: list[dict], speakers: list[dict],
):
    speaker_label_by_id = {
        str(s["id"]): s.get("speaker_label") or "" for s in speakers
    }
    for seg in segments:
        spk_uuid = seg.get("speaker_id")
        spk_label = speaker_label_by_id.get(str(spk_uuid)) if spk_uuid else ""
        await nsession.run(
            """
            MERGE (seg:Segment {id: $id})
            SET seg.session_id    = $sid,
                seg.segment_index = $idx,
                seg.start_ms      = $start_ms,
                seg.end_ms        = $end_ms,
                seg.text          = $text,
                seg.speaker_label = $spk_label,
                seg.sentiment     = $sentiment,
                seg.sentiment_score = $sentiment_score,
                seg.word_count    = $word_count
            WITH seg
            MATCH (sess:Session {id: $sid})
            MERGE (seg)-[:PART_OF]->(sess)
            """,
            id=str(seg["id"]),
            sid=session_id,
            idx=int(seg.get("segment_index") or 0),
            start_ms=int(seg.get("start_ms") or 0),
            end_ms=int(seg.get("end_ms") or 0),
            text=seg.get("text") or "",
            spk_label=spk_label or "",
            sentiment=seg.get("sentiment") or "",
            sentiment_score=float(seg.get("sentiment_score") or 0.0),
            word_count=int(seg.get("word_count") or 0),
        )

        if spk_uuid:
            await nsession.run(
                """
                MATCH (seg:Segment {id: $sid_seg})
                MATCH (spk:Speaker {id: $spk_id})
                MERGE (seg)-[:SPOKEN_BY]->(spk)
                """,
                sid_seg=str(seg["id"]),
                spk_id=str(spk_uuid),
            )


async def _create_topic_nodes(nsession, session_id: str, topics: list[dict]):
    for i, topic in enumerate(topics):
        topic_id = f"{session_id}::topic::{i}"
        await nsession.run(
            """
            MERGE (t:Topic {id: $id})
            SET t.session_id = $sid,
                t.name       = $name,
                t.start_ms   = $start_ms,
                t.end_ms     = $end_ms,
                t.order_idx  = $idx
            WITH t
            MATCH (sess:Session {id: $sid})
            MERGE (t)-[:OCCURRED_IN]->(sess)
            """,
            id=topic_id,
            sid=session_id,
            name=topic.get("name") or f"Phase {i+1}",
            start_ms=int(topic.get("start_ms") or 0),
            end_ms=int(topic.get("end_ms") or 0),
            idx=i,
        )


async def _create_signal_nodes(
    nsession, session_id: str, signals: list[dict], speakers: list[dict],
):
    speaker_label_by_id = {
        str(s["id"]): s.get("speaker_label") or "" for s in speakers
    }
    for sig in signals:
        spk_uuid = sig.get("speaker_id")
        spk_label = speaker_label_by_id.get(str(spk_uuid)) if spk_uuid else ""
        agent = sig.get("agent") or "unknown"
        # FusionInsight is a labeled subset of Signal so the same node can be
        # matched as either Signal or FusionInsight.
        labels = ":Signal" + (":FusionInsight" if agent == "fusion" else "")

        metadata = sig.get("metadata")
        if isinstance(metadata, dict):
            metadata_str = json.dumps(metadata)
        elif isinstance(metadata, str):
            metadata_str = metadata
        else:
            metadata_str = "{}"
        # Compute a sensible timestamp for the signal. Use midpoint of window
        # when available (avoids biasing to window_start), otherwise fall back
        # to the start.
        window_start = int(sig.get("window_start_ms") or 0)
        window_end = int(sig.get("window_end_ms") or 0)
        if window_end and window_end > window_start:
            timestamp_ms = (window_start + window_end) // 2
        else:
            timestamp_ms = window_start

        cypher = f"""
            MERGE (n{labels} {{id: $id}})
            SET n.session_id      = $sid,
                n.agent           = $agent,
                n.signal_type     = $signal_type,
                n.value           = $value,
                n.value_text      = $value_text,
                n.confidence      = $confidence,
                n.window_start_ms = $window_start_ms,
                n.window_end_ms   = $window_end_ms,
                n.timestamp_ms    = $timestamp_ms,
                n.speaker_label   = $speaker_label,
                n.metadata        = $metadata
        """
        await nsession.run(
            cypher,
            id=str(sig["id"]),
            sid=session_id,
            agent=agent,
            signal_type=sig.get("signal_type") or "",
            value=float(sig.get("value")) if sig.get("value") is not None else None,
            value_text=sig.get("value_text") or "",
            confidence=float(sig.get("confidence") or 0.0),
            window_start_ms=window_start,
            window_end_ms=window_end,
            timestamp_ms=timestamp_ms,
            speaker_label=spk_label or "",
            metadata=metadata_str,
        )

        if spk_uuid:
            await nsession.run(
                """
                MATCH (sig:Signal {id: $sig_id})
                MATCH (spk:Speaker {id: $spk_id})
                MERGE (sig)-[:EMITTED_BY]->(spk)
                """,
                sig_id=str(sig["id"]),
                spk_id=str(spk_uuid),
            )


async def _create_entity_nodes(
    nsession, session_id: str,
    people: list[dict], companies: list[dict], products: list[dict],
    objections: list[dict], commitments: list[dict],
):
    # ── People ──
    for i, p in enumerate(people):
        eid = f"{session_id}::person::{i}"
        await nsession.run(
            """
            MERGE (e:Entity:Person {id: $id})
            SET e.session_id      = $sid,
                e.name            = $name,
                e.role            = $role,
                e.speaker_label   = $spk_label,
                e.first_mention_ms = $first_mention_ms
            """,
            id=eid,
            sid=session_id,
            name=p.get("name") or "",
            role=p.get("role") or "",
            spk_label=p.get("speaker_label") or "",
            first_mention_ms=int(p.get("first_mention_ms") or 0),
        )
        # Link to Speaker by speaker_label if known
        spk_label = p.get("speaker_label") or ""
        if spk_label:
            await nsession.run(
                """
                MATCH (e:Entity:Person {id: $eid})
                MATCH (spk:Speaker {session_id: $sid, label: $label})
                MERGE (e)-[:IS_SPEAKER]->(spk)
                """,
                eid=eid,
                sid=session_id,
                label=spk_label,
            )

    # ── Companies ──
    for i, c in enumerate(companies):
        eid = f"{session_id}::company::{i}"
        await nsession.run(
            """
            MERGE (e:Entity:Company {id: $id})
            SET e.session_id      = $sid,
                e.name            = $name,
                e.context         = $context,
                e.first_mention_ms = $first_mention_ms
            """,
            id=eid,
            sid=session_id,
            name=c.get("name") or "",
            context=c.get("context") or "",
            first_mention_ms=int(c.get("first_mention_ms") or 0),
        )

    # ── Products / services ──
    for i, p in enumerate(products):
        eid = f"{session_id}::product::{i}"
        await nsession.run(
            """
            MERGE (e:Entity:Product {id: $id})
            SET e.session_id = $sid,
                e.name       = $name,
                e.context    = $context
            """,
            id=eid,
            sid=session_id,
            name=p.get("name") or "",
            context=p.get("context") or "",
        )

    # ── Objections ──
    for i, o in enumerate(objections):
        eid = f"{session_id}::objection::{i}"
        speaker_label = o.get("speaker") or ""
        await nsession.run(
            """
            MERGE (e:Entity:Objection {id: $id})
            SET e.session_id     = $sid,
                e.text           = $text,
                e.speaker_label  = $spk_label,
                e.timestamp_ms   = $timestamp_ms,
                e.resolved       = $resolved,
                e.resolved_at_ms = $resolved_at_ms
            """,
            id=eid,
            sid=session_id,
            text=o.get("text") or "",
            spk_label=speaker_label,
            timestamp_ms=int(o.get("timestamp_ms") or 0),
            resolved=bool(o.get("resolved", False)),
            resolved_at_ms=int(o.get("resolved_at_ms") or 0),
        )
        if speaker_label:
            await nsession.run(
                """
                MATCH (e:Entity:Objection {id: $eid})
                MATCH (spk:Speaker {session_id: $sid, label: $label})
                MERGE (e)-[:RAISED_BY]->(spk)
                """,
                eid=eid,
                sid=session_id,
                label=speaker_label,
            )

    # ── Commitments ──
    for i, c in enumerate(commitments):
        eid = f"{session_id}::commitment::{i}"
        speaker_label = c.get("speaker") or ""
        await nsession.run(
            """
            MERGE (e:Entity:Commitment {id: $id})
            SET e.session_id    = $sid,
                e.text          = $text,
                e.speaker_label = $spk_label,
                e.timestamp_ms  = $timestamp_ms
            """,
            id=eid,
            sid=session_id,
            text=c.get("text") or "",
            spk_label=speaker_label,
            timestamp_ms=int(c.get("timestamp_ms") or 0),
        )
        if speaker_label:
            await nsession.run(
                """
                MATCH (e:Entity:Commitment {id: $eid})
                MATCH (spk:Speaker {session_id: $sid, label: $label})
                MERGE (e)-[:RAISED_BY]->(spk)
                """,
                eid=eid,
                sid=session_id,
                label=speaker_label,
            )


async def _create_alert_nodes(nsession, session_id: str, alerts: list[dict]):
    for a in alerts:
        evidence = a.get("evidence")
        if isinstance(evidence, dict):
            evidence_str = json.dumps(evidence)
        elif isinstance(evidence, str):
            evidence_str = evidence
        else:
            evidence_str = "{}"
        await nsession.run(
            """
            MERGE (al:Alert {id: $id})
            SET al.session_id   = $sid,
                al.alert_type   = $alert_type,
                al.severity     = $severity,
                al.title        = $title,
                al.description  = $description,
                al.timestamp_ms = $timestamp_ms,
                al.evidence     = $evidence
            WITH al
            MATCH (sess:Session {id: $sid})
            MERGE (al)-[:RAISED_FOR]->(sess)
            """,
            id=str(a["id"]),
            sid=session_id,
            alert_type=a.get("alert_type") or "",
            severity=a.get("severity") or "",
            title=a.get("title") or "",
            description=a.get("description") or "",
            timestamp_ms=int(a.get("timestamp_ms") or 0),
            evidence=evidence_str,
        )


# ─────────────────────────────────────────────────────────
# Edge builders (run after all nodes exist)
# ─────────────────────────────────────────────────────────

async def _build_temporal_edges(nsession, session_id: str):
    """NEXT, REPLIED_TO, SPOKEN_TO, FOLLOWED_BY. NO longer builds PRECEDED (was noise)."""
    # Segment NEXT chain — build by ordering segments by start time so we
    # don't rely on contiguous segment_index values (indices can be missing).
    await nsession.run(
        """
        MATCH (s:Segment {session_id: $sid})
        WITH s ORDER BY s.start_ms ASC
        WITH collect(s) AS segments
        UNWIND range(0, size(segments) - 2) AS i
        WITH segments[i] AS a, segments[i+1] AS b
        MERGE (a)-[:NEXT]->(b)
        """,
        sid=session_id,
    )
    # REPLIED_TO — cross-speaker response chain.
    # Reuses the NEXT chain (built by start_ms order) so this is consistent
    # with the ordering above and works even when segment_index is non-contiguous.
    await nsession.run(
        """
        MATCH (a:Segment {session_id: $sid})-[:NEXT]->(b:Segment {session_id: $sid})
        WHERE a.speaker_label <> b.speaker_label
          AND a.speaker_label <> ''
          AND b.speaker_label <> ''
        MERGE (b)-[r:REPLIED_TO]->(a)
        SET r.gap_ms = b.start_ms - a.end_ms
        """,
        sid=session_id,
    )
    # SPOKEN_TO — who this segment was addressed to.
    # Determined by who responded next (the speaker of the following segment).
    # Enables "what did Speaker_0 say to Speaker_1?" queries.
    await nsession.run(
        """
        MATCH (a:Segment {session_id: $sid})-[:NEXT]->(b:Segment {session_id: $sid})
        WHERE a.speaker_label <> b.speaker_label
          AND a.speaker_label <> ''
          AND b.speaker_label <> ''
        MATCH (spk:Speaker {session_id: $sid, label: b.speaker_label})
        MERGE (a)-[:SPOKEN_TO]->(spk)
        """,
        sid=session_id,
    )
    # Topic FOLLOWED_BY chain
    await nsession.run(
        """
        MATCH (a:Topic {session_id: $sid}), (b:Topic {session_id: $sid})
        WHERE b.order_idx = a.order_idx + 1
        MERGE (a)-[:FOLLOWED_BY]->(b)
        """,
        sid=session_id,
    )
    # REMOVED: Signal PRECEDED chain. Was creating hundreds of meaningless
    # tone→tone→tone edges per speaker. Causal relationships between signals
    # are handled by TRIGGERED, CONTRADICTS, and REINFORCES in _build_causal_edges.


async def _build_containment_edges(nsession, session_id: str):
    """OCCURRED_DURING (signal→segment), DISCUSSES (segment→topic), MENTIONED_IN (entity→segment), RESOLVED_IN (objection→segment)."""
    # Signal OCCURRED_DURING Segment — Query 1: speaker-specific signals.
    # Match signal timestamp (midpoint) to the segment it falls inside,
    # enforcing speaker match so Speaker_0's stress never links to Speaker_1's segment.
    # Result: each speaker-labelled signal → exactly 1 segment (no fan-out).
    await nsession.run(
        """
        MATCH (sig:Signal {session_id: $sid})
        WHERE sig.speaker_label IS NOT NULL
          AND sig.speaker_label <> ''
          AND sig.speaker_label <> 'all'
        MATCH (seg:Segment {session_id: $sid})
        WHERE sig.timestamp_ms >= seg.start_ms
          AND sig.timestamp_ms < seg.end_ms
          AND sig.speaker_label = seg.speaker_label
        MERGE (sig)-[:OCCURRED_DURING]->(seg)
        """,
        sid=session_id,
    )
    # Signal OCCURRED_DURING Segment — Query 2: unlinked signals (session-level,
    # fusion, or speaker='all') → nearest segment by midpoint distance.
    # Covers rapport, balance, dominance, and any signal that didn't match above.
    await nsession.run(
        """
        MATCH (sig:Signal {session_id: $sid})
        WHERE NOT (sig)-[:OCCURRED_DURING]->()
        MATCH (seg:Segment {session_id: $sid})
        WITH sig, seg,
             abs(sig.timestamp_ms - (seg.start_ms + seg.end_ms) / 2) AS dist
        ORDER BY dist ASC
        WITH sig, collect(seg)[0] AS nearest
        WHERE nearest IS NOT NULL
        MERGE (sig)-[:OCCURRED_DURING]->(nearest)
        """,
        sid=session_id,
    )
    # Segment DISCUSSES Topic (segment start within topic range)
    await nsession.run(
        """
        MATCH (seg:Segment {session_id: $sid}),
              (t:Topic {session_id: $sid})
        WHERE seg.start_ms >= t.start_ms
          AND seg.start_ms <  t.end_ms
        MERGE (seg)-[:DISCUSSES]->(t)
        """,
        sid=session_id,
    )
    # Entity MENTIONED_IN Segment (Person/Company by first_mention_ms)
    await nsession.run(
        """
        MATCH (e:Entity {session_id: $sid}),
              (seg:Segment {session_id: $sid})
        WHERE e.first_mention_ms IS NOT NULL
          AND e.first_mention_ms >= seg.start_ms
          AND e.first_mention_ms <  seg.end_ms
        MERGE (e)-[:MENTIONED_IN]->(seg)
        """,
        sid=session_id,
    )
    # Objection MENTIONED_IN Segment (by timestamp_ms)
    await nsession.run(
        """
        MATCH (e:Entity:Objection {session_id: $sid}),
              (seg:Segment {session_id: $sid})
        WHERE e.timestamp_ms >= seg.start_ms
          AND e.timestamp_ms <  seg.end_ms
        MERGE (e)-[:MENTIONED_IN]->(seg)
        """,
        sid=session_id,
    )
    # Objection RESOLVED_IN Segment (when resolved=true and resolved_at_ms set)
    await nsession.run(
        """
        MATCH (e:Entity:Objection {session_id: $sid}),
              (seg:Segment {session_id: $sid})
        WHERE e.resolved = true
          AND e.resolved_at_ms > 0
          AND e.resolved_at_ms >= seg.start_ms
          AND e.resolved_at_ms <  seg.end_ms
        MERGE (e)-[:RESOLVED_IN]->(seg)
        """,
        sid=session_id,
    )


async def _build_causal_edges(nsession, session_id: str):
    """
    Build CONTRADICTS, REINFORCES, TRIGGERED edges from signal co-occurrence.

    Mirrors the heuristics in services/fusion_agent/signal_graph.py so the
    Neo4j projection has the same causal structure as the in-memory graph.
    """
    # CONTRADICTS — condition 1: positive sentiment + high stress (verbal/vocal incongruence)
    await nsession.run(
        """
        MATCH (sent:Signal {session_id: $sid, signal_type: 'sentiment_score'}),
              (stress:Signal {session_id: $sid, signal_type: 'vocal_stress_score'})
        WHERE sent.value > 0.5
          AND stress.value > 0.5
          AND sent.speaker_label = stress.speaker_label
          AND abs(sent.timestamp_ms - stress.timestamp_ms) < 3000
        MERGE (sent)-[:CONTRADICTS]->(stress)
        """,
        sid=session_id,
    )
    # CONTRADICTS — condition 2: confident/warm tone + filler spike (assurance + uncertainty)
    await nsession.run(
        """
        MATCH (tone:Signal {session_id: $sid, signal_type: 'tone_classification'}),
              (filler:Signal {session_id: $sid, signal_type: 'filler_detection'})
        WHERE tone.value_text IN ['confident', 'warm']
          AND filler.value_text = 'filler_spike'
          AND tone.speaker_label = filler.speaker_label
          AND abs(tone.timestamp_ms - filler.timestamp_ms) < 3000
        MERGE (tone)-[:CONTRADICTS]->(filler)
        """,
        sid=session_id,
    )
    # CONTRADICTS — condition 3: high power language + high stress (authority claim under pressure)
    await nsession.run(
        """
        MATCH (power:Signal {session_id: $sid, signal_type: 'power_language_score'}),
              (stress:Signal {session_id: $sid, signal_type: 'vocal_stress_score'})
        WHERE power.value > 0.6
          AND stress.value > 0.6
          AND power.speaker_label = stress.speaker_label
          AND abs(power.timestamp_ms - stress.timestamp_ms) < 3000
        MERGE (power)-[:CONTRADICTS]->(stress)
        """,
        sid=session_id,
    )

    # REINFORCES — condition 1: confident tone + powerful language
    await nsession.run(
        """
        MATCH (tone:Signal {session_id: $sid, signal_type: 'tone_classification'}),
              (power:Signal {session_id: $sid, signal_type: 'power_language_score'})
        WHERE tone.value_text = 'confident'
          AND power.value > 0.6
          AND tone.speaker_label = power.speaker_label
          AND abs(tone.timestamp_ms - power.timestamp_ms) < 3000
        MERGE (tone)-[:REINFORCES]->(power)
        """,
        sid=session_id,
    )
    # REINFORCES — condition 2: warm tone + positive sentiment (aligned emotional signal)
    await nsession.run(
        """
        MATCH (tone:Signal {session_id: $sid, signal_type: 'tone_classification'}),
              (sent:Signal {session_id: $sid, signal_type: 'sentiment_score'})
        WHERE tone.value_text = 'warm'
          AND sent.value > 0.5
          AND tone.speaker_label = sent.speaker_label
          AND abs(tone.timestamp_ms - sent.timestamp_ms) < 3000
        MERGE (tone)-[:REINFORCES]->(sent)
        """,
        sid=session_id,
    )
    # REINFORCES — condition 3: high stress + nervous tone (amplified anxiety signal)
    await nsession.run(
        """
        MATCH (stress:Signal {session_id: $sid, signal_type: 'vocal_stress_score'}),
              (tone:Signal {session_id: $sid, signal_type: 'tone_classification'})
        WHERE stress.value > 0.5
          AND tone.value_text IN ['nervous', 'uncertain', 'hesitant']
          AND stress.speaker_label = tone.speaker_label
          AND abs(stress.timestamp_ms - tone.timestamp_ms) < 3000
        MERGE (stress)-[:REINFORCES]->(tone)
        """,
        sid=session_id,
    )

    # TRIGGERED — condition 1: stress spike preceding tension cluster
    await nsession.run(
        """
        MATCH (stress:Signal {session_id: $sid, signal_type: 'vocal_stress_score'}),
              (cluster:Signal {session_id: $sid, signal_type: 'tension_cluster'})
        WHERE stress.value > 0.5
          AND stress.timestamp_ms < cluster.timestamp_ms
          AND cluster.timestamp_ms - stress.timestamp_ms < 5000
          AND stress.speaker_label = cluster.speaker_label
        MERGE (stress)-[:TRIGGERED]->(cluster)
        """,
        sid=session_id,
    )
    # TRIGGERED — condition 2: objection → vocal stress spike (objection causes stress response)
    await nsession.run(
        """
        MATCH (obj:Signal {session_id: $sid, signal_type: 'objection_signal'}),
              (stress:Signal {session_id: $sid})
        WHERE stress.signal_type IN ['vocal_stress_score', 'pitch_elevation_flag']
          AND stress.value > 0.5
          AND obj.timestamp_ms < stress.timestamp_ms
          AND stress.timestamp_ms - obj.timestamp_ms < 5000
        MERGE (obj)-[:TRIGGERED]->(stress)
        """,
        sid=session_id,
    )
    # TRIGGERED — condition 3: momentum shift → conversation engagement change (topic shift → reaction)
    await nsession.run(
        """
        MATCH (shift:Signal {session_id: $sid, signal_type: 'momentum_shift'}),
              (eng:Signal {session_id: $sid, signal_type: 'conversation_engagement'})
        WHERE shift.timestamp_ms < eng.timestamp_ms
          AND eng.timestamp_ms - shift.timestamp_ms < 10000
        MERGE (shift)-[:TRIGGERED]->(eng)
        """,
        sid=session_id,
    )


async def _link_fusion_to_signals(nsession, session_id: str):
    """
    COMBINES: link FusionInsight nodes to their contributing Signal nodes (typed).

    Each fusion type links to the TOP 3 signals per type (highest confidence,
    closest to window midpoint). This caps fan-out and prevents cartesian product
    explosion (was producing 30-50+ COMBINES per insight without the limit).
    """
    # stress_sentiment_incongruence / credibility_assessment — driven by stress + sentiment
    await nsession.run(
        """
        MATCH (fi:FusionInsight {session_id: $sid})
        WHERE fi.signal_type IN ['stress_sentiment_incongruence', 'credibility_assessment']
        MATCH (sig:Signal {session_id: $sid})
        WHERE NOT sig:FusionInsight
          AND sig.signal_type IN ['vocal_stress_score', 'sentiment_score',
                                   'filler_detection', 'pitch_elevation_flag']
          AND sig.timestamp_ms >= fi.window_start_ms
          AND sig.timestamp_ms <= fi.window_end_ms
          AND (fi.speaker_label = sig.speaker_label
               OR fi.speaker_label IN ['all', ''])
        WITH fi, sig
        ORDER BY sig.confidence DESC,
                 abs(sig.timestamp_ms - (fi.window_start_ms + fi.window_end_ms) / 2) ASC
        WITH fi, collect(sig)[0..3] AS top_sigs
        UNWIND top_sigs AS best
        MERGE (fi)-[:COMBINES]->(best)
        """,
        sid=session_id,
    )
    # verbal_incongruence — driven by sentiment + power language + objection
    await nsession.run(
        """
        MATCH (fi:FusionInsight {session_id: $sid})
        WHERE fi.signal_type = 'verbal_incongruence'
        MATCH (sig:Signal {session_id: $sid})
        WHERE NOT sig:FusionInsight
          AND sig.signal_type IN ['sentiment_score', 'power_language_score',
                                   'objection_signal', 'vocal_stress_score']
          AND sig.timestamp_ms >= fi.window_start_ms
          AND sig.timestamp_ms <= fi.window_end_ms
          AND (fi.speaker_label = sig.speaker_label
               OR fi.speaker_label IN ['all', ''])
        WITH fi, sig
        ORDER BY sig.confidence DESC,
                 abs(sig.timestamp_ms - (fi.window_start_ms + fi.window_end_ms) / 2) ASC
        WITH fi, collect(sig)[0..3] AS top_sigs
        UNWIND top_sigs AS best
        MERGE (fi)-[:COMBINES]->(best)
        """,
        sid=session_id,
    )
    # urgency_authenticity — driven by rate + persuasion + buying signals
    await nsession.run(
        """
        MATCH (fi:FusionInsight {session_id: $sid})
        WHERE fi.signal_type = 'urgency_authenticity'
        MATCH (sig:Signal {session_id: $sid})
        WHERE NOT sig:FusionInsight
          AND sig.signal_type IN ['speech_rate_anomaly', 'persuasion_technique',
                                   'buying_signal', 'tone_classification', 'sentiment_score']
          AND sig.timestamp_ms >= fi.window_start_ms
          AND sig.timestamp_ms <= fi.window_end_ms
          AND (fi.speaker_label = sig.speaker_label
               OR fi.speaker_label IN ['all', ''])
        WITH fi, sig
        ORDER BY sig.confidence DESC,
                 abs(sig.timestamp_ms - (fi.window_start_ms + fi.window_end_ms) / 2) ASC
        WITH fi, collect(sig)[0..3] AS top_sigs
        UNWIND top_sigs AS best
        MERGE (fi)-[:COMBINES]->(best)
        """,
        sid=session_id,
    )
    # tension_cluster / momentum_shift / persistent_incongruence — top 5 by confidence
    await nsession.run(
        """
        MATCH (fi:FusionInsight {session_id: $sid})
        WHERE fi.signal_type IN ['tension_cluster', 'momentum_shift',
                                  'persistent_incongruence']
        MATCH (sig:Signal {session_id: $sid})
        WHERE NOT sig:FusionInsight
          AND sig.confidence > 0.4
          AND sig.timestamp_ms >= fi.window_start_ms
          AND sig.timestamp_ms <= fi.window_end_ms
          AND (fi.speaker_label = sig.speaker_label
               OR fi.speaker_label IN ['all', ''])
        WITH fi, sig
        ORDER BY sig.confidence DESC
        WITH fi, collect(sig)[0..5] AS top_sigs
        UNWIND top_sigs AS best
        MERGE (fi)-[:COMBINES]->(best)
        """,
        sid=session_id,
    )


async def _build_influence_edges(nsession, session_id: str):
    """INFLUENCED: cross-speaker causality, constrained to adjacent segments (1-3 NEXT hops).

    Reduced window: 30s → 15s (eliminates long-range spurious connections).
    Adjacency guard: signals must be on segments within 3 NEXT hops of each other.
    Type guard: a.signal_type <> b.signal_type prevents tone→tone fan-out across speakers.
    """
    await nsession.run(
        """
        MATCH (a:Signal {session_id: $sid})-[:OCCURRED_DURING]->(segA:Segment),
              (b:Signal {session_id: $sid})-[:OCCURRED_DURING]->(segB:Segment)
        WHERE a.speaker_label <> b.speaker_label
          AND a.speaker_label <> ''
          AND b.speaker_label <> ''
          AND b.timestamp_ms > a.timestamp_ms
          AND b.timestamp_ms - a.timestamp_ms < 15000
          AND a.confidence > 0.4
          AND b.confidence > 0.4
          AND a.signal_type IN [
              'tone_classification', 'vocal_stress_score', 'objection_signal',
              'buying_signal', 'persuasion_technique', 'interruption_event'
          ]
          AND b.signal_type IN [
              'vocal_stress_score', 'tone_classification', 'sentiment_score',
              'buying_signal', 'objection_signal', 'filler_detection'
          ]
          AND a.signal_type <> b.signal_type
          AND EXISTS {
              MATCH (segA)-[:NEXT*1..3]->(segB)
          }
        MERGE (a)-[r:INFLUENCED]->(b)
        SET r.lag_ms = b.timestamp_ms - a.timestamp_ms
        """,
        sid=session_id,
    )


# ─────────────────────────────────────────────────────────
# Hybrid chat helper — query Neo4j for graph context
# ─────────────────────────────────────────────────────────

GRAPH_SCHEMA_HINT = """
NODE LABELS and their PROPERTIES:

(:Session)
  id, title, session_type, meeting_type, status, duration_ms, speaker_count

(:Speaker)
  id, label (e.g. "Speaker_0"), name, role, talk_time_ms, talk_time_pct, word_count

(:Segment)
  id, segment_index, start_ms, end_ms, text, speaker_label,
  sentiment, sentiment_score, word_count

(:Topic)
  id, name, start_ms, end_ms, order_idx

(:Signal)  -- and (:FusionInsight) which is the agent='fusion' subset
  id, agent, signal_type, value, value_text, confidence,
  window_start_ms, window_end_ms, timestamp_ms, speaker_label, metadata

(:Entity:Person)        id, name, role, speaker_label, first_mention_ms
(:Entity:Company)       id, name, context, first_mention_ms
(:Entity:Product)       id, name, context
(:Entity:Objection)     id, text, timestamp_ms, resolved, resolved_at_ms
(:Entity:Commitment)    id, text, speaker_label, timestamp_ms

(:Alert)
  id, alert_type, severity, title, description, timestamp_ms, evidence

Every node has a session_id property — ALWAYS filter with {session_id: $session_id}.

RELATIONSHIP TYPES (these are the ONLY valid relationships):

  Structural:    PARTICIPATED_IN, PART_OF, OCCURRED_IN, SPOKEN_BY,
                 EMITTED_BY, RAISED_FOR
  Temporal:      NEXT (segment→segment), FOLLOWED_BY (topic→topic),
                 REPLIED_TO (segment→segment, cross-speaker response, has gap_ms property),
                 SPOKEN_TO (segment→speaker, the speaker this turn was addressed to)
  Semantic:      OCCURRED_DURING (signal→segment), DISCUSSES (segment→topic),
                 MENTIONED_IN (entity→segment), IS_SPEAKER (person→speaker),
                 RAISED_BY (commitment→speaker), RESOLVED_IN (objection→segment),
                 COMBINES (fusionInsight→signal, component signals that produced the fusion output)
  Causal:        CONTRADICTS, REINFORCES, TRIGGERED  (all signal→signal)
  Cross-speaker: INFLUENCED  (signal→signal, has lag_ms property)

CRITICAL DISTINCTIONS:
  - Signal TYPES (like 'vocal_stress_score', 'tension_cluster', 'buying_signal',
    'objection_signal', 'rapport_indicator') are VALUES of the `signal_type`
    PROPERTY on (:Signal) nodes — they are NOT relationship types.
    To find tension clusters: MATCH (s:Signal {signal_type: 'tension_cluster'})
    NOT: MATCH ()-[r:TENSION_CLUSTER]->()
  - Time properties are *_ms (numeric milliseconds), NOT start_time/end_time.
    Use `timestamp_ms`, `start_ms`, `end_ms`, `window_start_ms`.
    To find a moment near 0:20 (= 20s = 20000ms):
      WHERE s.timestamp_ms > 18000 AND s.timestamp_ms < 22000
  - Confidence is `confidence` (0.0–1.0).
  - Sentiment score is `sentiment_score` on Segment OR `value` on
    Signal {signal_type: 'sentiment_score'}.

Common signal_type values: vocal_stress_score, sentiment_score, tone_classification,
filler_detection, pitch_elevation_flag, speech_rate_anomaly, buying_signal,
objection_signal, power_language_score, persuasion_technique, gottman_horsemen,
empathy_language, clarity_score, rapport_indicator, dominance_score,
conversation_engagement, conversation_balance, tension_cluster, momentum_shift,
verbal_incongruence, credibility_assessment, urgency_authenticity.

═══════════════════════════════════════════════════════════════
QUERY PATTERNS — copy these shapes; adjust signal_type / speaker_label / time
═══════════════════════════════════════════════════════════════

PATTERN A — "what was happening when X fired?" / "why was there X?"
  ALWAYS traverse to Segment.text. Return both the signal AND the words spoken.

  MATCH (sig:Signal {session_id: $session_id, signal_type: 'tension_cluster'})
  OPTIONAL MATCH (sig)-[:OCCURRED_DURING]->(seg:Segment)
  RETURN sig.timestamp_ms, sig.value, sig.confidence, sig.value_text,
         sig.speaker_label, seg.text, seg.speaker_label AS spoken_by
  ORDER BY sig.value DESC LIMIT 5

PATTERN B — "quote the line where X happened" / "what did Y say at time T?"
  Find the Segment directly by time, return text. NEVER reply with "transcript not available".

  MATCH (seg:Segment {session_id: $session_id})
  WHERE seg.start_ms <= $t_ms AND seg.end_ms >= $t_ms
  // Or filter by speaker_label if asked about a specific speaker:
  //   AND seg.speaker_label = 'Speaker_0'
  RETURN seg.start_ms, seg.end_ms, seg.speaker_label, seg.text
  ORDER BY seg.start_ms LIMIT 10

PATTERN C — "find moments where SIGNAL_A AND SIGNAL_B co-occur"
  Cross-signal temporal join. Use abs(time_diff) < window.

  MATCH (a:Signal {session_id: $session_id, signal_type: 'pitch_elevation_flag'})
  MATCH (b:Signal {session_id: $session_id, signal_type: 'filler_detection'})
  WHERE a.value_text = 'pitch_elevated_extreme'
    AND b.value_text = 'filler_spike'
    AND abs(a.timestamp_ms - b.timestamp_ms) < 5000
  OPTIONAL MATCH (a)-[:OCCURRED_DURING]->(seg:Segment)
  RETURN a.timestamp_ms, a.speaker_label, a.value, b.speaker_label, b.value,
         seg.text
  ORDER BY a.timestamp_ms LIMIT 10

PATTERN D — "what was SPEAKER_X's pitch / stress / tone like during EVENT?"
  Filter by both speaker_label AND signal_type. Don't default to vocal_stress_score.

  MATCH (s:Signal {session_id: $session_id})
  WHERE s.speaker_label = 'Speaker_0'
    AND s.signal_type = 'pitch_elevation_flag'
    AND s.timestamp_ms > $start_ms AND s.timestamp_ms < $end_ms
  OPTIONAL MATCH (s)-[:OCCURRED_DURING]->(seg:Segment)
  RETURN s.timestamp_ms, s.value, s.value_text, s.confidence, seg.text
  ORDER BY s.value DESC LIMIT 5

PATTERN E — "were any objections left unresolved?"
  Filter `resolved = false` directly on the Objection node.

  MATCH (o:Entity:Objection {session_id: $session_id})
  WHERE o.resolved = false
  RETURN o.text, o.timestamp_ms, o.resolved
  ORDER BY o.timestamp_ms LIMIT 10

  // To find resolved objections AND the segment that resolved them:
  MATCH (o:Entity:Objection {session_id: $session_id})
  WHERE o.resolved = true
  OPTIONAL MATCH (o)-[:RESOLVED_IN]->(seg:Segment)
  RETURN o.text, o.timestamp_ms, o.resolved_at_ms, seg.text AS resolution_text

PATTERN F — "who said X?" / "find segments containing keyword K"
  Use CONTAINS for substring search on Segment.text.

  MATCH (seg:Segment {session_id: $session_id})
  WHERE toLower(seg.text) CONTAINS toLower($keyword)
  RETURN seg.start_ms, seg.end_ms, seg.speaker_label, seg.text
  ORDER BY seg.start_ms LIMIT 10

PATTERN H — "how did Speaker B respond to Speaker A?" / "what was the reaction?"
  Use REPLIED_TO to trace the cross-speaker response chain.

  MATCH (response:Segment {session_id: $session_id})-[r:REPLIED_TO]->(trigger:Segment)
  WHERE trigger.speaker_label = 'Speaker_0'   // or whichever speaker triggered the response
  OPTIONAL MATCH (sig:Signal {session_id: $session_id})-[:OCCURRED_DURING]->(trigger)
  RETURN trigger.start_ms, trigger.speaker_label, trigger.text AS trigger_text,
         response.start_ms, response.speaker_label, response.text AS response_text,
         r.gap_ms AS silence_between_turns,
         collect(DISTINCT sig.signal_type) AS signals_on_trigger
  ORDER BY trigger.start_ms LIMIT 10

PATTERN G — "biggest stress / loudest moment / most nervous moment"
  Sort by signal value DESC, traverse to segment text.

  MATCH (s:Signal {session_id: $session_id, signal_type: 'vocal_stress_score'})
  WHERE s.value_text IN ['moderate_stress', 'elevated_stress', 'high_stress']
  OPTIONAL MATCH (s)-[:OCCURRED_DURING]->(seg:Segment)
  RETURN s.timestamp_ms, s.speaker_label, s.value, s.value_text,
         s.confidence, seg.text
  ORDER BY s.value DESC LIMIT 5

═══════════════════════════════════════════════════════════════
BEHAVIOURAL RULES — these matter more than the patterns
═══════════════════════════════════════════════════════════════

R1. **WHEN the question asks "what / why / what was being said / what was
    happening / quote / when / who said"** → the query MUST include
    `OPTIONAL MATCH (sig)-[:OCCURRED_DURING]->(seg:Segment)` and RETURN
    `seg.text`. NEVER answer "the transcript is not available" — it always is.

R2. **WHEN the question names a SPECIFIC SPEAKER** ("Holly's pitch",
    "Saad's stress") → filter on `speaker_label` (`'Speaker_0'` or
    `'Speaker_1'`) — do not omit the filter.

R3. **WHEN the question names a SPECIFIC SIGNAL TYPE** ("pitch", "fillers",
    "tone", "stress") → filter on the EXACT `signal_type` matching that
    word, not a default like `vocal_stress_score`. Mapping:
      "pitch"          → signal_type = 'pitch_elevation_flag'
      "fillers" / "ums" → signal_type = 'filler_detection'
      "tone"           → signal_type = 'tone_classification'
      "stress"         → signal_type = 'vocal_stress_score'
      "speed" / "pace" → signal_type = 'speech_rate_anomaly'
      "monotone"       → signal_type = 'monotone_flag'
      "pauses"         → signal_type = 'pause_classification'
      "interrupting"   → signal_type = 'interruption_event'

R4. **WHEN comparing TWO speakers or two signal types** → use Pattern C
    (cross-signal join with temporal window).

R5. **NEVER invent value ranges** — if the question says "extreme",
    filter `value_text = 'pitch_elevated_extreme'` (an actual stored value),
    not `value > 100`.

R6. **For unresolved objections**, filter `resolved = false` (boolean) on
    `Entity:Objection` nodes — the property exists on every objection node.
"""


async def search_graph_context(question: str, session_id: str, max_rows: int = 15) -> str:
    """
    Use the LLM to generate a Cypher query for `question`, execute it against
    Neo4j scoped to `session_id`, and return a text summary of the results
    suitable for stuffing into a chat context. Returns "" on any failure.
    """
    driver = _get_driver()
    if driver is None:
        return ""

    try:
        from shared.utils.llm_client import acomplete
    except ImportError:
        return ""

    system_prompt = (
        "You are a Cypher query generator for a Neo4j knowledge graph that "
        "stores conversation analysis data. Given a user question, return ONE "
        "Cypher query that answers it. Return ONLY the Cypher query — no "
        "explanation, no markdown fences, no commentary.\n\n"
        + GRAPH_SCHEMA_HINT
        + "\nFINAL RULES:\n"
        + "1. ALWAYS filter on session_id with the parameter $session_id.\n"
        + "2. NEVER use MERGE, CREATE, SET, DELETE — read-only queries only.\n"
        + "3. LIMIT results to at most 20 rows.\n"
        + "4. Return concrete columns, not whole nodes.\n"
        + "5. Pick the matching PATTERN A–G above and adapt it. Do not invent\n"
        + "   relationship types or properties not listed in the schema.\n"
        + "6. Follow the BEHAVIOURAL RULES R1–R6 — they override the patterns\n"
        + "   when in conflict."
    )

    try:
        cypher = await acomplete(
            system_prompt=system_prompt,
            user_prompt=f"Question: {question}\nSession ID: {session_id}",
            max_tokens=800,
            temperature=0.0,
        )
    except Exception as e:
        logger.warning(f"Cypher generation failed: {e}")
        return ""

    cypher = cypher.strip()
    if cypher.startswith("```"):
        cypher = cypher.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Safety: ensure the generated query is scoped by parameterized $session_id
    if "$session_id" not in cypher:
        logger.warning(f"Refusing to run Cypher missing $session_id: {cypher[:200]}")
        return ""

    # Safety: refuse anything that mutates the graph
    forbidden = ("CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "DETACH")
    upper = cypher.upper()
    if any(tok in upper for tok in forbidden):
        logger.warning(f"Refusing mutating Cypher: {cypher[:200]}")
        return ""

    try:
        async with driver.session() as nsession:
            result = await nsession.run(cypher, session_id=session_id)
            rows = []
            async for r in result:
                rows.append(dict(r))
                if len(rows) >= max_rows:
                    break
    except Exception as e:
        logger.warning(f"Cypher execution failed: {e}; cypher={cypher[:200]}")
        return ""

    if not rows:
        return ""

    # Format as compact text rows for the LLM context
    lines = ["GRAPH RESULTS:"]
    for row in rows:
        parts = []
        for k, v in row.items():
            v_str = str(v)
            if len(v_str) > 200:
                v_str = v_str[:200] + "..."
            parts.append(f"{k}={v_str}")
        lines.append("  - " + ", ".join(parts))
    return "\n".join(lines)
