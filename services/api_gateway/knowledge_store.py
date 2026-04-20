# services/api_gateway/knowledge_store.py
"""
NEXUS API Gateway — Knowledge Store
Stores session analysis data as vector embeddings in pgvector for RAG chat.
"""
import json
import hashlib
import logging
from typing import Optional

logger = logging.getLogger("nexus.gateway.knowledge")


async def store_session_knowledge(pool, session_id: str, session_data: dict):
    """
    After pipeline completes, chunk session data and store as vector embeddings.

    Chunks are created from:
    1. Transcript segments (grouped by 3 for context)
    2. Signal summaries (grouped by type + speaker)
    3. Entity extractions (topics, people, objections, commitments)
    4. Report sections (summary, key moments, recommendations)
    5. Graph analytics (tension clusters, momentum)
    6. Conversation dynamics summary
    """
    from shared.utils.llm_client import get_embedding

    chunks = []

    # ── 1. Transcript chunks (every 3 segments) ──
    segments = session_data.get("transcript_segments", [])
    for i in range(0, len(segments), 3):
        group = segments[i:i + 3]
        text = " ".join(
            f"{s.get('speaker_label', s.get('speaker', '?'))}: {s.get('text', '')}"
            for s in group
        )
        if not text.strip():
            continue
        start_ms = group[0].get("start_ms", 0)
        end_ms = group[-1].get("end_ms", 0)
        chunks.append({
            "type": "transcript",
            "text": text,
            "metadata": {"start_ms": start_ms, "end_ms": end_ms},
        })

    # ── 2. Signal summaries (group by type + speaker) ──
    signals = session_data.get("signals", [])
    signal_groups: dict[str, list] = {}
    for s in signals:
        key = f"{s.get('signal_type', '')}_{s.get('speaker_id', s.get('speaker_label', ''))}"
        signal_groups.setdefault(key, []).append(s)

    for key, group in signal_groups.items():
        sig_type = group[0].get("signal_type", "")
        speaker = group[0].get("speaker_label", group[0].get("speaker_id", "?"))
        values = [s.get("value", 0) or 0 for s in group]
        confs = [s.get("confidence", 0) or 0 for s in group]
        avg_value = sum(values) / len(values) if values else 0
        avg_conf = sum(confs) / len(confs) if confs else 0
        top_vals = sorted(values, reverse=True)[:5]

        text = (
            f"{speaker}'s {sig_type.replace('_', ' ')}: "
            f"{len(group)} signals, average value {avg_value:.2f}, "
            f"average confidence {avg_conf:.2f}. "
            f"Notable values: {', '.join(f'{v:.2f}' for v in top_vals)}"
        )
        chunks.append({
            "type": "signal_summary",
            "text": text,
            "metadata": {"signal_type": sig_type, "speaker": speaker, "count": len(group)},
        })

    # ── 3. Entity chunks ──
    entities = session_data.get("entities", {})
    if entities:
        for topic in entities.get("topics", []):
            chunks.append({
                "type": "topic",
                "text": f"Topic: {topic.get('name', '')} from {topic.get('start_ms', 0) // 1000}s to {topic.get('end_ms', 0) // 1000}s",
                "metadata": topic,
            })
        for obj in entities.get("objections", []):
            chunks.append({
                "type": "objection",
                "text": f"Objection: \"{obj.get('text', '')}\" at {obj.get('timestamp_ms', 0) // 1000}s. Resolved: {obj.get('resolved', False)}",
                "metadata": obj,
            })
        for commit in entities.get("commitments", []):
            chunks.append({
                "type": "commitment",
                "text": f"Commitment by {commit.get('speaker', '?')}: \"{commit.get('text', '')}\" at {commit.get('timestamp_ms', 0) // 1000}s",
                "metadata": commit,
            })
        for person in entities.get("people", []):
            chunks.append({
                "type": "person",
                "text": f"Person: {person.get('name', '')} (role: {person.get('role', 'unknown')}, speaker: {person.get('speaker_label', '?')})",
                "metadata": person,
            })

    # ── 4. Report sections ──
    report = session_data.get("report", {})
    if report:
        if report.get("executive_summary"):
            chunks.append({
                "type": "report_summary",
                "text": f"Executive Summary: {report['executive_summary']}",
                "metadata": {"section": "executive_summary"},
            })
        for moment in report.get("key_moments", []):
            chunks.append({
                "type": "report_moment",
                "text": (
                    f"Key Moment at {moment.get('time_description', '?')}: "
                    f"{moment.get('description', '')}. "
                    f"Significance: {moment.get('significance', '')}"
                ),
                "metadata": moment,
            })
        for insight in report.get("cross_modal_insights", []):
            if isinstance(insight, str):
                chunks.append({
                    "type": "report_insight",
                    "text": f"Cross-Modal Insight: {insight}",
                    "metadata": {"section": "cross_modal_insights"},
                })
        for rec in report.get("recommendations", []):
            if isinstance(rec, str):
                chunks.append({
                    "type": "report_recommendation",
                    "text": f"Recommendation: {rec}",
                    "metadata": {"section": "recommendations"},
                })

    # ── 5. Graph analytics ──
    analytics = session_data.get("graph_analytics", {})
    if analytics:
        for cluster in analytics.get("tension_clusters", []):
            chunks.append({
                "type": "analytics",
                "text": (
                    f"Tension cluster at {cluster.get('timestamp_ms', 0) // 1000}s: "
                    f"{cluster.get('signal_count', 0)} signals, "
                    f"severity {cluster.get('severity', '?')}, "
                    f"speaker {cluster.get('speaker_id', '?')}"
                ),
                "metadata": cluster,
            })
        momentum = analytics.get("momentum", {})
        if momentum.get("overall_trajectory"):
            chunks.append({
                "type": "analytics",
                "text": (
                    f"Conversation momentum: trajectory {momentum.get('overall_trajectory', '?')}, "
                    f"score {momentum.get('momentum_score', 0):.2f}, "
                    f"turning point at {(momentum.get('turning_point_ms') or 0) // 1000}s"
                ),
                "metadata": momentum,
            })

    # ── 6. Conversation dynamics ──
    convo = session_data.get("conversation_summary", {})
    if convo:
        session_info = convo.get("session", {})
        if session_info:
            chunks.append({
                "type": "conversation",
                "text": (
                    f"Conversation dynamics: "
                    f"{session_info.get('turn_rate_per_minute', 0):.1f} turns/min, "
                    f"dominance index {session_info.get('dominance_index', 0):.2f}, "
                    f"balance: {session_info.get('conversation_balance', 'unknown')}"
                ),
                "metadata": session_info,
            })

    # ── Generate embeddings and store ──
    stored = 0
    for chunk in chunks:
        try:
            embedding = await get_embedding(chunk["text"])
            if not embedding:
                continue
            chunk_id = hashlib.md5(
                f"{session_id}_{chunk['type']}_{chunk['text'][:80]}".encode()
            ).hexdigest()
            # pgvector expects embedding as a string like "[0.1, 0.2, ...]"
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            await pool.execute(
                """
                INSERT INTO knowledge_chunks (id, session_id, chunk_type, text, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector)
                ON CONFLICT (id) DO UPDATE SET text = $4, metadata = $5::jsonb, embedding = $6::vector
                """,
                chunk_id,
                session_id,
                chunk["type"],
                chunk["text"],
                json.dumps(chunk.get("metadata", {})),
                embedding_str,
            )
            stored += 1
        except Exception as e:
            logger.warning(f"Failed to store chunk [{chunk['type']}]: {e}")

    logger.info(f"[{session_id}] Stored {stored}/{len(chunks)} knowledge chunks")
    return stored
