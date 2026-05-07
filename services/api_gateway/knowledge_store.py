# services/api_gateway/knowledge_store.py
"""
NEXUS API Gateway — Knowledge Store
Stores session analysis data as vector embeddings in pgvector for RAG chat.
"""
import json
import hashlib
import logging

logger = logging.getLogger("nexus.gateway.knowledge")

_WINDOW_MS = 30_000  # 30-second windows for temporal signal chunks


def _fmt_ms(ms: int) -> str:
    """Convert milliseconds to mm:ss string."""
    ms = int(ms or 0)
    return f"{ms // 60000:02d}:{(ms % 60000) // 1000:02d}"


async def store_session_knowledge(pool, session_id: str, session_data: dict):
    """
    After pipeline completes, chunk session data and store as vector embeddings.

    Chunks created:
    1.  transcript        — every 3 segments, with speaker labels
    2.  signal_summary    — per (signal_type × speaker), all agents combined
    3.  window_signals    — 30-second behavioural snapshots with timestamps
    4.  signal_event      — individual high-confidence notable events
    5.  speaker_profile   — per-speaker overall behavioural summary
    6.  topic             — named topics with time ranges
    7.  objection         — objections with resolution status
    8.  commitment        — commitments with speaker and timestamp
    9.  person            — named participants and their roles
    10. report_summary    — executive summary narrative
    11. report_moment     — key moments from the fusion report
    12. report_insight    — cross-modal insights
    13. report_recommendation — actionable recommendations
    14. analytics         — tension clusters and momentum trajectory
    15. conversation      — turn-taking and dominance dynamics
    """
    from shared.utils.llm_client import get_embedding

    chunks = []
    signals = session_data.get("signals", [])

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
        chunks.append({
            "type": "transcript",
            "text": text,
            "metadata": {
                "start_ms": group[0].get("start_ms", 0),
                "end_ms": group[-1].get("end_ms", 0),
            },
        })

    # ── 2. Signal summaries (grouped by signal_type × speaker) ──
    # Collapses all occurrences into one chunk — good for "how often" questions.
    signal_groups: dict[str, list] = {}
    for s in signals:
        key = f"{s.get('signal_type', '')}_{s.get('speaker_id', s.get('speaker_label', ''))}"
        signal_groups.setdefault(key, []).append(s)

    for group in signal_groups.values():
        sig_type = group[0].get("signal_type", "")
        speaker = group[0].get("speaker_label", group[0].get("speaker_id", "?"))
        values = [s.get("value", 0) or 0 for s in group]
        confs = [s.get("confidence", 0) or 0 for s in group]
        avg_value = sum(values) / len(values) if values else 0
        avg_conf = sum(confs) / len(confs) if confs else 0
        top_vals = sorted(values, reverse=True)[:5]

        outcome_counts: dict[str, int] = {}
        for s in group:
            vt = (s.get("value_text") or "").strip()
            if vt:
                outcome_counts[vt] = outcome_counts.get(vt, 0) + 1

        outcome_phrase = ""
        if outcome_counts:
            parts = [
                f"{label.replace('_', ' ')} ({cnt}×)"
                for label, cnt in sorted(outcome_counts.items(), key=lambda x: -x[1])
            ]
            outcome_phrase = "Outcomes: " + ", ".join(parts) + ". "

        text = (
            f"{speaker}'s {sig_type.replace('_', ' ')}: "
            f"{len(group)} occurrence(s). "
            f"{outcome_phrase}"
            f"Average value {avg_value:.2f}, average confidence {avg_conf:.2f}. "
            f"Peak values: {', '.join(f'{v:.2f}' for v in top_vals)}"
        )
        chunks.append({
            "type": "signal_summary",
            "text": text,
            "metadata": {
                "signal_type": sig_type,
                "speaker": speaker,
                "count": len(group),
                "outcomes": outcome_counts,
            },
        })

    # ── 3. Time-windowed behavioural snapshots (30s windows) ──
    # Enables temporal questions: "what was happening at 5 minutes?"
    windowed: dict[str, dict[int, list]] = {}
    for s in signals:
        spk = s.get("speaker_label", s.get("speaker_id", "Unknown"))
        start = int(s.get("window_start_ms", 0) or 0)
        w_idx = start // _WINDOW_MS
        windowed.setdefault(spk, {}).setdefault(w_idx, []).append(s)

    for spk, windows in windowed.items():
        for w_idx, sigs in windows.items():
            if len(sigs) < 2:
                continue
            t_start = w_idx * _WINDOW_MS
            t_end = t_start + _WINDOW_MS

            sig_parts = []
            for s in sigs:
                vt = (s.get("value_text") or "").replace("_", " ").strip()
                val = s.get("value")
                st = (s.get("signal_type") or "").replace("_", " ")
                if vt:
                    sig_parts.append(f"{st}={vt}")
                elif val is not None:
                    sig_parts.append(f"{st}={val:.2f}")

            if not sig_parts:
                continue

            text = (
                f"[{_fmt_ms(t_start)}–{_fmt_ms(t_end)}] {spk}: "
                f"{', '.join(sig_parts[:10])}"
            )
            chunks.append({
                "type": "window_signals",
                "text": text,
                "metadata": {
                    "speaker": spk,
                    "window_start_ms": t_start,
                    "window_end_ms": t_end,
                    "signal_count": len(sigs),
                },
            })

    # ── 4. Individual notable signal events (high-confidence, named outcome) ──
    # Enables specific queries: "when did stress peak?", "were there buying signals?"
    for s in signals:
        conf = s.get("confidence", 0) or 0
        vt = (s.get("value_text") or "").strip()
        val = s.get("value")
        sig_type = (s.get("signal_type") or "")
        spk = s.get("speaker_label", s.get("speaker_id", "Unknown"))
        start_ms = int(s.get("window_start_ms", 0) or 0)

        if conf < 0.65 or not vt:
            continue

        val_str = f" ({val:.2f})" if val is not None else ""
        text = (
            f"{spk} at {_fmt_ms(start_ms)}: "
            f"{sig_type.replace('_', ' ')} — "
            f"{vt.replace('_', ' ')}{val_str}, confidence {conf:.2f}"
        )
        chunks.append({
            "type": "signal_event",
            "text": text,
            "metadata": {
                "signal_type": sig_type,
                "speaker": spk,
                "timestamp_ms": start_ms,
                "value": val,
                "value_text": vt,
                "confidence": conf,
            },
        })

    # ── 5. Per-speaker behavioural profile ──
    # Enables "how was Speaker_0 overall?" questions.
    speaker_signals: dict[str, list] = {}
    for s in signals:
        spk = s.get("speaker_label", s.get("speaker_id", "Unknown"))
        speaker_signals.setdefault(spk, []).append(s)

    for spk, spk_sigs in speaker_signals.items():
        type_outcomes: dict[str, dict[str, int]] = {}
        for s in spk_sigs:
            st = (s.get("signal_type") or "")
            vt = (s.get("value_text") or "").strip()
            if st and vt:
                type_outcomes.setdefault(st, {})
                type_outcomes[st][vt] = type_outcomes[st].get(vt, 0) + 1

        if not type_outcomes:
            continue

        summary_parts = []
        for st, outcomes in sorted(type_outcomes.items(), key=lambda x: -sum(x[1].values()))[:12]:
            dominant = max(outcomes, key=outcomes.get)
            total = sum(outcomes.values())
            summary_parts.append(
                f"{st.replace('_', ' ')}: {dominant.replace('_', ' ')} ({total}×)"
            )

        text = f"{spk} behavioural profile: " + "; ".join(summary_parts)
        chunks.append({
            "type": "speaker_profile",
            "text": text,
            "metadata": {"speaker": spk, "signal_count": len(spk_sigs)},
        })

    # ── 6–9. Entity chunks (topics, objections, commitments, people) ──
    entities = session_data.get("entities", {})
    if entities:
        for topic in entities.get("topics", []):
            name = topic.get("name", "")
            start_s = topic.get("start_ms", 0) // 1000
            end_s = topic.get("end_ms", 0) // 1000
            chunks.append({
                "type": "topic",
                "text": (
                    f"Topic discussed: \"{name}\" "
                    f"from {start_s}s to {end_s}s "
                    f"({_fmt_ms(topic.get('start_ms', 0))}–{_fmt_ms(topic.get('end_ms', 0))})"
                ),
                "metadata": topic,
            })
        for obj in entities.get("objections", []):
            resolved = "resolved" if obj.get("resolved") else "unresolved"
            chunks.append({
                "type": "objection",
                "text": (
                    f"Objection ({resolved}) at {_fmt_ms(obj.get('timestamp_ms', 0))}: "
                    f"\"{obj.get('text', '')}\""
                ),
                "metadata": obj,
            })
        for commit in entities.get("commitments", []):
            chunks.append({
                "type": "commitment",
                "text": (
                    f"Commitment / decision by {commit.get('speaker', '?')} "
                    f"at {_fmt_ms(commit.get('timestamp_ms', 0))}: "
                    f"\"{commit.get('text', '')}\""
                ),
                "metadata": commit,
            })
        for person in entities.get("people", []):
            chunks.append({
                "type": "person",
                "text": (
                    f"Participant: {person.get('name', '')} "
                    f"(role: {person.get('role', 'unknown')}, "
                    f"speaker label: {person.get('speaker_label', '?')})"
                ),
                "metadata": person,
            })
        for company in entities.get("companies", []):
            chunks.append({
                "type": "company",
                "text": (
                    f"Company mentioned: {company.get('name', '')} — "
                    f"{company.get('context', '')}"
                ),
                "metadata": company,
            })
        key_terms = entities.get("key_terms", [])
        if key_terms:
            chunks.append({
                "type": "key_terms",
                "text": f"Key terms and topics mentioned: {', '.join(key_terms)}",
                "metadata": {"terms": key_terms},
            })

    # ── 10–13. Report sections ──
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
            if isinstance(insight, str) and insight.strip():
                chunks.append({
                    "type": "report_insight",
                    "text": f"Cross-Modal Insight: {insight}",
                    "metadata": {"section": "cross_modal_insights"},
                })
        for rec in report.get("recommendations", []):
            if isinstance(rec, str) and rec.strip():
                chunks.append({
                    "type": "report_recommendation",
                    "text": f"Recommendation: {rec}",
                    "metadata": {"section": "recommendations"},
                })

    # ── 14. Graph analytics ──
    analytics = session_data.get("graph_analytics", {})
    if analytics:
        for cluster in analytics.get("tension_clusters", []):
            chunks.append({
                "type": "analytics",
                "text": (
                    f"Tension cluster at {_fmt_ms(cluster.get('timestamp_ms', 0))}: "
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
                    f"turning point at {_fmt_ms(momentum.get('turning_point_ms') or 0)}"
                ),
                "metadata": momentum,
            })

    # ── 15. Conversation dynamics ──
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
        # Per-speaker conversation stats
        for spk_id, spk_data in convo.get("per_speaker", {}).items():
            talk_pct = spk_data.get("talk_time_pct", 0) or 0
            interruptions = spk_data.get("interruptions_made", 0) or 0
            response_ms = spk_data.get("avg_response_latency_ms", 0) or 0
            chunks.append({
                "type": "conversation",
                "text": (
                    f"{spk_id} conversation stats: "
                    f"{talk_pct:.0f}% talk time, "
                    f"{interruptions} interruptions made, "
                    f"avg response latency {response_ms:.0f}ms"
                ),
                "metadata": {"speaker": spk_id, **spk_data},
            })

    # ── Embed and store all chunks ──
    stored = 0
    skipped = 0
    for chunk in chunks:
        try:
            embedding = await get_embedding(chunk["text"])
            if not embedding:
                skipped += 1
                continue
            chunk_id = hashlib.md5(
                f"{session_id}_{chunk['type']}_{chunk['text'][:80]}".encode()
            ).hexdigest()
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            await pool.execute(
                """
                INSERT INTO knowledge_chunks (id, session_id, chunk_type, text, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector)
                ON CONFLICT (id) DO UPDATE
                    SET text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
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
            logger.warning(f"[{session_id}] Failed to store chunk [{chunk['type']}]: {e}")
            skipped += 1

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    logger.info(
        f"[{session_id}] Knowledge store: {stored}/{len(chunks)} chunks embedded "
        f"({skipped} skipped). Types: {by_type}"
    )
    return stored
