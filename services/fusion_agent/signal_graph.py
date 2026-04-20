# services/fusion_agent/signal_graph.py
"""
NEXUS Fusion Agent — Signal Relationship Graph

In-memory graph that maps how signals from Voice, Language, and Fusion
agents connect to each other, to speakers, and to conversation topics.

Not a database — built per session at analysis time and serialised to JSON
for the dashboard to render.
"""
import logging
from typing import Optional

logger = logging.getLogger("nexus.fusion.graph")

# Temporal windows for edge detection
CO_OCCUR_WINDOW_MS = 10_000   # signals within 10s → co_occurred
PRECEDED_WINDOW_MS = 30_000   # signal A before B within 30s → preceded
STRESS_INCONGRUENCE = 0.40    # voice stress threshold for contradicts edge
SENTIMENT_POSITIVE = 0.25     # language sentiment threshold for contradicts edge


class SignalGraph:
    """
    Adjacency-list graph of session signals.

    Nodes represent speakers, signals (voice/language/fusion), topics, and moments.
    Edges represent temporal, causal, and incongruence relationships.
    """

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self._edge_set: set[tuple] = set()  # dedup (source, target, relationship)

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def build_from_session(
        self,
        voice_signals: list[dict],
        language_signals: list[dict],
        fusion_signals: list[dict],
        transcript_segments: list[dict],
        entities: Optional[dict] = None,
    ) -> None:
        """Build the complete signal relationship graph for a session."""
        self.nodes.clear()
        self.edges.clear()
        self._edge_set.clear()

        # 1. Speaker nodes
        all_signals = voice_signals + language_signals + fusion_signals
        speakers = set()
        for s in all_signals:
            sid = s.get("speaker_id") or "unknown"
            speakers.add(sid)
        for sid in speakers:
            self._add_node(f"spk_{sid}", "speaker", sid, agent=None)

        # 2. Signal nodes — only noteworthy ones
        voice_nodes = self._add_voice_signal_nodes(voice_signals)
        lang_nodes = self._add_language_signal_nodes(language_signals)
        fusion_nodes = self._add_fusion_signal_nodes(fusion_signals)

        # 3. Topic nodes (from entity extraction)
        topic_nodes = []
        if entities and entities.get("topics"):
            topic_nodes = self._add_topic_nodes(entities["topics"])

        # 4. Edges
        all_sig_nodes = voice_nodes + lang_nodes + fusion_nodes

        # a) speaker_produced
        for nid in all_sig_nodes:
            node = self.nodes[nid]
            sid = node.get("speaker_id")
            if sid:
                self._add_edge(f"spk_{sid}", nid, "speaker_produced", 1.0)

        # b) co_occurred — cross-agent signals within 10s window
        self._build_co_occurrence_edges(voice_nodes, lang_nodes)

        # c) triggered — fusion ← voice + language
        self._build_triggered_edges(fusion_nodes, voice_nodes + lang_nodes)

        # d) contradicts — high stress + positive sentiment
        self._build_contradiction_edges(voice_nodes, lang_nodes)

        # e) preceded — temporal ordering within 30s
        self._build_temporal_edges(all_sig_nodes)

        # f) about_topic — map signals to topic phases
        if topic_nodes:
            self._build_topic_edges(all_sig_nodes, topic_nodes)

        # g) resolved — objection → later buying signal from same speaker
        self._build_resolution_edges(lang_nodes)

        # 5. Moment nodes — co-located stress + objection
        self._build_moment_nodes(voice_nodes, lang_nodes)

        logger.info(
            f"Signal graph built: {len(self.nodes)} nodes, {len(self.edges)} edges"
        )

    def get_speaker_subgraph(self, speaker_id: str) -> dict:
        """Get all nodes and edges related to a specific speaker."""
        related_nodes = set()
        spk_key = f"spk_{speaker_id}"
        if spk_key in self.nodes:
            related_nodes.add(spk_key)

        # Find all nodes connected to this speaker
        for edge in self.edges:
            if edge["source"] == spk_key or edge["target"] == spk_key:
                related_nodes.add(edge["source"])
                related_nodes.add(edge["target"])

        # Also include edges between related nodes
        sub_edges = [
            e for e in self.edges
            if e["source"] in related_nodes and e["target"] in related_nodes
        ]
        sub_nodes = {nid: self.nodes[nid] for nid in related_nodes if nid in self.nodes}

        return {"nodes": list(sub_nodes.values()), "edges": sub_edges}

    def get_key_paths(self, max_paths: int = 5) -> list[dict]:
        """
        Find the most interesting signal chains — paths that cross agent
        boundaries (voice → language → fusion).

        Returns list of:
          {"nodes": [node, node, ...], "description": str, "score": float}
        """
        # Build adjacency map
        adj: dict[str, list[tuple[str, str]]] = {}
        for edge in self.edges:
            src, tgt = edge["source"], edge["target"]
            adj.setdefault(src, []).append((tgt, edge["relationship"]))

        # Start from fusion nodes (the most valuable endpoints)
        fusion_node_ids = [
            nid for nid, n in self.nodes.items() if n["type"] == "fusion_signal"
        ]

        paths = []
        for fid in fusion_node_ids:
            # Walk backwards through triggered/co_occurred edges
            chain = self._trace_back(fid, adj, max_depth=4)
            if len(chain) >= 2:
                agents_in_chain = set()
                for nid in chain:
                    a = self.nodes.get(nid, {}).get("agent")
                    if a:
                        agents_in_chain.add(a)

                score = len(chain) * 0.3 + len(agents_in_chain) * 0.5
                fusion_node = self.nodes[fid]
                score += (fusion_node.get("confidence") or 0) * 0.2

                desc = self._describe_path(chain)
                paths.append({
                    "nodes": [self.nodes[nid] for nid in chain if nid in self.nodes],
                    "description": desc,
                    "score": round(score, 3),
                })

        paths.sort(key=lambda p: p["score"], reverse=True)
        return paths[:max_paths]

    def to_json(self) -> dict:
        """Export graph as JSON for the dashboard."""
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "node_types": self._count_by("type"),
                "edge_types": self._count_edge_types(),
            },
        }

    # ──────────────────────────────────────────────
    # Node builders
    # ──────────────────────────────────────────────

    def _add_node(
        self, node_id: str, node_type: str, label: str,
        agent: Optional[str] = None, value: Optional[float] = None,
        value_text: str = "", confidence: Optional[float] = None,
        timestamp_ms: int = 0, end_ms: int = 0,
        speaker_id: str = "", signal_type: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "label": label,
            "agent": agent,
            "value": value,
            "value_text": value_text,
            "confidence": confidence,
            "timestamp_ms": timestamp_ms,
            "end_ms": end_ms,
            "speaker_id": speaker_id,
            "signal_type": signal_type,
            "metadata": metadata or {},
        }
        return node_id

    def _add_voice_signal_nodes(self, signals: list[dict]) -> list[str]:
        """Add nodes for noteworthy voice signals only."""
        ids = []
        for i, s in enumerate(signals):
            sig_type = s.get("signal_type", "")
            value = s.get("value")
            value_text = s.get("value_text", "")

            # Filter: only noteworthy
            if sig_type == "vocal_stress_score" and (value is None or value < 0.30):
                continue
            if sig_type == "filler_detection" and value_text == "normal":
                continue
            if sig_type == "tone_classification" and value_text == "neutral":
                continue

            label = self._signal_label(sig_type, value, value_text)
            nid = f"vs_{i}"
            self._add_node(
                nid, "voice_signal", label, agent="voice",
                value=value, value_text=value_text,
                confidence=s.get("confidence"),
                timestamp_ms=s.get("window_start_ms", 0),
                end_ms=s.get("window_end_ms", 0),
                speaker_id=s.get("speaker_id", ""),
                signal_type=sig_type,
                metadata=s.get("metadata", {}),
            )
            ids.append(nid)
        return ids

    def _add_language_signal_nodes(self, signals: list[dict]) -> list[str]:
        ids = []
        for i, s in enumerate(signals):
            sig_type = s.get("signal_type", "")
            value = s.get("value")
            value_text = s.get("value_text", "")

            # Filter: only noteworthy
            if sig_type == "sentiment_score" and value is not None and abs(value) < 0.30:
                continue
            if sig_type == "power_language_score" and value is not None and 0.35 < value < 0.75:
                continue
            if sig_type == "intent_classification" and value_text in ("INFORM", "QUESTION"):
                continue

            label = self._signal_label(sig_type, value, value_text)
            nid = f"ls_{i}"
            self._add_node(
                nid, "lang_signal", label, agent="language",
                value=value, value_text=value_text,
                confidence=s.get("confidence"),
                timestamp_ms=s.get("window_start_ms", 0),
                end_ms=s.get("window_end_ms", 0),
                speaker_id=s.get("speaker_id", ""),
                signal_type=sig_type,
                metadata=s.get("metadata", {}),
            )
            ids.append(nid)
        return ids

    def _add_fusion_signal_nodes(self, signals: list[dict]) -> list[str]:
        ids = []
        for i, s in enumerate(signals):
            label = self._signal_label(
                s.get("signal_type", ""), s.get("value"), s.get("value_text", "")
            )
            nid = f"fs_{i}"
            self._add_node(
                nid, "fusion_signal", label, agent="fusion",
                value=s.get("value"),
                value_text=s.get("value_text", ""),
                confidence=s.get("confidence"),
                timestamp_ms=s.get("window_start_ms", 0),
                end_ms=s.get("window_end_ms", 0),
                speaker_id=s.get("speaker_id", ""),
                signal_type=s.get("signal_type", ""),
                metadata=s.get("metadata", {}),
            )
            ids.append(nid)
        return ids

    def _add_topic_nodes(self, topics: list[dict]) -> list[str]:
        ids = []
        for i, t in enumerate(topics):
            nid = f"topic_{i}"
            self._add_node(
                nid, "topic", t.get("name", f"Phase {i+1}"),
                timestamp_ms=t.get("start_ms", 0),
                end_ms=t.get("end_ms", 0),
            )
            ids.append(nid)
        return ids

    # ──────────────────────────────────────────────
    # Edge builders
    # ──────────────────────────────────────────────

    def _add_edge(self, source: str, target: str, relationship: str, weight: float = 1.0):
        key = (source, target, relationship)
        if key in self._edge_set:
            return
        self._edge_set.add(key)
        self.edges.append({
            "source": source,
            "target": target,
            "relationship": relationship,
            "weight": round(weight, 3),
        })

    def _build_co_occurrence_edges(self, voice_ids: list[str], lang_ids: list[str]):
        for vid in voice_ids:
            vn = self.nodes[vid]
            v_start = vn["timestamp_ms"]
            v_end = vn.get("end_ms", v_start)
            for lid in lang_ids:
                ln = self.nodes[lid]
                if vn["speaker_id"] and ln["speaker_id"] and vn["speaker_id"] != ln["speaker_id"]:
                    continue
                l_start = ln["timestamp_ms"]
                l_end = ln.get("end_ms", l_start)
                overlap_start = max(v_start, l_start)
                overlap_end = min(v_end + CO_OCCUR_WINDOW_MS, l_end + CO_OCCUR_WINDOW_MS)
                if overlap_start < overlap_end:
                    weight = 1.0 - abs(v_start - l_start) / CO_OCCUR_WINDOW_MS
                    self._add_edge(vid, lid, "co_occurred", max(weight, 0.1))

    def _build_triggered_edges(self, fusion_ids: list[str], source_ids: list[str]):
        for fid in fusion_ids:
            fn = self.nodes[fid]
            f_start = fn["timestamp_ms"]
            f_end = fn.get("end_ms", f_start)
            f_speaker = fn["speaker_id"]
            for sid in source_ids:
                sn = self.nodes[sid]
                if f_speaker and sn["speaker_id"] and f_speaker != sn["speaker_id"]:
                    continue
                s_start = sn["timestamp_ms"]
                if abs(s_start - f_start) <= CO_OCCUR_WINDOW_MS or (f_start <= s_start <= f_end):
                    self._add_edge(sid, fid, "triggered", sn.get("confidence") or 0.5)

    def _build_contradiction_edges(self, voice_ids: list[str], lang_ids: list[str]):
        for vid in voice_ids:
            vn = self.nodes[vid]
            if vn["signal_type"] != "vocal_stress_score":
                continue
            if (vn["value"] or 0) < STRESS_INCONGRUENCE:
                continue
            for lid in lang_ids:
                ln = self.nodes[lid]
                if vn["speaker_id"] and ln["speaker_id"] and vn["speaker_id"] != ln["speaker_id"]:
                    continue
                if ln["signal_type"] != "sentiment_score":
                    continue
                if (ln["value"] or 0) < SENTIMENT_POSITIVE:
                    continue
                if abs(vn["timestamp_ms"] - ln["timestamp_ms"]) <= CO_OCCUR_WINDOW_MS:
                    self._add_edge(vid, lid, "contradicts", 0.8)

    def _build_temporal_edges(self, node_ids: list[str]):
        if len(node_ids) < 2:
            return
        sorted_ids = sorted(node_ids, key=lambda nid: self.nodes[nid]["timestamp_ms"])
        for i in range(len(sorted_ids) - 1):
            a = self.nodes[sorted_ids[i]]
            b = self.nodes[sorted_ids[i + 1]]
            if a["speaker_id"] and b["speaker_id"] and a["speaker_id"] != b["speaker_id"]:
                continue
            gap = b["timestamp_ms"] - a["timestamp_ms"]
            if 0 < gap <= PRECEDED_WINDOW_MS:
                weight = 1.0 - gap / PRECEDED_WINDOW_MS
                self._add_edge(sorted_ids[i], sorted_ids[i + 1], "preceded", max(weight, 0.1))

    def _build_topic_edges(self, signal_ids: list[str], topic_ids: list[str]):
        for tid in topic_ids:
            tn = self.nodes[tid]
            t_start = tn["timestamp_ms"]
            t_end = tn.get("end_ms", t_start)
            for sid in signal_ids:
                sn = self.nodes[sid]
                s_time = sn["timestamp_ms"]
                if t_start <= s_time <= t_end:
                    self._add_edge(sid, tid, "about_topic", 1.0)

    def _build_moment_nodes(self, voice_ids: list[str], lang_ids: list[str]):
        """Create 'moment' nodes where stress + objection co-locate."""
        stress_nodes = [
            nid for nid in voice_ids
            if self.nodes[nid]["signal_type"] == "vocal_stress_score"
            and (self.nodes[nid]["value"] or 0) >= 0.40
        ]
        obj_nodes = [
            nid for nid in lang_ids
            if self.nodes[nid]["signal_type"] == "objection_signal"
        ]
        moment_idx = 0
        for sn_id in stress_nodes:
            sn = self.nodes[sn_id]
            for on_id in obj_nodes:
                on = self.nodes[on_id]
                if abs(sn["timestamp_ms"] - on["timestamp_ms"]) <= CO_OCCUR_WINDOW_MS:
                    mid = f"moment_{moment_idx}"
                    self._add_node(
                        mid, "moment",
                        f"Stress + Objection @ {sn['timestamp_ms']//1000}s",
                        timestamp_ms=min(sn["timestamp_ms"], on["timestamp_ms"]),
                        end_ms=max(sn.get("end_ms", 0), on.get("end_ms", 0)),
                        speaker_id=sn["speaker_id"],
                    )
                    self._add_edge(sn_id, mid, "co_occurred", 0.9)
                    self._add_edge(on_id, mid, "co_occurred", 0.9)
                    moment_idx += 1

    def _build_resolution_edges(self, lang_ids: list[str]):
        """Link each objection to its EARLIEST subsequent buying signal only."""
        obj_nodes = [
            nid for nid in lang_ids
            if self.nodes[nid]["signal_type"] == "objection_signal"
        ]
        buy_nodes = sorted(
            [nid for nid in lang_ids if self.nodes[nid]["signal_type"] == "buying_signal"],
            key=lambda nid: self.nodes[nid]["timestamp_ms"],
        )
        for oid in obj_nodes:
            on = self.nodes[oid]
            for bid in buy_nodes:
                bn = self.nodes[bid]
                if on["speaker_id"] and bn["speaker_id"] and on["speaker_id"] != bn["speaker_id"]:
                    continue
                if bn["timestamp_ms"] > on["timestamp_ms"]:
                    self._add_edge(oid, bid, "resolved", 0.85)
                    break  # Only link to the first (earliest) resolution

    # ──────────────────────────────────────────────
    # Path tracing
    # ──────────────────────────────────────────────

    def _trace_back(self, start: str, adj: dict, max_depth: int = 4) -> list[str]:
        """BFS backwards from a node to find the longest chain."""
        # Build reverse adjacency
        rev: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge["relationship"] in ("triggered", "co_occurred", "preceded", "resolved"):
                rev.setdefault(edge["target"], []).append(edge["source"])

        best_path = [start]
        queue = [(start, [start])]
        visited = {start}

        while queue:
            node, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            for prev_node in rev.get(node, []):
                if prev_node not in visited and prev_node in self.nodes:
                    visited.add(prev_node)
                    new_path = [prev_node] + path
                    if len(new_path) > len(best_path):
                        best_path = new_path
                    queue.append((prev_node, new_path))

        return best_path

    def _describe_path(self, chain: list[str]) -> str:
        parts = []
        for nid in chain:
            node = self.nodes.get(nid, {})
            label = node.get("label", nid)
            ts = node.get("timestamp_ms", 0)
            parts.append(f"{label} ({ts // 1000}s)")
        return " → ".join(parts)

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _signal_label(sig_type: str, value, value_text: str) -> str:
        labels = {
            "vocal_stress_score": f"Stress {int((value or 0) * 100)}%",
            "filler_detection": f"Filler ({value_text})",
            "pitch_elevation_flag": f"Pitch ↑ ({value_text})",
            "speech_rate_anomaly": f"Rate ({value_text})",
            "tone_classification": f"Tone: {value_text}",
            "sentiment_score": f"Sentiment {value_text}",
            "buying_signal": "Buying Signal",
            "objection_signal": "Objection",
            "power_language_score": f"Power {int((value or 0) * 100)}%",
            "intent_classification": f"Intent: {value_text}",
            "credibility_assessment": f"Credibility: {value_text}",
            "verbal_incongruence": f"Incongruence: {value_text}",
            "urgency_authenticity": f"Urgency: {value_text}",
            # Voice Agent new rules
            "monotone_flag": "Monotone detected",
            "energy_level": f"Energy: {value_text}",
            "pause_classification": f"Pause: {value_text}",
            "interruption_event": f"Interruption ({value_text})",
            "talk_time_ratio": f"Talk: {int((value or 0) * 100)}%",
            # Language Agent new rules
            "emotional_intensity": f"Emotion: {value_text}",
            "persuasion_technique": f"Persuasion: {value_text}",
            "question_type": f"Q: {value_text}",
            "gottman_horsemen": f"Gottman: {value_text}",
            "empathy_language": "Empathy detected",
            "clarity_score": f"Clarity: {int((value or 0) * 100)}%",
            "topic_shift": "Topic Change",
            # Conversation Agent
            "turn_taking_pattern": f"Turns: {value_text}",
            "response_latency_pattern": f"Latency: {value_text}",
            "dominance_score": f"Dominance: {value_text}",
            "interruption_pattern": f"Interruptions: {value_text}",
            "rapport_indicator": f"Rapport: {value_text}",
            "conversation_engagement": f"Engagement: {value_text}",
            "conversation_balance": f"Balance: {value_text}",
            "conflict_score": f"Conflict: {value_text}",
        }
        return labels.get(sig_type, sig_type.replace("_", " ").title())

    def _count_by(self, field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes.values():
            val = n.get(field, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts

    def _count_edge_types(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.edges:
            r = e["relationship"]
            counts[r] = counts.get(r, 0) + 1
        return counts
