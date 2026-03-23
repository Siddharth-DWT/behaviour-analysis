"""
NEXUS Fusion Agent — Graph Analytics Engine

Uses the SignalGraph's topology to discover patterns that simple
pairwise rules miss: tension clusters, momentum shifts, resolution
paths, and persistent incongruence.
"""
import logging
from typing import Optional

logger = logging.getLogger("nexus.fusion.graph_analytics")

# Temporal bucket size for tension clustering
BUCKET_MS = 10_000


class GraphAnalytics:
    """Compute higher-order insights from a SignalGraph."""

    def __init__(self, graph):
        self.nodes = graph.nodes
        self.edges = graph.edges

    def compute_all(self) -> dict:
        return {
            "tension_clusters": self.find_tension_clusters(),
            "speaker_patterns": self.compute_speaker_patterns(),
            "topic_signal_density": self.compute_topic_signal_density(),
            "momentum": self.compute_momentum(),
            "resolution_paths": self.find_resolution_paths(),
            "incongruence_patterns": self.compute_incongruence_patterns(),
        }

    # ──────────────────────────────────────────
    # Tension Clusters
    # ──────────────────────────────────────────

    def find_tension_clusters(self) -> list[dict]:
        """Group negative signals into 10s buckets; 3+ signals = tension cluster."""
        negative_types = {
            "vocal_stress_score", "objection_signal", "filler_detection",
            "credibility_assessment", "verbal_incongruence",
        }
        negative_nodes = []
        for n in self.nodes.values():
            if n["type"] not in ("voice_signal", "lang_signal", "fusion_signal"):
                continue
            sig = n.get("signal_type", "")
            if sig in negative_types:
                negative_nodes.append(n)
            elif sig == "sentiment_score" and (n.get("value") or 0) < -0.3:
                negative_nodes.append(n)

        if not negative_nodes:
            return []

        # Bucket by time
        buckets: dict[int, list] = {}
        for n in negative_nodes:
            key = (n["timestamp_ms"] // BUCKET_MS) * BUCKET_MS
            buckets.setdefault(key, []).append(n)

        # Find clusters (3+ signals in a bucket)
        raw_clusters = []
        for ts, nodes in sorted(buckets.items()):
            if len(nodes) >= 3:
                peak_stress = max(
                    (n.get("value") or 0)
                    for n in nodes
                    if n.get("signal_type") == "vocal_stress_score"
                ) if any(n.get("signal_type") == "vocal_stress_score" for n in nodes) else 0
                has_objection = any(n.get("signal_type") == "objection_signal" for n in nodes)
                speaker_ids = list(set(n.get("speaker_id", "") for n in nodes if n.get("speaker_id")))
                raw_clusters.append({
                    "timestamp_ms": ts,
                    "duration_ms": BUCKET_MS,
                    "signal_count": len(nodes),
                    "peak_stress": round(peak_stress, 3),
                    "has_objection": has_objection,
                    "speaker_id": speaker_ids[0] if len(speaker_ids) == 1 else "multiple",
                    "severity": "high" if len(nodes) >= 5 else "moderate",
                })

        # Merge adjacent buckets
        merged = []
        for c in raw_clusters:
            if merged and c["timestamp_ms"] <= merged[-1]["timestamp_ms"] + merged[-1]["duration_ms"]:
                prev = merged[-1]
                prev["duration_ms"] = c["timestamp_ms"] + c["duration_ms"] - prev["timestamp_ms"]
                prev["signal_count"] += c["signal_count"]
                prev["peak_stress"] = max(prev["peak_stress"], c["peak_stress"])
                prev["has_objection"] = prev["has_objection"] or c["has_objection"]
                prev["severity"] = "high" if prev["signal_count"] >= 5 else "moderate"
            else:
                merged.append(c)

        return merged

    # ──────────────────────────────────────────
    # Speaker Patterns
    # ──────────────────────────────────────────

    def compute_speaker_patterns(self) -> dict[str, dict]:
        """Per-speaker: signal density, contradiction ratio, response pattern, escalation."""
        speakers: dict[str, dict] = {}

        # Collect signal nodes per speaker
        sig_nodes_by_speaker: dict[str, list] = {}
        for n in self.nodes.values():
            if n["type"] not in ("voice_signal", "lang_signal", "fusion_signal"):
                continue
            sid = n.get("speaker_id", "")
            if not sid:
                continue
            sig_nodes_by_speaker.setdefault(sid, []).append(n)

        # Count contradicts edges per speaker
        contradicts_by_speaker: dict[str, int] = {}
        for e in self.edges:
            if e["relationship"] == "contradicts":
                src = self.nodes.get(e["source"], {})
                sid = src.get("speaker_id", "")
                if sid:
                    contradicts_by_speaker[sid] = contradicts_by_speaker.get(sid, 0) + 1

        for sid, nodes in sig_nodes_by_speaker.items():
            if not nodes:
                continue

            # Time span
            times = [n["timestamp_ms"] for n in nodes]
            span_minutes = max(1, (max(times) - min(times)) / 60000)
            signal_density = len(nodes) / span_minutes

            # Contradiction ratio
            total_sigs = len(nodes)
            contradicts = contradicts_by_speaker.get(sid, 0)
            contradiction_ratio = contradicts / total_sigs if total_sigs > 0 else 0

            # Response pattern based on signal mix
            stress_count = sum(1 for n in nodes if n.get("signal_type") == "vocal_stress_score")
            buy_count = sum(1 for n in nodes if n.get("signal_type") == "buying_signal")
            obj_count = sum(1 for n in nodes if n.get("signal_type") == "objection_signal")

            if obj_count > buy_count and stress_count > len(nodes) * 0.3:
                response_pattern = "defensive"
            elif buy_count > 0 or contradiction_ratio < 0.1:
                response_pattern = "engaged"
            else:
                response_pattern = "passive"

            # Escalation trend: compare stress in first half vs second half
            sorted_stress = sorted(
                [n for n in nodes if n.get("signal_type") == "vocal_stress_score"],
                key=lambda n: n["timestamp_ms"],
            )
            if len(sorted_stress) >= 4:
                mid = len(sorted_stress) // 2
                first_half_avg = sum((n.get("value") or 0) for n in sorted_stress[:mid]) / mid
                second_half_avg = sum((n.get("value") or 0) for n in sorted_stress[mid:]) / (len(sorted_stress) - mid)
                if second_half_avg > first_half_avg + 0.05:
                    escalation_trend = "increasing"
                elif second_half_avg < first_half_avg - 0.05:
                    escalation_trend = "decreasing"
                else:
                    escalation_trend = "stable"
            else:
                escalation_trend = "stable"

            speakers[sid] = {
                "signal_density": round(signal_density, 2),
                "contradiction_ratio": round(contradiction_ratio, 3),
                "response_pattern": response_pattern,
                "escalation_trend": escalation_trend,
            }

        return speakers

    # ──────────────────────────────────────────
    # Topic Signal Density
    # ──────────────────────────────────────────

    def compute_topic_signal_density(self) -> list[dict]:
        """Per topic: signal counts by type, risk and opportunity levels."""
        topic_nodes = [n for n in self.nodes.values() if n["type"] == "topic"]
        if not topic_nodes:
            return []

        # Map signals to topics via about_topic edges
        topic_signals: dict[str, list] = {n["id"]: [] for n in topic_nodes}
        for e in self.edges:
            if e["relationship"] == "about_topic" and e["target"] in topic_signals:
                src_node = self.nodes.get(e["source"])
                if src_node:
                    topic_signals[e["target"]].append(src_node)

        results = []
        for tn in sorted(topic_nodes, key=lambda n: n["timestamp_ms"]):
            sigs = topic_signals.get(tn["id"], [])
            stress_sigs = [s for s in sigs if s.get("signal_type") == "vocal_stress_score"]
            buy_sigs = [s for s in sigs if s.get("signal_type") == "buying_signal"]
            obj_sigs = [s for s in sigs if s.get("signal_type") == "objection_signal"]

            avg_stress = (
                sum((s.get("value") or 0) for s in stress_sigs) / len(stress_sigs)
                if stress_sigs else 0
            )
            negative_count = len(obj_sigs) + len([
                s for s in sigs
                if s.get("signal_type") == "sentiment_score" and (s.get("value") or 0) < -0.3
            ])

            risk_level = "high" if negative_count >= 2 or avg_stress > 0.5 else (
                "moderate" if negative_count >= 1 or avg_stress > 0.3 else "low"
            )
            opportunity_level = "high" if len(buy_sigs) >= 2 else (
                "moderate" if len(buy_sigs) >= 1 else "low"
            )

            results.append({
                "topic_name": tn["label"],
                "timestamp_ms": tn["timestamp_ms"],
                "total_signals": len(sigs),
                "stress_signals": len(stress_sigs),
                "buying_signals": len(buy_sigs),
                "objection_signals": len(obj_sigs),
                "avg_stress": round(avg_stress, 3),
                "risk_level": risk_level,
                "opportunity_level": opportunity_level,
            })

        return results

    # ──────────────────────────────────────────
    # Momentum
    # ──────────────────────────────────────────

    def compute_momentum(self) -> dict:
        """Divide conversation into quartiles, compute trajectory."""
        sig_nodes = [
            n for n in self.nodes.values()
            if n["type"] in ("voice_signal", "lang_signal") and n["timestamp_ms"] > 0
        ]
        if len(sig_nodes) < 4:
            return {"overall_trajectory": "stable", "momentum_score": 0, "turning_point_ms": None, "quartiles": []}

        sig_nodes.sort(key=lambda n: n["timestamp_ms"])
        min_t = sig_nodes[0]["timestamp_ms"]
        max_t = sig_nodes[-1]["timestamp_ms"]
        span = max_t - min_t
        if span <= 0:
            return {"overall_trajectory": "stable", "momentum_score": 0, "turning_point_ms": None, "quartiles": []}

        quartile_dur = span / 4
        quartiles = []
        for q in range(4):
            q_start = min_t + q * quartile_dur
            q_end = q_start + quartile_dur
            q_nodes = [n for n in sig_nodes if q_start <= n["timestamp_ms"] < q_end]

            positive = sum(1 for n in q_nodes if n.get("signal_type") in ("buying_signal",)
                          or (n.get("signal_type") == "sentiment_score" and (n.get("value") or 0) > 0.3))
            negative = sum(1 for n in q_nodes if n.get("signal_type") in ("objection_signal",)
                          or (n.get("signal_type") == "vocal_stress_score" and (n.get("value") or 0) > 0.4)
                          or (n.get("signal_type") == "sentiment_score" and (n.get("value") or 0) < -0.3))
            stress_vals = [(n.get("value") or 0) for n in q_nodes if n.get("signal_type") == "vocal_stress_score"]
            avg_stress = sum(stress_vals) / len(stress_vals) if stress_vals else 0

            quartiles.append({
                "quartile": q + 1,
                "start_ms": int(q_start),
                "positive_signals": positive,
                "negative_signals": negative,
                "net_sentiment": positive - negative,
                "avg_stress": round(avg_stress, 3),
            })

        # Trajectory
        nets = [q["net_sentiment"] for q in quartiles]
        if nets[-1] > nets[0] + 1:
            trajectory = "positive"
        elif nets[-1] < nets[0] - 1:
            trajectory = "negative"
        elif max(nets) - min(nets) >= 3:
            trajectory = "volatile"
        else:
            trajectory = "stable"

        # Turning point: largest shift between consecutive quartiles
        turning_point_ms = None
        max_shift = 0
        for i in range(1, len(nets)):
            shift = abs(nets[i] - nets[i - 1])
            if shift > max_shift:
                max_shift = shift
                turning_point_ms = quartiles[i]["start_ms"]

        # Momentum score: -1 to +1
        total_pos = sum(q["positive_signals"] for q in quartiles)
        total_neg = sum(q["negative_signals"] for q in quartiles)
        total = total_pos + total_neg
        momentum_score = (total_pos - total_neg) / total if total > 0 else 0

        return {
            "overall_trajectory": trajectory,
            "momentum_score": round(momentum_score, 3),
            "turning_point_ms": turning_point_ms if max_shift >= 2 else None,
            "quartiles": quartiles,
        }

    # ──────────────────────────────────────────
    # Resolution Paths
    # ──────────────────────────────────────────

    def find_resolution_paths(self) -> list[dict]:
        """Trace objection → buying_signal resolution paths via 'resolved' edges."""
        paths = []
        for e in self.edges:
            if e["relationship"] != "resolved":
                continue
            obj_node = self.nodes.get(e["source"], {})
            buy_node = self.nodes.get(e["target"], {})
            if not obj_node or not buy_node:
                continue

            time_to_resolve = buy_node["timestamp_ms"] - obj_node["timestamp_ms"]

            paths.append({
                "objection_text": obj_node.get("label", ""),
                "objection_ms": obj_node["timestamp_ms"],
                "resolution_type": buy_node.get("label", "buying_signal"),
                "resolution_ms": buy_node["timestamp_ms"],
                "time_to_resolve_ms": max(0, time_to_resolve),
                "speaker_id": obj_node.get("speaker_id", ""),
            })

        return sorted(paths, key=lambda p: p["objection_ms"])

    # ──────────────────────────────────────────
    # Incongruence Patterns
    # ──────────────────────────────────────────

    def compute_incongruence_patterns(self) -> dict[str, dict]:
        """Per speaker: count contradicts edges, classify persistence."""
        contradicts_by_speaker: dict[str, list[int]] = {}

        for e in self.edges:
            if e["relationship"] != "contradicts":
                continue
            src = self.nodes.get(e["source"], {})
            sid = src.get("speaker_id", "")
            if sid:
                contradicts_by_speaker.setdefault(sid, []).append(src["timestamp_ms"])

        results = {}
        for sid, timestamps in contradicts_by_speaker.items():
            count = len(timestamps)
            # Group into 10s windows to determine persistence
            windows = set(ts // BUCKET_MS for ts in timestamps)
            if len(windows) >= 3:
                consistency = "persistent"
            elif len(windows) == 2:
                consistency = "occasional"
            else:
                consistency = "isolated"

            worst_ms = max(timestamps) if timestamps else 0

            results[sid] = {
                "total_contradicts_edges": count,
                "distinct_time_windows": len(windows),
                "consistency": consistency,
                "worst_incongruence_ms": worst_ms,
            }

        return results
