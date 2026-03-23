/**
 * Data transform and filtering for the Signal Network view.
 * Converts backend SignalGraphData → filtered VisualNode[] / VisualEdge[] + stats.
 */
import type {
  SignalGraphData, BackendGraphNode,
  SignalNetworkFilters, VisualNode, VisualEdge, NetworkStats, VisualNodeType,
} from "./types";
import {
  NODE_COLORS, SPEAKER_COLORS, EDGE_STYLES,
  SIGNAL_LABELS, NODE_TYPE_LABELS,
} from "./signalNetworkConstants";

// ─────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────

function humanLabel(n: BackendGraphNode): string {
  if (n.type === "speaker") return n.label || n.speaker_id || "Speaker";
  if (n.type === "topic")   return n.label || "Topic";
  // For signals, use readable name + value
  const base = SIGNAL_LABELS[n.signal_type ?? ""] || n.signal_type?.replace(/_/g, " ") || n.label;
  if (n.value != null && n.signal_type?.includes("score")) {
    return `${base} ${Math.round(n.value * 100)}%`;
  }
  if (n.value_text) return `${base}: ${n.value_text}`;
  return base;
}

function detailText(n: BackendGraphNode): string {
  const parts: string[] = [];
  if (n.signal_type) parts.push(n.signal_type);
  if (n.agent) parts.push(`agent: ${n.agent}`);
  if (n.value != null) parts.push(`value: ${n.value.toFixed(3)}`);
  if (n.confidence != null) parts.push(`confidence: ${Math.round(n.confidence * 100)}%`);
  if (n.speaker_id) parts.push(`speaker: ${n.speaker_id}`);
  return parts.join(" · ");
}

function nodeWidth(type: VisualNodeType, confidence: number): number {
  if (type === "speaker") return 88;
  if (type === "topic") return 100;
  if (type === "fusion_signal") return 140;
  // Scale signal node width by confidence
  return Math.round(100 + confidence * 60);
}

function nodeHeight(type: VisualNodeType): number {
  if (type === "speaker") return 88;
  if (type === "topic") return 32;
  if (type === "fusion_signal") return 40;
  return 32;
}

// ─────────────────────────────────────────
// Main build function
// ─────────────────────────────────────────

export function buildFilteredSignalNetwork(
  sg: SignalGraphData,
  speakerRoles: Record<string, string>,
  filters: SignalNetworkFilters,
): { nodes: VisualNode[]; edges: VisualEdge[]; stats: NetworkStats } {

  const allSpeakers = new Set<string>();
  for (const n of sg.nodes) if (n.type === "speaker" && n.speaker_id) allSpeakers.add(n.speaker_id);
  const speakerList = [...allSpeakers];

  // ── 1. Filter nodes ──
  const includedIds = new Set<string>();
  for (const n of sg.nodes) {
    // Always include speakers and topics
    if (n.type === "speaker" || n.type === "topic") {
      includedIds.add(n.id);
      continue;
    }
    // Agent filter
    if (filters.agents.size > 0 && n.agent && !filters.agents.has(n.agent)) continue;
    // Speaker filter
    if (filters.speakers.size > 0 && n.speaker_id && !filters.speakers.has(n.speaker_id)) continue;
    // Confidence filter
    if ((n.confidence ?? 0) < filters.confidenceMin) continue;
    // Node type filter
    if (filters.nodeTypes.size > 0 && !filters.nodeTypes.has(n.type)) continue;
    // Search
    if (filters.searchTerm) {
      const q = filters.searchTerm.toLowerCase();
      const haystack = `${n.label} ${n.signal_type ?? ""} ${n.value_text ?? ""}`.toLowerCase();
      if (!haystack.includes(q)) continue;
    }
    includedIds.add(n.id);
  }

  // ── 2. Build degree map ──
  const degreeMap = new Map<string, number>();
  for (const e of sg.edges) {
    if (!includedIds.has(e.source) || !includedIds.has(e.target)) continue;
    if (filters.edgeTypes.size > 0 && !filters.edgeTypes.has(e.relationship)) continue;
    degreeMap.set(e.source, (degreeMap.get(e.source) ?? 0) + 1);
    degreeMap.set(e.target, (degreeMap.get(e.target) ?? 0) + 1);
  }

  // ── 3. Convert to VisualNodes ──
  const nodeMap = new Map<string, BackendGraphNode>();
  for (const n of sg.nodes) nodeMap.set(n.id, n);

  const nodes: VisualNode[] = [];
  for (const n of sg.nodes) {
    if (!includedIds.has(n.id)) continue;
    const type = n.type as VisualNodeType;
    const conf = n.confidence ?? 0.5;
    const spkIdx = speakerList.indexOf(n.speaker_id ?? "");
    const isSpk = type === "speaker";

    nodes.push({
      id: n.id,
      type,
      label: humanLabel(n),
      detail: detailText(n),
      speaker: n.speaker_id || undefined,
      agent: n.agent || undefined,
      signalType: n.signal_type,
      value: n.value,
      valueText: n.value_text,
      confidence: conf,
      color: isSpk ? SPEAKER_COLORS[spkIdx >= 0 ? spkIdx : 0] : NODE_COLORS[type] ?? "#888",
      borderColor: isSpk ? SPEAKER_COLORS[spkIdx >= 0 ? spkIdx : 0] : NODE_COLORS[type] ?? "#888",
      width: nodeWidth(type, conf),
      height: nodeHeight(type),
      timeMs: n.timestamp_ms ?? 0,
      endMs: n.end_ms,
      degree: degreeMap.get(n.id) ?? 0,
      pinned: isSpk,
      metadata: n.metadata,
    });
  }

  // ── 4. Convert to VisualEdges ──
  const edges: VisualEdge[] = [];
  const edgeTypeCounts: Record<string, number> = {};
  let edgeIdx = 0;

  for (const e of sg.edges) {
    if (!includedIds.has(e.source) || !includedIds.has(e.target)) continue;
    if (filters.edgeTypes.size > 0 && !filters.edgeTypes.has(e.relationship)) continue;

    edgeTypeCounts[e.relationship] = (edgeTypeCounts[e.relationship] ?? 0) + 1;
    const style = EDGE_STYLES[e.relationship] ?? { color: "#88888866", dashed: true, width: 1, opacity: 0.3 };

    edges.push({
      id: `e_${edgeIdx++}`,
      source: e.source,
      target: e.target,
      edgeType: e.relationship,
      color: style.color,
      width: style.width,
      dashed: style.dashed,
      animated: style.animated,
      label: e.relationship,
      opacity: style.opacity,
    });
  }

  // ── 5. Compute stats ──
  const nodeTypeCounts: Record<string, number> = {};
  let totalConf = 0;
  let confCount = 0;
  const speakerSignalCounts: Record<string, number> = {};

  for (const n of nodes) {
    nodeTypeCounts[n.type] = (nodeTypeCounts[n.type] ?? 0) + 1;
    if (n.confidence != null && n.type !== "speaker" && n.type !== "topic") {
      totalConf += n.confidence;
      confCount++;
    }
    if (n.speaker && n.type !== "speaker") {
      speakerSignalCounts[n.speaker] = (speakerSignalCounts[n.speaker] ?? 0) + 1;
    }
  }

  // Top connected
  const topConnected = [...nodes]
    .filter(n => n.type !== "speaker")
    .sort((a, b) => b.degree - a.degree)
    .slice(0, 5)
    .map(n => ({ id: n.id, label: n.label, degree: n.degree, type: NODE_TYPE_LABELS[n.type] ?? n.type }));

  const stats: NetworkStats = {
    totalNodes: sg.nodes.length,
    totalEdges: sg.edges.length,
    visibleNodes: nodes.length,
    visibleEdges: edges.length,
    contradictionCount: edgeTypeCounts["contradicts"] ?? 0,
    triggerCount: edgeTypeCounts["triggered"] ?? 0,
    resolveCount: edgeTypeCounts["resolves"] ?? 0,
    topConnected,
    edgeTypeCounts,
    nodeTypeCounts,
    avgConfidence: confCount > 0 ? totalConf / confCount : 0,
    speakerSignalCounts,
  };

  return { nodes, edges, stats };
}
