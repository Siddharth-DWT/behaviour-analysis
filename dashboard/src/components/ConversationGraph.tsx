import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import * as d3 from "d3";
import type { Signal, TranscriptSegment } from "../api/client";

// ═══════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════

interface Topic {
  name: string;
  start_ms: number;
  end_ms: number;
}

interface Entities {
  topics?: Topic[];
  people?: Array<{ name: string; role: string; speaker_label?: string }>;
  objections?: Array<{ text: string; timestamp_ms: number; resolved: boolean }>;
  commitments?: Array<{ text: string; speaker: string; timestamp_ms: number }>;
  [key: string]: unknown;
}

type GraphNodeType = "speaker" | "utterance" | "topic" | "signal" | "insight";

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  type: GraphNodeType;
  label: string;
  detail: string;
  speaker?: string;
  color: string;
  borderColor: string;
  width: number;
  height: number;
  time_ms: number;
  badges: string[];
  confidence?: number;
  pinned?: boolean;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  id: string;
  edgeType: string;
  color: string;
  width: number;
  dashed: boolean;
  label?: string;
}

// Backend signal graph types
interface BackendGraphNode {
  id: string;
  type: string;
  label: string;
  agent?: string;
  value?: number;
  value_text?: string;
  confidence?: number;
  timestamp_ms?: number;
  speaker_id?: string;
  signal_type?: string;
  metadata?: Record<string, unknown>;
}

interface BackendGraphEdge {
  source: string;
  target: string;
  relationship: string;
}

interface SignalGraphData {
  nodes: BackendGraphNode[];
  edges: BackendGraphEdge[];
  stats?: Record<string, unknown>;
}

interface ConversationGraphProps {
  segments: TranscriptSegment[];
  signals: Signal[];
  entities: Entities;
  speakerRoles: Record<string, string>;
  durationMs: number;
  onClose: () => void;
  signalGraph?: SignalGraphData;
}

// ═══════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════

const SPEAKER_COLORS: Record<number, string> = {
  0: "#4F8BFF",
  1: "#F59E0B",
  2: "#8B5CF6",
  3: "#10B981",
};

const AGENT_COLORS: Record<string, string> = {
  voice: "#6366F1",
  language: "#06B6D4",
  fusion: "#F97316",
};

const TOPIC_COLORS = [
  "#4F8BFF", "#8B5CF6", "#10B981", "#F59E0B", "#EC4899", "#6366F1", "#22C55E",
];

// ═══════════════════════════════════════════
// DATA BUILDER
// ═══════════════════════════════════════════

function buildGraphData(
  segments: TranscriptSegment[],
  signals: Signal[],
  entities: Entities,
  speakerRoles: Record<string, string>,
  durationMs: number,
  simplified: boolean,
): { nodes: GraphNode[]; links: GraphLink[] } {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const nodeMap = new Map<string, GraphNode>();

  // ── Aggregate consecutive same-speaker segments into turns ──
  interface Turn {
    id: string;
    speaker: string;
    start_ms: number;
    end_ms: number;
    text: string;
    segIndices: number[];
    wordCount: number;
  }
  const turns: Turn[] = [];
  let currentTurn: Turn | null = null;

  for (const seg of segments) {
    const speaker = seg.speaker_label || "Unknown";
    if (
      currentTurn &&
      currentTurn.speaker === speaker &&
      currentTurn.segIndices.length < 4
    ) {
      currentTurn.end_ms = seg.end_ms;
      currentTurn.text += " " + seg.text;
      currentTurn.segIndices.push(seg.segment_index);
      currentTurn.wordCount += seg.word_count || seg.text.split(" ").length;
    } else {
      if (currentTurn) turns.push(currentTurn);
      currentTurn = {
        id: `turn_${turns.length}`,
        speaker,
        start_ms: seg.start_ms,
        end_ms: seg.end_ms,
        text: seg.text,
        segIndices: [seg.segment_index],
        wordCount: seg.word_count || seg.text.split(" ").length,
      };
    }
  }
  if (currentTurn) turns.push(currentTurn);

  // ── Speaker anchor nodes ──
  const speakerLabels = [...new Set(turns.map((t) => t.speaker))];
  speakerLabels.forEach((label, i) => {
    const role = speakerRoles[label] || "";
    const name = entities.people?.find(
      (p) => p.speaker_label === label || p.role?.toLowerCase() === role?.toLowerCase()
    )?.name;
    const node: GraphNode = {
      id: `spk_${label}`,
      type: "speaker",
      label: name || role || label,
      detail: `${role} (${label})`,
      speaker: label,
      color: SPEAKER_COLORS[i % 4],
      borderColor: SPEAKER_COLORS[i % 4],
      width: 90,
      height: 90,
      time_ms: 0,
      badges: [],
      pinned: true,
      fx: i === 0 ? 80 : undefined,
      fy: i === 0 ? 200 : undefined,
    };
    if (i === 1) {
      node.fx = 720;
      node.fy = 200;
    }
    nodes.push(node);
    nodeMap.set(node.id, node);
  });

  // ── Utterance (turn) nodes ──
  turns.forEach((turn, i) => {
    const spkIdx = speakerLabels.indexOf(turn.speaker);
    const truncText =
      turn.text.length > 50 ? turn.text.slice(0, 47) + "..." : turn.text;
    const w = Math.max(100, Math.min(220, 80 + turn.wordCount * 3));

    // Badges from signals in this time range
    const badges: string[] = [];
    for (const s of signals) {
      if (s.window_start_ms > turn.end_ms || s.window_end_ms < turn.start_ms) continue;
      if (s.speaker_label && s.speaker_label !== turn.speaker) continue;
      if (s.signal_type === "vocal_stress_score" && (s.value ?? 0) > 0.3)
        badges.push("stress");
      if (s.signal_type === "buying_signal") badges.push("buying");
      if (s.signal_type === "objection_signal") badges.push("objection");
      if (s.signal_type === "sentiment_score" && Math.abs(s.value ?? 0) > 0.35)
        badges.push("sentiment");
      if (s.agent === "fusion") badges.push("fusion");
    }

    const node: GraphNode = {
      id: turn.id,
      type: "utterance",
      label: truncText,
      detail: turn.text,
      speaker: turn.speaker,
      color: "var(--bg-surface, #1E1E2E)",
      borderColor: SPEAKER_COLORS[spkIdx % 4],
      width: w,
      height: 40,
      time_ms: turn.start_ms,
      badges: [...new Set(badges)],
    };
    nodes.push(node);
    nodeMap.set(node.id, node);

    // Link to speaker
    links.push({
      id: `spk_link_${i}`,
      source: `spk_${turn.speaker}`,
      target: turn.id,
      edgeType: "speaker_owns",
      color: "transparent",
      width: 0,
      dashed: false,
    });
  });

  // ── Conversation flow edges ──
  for (let i = 1; i < turns.length; i++) {
    const prev = turns[i - 1];
    const curr = turns[i];
    const crossSpeaker = prev.speaker !== curr.speaker;
    const latency = curr.start_ms - prev.end_ms;

    links.push({
      id: `flow_${i}`,
      source: prev.id,
      target: curr.id,
      edgeType: crossSpeaker ? "response_to" : "conversation_flow",
      color: crossSpeaker ? "var(--accent-blue, #4F8BFF)" : "var(--border, #333)",
      width: crossSpeaker ? 2 : 1,
      dashed: false,
      label: crossSpeaker && latency > 0 ? `${latency}ms` : undefined,
    });
  }

  if (simplified) {
    return { nodes, links };
  }

  // ── Topic nodes ──
  const topics = entities.topics || [];
  topics.forEach((topic, i) => {
    const node: GraphNode = {
      id: `topic_${i}`,
      type: "topic",
      label: topic.name,
      detail: `${Math.floor(topic.start_ms / 1000)}s - ${Math.floor(topic.end_ms / 1000)}s`,
      color: TOPIC_COLORS[i % TOPIC_COLORS.length],
      borderColor: TOPIC_COLORS[i % TOPIC_COLORS.length],
      width: 100,
      height: 36,
      time_ms: topic.start_ms,
      badges: [],
      fx: 850,
    };
    nodes.push(node);
    nodeMap.set(node.id, node);

    // Link utterances to topics
    for (const turn of turns) {
      if (turn.start_ms >= topic.start_ms && turn.start_ms < topic.end_ms) {
        links.push({
          id: `topic_link_${i}_${turn.id}`,
          source: turn.id,
          target: node.id,
          edgeType: "during_topic",
          color: TOPIC_COLORS[i % TOPIC_COLORS.length],
          width: 0.5,
          dashed: true,
        });
      }
    }
  });

  // ── Signal nodes (only notable ones) ──
  const notableSignals = signals.filter((s) => {
    if (s.confidence < 0.3) return false;
    if (s.signal_type === "vocal_stress_score" && (s.value ?? 0) < 0.35) return false;
    if (s.signal_type === "tone_classification" && s.value_text === "neutral") return false;
    if (s.signal_type === "filler_detection" && s.value_text === "normal") return false;
    if (s.signal_type === "sentiment_score" && Math.abs(s.value ?? 0) < 0.3) return false;
    if (s.signal_type === "power_language_score") return false; // too noisy
    if (s.signal_type === "intent_classification") return false;
    return true;
  });

  const isFusion = (s: Signal) => s.agent === "fusion";

  notableSignals.forEach((s, i) => {
    const agentColor = AGENT_COLORS[s.agent] || "#888";
    const labelMap: Record<string, string> = {
      vocal_stress_score: `Stress ${Math.round((s.value ?? 0) * 100)}%`,
      buying_signal: "Buying Signal",
      objection_signal: "Objection",
      credibility_assessment: "Credibility",
      verbal_incongruence: "Incongruence",
      urgency_authenticity: "Urgency",
      sentiment_score: `Sentiment ${s.value_text}`,
      pitch_elevation_flag: "Pitch Elevation",
      speech_rate_anomaly: `Rate: ${s.value_text}`,
    };

    const node: GraphNode = {
      id: `sig_${i}`,
      type: isFusion(s) ? "insight" : "signal",
      label: labelMap[s.signal_type] || s.signal_type.replace(/_/g, " "),
      detail: `${s.signal_type}: ${s.value_text} (confidence: ${Math.round(s.confidence * 100)}%)`,
      speaker: s.speaker_label || undefined,
      color: agentColor,
      borderColor: agentColor,
      width: isFusion(s) ? 90 : 70,
      height: isFusion(s) ? 36 : 28,
      time_ms: s.window_start_ms,
      badges: [],
      confidence: s.confidence,
    };
    nodes.push(node);
    nodeMap.set(node.id, node);

    // Link to parent utterance
    const parentTurn = turns.find(
      (t) =>
        s.window_start_ms >= t.start_ms &&
        s.window_start_ms <= t.end_ms &&
        (!s.speaker_label || s.speaker_label === t.speaker)
    );
    if (parentTurn) {
      links.push({
        id: `sig_link_${i}`,
        source: parentTurn.id,
        target: node.id,
        edgeType: isFusion(s) ? "triggered" : "has_signal",
        color: agentColor,
        width: isFusion(s) ? 2.5 : 1,
        dashed: isFusion(s),
      });
    }
  });

  return { nodes, links };
}

// ═══════════════════════════════════════════
// SIGNAL NETWORK BUILDER (from backend graph)
// ═══════════════════════════════════════════

const EDGE_STYLES: Record<string, { color: string; dashed: boolean; width: number }> = {
  speaker_produced: { color: "#4F8BFF44", dashed: false, width: 1 },
  co_occurred:      { color: "#888866", dashed: true, width: 1 },
  preceded:         { color: "#6366F188", dashed: false, width: 1.5 },
  contradicts:      { color: "#EF4444", dashed: true, width: 2 },
  about_topic:      { color: "#10B98188", dashed: false, width: 1 },
  triggered:        { color: "#F97316", dashed: false, width: 2 },
  resolves:         { color: "#22C55E", dashed: false, width: 1.5 },
};

function buildSignalNetworkData(
  sg: SignalGraphData,
  speakerRoles: Record<string, string>,
): { nodes: GraphNode[]; links: GraphLink[] } {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const includedIds = new Set<string>();

  // Include all speakers and topics, top signals by confidence
  const ranked = [...sg.nodes].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  for (const n of ranked) {
    if (n.type === "speaker" || n.type === "topic" || n.type === "fusion_signal") {
      includedIds.add(n.id);
    }
  }
  // Top 15 voice + top 15 lang signals
  let v = 0, l = 0;
  for (const n of ranked) {
    if (n.type === "voice_signal" && v < 15) { includedIds.add(n.id); v++; }
    if (n.type === "lang_signal" && l < 15) { includedIds.add(n.id); l++; }
  }

  const typeColors: Record<string, string> = {
    speaker:       "#4F8BFF",
    topic:         "#8B5CF6",
    voice_signal:  "#6366F1",
    lang_signal:   "#06B6D4",
    fusion_signal: "#F97316",
  };

  for (const n of sg.nodes) {
    if (!includedIds.has(n.id)) continue;
    const spkIdx = Object.keys(speakerRoles).indexOf(n.speaker_id ?? "");
    const color = typeColors[n.type] ?? "#888";
    const isFusion = n.type === "fusion_signal";
    const isSpeaker = n.type === "speaker";
    const isTopic = n.type === "topic";

    nodes.push({
      id: n.id,
      type: isFusion ? "insight" : isSpeaker ? "speaker" : isTopic ? "topic" : "signal",
      label: n.label || n.signal_type?.replace(/_/g, " ") || n.id,
      detail: [
        n.type,
        n.signal_type,
        n.value != null ? `val=${n.value.toFixed(2)}` : null,
        n.confidence != null ? `conf=${Math.round(n.confidence * 100)}%` : null,
      ].filter(Boolean).join(" · "),
      speaker: n.speaker_id || undefined,
      color: isSpeaker ? SPEAKER_COLORS[spkIdx >= 0 ? spkIdx : 0] : color,
      borderColor: isSpeaker ? SPEAKER_COLORS[spkIdx >= 0 ? spkIdx : 0] : color,
      width: isSpeaker ? 80 : isFusion ? 90 : isTopic ? 70 : 60,
      height: isSpeaker ? 80 : isFusion ? 34 : isTopic ? 26 : 24,
      time_ms: n.timestamp_ms ?? 0,
      badges: [],
      confidence: n.confidence,
      pinned: isSpeaker,
    });
  }

  // Only include edges where both endpoints are included
  sg.edges.forEach((e, i) => {
    if (!includedIds.has(e.source) || !includedIds.has(e.target)) return;
    const style = EDGE_STYLES[e.relationship] ?? { color: "#88888866", dashed: true, width: 1 };
    links.push({
      id: `sne_${i}`,
      source: e.source,
      target: e.target,
      edgeType: e.relationship,
      color: style.color,
      width: style.width,
      dashed: style.dashed,
      label: e.relationship,
    });
  });

  return { nodes, links };
}

// ═══════════════════════════════════════════
// BADGE RENDERING
// ═══════════════════════════════════════════

const BADGE_COLORS: Record<string, string> = {
  stress: "#EF4444",
  buying: "#22C55E",
  objection: "#F59E0B",
  sentiment: "#8B5CF6",
  fusion: "#F97316",
};

// ═══════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════

export default function ConversationGraph({
  segments,
  signals,
  entities,
  speakerRoles,
  durationMs,
  onClose,
  signalGraph,
}: ConversationGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [simplified, setSimplified] = useState(false);
  const [timeRange, setTimeRange] = useState<[number, number]>([0, durationMs]);
  const [searchTerm, setSearchTerm] = useState("");
  const [showTopics, setShowTopics] = useState(true);
  const [showSignals, setShowSignals] = useState(true);
  const [viewMode, setViewMode] = useState<"conversation" | "signal_network">("conversation");

  const convData = useMemo(
    () => buildGraphData(segments, signals, entities, speakerRoles, durationMs, simplified),
    [segments, signals, entities, speakerRoles, durationMs, simplified]
  );

  const netData = useMemo(
    () => signalGraph ? buildSignalNetworkData(signalGraph, speakerRoles) : { nodes: [], links: [] },
    [signalGraph, speakerRoles]
  );

  const { nodes, links } = viewMode === "signal_network" && signalGraph ? netData : convData;

  // Filter by time range and search
  const visibleNodeIds = useMemo(() => {
    const ids = new Set<string>();
    for (const n of nodes) {
      if (n.type === "speaker") { ids.add(n.id); continue; }
      if (n.type === "topic" && !showTopics) continue;
      if ((n.type === "signal" || n.type === "insight") && !showSignals) continue;
      if (n.time_ms < timeRange[0] || n.time_ms > timeRange[1]) continue;
      if (searchTerm && !n.label.toLowerCase().includes(searchTerm.toLowerCase()) && !n.detail.toLowerCase().includes(searchTerm.toLowerCase())) continue;
      ids.add(n.id);
    }
    return ids;
  }, [nodes, timeRange, searchTerm, showTopics, showSignals]);

  // D3 rendering
  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = 650;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    // Defs for arrow markers
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L10,0L0,4")
      .attr("fill", "var(--border, #444)");

    // Glow filter for insights
    const filter = defs.append("filter").attr("id", "glow");
    filter.append("feGaussianBlur").attr("stdDeviation", 3).attr("result", "blur");
    filter.append("feMerge").selectAll("feMergeNode")
      .data(["blur", "SourceGraphic"]).enter()
      .append("feMergeNode").attr("in", (d) => d);

    const g = svg.append("g");

    // Zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    // Filter visible
    const activeNodes = nodes.filter((n) => visibleNodeIds.has(n.id));
    const activeLinks = links.filter(
      (l) =>
        visibleNodeIds.has(typeof l.source === "string" ? l.source : (l.source as GraphNode).id) &&
        visibleNodeIds.has(typeof l.target === "string" ? l.target : (l.target as GraphNode).id)
    );

    // Force simulation
    const simulation = d3.forceSimulation<GraphNode>(activeNodes)
      .force("link", d3.forceLink<GraphNode, GraphLink>(activeLinks).id((d) => d.id).distance(80).strength(0.3))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("y", d3.forceY<GraphNode>((d) => {
        if (d.type === "speaker") return 200;
        return 50 + (d.time_ms / Math.max(durationMs, 1)) * (height - 100);
      }).strength(0.15))
      .force("x", d3.forceX<GraphNode>((d) => {
        if (d.pinned) return d.fx ?? width / 2;
        if (d.type === "topic") return width - 100;
        const spkIdx = d.speaker ? segments.findIndex((s) => s.speaker_label === d.speaker) >= 0 ? (d.speaker === segments[0]?.speaker_label ? 0 : 1) : 0 : 0;
        return spkIdx === 0 ? width * 0.3 : width * 0.7;
      }).strength(0.08))
      .alphaDecay(0.03);

    // Links
    const linkSel = g.append("g").selectAll("line")
      .data(activeLinks)
      .join("line")
      .attr("stroke", (d) => d.color)
      .attr("stroke-width", (d) => d.width)
      .attr("stroke-dasharray", (d) => d.dashed ? "6 3" : "none")
      .attr("opacity", (d) => d.edgeType === "speaker_owns" || d.edgeType === "during_topic" ? 0.2 : 0.6)
      .attr("marker-end", (d) => d.edgeType === "response_to" || d.edgeType === "conversation_flow" ? "url(#arrow)" : "");

    // Node groups
    const nodeSel = g.append("g").selectAll<SVGGElement, GraphNode>("g")
      .data(activeNodes)
      .join("g")
      .attr("cursor", "pointer")
      .call(d3.drag<SVGGElement, GraphNode>()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.2).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          if (!d.pinned) { d.fx = null; d.fy = null; }
        })
      );

    // Render shapes based on type
    nodeSel.each(function (d) {
      const el = d3.select(this);

      if (d.type === "speaker") {
        el.append("circle")
          .attr("r", 38)
          .attr("fill", d.color)
          .attr("fill-opacity", 0.15)
          .attr("stroke", d.borderColor)
          .attr("stroke-width", 3);
        el.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", -4)
          .attr("font-size", 12)
          .attr("font-weight", 600)
          .attr("fill", "var(--text-primary, #E8E8E8)")
          .text(d.label);
        el.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", 12)
          .attr("font-size", 9)
          .attr("fill", "var(--text-muted, #888)")
          .text(d.detail);

      } else if (d.type === "utterance") {
        el.append("rect")
          .attr("x", -d.width / 2)
          .attr("y", -d.height / 2)
          .attr("width", d.width)
          .attr("height", d.height)
          .attr("rx", 8)
          .attr("fill", "var(--bg-surface, #1E1E2E)")
          .attr("stroke", d.borderColor)
          .attr("stroke-width", 2);
        el.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", 4)
          .attr("font-size", 10)
          .attr("fill", "var(--text-primary, #E8E8E8)")
          .text(d.label.length > 35 ? d.label.slice(0, 32) + "..." : d.label);
        // Badges
        d.badges.forEach((badge, bi) => {
          el.append("circle")
            .attr("cx", -d.width / 2 + 10 + bi * 12)
            .attr("cy", -d.height / 2 - 4)
            .attr("r", 4)
            .attr("fill", BADGE_COLORS[badge] || "#888");
        });

      } else if (d.type === "topic") {
        const pts = [
          [-d.width / 2, 0],
          [-d.width / 2 + 14, -d.height / 2],
          [d.width / 2 - 14, -d.height / 2],
          [d.width / 2, 0],
          [d.width / 2 - 14, d.height / 2],
          [-d.width / 2 + 14, d.height / 2],
        ].map((p) => p.join(",")).join(" ");
        el.append("polygon")
          .attr("points", pts)
          .attr("fill", d.color)
          .attr("fill-opacity", 0.2)
          .attr("stroke", d.color)
          .attr("stroke-width", 1.5);
        el.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", 4)
          .attr("font-size", 9)
          .attr("font-weight", 500)
          .attr("fill", d.color)
          .text(d.label);

      } else if (d.type === "insight") {
        el.append("rect")
          .attr("x", -d.width / 2)
          .attr("y", -d.height / 2)
          .attr("width", d.width)
          .attr("height", d.height)
          .attr("rx", 6)
          .attr("fill", "#F97316")
          .attr("fill-opacity", 0.2)
          .attr("stroke", "#F97316")
          .attr("stroke-width", 2)
          .attr("filter", "url(#glow)");
        el.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", 4)
          .attr("font-size", 9)
          .attr("font-weight", 600)
          .attr("fill", "#F97316")
          .text(d.label);

      } else {
        // signal
        el.append("circle")
          .attr("r", 14)
          .attr("fill", d.color)
          .attr("fill-opacity", 0.2)
          .attr("stroke", d.color)
          .attr("stroke-width", 1.5);
        el.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", 3)
          .attr("font-size", 7)
          .attr("fill", d.color)
          .text(d.label.length > 12 ? d.label.slice(0, 10) + ".." : d.label);
      }
    });

    // Click handler
    nodeSel.on("click", (_event, d) => {
      setSelectedNode((prev) => (prev?.id === d.id ? null : d));

      // Highlight connected
      const connectedIds = new Set<string>([d.id]);
      for (const l of activeLinks) {
        const src = typeof l.source === "string" ? l.source : (l.source as GraphNode).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as GraphNode).id;
        if (src === d.id) connectedIds.add(tgt);
        if (tgt === d.id) connectedIds.add(src);
      }
      nodeSel.attr("opacity", (n) => connectedIds.has(n.id) ? 1 : 0.15);
      linkSel.attr("opacity", (l) => {
        const src = typeof l.source === "string" ? l.source : (l.source as GraphNode).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as GraphNode).id;
        return connectedIds.has(src) && connectedIds.has(tgt) ? 0.8 : 0.05;
      });
    });

    // Click background to deselect
    svg.on("click", (event) => {
      if (event.target === svgRef.current) {
        setSelectedNode(null);
        nodeSel.attr("opacity", 1);
        linkSel.attr("opacity", (d) =>
          d.edgeType === "speaker_owns" || d.edgeType === "during_topic" ? 0.2 : 0.6
        );
      }
    });

    // Tooltip on hover
    nodeSel.append("title").text((d) => d.detail);

    // Tick
    simulation.on("tick", () => {
      linkSel
        .attr("x1", (d) => ((d.source as GraphNode).x ?? 0))
        .attr("y1", (d) => ((d.source as GraphNode).y ?? 0))
        .attr("x2", (d) => ((d.target as GraphNode).x ?? 0))
        .attr("y2", (d) => ((d.target as GraphNode).y ?? 0));
      nodeSel.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => { simulation.stop(); };
  }, [nodes, links, visibleNodeIds, durationMs, segments]);

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-bg overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-nexus-border bg-nexus-surface flex-wrap">
        <span className="text-sm font-medium text-nexus-text-primary mr-auto">
          {viewMode === "signal_network" ? "Signal Relationship Network" : "Conversation Graph"}
          <span className="ml-2 text-[10px] text-nexus-text-muted font-normal">
            {nodes.length} nodes · {links.length} edges
          </span>
        </span>

        {/* View mode toggle */}
        {signalGraph && (
          <div className="flex rounded border border-nexus-border overflow-hidden text-[10px]">
            <button
              onClick={() => setViewMode("conversation")}
              className={`px-2 py-0.5 transition-colors ${viewMode === "conversation" ? "bg-nexus-blue text-white" : "bg-nexus-surface text-nexus-text-secondary hover:bg-nexus-surface-hover"}`}
            >
              Conversation
            </button>
            <button
              onClick={() => setViewMode("signal_network")}
              className={`px-2 py-0.5 transition-colors ${viewMode === "signal_network" ? "bg-nexus-blue text-white" : "bg-nexus-surface text-nexus-text-secondary hover:bg-nexus-surface-hover"}`}
            >
              Signal Network
            </button>
          </div>
        )}

        {viewMode === "conversation" && (<>
        <label className="flex items-center gap-1 text-[10px] text-nexus-text-secondary cursor-pointer">
          <input type="checkbox" checked={showTopics} onChange={(e) => setShowTopics(e.target.checked)} className="h-3 w-3" />
          Topics
        </label>
        <label className="flex items-center gap-1 text-[10px] text-nexus-text-secondary cursor-pointer">
          <input type="checkbox" checked={showSignals} onChange={(e) => setShowSignals(e.target.checked)} className="h-3 w-3" />
          Signals
        </label>
        <label className="flex items-center gap-1 text-[10px] text-nexus-text-secondary cursor-pointer">
          <input type="checkbox" checked={simplified} onChange={(e) => setSimplified(e.target.checked)} className="h-3 w-3" />
          Simplified
        </label>
        </>)}

        <input
          type="text"
          placeholder="Search..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="rounded border border-nexus-border bg-nexus-bg px-2 py-0.5 text-[11px] text-nexus-text-primary w-32"
        />

        <button
          onClick={onClose}
          className="rounded px-2 py-0.5 text-[11px] text-nexus-text-muted hover:text-nexus-text-primary hover:bg-nexus-surface-hover"
        >
          Close
        </button>
      </div>

      {/* Graph */}
      <div ref={containerRef} className="relative" style={{ height: 650 }}>
        <svg ref={svgRef} className="w-full h-full" />
      </div>

      {/* Signal Network legend */}
      {viewMode === "signal_network" && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 px-4 py-2 border-t border-nexus-border bg-nexus-surface">
          {[
            { color: "#4F8BFF", label: "Speaker" },
            { color: "#8B5CF6", label: "Topic" },
            { color: "#6366F1", label: "Voice Signal" },
            { color: "#06B6D4", label: "Language Signal" },
            { color: "#F97316", label: "Fusion Insight" },
          ].map(({ color, label }) => (
            <span key={label} className="flex items-center gap-1 text-[9px] text-nexus-text-secondary">
              <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
              {label}
            </span>
          ))}
          <span className="text-[9px] text-nexus-text-muted ml-auto">
            Red dashed = contradicts · Orange = triggered · Green = resolves
          </span>
        </div>
      )}

      {/* Timeline scrubber */}
      {viewMode === "conversation" && (
      <div className="px-4 py-2 border-t border-nexus-border bg-nexus-surface">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-nexus-text-muted w-10">
            {Math.floor(timeRange[0] / 1000)}s
          </span>
          <input
            type="range"
            min={0}
            max={durationMs}
            step={1000}
            value={timeRange[1]}
            onChange={(e) => setTimeRange([0, Number(e.target.value)])}
            className="flex-1 h-1 accent-blue-500"
          />
          <span className="text-[10px] text-nexus-text-muted w-10 text-right">
            {Math.floor(timeRange[1] / 1000)}s
          </span>
        </div>
      </div>
      )}

      {/* Detail panel */}
      {selectedNode && (
        <div className="border-t border-nexus-border bg-nexus-surface px-4 py-3">
          <div className="flex items-start gap-3">
            <span
              className="mt-1 h-3 w-3 rounded-full shrink-0"
              style={{ backgroundColor: selectedNode.borderColor || selectedNode.color }}
            />
            <div className="min-w-0">
              <div className="text-xs font-medium text-nexus-text-primary">
                {selectedNode.type === "utterance" ? selectedNode.detail : selectedNode.label}
              </div>
              {selectedNode.type !== "utterance" && (
                <div className="text-[10px] text-nexus-text-muted mt-0.5">
                  {selectedNode.detail}
                </div>
              )}
              {selectedNode.badges.length > 0 && (
                <div className="flex gap-1 mt-1">
                  {selectedNode.badges.map((b, i) => (
                    <span
                      key={i}
                      className="rounded-full px-1.5 py-0.5 text-[9px] text-white"
                      style={{ backgroundColor: BADGE_COLORS[b] || "#888" }}
                    >
                      {b}
                    </span>
                  ))}
                </div>
              )}
              {selectedNode.confidence != null && (
                <div className="text-[10px] text-nexus-text-muted mt-0.5">
                  Confidence: {Math.round(selectedNode.confidence * 100)}%
                </div>
              )}
              <div className="text-[10px] text-nexus-text-muted mt-0.5">
                Time: {Math.floor(selectedNode.time_ms / 1000)}s
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
