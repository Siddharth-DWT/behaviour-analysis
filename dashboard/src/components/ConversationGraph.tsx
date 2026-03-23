import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import type { Signal, TranscriptSegment } from "../api/client";
import { SignalNetworkView } from "./signal-network";
import type { SignalGraphData } from "./signal-network";

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
  agent?: string;
  signalType?: string;
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

interface TooltipData {
  node?: GraphNode;
  x: number;
  y: number;
  connections?: Array<{ label: string; type: string; edgeType: string }>;
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

const SPEAKER_COLORS = ["#4F8BFF", "#F59E0B", "#8B5CF6", "#10B981"];

const AGENT_COLORS: Record<string, string> = {
  voice: "#6366F1",
  language: "#06B6D4",
  fusion: "#F97316",
};

const TOPIC_COLORS = [
  "#4F8BFF", "#8B5CF6", "#10B981", "#F59E0B", "#EC4899", "#6366F1", "#22C55E",
];

const BADGE_COLORS: Record<string, string> = {
  stress: "#EF4444",
  buying: "#22C55E",
  objection: "#F59E0B",
  sentiment: "#8B5CF6",
  fusion: "#F97316",
};

const SIGNAL_LABELS: Record<string, string> = {
  vocal_stress_score: "Stress",
  buying_signal: "Buying Signal",
  objection_signal: "Objection",
  sentiment_score: "Sentiment",
  credibility_assessment: "Credibility",
  verbal_incongruence: "Incongruence",
  urgency_authenticity: "Urgency",
  pitch_elevation_flag: "Pitch",
  speech_rate_anomaly: "Rate",
  filler_detection: "Filler",
  tone_classification: "Tone",
  persuasion_indicator: "Persuasion",
};

function formatTime(ms: number): string {
  const sec = Math.floor(ms / 1000);
  return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
}

// ═══════════════════════════════════════════
// DATA BUILDER
// ═══════════════════════════════════════════

interface ConvFilters {
  speakers: Set<string>;
  agents: Set<string>;
  showTopics: boolean;
  showSignals: boolean;
  showUtterances: boolean;
  confidenceMin: number;
  searchTerm: string;
  simplified: boolean;
}

interface Turn {
  id: string;
  speaker: string;
  start_ms: number;
  end_ms: number;
  text: string;
  segIndices: number[];
  wordCount: number;
}

interface ConvStats {
  totalTurns: number;
  turnsPerSpeaker: Record<string, number>;
  wordsPerSpeaker: Record<string, number>;
  signalCounts: Record<string, number>;
  signalsByAgent: Record<string, number>;
  topicCount: number;
  avgTurnLength: number;
  crossSpeakerCount: number;
}

function buildConvData(
  segments: TranscriptSegment[],
  signals: Signal[],
  entities: Entities,
  speakerRoles: Record<string, string>,
  durationMs: number,
  filters: ConvFilters,
): { nodes: GraphNode[]; links: GraphLink[]; turns: Turn[]; stats: ConvStats } {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];

  // ── Aggregate segments into turns ──
  const turns: Turn[] = [];
  let currentTurn: Turn | null = null;
  for (const seg of segments) {
    const speaker = seg.speaker_label || "Unknown";
    if (currentTurn && currentTurn.speaker === speaker && currentTurn.segIndices.length < 4) {
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

  const speakerLabels = [...new Set(turns.map(t => t.speaker))];

  // ── Stats ──
  const turnsPerSpeaker: Record<string, number> = {};
  const wordsPerSpeaker: Record<string, number> = {};
  for (const t of turns) {
    turnsPerSpeaker[t.speaker] = (turnsPerSpeaker[t.speaker] ?? 0) + 1;
    wordsPerSpeaker[t.speaker] = (wordsPerSpeaker[t.speaker] ?? 0) + t.wordCount;
  }
  const signalCounts: Record<string, number> = {};
  const signalsByAgent: Record<string, number> = {};
  for (const s of signals) {
    const st = SIGNAL_LABELS[s.signal_type] || s.signal_type;
    signalCounts[st] = (signalCounts[st] ?? 0) + 1;
    signalsByAgent[s.agent] = (signalsByAgent[s.agent] ?? 0) + 1;
  }
  let crossSpeakerCount = 0;
  for (let i = 1; i < turns.length; i++) {
    if (turns[i].speaker !== turns[i - 1].speaker) crossSpeakerCount++;
  }
  const stats: ConvStats = {
    totalTurns: turns.length,
    turnsPerSpeaker,
    wordsPerSpeaker,
    signalCounts,
    signalsByAgent,
    topicCount: entities.topics?.length ?? 0,
    avgTurnLength: turns.length > 0 ? Math.round(turns.reduce((s, t) => s + t.wordCount, 0) / turns.length) : 0,
    crossSpeakerCount,
  };

  // ── Speaker nodes ──
  speakerLabels.forEach((label, i) => {
    if (filters.speakers.size > 0 && !filters.speakers.has(label)) return;
    const role = speakerRoles[label] || "";
    const name = entities.people?.find(
      p => p.speaker_label === label || p.role?.toLowerCase() === role?.toLowerCase()
    )?.name;
    nodes.push({
      id: `spk_${label}`,
      type: "speaker",
      label: name || role || label,
      detail: `${role} (${label}) · ${turnsPerSpeaker[label] ?? 0} turns · ${wordsPerSpeaker[label] ?? 0} words`,
      speaker: label,
      color: SPEAKER_COLORS[i % 4],
      borderColor: SPEAKER_COLORS[i % 4],
      width: 90,
      height: 90,
      time_ms: 0,
      badges: [],
      pinned: true,
    });
  });

  // ── Utterance nodes ──
  if (filters.showUtterances) {
    turns.forEach((turn, i) => {
      if (filters.speakers.size > 0 && !filters.speakers.has(turn.speaker)) return;
      const spkIdx = speakerLabels.indexOf(turn.speaker);
      const truncText = turn.text.length > 60 ? turn.text.slice(0, 57) + "..." : turn.text;
      const w = Math.max(120, Math.min(240, 90 + turn.wordCount * 3));

      // Badges from signals
      const badges: string[] = [];
      for (const s of signals) {
        if (s.window_start_ms > turn.end_ms || s.window_end_ms < turn.start_ms) continue;
        if (s.speaker_label && s.speaker_label !== turn.speaker) continue;
        if (s.signal_type === "vocal_stress_score" && (s.value ?? 0) > 0.3) badges.push("stress");
        if (s.signal_type === "buying_signal") badges.push("buying");
        if (s.signal_type === "objection_signal") badges.push("objection");
        if (s.signal_type === "sentiment_score" && Math.abs(s.value ?? 0) > 0.35) badges.push("sentiment");
        if (s.agent === "fusion") badges.push("fusion");
      }

      nodes.push({
        id: turn.id,
        type: "utterance",
        label: truncText,
        detail: turn.text,
        speaker: turn.speaker,
        color: "var(--bg-surface, #1E1E2E)",
        borderColor: SPEAKER_COLORS[spkIdx % 4],
        width: w,
        height: 44,
        time_ms: turn.start_ms,
        badges: [...new Set(badges)],
      });

      // Link to speaker
      if (nodes.find(n => n.id === `spk_${turn.speaker}`)) {
        links.push({
          id: `spk_link_${i}`,
          source: `spk_${turn.speaker}`,
          target: turn.id,
          edgeType: "speaker_owns",
          color: "transparent",
          width: 0,
          dashed: false,
        });
      }
    });

    // ── Flow edges ──
    const visibleTurnIds = new Set(nodes.filter(n => n.type === "utterance").map(n => n.id));
    for (let i = 1; i < turns.length; i++) {
      if (!visibleTurnIds.has(turns[i].id) || !visibleTurnIds.has(turns[i - 1].id)) continue;
      const prev = turns[i - 1];
      const curr = turns[i];
      const cross = prev.speaker !== curr.speaker;
      const latency = curr.start_ms - prev.end_ms;
      links.push({
        id: `flow_${i}`,
        source: prev.id,
        target: curr.id,
        edgeType: cross ? "response_to" : "conversation_flow",
        color: cross ? "#4F8BFF" : "#4B5563",
        width: cross ? 2 : 1,
        dashed: false,
        label: cross && latency > 0 ? `${latency}ms` : undefined,
      });
    }
  }

  if (filters.simplified) return { nodes, links, turns, stats };

  // ── Topic nodes ──
  if (filters.showTopics) {
    (entities.topics ?? []).forEach((topic, i) => {
      nodes.push({
        id: `topic_${i}`,
        type: "topic",
        label: topic.name,
        detail: `${formatTime(topic.start_ms)} – ${formatTime(topic.end_ms)}`,
        color: TOPIC_COLORS[i % TOPIC_COLORS.length],
        borderColor: TOPIC_COLORS[i % TOPIC_COLORS.length],
        width: 110,
        height: 34,
        time_ms: topic.start_ms,
        badges: [],
      });
      // Link utterances to topics
      if (filters.showUtterances) {
        for (const turn of turns) {
          if (turn.start_ms >= topic.start_ms && turn.start_ms < topic.end_ms) {
            if (nodes.find(n => n.id === turn.id)) {
              links.push({
                id: `topic_link_${i}_${turn.id}`,
                source: turn.id,
                target: `topic_${i}`,
                edgeType: "during_topic",
                color: TOPIC_COLORS[i % TOPIC_COLORS.length],
                width: 0.5,
                dashed: true,
              });
            }
          }
        }
      }
    });
  }

  // ── Signal nodes ──
  if (filters.showSignals) {
    const notableSignals = signals.filter(s => {
      if (s.confidence < filters.confidenceMin) return false;
      if (filters.agents.size > 0 && !filters.agents.has(s.agent)) return false;
      if (filters.speakers.size > 0 && s.speaker_label && !filters.speakers.has(s.speaker_label)) return false;
      if (filters.searchTerm) {
        const q = filters.searchTerm.toLowerCase();
        const hay = `${s.signal_type} ${s.value_text ?? ""} ${SIGNAL_LABELS[s.signal_type] ?? ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      // Filter out noise
      if (s.signal_type === "vocal_stress_score" && (s.value ?? 0) < 0.3) return false;
      if (s.signal_type === "tone_classification" && s.value_text === "neutral") return false;
      if (s.signal_type === "filler_detection" && s.value_text === "normal") return false;
      if (s.signal_type === "sentiment_score" && Math.abs(s.value ?? 0) < 0.25) return false;
      if (s.signal_type === "power_language_score") return false;
      if (s.signal_type === "intent_classification") return false;
      return true;
    });

    const isFusion = (s: Signal) => s.agent === "fusion";

    notableSignals.forEach((s, i) => {
      const agentColor = AGENT_COLORS[s.agent] || "#888";
      const base = SIGNAL_LABELS[s.signal_type] || s.signal_type.replace(/_/g, " ");
      let label = base;
      if (s.value != null && s.signal_type.includes("score")) label = `${base} ${Math.round(s.value * 100)}%`;
      else if (s.value_text && s.value_text !== "normal") label = `${base}: ${s.value_text}`;

      nodes.push({
        id: `sig_${i}`,
        type: isFusion(s) ? "insight" : "signal",
        label,
        detail: `${s.signal_type}: ${s.value_text ?? ""} (confidence: ${Math.round(s.confidence * 100)}%)`,
        speaker: s.speaker_label || undefined,
        agent: s.agent,
        signalType: s.signal_type,
        color: agentColor,
        borderColor: agentColor,
        width: isFusion(s) ? 130 : 110,
        height: isFusion(s) ? 36 : 30,
        time_ms: s.window_start_ms,
        badges: [],
        confidence: s.confidence,
      });

      // Link to parent utterance
      if (filters.showUtterances) {
        const parentTurn = turns.find(t =>
          s.window_start_ms >= t.start_ms && s.window_start_ms <= t.end_ms &&
          (!s.speaker_label || s.speaker_label === t.speaker)
        );
        if (parentTurn && nodes.find(n => n.id === parentTurn.id)) {
          links.push({
            id: `sig_link_${i}`,
            source: parentTurn.id,
            target: `sig_${i}`,
            edgeType: isFusion(s) ? "triggered" : "has_signal",
            color: agentColor,
            width: isFusion(s) ? 2 : 1,
            dashed: isFusion(s),
          });
        }
      }
    });
  }

  return { nodes, links, turns, stats };
}

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
  const [viewMode, setViewMode] = useState<"conversation" | "signal_network">("conversation");

  // ── Delegate to Signal Network view ──
  if (viewMode === "signal_network" && signalGraph) {
    return (
      <div>
        <div className="flex items-center gap-2 px-3 py-1.5 border border-nexus-border rounded-t-lg bg-nexus-surface">
          <div className="flex rounded border border-nexus-border overflow-hidden text-[10px]">
            <button onClick={() => setViewMode("conversation")}
              className="px-2 py-0.5 bg-nexus-surface text-nexus-text-secondary hover:bg-nexus-surface-hover">
              Conversation
            </button>
            <button className="px-2 py-0.5 bg-nexus-blue text-white">Signal Network</button>
          </div>
        </div>
        <SignalNetworkView signalGraph={signalGraph} speakerRoles={speakerRoles} onClose={onClose} />
      </div>
    );
  }

  // ═══ CONVERSATION FLOW VIEW ═══
  return <ConversationFlowView
    segments={segments} signals={signals} entities={entities}
    speakerRoles={speakerRoles} durationMs={durationMs}
    onClose={onClose} signalGraph={signalGraph}
    onSwitchToNetwork={() => setViewMode("signal_network")}
  />;
}

// ═══════════════════════════════════════════
// CONVERSATION FLOW VIEW (inner component)
// ═══════════════════════════════════════════

function ConversationFlowView({
  segments, signals, entities, speakerRoles, durationMs,
  onClose, signalGraph, onSwitchToNetwork,
}: ConversationGraphProps & { onSwitchToNetwork: () => void }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // ── State ──
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [showFilters, setShowFilters] = useState(true);
  const [showStats, setShowStats] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [timeRange, setTimeRange] = useState<[number, number]>([0, durationMs]);

  const allSpeakers = useMemo(() => {
    const s = new Set<string>();
    segments.forEach(seg => { if (seg.speaker_label) s.add(seg.speaker_label); });
    return [...s];
  }, [segments]);

  const [filters, setFilters] = useState<ConvFilters>({
    speakers: new Set<string>(),
    agents: new Set(["voice", "language", "fusion"]),
    showTopics: true,
    showSignals: true,
    showUtterances: true,
    confidenceMin: 0.2,
    searchTerm: "",
    simplified: false,
  });

  const updateFilter = useCallback(<K extends keyof ConvFilters>(key: K, val: ConvFilters[K]) => {
    setFilters(f => ({ ...f, [key]: val }));
  }, []);

  const toggleSet = useCallback(<T extends string>(prev: Set<T>, val: T): Set<T> => {
    const next = new Set(prev);
    if (next.has(val)) next.delete(val); else next.add(val);
    return next;
  }, []);

  // ── Build data ──
  const { nodes: allNodes, links: allLinks, stats } = useMemo(
    () => buildConvData(segments, signals, entities, speakerRoles, durationMs, filters),
    [segments, signals, entities, speakerRoles, durationMs, filters]
  );

  // Time range filter
  const nodes = useMemo(() => allNodes.filter(n => {
    if (n.type === "speaker") return true;
    if (n.time_ms < timeRange[0] || n.time_ms > timeRange[1]) return false;
    if (filters.searchTerm && n.type === "utterance") {
      const q = filters.searchTerm.toLowerCase();
      if (!n.label.toLowerCase().includes(q) && !n.detail.toLowerCase().includes(q)) return false;
    }
    return true;
  }), [allNodes, timeRange, filters.searchTerm]);

  const nodeIds = useMemo(() => new Set(nodes.map(n => n.id)), [nodes]);
  const links = useMemo(() => allLinks.filter(l => {
    const src = typeof l.source === "string" ? l.source : (l.source as GraphNode).id;
    const tgt = typeof l.target === "string" ? l.target : (l.target as GraphNode).id;
    return nodeIds.has(src) && nodeIds.has(tgt);
  }), [allLinks, nodeIds]);

  // ── Fullscreen ──
  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current?.closest(".conv-graph-root") as HTMLElement | null;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  }, []);

  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  // ── PNG Export ──
  const exportPng = useCallback(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.querySelectorAll("*").forEach(el => {
      const cs = window.getComputedStyle(el as Element);
      (el as HTMLElement).style.cssText = Array.from(cs).map(k => `${k}:${cs.getPropertyValue(k)}`).join(";");
    });
    const xml = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([xml], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth * 2;
      canvas.height = img.naturalHeight * 2;
      const ctx = canvas.getContext("2d")!;
      ctx.scale(2, 2);
      ctx.fillStyle = "#0F0F17";
      ctx.fillRect(0, 0, img.naturalWidth, img.naturalHeight);
      ctx.drawImage(img, 0, 0);
      canvas.toBlob(b => {
        if (!b) return;
        const a = document.createElement("a");
        a.href = URL.createObjectURL(b);
        a.download = `conversation-graph-${Date.now()}.png`;
        a.click();
      });
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }, []);

  // ═══════════════════════════════════════════
  // D3 RENDERING
  // ═══════════════════════════════════════════

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = isFullscreen ? window.innerHeight - 120 : 680;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    // ── Defs ──
    const defs = svg.append("defs");

    // Arrows
    ["#4F8BFF", "#4B5563", "#F97316"].forEach((col, i) => {
      defs.append("marker")
        .attr("id", `arrow-conv-${i}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 22).attr("markerWidth", 5).attr("markerHeight", 5)
        .attr("orient", "auto")
        .append("path").attr("d", "M0,-4L10,0L0,4").attr("fill", col);
    });

    // Glow
    const glow = defs.append("filter").attr("id", "conv-glow").attr("x", "-30%").attr("y", "-30%").attr("width", "160%").attr("height", "160%");
    glow.append("feGaussianBlur").attr("stdDeviation", 4).attr("result", "blur");
    glow.append("feMerge").selectAll("feMergeNode").data(["blur", "SourceGraphic"]).enter().append("feMergeNode").attr("in", d => d);

    // Shadow
    const shadow = defs.append("filter").attr("id", "conv-shadow").attr("x", "-10%").attr("y", "-10%").attr("width", "130%").attr("height", "130%");
    shadow.append("feDropShadow").attr("dx", 0).attr("dy", 2).attr("stdDeviation", 3).attr("flood-opacity", 0.25);

    const g = svg.append("g");

    // Zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.15, 4])
      .on("zoom", event => g.attr("transform", event.transform));
    svg.call(zoom);

    // ── Force simulation ──
    const speakerLabels = [...new Set(segments.map(s => s.speaker_label || "Unknown"))];

    const simulation = d3.forceSimulation<GraphNode>(nodes)
      .force("link",
        d3.forceLink<GraphNode, GraphLink>(links).id(d => d.id)
          .distance(d => {
            if (d.edgeType === "speaker_owns") return 60;
            if (d.edgeType === "during_topic") return 120;
            if (d.edgeType === "has_signal" || d.edgeType === "triggered") return 50;
            return 70;
          })
          .strength(d => {
            if (d.edgeType === "speaker_owns") return 0.2;
            if (d.edgeType === "during_topic") return 0.02;
            return 0.15;
          })
      )
      .force("charge", d3.forceManyBody<GraphNode>()
        .strength(d => d.type === "speaker" ? -350 : d.type === "topic" ? -150 : -80)
      )
      .force("center", d3.forceCenter(width / 2, height / 2).strength(0.04))
      .force("collision", d3.forceCollide<GraphNode>().radius(d => Math.max(d.width, d.height) / 2 + 3))
      .force("y", d3.forceY<GraphNode>(d => {
        if (d.type === "speaker") return height / 2;
        return 50 + (d.time_ms / Math.max(durationMs, 1)) * (height - 100);
      }).strength(0.1))
      .force("x", d3.forceX<GraphNode>(d => {
        if (d.type === "topic") return width - 80;
        if (d.type === "speaker") {
          const idx = speakerLabels.indexOf(d.speaker ?? "");
          return idx === 0 ? 80 : idx === 1 ? width - 80 : width / 2;
        }
        const spkIdx = d.speaker ? (d.speaker === speakerLabels[0] ? 0 : 1) : 0;
        return spkIdx === 0 ? width * 0.3 : width * 0.7;
      }).strength(0.06))
      .velocityDecay(0.45)
      .alphaDecay(0.025);

    // Pin speakers
    for (const n of nodes) {
      if (n.type === "speaker") {
        const idx = speakerLabels.indexOf(n.speaker ?? "");
        n.fx = idx === 0 ? 80 : idx === 1 ? width - 80 : width / 2;
        n.fy = height / 2;
      }
    }

    // ── Edges (curved) ──
    const edgeSel = g.append("g").selectAll<SVGPathElement, GraphLink>("path")
      .data(links)
      .join("path")
      .attr("fill", "none")
      .attr("stroke", d => d.color)
      .attr("stroke-width", d => d.width)
      .attr("stroke-dasharray", d => d.dashed ? "6 3" : "none")
      .attr("opacity", d => {
        if (d.edgeType === "speaker_owns") return 0;
        if (d.edgeType === "during_topic") return 0.15;
        return 0.5;
      })
      .attr("marker-end", d => {
        if (d.edgeType === "response_to") return "url(#arrow-conv-0)";
        if (d.edgeType === "conversation_flow") return "url(#arrow-conv-1)";
        if (d.edgeType === "triggered") return "url(#arrow-conv-2)";
        return "";
      });

    // ── Nodes ──
    const nodeSel = g.append("g").selectAll<SVGGElement, GraphNode>("g")
      .data(nodes)
      .join("g")
      .attr("cursor", "pointer")
      .call(d3.drag<SVGGElement, GraphNode>()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.15).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          if (!d.pinned) { d.fx = null; d.fy = null; }
        })
      );

    // ── Render node shapes ──
    nodeSel.each(function (d) {
      const el = d3.select(this);

      if (d.type === "speaker") {
        el.append("circle").attr("r", 42)
          .attr("fill", d.color).attr("fill-opacity", 0.12)
          .attr("stroke", d.color).attr("stroke-width", 3)
          .attr("filter", "url(#conv-shadow)");
        el.append("text").attr("text-anchor", "middle").attr("dy", -6)
          .attr("font-size", 13).attr("font-weight", 700).attr("fill", d.color)
          .text(d.label);
        el.append("text").attr("text-anchor", "middle").attr("dy", 12)
          .attr("font-size", 9).attr("fill", "#9CA3AF")
          .text(speakerRoles[d.speaker ?? ""] || d.speaker || "");
        // Turn count badge
        const tc = stats.turnsPerSpeaker[d.speaker ?? ""] ?? 0;
        el.append("circle").attr("cx", 30).attr("cy", -30).attr("r", 10)
          .attr("fill", d.color).attr("fill-opacity", 0.8);
        el.append("text").attr("x", 30).attr("y", -30).attr("dy", 3)
          .attr("text-anchor", "middle").attr("font-size", 8).attr("font-weight", 700)
          .attr("fill", "#fff").text(String(tc));

      } else if (d.type === "utterance") {
        // Rounded rect with speaker color left border
        el.append("rect")
          .attr("x", -d.width / 2).attr("y", -d.height / 2)
          .attr("width", d.width).attr("height", d.height)
          .attr("rx", 8)
          .attr("fill", "var(--bg-surface, #1A1A2E)")
          .attr("stroke", d.borderColor).attr("stroke-width", 1.5)
          .attr("filter", "url(#conv-shadow)");
        // Left color bar
        el.append("rect")
          .attr("x", -d.width / 2).attr("y", -d.height / 2 + 4)
          .attr("width", 4).attr("height", d.height - 8)
          .attr("rx", 2)
          .attr("fill", d.borderColor);
        // Time badge
        el.append("text")
          .attr("x", -d.width / 2 + 12).attr("y", -d.height / 2 + 12)
          .attr("font-size", 8).attr("fill", "#6B7280")
          .text(formatTime(d.time_ms));
        // Text
        const maxChars = Math.floor((d.width - 24) / 5.5);
        el.append("text")
          .attr("x", -d.width / 2 + 12).attr("dy", 5)
          .attr("font-size", 10).attr("fill", "var(--text-primary, #E8E8E8)")
          .text(d.label.length > maxChars ? d.label.slice(0, maxChars - 2) + ".." : d.label);
        // Badges
        d.badges.forEach((badge, bi) => {
          el.append("circle")
            .attr("cx", d.width / 2 - 8 - bi * 11).attr("cy", -d.height / 2 - 4)
            .attr("r", 5)
            .attr("fill", BADGE_COLORS[badge] || "#888")
            .attr("stroke", "var(--bg-surface, #1A1A2E)").attr("stroke-width", 1.5);
        });

      } else if (d.type === "topic") {
        const w = d.width / 2, h = d.height / 2;
        const pts = [[-w, 0], [-w + 14, -h], [w - 14, -h], [w, 0], [w - 14, h], [-w + 14, h]]
          .map(p => p.join(",")).join(" ");
        el.append("polygon").attr("points", pts)
          .attr("fill", d.color).attr("fill-opacity", 0.18)
          .attr("stroke", d.color).attr("stroke-width", 1.5);
        el.append("text").attr("text-anchor", "middle").attr("dy", 4)
          .attr("font-size", 10).attr("font-weight", 600).attr("fill", d.color)
          .text(d.label.length > 16 ? d.label.slice(0, 14) + ".." : d.label);

      } else if (d.type === "insight") {
        el.append("rect")
          .attr("x", -d.width / 2).attr("y", -d.height / 2)
          .attr("width", d.width).attr("height", d.height).attr("rx", 8)
          .attr("fill", "#F97316").attr("fill-opacity", 0.15)
          .attr("stroke", "#F97316").attr("stroke-width", 2)
          .attr("filter", "url(#conv-glow)");
        el.append("text").attr("x", -d.width / 2 + 12).attr("dy", 4)
          .attr("font-size", 12).text("⚡");
        el.append("text").attr("x", -d.width / 2 + 28).attr("dy", 4)
          .attr("font-size", 10).attr("font-weight", 600).attr("fill", "#F97316")
          .text(d.label.length > 16 ? d.label.slice(0, 14) + ".." : d.label);
        if (d.confidence != null) {
          const pw = 30, px = d.width / 2 - pw - 6;
          el.append("rect").attr("x", px).attr("y", -7).attr("width", pw).attr("height", 14).attr("rx", 7)
            .attr("fill", "#F97316").attr("fill-opacity", 0.3);
          el.append("text").attr("x", px + pw / 2).attr("dy", 3).attr("text-anchor", "middle")
            .attr("font-size", 8).attr("font-weight", 600).attr("fill", "#F97316")
            .text(`${Math.round(d.confidence * 100)}%`);
        }

      } else {
        // Signal — rounded rect with agent icon
        const isVoice = d.agent === "voice";
        const col = d.color;
        el.append("rect")
          .attr("x", -d.width / 2).attr("y", -d.height / 2)
          .attr("width", d.width).attr("height", d.height).attr("rx", 6)
          .attr("fill", col).attr("fill-opacity", 0.1)
          .attr("stroke", col).attr("stroke-width", 1.5);
        // Agent icon
        el.append("circle").attr("cx", -d.width / 2 + 12).attr("cy", 0).attr("r", 5)
          .attr("fill", col).attr("fill-opacity", 0.6);
        el.append("text").attr("x", -d.width / 2 + 12).attr("dy", 3).attr("text-anchor", "middle")
          .attr("font-size", 7).attr("fill", "#fff")
          .text(isVoice ? "V" : d.agent === "language" ? "L" : "F");
        // Label
        const maxW = d.width - 56;
        el.append("text").attr("x", -d.width / 2 + 24).attr("dy", 4)
          .attr("font-size", 9).attr("font-weight", 500).attr("fill", col)
          .text(d.label.length > maxW / 5 ? d.label.slice(0, Math.floor(maxW / 5)) + ".." : d.label);
        // Confidence pill
        if (d.confidence != null) {
          const pw = 28, px = d.width / 2 - pw - 4;
          el.append("rect").attr("x", px).attr("y", -6).attr("width", pw).attr("height", 12).attr("rx", 6)
            .attr("fill", col).attr("fill-opacity", Math.min(0.5, d.confidence));
          el.append("text").attr("x", px + pw / 2).attr("dy", 3).attr("text-anchor", "middle")
            .attr("font-size", 7).attr("font-weight", 600).attr("fill", col)
            .text(`${Math.round(d.confidence * 100)}%`);
        }
      }
    });

    // ── Interactions ──
    nodeSel
      .on("mouseenter", function (event, d) {
        d3.select(this).raise();
        const conns: TooltipData["connections"] = [];
        for (const l of links) {
          const src = typeof l.source === "string" ? l.source : (l.source as GraphNode).id;
          const tgt = typeof l.target === "string" ? l.target : (l.target as GraphNode).id;
          if (src === d.id || tgt === d.id) {
            const other = nodes.find(n => n.id === (src === d.id ? tgt : src));
            if (other && other.type !== "speaker") {
              conns.push({ label: other.label, type: other.type, edgeType: l.edgeType });
            }
          }
        }
        setTooltip({ node: d, x: event.clientX, y: event.clientY, connections: conns.slice(0, 6) });
      })
      .on("mouseleave", () => setTooltip(null))
      .on("click", (_, d) => {
        setSelectedNodeId(prev => prev === d.id ? null : d.id);
      });

    // Selection highlight
    const applyHighlight = (selId: string | null) => {
      if (!selId) {
        nodeSel.attr("opacity", 1);
        edgeSel.attr("opacity", d => {
          if (d.edgeType === "speaker_owns") return 0;
          if (d.edgeType === "during_topic") return 0.15;
          return 0.5;
        });
        return;
      }
      const connected = new Set<string>([selId]);
      for (const l of links) {
        const src = typeof l.source === "string" ? l.source : (l.source as GraphNode).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as GraphNode).id;
        if (src === selId) connected.add(tgt);
        if (tgt === selId) connected.add(src);
      }
      nodeSel.attr("opacity", n => connected.has(n.id) ? 1 : 0.08);
      edgeSel.attr("opacity", e => {
        const src = typeof e.source === "string" ? e.source : (e.source as GraphNode).id;
        const tgt = typeof e.target === "string" ? e.target : (e.target as GraphNode).id;
        return (connected.has(src) && connected.has(tgt)) ? 0.9 : 0.02;
      });
    };
    applyHighlight(selectedNodeId);

    svg.on("click", event => {
      if (event.target === svgRef.current) setSelectedNodeId(null);
    });

    // ── Tick ──
    simulation.on("tick", () => {
      edgeSel.attr("d", d => {
        const s = d.source as GraphNode, t = d.target as GraphNode;
        const sx = s.x ?? 0, sy = s.y ?? 0, tx = t.x ?? 0, ty = t.y ?? 0;
        const dx = tx - sx, dy = ty - sy;
        const cx = (sx + tx) / 2 - dy * 0.06;
        const cy = (sy + ty) / 2 + dx * 0.06;
        return `M${sx},${sy} Q${cx},${cy} ${tx},${ty}`;
      });
      nodeSel.attr("transform", d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => { simulation.stop(); };
  }, [nodes, links, selectedNodeId, isFullscreen, durationMs, segments, speakerRoles, stats]);

  // ── Active filter count ──
  const activeFilterCount = (
    (filters.speakers.size > 0 ? 1 : 0) +
    (filters.agents.size < 3 ? 1 : 0) +
    (!filters.showTopics ? 1 : 0) +
    (!filters.showSignals ? 1 : 0) +
    (!filters.showUtterances ? 1 : 0) +
    (filters.confidenceMin !== 0.2 ? 1 : 0) +
    (filters.searchTerm ? 1 : 0) +
    (filters.simplified ? 1 : 0)
  );

  // ═══════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════

  return (
    <div className="conv-graph-root rounded-lg border border-nexus-border bg-nexus-bg overflow-hidden flex flex-col" style={{ height: isFullscreen ? "100vh" : "auto" }}>

      {/* ── TOP TOOLBAR ── */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-nexus-border bg-nexus-surface flex-wrap shrink-0">
        <span className="text-sm font-semibold text-nexus-text-primary">
          Conversation Graph
        </span>
        <span className="text-[10px] text-nexus-text-muted">
          {nodes.length} nodes · {links.length} edges
        </span>

        {/* View mode toggle */}
        {signalGraph && (
          <div className="flex rounded border border-nexus-border overflow-hidden text-[10px] ml-2">
            <button className="px-2 py-0.5 bg-nexus-blue text-white">Conversation</button>
            <button onClick={onSwitchToNetwork}
              className="px-2 py-0.5 bg-nexus-surface text-nexus-text-secondary hover:bg-nexus-surface-hover">
              Signal Network
            </button>
          </div>
        )}

        <div className="ml-auto flex items-center gap-1">
          <button onClick={() => setShowFilters(v => !v)}
            className={`rounded px-2 py-0.5 text-[10px] transition-colors ${showFilters ? "bg-nexus-blue text-white" : "text-nexus-text-secondary hover:bg-nexus-surface-hover"}`}>
            Filters{activeFilterCount > 0 && ` (${activeFilterCount})`}
          </button>
          <button onClick={() => setShowStats(v => !v)}
            className={`rounded px-2 py-0.5 text-[10px] transition-colors ${showStats ? "bg-nexus-blue text-white" : "text-nexus-text-secondary hover:bg-nexus-surface-hover"}`}>
            Stats
          </button>
          <button onClick={toggleFullscreen}
            className="rounded px-2 py-0.5 text-[10px] text-nexus-text-secondary hover:bg-nexus-surface-hover">
            {isFullscreen ? "Exit FS" : "Fullscreen"}
          </button>
          <button onClick={exportPng}
            className="rounded px-2 py-0.5 text-[10px] text-nexus-text-secondary hover:bg-nexus-surface-hover">
            Export
          </button>
          <button onClick={onClose}
            className="rounded px-2 py-0.5 text-[10px] text-nexus-text-muted hover:text-nexus-text-primary hover:bg-nexus-surface-hover">
            Close
          </button>
        </div>
      </div>

      {/* ── BODY ── */}
      <div className="flex flex-1 min-h-0 overflow-hidden">

        {/* ── FILTER PANEL ── */}
        {showFilters && (
          <div className="w-52 shrink-0 border-r border-nexus-border bg-nexus-surface overflow-y-auto p-3 space-y-3 text-[10px]">

            {/* Search */}
            <input type="text" placeholder="Search utterances / signals..."
              value={filters.searchTerm}
              onChange={e => updateFilter("searchTerm", e.target.value)}
              className="w-full rounded border border-nexus-border bg-nexus-bg px-2 py-1 text-[11px] text-nexus-text-primary" />

            {/* Show toggles */}
            <div>
              <div className="text-nexus-text-secondary font-semibold mb-1 uppercase tracking-wide">Show</div>
              {([
                { key: "showUtterances" as const, label: "Utterances" },
                { key: "showSignals" as const, label: "Signals" },
                { key: "showTopics" as const, label: "Topics" },
                { key: "simplified" as const, label: "Simplified (turns only)" },
              ]).map(({ key, label }) => (
                <label key={key} className="flex items-center gap-1.5 cursor-pointer py-0.5">
                  <input type="checkbox"
                    checked={key === "simplified" ? filters.simplified : (filters[key] as boolean)}
                    onChange={e => updateFilter(key, e.target.checked)}
                    className="h-3 w-3" />
                  <span className="text-nexus-text-primary">{label}</span>
                </label>
              ))}
            </div>

            {/* Confidence */}
            <div>
              <div className="text-nexus-text-secondary font-semibold mb-1 uppercase tracking-wide">
                Signal confidence &ge; {Math.round(filters.confidenceMin * 100)}%
              </div>
              <input type="range" min={0} max={85} step={5}
                value={Math.round(filters.confidenceMin * 100)}
                onChange={e => updateFilter("confidenceMin", Number(e.target.value) / 100)}
                className="w-full h-1 accent-blue-500" />
            </div>

            {/* Speakers */}
            <div>
              <div className="text-nexus-text-secondary font-semibold mb-1 uppercase tracking-wide">Speakers</div>
              {allSpeakers.map((s, i) => (
                <label key={s} className="flex items-center gap-1.5 cursor-pointer py-0.5">
                  <input type="checkbox"
                    checked={filters.speakers.size === 0 || filters.speakers.has(s)}
                    onChange={() => updateFilter("speakers", toggleSet(filters.speakers, s))}
                    className="h-3 w-3" />
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: SPEAKER_COLORS[i] }} />
                  <span className="text-nexus-text-primary">{speakerRoles[s] || s}</span>
                </label>
              ))}
            </div>

            {/* Agents */}
            <div>
              <div className="text-nexus-text-secondary font-semibold mb-1 uppercase tracking-wide">Signal Agent</div>
              {(["voice", "language", "fusion"] as const).map(a => (
                <label key={a} className="flex items-center gap-1.5 cursor-pointer py-0.5">
                  <input type="checkbox"
                    checked={filters.agents.has(a)}
                    onChange={() => updateFilter("agents", toggleSet(filters.agents, a))}
                    className="h-3 w-3" />
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: AGENT_COLORS[a] }} />
                  <span className="text-nexus-text-primary capitalize">{a}</span>
                </label>
              ))}
            </div>

            {activeFilterCount > 0 && (
              <button onClick={() => setFilters({
                speakers: new Set(), agents: new Set(["voice", "language", "fusion"]),
                showTopics: true, showSignals: true, showUtterances: true,
                confidenceMin: 0.2, searchTerm: "", simplified: false,
              })} className="w-full text-center rounded border border-nexus-border py-1 text-nexus-text-secondary hover:bg-nexus-surface-hover">
                Reset all filters
              </button>
            )}
          </div>
        )}

        {/* ── GRAPH CANVAS ── */}
        <div ref={containerRef} className="flex-1 relative min-w-0" style={{ minHeight: 680 }}>
          <svg ref={svgRef} className="w-full h-full" />

          {/* ── TOOLTIP ── */}
          {tooltip && tooltip.node && (
            <div className="fixed z-50 pointer-events-none rounded-lg border border-nexus-border bg-nexus-surface shadow-xl px-3 py-2 max-w-xs"
              style={{
                left: Math.min(tooltip.x + 12, window.innerWidth - 300),
                top: Math.min(tooltip.y + 12, window.innerHeight - 300),
              }}>
              <div className="flex items-center gap-2 mb-1">
                <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: tooltip.node.borderColor }} />
                <span className="text-xs font-semibold text-nexus-text-primary truncate">{tooltip.node.type === "utterance" ? "Utterance" : tooltip.node.label}</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-nexus-bg text-nexus-text-muted ml-auto capitalize">
                  {tooltip.node.type}
                </span>
              </div>
              {tooltip.node.type === "utterance" && (
                <div className="text-[10px] text-nexus-text-primary mb-1 leading-snug">{tooltip.node.detail}</div>
              )}
              <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[9px] text-nexus-text-secondary">
                {tooltip.node.speaker && <><span>Speaker</span><span className="text-nexus-text-primary">{speakerRoles[tooltip.node.speaker] || tooltip.node.speaker}</span></>}
                {tooltip.node.confidence != null && <><span>Confidence</span><span className="text-nexus-text-primary">{Math.round(tooltip.node.confidence * 100)}%</span></>}
                <><span>Time</span><span className="text-nexus-text-primary">{formatTime(tooltip.node.time_ms)}</span></>
                {tooltip.node.agent && <><span>Agent</span><span className="text-nexus-text-primary capitalize">{tooltip.node.agent}</span></>}
              </div>
              {tooltip.node.badges.length > 0 && (
                <div className="flex gap-1 mt-1">
                  {tooltip.node.badges.map((b, i) => (
                    <span key={i} className="rounded-full px-1.5 py-0.5 text-[8px] text-white"
                      style={{ backgroundColor: BADGE_COLORS[b] || "#888" }}>
                      {b}
                    </span>
                  ))}
                </div>
              )}
              {tooltip.connections && tooltip.connections.length > 0 && (
                <div className="mt-1.5 pt-1.5 border-t border-nexus-border">
                  <div className="text-[8px] text-nexus-text-muted uppercase tracking-wide mb-0.5">Connected</div>
                  {tooltip.connections.map((c, i) => (
                    <div key={i} className="text-[9px] text-nexus-text-secondary py-0.5 truncate">
                      <span className="text-nexus-text-muted">{c.edgeType}</span> → <span className="text-nexus-text-primary">{c.label}</span>
                    </div>
                  ))}
                </div>
              )}
              <div className="text-[8px] text-nexus-text-muted mt-1">Click to pin selection</div>
            </div>
          )}
        </div>

        {/* ── STATS PANEL ── */}
        {showStats && (
          <div className="w-52 shrink-0 border-l border-nexus-border bg-nexus-surface overflow-y-auto p-3 space-y-3 text-[10px]">
            {/* Turn stats */}
            <div>
              <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-1">Turns</div>
              <div className="text-nexus-text-primary text-lg font-bold">{stats.totalTurns}</div>
              <div className="text-nexus-text-muted">avg {stats.avgTurnLength} words/turn · {stats.crossSpeakerCount} exchanges</div>
            </div>

            {/* Per speaker */}
            <div>
              <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-1">By Speaker</div>
              {Object.entries(stats.turnsPerSpeaker).map(([spk, count], i) => {
                const totalWords = stats.wordsPerSpeaker[spk] ?? 0;
                const pctOfTotal = stats.totalTurns > 0 ? Math.round((count / stats.totalTurns) * 100) : 0;
                return (
                  <div key={spk} className="py-0.5">
                    <div className="flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: SPEAKER_COLORS[i] }} />
                      <span className="text-nexus-text-primary">{speakerRoles[spk] || spk}</span>
                      <span className="ml-auto text-nexus-text-muted font-mono">{count} turns</span>
                    </div>
                    <div className="flex items-center gap-1 ml-3.5">
                      <div className="flex-1 h-1 bg-nexus-bg rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${pctOfTotal}%`, backgroundColor: SPEAKER_COLORS[i] }} />
                      </div>
                      <span className="text-nexus-text-muted text-[9px] w-10 text-right">{totalWords}w</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Signal counts */}
            <div>
              <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-1">Signals by Agent</div>
              {Object.entries(stats.signalsByAgent).map(([agent, count]) => (
                <div key={agent} className="flex items-center gap-1.5 py-0.5">
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: AGENT_COLORS[agent] ?? "#888" }} />
                  <span className="text-nexus-text-primary capitalize">{agent}</span>
                  <span className="ml-auto text-nexus-text-muted font-mono">{count}</span>
                </div>
              ))}
            </div>

            {/* Top signal types */}
            <div>
              <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-1">Top Signal Types</div>
              {Object.entries(stats.signalCounts)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 8)
                .map(([type, count]) => (
                  <div key={type} className="flex items-center gap-1.5 py-0.5">
                    <span className="text-nexus-text-primary truncate">{type}</span>
                    <span className="ml-auto text-nexus-text-muted font-mono">{count}</span>
                  </div>
                ))}
            </div>

            {/* Topics */}
            {stats.topicCount > 0 && (
              <div>
                <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-1">Topics</div>
                {(entities.topics ?? []).map((t, i) => (
                  <div key={i} className="flex items-center gap-1.5 py-0.5">
                    <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: TOPIC_COLORS[i] }} />
                    <span className="text-nexus-text-primary truncate">{t.name}</span>
                    <span className="ml-auto text-nexus-text-muted">{formatTime(t.start_ms)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── TIMELINE SCRUBBER ── */}
      <div className="px-4 py-2 border-t border-nexus-border bg-nexus-surface shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-nexus-text-muted w-10">{formatTime(timeRange[0])}</span>
          <input type="range" min={0} max={durationMs} step={1000}
            value={timeRange[1]}
            onChange={e => setTimeRange([0, Number(e.target.value)])}
            className="flex-1 h-1 accent-blue-500" />
          <span className="text-[10px] text-nexus-text-muted w-10 text-right">{formatTime(timeRange[1])}</span>
        </div>
      </div>

      {/* ── LEGEND ── */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-0.5 px-4 py-1 border-t border-nexus-border bg-nexus-surface shrink-0">
        {([
          { color: "#4F8BFF", shape: "circle", label: "Speaker" },
          { color: "var(--bg-surface, #1A1A2E)", shape: "rect", label: "Utterance" },
          { color: "#8B5CF6", shape: "hex", label: "Topic" },
          { color: "#6366F1", shape: "rect", label: "Voice Signal" },
          { color: "#06B6D4", shape: "rect", label: "Lang Signal" },
          { color: "#F97316", shape: "rect", label: "Fusion Insight" },
        ] as const).map(({ color, label }) => (
          <span key={label} className="flex items-center gap-1 text-[9px] text-nexus-text-secondary">
            <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
            {label}
          </span>
        ))}
        <span className="text-[9px] text-nexus-text-muted">|</span>
        {Object.entries(BADGE_COLORS).map(([name, col]) => (
          <span key={name} className="flex items-center gap-1 text-[9px] text-nexus-text-secondary">
            <span className="h-1.5 w-1.5 rounded-full shrink-0" style={{ backgroundColor: col }} />
            {name}
          </span>
        ))}
      </div>
    </div>
  );
}
