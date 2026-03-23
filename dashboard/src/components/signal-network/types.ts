import type * as d3 from "d3";

// ── Backend data types ──

export interface BackendGraphNode {
  id: string;
  type: string;         // speaker | topic | voice_signal | lang_signal | fusion_signal
  label: string;
  agent?: string | null;
  value?: number | null;
  value_text?: string;
  confidence?: number | null;
  timestamp_ms?: number;
  end_ms?: number;
  speaker_id?: string;
  signal_type?: string;
  metadata?: Record<string, unknown>;
}

export interface BackendGraphEdge {
  source: string;
  target: string;
  relationship: string; // contradicts | triggered | co_occurred | preceded | about_topic | speaker_produced | resolves
  weight?: number;
}

export interface SignalGraphData {
  nodes: BackendGraphNode[];
  edges: BackendGraphEdge[];
  stats?: {
    node_count: number;
    edge_count: number;
    node_types: Record<string, number>;
    edge_types: Record<string, number>;
  };
}

// ── Filter state ──

export interface SignalNetworkFilters {
  speakers: Set<string>;
  agents: Set<string>;
  edgeTypes: Set<string>;
  nodeTypes: Set<string>;
  confidenceMin: number;
  searchTerm: string;
  layoutMode: "force" | "by_speaker" | "by_topic" | "by_agent";
}

// ── Rendered graph types ──

export type VisualNodeType = "speaker" | "topic" | "voice_signal" | "lang_signal" | "fusion_signal";

export interface VisualNode extends d3.SimulationNodeDatum {
  id: string;
  type: VisualNodeType;
  label: string;
  detail: string;
  speaker?: string;
  agent?: string;
  signalType?: string;
  value?: number | null;
  valueText?: string;
  confidence?: number;
  color: string;
  borderColor: string;
  width: number;
  height: number;
  timeMs: number;
  endMs?: number;
  degree: number;         // connection count (computed)
  pinned?: boolean;
  metadata?: Record<string, unknown>;
}

export interface VisualEdge extends d3.SimulationLinkDatum<VisualNode> {
  id: string;
  edgeType: string;
  color: string;
  width: number;
  dashed: boolean;
  animated?: boolean;
  label: string;
  opacity: number;
}

// ── Stats ──

export interface NetworkStats {
  totalNodes: number;
  totalEdges: number;
  visibleNodes: number;
  visibleEdges: number;
  contradictionCount: number;
  triggerCount: number;
  resolveCount: number;
  topConnected: Array<{ id: string; label: string; degree: number; type: string }>;
  edgeTypeCounts: Record<string, number>;
  nodeTypeCounts: Record<string, number>;
  avgConfidence: number;
  speakerSignalCounts: Record<string, number>;
}

// ── Tooltip ──

export interface TooltipData {
  node?: VisualNode;
  edge?: VisualEdge;
  x: number;
  y: number;
  connections?: Array<{ label: string; type: string; edgeType: string; direction: "in" | "out" }>;
}
