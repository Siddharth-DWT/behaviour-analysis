import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import type {
  SignalGraphData, SignalNetworkFilters,
  VisualNode, VisualEdge, NetworkStats, TooltipData,
} from "./types";
import {
  NODE_COLORS, SPEAKER_COLORS,
  EDGE_STYLES, EDGE_TYPE_LABELS, NODE_TYPE_LABELS, SIGNAL_LABELS,
  ALL_AGENTS, DEFAULT_EDGE_TYPES, DEFAULT_CONFIDENCE_MIN, LAYOUT_MODES,
} from "./signalNetworkConstants";
import { buildFilteredSignalNetwork } from "./signalNetworkLayout";

// ═══════════════════════════════════════════
// PROPS
// ═══════════════════════════════════════════

interface Props {
  signalGraph: SignalGraphData;
  speakerRoles: Record<string, string>;
  onClose: () => void;
}

// ═══════════════════════════════════════════
// FORMAT HELPERS
// ═══════════════════════════════════════════

function formatTime(ms: number): string {
  const sec = Math.floor(ms / 1000);
  return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
}

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

// ═══════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════

export default function SignalNetworkView({ signalGraph, speakerRoles, onClose }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // ── Filter state ──
  const allSpeakers = useMemo(() => {
    const s = new Set<string>();
    signalGraph.nodes.forEach(n => { if (n.type === "speaker" && n.speaker_id) s.add(n.speaker_id); });
    return [...s];
  }, [signalGraph]);

  const allSignalTypes = useMemo(() => {
    const s = new Set<string>();
    signalGraph.nodes.forEach(n => { if (n.signal_type) s.add(n.signal_type); });
    return [...s].sort();
  }, [signalGraph]);

  const [filters, setFilters] = useState<SignalNetworkFilters>({
    speakers: new Set<string>(),
    agents: new Set(ALL_AGENTS),
    edgeTypes: new Set(DEFAULT_EDGE_TYPES),
    nodeTypes: new Set<string>(),
    confidenceMin: DEFAULT_CONFIDENCE_MIN,
    searchTerm: "",
    layoutMode: "force",
  });
  const [showFilters, setShowFilters] = useState(true);
  const [showStats, setShowStats] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showEdgeLabels, setShowEdgeLabels] = useState(false);

  // ── Compute filtered data ──
  const { nodes, edges, stats } = useMemo(
    () => buildFilteredSignalNetwork(signalGraph, speakerRoles, filters),
    [signalGraph, speakerRoles, filters]
  );

  // ── Filter helpers ──
  const toggleSet = useCallback(<T extends string>(prev: Set<T>, val: T): Set<T> => {
    const next = new Set(prev);
    if (next.has(val)) next.delete(val); else next.add(val);
    return next;
  }, []);

  const updateFilter = useCallback(<K extends keyof SignalNetworkFilters>(key: K, val: SignalNetworkFilters[K]) => {
    setFilters(f => ({ ...f, [key]: val }));
  }, []);

  // ── Fullscreen ──
  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current?.parentElement;
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

  // ── PNG export ──
  const exportPng = useCallback(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const clone = svg.cloneNode(true) as SVGSVGElement;
    // Inline computed styles for canvas rendering
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
        a.download = `signal-network-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
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
    svg.attr("width", width).attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`);

    // ── Defs ──
    const defs = svg.append("defs");

    // Arrow markers per edge type
    for (const [type, style] of Object.entries(EDGE_STYLES)) {
      defs.append("marker")
        .attr("id", `arrow-${type}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 18)
        .attr("markerWidth", 5).attr("markerHeight", 5)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-4L10,0L0,4")
        .attr("fill", style.color);
    }

    // Glow filter
    const glow = defs.append("filter").attr("id", "glow").attr("x", "-30%").attr("y", "-30%").attr("width", "160%").attr("height", "160%");
    glow.append("feGaussianBlur").attr("stdDeviation", 4).attr("result", "blur");
    glow.append("feMerge").selectAll("feMergeNode")
      .data(["blur", "SourceGraphic"]).enter()
      .append("feMergeNode").attr("in", d => d);

    // Drop shadow for nodes
    const shadow = defs.append("filter").attr("id", "shadow").attr("x", "-10%").attr("y", "-10%").attr("width", "130%").attr("height", "130%");
    shadow.append("feDropShadow").attr("dx", 0).attr("dy", 2).attr("stdDeviation", 3).attr("flood-opacity", 0.3);

    const g = svg.append("g");

    // ── Zoom ──
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.15, 4])
      .on("zoom", event => g.attr("transform", event.transform));
    svg.call(zoom);

    // ── Force simulation ──
    const layoutMode = filters.layoutMode;

    const simulation = d3.forceSimulation<VisualNode>(nodes)
      .force("link",
        d3.forceLink<VisualNode, VisualEdge>(edges)
          .id(d => d.id)
          .distance(d => {
            if (d.edgeType === "speaker_produced") return 80;
            if (d.edgeType === "co_occurred") return 40;
            if (d.edgeType === "contradicts" || d.edgeType === "triggered") return 100;
            return 60;
          })
          .strength(d => {
            if (d.edgeType === "speaker_produced") return 0.3;
            if (d.edgeType === "co_occurred") return 0.05;
            return 0.15;
          })
      )
      .force("charge", d3.forceManyBody<VisualNode>()
        .strength(d => d.type === "speaker" ? -400 : d.type === "topic" ? -200 : -80)
      )
      .force("center", d3.forceCenter(width / 2, height / 2).strength(0.05))
      .force("collision", d3.forceCollide<VisualNode>().radius(d => Math.max(d.width, d.height) / 2 + 4))
      .velocityDecay(0.45)
      .alphaDecay(0.025);

    // Layout-specific forces
    if (layoutMode === "by_speaker") {
      const speakerCenters: Record<string, { x: number; y: number }> = {};
      allSpeakers.forEach((s, i) => {
        const angle = (i / allSpeakers.length) * 2 * Math.PI - Math.PI / 2;
        speakerCenters[s] = { x: width / 2 + Math.cos(angle) * 200, y: height / 2 + Math.sin(angle) * 200 };
      });
      simulation.force("cluster-x", d3.forceX<VisualNode>(d => {
        if (d.speaker && speakerCenters[d.speaker]) return speakerCenters[d.speaker].x;
        return width / 2;
      }).strength(0.15));
      simulation.force("cluster-y", d3.forceY<VisualNode>(d => {
        if (d.speaker && speakerCenters[d.speaker]) return speakerCenters[d.speaker].y;
        return height / 2;
      }).strength(0.15));
    } else if (layoutMode === "by_agent") {
      const agentX: Record<string, number> = { voice: width * 0.25, language: width * 0.5, fusion: width * 0.75 };
      simulation.force("cluster-x", d3.forceX<VisualNode>(d => {
        if (d.agent && agentX[d.agent]) return agentX[d.agent];
        if (d.type === "topic") return width * 0.5;
        return width * 0.5;
      }).strength(0.2));
    } else if (layoutMode === "by_topic") {
      simulation.force("y-time", d3.forceY<VisualNode>(d => {
        return 50 + (d.timeMs / Math.max(1, d3.max(nodes, n => n.timeMs) ?? 1)) * (height - 100);
      }).strength(0.12));
    }

    // Pin speaker nodes
    for (const n of nodes) {
      if (n.type === "speaker") {
        const idx = allSpeakers.indexOf(n.speaker ?? "");
        if (layoutMode !== "by_speaker") {
          n.fx = idx === 0 ? 100 : idx === 1 ? width - 100 : width / 2;
          n.fy = height / 2;
        }
      }
    }

    // ── Edges (curved paths) ──
    const edgeGroup = g.append("g").attr("class", "edges");

    const edgeSel = edgeGroup.selectAll<SVGPathElement, VisualEdge>("path")
      .data(edges)
      .join("path")
      .attr("fill", "none")
      .attr("stroke", d => d.color)
      .attr("stroke-width", d => d.width)
      .attr("stroke-dasharray", d => d.dashed ? (d.animated ? "8 4" : "5 3") : "none")
      .attr("opacity", d => d.opacity)
      .attr("marker-end", d => {
        if (d.edgeType === "triggered" || d.edgeType === "preceded" || d.edgeType === "resolves" || d.edgeType === "about_topic")
          return `url(#arrow-${d.edgeType})`;
        return "";
      })
      .attr("cursor", "pointer");

    // Animated dash for contradicts
    edgeSel.filter(d => !!d.animated)
      .each(function () {
        const el = d3.select(this);
        const len = 12;
        el.attr("stroke-dashoffset", 0);
        (function animate() {
          el.transition()
            .duration(800)
            .ease(d3.easeLinear)
            .attr("stroke-dashoffset", -len)
            .on("end", animate);
        })();
      });

    // Edge hover
    edgeSel
      .on("mouseenter", function (event, d) {
        d3.select(this).attr("stroke-width", d.width + 1.5).attr("opacity", 1);
        setTooltip({
          edge: d,
          x: event.clientX,
          y: event.clientY,
        });
      })
      .on("mouseleave", function (_, d) {
        d3.select(this).attr("stroke-width", d.width).attr("opacity", d.opacity);
        setTooltip(prev => prev?.edge ? null : prev);
      });

    // Edge labels (optional)
    if (showEdgeLabels) {
      const importantEdges = edges.filter(e =>
        e.edgeType === "contradicts" || e.edgeType === "triggered" || e.edgeType === "resolves"
      );
      const edgeLabelSel = edgeGroup.selectAll<SVGTextElement, VisualEdge>("text.edge-label")
        .data(importantEdges)
        .join("text")
        .attr("class", "edge-label")
        .attr("font-size", 8)
        .attr("fill", d => d.color)
        .attr("text-anchor", "middle")
        .attr("pointer-events", "none")
        .text(d => EDGE_TYPE_LABELS[d.edgeType] ?? d.edgeType);

      simulation.on("tick.edgeLabels", () => {
        edgeLabelSel
          .attr("x", d => {
            const s = d.source as VisualNode, t = d.target as VisualNode;
            return ((s.x ?? 0) + (t.x ?? 0)) / 2;
          })
          .attr("y", d => {
            const s = d.source as VisualNode, t = d.target as VisualNode;
            return ((s.y ?? 0) + (t.y ?? 0)) / 2 - 6;
          });
      });
    }

    // ── Nodes ──
    const nodeGroup = g.append("g").attr("class", "nodes");

    const nodeSel = nodeGroup.selectAll<SVGGElement, VisualNode>("g")
      .data(nodes)
      .join("g")
      .attr("cursor", "pointer")
      .call(d3.drag<SVGGElement, VisualNode>()
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
        // Large circle with label
        el.append("circle").attr("r", 40)
          .attr("fill", d.color).attr("fill-opacity", 0.12)
          .attr("stroke", d.color).attr("stroke-width", 3)
          .attr("filter", "url(#shadow)");
        el.append("text").attr("text-anchor", "middle").attr("dy", -6)
          .attr("font-size", 13).attr("font-weight", 700)
          .attr("fill", d.color)
          .text(d.label);
        el.append("text").attr("text-anchor", "middle").attr("dy", 12)
          .attr("font-size", 9).attr("fill", "#9CA3AF")
          .text(speakerRoles[d.speaker ?? ""] || d.speaker || "");

      } else if (d.type === "topic") {
        // Hexagonal shape
        const w = d.width / 2, h = d.height / 2;
        const pts = [
          [-w, 0], [-w + 12, -h], [w - 12, -h], [w, 0], [w - 12, h], [-w + 12, h],
        ].map(p => p.join(",")).join(" ");
        el.append("polygon").attr("points", pts)
          .attr("fill", d.color).attr("fill-opacity", 0.18)
          .attr("stroke", d.color).attr("stroke-width", 1.5);
        el.append("text").attr("text-anchor", "middle").attr("dy", 4)
          .attr("font-size", 10).attr("font-weight", 600).attr("fill", d.color)
          .text(d.label.length > 16 ? d.label.slice(0, 14) + ".." : d.label);

      } else if (d.type === "fusion_signal") {
        // Rounded rect with glow
        el.append("rect")
          .attr("x", -d.width / 2).attr("y", -d.height / 2)
          .attr("width", d.width).attr("height", d.height).attr("rx", 8)
          .attr("fill", "#F97316").attr("fill-opacity", 0.15)
          .attr("stroke", "#F97316").attr("stroke-width", 2)
          .attr("filter", "url(#glow)");
        // Icon
        el.append("text").attr("x", -d.width / 2 + 12).attr("dy", 4)
          .attr("font-size", 12).text("⚡");
        // Label
        el.append("text").attr("x", -d.width / 2 + 28).attr("dy", 4)
          .attr("font-size", 10).attr("font-weight", 600).attr("fill", "#F97316")
          .text(d.label.length > 18 ? d.label.slice(0, 16) + ".." : d.label);
        // Confidence pill
        if (d.confidence != null) {
          const pw = 30, px = d.width / 2 - pw - 6;
          el.append("rect")
            .attr("x", px).attr("y", -7).attr("width", pw).attr("height", 14).attr("rx", 7)
            .attr("fill", "#F97316").attr("fill-opacity", 0.3);
          el.append("text")
            .attr("x", px + pw / 2).attr("dy", 3).attr("text-anchor", "middle")
            .attr("font-size", 8).attr("font-weight", 600).attr("fill", "#F97316")
            .text(pct(d.confidence));
        }

      } else {
        // Voice or Language signal — rounded rect
        const isVoice = d.type === "voice_signal";
        const col = isVoice ? "#6366F1" : "#06B6D4";
        el.append("rect")
          .attr("x", -d.width / 2).attr("y", -d.height / 2)
          .attr("width", d.width).attr("height", d.height).attr("rx", 6)
          .attr("fill", col).attr("fill-opacity", 0.1)
          .attr("stroke", col).attr("stroke-width", 1.5);
        // Agent icon (small circle)
        el.append("circle")
          .attr("cx", -d.width / 2 + 12).attr("cy", 0).attr("r", 5)
          .attr("fill", col).attr("fill-opacity", 0.6);
        el.append("text")
          .attr("x", -d.width / 2 + 12).attr("dy", 3).attr("text-anchor", "middle")
          .attr("font-size", 7).attr("fill", "#fff")
          .text(isVoice ? "V" : "L");
        // Label
        const maxLabelW = d.width - 56;
        el.append("text").attr("x", -d.width / 2 + 24).attr("dy", 4)
          .attr("font-size", 9).attr("font-weight", 500).attr("fill", col)
          .text(d.label.length > maxLabelW / 5 ? d.label.slice(0, Math.floor(maxLabelW / 5)) + ".." : d.label);
        // Confidence pill
        if (d.confidence != null) {
          const pw = 28, px = d.width / 2 - pw - 4;
          el.append("rect")
            .attr("x", px).attr("y", -6).attr("width", pw).attr("height", 12).attr("rx", 6)
            .attr("fill", col).attr("fill-opacity", Math.min(0.5, d.confidence));
          el.append("text")
            .attr("x", px + pw / 2).attr("dy", 3).attr("text-anchor", "middle")
            .attr("font-size", 7).attr("font-weight", 600).attr("fill", col)
            .text(pct(d.confidence));
        }
      }
    });

    // ── Node interactions ──
    nodeSel
      .on("mouseenter", function (event, d) {
        d3.select(this).raise();
        // Compute connections
        const conns: TooltipData["connections"] = [];
        for (const e of edges) {
          const src = typeof e.source === "string" ? e.source : (e.source as VisualNode).id;
          const tgt = typeof e.target === "string" ? e.target : (e.target as VisualNode).id;
          if (src === d.id) {
            const tn = nodes.find(n => n.id === tgt);
            if (tn) conns.push({ label: tn.label, type: NODE_TYPE_LABELS[tn.type] ?? tn.type, edgeType: e.edgeType, direction: "out" });
          } else if (tgt === d.id) {
            const sn = nodes.find(n => n.id === src);
            if (sn) conns.push({ label: sn.label, type: NODE_TYPE_LABELS[sn.type] ?? sn.type, edgeType: e.edgeType, direction: "in" });
          }
        }
        setTooltip({ node: d, x: event.clientX, y: event.clientY, connections: conns.slice(0, 8) });
      })
      .on("mouseleave", () => {
        setTooltip(prev => prev?.node ? null : prev);
      })
      .on("click", (_event, d) => {
        setSelectedNodeId(prev => prev === d.id ? null : d.id);
      });

    // ── Selection highlighting ──
    const applyHighlight = (selId: string | null) => {
      if (!selId) {
        nodeSel.attr("opacity", 1);
        edgeSel.attr("opacity", d => d.opacity);
        return;
      }
      const connected = new Set<string>([selId]);
      for (const e of edges) {
        const src = typeof e.source === "string" ? e.source : (e.source as VisualNode).id;
        const tgt = typeof e.target === "string" ? e.target : (e.target as VisualNode).id;
        if (src === selId) connected.add(tgt);
        if (tgt === selId) connected.add(src);
      }
      nodeSel.attr("opacity", n => connected.has(n.id) ? 1 : 0.08);
      edgeSel.attr("opacity", e => {
        const src = typeof e.source === "string" ? e.source : (e.source as VisualNode).id;
        const tgt = typeof e.target === "string" ? e.target : (e.target as VisualNode).id;
        return (connected.has(src) && connected.has(tgt)) ? 0.9 : 0.02;
      });
    };
    applyHighlight(selectedNodeId);

    // Click background to deselect
    svg.on("click", (event) => {
      if (event.target === svgRef.current) {
        setSelectedNodeId(null);
      }
    });

    // ── Tick ──
    const curvedPath = (d: VisualEdge) => {
      const s = d.source as VisualNode, t = d.target as VisualNode;
      const sx = s.x ?? 0, sy = s.y ?? 0, tx = t.x ?? 0, ty = t.y ?? 0;
      const dx = tx - sx, dy = ty - sy;
      // Slight curve perpendicular to the line
      const cx = (sx + tx) / 2 - dy * 0.08;
      const cy = (sy + ty) / 2 + dx * 0.08;
      return `M${sx},${sy} Q${cx},${cy} ${tx},${ty}`;
    };

    simulation.on("tick", () => {
      edgeSel.attr("d", curvedPath);
      nodeSel.attr("transform", d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => { simulation.stop(); };
  }, [nodes, edges, filters.layoutMode, selectedNodeId, showEdgeLabels, allSpeakers, speakerRoles, isFullscreen]);

  // ═══════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════

  const activeFilterCount = (
    (filters.speakers.size > 0 ? 1 : 0) +
    (filters.agents.size < 3 ? 1 : 0) +
    (filters.edgeTypes.size !== DEFAULT_EDGE_TYPES.size ? 1 : 0) +
    (filters.confidenceMin !== DEFAULT_CONFIDENCE_MIN ? 1 : 0) +
    (filters.searchTerm ? 1 : 0) +
    (filters.layoutMode !== "force" ? 1 : 0)
  );

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-bg overflow-hidden flex flex-col" style={{ height: isFullscreen ? "100vh" : "auto" }}>

      {/* ── TOP TOOLBAR ── */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-nexus-border bg-nexus-surface flex-wrap shrink-0">
        <span className="text-sm font-semibold text-nexus-text-primary">
          Signal Relationship Network
        </span>
        <span className="text-[10px] text-nexus-text-muted">
          {stats.visibleNodes} / {stats.totalNodes} nodes · {stats.visibleEdges} / {stats.totalEdges} edges
        </span>

        <div className="ml-auto flex items-center gap-1">
          {/* Edge labels toggle */}
          <label className="flex items-center gap-1 text-[10px] text-nexus-text-secondary cursor-pointer px-1">
            <input type="checkbox" checked={showEdgeLabels} onChange={e => setShowEdgeLabels(e.target.checked)} className="h-3 w-3" />
            Labels
          </label>

          {/* Filters toggle */}
          <button
            onClick={() => setShowFilters(v => !v)}
            className={`rounded px-2 py-0.5 text-[10px] transition-colors ${showFilters ? "bg-nexus-blue text-white" : "text-nexus-text-secondary hover:bg-nexus-surface-hover"}`}
          >
            Filters{activeFilterCount > 0 && ` (${activeFilterCount})`}
          </button>

          {/* Stats toggle */}
          <button
            onClick={() => setShowStats(v => !v)}
            className={`rounded px-2 py-0.5 text-[10px] transition-colors ${showStats ? "bg-nexus-blue text-white" : "text-nexus-text-secondary hover:bg-nexus-surface-hover"}`}
          >
            Stats
          </button>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            className="rounded px-2 py-0.5 text-[10px] text-nexus-text-secondary hover:bg-nexus-surface-hover"
            title="Toggle fullscreen"
          >
            {isFullscreen ? "Exit FS" : "Fullscreen"}
          </button>

          {/* Export */}
          <button
            onClick={exportPng}
            className="rounded px-2 py-0.5 text-[10px] text-nexus-text-secondary hover:bg-nexus-surface-hover"
            title="Export as PNG"
          >
            Export
          </button>

          <button onClick={onClose}
            className="rounded px-2 py-0.5 text-[10px] text-nexus-text-muted hover:text-nexus-text-primary hover:bg-nexus-surface-hover"
          >
            Close
          </button>
        </div>
      </div>

      {/* ── BODY: filters | graph | stats ── */}
      <div className="flex flex-1 min-h-0 overflow-hidden">

        {/* ── FILTER PANEL (left) ── */}
        {showFilters && (
          <div className="w-56 shrink-0 border-r border-nexus-border bg-nexus-surface overflow-y-auto p-3 space-y-4 text-[10px]">

            {/* Search */}
            <input
              type="text"
              placeholder="Search signals..."
              value={filters.searchTerm}
              onChange={e => updateFilter("searchTerm", e.target.value)}
              className="w-full rounded border border-nexus-border bg-nexus-bg px-2 py-1 text-[11px] text-nexus-text-primary"
            />

            {/* Layout */}
            <div>
              <div className="text-nexus-text-secondary font-semibold mb-1 uppercase tracking-wide">Layout</div>
              <select
                value={filters.layoutMode}
                onChange={e => updateFilter("layoutMode", e.target.value as SignalNetworkFilters["layoutMode"])}
                className="w-full rounded border border-nexus-border bg-nexus-bg px-2 py-1 text-nexus-text-primary"
              >
                {LAYOUT_MODES.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>
            </div>

            {/* Confidence */}
            <div>
              <div className="text-nexus-text-secondary font-semibold mb-1 uppercase tracking-wide">
                Confidence &ge; {pct(filters.confidenceMin)}
              </div>
              <input
                type="range" min={0} max={85} step={5}
                value={Math.round(filters.confidenceMin * 100)}
                onChange={e => updateFilter("confidenceMin", Number(e.target.value) / 100)}
                className="w-full h-1 accent-blue-500"
              />
            </div>

            {/* Speakers */}
            <div>
              <div className="text-nexus-text-secondary font-semibold mb-1 uppercase tracking-wide">Speakers</div>
              {allSpeakers.map((s, i) => (
                <label key={s} className="flex items-center gap-1.5 cursor-pointer py-0.5">
                  <input type="checkbox"
                    checked={filters.speakers.size === 0 || filters.speakers.has(s)}
                    onChange={() => updateFilter("speakers", toggleSet(filters.speakers, s))}
                    className="h-3 w-3"
                  />
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: SPEAKER_COLORS[i] }} />
                  <span className="text-nexus-text-primary">{speakerRoles[s] || s}</span>
                </label>
              ))}
            </div>

            {/* Agents */}
            <div>
              <div className="text-nexus-text-secondary font-semibold mb-1 uppercase tracking-wide">Agent</div>
              {["voice", "language", "fusion"].map(a => (
                <label key={a} className="flex items-center gap-1.5 cursor-pointer py-0.5">
                  <input type="checkbox"
                    checked={filters.agents.has(a)}
                    onChange={() => updateFilter("agents", toggleSet(filters.agents, a))}
                    className="h-3 w-3"
                  />
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: NODE_COLORS[a === "voice" ? "voice_signal" : a === "language" ? "lang_signal" : "fusion_signal"] }} />
                  <span className="text-nexus-text-primary capitalize">{a}</span>
                </label>
              ))}
            </div>

            {/* Edge types */}
            <div>
              <div className="text-nexus-text-secondary font-semibold mb-1 uppercase tracking-wide">Relationships</div>
              {Object.entries(EDGE_TYPE_LABELS).map(([key, label]) => (
                <label key={key} className="flex items-center gap-1.5 cursor-pointer py-0.5">
                  <input type="checkbox"
                    checked={filters.edgeTypes.has(key)}
                    onChange={() => updateFilter("edgeTypes", toggleSet(filters.edgeTypes, key))}
                    className="h-3 w-3"
                  />
                  <span className="w-4 h-0.5 shrink-0 rounded" style={{
                    backgroundColor: EDGE_STYLES[key]?.color ?? "#888",
                    borderTop: EDGE_STYLES[key]?.dashed ? "1px dashed" : "none",
                  }} />
                  <span className="text-nexus-text-primary">{label}</span>
                  <span className="ml-auto text-nexus-text-muted">{stats.edgeTypeCounts[key] ?? 0}</span>
                </label>
              ))}
            </div>

            {/* Reset */}
            {activeFilterCount > 0 && (
              <button
                onClick={() => setFilters({
                  speakers: new Set(),
                  agents: new Set(ALL_AGENTS),
                  edgeTypes: new Set(DEFAULT_EDGE_TYPES),
                  nodeTypes: new Set(),
                  confidenceMin: DEFAULT_CONFIDENCE_MIN,
                  searchTerm: "",
                  layoutMode: "force",
                })}
                className="w-full text-center rounded border border-nexus-border py-1 text-nexus-text-secondary hover:bg-nexus-surface-hover"
              >
                Reset all filters
              </button>
            )}
          </div>
        )}

        {/* ── GRAPH CANVAS ── */}
        <div ref={containerRef} className="flex-1 relative min-w-0" style={{ minHeight: 680 }}>
          <svg ref={svgRef} className="w-full h-full" />

          {/* ── TOOLTIP OVERLAY ── */}
          {tooltip && (
            <div
              className="fixed z-50 pointer-events-none rounded-lg border border-nexus-border bg-nexus-surface shadow-xl px-3 py-2 max-w-xs"
              style={{
                left: Math.min(tooltip.x + 12, window.innerWidth - 300),
                top: Math.min(tooltip.y + 12, window.innerHeight - 300),
              }}
            >
              {tooltip.node && (
                <>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: tooltip.node.color }} />
                    <span className="text-xs font-semibold text-nexus-text-primary truncate">{tooltip.node.label}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-nexus-bg text-nexus-text-muted ml-auto">
                      {NODE_TYPE_LABELS[tooltip.node.type]}
                    </span>
                  </div>
                  {tooltip.node.signalType && (
                    <div className="text-[9px] font-mono text-nexus-text-muted mb-1">{tooltip.node.signalType}</div>
                  )}
                  <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[9px] text-nexus-text-secondary">
                    {tooltip.node.speaker && <><span>Speaker</span><span className="text-nexus-text-primary">{speakerRoles[tooltip.node.speaker] || tooltip.node.speaker}</span></>}
                    {tooltip.node.confidence != null && <><span>Confidence</span><span className="text-nexus-text-primary">{pct(tooltip.node.confidence)}</span></>}
                    {tooltip.node.value != null && <><span>Value</span><span className="text-nexus-text-primary">{tooltip.node.value.toFixed(3)}</span></>}
                    {tooltip.node.valueText && <><span>Status</span><span className="text-nexus-text-primary">{tooltip.node.valueText}</span></>}
                    <><span>Time</span><span className="text-nexus-text-primary">{formatTime(tooltip.node.timeMs)}</span></>
                    <><span>Connections</span><span className="text-nexus-text-primary">{tooltip.node.degree}</span></>
                  </div>
                  {tooltip.connections && tooltip.connections.length > 0 && (
                    <div className="mt-1.5 pt-1.5 border-t border-nexus-border">
                      <div className="text-[8px] text-nexus-text-muted uppercase tracking-wide mb-0.5">Connected to</div>
                      {tooltip.connections.map((c, i) => (
                        <div key={i} className="flex items-center gap-1 text-[9px] text-nexus-text-secondary py-0.5">
                          <span className="w-3 shrink-0 text-center" style={{ color: EDGE_STYLES[c.edgeType]?.color ?? "#888" }}>
                            {c.direction === "out" ? "→" : "←"}
                          </span>
                          <span className="text-nexus-text-primary truncate">{c.label}</span>
                          <span className="text-nexus-text-muted ml-auto text-[8px]">{EDGE_TYPE_LABELS[c.edgeType] ?? c.edgeType}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="text-[8px] text-nexus-text-muted mt-1">Click to pin selection</div>
                </>
              )}
              {tooltip.edge && (
                <>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="h-0.5 w-4 rounded" style={{ backgroundColor: tooltip.edge.color }} />
                    <span className="text-xs font-semibold text-nexus-text-primary">
                      {EDGE_TYPE_LABELS[tooltip.edge.edgeType] ?? tooltip.edge.edgeType}
                    </span>
                  </div>
                  <div className="text-[9px] text-nexus-text-secondary">
                    {EDGE_STYLES[tooltip.edge.edgeType]?.description ?? ""}
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* ── STATS PANEL (right) ── */}
        {showStats && (
          <div className="w-52 shrink-0 border-l border-nexus-border bg-nexus-surface overflow-y-auto p-3 space-y-3 text-[10px]">

            {/* Headline badges */}
            <div className="flex flex-wrap gap-1.5">
              {stats.contradictionCount > 0 && (
                <span className="rounded px-2 py-0.5 text-white text-[9px] font-semibold" style={{ backgroundColor: "#EF4444" }}>
                  {stats.contradictionCount} contradictions
                </span>
              )}
              {stats.triggerCount > 0 && (
                <span className="rounded px-2 py-0.5 text-white text-[9px] font-semibold" style={{ backgroundColor: "#F97316" }}>
                  {stats.triggerCount} triggers
                </span>
              )}
              {stats.resolveCount > 0 && (
                <span className="rounded px-2 py-0.5 text-white text-[9px] font-semibold" style={{ backgroundColor: "#22C55E" }}>
                  {stats.resolveCount} resolved
                </span>
              )}
            </div>

            {/* Avg confidence */}
            <div>
              <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-0.5">Avg Confidence</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-1.5 bg-nexus-bg rounded-full overflow-hidden">
                  <div className="h-full rounded-full" style={{
                    width: pct(stats.avgConfidence),
                    backgroundColor: stats.avgConfidence > 0.6 ? "#22C55E" : stats.avgConfidence > 0.4 ? "#F59E0B" : "#EF4444",
                  }} />
                </div>
                <span className="text-nexus-text-primary font-semibold">{pct(stats.avgConfidence)}</span>
              </div>
            </div>

            {/* Node type breakdown */}
            <div>
              <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-1">Nodes by Type</div>
              {Object.entries(stats.nodeTypeCounts).map(([type, count]) => (
                <div key={type} className="flex items-center gap-1.5 py-0.5">
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: NODE_COLORS[type as keyof typeof NODE_COLORS] ?? "#888" }} />
                  <span className="text-nexus-text-primary">{NODE_TYPE_LABELS[type as keyof typeof NODE_TYPE_LABELS] ?? type}</span>
                  <span className="ml-auto text-nexus-text-muted font-mono">{count}</span>
                </div>
              ))}
            </div>

            {/* Edge type breakdown */}
            <div>
              <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-1">Edges by Type</div>
              {Object.entries(stats.edgeTypeCounts)
                .sort(([, a], [, b]) => b - a)
                .map(([type, count]) => (
                <div key={type} className="flex items-center gap-1.5 py-0.5">
                  <span className="w-3 h-0.5 rounded shrink-0" style={{ backgroundColor: EDGE_STYLES[type]?.color ?? "#888" }} />
                  <span className="text-nexus-text-primary">{EDGE_TYPE_LABELS[type] ?? type}</span>
                  <span className="ml-auto text-nexus-text-muted font-mono">{count}</span>
                </div>
              ))}
            </div>

            {/* Top connected */}
            <div>
              <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-1">Most Connected</div>
              {stats.topConnected.map((n, i) => (
                <div key={n.id}
                  className="flex items-center gap-1.5 py-0.5 cursor-pointer hover:bg-nexus-surface-hover rounded px-1 -mx-1"
                  onClick={() => setSelectedNodeId(n.id)}
                >
                  <span className="text-nexus-text-muted w-3">{i + 1}.</span>
                  <span className="text-nexus-text-primary truncate">{n.label}</span>
                  <span className="ml-auto text-nexus-text-muted font-mono">{n.degree}</span>
                </div>
              ))}
            </div>

            {/* Per-speaker signal counts */}
            <div>
              <div className="text-nexus-text-secondary font-semibold uppercase tracking-wide mb-1">Signals by Speaker</div>
              {Object.entries(stats.speakerSignalCounts).map(([spk, count], i) => (
                <div key={spk} className="flex items-center gap-1.5 py-0.5">
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: SPEAKER_COLORS[i] }} />
                  <span className="text-nexus-text-primary">{speakerRoles[spk] || spk}</span>
                  <span className="ml-auto text-nexus-text-muted font-mono">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── LEGEND BAR ── */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-1.5 border-t border-nexus-border bg-nexus-surface shrink-0">
        {Object.entries(NODE_TYPE_LABELS).map(([type, label]) => (
          <span key={type} className="flex items-center gap-1 text-[9px] text-nexus-text-secondary">
            <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: NODE_COLORS[type as keyof typeof NODE_COLORS] }} />
            {label}
          </span>
        ))}
        <span className="text-[9px] text-nexus-text-muted">|</span>
        {Object.entries(EDGE_TYPE_LABELS).filter(([k]) => k !== "speaker_produced").map(([type, label]) => (
          <span key={type} className="flex items-center gap-1 text-[9px] text-nexus-text-secondary">
            <span className="w-3 shrink-0 inline-block" style={{
              height: EDGE_STYLES[type]?.width ?? 1,
              backgroundColor: EDGE_STYLES[type]?.color ?? "#888",
              borderStyle: EDGE_STYLES[type]?.dashed ? "dashed" : "solid",
            }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
