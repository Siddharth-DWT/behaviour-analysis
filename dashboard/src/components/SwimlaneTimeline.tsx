/**
 * SwimlaneTimeline — Gantt-chart-style conversation visualisation.
 *
 * Each speaker gets a horizontal row.  Speech blocks are drawn at their
 * real timestamps and scaled by duration.  Noteworthy signals are embedded
 * inside the blocks as colour-coded dots.  A fusion-events row runs along
 * the bottom.  Supports zoom (wheel), click-to-seek, hover tooltips, and
 * speaker/signal filtering.
 */
import { useRef, useState, useMemo, useCallback, useEffect } from "react";
import * as d3 from "d3";
import type { Signal, TranscriptSegment } from "../api/client";
import {
  buildSwimlaneData,
  type SwimlaneData,
  type SpeechBlock,
  type BlockSignal,
  type FusionEvent,
} from "../utils/buildSwimlaneData";
import TranscriptView from "./TranscriptView";

/* ------------------------------------------------------------------ */
/* Props                                                               */
/* ------------------------------------------------------------------ */

interface Props {
  segments: TranscriptSegment[];
  signals: Signal[];
  durationMs: number;
  entities: Record<string, unknown>;
  speakerRoles: Record<string, string>;
  onSeek?: (ms: number) => void;
  onSegmentClick?: (segmentId: string) => void;
  /** Hide the Timeline/Transcript toggle — always show timeline only */
  hideTranscriptToggle?: boolean;
}

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const LABEL_W = 100;
const HEADER_H = 24;
const TOPIC_H = 28;
const LANE_H = 56;
const LANE_GAP = 3;
const FUSION_ROW_H = 28;
const MIN_BLOCK_PX = 4;

const BORDER_COLORS: Record<string, string> = {
  stress: "#EF4444",
  objection: "#EF4444",
  buying: "#22C55E",
  fusion: "#F97316",
  sentiment: "#8B5CF6",
  normal: "#4B5563",
};

const SIGNAL_DOTS: Record<string, { color: string; symbol: string }> = {
  stress:    { color: "#EF4444", symbol: "S" },
  buying:    { color: "#22C55E", symbol: "B" },
  objection: { color: "#EF4444", symbol: "O" },
  fusion:    { color: "#F97316", symbol: "!" },
  sentiment: { color: "#8B5CF6", symbol: "~" },
  filler:    { color: "#6B7280", symbol: "F" },
  pitch:     { color: "#F59E0B", symbol: "P" },
  tone:      { color: "#3B82F6", symbol: "T" },
  rate:      { color: "#F59E0B", symbol: "R" },
};

const SEVERITY_COLORS: Record<string, string> = {
  critical: "#EF4444",
  warning: "#F97316",
  positive: "#22C55E",
  info: "#3B82F6",
};

const ZOOM_PRESETS = [
  { label: "Full", ms: Infinity },
  { label: "5 min", ms: 300_000 },
  { label: "1 min", ms: 60_000 },
  { label: "30 s", ms: 30_000 },
];

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function fmtMs(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function SwimlaneTimeline({
  segments,
  signals,
  durationMs,
  entities,
  speakerRoles,
  onSeek,
  onSegmentClick,
  hideTranscriptToggle = false,
}: Props) {
  const [viewMode, setViewMode] = useState<"timeline" | "transcript">("timeline");
  const containerRef = useRef<HTMLDivElement>(null);
  const [chartW, setChartW] = useState(900);
  const [viewRange, setViewRange] = useState<[number, number]>([0, durationMs]);
  const [hoveredBlock, setHoveredBlock] = useState<{ block: SpeechBlock; spkId: string; x: number; y: number } | null>(null);
  const [selectedBlock, setSelectedBlock] = useState<{ block: SpeechBlock; spkId: string } | null>(null);
  const [hiddenSpeakers, setHiddenSpeakers] = useState<Set<string>>(new Set());
  const [hiddenSignalTypes, setHiddenSignalTypes] = useState<Set<string>>(new Set());

  // Responsive width
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(([e]) => setChartW(Math.max(500, e.contentRect.width)));
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => { setViewRange([0, durationMs]); }, [durationMs]);

  // Layout calculations
  const timelineW = chartW - LABEL_W - 8;

  // Native wheel listener with { passive: false } so preventDefault works
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      const rect = el.getBoundingClientRect();
      const ry = e.clientY - rect.top;
      if (ry < 0) return;
      e.preventDefault();
      e.stopPropagation();
      const tw = rect.width - LABEL_W - 8;
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left - LABEL_W) / tw));
      setViewRange(([lo, hi]) => {
        const span = hi - lo;
        const factor = e.deltaY > 0 ? 1.3 : 0.75;
        const newSpan = Math.min(durationMs, Math.max(5000, span * factor));
        const pivot = lo + frac * span;
        const nLo = Math.max(0, pivot - frac * newSpan);
        const nHi = Math.min(durationMs, nLo + newSpan);
        return [nLo, nHi];
      });
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [durationMs, chartW]);

  // Build structured data
  const data: SwimlaneData = useMemo(
    () => buildSwimlaneData(segments, signals, entities, speakerRoles, durationMs),
    [segments, signals, entities, speakerRoles, durationMs]
  );

  const visibleSpeakers = data.speakers.filter((s) => !hiddenSpeakers.has(s.id));
  const xScale = useMemo(
    () => d3.scaleLinear().domain(viewRange).range([0, timelineW]),
    [viewRange, timelineW]
  );

  const bodyH = TOPIC_H + visibleSpeakers.length * (LANE_H + LANE_GAP) + FUSION_ROW_H;
  const totalH = HEADER_H + bodyH + 8;

  // Time ticks
  const ticks = xScale.ticks(Math.max(4, Math.floor(timelineW / 90)));

  // Pan via arrow keys
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    const [lo, hi] = viewRange;
    const span = hi - lo;
    const step = span * 0.2;
    if (e.key === "ArrowRight") {
      const nLo = Math.min(durationMs - span, lo + step);
      setViewRange([nLo, nLo + span]);
    } else if (e.key === "ArrowLeft") {
      const nLo = Math.max(0, lo - step);
      setViewRange([nLo, nLo + span]);
    }
  }, [viewRange, durationMs]);

  const handleZoomPreset = (ms: number) => {
    if (ms === Infinity) { setViewRange([0, durationMs]); return; }
    const center = (viewRange[0] + viewRange[1]) / 2;
    const lo = Math.max(0, center - ms / 2);
    const hi = Math.min(durationMs, lo + ms);
    setViewRange([lo, hi]);
  };

  const handleBgClick = useCallback((e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const rx = e.clientX - rect.left - LABEL_W;
    if (rx < 0) return;
    onSeek?.(Math.round(Math.max(0, Math.min(durationMs, xScale.invert(rx)))));
  }, [xScale, onSeek, durationMs]);

  const toggleSpeaker = (id: string) => {
    setHiddenSpeakers((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const toggleSignalType = (type: string) => {
    setHiddenSignalTypes((prev) => {
      const next = new Set(prev);
      next.has(type) ? next.delete(type) : next.add(type);
      return next;
    });
  };

  if (segments.length === 0 || durationMs <= 0) return null;

  // Y positions for visible speakers
  const laneYs = visibleSpeakers.map((_, i) => HEADER_H + TOPIC_H + i * (LANE_H + LANE_GAP));
  const fusionRowY = HEADER_H + TOPIC_H + visibleSpeakers.length * (LANE_H + LANE_GAP);
  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
      {/* Header bar with view toggle */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-nexus-text-primary">
          {viewMode === "timeline" ? "Conversation Timeline" : "Transcript View"}
        </h3>
        <div className="flex items-center gap-3">
          {/* View toggle (hidden when used on Insights tab) */}
          {!hideTranscriptToggle && (
            <div className="flex overflow-hidden rounded-full border border-nexus-border" style={{ height: 28 }}>
              <button
                onClick={() => setViewMode("timeline")}
                className={`px-3 text-[11px] font-medium transition-colors ${
                  viewMode === "timeline"
                    ? "bg-blue-600 text-white"
                    : "bg-transparent text-nexus-text-muted hover:text-nexus-text-primary"
                }`}
              >
                Timeline Graph
              </button>
              <button
                onClick={() => setViewMode("transcript")}
                className={`px-3 text-[11px] font-medium transition-colors ${
                  viewMode === "transcript"
                    ? "bg-blue-600 text-white"
                    : "bg-transparent text-nexus-text-muted hover:text-nexus-text-primary"
                }`}
              >
                Transcript View
              </button>
            </div>
          )}

          {/* Zoom presets (only show in timeline mode) */}
          {viewMode === "timeline" && (
            <div className="flex rounded border border-nexus-border text-[10px]">
              {ZOOM_PRESETS.map((z) => (
                <button
                  key={z.label}
                  onClick={() => handleZoomPreset(z.ms)}
                  className="px-2 py-0.5 text-nexus-text-muted hover:bg-nexus-surface-hover hover:text-nexus-text-primary transition-colors first:rounded-l last:rounded-r"
                >
                  {z.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Transcript View */}
      {viewMode === "transcript" && (
        <TranscriptView
          segments={segments}
          signals={signals}
          speakerRoles={speakerRoles}
          durationMs={durationMs}
        />
      )}

      {/* Timeline View — only render DOM when active */}
      {viewMode === "timeline" && (
      <>

      {/* Filters */}
      <div className="mb-2 flex flex-wrap gap-3">
        {/* Speaker toggles */}
        <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
          <span className="text-nexus-text-muted mr-1">Speakers:</span>
          {data.speakers.map((s) => (
            <button
              key={s.id}
              onClick={() => toggleSpeaker(s.id)}
              className={`flex items-center gap-1 rounded-full px-2 py-0.5 border transition-colors ${
                hiddenSpeakers.has(s.id)
                  ? "border-nexus-border text-nexus-text-muted opacity-40"
                  : "border-transparent text-nexus-text-primary"
              }`}
              style={{ backgroundColor: hiddenSpeakers.has(s.id) ? "transparent" : s.color + "22" }}
            >
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: s.color }} />
              {s.name !== s.label ? s.name : s.label}
              {s.role && <span className="opacity-60">({s.role})</span>}
            </button>
          ))}
        </div>
        {/* Signal type toggles */}
        <div className="flex flex-wrap items-center gap-1 text-[10px]">
          <span className="text-nexus-text-muted mr-1">Signals:</span>
          {Object.entries(SIGNAL_DOTS).map(([type, cfg]) => (
            <button
              key={type}
              onClick={() => toggleSignalType(type)}
              className={`rounded px-1.5 py-0.5 transition-opacity ${
                hiddenSignalTypes.has(type) ? "opacity-30" : "opacity-100"
              }`}
              style={{ backgroundColor: cfg.color + "22", color: cfg.color, border: `1px solid ${cfg.color}44` }}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* SVG canvas */}
      <div ref={containerRef} className="relative overflow-hidden" tabIndex={0} onKeyDown={handleKeyDown}>
        {/* ── Fixed speaker labels (HTML, left column) ── */}
        <div className="absolute left-0 top-0 z-10 pointer-events-none" style={{ width: LABEL_W }}>
          {/* Topic label */}
          {data.topics.length > 0 && (
            <div
              className="flex items-center justify-end pr-2 text-[9px] text-nexus-text-muted"
              style={{ height: LANE_H * 0.5, marginTop: HEADER_H }}
            >
              Topics
            </div>
          )}
          {/* Speaker labels */}
          {visibleSpeakers.map((spk, li) => (
            <div
              key={spk.id}
              className="flex flex-col items-end justify-center pr-2"
              style={{
                height: LANE_H,
                marginTop: li === 0 ? (data.topics.length > 0 ? TOPIC_H - LANE_H * 0.5 : HEADER_H) : LANE_GAP,
              }}
            >
              <span className="text-[11px] font-semibold text-nexus-text-primary truncate max-w-full text-right">
                {spk.name !== spk.label ? spk.name : spk.label}
              </span>
              {spk.role && (
                <span className="text-[9px] truncate max-w-full text-right" style={{ color: spk.color }}>
                  {spk.role}
                </span>
              )}
              <span className="text-[9px] text-nexus-text-muted">
                {Math.round(spk.talkTimePct)}% talk
              </span>
            </div>
          ))}
          {/* Fusion label */}
          <div
            className="flex items-center justify-end pr-2 text-[9px] text-nexus-text-muted"
            style={{ height: FUSION_ROW_H, marginTop: LANE_GAP }}
          >
            Fusion
          </div>
        </div>

        <svg width={chartW} height={totalH} className="select-none">

          {/* ── Time axis ── */}
          <g transform={`translate(${LABEL_W}, 0)`}>
            {ticks.map((t) => (
              <g key={t} transform={`translate(${xScale(t)}, 0)`}>
                <line y1={HEADER_H} y2={totalH - 8} stroke="#2D3348" strokeWidth={0.5} strokeDasharray="2,4" />
                <text y={HEADER_H - 6} textAnchor="middle" fontSize={9} fill="#565E73">{fmtMs(t)}</text>
              </g>
            ))}
          </g>

          {/* ── Topic bar ── */}
          {data.topics.length > 0 && (
            <g transform={`translate(${LABEL_W}, ${HEADER_H})`}>
              {/* Topic label rendered in HTML overlay */}
              {data.topics.map((t, i) => {
                const tx = xScale(t.start_ms);
                const tw = Math.max(2, xScale(t.end_ms) - tx);
                if (tx + tw < 0 || tx > timelineW) return null;
                return (
                  <g key={i}>
                    <rect x={tx} y={4} width={tw} height={TOPIC_H - 8} rx={3} fill={t.color} opacity={0.25} />
                    {tw > 40 && (
                      <text x={tx + 6} y={TOPIC_H / 2} dominantBaseline="central" fontSize={8} fill={t.color}>
                        {t.name.length > Math.floor(tw / 5) ? t.name.slice(0, Math.floor(tw / 5)) + "..." : t.name}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          )}

          {/* ── Speaker lanes ── */}
          {visibleSpeakers.map((spk, li) => {
            const ly = laneYs[li];
            return (
              <g key={spk.id}>
                {/* Lane bg */}
                <rect
                  x={LABEL_W} y={ly} width={timelineW} height={LANE_H}
                  fill={li % 2 === 0 ? "#1A1D27" : "#1E2130"} rx={2}
                  onClick={handleBgClick} className="cursor-pointer"
                />

                {/* Speech blocks */}
                {spk.blocks.map((blk, bi) => {
                  const bx = xScale(blk.start_ms) + LABEL_W;
                  const bw = Math.max(MIN_BLOCK_PX, xScale(blk.end_ms) - xScale(blk.start_ms));
                  if (bx + bw < LABEL_W || bx > chartW) return null;

                  const isSelected = selectedBlock?.block === blk;
                  const borderColor = BORDER_COLORS[blk.borderType] || BORDER_COLORS.normal;
                  const visibleSigs = blk.signals.filter((s) => !hiddenSignalTypes.has(s.type));

                  return (
                    <g key={bi}>
                      {/* Block body */}
                      <rect
                        x={bx} y={ly + 3} width={bw} height={LANE_H - 6} rx={3}
                        fill={spk.color} opacity={isSelected ? 0.35 : 0.2}
                        stroke={isSelected ? "#E8ECF4" : "transparent"} strokeWidth={isSelected ? 1.5 : 0}
                        className="cursor-pointer transition-opacity duration-100 hover:opacity-30"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedBlock(isSelected ? null : { block: blk, spkId: spk.id });
                          onSeek?.(blk.start_ms);
                          if (blk.segmentIds[0]) onSegmentClick?.(blk.segmentIds[0]);
                        }}
                        onMouseEnter={(e) => setHoveredBlock({ block: blk, spkId: spk.id, x: e.clientX, y: e.clientY })}
                        onMouseLeave={() => setHoveredBlock(null)}
                      />
                      {/* Left border accent */}
                      <rect x={bx} y={ly + 3} width={3} height={LANE_H - 6} rx={1} fill={borderColor} />

                      {/* Text inside */}
                      {bw > 50 && (
                        <text
                          x={bx + 7} y={ly + 20} dominantBaseline="central"
                          fontSize={9} fill="#E8ECF4" opacity={0.75}
                        >
                          {blk.text.length > Math.floor(bw / 5.5) - 2
                            ? blk.text.slice(0, Math.floor(bw / 5.5) - 2) + "..."
                            : blk.text}
                        </text>
                      )}

                      {/* Signal dots at bottom of block */}
                      {visibleSigs.slice(0, 5).map((sig, di) => {
                        const dot = SIGNAL_DOTS[sig.type];
                        if (!dot) return null;
                        return (
                          <g key={di} transform={`translate(${bx + 8 + di * 16}, ${ly + LANE_H - 10})`}>
                            <circle r={5} fill={dot.color} opacity={0.85} stroke="#1A1D27" strokeWidth={1} />
                            <text textAnchor="middle" dominantBaseline="central" fontSize={6} fill="white" fontWeight={700}>
                              {dot.symbol}
                            </text>
                          </g>
                        );
                      })}
                      {visibleSigs.length > 5 && (
                        <text
                          x={bx + 8 + 5 * 16} y={ly + LANE_H - 10}
                          dominantBaseline="central" fontSize={7} fill="#8B93A7"
                        >
                          +{visibleSigs.length - 5}
                        </text>
                      )}
                    </g>
                  );
                })}

                {/* Lane bottom border */}
                <line x1={LABEL_W} x2={chartW - 8} y1={ly + LANE_H} y2={ly + LANE_H} stroke="#2D3348" strokeWidth={0.5} />
              </g>
            );
          })}

          {/* ── Fusion events row ── */}
          <g transform={`translate(0, ${fusionRowY})`}>
            {/* Fusion label rendered in HTML overlay */}
            <rect x={LABEL_W} y={0} width={timelineW} height={FUSION_ROW_H} fill="#16181F" rx={2} />
            {data.fusion_events.map((ev, i) => {
              const ex = xScale(ev.timestamp_ms) + LABEL_W;
              if (ex < LABEL_W || ex > chartW) return null;
              const col = SEVERITY_COLORS[ev.severity] || SEVERITY_COLORS.info;
              return (
                <g key={i} transform={`translate(${ex}, ${FUSION_ROW_H / 2})`} className="cursor-pointer">
                  {/* Diamond shape */}
                  <rect x={-5} y={-5} width={10} height={10} rx={1} fill={col} opacity={0.8}
                    transform="rotate(45)" />
                  <title>{ev.label} ({fmtMs(ev.timestamp_ms)})</title>
                </g>
              );
            })}
          </g>

        </svg>

        {/* ── Navigation slider (HTML, below SVG) ── */}
        {(() => {
          const span = viewRange[1] - viewRange[0];
          const isZoomed = span < durationMs - 100;
          if (!isZoomed) return null;

          const thumbLeftPct = (viewRange[0] / durationMs) * 100;
          const thumbWidthPct = (span / durationMs) * 100;

          return (
            <div className="mt-1" style={{ paddingLeft: LABEL_W }}>
              <div className="flex items-center gap-2">
                {/* Left arrow */}
                <button
                  onClick={() => {
                    const step = span * 0.3;
                    const nLo = Math.max(0, viewRange[0] - step);
                    setViewRange([nLo, nLo + span]);
                  }}
                  className="flex h-5 w-5 items-center justify-center rounded text-[10px] text-nexus-text-muted hover:bg-nexus-surface-hover hover:text-nexus-text-primary"
                >
                  ◄
                </button>

                {/* Track with draggable thumb */}
                <div
                  className="relative h-3 flex-1 cursor-pointer rounded-full bg-[#2D3348]"
                  onMouseDown={(e) => {
                    const track = e.currentTarget;
                    const rect = track.getBoundingClientRect();
                    const startX = e.clientX;
                    const startLo = viewRange[0];

                    const onMove = (me: MouseEvent) => {
                      const dx = me.clientX - startX;
                      const dtMs = (dx / rect.width) * durationMs;
                      const nLo = Math.max(0, Math.min(durationMs - span, startLo + dtMs));
                      setViewRange([nLo, nLo + span]);
                    };
                    const onUp = () => {
                      document.removeEventListener("mousemove", onMove);
                      document.removeEventListener("mouseup", onUp);
                    };
                    document.addEventListener("mousemove", onMove);
                    document.addEventListener("mouseup", onUp);
                  }}
                >
                  {/* Thumb */}
                  <div
                    className="absolute top-0 h-full rounded-full bg-[#4F8BFF] opacity-60 hover:opacity-80 transition-opacity"
                    style={{
                      left: `${thumbLeftPct}%`,
                      width: `${Math.max(2, thumbWidthPct)}%`,
                    }}
                  />
                  {/* Tick marks */}
                  {[0, 25, 50, 75, 100].map((pct) => (
                    <div
                      key={pct}
                      className="absolute top-0 h-full w-px bg-[#565E73] opacity-30"
                      style={{ left: `${pct}%` }}
                    />
                  ))}
                </div>

                {/* Right arrow */}
                <button
                  onClick={() => {
                    const step = span * 0.3;
                    const nLo = Math.min(durationMs - span, viewRange[0] + step);
                    setViewRange([nLo, nLo + span]);
                  }}
                  className="flex h-5 w-5 items-center justify-center rounded text-[10px] text-nexus-text-muted hover:bg-nexus-surface-hover hover:text-nexus-text-primary"
                >
                  ►
                </button>
              </div>

              {/* Time labels */}
              <div className="mt-0.5 flex justify-between text-[8px] text-nexus-text-muted" style={{ paddingLeft: 20, paddingRight: 20 }}>
                <span>{fmtMs(viewRange[0])}</span>
                <span>{fmtMs((viewRange[0] + viewRange[1]) / 2)}</span>
                <span>{fmtMs(viewRange[1])}</span>
              </div>
            </div>
          );
        })()}

        {/* ── Hover tooltip ── */}
        {hoveredBlock && (
          <div
            className="pointer-events-none fixed z-50 max-w-sm rounded-lg border border-nexus-border bg-nexus-surface px-3 py-2 shadow-xl"
            style={{ left: hoveredBlock.x + 14, top: hoveredBlock.y - 20 }}
          >
            <div className="mb-1 flex items-center gap-2 text-[10px] text-nexus-text-muted">
              <span>{fmtMs(hoveredBlock.block.start_ms)} - {fmtMs(hoveredBlock.block.end_ms)}</span>
              <span className="font-medium">{hoveredBlock.spkId}</span>
            </div>
            <p className="text-xs leading-relaxed text-nexus-text-primary line-clamp-4">
              {hoveredBlock.block.text}
            </p>
            {hoveredBlock.block.signals.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {hoveredBlock.block.signals.map((s, i) => {
                  const dot = SIGNAL_DOTS[s.type];
                  return (
                    <span
                      key={i}
                      className="inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[9px] font-medium"
                      style={{
                        backgroundColor: (dot?.color || "#666") + "22",
                        color: dot?.color || "#888",
                        border: `1px solid ${dot?.color || "#444"}44`,
                      }}
                    >
                      {s.label}
                    </span>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Selected block detail panel ── */}
      {selectedBlock && (
        <div className="mt-3 rounded-lg border border-nexus-border bg-nexus-bg p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-nexus-text-muted">
              <span className="font-medium text-nexus-text-primary">{selectedBlock.spkId}</span>
              <span>{fmtMs(selectedBlock.block.start_ms)} - {fmtMs(selectedBlock.block.end_ms)}</span>
            </div>
            <button
              onClick={() => setSelectedBlock(null)}
              className="text-xs text-nexus-text-muted hover:text-nexus-text-primary"
            >
              Close
            </button>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-nexus-text-primary">
            {selectedBlock.block.text}
          </p>
          {selectedBlock.block.signals.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {selectedBlock.block.signals.map((s, i) => {
                const dot = SIGNAL_DOTS[s.type];
                return (
                  <div
                    key={i}
                    className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs"
                    style={{ backgroundColor: (dot?.color || "#666") + "15", border: `1px solid ${dot?.color || "#444"}33` }}
                  >
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: dot?.color }} />
                    <span style={{ color: dot?.color }}>{s.label}</span>
                    <span className="text-nexus-text-muted">conf: {Math.round(s.confidence * 100)}%</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ── Legend ── */}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[9px] text-nexus-text-muted">
        <span className="font-medium">Block borders:</span>
        {Object.entries(BORDER_COLORS).filter(([k]) => k !== "normal").map(([k, c]) => (
          <span key={k} className="flex items-center gap-1">
            <span className="inline-block h-2 w-1 rounded-sm" style={{ backgroundColor: c }} />
            {k}
          </span>
        ))}
        <span className="ml-2 font-medium">Signal dots:</span>
        {Object.entries(SIGNAL_DOTS).map(([k, v]) => (
          <span key={k} className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: v.color }} />
            {v.symbol}={k}
          </span>
        ))}
      </div>
      </>
      )}
    </div>
  );
}
