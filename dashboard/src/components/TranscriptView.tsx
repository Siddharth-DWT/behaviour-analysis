/**
 * TranscriptView — Chat-style transcript with signal minimap sidebar.
 *
 * Shows diarised transcript as alternating chat bubbles with embedded
 * signal badges and a vertical minimap for quick navigation.
 */
import { useRef, useState, useCallback, useEffect, useMemo } from "react";
import type { Signal, TranscriptSegment } from "../api/client";

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const SPEAKER_STYLES: { bg: string; border: string; text: string }[] = [
  { bg: "#e6f1fb", border: "#3266ad", text: "#3266ad" }, // blue
  { bg: "#eaf3de", border: "#639922", text: "#639922" }, // green
  { bg: "#faeeda", border: "#ba7517", text: "#ba7517" }, // amber
  { bg: "#fcebeb", border: "#a32d2d", text: "#a32d2d" }, // red
  { bg: "#eeedfe", border: "#534ab7", text: "#534ab7" }, // purple
];

const SIGNAL_COLORS: Record<string, string> = {
  stress: "#e24b4a",
  buying: "#378add",
  objection: "#ef9f27",
  sentiment: "#63991a",
  fusion: "#ef9f27",
  pitch: "#7f77dd",
  tone: "#1d9e75",
  filler: "#888",
  rate: "#7f77dd",
};

const FILTER_TYPES = ["All", "Stress", "Buying", "Objection", "Sentiment", "Fusion"] as const;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function fmtMs(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function signalCategory(sig: Signal): string {
  const t = sig.signal_type;
  if (t.includes("stress")) return "stress";
  if (t.includes("buying")) return "buying";
  if (t.includes("objection")) return "objection";
  if (t.includes("sentiment")) return "sentiment";
  if (t.includes("pitch")) return "pitch";
  if (t.includes("tone")) return "tone";
  if (t.includes("filler")) return "filler";
  if (t.includes("rate")) return "rate";
  if (["tension_cluster", "momentum_shift", "persistent_incongruence", "verbal_incongruence"].includes(t))
    return "fusion";
  return "other";
}

function isNoteworthy(sig: Signal): boolean {
  const cat = signalCategory(sig);
  const v = sig.value ?? 0;
  if (cat === "stress" && v <= 0.3) return false;
  if (cat === "sentiment" && Math.abs(v) <= 0.35) return false;
  if (cat === "other") return false;
  return true;
}

function matchSignalsToSegment(seg: TranscriptSegment, signals: Signal[]): Signal[] {
  const raw = signals.filter(
    (s) =>
      isNoteworthy(s) &&
      (s.speaker_label === seg.speaker_label || s.speaker_label === seg.speaker_id) &&
      (s.window_start_ms ?? 0) <= seg.end_ms &&
      (s.window_end_ms ?? s.window_start_ms ?? 0) >= seg.start_ms
  );

  // Deduplicate: keep only the highest-value signal per category
  const bestByCategory = new Map<string, Signal>();
  for (const s of raw) {
    const cat = signalCategory(s);
    const existing = bestByCategory.get(cat);
    if (!existing || Math.abs(s.value ?? 0) > Math.abs(existing.value ?? 0)) {
      bestByCategory.set(cat, s);
    }
  }
  return Array.from(bestByCategory.values());
}

/* ------------------------------------------------------------------ */
/* Props                                                               */
/* ------------------------------------------------------------------ */

interface Props {
  segments: TranscriptSegment[];
  signals: Signal[];
  speakerRoles?: Record<string, string>;
  durationMs: number;
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function TranscriptView({ segments, signals, speakerRoles, durationMs }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const minimapRef = useRef<HTMLDivElement>(null);
  const [search, setSearch] = useState("");
  const [activeFilter, setActiveFilter] = useState<string>("All");
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [viewportTop, setViewportTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(20);

  // Speaker ordering
  const speakerOrder = useMemo(() => {
    const seen: string[] = [];
    for (const seg of segments) {
      const spk = seg.speaker_label || seg.speaker_id || "Unknown";
      if (!seen.includes(spk)) seen.push(spk);
    }
    return seen;
  }, [segments]);

  const getSpeakerIdx = (spk: string) => {
    const idx = speakerOrder.indexOf(spk);
    return idx >= 0 ? idx : 0;
  };

  // Pre-compute signals per segment
  const segmentSignals = useMemo(
    () => segments.map((seg) => matchSignalsToSegment(seg, signals)),
    [segments, signals]
  );

  // Noteworthy signals for minimap
  const minimapSignals = useMemo(
    () => signals.filter(isNoteworthy).sort((a, b) => (a.window_start_ms ?? 0) - (b.window_start_ms ?? 0)),
    [signals]
  );

  // Filtered segments
  const filteredIndices = useMemo(() => {
    return segments.map((seg, i) => {
      // Search filter
      if (search && !seg.text.toLowerCase().includes(search.toLowerCase())) return false;
      // Signal type filter
      if (activeFilter !== "All") {
        const cat = activeFilter.toLowerCase();
        const hasCat = segmentSignals[i].some((s) => signalCategory(s) === cat);
        if (!hasCat) return false;
      }
      return true;
    });
  }, [segments, segmentSignals, search, activeFilter]);

  // Scroll sync → minimap viewport
  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    const mapH = minimapRef.current?.clientHeight || 300;
    setViewportTop((scrollTop / scrollHeight) * mapH);
    setViewportHeight((clientHeight / scrollHeight) * mapH);
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.addEventListener("scroll", handleScroll);
    handleScroll();
    return () => el.removeEventListener("scroll", handleScroll);
  }, [handleScroll, segments]);

  // Click minimap dot → scroll to segment
  const scrollToTime = useCallback(
    (ms: number) => {
      const el = scrollRef.current;
      if (!el) return;
      // Find nearest segment
      let bestIdx = 0;
      let bestDist = Infinity;
      segments.forEach((seg, i) => {
        const dist = Math.abs(seg.start_ms - ms);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      });
      const bubble = el.querySelector(`[data-seg-idx="${bestIdx}"]`);
      if (bubble) bubble.scrollIntoView({ behavior: "smooth", block: "center" });
    },
    [segments]
  );

  if (segments.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-nexus-border bg-nexus-surface text-sm text-nexus-text-muted">
        No transcript available.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface flex flex-col" style={{ height: 520 }}>
      {/* ── A) Toolbar ── */}
      <div className="flex flex-wrap items-center gap-2 border-b border-nexus-border px-3 py-2">
        <input
          type="text"
          placeholder="Search transcript..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-7 w-48 rounded border border-nexus-border bg-nexus-bg px-2 text-xs text-nexus-text-primary placeholder:text-nexus-text-muted focus:border-blue-500 focus:outline-none"
        />
        <div className="flex gap-1">
          {FILTER_TYPES.map((f) => (
            <button
              key={f}
              onClick={() => setActiveFilter(f)}
              className={`rounded-full px-2.5 py-0.5 text-[10px] font-medium transition-colors ${
                activeFilter === f
                  ? "bg-blue-600 text-white"
                  : "bg-nexus-bg text-nexus-text-muted hover:text-nexus-text-primary border border-nexus-border"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
        <span className="ml-auto text-[10px] text-nexus-text-muted">
          {filteredIndices.filter(Boolean).length} / {segments.length} segments
        </span>
      </div>

      {/* ── B) Main area ── */}
      <div className="flex flex-1 min-h-0">
        {/* Signal minimap */}
        <div
          ref={minimapRef}
          className="relative w-[52px] shrink-0 border-r border-nexus-border bg-[#16181F] cursor-pointer"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const frac = (e.clientY - rect.top) / rect.height;
            scrollToTime(frac * durationMs);
          }}
        >
          {/* Signal dots */}
          {minimapSignals.map((sig, i) => {
            const top = ((sig.window_start_ms ?? 0) / Math.max(durationMs, 1)) * 100;
            const cat = signalCategory(sig);
            const color = SIGNAL_COLORS[cat] || "#666";
            return (
              <div
                key={i}
                className="absolute left-1/2 -translate-x-1/2 h-[5px] w-[5px] rounded-full"
                style={{ top: `${top}%`, backgroundColor: color }}
                title={`${sig.signal_type} ${fmtMs(sig.window_start_ms ?? 0)}`}
              />
            );
          })}
          {/* Viewport rectangle */}
          <div
            className="absolute left-1 right-1 rounded border border-blue-400/50 bg-blue-500/15 pointer-events-none"
            style={{ top: viewportTop, height: Math.max(8, viewportHeight) }}
          />
        </div>

        {/* Transcript scroll area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5">
          {segments.map((seg, i) => {
            if (!filteredIndices[i]) return null;

            const spk = seg.speaker_label || seg.speaker_id || "Unknown";
            const spkIdx = getSpeakerIdx(spk);
            const style = SPEAKER_STYLES[spkIdx % SPEAKER_STYLES.length];
            const isLeft = spkIdx % 2 === 0;
            const sigs = segmentSignals[i];
            const hasHighStress = sigs.some(
              (s) => signalCategory(s) === "stress" && (s.value ?? 0) > 0.55
            );
            const isSelected = selectedIdx === i;
            const role = speakerRoles?.[spk];

            return (
              <div
                key={i}
                data-seg-idx={i}
                className={`flex ${isLeft ? "justify-start" : "justify-end"}`}
              >
                <div
                  className={`max-w-[75%] rounded-lg px-3 py-1.5 cursor-pointer transition-all ${
                    isSelected ? "ring-2 ring-blue-400" : ""
                  }`}
                  style={{
                    backgroundColor: hasHighStress ? "#332211" : style.bg + "18",
                    borderLeft: isLeft ? `3px solid ${style.border}` : "none",
                    borderRight: !isLeft ? `3px solid ${style.border}` : "none",
                  }}
                  onClick={() => setSelectedIdx(isSelected ? null : i)}
                >
                  {/* Header: speaker + time */}
                  <div className={`flex items-center gap-2 mb-0.5 ${isLeft ? "" : "justify-end"}`}>
                    <span className="text-[10px] font-semibold" style={{ color: style.text }}>
                      {spk}
                      {role && <span className="font-normal opacity-70"> ({role})</span>}
                    </span>
                    <span className="text-[9px] text-nexus-text-muted">{fmtMs(seg.start_ms)}</span>
                  </div>

                  {/* Signal pills (max 2) */}
                  {sigs.length > 0 && (
                    <div className={`flex gap-1 mb-0.5 flex-wrap ${isLeft ? "" : "justify-end"}`}>
                      {sigs.slice(0, 2).map((s, si) => {
                        const cat = signalCategory(s);
                        const color = SIGNAL_COLORS[cat] || "#666";
                        return (
                          <span
                            key={si}
                            className="inline-flex items-center gap-0.5 rounded px-1.5 py-0 text-[8px] font-medium"
                            style={{
                              backgroundColor: color + "22",
                              color,
                              border: `1px solid ${color}44`,
                            }}
                          >
                            {cat}
                            {s.value != null && ` ${Math.round((s.value as number) * 100)}%`}
                          </span>
                        );
                      })}
                      {sigs.length > 2 && (
                        <span className="text-[8px] text-nexus-text-muted">+{sigs.length - 2}</span>
                      )}
                    </div>
                  )}

                  {/* Text */}
                  <p className="text-[11px] leading-relaxed text-nexus-text-primary">{seg.text}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── C) Detail panel ── */}
      <div className="shrink-0 border-t border-nexus-border px-3 py-2" style={{ minHeight: 60 }}>
        {selectedIdx !== null && segmentSignals[selectedIdx]?.length > 0 ? (
          <div>
            <div className="mb-1 text-[10px] text-nexus-text-muted">
              Signals for {segments[selectedIdx].speaker_label || "Unknown"} at{" "}
              {fmtMs(segments[selectedIdx].start_ms)} - {fmtMs(segments[selectedIdx].end_ms)}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {segmentSignals[selectedIdx].map((s, i) => {
                const cat = signalCategory(s);
                const color = SIGNAL_COLORS[cat] || "#666";
                return (
                  <div
                    key={i}
                    className="flex items-center gap-1 rounded-md px-2 py-0.5 text-[10px]"
                    style={{ backgroundColor: color + "18", border: `1px solid ${color}33` }}
                  >
                    <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
                    <span style={{ color }}>{s.signal_type.replace(/_/g, " ")}</span>
                    {s.value != null && (
                      <span className="text-nexus-text-muted">{Math.round((s.value as number) * 100)}%</span>
                    )}
                    <span className="text-nexus-text-muted">conf {Math.round(s.confidence * 100)}%</span>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="flex h-full items-center text-[10px] text-nexus-text-muted">
            {selectedIdx !== null ? "No notable signals for this segment." : "Click a bubble to see signal details."}
          </div>
        )}
      </div>
    </div>
  );
}
