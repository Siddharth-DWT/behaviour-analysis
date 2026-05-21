import { useMemo, useState } from "react";
import type { VideoSignal } from "../api/client";

interface Props {
  signals: VideoSignal[];
  durationMs: number;
}

const INTERROGATION_TYPES = new Set([
  "false_confession_risk",
  "denial_weakening",
  "interrogator_technique",
  "statement_contamination",
  "capitulation_cascade",
  "resistance_hardening",
  "blink_suppression_spike",
  "motor_inhibition",
  "freezing_response",
  "evidence_response_processing_delay",
  "narrative_consistency_drift",
]);

function fmtMs(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function RiskGauge({ score }: { score: number }) {
  const color =
    score < 0.3
      ? "#10B981"
      : score < 0.55
      ? "#F59E0B"
      : score < 0.8
      ? "#F97316"
      : "#EF4444";
  const label =
    score < 0.3
      ? "Low"
      : score < 0.55
      ? "Low-Moderate"
      : score < 0.8
      ? "Elevated"
      : "High";
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px]">
        <span style={{ color }} className="font-semibold">
          {label} Risk
        </span>
        <span className="text-gray-400">{(score * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-gray-700">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(score * 100, 100)}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function RiskFactors({ metadata }: { metadata: Record<string, unknown> }) {
  const rf = (metadata.risk_factors ?? {}) as Record<string, Record<string, unknown>>;

  const isActive = (key: string): boolean => {
    const f = rf[key];
    if (!f) return false;
    if (key === "duration_risk")        return ((f.contribution as number) ?? 0) > 0;
    if (key === "resistance_hardening") return !(f.present as boolean);  // absence = risk
    const sc = (f.signal_count as number) ?? (f.weakening_count as number) ?? 0;
    return sc > 0;
  };

  const factors: { key: string; label: string }[] = [
    { key: "contamination",        label: "contamination" },
    { key: "capitulation_cascade", label: "capitulation" },
    { key: "denial_evolution",     label: "denial drop" },
    { key: "duration_risk",        label: "long duration" },
    { key: "processing_delays",    label: "response delays" },
    { key: "resistance_hardening", label: "no resistance" },
  ];
  const present = factors.filter((f) => isActive(f.key));
  const absent  = factors.filter((f) => !isActive(f.key));
  if (present.length === 0 && absent.length === 0) return null;
  return (
    <div className="mt-1 flex flex-wrap gap-1">
      {present.map((f) => (
        <span key={f.key} className="rounded px-1 py-0.5 text-[9px] bg-red-900/40 text-red-300">
          ✓ {f.label}
        </span>
      ))}
      {absent.map((f) => (
        <span key={f.key} className="rounded px-1 py-0.5 text-[9px] bg-gray-800 text-gray-500">
          ✗ {f.label}
        </span>
      ))}
    </div>
  );
}

function DenialTrajectory({ signal }: { signal: VideoSignal }) {
  const meta = (signal.metadata ?? {}) as Record<string, unknown>;
  const earlyMean = typeof meta.first_strength === "number" ? meta.first_strength : 1.0;
  const lateMean = typeof meta.last_strength === "number" ? meta.last_strength : 0.5;
  const firstLabel = typeof meta.first_label === "string" ? meta.first_label : "categorical";
  const lastLabel = typeof meta.last_label === "string" ? meta.last_label : "weak";
  const drop = Math.max(0, earlyMean - lateMean);
  return (
    <div className="space-y-1">
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-gray-700">
        {/* Early strength indicator */}
        <div
          className="absolute left-0 top-0 h-full rounded-l-full"
          style={{
            width: `${earlyMean * 100}%`,
            background: "linear-gradient(to right, #10B981, #F59E0B)",
          }}
        />
        {/* Late strength indicator */}
        <div
          className="absolute top-0 h-full"
          style={{
            left: `${(1 - lateMean) * 100}%`,
            right: 0,
            background: "#EF4444",
            opacity: 0.5,
          }}
        />
      </div>
      <div className="flex items-center justify-between text-[9px] text-gray-400">
        <span>{firstLabel}</span>
        <span className="text-amber-400">−{(drop * 100).toFixed(0)}%</span>
        <span className="text-red-400">{lastLabel}</span>
      </div>
    </div>
  );
}

function TechniqueBadge({ signal }: { signal: VideoSignal }) {
  const technique = signal.value_text ?? "unknown";
  const meta = (signal.metadata ?? {}) as Record<string, unknown>;
  const peaceCount = typeof meta.peace_count === "number" ? meta.peace_count : 0;
  const reidCount = typeof meta.reid_count === "number" ? meta.reid_count : 0;
  const colorMap: Record<string, string> = {
    peace: "#10B981",
    reid: "#F59E0B",
    coercive: "#EF4444",
    mixed: "#6B7280",
  };
  const color = colorMap[technique] ?? "#6B7280";
  return (
    <div className="flex items-center gap-2">
      <span
        className="rounded px-2 py-0.5 text-[10px] font-semibold uppercase"
        style={{ backgroundColor: `${color}22`, color }}
      >
        {technique}
      </span>
      {peaceCount > 0 && (
        <span className="text-[9px] text-emerald-400">PEACE ×{peaceCount}</span>
      )}
      {reidCount > 0 && (
        <span className="text-[9px] text-amber-400">Reid ×{reidCount}</span>
      )}
    </div>
  );
}

function ContaminationList({ signals }: { signals: VideoSignal[] }) {
  const [expanded, setExpanded] = useState(false);

  const allTerms = useMemo(() => {
    const seen = new Set<string>();
    const items: { term: string; startMs: number; endMs: number }[] = [];
    for (const s of signals) {
      const terms = (s.metadata?.contaminated_terms ?? []) as string[];
      for (const t of terms) {
        if (!seen.has(t)) {
          seen.add(t);
          items.push({ term: t, startMs: s.start_ms, endMs: s.end_ms });
        }
      }
    }
    return items;
  }, [signals]);

  if (allTerms.length === 0) {
    return (
      <span className="text-[10px] text-gray-500">No terms recorded in metadata</span>
    );
  }

  const visible = expanded ? allTerms : allTerms.slice(0, 4);
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap gap-1">
        {visible.map(({ term, startMs, endMs }) => (
          <span
            key={term}
            className="rounded border border-red-800/50 bg-red-900/20 px-1.5 py-0.5 text-[9px] text-red-300"
            title={`Adopted between ${fmtMs(startMs)} – ${fmtMs(endMs)}`}
          >
            "{term}"
          </span>
        ))}
      </div>
      {allTerms.length > 4 && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-[9px] text-gray-500 hover:text-gray-300 transition-colors"
        >
          {expanded ? "Show less" : `+${allTerms.length - 4} more`}
        </button>
      )}
      <p className="text-[9px] text-gray-500 italic">
        Garrett 2011: present in 97.5% of proven false confessions
      </p>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function InterrogationSummaryPanel({ signals }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  const interrogationSignals = useMemo(
    () => signals.filter((s) => INTERROGATION_TYPES.has(s.signal_type)),
    [signals]
  );

  if (interrogationSignals.length === 0) return null;

  const riskSignal = interrogationSignals.find(
    (s) => s.signal_type === "false_confession_risk"
  );
  const denialSignal = interrogationSignals.find(
    (s) => s.signal_type === "denial_weakening"
  );
  const techniqueSignal = interrogationSignals.find(
    (s) => s.signal_type === "interrogator_technique"
  );
  const contaminationSignals = interrogationSignals.filter(
    (s) => s.signal_type === "statement_contamination"
  );
  const capitulationSignals = interrogationSignals.filter(
    (s) => s.signal_type === "capitulation_cascade"
  );

  return (
    <div className="rounded-lg border border-amber-900/40 bg-gray-900/60 p-3 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold text-amber-300">
          Interrogation Analysis
        </span>
        <button
          onClick={() => setCollapsed((v) => !v)}
          className="text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
        >
          {collapsed ? "Expand" : "Collapse"}
        </button>
      </div>

      {!collapsed && (
        <div className="space-y-3 divide-y divide-gray-800">

          {/* False Confession Risk */}
          {riskSignal && (
            <div className="space-y-1.5 pt-2 first:pt-0">
              <span className="text-[10px] font-medium text-gray-300">
                ⚖️ False Confession Risk
              </span>
              <RiskGauge score={riskSignal.value} />
              <RiskFactors
                metadata={(riskSignal.metadata ?? {}) as Record<string, unknown>}
              />
            </div>
          )}

          {/* Denial Trajectory */}
          {denialSignal && (
            <div className="space-y-1.5 pt-2">
              <span className="text-[10px] font-medium text-gray-300">
                📊 Denial Trajectory
              </span>
              <DenialTrajectory signal={denialSignal} />
            </div>
          )}

          {/* Interrogation Technique */}
          {techniqueSignal && (
            <div className="space-y-1.5 pt-2">
              <span className="text-[10px] font-medium text-gray-300">
                🎭 Technique
              </span>
              <TechniqueBadge signal={techniqueSignal} />
            </div>
          )}

          {/* Contamination */}
          {contaminationSignals.length > 0 && (
            <div className="space-y-1.5 pt-2">
              <span className="text-[10px] font-medium text-gray-300">
                ⚠️ Information Adopted ({contaminationSignals.length} signal
                {contaminationSignals.length !== 1 ? "s" : ""})
              </span>
              <ContaminationList signals={contaminationSignals} />
            </div>
          )}

          {/* Capitulation Cascade */}
          {capitulationSignals.length > 0 && (
            <div className="space-y-1 pt-2">
              <span className="text-[10px] font-medium text-gray-300">
                📉 Capitulation Pattern
              </span>
              {capitulationSignals.map((s, i) => (
                <div key={i} className="text-[9px] text-gray-400">
                  {fmtMs(s.start_ms)} → {fmtMs(s.end_ms)}
                  {s.value_text && (
                    <span className="ml-1 text-gray-500">
                      ({s.value_text.replace(/_/g, " ")})
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}

        </div>
      )}
    </div>
  );
}
