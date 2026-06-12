import { useMemo, useState } from "react";
import type { VideoSignal, Report } from "../api/client";

type TechniqueAnalysis = NonNullable<Report["content"]["technique_analysis"]>;

interface Props {
  signals: VideoSignal[];
  durationMs: number;
  techniqueAnalysis?: TechniqueAnalysis;
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
  "verbal_uncertainty_cluster",
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
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span style={{ color }} className="font-semibold">
          {label} Risk
        </span>
        <span className="font-medium text-gray-200">{(score * 100).toFixed(0)}%</span>
      </div>
      <div className="h-3 w-full overflow-hidden rounded-full bg-gray-600">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${Math.max(score * 100, score > 0 ? 2 : 0)}%`,
            backgroundColor: color,
            boxShadow: score > 0 ? `0 0 6px ${color}88` : "none",
          }}
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
    if (key === "duration_risk") return ((f.contribution as number) ?? 0) > 0;
    const sc = (f.signal_count as number) ?? (f.weakening_count as number) ?? 0;
    return sc > 0;
  };

  // Risk factors — presence of these signals INCREASES risk (shown red when active)
  const riskFactors: { key: string; label: string }[] = [
    { key: "contamination",        label: "contamination" },
    { key: "capitulation_cascade", label: "capitulation" },
    { key: "denial_evolution",     label: "denial drop" },
    { key: "duration_risk",        label: "long duration" },
    { key: "processing_delays",    label: "response delays" },
  ];
  const present = riskFactors.filter((f) => isActive(f.key));
  const absent  = riskFactors.filter((f) => !isActive(f.key));

  // Resistance hardening is PROTECTIVE (−0.05) — shown separately in green when detected,
  // silent when absent (absence is the baseline, not a risk indicator)
  const hardeningPresent = (rf["resistance_hardening"] as Record<string, unknown> | undefined)?.present === true;

  if (present.length === 0 && absent.length === 0 && !hardeningPresent) return null;
  return (
    <div className="mt-1.5 flex flex-wrap gap-1.5">
      {present.map((f) => (
        <span
          key={f.key}
          className="rounded border border-red-600/60 bg-red-900/60 px-2 py-0.5 text-xs font-medium text-red-200"
        >
          ✓ {f.label}
        </span>
      ))}
      {absent.map((f) => (
        <span
          key={f.key}
          className="rounded border border-gray-600 bg-gray-700/60 px-2 py-0.5 text-xs text-gray-300"
        >
          ✗ {f.label}
        </span>
      ))}
      {hardeningPresent && (
        <span className="rounded border border-emerald-600/60 bg-emerald-900/40 px-2 py-0.5 text-xs font-medium text-emerald-300">
          ✓ resistance confirmed
        </span>
      )}
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
    <div className="space-y-1.5">
      <div className="relative h-3 w-full overflow-hidden rounded-full bg-gray-600">
        <div
          className="absolute left-0 top-0 h-full rounded-l-full"
          style={{
            width: `${earlyMean * 100}%`,
            background: "linear-gradient(to right, #10B981, #F59E0B)",
          }}
        />
        <div
          className="absolute top-0 h-full"
          style={{
            left: `${(1 - lateMean) * 100}%`,
            right: 0,
            background: "#EF4444",
            opacity: 0.6,
          }}
        />
      </div>
      <div className="flex items-center justify-between text-xs text-gray-300">
        <span>{firstLabel}</span>
        <span className="font-semibold text-amber-400">−{(drop * 100).toFixed(0)}%</span>
        <span className="text-red-400">{lastLabel}</span>
      </div>
    </div>
  );
}

function TechniqueBadge({
  signal,
  reportTechnique,
}: {
  signal?: VideoSignal;
  reportTechnique?: TechniqueAnalysis;
}) {
  // Prefer the LLM report analysis over the rule-based signal when available
  const technique = reportTechnique?.primary ?? signal?.value_text ?? "unknown";
  const peaceCount = reportTechnique?.peace_markers
    ?? (typeof (signal?.metadata as Record<string, unknown>)?.peace_count === "number"
      ? (signal!.metadata as Record<string, unknown>).peace_count as number
      : 0);
  const reidCount = reportTechnique?.reid_markers
    ?? (typeof (signal?.metadata as Record<string, unknown>)?.reid_count === "number"
      ? (signal!.metadata as Record<string, unknown>).reid_count as number
      : 0);
  const coerciveCount = reportTechnique?.coercive_markers ?? 0;
  const colorMap: Record<string, string> = {
    peace:    "#10B981",
    reid:     "#F59E0B",
    coercive: "#EF4444",
    mixed:    "#94A3B8",
  };
  const color = colorMap[technique] ?? "#94A3B8";
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <span
          className="rounded border px-2.5 py-1 text-xs font-bold uppercase tracking-wide"
          style={{
            backgroundColor: `${color}30`,
            borderColor: `${color}60`,
            color,
          }}
        >
          {technique}
        </span>
        {peaceCount > 0 && (
          <span className="text-xs font-medium text-emerald-400">PEACE ×{peaceCount}</span>
        )}
        {reidCount > 0 && (
          <span className="text-xs font-medium text-amber-400">Reid ×{reidCount}</span>
        )}
        {coerciveCount > 0 && (
          <span className="text-xs font-medium text-red-400">Coercive ×{coerciveCount}</span>
        )}
      </div>
      {reportTechnique?.assessment && (
        <p className="text-xs text-gray-400 leading-relaxed">{reportTechnique.assessment}</p>
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
      <span className="text-xs text-gray-400">No terms recorded in metadata</span>
    );
  }

  const visible = expanded ? allTerms : allTerms.slice(0, 4);
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap gap-1.5">
        {visible.map(({ term, startMs, endMs }) => (
          <span
            key={term}
            className="rounded border border-red-700/60 bg-red-900/30 px-2 py-0.5 text-xs text-red-200"
            title={`Adopted between ${fmtMs(startMs)} – ${fmtMs(endMs)}`}
          >
            "{term}"
          </span>
        ))}
      </div>
      {allTerms.length > 4 && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-xs text-gray-400 hover:text-gray-200 transition-colors"
        >
          {expanded ? "Show less" : `+${allTerms.length - 4} more`}
        </button>
      )}
      <p className="text-xs text-gray-400 italic">
        Garrett 2011: present in 97.5% of proven false confessions
      </p>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function InterrogationSummaryPanel({ signals, techniqueAnalysis }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  const interrogationSignals = useMemo(
    () => signals.filter((s) => INTERROGATION_TYPES.has(s.signal_type)),
    [signals]
  );

  if (interrogationSignals.length === 0 && !techniqueAnalysis) return null;

  // Collect ALL speakers with a non-zero risk score — one signal emitted per speaker.
  // Detectives score 0 and are excluded. Suspects with any measured risk are shown,
  // each with their own gauge and factor breakdown, sorted highest-first.
  const riskSignals = useMemo(
    () =>
      interrogationSignals
        .filter((s) => s.signal_type === "false_confession_risk" && (s.value ?? 0) > 0)
        .sort((a, b) => (b.value ?? 0) - (a.value ?? 0)),
    [interrogationSignals]
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
  const uncertaintyClusters = interrogationSignals.filter(
    (s) => s.signal_type === "verbal_uncertainty_cluster"
  );

  return (
    <div className="rounded-lg border border-amber-700/50 bg-gray-900/70 p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-amber-300">
          Interrogation Analysis
        </span>
        <button
          onClick={() => setCollapsed((v) => !v)}
          className="text-xs text-gray-400 hover:text-gray-200 transition-colors"
        >
          {collapsed ? "Expand" : "Collapse"}
        </button>
      </div>

      {!collapsed && (
        <div className="space-y-3 divide-y divide-gray-700">

          {/* False Confession Risk — one row per speaker with non-zero score */}
          {riskSignals.length > 0 && (
            <div className="space-y-3 pt-2 first:pt-0">
              <span className="text-xs font-semibold text-gray-200">
                ⚖️ False Confession Risk
              </span>
              {riskSignals.map((sig) => (
                <div key={sig.speaker_id ?? sig.start_ms} className="space-y-1.5">
                  {riskSignals.length > 1 && sig.speaker_id && (
                    <span className="text-[10px] font-medium uppercase tracking-wide text-amber-400/70">
                      {sig.speaker_id}
                    </span>
                  )}
                  <RiskGauge score={sig.value ?? 0} />
                  <RiskFactors
                    metadata={(sig.metadata ?? {}) as Record<string, unknown>}
                  />
                </div>
              ))}
            </div>
          )}

          {/* Denial Trajectory */}
          {denialSignal && (
            <div className="space-y-2 pt-3">
              <span className="text-xs font-semibold text-gray-200">
                📊 Denial Trajectory
              </span>
              <DenialTrajectory signal={denialSignal} />
            </div>
          )}

          {/* Interrogation Technique — prefer LLM report over rule-based signal */}
          {(techniqueAnalysis ?? techniqueSignal) && (
            <div className="space-y-2 pt-3">
              <span className="text-xs font-semibold text-gray-200">
                🎭 Technique
              </span>
              <TechniqueBadge
                signal={techniqueSignal}
                reportTechnique={techniqueAnalysis}
              />
            </div>
          )}

          {/* Contamination */}
          {contaminationSignals.length > 0 && (
            <div className="space-y-2 pt-3">
              <span className="text-xs font-semibold text-gray-200">
                ⚠️ Information Adopted ({contaminationSignals.length} signal
                {contaminationSignals.length !== 1 ? "s" : ""})
              </span>
              <ContaminationList signals={contaminationSignals} />
            </div>
          )}

          {/* Capitulation Cascade */}
          {capitulationSignals.length > 0 && (
            <div className="space-y-1.5 pt-3">
              <span className="text-xs font-semibold text-gray-200">
                📉 Capitulation Pattern
              </span>
              {capitulationSignals.map((s, i) => (
                <div key={i} className="text-xs text-gray-300">
                  {fmtMs(s.start_ms)} → {fmtMs(s.end_ms)}
                  {s.value_text && (
                    <span className="ml-1.5 text-gray-400">
                      ({s.value_text.replace(/_/g, " ")})
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Verbal Uncertainty Clusters */}
          {uncertaintyClusters.length > 0 && (
            <div className="space-y-2 pt-3">
              <span className="text-xs font-semibold text-gray-200">
                💭 Verbal Uncertainty Clusters
              </span>
              {uncertaintyClusters.map((s, i) => {
                const meta = (s.metadata ?? {}) as Record<string, unknown>;
                const pairs = meta.uncertain_pairs as number ?? 0;
                const total = meta.window_pairs as number ?? 5;
                const markers = (meta.sample_markers as string[] ?? []).slice(0, 3);
                return (
                  <div key={i} className="space-y-1">
                    <div className="text-xs text-gray-300">
                      {fmtMs(s.start_ms)} → {fmtMs(s.end_ms)}
                      <span className="ml-1.5 text-slate-400">
                        {pairs}/{total} responses hedged
                      </span>
                    </div>
                    {markers.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {markers.map((m) => (
                          <span
                            key={m}
                            className="rounded border border-slate-600 bg-slate-800/60 px-1.5 py-0.5 text-[10px] text-slate-300"
                          >
                            "{m}"
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
              <p className="text-[10px] text-gray-500 italic leading-relaxed">
                CBCA Criterion 15: memory admissions more common in truthful accounts (Steller &amp; Köhnken 1989). Cognitive load indicator — not a deception cue.
              </p>
            </div>
          )}

        </div>
      )}
    </div>
  );
}
