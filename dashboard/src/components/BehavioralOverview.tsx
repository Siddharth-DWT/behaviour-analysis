import type { Signal } from "../api/client";

interface Props {
  signals: Signal[];
  meetingType: string;
  durationMs: number;
}

// ── Gauge config per meeting type ──
interface GaugeConfig {
  key: string;
  label: string;
  sub: string;
  compute: (signals: Signal[]) => number;   // returns 0-100
  colorFn: (pct: number) => string;
}

function avg(vals: number[]): number {
  return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

function sigVals(signals: Signal[], type: string, agent?: string): number[] {
  return signals
    .filter((s) => s.signal_type === type && s.value != null && (!agent || s.agent === agent))
    .map((s) => s.value!);
}

// Shared gauge computations
const COMPUTATIONS: Record<string, (s: Signal[]) => number> = {
  engagement: (s) => {
    const attn = sigVals(s, "attention_level", "video");
    const face = sigVals(s, "facial_engagement", "video");
    const conv = s
      .filter((x) => x.signal_type === "conversation_engagement")
      .map((x) =>
        x.value_text === "highly_engaged" ? 0.9
        : x.value_text === "engaged" ? 0.7
        : x.value_text === "passive" ? 0.4
        : 0.2
      );
    const all = [...attn, ...face, ...conv];
    return Math.round(avg(all) * 100);
  },
  rapport: (s) => {
    const r = s.find((x) => x.signal_type === "rapport_indicator");
    const rapportConf = s
      .filter((x) => x.signal_type === "rapport_confirmation" && x.agent === "fusion")
      .map((x) => x.confidence ?? 0);
    const base = r?.value ?? 0;
    const boost = avg(rapportConf) * 0.15;
    return Math.round(Math.min(base + boost, 1) * 100);
  },
  stress: (s) => {
    const v = sigVals(s, "vocal_stress_score", "voice");
    const f = sigVals(s, "facial_stress", "video");
    return Math.round(avg([...v, ...f]) * 100);
  },
  alignment: (s) => {
    const masking = s.filter(
      (x) => x.agent === "fusion" && ["tone_face_masking", "smile_sentiment_incongruence", "stress_suppression"].includes(x.signal_type)
    );
    if (masking.length === 0) return sigVals(s, "vocal_stress_score").length > 0 ? 85 : 0;
    const avgIncon = avg(masking.map((x) => (x.value ?? 0) * (x.confidence ?? 0)));
    return Math.round(Math.max(0, 1 - avgIncon) * 100);
  },
  attention: (s) => {
    const screen = sigVals(s, "screen_contact", "video");
    const attn = sigVals(s, "attention_level", "video");
    const all = [...screen, ...attn];
    return all.length > 0 ? Math.round(avg(all) * 100) : 0;
  },
  composure: (s) => {
    const stress = sigVals(s, "vocal_stress_score");
    const fstress = sigVals(s, "facial_stress", "video");
    const avgStress = avg([...stress, ...fstress]);
    // Recovery signals: does stress come down after spikes?
    return Math.round(Math.max(0, 1 - avgStress * 1.2) * 100);
  },
  participation: (s) => {
    const bal = s.find((x) => x.signal_type === "conversation_balance");
    if (!bal) return 50;
    return bal.value_text === "well_balanced" ? 85
      : bal.value_text === "moderately_balanced" ? 60
      : 35;
  },
  energy: (s) => {
    const rate = s.filter(
      (x) => x.signal_type === "speech_rate_anomaly" && x.value_text === "rate_elevated"
    ).length;
    const tone = s.filter(
      (x) =>
        (x.signal_type === "tone_classification" || x.signal_type === "tone_analysis") &&
        ["confident", "enthusiastic"].includes(x.value_text ?? "")
    ).length;
    const base = sigVals(s, "gesture_animation", "video");
    return Math.min(Math.round((rate * 5 + tone * 4 + avg(base) * 20 + 40)), 100);
  },
  dealRisk: (s) => {
    const buying = s.filter((x) => x.signal_type === "buying_signal").length;
    const obj = s.filter((x) => x.signal_type === "objection_signal").length;
    const stress = avg(sigVals(s, "vocal_stress_score"));
    const risk = Math.min(Math.max((obj * 10 - buying * 8 + stress * 20) / 60, 0), 1);
    return Math.round(risk * 100);
  },
};

const C_GREEN  = "var(--color-stress-low,  #22C55E)";
const C_AMBER  = "var(--color-stress-med,  #F59E0B)";
const C_RED    = "var(--color-stress-high, #EF4444)";

function neutralGauge(pct: number): string {
  return pct >= 70 ? C_GREEN : pct >= 45 ? C_AMBER : C_RED;
}
function stressGauge(pct: number): string {
  return pct <= 30 ? C_GREEN : pct <= 55 ? C_AMBER : C_RED;
}
function riskGauge(pct: number): string {
  return pct <= 30 ? C_GREEN : pct <= 55 ? C_AMBER : C_RED;
}

const GAUGE_SETS: Record<string, GaugeConfig[]> = {
  sales_call: [
    { key: "engagement",   label: "Engagement",        sub: "combined engagement",       compute: COMPUTATIONS.engagement,   colorFn: neutralGauge },
    { key: "rapport",      label: "Rapport",           sub: "relational warmth",          compute: COMPUTATIONS.rapport,      colorFn: neutralGauge },
    { key: "stress",       label: "Stress",            sub: "voice + face combined",      compute: COMPUTATIONS.stress,       colorFn: stressGauge  },
    { key: "alignment",    label: "Alignment",         sub: "voice-face agreement",       compute: COMPUTATIONS.alignment,    colorFn: neutralGauge },
    { key: "attention",    label: "Attention",         sub: "screen gaze %",              compute: COMPUTATIONS.attention,    colorFn: neutralGauge },
    { key: "dealRisk",     label: "Deal Risk",         sub: "objection vs buying signal", compute: COMPUTATIONS.dealRisk,     colorFn: riskGauge    },
  ],
  client_meeting: [
    { key: "rapport",      label: "Rapport",           sub: "relational warmth",          compute: COMPUTATIONS.rapport,      colorFn: neutralGauge },
    { key: "engagement",   label: "Engagement",        sub: "combined engagement",        compute: COMPUTATIONS.engagement,   colorFn: neutralGauge },
    { key: "stress",       label: "Stress",            sub: "voice + face combined",      compute: COMPUTATIONS.stress,       colorFn: stressGauge  },
    { key: "alignment",    label: "Alignment",         sub: "voice-face agreement",       compute: COMPUTATIONS.alignment,    colorFn: neutralGauge },
    { key: "attention",    label: "Attention",         sub: "screen gaze %",              compute: COMPUTATIONS.attention,    colorFn: neutralGauge },
    { key: "dealRisk",     label: "Churn Risk",        sub: "disengagement signals",      compute: COMPUTATIONS.dealRisk,     colorFn: riskGauge    },
  ],
  interview: [
    { key: "engagement",   label: "Engagement",        sub: "candidate interest",         compute: COMPUTATIONS.engagement,   colorFn: neutralGauge },
    { key: "composure",    label: "Composure",         sub: "stress management",          compute: COMPUTATIONS.composure,    colorFn: neutralGauge },
    { key: "alignment",    label: "Authenticity",      sub: "voice-face agreement",       compute: COMPUTATIONS.alignment,    colorFn: neutralGauge },
    { key: "attention",    label: "Attention",         sub: "screen gaze %",              compute: COMPUTATIONS.attention,    colorFn: neutralGauge },
    { key: "rapport",      label: "Connection",        sub: "interview rapport",          compute: COMPUTATIONS.rapport,      colorFn: neutralGauge },
    { key: "energy",       label: "Energy",            sub: "pace + gesture activity",    compute: COMPUTATIONS.energy,       colorFn: neutralGauge },
  ],
  internal: [
    { key: "participation", label: "Participation",    sub: "balance of voices",          compute: COMPUTATIONS.participation, colorFn: neutralGauge },
    { key: "engagement",   label: "Engagement",        sub: "combined engagement",        compute: COMPUTATIONS.engagement,   colorFn: neutralGauge },
    { key: "stress",       label: "Conflict Level",    sub: "stress + interruption",      compute: COMPUTATIONS.stress,       colorFn: stressGauge  },
    { key: "rapport",      label: "Team Rapport",      sub: "relational warmth",          compute: COMPUTATIONS.rapport,      colorFn: neutralGauge },
    { key: "attention",    label: "Focus",             sub: "screen gaze %",              compute: COMPUTATIONS.attention,    colorFn: neutralGauge },
    { key: "alignment",    label: "Alignment",         sub: "voice-face agreement",       compute: COMPUTATIONS.alignment,    colorFn: neutralGauge },
  ],
  podcast: [
    { key: "energy",       label: "Energy",            sub: "pace + gesture activity",    compute: COMPUTATIONS.energy,       colorFn: neutralGauge },
    { key: "engagement",   label: "Engagement",        sub: "combined engagement",        compute: COMPUTATIONS.engagement,   colorFn: neutralGauge },
    { key: "rapport",      label: "Host-Guest Bond",   sub: "relational warmth",          compute: COMPUTATIONS.rapport,      colorFn: neutralGauge },
    { key: "attention",    label: "Focus",             sub: "screen gaze %",              compute: COMPUTATIONS.attention,    colorFn: neutralGauge },
    { key: "alignment",    label: "Authenticity",      sub: "voice-face agreement",       compute: COMPUTATIONS.alignment,    colorFn: neutralGauge },
    { key: "composure",    label: "Composure",         sub: "stress management",          compute: COMPUTATIONS.composure,    colorFn: neutralGauge },
  ],
};

function getGauges(meetingType: string): GaugeConfig[] {
  return GAUGE_SETS[meetingType] ?? GAUGE_SETS["sales_call"];
}

// ── Summary strip counters ──
interface StripItem { icon: string; label: string; color: string }

function buildStrip(signals: Signal[], meetingType: string): StripItem[] {
  const incong = signals.filter(
    (s) =>
      s.agent === "fusion" &&
      ["tone_face_masking", "smile_sentiment_incongruence", "stress_suppression", "verbal_incongruence"].includes(s.signal_type) &&
      (s.confidence ?? 0) >= 0.40
  ).length;

  const rapportPeaks = signals.filter(
    (s) =>
      (s.signal_type === "rapport_confirmation" || s.signal_type === "rapport_indicator") &&
      (s.value ?? 0) >= 0.65
  ).length;

  const stressSpikes = signals.filter(
    (s) => s.signal_type === "vocal_stress_score" && (s.value ?? 0) >= 0.60
  ).length;

  const unresolved = signals.filter(
    (s) => s.signal_type === "objection_signal" && (s.value ?? 0) > 0.4
  ).length;

  const items: StripItem[] = [];

  if (incong > 0) items.push({ icon: "⚠️", label: `${incong} incongruence moment${incong !== 1 ? "s" : ""}`, color: "#F59E0B" });
  if (rapportPeaks > 0) items.push({ icon: "✅", label: `${rapportPeaks} rapport peak${rapportPeaks !== 1 ? "s" : ""}`, color: "#22C55E" });
  if (stressSpikes > 0) items.push({ icon: "📊", label: `${stressSpikes} stress spike${stressSpikes !== 1 ? "s" : ""}`, color: "#EF4444" });
  if (unresolved > 0 && meetingType !== "podcast") {
    items.push({ icon: "👎", label: `${unresolved} objection${unresolved !== 1 ? "s" : ""}`, color: "#EF4444" });
  }

  return items;
}

// ── Gauge card ──
function GaugeCard({ config, value }: { config: GaugeConfig; value: number }) {
  const color = config.colorFn(value);
  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface p-3 flex flex-col gap-2">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium text-nexus-text-primary">{config.label}</div>
          <div className="text-[10px] text-nexus-text-muted mt-0.5">{config.sub}</div>
        </div>
        <span
          className="text-lg font-bold tabular-nums"
          style={{ color }}
        >
          {value > 0 ? `${value}%` : "—"}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-nexus-surface-hover overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(value, 100)}%`, background: color }}
        />
      </div>
    </div>
  );
}

export default function BehavioralOverview({ signals, meetingType, durationMs }: Props) {
  const hasVideoSignals = signals.some((s) => s.agent === "video");
  const hasFusionSignals = signals.some((s) => s.agent === "fusion");
  if (!hasVideoSignals && !hasFusionSignals) return null;

  const gauges = getGauges(meetingType);
  const strip = buildStrip(signals, meetingType);

  return (
    <section className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-text-primary">
        <span>🧠</span>
        Behavioral Overview
        {!hasVideoSignals && (
          <span className="ml-auto text-[10px] font-normal text-nexus-text-muted">
            audio-only — upload video for visual signals
          </span>
        )}
      </h3>

      {/* 6 gauge cards */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {gauges.map((g) => (
          <GaugeCard key={g.key} config={g} value={g.compute(signals)} />
        ))}
      </div>

      {/* Summary strip */}
      {strip.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-3 border-t border-nexus-border pt-3">
          {strip.map((item, i) => (
            <span
              key={i}
              className="flex items-center gap-1 text-[11px] font-medium"
              style={{ color: item.color }}
            >
              {item.icon} {item.label}
            </span>
          ))}
        </div>
      )}
    </section>
  );
}
