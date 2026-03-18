import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  FileText,
  Loader2,
  Clock,
  Users,
  AlertTriangle,
  Sparkles,
  Target,
  Lightbulb,
  CheckCircle,
  XCircle,
  TrendingUp,
  ShieldCheck,
  MessageSquare,
  AlertCircle,
  Info,
  Zap,
} from "lucide-react";
import { format } from "date-fns";
import { getSession, getSignals, getTranscript, getReport } from "../api/client";
import type { Signal, TranscriptSegment } from "../api/client";
import TranscriptBlock from "../components/TranscriptBlock";
import StressTimeline from "../components/StressTimeline";
import AlertCard from "../components/AlertCard";
import SignalExplorer from "../components/SignalExplorer";

// ── Helpers ──

function formatDuration(ms: number | null): string {
  if (!ms) return "--";
  const totalSec = Math.round(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return sec > 0 ? `${min}m ${sec}s` : `${min}m`;
}

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

function matchSignalsToSegment(
  segment: TranscriptSegment,
  signals: Signal[]
): Signal[] {
  return signals.filter((s) => {
    const overlapStart = Math.max(s.window_start_ms, segment.start_ms);
    const overlapEnd = Math.min(s.window_end_ms, segment.end_ms);
    if (overlapEnd <= overlapStart) return false;
    if (
      s.speaker_label &&
      segment.speaker_label &&
      s.speaker_label !== segment.speaker_label
    ) {
      return false;
    }
    return true;
  });
}

function getCSSVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ── Gauge Bar ──

function GaugeBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-[11px] text-nexus-text-secondary mb-1">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-nexus-surface-hover overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(pct, 100)}%`, background: color }}
        />
      </div>
    </div>
  );
}

// ── Stat Chip ──

function StatChip({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-nexus-surface-hover px-2.5 py-1 text-[11px]">
      {color && <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />}
      <span className="text-nexus-text-secondary">{label}</span>
      <span className="font-mono font-medium text-nexus-text-primary">{value}</span>
    </span>
  );
}

// ── Speaker Analysis ──

interface SpeakerStats {
  label: string;
  role?: string;
  avgStress: number;
  maxStress: number;
  dominantTone: string;
  avgSentiment: number;
  avgPower: number;
  avgConfidence: number;
  fillerCount: number;
  buyingSignalCount: number;
  objectionCount: number;
}

function computeSpeakerStats(signals: Signal[]): SpeakerStats[] {
  const speakerMap = new Map<string, Signal[]>();

  for (const s of signals) {
    const label = s.speaker_label || "Unknown";
    if (!speakerMap.has(label)) speakerMap.set(label, []);
    speakerMap.get(label)!.push(s);
  }

  const stats: SpeakerStats[] = [];

  for (const [label, sigs] of speakerMap) {
    const stressVals = sigs
      .filter((s) => s.signal_type === "vocal_stress_score" && s.value != null)
      .map((s) => s.value!);
    const sentimentVals = sigs
      .filter((s) => s.signal_type === "sentiment_score" && s.value != null)
      .map((s) => s.value!);
    const powerVals = sigs
      .filter((s) => s.signal_type === "power_language_score" && s.value != null)
      .map((s) => s.value!);

    // Dominant tone from tone_classification signals
    const tones = sigs
      .filter((s) => (s.signal_type === "tone_classification" || s.signal_type === "tone_analysis") && s.value_text)
      .map((s) => s.value_text);
    const toneFreq = new Map<string, number>();
    for (const t of tones) {
      if (t && t.toLowerCase() !== "neutral") {
        toneFreq.set(t, (toneFreq.get(t) || 0) + 1);
      }
    }
    let dominantTone = "Neutral";
    let maxFreq = 0;
    for (const [tone, freq] of toneFreq) {
      if (freq > maxFreq) {
        dominantTone = tone;
        maxFreq = freq;
      }
    }

    // Filler count — count ALL filler_detection signals (each represents a filler event)
    const fillerCount = sigs.filter(
      (s) => s.signal_type === "filler_detection"
    ).length;

    // Buying signals — any signal with type buying_signal, regardless of value
    const buyingSignalCount = sigs.filter(
      (s) => s.signal_type === "buying_signal"
    ).length;

    // Objections
    const objectionCount = sigs.filter(
      (s) => s.signal_type === "objection_signal"
    ).length;

    const avg = (arr: number[]) => (arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);

    stats.push({
      label,
      avgStress: avg(stressVals),
      maxStress: stressVals.length > 0 ? Math.max(...stressVals) : 0,
      dominantTone: dominantTone.charAt(0).toUpperCase() + dominantTone.slice(1),
      avgSentiment: avg(sentimentVals),
      avgPower: powerVals.length > 0 ? avg(powerVals) : 0.5,
      avgConfidence: avg(
        sigs.filter((s) => s.confidence != null).map((s) => s.confidence)
      ),
      fillerCount,
      buyingSignalCount,
      objectionCount,
    });
  }

  return stats;
}

// ── Call Outcome ──

function computeCallOutcome(speakerStats: SpeakerStats[]) {
  const totalBuying = speakerStats.reduce((a, s) => a + s.buyingSignalCount, 0);
  const totalObjections = speakerStats.reduce((a, s) => a + s.objectionCount, 0);
  const avgSentiment =
    speakerStats.length > 0
      ? speakerStats.reduce((a, s) => a + s.avgSentiment, 0) / speakerStats.length
      : 0;

  let outcome: "Positive" | "Neutral" | "Negative" = "Neutral";
  let outcomeColor = getCSSVar("--stress-med") || "#F59E0B";
  if (totalBuying >= 2 && totalObjections === 0) {
    outcome = "Positive";
    outcomeColor = getCSSVar("--stress-low") || "#22C55E";
  } else if (totalBuying >= 1 && avgSentiment > 0) {
    outcome = "Positive";
    outcomeColor = getCSSVar("--stress-low") || "#22C55E";
  } else if (totalObjections >= 3 || avgSentiment < -0.3) {
    outcome = "Negative";
    outcomeColor = getCSSVar("--stress-high") || "#EF4444";
  }

  // Decision readiness
  let readinessLabel = "Uncertain";
  let readinessColor = getCSSVar("--stress-med") || "#F59E0B";
  if (totalBuying >= 2) {
    readinessLabel = "Ready";
    readinessColor = getCSSVar("--stress-low") || "#22C55E";
  } else if (totalObjections > totalBuying) {
    readinessLabel = "Not Ready";
    readinessColor = getCSSVar("--stress-high") || "#EF4444";
  }

  // Objection handled
  let objHandledLabel = "N/A";
  let objHandledColor = getCSSVar("--text-secondary") || "#8B93A7";
  if (totalObjections > 0) {
    if (totalBuying > totalObjections) {
      objHandledLabel = "Yes";
      objHandledColor = getCSSVar("--stress-low") || "#22C55E";
    } else if (totalBuying > 0) {
      objHandledLabel = "Partially";
      objHandledColor = getCSSVar("--stress-med") || "#F59E0B";
    } else {
      objHandledLabel = "No";
      objHandledColor = getCSSVar("--stress-high") || "#EF4444";
    }
  }

  return {
    outcome, outcomeColor,
    readinessLabel, readinessColor,
    objHandledLabel, objHandledColor,
    totalBuying, totalObjections,
  };
}

// ── Infer Speaker Roles ──

function inferSpeakerRoles(
  speakerStats: SpeakerStats[],
  meetingType: string
): Record<string, string> {
  const roles: Record<string, string> = {};
  if (meetingType !== "sales_call" || speakerStats.length < 2) return roles;

  // In a sales call: the speaker with more buying signals / questions is the Prospect
  // The speaker with more assertive language is the Seller
  const sorted = [...speakerStats].sort((a, b) => {
    // Higher buying signals + lower power = more likely Prospect
    const aScore = a.buyingSignalCount * 2 - a.avgPower * 10;
    const bScore = b.buyingSignalCount * 2 - b.avgPower * 10;
    return bScore - aScore; // Higher score = more likely Prospect
  });

  roles[sorted[0].label] = "Prospect";
  roles[sorted[1].label] = "Seller";

  return roles;
}

// ── Fusion Signal Display ──

const FUSION_SIGNAL_LABELS: Record<string, { label: string; icon: string }> = {
  credibility_assessment: { label: "Credibility Concern", icon: "🔴" },
  verbal_incongruence: { label: "Verbal Mismatch", icon: "⚠️" },
  urgency_authenticity: { label: "Urgency Pattern", icon: "⚠️" },
};

const FUSION_VALUE_LABELS: Record<string, string> = {
  credibility_concern: "Content contradicts vocal stress patterns",
  mild_incongruence: "Slight mismatch between verbal content and vocal indicators",
  strong_verbal_incongruence: "Positive sentiment expressed with heavy hedging",
  moderate_verbal_incongruence: "Agreement with notable hedging language",
  incongruence_with_objection: "Positive sentiment combined with hidden objection markers",
  manufactured_urgency: "Fast-paced persuasion with concurrent stress indicators",
  authentic_urgency: "Persuasive language supported by confident vocal patterns",
  ambiguous_urgency: "Urgency pattern with mixed vocal signals",
};

// ── Main Component ──

export default function SessionDetail() {
  const { id } = useParams<{ id: string }>();

  const { data: detail, isLoading: loadingDetail } = useQuery({
    queryKey: ["session", id],
    queryFn: () => getSession(id!),
    enabled: !!id,
  });

  const { data: signalData } = useQuery({
    queryKey: ["signals", id],
    queryFn: () => getSignals(id!, { limit: 500 }),
    enabled: !!id,
  });

  const { data: transcriptData } = useQuery({
    queryKey: ["transcript", id],
    queryFn: () => getTranscript(id!),
    enabled: !!id,
  });

  const { data: reportData } = useQuery({
    queryKey: ["report", id],
    queryFn: () => getReport(id!),
    enabled: !!id && detail?.has_report === true,
  });

  if (loadingDetail) {
    return (
      <div className="flex items-center justify-center py-20 text-sm text-nexus-text-muted">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading session...
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="py-20 text-center text-sm text-nexus-text-muted">
        Session not found
      </div>
    );
  }

  const { session, alerts } = detail;
  const signals = signalData?.signals ?? [];
  const segments = transcriptData?.segments ?? [];
  const report = reportData?.report;
  const content = report?.content;

  // Compute analytics from signals
  const speakerStats = computeSpeakerStats(signals);
  const callOutcome = computeCallOutcome(speakerStats);

  // Infer speaker roles
  const speakerRoles = inferSpeakerRoles(speakerStats, session.meeting_type);

  // Extract fusion signals
  const fusionSignals = signals.filter((s) => s.agent === "fusion");

  // Theme-aware speaker colors
  const speakerColors = [
    getCSSVar("--accent-blue") || "#4F8BFF",
    getCSSVar("--accent-purple") || "#8B5CF6",
    getCSSVar("--stress-med") || "#F59E0B",
    getCSSVar("--engagement") || "#10B981",
    getCSSVar("--agent-gaze") || "#EC4899",
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      {/* 1. HEADER */}
      <div>
        <Link
          to="/sessions"
          className="mb-3 inline-flex items-center gap-1 text-xs text-nexus-text-muted hover:text-nexus-accent-blue"
        >
          <ArrowLeft className="h-3 w-3" />
          Back to Sessions
        </Link>

        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-lg font-semibold text-nexus-text-primary">
              {session.title || "Untitled Session"}
            </h1>
            <div className="mt-1 flex items-center gap-4 text-xs text-nexus-text-muted">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatDuration(session.duration_ms)}
              </span>
              <span className="flex items-center gap-1">
                <Users className="h-3 w-3" />
                {session.speaker_count ?? "--"} speakers
              </span>
              <span>
                {format(new Date(session.created_at), "MMM d, yyyy 'at' h:mm a")}
              </span>
              {detail.alert_count > 0 && (
                <span className="flex items-center gap-1 text-nexus-alert">
                  <AlertTriangle className="h-3 w-3" />
                  {detail.alert_count} alert{detail.alert_count !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          </div>

          {detail.has_report && (
            <Link
              to={`/sessions/${id}/report`}
              className="flex items-center gap-1.5 rounded bg-accent-purple-20 px-3 py-1.5 text-xs font-medium text-nexus-accent-purple transition-colors hover:bg-accent-purple-30"
            >
              <FileText className="h-3.5 w-3.5" />
              View Report
            </Link>
          )}
        </div>

        {/* Signal count summary */}
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(detail.signals_by_agent).map(([agent, count]) => {
            const agentColors: Record<string, string> = {
              voice: "var(--agent-voice)",
              language: "var(--agent-language)",
              fusion: "var(--agent-fusion)",
            };
            return (
              <span
                key={agent}
                className="inline-flex items-center gap-1.5 rounded bg-nexus-surface px-2 py-0.5 text-[10px] font-mono text-nexus-text-secondary"
              >
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ background: agentColors[agent] || "var(--neutral)" }}
                />
                {agent}: {count}
              </span>
            );
          })}
          <span className="rounded bg-nexus-surface px-2 py-0.5 text-[10px] font-mono text-nexus-text-secondary">
            {detail.signal_count} signals
          </span>
        </div>
      </div>

      {/* 2. EXECUTIVE SUMMARY */}
      {content?.executive_summary ? (
        <section className="rounded-lg border border-accent-purple-30 bg-nexus-surface p-5">
          <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
            <Sparkles className="h-4 w-4" />
            Executive Summary
          </h2>
          <p className="text-sm leading-relaxed text-nexus-text-primary">
            {content.executive_summary}
          </p>
        </section>
      ) : report?.narrative ? (
        <section className="rounded-lg border border-accent-purple-30 bg-nexus-surface p-5">
          <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
            <Sparkles className="h-4 w-4" />
            Executive Summary
          </h2>
          <p className="text-sm leading-relaxed text-nexus-text-primary">
            {report.narrative}
          </p>
        </section>
      ) : null}

      {/* 3. CALL OUTCOME (sales_call only) */}
      {session.meeting_type === "sales_call" && speakerStats.length > 0 && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {/* Estimated Outcome */}
            <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 text-center">
              <div className="text-[11px] text-nexus-text-secondary mb-2">
                Estimated Outcome
              </div>
              <div className="flex items-center justify-center gap-2">
                {callOutcome.outcome === "Positive" ? (
                  <CheckCircle className="h-5 w-5" style={{ color: callOutcome.outcomeColor }} />
                ) : callOutcome.outcome === "Negative" ? (
                  <XCircle className="h-5 w-5" style={{ color: callOutcome.outcomeColor }} />
                ) : (
                  <TrendingUp className="h-5 w-5" style={{ color: callOutcome.outcomeColor }} />
                )}
                <span className="text-xl font-bold" style={{ color: callOutcome.outcomeColor }}>
                  {callOutcome.outcome}
                </span>
              </div>
            </div>

            {/* Decision Readiness */}
            <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 text-center">
              <div className="text-[11px] text-nexus-text-secondary mb-2">
                Decision Readiness
              </div>
              <div className="flex items-center justify-center gap-2">
                <ShieldCheck className="h-5 w-5" style={{ color: callOutcome.readinessColor }} />
                <span className="text-xl font-bold" style={{ color: callOutcome.readinessColor }}>
                  {callOutcome.readinessLabel}
                </span>
              </div>
            </div>

            {/* Objection Handled */}
            <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 text-center">
              <div className="text-[11px] text-nexus-text-secondary mb-2">
                Objection Handled
              </div>
              <div className="flex items-center justify-center gap-2">
                <MessageSquare className="h-5 w-5" style={{ color: callOutcome.objHandledColor }} />
                <span className="text-xl font-bold" style={{ color: callOutcome.objHandledColor }}>
                  {callOutcome.objHandledLabel}
                </span>
              </div>
            </div>
          </div>

          {/* Signal Explorer */}
          <SignalExplorer
            signals={signals}
            signalsByAgent={detail.signals_by_agent}
            totalCount={detail.signal_count}
            speakerRoles={speakerRoles}
          />
        </>
      )}

      {/* 4. SPEAKER ANALYSIS + STRESS TIMELINE (two-column) */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* LEFT: Speaker Analysis Cards */}
        <div className="lg:col-span-3 space-y-4">
          {speakerStats.length > 0 && (
            <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
              <h3 className="mb-4 text-sm font-medium text-nexus-text-primary">
                Speaker Analysis
              </h3>
              <div className="space-y-5">
                {speakerStats.map((speaker, i) => {
                  const color = speakerColors[i % speakerColors.length];
                  const role = speakerRoles[speaker.label];
                  const stressPct = Math.round(speaker.avgStress * 100);
                  const sentPct = Math.round(Math.max(0, (speaker.avgSentiment + 1) / 2 * 100));
                  const powerPct = Math.round(speaker.avgPower * 100);
                  const confPct = Math.round(speaker.avgConfidence * 100);

                  const stressColor = stressPct > 60
                    ? (getCSSVar("--stress-high") || "#EF4444")
                    : stressPct > 35
                    ? (getCSSVar("--stress-med") || "#F59E0B")
                    : (getCSSVar("--stress-low") || "#22C55E");
                  const sentColor = sentPct > 55
                    ? (getCSSVar("--stress-low") || "#22C55E")
                    : sentPct < 45
                    ? (getCSSVar("--stress-high") || "#EF4444")
                    : (getCSSVar("--stress-med") || "#F59E0B");

                  return (
                    <div key={speaker.label}>
                      <div className="flex items-center gap-2 mb-3">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ background: color }}
                        />
                        <span className="text-sm font-semibold text-nexus-text-primary">
                          {role ? `${role}` : speaker.label}
                        </span>
                        <span className="text-xs text-nexus-text-muted">
                          ({speaker.label})
                        </span>
                      </div>

                      <GaugeBar label="Stress" pct={stressPct} color={stressColor} />
                      <GaugeBar label="Sentiment" pct={sentPct} color={sentColor} />
                      <GaugeBar label="Power" pct={powerPct} color={getCSSVar("--accent-purple") || "#8B5CF6"} />
                      <GaugeBar label="Confidence" pct={confPct} color={getCSSVar("--accent-blue") || "#4F8BFF"} />

                      <div className="mt-2 flex flex-wrap gap-1.5">
                        <StatChip label="Tone" value={speaker.dominantTone} />
                        <StatChip label="Fillers" value={speaker.fillerCount} color={getCSSVar("--stress-med") || "#F59E0B"} />
                        {speaker.buyingSignalCount > 0 && (
                          <StatChip label="Buying" value={speaker.buyingSignalCount} color={getCSSVar("--stress-low") || "#22C55E"} />
                        )}
                        {speaker.objectionCount > 0 && (
                          <StatChip label="Objections" value={speaker.objectionCount} color={getCSSVar("--stress-high") || "#EF4444"} />
                        )}
                      </div>

                      {i < speakerStats.length - 1 && (
                        <div className="mt-4 border-b border-nexus-border" />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT: Stress Timeline */}
        <div className="lg:col-span-2">
          <StressTimeline signals={signals} speakerRoles={speakerRoles} />
        </div>
      </div>

      {/* 5. ALERTS & FUSION INSIGHTS */}
      <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
        <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-nexus-text-primary">
          <Zap className="h-4 w-4 text-nexus-alert" />
          Alerts & Fusion Insights
          <span className="ml-auto text-xs font-normal text-nexus-text-muted">
            {alerts.length} alert{alerts.length !== 1 ? "s" : ""}, {fusionSignals.length} fusion signal{fusionSignals.length !== 1 ? "s" : ""}
          </span>
        </h2>

        {alerts.length === 0 && fusionSignals.length === 0 ? (
          <p className="text-sm text-nexus-text-muted italic">
            No alerts detected in this session.
          </p>
        ) : (
          <div className="space-y-3">
            {/* Alerts */}
            {alerts.map((alert) => (
              <AlertCard key={alert.id} alert={alert} />
            ))}

            {/* Fusion signals */}
            {fusionSignals.map((fs, i) => {
              const sigConfig = FUSION_SIGNAL_LABELS[fs.signal_type] || {
                label: fs.signal_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
                icon: "🟠",
              };
              const valueDesc = FUSION_VALUE_LABELS[fs.value_text] || fs.value_text.replace(/_/g, " ");
              const speaker = fs.speaker_label || "Unknown";
              const role = speakerRoles[speaker];
              const speakerDisplay = role ? `${role} (${speaker})` : speaker;

              return (
                <div
                  key={`fusion-${i}`}
                  className="rounded-lg border-l-[3px] border-nexus-alert bg-nexus-surface-hover p-3"
                >
                  <div className="flex items-start gap-2">
                    <span className="text-base leading-none mt-0.5">{sigConfig.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 text-xs mb-0.5">
                        <span className="font-semibold text-nexus-alert">
                          {sigConfig.label}
                        </span>
                        <span className="text-nexus-text-muted">
                          {speakerDisplay}
                        </span>
                        <span className="ml-auto font-mono text-nexus-text-muted">
                          {formatTime(fs.window_start_ms)}–{formatTime(fs.window_end_ms)}
                        </span>
                      </div>
                      <p className="text-sm text-nexus-text-primary">
                        {valueDesc}
                      </p>
                      <p className="mt-0.5 text-xs text-nexus-text-muted">
                        Confidence: {(fs.confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* 6. TRANSCRIPT */}
      <div>
        <h2 className="mb-3 text-sm font-medium text-nexus-text-secondary">
          Transcript
          {segments.length > 0 && (
            <span className="ml-2 text-nexus-text-muted">
              ({segments.length} segments)
            </span>
          )}
        </h2>

        {segments.length === 0 ? (
          <div className="flex h-48 items-center justify-center rounded-lg border border-nexus-border bg-nexus-surface text-sm text-nexus-text-muted">
            No transcript available
          </div>
        ) : (
          <div className="space-y-2 max-h-[700px] overflow-y-auto pr-1">
            {segments.map((segment) => (
              <TranscriptBlock
                key={segment.id}
                segment={segment}
                signals={matchSignalsToSegment(segment, signals)}
                speakerRole={segment.speaker_label ? speakerRoles[segment.speaker_label] : undefined}
              />
            ))}
          </div>
        )}
      </div>

      {/* 7. KEY MOMENTS */}
      {content?.key_moments && content.key_moments.length > 0 && (
        <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
          <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-nexus-text-primary">
            <Target className="h-4 w-4 text-nexus-accent-blue" />
            Key Moments
          </h2>
          <div className="space-y-4">
            {content.key_moments.map((moment, i) => (
              <div
                key={i}
                className="border-l-2 border-accent-blue-40 pl-3"
              >
                <div className="flex items-center gap-2 text-xs">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-accent-blue-15 font-mono text-[10px] font-bold text-nexus-accent-blue">
                    {i + 1}
                  </span>
                  {moment.time_description && (
                    <span className="font-mono text-nexus-text-muted">
                      {moment.time_description}
                    </span>
                  )}
                </div>
                <p className="mt-1 text-sm text-nexus-text-primary">
                  {moment.description}
                </p>
                {moment.significance && (
                  <p className="mt-0.5 text-xs text-nexus-text-secondary italic">
                    {moment.significance}
                  </p>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 8. CROSS-MODAL INSIGHTS */}
      <section className="rounded-lg border-l-[3px] border-nexus-accent-purple bg-nexus-surface p-5">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
          <Lightbulb className="h-4 w-4" />
          Cross-Modal Insights
        </h2>
        {content?.cross_modal_insights && content.cross_modal_insights.length > 0 ? (
          <ul className="space-y-2">
            {content.cross_modal_insights.map((insight, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-sm text-nexus-text-primary"
              >
                <span className="mt-1 text-nexus-accent-purple font-bold">→</span>
                {insight}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-nexus-text-muted italic">
            Insufficient cross-modal data for insights in this session.
          </p>
        )}
      </section>

      {/* 9. COACHING RECOMMENDATIONS */}
      {content?.recommendations && content.recommendations.length > 0 && (
        <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
          <h2 className="mb-3 text-sm font-semibold text-nexus-text-primary">
            Coaching Recommendations
          </h2>
          <ul className="space-y-2">
            {content.recommendations.map((rec, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-sm text-nexus-text-primary"
              >
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-nexus-stress-low" />
                {rec}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
