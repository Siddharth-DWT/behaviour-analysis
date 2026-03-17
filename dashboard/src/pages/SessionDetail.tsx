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
} from "lucide-react";
import { format } from "date-fns";
import { getSession, getSignals, getTranscript, getReport } from "../api/client";
import type { Signal, TranscriptSegment } from "../api/client";
import TranscriptBlock from "../components/TranscriptBlock";
import StressTimeline from "../components/StressTimeline";
import AlertCard from "../components/AlertCard";

// ── Helpers ──

function formatDuration(ms: number | null): string {
  if (!ms) return "--";
  const totalSec = Math.round(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return sec > 0 ? `${min}m ${sec}s` : `${min}m`;
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
    const label = s.speaker_label || s.speaker_id || "Unknown";
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

    const tones = sigs
      .filter((s) => s.signal_type === "tone_analysis" && s.value_text)
      .map((s) => s.value_text);
    const toneFreq = new Map<string, number>();
    for (const t of tones) {
      toneFreq.set(t, (toneFreq.get(t) || 0) + 1);
    }
    let dominantTone = "Neutral";
    let maxFreq = 0;
    for (const [tone, freq] of toneFreq) {
      if (freq > maxFreq) {
        dominantTone = tone;
        maxFreq = freq;
      }
    }

    const fillerCount = sigs.filter(
      (s) =>
        s.signal_type === "filler_detection" &&
        s.value_text &&
        ["elevated", "high", "excessive"].includes(s.value_text.toLowerCase())
    ).length;

    const buyingSignalCount = sigs.filter(
      (s) =>
        (s.signal_type === "buying_signal" || s.signal_type === "buying_signal_detected") &&
        (s.value_text?.toLowerCase() === "true" || (s.value != null && s.value > 0.5))
    ).length;

    const objectionCount = sigs.filter(
      (s) =>
        (s.signal_type === "objection_signal" || s.signal_type === "objection_detected") &&
        (s.value_text?.toLowerCase() === "true" || (s.value != null && s.value > 0.5))
    ).length;

    const avg = (arr: number[]) => (arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);

    stats.push({
      label,
      avgStress: avg(stressVals),
      dominantTone: dominantTone.charAt(0).toUpperCase() + dominantTone.slice(1),
      avgSentiment: avg(sentimentVals),
      avgPower: avg(powerVals),
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

  // Estimated outcome
  let outcome: "Positive" | "Neutral" | "Negative" = "Neutral";
  let outcomeColor = "text-nexus-stress-med";
  if (totalBuying >= 2 && avgSentiment > 0.3) {
    outcome = "Positive";
    outcomeColor = "text-nexus-stress-low";
  } else if (totalObjections >= 3 || avgSentiment < -0.2) {
    outcome = "Negative";
    outcomeColor = "text-nexus-stress-high";
  }

  // Decision readiness (heuristic)
  const readiness = Math.min(100, Math.round((totalBuying * 20 + Math.max(0, avgSentiment * 50))));

  // Objection handled ratio
  const handledRatio =
    totalObjections > 0
      ? Math.round(Math.max(0, 100 - (totalObjections / Math.max(totalBuying + totalObjections, 1)) * 100))
      : 100;

  return { outcome, outcomeColor, readiness, handledRatio, totalBuying, totalObjections };
}

// ── Speaker Colors ──

const SPEAKER_COLORS = ["#4F8BFF", "#8B5CF6", "#F59E0B", "#10B981", "#EC4899"];

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
    enabled: !!id && !!detail?.has_report,
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

  const speakerStats = computeSpeakerStats(signals);
  const callOutcome = computeCallOutcome(speakerStats);

  // Build speaker role map (from report metadata or fusion data if available)
  const speakerRoles: Record<string, string> = {};
  // Try to extract from report narrative or content
  if (content && (content as Record<string, unknown>)["speaker_roles"]) {
    const roles = (content as Record<string, unknown>)["speaker_roles"] as Record<string, string>;
    Object.assign(speakerRoles, roles);
  }
  // Fallback: try to infer from signal metadata
  for (const sig of signals) {
    if (sig.metadata && typeof sig.metadata === "object") {
      const meta = sig.metadata as Record<string, unknown>;
      if (meta["speaker_role"] && sig.speaker_label) {
        speakerRoles[sig.speaker_label] = String(meta["speaker_role"]);
      }
    }
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      {/* ── HEADER ── */}
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
              className="flex items-center gap-1.5 rounded bg-nexus-accent-purple/20 px-3 py-1.5 text-xs font-medium text-nexus-accent-purple transition-colors hover:bg-nexus-accent-purple/30"
            >
              <FileText className="h-3.5 w-3.5" />
              View Report
            </Link>
          )}
        </div>

        {/* Signal count pills */}
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(detail.signals_by_agent).map(([agent, count]) => (
            <span
              key={agent}
              className="rounded bg-nexus-surface px-2 py-0.5 text-[10px] font-mono text-nexus-text-secondary"
            >
              {agent}: {count}
            </span>
          ))}
          <span className="rounded bg-nexus-surface px-2 py-0.5 text-[10px] font-mono text-nexus-text-secondary">
            total: {detail.signal_count}
          </span>
        </div>
      </div>

      {/* ── TOP: EXECUTIVE SUMMARY ── */}
      {content?.executive_summary && (
        <section className="rounded-lg border border-nexus-accent-purple/30 bg-nexus-accent-purple/5 p-5">
          <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
            <Sparkles className="h-4 w-4" />
            Executive Summary
          </h2>
          <p className="text-sm leading-relaxed text-nexus-text-primary">
            {content.executive_summary}
          </p>
        </section>
      )}

      {/* ── TOP: CALL OUTCOME ── */}
      {speakerStats.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {/* Estimated Outcome */}
          <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
            <div className="flex items-center gap-2 text-xs text-nexus-text-secondary mb-2">
              {callOutcome.outcome === "Positive" ? (
                <CheckCircle className="h-3.5 w-3.5 text-nexus-stress-low" />
              ) : callOutcome.outcome === "Negative" ? (
                <XCircle className="h-3.5 w-3.5 text-nexus-stress-high" />
              ) : (
                <TrendingUp className="h-3.5 w-3.5 text-nexus-stress-med" />
              )}
              Estimated Outcome
            </div>
            <p className={`text-lg font-semibold ${callOutcome.outcomeColor}`}>
              {callOutcome.outcome}
            </p>
          </div>

          {/* Decision Readiness */}
          <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
            <div className="flex items-center gap-2 text-xs text-nexus-text-secondary mb-2">
              <ShieldCheck className="h-3.5 w-3.5 text-nexus-accent-blue" />
              Decision Readiness
            </div>
            <p className="text-lg font-semibold text-nexus-accent-blue">
              {callOutcome.readiness}%
            </p>
          </div>

          {/* Objection Handled */}
          <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
            <div className="flex items-center gap-2 text-xs text-nexus-text-secondary mb-2">
              <MessageSquare className="h-3.5 w-3.5 text-nexus-accent-purple" />
              Objection Handled
            </div>
            <p className="text-lg font-semibold text-nexus-accent-purple">
              {callOutcome.handledRatio}%
            </p>
          </div>
        </div>
      )}

      {/* Stat chips row */}
      {speakerStats.length > 0 && (
        <div className="flex flex-wrap gap-2">
          <StatChip label="Buying Signals" value={callOutcome.totalBuying} color="#22C55E" />
          <StatChip label="Objections" value={callOutcome.totalObjections} color="#EF4444" />
          <StatChip label="Segments" value={segments.length} color="#4F8BFF" />
          <StatChip label="Signals" value={detail.signal_count} color="#8B5CF6" />
        </div>
      )}

      {/* ── MIDDLE: TRANSCRIPT + RIGHT PANEL ── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* Transcript Panel */}
        <div className="lg:col-span-3">
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

        {/* Right column */}
        <div className="lg:col-span-2 space-y-4">
          {/* Speaker Analysis Cards */}
          {speakerStats.length > 0 && (
            <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
              <h3 className="mb-4 text-sm font-medium text-nexus-text-primary">
                Speaker Analysis
              </h3>
              <div className="space-y-5">
                {speakerStats.map((speaker, i) => {
                  const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
                  const role = speakerRoles[speaker.label];
                  return (
                    <div key={speaker.label}>
                      <div className="flex items-center gap-2 mb-3">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ background: color }}
                        />
                        <span className="text-sm font-medium text-nexus-text-primary">
                          {speaker.label}
                        </span>
                        {role && (
                          <span className="rounded bg-nexus-accent-purple/15 px-1.5 py-0.5 text-[10px] font-medium text-nexus-accent-purple">
                            {role}
                          </span>
                        )}
                      </div>

                      <GaugeBar
                        label="Stress"
                        pct={Math.round(speaker.avgStress * 100)}
                        color={speaker.avgStress > 0.6 ? "#EF4444" : speaker.avgStress > 0.3 ? "#F59E0B" : "#22C55E"}
                      />
                      <GaugeBar
                        label="Sentiment"
                        pct={Math.round(Math.max(0, (speaker.avgSentiment + 1) / 2 * 100))}
                        color={speaker.avgSentiment > 0.3 ? "#22C55E" : speaker.avgSentiment < -0.2 ? "#EF4444" : "#F59E0B"}
                      />
                      <GaugeBar
                        label="Power"
                        pct={Math.round(speaker.avgPower * 100)}
                        color="#8B5CF6"
                      />
                      <GaugeBar
                        label="Confidence"
                        pct={Math.round(speaker.avgConfidence * 100)}
                        color="#4F8BFF"
                      />

                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {speaker.dominantTone !== "Neutral" && (
                          <StatChip label="Tone" value={speaker.dominantTone} />
                        )}
                        {speaker.fillerCount > 0 && (
                          <StatChip label="Fillers" value={speaker.fillerCount} color="#F59E0B" />
                        )}
                        {speaker.buyingSignalCount > 0 && (
                          <StatChip label="Buying" value={speaker.buyingSignalCount} color="#22C55E" />
                        )}
                        {speaker.objectionCount > 0 && (
                          <StatChip label="Objections" value={speaker.objectionCount} color="#EF4444" />
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

          {/* Stress Timeline Chart */}
          <StressTimeline signals={signals} speakerRoles={speakerRoles} />

          {/* Alerts */}
          {alerts.length > 0 && (
            <div>
              <h2 className="mb-3 text-sm font-medium text-nexus-text-secondary">
                Alerts
                <span className="ml-2 text-nexus-text-muted">
                  ({alerts.length})
                </span>
              </h2>
              <div className="space-y-2">
                {alerts.map((alert) => (
                  <AlertCard key={alert.id} alert={alert} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── BOTTOM: KEY MOMENTS ── */}
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
                className="border-l-2 border-nexus-accent-blue/40 pl-3"
              >
                <div className="flex items-center gap-2 text-xs">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-nexus-accent-blue/15 font-mono text-[10px] font-bold text-nexus-accent-blue">
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

      {/* ── BOTTOM: CROSS-MODAL INSIGHTS ── */}
      {content?.cross_modal_insights && content.cross_modal_insights.length > 0 && (
        <section className="rounded-lg border border-nexus-accent-purple/30 bg-nexus-accent-purple/5 p-5">
          <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
            <Lightbulb className="h-4 w-4" />
            Cross-Modal Insights
          </h2>
          <ul className="space-y-2">
            {content.cross_modal_insights.map((insight, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-sm text-nexus-text-primary"
              >
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-nexus-accent-purple" />
                {insight}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ── BOTTOM: RECOMMENDATIONS ── */}
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
