import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  FileText,
  Loader2,
  Clock,
  Users,
  AlertTriangle,
} from "lucide-react";
import { format } from "date-fns";
import { getSession, getSignals, getTranscript } from "../api/client";
import type { Signal, TranscriptSegment } from "../api/client";
import TranscriptBlock from "../components/TranscriptBlock";
import StressTimeline from "../components/StressTimeline";
import AlertCard from "../components/AlertCard";

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
    // Match by time overlap
    const overlapStart = Math.max(s.window_start_ms, segment.start_ms);
    const overlapEnd = Math.min(s.window_end_ms, segment.end_ms);
    if (overlapEnd <= overlapStart) return false;

    // Match by speaker if available
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

  // Agent signal counts
  const agentCounts = detail.signals_by_agent;

  return (
    <div className="mx-auto max-w-6xl">
      {/* Header */}
      <div className="mb-6">
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
          {Object.entries(agentCounts).map(([agent, count]) => (
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

      {/* Main content: transcript + stress chart */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* Transcript Panel — takes 3/5 width on desktop */}
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
            <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
              {segments.map((segment) => (
                <TranscriptBlock
                  key={segment.id}
                  segment={segment}
                  signals={matchSignalsToSegment(segment, signals)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Right column — stress chart + alerts */}
        <div className="lg:col-span-2 space-y-4">
          {/* Stress Timeline Chart */}
          <StressTimeline signals={signals} />

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

          {/* Quick stats */}
          <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
            <h3 className="mb-3 text-sm font-medium text-nexus-text-primary">
              Signal Summary
            </h3>
            <div className="space-y-2 text-xs">
              {Object.entries(agentCounts).map(([agent, count]) => {
                const agentColorMap: Record<string, string> = {
                  voice: "bg-nexus-agent-voice",
                  language: "bg-nexus-agent-language",
                  fusion: "bg-nexus-agent-fusion",
                };
                return (
                  <div key={agent} className="flex items-center gap-2">
                    <span
                      className={`h-2 w-2 rounded-full ${
                        agentColorMap[agent] ?? "bg-nexus-neutral"
                      }`}
                    />
                    <span className="text-nexus-text-secondary capitalize">
                      {agent}
                    </span>
                    <span className="ml-auto font-mono text-nexus-text-primary">
                      {count}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
