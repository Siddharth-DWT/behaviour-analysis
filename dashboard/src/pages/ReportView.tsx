import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  Loader2,
  Clock,
  Users,
  Lightbulb,
  Target,
  Sparkles,
  RefreshCw,
} from "lucide-react";
import { format } from "date-fns";
import { getSession, getReport, getSignals } from "../api/client";
import StressTimeline from "../components/StressTimeline";

function formatDuration(ms: number | null): string {
  if (!ms) return "--";
  const totalSec = Math.round(ms / 1000);
  const min = Math.floor(totalSec / 60);
  return `${min} min`;
}

export default function ReportView() {
  const { id } = useParams<{ id: string }>();

  const { data: detail } = useQuery({
    queryKey: ["session", id],
    queryFn: () => getSession(id!),
    enabled: !!id,
  });

  const {
    data: reportData,
    isLoading: loadingReport,
    error: reportError,
    refetch,
  } = useQuery({
    queryKey: ["report", id],
    queryFn: () => getReport(id!),
    enabled: !!id,
  });

  const { data: signalData } = useQuery({
    queryKey: ["signals", id],
    queryFn: () => getSignals(id!, { limit: 5000 }),
    enabled: !!id,
  });

  const session = detail?.session;
  const report = reportData?.report;
  const content = report?.content;
  const signals = signalData?.signals ?? [];

  // Build speaker role map from signal metadata
  const speakerRoles: Record<string, string> = {};
  if (content && (content as Record<string, unknown>)["speaker_roles"]) {
    const roles = (content as Record<string, unknown>)["speaker_roles"] as Record<string, string>;
    Object.assign(speakerRoles, roles);
  }
  for (const sig of signals) {
    if (sig.metadata && typeof sig.metadata === "object") {
      const meta = sig.metadata as Record<string, unknown>;
      if (meta["speaker_role"] && sig.speaker_label) {
        speakerRoles[sig.speaker_label] = String(meta["speaker_role"]);
      }
    }
  }

  if (loadingReport) {
    return (
      <div className="flex items-center justify-center py-20 text-sm text-nexus-text-muted">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading report...
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl">
      {/* Header */}
      <div className="mb-6">
        <Link
          to={`/sessions/${id}`}
          className="mb-3 inline-flex items-center gap-1 text-xs text-nexus-text-muted hover:text-nexus-accent-blue"
        >
          <ArrowLeft className="h-3 w-3" />
          Back to Session
        </Link>

        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-lg font-semibold text-nexus-text-primary">
              Session Report
            </h1>
            {session && (
              <div className="mt-1 flex items-center gap-4 text-xs text-nexus-text-muted">
                <span>{session.title || "Untitled"}</span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {formatDuration(session.duration_ms)}
                </span>
                <span className="flex items-center gap-1">
                  <Users className="h-3 w-3" />
                  {session.speaker_count ?? "--"} speakers
                </span>
                <span>
                  {format(new Date(session.created_at), "MMM d, yyyy")}
                </span>
              </div>
            )}
          </div>

          <button
            onClick={() => refetch()}
            className="flex items-center gap-1 rounded bg-nexus-surface px-2 py-1 text-xs text-nexus-text-secondary hover:bg-nexus-surface-hover"
            title="Regenerate report"
          >
            <RefreshCw className="h-3 w-3" />
            Regenerate
          </button>
        </div>
      </div>

      {/* Error state */}
      {reportError && (
        <div className="mb-6 rounded-lg border border-stress-high-30 bg-stress-high-10 p-4 text-sm text-nexus-stress-high">
          Failed to load report: {(reportError as Error).message}
        </div>
      )}

      {/* No report */}
      {!report && !reportError && (
        <div className="flex flex-col items-center justify-center py-20 text-nexus-text-muted">
          <Sparkles className="mb-3 h-8 w-8 opacity-40" />
          <p className="text-sm">No report available yet</p>
          <p className="mt-1 text-xs">
            Reports are generated automatically after analysis
          </p>
        </div>
      )}

      {/* Report content */}
      {content && (
        <div className="space-y-6">
          {/* Executive Summary */}
          {content.executive_summary && (
            <section className="rounded-lg border border-accent-purple-30 bg-accent-purple-5 p-5">
              <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
                <Sparkles className="h-4 w-4" />
                Executive Summary
              </h2>
              <p className="text-sm leading-relaxed text-nexus-text-primary">
                {content.executive_summary}
              </p>
            </section>
          )}

          {/* Key Moments */}
          {content.key_moments && content.key_moments.length > 0 && (
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

          {/* Stress Timeline */}
          <StressTimeline signals={signals} speakerRoles={speakerRoles} />

          {/* Cross-modal Insights */}
          {content.cross_modal_insights &&
            content.cross_modal_insights.length > 0 && (
              <section className="rounded-lg border border-accent-purple-30 bg-accent-purple-5 p-5">
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

          {/* Recommendations */}
          {content.recommendations && content.recommendations.length > 0 && (
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

          {/* Generated timestamp */}
          {report?.generated_at && (
            <p className="text-center text-[10px] text-nexus-text-muted">
              Generated{" "}
              {format(new Date(report.generated_at), "MMM d, yyyy 'at' h:mm a")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
