import { Link } from "react-router-dom";
import { Clock, Users, AlertTriangle, ChevronRight } from "lucide-react";
import { format } from "date-fns";
import type { Session } from "../api/client";

function formatDuration(ms: number | null): string {
  if (!ms) return "--";
  const totalSec = Math.round(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  if (min === 0) return `${sec}s`;
  return sec > 0 ? `${min}m ${sec}s` : `${min}m`;
}

const STATUS_STYLES: Record<string, string> = {
  completed: "bg-stress-low-20 text-nexus-stress-low",
  processing: "bg-stress-med-20 text-nexus-stress-med",
  analysing: "bg-accent-blue-20 text-nexus-accent-blue",
  failed: "bg-stress-high-20 text-nexus-stress-high",
  created: "bg-neutral-20 text-nexus-neutral",
};

const MEETING_TYPE_LABELS: Record<string, string> = {
  sales_call: "Sales Call",
  client_meeting: "Client Meeting",
  internal: "Internal",
  interview: "Interview",
  other: "Other",
};

export default function SessionCard({ session }: { session: Session }) {
  const statusStyle = STATUS_STYLES[session.status] ?? STATUS_STYLES.created;

  return (
    <Link
      to={`/sessions/${session.id}`}
      className="block rounded-lg border border-nexus-border bg-nexus-surface p-4 transition-colors hover:border-accent-blue-40 hover:bg-nexus-surface-hover"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-sm font-medium text-nexus-text-primary truncate">
              {session.title || "Untitled Session"}
            </h3>
            <span
              className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-mono font-bold uppercase ${statusStyle}`}
            >
              {session.status}
            </span>
          </div>

          <div className="flex items-center gap-4 text-xs text-nexus-text-muted">
            <span className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {formatDuration(session.duration_ms)}
            </span>
            <span className="flex items-center gap-1">
              <Users className="h-3 w-3" />
              {session.speaker_count ?? "--"} speakers
              {session.participant_count != null &&
                session.participant_count !== session.speaker_count && (
                  <span className="text-nexus-accent-blue">
                    · {session.participant_count} visible
                  </span>
              )}
            </span>
            <span className="rounded bg-nexus-surface-hover px-1.5 py-0.5 font-mono">
              {MEETING_TYPE_LABELS[session.meeting_type] ?? session.meeting_type}
            </span>
          </div>

          <div className="mt-2 text-xs text-nexus-text-muted">
            {format(new Date(session.created_at), "MMM d, yyyy 'at' h:mm a")}
          </div>
        </div>

        <ChevronRight className="h-4 w-4 shrink-0 text-nexus-text-muted mt-1" />
      </div>
    </Link>
  );
}
