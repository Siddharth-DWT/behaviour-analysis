import { AlertTriangle, AlertCircle, Info, Star } from "lucide-react";
import type { Alert } from "../api/client";

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

const SEVERITY_CONFIG: Record<
  string,
  { icon: typeof AlertTriangle; color: string; bg: string; borderColor: string; label: string }
> = {
  red: {
    icon: AlertCircle,
    color: "text-nexus-stress-high",
    bg: "bg-stress-high-10",
    borderColor: "border-l-nexus-stress-high",
    label: "CRITICAL",
  },
  orange: {
    icon: AlertTriangle,
    color: "text-nexus-alert",
    bg: "bg-alert-10",
    borderColor: "border-l-nexus-alert",
    label: "ALERT",
  },
  yellow: {
    icon: Info,
    color: "text-nexus-stress-med",
    bg: "bg-stress-med-10",
    borderColor: "border-l-nexus-stress-med",
    label: "NOTICE",
  },
  green: {
    icon: Star,
    color: "text-nexus-stress-low",
    bg: "bg-stress-low-10",
    borderColor: "border-l-nexus-stress-low",
    label: "INSIGHT",
  },
};

export default function AlertCard({ alert }: { alert: Alert }) {
  const config = SEVERITY_CONFIG[alert.severity] ?? SEVERITY_CONFIG.yellow;
  const Icon = config.icon;

  return (
    <div
      className={`rounded-lg border border-nexus-border border-l-[3px] ${config.borderColor} ${config.bg} p-3`}
    >
      <div className="flex items-start gap-2">
        <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${config.color}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-xs">
            <span className={`font-mono font-bold ${config.color}`}>
              {config.label}
            </span>
            <span className="text-nexus-text-muted">
              {formatTime(alert.timestamp_ms)}
            </span>
            {alert.speaker_label && (
              <span className="text-nexus-text-secondary">
                {alert.speaker_label}
              </span>
            )}
          </div>
          <p className="mt-0.5 text-sm font-medium text-nexus-text-primary">
            {alert.title}
          </p>
          {alert.description && (
            <p className="mt-0.5 text-xs text-nexus-text-secondary leading-relaxed">
              {alert.description}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
