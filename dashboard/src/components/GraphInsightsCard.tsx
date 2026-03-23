interface TensionCluster {
  timestamp_ms: number;
  duration_ms: number;
  signal_count: number;
  peak_stress: number;
  has_objection: boolean;
  speaker_id: string;
  severity: string;
}

interface TopicDensity {
  topic_name: string;
  total_signals: number;
  risk_level: string;
  opportunity_level: string;
}

interface Momentum {
  overall_trajectory: string;
  momentum_score: number;
  turning_point_ms: number | null;
}

interface SpeakerPattern {
  response_pattern: string;
  escalation_trend: string;
  contradiction_ratio: number;
}

interface ResolutionPath {
  objection_text: string;
  objection_ms: number;
  resolution_type: string;
  resolution_ms: number;
  time_to_resolve_ms: number;
}

interface GraphAnalytics {
  tension_clusters?: TensionCluster[];
  momentum?: Momentum;
  topic_signal_density?: TopicDensity[];
  speaker_patterns?: Record<string, SpeakerPattern>;
  resolution_paths?: ResolutionPath[];
}

interface GraphInsightsCardProps {
  analytics: GraphAnalytics;
  speakerRoles: Record<string, string>;
}

function formatTime(ms: number): string {
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

const RISK_COLORS: Record<string, string> = {
  high: "var(--stress-high, #EF4444)",
  moderate: "var(--stress-med, #F59E0B)",
  low: "var(--stress-low, #22C55E)",
};

const OPP_COLORS: Record<string, string> = {
  high: "var(--stress-low, #22C55E)",
  moderate: "var(--stress-med, #F59E0B)",
  low: "var(--text-muted, #666)",
};

const TRAJECTORY_ICONS: Record<string, string> = {
  positive: "📈",
  negative: "📉",
  stable: "➡️",
  volatile: "〰️",
};

export default function GraphInsightsCard({ analytics, speakerRoles }: GraphInsightsCardProps) {
  if (!analytics || Object.keys(analytics).length === 0) return null;

  const clusters = analytics.tension_clusters || [];
  const momentum = analytics.momentum;
  const topics = analytics.topic_signal_density || [];
  const patterns = analytics.speaker_patterns || {};
  const resolutions = analytics.resolution_paths || [];

  const hasContent = clusters.length > 0 || momentum?.turning_point_ms || topics.length > 0 ||
    Object.keys(patterns).length > 0 || resolutions.length > 0;

  if (!hasContent) return null;

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 space-y-4">
      <h3 className="text-sm font-medium text-nexus-text-primary">
        Graph Insights
        <span className="ml-2 text-[10px] font-normal text-nexus-text-muted">
          Topology-derived patterns
        </span>
      </h3>

      {/* Tension Clusters */}
      {clusters.length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary font-medium mb-1.5">
            🔴 Tension Clusters
          </div>
          <div className="space-y-1">
            {clusters.map((c, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-nexus-text-primary">
                <span
                  className="h-2 w-2 rounded-full shrink-0"
                  style={{ backgroundColor: c.severity === "high" ? RISK_COLORS.high : RISK_COLORS.moderate }}
                />
                <span>
                  {c.severity === "high" ? "High" : "Moderate"} tension at{" "}
                  <span className="font-mono">{formatTime(c.timestamp_ms)}</span>
                  {" — "}
                  {c.signal_count} signals
                  {c.has_objection && ", includes objection"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Momentum */}
      {momentum && momentum.overall_trajectory !== "stable" && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary font-medium mb-1">
            {TRAJECTORY_ICONS[momentum.overall_trajectory] || "📊"} Momentum
          </div>
          <div className="text-xs text-nexus-text-primary">
            Conversation trajectory:{" "}
            <span className="font-medium" style={{
              color: momentum.overall_trajectory === "positive"
                ? "var(--stress-low, #22C55E)"
                : momentum.overall_trajectory === "negative"
                ? "var(--stress-high, #EF4444)"
                : "var(--text-primary)"
            }}>
              {momentum.overall_trajectory}
            </span>
            {momentum.turning_point_ms && (
              <span className="text-nexus-text-muted">
                {" "}(shifted at <span className="font-mono">{formatTime(momentum.turning_point_ms)}</span>)
              </span>
            )}
          </div>
        </div>
      )}

      {/* Topic Analysis */}
      {topics.length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary font-medium mb-1.5">
            📊 Topic Analysis
          </div>
          <div className="space-y-1">
            {topics.map((t, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="text-nexus-text-primary flex-1 min-w-0 truncate">{t.topic_name}</span>
                <span
                  className="rounded-full px-1.5 py-0.5 text-[9px] font-medium"
                  style={{
                    backgroundColor: `color-mix(in srgb, ${RISK_COLORS[t.risk_level]} 15%, transparent)`,
                    color: RISK_COLORS[t.risk_level],
                  }}
                >
                  {t.risk_level} risk
                </span>
                <span
                  className="rounded-full px-1.5 py-0.5 text-[9px] font-medium"
                  style={{
                    backgroundColor: `color-mix(in srgb, ${OPP_COLORS[t.opportunity_level]} 15%, transparent)`,
                    color: OPP_COLORS[t.opportunity_level],
                  }}
                >
                  {t.opportunity_level} opp
                </span>
                <span className="font-mono text-nexus-text-muted text-[10px] w-6 text-right">
                  {t.total_signals}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Speaker Patterns */}
      {Object.keys(patterns).length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary font-medium mb-1.5">
            👤 Speaker Patterns
          </div>
          <div className="space-y-1">
            {Object.entries(patterns).map(([sid, p]) => {
              const role = speakerRoles[sid] || sid;
              return (
                <div key={sid} className="text-xs text-nexus-text-primary">
                  <span className="font-medium">{role}</span>:{" "}
                  <span className="capitalize">{p.response_pattern}</span>
                  {p.escalation_trend !== "stable" && (
                    <span className="text-nexus-text-muted"> (stress {p.escalation_trend})</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Resolution Paths */}
      {resolutions.length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary font-medium mb-1.5">
            ✅ Resolution Paths
          </div>
          <div className="space-y-1">
            {resolutions.map((r, i) => (
              <div key={i} className="text-xs text-nexus-text-primary">
                Objection at{" "}
                <span className="font-mono">{formatTime(r.objection_ms)}</span>
                {" → resolved in "}
                <span className="font-medium text-nexus-stress-low">
                  {Math.round(r.time_to_resolve_ms / 1000)}s
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
