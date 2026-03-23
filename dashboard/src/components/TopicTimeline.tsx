import type { Signal } from "../api/client";

interface Topic {
  name: string;
  start_ms: number;
  end_ms: number;
}

interface TopicTimelineProps {
  topics: Topic[];
  signals: Signal[];
  durationMs: number;
}

const TOPIC_COLORS = [
  "var(--accent-blue, #4F8BFF)",
  "var(--accent-purple, #8B5CF6)",
  "var(--engagement, #10B981)",
  "var(--stress-med, #F59E0B)",
  "var(--agent-gaze, #EC4899)",
  "var(--agent-voice, #6366F1)",
  "var(--stress-low, #22C55E)",
];

const SIGNAL_DOT_MAP: Record<string, { color: string; label: string }> = {
  vocal_stress_score: { color: "var(--stress-high, #EF4444)", label: "Stress" },
  buying_signal: { color: "var(--stress-low, #22C55E)", label: "Buy" },
  objection_signal: { color: "var(--stress-high, #EF4444)", label: "Obj" },
  pitch_elevation_flag: { color: "var(--stress-med, #F59E0B)", label: "Pitch" },
  verbal_incongruence: { color: "var(--agent-fusion, #F97316)", label: "Incong" },
  credibility_assessment: { color: "var(--agent-fusion, #F97316)", label: "Cred" },
  tension_cluster: { color: "var(--agent-fusion, #F97316)", label: "Tension" },
  momentum_shift: { color: "var(--accent-purple, #8B5CF6)", label: "Momentum" },
};

function formatTime(ms: number): string {
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export default function TopicTimeline({ topics, signals, durationMs }: TopicTimelineProps) {
  if (!topics.length || !durationMs) return null;

  // Filter to only noteworthy signals for dots
  const dotSignals = signals.filter((s) => {
    if (s.signal_type === "vocal_stress_score" && (s.value == null || s.value < 0.35)) return false;
    return s.signal_type in SIGNAL_DOT_MAP;
  });

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
      <h3 className="mb-3 text-sm font-medium text-nexus-text-primary">
        Conversation Phases
      </h3>

      {/* Phase bar */}
      <div className="relative h-10 w-full rounded-md overflow-hidden flex">
        {topics.map((topic, i) => {
          const widthPct = ((topic.end_ms - topic.start_ms) / durationMs) * 100;
          return (
            <div
              key={i}
              className="relative h-full flex items-center justify-center overflow-hidden border-r border-nexus-bg last:border-r-0"
              style={{
                width: `${widthPct}%`,
                backgroundColor: TOPIC_COLORS[i % TOPIC_COLORS.length],
                opacity: 0.8,
              }}
              title={`${topic.name} (${formatTime(topic.start_ms)} - ${formatTime(topic.end_ms)})`}
            >
              {widthPct > 12 && (
                <span className="text-[10px] font-medium text-white truncate px-1">
                  {topic.name}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Signal dots overlay */}
      <div className="relative h-5 w-full mt-1">
        {dotSignals.map((s, i) => {
          const leftPct = (s.window_start_ms / durationMs) * 100;
          const config = SIGNAL_DOT_MAP[s.signal_type];
          if (!config) return null;
          return (
            <div
              key={i}
              className="absolute top-0.5 h-3 w-3 rounded-full border border-nexus-bg"
              style={{
                left: `${Math.min(leftPct, 98)}%`,
                backgroundColor: config.color,
              }}
              title={`${config.label} (${formatTime(s.window_start_ms)}) — ${s.value_text || ""} ${s.value != null ? Math.round(s.value * 100) + "%" : ""}`}
            />
          );
        })}
      </div>

      {/* Time labels */}
      <div className="flex justify-between text-[10px] text-nexus-text-muted mt-0.5">
        <span>0:00</span>
        <span>{formatTime(durationMs)}</span>
      </div>

      {/* Legend — topics */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2">
        {topics.map((t, i) => (
          <span key={i} className="flex items-center gap-1 text-[10px] text-nexus-text-secondary">
            <span
              className="h-2 w-2 rounded-sm"
              style={{ backgroundColor: TOPIC_COLORS[i % TOPIC_COLORS.length], opacity: 0.8 }}
            />
            {t.name}
          </span>
        ))}
      </div>
      {/* Legend — signal dots */}
      {dotSignals.length > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1">
          {Object.entries(SIGNAL_DOT_MAP)
            .filter(([type]) => dotSignals.some((s) => s.signal_type === type))
            .map(([type, config]) => (
              <span key={type} className="flex items-center gap-1 text-[10px] text-nexus-text-muted">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: config.color }} />
                {config.label}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}
