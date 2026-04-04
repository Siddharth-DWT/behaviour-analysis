import { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { Signal } from "../api/client";

interface Props {
  signals: Signal[];
  speakerRoles?: Record<string, string>;
  speakerNames?: Record<string, string>;
}

interface DataPoint {
  time: number;
  timeLabel: string;
  [speaker: string]: number | string;
}

function getCSSVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

export default function StressTimeline({ signals, speakerRoles, speakerNames }: Props) {
  const [themeColors, setThemeColors] = useState({
    speakerColors: ["#4F8BFF", "#8B5CF6", "#F59E0B", "#10B981", "#EC4899"],
    border: "#2D3348",
    textSecondary: "#8B93A7",
    textPrimary: "#E8ECF4",
    surface: "#1A1D27",
  });

  useEffect(() => {
    function updateColors() {
      setThemeColors({
        speakerColors: [
          getCSSVar("--accent-blue") || "#4F8BFF",
          getCSSVar("--accent-purple") || "#8B5CF6",
          getCSSVar("--stress-med") || "#F59E0B",
          getCSSVar("--engagement") || "#10B981",
          getCSSVar("--agent-gaze") || "#EC4899",
        ],
        border: getCSSVar("--border") || "#2D3348",
        textSecondary: getCSSVar("--text-secondary") || "#8B93A7",
        textPrimary: getCSSVar("--text-primary") || "#E8ECF4",
        surface: getCSSVar("--bg-surface") || "#1A1D27",
      });
    }

    updateColors();

    // Re-read colors when theme class changes
    const observer = new MutationObserver(() => updateColors());
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  // Filter to stress signals only
  const stressSignals = signals.filter(
    (s) => s.agent === "voice" && s.signal_type === "vocal_stress_score" && s.value != null
  );

  if (stressSignals.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center rounded-lg border border-nexus-border bg-nexus-surface text-sm text-nexus-text-muted">
        No stress data available
      </div>
    );
  }

  // Get unique speakers
  const speakerLabels = Array.from(
    new Set(stressSignals.map((s) => s.speaker_label || s.speaker_id || "Unknown"))
  );

  // Build display names with real names + roles
  const speakerDisplayNames: Record<string, string> = {};
  for (const label of speakerLabels) {
    const name = speakerNames?.[label];
    const role = speakerRoles?.[label];
    if (name && role) speakerDisplayNames[label] = `${name} (${role})`;
    else if (name) speakerDisplayNames[label] = name;
    else if (role) speakerDisplayNames[label] = `${label} (${role})`;
    else speakerDisplayNames[label] = label;
  }

  // Build time-series data points
  const timeMap = new Map<number, DataPoint>();

  for (const signal of stressSignals) {
    const timeKey = signal.window_start_ms;
    const speaker = signal.speaker_label || signal.speaker_id || "Unknown";
    const displayName = speakerDisplayNames[speaker] || speaker;

    if (!timeMap.has(timeKey)) {
      timeMap.set(timeKey, {
        time: timeKey,
        timeLabel: formatTime(timeKey),
      });
    }

    const point = timeMap.get(timeKey)!;
    point[displayName] = Number((signal.value ?? 0).toFixed(3));
  }

  const data = Array.from(timeMap.values()).sort((a, b) => a.time - b.time);
  const displaySpeakers = speakerLabels.map((l) => speakerDisplayNames[l] || l);

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
      <h3 className="mb-3 text-sm font-medium text-nexus-text-primary">
        Stress Timeline
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
          <defs>
            {displaySpeakers.map((speaker, i) => (
              <linearGradient
                key={speaker}
                id={`stress-gradient-${i}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop
                  offset="5%"
                  stopColor={themeColors.speakerColors[i % themeColors.speakerColors.length]}
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor={themeColors.speakerColors[i % themeColors.speakerColors.length]}
                  stopOpacity={0.05}
                />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={themeColors.border} />
          <XAxis
            dataKey="timeLabel"
            tick={{ fill: themeColors.textSecondary, fontSize: 10 }}
            tickLine={{ stroke: themeColors.border }}
            axisLine={{ stroke: themeColors.border }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: themeColors.textSecondary, fontSize: 10 }}
            tickLine={{ stroke: themeColors.border }}
            axisLine={{ stroke: themeColors.border }}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: themeColors.surface,
              border: `1px solid ${themeColors.border}`,
              borderRadius: "8px",
              fontSize: 12,
              color: themeColors.textPrimary,
            }}
            formatter={(value: number) => [value.toFixed(3), ""]}
            labelStyle={{ color: themeColors.textSecondary }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: themeColors.textSecondary }}
          />
          {displaySpeakers.map((speaker, i) => (
            <Area
              key={speaker}
              type="monotone"
              dataKey={speaker}
              stroke={themeColors.speakerColors[i % themeColors.speakerColors.length]}
              fillOpacity={1}
              fill={`url(#stress-gradient-${i})`}
              strokeWidth={1.5}
              dot={false}
              name={speaker}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>

      {/* Stress level legend */}
      <div className="mt-2 flex items-center gap-4 text-[10px] text-nexus-text-muted">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-nexus-stress-low" />
          Low (&lt;0.30)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-nexus-stress-med" />
          Moderate (0.30-0.60)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-nexus-stress-high" />
          High (&gt;0.60)
        </span>
      </div>
    </div>
  );
}
