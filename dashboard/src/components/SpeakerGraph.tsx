import type { Signal } from "../api/client";

interface SpeakerInfo {
  label: string;
  role?: string;
  name?: string;
  avgStress: number;
  talkTimePct: number;
  buyingSignalCount: number;
  objectionCount: number;
  avgPower: number;
  avgSentiment: number;
  fillerCount: number;
}

interface Entities {
  people?: Array<{ name: string; role: string; first_mention_ms: number }>;
  [key: string]: unknown;
}

interface SpeakerGraphProps {
  speakers: SpeakerInfo[];
  contentType: string;
  entities: Entities;
  signals: Signal[];
  speakerRoles: Record<string, string>;
}

function stressColor(stress: number): string {
  if (stress > 0.5) return "var(--stress-high, #EF4444)";
  if (stress > 0.3) return "var(--stress-med, #F59E0B)";
  return "var(--stress-low, #22C55E)";
}

function speakerFill(idx: number): string {
  const fills = [
    "var(--accent-blue, #4F8BFF)",
    "var(--accent-purple, #8B5CF6)",
    "var(--stress-med, #F59E0B)",
    "var(--engagement, #10B981)",
  ];
  return fills[idx % fills.length];
}

function getSpeakerName(
  label: string,
  role: string | undefined,
  entities: Entities
): { name: string; role: string } {
  let displayName = "";
  let displayRole = role || "";

  if (entities.people) {
    // Match by speaker_label first, then by role
    const byLabel = entities.people.find(
      (p) => (p as Record<string, unknown>).speaker_label === label
    );
    const byRole = entities.people.find(
      (p) => p.role?.toLowerCase() === role?.toLowerCase()
    );
    const match = byLabel || byRole;
    if (match) {
      displayName = match.name;
      if (!displayRole && match.role) {
        displayRole = match.role.charAt(0).toUpperCase() + match.role.slice(1);
      }
    }
  }

  // If no name extracted, show role as primary text (not "Speaker_0")
  if (!displayName || displayName === label) {
    displayName = displayRole || label;
    displayRole = displayRole && displayRole !== displayName ? displayRole : "";
  }

  return { name: displayName, role: displayRole };
}

function statLine(contentType: string, speaker: SpeakerInfo): string {
  const role = speaker.role?.toLowerCase() || "";
  if (contentType === "sales_call") {
    if (role === "prospect" || role === "buyer") {
      if (speaker.buyingSignalCount > 0)
        return `${speaker.buyingSignalCount} buying signal${speaker.buyingSignalCount > 1 ? "s" : ""}`;
      if (speaker.objectionCount > 0)
        return `${speaker.objectionCount} objection${speaker.objectionCount > 1 ? "s" : ""}`;
      return "No buying signals";
    }
    return `Power ${Math.round(speaker.avgPower * 100)}%`;
  }
  if (contentType === "interview") {
    if (role === "candidate") return `Confidence ${Math.round(speaker.avgPower * 100)}%`;
    return `${speaker.fillerCount} questions`;
  }
  return `${Math.round(speaker.talkTimePct)}% talk time`;
}

export default function SpeakerGraph({
  speakers,
  contentType,
  entities,
  signals,
  speakerRoles,
}: SpeakerGraphProps) {
  if (speakers.length < 2) return null;

  // Count actual speaker alternations (not signal count)
  let turnCount = 0;
  for (let i = 1; i < signals.length; i++) {
    if (
      signals[i].speaker_label &&
      signals[i - 1].speaker_label &&
      signals[i].speaker_label !== signals[i - 1].speaker_label &&
      signals[i].signal_type === "vocal_stress_score" &&
      signals[i - 1].signal_type === "vocal_stress_score"
    ) {
      turnCount++;
    }
  }

  const W = 460;
  const H = 220;
  const CY = H / 2;

  // Position speakers
  const positions = speakers.map((_, i) => {
    const count = speakers.length;
    if (count === 2) {
      return { cx: i === 0 ? 110 : W - 110, cy: CY };
    }
    const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
    return {
      cx: W / 2 + Math.cos(angle) * 130,
      cy: H / 2 + Math.sin(angle) * 70,
    };
  });

  // Circle radii based on talk time
  const radii = speakers.map((s) => {
    const pct = s.talkTimePct || 50;
    return Math.max(36, Math.min(58, 30 + pct * 0.5));
  });

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
      <h3 className="mb-2 text-sm font-medium text-nexus-text-primary">
        Speaker Map
      </h3>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ maxHeight: 240 }}
      >
        {/* Connection lines */}
        {speakers.length >= 2 &&
          speakers.map((_, i) =>
            speakers.map((_, j) => {
              if (j <= i) return null;
              const lineWidth = Math.min(4, 1.5 + turnCount * 0.02);
              return (
                <g key={`line-${i}-${j}`}>
                  <line
                    x1={positions[i].cx}
                    y1={positions[i].cy}
                    x2={positions[j].cx}
                    y2={positions[j].cy}
                    stroke="var(--border, #333)"
                    strokeWidth={lineWidth}
                    strokeDasharray={turnCount < 5 ? "6 4" : "none"}
                    opacity={0.5}
                  />
                  {/* Turn count label */}
                  <text
                    x={(positions[i].cx + positions[j].cx) / 2}
                    y={(positions[i].cy + positions[j].cy) / 2 - 8}
                    textAnchor="middle"
                    fontSize={10}
                    fill="var(--text-muted, #666)"
                  >
                    {turnCount > 0 ? `${turnCount} exchanges` : ""}
                  </text>
                </g>
              );
            })
          )}

        {/* Speaker circles */}
        {speakers.map((speaker, i) => {
          const { cx, cy } = positions[i];
          const r = radii[i];
          const assignedRole = speakerRoles[speaker.label] || speaker.role;
          const { name: displayName, role: displayRole } = getSpeakerName(
            speaker.label, assignedRole, entities
          );
          const borderColor = stressColor(speaker.avgStress);
          const fillColor = speakerFill(i);
          const stat = statLine(contentType, { ...speaker, role: assignedRole });

          return (
            <g key={speaker.label}>
              {/* Stress ring */}
              <circle
                cx={cx}
                cy={cy}
                r={r + 3}
                fill="none"
                stroke={borderColor}
                strokeWidth={3}
              />
              {/* Main circle */}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={fillColor}
                opacity={0.15}
                stroke={fillColor}
                strokeWidth={1.5}
              />
              {/* Name */}
              <text
                x={cx}
                y={cy - 8}
                textAnchor="middle"
                fontSize={12}
                fontWeight={600}
                fill="var(--text-primary, #E8E8E8)"
              >
                {displayName}
              </text>
              {/* Role subtitle */}
              {displayRole && (
                <text
                  x={cx}
                  y={cy + 6}
                  textAnchor="middle"
                  fontSize={10}
                  fill={fillColor}
                  opacity={0.9}
                >
                  {displayRole}
                </text>
              )}
              {/* Speaker label */}
              <text
                x={cx}
                y={cy + 18}
                textAnchor="middle"
                fontSize={8}
                fill="var(--text-muted, #888)"
              >
                ({speaker.label})
              </text>
              {/* Stat below circle */}
              <text
                x={cx}
                y={cy + r + 18}
                textAnchor="middle"
                fontSize={10}
                fill="var(--text-secondary, #AAA)"
              >
                {stat}
              </text>
              {/* Stress badge */}
              <text
                x={cx}
                y={cy - r - 8}
                textAnchor="middle"
                fontSize={9}
                fill={borderColor}
              >
                Stress {Math.round(speaker.avgStress * 100)}%
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
