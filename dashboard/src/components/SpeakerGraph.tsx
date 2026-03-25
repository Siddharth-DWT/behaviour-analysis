import { useState } from "react";
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
  if (stress > 0.5) return "#EF4444";
  if (stress > 0.3) return "#F59E0B";
  return "#22C55E";
}

const SPEAKER_FILLS = ["#4F8BFF", "#8B5CF6", "#F59E0B", "#10B981", "#EC4899", "#06B6D4", "#F97316", "#6366F1"];

function getSpeakerName(
  label: string,
  role: string | undefined,
  entities: Entities
): { name: string; role: string } {
  let displayName = "";
  let displayRole = role || "";

  if (entities.people) {
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

/** Count per-pair speaker exchanges from sorted vocal_stress_score signals */
function computePairExchanges(
  signals: Signal[],
  speakerLabels: string[]
): Map<string, number> {
  const pairKey = (a: string, b: string) => [a, b].sort().join("↔");
  const exchanges = new Map<string, number>();

  // Init all pairs to 0
  for (let i = 0; i < speakerLabels.length; i++) {
    for (let j = i + 1; j < speakerLabels.length; j++) {
      exchanges.set(pairKey(speakerLabels[i], speakerLabels[j]), 0);
    }
  }

  // Count transitions between consecutive stress signals
  const stressSignals = signals
    .filter((s) => s.signal_type === "vocal_stress_score" && s.speaker_label)
    .sort((a, b) => (a.window_start_ms || 0) - (b.window_start_ms || 0));

  for (let i = 1; i < stressSignals.length; i++) {
    const prev = stressSignals[i - 1].speaker_label!;
    const curr = stressSignals[i].speaker_label!;
    if (prev !== curr) {
      const key = pairKey(prev, curr);
      exchanges.set(key, (exchanges.get(key) || 0) + 1);
    }
  }

  return exchanges;
}

export default function SpeakerGraph({
  speakers,
  contentType,
  entities,
  signals,
  speakerRoles,
}: SpeakerGraphProps) {
  const [hoveredSpeaker, setHoveredSpeaker] = useState<string | null>(null);

  if (speakers.length < 2) return null;

  const labels = speakers.map((s) => s.label);
  const pairExchanges = computePairExchanges(signals, labels);
  const totalExchanges = Array.from(pairExchanges.values()).reduce((a, b) => a + b, 0);
  const pairKey = (a: string, b: string) => [a, b].sort().join("↔");

  const W = 460;
  const H = speakers.length <= 3 ? 240 : 280;
  const CY = H / 2;

  // Position speakers in a circle (or left-right for 2)
  const positions = speakers.map((_, i) => {
    const count = speakers.length;
    if (count === 2) {
      return { cx: i === 0 ? 120 : W - 120, cy: CY };
    }
    const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
    const rx = Math.min(150, W / 2 - 80);
    const ry = Math.min(80, H / 2 - 50);
    return {
      cx: W / 2 + Math.cos(angle) * rx,
      cy: H / 2 + Math.sin(angle) * ry,
    };
  });

  // Circle radii proportional to talk time (bigger = more talk time)
  const maxPct = Math.max(...speakers.map((s) => s.talkTimePct), 1);
  const radii = speakers.map((s) => {
    const norm = s.talkTimePct / maxPct; // 0 to 1
    return Math.max(32, Math.min(60, 28 + norm * 32));
  });

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
      <h3 className="mb-2 text-sm font-medium text-nexus-text-primary">
        Speaker Map
      </h3>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ maxHeight: 300 }}
      >
        {/* Connection lines with per-pair exchange counts */}
        {speakers.map((_, i) =>
          speakers.map((_, j) => {
            if (j <= i) return null;
            const key = pairKey(labels[i], labels[j]);
            const count = pairExchanges.get(key) || 0;
            const maxPairCount = Math.max(...Array.from(pairExchanges.values()), 1);
            const lineWidth = Math.max(1, Math.min(5, 1 + (count / maxPairCount) * 4));
            const opacity = count === 0 ? 0.2 : 0.3 + (count / maxPairCount) * 0.5;

            const mx = (positions[i].cx + positions[j].cx) / 2;
            const my = (positions[i].cy + positions[j].cy) / 2;

            // Offset label perpendicular to the line to avoid overlap
            const dx = positions[j].cx - positions[i].cx;
            const dy = positions[j].cy - positions[i].cy;
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            const offsetX = (-dy / len) * 12;
            const offsetY = (dx / len) * 12;

            return (
              <g key={`line-${i}-${j}`}>
                <line
                  x1={positions[i].cx}
                  y1={positions[i].cy}
                  x2={positions[j].cx}
                  y2={positions[j].cy}
                  stroke="var(--border, #444)"
                  strokeWidth={lineWidth}
                  strokeDasharray={count < 3 ? "6 4" : "none"}
                  opacity={opacity}
                />
                {count > 0 && (
                  <text
                    x={mx + offsetX}
                    y={my + offsetY}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={9}
                    fill="var(--text-muted, #888)"
                  >
                    {count} exchange{count !== 1 ? "s" : ""}
                  </text>
                )}
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
          const fillColor = SPEAKER_FILLS[i % SPEAKER_FILLS.length];
          const stat = statLine(contentType, { ...speaker, role: assignedRole });
          const isHovered = hoveredSpeaker === speaker.label;
          const scale = isHovered ? 1.08 : 1;

          return (
            <g
              key={speaker.label}
              onMouseEnter={() => setHoveredSpeaker(speaker.label)}
              onMouseLeave={() => setHoveredSpeaker(null)}
              style={{ cursor: "pointer", transition: "transform 0.2s" }}
              transform={`translate(${cx}, ${cy}) scale(${scale}) translate(${-cx}, ${-cy})`}
            >
              {/* Stress ring */}
              <circle
                cx={cx}
                cy={cy}
                r={r + 3}
                fill="none"
                stroke={borderColor}
                strokeWidth={3}
                opacity={isHovered ? 1 : 0.8}
              />
              {/* Main circle */}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={fillColor}
                opacity={isHovered ? 0.25 : 0.15}
                stroke={fillColor}
                strokeWidth={1.5}
              />
              {/* Name */}
              <text
                x={cx}
                y={cy - (displayRole ? 6 : 0)}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={11}
                fontWeight={600}
                fill="var(--text-primary, #E8E8E8)"
              >
                {displayName.length > 12 ? displayName.slice(0, 11) + "…" : displayName}
              </text>
              {/* Role subtitle (inside circle, below name) */}
              {displayRole && (
                <text
                  x={cx}
                  y={cy + 10}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={8}
                  fill={fillColor}
                  opacity={0.9}
                >
                  ({displayRole})
                </text>
              )}
              {/* Stat below circle — outside, no overlap */}
              <text
                x={cx}
                y={cy + r + 14}
                textAnchor="middle"
                fontSize={9}
                fill="var(--text-secondary, #AAA)"
              >
                {stat}
              </text>
              {/* Stress badge above circle */}
              <text
                x={cx}
                y={cy - r - 8}
                textAnchor="middle"
                fontSize={8}
                fill={borderColor}
              >
                Stress {Math.round(speaker.avgStress * 100)}%
              </text>

              {/* Talk time badge — small number inside circle at bottom */}
              <text
                x={cx}
                y={cy + (displayRole ? 22 : 14)}
                textAnchor="middle"
                fontSize={8}
                fill="var(--text-muted, #999)"
              >
                {Math.round(speaker.talkTimePct)}%
              </text>
            </g>
          );
        })}

        {/* Total exchanges label */}
        {totalExchanges > 0 && (
          <text
            x={W / 2}
            y={H - 6}
            textAnchor="middle"
            fontSize={9}
            fill="var(--text-muted, #666)"
          >
            {totalExchanges} total exchanges
          </text>
        )}
      </svg>
    </div>
  );
}
