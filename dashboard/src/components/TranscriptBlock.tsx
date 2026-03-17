import type { TranscriptSegment, Signal } from "../api/client";
import SignalBadge, { filterSmartBadges } from "./SignalBadge";

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

/**
 * Determine left-border color based on signal content.
 * Green = buying signal or very positive sentiment
 * Red = high stress or objection
 * Orange = fusion alert
 * Default = transparent (no accent)
 */
function getBorderColor(signals: Signal[]): string {
  let hasHighStress = false;
  let hasObjection = false;
  let hasFusion = false;
  let hasBuyingSignal = false;
  let hasPositiveSentiment = false;

  for (const s of signals) {
    if (s.signal_type === "vocal_stress_score" && s.value != null && s.value > 0.5) {
      hasHighStress = true;
    }
    if (
      (s.signal_type === "objection_signal" || s.signal_type === "objection_detected") &&
      (s.value_text?.toLowerCase() === "true" || (s.value != null && s.value > 0.5))
    ) {
      hasObjection = true;
    }
    if (s.agent === "fusion") {
      hasFusion = true;
    }
    if (
      (s.signal_type === "buying_signal" || s.signal_type === "buying_signal_detected") &&
      (s.value_text?.toLowerCase() === "true" || (s.value != null && s.value > 0.5))
    ) {
      hasBuyingSignal = true;
    }
    if (s.signal_type === "sentiment_score" && s.value != null && s.value > 0.8) {
      hasPositiveSentiment = true;
    }
  }

  // Priority: red > orange > green > default
  if (hasHighStress || hasObjection) return "border-l-nexus-stress-high";
  if (hasFusion) return "border-l-nexus-alert";
  if (hasBuyingSignal || hasPositiveSentiment) return "border-l-nexus-stress-low";
  return "border-l-transparent";
}

interface Props {
  segment: TranscriptSegment;
  signals: Signal[];
  speakerRole?: string;
}

export default function TranscriptBlock({ segment, signals, speakerRole }: Props) {
  const speaker = segment.speaker_label || "Speaker";
  const badges = filterSmartBadges(signals, 3);
  const borderColor = getBorderColor(signals);

  return (
    <div
      className={`group rounded-lg border border-nexus-border border-l-[3px] ${borderColor} bg-nexus-surface p-3 transition-colors hover:border-nexus-border/80`}
    >
      {/* Header: time + speaker + role */}
      <div className="mb-1.5 flex items-center gap-2 text-xs">
        <span className="font-mono text-nexus-text-muted">
          {formatTime(segment.start_ms)}
        </span>
        <span className="font-medium text-nexus-accent-blue">{speaker}</span>
        {speakerRole && (
          <span className="rounded bg-nexus-accent-purple/15 px-1.5 py-0.5 text-[10px] font-medium text-nexus-accent-purple">
            {speakerRole}
          </span>
        )}
      </div>

      {/* Transcript text */}
      <p className="text-sm leading-relaxed text-nexus-text-primary">
        {segment.text}
      </p>

      {/* Smart signal badges (max 3 noteworthy) */}
      {badges.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {badges.map((badge, i) => (
            <SignalBadge key={i} badge={badge} />
          ))}
        </div>
      )}
    </div>
  );
}
