import type { TranscriptSegment, Signal } from "../api/client";
import SignalBadge from "./SignalBadge";

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

interface Props {
  segment: TranscriptSegment;
  signals: Signal[];
}

export default function TranscriptBlock({ segment, signals }: Props) {
  const speaker = segment.speaker_label || "Speaker";

  return (
    <div className="group rounded-lg border border-nexus-border bg-nexus-surface p-3 transition-colors hover:border-nexus-border/80">
      {/* Header: time + speaker */}
      <div className="mb-1.5 flex items-center gap-2 text-xs">
        <span className="font-mono text-nexus-text-muted">
          {formatTime(segment.start_ms)}
        </span>
        <span className="font-medium text-nexus-accent-blue">{speaker}</span>
      </div>

      {/* Transcript text */}
      <p className="text-sm leading-relaxed text-nexus-text-primary">
        {segment.text}
      </p>

      {/* Inline signal badges */}
      {signals.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {signals.map((signal, i) => (
            <SignalBadge key={`${signal.id}-${i}`} signal={signal} />
          ))}
        </div>
      )}
    </div>
  );
}
