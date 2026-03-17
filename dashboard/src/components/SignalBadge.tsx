import type { Signal } from "../api/client";

const AGENT_CONFIG: Record<string, { label: string; color: string; bg: string }> = {
  voice: { label: "VOICE", color: "text-nexus-agent-voice", bg: "bg-nexus-agent-voice/15" },
  language: { label: "LANG", color: "text-nexus-agent-language", bg: "bg-nexus-agent-language/15" },
  fusion: { label: "FUSION", color: "text-nexus-agent-fusion", bg: "bg-nexus-agent-fusion/15" },
  facial: { label: "FACE", color: "text-nexus-agent-facial", bg: "bg-nexus-agent-facial/15" },
  body: { label: "BODY", color: "text-nexus-agent-body", bg: "bg-nexus-agent-body/15" },
  gaze: { label: "GAZE", color: "text-nexus-agent-gaze", bg: "bg-nexus-agent-gaze/15" },
  conversation: { label: "CONVO", color: "text-nexus-agent-conversation", bg: "bg-nexus-agent-conversation/15" },
};

function formatSignal(signal: Signal): string {
  const { signal_type, value, value_text } = signal;

  const type = signal_type
    .replace(/_/g, " ")
    .replace(/\bscore\b/, "")
    .trim();

  if (value_text && value != null) {
    return `${type}: ${value_text} (${value.toFixed(2)})`;
  }
  if (value_text) return `${type}: ${value_text}`;
  if (value != null) return `${type}: ${value.toFixed(2)}`;
  return type;
}

export default function SignalBadge({ signal }: { signal: Signal }) {
  const config = AGENT_CONFIG[signal.agent] ?? {
    label: signal.agent.toUpperCase(),
    color: "text-nexus-text-secondary",
    bg: "bg-nexus-surface-hover",
  };

  return (
    <span
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-mono leading-tight ${config.bg} ${config.color}`}
      title={`Confidence: ${signal.confidence.toFixed(2)}`}
    >
      <span className="font-bold">{config.label}</span>
      <span className="opacity-80">{formatSignal(signal)}</span>
    </span>
  );
}
