interface PathNode {
  id: string;
  type: string;
  label: string;
  agent: string | null;
  confidence: number | null;
  timestamp_ms: number;
}

interface KeyPath {
  nodes: PathNode[];
  description: string;
  score: number;
}

interface SignalChainCardsProps {
  keyPaths: KeyPath[];
}

const AGENT_COLORS: Record<string, string> = {
  voice: "var(--agent-voice, #6366F1)",
  language: "var(--agent-language, #06B6D4)",
  fusion: "var(--agent-fusion, #F97316)",
};

// Human-readable names for internal signal value_text codes
const READABLE_LABELS: Record<string, string> = {
  mild_verbal_incongruence: "Mild Verbal Mismatch",
  moderate_verbal_incongruence: "Moderate Verbal Mismatch",
  strong_verbal_incongruence: "Strong Verbal Mismatch",
  hedged_agreement: "Hedged Agreement",
  incongruence_with_objection: "Incongruence + Objection",
  credibility_concern: "Credibility Concern",
  mild_incongruence: "Mild Incongruence",
  manufactured_urgency: "Manufactured Urgency",
  authentic_urgency: "Authentic Urgency",
  ambiguous_urgency: "Ambiguous Urgency",
  filler_spike: "Filler Spike",
  filler_elevated: "Elevated Fillers",
};

function humanizeLabel(label: string): string {
  // Check if label contains a known code after ": "
  const colonIdx = label.indexOf(": ");
  if (colonIdx >= 0) {
    const code = label.slice(colonIdx + 2).trim();
    if (READABLE_LABELS[code]) {
      return label.slice(0, colonIdx + 2) + READABLE_LABELS[code];
    }
  }
  // Check whole label
  if (READABLE_LABELS[label]) return READABLE_LABELS[label];
  return label;
}

const TYPE_ICONS: Record<string, string> = {
  voice_signal: "🎙️",
  lang_signal: "💬",
  fusion_signal: "⚡",
  moment: "⭐",
  speaker: "👤",
  topic: "📋",
};

export default function SignalChainCards({ keyPaths }: SignalChainCardsProps) {
  if (!keyPaths.length) return null;

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
      <h3 className="mb-3 text-sm font-medium text-nexus-text-primary">
        Signal Chains
        <span className="ml-2 text-[10px] font-normal text-nexus-text-muted">
          Cross-modal insight paths
        </span>
      </h3>

      <div className="space-y-2">
        {keyPaths.map((path, i) => (
          <div
            key={i}
            className="rounded-lg border border-nexus-border bg-nexus-surface-hover p-3"
          >
            {/* Chain visualization */}
            <div className="flex items-center gap-1 flex-wrap">
              {path.nodes.map((node, j) => (
                <span key={j} className="flex items-center gap-1">
                  {j > 0 && (
                    <span className="text-nexus-text-muted text-xs mx-0.5">→</span>
                  )}
                  <span
                    className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium leading-tight"
                    style={{
                      backgroundColor: node.agent
                        ? `color-mix(in srgb, ${AGENT_COLORS[node.agent] || "#888"} 15%, transparent)`
                        : "var(--surface-hover)",
                      color: AGENT_COLORS[node.agent || ""] || "var(--text-secondary)",
                      border: `1px solid ${AGENT_COLORS[node.agent || ""] || "var(--border)"}`,
                      opacity: node.confidence != null
                        ? Math.max(0.6, Math.min(1, node.confidence + 0.3))
                        : 0.9,
                    }}
                    title={`${node.label} — confidence: ${node.confidence != null ? Math.round(node.confidence * 100) + "%" : "N/A"}`}
                  >
                    <span>{TYPE_ICONS[node.type] || "•"}</span>
                    <span className="break-words">{humanizeLabel(node.label)}</span>
                  </span>
                </span>
              ))}
            </div>

            {/* Score */}
            <div className="mt-1.5 flex items-center justify-between">
              <span className="text-[10px] text-nexus-text-muted">
                {path.nodes.length} signals across{" "}
                {new Set(path.nodes.map((n) => n.agent).filter(Boolean)).size} agents
              </span>
              <span className="text-[10px] font-mono text-nexus-text-secondary">
                relevance: {path.score.toFixed(1)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
