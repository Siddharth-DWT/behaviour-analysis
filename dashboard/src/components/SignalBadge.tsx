import type { Signal } from "../api/client";

export interface SmartBadge {
  label: string;
  color: string;
  bg: string;
  priority: number;
}

const INTENT_LABELS: Record<string, string> = {
  PROPOSE: "Proposing",
  QUESTION: "Question",
  AGREE: "Agreeing",
  COMMIT: "Committing",
  OBJECTION: "Objecting",
  RAPPORT: "Rapport",
  CONVINCE: "Persuading",
  CHALLENGE: "Challenging",
};

const FUSION_LABELS: Record<string, string> = {
  urgency_authenticity: "Urgency Mismatch",
  credibility_assessment: "Credibility Issue",
  verbal_incongruence: "Verbal Mismatch",
  stress_language_mismatch: "Stress Mismatch",
  engagement_contradiction: "Engagement Conflict",
};

const SKIP_VALUES = new Set(["normal", "low_stress", "neutral", "none", "baseline"]);

/**
 * Convert a raw signal into a human-readable smart badge, or null if not noteworthy.
 */
export function toSmartBadge(signal: Signal): SmartBadge | null {
  const { signal_type, value, value_text, agent } = signal;

  // Skip signals with zero value or uninteresting status
  if (value === 0 || value === 0.0) return null;
  if (value_text && SKIP_VALUES.has(value_text.toLowerCase())) return null;

  // Voice: stress
  if (signal_type === "vocal_stress_score" && value != null && value > 0.3) {
    const isHigh = value > 0.6;
    return {
      label: `Stress ${(value * 100).toFixed(0)}%`,
      color: isHigh ? "text-nexus-stress-high" : "text-nexus-alert",
      bg: isHigh ? "bg-nexus-stress-high/15" : "bg-nexus-alert/15",
      priority: value > 0.6 ? 95 : 70,
    };
  }

  // Language: sentiment
  if (signal_type === "sentiment_score" && value != null) {
    if (value > 0.6) {
      return {
        label: "Positive",
        color: "text-nexus-stress-low",
        bg: "bg-nexus-stress-low/15",
        priority: 40,
      };
    }
    if (value < -0.3) {
      return {
        label: "Negative",
        color: "text-nexus-stress-high",
        bg: "bg-nexus-stress-high/15",
        priority: 60,
      };
    }
    return null;
  }

  // Language: intent
  if (signal_type === "intent_classification" && value_text) {
    const upper = value_text.toUpperCase();
    if (upper === "INFORM") return null; // Skip default intent
    const label = INTENT_LABELS[upper] || value_text;
    return {
      label,
      color: "text-nexus-accent-blue",
      bg: "bg-nexus-accent-blue/15",
      priority: 50,
    };
  }

  // Language: buying signal
  if (signal_type === "buying_signal" && value_text?.toLowerCase() === "true") {
    return {
      label: "Buying Signal",
      color: "text-nexus-stress-low",
      bg: "bg-nexus-stress-low/15",
      priority: 90,
    };
  }
  if (signal_type === "buying_signal_detected" || (signal_type === "buying_signal" && value != null && value > 0.5)) {
    return {
      label: "Buying Signal",
      color: "text-nexus-stress-low",
      bg: "bg-nexus-stress-low/15",
      priority: 90,
    };
  }

  // Language: objection
  if (signal_type === "objection_signal" && value_text?.toLowerCase() === "true") {
    return {
      label: "Objection",
      color: "text-nexus-stress-high",
      bg: "bg-nexus-stress-high/15",
      priority: 85,
    };
  }
  if (signal_type === "objection_detected" || (signal_type === "objection_signal" && value != null && value > 0.5)) {
    return {
      label: "Objection",
      color: "text-nexus-stress-high",
      bg: "bg-nexus-stress-high/15",
      priority: 85,
    };
  }

  // Voice: filler detection
  if (signal_type === "filler_detection" && value_text) {
    const lower = value_text.toLowerCase();
    if (lower === "elevated" || lower === "high" || lower === "excessive") {
      return {
        label: "Fillers",
        color: "text-nexus-stress-med",
        bg: "bg-nexus-stress-med/15",
        priority: 55,
      };
    }
    return null;
  }

  // Voice: pitch elevated
  if (signal_type === "pitch_analysis" && value_text) {
    const lower = value_text.toLowerCase();
    if (lower.includes("elevated") || lower.includes("high")) {
      return {
        label: "Pitch Elevated",
        color: "text-nexus-alert",
        bg: "bg-nexus-alert/15",
        priority: 50,
      };
    }
    return null;
  }

  // Voice: rate anomaly
  if (signal_type === "speech_rate_analysis" && value_text) {
    const lower = value_text.toLowerCase();
    if (lower.includes("fast") || lower.includes("slow") || lower.includes("rapid")) {
      return {
        label: lower.includes("fast") || lower.includes("rapid") ? "Fast Speech" : "Slow Speech",
        color: "text-nexus-stress-med",
        bg: "bg-nexus-stress-med/15",
        priority: 45,
      };
    }
    return null;
  }

  // Language: power language
  if (signal_type === "power_language_score" && value != null && value < 0.4) {
    return {
      label: "Low Power",
      color: "text-nexus-accent-purple",
      bg: "bg-nexus-accent-purple/15",
      priority: 45,
    };
  }

  // Voice: tone
  if (signal_type === "tone_analysis" && value_text) {
    const lower = value_text.toLowerCase();
    if (lower === "neutral" || lower === "calm" || lower === "normal") return null;
    return {
      label: `Tone: ${value_text}`,
      color: "text-nexus-accent-blue",
      bg: "bg-nexus-accent-blue/15",
      priority: 35,
    };
  }

  // Fusion signals
  if (agent === "fusion") {
    const fusionLabel = FUSION_LABELS[signal_type] || signal_type.replace(/_/g, " ");
    return {
      label: fusionLabel,
      color: "text-nexus-agent-fusion",
      bg: "bg-nexus-agent-fusion/15",
      priority: 80,
    };
  }

  // Default: skip everything else
  return null;
}

/**
 * Filter signals to at most `max` noteworthy smart badges, sorted by priority.
 */
export function filterSmartBadges(signals: Signal[], max = 3): SmartBadge[] {
  const badges: SmartBadge[] = [];
  for (const signal of signals) {
    const badge = toSmartBadge(signal);
    if (badge) badges.push(badge);
  }
  badges.sort((a, b) => b.priority - a.priority);
  return badges.slice(0, max);
}

export default function SignalBadge({ badge }: { badge: SmartBadge }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium leading-tight ${badge.bg} ${badge.color}`}
    >
      <span className="h-1 w-1 rounded-full bg-current opacity-60" />
      {badge.label}
    </span>
  );
}
