import type { Signal } from "../api/client";

export interface SmartBadge {
  label: string;
  color: string;
  bg: string;
  priority: number;
  /** Used for deduplication — only one badge per baseType per segment */
  baseType: string;
  /** The raw numeric value (for keeping the max when deduplicating) */
  rawValue: number;
  /** Signal confidence 0-1 — drives opacity and sort order */
  confidence: number;
}

const INTENT_LABELS: Record<string, string> = {
  PROPOSE: "Proposing",
  COMMIT: "Committing",
  OBJECTION: "Objecting",
  REQUEST: "Request",
  RAPPORT: "Rapport",
  CONVINCE: "Persuading",
  CHALLENGE: "Challenging",
  AGREE: "Agreeing",
};

// Intents that are too common / uninteresting to badge
const SKIP_INTENTS = new Set(["INFORM", "QUESTION"]);

const FUSION_LABELS: Record<string, string> = {
  urgency_authenticity: "⚠️ Urgency Mismatch",
  credibility_assessment: "⚠️ Credibility Issue",
  verbal_incongruence: "⚠️ Verbal Mismatch",
  stress_language_mismatch: "⚠️ Stress Mismatch",
  engagement_contradiction: "⚠️ Engagement Conflict",
};

const SKIP_VALUES = new Set([
  "normal", "low_stress", "neutral", "none", "baseline",
  "mild_positive", "mild_negative", "moderate_stress",
  "neutral_power", "moderate_power",
]);

/**
 * Convert a raw signal into a human-readable smart badge, or null if not noteworthy.
 */
export function toSmartBadge(signal: Signal): SmartBadge | null {
  const { signal_type, value, value_text, agent } = signal;
  const conf = (signal as any).confidence ?? 0.5;

  // Skip zero-value signals and known noise labels
  if (value === 0 || value === 0.0) return null;
  if (value_text && SKIP_VALUES.has(value_text.toLowerCase())) return null;

  // ── Voice: stress (only show > 0.30) ──
  if (signal_type === "vocal_stress_score" && value != null && value > 0.3) {
    const isHigh = value > 0.6;
    return {
      label: `Stress ${(value * 100).toFixed(0)}%`,
      color: isHigh ? "text-nexus-stress-high" : "text-nexus-alert",
      bg: isHigh ? "bg-stress-high-15" : "bg-alert-15",
      priority: value > 0.6 ? 95 : 70,
      baseType: "stress",
      rawValue: value,
      confidence: conf,
    };
  }

  // ── Language: sentiment (only show |score| > 0.35) ──
  if (signal_type === "sentiment_score" && value != null) {
    if (value > 0.35) {
      return {
        label: "Positive",
        color: "text-nexus-stress-low",
        bg: "bg-stress-low-15",
        priority: 40,
        baseType: "sentiment",
        rawValue: value,
        confidence: conf,
      };
    }
    if (value < -0.35) {
      return {
        label: "Negative",
        color: "text-nexus-stress-high",
        bg: "bg-stress-high-15",
        priority: 60,
        baseType: "sentiment",
        rawValue: Math.abs(value),
        confidence: conf,
      };
    }
    return null; // Neutral zone — no badge
  }

  // ── Language: intent (skip INFORM and QUESTION — too common) ──
  if (signal_type === "intent_classification" && value_text) {
    const upper = value_text.toUpperCase();
    if (SKIP_INTENTS.has(upper)) return null;
    const label = INTENT_LABELS[upper] || value_text;
    return {
      label,
      color: "text-nexus-accent-blue",
      bg: "bg-accent-blue-15",
      priority: 50,
      baseType: "intent",
      rawValue: value ?? 0,
      confidence: conf,
    };
  }

  // ── Language: buying signal (always show when detected) ──
  if (signal_type === "buying_signal" && value != null && value > 0.3) {
    return {
      label: "🟢 Buying Signal",
      color: "text-nexus-stress-low",
      bg: "bg-stress-low-15",
      priority: 90,
      baseType: "buying",
      rawValue: value,
      confidence: conf,
    };
  }

  // ── Language: objection (always show when detected) ──
  if (signal_type === "objection_signal" && value != null && value > 0.3) {
    return {
      label: "🔴 Objection",
      color: "text-nexus-stress-high",
      bg: "bg-stress-high-15",
      priority: 85,
      baseType: "objection",
      rawValue: value,
      confidence: conf,
    };
  }

  // ── Voice: filler detection (only elevated+) ──
  if (signal_type === "filler_detection" && value_text) {
    const lower = value_text.toLowerCase();
    if (lower === "elevated" || lower === "high" || lower === "excessive") {
      return {
        label: "Filler Spike",
        color: "text-nexus-stress-med",
        bg: "bg-stress-med-15",
        priority: 55,
        baseType: "filler",
        rawValue: value ?? 0,
        confidence: conf,
      };
    }
    return null;
  }

  // ── Voice: pitch elevated ──
  if (signal_type === "pitch_analysis" && value_text) {
    const lower = value_text.toLowerCase();
    if (lower.includes("elevated") || lower.includes("high")) {
      return {
        label: "Pitch ↑",
        color: "text-nexus-alert",
        bg: "bg-alert-15",
        priority: 50,
        baseType: "pitch",
        rawValue: value ?? 0,
        confidence: conf,
      };
    }
    return null;
  }

  // ── Voice: rate anomaly ──
  if (signal_type === "speech_rate_analysis" && value_text) {
    const lower = value_text.toLowerCase();
    if (lower.includes("fast") || lower.includes("slow") || lower.includes("rapid")) {
      return {
        label: lower.includes("fast") || lower.includes("rapid") ? "Fast" : "Slow",
        color: "text-nexus-accent-blue",
        bg: "bg-accent-blue-15",
        priority: 45,
        baseType: "rate",
        rawValue: value ?? 0,
        confidence: conf,
      };
    }
    return null;
  }

  // ── Language: power language (only extremes) ──
  if (signal_type === "power_language_score" && value != null) {
    if (value < 0.3) {
      return {
        label: "Weak Language",
        color: "text-nexus-stress-high",
        bg: "bg-stress-high-15",
        priority: 45,
        baseType: "power",
        rawValue: 1 - value,
        confidence: conf,
      };
    }
    if (value > 0.8) {
      return {
        label: "Strong Language",
        color: "text-nexus-stress-low",
        bg: "bg-stress-low-15",
        priority: 45,
        baseType: "power",
        rawValue: value,
        confidence: conf,
      };
    }
    return null; // Middle range — not noteworthy
  }

  // ── Voice: tone (skip neutral/calm/normal) ──
  if (signal_type === "tone_analysis" && value_text) {
    const lower = value_text.toLowerCase();
    if (lower === "neutral" || lower === "calm" || lower === "normal") return null;
    return {
      label: `Tone: ${value_text}`,
      color: "text-nexus-accent-blue",
      bg: "bg-accent-blue-15",
      priority: 35,
      baseType: "tone",
      rawValue: value ?? 0,
      confidence: conf,
    };
  }

  // ── Fusion signals (always show — highest value cross-modal insights) ──
  if (agent === "fusion") {
    const fusionLabel = FUSION_LABELS[signal_type] || `⚠️ ${signal_type.replace(/_/g, " ")}`;
    return {
      label: fusionLabel,
      color: "text-nexus-agent-fusion",
      bg: "bg-agent-fusion-15",
      priority: 80,
      baseType: `fusion_${signal_type}`,
      rawValue: value ?? 0,
      confidence: conf,
    };
  }

  // Default: skip everything else
  return null;
}

/**
 * Filter signals to at most `max` noteworthy smart badges.
 * Deduplicates by baseType — keeps only the highest-value badge per type.
 */
export function filterSmartBadges(signals: Signal[], max = 4): SmartBadge[] {
  const all: SmartBadge[] = [];
  for (const signal of signals) {
    const badge = toSmartBadge(signal);
    if (badge) all.push(badge);
  }

  // Deduplicate: keep only the highest rawValue per baseType
  const best = new Map<string, SmartBadge>();
  for (const badge of all) {
    const existing = best.get(badge.baseType);
    if (!existing || badge.rawValue > existing.rawValue) {
      best.set(badge.baseType, badge);
    }
  }

  const deduped = Array.from(best.values());
  // Sort by priority first, then by confidence descending within same priority
  deduped.sort((a, b) => b.priority - a.priority || b.confidence - a.confidence);
  return deduped.slice(0, max);
}

/** Map confidence to opacity: <0.4 → 70%, 0.4-0.6 → 85%, >0.6 → 100% */
function confidenceOpacity(conf: number): string {
  if (conf < 0.4) return "opacity-70";
  if (conf <= 0.6) return "opacity-85";
  return "";
}

export default function SignalBadge({ badge }: { badge: SmartBadge }) {
  const opClass = confidenceOpacity(badge.confidence);
  const tooltip = `Confidence: ${(badge.confidence * 100).toFixed(0)}%`;
  return (
    <span
      title={tooltip}
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium leading-tight ${badge.bg} ${badge.color} ${opClass}`}
    >
      <span className="h-1 w-1 rounded-full bg-current opacity-60" />
      {badge.label}
    </span>
  );
}
