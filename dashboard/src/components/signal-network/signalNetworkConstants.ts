import type { VisualNodeType } from "./types";

// ── Node colors ──

export const NODE_COLORS: Record<VisualNodeType, string> = {
  speaker:       "#4F8BFF",
  topic:         "#8B5CF6",
  voice_signal:  "#6366F1",
  lang_signal:   "#06B6D4",
  fusion_signal: "#F97316",
};

export const SPEAKER_COLORS = ["#4F8BFF", "#F59E0B", "#8B5CF6", "#10B981", "#EC4899", "#6366F1"];

// ── Edge styles ──

export interface EdgeStyle {
  color: string;
  dashed: boolean;
  width: number;
  opacity: number;
  animated?: boolean;
  description: string;
}

export const EDGE_STYLES: Record<string, EdgeStyle> = {
  contradicts:      { color: "#EF4444", dashed: true,  width: 2.5, opacity: 0.85, animated: true, description: "Conflicting signals from different modalities" },
  triggered:        { color: "#F97316", dashed: false, width: 2.0, opacity: 0.8,  description: "Cross-modal signal that triggered a fusion insight" },
  resolves:         { color: "#22C55E", dashed: false, width: 2.0, opacity: 0.8,  description: "Objection or tension resolved later in conversation" },
  preceded:         { color: "#6366F1", dashed: false, width: 1.2, opacity: 0.4,  description: "Signal appeared before another in time" },
  co_occurred:      { color: "#6B7280", dashed: true,  width: 0.8, opacity: 0.2,  description: "Signals co-occurring within the same time window" },
  about_topic:      { color: "#10B981", dashed: false, width: 1.0, opacity: 0.35, description: "Signal related to a specific conversation topic" },
  speaker_produced: { color: "#9CA3AF", dashed: false, width: 0.5, opacity: 0.15, description: "Signal produced by this speaker" },
};

export const EDGE_TYPE_LABELS: Record<string, string> = {
  contradicts:      "Contradicts",
  triggered:        "Triggered",
  resolves:         "Resolves",
  preceded:         "Preceded",
  co_occurred:      "Co-occurred",
  about_topic:      "About Topic",
  speaker_produced: "Speaker Produced",
};

// ── Node labels ──

export const NODE_TYPE_LABELS: Record<VisualNodeType, string> = {
  speaker:       "Speaker",
  topic:         "Topic",
  voice_signal:  "Voice",
  lang_signal:   "Language",
  fusion_signal: "Fusion",
};

// Signal type → human-readable
export const SIGNAL_LABELS: Record<string, string> = {
  vocal_stress_score:    "Stress",
  buying_signal:         "Buying Signal",
  objection_signal:      "Objection",
  sentiment_score:       "Sentiment",
  credibility_assessment: "Credibility",
  verbal_incongruence:   "Incongruence",
  urgency_authenticity:  "Urgency",
  pitch_elevation_flag:  "Pitch Elevation",
  speech_rate_anomaly:   "Speech Rate",
  filler_detection:      "Filler",
  tone_classification:   "Tone",
  power_language_score:  "Power Language",
  persuasion_indicator:  "Persuasion",
  intent_classification: "Intent",
};

// ── Default filter state ──

export const ALL_AGENTS = new Set(["voice", "language", "fusion"]);
export const ALL_NODE_TYPES = new Set<string>(["speaker", "topic", "voice_signal", "lang_signal", "fusion_signal"]);

// The noisier edge types are off by default
export const DEFAULT_EDGE_TYPES = new Set(["contradicts", "triggered", "resolves", "about_topic", "preceded"]);

export const DEFAULT_CONFIDENCE_MIN = 0.15;

// ── Layout config ──

export const LAYOUT_MODES = [
  { value: "force",      label: "Force-directed" },
  { value: "by_speaker", label: "Cluster by Speaker" },
  { value: "by_topic",   label: "Cluster by Topic" },
  { value: "by_agent",   label: "Cluster by Agent" },
] as const;
