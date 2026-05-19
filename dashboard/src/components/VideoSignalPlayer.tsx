import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { getAccessToken, getVideoSpeakers } from "../api/client";
import type { VideoSignal, SpeakerInfo } from "../api/client";
import { getSignalDisplay } from "../config/signalDisplayConfig";
import InterrogationSummaryPanel from "./InterrogationSummaryPanel";

// ── Signal display configuration ──────────────────────────────────────────────

type SignalConfigEntry = {
  icon: string;
  label: (s: VideoSignal) => string;
  color: string | ((s: VideoSignal) => string);
  category: "face" | "body" | "gaze" | "voice" | "compound";
  hidden?: boolean;
};

const SIGNAL_CONFIG: Record<string, SignalConfigEntry> = {
  presence_detected: {
    icon: "",
    label: () => "",
    color: "transparent",
    category: "face",
    hidden: true,
  },
  facial_stress: {
    icon: "●",
    label: (s) =>
      `Tension: ${s.value_text === "high_facial_stress" ? "High" : "Moderate"} (${Math.round(s.value * 100)}%)`,
    color: "#EF4444",
    category: "face",
  },
  facial_emotion: {
    icon: "◉",
    label: (s) => `Expression: ${s.value_text}`,
    color: "#8B5CF6",
    category: "face",
  },
  smile_type: {
    icon: "◗",
    label: (s) => `Smile: ${s.value_text === "duchenne" ? "Genuine" : "Social"}`,
    color: (s) => (s.value_text === "duchenne" ? "#10B981" : "#F59E0B"),
    category: "face",
  },
  facial_engagement: {
    icon: "⚡",
    label: (s) =>
      `Engagement: ${s.value_text === "high_engagement" ? "High" : "Low"}`,
    color: (s) => (s.value_text === "high_engagement" ? "#10B981" : "#6B7280"),
    category: "face",
  },
  valence_arousal: {
    icon: "◈",
    label: (s) => `State: ${s.value_text.replace(/_/g, " ")}`,
    color: "#8B5CF6",
    category: "face",
  },
  head_nod: {
    icon: "↓",
    label: () => "Nod",
    color: "#10B981",
    category: "body",
  },
  head_shake: {
    icon: "↔",
    label: () => "Shake",
    color: "#F59E0B",
    category: "body",
  },
  body_lean: {
    icon: "↗",
    label: (s) =>
      s.value_text === "forward_lean" ? "Forward Lean" : "Backward Lean",
    color: (s) => (s.value_text === "forward_lean" ? "#10B981" : "#EF4444"),
    category: "body",
  },
  posture: {
    icon: "▲",
    label: (s) =>
      s.value_text === "upright_power_posture" ? "Upright" : "Slumped",
    color: (s) =>
      s.value_text === "upright_power_posture" ? "#10B981" : "#F59E0B",
    category: "body",
  },
  self_touch: {
    icon: "✕",
    label: () => "Self-touch",
    color: "#F59E0B",
    category: "body",
  },
  body_fidgeting: {
    icon: "~",
    label: (s) => `Fidgeting: ${s.value_text.replace(/_/g, " ")}`,
    color: "#F59E0B",
    category: "body",
  },
  shoulder_tension: {
    icon: "△",
    label: () => "Shoulder Tension",
    color: "#F59E0B",
    category: "body",
  },
  head_body_incongruence: {
    icon: "!",
    label: (s) =>
      s.value_text === "nod_but_withdrawing"
        ? "Nod + Withdrawing"
        : "Shake + Engaged",
    color: "#EF4444",
    category: "body",
  },
  gaze_direction_shift: {
    icon: "→",
    label: (s) =>
      `Gaze: ${s.value_text.replace("gaze_shift_", "").replace("gaze_", "")}`,
    color: "#6B7280",
    category: "gaze",
  },
  screen_contact: {
    icon: "◎",
    label: (s) =>
      s.value_text === "sustained_eye_contact"
        ? "Sustained Eye Contact"
        : "Low Screen Contact",
    color: (s) =>
      s.value_text === "sustained_eye_contact" ? "#10B981" : "#EF4444",
    category: "gaze",
  },
  sustained_distraction: {
    icon: "○",
    label: () => "Distracted (>8s)",
    color: "#EF4444",
    category: "gaze",
  },
  attention_level: {
    icon: "◎",
    label: (s) =>
      `Attention: ${s.value_text === "high_attention" ? "High" : "Low"}`,
    color: (s) =>
      s.value_text === "high_attention" ? "#10B981" : "#F59E0B",
    category: "gaze",
  },
  blink_rate_anomaly: {
    icon: "◦",
    label: (s) =>
      s.value_text === "elevated_blink_rate" ? "Rapid Blinking" : "Slow Blinks",
    color: "#F59E0B",
    category: "gaze",
  },
  genuine_engagement: {
    icon: "●",
    label: () => "Genuine Engagement",
    color: "#10B981",
    category: "compound",
  },
  active_disengagement: {
    icon: "●",
    label: () => "Disengaged",
    color: "#EF4444",
    category: "compound",
  },
  peak_performance: {
    icon: "★",
    label: () => "Peak Performance",
    color: "#F59E0B",
    category: "compound",
  },
  cognitive_overload: {
    icon: "▲",
    label: () => "Cognitive Overload",
    color: "#EF4444",
    category: "compound",
  },
  tone_face_masking: {
    icon: "◑",
    label: () => "Masking Detected",
    color: "#EF4444",
    category: "compound",
  },
  stress_suppression: {
    icon: "◑",
    label: () => "Stress Suppression",
    color: "#EF4444",
    category: "compound",
  },
  conflict_escalation: {
    icon: "▲",
    label: () => "Conflict Escalating",
    color: "#EF4444",
    category: "compound",
  },
  emotional_suppression: {
    icon: "◑",
    label: () => "Emotional Suppression",
    color: "#EF4444",
    category: "compound",
  },
  decision_engagement: {
    icon: "★",
    label: () => "Decision Ready",
    color: "#10B981",
    category: "compound",
  },
  verbal_nonverbal_discordance: {
    icon: "≠",
    label: () => "Words/Body Mismatch",
    color: "#EF4444",
    category: "compound",
  },
  voice_face_alignment: {
    icon: "≈",
    label: (s) => {
      const labels: Record<string, string> = {
        congruent:                    "Authentic",
        voice_positive_face_negative: "Forced Positivity",
        voice_negative_face_positive: "Polite Masking",
        voice_stressed_face_calm:     "Hidden Stress",
        voice_calm_face_stressed:     "Face Leaking",
        energy_mismatch:              "Energy Mismatch",
      };
      return labels[s.value_text] || "Voice-Face Sync";
    },
    color: (s) => {
      if (s.value_text === "congruent") return "#10B981";
      if (s.value_text === "energy_mismatch") return "#F59E0B";
      return "#EF4444";
    },
    category: "compound",
  },
  rapport_building: {
    icon: "♥",
    label: () => "Strong Rapport",
    color: "#10B981",
    category: "compound",
  },
  rapport_confirmation: {
    icon: "◉",
    label: () => "Rapport Confirmed",
    color: "#10B981",
    category: "compound",
  },
  dominance_display: {
    icon: "▲",
    label: () => "Dominance Display",
    color: "#F59E0B",
    category: "compound",
  },
  submission_signal: {
    icon: "▽",
    label: () => "Submission Signal",
    color: "#6B7280",
    category: "compound",
  },
  deception_cluster: {
    icon: "⚠",
    label: () => "Review Required",
    color: "#EF4444",
    category: "compound",
  },
  // ── Temporal patterns (T-01 through T-08) ───────────────────────────────────
  stress_trajectory: {
    icon: "↑",
    label: (s) => `Stress ${s.value_text === "stress_trajectory_rising" ? "Rising" : "Falling"}`,
    color: (s) => (s.value_text === "stress_trajectory_rising" ? "#EF4444" : "#10B981"),
    category: "compound",
  },
  engagement_decay: {
    icon: "↓",
    label: () => "Engagement Declining",
    color: "#F59E0B",
    category: "compound",
  },
  rapport_evolution: {
    icon: "↗",
    label: (s) => `Rapport ${s.value_text?.includes("building") ? "Building" : "Declining"}`,
    color: (s) => (s.value_text?.includes("building") ? "#10B981" : "#EF4444"),
    category: "compound",
  },
  behavioral_shift: {
    icon: "⇔",
    label: () => "Behavioral Shift",
    color: "#8B5CF6",
    category: "compound",
  },
  adaptation_pattern: {
    icon: "↗",
    label: () => "Positive Adaptation",
    color: "#10B981",
    category: "compound",
  },
  fatigue_detection: {
    icon: "↓",
    label: () => "Energy Declining",
    color: "#F59E0B",
    category: "compound",
  },
  stress_recovery: {
    icon: "↓",
    label: () => "Stress Recovery",
    color: "#10B981",
    category: "compound",
  },
  escalation_ladder: {
    icon: "▲",
    label: (s) => `Escalation: ${(s.value_text || "").replace(/_/g, " ")}`,
    color: "#EF4444",
    category: "compound",
  },
  // ── Graph-based patterns ─────────────────────────────────────────────────────
  tension_cluster: {
    icon: "⊗",
    label: (s) => `Tension: ${s.value_text === "high_tension" ? "High" : "Moderate"}`,
    color: (s) => (s.value_text === "high_tension" ? "#EF4444" : "#F59E0B"),
    category: "compound",
  },
  // ── New body signals (Phase 2E) ───────────────────────────────────────────────
  face_region_touch: {
    icon: "✋",
    label: (s) => {
      const zoneLabels: Record<string, string> = {
        chin_touch_evaluation:     "Chin Touch: Evaluating",
        mouth_cover_suppression:   "Mouth Cover: Holding Back",
        nose_touch_discomfort:     "Nose Touch: Discomfort",
        cheek_touch_listening:     "Cheek Touch: Listening",
        cheek_rest_fatigue:        "Head Resting: Fatigue",
        ear_touch_soothing:        "Ear Touch: Self-Soothing",
        neck_touch_vulnerability:  "Neck Touch: Feeling Exposed",
        forehead_touch_frustration:"Forehead: Frustration",
      };
      return zoneLabels[s.value_text] || `Touch: ${s.value_text}`;
    },
    color: "#F59E0B",
    category: "body",
  },
  arms_crossed: {
    icon: "⊠",
    label: () => "Arms Crossed",
    color: "#F59E0B",
    category: "body",
  },
  finger_steepling: {
    icon: "△",
    label: () => "Steepling: Confidence",
    color: "#10B981",
    category: "body",
  },
  head_supported: {
    icon: "🤲",
    label: (s) =>
      s.value_text === "head_resting_disengagement"
        ? "Head Resting: Disengaged"
        : "Head Resting: Thinking",
    color: (s) =>
      s.value_text === "head_resting_disengagement" ? "#EF4444" : "#F59E0B",
    category: "body",
  },
  hands_clasped: {
    icon: "🤝",
    label: (s) =>
      s.value_text === "hands_clasped_waiting"
        ? "Hands Clasped: Waiting"
        : "Hands Clasped: Restraint",
    color: "#6B7280",
    category: "body",
  },
  cross_speaker_interaction: {
    icon: "🔗",
    label: (s) => {
      const labels: Record<string, string> = {
        agreement_reaction:    "Agreeing with speaker",
        disagreement_reaction: "Disagreeing with speaker",
        discomfort_reaction:   "Uncomfortable",
        incongruent_reaction:  "Mixed signals",
        disengagement_reaction: "Checked out",
      };
      return labels[s.value_text ?? ""] ?? s.value_text ?? "Interaction";
    },
    color: (s) => {
      if (s.value_text === "agreement_reaction")    return "#10B981";
      if (s.value_text === "disagreement_reaction") return "#EF4444";
      if (s.value_text === "incongruent_reaction")  return "#F59E0B";
      return "#6B7280";
    },
    category: "body",
  },
  lip_pursing: {
    icon: "◑",
    label: () => "Lip Pursing: Holding Back",
    color: "#F59E0B",
    category: "face",
  },
  laughter: {
    icon: "😂",
    label: (s) => s.value_text === "genuine_laughter" ? "Genuine Laughter" : "Big Smile",
    color: (s) => s.value_text === "genuine_laughter" ? "#10B981" : "#F59E0B",
    category: "face",
  },
  posture_transition: {
    icon: "⇄",
    label: (s) => {
      const labels: Record<string, string> = {
        closing_up:       "Closing Up",
        opening_up:       "Opening Up",
        disengaging:      "Disengaging",
        re_engaging:      "Re-Engaging",
        defensive_shift:  "Defensive Shift",
        losing_interest:  "Losing Interest",
      };
      return labels[s.value_text] || s.value_text;
    },
    color: (s) =>
      ["opening_up", "re_engaging"].includes(s.value_text)
        ? "#10B981"
        : ["closing_up", "defensive_shift", "losing_interest"].includes(s.value_text)
        ? "#EF4444"
        : "#F59E0B",
    category: "body",
  },
  body_language_cluster: {
    icon: "◉",
    label: (s) => {
      const labels: Record<string, string> = {
        skepticism_cluster:           "Skepticism Detected",
        stress_anxiety_cluster:       "Stress Indicators",
        confidence_authority_cluster: "Confidence Display",
        disengagement_boredom_cluster:"Disengagement Pattern",
      };
      return labels[s.value_text] || s.value_text;
    },
    color: (s) =>
      s.value_text === "confidence_authority_cluster"
        ? "#10B981"
        : s.value_text === "skepticism_cluster"
        ? "#F59E0B"
        : "#EF4444",
    category: "compound",
  },
  hand_gesture: {
    icon: "✋",
    label: (s) => {
      const labels: Record<string, string> = {
        approval:    "Thumbs Up",
        disapproval: "Thumbs Down",
        emphasis:    "Emphasizing",
        victory:     "Victory Sign",
        tension:     "Clenched Fist",
      };
      return labels[s.value_text] || `Gesture: ${s.value_text}`;
    },
    color: (s) => {
      if (s.value_text === "approval" || s.value_text === "victory") return "#10B981";
      if (s.value_text === "disapproval" || s.value_text === "tension") return "#EF4444";
      return "#F59E0B";
    },
    category: "body",
  },
  evaluation_cluster: {
    icon: "🤔",
    label: () => "Evaluating",
    color: "#F59E0B",
    category: "body",
  },
  hidden_disagreement: {
    icon: "◑",
    label: () => "Suppressed Disagreement",
    color: "#EF4444",
    category: "body",
  },
  frustration_cluster: {
    icon: "▲",
    label: () => "Frustration",
    color: "#EF4444",
    category: "body",
  },
  arm_posture: {
    icon: "↔",
    label: (s) => s.value_text === "expansive" ? "Open Posture" : "Closed Posture",
    color: (s) => s.value_text === "expansive" ? "#10B981" : "#F59E0B",
    category: "body",
  },
  gesture_animation: {
    icon: "✋",
    label: (s) =>
      s.value_text === "very_animated_gestures" ? "Very Animated" : "Gesturing",
    color: "#10B981",
    category: "body",
  },
  body_mirroring: {
    icon: "⇔",
    label: () => "Mirroring",
    color: "#10B981",
    category: "body",
  },
  gaze_synchrony: {
    icon: "👀",
    label: () => "Mutual Look-Away",
    color: "#F59E0B",
    category: "gaze",
  },

  // ── Voice signals (shown for Speaker_* who have diarization) ─────────────────
  vocal_stress_score: {
    icon: "◆",
    label: (s) => {
      if (s.value > 0.70) return "Voice Stress: High";
      if (s.value > 0.50) return "Voice Stress: Elevated";
      return "Voice Stress: Moderate";
    },
    color: (s) => (s.value > 0.70 ? "#EF4444" : s.value > 0.50 ? "#F59E0B" : "#6B7280"),
    category: "voice" as const,
  },
  tone_classification: {
    icon: "♪",
    label: (s) => {
      const labels: Record<string, string> = {
        warm: "Warm Tone", cold: "Cold Tone",
        aggressive: "Aggressive Tone", excited: "Excited Tone",
        nervous: "Nervous Tone", confident: "Confident Tone",
        neutral: "Neutral Tone",
      };
      return labels[s.value_text ?? ""] || `Tone: ${s.value_text ?? ""}`;
    },
    color: (s) => {
      if (["warm", "confident", "excited"].includes(s.value_text ?? "")) return "#10B981";
      if (["cold", "aggressive"].includes(s.value_text ?? "")) return "#EF4444";
      if (s.value_text === "nervous") return "#F59E0B";
      return "#6B7280";
    },
    category: "voice" as const,
  },
  filler_detection: {
    icon: "…",
    label: (s) => `Fillers: ${(s.value_text ?? "detected").replace(/_/g, " ")}`,
    color: "#F59E0B",
    category: "voice" as const,
  },
  speech_rate_anomaly: {
    icon: "⏩",
    label: (s) => {
      if ((s.value_text ?? "").includes("fast")) return "Speaking Fast";
      if ((s.value_text ?? "").includes("slow")) return "Speaking Slow";
      return `Pace: ${(s.value_text ?? "anomaly").replace(/_/g, " ")}`;
    },
    color: (s) => ((s.value_text ?? "").includes("fast") ? "#F59E0B" : "#6B7280"),
    category: "voice" as const,
  },
  energy_level: {
    icon: "⚡",
    label: (s) => {
      if (s.value_text === "elevated") return "High Energy";
      if (s.value_text === "depressed") return "Low Energy";
      return `Energy: ${s.value_text ?? "normal"}`;
    },
    color: (s) =>
      s.value_text === "elevated" ? "#10B981"
      : s.value_text === "depressed" ? "#6B7280"
      : "#F59E0B",
    category: "voice" as const,
  },
  pitch_elevation_flag: {
    icon: "↑",
    label: (s) => {
      if (s.value_text === "extreme_pitch_spike") return "Pitch Spike: Extreme";
      if (s.value_text === "significant_elevation") return "Pitch: Elevated";
      return "Pitch Shift";
    },
    color: "#F59E0B",
    category: "voice" as const,
  },
  monotone_flag: {
    icon: "─",
    label: () => "Monotone Voice",
    color: "#6B7280",
    category: "voice" as const,
  },
  interruption_event: {
    icon: "⚡",
    label: (s) => (s.value_text === "competitive" ? "Interruption" : "Overlap"),
    color: (s) => (s.value_text === "competitive" ? "#EF4444" : "#F59E0B"),
    category: "voice" as const,
  },

  // ── Interrogation video signals ───────────────────────────────────────────────
  blink_suppression_spike: {
    icon: "◦",
    label: () => "Blink: Suppression → Spike",
    color: "#EF4444",
    category: "gaze",
  },
  motor_inhibition: {
    icon: "⊖",
    label: () => "Motor Inhibition",
    color: "#6B7280",
    category: "body",
  },
  smile_context_incongruence: {
    icon: "◑",
    label: () => "Smile: Wrong Context",
    color: "#EF4444",
    category: "face",
  },
  erratic_gaze_pattern: {
    icon: "⇝",
    label: () => "Erratic Gaze",
    color: "#F59E0B",
    category: "gaze",
  },
  freezing_response: {
    icon: "❄",
    label: (s) => `Freeze: ${s.value_text?.replace(/_/g, " ") ?? "detected"}`,
    color: "#8B5CF6",
    category: "body",
  },
  barrier_behavior: {
    icon: "⊠",
    label: () => "Barrier Posture",
    color: "#F59E0B",
    category: "body",
  },

  // ── Interrogation voice / conversation signals ───────────────────────────────
  agitated_high_arousal_tone: {
    icon: "⚡",
    label: () => "Agitated High Arousal",
    color: "#EF4444",
    category: "compound",
  },
  evidence_response_processing_delay: {
    icon: "⏱",
    label: (s) => `Evidence Response Delay: ${((s.value ?? 0) * 10).toFixed(1)}s`,
    color: "#F59E0B",
    category: "compound",
  },
  low_autonomic_reactivity: {
    icon: "◉",
    label: () => "Low Autonomic Reactivity",
    color: "#6B7280",
    category: "body",
  },

  // ── Interrogation language signals (shown in Patterns row) ───────────────────
  pronoun_distancing: {
    icon: "↔",
    label: () => "Pronoun Distancing",
    color: "#EF4444",
    category: "compound",
  },
  tense_inconsistency: {
    icon: "⚠",
    label: () => "Tense Inconsistency",
    color: "#F59E0B",
    category: "compound",
  },
  statement_contamination: {
    icon: "⊗",
    label: () => "Statement Contamination",
    color: "#EF4444",
    category: "compound",
  },
  denial_weakening: {
    icon: "↓",
    label: () => "Denial Weakening",
    color: "#EF4444",
    category: "compound",
    hidden: true,
  },
  capitulation_cascade: {
    icon: "📉",
    label: () => "Capitulation Pattern",
    color: "#EF4444",
    category: "compound",
  },
  resistance_hardening: {
    icon: "📈",
    label: () => "Resistance Pattern",
    color: "#3B82F6",
    category: "compound",
  },
  false_confession_risk: {
    icon: "⚖",
    label: () => "False Confession Risk",
    color: "#EF4444",
    category: "compound",
    hidden: true,
  },
  interrogator_technique: {
    icon: "◈",
    label: (s) => `Technique: ${s.value_text ?? "unknown"}`,
    color: "#6B7280",
    category: "compound",
    hidden: true,
  },
};

const CATEGORIES: { key: string; label: string }[] = [
  { key: "face", label: "Face" },
  { key: "body", label: "Body" },
  { key: "gaze", label: "Gaze" },
  { key: "compound", label: "Patterns" },
];


function resolveColor(
  config: SignalConfigEntry,
  signal: VideoSignal
): string {
  return typeof config.color === "function" ? config.color(signal) : config.color;
}

const GENERIC_LABEL = /^(Face|Speaker|Person)_\d+$/;

function resolveDisplayLabel(
  rosterEntry: SpeakerInfo | undefined,
  spkId: string
): string {
  const name = rosterEntry?.display_name;
  if (name && !GENERIC_LABEL.test(name)) return name;
  return spkId;
}

// ── Sub-components ────────────────────────────────────────────────────────────

interface SpeakerGroupHeaderProps {
  speakerLabel: string;
  displayLabel: string;
  roster: Record<string, SpeakerInfo>;
  highlighted: boolean;
  onToggle: () => void;
}

function SpeakerGroupHeader({
  speakerLabel,
  displayLabel,
  roster,
  highlighted,
  onToggle,
}: SpeakerGroupHeaderProps) {
  const [imgFailed, setImgFailed] = useState(false);
  const token = getAccessToken();
  const info = roster[speakerLabel];
  const thumbPath = !imgFailed && info?.thumbnail_url
    ? `/api${info.thumbnail_url}${token ? `?token=${encodeURIComponent(token)}` : ""}`
    : null;
  const initials = displayLabel
    .split(" ")
    .map((w) => w[0] ?? "")
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <button
      onClick={onToggle}
      className={`flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left transition-colors ${
        highlighted ? "bg-yellow-500/20" : "hover:bg-white/5"
      }`}
      title={highlighted ? `Unhighlight ${displayLabel}` : `Highlight ${displayLabel}`}
    >
      {thumbPath ? (
        <img
          src={thumbPath}
          alt={displayLabel}
          className="h-7 w-7 flex-shrink-0 rounded-full object-cover border border-gray-600"
          onError={() => setImgFailed(true)}
        />
      ) : (
        <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-gray-700">
          <span className="text-[9px] font-bold text-gray-400">{initials}</span>
        </div>
      )}
      <div className="min-w-0 flex-1">
        <span className="block truncate text-[10px] font-medium text-gray-200">{displayLabel}</span>

        {info?.match_method && info.match_method !== "new_registration" && info.match_method !== "" && (
          <span className="text-[9px] text-gray-500">
            {info.match_method === "face_voice_fused"
              ? "biometric"
              : info.match_method === "face_embedding"
              ? "face"
              : info.match_method === "voice_embedding"
              ? "voice"
              : info.match_method}{" "}
            match ({Math.round((info.match_confidence ?? 0) * 100)}%)
          </span>
        )}
      </div>
      {highlighted && <span className="ml-auto shrink-0 text-[8px] text-yellow-400">●</span>}
    </button>
  );
}

interface FaceHighlightProps {
  speaker: string;
  signals: VideoSignal[];
}

function FaceHighlight({ speaker, signals }: FaceHighlightProps) {
  // Use the most recently started active signal for face position — avoids
  // showing stale coordinates from older windows when newer detections exist.
  const sig = signals
    .filter((s) => s.speaker_id === speaker && s.metadata?.face_centre_x != null)
    .reduce<VideoSignal | null>(
      (best, s) => (!best || Number(s.start_ms) > Number(best.start_ms) ? s : best),
      null
    );
  if (!sig) return null;

  const cx = sig.metadata!.face_centre_x as number;
  const cy = sig.metadata!.face_centre_y as number;

  return (
    <div
      className="pointer-events-none absolute rounded-lg border-2 border-yellow-400"
      style={{
        left: `${(cx - 0.08) * 100}%`,
        top: `${(cy - 0.12) * 100}%`,
        width: "16%",
        height: "24%",
        transition: "opacity 0.3s",
        zIndex: 5,
      }}
    />
  );
}

// ── Component ──────────────────────────────────────────────────────────────────

interface Props {
  sessionId: string;
  signals: VideoSignal[];
}

export default function VideoSignalPlayer({ sessionId, signals }: Props) {
  const token = getAccessToken();
  const rawVideoUrl      = `/api/sessions/${sessionId}/video${token ? `?token=${encodeURIComponent(token)}` : ""}`;
  const annotatedVideoUrl = `/api/sessions/${sessionId}/video/annotated${token ? `?token=${encodeURIComponent(token)}` : ""}`;

  const videoRef = useRef<HTMLVideoElement>(null);
  const animFrameRef = useRef<number | null>(null);

  const [currentTimeMs, setCurrentTimeMs] = useState<number>(-1);
  const [hasPlayed, setHasPlayed] = useState(false);
  const [durationMs, setDurationMs] = useState(0);
  const [activeSignals, setActiveSignals] = useState<VideoSignal[]>([]);
  const [enabledCategories, setEnabledCategories] = useState<Set<string>>(
    new Set(["face", "body", "gaze", "compound"])
  );
  const [selectedSpeaker, setSelectedSpeaker] = useState("all");
  const [showAnnotated, setShowAnnotated] = useState(false);
  const [annotatedAvailable, setAnnotatedAvailable] = useState(false);
  const [showExpanded, setShowExpanded] = useState(false);
  const [speakerRoster, setSpeakerRoster] = useState<Record<string, SpeakerInfo>>({});
  const [highlightedSpeaker, setHighlightedSpeaker] = useState<string | null>(null);

  // Fetch speaker roster once on mount to resolve display names and thumbnails.
  useEffect(() => {
    getVideoSpeakers(sessionId)
      .then((data) => {
        const roster: Record<string, SpeakerInfo> = {};
        for (const spk of data.speakers) {
          roster[spk.speaker_label] = spk;
        }
        setSpeakerRoster(roster);
      })
      .catch(() => {});
  }, [sessionId]);

  // Poll until the annotated video is ready.
  // Landmark burn (full MediaPipe re-pass) takes several minutes for long videos.
  // Schedule: first 3 checks every 10s, then every 30s up to 20 min total.
  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    let attempts = 0;
    const MAX_ATTEMPTS = 40; // 3×10s + 37×30s ≈ 18.5 min
    const check = () => {
      if (cancelled || attempts >= MAX_ATTEMPTS) return;
      const delay = attempts < 3 ? 10_000 : 30_000;
      attempts++;
      fetch(annotatedVideoUrl, { method: "HEAD" })
        .then((r) => {
          if (cancelled) return;
          if (r.ok) {
            setAnnotatedAvailable(true);
          } else if (attempts < MAX_ATTEMPTS) {
            setTimeout(check, delay);
          }
        })
        .catch(() => { if (!cancelled && attempts < MAX_ATTEMPTS) setTimeout(check, delay); });
    };
    check();
    return () => { cancelled = true; };
  }, [annotatedVideoUrl, token]);

  const videoUrl = showAnnotated && annotatedAvailable ? annotatedVideoUrl : rawVideoUrl;

  // Holds the non-canonical→canonical face-track mapping so computeActive can
  // include signals from all fragmented tracks when a canonical speaker is selected.
  const toCanonicalRef = useRef<Record<string, string>>({});

  const computeActive = useCallback(
    (ms: number): VideoSignal[] => {
      if (ms < 0 || !hasPlayed) return [];
      const canon = toCanonicalRef.current;
      const FORWARD_GRACE_MS = 250;
      const filtered = signals.filter((s) => {
        const start = Number(s.start_ms ?? 0);
        const rawEnd = Number(s.end_ms ?? s.start_ms ?? 0);
        const end = Math.max(rawEnd, start + 1) + FORWARD_GRACE_MS;
        if (ms < start || ms > end) return false;
        const cfg = SIGNAL_CONFIG[s.signal_type];
        if (!cfg) return false;
        if (!enabledCategories.has(cfg.category)) return false;
        if (selectedSpeaker !== "all") {
          const resolvedId = canon[s.speaker_id ?? ""] ?? s.speaker_id ?? "";
          if (resolvedId !== selectedSpeaker && s.speaker_id !== selectedSpeaker) return false;
        }
        if (s.confidence < 0.15) return false;
        return true;
      });
      // Multiple overlapping 2-second windows can each emit the same signal_type for the
      // same speaker. Deduplicate by keeping only the highest-confidence entry per pair.
      const best = new Map<string, VideoSignal>();
      for (const s of filtered) {
        const key = `${s.speaker_id ?? ""}::${s.signal_type}`;
        const prev = best.get(key);
        if (!prev || s.confidence > prev.confidence) best.set(key, s);
      }
      return Array.from(best.values());
    },
    [signals, enabledCategories, selectedSpeaker, hasPlayed]
  );

  const tick = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    const ms = video.currentTime * 1000;
    setCurrentTimeMs(ms);
    if (ms > 0 && !hasPlayed) setHasPlayed(true);
    setActiveSignals(computeActive(ms));
    if (!video.paused && !video.ended) {
      animFrameRef.current = requestAnimationFrame(tick);
    }
  }, [computeActive, hasPlayed]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onPlay = () => { animFrameRef.current = requestAnimationFrame(tick); };
    const onPause = () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      tick();
    };
    const onSeeked = () => tick();
    const onMeta = () => { if (video.duration > 0) setDurationMs(video.duration * 1000); };

    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("loadedmetadata", onMeta);
    video.addEventListener("loadeddata", onMeta);
    video.addEventListener("canplay", onMeta);

    // When tick is recreated (e.g. hasPlayed transitions false→true), the effect
    // cleanup cancels the RAF and re-runs this effect. If the video is already
    // playing at that point no "play" event fires, so the loop must be restarted
    // here explicitly — otherwise currentTimeMs freezes within the first few frames.
    if (!video.paused && !video.ended) {
      animFrameRef.current = requestAnimationFrame(tick);
    }

    return () => {
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("loadedmetadata", onMeta);
      video.removeEventListener("loadeddata", onMeta);
      video.removeEventListener("canplay", onMeta);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [tick]);

  // Re-filter active signals when filters change (without requiring seek)
  useEffect(() => {
    setActiveSignals(computeActive(currentTimeMs));
  }, [enabledCategories, selectedSpeaker, computeActive, currentTimeMs]);

  const seekTo = (ms: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = ms / 1000;
  };

  const toggleCategory = (cat: string) => {
    setEnabledCategories((prev) => {
      const next = new Set(prev);
      next.has(cat) ? next.delete(cat) : next.add(cat);
      return next;
    });
  };

  // ── Speaker identity resolution ───────────────────────────────────────────────
  // Two-pass approach:
  //   1. Registry merge — collapse multiple Face_N entries for the same person
  //      when ArcFace matched them to the same registry_id (safety net for when
  //      the backend merge didn't run).
  //   2. Window-bounded visibility — entries appear and disappear with their
  //      signals; no session-level pre-seeding, no cluster gate.

  const rosterLoaded = Object.keys(speakerRoster).length > 0;
  const allSpeakerIds = [...new Set(signals.map((s) => s.speaker_id).filter(Boolean))] as string[];

  // Pass 1 — group by registry_id
  const registryGroups: Record<string, string[]> = {};
  for (const spkId of allSpeakerIds) {
    const regId = rosterLoaded ? speakerRoster[spkId]?.registry_id : undefined;
    if (regId) (registryGroups[regId] ??= []).push(spkId);
  }

  // Pass 2 — pick canonical (most signals) per registry group; build toCanonical map
  const toCanonical: Record<string, string> = {};
  for (const spkIds of Object.values(registryGroups)) {
    if (spkIds.length <= 1) continue;
    const sigCount = (id: string) => signals.filter((s) => s.speaker_id === id).length;
    const canonical = [...spkIds].sort((a, b) => sigCount(b) - sigCount(a))[0];
    for (const id of spkIds) {
      if (id !== canonical) toCanonical[id] = canonical;
    }
  }
  // Keep the ref in sync so computeActive can use it without being a dependency
  toCanonicalRef.current = toCanonical;

  // Window-bounded grouping — entries exist only while their signals are active.
  // No cluster gate, no pre-seeding. Entries self-clean when their windows close.
  const visibleByFace = useMemo(() => {
    const groups: Record<
      string,
      {
        rawId: string;
        label: string;
        signals: VideoSignal[];
        faceCentreX: number;
        faceCentreY: number;
      }
    > = {};

    for (const s of activeSignals) {
      const rawId = s.speaker_id;
      if (!rawId) continue;
      const cfg = SIGNAL_CONFIG[s.signal_type];
      if (cfg?.category === "voice") continue;
      // Session-spanning hidden signals (denial_weakening, false_confession_risk, etc.)
      // feed InterrogationSummaryPanel but must not keep face entries alive in the
      // sidebar after the person's face is gone from frame.
      if (cfg?.hidden) continue;

      const canonId = toCanonical[rawId] ?? rawId;
      const rosterEntry = speakerRoster[canonId];
      const hasFacePos = s.metadata?.face_centre_x != null && s.metadata?.face_centre_y != null;
      const cx = hasFacePos ? (s.metadata!.face_centre_x as number) : undefined;
      const cy = hasFacePos ? (s.metadata!.face_centre_y as number) : undefined;

      const g = groups[canonId];
      if (!g) {
        groups[canonId] = {
          rawId: canonId,
          label: resolveDisplayLabel(rosterEntry, canonId),
          signals: [s],
          faceCentreX: cx ?? 0,
          faceCentreY: cy ?? 0,
        };
      } else {
        g.signals.push(s);
        if (cx != null && cy != null) {
          // Only update face position from signals that carry actual face metadata,
          // using the most recent such signal to follow natural head movement.
          const latestWithPos = g.signals
            .filter((x) => x.metadata?.face_centre_x != null)
            .reduce((a, b) => ((a.start_ms ?? 0) > (b.start_ms ?? 0) ? a : b));
          if (latestWithPos === s) {
            g.faceCentreX = cx;
            g.faceCentreY = cy;
          }
        }
      }
    }

    return groups;
  }, [activeSignals, toCanonical, speakerRoster]);

  const hasIncongruence = activeSignals.some((s) =>
    (s.signal_type === "head_body_incongruence"
      || s.signal_type === "verbal_nonverbal_discordance"
      || s.signal_type === "tone_face_masking")
    && (s.confidence ?? 0) > 0.40
  );

  const playheadPct = durationMs > 0 ? (currentTimeMs / durationMs) * 100 : 0;

  // Deduplicate timeline dots — one dot per signal_type per segment
  const timelineDots = signals.filter((s) => {
    const cfg = SIGNAL_CONFIG[s.signal_type];
    return cfg && enabledCategories.has(cfg.category);
  });

  const handcuffedDetected = useMemo(
    () =>
      signals.some(
        (s) => s.signal_type === "presence_detected" && s.metadata?.handcuffed === true
      ),
    [signals]
  );

  const KEY_INTERROGATION_EVENTS = useMemo(
    () =>
      new Set([
        "evidence_response_processing_delay",
        "statement_contamination",
        "capitulation_cascade",
        "freezing_response",
      ]),
    []
  );


  return (
    <div className="w-full space-y-3">
      {/* Video Playback — badges panel left, video right */}
      <div className="flex overflow-hidden rounded-lg bg-black" style={{ minHeight: 580 }}>

        {/* Left panel: signal badges in the black area */}
        <div className="flex w-48 flex-shrink-0 flex-col justify-start gap-1 overflow-y-auto p-3">
          {Object.keys(visibleByFace).length === 0 ? (
            <span className="mt-4 text-center text-[10px] text-white/20">No active signals</span>
          ) : (
            <>
              {Object.entries(visibleByFace).map(([speakerId, { label, rawId, signals: sigs }]) => {
                const prioritySigs = sigs
                  .filter((s: VideoSignal) => {
                    if (SIGNAL_CONFIG[s.signal_type]?.hidden) return false;
                    const display = getSignalDisplay(s.signal_type, s.value_text ?? "");
                    return display.priority <= (showExpanded ? 2 : 1);
                  })
                  .sort((a: VideoSignal, b: VideoSignal) => (b.confidence || 0) - (a.confidence || 0));
                const visibleSigs = showExpanded ? prioritySigs : prioritySigs.slice(0, 3);
                const hasPresence = sigs.some((s: VideoSignal) => s.signal_type === "presence_detected");
                if (visibleSigs.length === 0 && !hasPresence) return null;
                // Hide any speaker with no face thumbnail from the video sidebar.
                // Face_N without thumbnail = unconfirmed detection (ArcFace failure, photo).
                // Speaker_N without thumbnail = voice-only speaker, no face matched — their
                // signals belong in the voice/language panels, not the video player sidebar.
                if (!speakerRoster[rawId]?.thumbnail_url) return null;
                return (
                  <div key={speakerId} className="flex flex-col gap-1">
                    <SpeakerGroupHeader
                      speakerLabel={rawId}
                      displayLabel={label}
                      roster={speakerRoster}
                      highlighted={!!rawId && highlightedSpeaker === rawId}
                      onToggle={() =>
                        setHighlightedSpeaker((prev) =>
                          rawId && prev !== rawId ? rawId : null
                        )
                      }
                    />
                    {visibleSigs.map((s: VideoSignal, i: number) => {
                      const display = getSignalDisplay(s.signal_type, s.value_text ?? "");
                      const color = display.color;
                      return (
                        <div
                          key={`${s.signal_type}-${i}`}
                          className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium text-white"
                          style={{
                            backgroundColor: `${color}22`,
                            border: `1px solid ${color}55`,
                          }}
                          title={display.description}
                        >
                          <span className="font-mono text-[11px]">{display.icon}</span>
                          <span className="truncate">{display.label}</span>
                          {s.confidence >= 0.5 && (
                            <span className="ml-auto shrink-0 text-[10px] opacity-50">
                              {Math.round(s.confidence * 100)}%
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
              {(() => {
                const hiddenCount = showExpanded ? 0 : Object.values(visibleByFace).reduce(
                  (total: number, { signals: sigs }) => {
                    if (sigs.length === 0) return total;
                    const sort = (a: VideoSignal, b: VideoSignal) =>
                      (b.confidence || 0) - (a.confidence || 0);
                    const shownNow = sigs
                      .filter((s: VideoSignal) => getSignalDisplay(s.signal_type, s.value_text ?? "").priority <= 1)
                      .sort(sort)
                      .slice(0, 3).length;
                    const shownExpanded = sigs
                      .filter((s: VideoSignal) => getSignalDisplay(s.signal_type, s.value_text ?? "").priority <= 2)
                      .sort(sort).length;
                    return total + (shownExpanded - shownNow);
                  },
                  0,
                );
                return hiddenCount > 0 || showExpanded ? (
                  <button
                    onClick={() => setShowExpanded((v) => !v)}
                    className="mt-1 text-[9px] text-white/30 hover:text-white/60 transition-colors text-center"
                  >
                    {showExpanded ? "Show less" : `+${hiddenCount} more`}
                  </button>
                ) : null;
              })()}
            </>
          )}
        </div>

        {/* Right panel: video — 9:16 portrait */}
        <div className="relative flex flex-1 items-center justify-center">
          <div className="relative w-full" style={{ aspectRatio: "16/9" }}>
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="h-full w-full object-contain"
            />
            {highlightedSpeaker && (
              <FaceHighlight speaker={highlightedSpeaker} signals={activeSignals} />
            )}
            {hasIncongruence && (
              <div className="pointer-events-none absolute right-2 top-2 animate-pulse rounded-full bg-red-500/80 px-3 py-1.5 text-xs font-bold text-white backdrop-blur-sm">
                ! Incongruence Detected
              </div>
            )}
            {handcuffedDetected && (
              <div className="pointer-events-none absolute left-2 top-2 rounded-full bg-amber-500/70 px-3 py-1.5 text-xs font-medium text-white backdrop-blur-sm">
                Handcuffed — upper-body gesture analysis adjusted
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Interrogation summary panel — mounts only when interrogation signals present */}
      <InterrogationSummaryPanel signals={signals} durationMs={durationMs} />

      {/* Signal timeline bar */}
      <div className="rounded-lg border border-nexus-border bg-nexus-surface p-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-[11px] font-medium text-nexus-text-secondary">
            Signal Timeline
          </span>
          <span className="text-[11px] text-nexus-text-muted">
            {activeSignals.length} active / {signals.length} total
          </span>
        </div>

        {/* Lane rows — one per category, labels & bars perfectly aligned */}
        <div className="flex flex-col gap-px">
          {CATEGORIES.map((c) => {
            const laneDots = timelineDots.filter(
              (s) => SIGNAL_CONFIG[s.signal_type]?.category === c.key
            );
            const active = enabledCategories.has(c.key);
            return (
              <div key={c.key} className="flex items-stretch gap-2" style={{ height: 16 }}>
                {/* Label */}
                <span
                  className={`w-12 flex-shrink-0 self-center text-[8px] leading-none ${
                    active ? "text-nexus-text-muted" : "text-nexus-text-muted/30"
                  }`}
                >
                  {c.label}
                </span>

                {/* Bar for this lane */}
                <div
                  className="relative flex-1 overflow-hidden cursor-crosshair rounded-sm bg-nexus-bg"
                  style={{ height: 16 }}
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const pct = (e.clientX - rect.left) / rect.width;
                    seekTo(pct * durationMs);
                  }}
                >
                  {/* Playhead */}
                  {durationMs > 0 && (
                    <div
                      className="pointer-events-none absolute z-10 w-px bg-white/80"
                      style={{ top: 0, bottom: 0, left: `${playheadPct}%` }}
                    />
                  )}
                  {/* Signal segments */}
                  {(() => {
                    const effectiveDuration = durationMs > 0 ? durationMs : (signals.length > 0 ? Math.max(...signals.map(x => x.end_ms ?? 0)) : 1);
                    return laneDots.map((s, i) => {
                      const cfg = SIGNAL_CONFIG[s.signal_type];
                      if (!cfg) return null;
                      const color = resolveColor(cfg, s);
                      const segStart = s.start_ms;
                      const segEnd   = s.end_ms;
                      const left = Math.max(0, Math.min((segStart / effectiveDuration) * 100, 100));
                      const rawWidth = Math.max(((segEnd - segStart) / effectiveDuration) * 100, 0.5);
                      const width = Math.min(rawWidth, 100 - left);
                      const isKeyEvent = KEY_INTERROGATION_EVENTS.has(s.signal_type);
                      const title = `${getSignalDisplay(s.signal_type, s.value_text ?? "").label} @ ${(s.start_ms / 1000).toFixed(1)}s`;
                      if (isKeyEvent) {
                        return (
                          <div
                            key={`${s.signal_type}-${s.start_ms}-${i}`}
                            className="absolute flex flex-col items-center opacity-90 hover:opacity-100 transition-opacity cursor-pointer"
                            style={{ left: `${left}%`, width: 6, top: 0, bottom: 0, marginLeft: -3 }}
                            title={title}
                            onClick={(e) => { e.stopPropagation(); seekTo(s.start_ms); }}
                          >
                            <div style={{ width: 0, height: 0, borderLeft: '3px solid transparent', borderRight: '3px solid transparent', borderBottom: `5px solid ${color}`, flexShrink: 0 }} />
                            <div style={{ flex: 1, width: 2, backgroundColor: color }} />
                          </div>
                        );
                      }
                      return (
                        <div
                          key={`${s.signal_type}-${s.start_ms}-${i}`}
                          className="absolute rounded-sm opacity-80 hover:opacity-100 transition-opacity"
                          style={{ left: `${left}%`, width: `${width}%`, minWidth: '3px', top: 1, bottom: 1, backgroundColor: color }}
                          title={title}
                          onClick={(e) => { e.stopPropagation(); seekTo(s.start_ms); }}
                        />
                      );
                    });
                  })()}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Filter controls */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Raw / Annotated toggle — or "processing" indicator while burn runs */}
        {!annotatedAvailable && (
          <span className="text-[10px] text-nexus-text-muted animate-pulse">
            Landmark video processing…
          </span>
        )}
        {annotatedAvailable && (
          <div className="flex rounded-lg border border-nexus-border overflow-hidden">
            <button
              onClick={() => setShowAnnotated(false)}
              className={`px-3 py-1 text-xs font-medium transition-colors ${
                !showAnnotated
                  ? "bg-blue-600 text-white"
                  : "bg-nexus-surface text-nexus-text-muted hover:text-nexus-text-secondary"
              }`}
            >
              Raw
            </button>
            <button
              onClick={() => setShowAnnotated(true)}
              className={`px-3 py-1 text-xs font-medium transition-colors ${
                showAnnotated
                  ? "bg-blue-600 text-white"
                  : "bg-nexus-surface text-nexus-text-muted hover:text-nexus-text-secondary"
              }`}
            >
              Landmarks
            </button>
          </div>
        )}

        <div className="flex gap-1.5">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.key}
              onClick={() => toggleCategory(cat.key)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                enabledCategories.has(cat.key)
                  ? "bg-blue-600 text-white"
                  : "bg-nexus-surface text-nexus-text-muted hover:text-nexus-text-secondary"
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {allSpeakerIds.length > 1 && (
          <select
            value={selectedSpeaker}
            onChange={(e) => setSelectedSpeaker(e.target.value)}
            className="rounded-lg border border-nexus-border bg-nexus-surface px-3 py-1 text-xs text-nexus-text-primary"
          >
            <option value="all">All Speakers</option>
            {allSpeakerIds.map((spk) => (
              <option key={spk} value={spk}>
                {resolveDisplayLabel(speakerRoster[spk], spk)}
              </option>
            ))}
          </select>
        )}
      </div>
    </div>
  );
}
