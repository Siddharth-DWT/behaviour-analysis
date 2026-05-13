import { useState, useMemo } from "react";
import {
  Mic,
  MessageSquareText,
  Combine,
  Eye,
  ChevronDown,
  ChevronRight,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";
import type { Signal } from "../api/client";

// ── Signal type metadata ──

interface SignalTypeMeta {
  label: string;
  description: string;
  category: "voice" | "language" | "fusion" | "video";
  interpret: (s: Signal) => { label: string; severity: "high" | "med" | "low" | "neutral" | "info" } | null;
}

const SIGNAL_TYPE_META: Record<string, SignalTypeMeta> = {
  vocal_stress_score: {
    label: "Vocal Stress",
    description: "Composite stress score derived from pitch, jitter, shimmer, and HNR deviation",
    category: "voice",
    interpret: (s) => {
      if (s.value == null) return null;
      if (s.value > 0.6) return { label: `${(s.value * 100).toFixed(0)}% — High stress`, severity: "high" };
      if (s.value > 0.35) return { label: `${(s.value * 100).toFixed(0)}% — Moderate`, severity: "med" };
      return { label: `${(s.value * 100).toFixed(0)}% — Normal`, severity: "low" };
    },
  },
  pitch_elevation_flag: {
    label: "Pitch Elevation",
    description: "Flagged when speaker's pitch deviates significantly from their baseline",
    category: "voice",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("significant") || txt.includes("high")) return { label: "Significantly elevated", severity: "high" };
      if (txt.includes("mild")) return { label: "Mildly elevated", severity: "med" };
      return { label: s.value_text || "Elevated", severity: "med" };
    },
  },
  filler_detection: {
    label: "Filler Words",
    description: "Frequency of hesitation markers (um, uh, like, you know)",
    category: "voice",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "normal" || txt === "none" || s.value === 0) return null; // skip normal
      if (txt === "excessive" || txt === "high") return { label: "Excessive fillers", severity: "high" };
      if (txt === "elevated") return { label: "Elevated fillers", severity: "med" };
      return { label: s.value_text || "Detected", severity: "med" };
    },
  },
  tone_classification: {
    label: "Tone",
    description: "Vocal tone classification — nervous, confident, assertive, etc.",
    category: "voice",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "neutral" || txt === "calm" || txt === "normal") return null;
      if (txt === "nervous" || txt === "anxious") return { label: s.value_text!, severity: "high" };
      if (txt === "confident" || txt === "assertive") return { label: s.value_text!, severity: "low" };
      return { label: s.value_text || "Unknown", severity: "neutral" };
    },
  },
  speech_rate_anomaly: {
    label: "Speech Rate",
    description: "Deviation from speaker's baseline speaking pace",
    category: "voice",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "normal" || txt === "baseline") return null;
      if (txt.includes("rapid") || txt.includes("fast")) return { label: `Rapid speech (${s.value != null ? Math.round(Math.abs(s.value) * 100) : ""}% dev)`, severity: "med" };
      if (txt.includes("slow") || txt.includes("depress")) return { label: `Slow speech (${s.value != null ? Math.round(Math.abs(s.value) * 100) : ""}% dev)`, severity: "med" };
      return { label: s.value_text || "Anomaly", severity: "med" };
    },
  },
  sentiment_score: {
    label: "Sentiment",
    description: "Emotional valence of language — positive, negative, or neutral",
    category: "language",
    interpret: (s) => {
      if (s.value == null) return null;
      if (s.value > 0.6) return { label: `Positive (${(s.value * 100).toFixed(0)}%)`, severity: "low" };
      if (s.value < -0.3) return { label: `Negative (${(s.value * 100).toFixed(0)}%)`, severity: "high" };
      return { label: `Neutral (${(s.value * 100).toFixed(0)}%)`, severity: "neutral" };
    },
  },
  power_language_score: {
    label: "Power Language",
    description: "Use of assertive, authoritative, or hedging language patterns",
    category: "language",
    interpret: (s) => {
      if (s.value == null) return null;
      if (s.value > 0.7) return { label: "Powerful / Assertive", severity: "low" };
      if (s.value < 0.4) return { label: "Hedging / Tentative", severity: "med" };
      return { label: "Moderate", severity: "neutral" };
    },
  },
  intent_classification: {
    label: "Intent",
    description: "Conversational intent — proposing, questioning, agreeing, objecting, etc.",
    category: "language",
    interpret: (s) => {
      const txt = (s.value_text || "").toUpperCase();
      if (txt === "INFORM") return null;
      const labels: Record<string, string> = {
        PROPOSE: "Proposing", QUESTION: "Question", AGREE: "Agreeing",
        COMMIT: "Committing", OBJECTION: "Objecting", RAPPORT: "Rapport",
        CONVINCE: "Persuading", CHALLENGE: "Challenging",
      };
      return { label: labels[txt] || s.value_text || txt, severity: "info" };
    },
  },
  buying_signal: {
    label: "Buying Signal",
    description: "Indicators of purchase intent or positive decision-making",
    category: "language",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "true" || txt.includes("strong")) return { label: "Strong buying signal", severity: "low" };
      if (txt.includes("weak")) return { label: "Weak buying signal", severity: "neutral" };
      return { label: "Buying signal detected", severity: "low" };
    },
  },
  objection_signal: {
    label: "Objection",
    description: "Verbal resistance, pushback, or concerns raised",
    category: "language",
    interpret: (s) => ({ label: "Objection raised", severity: "high" }),
  },
  urgency_authenticity: {
    label: "Urgency Authenticity",
    description: "Cross-modal check — is the urgency genuine or manufactured?",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("manufactured")) return { label: "Manufactured urgency", severity: "high" };
      if (txt.includes("authentic")) return { label: "Authentic urgency", severity: "low" };
      return { label: "Ambiguous urgency", severity: "med" };
    },
  },
  credibility_assessment: {
    label: "Credibility",
    description: "Voice stress vs. language content alignment check",
    category: "fusion",
    interpret: (s) => ({ label: "Credibility concern", severity: "high" }),
  },
  verbal_incongruence: {
    label: "Verbal Incongruence",
    description: "Mismatch between what is said and how it is said",
    category: "fusion",
    interpret: (s) => ({ label: s.value_text?.replace(/_/g, " ") || "Mismatch detected", severity: "med" }),
  },
  tension_cluster: {
    label: "Tension Cluster",
    description: "Multiple negative signals concentrated in a short time window",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("high")) return { label: "High tension cluster", severity: "high" };
      return { label: "Moderate tension cluster", severity: "med" };
    },
  },
  momentum_shift: {
    label: "Momentum Shift",
    description: "Detected shift in conversation trajectory (positive or negative)",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("positive")) return { label: "Positive momentum shift", severity: "low" };
      if (txt.includes("negative")) return { label: "Negative momentum shift", severity: "high" };
      return { label: "Momentum change detected", severity: "med" };
    },
  },
  persistent_incongruence: {
    label: "Persistent Incongruence",
    description: "Speaker shows sustained mismatch between verbal and vocal signals across multiple time windows",
    category: "fusion",
    interpret: (s) => ({ label: "Persistent incongruence pattern", severity: "high" }),
  },

  // ── Phase 2E fusion (audio × video) ──────────────────────────────────────
  tone_face_masking: {
    label: "Voice-Face Masking",
    description: "Voice tone and facial emotion strongly contradict each other",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("strong")) return { label: "Strong masking detected", severity: "high" };
      if (txt.includes("moderate")) return { label: "Moderate masking", severity: "med" };
      return { label: "Mild voice-face mismatch", severity: "neutral" };
    },
  },
  stress_suppression: {
    label: "Stress Suppression",
    description: "Stress visible in one channel (voice or face) but suppressed in the other",
    category: "fusion",
    interpret: (s) => ({ label: "Stress suppression detected", severity: "high" }),
  },
  cognitive_load: {
    label: "Cognitive Overload",
    description: "Filler words and gaze breaks co-occurring — high cognitive demand",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("high")) return { label: "High cognitive load", severity: "high" };
      if (txt.includes("moderate")) return { label: "Moderate cognitive load", severity: "med" };
      return { label: "Mild cognitive load", severity: "neutral" };
    },
  },
  nonverbal_disagreement: {
    label: "Nonverbal Disagreement",
    description: "Head shake co-occurs with objection language — explicit disagreement",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("explicit")) return { label: "Explicit disagreement", severity: "high" };
      return { label: "Polite but resistant", severity: "med" };
    },
  },
  physical_engagement: {
    label: "Physical Engagement",
    description: "Forward lean and strong visual attention — fully engaged",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "disengagement" || txt.includes("disengage")) return { label: "Physical disengagement", severity: "high" };
      if (txt.includes("elsewhere")) return { label: "Body present, mind elsewhere", severity: "med" };
      return { label: "High physical engagement", severity: "low" };
    },
  },
  false_confidence: {
    label: "False Confidence",
    description: "Gaze breaks align with hedging language — low genuine confidence",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("low_confidence")) return { label: "Low genuine confidence", severity: "high" };
      if (txt.includes("mild")) return { label: "Mild uncertainty", severity: "med" };
      return { label: "Hedged statement", severity: "neutral" };
    },
  },
  smile_sentiment_incongruence: {
    label: "Smile Masks Sentiment",
    description: "Social smile co-occurring with negative sentiment — masking displeasure",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("sarcasm")) return { label: "Possible sarcasm", severity: "med" };
      return { label: "Smile masking negative sentiment", severity: "high" };
    },
  },
  processing_load: {
    label: "Processing Load",
    description: "Long response latency with facial stress — overwhelmed",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("high")) return { label: "High processing load", severity: "high" };
      if (txt.includes("elevated")) return { label: "Elevated processing load", severity: "med" };
      return { label: "Mild processing delay", severity: "neutral" };
    },
  },
  dominance_anxiety: {
    label: "Dominance Anxiety",
    description: "Dominant language but gaze avoidance — anxiety under dominant facade",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "dominance_anxiety") return { label: "Dominance anxiety", severity: "high" };
      return { label: "Dominance with uncertainty", severity: "med" };
    },
  },
  interrupt_intent: {
    label: "Interrupt Intent",
    description: "Forward lean during interruption — assertive or defensive intent",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("competitive")) return { label: "Competitive interrupt", severity: "med" };
      return { label: "Reactive interrupt", severity: "neutral" };
    },
  },
  rapport_confirmation: {
    label: "Rapport Confirmed",
    description: "Empathy language and head nods strongly aligned — genuine rapport",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("strong")) return { label: "Strong rapport", severity: "low" };
      if (txt.includes("building")) return { label: "Building rapport", severity: "low" };
      return { label: "Rapport indicator", severity: "low" };
    },
  },
  head_body_incongruence: {
    label: "Head-Body Incongruence",
    description: "Head and body signals contradict each other",
    category: "fusion",
    interpret: (s) => ({ label: "Head-body mismatch", severity: "med" }),
  },
  voice_face_alignment: {
    label: "Voice-Face Alignment",
    description: "Alignment (or mismatch) between vocal tone and facial expression",
    category: "fusion",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "congruent") return null;
      return { label: `Misaligned: ${txt.replace(/_/g, " ")}`, severity: "med" };
    },
  },

  // ── Video signals ─────────────────────────────────────────────────────────
  facial_emotion: {
    label: "Facial Emotion",
    description: "Dominant facial emotion detected via DeepFace/FACS analysis",
    category: "video",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (!txt || txt === "neutral") return null;
      const severity: Record<string, "high" | "med" | "low" | "neutral"> = {
        angry: "high", fearful: "high", contempt: "high",
        disgusted: "med", sad: "med",
        surprised: "neutral",
        happy: "low",
      };
      return { label: txt.charAt(0).toUpperCase() + txt.slice(1), severity: severity[txt] ?? "neutral" };
    },
  },
  facial_engagement: {
    label: "Facial Engagement",
    description: "Engagement level inferred from facial expression activity",
    category: "video",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "high_engagement") return { label: "High engagement", severity: "low" };
      if (txt === "low_engagement") return { label: "Low engagement", severity: "med" };
      if (txt === "disengaged") return { label: "Disengaged", severity: "high" };
      return null;
    },
  },
  facial_stress: {
    label: "Facial Stress",
    description: "Visible stress indicators in facial muscle activity (AU4, AU7, AU23)",
    category: "video",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "high_facial_stress") return { label: "High facial stress", severity: "high" };
      if (txt === "moderate") return { label: "Moderate facial stress", severity: "med" };
      return null;
    },
  },
  smile_type: {
    label: "Smile Type",
    description: "Duchenne (genuine) vs. social vs. forced smile classification",
    category: "video",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "duchenne") return { label: "Genuine (Duchenne) smile", severity: "low" };
      if (txt === "forced") return { label: "Forced smile", severity: "med" };
      if (txt === "social") return { label: "Social smile", severity: "neutral" };
      return null;
    },
  },
  body_lean: {
    label: "Body Lean",
    description: "Direction of body lean — forward lean signals engagement, back lean signals withdrawal",
    category: "video",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "forward_lean") return { label: "Forward lean — engaged", severity: "low" };
      if (txt === "back_lean") return { label: "Back lean — withdrawing", severity: "med" };
      return null;
    },
  },
  posture: {
    label: "Posture",
    description: "Body posture classification — upright, slumped, open, or closed",
    category: "video",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (!txt || txt === "upright") return null;
      if (txt.includes("slump") || txt.includes("closed")) return { label: txt.replace(/_/g, " "), severity: "med" };
      return { label: txt.replace(/_/g, " "), severity: "neutral" };
    },
  },
  head_nod: {
    label: "Head Nod",
    description: "Head nodding — agreement or active listening",
    category: "video",
    interpret: (s) => ({ label: "Head nod", severity: "low" }),
  },
  head_shake: {
    label: "Head Shake",
    description: "Head shaking — disagreement or uncertainty",
    category: "video",
    interpret: (s) => ({ label: "Head shake", severity: "med" }),
  },
  body_fidgeting: {
    label: "Body Fidgeting",
    description: "Excessive body movement or restlessness — discomfort or stress indicator",
    category: "video",
    interpret: (s) => {
      if (s.value == null) return null;
      if (s.value > 0.6) return { label: `High fidget (${(s.value * 100).toFixed(0)}%)`, severity: "high" };
      if (s.value > 0.3) return { label: `Moderate fidget (${(s.value * 100).toFixed(0)}%)`, severity: "med" };
      return null;
    },
  },
  screen_contact: {
    label: "Screen Contact",
    description: "Gaze direction relative to camera — eye contact and attention measure",
    category: "video",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt === "low_screen_contact") return { label: `Low eye contact (${s.value != null ? Math.round(s.value * 100) : "?"}%)`, severity: "med" };
      if (txt === "sustained_high") return { label: "Sustained eye contact", severity: "low" };
      return null;
    },
  },
  attention_level: {
    label: "Attention Level",
    description: "Overall attentiveness based on gaze pattern and direction analysis",
    category: "video",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("high")) return { label: "High attention", severity: "low" };
      if (txt.includes("reduced")) return { label: "Reduced attention", severity: "med" };
      if (txt.includes("low")) return { label: "Low attention", severity: "high" };
      return null;
    },
  },
  sustained_distraction: {
    label: "Sustained Distraction",
    description: "Extended gaze away from screen — attention lost to off-screen stimulus",
    category: "video",
    interpret: (s) => ({ label: "Sustained distraction", severity: "high" }),
  },
  gaze_synchrony: {
    label: "Gaze Synchrony",
    description: "Coordination of gaze patterns between participants — rapport indicator",
    category: "video",
    interpret: (s) => {
      const txt = (s.value_text || "").toLowerCase();
      if (txt.includes("high") || txt.includes("strong")) return { label: "High gaze synchrony", severity: "low" };
      if (txt.includes("low") || txt.includes("none")) return { label: "Low gaze synchrony", severity: "med" };
      return { label: "Moderate gaze synchrony", severity: "neutral" };
    },
  },
};

// ── Helpers ──

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

const SEVERITY_STYLES: Record<string, { dot: string; text: string; bg: string }> = {
  high: { dot: "bg-[var(--stress-high)]", text: "text-nexus-stress-high", bg: "bg-stress-high-10" },
  med: { dot: "bg-[var(--stress-med)]", text: "text-nexus-stress-med", bg: "bg-stress-med-10" },
  low: { dot: "bg-[var(--stress-low)]", text: "text-nexus-stress-low", bg: "bg-stress-low-10" },
  neutral: { dot: "bg-[var(--neutral)]", text: "text-nexus-text-secondary", bg: "bg-neutral-20" },
  info: { dot: "bg-[var(--accent-blue)]", text: "text-nexus-accent-blue", bg: "bg-accent-blue-10" },
};

const AGENT_CONFIG = {
  voice: { label: "Voice", icon: Mic, color: "var(--agent-voice)", bgClass: "bg-accent-blue-10" },
  language: { label: "Language", icon: MessageSquareText, color: "var(--agent-language)", bgClass: "bg-accent-purple-10" },
  fusion: { label: "Fusion", icon: Combine, color: "var(--agent-fusion)", bgClass: "bg-alert-10" },
  video: { label: "Video", icon: Eye, color: "var(--agent-gaze, #EC4899)", bgClass: "bg-pink-10" },
};

type AgentFilter = "all" | "voice" | "language" | "fusion" | "video";
type SeverityFilter = "all" | "high" | "med" | "low";

interface SignalGroup {
  signalType: string;
  meta: SignalTypeMeta;
  signals: Signal[];
  noteworthyCount: number;
  highCount: number;
  medCount: number;
  lowCount: number;
}

// ── Component ──

interface Props {
  signals: Signal[];
  signalsByAgent: Record<string, number>;
  totalCount: number;
  speakerRoles?: Record<string, string>;
}

export default function SignalExplorer({ signals, signalsByAgent, totalCount, speakerRoles }: Props) {
  const [agentFilter, setAgentFilter] = useState<AgentFilter>("all");
  const [severityFilter, setSeverityFilter] = useState<SeverityFilter>("all");
  const [expandedType, setExpandedType] = useState<string | null>(null);

  // Group and classify signals
  const groups = useMemo(() => {
    const map = new Map<string, Signal[]>();
    for (const s of signals) {
      const key = s.signal_type;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(s);
    }

    const result: SignalGroup[] = [];
    for (const [signalType, sigs] of map) {
      const meta = SIGNAL_TYPE_META[signalType];
      if (!meta) continue; // skip unknown types

      let noteworthy = 0, high = 0, med = 0, low = 0;
      for (const s of sigs) {
        const interp = meta.interpret(s);
        if (interp) {
          noteworthy++;
          if (interp.severity === "high") high++;
          else if (interp.severity === "med") med++;
          else if (interp.severity === "low") low++;
        }
      }

      result.push({ signalType, meta, signals: sigs, noteworthyCount: noteworthy, highCount: high, medCount: med, lowCount: low });
    }

    // Sort: fusion first, then language, voice, video; then by noteworthy count desc
    result.sort((a, b) => {
      const catOrder = { fusion: 0, language: 1, voice: 2, video: 3 };
      const catDiff = (catOrder[a.meta.category] ?? 4) - (catOrder[b.meta.category] ?? 4);
      if (catDiff !== 0) return catDiff;
      return b.highCount - a.highCount || b.noteworthyCount - a.noteworthyCount;
    });

    return result;
  }, [signals]);

  // Apply filters
  const filtered = useMemo(() => {
    return groups.filter((g) => {
      if (agentFilter !== "all" && g.meta.category !== agentFilter) return false;
      if (severityFilter === "high" && g.highCount === 0) return false;
      if (severityFilter === "med" && g.medCount === 0 && g.highCount === 0) return false;
      if (severityFilter === "low" && g.noteworthyCount === 0) return false;
      return true;
    });
  }, [groups, agentFilter, severityFilter]);

  // Summary stats
  const totalNoteworthy = groups.reduce((a, g) => a + g.noteworthyCount, 0);
  const totalHigh = groups.reduce((a, g) => a + g.highCount, 0);

  return (
    <section className="rounded-lg border border-nexus-border bg-nexus-surface overflow-hidden">
      {/* Header */}
      <div className="p-4 pb-3 border-b border-nexus-border">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-nexus-accent-blue" />
            <h2 className="text-sm font-semibold text-nexus-text-primary">Signal Explorer</h2>
          </div>
          <div className="flex items-center gap-3 text-[11px] text-nexus-text-muted font-mono">
            <span>{totalCount} total</span>
            <span className="text-nexus-stress-high">{totalHigh} high</span>
            <span>{totalNoteworthy} noteworthy</span>
          </div>
        </div>

        {/* Agent breakdown bar */}
        <div className="flex h-2 rounded-full overflow-hidden bg-nexus-surface-hover mb-3">
          {Object.entries(signalsByAgent).map(([agent, count]) => {
            const cfg = AGENT_CONFIG[agent as keyof typeof AGENT_CONFIG];
            if (!cfg) return null;
            const pct = (count / totalCount) * 100;
            return (
              <div
                key={agent}
                className="h-full transition-all"
                style={{ width: `${pct}%`, background: cfg.color }}
                title={`${cfg.label}: ${count} signals (${pct.toFixed(0)}%)`}
              />
            );
          })}
        </div>

        {/* Agent legend + filters */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={() => setAgentFilter("all")}
            className={`rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors ${
              agentFilter === "all"
                ? "bg-nexus-surface-hover text-nexus-text-primary"
                : "text-nexus-text-muted hover:text-nexus-text-secondary"
            }`}
          >
            All Agents
          </button>
          {Object.entries(signalsByAgent).map(([agent, count]) => {
            const cfg = AGENT_CONFIG[agent as keyof typeof AGENT_CONFIG];
            if (!cfg) return null;
            const Icon = cfg.icon;
            const isActive = agentFilter === agent;
            return (
              <button
                key={agent}
                onClick={() => setAgentFilter(isActive ? "all" : agent as AgentFilter)}
                className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors ${
                  isActive
                    ? `${cfg.bgClass} text-nexus-text-primary`
                    : "text-nexus-text-muted hover:text-nexus-text-secondary"
                }`}
              >
                <Icon className="h-3 w-3" style={{ color: cfg.color }} />
                {cfg.label}
                <span className="font-mono">{count}</span>
              </button>
            );
          })}

          <span className="mx-1 h-3 w-px bg-nexus-border" />

          {(["all", "high", "med", "low"] as const).map((sev) => {
            const labels = { all: "Any", high: "High", med: "Medium", low: "Low" };
            const isActive = severityFilter === sev;
            return (
              <button
                key={sev}
                onClick={() => setSeverityFilter(isActive ? "all" : sev)}
                className={`rounded-full px-2 py-1 text-[10px] font-medium transition-colors ${
                  isActive
                    ? "bg-nexus-surface-hover text-nexus-text-primary"
                    : "text-nexus-text-muted hover:text-nexus-text-secondary"
                }`}
              >
                {labels[sev]}
              </button>
            );
          })}
        </div>
      </div>

      {/* Signal type rows */}
      <div className="divide-y divide-nexus-border">
        {filtered.map((group) => {
          const isExpanded = expandedType === group.signalType;
          const cfg = AGENT_CONFIG[group.meta.category];
          const Icon = cfg?.icon || Activity;

          return (
            <div key={group.signalType}>
              {/* Row header */}
              <button
                onClick={() => setExpandedType(isExpanded ? null : group.signalType)}
                className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-nexus-surface-hover transition-colors"
              >
                {isExpanded ? (
                  <ChevronDown className="h-3 w-3 text-nexus-text-muted shrink-0" />
                ) : (
                  <ChevronRight className="h-3 w-3 text-nexus-text-muted shrink-0" />
                )}

                <Icon className="h-3.5 w-3.5 shrink-0" style={{ color: cfg?.color }} />

                <span className="text-xs font-medium text-nexus-text-primary flex-1">
                  {group.meta.label}
                </span>

                {/* Mini severity dots */}
                <div className="flex items-center gap-2">
                  {group.highCount > 0 && (
                    <span className="inline-flex items-center gap-1 text-[10px] font-mono text-nexus-stress-high">
                      <span className="h-1.5 w-1.5 rounded-full bg-[var(--stress-high)]" />
                      {group.highCount}
                    </span>
                  )}
                  {group.medCount > 0 && (
                    <span className="inline-flex items-center gap-1 text-[10px] font-mono text-nexus-stress-med">
                      <span className="h-1.5 w-1.5 rounded-full bg-[var(--stress-med)]" />
                      {group.medCount}
                    </span>
                  )}
                  {group.lowCount > 0 && (
                    <span className="inline-flex items-center gap-1 text-[10px] font-mono text-nexus-stress-low">
                      <span className="h-1.5 w-1.5 rounded-full bg-[var(--stress-low)]" />
                      {group.lowCount}
                    </span>
                  )}
                </div>

                <span className="text-[10px] font-mono text-nexus-text-muted w-8 text-right">
                  {group.signals.length}
                </span>
              </button>

              {/* Expanded detail */}
              {isExpanded && (
                <div className="bg-nexus-bg px-4 pb-3">
                  <p className="text-[11px] text-nexus-text-muted mb-2 pl-6">
                    {group.meta.description}
                  </p>
                  <div className="space-y-1 pl-6 max-h-[240px] overflow-y-auto">
                    {group.signals.map((sig, i) => {
                      const interp = group.meta.interpret(sig);
                      if (!interp) return null;
                      const style = SEVERITY_STYLES[interp.severity] || SEVERITY_STYLES.neutral;
                      const speaker = sig.speaker_label || "Unknown";
                      const role = speakerRoles?.[speaker];
                      const speakerDisplay = role ? `${role}` : speaker;

                      return (
                        <div
                          key={sig.id || i}
                          className={`flex items-center gap-2 rounded px-2 py-1.5 text-[11px] ${style.bg}`}
                        >
                          <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${style.dot}`} />
                          <span className={`font-medium ${style.text}`}>
                            {interp.label}
                          </span>
                          <span className="text-nexus-text-muted">
                            {speakerDisplay}
                          </span>
                          <span className="ml-auto font-mono text-nexus-text-muted">
                            {formatTime(sig.window_start_ms)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {filtered.length === 0 && (
          <div className="px-4 py-6 text-center text-xs text-nexus-text-muted">
            No signals match the current filters.
          </div>
        )}
      </div>
    </section>
  );
}
