import { useState, useEffect, useMemo } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  FileText,
  Loader2,
  Clock,
  Users,
  AlertTriangle,
  Sparkles,
  Target,
  Lightbulb,
  CheckCircle,
  XCircle,
  TrendingUp,
  ShieldCheck,
  MessageSquare,
  AlertCircle,
  Info,
  Zap,
} from "lucide-react";
import { format } from "date-fns";
import { getSession, getSignals, getTranscript, getReport, getVideoSignals } from "../api/client";
import type { Signal, TranscriptSegment } from "../api/client";
import TranscriptBlock from "../components/TranscriptBlock";
import StressTimeline from "../components/StressTimeline";
import AlertCard from "../components/AlertCard";
import SignalExplorer from "../components/SignalExplorer";
import TopicTimeline from "../components/TopicTimeline";
import SignalChainCards from "../components/SignalChainCards";
import SpeakerGraph from "../components/SpeakerGraph";
import InsightPanel from "../components/InsightPanel";
import ConversationGraph from "../components/ConversationGraph";
import SessionChat from "../components/SessionChat";
import SwimlaneTimeline from "../components/SwimlaneTimeline";
import TranscriptView from "../components/TranscriptView";
import GraphInsightsCard from "../components/GraphInsightsCard";
import BehavioralOverview from "../components/BehavioralOverview";
import VideoSignalPlayer from "../components/VideoSignalPlayer";

// ── Helpers ──

const VIDEO_EXTENSIONS = new Set(["mp4", "webm", "mov", "avi", "mkv", "m4v"]);

function isVideoFile(url: string | null | undefined): boolean {
  if (!url) return false;
  const ext = url.split("?")[0].split(".").pop()?.toLowerCase() ?? "";
  return VIDEO_EXTENSIONS.has(ext);
}

function formatDuration(ms: number | null): string {
  if (!ms) return "--";
  const totalSec = Math.round(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return sec > 0 ? `${min}m ${sec}s` : `${min}m`;
}

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

function matchSignalsToSegment(
  segment: TranscriptSegment,
  signals: Signal[]
): Signal[] {
  return signals.filter((s) => {
    const overlapStart = Math.max(s.window_start_ms, segment.start_ms);
    const overlapEnd = Math.min(s.window_end_ms, segment.end_ms);
    if (overlapEnd <= overlapStart) return false;
    if (
      s.speaker_label &&
      segment.speaker_label &&
      s.speaker_label !== segment.speaker_label
    ) {
      return false;
    }
    return true;
  });
}

function getCSSVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ── Gauge Bar ──

function GaugeBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-[11px] text-nexus-text-secondary mb-1">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-nexus-surface-hover overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(pct, 100)}%`, background: color }}
        />
      </div>
    </div>
  );
}

// ── Stat Chip ──

function StatChip({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-nexus-surface-hover px-2.5 py-1 text-[11px]">
      {color && <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />}
      <span className="text-nexus-text-secondary">{label}</span>
      <span className="font-mono font-medium text-nexus-text-primary">{value}</span>
    </span>
  );
}

// ── Speaker Analysis ──

interface SpeakerStats {
  label: string;
  role?: string;
  avgStress: number;
  maxStress: number;
  dominantTone: string;
  avgSentiment: number;
  avgPower: number;
  avgConfidence: number;
  fillerCount: number;
  buyingSignalCount: number;
  objectionCount: number;
}

function computeSpeakerStats(signals: Signal[]): SpeakerStats[] {
  const speakerMap = new Map<string, Signal[]>();

  for (const s of signals) {
    if (!s.speaker_label) continue;  // Skip signals without speaker attribution
    const label = s.speaker_label;
    if (!speakerMap.has(label)) speakerMap.set(label, []);
    speakerMap.get(label)!.push(s);
  }

  const stats: SpeakerStats[] = [];

  for (const [label, sigs] of speakerMap) {
    // Only include speakers who actually spoke — face-only participants (Face_N)
    // have no voice or language signals and would show meaningless default values.
    if (!sigs.some((s) => s.agent === "voice" || s.agent === "language")) continue;

    const stressVals = sigs
      .filter((s) => s.signal_type === "vocal_stress_score" && s.value != null)
      .map((s) => s.value!);
    const sentimentVals = sigs
      .filter((s) => s.signal_type === "sentiment_score" && s.value != null)
      .map((s) => s.value!);
    const powerVals = sigs
      .filter((s) => s.signal_type === "power_language_score" && s.value != null)
      .map((s) => s.value!);

    // Dominant tone from tone_classification signals
    const tones = sigs
      .filter((s) => (s.signal_type === "tone_classification" || s.signal_type === "tone_analysis") && s.value_text)
      .map((s) => s.value_text);
    const toneFreq = new Map<string, number>();
    for (const t of tones) {
      if (t && t.toLowerCase() !== "neutral") {
        toneFreq.set(t, (toneFreq.get(t) || 0) + 1);
      }
    }
    let dominantTone = "Neutral";
    let maxFreq = 0;
    for (const [tone, freq] of toneFreq) {
      if (freq > maxFreq) {
        dominantTone = tone;
        maxFreq = freq;
      }
    }

    // Filler count — count ALL filler_detection signals (each represents a filler event)
    const fillerCount = sigs.filter(
      (s) => s.signal_type === "filler_detection"
    ).length;

    // Buying signals — any signal with type buying_signal, regardless of value
    const buyingSignalCount = sigs.filter(
      (s) => s.signal_type === "buying_signal"
    ).length;

    // Objections
    const objectionCount = sigs.filter(
      (s) => s.signal_type === "objection_signal"
    ).length;

    const avg = (arr: number[]) => (arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);

    stats.push({
      label,
      avgStress: avg(stressVals),
      maxStress: stressVals.length > 0 ? Math.max(...stressVals) : 0,
      dominantTone: dominantTone.charAt(0).toUpperCase() + dominantTone.slice(1),
      avgSentiment: avg(sentimentVals),
      avgPower: powerVals.length > 0 ? avg(powerVals) : 0.5,
      avgConfidence: avg(
        sigs.filter((s) => s.confidence != null).map((s) => s.confidence)
      ),
      fillerCount,
      buyingSignalCount,
      objectionCount,
    });
  }

  return stats;
}

// ── Video Stats per speaker ──

interface VideoStats {
  // Face
  dominantEmotion: string | null;
  dominantEmotionIntensity: number;   // 0.0–1.0 peak value for emoji selection
  facialEngagement: "high_engagement" | "disengaged" | null;
  facialStress: "high_facial_stress" | "moderate" | null;
  duchenneSmilesCount: number;
  // Body
  lean: "forward_lean" | "back_lean" | null;
  posture: string | null;
  nodCount: number;
  shakeCount: number;
  fidgetLevel: "low" | "moderate" | "high" | null;
  // Gaze
  screenEngagementPct: number;
  attentionLevel: "high" | "reduced" | null;
  distractionCount: number;
  // Alignment — only populated when incongruence signals exist; null = no data
  incongruenceLevel: "low" | "moderate" | "high" | null;
}

function computeVideoStats(signals: Signal[]): Record<string, VideoStats> {
  const speakers = [...new Set(signals.filter((s) => s.speaker_label).map((s) => s.speaker_label!))];
  const result: Record<string, VideoStats> = {};

  for (const label of speakers) {
    const sigs = signals.filter((s) => s.speaker_label === label);
    const video = sigs.filter((s) => s.agent === "video");
    const fusion = sigs.filter((s) => s.agent === "fusion");

    if (video.length === 0) continue;

    // ── Face ──────────────────────────────────────────────────────────

    // Dominant non-neutral emotion (weighted by confidence × value)
    const emotionScore = new Map<string, number>();
    const emotionIntensity = new Map<string, number>();
    for (const s of video.filter((s) => s.signal_type === "facial_emotion" && s.value_text)) {
      const e = s.value_text!;
      if (e === "neutral") continue;
      const weight = (s.confidence ?? 0.5) * (s.value ?? 0.5);
      emotionScore.set(e, (emotionScore.get(e) ?? 0) + weight);
      emotionIntensity.set(e, Math.max(emotionIntensity.get(e) ?? 0, s.value ?? 0));
    }
    let dominantEmotion: string | null = null;
    let dominantEmotionIntensity = 0;
    let maxScore = 0;
    for (const [e, sc] of emotionScore) {
      if (sc > maxScore) { dominantEmotion = e; maxScore = sc; dominantEmotionIntensity = emotionIntensity.get(e) ?? 0; }
    }

    // Facial engagement: pick highest-confidence window
    const engSigs = video.filter((s) => s.signal_type === "facial_engagement" && s.value_text);
    let facialEngagement: VideoStats["facialEngagement"] = null;
    if (engSigs.length > 0) {
      const top = engSigs.reduce((a, b) => (b.confidence ?? 0) > (a.confidence ?? 0) ? b : a);
      if (top.value_text === "high_engagement") facialEngagement = "high_engagement";
      else if (top.value_text === "low_engagement" || top.value_text === "disengaged") facialEngagement = "disengaged";
    }

    // Facial stress level
    const stressSigs = video.filter((s) => s.signal_type === "facial_stress" && s.value_text);
    let facialStress: VideoStats["facialStress"] = null;
    if (stressSigs.length > 0) {
      const hasHigh = stressSigs.some((s) => s.value_text === "high_facial_stress");
      facialStress = hasHigh ? "high_facial_stress" : "moderate";
    }

    // Duchenne (genuine) smiles only
    const duchenneSmilesCount = video.filter(
      (s) => s.signal_type === "smile_type" && s.value_text === "duchenne"
    ).length;

    // ── Body ──────────────────────────────────────────────────────────

    // Lean direction (highest-confidence signal)
    const leanSigs = video.filter((s) => s.signal_type === "body_lean" && s.value_text);
    let lean: VideoStats["lean"] = null;
    if (leanSigs.length > 0) {
      const top = leanSigs.reduce((a, b) => (b.confidence ?? 0) > (a.confidence ?? 0) ? b : a);
      lean = (top.value_text as VideoStats["lean"]) ?? null;
    }

    // Posture (most frequent non-upright state)
    const postureFreq = new Map<string, number>();
    for (const s of video.filter((s) => s.signal_type === "posture" && s.value_text)) {
      const p = s.value_text!;
      if (p === "upright") continue;
      postureFreq.set(p, (postureFreq.get(p) ?? 0) + 1);
    }
    let posture: string | null = null;
    let maxPFreq = 0;
    for (const [p, f] of postureFreq) { if (f > maxPFreq) { posture = p; maxPFreq = f; } }

    const nodCount = video.filter((s) => s.signal_type === "head_nod").length;
    const shakeCount = video.filter((s) => s.signal_type === "head_shake").length;

    // Fidget level from actual values
    const fidgetVals = video
      .filter((s) => s.signal_type === "body_fidgeting" && s.value != null)
      .map((s) => s.value!);
    const avgFidget = fidgetVals.length > 0
      ? fidgetVals.reduce((a, b) => a + b, 0) / fidgetVals.length : -1;
    const fidgetLevel: VideoStats["fidgetLevel"] =
      avgFidget < 0 ? null : avgFidget > 0.6 ? "high" : avgFidget > 0.3 ? "moderate" : "low";

    // ── Gaze ──────────────────────────────────────────────────────────

    // Screen engagement: only meaningful when contact was actually low.
    // The gaze rule only fires for low (<40%) or sustained high (>90%) contact.
    // High values (99%) just mean "looking at camera" — normal video-call behavior.
    // Show the chip only when low_screen_contact signals exist (the anomaly case).
    const lowContactSigs = video.filter(
      (s) => s.signal_type === "screen_contact" && s.value_text === "low_screen_contact" && s.value != null
    );
    const screenEngagementPct = lowContactSigs.length > 0
      ? Math.round((lowContactSigs.reduce((a, b) => a + (b.value ?? 0), 0) / lowContactSigs.length) * 100) : 0;

    // Attention level from attention_level signals
    const attSigs = video.filter((s) => s.signal_type === "attention_level" && s.value_text);
    let attentionLevel: VideoStats["attentionLevel"] = null;
    if (attSigs.length > 0) {
      const top = attSigs.reduce((a, b) => (b.confidence ?? 0) > (a.confidence ?? 0) ? b : a);
      if (top.value_text?.includes("high")) attentionLevel = "high";
      else if (top.value_text?.includes("reduced") || top.value_text?.includes("low")) attentionLevel = "reduced";
    }

    const distractionCount = video.filter((s) => s.signal_type === "sustained_distraction").length;

    // ── Alignment — real data only ─────────────────────────────────────
    const incongSigs = fusion.filter((s) =>
      ["tone_face_masking", "smile_sentiment_incongruence", "stress_suppression",
       "head_body_incongruence", "verbal_incongruence", "voice_face_alignment"].includes(s.signal_type)
      // voice_face_alignment: only count mismatch value_texts, not "congruent"
      && !(s.signal_type === "voice_face_alignment" && s.value_text === "congruent")
    );
    let incongruenceLevel: VideoStats["incongruenceLevel"] = null;
    if (incongSigs.length > 0) {
      const avgIncon =
        incongSigs.reduce((a, s) => a + (s.value ?? 0) * (s.confidence ?? 0.5), 0) / incongSigs.length;
      incongruenceLevel = avgIncon > 0.6 ? "high" : avgIncon > 0.35 ? "moderate" : "low";
    }

    result[label] = {
      dominantEmotion, dominantEmotionIntensity, facialEngagement, facialStress, duchenneSmilesCount,
      lean, posture, nodCount, shakeCount, fidgetLevel,
      screenEngagementPct, attentionLevel, distractionCount,
      incongruenceLevel,
    };
  }

  return result;
}

// ── Call Outcome ──

function computeCallOutcome(speakerStats: SpeakerStats[]) {
  const totalBuying = speakerStats.reduce((a, s) => a + s.buyingSignalCount, 0);
  const totalObjections = speakerStats.reduce((a, s) => a + s.objectionCount, 0);
  const avgSentiment =
    speakerStats.length > 0
      ? speakerStats.reduce((a, s) => a + s.avgSentiment, 0) / speakerStats.length
      : 0;

  let outcome: "Positive" | "Neutral" | "Negative" = "Neutral";
  let outcomeColor = getCSSVar("--stress-med") || "#F59E0B";
  if (totalBuying >= 2 && totalObjections === 0) {
    outcome = "Positive";
    outcomeColor = getCSSVar("--stress-low") || "#22C55E";
  } else if (totalBuying >= 1 && avgSentiment > 0) {
    outcome = "Positive";
    outcomeColor = getCSSVar("--stress-low") || "#22C55E";
  } else if (totalObjections >= 3 || avgSentiment < -0.3) {
    outcome = "Negative";
    outcomeColor = getCSSVar("--stress-high") || "#EF4444";
  }

  // Decision readiness
  let readinessLabel = "Uncertain";
  let readinessColor = getCSSVar("--stress-med") || "#F59E0B";
  if (totalBuying >= 2) {
    readinessLabel = "Ready";
    readinessColor = getCSSVar("--stress-low") || "#22C55E";
  } else if (totalObjections > totalBuying) {
    readinessLabel = "Not Ready";
    readinessColor = getCSSVar("--stress-high") || "#EF4444";
  }

  // Objection handled
  let objHandledLabel = "N/A";
  let objHandledColor = getCSSVar("--text-secondary") || "#8B93A7";
  if (totalObjections > 0) {
    if (totalBuying > totalObjections) {
      objHandledLabel = "Yes";
      objHandledColor = getCSSVar("--stress-low") || "#22C55E";
    } else if (totalBuying > 0) {
      objHandledLabel = "Partially";
      objHandledColor = getCSSVar("--stress-med") || "#F59E0B";
    } else {
      objHandledLabel = "No";
      objHandledColor = getCSSVar("--stress-high") || "#EF4444";
    }
  }

  return {
    outcome, outcomeColor,
    readinessLabel, readinessColor,
    objHandledLabel, objHandledColor,
    totalBuying, totalObjections,
  };
}

// ── Infer Speaker Roles ──

function inferSpeakerRoles(
  speakerStats: SpeakerStats[],
  meetingType: string,
  segments: TranscriptSegment[] = []
): Record<string, string> {
  const roles: Record<string, string> = {};
  if (meetingType !== "sales_call" || speakerStats.length < 2) return roles;

  // Strategy 1: Check who introduces themselves in the first 6 segments.
  // The speaker who says "calling from", "my name is X from", "this is X from"
  // is the Seller. The other is the Prospect.
  const SELLER_PATTERNS = [
    /calling (?:you )?from/i,
    /this is .{1,30} from/i,
    /my name is .{1,30} from/i,
    /i['']m .{1,20} (?:calling|reaching out)/i,
    /we(?:['']re| are) a .{1,40} company/i,
    /quick call to see if/i,
    /wanted to (?:talk|reach|connect|check)/i,
  ];

  const early = segments.slice(0, 8);
  let sellerLabel: string | null = null;
  for (const seg of early) {
    const text = seg.text || "";
    const speaker = seg.speaker_label;
    if (!speaker) continue;
    for (const pat of SELLER_PATTERNS) {
      if (pat.test(text)) {
        sellerLabel = speaker;
        break;
      }
    }
    if (sellerLabel) break;
  }

  if (sellerLabel) {
    const labels = speakerStats.map((s) => s.label);
    roles[sellerLabel] = "Seller";
    const prospect = labels.find((l) => l !== sellerLabel);
    if (prospect) roles[prospect] = "Prospect";
    return roles;
  }

  // Strategy 2 (fallback): Score by signals.
  // Objections strongly indicate Prospect. Talks-more indicates Seller.
  const sorted = [...speakerStats].sort((a, b) => {
    const aScore =
      (a.objectionCount ?? 0) * 5 -
      a.avgPower * 3 +
      a.avgStress * 2;
    const bScore =
      (b.objectionCount ?? 0) * 5 -
      b.avgPower * 3 +
      b.avgStress * 2;
    return bScore - aScore;
  });

  roles[sorted[0].label] = "Prospect";
  roles[sorted[1].label] = "Seller";

  return roles;
}

// ── Fusion Signal Display ──

const FUSION_SIGNAL_LABELS: Record<string, { label: string; icon: string }> = {
  // Phase 1 (audio-only)
  stress_sentiment_incongruence: { label: "Credibility Concern",         icon: "🔴" },
  credibility_assessment:        { label: "Credibility Concern",         icon: "🔴" },
  verbal_incongruence:           { label: "Verbal Mismatch",             icon: "⚠️" },
  urgency_authenticity:          { label: "Urgency Pattern",             icon: "⚠️" },
  // Phase 2E (audio × video)
  tone_face_masking:             { label: "Voice-Face Masking",          icon: "🎭" },
  stress_suppression:            { label: "Stress Suppression",          icon: "😬" },
  cognitive_load:                { label: "Cognitive Overload",          icon: "🧠" },
  nonverbal_disagreement:        { label: "Nonverbal Disagreement",      icon: "👎" },
  physical_engagement:           { label: "Physical Engagement",         icon: "🙌" },
  false_confidence:              { label: "False Confidence",            icon: "🎯" },
  smile_sentiment_incongruence:  { label: "Smile Masks Sentiment",       icon: "😊⚠️" },
  processing_load:               { label: "Processing Load",             icon: "⏳" },
  dominance_anxiety:             { label: "Dominance Anxiety",           icon: "😰" },
  interrupt_intent:              { label: "Interrupt Intent",            icon: "✋" },
  rapport_confirmation:          { label: "Rapport Confirmed",           icon: "🤝" },
};

const FUSION_VALUE_LABELS: Record<string, string> = {
  // Phase 1
  credibility_concern:            "Content contradicts vocal stress patterns",
  mild_incongruence:              "Slight mismatch between verbal content and vocal indicators",
  strong_verbal_incongruence:     "Positive sentiment expressed with heavy hedging",
  moderate_verbal_incongruence:   "Agreement with notable hedging language",
  mild_verbal_incongruence:       "Mild verbal hedging alongside positive sentiment",
  hedged_agreement:               "Agreement language with underlying uncertainty markers",
  incongruence_with_objection:    "Positive sentiment combined with hidden objection markers",
  manufactured_urgency:           "Fast-paced persuasion with concurrent stress indicators",
  authentic_urgency:              "Persuasive language supported by confident vocal patterns",
  ambiguous_urgency:              "Urgency pattern with mixed vocal signals",
  // Phase 2E
  strong_masking:                 "Voice tone and facial emotion strongly contradict each other",
  moderate_masking:               "Facial expression partially contradicts voice tone",
  mild_masking:                   "Slight misalignment between voice and face",
  corroborated_stress:            "Both voice and face confirm elevated stress",
  stress_suppression:             "Stress visible in one channel but suppressed in the other",
  high_cognitive_load:            "Filler words and gaze breaks co-occurring — high cognitive demand",
  moderate_cognitive_load:        "Some filler spikes with gaze breaks indicating mental effort",
  mild_cognitive_load:            "Minor signs of cognitive processing",
  explicit_disagreement:          "Head shake co-occurs with objection language — clear disagreement",
  polite_disagreement:            "Head nod with objection language — polite but resistant",
  high_engagement:                "Forward lean and strong visual attention — fully engaged",
  disengagement:                  "Backward lean with low attention — disengaging",
  body_engaged_mind_elsewhere:    "Physical presence but attention is elsewhere",
  low_confidence_detected:        "Gaze breaks align with hedging language — low genuine confidence",
  mild_uncertainty:               "Some gaze avoidance alongside hedged statements",
  hedged_statement:               "Hedged language with mild gaze avoidance",
  emotion_masking:                "Social smile co-occurring with negative sentiment — masking displeasure",
  possible_sarcasm:               "Genuine smile with negative sentiment — possible sarcasm or irony",
  high_processing_load:           "Long response latency with facial stress — overwhelmed",
  elevated_processing_load:       "Slightly delayed response with visible facial tension",
  mild_processing_load:           "Minor processing delay with mild facial stress",
  dominance_anxiety:              "Dominant language but gaze avoidance — anxiety under dominant facade",
  mild_dominance_anxiety:         "Dominant tone with occasional gaze breaks",
  dominance_with_uncertainty:     "Assertive language with mild nonverbal uncertainty",
  competitive_interrupt:          "Forward lean during interruption — assertive intent",
  reactive_interrupt:             "Backward lean during interruption — defensive or confused",
  strong_rapport:                 "Empathy language and head nods strongly aligned — deep rapport",
  building_rapport:               "Empathy and nodding building connection",
  rapport_indicator:              "Mild rapport signal from verbal-nonverbal alignment",
};

// ── Main Component ──

type TabKey = "transcript" | "insights" | "report" | "chat";

export default function SessionDetail() {
  const { id } = useParams<{ id: string }>();
  const [activeTab, setActiveTab] = useState<TabKey>("transcript");
  const [transcriptViewMode, setTranscriptViewMode] = useState<"list" | "chat">("list");
  const [showConvoGraph, setShowConvoGraph] = useState(false);

  const { data: detail, isLoading: loadingDetail } = useQuery({
    queryKey: ["session", id],
    queryFn: () => getSession(id!),
    enabled: !!id,
  });

  const { data: signalData } = useQuery({
    queryKey: ["signals", id],
    queryFn: () => getSignals(id!, { limit: 50000 }),
    enabled: !!id,
  });

  const { data: transcriptData } = useQuery({
    queryKey: ["transcript", id],
    queryFn: () => getTranscript(id!),
    enabled: !!id,
  });

  const { data: reportData } = useQuery({
    queryKey: ["report", id],
    queryFn: () => getReport(id!),
    enabled: !!id && detail?.has_report === true,
  });

  const { data: videoSignalData } = useQuery({
    queryKey: ["video-signals", id],
    queryFn: () => getVideoSignals(id!),
    enabled: !!id && isVideoFile(detail?.session?.media_url),
    // Poll every 30 s until Face/Body/Gaze signals arrive, then stop
    refetchInterval: (query) => {
      const d = query.state.data as { signals?: { agent: string }[] } | undefined;
      return d?.signals?.some((s) => s.agent === "video") ? false : 30_000;
    },
  });


  // useMemo hooks must be declared before any early returns (React rules of hooks)
  const speakerBaselines = useMemo(() => {
    const rawSignals = signalData?.signals ?? [];
    const buckets: Record<string, number[]> = {};
    for (const s of rawSignals) {
      if (s.signal_type === "vocal_stress_score" && s.speaker_id) {
        (buckets[s.speaker_id] ??= []).push(s.value ?? 0);
      }
    }
    const result: Record<string, number> = {};
    for (const [spk, vals] of Object.entries(buckets)) {
      result[spk] = vals.reduce((a, b) => a + b, 0) / vals.length;
    }
    return result;
  }, [signalData]);

  if (loadingDetail) {
    return (
      <div className="flex items-center justify-center py-20 text-sm text-nexus-text-muted">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading session...
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="py-20 text-center text-sm text-nexus-text-muted">
        Session not found
      </div>
    );
  }

  const { session, alerts } = detail;
  const signals = signalData?.signals ?? [];
  const segments = transcriptData?.segments ?? [];
  const report = reportData?.report;
  const content = report?.content;

  // Compute analytics from signals
  const speakerStats = computeSpeakerStats(signals);
  const videoStats = computeVideoStats(signals);
  const hasVideoSignals = signals.some((s) => s.agent === "video");
  const callOutcome = computeCallOutcome(speakerStats);

  // Infer speaker roles (uses transcript to detect who introduces themselves)
  const speakerRoles = inferSpeakerRoles(speakerStats, session.meeting_type, segments);

  // Build speaker name map from entity extraction (Speaker_0 → "Rita")
  const speakerNames: Record<string, string> = (() => {
    const names: Record<string, string> = {};
    const people = (content?.entities as any)?.people as Array<{ name: string; role: string; speaker_label: string }> | undefined;
    if (!people) return names;
    const byLabel: Record<string, Array<{ name: string; role: string }>> = {};
    for (const p of people) {
      if (p.speaker_label) {
        (byLabel[p.speaker_label] ||= []).push(p);
      }
    }
    for (const [label, candidates] of Object.entries(byLabel)) {
      const best = candidates.find((c) => c.role && c.role.toLowerCase() !== "participant") || candidates[0];
      if (best) names[label] = best.name;
    }
    return names;
  })();

  // Helper: get display name for a speaker label
  const displayName = (label: string | null | undefined): string => {
    if (!label) return "Unknown";
    return speakerNames[label] || label;
  };

  // Mirrors SIGNAL_TO_CATEGORY in VideoSignalPlayer.tsx
  const SIGNAL_TO_CAT: Record<string, string> = {
    vocal_stress_score: "Stressed", facial_stress: "Stressed",
    shoulder_tension: "Stressed", freezing_response: "Stressed",
    agitated_high_arousal_tone: "Stressed", tension_cluster: "Stressed",
    stress_anxiety_cluster: "Stressed",
    emotional_suppression: "Guarded", hidden_disagreement: "Guarded",
    lip_pursing: "Guarded", motor_inhibition: "Guarded",
    arms_crossed: "Guarded", self_touch: "Guarded", face_region_touch: "Guarded",
    head_nod: "Engaged", smile_type: "Engaged", body_lean: "Engaged",
    buying_signal: "Engaged", facial_engagement: "Engaged",
    attention_level: "Engaged", genuine_engagement: "Engaged",
    gesture_animation: "Engaged", laughter: "Engaged",
    gaze_direction_shift: "Deflecting", sustained_distraction: "Deflecting",
    topic_shift: "Deflecting", active_disengagement: "Deflecting",
    blink_rate_anomaly: "Deflecting",
    head_shake: "Resistant", objection_signal: "Resistant",
    conflict_detection: "Resistant", frustration_cluster: "Resistant",
    resistance_hardening: "Resistant", head_body_incongruence: "Resistant",
    pause_classification: "Processing", strategic_pause: "Processing",
    evaluation_cluster: "Processing", cognitive_overload: "Processing",
    evidence_response_processing_delay: "Processing", decision_engagement: "Processing",
    verbal_uncertainty_cluster: "Processing",
    interruption_event: "Dominant", dominance_display: "Dominant",
    dominance_score: "Dominant", arm_posture: "Dominant",
    finger_steepling: "Dominant", peak_performance: "Dominant",
  };

  function getActiveCategories(segSignals: Signal[], speakerId: string | null): { categories: string[]; stressRatio: number } {
    const baseline = speakerId ? (speakerBaselines[speakerId] ?? 0.25) : 0.25;
    const stress = segSignals.find((s) => s.signal_type === "vocal_stress_score");
    const stressRatio = stress ? (stress.value ?? 0) / Math.max(baseline, 0.01) : 1.0;
    const seen = new Set<string>();
    const categories: string[] = [];
    for (const s of segSignals) {
      if ((s.confidence ?? 0) < 0.30) continue;
      if (s.signal_type === "presence_detected") continue;
      const cat = SIGNAL_TO_CAT[s.signal_type];
      if (!cat) continue;
      if (cat === "Stressed" && s.signal_type === "vocal_stress_score") {
        if ((s.value ?? 0) < baseline * 1.5) continue;
      }
      if (s.signal_type === "body_lean" && s.value_text !== "forward_lean") continue;
      if (!seen.has(cat)) { seen.add(cat); categories.push(cat); }
    }
    return { categories, stressRatio };
  }

  function phaseToColor(phase: string): string {
    const map: Record<string, string> = {
      Stressed: "#EF4444", Resistant: "#F97316", Guarded: "#F59E0B",
      Engaged: "#22C55E", Deflecting: "#A855F7", Processing: "#3B82F6",
      Dominant: "#EC4899",
    };
    return map[phase] ?? "#6B7280";
  }

  // Extract fusion signals
  const fusionSignals = signals.filter((s) => s.agent === "fusion");

  // Theme-aware speaker colors
  const speakerColors = [
    getCSSVar("--accent-blue") || "#4F8BFF",
    getCSSVar("--accent-purple") || "#8B5CF6",
    getCSSVar("--stress-med") || "#F59E0B",
    getCSSVar("--engagement") || "#10B981",
    getCSSVar("--agent-gaze") || "#EC4899",
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      {/* 1. HEADER */}
      <div>
        <Link
          to="/sessions"
          className="mb-3 inline-flex items-center gap-1 text-xs text-nexus-text-muted hover:text-nexus-accent-blue"
        >
          <ArrowLeft className="h-3 w-3" />
          Back to Sessions
        </Link>

        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 className="text-lg font-semibold text-nexus-text-primary">
              {session.title || "Untitled Session"}
            </h1>
            <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-nexus-text-muted">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatDuration(session.duration_ms)}
              </span>
              <span className="flex items-center gap-1">
                <Users className="h-3 w-3" />
                {session.speaker_count ?? "--"} audio speakers
                {session.participant_count != null &&
                  session.participant_count !== session.speaker_count && (
                    <span className="text-nexus-accent-blue">
                      · {session.participant_count} visible
                    </span>
                )}
              </span>
              <span>
                {format(new Date(session.created_at), "MMM d, yyyy 'at' h:mm a")}
              </span>
              {detail.alert_count > 0 && (
                <span className="flex items-center gap-1 text-nexus-alert">
                  <AlertTriangle className="h-3 w-3" />
                  {detail.alert_count} alert{detail.alert_count !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          </div>

          {detail.has_report && (
            <Link
              to={`/sessions/${id}/report`}
              className="self-start flex items-center gap-1.5 rounded bg-accent-purple-20 px-3 py-1.5 text-xs font-medium text-nexus-accent-purple transition-colors hover:bg-accent-purple-30"
            >
              <FileText className="h-3.5 w-3.5" />
              View Report
            </Link>
          )}
        </div>

        {/* Signal count summary */}
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(detail.signals_by_agent).map(([agent, count]) => {
            const agentColors: Record<string, string> = {
              voice: "var(--agent-voice)",
              language: "var(--agent-language)",
              fusion: "var(--agent-fusion)",
              conversation: "var(--accent-blue, #4F8BFF)",
              video: "var(--agent-gaze, #EC4899)",
            };
            return (
              <span
                key={agent}
                className="inline-flex items-center gap-1.5 rounded bg-nexus-surface px-2 py-0.5 text-[10px] font-mono text-nexus-text-secondary"
              >
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ background: agentColors[agent] || "var(--neutral)" }}
                />
                {agent}: {count}
              </span>
            );
          })}
          <span className="rounded bg-nexus-surface px-2 py-0.5 text-[10px] font-mono text-nexus-text-secondary">
            {detail.signal_count} signals
          </span>
        </div>
      </div>

      {/* TAB BAR */}
      <div className="flex gap-1 rounded-lg bg-nexus-surface p-1 border border-nexus-border">
        {([
          { key: "transcript" as TabKey, label: "Transcript", icon: "📝" },
          { key: "insights" as TabKey, label: "Insights", icon: "💡" },
          { key: "report" as TabKey, label: "Report", icon: "📊" },
          { key: "chat" as TabKey, label: "Chat", icon: "💬" },
        ]).map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex-1 rounded-md px-3 py-2 text-xs font-medium transition-colors ${
              activeTab === tab.key
                ? "bg-nexus-surface-hover text-nexus-text-primary shadow-sm"
                : "text-nexus-text-muted hover:text-nexus-text-secondary"
            }`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* ═══ TRANSCRIPT TAB ═══ */}
      {activeTab === "transcript" && (<>

      {/* 3. CALL OUTCOME (sales_call only) */}
      {session.meeting_type === "sales_call" && speakerStats.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {/* Estimated Outcome */}
          <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 text-center">
            <div className="text-[11px] text-nexus-text-secondary mb-2">
              Estimated Outcome
            </div>
            <div className="flex items-center justify-center gap-2">
              {callOutcome.outcome === "Positive" ? (
                <CheckCircle className="h-5 w-5" style={{ color: callOutcome.outcomeColor }} />
              ) : callOutcome.outcome === "Negative" ? (
                <XCircle className="h-5 w-5" style={{ color: callOutcome.outcomeColor }} />
              ) : (
                <TrendingUp className="h-5 w-5" style={{ color: callOutcome.outcomeColor }} />
              )}
              <span className="text-xl font-bold" style={{ color: callOutcome.outcomeColor }}>
                {callOutcome.outcome}
              </span>
            </div>
          </div>

          {/* Decision Readiness */}
          <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 text-center">
            <div className="text-[11px] text-nexus-text-secondary mb-2">
              Decision Readiness
            </div>
            <div className="flex items-center justify-center gap-2">
              <ShieldCheck className="h-5 w-5" style={{ color: callOutcome.readinessColor }} />
              <span className="text-xl font-bold" style={{ color: callOutcome.readinessColor }}>
                {callOutcome.readinessLabel}
              </span>
            </div>
          </div>

          {/* Objection Handled */}
          <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 text-center">
            <div className="text-[11px] text-nexus-text-secondary mb-2">
              Objection Handled
            </div>
            <div className="flex items-center justify-center gap-2">
              <MessageSquare className="h-5 w-5" style={{ color: callOutcome.objHandledColor }} />
              <span className="text-xl font-bold" style={{ color: callOutcome.objHandledColor }}>
                {callOutcome.objHandledLabel}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Signal Explorer — all meeting types */}
      {signals.length > 0 && (
        <SignalExplorer
          signals={signals}
          signalsByAgent={detail.signals_by_agent}
          totalCount={detail.signal_count}
          speakerRoles={speakerRoles}
        />
      )}

      {/* 3b. BEHAVIORAL OVERVIEW (shows when video or fusion signals present) */}
      <BehavioralOverview
        signals={signals}
        meetingType={session.meeting_type}
        durationMs={session.duration_ms || 0}
      />

      {/* 4. SPEAKER ANALYSIS + STRESS TIMELINE (two-column) */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* LEFT: Speaker Analysis Cards */}
        <div className="lg:col-span-3 space-y-4">
          {speakerStats.length > 0 && (
            <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
              <h3 className="mb-4 text-sm font-medium text-nexus-text-primary">
                Speaker Analysis
              </h3>
              <div className="space-y-5">
                {speakerStats.map((speaker, i) => {
                  const color = speakerColors[i % speakerColors.length];
                  const role = speakerRoles[speaker.label];
                  const stressPct = Math.round(speaker.avgStress * 100);
                  const sentPct = Math.round(Math.max(0, (speaker.avgSentiment + 1) / 2 * 100));
                  const powerPct = Math.round(speaker.avgPower * 100);
                  const confPct = Math.round(speaker.avgConfidence * 100);

                  const stressColor = stressPct > 60
                    ? (getCSSVar("--stress-high") || "#EF4444")
                    : stressPct > 35
                    ? (getCSSVar("--stress-med") || "#F59E0B")
                    : (getCSSVar("--stress-low") || "#22C55E");
                  const sentColor = sentPct > 55
                    ? (getCSSVar("--stress-low") || "#22C55E")
                    : sentPct < 45
                    ? (getCSSVar("--stress-high") || "#EF4444")
                    : (getCSSVar("--stress-med") || "#F59E0B");

                  return (
                    <div key={speaker.label}>
                      <div className="flex items-center gap-2 mb-3">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ background: color }}
                        />
                        <span className="text-sm font-semibold text-nexus-text-primary">
                          {displayName(speaker.label)}
                        </span>
                        <span className="text-xs text-nexus-text-muted">
                          ({speaker.label}{role ? ` · ${role}` : ""})
                        </span>
                      </div>

                      <GaugeBar label="Stress" pct={stressPct} color={stressColor} />
                      <GaugeBar label="Sentiment" pct={sentPct} color={sentColor} />
                      <GaugeBar label="Power" pct={powerPct} color={getCSSVar("--accent-purple") || "#8B5CF6"} />
                      <GaugeBar label="Confidence" pct={confPct} color={getCSSVar("--accent-blue") || "#4F8BFF"} />

                      <div className="mt-2 flex flex-wrap gap-1.5">
                        <StatChip label="Tone" value={speaker.dominantTone} />
                        <StatChip label="Fillers" value={speaker.fillerCount} color={getCSSVar("--stress-med") || "#F59E0B"} />
                        {speaker.buyingSignalCount > 0 && (
                          <StatChip label="Buying" value={speaker.buyingSignalCount} color={getCSSVar("--stress-low") || "#22C55E"} />
                        )}
                        {speaker.objectionCount > 0 && (
                          <StatChip label="Objections" value={speaker.objectionCount} color={getCSSVar("--stress-high") || "#EF4444"} />
                        )}
                        {/* Conversation agent: dominance & engagement per speaker */}
                        {(() => {
                          const domSig = signals.find(
                            (s) => s.agent === "conversation" && s.signal_type === "dominance_score" && s.speaker_label === speaker.label
                          );
                          const engSig = signals.find(
                            (s) => s.agent === "conversation" && s.signal_type === "conversation_engagement" && s.speaker_label === speaker.label
                          );
                          return (
                            <>
                              {domSig && (
                                <StatChip
                                  label="Dominance"
                                  value={(domSig.value_text || "").replace(/_/g, " ")}
                                  color={
                                    domSig.value_text === "dominant"
                                      ? (getCSSVar("--stress-high") || "#EF4444")
                                      : domSig.value_text === "balanced"
                                      ? (getCSSVar("--stress-low") || "#22C55E")
                                      : (getCSSVar("--stress-med") || "#F59E0B")
                                  }
                                />
                              )}
                              {engSig && (
                                <StatChip
                                  label="Engagement"
                                  value={(engSig.value_text || "").replace(/_/g, " ")}
                                  color={
                                    engSig.value_text === "highly_engaged" || engSig.value_text === "engaged"
                                      ? (getCSSVar("--stress-low") || "#22C55E")
                                      : engSig.value_text === "passive"
                                      ? (getCSSVar("--stress-med") || "#F59E0B")
                                      : (getCSSVar("--stress-high") || "#EF4444")
                                  }
                                />
                              )}
                            </>
                          );
                        })()}
                      </div>

                      {/* VISUAL section — only shown when video signals exist for this speaker */}
                      {hasVideoSignals && videoStats[speaker.label] && (() => {
                        const vs = videoStats[speaker.label];
                        // Intensity-graded emoji: low → medium → high
                        const EMOTION_EMOJI: Record<string, [string, string, string]> = {
                          happy:     ["🙂 happy",      "😊 happy",     "😄 happy"],
                          sad:       ["😔 sad",         "😢 sad",       "😭 sad"],
                          angry:     ["😤 angry",       "😠 angry",     "🤬 angry"],
                          surprised: ["😮 surprised",   "😲 surprised", "🤯 surprised"],
                          disgusted: ["😒 disgusted",   "🤢 disgusted", "🤮 disgusted"],
                          contempt:  ["🙄 contempt",    "😏 contempt",  "😤 contempt"],
                          fearful:   ["😟 fearful",     "😨 fearful",   "😱 fearful"],
                        };
                        const EMOTION_COLOR: Record<string, string> = {
                          happy: "#22C55E", sad: "#3B82F6", angry: "#EF4444",
                          surprised: "#F59E0B", disgusted: "#6B7280",
                          contempt: "#EF4444", fearful: "#8B5CF6",
                        };
                        const emotionLabel = (emotion: string, intensity: number) => {
                          const tiers = EMOTION_EMOJI[emotion];
                          if (!tiers) return emotion;
                          if (intensity > 0.65) return tiers[2];
                          if (intensity > 0.35) return tiers[1];
                          return tiers[0];
                        };
                        return (
                          <div className="mt-3 rounded-md bg-nexus-surface-hover p-2.5">
                            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-nexus-text-muted">
                              Visual
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {/* Facial engagement */}
                              {vs.facialEngagement && (
                                <StatChip
                                  label="Engagement"
                                  value={vs.facialEngagement === "high_engagement" ? "engaged" : "disengaged"}
                                  color={vs.facialEngagement === "high_engagement" ? "#22C55E" : "#EF4444"}
                                />
                              )}
                              {/* Facial stress */}
                              {vs.facialStress && (
                                <StatChip
                                  label="Face stress"
                                  value={vs.facialStress === "high_facial_stress" ? "high" : "moderate"}
                                  color={vs.facialStress === "high_facial_stress" ? "#EF4444" : "#F59E0B"}
                                />
                              )}
                              {/* Dominant emotion */}
                              {vs.dominantEmotion && (
                                <StatChip
                                  label="Emotion"
                                  value={emotionLabel(vs.dominantEmotion, vs.dominantEmotionIntensity)}
                                  color={EMOTION_COLOR[vs.dominantEmotion] ?? "#8B5CF6"}
                                />
                              )}
                              {/* Duchenne smiles */}
                              {vs.duchenneSmilesCount > 0 && (
                                <StatChip label="Genuine smiles" value={vs.duchenneSmilesCount} color="#22C55E" />
                              )}
                              {/* Body lean */}
                              {vs.lean && (
                                <StatChip
                                  label="Lean"
                                  value={vs.lean === "forward_lean" ? "forward" : "back"}
                                  color={vs.lean === "forward_lean" ? "#22C55E" : "#F59E0B"}
                                />
                              )}
                              {/* Posture */}
                              {vs.posture && (
                                <StatChip label="Posture" value={vs.posture.replace(/_/g, " ")} color="#F59E0B" />
                              )}
                              {/* Head gestures */}
                              {vs.nodCount > 0 && (
                                <StatChip label="Nods" value={vs.nodCount} color="#4F8BFF" />
                              )}
                              {vs.shakeCount > 0 && (
                                <StatChip label="Shakes" value={vs.shakeCount} color="#F97316" />
                              )}
                              {/* Fidget */}
                              {vs.fidgetLevel && vs.fidgetLevel !== "low" && (
                                <StatChip label="Fidget" value={vs.fidgetLevel} color={vs.fidgetLevel === "high" ? "#EF4444" : "#F59E0B"} />
                              )}
                              {/* Gaze */}
                              {vs.screenEngagementPct > 0 && (
                                <StatChip label="Eye contact" value={`${vs.screenEngagementPct}%`} color={vs.screenEngagementPct >= 70 ? "#22C55E" : vs.screenEngagementPct >= 45 ? "#F59E0B" : "#EF4444"} />
                              )}
                              {vs.attentionLevel && (
                                <StatChip
                                  label="Attention"
                                  value={vs.attentionLevel}
                                  color={vs.attentionLevel === "high" ? "#22C55E" : "#EF4444"}
                                />
                              )}
                              {vs.distractionCount > 0 && (
                                <StatChip label="Distracted" value={`${vs.distractionCount}×`} color="#EF4444" />
                              )}
                              {/* Incongruence — only from real fusion signals */}
                              {vs.incongruenceLevel && (
                                <StatChip
                                  label="Mismatch"
                                  value={vs.incongruenceLevel}
                                  color={vs.incongruenceLevel === "high" ? "#EF4444" : vs.incongruenceLevel === "moderate" ? "#F59E0B" : "#22C55E"}
                                />
                              )}
                            </div>
                          </div>
                        );
                      })()}

                      {i < speakerStats.length - 1 && (
                        <div className="mt-4 border-b border-nexus-border" />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT: Stress Timeline */}
        <div className="lg:col-span-2">
          <StressTimeline signals={signals} speakerRoles={speakerRoles} speakerNames={speakerNames} />
        </div>
      </div>

      {/* 4b. CONVERSATION DYNAMICS SUMMARY */}
      {(() => {
        const convoSignals = signals.filter((s) => s.agent === "conversation");
        if (convoSignals.length === 0) return null;
        const turnTaking = convoSignals.find((s) => s.signal_type === "turn_taking_pattern");
        const rapportSig = convoSignals.find((s) => s.signal_type === "rapport_indicator");
        const balanceSig = convoSignals.find((s) => s.signal_type === "conversation_balance");
        return (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {turnTaking && (
              <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 text-center">
                <div className="text-[11px] text-nexus-text-secondary mb-2">Turn Rate</div>
                <div className="text-lg font-bold text-nexus-text-primary">
                  {turnTaking.value != null ? `${turnTaking.value.toFixed(1)}/min` : "--"}
                </div>
                <div className="text-[10px] text-nexus-text-muted mt-0.5">
                  {(turnTaking.value_text || "").replace(/_/g, " ")}
                </div>
              </div>
            )}
            {rapportSig && (
              <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 text-center">
                <div className="text-[11px] text-nexus-text-secondary mb-2">Rapport</div>
                <div className="text-lg font-bold" style={{
                  color: (rapportSig.value ?? 0) >= 0.65 ? "#22C55E" : (rapportSig.value ?? 0) >= 0.4 ? "#F59E0B" : "#EF4444"
                }}>
                  {rapportSig.value != null ? rapportSig.value.toFixed(2) : "--"}
                </div>
                <div className="text-[10px] text-nexus-text-muted mt-0.5">
                  {(rapportSig.value_text || "").replace(/_/g, " ")}
                </div>
              </div>
            )}
            {balanceSig && (
              <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4 text-center">
                <div className="text-[11px] text-nexus-text-secondary mb-2">Balance</div>
                <div className="text-lg font-bold" style={{
                  color: balanceSig.value_text === "well_balanced" ? "#22C55E" : balanceSig.value_text === "moderately_balanced" ? "#F59E0B" : "#EF4444"
                }}>
                  {(balanceSig.value_text || "").replace(/_/g, " ")}
                </div>
                <div className="text-[10px] text-nexus-text-muted mt-0.5">
                  index: {balanceSig.value != null ? balanceSig.value.toFixed(2) : "--"}
                </div>
              </div>
            )}
          </div>
        );
      })()}

      {/* 5. ALERTS & FUSION INSIGHTS */}
      <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
        <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-nexus-text-primary">
          <Zap className="h-4 w-4 text-nexus-alert" />
          Alerts & Fusion Insights
          <span className="ml-auto text-xs font-normal text-nexus-text-muted">
            {alerts.length} alert{alerts.length !== 1 ? "s" : ""}, {fusionSignals.length} fusion signal{fusionSignals.length !== 1 ? "s" : ""}
          </span>
        </h2>

        {alerts.length === 0 && fusionSignals.length === 0 ? (
          <p className="text-sm text-nexus-text-muted italic">
            No alerts detected in this session.
          </p>
        ) : (
          <div className="space-y-3">
            {/* Alerts */}
            {alerts.map((alert) => (
              <AlertCard key={alert.id} alert={alert} />
            ))}

            {/* Fusion signals */}
            {fusionSignals.map((fs, i) => {
              const sigConfig = FUSION_SIGNAL_LABELS[fs.signal_type] || {
                label: fs.signal_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
                icon: "🟠",
              };
              const valueDesc = FUSION_VALUE_LABELS[fs.value_text] || fs.value_text.replace(/_/g, " ");
              const speaker = displayName(fs.speaker_label);
              const role = speakerRoles[speaker];
              const speakerDisplay = role ? `${role} (${speaker})` : speaker;

              return (
                <div
                  key={`fusion-${i}`}
                  className="rounded-lg border-l-[3px] border-nexus-alert bg-nexus-surface-hover p-3"
                >
                  <div className="flex items-start gap-2">
                    <span className="text-base leading-none mt-0.5">{sigConfig.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 text-xs mb-0.5">
                        <span className="font-semibold text-nexus-alert">
                          {sigConfig.label}
                        </span>
                        <span className="text-nexus-text-muted">
                          {speakerDisplay}
                        </span>
                        <span className="ml-auto font-mono text-nexus-text-muted">
                          {formatTime(fs.window_start_ms)}–{formatTime(fs.window_end_ms)}
                        </span>
                      </div>
                      <p className="text-sm text-nexus-text-primary">
                        {valueDesc}
                      </p>
                      <p className="mt-0.5 text-xs text-nexus-text-muted">
                        Confidence: {(fs.confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* 5b. VIDEO PLAYER with signal overlay — audio-only sessions skip this */}
      {isVideoFile(session.media_url) && videoSignalData && (
        <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
          <h2 className="mb-3 text-sm font-medium text-nexus-text-secondary">
            Video Playback
          </h2>
          <VideoSignalPlayer
            sessionId={id!}
            signals={videoSignalData.signals}
            techniqueAnalysis={content?.technique_analysis}
          />
        </div>
      )}

      {/* 6. TRANSCRIPT with view toggle */}
      <div>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-medium text-nexus-text-secondary">
            Transcript
            {segments.length > 0 && (
              <span className="ml-2 text-nexus-text-muted">
                ({segments.length} segments)
              </span>
            )}
          </h2>
          {segments.length > 0 && (
            <div className="flex overflow-hidden rounded-full border border-nexus-border" style={{ height: 28 }}>
              <button
                onClick={() => setTranscriptViewMode("list")}
                className={`px-3 text-[11px] font-medium transition-colors ${
                  transcriptViewMode === "list"
                    ? "bg-blue-600 text-white"
                    : "bg-transparent text-nexus-text-muted hover:text-nexus-text-primary"
                }`}
              >
                List View
              </button>
              <button
                onClick={() => setTranscriptViewMode("chat")}
                className={`px-3 text-[11px] font-medium transition-colors ${
                  transcriptViewMode === "chat"
                    ? "bg-blue-600 text-white"
                    : "bg-transparent text-nexus-text-muted hover:text-nexus-text-primary"
                }`}
              >
                Chat View
              </button>
            </div>
          )}
        </div>

        {segments.length === 0 ? (
          <div className="flex h-48 items-center justify-center rounded-lg border border-nexus-border bg-nexus-surface text-sm text-nexus-text-muted">
            No transcript available
          </div>
        ) : transcriptViewMode === "list" ? (
          <div className="space-y-2 max-h-[700px] overflow-y-auto pr-1">
            {segments.map((segment) => {
              const segSigs = matchSignalsToSegment(segment, signals);
              const { categories, stressRatio } = getActiveCategories(segSigs, segment.speaker_label ?? null);
              return (
                <TranscriptBlock
                  key={segment.id}
                  segment={segment}
                  signals={segSigs}
                  speakerRole={segment.speaker_label ? speakerRoles[segment.speaker_label] : undefined}
                  speakerName={segment.speaker_label ? speakerNames[segment.speaker_label] : undefined}
                  categories={categories}
                  stressRatio={stressRatio}
                />
              );
            })}
          </div>
        ) : (
          <TranscriptView
            segments={segments}
            signals={signals}
            speakerRoles={speakerRoles}
            speakerNames={speakerNames}
            durationMs={session.duration_ms || 0}
          />
        )}
      </div>

      </>)}

      {/* ═══ INSIGHTS TAB ═══ */}
      {activeTab === "insights" && (
        <div className="space-y-6">
          {content?.entities?.topics || content?.key_paths ? (
            <>
              {/* Topic Timeline (full width) */}
              {content?.entities?.topics && content.entities.topics.length > 0 && (
                <TopicTimeline
                  topics={content.entities.topics}
                  signals={signals}
                  durationMs={session.duration_ms || 0}
                />
              )}

              {/* Swimlane Conversation Timeline (timeline only, no transcript toggle) */}
              {segments.length > 0 && (
                <SwimlaneTimeline
                  segments={segments}
                  signals={signals}
                  durationMs={session.duration_ms || 0}
                  entities={content?.entities || {}}
                  speakerRoles={speakerRoles}
                  hideTranscriptToggle
                />
              )}

              {/* SpeakerGraph + InsightPanel (two-column) */}
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
                <div className="lg:col-span-2">
                  <SpeakerGraph
                    speakers={(() => {
                      // Calculate real talk time from transcript segments
                      const talkMs: Record<string, number> = {};
                      for (const seg of segments) {
                        const spk = seg.speaker_label || "unknown";
                        talkMs[spk] = (talkMs[spk] || 0) + Math.max(0, (seg.end_ms || 0) - (seg.start_ms || 0));
                      }
                      const totalMs = Object.values(talkMs).reduce((a, b) => a + b, 0) || 1;
                      return speakerStats.map((s) => ({
                        ...s,
                        talkTimePct: ((talkMs[s.label] || 0) / totalMs) * 100,
                      }));
                    })()}
                    contentType={session.meeting_type}
                    entities={content?.entities || {}}
                    signals={signals}
                    speakerRoles={speakerRoles}
                  />
                </div>
                <div className="lg:col-span-3">
                  <InsightPanel
                    contentType={session.meeting_type}
                    entities={content?.entities || {}}
                    signals={signals}
                    speakers={(() => {
                      const talkMs: Record<string, number> = {};
                      for (const seg of segments) {
                        const spk = seg.speaker_label || "unknown";
                        talkMs[spk] = (talkMs[spk] || 0) + Math.max(0, (seg.end_ms || 0) - (seg.start_ms || 0));
                      }
                      const totalMs = Object.values(talkMs).reduce((a, b) => a + b, 0) || 1;
                      return speakerStats.map((s) => ({
                        ...s,
                        role: speakerRoles[s.label],
                        talkTimePct: ((talkMs[s.label] || 0) / totalMs) * 100,
                      }));
                    })()}
                    speakerRoles={speakerRoles}
                  />
                </div>
              </div>

              {/* Signal Chain Cards (full width) */}
              {content?.key_paths && content.key_paths.length > 0 && (
                <SignalChainCards keyPaths={content.key_paths} />
              )}

              {/* Graph Insights */}
              {content?.graph_analytics && (
                <GraphInsightsCard
                  analytics={content.graph_analytics as Record<string, unknown>}
                  speakerRoles={speakerRoles}
                  signals={signals}
                />
              )}

              {/* Conversation Graph toggle */}
              {!showConvoGraph ? (
                <button
                  onClick={() => setShowConvoGraph(true)}
                  className="w-full rounded-lg border border-dashed border-nexus-border bg-nexus-surface px-4 py-3 text-sm text-nexus-text-secondary hover:bg-nexus-surface-hover hover:text-nexus-text-primary transition-colors"
                >
                  Open Advanced Signal Node Graph
                </button>
              ) : (
                <ConversationGraph
                  segments={segments}
                  signals={signals}
                  entities={content?.entities || {}}
                  speakerRoles={speakerRoles}
                  durationMs={session.duration_ms || 0}
                  onClose={() => setShowConvoGraph(false)}
                  signalGraph={content?.signal_graph as any}
                />
              )}
            </>
          ) : (
            <div className="flex h-48 items-center justify-center rounded-lg border border-nexus-border bg-nexus-surface text-sm text-nexus-text-muted">
              Insights not available for this session. Re-analyse to generate.
            </div>
          )}
        </div>
      )}

      {/* ═══ REPORT TAB ═══ */}
      {activeTab === "report" && (
        <div className="space-y-6">

          {/* 1 — Executive Summary */}
          {content?.executive_summary ? (
            <section className="rounded-lg border border-accent-purple-30 bg-nexus-surface p-5">
              <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
                <Sparkles className="h-4 w-4" />
                Executive Summary
              </h2>
              <p className="text-sm leading-relaxed text-nexus-text-primary">
                {content.executive_summary}
              </p>
            </section>
          ) : report?.narrative ? (
            <section className="rounded-lg border border-accent-purple-30 bg-nexus-surface p-5">
              <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
                <Sparkles className="h-4 w-4" />
                Executive Summary
              </h2>
              <p className="text-sm leading-relaxed text-nexus-text-primary">
                {report.narrative}
              </p>
            </section>
          ) : null}

          {/* 2 — Key Facts & Commitments (all types) */}
          {content?.key_facts && (content.key_facts as any[]).length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-nexus-text-primary">
                <span>📋</span>
                Key Facts &amp; Commitments
              </h2>
              <div className="space-y-2">
                {(content.key_facts as any[]).map((fact: any, i: number) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 rounded border-l-2 bg-nexus-surface-hover p-2 text-xs"
                    style={{
                      borderLeftColor:
                        fact.type === "commitment" ? "var(--accent-blue)"
                        : fact.type === "objection" ? "var(--stress-high)"
                        : fact.type === "admission" ? "var(--stress-med)"
                        : "var(--text-muted)",
                    }}
                  >
                    <span className="mt-0.5 shrink-0">
                      {fact.type === "commitment" ? "✅"
                        : fact.type === "objection" ? "❌"
                        : fact.type === "admission" ? "⚠️"
                        : "📌"}
                    </span>
                    <div className="flex-1 min-w-0">
                      <span className="text-nexus-text-primary">{fact.text}</span>
                      <span className="ml-2 text-nexus-text-muted">
                        {fact.speaker && `— ${fact.speaker} `}
                        {fact.timestamp && `at ${fact.timestamp}`}
                        {fact.status && ` (${fact.status})`}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Notes (topic-grouped discussion details) */}
          {content?.notes && (content.notes as any[]).length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-4 text-sm font-semibold text-nexus-text-primary">
                Notes
              </h2>
              <div className="space-y-5">
                {(content.notes as any[]).map((topic: any, i: number) => (
                  <div key={i}>
                    <h3 className="text-sm font-semibold text-nexus-accent-purple mb-1">
                      {topic.topic}
                    </h3>
                    {topic.summary && (
                      <p className="text-xs text-nexus-text-muted mb-2 italic">{topic.summary}</p>
                    )}
                    <ul className="space-y-2 ml-2 border-l-2 border-nexus-border pl-3">
                      {(topic.details || []).map((detail: any, j: number) => (
                        <li key={j} className="text-sm text-nexus-text-primary">
                          <div className="flex items-start gap-2">
                            <span className="mt-0.5 text-nexus-accent-purple">→</span>
                            <div>
                              <span className="font-medium">{detail.speaker}</span>
                              {detail.timestamp && (
                                <span className="ml-1.5 text-[10px] text-nexus-text-muted">
                                  ({detail.timestamp})
                                </span>
                              )}
                              <span className="ml-1">{detail.text}</span>
                              {detail.sub_details?.length > 0 && (
                                <ul className="mt-1 ml-3 space-y-0.5">
                                  {detail.sub_details.map((sub: string, k: number) => (
                                    <li key={k} className="text-xs text-nexus-text-secondary flex items-start gap-1.5">
                                      <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-nexus-text-muted" />
                                      {sub}
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 3a — Statement Contradictions (all types) */}
          {content?.contradiction_analysis && (content.contradiction_analysis as any[]).length > 0 && (
            <section className="rounded-lg border border-amber-500/20 bg-nexus-surface p-5">
              <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-amber-400">
                <Zap className="h-4 w-4" />
                Statement Contradictions
              </h2>
              <div className="space-y-4">
                {(content.contradiction_analysis as any[]).map((c: any, i: number) => (
                  <div key={i} className="rounded border border-nexus-border p-3">
                    <div className="mb-2 text-xs font-medium text-nexus-text-muted">
                      {c.speaker} —{" "}
                      <span className="capitalize text-amber-400/80">
                        {(c.contradiction_type || "").replace(/_/g, " ")}
                      </span>
                    </div>
                    <div className="mb-2 grid grid-cols-2 gap-3">
                      <div className="rounded bg-nexus-surface-hover p-2">
                        <div className="text-[10px] text-nexus-text-muted">{c.statement_a?.timestamp}</div>
                        <p className="mt-0.5 text-sm text-nexus-text-primary">
                          &ldquo;{c.statement_a?.text}&rdquo;
                        </p>
                        <div className="mt-1 text-[10px]">
                          voice stress:{" "}
                          {c.statement_a?.voice_stress != null ? (
                            <span className={c.statement_a.voice_stress > 0.5 ? "text-red-400" : "text-green-400"}>
                              {c.statement_a.voice_stress.toFixed(2)}
                            </span>
                          ) : (
                            <span className="text-nexus-text-muted">N/A</span>
                          )}
                        </div>
                      </div>
                      <div className="rounded bg-nexus-surface-hover p-2">
                        <div className="text-[10px] text-nexus-text-muted">{c.statement_b?.timestamp}</div>
                        <p className="mt-0.5 text-sm text-nexus-text-primary">
                          &ldquo;{c.statement_b?.text}&rdquo;
                        </p>
                        <div className="mt-1 text-[10px]">
                          voice stress:{" "}
                          {c.statement_b?.voice_stress != null ? (
                            <span className={c.statement_b.voice_stress > 0.5 ? "text-red-400" : "text-green-400"}>
                              {c.statement_b.voice_stress.toFixed(2)}
                            </span>
                          ) : (
                            <span className="text-nexus-text-muted">N/A</span>
                          )}
                        </div>
                      </div>
                    </div>
                    {c.voice_delta && !c.voice_delta.includes("No voice stress data") && !c.voice_delta.includes("Voice stress data not available") && (
                      <p className="text-xs italic text-nexus-accent-purple">{c.voice_delta}</p>
                    )}
                    {c.significance && (
                      <p className="mt-1 text-xs text-nexus-text-secondary">{c.significance}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 3b — Voice Anomalies with Transcript Context (all types) */}
          {content?.voice_text_correlations && (content.voice_text_correlations as any[]).length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
                <MessageSquare className="h-4 w-4" />
                Voice Anomalies — What Was Being Said
              </h2>
              <div className="space-y-3">
                {(content.voice_text_correlations as any[]).map((v: any, i: number) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 rounded border-l-2 border-nexus-accent-purple bg-nexus-surface-hover p-2"
                  >
                    <div className="shrink-0 text-center">
                      <div className="font-mono text-xs text-nexus-text-muted">{v.timestamp}</div>
                      <div className="text-[10px] text-red-400">{v.anomaly_value}</div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="mb-0.5 text-xs font-medium text-nexus-text-muted">
                        {v.speaker} —{" "}
                        <span className="capitalize">
                          {(v.anomaly_type || "").replace(/_/g, " ")}
                        </span>
                      </div>
                      <p className="text-sm text-nexus-text-primary">
                        &ldquo;{v.transcript_text}&rdquo;
                      </p>
                      {v.context && (
                        <p className="mt-1 text-xs italic text-nexus-text-secondary">{v.context}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 3 — Risk Assessment (interrogation_video only) */}
          {session.meeting_type === "interrogation_video" && content?.risk_assessment && (
            <section className="rounded-lg border border-amber-500/30 bg-nexus-surface p-5">
              <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-amber-400">
                <span>⚖️</span>
                False Confession Risk Assessment
              </h2>
              {/* Risk gauge */}
              <div className="mb-3">
                <div className="flex items-center justify-between mb-1 text-xs">
                  <span className="font-semibold capitalize" style={{
                    color: (content.risk_assessment as any).risk_score < 0.3 ? "#10B981"
                      : (content.risk_assessment as any).risk_score < 0.55 ? "#F59E0B"
                      : (content.risk_assessment as any).risk_score < 0.8 ? "#F97316"
                      : "#EF4444",
                  }}>
                    {String((content.risk_assessment as any).false_confession_risk ?? "").replace(/_/g, " ")}
                  </span>
                  <span className="text-nexus-text-muted font-mono">
                    {((content.risk_assessment as any).risk_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-gray-700">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.min((content.risk_assessment as any).risk_score * 100, 100)}%`,
                      backgroundColor: (content.risk_assessment as any).risk_score < 0.3 ? "#10B981"
                        : (content.risk_assessment as any).risk_score < 0.55 ? "#F59E0B"
                        : (content.risk_assessment as any).risk_score < 0.8 ? "#F97316"
                        : "#EF4444",
                    }}
                  />
                </div>
              </div>
              {/* Contributing factors */}
              {(content.risk_assessment as any).contributing_factors?.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {(content.risk_assessment as any).contributing_factors.map((f: any, i: number) => (
                    <span
                      key={i}
                      className={`rounded px-2 py-1 text-[10px] leading-snug ${
                        f.present
                          ? "bg-red-900/60 text-red-200 border border-red-700/50"
                          : "bg-gray-700/60 text-gray-300 border border-gray-600/40"
                      }`}
                    >
                      <span className={f.present ? "text-red-400" : "text-gray-500"}>{f.present ? "✓" : "✗"}</span>
                      {" "}{f.factor.replace(/_/g, " ")}
                      {f.detail && <span className={`ml-1 ${f.present ? "text-red-300/80" : "text-gray-400"}`}>— {f.detail}</span>}
                    </span>
                  ))}
                </div>
              )}
              {/* Ethical note */}
              {(content.risk_assessment as any).ethical_note && (
                <p className="text-[10px] text-nexus-text-muted italic border-t border-nexus-border pt-2 mt-2">
                  {(content.risk_assessment as any).ethical_note}
                </p>
              )}
            </section>
          )}

          {/* 4 — Speaker Analyses (all types) */}
          {content?.speaker_analyses && Object.keys(content.speaker_analyses as any).length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-4 text-sm font-semibold text-nexus-text-primary">
                👤 Speaker Analysis
              </h2>
              <div className="space-y-4">
                {Object.entries(content.speaker_analyses as Record<string, any>).map(([spk, analysis]) => (
                  <div key={spk} className="border-l-2 border-accent-purple-40 pl-3">
                    <div className="text-xs font-semibold text-nexus-accent-purple mb-1">
                      {analysis.role ? `${spk} (${analysis.role})` : spk}
                    </div>
                    {analysis.behavioral_profile && (
                      <p className="text-sm text-nexus-text-primary mb-1">
                        {analysis.behavioral_profile}
                      </p>
                    )}
                    {analysis.voice_patterns && (
                      <p className="text-xs text-nexus-text-secondary mb-0.5">
                        🎙️ {analysis.voice_patterns}
                      </p>
                    )}
                    {analysis.body_language && (
                      <p className="text-xs text-nexus-text-secondary mb-0.5">
                        🧍 {analysis.body_language}
                      </p>
                    )}
                    {analysis.key_moments?.length > 0 && (
                      <div className="mt-1 space-y-0.5">
                        {analysis.key_moments.map((km: string, ki: number) => (
                          <p key={ki} className="text-[10px] text-nexus-text-muted font-mono">
                            • {km}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 5 — Key Moments */}
          {content?.key_moments && (content.key_moments as any[]).length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-nexus-text-primary">
                <Target className="h-4 w-4 text-nexus-accent-blue" />
                Key Moments
              </h2>
              <div className="space-y-4">
                {(content.key_moments as any[]).map((moment: any, i: number) => (
                  <div key={i} className="border-l-2 border-accent-blue-40 pl-3">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-accent-blue-15 font-mono text-[10px] font-bold text-nexus-accent-blue">
                        {i + 1}
                      </span>
                      {moment.time_description && (
                        <span className="font-mono text-nexus-text-muted">
                          {moment.time_description}
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-sm text-nexus-text-primary">{moment.description}</p>
                    {moment.significance && (
                      <p className="mt-0.5 text-xs text-nexus-text-secondary italic">
                        {moment.significance}
                      </p>
                    )}
                    {moment.signals_involved?.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {moment.signals_involved.map((sig: string, si: number) => (
                          <span key={si} className="rounded-full bg-accent-blue-15 px-1.5 py-0.5 text-[9px] text-nexus-accent-blue">
                            {sig.replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 6 — Cross-Modal Insights */}
          <section className="rounded-lg border-l-[3px] border-nexus-accent-purple bg-nexus-surface p-5">
            <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
              <Lightbulb className="h-4 w-4" />
              Cross-Modal Insights
            </h2>
            {content?.cross_modal_insights && (content.cross_modal_insights as any[]).length > 0 ? (
              <ul className="space-y-3">
                {(content.cross_modal_insights as any[]).map((insight: any, i: number) => {
                  const text = typeof insight === "string" ? insight : insight.insight;
                  const modalities: string[] = typeof insight === "object" ? (insight.modalities ?? []) : [];
                  const significance: string = typeof insight === "object" ? (insight.significance ?? "") : "";
                  return (
                    <li key={i} className="flex items-start gap-2 text-sm text-nexus-text-primary">
                      <span className="mt-1 font-bold text-nexus-accent-purple shrink-0">⚡</span>
                      <div>
                        <span>{text}</span>
                        {significance && (
                          <p className="mt-0.5 text-xs text-nexus-text-muted italic">{significance}</p>
                        )}
                        {modalities.length > 0 && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {modalities.map((m: string) => (
                              <span key={m} className="rounded-full bg-accent-purple-15 px-2 py-0.5 text-[10px] text-nexus-accent-purple">
                                {m}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="text-sm text-nexus-text-muted italic">
                Insufficient cross-modal data for insights in this session.
              </p>
            )}
          </section>

          {/* 7 — Contamination Timeline (interrogation_video only) */}
          {session.meeting_type === "interrogation_video" &&
            content?.contamination_timeline &&
            (content.contamination_timeline as any[]).length > 0 && (
            <section className="rounded-lg border border-red-900/40 bg-nexus-surface p-5">
              <h2 className="mb-3 text-sm font-semibold text-red-400">
                ⚠️ Contamination Timeline
              </h2>
              <div className="space-y-2">
                {(content.contamination_timeline as any[]).map((item: any, i: number) => (
                  <div key={i} className="rounded bg-nexus-surface-hover p-2 text-xs">
                    <span className="font-mono text-red-300">"{item.term}"</span>
                    <span className="ml-2 text-nexus-text-muted">
                      Interrogator at {item.interrogator_first} → Suspect adopted at {item.suspect_adopted}
                    </span>
                    {item.context && (
                      <p className="mt-0.5 text-nexus-text-muted italic">{item.context}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 8 — Technique Analysis (interrogation_video only) */}
          {session.meeting_type === "interrogation_video" && content?.technique_analysis && (
            <section className="rounded-lg border border-amber-900/30 bg-nexus-surface p-5">
              <h2 className="mb-3 text-sm font-semibold text-amber-400">
                🎭 Technique Analysis
              </h2>
              <div className="flex items-center gap-3 mb-2">
                <span className="rounded px-2 py-0.5 text-[11px] font-semibold uppercase"
                  style={{
                    backgroundColor: (content.technique_analysis as any).primary === "peace" ? "#10B98122"
                      : (content.technique_analysis as any).primary === "reid" ? "#F59E0B22"
                      : (content.technique_analysis as any).primary === "coercive" ? "#EF444422"
                      : "#6B728022",
                    color: (content.technique_analysis as any).primary === "peace" ? "#10B981"
                      : (content.technique_analysis as any).primary === "reid" ? "#F59E0B"
                      : (content.technique_analysis as any).primary === "coercive" ? "#EF4444"
                      : "#6B7280",
                  }}>
                  {(content.technique_analysis as any).primary}
                </span>
                {(content.technique_analysis as any).peace_markers > 0 && (
                  <span className="text-[10px] text-emerald-400">PEACE ×{(content.technique_analysis as any).peace_markers}</span>
                )}
                {(content.technique_analysis as any).reid_markers > 0 && (
                  <span className="text-[10px] text-amber-400">Reid ×{(content.technique_analysis as any).reid_markers}</span>
                )}
                {(content.technique_analysis as any).coercive_markers > 0 && (
                  <span className="text-[10px] text-red-400">Coercive ×{(content.technique_analysis as any).coercive_markers}</span>
                )}
              </div>
              {(content.technique_analysis as any).assessment && (
                <p className="text-xs text-nexus-text-secondary">{(content.technique_analysis as any).assessment}</p>
              )}
            </section>
          )}

          {/* 9 — Deal Assessment (sales_call only) */}
          {session.meeting_type === "sales_call" && content?.deal_assessment && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-3 text-sm font-semibold text-nexus-text-primary">
                📈 Deal Assessment
              </h2>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <span className="text-nexus-text-muted">Close Probability</span>
                  <div className="mt-1 font-semibold" style={{
                    color: (content.deal_assessment as any).close_probability === "high" ? "#10B981"
                      : (content.deal_assessment as any).close_probability === "medium" ? "#F59E0B"
                      : "#EF4444",
                  }}>
                    {(content.deal_assessment as any).close_probability?.toUpperCase()}
                  </div>
                </div>
                <div>
                  <span className="text-nexus-text-muted">Stage Reached</span>
                  <div className="mt-1 text-nexus-text-primary font-medium">
                    {(content.deal_assessment as any).stage_reached}
                  </div>
                </div>
                <div>
                  <span className="text-nexus-text-muted">Buying Signals</span>
                  <div className="mt-1 text-emerald-400 font-semibold">
                    {(content.deal_assessment as any).buying_signals ?? 0}
                  </div>
                </div>
                <div>
                  <span className="text-nexus-text-muted">Unresolved Objections</span>
                  <div className="mt-1 font-semibold" style={{
                    color: (content.deal_assessment as any).unresolved_objections > 0 ? "#EF4444" : "#10B981",
                  }}>
                    {(content.deal_assessment as any).unresolved_objections ?? 0}
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* 10 — Objection Handling (sales_call + client_meeting) */}
          {["sales_call", "client_meeting"].includes(session.meeting_type) &&
            content?.objection_handling &&
            (content.objection_handling as any[]).length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-3 text-sm font-semibold text-nexus-text-primary">
                🛡️ Objection Handling
              </h2>
              <div className="space-y-2">
                {(content.objection_handling as any[]).map((obj: any, i: number) => (
                  <div key={i} className="rounded border-l-2 bg-nexus-surface-hover p-2 text-xs"
                    style={{ borderLeftColor: obj.resolved ? "var(--stress-low)" : "var(--stress-high)" }}>
                    <div className="flex items-center gap-2 mb-1">
                      <span>{obj.resolved ? "✅" : "❌"}</span>
                      <span className="font-medium text-nexus-text-primary">{obj.objection}</span>
                      {obj.timestamp && (
                        <span className="ml-auto font-mono text-nexus-text-muted">{obj.timestamp}</span>
                      )}
                    </div>
                    {obj.handling_quality && (
                      <p className="text-nexus-text-secondary mb-0.5">Quality: {obj.handling_quality}</p>
                    )}
                    {obj.prospect_reaction && (
                      <p className="text-nexus-text-muted italic">{obj.prospect_reaction}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 11 — Recommendations */}
          {content?.recommendations && (content.recommendations as any[]).length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-3 text-sm font-semibold text-nexus-text-primary">
                Coaching Recommendations
              </h2>
              <ul className="space-y-3">
                {(content.recommendations as any[]).map((rec: any, i: number) => {
                  const text = typeof rec === "string" ? rec : rec.action;
                  const priority: string = typeof rec === "object" ? (rec.priority ?? "") : "";
                  const rationale: string = typeof rec === "object" ? (rec.rationale ?? "") : "";
                  return (
                    <li key={i} className="flex items-start gap-2 text-sm text-nexus-text-primary">
                      <span
                        className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
                        style={{
                          backgroundColor: priority === "high" ? "#EF4444"
                            : priority === "medium" ? "#F59E0B"
                            : "#22C55E",
                        }}
                      />
                      <div>
                        <span>{text}</span>
                        {rationale && (
                          <p className="mt-0.5 text-xs text-nexus-text-muted italic">{rationale}</p>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            </section>
          )}

          {/* Action Items (grouped by assignee) */}
          {content?.action_items && (content.action_items as any[]).length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-4 text-sm font-semibold text-nexus-text-primary">
                Action Items
              </h2>
              <div className="space-y-4">
                {Object.entries(
                  (content.action_items as any[]).reduce((groups: Record<string, any[]>, item: any) => {
                    const key = item.assignee || "Unassigned";
                    (groups[key] = groups[key] || []).push(item);
                    return groups;
                  }, {})
                ).map(([assignee, items]) => (
                  <div key={assignee}>
                    <h3 className="text-xs font-semibold text-nexus-text-primary mb-1.5 uppercase tracking-wide">
                      {assignee}
                    </h3>
                    <ul className="space-y-1.5 ml-2">
                      {(items as any[]).map((item: any, i: number) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-nexus-text-primary">
                          <span className="mt-0.5">•</span>
                          <div>
                            {item.task}
                            {item.deadline && (
                              <span className="ml-1.5 text-xs text-amber-400 font-medium">
                                — {item.deadline}
                              </span>
                            )}
                            {item.timestamp && (
                              <span className="ml-1.5 text-[10px] text-nexus-text-muted">
                                ({item.timestamp})
                              </span>
                            )}
                            {item.context && (
                              <p className="text-xs text-nexus-text-muted mt-0.5">{item.context}</p>
                            )}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Behavioral Analysis — 4 subsections */}
          {content?.behavioral_analysis && Object.keys(content.behavioral_analysis).length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-4 text-sm font-semibold text-nexus-text-primary">
                Behavioral Analysis
              </h2>

              {/* Key Moments */}
              {(content.behavioral_analysis.hotspots ?? []).length > 0 && (
                <div className="mb-5">
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-nexus-text-muted">
                    Key Moments
                  </h3>
                  <div className="space-y-1">
                    {content.behavioral_analysis.hotspots!.map((h, i) => (
                      <div key={i} className="flex items-start gap-2 rounded p-1.5 text-sm hover:bg-nexus-surface-hover">
                        <span className="w-10 shrink-0 font-mono text-xs text-nexus-text-muted">{h.timestamp}</span>
                        <span className="shrink-0 font-medium">{h.speaker}</span>
                        <span className="flex-1 text-nexus-text-secondary">&ldquo;{(h.text ?? "").slice(0, 80)}&rdquo;</span>
                        <span className="shrink-0 text-[10px] font-medium text-nexus-text-muted">
                          {(h.categories ?? []).join(" · ")}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Behavioral Arc */}
              {content.behavioral_analysis.speaker_trajectories &&
               Object.keys(content.behavioral_analysis.speaker_trajectories).length > 0 && (
                <div className="mb-5">
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-nexus-text-muted">
                    Behavioral Arc
                  </h3>
                  {Object.entries(content.behavioral_analysis.speaker_trajectories).map(([spk, tl]) => (
                    <div key={spk} className="mb-3">
                      <div className="mb-1 text-xs font-medium">{spk}</div>
                      {tl.summary && (
                        <div className="mb-1.5 text-xs italic text-nexus-text-secondary">{tl.summary}</div>
                      )}
                      {(tl.phases ?? []).length > 0 && (
                        <div className="flex h-6 gap-0.5 overflow-hidden rounded">
                          {tl.phases!.map((phase, i) => (
                            <div
                              key={i}
                              className="flex flex-1 items-center justify-center text-[8px] font-medium text-white"
                              style={{ backgroundColor: phaseToColor(phase) }}
                              title={phase}
                            >
                              {phase}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Key Exchanges */}
              {(content.behavioral_analysis.exchange_analysis ?? []).length > 0 && (
                <div className="mb-5">
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-nexus-text-muted">
                    Key Exchanges
                  </h3>
                  <div className="space-y-2">
                    {content.behavioral_analysis.exchange_analysis!.map((ex, i) => (
                      <div key={i} className="rounded border border-nexus-border p-2.5">
                        <div className="text-xs text-nexus-text-muted">{ex.stimulus}</div>
                        <div className="mt-1 flex items-center gap-2">
                          <span className="text-nexus-text-muted">→</span>
                          <span className="text-sm font-medium">
                            {(ex.response_categories ?? []).join(" · ")}
                          </span>
                          {(ex.stress_impact ?? 0) > 1.5 && (
                            <span className="text-[10px] text-nexus-text-muted">
                              {ex.stress_impact!.toFixed(1)}× baseline
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Topic Sensitivity */}
              {content.behavioral_analysis.topic_sensitivity &&
               Object.keys(content.behavioral_analysis.topic_sensitivity).length > 0 && (
                <div>
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-nexus-text-muted">
                    Topic Sensitivity
                  </h3>
                  {Object.entries(content.behavioral_analysis.topic_sensitivity).map(([spk, data]) => (
                    <div key={spk} className="mb-2 text-xs">
                      <div className="mb-1 font-medium">{spk}</div>
                      {data.most_sensitive && (
                        <div className="text-nexus-text-secondary">
                          Most reactive: <span className="font-medium">{data.most_sensitive}</span>
                        </div>
                      )}
                      {data.least_sensitive && (
                        <div className="text-nexus-text-muted">
                          Least reactive: {data.least_sensitive}
                        </div>
                      )}
                      {(data.topics ?? []).map((t, i) => (
                        <div key={i} className="mt-0.5 flex items-center gap-2">
                          <span className="w-4 text-nexus-text-muted">#{i + 1}</span>
                          <span className="flex-1">{t.topic}</span>
                          <span className="w-16 text-nexus-text-muted">{t.dominant_category}</span>
                          <span className="w-8 text-nexus-text-muted">{(t.stress_ratio ?? 1).toFixed(1)}×</span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {!content?.executive_summary && !report?.narrative && (
            <div className="flex h-48 items-center justify-center rounded-lg border border-nexus-border bg-nexus-surface text-sm text-nexus-text-muted">
              No report available for this session.
            </div>
          )}
        </div>
      )}

      {/* ═══ CHAT TAB ═══ */}
      {activeTab === "chat" && (
        <div className="rounded-lg border border-nexus-border bg-nexus-surface" style={{ height: "calc(100vh - 260px)", minHeight: 400 }}>
          <SessionChat sessionId={session.id} meetingType={session.meeting_type} />
        </div>
      )}
    </div>
  );
}
