import { useState } from "react";
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
  screenEngagementPct: number;   // avg screen_contact value × 100
  dominantEmotion: string | null;
  smileCount: number;
  nodCount: number;
  fidgetLevel: "low" | "moderate" | "high" | null;
  voiceFaceAlignmentPct: number; // 0 = no data, else 0-100
}

function computeVideoStats(signals: Signal[]): Record<string, VideoStats> {
  const speakers = [...new Set(signals.filter((s) => s.speaker_label).map((s) => s.speaker_label!))];
  const result: Record<string, VideoStats> = {};

  for (const label of speakers) {
    const sigs = signals.filter((s) => s.speaker_label === label);
    const video = sigs.filter((s) => s.agent === "video");
    const fusion = sigs.filter((s) => s.agent === "fusion");

    if (video.length === 0) continue;

    // Screen engagement
    const screenVals = video
      .filter((s) => s.signal_type === "screen_contact" && s.value != null)
      .map((s) => s.value!);
    const screenEngagementPct = screenVals.length > 0
      ? Math.round((screenVals.reduce((a, b) => a + b, 0) / screenVals.length) * 100)
      : 0;

    // Dominant emotion
    const emotionFreq = new Map<string, number>();
    for (const s of video.filter((s) => s.signal_type === "facial_emotion" && s.value_text)) {
      const e = s.value_text!;
      emotionFreq.set(e, (emotionFreq.get(e) ?? 0) + 1);
    }
    let dominantEmotion: string | null = null;
    let maxFreq = 0;
    for (const [e, freq] of emotionFreq) {
      if (freq > maxFreq && e !== "neutral") { dominantEmotion = e; maxFreq = freq; }
    }

    // Counts
    const smileCount = video.filter((s) => s.signal_type === "smile_type").length;
    const nodCount = video.filter((s) => s.signal_type === "head_nod").length;

    // Fidget level
    const fidgetVals = video
      .filter((s) => s.signal_type === "body_fidgeting" && s.value != null)
      .map((s) => s.value!);
    const avgFidget = fidgetVals.length > 0
      ? fidgetVals.reduce((a, b) => a + b, 0) / fidgetVals.length
      : -1;
    const fidgetLevel: VideoStats["fidgetLevel"] =
      avgFidget < 0 ? null
      : avgFidget > 0.6 ? "high"
      : avgFidget > 0.3 ? "moderate"
      : "low";

    // Voice-face alignment (inverse of incongruence confidence × score)
    const incongSigs = fusion.filter(
      (s) => ["tone_face_masking", "smile_sentiment_incongruence", "stress_suppression"].includes(s.signal_type)
    );
    let voiceFaceAlignmentPct = 0;
    if (video.length > 0) {
      if (incongSigs.length === 0) {
        voiceFaceAlignmentPct = 88; // No incongruence detected → high alignment
      } else {
        const avgIncon =
          incongSigs.reduce((a, s) => a + (s.value ?? 0) * (s.confidence ?? 0), 0) / incongSigs.length;
        voiceFaceAlignmentPct = Math.round(Math.max(0, 1 - avgIncon) * 100);
      }
    }

    result[label] = {
      screenEngagementPct,
      dominantEmotion,
      smileCount,
      nodCount,
      fidgetLevel,
      voiceFaceAlignmentPct,
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
    queryFn: () => getSignals(id!, { limit: 5000 }),
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
    enabled: !!id && !!detail?.session?.media_url,
  });

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
                {session.speaker_count ?? "--"} speakers
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
        <>
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

          {/* Signal Explorer */}
          <SignalExplorer
            signals={signals}
            signalsByAgent={detail.signals_by_agent}
            totalCount={detail.signal_count}
            speakerRoles={speakerRoles}
          />
        </>
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
                        const EMOTION_LABEL: Record<string, string> = {
                          happy: "😊 happy", joy: "😊 joyful", excited: "😄 excited",
                          sad: "😢 sad", angry: "😠 angry", fearful: "😨 fearful",
                          disgusted: "😒 disgusted", surprised: "😲 surprised",
                          contempt: "🙄 contempt", stressed: "😬 stressed",
                        };
                        return (
                          <div className="mt-3 rounded-md bg-nexus-surface-hover p-2.5">
                            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-nexus-text-muted">
                              Visual
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {vs.screenEngagementPct > 0 && (
                                <StatChip label="Screen" value={`${vs.screenEngagementPct}%`} color={vs.screenEngagementPct >= 70 ? "#22C55E" : vs.screenEngagementPct >= 45 ? "#F59E0B" : "#EF4444"} />
                              )}
                              {vs.dominantEmotion && (
                                <StatChip label="Emotion" value={EMOTION_LABEL[vs.dominantEmotion] ?? vs.dominantEmotion} />
                              )}
                              {vs.smileCount > 0 && (
                                <StatChip label="Smiles" value={vs.smileCount} color="#22C55E" />
                              )}
                              {vs.nodCount > 0 && (
                                <StatChip label="Nods" value={vs.nodCount} color="#4F8BFF" />
                              )}
                              {vs.fidgetLevel && (
                                <StatChip label="Fidget" value={vs.fidgetLevel} color={vs.fidgetLevel === "high" ? "#EF4444" : vs.fidgetLevel === "moderate" ? "#F59E0B" : "#22C55E"} />
                              )}
                              {vs.voiceFaceAlignmentPct > 0 && (
                                <StatChip
                                  label="Alignment"
                                  value={`${vs.voiceFaceAlignmentPct}%`}
                                  color={vs.voiceFaceAlignmentPct >= 75 ? "#22C55E" : vs.voiceFaceAlignmentPct >= 55 ? "#F59E0B" : "#EF4444"}
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

      {/* 5b. VIDEO PLAYER with signal overlay */}
      {session.media_url && videoSignalData && (
        <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
          <h2 className="mb-3 text-sm font-medium text-nexus-text-secondary">
            Video Playback
          </h2>
          <VideoSignalPlayer
            sessionId={id!}
            signals={videoSignalData.signals}
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
            {segments.map((segment) => (
              <TranscriptBlock
                key={segment.id}
                segment={segment}
                signals={matchSignalsToSegment(segment, signals)}
                speakerRole={segment.speaker_label ? speakerRoles[segment.speaker_label] : undefined}
                speakerName={segment.speaker_label ? speakerNames[segment.speaker_label] : undefined}
              />
            ))}
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
          {/* Executive Summary */}
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

          {/* Key Moments */}
          {content?.key_moments && content.key_moments.length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-nexus-text-primary">
                <Target className="h-4 w-4 text-nexus-accent-blue" />
                Key Moments
              </h2>
              <div className="space-y-4">
                {content.key_moments.map((moment, i) => (
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
                      <p className="mt-0.5 text-xs text-nexus-text-secondary italic">{moment.significance}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Cross-Modal Insights */}
          <section className="rounded-lg border-l-[3px] border-nexus-accent-purple bg-nexus-surface p-5">
            <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-nexus-accent-purple">
              <Lightbulb className="h-4 w-4" />
              Cross-Modal Insights
            </h2>
            {content?.cross_modal_insights && content.cross_modal_insights.length > 0 ? (
              <ul className="space-y-2">
                {content.cross_modal_insights.map((insight, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-nexus-text-primary">
                    <span className="mt-1 text-nexus-accent-purple font-bold">→</span>
                    {insight}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-nexus-text-muted italic">
                Insufficient cross-modal data for insights in this session.
              </p>
            )}
          </section>

          {/* Coaching Recommendations */}
          {content?.recommendations && content.recommendations.length > 0 && (
            <section className="rounded-lg border border-nexus-border bg-nexus-surface p-5">
              <h2 className="mb-3 text-sm font-semibold text-nexus-text-primary">
                Coaching Recommendations
              </h2>
              <ul className="space-y-2">
                {content.recommendations.map((rec, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-nexus-text-primary">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-nexus-stress-low" />
                    {rec}
                  </li>
                ))}
              </ul>
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
