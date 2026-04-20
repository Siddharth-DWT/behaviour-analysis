import type { TranscriptSegment, Signal } from "../api/client";
import SignalBadge, { filterSmartBadges } from "./SignalBadge";

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${String(sec).padStart(2, "0")}`;
}

function getBorderColor(signals: Signal[]): string {
  let hasHighStress = false;
  let hasObjection = false;
  let hasFusion = false;
  let hasBuyingSignal = false;
  let hasPositiveSentiment = false;

  for (const s of signals) {
    if (s.signal_type === "vocal_stress_score" && s.value != null && s.value > 0.5) hasHighStress = true;
    if (
      (s.signal_type === "objection_signal" || s.signal_type === "objection_detected") &&
      (s.value_text?.toLowerCase() === "true" || (s.value != null && s.value > 0.5))
    ) hasObjection = true;
    if (s.agent === "fusion") hasFusion = true;
    if (
      (s.signal_type === "buying_signal" || s.signal_type === "buying_signal_detected") &&
      (s.value_text?.toLowerCase() === "true" || (s.value != null && s.value > 0.5))
    ) hasBuyingSignal = true;
    if (s.signal_type === "sentiment_score" && s.value != null && s.value > 0.8) hasPositiveSentiment = true;
  }

  if (hasHighStress || hasObjection) return "border-l-nexus-stress-high";
  if (hasFusion) return "border-l-nexus-alert";
  if (hasBuyingSignal || hasPositiveSentiment) return "border-l-nexus-stress-low";
  return "border-l-transparent";
}

// ── Emotion emoji mapping ──
const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊", joy: "😊", excited: "😄",
  sad: "😢", sadness: "😢",
  angry: "😠", anger: "😠",
  fearful: "😨", fear: "😨",
  disgusted: "😒", disgust: "😒",
  surprised: "😲", surprise: "😲",
  contempt: "🙄",
  stressed: "😬",
};

// ── Build inline video indicators for a segment ──
interface VideoIndicators {
  emotionEmoji: string | null;
  stressLevel: "high" | "med" | "low" | null;   // combined voice+face
  incongruence: string | null;                    // fusion signal label
  headGesture: "nod" | "shake" | null;
  gazeAway: boolean;
}

function getVideoIndicators(signals: Signal[]): VideoIndicators {
  const video = signals.filter((s) => s.agent === "video");
  const fusion = signals.filter((s) => s.agent === "fusion");

  if (video.length === 0 && !fusion.some((s) =>
    ["tone_face_masking", "smile_sentiment_incongruence", "stress_suppression"].includes(s.signal_type)
  )) {
    return { emotionEmoji: null, stressLevel: null, incongruence: null, headGesture: null, gazeAway: false };
  }

  // Dominant emotion (highest confidence)
  const emotionSigs = video
    .filter((s) => s.signal_type === "facial_emotion" && s.value_text)
    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  const emotion = emotionSigs[0]?.value_text?.toLowerCase() ?? null;
  const emotionEmoji = emotion ? (EMOTION_EMOJI[emotion] ?? null) : null;

  // Combined stress (voice + face, take max)
  const voiceStress = signals.find((s) => s.signal_type === "vocal_stress_score")?.value ?? 0;
  const faceStress = video.find((s) => s.signal_type === "facial_stress")?.value ?? 0;
  const combined = Math.max(voiceStress, faceStress);
  const stressLevel: VideoIndicators["stressLevel"] =
    combined > 0.55 ? "high" : combined > 0.30 ? "med" : combined > 0.0 ? "low" : null;

  // Incongruence — only flag it (don't show label), let the badge text speak
  const INCONG_TYPES = [
    "tone_face_masking", "smile_sentiment_incongruence",
    "stress_suppression", "verbal_incongruence",
  ];
  const incongSig = fusion.find(
    (s) => INCONG_TYPES.includes(s.signal_type) && (s.confidence ?? 0) >= 0.40
  );
  const INCONG_LABELS: Record<string, string> = {
    tone_face_masking: "voice-face mismatch",
    smile_sentiment_incongruence: "smile masks sentiment",
    stress_suppression: "stress suppressed",
    verbal_incongruence: "verbal mismatch",
  };
  const incongruence = incongSig ? (INCONG_LABELS[incongSig.signal_type] ?? "incongruence") : null;

  // Head gesture (prefer most recent)
  const nodSig = video.find((s) => s.signal_type === "head_nod");
  const shakeSig = video.find((s) => s.signal_type === "head_shake");
  const headGesture: VideoIndicators["headGesture"] = shakeSig ? "shake" : nodSig ? "nod" : null;

  // Gaze — any break or sustained distraction
  const gazeAway = video.some(
    (s) => s.signal_type === "gaze_direction_shift" || s.signal_type === "sustained_distraction"
  );

  return { emotionEmoji, stressLevel, incongruence, headGesture, gazeAway };
}

const STRESS_DOT_COLOR = { high: "#EF4444", med: "#F59E0B", low: "#22C55E" };

interface Props {
  segment: TranscriptSegment;
  signals: Signal[];
  speakerRole?: string;
  speakerName?: string;
}

export default function TranscriptBlock({ segment, signals, speakerRole, speakerName }: Props) {
  const speaker = speakerName || segment.speaker_label || "Speaker";
  const badges = filterSmartBadges(signals, 3);
  const borderColor = getBorderColor(signals);
  const vi = getVideoIndicators(signals);

  const hasVideoIndicators =
    vi.emotionEmoji || vi.stressLevel || vi.incongruence || vi.headGesture || vi.gazeAway;

  return (
    <div
      className={`group rounded-lg border border-nexus-border border-l-[3px] ${borderColor} bg-nexus-surface p-3 transition-colors`}
    >
      {/* Header: time + speaker + role */}
      <div className="mb-1.5 flex items-center gap-2 text-xs">
        <span className="font-mono text-nexus-text-muted">
          {formatTime(segment.start_ms)}
        </span>
        <span className="font-medium text-nexus-accent-blue">{speaker}</span>
        {speakerRole && (
          <span className="rounded bg-accent-purple-15 px-1.5 py-0.5 text-[10px] font-medium text-nexus-accent-purple">
            {speakerRole}
          </span>
        )}
      </div>

      {/* Transcript text */}
      <p className="text-sm leading-relaxed text-nexus-text-primary">
        {segment.text}
      </p>

      {/* Smart signal badges (audio/language — max 3) */}
      {badges.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {badges.map((badge, i) => (
            <SignalBadge key={i} badge={badge} />
          ))}
        </div>
      )}

      {/* Video indicators row — only shown when video signals present */}
      {hasVideoIndicators && (
        <div className="mt-1.5 flex flex-wrap items-center gap-2 border-t border-nexus-border pt-1.5">
          {/* Emotion emoji */}
          {vi.emotionEmoji && (
            <span title="Detected facial emotion" className="text-sm leading-none">
              {vi.emotionEmoji}
            </span>
          )}

          {/* Combined stress dot */}
          {vi.stressLevel && (
            <span
              className="flex items-center gap-1 text-[10px] text-nexus-text-muted"
              title="Combined voice + face stress"
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: STRESS_DOT_COLOR[vi.stressLevel] }}
              />
              stress
            </span>
          )}

          {/* Head gesture */}
          {vi.headGesture === "nod" && (
            <span
              className="text-[10px] text-nexus-text-muted"
              title="Head nod detected"
            >
              👍 nod
            </span>
          )}
          {vi.headGesture === "shake" && (
            <span
              className="text-[10px] text-nexus-text-muted"
              title="Head shake detected"
            >
              👎 shake
            </span>
          )}

          {/* Gaze away */}
          {vi.gazeAway && (
            <span
              className="text-[10px] text-nexus-text-muted"
              title="Gaze break detected"
            >
              👁️‍🗨️ gaze break
            </span>
          )}

          {/* Incongruence warning — shown last, most prominent */}
          {vi.incongruence && (
            <span
              className="ml-auto flex items-center gap-1 rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-400"
              title="Voice and face signals disagree"
            >
              ⚠️ {vi.incongruence}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
