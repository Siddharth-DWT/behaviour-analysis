/**
 * Transforms raw session data into structured SwimlaneData for the
 * conversation timeline visualisation.
 */
import type { Signal, TranscriptSegment } from "../api/client";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

export interface BlockSignal {
  type: "stress" | "buying" | "objection" | "sentiment" | "filler" | "pitch" | "fusion" | "tone" | "rate"
      | "face" | "body" | "gaze";
  value: number;
  label: string;
  confidence: number;
}

export interface SpeechBlock {
  start_ms: number;
  end_ms: number;
  text: string;
  segmentIds: string[];
  signals: BlockSignal[];
  /** Dominant border color key */
  borderType: "stress" | "buying" | "objection" | "fusion" | "sentiment" | "face" | "normal";
}

export interface SwimlaneSpaker {
  id: string;
  label: string;
  name: string;
  role: string;
  color: string;
  talkTimePct: number;
  blocks: SpeechBlock[];
}

export interface FusionEvent {
  timestamp_ms: number;
  type: string;
  label: string;
  severity: "critical" | "warning" | "info" | "positive";
  speaker_id?: string;
}

export interface TopicSegment {
  name: string;
  start_ms: number;
  end_ms: number;
  color: string;
}

export interface SwimlaneData {
  duration_ms: number;
  topics: TopicSegment[];
  speakers: SwimlaneSpaker[];
  fusion_events: FusionEvent[];
}

/* ------------------------------------------------------------------ */
/* Speaker colors                                                      */
/* ------------------------------------------------------------------ */

const SPEAKER_COLORS = [
  "#4F8BFF", // blue
  "#F59E0B", // orange
  "#10B981", // green
  "#8B5CF6", // purple
  "#EC4899", // pink
  "#06B6D4", // cyan
  "#F97316", // deep orange
  "#6366F1", // indigo
];

const TOPIC_COLORS = [
  "#3B82F6", "#8B5CF6", "#10B981", "#F59E0B",
  "#EF4444", "#EC4899", "#06B6D4", "#6366F1",
];

/* ------------------------------------------------------------------ */
/* Video signal type maps                                              */
/* ------------------------------------------------------------------ */

const FACE_SIGNAL_LABELS: Record<string, (sig: Signal) => string> = {
  facial_stress:     (s) => `Face stress: ${s.value_text === "high_facial_stress" ? "High" : "Moderate"}`,
  facial_emotion:    (s) => `Emotion: ${s.value_text || ""}`,
  facial_engagement: (s) => s.value_text === "high_engagement" ? "Engaged" : "Disengaged",
  smile_type:        (s) => s.value_text === "duchenne" ? "Genuine smile" : "Social smile",
  valence_arousal:   (s) => (s.value_text || "").replace(/_/g, " "),
};

const BODY_SIGNAL_LABELS: Record<string, (sig: Signal) => string> = {
  head_nod:               () => "Head nod",
  head_shake:             () => "Head shake",
  posture:                (s) => `Posture: ${(s.value_text || "").replace(/_/g, " ")}`,
  body_lean:              (s) => s.value_text === "forward_lean" ? "Forward lean" : "Back lean",
  body_fidgeting:         () => "Fidgeting",
  self_touch:             () => "Self-touch",
  shoulder_tension:       () => "Shoulder tension",
  head_body_incongruence: (s) => `Mismatch: ${(s.value_text || "").replace(/_/g, " ")}`,
  gesture_animation:      () => "Gesturing",
  body_mirroring:         () => "Mirroring",
};

const GAZE_SIGNAL_LABELS: Record<string, (sig: Signal) => string> = {
  gaze_direction_shift: (s) => `Gaze: ${(s.value_text || "shift").replace(/_/g, " ")}`,
  screen_contact:       (s) => s.value_text === "sustained_contact" ? "Eye contact" : "Gaze avoidance",
  sustained_distraction:() => "Distracted",
  attention_level:      (s) => `Attention: ${(s.value_text || "").replace(/_/g, " ")}`,
  blink_rate_anomaly:   (s) => `Blink: ${(s.value_text || "anomaly").replace(/_/g, " ")}`,
  gaze_synchrony:       () => "Gaze sync",
};

/* ------------------------------------------------------------------ */
/* Signal classification                                               */
/* ------------------------------------------------------------------ */

function classifySignal(sig: Signal): BlockSignal | null {
  const v = sig.value ?? 0;
  const c = sig.confidence ?? 0;

  // ── Voice / Language signals ───────────────────────────────────────
  switch (sig.signal_type) {
    case "vocal_stress_score":
      return { type: "stress", value: v, label: `Stress ${Math.round(v * 100)}%`, confidence: c };
    case "buying_signal":
      return { type: "buying", value: v, label: sig.value_text || "Buying signal", confidence: c };
    case "objection_signal":
      return { type: "objection", value: v, label: sig.value_text || "Objection", confidence: c };
    case "filler_detection":
      if (sig.value_text === "filler_spike" || sig.value_text === "filler_elevated")
        return { type: "filler", value: v, label: "Filler spike", confidence: c };
      return null;
    case "pitch_elevation_flag":
      return { type: "pitch", value: v, label: sig.value_text || "Pitch elevated", confidence: c };
    case "speech_rate_anomaly":
      return { type: "rate", value: v, label: sig.value_text || "Rate anomaly", confidence: c };
    case "tone_classification":
      if (sig.value_text === "nervous" || sig.value_text === "confident")
        return { type: "tone", value: v, label: `Tone: ${sig.value_text}`, confidence: c };
      return null;
    case "sentiment_score":
      return { type: "sentiment", value: v, label: v > 0 ? "Positive" : "Negative", confidence: c };
    case "tension_cluster":
    case "momentum_shift":
    case "persistent_incongruence":
    case "verbal_incongruence":
      return { type: "fusion", value: v, label: sig.value_text || sig.signal_type.replace(/_/g, " "), confidence: c };
  }

  // ── Video: Facial / Body / Gaze signals ──────────────────────────
  // Confidence filtering is done by the API gateway; render whatever arrives.
  if (sig.signal_type in FACE_SIGNAL_LABELS)
    return { type: "face", value: v, label: FACE_SIGNAL_LABELS[sig.signal_type](sig), confidence: c };

  if (sig.signal_type in BODY_SIGNAL_LABELS)
    return { type: "body", value: v, label: BODY_SIGNAL_LABELS[sig.signal_type](sig), confidence: c };

  if (sig.signal_type in GAZE_SIGNAL_LABELS)
    return { type: "gaze", value: v, label: GAZE_SIGNAL_LABELS[sig.signal_type](sig), confidence: c };

  return null;
}

function pickBorderType(signals: BlockSignal[]): SpeechBlock["borderType"] {
  for (const s of signals) {
    if (s.type === "stress" && s.value > 0.65) return "stress";
    if (s.type === "objection") return "objection";
  }
  for (const s of signals) {
    if (s.type === "fusion") return "fusion";
    if (s.type === "buying") return "buying";
    if (s.type === "face" && s.value > 0.7) return "face";
  }
  for (const s of signals) {
    if (s.type === "sentiment") return "sentiment";
    if (s.type === "stress" && s.value > 0.55) return "stress";
  }
  return "normal";
}

/* ------------------------------------------------------------------ */
/* Merge consecutive same-speaker segments < gap apart                 */
/* ------------------------------------------------------------------ */

function mergeSegments(
  segs: TranscriptSegment[],
  maxGapMs: number
): { start_ms: number; end_ms: number; text: string; ids: string[] }[] {
  if (segs.length === 0) return [];
  const sorted = [...segs].sort((a, b) => a.start_ms - b.start_ms);
  const merged: { start_ms: number; end_ms: number; text: string; ids: string[] }[] = [];
  let current = {
    start_ms: sorted[0].start_ms,
    end_ms: sorted[0].end_ms,
    text: sorted[0].text,
    ids: [sorted[0].id],
  };

  for (let i = 1; i < sorted.length; i++) {
    const seg = sorted[i];
    if (seg.start_ms - current.end_ms <= maxGapMs) {
      current.end_ms = Math.max(current.end_ms, seg.end_ms);
      current.text += " " + seg.text;
      current.ids.push(seg.id);
    } else {
      merged.push(current);
      current = {
        start_ms: seg.start_ms,
        end_ms: seg.end_ms,
        text: seg.text,
        ids: [seg.id],
      };
    }
  }
  merged.push(current);
  return merged;
}

/* ------------------------------------------------------------------ */
/* Build fusion events from fusion signals                             */
/* ------------------------------------------------------------------ */

function buildFusionEvents(signals: Signal[]): FusionEvent[] {
  return signals
    .filter((s) => s.agent === "fusion")
    .map((s) => {
      let severity: FusionEvent["severity"] = "info";
      if (s.signal_type === "tension_cluster") {
        severity = (s.value ?? 0) > 1.0 ? "critical" : "warning";
      } else if (s.signal_type === "persistent_incongruence" || s.signal_type === "verbal_incongruence") {
        severity = "warning";
      } else if (s.signal_type === "momentum_shift") {
        severity = s.value_text?.includes("positive") ? "positive" : "warning";
      }
      return {
        timestamp_ms: s.window_start_ms,
        type: s.signal_type,
        label: s.value_text || s.signal_type.replace(/_/g, " "),
        severity,
        speaker_id: s.speaker_label || undefined,
      };
    })
    .sort((a, b) => a.timestamp_ms - b.timestamp_ms);
}

/* ------------------------------------------------------------------ */
/* Build topic segments from entities                                  */
/* ------------------------------------------------------------------ */

function buildTopics(
  entities: Record<string, unknown>,
  durationMs: number
): TopicSegment[] {
  const rawTopics = (entities?.topics as Array<{ name: string; start_ms?: number; end_ms?: number }>) || [];
  if (rawTopics.length === 0) return [];

  return rawTopics.map((t, i) => ({
    name: t.name || `Topic ${i + 1}`,
    start_ms: t.start_ms || (durationMs / rawTopics.length) * i,
    end_ms: t.end_ms || (durationMs / rawTopics.length) * (i + 1),
    color: TOPIC_COLORS[i % TOPIC_COLORS.length],
  }));
}

/* ------------------------------------------------------------------ */
/* Main builder                                                        */
/* ------------------------------------------------------------------ */

export function buildSwimlaneData(
  transcriptSegments: TranscriptSegment[],
  signals: Signal[],
  entities: Record<string, unknown>,
  speakerRoles: Record<string, string>,
  durationMs: number,
): SwimlaneData {
  // 1. Group segments by speaker
  const bySpeaker = new Map<string, TranscriptSegment[]>();
  for (const seg of transcriptSegments) {
    const key = seg.speaker_label || seg.speaker_id || "Unknown";
    if (!bySpeaker.has(key)) bySpeaker.set(key, []);
    bySpeaker.get(key)!.push(seg);
  }

  // Index signals by speaker for fast lookup
  const sigsBySpeaker = new Map<string, Signal[]>();
  for (const sig of signals) {
    const key = sig.speaker_label || "Unknown";
    if (!sigsBySpeaker.has(key)) sigsBySpeaker.set(key, []);
    sigsBySpeaker.get(key)!.push(sig);
  }

  // 2. Compute talk time
  const talkMs = new Map<string, number>();
  for (const [spk, segs] of bySpeaker) {
    talkMs.set(spk, segs.reduce((sum, s) => sum + Math.max(0, s.end_ms - s.start_ms), 0));
  }
  const totalTalkMs = Array.from(talkMs.values()).reduce((a, b) => a + b, 0) || 1;

  // 3. Build speakers with merged blocks + embedded signals
  const speakerEntries = Array.from(bySpeaker.entries());
  const speakers: SwimlaneSpaker[] = speakerEntries.map(([spkLabel, segs], idx) => {
    const merged = mergeSegments(segs, 1000); // merge if < 1s gap
    const speakerSigs = sigsBySpeaker.get(spkLabel) || [];

    const blocks: SpeechBlock[] = merged.map((block) => {
      // Find overlapping signals
      const overlapping = speakerSigs.filter(
        (s) => s.window_start_ms < block.end_ms && (s.window_end_ms || s.window_start_ms) > block.start_ms
      );
      const classified = overlapping.map(classifySignal).filter(Boolean) as BlockSignal[];

      // Deduplicate by type (keep highest value per type)
      const bestByType = new Map<string, BlockSignal>();
      for (const bs of classified) {
        const existing = bestByType.get(bs.type);
        if (!existing || bs.value > existing.value) {
          bestByType.set(bs.type, bs);
        }
      }
      const blockSignals = Array.from(bestByType.values());

      return {
        start_ms: block.start_ms,
        end_ms: block.end_ms,
        text: block.text,
        segmentIds: block.ids,
        signals: blockSignals,
        borderType: pickBorderType(blockSignals),
      };
    });

    // Resolve name from entities
    let name = spkLabel;
    const people = entities?.people as Array<Record<string, string>> | undefined;
    if (people) {
      const match = people.find((p) => p.speaker_label === spkLabel);
      if (match?.name) name = match.name;
    }

    return {
      id: spkLabel,
      label: spkLabel,
      name,
      role: speakerRoles[spkLabel] || "",
      color: SPEAKER_COLORS[idx % SPEAKER_COLORS.length],
      talkTimePct: ((talkMs.get(spkLabel) || 0) / totalTalkMs) * 100,
      blocks,
    };
  });

  // 4. Build fusion events
  const fusion_events = buildFusionEvents(signals);

  // 5. Build topics
  const topics = buildTopics(entities, durationMs);

  return { duration_ms: durationMs, topics, speakers, fusion_events };
}
