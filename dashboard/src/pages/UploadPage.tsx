import { useState, useRef, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  FileAudio,
  Loader2,
  RotateCcw,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  X,
  Copy,
  Check,
  ChevronDown,
} from "lucide-react";
import { uploadSession, quickTranscribe, getSession, getTranscript, getSessionProgress, getReport, listSessions } from "../api/client";
import type { TranscriptSegment, QuickSegment, Session } from "../api/client";
import { DEFAULT_CONFIG, type UploadConfig } from "../components/UploadSettings";

// ─── Option lists ──────────────────────────────────────────────────────────────

const CONTENT_TYPE_OPTIONS = [
  { value: "sales_call", label: "Sales Call" },
  { value: "client_meeting", label: "Client Meeting" },
  { value: "internal", label: "Internal Meeting" },
  { value: "interview", label: "Interview" },
  { value: "podcast", label: "Podcast" },
];

const LANGUAGE_OPTIONS = [
  { value: null, label: "Auto Detect" },
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "hi", label: "Hindi" },
  { value: "ar", label: "Arabic" },
  { value: "pt", label: "Portuguese" },
  { value: "ja", label: "Japanese" },
  { value: "zh", label: "Chinese" },
  { value: "ko", label: "Korean" },
  { value: "ru", label: "Russian" },
];

// ─── Model capability matrix ──────────────────────────────────────────────────

interface ModelFeatures {
  language: boolean;
  custom_prompt: boolean;
  key_terms: boolean;
  keep_filler_words: boolean;
  text_formatting: boolean;
  auto_punctuation: boolean;
  temperature: boolean;
  multichannel: boolean;
}

const MODEL_FEATURES: Record<string, ModelFeatures> = {
  auto:       { language: false, custom_prompt: false, key_terms: false, keep_filler_words: false, text_formatting: false, auto_punctuation: false, temperature: false, multichannel: false },
  parakeet:   { language: false, custom_prompt: false, key_terms: false, keep_filler_words: false, text_formatting: false, auto_punctuation: false, temperature: false, multichannel: false },
  assemblyai: { language: true,  custom_prompt: true,  key_terms: true,  keep_filler_words: true,  text_formatting: true,  auto_punctuation: true,  temperature: true,  multichannel: true  },
  whisper:    { language: true,  custom_prompt: true,  key_terms: true,  keep_filler_words: false, text_formatting: false, auto_punctuation: false, temperature: true,  multichannel: false },
  deepgram:   { language: true,  custom_prompt: false, key_terms: true,  keep_filler_words: true,  text_formatting: true,  auto_punctuation: true,  temperature: false, multichannel: true  },
};

function getFeatures(modelPref: string | null): ModelFeatures {
  return MODEL_FEATURES[modelPref ?? "auto"] ?? MODEL_FEATURES.auto;
}

const MODEL_CARDS = [
  {
    value: null,
    label: "Auto",
    subtitle: "Best Available",
    description: "Uses the fastest configured backend with smart fallback",
    chips: ["Smart fallback", "Fast"],
  },
  {
    value: "assemblyai",
    label: "AssemblyAI",
    subtitle: "Universal-3 Pro",
    description: "Most configurable — prompt, key terms, filler words, temperature",
    chips: ["Prompt", "Key terms", "Temperature"],
  },
  {
    value: "whisper",
    label: "Whisper",
    subtitle: "Large V3",
    description: "High accuracy, great for non-English and technical content",
    chips: ["Language", "Temperature", "Prompt"],
  },
  {
    value: "deepgram",
    label: "Deepgram",
    subtitle: "Nova 3",
    description: "Fast, accurate, supports filler words and smart formatting",
    chips: ["Language", "Key terms", "Smart format"],
  },
  {
    value: "parakeet",
    label: "Parakeet",
    subtitle: "NVIDIA",
    description: "Parakeet-CTC — fastest transcription, requires NVIDIA GPU server",
    chips: ["Fastest", "Parakeet-CTC"],
  },
];

const SPEAKER_OPTIONS = [
  { value: null, label: "Auto-detect" },
  ...Array.from({ length: 9 }, (_, i) => ({ value: i + 2, label: String(i + 2) })),
];

const TRANSLATE_OPTIONS = [
  { value: null, label: "No translation" },
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "hi", label: "Hindi" },
  { value: "ar", label: "Arabic" },
  { value: "pt", label: "Portuguese" },
];

// ─── Form helpers ──────────────────────────────────────────────────────────────

function FormSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string | number | null;
  options: { value: string | number | null; label: string }[];
  onChange: (v: string | number | null) => void;
}) {
  return (
    <div>
      <label className="mb-1 block text-[11px] font-medium text-nexus-text-muted uppercase tracking-wide">
        {label}
      </label>
      <select
        value={value === null ? "__null__" : String(value)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "__null__") onChange(null);
          else if (!isNaN(Number(raw)) && options.some((o) => typeof o.value === "number"))
            onChange(Number(raw));
          else onChange(raw);
        }}
        className="w-full rounded border border-nexus-border bg-nexus-bg px-2 py-1.5 text-xs text-nexus-text-secondary outline-none focus:border-nexus-accent-blue transition-colors"
      >
        {options.map((o) => (
          <option key={String(o.value)} value={o.value === null ? "__null__" : String(o.value)}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function FormToggle({
  label,
  checked,
  onChange,
  help,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  help?: string;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2.5 py-0.5">
      <div
        onClick={() => onChange(!checked)}
        className={`relative h-4 w-7 rounded-full transition-colors ${
          checked ? "bg-nexus-accent-blue" : "bg-nexus-border"
        }`}
      >
        <div
          className={`absolute top-0.5 h-3 w-3 rounded-full bg-white shadow transition-transform ${
            checked ? "translate-x-3" : "translate-x-0.5"
          }`}
        />
      </div>
      <span className="text-xs text-nexus-text-secondary">{label}</span>
      {help && <span className="text-[10px] text-nexus-text-muted">— {help}</span>}
    </label>
  );
}

// ─── Model radio cards ─────────────────────────────────────────────────────────

function ModelRadioCards({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (v: string | null) => void;
}) {
  return (
    <div className="space-y-1.5">
      {MODEL_CARDS.map((card) => {
        const selected = value === card.value;
        return (
          <button
            key={String(card.value)}
            type="button"
            onClick={() => onChange(card.value)}
            className={`w-full rounded-lg border px-3 py-2.5 text-left transition-colors ${
              selected
                ? "border-nexus-accent-blue bg-nexus-accent-blue/5"
                : "border-nexus-border bg-nexus-bg hover:border-nexus-accent-blue/40"
            }`}
          >
            <div className="flex items-center justify-between mb-0.5">
              <span className={`text-xs font-semibold ${selected ? "text-nexus-accent-blue" : "text-nexus-text-primary"}`}>
                {card.label}
              </span>
              <span className="text-[10px] text-nexus-text-muted">{card.subtitle}</span>
            </div>
            <p className="text-[10px] text-nexus-text-muted mb-1.5">{card.description}</p>
            <div className="flex flex-wrap gap-1">
              {card.chips.map((chip) => (
                <span
                  key={chip}
                  className={`rounded px-1.5 py-0.5 text-[9px] font-medium ${
                    selected
                      ? "bg-nexus-accent-blue/15 text-nexus-accent-blue"
                      : "bg-nexus-border/50 text-nexus-text-muted"
                  }`}
                >
                  {chip}
                </span>
              ))}
            </div>
          </button>
        );
      })}
    </div>
  );
}

// ─── Transcript chat preview (transcription only, no signal badges) ────────────

const BUBBLE_STYLES = [
  { border: "#3266ad", text: "#3266ad", bg: "#3266ad0f" },
  { border: "#639922", text: "#639922", bg: "#6399220f" },
  { border: "#ba7517", text: "#ba7517", bg: "#ba75170f" },
  { border: "#a32d2d", text: "#a32d2d", bg: "#a32d2d0f" },
  { border: "#7f5bbf", text: "#7f5bbf", bg: "#7f5bbf0f" },
];

function fmtMs(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function TranscriptChatPreview({ segments }: { segments: TranscriptSegment[] }) {
  const speakerOrder: string[] = [];
  for (const seg of segments) {
    const spk = seg.speaker_label || seg.speaker_id || "Unknown";
    if (!speakerOrder.includes(spk)) speakerOrder.push(spk);
  }

  if (segments.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center text-xs text-nexus-text-muted">
        No transcript available
      </div>
    );
  }

  return (
    <div className="overflow-y-auto space-y-2 pr-1" style={{ maxHeight: 380 }}>
      {segments.map((seg, i) => {
        const spk = seg.speaker_label || seg.speaker_id || "Unknown";
        const idx = speakerOrder.indexOf(spk);
        const style = BUBBLE_STYLES[idx % BUBBLE_STYLES.length];
        const isLeft = idx % 2 === 0;

        return (
          <div key={i} className={`flex ${isLeft ? "justify-start" : "justify-end"}`}>
            <div
              className="max-w-[82%] rounded-lg px-3 py-2"
              style={{
                backgroundColor: style.bg,
                borderLeft: isLeft ? `3px solid ${style.border}` : "none",
                borderRight: !isLeft ? `3px solid ${style.border}` : "none",
              }}
            >
              <div className={`flex items-center gap-2 mb-0.5 ${isLeft ? "" : "justify-end"}`}>
                <span className="text-[10px] font-semibold" style={{ color: style.text }}>
                  {spk}
                </span>
                <span className="text-[9px] text-nexus-text-muted">{fmtMs(seg.start_ms)}</span>
              </div>
              <p className="text-[11px] leading-relaxed text-nexus-text-primary">{seg.text}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Quick-mode done state (in-memory, no session) ────────────────────────────

const QUICK_SPEAKER_COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
];

function QuickDoneView({
  segments,
  meta,
  diarization,
  onReset,
}: {
  segments: QuickSegment[];
  meta: { duration: number; backend: string; model: string } | null;
  diarization: boolean;
  onReset: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [copyMenuOpen, setCopyMenuOpen] = useState(false);
  const copyMenuRef = useRef<HTMLDivElement>(null);

  // Close copy dropdown when clicking outside
  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (copyMenuRef.current && !copyMenuRef.current.contains(e.target as Node)) {
        setCopyMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);

  // Build unique speaker list from segments (preserves first-appearance order)
  const speakerOrder: string[] = [];
  for (const seg of segments) {
    if (!speakerOrder.includes(seg.speaker)) speakerOrder.push(seg.speaker);
  }
  const multiSpeaker = diarization && speakerOrder.length > 1;

  // Format ms → M:SS (matches TranscriptBlock display)
  const fmtTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };

  const doCopy = (withTimestamps: boolean) => {
    const text = segments
      .map((s) => {
        if (withTimestamps) {
          const time = fmtTime(s.start_ms);
          return multiSpeaker
            ? `${time}  ${s.speaker}  ${s.text}`
            : `${time}  ${s.text}`;
        }
        return multiSpeaker ? `${s.speaker}  ${s.text}` : s.text;
      })
      .join("\n");
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setCopyMenuOpen(false);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const durationLabel = meta
    ? `${Math.floor(meta.duration / 60)}m ${Math.round(meta.duration % 60)}s`
    : null;

  return (
    <>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-nexus-border px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-nexus-text-primary">Transcript</span>
          <span className="rounded-full bg-emerald-500/10 px-1.5 py-0.5 text-[9px] font-medium text-emerald-400 border border-emerald-500/20">
            DONE
          </span>
          {multiSpeaker && (
            <span className="text-[10px] text-nexus-text-muted">
              {speakerOrder.length} speakers
            </span>
          )}
          {durationLabel && (
            <span className="text-[10px] text-nexus-text-muted">{durationLabel}</span>
          )}
          {meta && (
            <span className="text-[10px] text-nexus-text-muted opacity-60">{meta.model}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Copy split-button with dropdown */}
          <div className="relative flex" ref={copyMenuRef}>
            <button
              onClick={() => doCopy(true)}
              className="flex items-center gap-1 rounded-l px-2 py-1 text-[10px] text-nexus-text-muted hover:text-nexus-text-primary hover:bg-nexus-surface-hover transition-colors border-r border-nexus-border"
            >
              {copied ? (
                <><Check className="h-3 w-3 text-emerald-400" /><span className="text-emerald-400">Copied</span></>
              ) : (
                <><Copy className="h-3 w-3" />Copy text</>
              )}
            </button>
            <button
              onClick={() => setCopyMenuOpen((o) => !o)}
              className="flex items-center rounded-r px-1.5 py-1 text-[10px] text-nexus-text-muted hover:text-nexus-text-primary hover:bg-nexus-surface-hover transition-colors"
            >
              <ChevronDown className={`h-3 w-3 transition-transform ${copyMenuOpen ? "rotate-180" : ""}`} />
            </button>
            {copyMenuOpen && (
              <div className="absolute right-0 top-full z-50 mt-1 w-44 rounded border border-nexus-border bg-nexus-surface py-1 shadow-lg">
                <button
                  onClick={() => doCopy(true)}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-[10px] text-nexus-text-secondary hover:bg-nexus-surface-hover"
                >
                  <Copy className="h-3 w-3" />
                  Copy with timestamps
                </button>
                <button
                  onClick={() => doCopy(false)}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-[10px] text-nexus-text-secondary hover:bg-nexus-surface-hover"
                >
                  <Copy className="h-3 w-3" />
                  Copy plain text
                </button>
              </div>
            )}
          </div>
          <button
            onClick={onReset}
            className="flex items-center gap-1 text-[10px] text-nexus-text-muted hover:text-nexus-text-primary transition-colors"
          >
            <RotateCcw className="h-3 w-3" />
            New Upload
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="p-4">
        {segments.length === 0 ? (
          <p className="text-xs text-nexus-text-muted text-center py-8">No transcript returned.</p>
        ) : multiSpeaker ? (
          /* Chat bubble view for diarized multi-speaker */
          <div className="overflow-y-auto space-y-2 pr-1" style={{ maxHeight: 380 }}>
            {segments.map((seg, i) => {
              const idx = speakerOrder.indexOf(seg.speaker);
              const color = QUICK_SPEAKER_COLORS[idx % QUICK_SPEAKER_COLORS.length];
              const isLeft = idx % 2 === 0;
              return (
                <div key={i} className={`flex ${isLeft ? "justify-start" : "justify-end"}`}>
                  <div
                    className="max-w-[82%] rounded-lg px-3 py-2"
                    style={{
                      backgroundColor: `${color}0f`,
                      borderLeft: isLeft ? `3px solid ${color}` : "none",
                      borderRight: !isLeft ? `3px solid ${color}` : "none",
                    }}
                  >
                    <div className={`flex items-center gap-2 mb-0.5 ${isLeft ? "" : "justify-end"}`}>
                      <span className="text-[10px] font-semibold" style={{ color }}>
                        {seg.speaker}
                      </span>
                      <span className="text-[9px] text-nexus-text-muted">{fmtMs(seg.start_ms)}</span>
                    </div>
                    <p className="text-[11px] leading-relaxed text-nexus-text-primary">{seg.text}</p>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          /* Plain text card for no-diarization or single speaker */
          <div className="overflow-y-auto space-y-1 pr-1" style={{ maxHeight: 380 }}>
            {segments.map((seg, i) => (
              <div key={i} className="flex gap-2 items-start">
                <span className="text-[9px] text-nexus-text-muted shrink-0 mt-0.5 w-8 text-right tabular-nums">
                  {fmtMs(seg.start_ms)}
                </span>
                <p className="text-[11px] leading-relaxed text-nexus-text-primary flex-1">{seg.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

// ─── Processing steps ──────────────────────────────────────────────────────────

function buildSteps(cfg: UploadConfig, hasVideo?: boolean): string[] {
  const steps = [
    "Uploading audio file",
    "Transcribing speech",
  ];
  if (cfg.run_diarization) {
    steps.push("Speaker diarization");
  }
  if (cfg.run_behavioural) {
    steps.push("Acoustic feature extraction");
    steps.push("Language & sentiment analysis");
    steps.push("Conversation dynamics");
    if (hasVideo) steps.push("Video analysis");
    steps.push("Fusion & signal scoring");
    steps.push("Report generation");
  }
  if (cfg.run_entity_extraction) {
    steps.push("Entity extraction");
  }
  if (cfg.run_behavioural && cfg.run_knowledge_graph) {
    steps.push("Knowledge graph sync");
  }
  return steps;
}

// Map a backend pipeline_step name to the active step index in buildSteps()
const STEP_LABEL: Record<string, string> = {
  transcribing:       "Transcribing speech",
  diarization:        "Speaker diarization",
  language:           "Language & sentiment analysis",
  conversation:       "Conversation dynamics",
  video:              "Video analysis",
  fusion:             "Fusion & signal scoring",
  report:             "Report generation",
  entity_extraction:  "Entity extraction",
  knowledge_graph:    "Knowledge graph sync",
};

// Ordered backend pipeline — used to find the nearest visible step when the
// current backend step isn't present in this config's step list.
const PIPELINE_ORDER = [
  "transcribing", "diarization", "language", "conversation",
  "video", "fusion", "report", "entity_extraction", "knowledge_graph",
];

function pipelineStepToIndex(stepName: string | null, steps: string[]): number {
  if (!stepName) return steps.length;  // null = pipeline done, all steps complete

  // Direct match
  const label = STEP_LABEL[stepName];
  if (label) {
    const idx = steps.indexOf(label);
    if (idx >= 0) return idx;
  }

  // Step not visible in this config — walk backwards to find the last visible step
  const pos = PIPELINE_ORDER.indexOf(stepName);
  for (let i = pos - 1; i >= 0; i--) {
    const prevLabel = STEP_LABEL[PIPELINE_ORDER[i]];
    if (prevLabel) {
      const idx = steps.indexOf(prevLabel);
      if (idx >= 0) return idx;
    }
  }

  return 0;
}

// ─── Inline step list (used in chat area while processing) ─────────────────────

function InlineStepList({ cfg, pipelineStep, hasVideo }: { cfg: UploadConfig; pipelineStep: string | null; hasVideo?: boolean }) {
  const steps = buildSteps(cfg, hasVideo);
  const activeStep = pipelineStepToIndex(pipelineStep, steps);

  return (
    <div className="space-y-2">
      {steps.map((step, i) => {
        const done = i < activeStep;
        const active = i === activeStep;
        return (
          <div key={i} className="flex items-center gap-2.5">
            <div className="w-3.5 shrink-0">
              {done ? (
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
              ) : active ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-nexus-accent-blue" />
              ) : (
                <div className="h-3.5 w-3.5 rounded-full border border-nexus-border" />
              )}
            </div>
            <span className={`text-[11px] transition-colors ${
              done ? "text-nexus-text-muted line-through" : active ? "text-nexus-text-primary font-medium" : "text-nexus-text-muted"
            }`}>
              {step}
            </span>
          </div>
        );
      })}
      <p className="pt-2 text-[10px] text-nexus-text-muted">
        This may take 1–3 minutes depending on file length
      </p>
    </div>
  );
}

// ─── Section card ──────────────────────────────────────────────────────────────

function SettingsCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-nexus-border bg-nexus-surface p-5">
      <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-nexus-text-muted">
        {title}
      </h3>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

type Phase = "form" | "processing" | "done" | "error";

export default function UploadPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [phase, setPhase] = useState<Phase>("form");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pipelineStep, setPipelineStep] = useState<string | null>("transcribing");
  const [title, setTitle] = useState("");
  const [cfg, setCfg] = useState<UploadConfig>({ ...DEFAULT_CONFIG });
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [sessionInfo, setSessionInfo] = useState<Session | null>(null);
  // Quick-mode (transcript-only): result lives in memory, never stored in DB
  const [quickSegments, setQuickSegments] = useState<QuickSegment[] | null>(null);
  const [quickMeta, setQuickMeta] = useState<{ duration: number; backend: string; model: string } | null>(null);
  // Entity summary shown in done state when entity extraction was enabled
  const [entitySummary, setEntitySummary] = useState<{
    people: Array<{ name: string; role: string }>;
    topics: Array<{ name: string }>;
    objections: Array<{ text: string; resolved: boolean }>;
    commitments: Array<{ text: string }>;
    companies: Array<{ name: string }>;
  } | null>(null);

  // Lightweight session history (transcript/diarize/entity sessions stored in DB)
  const [lightSessions, setLightSessions] = useState<Session[]>([]);
  const [lightLoading, setLightLoading] = useState(false);

  // Fetch lightweight session history on mount
  useEffect(() => {
    setLightLoading(true);
    listSessions({ session_type: "lightweight", limit: 50 })
      .then((r: { sessions: Session[] }) => setLightSessions(r.sessions))
      .catch(() => {})
      .finally(() => setLightLoading(false));
  }, []);

  // Load a lightweight session result into the done view
  const loadLightSession = useCallback(async (s: Session) => {
    try {
      const [txRes, reportRes] = await Promise.all([
        getTranscript(s.id),
        getReport(s.id).catch(() => null),
      ]);
      setSegments(txRes.segments);
      setSessionInfo(s);
      setSessionId(s.id);
      const ents = reportRes?.report?.content?.entities as any;
      if (ents) setEntitySummary(ents);
      else setEntitySummary(null);
      setQuickSegments(null);
      setPhase("done");
    } catch {
      // silently ignore
    }
  }, []);

  const set = <K extends keyof UploadConfig>(key: K, val: UploadConfig[K]) =>
    setCfg((prev) => ({ ...prev, [key]: val }));

  // ── File handling ──

  const pickFile = (f: File) => {
    setFile(f);
    setTitle(f.name.replace(/\.[^.]+$/, ""));
    setPhase("form");
    setQuickSegments(null);
    setQuickMeta(null);
    setSegments([]);
    setSessionId(null);
    setSessionInfo(null);
    setErrorMsg(null);
    setPipelineStep("transcribing");
  };

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) pickFile(f);
    },
    [] // eslint-disable-line react-hooks/exhaustive-deps
  );

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) pickFile(f);
    e.target.value = "";
  };

  // ── Polling ──

  const startPolling = useCallback((sid: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    setPipelineStep("transcribing");
    pollRef.current = setInterval(async () => {
      try {
        // Fetch step + session status in parallel
        const [progress, detail] = await Promise.all([
          getSessionProgress(sid).catch(() => ({ pipeline_step: null })),
          getSession(sid),
        ]);

        if (progress.pipeline_step) {
          setPipelineStep(progress.pipeline_step);
        }

        const status = detail.session.status;
        // "partial" = some agents failed but transcript succeeded — show it
        if (status === "completed" || status === "partial" || status === "failed") {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          setPipelineStep(null);
          if (status === "completed" || status === "partial") {
            const [txRes, reportRes] = await Promise.all([
              getTranscript(sid),
              getReport(sid).catch(() => null),
            ]);
            setSegments(txRes.segments);
            setSessionInfo(detail.session);
            const ents = reportRes?.report?.content?.entities as any;
            if (ents) setEntitySummary(ents);
            setPhase("done");
            // Refresh lightweight session list if this was a lightweight session
            if (detail.session.session_type === "lightweight") {
              listSessions({ session_type: "lightweight", limit: 50 })
                .then((r: { sessions: Session[] }) => setLightSessions(r.sessions))
                .catch(() => {});
            }
          } else {
            setErrorMsg("Analysis failed. Check logs or try again.");
            setPhase("error");
          }
        }
      } catch {
        // keep polling
      }
    }, 3000);
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // ── Submit ──

  const handleSubmit = async () => {
    if (!file) return;
    setPhase("processing");
    setErrorMsg(null);

    const configPayload = {
      meeting_type: cfg.meeting_type,
      transcription: {
        language: cfg.language,
        model_preference: cfg.model_preference,
        custom_prompt: cfg.custom_prompt || null,
        key_terms: cfg.key_terms
          ? cfg.key_terms.split(",").map((t) => t.trim()).filter(Boolean)
          : null,
        multichannel: cfg.multichannel,
        keep_filler_words: cfg.keep_filler_words,
        auto_punctuation: cfg.auto_punctuation,
        text_formatting: cfg.text_formatting,
        word_timestamps: cfg.word_timestamps,
        temperature: cfg.temperature,
      },
      analysis: {
        run_behavioural: cfg.run_behavioural,
        run_sentiment: cfg.run_sentiment,
        run_diarization: cfg.run_diarization,
        run_entity_extraction: cfg.run_entity_extraction,
        run_knowledge_graph: cfg.run_knowledge_graph,
        run_video: cfg.run_video,
        sensitivity: cfg.sensitivity,
        translate_to: cfg.translate_to,
      },
    };

    // ── Quick path: transcript-only — no session, no DB, result lives in memory ──
    // Only for bare transcript with no diarization, entity extraction, or behavioural analysis
    if (!cfg.run_behavioural && !cfg.run_entity_extraction && !cfg.run_diarization) {
      try {
        const result = await quickTranscribe(file, configPayload);
        setQuickSegments(result.segments);
        setQuickMeta({
          duration: result.duration_seconds,
          backend: result.backend,
          model: result.model,
        });
        setPhase("done");
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : "Transcription failed");
        setPhase("error");
      }
      return;
    }

    // ── Full path: create session + background pipeline ──
    try {
      const result = await uploadSession(
        file,
        title || file.name.replace(/\.[^.]+$/, ""),
        cfg.meeting_type,
        configPayload
      );
      setSessionId(result.session_id);
      startPolling(result.session_id);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Upload failed");
      setPhase("error");
    }
  };

  // ─── Phase: Processing ───────────────────────────────────────────────────────

  // ─── Unified layout (all phases) ────────────────────────────────────────────

  const resetUpload = () => {
    setPhase("form");
    setFile(null);
    setTitle("");
    setCfg({ ...DEFAULT_CONFIG });
    setSessionId(null);
    setSegments([]);
    setSessionInfo(null);
    setQuickSegments(null);
    setQuickMeta(null);
    setEntitySummary(null);
    setErrorMsg(null);
  };

  return (
    <div className="mx-auto max-w-5xl">
      <h1 className="mb-5 text-xl font-semibold text-nexus-text-primary">Upload Recording</h1>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_380px]">
        {/* ── Left: File + Title + Content Type + Prompt ── */}
        <div className="space-y-4">
          {/* File dropzone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`rounded-xl border-2 border-dashed transition-colors ${
              dragging
                ? "border-nexus-accent-blue bg-nexus-accent-blue/5 cursor-copy"
                : file
                ? "border-nexus-border bg-nexus-surface cursor-default"
                : "border-nexus-border hover:border-nexus-accent-blue/50 cursor-pointer"
            }`}
          >
            {file ? (
              <div className="flex items-center gap-4 p-5">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-nexus-accent-blue/10">
                  <FileAudio className="h-6 w-6 text-nexus-accent-blue" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-nexus-text-primary">{file.name}</p>
                  <p className="text-xs text-nexus-text-muted">
                    {(file.size / 1024 / 1024).toFixed(1)} MB
                  </p>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); setFile(null); }}
                  className="shrink-0 rounded-lg p-1.5 text-nexus-text-muted hover:text-nexus-text-primary hover:bg-nexus-surface-hover transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-3 py-12">
                <div className="flex h-14 w-14 items-center justify-center rounded-full border border-nexus-border bg-nexus-surface">
                  <Upload className="h-6 w-6 text-nexus-text-muted" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-nexus-text-secondary">
                    Drop audio or video file here
                  </p>
                  <p className="mt-0.5 text-xs text-nexus-text-muted">
                    or click to browse — WAV, MP3, M4A, FLAC, OGG, WEBM, MP4
                  </p>
                  <p className="mt-0.5 text-[10px] text-nexus-text-muted opacity-70">
                    MP4 / WEBM files also run facial, gaze &amp; body analysis
                  </p>
                </div>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.mp3,.m4a,.flac,.ogg,.webm,.mp4"
              onChange={onFileChange}
              className="hidden"
            />
          </div>

          {/* Title */}
          <div>
            <label className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-nexus-text-muted">
              Title
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Recording title..."
              className="w-full rounded-lg border border-nexus-border bg-nexus-surface px-3 py-2 text-sm text-nexus-text-primary outline-none focus:border-nexus-accent-blue transition-colors"
            />
          </div>

          {/* Content Type */}
          <FormSelect
            label="Content Type"
            value={cfg.meeting_type}
            options={CONTENT_TYPE_OPTIONS}
            onChange={(v) => set("meeting_type", v as string)}
          />

          {/* Custom Prompt — only for models that support it */}
          {getFeatures(cfg.model_preference).custom_prompt && (
            <div>
              <label className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-nexus-text-muted">
                Custom Prompt{" "}
                <span className="normal-case font-normal text-nexus-text-muted">(optional)</span>
              </label>
              <textarea
                value={cfg.custom_prompt}
                onChange={(e) => set("custom_prompt", e.target.value)}
                placeholder="Guide the model — disfluencies, jargon, formatting style..."
                rows={4}
                className="w-full resize-none rounded-lg border border-nexus-border bg-nexus-surface px-3 py-2 text-xs text-nexus-text-secondary outline-none focus:border-nexus-accent-blue transition-colors"
              />
            </div>
          )}

          {/* Key Terms — only for models that support word boost */}
          {getFeatures(cfg.model_preference).key_terms && (
            <div>
              <label className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-nexus-text-muted">
                Key Terms{" "}
                <span className="normal-case font-normal text-nexus-text-muted">(optional)</span>
              </label>
              <input
                type="text"
                value={cfg.key_terms}
                onChange={(e) => set("key_terms", e.target.value)}
                placeholder="CRM, Salesforce, NEXUS — comma separated"
                className="w-full rounded-lg border border-nexus-border bg-nexus-surface px-3 py-2 text-xs text-nexus-text-secondary outline-none focus:border-nexus-accent-blue transition-colors"
              />
              <p className="mt-1 text-[10px] text-nexus-text-muted">
                Boost recognition for technical terms, brand names, or jargon
              </p>
            </div>
          )}

          {/* ── Analyse button — appears when file is selected, disappears when running ── */}
          {phase === "form" && file && (
            <button
              onClick={handleSubmit}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-nexus-accent-blue py-3 text-sm font-semibold text-white hover:bg-accent-blue-80 transition-colors"
            >
              Analyse
              <ArrowRight className="h-4 w-4" />
            </button>
          )}

          {/* ── Inline chat area — processing indicator or transcript ── */}
          {(phase === "processing" || phase === "done" || phase === "error") && (
            <div className="rounded-xl border border-nexus-border bg-nexus-surface overflow-hidden">
              {phase === "processing" && (
                <div className="p-5">
                  {/* Compact processing header */}
                  <div className="flex items-center gap-3 mb-4">
                    <div className="relative h-8 w-8 shrink-0">
                      <div className="absolute inset-0 rounded-full border-2 border-nexus-border" />
                      <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-nexus-accent-blue animate-spin" />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <FileAudio className="h-3.5 w-3.5 text-nexus-accent-blue" />
                      </div>
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-nexus-text-primary">Analysing Recording</p>
                      <p className="text-[10px] text-nexus-text-muted truncate">{file?.name}</p>
                    </div>
                  </div>
                  {/* Step list — built from config */}
                  <InlineStepList cfg={cfg} pipelineStep={pipelineStep} hasVideo={file ? /\.(mp4|webm)$/i.test(file.name) : false} />
                </div>
              )}

              {phase === "done" && quickSegments !== null && (
                /* ── Quick mode: in-memory, no session ── */
                <QuickDoneView
                  segments={quickSegments}
                  meta={quickMeta}
                  diarization={cfg.run_diarization}
                  onReset={resetUpload}
                />
              )}

              {phase === "done" && quickSegments === null && (
                /* ── Full analysis mode: session in DB ── */
                <>
                  <div className="flex items-center justify-between border-b border-nexus-border px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-nexus-text-primary">Transcript</span>
                      <span className="rounded-full bg-emerald-500/10 px-1.5 py-0.5 text-[9px] font-medium text-emerald-400 border border-emerald-500/20">
                        DONE
                      </span>
                      {sessionInfo?.speaker_count ? (
                        <span className="text-[10px] text-nexus-text-muted">
                          {sessionInfo.speaker_count} speaker{sessionInfo.speaker_count !== 1 ? "s" : ""}
                        </span>
                      ) : null}
                      <span className="text-[10px] text-nexus-text-muted">{segments.length} segments</span>
                      {sessionId && (
                        <span
                          className="font-mono text-[9px] text-nexus-text-muted bg-nexus-surface-hover border border-nexus-border rounded px-1.5 py-0.5 cursor-pointer select-all"
                          title="Session ID — click to copy"
                          onClick={() => navigator.clipboard.writeText(sessionId)}
                        >
                          {sessionId.slice(0, 8)}…
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={resetUpload}
                        className="flex items-center gap-1 text-[10px] text-nexus-text-muted hover:text-nexus-text-primary transition-colors"
                      >
                        <RotateCcw className="h-3 w-3" />
                        New Upload
                      </button>
                      {sessionInfo?.session_type !== "lightweight" && (
                        <button
                          onClick={() => navigate(`/sessions/${sessionId}`)}
                          className="flex items-center gap-1 rounded-md bg-nexus-accent-blue px-2.5 py-1 text-[10px] font-semibold text-white hover:bg-accent-blue-80 transition-colors"
                        >
                          View Report
                          <ArrowRight className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                  </div>
                  {entitySummary && (
                    <div className="border-b border-nexus-border px-4 py-3 space-y-2">
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-nexus-text-muted">Extracted Entities</p>
                      <div className="flex flex-wrap gap-2">
                        {entitySummary.people?.length > 0 && (
                          <div className="space-y-1">
                            <p className="text-[9px] uppercase tracking-wider text-nexus-text-muted">People</p>
                            <div className="flex flex-wrap gap-1">
                              {entitySummary.people.map((p, i) => (
                                <span key={i} className="rounded-full bg-nexus-accent-blue/10 border border-nexus-accent-blue/20 px-2 py-0.5 text-[10px] text-nexus-accent-blue">
                                  {p.name} <span className="opacity-60">({p.role})</span>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        {entitySummary.companies?.length > 0 && (
                          <div className="space-y-1">
                            <p className="text-[9px] uppercase tracking-wider text-nexus-text-muted">Companies</p>
                            <div className="flex flex-wrap gap-1">
                              {entitySummary.companies.map((c, i) => (
                                <span key={i} className="rounded-full bg-purple-500/10 border border-purple-500/20 px-2 py-0.5 text-[10px] text-purple-400">
                                  {c.name}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="flex flex-wrap gap-3 pt-1">
                        {entitySummary.topics?.length > 0 && (
                          <span className="text-[10px] text-nexus-text-muted">
                            📌 <span className="text-nexus-text-secondary">{entitySummary.topics.length}</span> topics
                          </span>
                        )}
                        {entitySummary.objections?.length > 0 && (
                          <span className="text-[10px] text-nexus-text-muted">
                            ⚠️ <span className="text-nexus-text-secondary">{entitySummary.objections.length}</span> objection{entitySummary.objections.length !== 1 ? "s" : ""}
                            {entitySummary.objections.filter(o => !o.resolved).length > 0 && (
                              <span className="text-red-400"> ({entitySummary.objections.filter(o => !o.resolved).length} unresolved)</span>
                            )}
                          </span>
                        )}
                        {entitySummary.commitments?.length > 0 && (
                          <span className="text-[10px] text-nexus-text-muted">
                            ✅ <span className="text-nexus-text-secondary">{entitySummary.commitments.length}</span> commitment{entitySummary.commitments.length !== 1 ? "s" : ""}
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                  <div className="p-4">
                    <TranscriptChatPreview segments={segments} />
                  </div>
                </>
              )}

              {phase === "error" && (
                <div className="flex items-center gap-2 p-4 text-xs text-red-400">
                  <AlertCircle className="h-4 w-4 shrink-0" />
                  {errorMsg ?? "Analysis failed. Check logs or try again."}
                  <button
                    onClick={resetUpload}
                    className="ml-auto flex items-center gap-1 text-nexus-text-muted hover:text-nexus-text-primary"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Retry
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ── Lightweight session history ── */}
          {(lightLoading || lightSessions.length > 0) && (
            <div className="rounded-xl border border-nexus-border bg-nexus-surface overflow-hidden">
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-nexus-border">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-nexus-text-muted">
                  Recent Transcriptions
                </h3>
                {lightLoading && <Loader2 className="h-3 w-3 animate-spin text-nexus-text-muted" />}
              </div>
              <div className="divide-y divide-nexus-border">
                {lightSessions.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => loadLightSession(s)}
                    className={`w-full text-left px-4 py-2.5 hover:bg-nexus-surface-hover transition-colors flex items-center justify-between gap-2 ${
                      sessionId === s.id ? "bg-nexus-accent-blue/5 border-l-2 border-nexus-accent-blue" : ""
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <p className="text-[11px] font-medium text-nexus-text-primary truncate">{s.title}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-[9px] text-nexus-text-muted">
                          {new Date(s.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}
                        </span>
                        {s.speaker_count != null && (
                          <span className="text-[9px] text-nexus-text-muted">
                            · {s.speaker_count} speaker{s.speaker_count !== 1 ? "s" : ""}
                          </span>
                        )}
                      </div>
                    </div>
                    <span
                      className="font-mono text-[9px] text-nexus-text-muted bg-nexus-surface-hover border border-nexus-border rounded px-1.5 py-0.5 shrink-0"
                      title={s.id}
                    >
                      {s.id.slice(0, 8)}…
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Right: All settings panels ── */}
        {(() => {
          const feat = getFeatures(cfg.model_preference);
          return (
            <div className="space-y-4">
              {/* Model selection */}
              <SettingsCard title="Transcription Model">
                <ModelRadioCards
                  value={cfg.model_preference}
                  onChange={(v) => set("model_preference", v as string | null)}
                />
              </SettingsCard>

              {/* Transcription settings — only shown when model supports them */}
              {(feat.language || feat.temperature) && (
                <SettingsCard title="Transcription">
                  {feat.language && (
                    <FormSelect
                      label="Language"
                      value={cfg.language}
                      options={LANGUAGE_OPTIONS}
                      onChange={(v) => set("language", v as string | null)}
                    />
                  )}
                  <FormSelect
                    label="Number of Speakers"
                    value={cfg.num_speakers}
                    options={SPEAKER_OPTIONS}
                    onChange={(v) => set("num_speakers", v as number | null)}
                  />
                  <FormSelect
                    label="Translate Output To"
                    value={cfg.translate_to}
                    options={TRANSLATE_OPTIONS}
                    onChange={(v) => set("translate_to", v as string | null)}
                  />
                  {feat.temperature && (
                    <div className="pt-1">
                      <div className="mb-1.5 flex items-center justify-between">
                        <label className="text-[11px] font-medium uppercase tracking-wide text-nexus-text-muted">
                          Temperature
                        </label>
                        <span className="text-[11px] font-mono text-nexus-text-muted">
                          {cfg.temperature !== null ? cfg.temperature.toFixed(2) : "default"}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={cfg.temperature ?? 0}
                        onChange={(e) => set("temperature", parseFloat(e.target.value))}
                        className="w-full accent-nexus-accent-blue"
                      />
                      <div className="flex justify-between text-[10px] text-nexus-text-muted mt-0.5">
                        <span>0.0 — deterministic</span>
                        <span>1.0 — creative</span>
                      </div>
                    </div>
                  )}
                </SettingsCard>
              )}

              {/* Formatting — only shown when model supports any formatting option */}
              {(feat.auto_punctuation || feat.keep_filler_words || feat.text_formatting || feat.multichannel) && (
                <SettingsCard title="Formatting">
                  {feat.auto_punctuation && (
                    <FormToggle
                      label="Auto Punctuation"
                      checked={cfg.auto_punctuation}
                      onChange={(v) => set("auto_punctuation", v)}
                    />
                  )}
                  {feat.multichannel && (
                    <FormToggle
                      label="Multichannel"
                      checked={cfg.multichannel}
                      onChange={(v) => set("multichannel", v)}
                      help="each speaker on a separate audio channel"
                    />
                  )}
                  {feat.keep_filler_words && (
                    <FormToggle
                      label="Keep Filler Words"
                      checked={cfg.keep_filler_words}
                      onChange={(v) => set("keep_filler_words", v)}
                      help='keep "um", "uh" in transcript'
                    />
                  )}
                  {feat.text_formatting && (
                    <FormToggle
                      label="Text Formatting"
                      checked={cfg.text_formatting}
                      onChange={(v) => set("text_formatting", v)}
                      help="dates, numbers, URLs"
                    />
                  )}
                  <FormToggle
                    label="Word Timestamps"
                    checked={cfg.word_timestamps}
                    onChange={(v) => set("word_timestamps", v)}
                    help="per-word timing"
                  />
                </SettingsCard>
              )}

              {/* Analysis settings */}
              <SettingsCard title="Analysis">
                <FormToggle
                  label="Behavioural Analysis"
                  checked={cfg.run_behavioural}
                  onChange={(v) => {
                    set("run_behavioural", v);
                    if (v) {
                      set("run_diarization", true);
                      set("run_entity_extraction", true);
                    }
                  }}
                  help="voice, language, conversation, fusion rules"
                />
                {/* Sub-options only relevant when behavioural analysis is on */}
                {cfg.run_behavioural && (
                  <div className="pt-0.5 pl-1 border-l-2 border-nexus-accent-blue/30 space-y-3">
                    {/* Sensitivity */}
                    <div>
                      <div className="mb-1.5 flex items-center justify-between">
                        <label className="text-[11px] font-medium uppercase tracking-wide text-nexus-text-muted">
                          Sensitivity
                        </label>
                        <span className="text-[11px] text-nexus-text-muted">
                          {cfg.sensitivity < 0.35
                            ? "Low"
                            : cfg.sensitivity > 0.65
                            ? "High"
                            : "Normal"}{" "}
                          <span className="font-mono">({cfg.sensitivity.toFixed(2)})</span>
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={cfg.sensitivity}
                        onChange={(e) => set("sensitivity", parseFloat(e.target.value))}
                        className="w-full accent-nexus-accent-blue"
                      />
                      <div className="flex justify-between text-[10px] text-nexus-text-muted mt-0.5">
                        <span>Low — fewer false positives</span>
                        <span>High</span>
                      </div>
                    </div>
                  </div>
                )}
                <FormToggle
                  label="Speaker Diarization"
                  checked={cfg.run_diarization}
                  onChange={(v) => set("run_diarization", v)}
                  help="separate speakers in transcript"
                />
                <FormToggle
                  label="Entity Extraction"
                  checked={cfg.run_entity_extraction}
                  onChange={(v) => set("run_entity_extraction", v)}
                  help="people, companies, topics, decisions"
                />
              </SettingsCard>
            </div>
          );
        })()}
      </div>

    </div>
  );
}
