import { useState } from "react";
import { ChevronDown, ChevronRight, FileAudio } from "lucide-react";

// ─── Types ───────────────────────────────────────────────

export interface UploadConfig {
  // Content type (also sent as top-level Form field)
  meeting_type: string;

  // Transcription
  language: string | null;
  model_preference: string | null;
  custom_prompt: string;
  key_terms: string; // comma-separated; split on submit

  // Diarization
  num_speakers: number | null; // null = auto
  multichannel: boolean;

  // Formatting
  auto_punctuation: boolean;
  keep_filler_words: boolean;
  text_formatting: boolean;
  word_timestamps: boolean;

  // Model-specific
  temperature: number | null; // 0.0–1.0; AssemblyAI + Whisper only

  // Analysis
  run_behavioural: boolean;
  run_sentiment: boolean;
  run_diarization: boolean;
  run_entity_extraction: boolean;
  run_knowledge_graph: boolean;
  run_video: boolean;
  sensitivity: number; // 0.0 – 1.0

  // Speech understanding
  translate_to: string | null;
}

export const DEFAULT_CONFIG: UploadConfig = {
  meeting_type: "sales_call",
  language: null,
  model_preference: null,
  custom_prompt: "",
  key_terms: "",
  num_speakers: null,
  multichannel: false,
  auto_punctuation: true,
  keep_filler_words: false,
  text_formatting: false,
  word_timestamps: false,
  run_behavioural: true,
  run_sentiment: false,
  run_diarization: true,
  run_entity_extraction: true,
  run_knowledge_graph: true,
  run_video: true,
  sensitivity: 0.5,
  translate_to: null,
  temperature: null,
};

// ─── Option lists ─────────────────────────────────────────

const CONTENT_TYPE_OPTIONS = [
  { value: "sales_call", label: "Sales Call" },
  { value: "client_meeting", label: "Client Meeting" },
  { value: "internal", label: "Internal Meeting" },
  { value: "interview", label: "Interview" },
  { value: "podcast", label: "Podcast" },
];

const LANGUAGE_OPTIONS = [
  { value: null, label: "Automatic Detection" },
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
  { value: "it", label: "Italian" },
  { value: "nl", label: "Dutch" },
  { value: "tr", label: "Turkish" },
  { value: "pl", label: "Polish" },
  { value: "sv", label: "Swedish" },
];

const MODEL_OPTIONS = [
  { value: null, label: "Auto (Best Available)" },
  { value: "parakeet", label: "Parakeet (fastest)" },
  { value: "assemblyai", label: "AssemblyAI Universal" },
  { value: "whisper", label: "Whisper Large V3" },
  { value: "deepgram", label: "Deepgram Nova 3" },
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

// ─── Sub-components ───────────────────────────────────────

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-b border-nexus-border last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left text-xs font-medium text-nexus-text-secondary hover:text-nexus-text-primary transition-colors"
      >
        {title}
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 opacity-60" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 opacity-60" />
        )}
      </button>
      {open && <div className="px-4 pb-4 pt-1 space-y-3">{children}</div>}
    </div>
  );
}

function SelectField({
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
      <label className="mb-1 block text-xs text-nexus-text-muted">{label}</label>
      <select
        value={value === null ? "__null__" : String(value)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "__null__") onChange(null);
          else if (!isNaN(Number(raw)) && options.some((o) => typeof o.value === "number"))
            onChange(Number(raw));
          else onChange(raw);
        }}
        className="w-full rounded border border-nexus-border bg-nexus-surface px-2 py-1.5 text-xs text-nexus-text-secondary outline-none focus:border-nexus-accent-blue"
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

function Toggle({
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
    <label className="flex cursor-pointer items-start gap-2">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-3.5 w-3.5 rounded accent-nexus-accent-blue"
      />
      <span>
        <span className="text-xs text-nexus-text-secondary">{label}</span>
        {help && <span className="ml-1 text-xs text-nexus-text-muted">— {help}</span>}
      </span>
    </label>
  );
}

// ─── Settings summary line (below file name when defaults are changed) ────

function settingsSummary(cfg: UploadConfig): string {
  const parts: string[] = [];
  const ct = CONTENT_TYPE_OPTIONS.find((o) => o.value === cfg.meeting_type);
  if (ct && cfg.meeting_type !== "sales_call") parts.push(ct.label);
  const lang = LANGUAGE_OPTIONS.find((o) => o.value === cfg.language);
  if (lang && cfg.language !== null) parts.push(lang.label);
  const model = MODEL_OPTIONS.find((o) => o.value === cfg.model_preference);
  if (model && cfg.model_preference !== null) parts.push(model.label);
  if (cfg.num_speakers !== null) parts.push(`${cfg.num_speakers} speakers`);
  if (!cfg.run_behavioural) parts.push("transcript only");
  return parts.join(" · ");
}

// ─── Main component ───────────────────────────────────────

interface Props {
  file: File;
  onSubmit: (title: string, config: UploadConfig) => void;
  onCancel: () => void;
  loading: boolean;
}

export default function UploadSettings({ file, onSubmit, onCancel, loading }: Props) {
  const [title, setTitle] = useState(file.name.replace(/\.[^.]+$/, ""));
  const [cfg, setCfg] = useState<UploadConfig>({ ...DEFAULT_CONFIG });

  const set = <K extends keyof UploadConfig>(key: K, val: UploadConfig[K]) =>
    setCfg((prev) => ({ ...prev, [key]: val }));

  const summary = settingsSummary(cfg);
  const fileSizeMB = (file.size / 1024 / 1024).toFixed(1);

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface">
      {/* File header */}
      <div className="flex items-start gap-3 border-b border-nexus-border px-4 py-3">
        <FileAudio className="mt-0.5 h-4 w-4 shrink-0 text-nexus-accent-blue" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium text-nexus-text-primary">{file.name}</p>
          <p className="text-xs text-nexus-text-muted">{fileSizeMB} MB</p>
        </div>
      </div>

      {/* Title */}
      <div className="border-b border-nexus-border px-4 py-3">
        <label className="mb-1 block text-xs text-nexus-text-muted">Title</label>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="w-full rounded border border-nexus-border bg-nexus-bg px-2 py-1.5 text-xs text-nexus-text-primary outline-none focus:border-nexus-accent-blue"
        />
      </div>

      {/* Content type — always visible */}
      <div className="border-b border-nexus-border px-4 py-3">
        <SelectField
          label="Content Type"
          value={cfg.meeting_type}
          options={CONTENT_TYPE_OPTIONS}
          onChange={(v) => set("meeting_type", v as string)}
        />
      </div>

      {/* Settings summary (only shown when non-default settings active) */}
      {summary && (
        <div className="border-b border-nexus-border px-4 py-2">
          <p className="text-xs text-nexus-text-muted">{summary}</p>
        </div>
      )}

      {/* Collapsible sections */}
      <Section title="Transcription Settings">
        <SelectField
          label="Language"
          value={cfg.language}
          options={LANGUAGE_OPTIONS}
          onChange={(v) => set("language", v as string | null)}
        />
        <SelectField
          label="Transcription Model"
          value={cfg.model_preference}
          options={MODEL_OPTIONS}
          onChange={(v) => set("model_preference", v as string | null)}
        />
        <div>
          <label className="mb-1 block text-xs text-nexus-text-muted">Custom Prompt (optional)</label>
          <textarea
            value={cfg.custom_prompt}
            onChange={(e) => set("custom_prompt", e.target.value)}
            placeholder="Add instructions for transcription style..."
            rows={3}
            className="w-full rounded border border-nexus-border bg-nexus-bg px-2 py-1.5 text-xs text-nexus-text-secondary outline-none focus:border-nexus-accent-blue resize-none"
          />
          <p className="mt-0.5 text-xs text-nexus-text-muted">
            Guide how the model handles disfluencies, jargon, and formatting
          </p>
        </div>
        <div>
          <label className="mb-1 block text-xs text-nexus-text-muted">Key Terms (optional)</label>
          <input
            type="text"
            value={cfg.key_terms}
            onChange={(e) => set("key_terms", e.target.value)}
            placeholder="Add domain-specific terms separated by commas..."
            className="w-full rounded border border-nexus-border bg-nexus-bg px-2 py-1.5 text-xs text-nexus-text-secondary outline-none focus:border-nexus-accent-blue"
          />
          <p className="mt-0.5 text-xs text-nexus-text-muted">
            Boost recognition accuracy for technical terms, brand names, or jargon
          </p>
        </div>
      </Section>

      <Section title="Speaker Diarization">
        <SelectField
          label="Number of Speakers"
          value={cfg.num_speakers}
          options={SPEAKER_OPTIONS}
          onChange={(v) => set("num_speakers", v as number | null)}
        />
        <Toggle
          label="Multichannel"
          checked={cfg.multichannel}
          onChange={(v) => set("multichannel", v)}
          help="each speaker is on a separate audio channel"
        />
      </Section>

      <Section title="Transcript Formatting">
        <Toggle
          label="Auto Punctuation"
          checked={cfg.auto_punctuation}
          onChange={(v) => set("auto_punctuation", v)}
        />
        <Toggle
          label="Keep Filler Words"
          checked={cfg.keep_filler_words}
          onChange={(v) => set("keep_filler_words", v)}
          help='keep "um", "uh" in transcript text'
        />
        <Toggle
          label="Text Formatting"
          checked={cfg.text_formatting}
          onChange={(v) => set("text_formatting", v)}
          help="smart formatting for dates, numbers, URLs"
        />
        <Toggle
          label="Word Timestamps"
          checked={cfg.word_timestamps}
          onChange={(v) => set("word_timestamps", v)}
          help="per-word timing in transcript"
        />
      </Section>

      <Section title="Analysis Options">
        <Toggle
          label="Behavioural Analysis"
          checked={cfg.run_behavioural}
          onChange={(v) => set("run_behavioural", v)}
          help="Voice + Language + Conversation + Fusion"
        />
        <Toggle
          label="Sentiment Analysis"
          checked={cfg.run_sentiment}
          onChange={(v) => set("run_sentiment", v)}
          help="Language sentiment pass only (no full behavioural pipeline)"
        />
        <Toggle
          label="Speaker Diarization"
          checked={cfg.run_diarization}
          onChange={(v) => set("run_diarization", v)}
          help="separate speakers in transcript"
        />
        <Toggle
          label="Entity Extraction"
          checked={cfg.run_entity_extraction}
          onChange={(v) => set("run_entity_extraction", v)}
          help="people, companies, topics, objections"
        />
        <Toggle
          label="Knowledge Graph"
          checked={cfg.run_knowledge_graph}
          onChange={(v) => set("run_knowledge_graph", v)}
          help="Neo4j sync for graph queries"
        />
        <div>
          <div className="mb-1 flex items-center justify-between">
            <label className="text-xs text-nexus-text-muted">Sensitivity</label>
            <span className="text-xs text-nexus-text-muted">
              {cfg.sensitivity < 0.35 ? "Low" : cfg.sensitivity > 0.65 ? "High" : "Normal"}
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
          <div className="flex justify-between text-xs text-nexus-text-muted mt-0.5">
            <span>Low</span>
            <span>High</span>
          </div>
          <p className="mt-1 text-xs text-nexus-text-muted">
            Higher sensitivity detects more signals but may produce more false positives
          </p>
        </div>
      </Section>

      <Section title="Speech Understanding">
        <SelectField
          label="Translate to"
          value={cfg.translate_to}
          options={TRANSLATE_OPTIONS}
          onChange={(v) => set("translate_to", v as string | null)}
        />
      </Section>

      {/* Action buttons */}
      <div className="flex items-center justify-between border-t border-nexus-border px-4 py-3">
        <button
          type="button"
          onClick={onCancel}
          disabled={loading}
          className="rounded px-3 py-1.5 text-xs text-nexus-text-muted hover:text-nexus-text-primary transition-colors disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={() => onSubmit(title, cfg)}
          disabled={loading}
          className="flex items-center gap-1.5 rounded bg-nexus-accent-blue px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
        >
          {loading ? "Uploading…" : "Analyse →"}
        </button>
      </div>
    </div>
  );
}
