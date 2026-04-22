import { useRef, useState, useEffect, useCallback } from "react";
import { getAccessToken } from "../api/client";
import type { VideoSignal } from "../api/client";

// ── Signal display configuration ──────────────────────────────────────────────

type SignalConfigEntry = {
  icon: string;
  label: (s: VideoSignal) => string;
  color: string | ((s: VideoSignal) => string);
  category: "face" | "body" | "gaze" | "compound";
};

const SIGNAL_CONFIG: Record<string, SignalConfigEntry> = {
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
  rapport_building: {
    icon: "♥",
    label: () => "Strong Rapport",
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

  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const [durationMs, setDurationMs] = useState(0);
  const [activeSignals, setActiveSignals] = useState<VideoSignal[]>([]);
  const [enabledCategories, setEnabledCategories] = useState<Set<string>>(
    new Set(["face", "body", "gaze", "compound"])
  );
  const [selectedSpeaker, setSelectedSpeaker] = useState("all");
  const [showAnnotated, setShowAnnotated] = useState(false);
  const [annotatedAvailable, setAnnotatedAvailable] = useState(false);

  // Poll until the annotated video is ready (video agent can take 30-90s)
  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    const check = () => {
      fetch(annotatedVideoUrl, { method: "HEAD" })
        .then((r) => {
          if (cancelled) return;
          if (r.ok) {
            setAnnotatedAvailable(true);
          } else {
            setTimeout(check, 5000);
          }
        })
        .catch(() => { if (!cancelled) setTimeout(check, 5000); });
    };
    check();
    return () => { cancelled = true; };
  }, [annotatedVideoUrl, token]);

  const videoUrl = showAnnotated && annotatedAvailable ? annotatedVideoUrl : rawVideoUrl;

  const speakers = [...new Set(signals.map((s) => s.speaker_id).filter(Boolean))];

  const computeActive = useCallback(
    (ms: number): VideoSignal[] =>
      signals.filter((s) => {
        if (ms < s.start_ms || ms > s.end_ms) return false;
        const cfg = SIGNAL_CONFIG[s.signal_type];
        if (!cfg) return false;
        if (!enabledCategories.has(cfg.category)) return false;
        if (selectedSpeaker !== "all" && s.speaker_id !== selectedSpeaker) return false;
        if (s.confidence < 0.3) return false;
        return true;
      }),
    [signals, enabledCategories, selectedSpeaker]
  );

  const tick = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    const ms = video.currentTime * 1000;
    setCurrentTimeMs(ms);
    setActiveSignals(computeActive(ms));
    if (!video.paused && !video.ended) {
      animFrameRef.current = requestAnimationFrame(tick);
    }
  }, [computeActive]);

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

  // Group active signals by speaker for badge display
  const bySpeaker: Record<string, VideoSignal[]> = {};
  for (const s of activeSignals) {
    const key = s.speaker_id || "unknown";
    (bySpeaker[key] ??= []).push(s);
  }

  const hasIncongruence = activeSignals.some((s) =>
    ["tone_face_masking", "head_body_incongruence", "cognitive_overload",
     "stress_suppression", "emotional_suppression"].includes(s.signal_type)
  );

  const playheadPct = durationMs > 0 ? (currentTimeMs / durationMs) * 100 : 0;

  // Deduplicate timeline dots — one dot per signal_type per segment
  const timelineDots = signals.filter((s) => {
    const cfg = SIGNAL_CONFIG[s.signal_type];
    return cfg && enabledCategories.has(cfg.category);
  });

  return (
    <div className="w-full space-y-3">
      {/* Video Playback — badges panel left, video right */}
      <div className="flex overflow-hidden rounded-lg bg-black" style={{ minHeight: 410 }}>

        {/* Left panel: signal badges in the black area */}
        <div className="flex w-48 flex-shrink-0 flex-col justify-start gap-1 overflow-y-auto p-3">
          {activeSignals.length === 0 ? (
            <span className="mt-4 text-center text-[10px] text-white/20">No active signals</span>
          ) : (
            Object.entries(bySpeaker).map(([speaker, sigs]) => (
              <div key={speaker} className="flex flex-col gap-1">
                {speakers.length > 1 && (
                  <span className="mb-0.5 text-[9px] font-semibold uppercase tracking-wider text-white/40">
                    {speaker}
                  </span>
                )}
                {sigs.slice(0, 6).map((s, i) => {
                  const cfg = SIGNAL_CONFIG[s.signal_type];
                  if (!cfg) return null;
                  const color = resolveColor(cfg, s);
                  return (
                    <div
                      key={`${s.signal_type}-${i}`}
                      className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium text-white"
                      style={{
                        backgroundColor: `${color}22`,
                        border: `1px solid ${color}55`,
                      }}
                    >
                      <span className="font-mono text-[11px]">{cfg.icon}</span>
                      <span className="truncate">{cfg.label(s)}</span>
                      {s.confidence >= 0.5 && (
                        <span className="ml-auto shrink-0 text-[10px] opacity-50">
                          {Math.round(s.confidence * 100)}%
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))
          )}
        </div>

        {/* Right panel: video — 9:16 portrait */}
        <div className="relative flex flex-1 items-center justify-center">
          <div className="relative" style={{ aspectRatio: "9/16", maxHeight: 410, width: "auto" }}>
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            className="h-full w-full object-contain"
          />
          {hasIncongruence && (
            <div className="pointer-events-none absolute right-2 top-2 animate-pulse rounded-full bg-red-500/80 px-3 py-1.5 text-xs font-bold text-white backdrop-blur-sm">
              ! Incongruence Detected
            </div>
          )}
          </div>
        </div>

      </div>

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
              <div key={c.key} className="flex items-stretch gap-2" style={{ height: 12 }}>
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
                  style={{ height: 12 }}
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
                  {laneDots.map((s, i) => {
                    const cfg = SIGNAL_CONFIG[s.signal_type];
                    if (!cfg) return null;
                    const effectiveDuration = durationMs > 0 ? durationMs : (signals.length > 0 ? Math.max(...signals.map(x => x.end_ms)) : 1);
                    const color = resolveColor(cfg, s);
                    const left = Math.max(0, Math.min((s.start_ms / effectiveDuration) * 100, 100));
                    const rawWidth = Math.max(((s.end_ms - s.start_ms) / effectiveDuration) * 100, 0.5);
                    const width = Math.min(rawWidth, 100 - left);
                    return (
                      <div
                        key={`${s.signal_type}-${s.start_ms}-${i}`}
                        className="absolute rounded-full opacity-75 hover:opacity-100 transition-opacity"
                        style={{ left: `${left}%`, width: `${width}%`, top: 2, bottom: 2, backgroundColor: color }}
                        title={`${cfg.label(s)} @ ${(s.start_ms / 1000).toFixed(1)}s`}
                        onClick={(e) => { e.stopPropagation(); seekTo(s.start_ms); }}
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Filter controls */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Raw / Annotated toggle */}
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

        {speakers.length > 1 && (
          <select
            value={selectedSpeaker}
            onChange={(e) => setSelectedSpeaker(e.target.value)}
            className="rounded-lg border border-nexus-border bg-nexus-surface px-3 py-1 text-xs text-nexus-text-primary"
          >
            <option value="all">All Speakers</option>
            {speakers.map((spk) => (
              <option key={spk} value={spk}>
                {spk}
              </option>
            ))}
          </select>
        )}
      </div>
    </div>
  );
}
