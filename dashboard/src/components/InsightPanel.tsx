import type { Signal } from "../api/client";

interface Entities {
  people?: Array<{ name: string; role: string; first_mention_ms: number }>;
  companies?: Array<{ name: string; context: string; first_mention_ms: number }>;
  products_services?: Array<{ name: string; context: string }>;
  topics?: Array<{ name: string; start_ms: number; end_ms: number }>;
  objections?: Array<{ text: string; timestamp_ms: number; resolved: boolean }>;
  commitments?: Array<{ text: string; speaker: string; timestamp_ms: number }>;
  key_terms?: string[];
  [key: string]: unknown;
}

interface SpeakerInfo {
  label: string;
  role?: string;
  avgStress: number;
  avgSentiment: number;
  avgPower: number;
  buyingSignalCount: number;
  objectionCount: number;
  talkTimePct: number;
}

interface InsightPanelProps {
  contentType: string;
  entities: Entities;
  signals: Signal[];
  speakers: SpeakerInfo[];
  speakerRoles: Record<string, string>;
}

function formatTime(ms: number): string {
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ── Sales Call Panel ──

function SalesPanel({ entities, speakers, speakerRoles }: {
  entities: Entities; speakers: SpeakerInfo[]; speakerRoles: Record<string, string>;
}) {
  const STAGES = ["Opening", "Qualification", "Objection", "Pitch", "Closing"];
  const topicNames = (entities.topics || []).map((t) => t.name.toLowerCase());

  const buySignals = speakers.reduce((a, s) => a + s.buyingSignalCount, 0);
  const totalObjections = speakers.reduce((a, s) => a + s.objectionCount, 0);
  const objections = entities.objections || [];
  const commitments = entities.commitments || [];

  // Derive which stages actually occurred from hard evidence
  const activeStages = STAGES.map((stage) => {
    if (stage === "Opening") return true; // always happens

    if (stage === "Qualification") {
      // Only active if topic names explicitly mention discovery/qualification
      // OR if specification questions were asked (buying signals from prospect early)
      const topicMatch = topicNames.some((t) =>
        t.includes("discov") || t.includes("qualif") || t.includes("needs")
      );
      return topicMatch;
    }

    if (stage === "Objection") {
      return totalObjections > 0 || objections.length > 0 ||
        topicNames.some((t) => t.includes("objection"));
    }

    if (stage === "Pitch") {
      return topicNames.some((t) =>
        t.includes("pitch") || t.includes("value") || t.includes("propos") ||
        t.includes("discussion") || t.includes("presentation")
      ) || buySignals > 0;
    }

    if (stage === "Closing") {
      return commitments.length > 0 ||
        topicNames.some((t) => t.includes("clos") || t.includes("wrap") || t.includes("next"));
    }

    return false;
  });
  const lastActive = activeStages.lastIndexOf(true);

  return (
    <div className="space-y-4">
      {/* Deal Progression */}
      <div>
        <div className="text-[11px] text-nexus-text-secondary mb-2 font-medium">
          Deal Progression
        </div>
        <div className="flex gap-1">
          {STAGES.map((stage, i) => {
            const isActive = activeStages[i];
            const isCurrent = i === lastActive;
            const bg = isActive
              ? isCurrent
                ? "var(--accent-blue, #4F8BFF)"
                : "var(--stress-low, #22C55E)"
              : "var(--surface-hover, #2A2A2A)";
            const textColor = isActive ? "#fff" : "var(--text-muted, #666)";
            return (
              <div
                key={stage}
                className="flex-1 rounded px-1.5 py-1.5 text-center text-[10px] font-medium"
                style={{ backgroundColor: bg, color: textColor }}
              >
                {stage}
              </div>
            );
          })}
        </div>
      </div>

      {/* Objections */}
      {objections.length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary mb-1.5 font-medium">
            Objections
          </div>
          <div className="space-y-1.5">
            {objections.map((obj, i) => (
              <div
                key={i}
                className="flex items-start gap-2 rounded border-l-2 bg-nexus-surface-hover p-2 text-xs"
                style={{
                  borderLeftColor: obj.resolved
                    ? "var(--stress-low, #22C55E)"
                    : "var(--stress-high, #EF4444)",
                }}
              >
                <span className="shrink-0 mt-0.5">{obj.resolved ? "✅" : "❌"}</span>
                <div className="flex-1 min-w-0">
                  <span className="text-nexus-text-primary">{obj.text}</span>
                  <span className="ml-2 text-nexus-text-muted font-mono">
                    {formatTime(obj.timestamp_ms)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Buying Signals summary */}
      {buySignals > 0 && (
        <div className="flex items-center gap-2 rounded bg-nexus-surface-hover p-2 border-l-2"
          style={{ borderLeftColor: "var(--stress-low, #22C55E)" }}>
          <span className="text-sm">🟢</span>
          <span className="text-xs text-nexus-text-primary">
            {buySignals} buying signal{buySignals > 1 ? "s" : ""} detected
          </span>
        </div>
      )}

      {/* Commitments */}
      {commitments.length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary mb-1.5 font-medium">
            Commitments
          </div>
          <div className="space-y-1.5">
            {commitments.map((c, i) => (
              <div
                key={i}
                className="rounded border-l-2 bg-nexus-surface-hover p-2 text-xs"
                style={{ borderLeftColor: "var(--accent-blue, #4F8BFF)" }}
              >
                <span className="text-nexus-text-primary">{c.text}</span>
                <span className="ml-2 text-nexus-text-muted">
                  {c.speaker} at {formatTime(c.timestamp_ms)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key terms */}
      {entities.key_terms && entities.key_terms.length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary mb-1.5 font-medium">
            Key Terms
          </div>
          <div className="flex flex-wrap gap-1">
            {entities.key_terms.map((term, i) => (
              <span
                key={i}
                className="rounded-full bg-nexus-surface-hover px-2 py-0.5 text-[10px] text-nexus-text-secondary"
              >
                {term}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Internal / Client Meeting Panel ──

function MeetingPanel({ speakers, entities }: {
  speakers: SpeakerInfo[]; entities: Entities;
}) {
  const totalTalk = speakers.reduce((a, s) => a + s.talkTimePct, 0) || 100;
  const dominant = speakers.find((s) => s.talkTimePct > 70);
  const commitments = (entities.commitments || []) as Array<{ text: string; speaker: string; timestamp_ms: number }>;

  return (
    <div className="space-y-4">
      {/* Participation Balance */}
      <div>
        <div className="text-[11px] text-nexus-text-secondary mb-2 font-medium">
          Participation Balance
        </div>
        <div className="flex h-6 w-full overflow-hidden rounded">
          {speakers.map((s, i) => {
            const pct = Math.max(5, (s.talkTimePct / totalTalk) * 100);
            const colors = [
              "var(--accent-blue, #4F8BFF)",
              "var(--accent-purple, #8B5CF6)",
              "var(--stress-med, #F59E0B)",
              "var(--engagement, #10B981)",
            ];
            return (
              <div
                key={s.label}
                className="flex items-center justify-center text-[10px] font-medium text-white"
                style={{ width: `${pct}%`, backgroundColor: colors[i % colors.length] }}
              >
                {pct > 15 && `${s.role || s.label} ${Math.round(s.talkTimePct)}%`}
              </div>
            );
          })}
        </div>
        {dominant && (
          <div className="mt-1 flex items-center gap-1 text-[10px] text-nexus-alert">
            ⚠️ Meeting dominated by {dominant.role || dominant.label} ({Math.round(dominant.talkTimePct)}%)
          </div>
        )}
      </div>

      {commitments.length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary mb-1.5 font-medium">
            Action Items & Commitments
          </div>
          <div className="space-y-1">
            {commitments.map((c, i) => (
              <div key={i} className="flex items-start gap-1.5 text-xs text-nexus-text-primary">
                <span className="text-nexus-accent-blue mt-0.5">•</span>
                {c.text}
                <span className="text-nexus-text-muted ml-auto shrink-0">
                  {c.speaker}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {entities.key_terms && entities.key_terms.length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary mb-1.5 font-medium">
            Key Terms
          </div>
          <div className="flex flex-wrap gap-1">
            {entities.key_terms.map((term, i) => (
              <span key={i} className="rounded-full bg-nexus-surface-hover px-2 py-0.5 text-[10px] text-nexus-text-secondary">
                {term}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Interview Panel ──

function InterviewPanel({ speakers, entities, speakerRoles }: {
  speakers: SpeakerInfo[]; entities: Entities; speakerRoles: Record<string, string>;
}) {
  const candidate = speakers.find(
    (s) => (speakerRoles[s.label] || s.role)?.toLowerCase() === "candidate"
  );

  return (
    <div className="space-y-4">
      {candidate && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary mb-2 font-medium">
            Candidate Assessment
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-nexus-text-muted">Avg Confidence</span>
              <div className="mt-1 text-lg font-bold" style={{
                color: candidate.avgPower > 0.6
                  ? "var(--stress-low, #22C55E)"
                  : candidate.avgPower > 0.4
                  ? "var(--stress-med, #F59E0B)"
                  : "var(--stress-high, #EF4444)",
              }}>
                {Math.round(candidate.avgPower * 100)}%
              </div>
            </div>
            <div>
              <span className="text-nexus-text-muted">Stress Level</span>
              <div className="mt-1 text-lg font-bold" style={{
                color: candidate.avgStress < 0.3
                  ? "var(--stress-low, #22C55E)"
                  : candidate.avgStress < 0.5
                  ? "var(--stress-med, #F59E0B)"
                  : "var(--stress-high, #EF4444)",
              }}>
                {Math.round(candidate.avgStress * 100)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {entities.key_terms && entities.key_terms.length > 0 && (
        <div>
          <div className="text-[11px] text-nexus-text-secondary mb-1.5 font-medium">
            Key Terms
          </div>
          <div className="flex flex-wrap gap-1">
            {entities.key_terms.map((term, i) => (
              <span key={i} className="rounded-full bg-nexus-surface-hover px-2 py-0.5 text-[10px] text-nexus-text-secondary">
                {term}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Component ──

export default function InsightPanel({
  contentType,
  entities,
  signals,
  speakers,
  speakerRoles,
}: InsightPanelProps) {
  if (!entities || Object.keys(entities).length === 0) {
    return (
      <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
        <p className="text-sm text-nexus-text-muted italic">
          Limited insights available. Entity extraction may not have run for this session.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-nexus-border bg-nexus-surface p-4">
      <h3 className="mb-3 text-sm font-medium text-nexus-text-primary">
        {contentType === "sales_call" ? "Deal Insights" :
         contentType === "interview" ? "Interview Insights" :
         contentType === "client_meeting" ? "Meeting Insights" :
         "Session Insights"}
      </h3>

      {contentType === "sales_call" && (
        <SalesPanel entities={entities} speakers={speakers} speakerRoles={speakerRoles} />
      )}
      {contentType === "interview" && (
        <InterviewPanel speakers={speakers} entities={entities} speakerRoles={speakerRoles} />
      )}
      {(contentType === "internal" || contentType === "client_meeting") && (
        <MeetingPanel speakers={speakers} entities={entities} />
      )}
      {!["sales_call", "interview", "internal", "client_meeting"].includes(contentType) && (
        <MeetingPanel speakers={speakers} entities={entities} />
      )}
    </div>
  );
}
