const API_BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

// ── Types ──

export interface Session {
  id: string;
  org_id: string;
  title: string;
  session_type: string;
  meeting_type: string;
  status: string;
  media_url: string | null;
  duration_ms: number | null;
  speaker_count: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface Signal {
  id: number;
  session_id: string;
  speaker_id: string | null;
  speaker_label: string | null;
  agent: string;
  signal_type: string;
  value: number | null;
  value_text: string;
  confidence: number;
  window_start_ms: number;
  window_end_ms: number;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface Alert {
  id: string;
  session_id: string;
  speaker_id: string | null;
  speaker_label: string | null;
  alert_type: string;
  severity: string;
  title: string;
  description: string;
  evidence: Record<string, unknown> | null;
  timestamp_ms: number;
  acknowledged: boolean;
  created_at: string;
}

export interface TranscriptSegment {
  id: string;
  session_id: string;
  speaker_id: string | null;
  speaker_label: string | null;
  segment_index: number;
  start_ms: number;
  end_ms: number;
  text: string;
  word_count: number | null;
  sentiment: string | null;
  sentiment_score: number | null;
}

export interface Report {
  id: string;
  session_id: string;
  report_type: string;
  content: {
    executive_summary?: string;
    key_moments?: Array<{
      time_description?: string;
      description?: string;
      significance?: string;
    }>;
    cross_modal_insights?: string[];
    recommendations?: string[];
    raw_response?: string;
  };
  narrative: string | null;
  generated_at: string;
}

export interface SessionDetail {
  session: Session;
  signal_count: number;
  signals_by_agent: Record<string, number>;
  alerts: Alert[];
  alert_count: number;
  has_report: boolean;
  transcript_segment_count: number;
}

// ── API Functions ──

export async function listSessions(params?: {
  limit?: number;
  offset?: number;
  status?: string;
  meeting_type?: string;
}): Promise<{ sessions: Session[]; total: number; limit: number; offset: number }> {
  const search = new URLSearchParams();
  if (params?.limit) search.set("limit", String(params.limit));
  if (params?.offset) search.set("offset", String(params.offset));
  if (params?.status) search.set("status", params.status);
  if (params?.meeting_type) search.set("meeting_type", params.meeting_type);
  const qs = search.toString();
  return request(`/sessions${qs ? `?${qs}` : ""}`);
}

export async function getSession(id: string): Promise<SessionDetail> {
  return request(`/sessions/${id}`);
}

export async function getSignals(
  id: string,
  params?: { agent?: string; signal_type?: string; limit?: number; offset?: number }
): Promise<{ session_id: string; signals: Signal[]; count: number }> {
  const search = new URLSearchParams();
  if (params?.agent) search.set("agent", params.agent);
  if (params?.signal_type) search.set("signal_type", params.signal_type);
  if (params?.limit) search.set("limit", String(params.limit));
  if (params?.offset) search.set("offset", String(params.offset));
  const qs = search.toString();
  return request(`/sessions/${id}/signals${qs ? `?${qs}` : ""}`);
}

export async function getTranscript(
  id: string
): Promise<{ session_id: string; segments: TranscriptSegment[]; count: number }> {
  return request(`/sessions/${id}/transcript`);
}

export async function getReport(
  id: string,
  regenerate = false
): Promise<{ session_id: string; report: Report }> {
  return request(`/sessions/${id}/report${regenerate ? "?regenerate=true" : ""}`);
}

export async function uploadSession(
  file: File,
  title: string,
  meetingType: string
): Promise<{
  session_id: string;
  status: string;
  voice_signal_count: number;
  language_signal_count: number;
  fusion_signal_count: number;
  alert_count: number;
}> {
  const form = new FormData();
  form.append("file", file);
  form.append("title", title);
  form.append("meeting_type", meetingType);

  const res = await fetch(`${API_BASE}/sessions`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Upload failed ${res.status}: ${body}`);
  }
  return res.json();
}

export async function getHealth(): Promise<{
  status: string;
  database: string;
  agents: Record<string, string>;
}> {
  return request("/health");
}
