const API_BASE = "/api";

// ── Auth token management ──
// Access token is stored in memory (not localStorage) for XSS protection.
// It's set by AuthContext after login/signup/refresh.

let _accessToken: string | null = null;

export function setAccessToken(token: string | null) {
  _accessToken = token;
}

export function getAccessToken(): string | null {
  return _accessToken;
}

// ── Core request function with auth ──

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options?.headers as Record<string, string>),
  };

  if (_accessToken) {
    headers["Authorization"] = `Bearer ${_accessToken}`;
  }

  // Don't set Content-Type for FormData (browser sets it with boundary)
  if (options?.body instanceof FormData) {
    delete headers["Content-Type"];
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (res.status === 401 && _accessToken) {
    // Try to refresh the token
    const refreshToken = localStorage.getItem("nexus_refresh_token");
    if (refreshToken) {
      try {
        const refreshRes = await fetch(`${API_BASE}/auth/refresh`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (refreshRes.ok) {
          const data = await refreshRes.json();
          _accessToken = data.access_token;
          localStorage.setItem("nexus_refresh_token", data.refresh_token);

          // Retry original request with new token
          const retryHeaders: Record<string, string> = { ...headers, Authorization: `Bearer ${data.access_token}` };
          if (options?.body instanceof FormData) {
            delete retryHeaders["Content-Type"];
          }
          const retryRes = await fetch(`${API_BASE}${path}`, {
            ...options,
            headers: retryHeaders,
          });
          if (!retryRes.ok) {
            const body = await retryRes.text();
            throw new Error(`API ${retryRes.status}: ${body}`);
          }
          return retryRes.json();
        }
      } catch {
        // Refresh failed — clear auth and redirect to login
      }
      _accessToken = null;
      localStorage.removeItem("nexus_refresh_token");
      window.location.href = "/login";
      throw new Error("Session expired");
    }
  }

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
    entities?: {
      people?: Array<{ name: string; role: string; first_mention_ms: number }>;
      companies?: Array<{ name: string; context: string; first_mention_ms: number }>;
      products_services?: Array<{ name: string; context: string }>;
      topics?: Array<{ name: string; start_ms: number; end_ms: number }>;
      objections?: Array<{ text: string; timestamp_ms: number; resolved: boolean }>;
      commitments?: Array<{ text: string; speaker: string; timestamp_ms: number }>;
      key_terms?: string[];
    };
    signal_graph?: {
      nodes: Array<Record<string, unknown>>;
      edges: Array<Record<string, unknown>>;
      stats?: Record<string, unknown>;
    };
    key_paths?: Array<{
      nodes: Array<{ id: string; type: string; label: string; agent: string | null; confidence: number | null; timestamp_ms: number }>;
      description: string;
      score: number;
    }>;
    graph_analytics?: Record<string, unknown>;
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
  const result = await request<{ session_id: string; report: Report }>(
    `/sessions/${id}/report${regenerate ? "?regenerate=true" : ""}`
  );
  // API may return content as JSON string — parse it
  if (result.report && typeof result.report.content === "string") {
    try {
      const parsed = JSON.parse(result.report.content as unknown as string);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (result.report as any).content = parsed;
    } catch {
      // If parse fails, wrap raw string as executive_summary
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (result.report as any).content = { executive_summary: result.report.content };
    }
  }
  return result;
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

  return request("/sessions", { method: "POST", body: form });
}

export async function getHealth(): Promise<{
  status: string;
  database: string;
  agents: Record<string, string>;
}> {
  return request("/health");
}
