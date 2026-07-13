import { useEffect, useState, FormEvent } from "react";
import { Loader2, KeyRound, Plus, AlertTriangle, Terminal } from "lucide-react";
import {
  listApiKeys, createApiKey, revokeApiKey, ApiKey, ApiKeyCreated,
} from "../../api/client";
import {
  Modal, CopyButton, ConfirmButton, CodeBlock, API_DOCS_BASE, inputCls, primaryBtnCls, cardCls,
} from "./ui";

const DOC_ANALYZE = `curl -X POST ${API_DOCS_BASE}/v1/analyze \\
  -H "Authorization: Bearer nxs_live_..." \\
  -F "file=@recording.mp3" \\
  -F "meeting_type=sales_call"
# → { "session_id": "…", "status": "processing" }`;

const DOC_POLL = `curl -H "Authorization: Bearer nxs_live_..." \\
  ${API_DOCS_BASE}/v1/sessions/<session_id>
# also: /v1/sessions/<id>/signals   /v1/sessions/<id>/report`;

function fmtDate(s: string | null): string {
  if (!s) return "—";
  try {
    return new Date(s).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  } catch {
    return "—";
  }
}

export default function ApiTokensTab() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // create modal
  const [modalOpen, setModalOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newRate, setNewRate] = useState<string>("60");
  const [newExpiry, setNewExpiry] = useState<string>(""); // days; "" = never
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [created, setCreated] = useState<ApiKeyCreated | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listApiKeys();
      setKeys(res.api_keys);
    } catch (err) {
      setError((err as Error).message || "Failed to load API keys");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const openCreate = () => {
    setNewName("");
    setNewRate("60");
    setNewExpiry("");
    setCreateError(null);
    setCreated(null);
    setModalOpen(true);
  };

  const handleCreate = async (e: FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    setCreateError(null);
    try {
      const rate = parseInt(newRate, 10);
      const days = parseInt(newExpiry, 10);
      const res = await createApiKey(
        newName.trim(),
        Number.isFinite(rate) ? rate : undefined,
        Number.isFinite(days) && days > 0 ? days : undefined,
      );
      setCreated(res);
      await load();
    } catch (err) {
      setCreateError((err as Error).message || "Failed to create key");
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (id: string) => {
    try {
      await revokeApiKey(id);
      await load();
    } catch (err) {
      setError((err as Error).message || "Failed to revoke key");
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xs font-semibold text-nexus-text-primary">API Tokens</h2>
          <p className="mt-1 text-[11px] text-nexus-text-muted">
            Use a token as <code className="text-nexus-text-secondary">Authorization: Bearer nxs_live_…</code> to call the API.
          </p>
        </div>
        <button onClick={openCreate} className={primaryBtnCls}>
          <Plus className="h-3.5 w-3.5" /> Create Token
        </button>
      </div>

      {error && (
        <div className="rounded border border-stress-high-30 bg-stress-high-10 px-3 py-2 text-xs text-nexus-stress-high">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center gap-2 py-10 text-xs text-nexus-text-muted">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading tokens…
        </div>
      ) : keys.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-nexus-border py-10 text-center">
          <KeyRound className="h-6 w-6 text-nexus-text-muted" />
          <p className="text-xs text-nexus-text-secondary">No API tokens yet</p>
          <button onClick={openCreate} className="text-xs text-nexus-accent-blue hover:underline">
            Create your first token
          </button>
        </div>
      ) : (
        <div className="divide-y divide-nexus-border overflow-hidden rounded-lg border border-nexus-border">
          {keys.map((k) => (
            <div key={k.id} className="flex items-center justify-between gap-3 bg-nexus-surface px-4 py-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="truncate text-xs font-medium text-nexus-text-primary">{k.name}</span>
                  {!k.active && (
                    <span className="rounded bg-stress-high-10 px-1.5 py-0.5 text-[10px] text-nexus-stress-high">revoked</span>
                  )}
                  {k.active && k.expires_at && new Date(k.expires_at) < new Date() && (
                    <span className="rounded bg-stress-high-10 px-1.5 py-0.5 text-[10px] text-nexus-stress-high">expired</span>
                  )}
                </div>
                <p className="mt-0.5 font-mono text-[11px] text-nexus-text-muted">{k.masked_key}</p>
                <p className="mt-0.5 text-[10px] text-nexus-text-muted">
                  {k.rate_limit_per_min}/min · {k.request_count} calls · last used {fmtDate(k.last_used_at)} · created {fmtDate(k.created_at)} · {k.expires_at ? `expires ${fmtDate(k.expires_at)}` : "never expires"}
                </p>
              </div>
              {k.active && (
                <ConfirmButton onConfirm={() => handleRevoke(k.id)} idle="Revoke" confirm="Confirm revoke" />
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── How to use ── */}
      <div className={cardCls}>
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-nexus-accent-blue" />
          <h3 className="text-xs font-semibold text-nexus-text-primary">How to use</h3>
        </div>
        <p className="text-[11px] text-nexus-text-muted">
          Send your token in the <code className="text-nexus-text-secondary">Authorization</code> header on every
          request. Tokens are rate-limited (default 60 requests/min) and act as your account.
        </p>

        <div className="space-y-1.5">
          <p className="text-[11px] font-medium text-nexus-text-secondary">1. Submit a file for analysis</p>
          <CodeBlock code={DOC_ANALYZE} />
          <p className="text-[10px] text-nexus-text-muted">
            By default the uploaded file is <span className="text-nexus-text-secondary">deleted after analysis</span>{" "}
            (<code>retain_media=false</code>) — only the signals &amp; report are kept. Pass{" "}
            <code>retain_media=true</code> to keep the media for video playback. Other optional fields:{" "}
            <code>callback_url</code> (webhook), <code>config</code> (JSON transcription/analysis options).
          </p>
        </div>

        <div className="space-y-1.5">
          <p className="text-[11px] font-medium text-nexus-text-secondary">2. Poll for results</p>
          <CodeBlock code={DOC_POLL} />
        </div>
      </div>

      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title={created ? "Token created" : "Create API token"}>
        {created ? (
          <div className="space-y-3">
            <div className="flex items-start gap-2 rounded border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-xs text-yellow-500">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              Copy this token now — you won't be able to see it again.
            </div>
            <div className="rounded border border-nexus-border bg-nexus-bg px-3 py-2">
              <p className="break-all font-mono text-xs text-nexus-text-primary">{created.key}</p>
            </div>
            <p className="text-[10px] text-nexus-text-muted">
              {created.expires_at ? `Expires ${fmtDate(created.expires_at)}.` : "This token never expires."}
            </p>
            <div className="flex items-center justify-between">
              <CopyButton value={created.key} label="Copy token" />
              <button onClick={() => setModalOpen(false)} className={primaryBtnCls}>Done</button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleCreate} className="space-y-4">
            {createError && (
              <div className="rounded border border-stress-high-30 bg-stress-high-10 px-3 py-2 text-xs text-nexus-stress-high">
                {createError}
              </div>
            )}
            <div>
              <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">Name</label>
              <input value={newName} onChange={(e) => setNewName(e.target.value)} required className={inputCls} placeholder="e.g. CI pipeline" />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">Rate limit (requests / min)</label>
              <input type="number" min={1} max={6000} value={newRate} onChange={(e) => setNewRate(e.target.value)} className={inputCls} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">Expires in (days)</label>
              <input type="number" min={1} max={3650} value={newExpiry} onChange={(e) => setNewExpiry(e.target.value)} className={inputCls} placeholder="Leave blank for no expiry" />
            </div>
            <div className="flex justify-end gap-2">
              <button type="button" onClick={() => setModalOpen(false)} className="rounded border border-nexus-border px-4 py-2 text-xs text-nexus-text-secondary hover:text-nexus-text-primary">Cancel</button>
              <button type="submit" disabled={creating} className={primaryBtnCls}>
                {creating && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                {creating ? "Creating…" : "Create"}
              </button>
            </div>
          </form>
        )}
      </Modal>
    </div>
  );
}
