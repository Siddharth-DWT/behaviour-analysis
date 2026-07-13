import { useEffect, useState, FormEvent } from "react";
import { Loader2, Webhook, Plus, AlertTriangle, RefreshCw, CheckCircle2, XCircle, Terminal } from "lucide-react";
import {
  listWebhooks, createWebhook, updateWebhook, deleteWebhook, rotateWebhookSecret,
  listWebhookDeliveries, WebhookEndpoint, WebhookDelivery,
} from "../../api/client";
import {
  Modal, CopyButton, ConfirmButton, CodeBlock, API_DOCS_BASE, inputCls, primaryBtnCls, cardCls,
} from "./ui";

const DOC_PAYLOAD = `POST <your endpoint>
X-Nexus-Event: analysis.completed
X-Nexus-Signature: sha256=<hmac>

{
  "event": "analysis.completed",
  "session_id": "…",
  "status": "completed",
  "duration_seconds": 812.0,
  "signal_counts": { "voice": 12, "video": 40, "fusion": 3 },
  "results_url": "${API_DOCS_BASE}/v1/sessions/…",
  "sent_at": "2026-07-13T09:50:00Z"
}`;

const DOC_VERIFY = `# Python — verify the signature (timing-safe)
import hmac, hashlib

def verify(body: bytes, header: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    got = header.split("=", 1)[1]          # strip "sha256="
    return hmac.compare_digest(expected, got)`;

const DOC_ONEOFF = `curl -X POST ${API_DOCS_BASE}/v1/analyze \\
  -H "Authorization: Bearer nxs_live_..." \\
  -F "file=@clip.mp4" \\
  -F "callback_url=https://you.com/hooks/nexus"`;

function fmtTime(s: string | null): string {
  if (!s) return "—";
  try {
    return new Date(s).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return "—";
  }
}

export default function WebhooksTab() {
  const [endpoints, setEndpoints] = useState<WebhookEndpoint[]>([]);
  const [deliveries, setDeliveries] = useState<WebhookDelivery[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // add modal
  const [modalOpen, setModalOpen] = useState(false);
  const [newUrl, setNewUrl] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [saving, setSaving] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  // secret reveal (create or rotate)
  const [revealSecret, setRevealSecret] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [w, d] = await Promise.all([listWebhooks(), listWebhookDeliveries(50)]);
      setEndpoints(w.webhooks);
      setDeliveries(d.deliveries);
    } catch (err) {
      setError((err as Error).message || "Failed to load webhooks");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const openAdd = () => {
    setNewUrl("");
    setNewDesc("");
    setCreateError(null);
    setRevealSecret(null);
    setModalOpen(true);
  };

  const handleCreate = async (e: FormEvent) => {
    e.preventDefault();
    if (!newUrl.trim()) return;
    setSaving(true);
    setCreateError(null);
    try {
      const res = await createWebhook(newUrl.trim(), newDesc.trim() || undefined);
      setRevealSecret(res.secret);
      await load();
    } catch (err) {
      setCreateError((err as Error).message || "Failed to add endpoint");
    } finally {
      setSaving(false);
    }
  };

  const handleToggle = async (ep: WebhookEndpoint) => {
    try {
      await updateWebhook(ep.id, { active: !ep.active });
      await load();
    } catch (err) {
      setError((err as Error).message || "Failed to update endpoint");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteWebhook(id);
      await load();
    } catch (err) {
      setError((err as Error).message || "Failed to delete endpoint");
    }
  };

  const handleRotate = async (id: string) => {
    try {
      const res = await rotateWebhookSecret(id);
      setRevealSecret(res.secret);
      setModalOpen(true);
    } catch (err) {
      setError((err as Error).message || "Failed to rotate secret");
    }
  };

  return (
    <div className="space-y-6">
      {/* ── Endpoints ── */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xs font-semibold text-nexus-text-primary">Webhook Endpoints</h2>
            <p className="mt-1 text-[11px] text-nexus-text-muted">
              Registered URLs receive a signed <code className="text-nexus-text-secondary">POST</code> when your analyses complete.
            </p>
          </div>
          <button onClick={openAdd} className={primaryBtnCls}>
            <Plus className="h-3.5 w-3.5" /> Add Endpoint
          </button>
        </div>

        {error && (
          <div className="rounded border border-stress-high-30 bg-stress-high-10 px-3 py-2 text-xs text-nexus-stress-high">
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-8 text-xs text-nexus-text-muted">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading…
          </div>
        ) : endpoints.length === 0 ? (
          <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-nexus-border py-10 text-center">
            <Webhook className="h-6 w-6 text-nexus-text-muted" />
            <p className="text-xs text-nexus-text-secondary">No webhook endpoints yet</p>
            <button onClick={openAdd} className="text-xs text-nexus-accent-blue hover:underline">Add your first endpoint</button>
          </div>
        ) : (
          <div className="divide-y divide-nexus-border overflow-hidden rounded-lg border border-nexus-border">
            {endpoints.map((ep) => (
              <div key={ep.id} className="flex items-center justify-between gap-3 bg-nexus-surface px-4 py-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="truncate font-mono text-xs text-nexus-text-primary">{ep.url}</span>
                    <span className={`rounded px-1.5 py-0.5 text-[10px] ${ep.active ? "bg-green-500/10 text-green-400" : "bg-nexus-surface-hover text-nexus-text-muted"}`}>
                      {ep.active ? "active" : "paused"}
                    </span>
                  </div>
                  {ep.description && <p className="mt-0.5 text-[11px] text-nexus-text-secondary">{ep.description}</p>}
                  <p className="mt-0.5 text-[10px] text-nexus-text-muted">last delivery {fmtTime(ep.last_delivery_at)}</p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button onClick={() => handleToggle(ep)} className="rounded border border-nexus-border bg-nexus-bg px-2.5 py-1.5 text-xs text-nexus-text-secondary hover:text-nexus-text-primary">
                    {ep.active ? "Pause" : "Enable"}
                  </button>
                  <button onClick={() => handleRotate(ep.id)} className="flex items-center gap-1 rounded border border-nexus-border bg-nexus-bg px-2.5 py-1.5 text-xs text-nexus-text-secondary hover:text-nexus-text-primary">
                    <RefreshCw className="h-3.5 w-3.5" /> Rotate
                  </button>
                  <ConfirmButton onConfirm={() => handleDelete(ep.id)} idle="Delete" confirm="Confirm" />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Recent Deliveries ── */}
      <div className="space-y-3">
        <h2 className="text-xs font-semibold text-nexus-text-primary">Recent Deliveries</h2>
        {deliveries.length === 0 ? (
          <p className="rounded-lg border border-dashed border-nexus-border px-4 py-6 text-center text-[11px] text-nexus-text-muted">
            No deliveries yet.
          </p>
        ) : (
          <div className="divide-y divide-nexus-border overflow-hidden rounded-lg border border-nexus-border">
            {deliveries.map((d) => (
              <div key={d.id} className="flex items-center justify-between gap-3 bg-nexus-surface px-4 py-2.5">
                <div className="flex min-w-0 items-center gap-2">
                  {d.delivered ? (
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-green-400" />
                  ) : (
                    <XCircle className="h-4 w-4 shrink-0 text-nexus-stress-high" />
                  )}
                  <div className="min-w-0">
                    <p className="truncate text-xs text-nexus-text-primary">{d.event}</p>
                    <p className="truncate font-mono text-[10px] text-nexus-text-muted" title={d.last_error || ""}>
                      {d.callback_url}
                    </p>
                  </div>
                </div>
                <div className="shrink-0 text-right">
                  <p className="text-[11px] text-nexus-text-secondary">
                    {d.last_status ?? "—"} · {d.attempts} {d.attempts === 1 ? "try" : "tries"}
                  </p>
                  <p className="text-[10px] text-nexus-text-muted">{fmtTime(d.created_at)}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── How it works ── */}
      <div className={cardCls}>
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-nexus-accent-blue" />
          <h3 className="text-xs font-semibold text-nexus-text-primary">How webhooks work</h3>
        </div>
        <p className="text-[11px] text-nexus-text-muted">
          When an analysis finishes, NEXUS sends a signed <code className="text-nexus-text-secondary">POST</code> to
          each active endpoint above. To send to a one-off URL instead, pass <code>callback_url</code> on
          <code> /v1/analyze</code> — it takes priority over your registered endpoints.
        </p>

        <div className="space-y-1.5">
          <p className="text-[11px] font-medium text-nexus-text-secondary">Payload delivered to your endpoint</p>
          <CodeBlock code={DOC_PAYLOAD} />
          <p className="text-[10px] text-nexus-text-muted">
            A failed run sends <code>event: "analysis.failed"</code> with an <code>error</code> field.
            Delivery retries 3× with backoff; see “Recent Deliveries” above.
          </p>
        </div>

        <div className="space-y-1.5">
          <p className="text-[11px] font-medium text-nexus-text-secondary">Verify the signature</p>
          <p className="text-[10px] text-nexus-text-muted">
            Each request carries <code>X-Nexus-Signature: sha256=&lt;hmac&gt;</code> — an HMAC-SHA256 of the raw
            body using this endpoint's signing secret (shown once on create / rotate).
          </p>
          <CodeBlock code={DOC_VERIFY} />
        </div>

        <div className="space-y-1.5">
          <p className="text-[11px] font-medium text-nexus-text-secondary">One-off callback (no registration)</p>
          <CodeBlock code={DOC_ONEOFF} />
        </div>
      </div>

      {/* Add / secret-reveal modal */}
      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title={revealSecret ? "Signing secret" : "Add webhook endpoint"}>
        {revealSecret ? (
          <div className="space-y-3">
            <div className="flex items-start gap-2 rounded border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-xs text-yellow-500">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              Copy this secret now — it won't be shown again. Use it to verify the <code>X-Nexus-Signature</code> header (HMAC-SHA256).
            </div>
            <div className="rounded border border-nexus-border bg-nexus-bg px-3 py-2">
              <p className="break-all font-mono text-xs text-nexus-text-primary">{revealSecret}</p>
            </div>
            <div className="flex items-center justify-between">
              <CopyButton value={revealSecret} label="Copy secret" />
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
              <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">Endpoint URL</label>
              <input value={newUrl} onChange={(e) => setNewUrl(e.target.value)} required className={inputCls} placeholder="https://example.com/hooks/nexus" />
              <p className="mt-1 text-[10px] text-nexus-text-muted">Must be a public https URL. Private/loopback addresses are rejected.</p>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">Description (optional)</label>
              <input value={newDesc} onChange={(e) => setNewDesc(e.target.value)} className={inputCls} placeholder="e.g. Production receiver" />
            </div>
            <div className="flex justify-end gap-2">
              <button type="button" onClick={() => setModalOpen(false)} className="rounded border border-nexus-border px-4 py-2 text-xs text-nexus-text-secondary hover:text-nexus-text-primary">Cancel</button>
              <button type="submit" disabled={saving} className={primaryBtnCls}>
                {saving && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                {saving ? "Adding…" : "Add Endpoint"}
              </button>
            </div>
          </form>
        )}
      </Modal>
    </div>
  );
}
