import { ReactNode, useEffect, useState } from "react";
import { X, Copy, Check } from "lucide-react";

// ── Modal ──────────────────────────────────────────────────────────────────
// Centered card over a dimmed backdrop. Closes on Esc or backdrop click.
export function Modal({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onMouseDown={onClose}
    >
      <div
        className="w-full max-w-md rounded-lg border border-nexus-border bg-nexus-surface p-5 shadow-lg"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-xs font-semibold text-nexus-text-primary">{title}</h3>
          <button
            onClick={onClose}
            className="text-nexus-text-muted hover:text-nexus-text-secondary"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

// ── CopyButton ───────────────────────────────────────────────────────────────
export function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard blocked — ignore */
    }
  };
  return (
    <button
      type="button"
      onClick={onCopy}
      className="flex items-center gap-1.5 rounded border border-nexus-border bg-nexus-bg px-2.5 py-1.5 text-xs text-nexus-text-secondary transition-colors hover:text-nexus-text-primary"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-green-400" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Copied!" : label}
    </button>
  );
}

// ── ConfirmButton ────────────────────────────────────────────────────────────
// Two-step inline confirm. First click arms; second click (within 4s) fires.
export function ConfirmButton({
  onConfirm,
  idle,
  confirm = "Confirm?",
  danger = true,
  disabled = false,
}: {
  onConfirm: () => void;
  idle: ReactNode;
  confirm?: ReactNode;
  danger?: boolean;
  disabled?: boolean;
}) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(t);
  }, [armed]);

  const base =
    "rounded px-2.5 py-1.5 text-xs font-medium transition-colors disabled:opacity-50";
  const tone = danger
    ? "border border-stress-high-30 bg-stress-high-10 text-nexus-stress-high hover:bg-stress-high-30"
    : "border border-nexus-border bg-nexus-bg text-nexus-text-secondary hover:text-nexus-text-primary";

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => (armed ? (setArmed(false), onConfirm()) : setArmed(true))}
      className={`${base} ${tone}`}
    >
      {armed ? confirm : idle}
    </button>
  );
}

// ── CodeBlock ────────────────────────────────────────────────────────────────
// Monospace, horizontally-scrollable snippet with a corner copy button.
export function CodeBlock({ code }: { code: string }) {
  return (
    <div className="relative rounded border border-nexus-border bg-nexus-bg">
      <div className="absolute right-1.5 top-1.5 z-10">
        <CopyButton value={code} label="" />
      </div>
      <pre className="overflow-x-auto p-3 pr-14 text-[11px] leading-relaxed text-nexus-text-secondary">
        <code>{code}</code>
      </pre>
    </div>
  );
}

// Public API base shown in docs (matches the deployed backend / Vite proxy target).
export const API_DOCS_BASE = "https://analysis-be.pathtodeal.com";

// ── Shared class tokens (mirror ProfilePage) ─────────────────────────────────
export const inputCls =
  "w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue";

export const cardCls =
  "rounded-lg border border-nexus-border bg-nexus-surface p-5 space-y-4";

export const primaryBtnCls =
  "flex items-center gap-2 rounded bg-nexus-accent-blue px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50";
