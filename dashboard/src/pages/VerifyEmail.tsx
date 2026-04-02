import { useState, useEffect } from "react";
import { useSearchParams, Link } from "react-router-dom";
import {
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { verifyEmail, resendVerification } from "../api/client";

type VerifyStatus = "loading" | "success" | "expired" | "error";

export default function VerifyEmail() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");
  const [status, setStatus] = useState<VerifyStatus>("loading");
  const [message, setMessage] = useState("");
  const [resendEmail, setResendEmail] = useState("");
  const [resendStatus, setResendStatus] = useState<string | null>(null);
  const [resending, setResending] = useState(false);

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setMessage("No verification token provided.");
      return;
    }

    verifyEmail(token)
      .then((data) => {
        setStatus("success");
        setMessage(data.message || "Email verified successfully!");
      })
      .catch((err) => {
        const msg = err.message || "";
        if (msg.toLowerCase().includes("expired")) {
          setStatus("expired");
          setMessage("This verification link has expired.");
        } else if (msg.toLowerCase().includes("already verified")) {
          setStatus("success");
          setMessage("Your email is already verified. You can log in.");
        } else {
          setStatus("error");
          setMessage(msg || "Invalid verification link.");
        }
      });
  }, [token]);

  const handleResend = async () => {
    if (!resendEmail || resending) return;
    setResending(true);
    setResendStatus(null);
    try {
      await resendVerification(resendEmail);
      setResendStatus("Verification email sent! Check your inbox.");
    } catch (err) {
      setResendStatus((err as Error).message || "Failed to resend");
    } finally {
      setResending(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-nexus-bg px-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="mb-8 text-center">
          <div className="mb-3 flex items-center justify-center gap-2">
            <Activity className="h-8 w-8 text-nexus-accent-blue" />
            <span className="font-mono text-2xl font-bold tracking-wider text-nexus-text-primary">
              NEXUS
            </span>
          </div>
          <p className="text-xs text-nexus-text-muted">
            Behavioural Analysis System
          </p>
        </div>

        <div className="rounded-lg border border-nexus-border bg-nexus-surface p-6 text-center">
          {/* Loading */}
          {status === "loading" && (
            <>
              <Loader2 className="mx-auto mb-4 h-12 w-12 animate-spin text-nexus-accent-blue" />
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Verifying your email...
              </h2>
              <p className="text-xs text-nexus-text-muted">
                Please wait a moment.
              </p>
            </>
          )}

          {/* Success */}
          {status === "success" && (
            <>
              <CheckCircle2 className="mx-auto mb-4 h-12 w-12 text-emerald-400" />
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Email Verified
              </h2>
              <p className="mb-6 text-xs text-nexus-text-secondary">
                {message}
              </p>
              <Link
                to="/login"
                className="inline-flex items-center justify-center rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80"
              >
                Go to Login
              </Link>
            </>
          )}

          {/* Expired */}
          {status === "expired" && (
            <>
              <Clock className="mx-auto mb-4 h-12 w-12 text-yellow-400" />
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Link Expired
              </h2>
              <p className="mb-6 text-xs text-nexus-text-secondary">
                {message}
              </p>

              <div className="mb-4">
                <label className="mb-1 block text-left text-xs font-medium text-nexus-text-secondary">
                  Email address
                </label>
                <input
                  type="email"
                  value={resendEmail}
                  onChange={(e) => setResendEmail(e.target.value)}
                  className="w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue"
                  placeholder="you@company.com"
                />
              </div>

              {resendStatus && (
                <div className="mb-4 rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-xs text-nexus-text-secondary">
                  {resendStatus}
                </div>
              )}

              <button
                type="button"
                onClick={handleResend}
                disabled={!resendEmail || resending}
                className="mb-4 flex w-full items-center justify-center gap-2 rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
              >
                {resending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="h-3.5 w-3.5" />
                )}
                {resending ? "Sending..." : "Resend verification email"}
              </button>

              <Link
                to="/login"
                className="text-xs text-nexus-accent-blue hover:underline"
              >
                Back to Login
              </Link>
            </>
          )}

          {/* Error */}
          {status === "error" && (
            <>
              <XCircle className="mx-auto mb-4 h-12 w-12 text-nexus-stress-high" />
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Verification Failed
              </h2>
              <p className="mb-6 text-xs text-nexus-text-secondary">
                {message}
              </p>
              <Link
                to="/login"
                className="text-xs text-nexus-accent-blue hover:underline"
              >
                Back to Login
              </Link>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
