import { useState, useEffect, useCallback, useRef } from "react";
import { useSearchParams, Link } from "react-router-dom";
import {
  Activity,
  Loader2,
  CheckCircle,
  AlertCircle,
  XCircle,
} from "lucide-react";
import { verifyEmail, resendVerification } from "../api/client";

type VerifyState = "loading" | "success" | "expired" | "error";

export default function VerifyEmail() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");

  const [state, setState] = useState<VerifyState>("loading");
  const [message, setMessage] = useState("");
  const [resendEmail, setResendEmail] = useState("");
  const [resendCooldown, setResendCooldown] = useState(0);
  const [resendLoading, setResendLoading] = useState(false);
  const [resendMessage, setResendMessage] = useState<string | null>(null);
  const cooldownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const verifiedRef = useRef(false);

  // Cooldown timer
  useEffect(() => {
    if (resendCooldown <= 0) {
      if (cooldownRef.current) {
        clearInterval(cooldownRef.current);
        cooldownRef.current = null;
      }
      return;
    }
    cooldownRef.current = setInterval(() => {
      setResendCooldown((prev) => {
        if (prev <= 1) {
          if (cooldownRef.current) clearInterval(cooldownRef.current);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => {
      if (cooldownRef.current) clearInterval(cooldownRef.current);
    };
  }, [resendCooldown > 0]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!token) {
      setState("error");
      setMessage("No verification token provided.");
      return;
    }

    // Prevent double-call in React strict mode
    if (verifiedRef.current) return;
    verifiedRef.current = true;

    verifyEmail(token)
      .then((data) => {
        setState("success");
        setMessage(data.message || "Email verified successfully!");
      })
      .catch((err) => {
        const msg = (err as Error).message || "Verification failed";
        if (
          msg.toLowerCase().includes("expired") ||
          msg.toLowerCase().includes("expire")
        ) {
          setState("expired");
          setMessage(msg);
        } else {
          setState("error");
          setMessage(msg);
        }
      });
  }, [token]);

  const handleResend = useCallback(async () => {
    if (resendCooldown > 0 || resendLoading || !resendEmail) return;
    setResendLoading(true);
    setResendMessage(null);
    try {
      const result = await resendVerification(resendEmail);
      setResendMessage(result.message || "Verification email resent.");
      setResendCooldown(60);
    } catch (err) {
      setResendMessage(
        (err as Error).message || "Failed to resend verification email."
      );
    } finally {
      setResendLoading(false);
    }
  }, [resendCooldown, resendLoading, resendEmail]);

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
          {/* Loading state */}
          {state === "loading" && (
            <>
              <div className="mb-4 flex justify-center">
                <Loader2 className="h-10 w-10 animate-spin text-nexus-accent-blue" />
              </div>
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Verifying your email...
              </h2>
              <p className="text-xs text-nexus-text-muted">
                Please wait while we verify your email address.
              </p>
            </>
          )}

          {/* Success state */}
          {state === "success" && (
            <>
              <div className="mb-4 flex justify-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/10">
                  <CheckCircle className="h-6 w-6 text-emerald-400" />
                </div>
              </div>
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Email verified!
              </h2>
              <p className="mb-5 text-xs text-nexus-text-secondary">
                {message}
              </p>
              <Link
                to="/login"
                className="inline-flex w-full items-center justify-center rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80"
              >
                Go to Login
              </Link>
            </>
          )}

          {/* Expired state */}
          {state === "expired" && (
            <>
              <div className="mb-4 flex justify-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-yellow-500/10">
                  <AlertCircle className="h-6 w-6 text-yellow-400" />
                </div>
              </div>
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Verification link expired
              </h2>
              <p className="mb-5 text-xs text-nexus-text-secondary">
                {message}
              </p>

              <div className="mb-3">
                <input
                  type="email"
                  value={resendEmail}
                  onChange={(e) => setResendEmail(e.target.value)}
                  placeholder="Enter your email to resend"
                  className="w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue"
                />
              </div>

              {resendMessage && (
                <p className="mb-3 text-xs text-nexus-text-secondary">
                  {resendMessage}
                </p>
              )}

              <button
                onClick={handleResend}
                disabled={resendCooldown > 0 || resendLoading || !resendEmail}
                className="mb-3 flex w-full items-center justify-center gap-2 rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
              >
                {resendLoading && <Loader2 className="h-3 w-3 animate-spin" />}
                {resendCooldown > 0
                  ? `Resend verification email (${resendCooldown}s)`
                  : "Resend verification email"}
              </button>

              <Link
                to="/login"
                className="text-xs text-nexus-accent-blue hover:underline"
              >
                Back to Login
              </Link>
            </>
          )}

          {/* Error state */}
          {state === "error" && (
            <>
              <div className="mb-4 flex justify-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-500/10">
                  <XCircle className="h-6 w-6 text-nexus-stress-high" />
                </div>
              </div>
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Verification failed
              </h2>
              <p className="mb-5 text-xs text-nexus-text-secondary">
                {message}
              </p>
              <Link
                to="/login"
                className="inline-flex w-full items-center justify-center rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80"
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
