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
<<<<<<< HEAD

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
=======
  const [status, setStatus] = useState<VerifyStatus>("loading");
  const [message, setMessage] = useState("");
  const [resendEmail, setResendEmail] = useState("");
  const [resendStatus, setResendStatus] = useState<string | null>(null);
  const [resending, setResending] = useState(false);

  useEffect(() => {
    if (!token) {
      setStatus("error");
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
      setMessage("No verification token provided.");
      return;
    }

<<<<<<< HEAD
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
=======
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
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
        }
      });
  }, [token]);

<<<<<<< HEAD
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
=======
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
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172

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
<<<<<<< HEAD
          {/* Loading state */}
          {state === "loading" && (
            <>
              <div className="mb-4 flex justify-center">
                <Loader2 className="h-10 w-10 animate-spin text-nexus-accent-blue" />
              </div>
=======
          {/* Loading */}
          {status === "loading" && (
            <>
              <Loader2 className="mx-auto mb-4 h-12 w-12 animate-spin text-nexus-accent-blue" />
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Verifying your email...
              </h2>
              <p className="text-xs text-nexus-text-muted">
<<<<<<< HEAD
                Please wait while we verify your email address.
=======
                Please wait a moment.
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
              </p>
            </>
          )}

<<<<<<< HEAD
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
=======
          {/* Success */}
          {status === "success" && (
            <>
              <CheckCircle2 className="mx-auto mb-4 h-12 w-12 text-emerald-400" />
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Email Verified
              </h2>
              <p className="mb-6 text-xs text-nexus-text-secondary">
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
                {message}
              </p>
              <Link
                to="/login"
<<<<<<< HEAD
                className="inline-flex w-full items-center justify-center rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80"
=======
                className="inline-flex items-center justify-center rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80"
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
              >
                Go to Login
              </Link>
            </>
          )}

<<<<<<< HEAD
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
=======
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
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
                <input
                  type="email"
                  value={resendEmail}
                  onChange={(e) => setResendEmail(e.target.value)}
<<<<<<< HEAD
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
=======
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
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
              </button>

              <Link
                to="/login"
                className="text-xs text-nexus-accent-blue hover:underline"
              >
                Back to Login
              </Link>
            </>
          )}

<<<<<<< HEAD
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
=======
          {/* Error */}
          {status === "error" && (
            <>
              <XCircle className="mx-auto mb-4 h-12 w-12 text-nexus-stress-high" />
              <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
                Verification Failed
              </h2>
              <p className="mb-6 text-xs text-nexus-text-secondary">
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
                {message}
              </p>
              <Link
                to="/login"
<<<<<<< HEAD
                className="inline-flex w-full items-center justify-center rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80"
=======
                className="text-xs text-nexus-accent-blue hover:underline"
>>>>>>> 6a3ef65253290b82d23c749daf2cc5b90f16c172
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
