import { useState, useCallback, useEffect, useRef, FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Activity, Eye, EyeOff, Loader2, AlertCircle } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { resendVerification } from "../api/client";

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Email not verified state
  const [notVerified, setNotVerified] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);
  const [resendLoading, setResendLoading] = useState(false);
  const [resendMessage, setResendMessage] = useState<string | null>(null);
  const cooldownRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  const handleResend = useCallback(async () => {
    if (resendCooldown > 0 || resendLoading || !email) return;
    setResendLoading(true);
    setResendMessage(null);
    try {
      const result = await resendVerification(email);
      setResendMessage(result.message || "Verification email resent.");
      setResendCooldown(60);
    } catch (err) {
      setResendMessage((err as Error).message || "Failed to resend verification email.");
    } finally {
      setResendLoading(false);
    }
  }, [resendCooldown, resendLoading, email]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setNotVerified(false);
    setResendMessage(null);
    setLoading(true);
    try {
      await login(email, password);
      navigate("/sessions", { replace: true });
    } catch (err) {
      const message = (err as Error).message || "Login failed";
      // Detect 403 "not verified" error from the backend
      if (message.includes("403") && message.toLowerCase().includes("not verified")) {
        setNotVerified(true);
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
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

        {/* Form */}
        <form
          onSubmit={handleSubmit}
          className="rounded-lg border border-nexus-border bg-nexus-surface p-6"
        >
          <h2 className="mb-5 text-center text-sm font-semibold text-nexus-text-primary">
            Sign in to your account
          </h2>

          {error && (
            <div className="mb-4 rounded border border-stress-high-30 bg-stress-high-10 px-3 py-2 text-xs text-nexus-stress-high">
              {error}
            </div>
          )}

          {notVerified && (
            <div className="mb-4 rounded border border-yellow-500/30 bg-yellow-500/10 px-3 py-3 text-xs">
              <div className="mb-2 flex items-center gap-1.5 text-yellow-400">
                <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
                <span className="font-medium">Your email hasn't been verified yet.</span>
              </div>
              <p className="mb-3 text-nexus-text-secondary">
                Please check your inbox for the verification link, or resend it below.
              </p>
              {resendMessage && (
                <p className="mb-2 text-nexus-text-secondary">{resendMessage}</p>
              )}
              <button
                type="button"
                onClick={handleResend}
                disabled={resendCooldown > 0 || resendLoading}
                className="flex w-full items-center justify-center gap-1.5 rounded border border-nexus-border bg-nexus-bg px-3 py-1.5 text-xs font-medium text-nexus-text-secondary transition-colors hover:bg-nexus-border hover:text-nexus-text-primary disabled:opacity-50"
              >
                {resendLoading && <Loader2 className="h-3 w-3 animate-spin" />}
                {resendCooldown > 0
                  ? `Resend verification email (${resendCooldown}s)`
                  : "Resend verification email"}
              </button>
            </div>
          )}

          <div className="mb-4">
            <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
              className="w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue"
              placeholder="you@company.com"
            />
          </div>

          <div className="mb-5">
            <div className="mb-1 flex items-center justify-between">
              <label className="text-xs font-medium text-nexus-text-secondary">Password</label>
              <Link to="/forgot-password" className="text-[10px] text-nexus-accent-blue hover:underline">
                Forgot password?
              </Link>
            </div>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                className="w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 pr-9 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue"
                placeholder="Enter password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-nexus-text-muted hover:text-nexus-text-secondary"
                tabIndex={-1}
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            {loading ? "Signing in..." : "Sign In"}
          </button>

          <p className="mt-4 text-center text-xs text-nexus-text-muted">
            Don't have an account?{" "}
            <Link
              to="/signup"
              className="text-nexus-accent-blue hover:underline"
            >
              Sign Up
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}
