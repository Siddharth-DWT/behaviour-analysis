import { useState, FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Activity, Eye, EyeOff, Loader2, RefreshCw } from "lucide-react";
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
  const [needsVerification, setNeedsVerification] = useState(false);
  const [resendStatus, setResendStatus] = useState<string | null>(null);

  const handleResendVerification = async () => {
    setResendStatus(null);
    try {
      await resendVerification(email);
      setResendStatus("Verification email sent! Check your inbox.");
    } catch (err) {
      setResendStatus((err as Error).message || "Failed to resend");
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setNeedsVerification(false);
    setResendStatus(null);
    setLoading(true);
    try {
      await login(email, password);
      navigate("/sessions", { replace: true });
    } catch (err) {
      const msg = (err as Error).message || "Login failed";
      if (msg.toLowerCase().includes("not verified")) {
        setNeedsVerification(true);
      } else {
        setError(msg);
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

          {needsVerification && (
            <div className="mb-4 rounded border border-yellow-500/30 bg-yellow-500/10 px-3 py-3">
              <p className="mb-2 text-xs text-yellow-400">
                Your email hasn't been verified yet.
              </p>
              {resendStatus && (
                <p className="mb-2 text-xs text-nexus-text-secondary">
                  {resendStatus}
                </p>
              )}
              <button
                type="button"
                onClick={handleResendVerification}
                className="flex items-center gap-1.5 text-xs font-medium text-yellow-400 hover:text-yellow-300"
              >
                <RefreshCw className="h-3 w-3" />
                Resend verification email
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
            <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
              Password
            </label>
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
