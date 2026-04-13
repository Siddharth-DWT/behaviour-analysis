import { useState, FormEvent } from "react";
import { Link } from "react-router-dom";
import { Activity, Loader2, CheckCircle } from "lucide-react";
import { forgotPassword } from "../api/client";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await forgotPassword(email.trim().toLowerCase());
      setSent(true);
    } catch (err) {
      setError((err as Error).message || "Something went wrong. Please try again.");
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
          <p className="text-xs text-nexus-text-muted">Behavioural Analysis System</p>
        </div>

        <div className="rounded-lg border border-nexus-border bg-nexus-surface p-6">
          {sent ? (
            /* Success state */
            <div className="text-center space-y-3">
              <CheckCircle className="mx-auto h-10 w-10 text-green-400" />
              <h2 className="text-sm font-semibold text-nexus-text-primary">Check your inbox</h2>
              <p className="text-xs text-nexus-text-muted leading-relaxed">
                If <span className="text-nexus-text-secondary">{email}</span> is registered,
                you'll receive a reset link shortly. The link expires in 1 hour.
              </p>
              <Link
                to="/login"
                className="mt-2 inline-block text-xs text-nexus-accent-blue hover:underline"
              >
                Back to Sign In
              </Link>
            </div>
          ) : (
            /* Email form */
            <>
              <h2 className="mb-1 text-center text-sm font-semibold text-nexus-text-primary">
                Forgot your password?
              </h2>
              <p className="mb-5 text-center text-xs text-nexus-text-muted">
                Enter your email and we'll send you a reset link.
              </p>

              {error && (
                <div className="mb-4 rounded border border-stress-high-30 bg-stress-high-10 px-3 py-2 text-xs text-nexus-stress-high">
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
                    Email address
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

                <button
                  type="submit"
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-2 rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
                >
                  {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                  {loading ? "Sending..." : "Send Reset Link"}
                </button>
              </form>

              <p className="mt-4 text-center text-xs text-nexus-text-muted">
                Remember it?{" "}
                <Link to="/login" className="text-nexus-accent-blue hover:underline">
                  Sign In
                </Link>
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
