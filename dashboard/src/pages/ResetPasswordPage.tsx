import { useState, FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Activity, Eye, EyeOff, Loader2, AlertCircle, CheckCircle } from "lucide-react";
import { resetPassword } from "../api/client";

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get("token") ?? "";

  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  if (!token) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-nexus-bg px-4">
        <div className="w-full max-w-sm rounded-lg border border-nexus-border bg-nexus-surface p-6 text-center space-y-3">
          <AlertCircle className="mx-auto h-10 w-10 text-nexus-stress-high" />
          <p className="text-sm text-nexus-text-primary">Invalid reset link.</p>
          <Link to="/forgot-password" className="text-xs text-nexus-accent-blue hover:underline">
            Request a new one
          </Link>
        </div>
      </div>
    );
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (newPw !== confirmPw) {
      setError("Passwords do not match.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      await resetPassword(token, newPw);
      setSuccess(true);
      setTimeout(() => navigate("/login", { replace: true }), 2500);
    } catch (err) {
      setError((err as Error).message || "Reset failed. The link may have expired.");
    } finally {
      setLoading(false);
    }
  };

  const inputCls =
    "w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue";

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
          {success ? (
            <div className="text-center space-y-3">
              <CheckCircle className="mx-auto h-10 w-10 text-green-400" />
              <h2 className="text-sm font-semibold text-nexus-text-primary">Password updated!</h2>
              <p className="text-xs text-nexus-text-muted">Redirecting you to sign in...</p>
            </div>
          ) : (
            <>
              <h2 className="mb-1 text-center text-sm font-semibold text-nexus-text-primary">
                Create new password
              </h2>
              <p className="mb-5 text-center text-xs text-nexus-text-muted">
                Min 8 characters, 1 uppercase letter, 1 number.
              </p>

              {error && (
                <div className="mb-4 rounded border border-stress-high-30 bg-stress-high-10 px-3 py-2 text-xs text-nexus-stress-high">
                  {error}{" "}
                  {error.toLowerCase().includes("expired") && (
                    <Link to="/forgot-password" className="underline">
                      Request a new link.
                    </Link>
                  )}
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
                    New Password
                  </label>
                  <div className="relative">
                    <input
                      type={showNew ? "text" : "password"}
                      value={newPw}
                      onChange={(e) => setNewPw(e.target.value)}
                      required
                      autoComplete="new-password"
                      className={`${inputCls} pr-9`}
                      placeholder="New password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowNew(!showNew)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-nexus-text-muted hover:text-nexus-text-secondary"
                      tabIndex={-1}
                    >
                      {showNew ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
                    Confirm Password
                  </label>
                  <input
                    type="password"
                    value={confirmPw}
                    onChange={(e) => setConfirmPw(e.target.value)}
                    required
                    autoComplete="new-password"
                    className={inputCls}
                    placeholder="Repeat new password"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-2 rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
                >
                  {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                  {loading ? "Updating..." : "Set New Password"}
                </button>
              </form>

              <p className="mt-4 text-center text-xs text-nexus-text-muted">
                <Link to="/login" className="text-nexus-accent-blue hover:underline">
                  Back to Sign In
                </Link>
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
