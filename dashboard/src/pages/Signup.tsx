import { useState, useEffect, FormEvent, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Activity, Eye, EyeOff, Loader2, CheckCircle2, Mail, RefreshCw } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { resendVerification } from "../api/client";

type PasswordStrength = "weak" | "fair" | "strong";

function getPasswordStrength(pw: string): PasswordStrength {
  if (pw.length < 8) return "weak";
  const hasUpper = /[A-Z]/.test(pw);
  const hasDigit = /\d/.test(pw);
  if (hasUpper && hasDigit) return "strong";
  return "fair";
}

const STRENGTH_CONFIG: Record<
  PasswordStrength,
  { label: string; color: string; bg: string; width: string }
> = {
  weak: {
    label: "Weak — need 8+ characters",
    color: "text-nexus-stress-high",
    bg: "bg-nexus-stress-high",
    width: "w-1/3",
  },
  fair: {
    label: "Fair — add number + uppercase",
    color: "text-yellow-400",
    bg: "bg-yellow-400",
    width: "w-2/3",
  },
  strong: {
    label: "Strong",
    color: "text-emerald-400",
    bg: "bg-emerald-400",
    width: "w-full",
  },
};

export default function Signup() {
  const navigate = useNavigate();
  const { signup } = useAuth();

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [company, setCompany] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [verificationSent, setVerificationSent] = useState(false);
  const [sentEmail, setSentEmail] = useState("");
  const [resendCooldown, setResendCooldown] = useState(0);
  const [resendStatus, setResendStatus] = useState<string | null>(null);

  const strength = useMemo(() => getPasswordStrength(password), [password]);
  const cfg = STRENGTH_CONFIG[strength];

  // Cooldown timer for resend button
  useEffect(() => {
    if (resendCooldown <= 0) return;
    const interval = setInterval(() => {
      setResendCooldown((prev) => prev - 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [resendCooldown]);

  const handleResend = async () => {
    if (resendCooldown > 0) return;
    setResendStatus(null);
    try {
      await resendVerification(sentEmail);
      setResendStatus("Verification email sent!");
      setResendCooldown(60);
    } catch (err) {
      setResendStatus((err as Error).message || "Failed to resend");
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    if (strength !== "strong") {
      setError(
        "Password must be at least 8 characters with 1 uppercase letter and 1 number"
      );
      return;
    }

    setLoading(true);
    try {
      const result = await signup(email, password, fullName, company || undefined);
      if (result.requiresVerification) {
        setSentEmail(email);
        setVerificationSent(true);
      } else {
        navigate("/sessions", { replace: true });
      }
    } catch (err) {
      setError((err as Error).message || "Signup failed");
    } finally {
      setLoading(false);
    }
  };

  if (verificationSent) {
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
            <CheckCircle2 className="mx-auto mb-4 h-12 w-12 text-emerald-400" />
            <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
              Check your email!
            </h2>
            <div className="mb-1 flex items-center justify-center gap-1.5 text-nexus-text-muted">
              <Mail className="h-3.5 w-3.5" />
              <span className="text-xs">{sentEmail}</span>
            </div>
            <p className="mb-6 text-xs text-nexus-text-secondary">
              We've sent a verification link to your email. Click the link to
              activate your account.
            </p>

            {resendStatus && (
              <div className="mb-4 rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-xs text-nexus-text-secondary">
                {resendStatus}
              </div>
            )}

            <button
              type="button"
              onClick={handleResend}
              disabled={resendCooldown > 0}
              className="mb-4 flex w-full items-center justify-center gap-2 rounded border border-nexus-border bg-nexus-bg px-4 py-2 text-xs font-medium text-nexus-text-primary transition-colors hover:bg-nexus-border disabled:opacity-50"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              {resendCooldown > 0
                ? `Resend in ${resendCooldown}s`
                : "Resend verification email"}
            </button>

            <Link
              to="/login"
              className="text-xs text-nexus-accent-blue hover:underline"
            >
              Back to Login
            </Link>
          </div>
        </div>
      </div>
    );
  }

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
            Create your account
          </h2>

          {error && (
            <div className="mb-4 rounded border border-stress-high-30 bg-stress-high-10 px-3 py-2 text-xs text-nexus-stress-high">
              {error}
            </div>
          )}

          <div className="mb-4">
            <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
              Full Name
            </label>
            <input
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              required
              autoComplete="name"
              className="w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue"
              placeholder="Jane Smith"
            />
          </div>

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

          <div className="mb-4">
            <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
              Company{" "}
              <span className="text-nexus-text-muted">(optional)</span>
            </label>
            <input
              type="text"
              value={company}
              onChange={(e) => setCompany(e.target.value)}
              autoComplete="organization"
              className="w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue"
              placeholder="Acme Inc."
            />
          </div>

          <div className="mb-2">
            <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="new-password"
                className="w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 pr-9 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue"
                placeholder="Min 8 chars, 1 uppercase, 1 number"
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

          {/* Strength indicator */}
          {password.length > 0 && (
            <div className="mb-4">
              <div className="mb-1 h-1 w-full rounded bg-nexus-border">
                <div
                  className={`h-1 rounded transition-all ${cfg.bg} ${cfg.width}`}
                />
              </div>
              <p className={`text-[10px] ${cfg.color}`}>{cfg.label}</p>
            </div>
          )}

          <div className="mb-5">
            <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
              Confirm Password
            </label>
            <input
              type={showPassword ? "text" : "password"}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              autoComplete="new-password"
              className={`w-full rounded border bg-nexus-bg px-3 py-2 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue ${
                confirmPassword && confirmPassword !== password
                  ? "border-nexus-stress-high"
                  : "border-nexus-border"
              }`}
              placeholder="Re-enter password"
            />
            {confirmPassword && confirmPassword !== password && (
              <p className="mt-1 text-[10px] text-nexus-stress-high">
                Passwords do not match
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            {loading ? "Creating account..." : "Create Account"}
          </button>

          <p className="mt-4 text-center text-xs text-nexus-text-muted">
            Already have an account?{" "}
            <Link
              to="/login"
              className="text-nexus-accent-blue hover:underline"
            >
              Sign In
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}
