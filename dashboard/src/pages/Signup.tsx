import { useState, FormEvent, useMemo, useCallback, useEffect, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Activity, Eye, EyeOff, Loader2, Mail, CheckCircle } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { resendVerification } from "../api/client";
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

  // Email verification state
  const [verificationSent, setVerificationSent] = useState(false);
  const [verificationEmail, setVerificationEmail] = useState("");
  const [resendCooldown, setResendCooldown] = useState(0);
  const [resendLoading, setResendLoading] = useState(false);
  const [resendMessage, setResendMessage] = useState<string | null>(null);
  const cooldownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const strength = useMemo(() => getPasswordStrength(password), [password]);
  const cfg = STRENGTH_CONFIG[strength];

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
    if (resendCooldown > 0 || resendLoading) return;
    setResendLoading(true);
    setResendMessage(null);
    try {
      const result = await resendVerification(verificationEmail);
      setResendMessage(result.message || "Verification email resent.");
      setResendCooldown(60);
    } catch (err) {
      setResendMessage((err as Error).message || "Failed to resend verification email.");
    } finally {
      setResendLoading(false);
    }
  }, [resendCooldown, resendLoading, verificationEmail]);

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
      await signup(email, password, fullName, company || undefined);
      navigate("/sessions", { replace: true });
    } catch (err: unknown) {
      const error = err as Error & { requiresVerification?: boolean; email?: string };
      if (error.requiresVerification) {
        setVerificationSent(true);
        setVerificationEmail(error.email || email);
        setResendCooldown(60);
      } else {
        setError(error.message || "Signup failed");
      }
    } finally {
      setLoading(false);
    }
  };

  // Verification sent screen
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
            <div className="mb-4 flex justify-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-nexus-accent-blue/10">
                <Mail className="h-6 w-6 text-nexus-accent-blue" />
              </div>
            </div>

            <h2 className="mb-2 text-sm font-semibold text-nexus-text-primary">
              Check your email
            </h2>
            <p className="mb-5 text-xs text-nexus-text-secondary">
              We've sent a verification link to{" "}
              <span className="font-medium text-nexus-text-primary">
                {verificationEmail}
              </span>
              . Please click the link to verify your account.
            </p>

            {resendMessage && (
              <div className="mb-4 rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-xs text-nexus-text-secondary">
                <CheckCircle className="mr-1 inline h-3 w-3 text-emerald-400" />
                {resendMessage}
              </div>
            )}

            <button
              onClick={handleResend}
              disabled={resendCooldown > 0 || resendLoading}
              className="flex w-full items-center justify-center gap-2 rounded border border-nexus-border bg-nexus-bg px-4 py-2 text-xs font-medium text-nexus-text-secondary transition-colors hover:bg-nexus-border hover:text-nexus-text-primary disabled:opacity-50"
            >
              {resendLoading && <Loader2 className="h-3 w-3 animate-spin" />}
              {resendCooldown > 0
                ? `Resend verification email (${resendCooldown}s)`
                : "Resend verification email"}
            </button>

            <Link
              to="/login"
              className="mt-4 block text-xs text-nexus-accent-blue hover:underline"
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
