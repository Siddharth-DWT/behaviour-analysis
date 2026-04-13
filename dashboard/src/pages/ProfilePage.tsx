import { useState, FormEvent, useEffect } from "react";
import { Activity, Loader2, Check, Eye, EyeOff } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { updateProfile, changePassword } from "../api/client";

export default function ProfilePage() {
  const { user, updateUser } = useAuth();

  // ── Profile form ──
  const [name, setName] = useState(user?.full_name ?? "");
  const [company, setCompany] = useState(user?.company ?? "");
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [profileSuccess, setProfileSuccess] = useState(false);

  useEffect(() => {
    if (user) {
      setName(user.full_name ?? "");
      setCompany(user.company ?? "");
    }
  }, [user]);

  const handleProfileSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setProfileLoading(true);
    setProfileError(null);
    setProfileSuccess(false);
    try {
      const updated = await updateProfile({ full_name: name.trim(), company: company.trim() || undefined });
      if (user) updateUser({ ...user, full_name: updated.full_name, company: updated.company ?? undefined });
      setProfileSuccess(true);
      setTimeout(() => setProfileSuccess(false), 3000);
    } catch (err) {
      setProfileError((err as Error).message || "Failed to update profile");
    } finally {
      setProfileLoading(false);
    }
  };

  // ── Password form ──
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [pwLoading, setPwLoading] = useState(false);
  const [pwError, setPwError] = useState<string | null>(null);
  const [pwSuccess, setPwSuccess] = useState(false);

  const handlePasswordSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (newPw !== confirmPw) {
      setPwError("New passwords do not match.");
      return;
    }
    setPwLoading(true);
    setPwError(null);
    setPwSuccess(false);
    try {
      await changePassword(currentPw, newPw);
      setPwSuccess(true);
      setCurrentPw("");
      setNewPw("");
      setConfirmPw("");
      setTimeout(() => setPwSuccess(false), 3000);
    } catch (err) {
      setPwError((err as Error).message || "Failed to change password");
    } finally {
      setPwLoading(false);
    }
  };

  const inputCls =
    "w-full rounded border border-nexus-border bg-nexus-bg px-3 py-2 text-sm text-nexus-text-primary outline-none placeholder:text-nexus-text-muted focus:border-nexus-accent-blue";

  return (
    <div className="mx-auto max-w-lg space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Activity className="h-5 w-5 text-nexus-accent-blue" />
        <h1 className="text-sm font-semibold text-nexus-text-primary">Profile</h1>
      </div>

      {/* Account info strip */}
      <div className="rounded-lg border border-nexus-border bg-nexus-surface px-4 py-3 flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-nexus-accent-blue text-xs font-bold text-white">
          {(user?.full_name ?? "?")
            .split(" ")
            .map((w) => w[0])
            .join("")
            .toUpperCase()
            .slice(0, 2)}
        </div>
        <div>
          <p className="text-xs font-medium text-nexus-text-primary">{user?.email}</p>
          <p className="text-[10px] text-nexus-text-muted capitalize">{user?.role}</p>
        </div>
      </div>

      {/* ── Update Profile ── */}
      <form
        onSubmit={handleProfileSubmit}
        className="rounded-lg border border-nexus-border bg-nexus-surface p-5 space-y-4"
      >
        <h2 className="text-xs font-semibold text-nexus-text-primary">Update Profile</h2>

        {profileError && (
          <div className="rounded border border-stress-high-30 bg-stress-high-10 px-3 py-2 text-xs text-nexus-stress-high">
            {profileError}
          </div>
        )}
        {profileSuccess && (
          <div className="flex items-center gap-1.5 rounded border border-green-500/30 bg-green-500/10 px-3 py-2 text-xs text-green-400">
            <Check className="h-3.5 w-3.5" /> Profile updated.
          </div>
        )}

        <div>
          <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
            Full Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className={inputCls}
            placeholder="Your name"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
            Company
          </label>
          <input
            type="text"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            className={inputCls}
            placeholder="Your company (optional)"
          />
        </div>

        <button
          type="submit"
          disabled={profileLoading}
          className="flex items-center gap-2 rounded bg-nexus-accent-blue px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
        >
          {profileLoading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {profileLoading ? "Saving..." : "Save Changes"}
        </button>
      </form>

      {/* ── Change Password ── */}
      <form
        onSubmit={handlePasswordSubmit}
        className="rounded-lg border border-nexus-border bg-nexus-surface p-5 space-y-4"
      >
        <h2 className="text-xs font-semibold text-nexus-text-primary">Change Password</h2>

        {pwError && (
          <div className="rounded border border-stress-high-30 bg-stress-high-10 px-3 py-2 text-xs text-nexus-stress-high">
            {pwError}
          </div>
        )}
        {pwSuccess && (
          <div className="flex items-center gap-1.5 rounded border border-green-500/30 bg-green-500/10 px-3 py-2 text-xs text-green-400">
            <Check className="h-3.5 w-3.5" /> Password changed.
          </div>
        )}

        <div>
          <label className="mb-1 block text-xs font-medium text-nexus-text-secondary">
            Current Password
          </label>
          <div className="relative">
            <input
              type={showCurrent ? "text" : "password"}
              value={currentPw}
              onChange={(e) => setCurrentPw(e.target.value)}
              required
              className={`${inputCls} pr-9`}
              placeholder="Current password"
            />
            <button
              type="button"
              onClick={() => setShowCurrent(!showCurrent)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-nexus-text-muted hover:text-nexus-text-secondary"
              tabIndex={-1}
            >
              {showCurrent ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
        </div>

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
              className={`${inputCls} pr-9`}
              placeholder="Min 8 chars, 1 uppercase, 1 number"
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
            Confirm New Password
          </label>
          <input
            type="password"
            value={confirmPw}
            onChange={(e) => setConfirmPw(e.target.value)}
            required
            className={inputCls}
            placeholder="Repeat new password"
          />
        </div>

        <button
          type="submit"
          disabled={pwLoading}
          className="flex items-center gap-2 rounded bg-nexus-accent-blue px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
        >
          {pwLoading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {pwLoading ? "Updating..." : "Change Password"}
        </button>
      </form>
    </div>
  );
}
