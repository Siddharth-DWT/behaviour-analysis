import { ReactNode, useState, useRef, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  Activity,
  FolderOpen,
  ChevronLeft,
  ChevronRight,
  Sun,
  Moon,
  LogOut,
  User,
  ChevronDown,
  UploadCloud,
} from "lucide-react";
import { useTheme } from "../contexts/ThemeContext";
import { useAuth } from "../contexts/AuthContext";

const NAV_ITEMS = [
  { path: "/upload", label: "Upload", icon: UploadCloud },
  { path: "/sessions", label: "Sessions", icon: FolderOpen },
];

function UserInitials({ name }: { name: string }) {
  const initials = name
    .split(" ")
    .map((w) => w[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  return (
    <div className="flex h-7 w-7 items-center justify-center rounded-full bg-nexus-accent-blue text-[10px] font-bold text-white">
      {initials}
    </div>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const menuRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLogout = () => {
    setMenuOpen(false);
    logout();
    navigate("/login", { replace: true });
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`flex flex-col border-r border-nexus-border bg-nexus-surface transition-all duration-200 ${
          collapsed ? "w-14" : "w-48"
        }`}
      >
        {/* Logo */}
        <div className="flex h-14 items-center justify-center border-b border-nexus-border px-3">
          <Activity className="h-6 w-6 text-nexus-accent-blue shrink-0" />
          {!collapsed && (
            <span className="ml-2 font-mono text-sm font-bold tracking-wider text-nexus-accent-blue">
              NEXUS
            </span>
          )}
        </div>

        {/* Nav items */}
        <nav className="flex-1 py-3">
          {NAV_ITEMS.map((item) => {
            // /upload is top-level only; /sessions matches all sub-routes
            const active =
              item.path === "/upload"
                ? location.pathname === "/upload"
                : location.pathname.startsWith(item.path);
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.disabled ? "#" : item.path}
                className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                  item.disabled
                    ? "cursor-not-allowed opacity-30"
                    : active
                    ? "bg-accent-blue-10 text-nexus-accent-blue"
                    : "text-nexus-text-secondary hover:bg-nexus-surface-hover hover:text-nexus-text-primary"
                }`}
                onClick={(e) => item.disabled && e.preventDefault()}
                title={item.label}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {!collapsed && <span>{item.label}</span>}
              </Link>
            );
          })}
        </nav>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex h-10 items-center justify-center border-t border-nexus-border text-nexus-text-muted hover:text-nexus-text-secondary"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </button>
      </aside>

      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Topbar */}
        <header className="flex h-14 items-center justify-between border-b border-nexus-border bg-nexus-surface px-6">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-nexus-accent-blue" />
            <span className="font-mono text-sm font-bold tracking-wider text-nexus-text-primary">
              NEXUS
            </span>
            <span className="hidden sm:block text-xs text-nexus-text-muted">
              Behavioural Analysis System
            </span>
          </div>
          <div className="flex items-center gap-2 sm:gap-3 text-xs text-nexus-text-muted">
            <button
              onClick={toggleTheme}
              className="rounded-lg p-1.5 text-nexus-text-muted hover:bg-nexus-surface-hover hover:text-nexus-text-primary transition-colors"
              title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
            <span className="hidden sm:block rounded bg-nexus-surface-hover px-2 py-1 font-mono">
              v0.1.0
            </span>

            {/* User menu */}
            {user && (
              <div className="relative" ref={menuRef}>
                <button
                  onClick={() => setMenuOpen(!menuOpen)}
                  className="flex items-center gap-2 rounded-lg px-2 py-1 transition-colors hover:bg-nexus-surface-hover"
                >
                  <UserInitials name={user.full_name} />
                  <span className="hidden sm:block max-w-[120px] truncate text-xs text-nexus-text-secondary">
                    {user.full_name}
                  </span>
                  <ChevronDown
                    className={`h-3 w-3 text-nexus-text-muted transition-transform ${
                      menuOpen ? "rotate-180" : ""
                    }`}
                  />
                </button>

                {menuOpen && (
                  <div className="absolute right-0 top-full z-50 mt-1 w-48 rounded-lg border border-nexus-border bg-nexus-surface py-1 shadow-lg">
                    <div className="border-b border-nexus-border px-3 py-2">
                      <p className="truncate text-xs font-medium text-nexus-text-primary">
                        {user.full_name}
                      </p>
                      <p className="truncate text-[10px] text-nexus-text-muted">
                        {user.email}
                      </p>
                    </div>

                    <button
                      onClick={() => { setMenuOpen(false); navigate("/profile"); }}
                      className="flex w-full items-center gap-2 px-3 py-2 text-xs text-nexus-text-secondary hover:bg-nexus-surface-hover"
                    >
                      <User className="h-3.5 w-3.5" />
                      Profile
                    </button>
                    <button
                      onClick={() => setMenuOpen(false)}
                      className="flex w-full items-center gap-2 px-3 py-2 text-xs text-nexus-text-secondary hover:bg-nexus-surface-hover"
                    >
                      <Settings className="h-3.5 w-3.5" />
                      Settings
                    </button>

                    <div className="border-t border-nexus-border" />

                    <button
                      onClick={handleLogout}
                      className="flex w-full items-center gap-2 px-3 py-2 text-xs text-nexus-stress-high hover:bg-nexus-surface-hover"
                    >
                      <LogOut className="h-3.5 w-3.5" />
                      Sign Out
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto bg-nexus-bg p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
