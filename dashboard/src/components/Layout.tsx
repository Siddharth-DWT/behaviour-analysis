import { ReactNode, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  BarChart3,
  FolderOpen,
  Settings,
  Activity,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

const NAV_ITEMS = [
  { path: "/sessions", label: "Sessions", icon: FolderOpen },
  { path: "/analytics", label: "Analytics", icon: BarChart3, disabled: true },
  { path: "/settings", label: "Settings", icon: Settings, disabled: true },
];

export default function Layout({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(true);
  const location = useLocation();

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
            const active = location.pathname.startsWith(item.path);
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.disabled ? "#" : item.path}
                className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                  item.disabled
                    ? "cursor-not-allowed opacity-30"
                    : active
                    ? "bg-nexus-accent-blue/10 text-nexus-accent-blue"
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
            <span className="text-xs text-nexus-text-muted">
              Behavioural Analysis System
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs text-nexus-text-muted">
            <span className="rounded bg-nexus-surface-hover px-2 py-1 font-mono">
              v0.1.0
            </span>
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
