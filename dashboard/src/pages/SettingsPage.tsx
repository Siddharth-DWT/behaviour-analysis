import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Settings, User, KeyRound, Webhook } from "lucide-react";
import ProfileTab from "../components/settings/ProfileTab";
import ApiTokensTab from "../components/settings/ApiTokensTab";
import WebhooksTab from "../components/settings/WebhooksTab";

type TabId = "profile" | "api-tokens" | "webhooks";

const TABS: { id: TabId; label: string; icon: typeof User }[] = [
  { id: "profile", label: "Profile", icon: User },
  { id: "api-tokens", label: "API Tokens", icon: KeyRound },
  { id: "webhooks", label: "Webhooks", icon: Webhook },
];

export default function SettingsPage() {
  const [params, setParams] = useSearchParams();
  const initial = (params.get("tab") as TabId) || "profile";
  const [tab, setTab] = useState<TabId>(
    TABS.some((t) => t.id === initial) ? initial : "profile"
  );

  const selectTab = (id: TabId) => {
    setTab(id);
    setParams({ tab: id }, { replace: true });
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Settings className="h-5 w-5 text-nexus-accent-blue" />
        <h1 className="text-sm font-semibold text-nexus-text-primary">Settings</h1>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-nexus-border">
        {TABS.map((t) => {
          const Icon = t.icon;
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => selectTab(t.id)}
              className={`flex items-center gap-1.5 border-b-2 px-3 py-2 text-xs font-medium transition-colors ${
                active
                  ? "border-nexus-accent-blue text-nexus-accent-blue"
                  : "border-transparent text-nexus-text-secondary hover:text-nexus-text-primary"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {t.label}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      {tab === "profile" && <ProfileTab />}
      {tab === "api-tokens" && <ApiTokensTab />}
      {tab === "webhooks" && <WebhooksTab />}
    </div>
  );
}
