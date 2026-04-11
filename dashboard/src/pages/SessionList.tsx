import { useState, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Upload, Loader2 } from "lucide-react";
import { listSessions } from "../api/client";
import SessionCard from "../components/SessionCard";


export default function SessionList() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [processingIds, setProcessingIds] = useState<Set<string>>(new Set());

  const { data, isLoading, error } = useQuery({
    queryKey: ["sessions"],
    queryFn: () => listSessions({ limit: 50 }),
    refetchInterval: (query) => {
      if (processingIds.size > 0) return 3000;
      const sessions = (query.state.data as any)?.sessions ?? [];
      if (sessions.some((s: any) => s.status === "processing" || s.status === "analysing"))
        return 3000;
      return false;
    },
  });

  // Clear completed sessions from polling set
  useEffect(() => {
    if (!data?.sessions || processingIds.size === 0) return;
    const completed = new Set<string>();
    for (const s of data.sessions) {
      if (
        processingIds.has(s.id) &&
        s.status !== "processing" &&
        s.status !== "analysing"
      ) {
        completed.add(s.id);
      }
    }
    if (completed.size > 0) {
      setProcessingIds((prev) => {
        const next = new Set(prev);
        for (const id of completed) next.delete(id);
        return next;
      });
    }
  }, [data, processingIds]);

  return (
    <div className="mx-auto max-w-4xl">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-nexus-text-primary">Sessions</h1>
          <p className="mt-0.5 text-xs text-nexus-text-muted">
            {data
              ? `${data.total} session${data.total !== 1 ? "s" : ""}`
              : "Loading..."}
          </p>
        </div>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center justify-center py-20 text-sm text-nexus-text-muted">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Loading sessions...
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-stress-high-30 bg-stress-high-10 p-4 text-sm text-nexus-stress-high">
          Failed to load sessions: {(error as Error).message}
        </div>
      )}

      {/* Session list */}
      {data && (
        <div className="space-y-2">
          {data.sessions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-nexus-text-muted">
              <Upload className="mb-3 h-8 w-8 opacity-40" />
              <p className="text-sm">No sessions yet</p>
              <p className="mt-1 text-xs">Upload an audio recording to get started</p>
              <button
                onClick={() => navigate("/upload")}
                className="mt-4 flex items-center gap-1.5 rounded bg-nexus-accent-blue px-4 py-2 text-xs font-medium text-white hover:bg-accent-blue-80 transition-colors"
              >
                <Upload className="h-3.5 w-3.5" />
                Upload Recording
              </button>
            </div>
          ) : (
            data.sessions.map((session) => (
              <SessionCard key={session.id} session={session} />
            ))
          )}
        </div>
      )}
    </div>
  );
}
