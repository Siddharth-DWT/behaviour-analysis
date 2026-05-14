import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, Loader2, ChevronDown } from "lucide-react";
import { listSessions, getSession } from "../api/client";
import SessionCard from "../components/SessionCard";
import type { Session } from "../api/client";

const PAGE_SIZE = 50;

export default function SessionList() {
  const navigate = useNavigate();

  const [sessions, setSessions] = useState<Session[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [offset, setOffset] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isLoadingMore, setIsLoadingMore] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [processingIds, setProcessingIds] = useState<Set<string>>(new Set());

  const fetchSessions = async (currentOffset: number, append: boolean) => {
    try {
      const data = await listSessions({ limit: PAGE_SIZE, offset: currentOffset });
      setTotal(data.total);
      setSessions((prev) => append ? [...prev, ...data.sessions] : data.sessions);

      // Track any processing sessions for polling
      const newProcessing = data.sessions
        .filter((s: Session) => s.status === "processing" || s.status === "analysing")
        .map((s: Session) => s.id);
      if (newProcessing.length > 0) {
        setProcessingIds((prev) => new Set([...prev, ...newProcessing]));
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setIsLoading(false);
      setIsLoadingMore(false);
    }
  };

  // Initial load
  useEffect(() => {
    fetchSessions(0, false);
  }, []);

  // Poll only the specific sessions that are still processing — one request per session,
  // not a full list reload. Remove each ID individually when it finishes.
  useEffect(() => {
    if (processingIds.size === 0) return;
    const interval = setInterval(async () => {
      const results = await Promise.allSettled(
        [...processingIds].map((id) => getSession(id))
      );
      const completed = new Set<string>();
      results.forEach((result, i) => {
        if (result.status !== "fulfilled") return;
        const session: Session = result.value.session;
        const isDone =
          session.status !== "processing" && session.status !== "analysing";
        setSessions((prev) =>
          prev.map((s) => (s.id === session.id ? session : s))
        );
        if (isDone) completed.add([...processingIds][i]);
      });
      if (completed.size > 0) {
        setProcessingIds((prev) => {
          const next = new Set(prev);
          completed.forEach((id) => next.delete(id));
          return next;
        });
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [processingIds]);

  const handleLoadMore = () => {
    const nextOffset = offset + PAGE_SIZE;
    setOffset(nextOffset);
    setIsLoadingMore(true);
    fetchSessions(nextOffset, true);
  };

  const hasMore = sessions.length < total;

  return (
    <div className="mx-auto max-w-4xl">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-nexus-text-primary">Sessions</h1>
          <p className="mt-0.5 text-xs text-nexus-text-muted">
            {total > 0
              ? `${sessions.length} of ${total} session${total !== 1 ? "s" : ""}`
              : isLoading ? "Loading..." : "No sessions"}
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
          Failed to load sessions: {error}
        </div>
      )}

      {/* Session list */}
      {!isLoading && (
        <div className="space-y-2">
          {sessions.length === 0 ? (
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
            <>
              {sessions.map((session) => (
                <SessionCard key={session.id} session={session} />
              ))}

              {hasMore && (
                <div className="flex justify-center pt-4 pb-8">
                  <button
                    onClick={handleLoadMore}
                    disabled={isLoadingMore}
                    className="flex items-center gap-2 rounded-lg border border-nexus-border px-5 py-2.5 text-sm text-nexus-text-muted hover:border-nexus-accent-blue hover:text-nexus-accent-blue transition-colors disabled:opacity-50"
                  >
                    {isLoadingMore ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5" />
                    )}
                    {isLoadingMore ? "Loading..." : `Load more (${total - sessions.length} remaining)`}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
