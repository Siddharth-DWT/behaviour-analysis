import { useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Upload, Loader2 } from "lucide-react";
import { listSessions, uploadSession } from "../api/client";
import SessionCard from "../components/SessionCard";

export default function SessionList() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [meetingType, setMeetingType] = useState("sales_call");

  const { data, isLoading, error } = useQuery({
    queryKey: ["sessions"],
    queryFn: () => listSessions({ limit: 50 }),
  });

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      setUploading(true);
      setUploadError(null);
      const title = file.name.replace(/\.[^.]+$/, "");
      return uploadSession(file, title, meetingType);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      setUploading(false);
    },
    onError: (err: Error) => {
      setUploadError(err.message);
      setUploading(false);
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadMutation.mutate(file);
    e.target.value = "";
  };

  return (
    <div className="mx-auto max-w-4xl">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-nexus-text-primary">
            Sessions
          </h1>
          <p className="mt-0.5 text-xs text-nexus-text-muted">
            {data ? `${data.total} session${data.total !== 1 ? "s" : ""}` : "Loading..."}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={meetingType}
            onChange={(e) => setMeetingType(e.target.value)}
            className="rounded border border-nexus-border bg-nexus-surface px-2 py-1.5 text-xs text-nexus-text-secondary outline-none focus:border-nexus-accent-blue"
          >
            <option value="sales_call">Sales Call</option>
            <option value="client_meeting">Client Meeting</option>
            <option value="internal">Internal</option>
            <option value="interview">Interview</option>
          </select>

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-1.5 rounded bg-nexus-accent-blue px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent-blue-80 disabled:opacity-50"
          >
            {uploading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Upload className="h-3.5 w-3.5" />
            )}
            {uploading ? "Analysing..." : "Upload"}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a,.flac,.ogg,.webm,.mp4"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      </div>

      {/* Upload status */}
      {uploading && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-accent-blue-30 bg-accent-blue-10 p-3 text-xs text-nexus-accent-blue">
          <Loader2 className="h-4 w-4 animate-spin" />
          Running full pipeline: Voice → Language → Fusion. This may take a few minutes...
        </div>
      )}

      {uploadError && (
        <div className="mb-4 rounded-lg border border-stress-high-30 bg-stress-high-10 p-3 text-xs text-nexus-stress-high">
          Upload failed: {uploadError}
        </div>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-20 text-sm text-nexus-text-muted">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Loading sessions...
        </div>
      )}

      {/* Error state */}
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
