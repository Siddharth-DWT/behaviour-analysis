import { getAccessToken } from "./client";

const API_BASE = "/api";
const CHUNK_SIZE = 10 * 1024 * 1024; // 10 MB
const MAX_PARALLEL = 3;              // 3 simultaneous chunk uploads
const MAX_RETRIES  = 3;              // retries per chunk before giving up

export interface ChunkProgress {
  phase: "initializing" | "uploading" | "assembling" | "done" | "error";
  chunksUploaded: number;
  totalChunks: number;
  bytesUploaded: number;
  totalBytes: number;
  progressPct: number;
  sessionId?: string;
  error?: string;
}

export type ProgressCallback = (p: ChunkProgress) => void;

export interface ChunkedUploadOptions {
  title?: string;
  meetingType?: string;
  config?: Record<string, unknown>;
  onProgress?: ProgressCallback;
}

function authHeaders(): Record<string, string> {
  const token = getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function apiPost(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function uploadOneChunk(
  uploadId: string,
  chunkNumber: number,
  blob: Blob,
): Promise<void> {
  const form = new FormData();
  form.append("chunk_number", String(chunkNumber));
  form.append("chunk", blob, `chunk_${chunkNumber}`);

  let lastErr: unknown;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(`${API_BASE}/uploads/${uploadId}/chunk`, {
        method: "POST",
        headers: authHeaders(),
        body: form,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `chunk ${chunkNumber} failed: ${res.status}`);
      }
      return;
    } catch (err) {
      lastErr = err;
      if (attempt < MAX_RETRIES - 1) {
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  }
  throw lastErr;
}

/**
 * Upload a file in 10 MB chunks with parallel transfers and per-chunk retry.
 * Falls back gracefully — the caller should use this only for files > 20 MB.
 */
export async function uploadFileChunked(
  file: File,
  options: ChunkedUploadOptions = {},
): Promise<string> {
  const { title, meetingType, config, onProgress } = options;

  const report = (partial: Partial<ChunkProgress>) =>
    onProgress?.({
      phase: "uploading",
      chunksUploaded: 0,
      totalChunks: 0,
      bytesUploaded: 0,
      totalBytes: file.size,
      progressPct: 0,
      ...partial,
    });

  // ── 1. Init ──────────────────────────────────────────────────────────────────
  report({ phase: "initializing", progressPct: 0 });

  const init = (await apiPost("/uploads/init", {
    filename:     file.name,
    file_size:    file.size,
    chunk_size:   CHUNK_SIZE,
    meeting_type: meetingType ?? "sales_call",
    title:        title ?? file.name.replace(/\.[^.]+$/, ""),
    config:       JSON.stringify(config ?? {}),
  })) as { upload_id: string; total_chunks: number; chunk_size: number };

  const { upload_id, total_chunks } = init;
  let chunksUploaded = 0;

  // ── 2. Upload chunks in parallel batches ─────────────────────────────────────
  const indices = Array.from({ length: total_chunks }, (_, i) => i);

  for (let i = 0; i < indices.length; i += MAX_PARALLEL) {
    const batch = indices.slice(i, i + MAX_PARALLEL);
    await Promise.all(
      batch.map(async (idx) => {
        const start = idx * CHUNK_SIZE;
        const end   = Math.min(start + CHUNK_SIZE, file.size);
        await uploadOneChunk(upload_id, idx, file.slice(start, end));
        chunksUploaded++;
        report({
          phase:          "uploading",
          chunksUploaded,
          totalChunks:    total_chunks,
          bytesUploaded:  Math.min(chunksUploaded * CHUNK_SIZE, file.size),
          progressPct:    Math.round((chunksUploaded / total_chunks) * 100),
        });
      }),
    );
  }

  // ── 3. Complete ───────────────────────────────────────────────────────────────
  report({ phase: "assembling", progressPct: 100, chunksUploaded: total_chunks, totalChunks: total_chunks });

  const result = (await apiPost(`/uploads/${upload_id}/complete`, {})) as {
    session_id: string;
  };

  report({
    phase:          "done",
    chunksUploaded: total_chunks,
    totalChunks:    total_chunks,
    bytesUploaded:  file.size,
    progressPct:    100,
    sessionId:      result.session_id,
  });

  return result.session_id;
}
