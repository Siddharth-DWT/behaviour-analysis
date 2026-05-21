-- ============================================================================
-- NEXUS Schema - Migration 08
-- Adds face_thumbnails table and the idx_sr_face ivfflat index on
-- speakers_registry.face_embedding — both were created only at runtime via
-- database.py (ensure_speaker_registry_tables) and were never written here.
-- Safe to run on any DB regardless of state (all statements are idempotent).
-- ============================================================================

-- ── face_thumbnails ──
-- Stores one or more face images per registry entry (different sessions/angles).
-- is_primary = TRUE for the best-quality reference thumbnail shown in the UI.
CREATE TABLE IF NOT EXISTS face_thumbnails (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    registry_id     UUID NOT NULL REFERENCES speakers_registry(id) ON DELETE CASCADE,
    session_id      UUID REFERENCES sessions(id) ON DELETE SET NULL,
    thumbnail       BYTEA NOT NULL,
    quality_score   FLOAT DEFAULT 0.0,
    is_primary      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ft_registry ON face_thumbnails(registry_id);

-- ── speakers_registry face embedding ivfflat index ──
-- Guarded with a DO block: ivfflat requires pgvector and enough rows for
-- the index to build; the runtime code in database.py uses the same
-- try/except pattern so the failure is non-fatal.
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_sr_face ON speakers_registry
        USING ivfflat (face_embedding vector_cosine_ops) WITH (lists = 10);
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Could not create idx_sr_face (non-fatal): %', SQLERRM;
END $$;

-- ── Record this migration ──
INSERT INTO schema_version (version, description) VALUES
    (8, 'face_thumbnails table + idx_sr_face ivfflat index on speakers_registry')
ON CONFLICT (version) DO NOTHING;
