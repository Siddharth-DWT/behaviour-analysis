-- ============================================================================
-- NEXUS Schema - Migration 08
-- Adds face_thumbnails table for speaker thumbnail storage.
-- Was referenced in speaker_registry.py but never added to the init scripts.
-- ============================================================================

CREATE TABLE IF NOT EXISTS face_thumbnails (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    registry_id   uuid NOT NULL REFERENCES speakers_registry(id) ON DELETE CASCADE,
    session_id    uuid REFERENCES sessions(id) ON DELETE SET NULL,
    thumbnail     bytea NOT NULL,
    quality_score double precision NOT NULL DEFAULT 0.0,
    is_primary    boolean NOT NULL DEFAULT false,
    created_at    timestamp with time zone DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ft_registry ON face_thumbnails(registry_id);
CREATE INDEX IF NOT EXISTS idx_ft_primary  ON face_thumbnails(registry_id, is_primary);

INSERT INTO schema_version (version, description) VALUES
    (8, 'face_thumbnails: speaker face crops for thumbnail display')
ON CONFLICT (version) DO NOTHING;
