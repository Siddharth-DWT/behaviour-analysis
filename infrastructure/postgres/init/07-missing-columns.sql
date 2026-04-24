-- ============================================================================
-- NEXUS Schema - Migration 07
-- Backfills columns that were added via runtime ALTER TABLE in database.py
-- and records the missing schema_version entry for migration 06.
-- Safe to run on any DB regardless of state (all statements are idempotent).
-- ============================================================================

-- ── sessions.upload_config ──
-- Added at runtime by _ensure_upload_config_column() — now tracked here.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS upload_config JSONB DEFAULT '{}';

-- ── sessions.participant_count ──
-- Added at runtime by _ensure_participant_count_column() — now tracked here.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS participant_count INT;

-- ── Backfill schema_version v6 ──
-- 06-password-reset.sql created the table but forgot the version insert.
INSERT INTO schema_version (version, description) VALUES
    (6, 'Password reset tokens table')
ON CONFLICT (version) DO NOTHING;

-- ── Record this migration ──
INSERT INTO schema_version (version, description) VALUES
    (7, 'Backfill: sessions.upload_config, sessions.participant_count, schema_version v6')
ON CONFLICT (version) DO NOTHING;
