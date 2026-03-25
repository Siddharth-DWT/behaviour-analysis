-- ============================================================================
-- NEXUS Auth Schema - Migration 02
-- Adds authentication tables: users auth fields + auth_tokens + session ownership
-- ============================================================================

-- ── 1. Extend users table with auth fields ──
-- The users table already exists in 01-schema.sql with (id, org_id, email, name, role, created_at).
-- We add the columns needed for password-based authentication.

ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS company VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url VARCHAR(500);
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT false;
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ;

-- Backfill full_name from existing name column for seed data
UPDATE users SET full_name = name WHERE full_name IS NULL;


-- ── 2. Auth tokens (refresh tokens) ──

CREATE TABLE IF NOT EXISTS auth_tokens (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    refresh_token VARCHAR(500) UNIQUE NOT NULL,
    device_info VARCHAR(255),
    expires_at  TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_token ON auth_tokens(refresh_token);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_expires ON auth_tokens(expires_at);


-- ── 3. Link sessions to users ──

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);


-- ── 4. Version tracking ──

INSERT INTO schema_version (version, description) VALUES
    (2, 'Auth tables: user password fields, auth_tokens, session ownership')
ON CONFLICT (version) DO NOTHING;
