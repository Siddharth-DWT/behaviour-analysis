-- 09-api-keys.sql
-- Programmatic API access: personal access tokens (PATs), per-request usage, and
-- webhook delivery audit. Enables external systems to submit media and receive
-- signed completion callbacks.

-- ── API keys (personal access tokens) ───────────────────────────────────────
-- Lookup is by key_lookup = sha256(raw_key) in a UNIQUE index (O(1) probe).
-- key_hash (bcrypt) is optional defense-in-depth verified against the single row.
CREATE TABLE IF NOT EXISTS api_keys (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    org_id             UUID,                         -- snapshot of owner's org at creation
    name               VARCHAR(120) NOT NULL,        -- human label, e.g. "CI pipeline"
    key_prefix         VARCHAR(32)  NOT NULL,        -- display prefix, e.g. 'nxs_live_'
    key_last4          VARCHAR(8)   NOT NULL,        -- last 4 chars for display
    key_lookup         CHAR(64)     NOT NULL,        -- sha256 hex of full raw key
    key_hash           TEXT         NOT NULL,        -- bcrypt of raw key (defense-in-depth)
    role               VARCHAR(16),                  -- optional role override; NULL = inherit user's role
    rate_limit_per_min INTEGER      NOT NULL DEFAULT 60,
    request_count      BIGINT       NOT NULL DEFAULT 0,
    last_used_at       TIMESTAMPTZ,
    expires_at         TIMESTAMPTZ,                  -- NULL = never expires
    revoked_at         TIMESTAMPTZ,                  -- NULL = active
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_lookup ON api_keys(key_lookup);
CREATE INDEX        IF NOT EXISTS idx_api_keys_user   ON api_keys(user_id);
CREATE INDEX        IF NOT EXISTS idx_api_keys_active ON api_keys(user_id) WHERE revoked_at IS NULL;

-- ── Per-request / per-session usage audit ───────────────────────────────────
CREATE TABLE IF NOT EXISTS api_key_usage (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_id      UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    session_id  UUID REFERENCES sessions(id) ON DELETE SET NULL,
    endpoint    VARCHAR(128) NOT NULL,
    status_code INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_key_usage_key     ON api_key_usage(key_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_key_usage_session ON api_key_usage(session_id);

-- ── Webhook delivery audit / retry ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    key_id        UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    callback_url  TEXT NOT NULL,
    event         VARCHAR(48) NOT NULL,             -- 'analysis.completed' | 'analysis.failed'
    payload       JSONB NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    delivered     BOOLEAN NOT NULL DEFAULT false,
    last_status   INTEGER,                          -- last HTTP status from callback
    last_error    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_session ON webhook_deliveries(session_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_pending ON webhook_deliveries(created_at) WHERE delivered = false;

INSERT INTO schema_version (version, description) VALUES
    (9, 'api_keys, api_key_usage, webhook_deliveries: programmatic access + webhook delivery')
ON CONFLICT (version) DO NOTHING;
