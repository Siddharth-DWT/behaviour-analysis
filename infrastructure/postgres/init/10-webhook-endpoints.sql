-- 10-webhook-endpoints.sql
-- Managed webhook endpoints: a user registers a callback URL once (with its own
-- signing secret) and it is auto-used for their sessions when a request does not
-- pass its own callback_url. The server signs outgoing payloads with `secret`, so
-- it is stored in plaintext (same model as Stripe endpoint secrets).

CREATE TABLE IF NOT EXISTS webhook_endpoints (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    org_id           UUID,
    url              TEXT NOT NULL,
    secret           TEXT NOT NULL,          -- 'whsec_…' — server signs with it (plaintext)
    description      VARCHAR(200),
    active           BOOLEAN NOT NULL DEFAULT true,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_delivery_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_webhook_endpoints_user   ON webhook_endpoints(user_id);
CREATE INDEX IF NOT EXISTS idx_webhook_endpoints_active ON webhook_endpoints(user_id) WHERE active;

INSERT INTO schema_version (version, description) VALUES
    (10, 'webhook_endpoints: managed per-user webhook callback URLs with signing secrets')
ON CONFLICT (version) DO NOTHING;
