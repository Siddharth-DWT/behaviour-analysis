-- 11-media-buffer.sql
-- Short-lived webhook media buffer: when a caller opts into
-- include_media_in_webhook on a session that is NOT otherwise retained
-- (retain_media=false), the raw upload is held on disk past pipeline
-- completion for WEBHOOK_MEDIA_BUFFER_HOURS so the webhook consumer can
-- fetch it via GET /v1/sessions/{id}/media before it is swept.

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS media_buffer_expires_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_sessions_media_buffer_expiry
    ON sessions(media_buffer_expires_at) WHERE media_buffer_expires_at IS NOT NULL;

INSERT INTO schema_version (version, description) VALUES
    (11, 'sessions.media_buffer_expires_at: short-lived webhook media buffer window')
ON CONFLICT (version) DO NOTHING;
