# Buffer & Reference Audio Files in the NEXUS Outbound Webhook

## Context

NEXUS already has an outbound webhook system (`analysis.completed` / `analysis.failed`
events, fired from `PipelineOrchestrator.run()` in
`backend/pipeline/analysis_pipeline.py`) used by programmatic-API consumers to get
notified when a session finishes processing. Today that payload is **JSON-metadata
only** — it never references the audio/video file itself, because by default
(`retain_media=False`) NEXUS deletes the uploaded file from disk immediately after
the pipeline completes (`analysis_pipeline.py:783-788`), right after the webhook
fires. Even when a caller opts into `retain_media=True`, that flag exists purely for
*dashboard playback* and is a separate, longer-lived concern (swept after
`RECORDING_RETENTION_DAYS`, default 3 days).

The user wants webhook consumers to be able to retrieve the processed audio itself
as part of the completion notification — i.e., the payload should carry a reference
to the file, and NEXUS should **buffer** (temporarily hold) that file on disk long
enough for the consumer to pull it, even for the common case where the caller does
not want long-term retention. This is a new, distinct concept from `retain_media`:
a short-lived staging window purely in service of webhook delivery, not dashboard
UX.

Goal: add an opt-in `include_media_in_webhook` flag to session submission, a new
authenticated media-download endpoint under the `/v1` programmatic API, a
`media_url`/`media_expires_at` field pair on the webhook payload, and a bounded
buffer window (env-configurable, default 24h) after which the buffered file is
purged — all fully backward compatible (default off, zero change to existing
consumers or the default deletion behavior).

## Design

### 1. New opt-in flag: `include_media_in_webhook`

A new boolean, mirroring how `retain_media` is already threaded end-to-end, added
alongside it at every call site (default `False` — no behavior change unless a
caller explicitly asks for it):

- `POST /v1/analyze` — new `Form(default=False)` param next to `retain_media` in
  `backend/api/v1.py:99-140` (params block at lines 106-107), forwarded into
  `create_session_and_dispatch(...)`.
- `POST /uploads/complete` (dashboard chunked-upload path) — read out of the JSON
  `config` body the same way `retain_media` already is, in
  `backend/api/uploads.py:206-207` (`config_dict.get("include_media_in_webhook", False)`),
  so dashboard-originated sessions with a **managed** webhook registered can opt in
  too (this path has no per-request `callback_url`, only managed-endpoint delivery).
- `create_session_and_dispatch()` (`backend/api/_session_launch.py:25-37` signature,
  body ~73-123) gains the new keyword and forwards it into `pipeline.run(...)`
  alongside the existing `retain_media=retain_media` (line 121).
- `PipelineOrchestrator.run()` (`backend/pipeline/analysis_pipeline.py:86-103`) gains
  `include_media_in_webhook: bool = False` as a new parameter next to
  `retain_media: bool = False` (line 101).

### 2. Buffer window config

New env var `WEBHOOK_MEDIA_BUFFER_HOURS` (default `24`), read the same way
`_RECORDING_RETENTION_DAYS` already is, next to it:

```python
# backend/pipeline/analysis_pipeline.py:52
_WEBHOOK_MEDIA_BUFFER_HOURS = int(os.getenv("WEBHOOK_MEDIA_BUFFER_HOURS", "24"))
```

Add to `.env.example` (new lines near the existing webhook-adjacent section, e.g.
after `JWT_REFRESH_TOKEN_EXPIRE_DAYS` at line 57) and to the `backend.environment`
block in `docker-compose.yml` (near `WEBHOOK_SIGNING_SECRET` / `PUBLIC_BASE_URL`,
lines 162-163):

```yaml
- WEBHOOK_MEDIA_BUFFER_HOURS=${WEBHOOK_MEDIA_BUFFER_HOURS:-24}
```

### 3. DB: track buffer expiry per session

Two files/sessions on disk are otherwise indistinguishable inside `UPLOAD_DIR` — a
`retain_media=True` file (long-lived, 3-day sweep) and a buffered-for-webhook file
(short-lived, hour-scale sweep) live in the same directory with no way to tell them
apart from mtime alone. A DB column is the minimal correct way to disambiguate.

New migration `infrastructure/postgres/init/11-media-buffer.sql`, following the
existing numbered-migration convention (see `10-webhook-endpoints.sql`):

```sql
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS media_buffer_expires_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_sessions_media_buffer_expiry
    ON sessions(media_buffer_expires_at) WHERE media_buffer_expires_at IS NOT NULL;

INSERT INTO schema_version (version, description) VALUES
    (11, 'sessions.media_buffer_expires_at: short-lived webhook media buffer window')
ON CONFLICT (version) DO NOTHING;
```

New helpers in `backend/core/database.py` (next to `create_session`/`get_session`/
`update_session_status`, lines 188-341):

- `set_session_media_buffer(session_id, media_url, expires_at)` — `UPDATE sessions
  SET media_url = $2, media_buffer_expires_at = $3 WHERE id = $1`. Used when a
  `retain_media=False` session opts into `include_media_in_webhook` — since
  `media_url` is currently only written once at row-creation time and stays `NULL`
  for ephemeral sessions (`analysis_pipeline.py:782` comment confirms this), this is
  the one place that needs to retroactively set it.
- `clear_expired_media_buffers()` — selects `id, media_url FROM sessions WHERE
  media_buffer_expires_at IS NOT NULL AND media_buffer_expires_at < NOW()`, and for
  each row the caller (see §6) deletes the file then this helper nulls
  `media_url`/`media_buffer_expires_at` on that row (`UPDATE ... SET media_url =
  NULL, media_buffer_expires_at = NULL WHERE id = ANY($1)`).

### 4. New endpoint: `GET /v1/sessions/{id}/media`

Added in `backend/api/v1.py`, immediately after the existing ownership-checked read
endpoints (`/sessions/{id}`, `/signals`, `/report`, lines 158-215), reusing the
**exact same** `_owned_session()` helper (`v1.py:145-155`) those already use:

```python
@router.get("/sessions/{session_id}/media")
async def v1_get_session_media(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    session = await _owned_session(session_id, current_user)
    media_path = session.get("media_url")
    if not media_path or not Path(media_path).exists():
        raise HTTPException(404, "Media not available (expired, not retained, or never buffered)")
    return FileResponse(media_path, media_type=..., headers={"Accept-Ranges": "bytes"})
```

This deliberately does **not** reuse `backend/api/sessions.py`'s existing
`GET /sessions/{session_id}/video` route (lines 670-706) as a template — that route
authenticates the JWT but performs **no ownership comparison** against
`session["user_id"]` (a pre-existing gap, unrelated to this feature, flagged
separately by a full-codebase audit run earlier in this session). The new `/v1`
endpoint must use `_owned_session()` so it inherits the same 404-on-mismatch
ownership guarantee as its sibling `/v1` read endpoints.

Auth: plain `Depends(get_current_user)` — this already accepts both dashboard JWTs
and `nxs_...` API keys (`backend/core/auth.py:122-178`), so a webhook consumer (who
by definition already holds an API key, since they had to call `/v1/analyze` or
register a managed webhook) authenticates with the *same* key they already have.
No new signed-URL/token scheme needed. (Confirmed with the user: reuse existing API
key auth rather than building a new pre-signed-URL mechanism.)

Content-type detection: mirror whatever extension→mimetype logic
`get_session_video` already uses in `sessions.py` (read it at implementation time
and reuse rather than reinvent). `FileResponse` (Starlette) handles `Range` requests
natively, so no manual byte-range logic is needed.

### 5. Webhook payload: add `media_url` / `media_expires_at`

In `_fire_webhook()` (`analysis_pipeline.py:798-884`), thread two new optional
kwargs and add them to the payload dict (currently built at lines 825-835):

```python
payload = {
    ...,
    "results_url": f"{public_base}/v1/sessions/{session_id}" if public_base else None,
    "media_url": f"{public_base}/v1/sessions/{session_id}/media" if (media_available and public_base) else None,
    "media_expires_at": media_expires_at.isoformat() if media_expires_at else None,
    "sent_at": now_iso,
}
```

`media_available` / `media_expires_at` are computed by the caller (§6) and passed
in as new `_fire_webhook(..., media_available: bool = False, media_expires_at:
Optional[datetime] = None)` kwargs. When `include_media_in_webhook` was never
requested, both stay `None`/`False` and the payload is byte-for-byte identical to
today — fully backward compatible with `docs/API_GUIDE.md`'s documented schema for
existing consumers.

### 6. Pipeline completion logic — buffer instead of (or in addition to) delete

Replace the block at `analysis_pipeline.py:756-788` with:

```python
_cleanup_old_recordings()
_cleanup_expired_media_buffers()          # new — see below
await self._redis_repo.set_session_state(...)

media_expires_at = None
media_available = False
if include_media_in_webhook:
    if retain_media:
        # Already kept long-term by the existing retain_media path; just reference it.
        media_available = True
    else:
        media_expires_at = datetime.now(timezone.utc) + timedelta(hours=_WEBHOOK_MEDIA_BUFFER_HOURS)
        await set_session_media_buffer(
            session_id, media_url=str(Path(file_path).resolve()), expires_at=media_expires_at,
        )
        media_available = True

await self._fire_webhook(
    ..., media_available=media_available, media_expires_at=media_expires_at,
)

# Ephemeral media: delete only if neither retained nor buffered for the webhook.
if not retain_media and media_expires_at is None:
    try:
        Path(file_path).unlink(missing_ok=True)
        logger.info("[%s] Raw media deleted (retain_media=false)", session_id)
    except Exception as exc:
        logger.warning("[%s] Raw media delete failed (non-fatal): %s", session_id, exc)
```

New module-level function, next to `_cleanup_old_recordings`
(`analysis_pipeline.py:1210-1224`):

```python
async def _cleanup_expired_media_buffers() -> None:
    """Delete files whose webhook media-buffer window has elapsed; null the DB pointer."""
    expired = await clear_expired_media_buffers_select()   # SELECT-only half of §3
    if not expired:
        return
    deleted_ids = []
    for row in expired:
        try:
            Path(row["media_url"]).unlink(missing_ok=True)
            deleted_ids.append(row["id"])
        except OSError:
            pass
    if deleted_ids:
        await clear_expired_media_buffers(deleted_ids)      # nulls media_url + expiry
        logger.info("Cleaned up %d buffered media file(s) past their webhook window", len(deleted_ids))
```

This runs opportunistically once per completed pipeline run, at the same point the
existing `_cleanup_old_recordings()` mtime sweep already runs (line 756) — same
inherited limitation as today (no dedicated scheduler; a quiet deployment with no
new sessions won't sweep), kept consistent with existing code rather than
introducing new infrastructure.

**Failure path unchanged.** The early-exit failure webhook
(`analysis_pipeline.py:148-172`, `analysis.failed`) does not gain media buffering —
there are no results to correlate the file with, and the existing immediate-delete
behavior on failure is left as-is. `include_media_in_webhook` only affects the
`analysis.completed` success path.

### 7. Docs

Update `docs/API_GUIDE.md`:
- §6 "Webhooks" (lines 215-327): document the new `media_url`/`media_expires_at`
  payload fields and when they're populated.
- §3 "Submit a file for analysis" (lines 77-131): document the new
  `include_media_in_webhook` form field on `POST /v1/analyze`.
- §12 endpoint reference (~596-612): add `GET /v1/sessions/{id}/media`.
- Note the `WEBHOOK_MEDIA_BUFFER_HOURS` env var and that media is deleted
  immediately (today's behavior) unless this flag is set.

## Files touched

| File | Change |
|---|---|
| `backend/pipeline/analysis_pipeline.py` | New env const, `run()` param, completion-block buffer/delete logic, `_fire_webhook()` payload fields, new `_cleanup_expired_media_buffers()` |
| `backend/api/v1.py` | New `include_media_in_webhook` form param on `/analyze`, new `GET /sessions/{id}/media` route |
| `backend/api/uploads.py` | Read `include_media_in_webhook` from `config` body (lines ~206-207 pattern) |
| `backend/api/_session_launch.py` | Thread new kwarg through `create_session_and_dispatch()` |
| `backend/core/database.py` | `set_session_media_buffer()`, `clear_expired_media_buffers()` (+ select variant) |
| `infrastructure/postgres/init/11-media-buffer.sql` | New migration: `media_buffer_expires_at` column + index |
| `.env.example`, `docker-compose.yml` | `WEBHOOK_MEDIA_BUFFER_HOURS` |
| `docs/API_GUIDE.md` | Document new flag, payload fields, endpoint |

Existing utilities reused as-is: `_owned_session()` (`v1.py:145`), `get_current_user`
dual JWT/API-key auth (`core/auth.py:122`), `deliver_webhook()` signing/retry
(`backend/services/webhook_service.py`), the `retain_media` threading pattern
throughout as the template for `include_media_in_webhook`.

## Verification

1. **Unit-level sanity**: run existing `backend/tests/` suite to confirm nothing
   in the pipeline/webhook path regresses (`pytest backend/tests/`).
2. **Manual end-to-end** (requires `docker compose up -d`, not run without explicit
   go-ahead per project rules):
   - `POST /v1/analyze` with `retain_media=false&include_media_in_webhook=true` and
     a `callback_url` pointing at a local test receiver (e.g. `webhook.site` or a
     throwaway `python -m http.server` + ngrok/localtunnel, or a simple FastAPI
     stub logging the POST body).
   - Confirm the received `analysis.completed` payload includes a populated
     `media_url` + `media_expires_at`.
   - Fetch that `media_url` with the same API key (`Authorization: Bearer nxs_...`)
     and confirm the audio streams back correctly (verify `Content-Type` and that
     the file plays).
   - Confirm the file is **not** deleted immediately after completion (check
     `data/recordings/` on disk).
   - Wait past `WEBHOOK_MEDIA_BUFFER_HOURS` (or temporarily set it to a small value
     like `0.01` for testing) and trigger another pipeline completion; confirm the
     buffered file is swept and `sessions.media_url`/`media_buffer_expires_at` are
     nulled.
   - Repeat with `include_media_in_webhook=false` (default) and confirm payload/
     deletion behavior is byte-for-byte unchanged from current production behavior.
   - Repeat with `retain_media=true&include_media_in_webhook=true` and confirm
     `media_expires_at` is `null` in the payload (long-lived, governed by
     `RECORDING_RETENTION_DAYS` instead) while `media_url` is still populated.
3. **Ownership check**: as a second user (different API key), attempt
   `GET /v1/sessions/{other_user_session_id}/media` and confirm a 404, matching
   the existing `_owned_session()` behavior on `/v1/sessions/{id}`.
