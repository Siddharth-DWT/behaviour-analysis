# NEXUS API Guide — Tokens, Analysis & Webhooks

Programmatic access to the NEXUS behavioural-analysis engine. Create an API token,
submit an audio/video file, and receive results by polling or via a signed webhook.

- **Base URL (production/staging):** `https://analysis-be.pathtodeal.com`
- **Base URL (local dev):** `http://localhost:8000`
- All request/response bodies are JSON unless noted. File uploads use `multipart/form-data`.
- Every example below uses `$BASE` for the base URL and `$TOKEN` for your API token.

```bash
export BASE="https://analysis-be.pathtodeal.com"
export TOKEN="nxs_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## 1. Concepts in 30 seconds

| Thing | What it is |
|-------|-----------|
| **API token** (`nxs_live_…`) | A long-lived credential that acts as *your user account*. Send it as `Authorization: Bearer <token>`. |
| **Session** | One analysis job for one uploaded file. Identified by a `session_id` (UUID). |
| **Signals** | The behavioural findings produced by the 6 analysis domains (voice, language, facial, body, gaze, conversation) + fusion. |
| **Webhook** | An HTTPS URL NEXUS calls with a **signed** JSON payload when a session finishes. |

A token **acts as you** — it inherits your role and can call every `/v1` endpoint you can,
including creating more tokens. Treat it like a password.

---

## 2. Get an API token

### Option A — Dashboard (easiest)
1. Log in to the dashboard.
2. Go to **Settings → API Tokens**.
3. Click **Create Token**, give it a name, optionally set a rate limit and an expiry.
4. **Copy the token immediately** — it is shown **once** and never again.

### Option B — API
Token management is authenticated with your **dashboard login (JWT)** *or* an existing API token.

```bash
curl -X POST "$BASE/v1/api-keys" \
  -H "Authorization: Bearer <your-JWT-or-existing-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "CI pipeline",
    "rate_limit_per_min": 60,
    "expires_in_days": 90
  }'
```

Response (the plaintext `key` appears **only here, once**):

```json
{
  "id": "89a655cf-1ce8-4e28-b564-7b5f0147c592",
  "name": "CI pipeline",
  "key": "nxs_live_miDyO2CpQsAdKYE4w7usaX8L3pQppmMvRyZfIHJ_E50",
  "key_prefix": "nxs_live_",
  "last4": "_E50",
  "rate_limit_per_min": 60,
  "expires_at": "2026-10-11T09:48:32Z",
  "created_at": "2026-07-13T09:48:32Z"
}
```

**Field notes**
- `rate_limit_per_min` — optional, default **60**. Requests above this in a rolling minute get `429`.
- `expires_in_days` — optional, `1`–`3650`. Omit for a token that **never expires**.

> If you lose a token you cannot recover it — revoke it and create a new one.

---

## 3. Submit a file for analysis

`POST /v1/analyze` — `multipart/form-data`, authenticated with an **API token**.

```bash
curl -X POST "$BASE/v1/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@meeting.mp4" \
  -F "meeting_type=sales_call" \
  -F "title=Acme discovery call" \
  -F "retain_media=false"
```

Response (returns immediately — analysis runs in the background):

```json
{
  "session_id": "624fc5fd-02f8-454f-9400-518bc8e36d85",
  "status": "processing",
  "title": "Acme discovery call",
  "meeting_type": "sales_call",
  "retain_media": false
}
```

### Form fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `file` | ✅ | — | The media file (see formats below). Max **300 MB**. |
| `meeting_type` | — | `sales_call` | Analysis profile: `sales_call`, `interview`, `general`, `interrogation_video`, … |
| `title` | — | filename | Human label for the session. |
| `callback_url` | — | — | One-off webhook URL for **this** run (see §6). Must be public HTTPS. |
| `retain_media` | — | `false` | `false` = the uploaded file is **deleted after analysis** (only signals/report are kept). `true` = keep it (enables video playback in the dashboard). For an **audio-only** file (`.wav`/`.mp3`/`.m4a`/`.flac`/`.ogg`) submitted here with `retain_media=true`, the file is kept **indefinitely** rather than swept after `RECORDING_RETENTION_DAYS` — this only applies to files submitted via `/v1/analyze`; video files, and anything uploaded through the dashboard, still follow the normal 3-day sweep. |
| `include_media_in_webhook` | — | `false` | `true` = buffer the raw file on disk after completion (even if `retain_media=false`) and include a `media_url` in the completion webhook payload — see §6. Buffer window is `WEBHOOK_MEDIA_BUFFER_HOURS` (default 24h) unless `retain_media=true`, in which case the file is already kept long-term. |
| `config` | — | `{}` | JSON string for advanced options (see below). |

### Accepted file formats
`.wav` · `.mp3` · `.m4a` · `.flac` · `.ogg` — audio-only pipeline
`.mp4` · `.webm` — full audio **+ video** pipeline

Audio-only files skip the facial/body/gaze/video agents automatically.

### The `config` object (optional)
Pass as a JSON **string** in the `config` form field:

```json
{
  "num_speakers": 2,
  "analysis":      { "run_behavioural": true, "sensitivity": 0.5 },
  "transcription": { "model_preference": "assemblyai", "language": null }
}
```

- `analysis.run_behavioural` — `false` returns transcript-only (skips language/conversation/fusion).
  When `false` **and** you haven't set `transcription.model_preference` yourself, the transcription
  backend defaults to **Parakeet** instead of the normal AssemblyAI-first cascade — faster/cheaper
  for pure transcription. Set `transcription.model_preference` explicitly to override this.
- `num_speakers` — hint for diarization when you know the count.

---

## 4. Get results (polling)

After submitting, poll the session until `status` is `completed` (or `partial`/`failed`).

### Session status + summary
```bash
curl -H "Authorization: Bearer $TOKEN" "$BASE/v1/sessions/<session_id>"
```
```json
{
  "session_id": "624fc5fd-…",
  "status": "completed",
  "title": "Acme discovery call",
  "meeting_type": "sales_call",
  "duration_ms": 1267000,
  "speaker_count": 3,
  "signal_count": 2690,
  "signals_by_agent": { "voice": 2403, "language": 448, "video": 2690, "fusion": 42 },
  "has_report": true,
  "created_at": "…",
  "completed_at": "…"
}
```

`status` values: `processing` → `completed` (all good) · `partial` (some agent failed) · `failed`.

### Signals (the raw findings)
```bash
curl -H "Authorization: Bearer $TOKEN" \
  "$BASE/v1/sessions/<session_id>/signals?agent=voice&limit=1000"
```
Query params: `agent` (`voice|language|conversation|video|fusion`), `signal_type`, `limit` (≤50000), `offset`.

Each signal looks like:
```json
{
  "agent": "voice",
  "speaker_id": "…",
  "signal_type": "vocal_stress_score",
  "value": 0.72,
  "value_text": "elevated_stress",
  "confidence": 0.61,
  "window_start_ms": 45000,
  "window_end_ms": 47000,
  "metadata": { "...": "..." }
}
```

### Narrative report
```bash
curl -H "Authorization: Bearer $TOKEN" "$BASE/v1/sessions/<session_id>/report"
```
Returns `404` until the report has been generated.

> **Ownership:** you can only read sessions created by *your* token's user. Anyone else's
> session id returns `404` (not `403`, so existence isn't leaked).

---

## 5. Suggested polling loop

```bash
SID=$(curl -s -X POST "$BASE/v1/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@meeting.mp4" -F "meeting_type=sales_call" \
  | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")

while true; do
  STATUS=$(curl -s -H "Authorization: Bearer $TOKEN" "$BASE/v1/sessions/$SID" \
    | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "status: $STATUS"
  [ "$STATUS" = "completed" ] || [ "$STATUS" = "partial" ] || [ "$STATUS" = "failed" ] && break
  sleep 15
done
```

A full behavioural run takes several minutes for a ~20-minute video. **Webhooks (next) avoid polling.**

---

## 6. Webhooks — get notified automatically

Two ways to receive a signed callback when a session finishes:

1. **Managed endpoint** — register a URL once; it's used automatically for **all** your sessions.
2. **One-off** — pass `callback_url` on a single `POST /v1/analyze` (takes priority for that run).

All callback URLs must be **public HTTPS**. Private, loopback, link-local, and cloud-metadata
addresses (e.g. `169.254.169.254`, `127.0.0.1`, `10.x`, `192.168.x`) are rejected with `422`.

### Register a managed endpoint
```bash
curl -X POST "$BASE/v1/webhooks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "url": "https://you.com/hooks/nexus", "description": "Production receiver" }'
```
Response — the signing `secret` is shown **once**:
```json
{
  "id": "13265ad4-…",
  "url": "https://you.com/hooks/nexus",
  "description": "Production receiver",
  "active": true,
  "secret": "whsec_7N1qaS6-HR-Alwsdo_D5uhdm9-NR0PujPYeyH9PGkZE",
  "created_at": "…",
  "last_delivery_at": null
}
```

### The payload NEXUS sends to your endpoint
```
POST https://you.com/hooks/nexus
Content-Type: application/json
X-Nexus-Event: analysis.completed
X-Nexus-Signature: sha256=<hex-hmac>
X-Nexus-Delivery: <delivery-uuid>
```
```json
{
  "event": "analysis.completed",
  "session_id": "624fc5fd-…",
  "status": "completed",
  "title": "Acme discovery call",
  "meeting_type": "sales_call",
  "duration_seconds": 1267.0,
  "signal_counts": { "voice": 2403, "language": 448, "conversation": 15, "video": 2690, "fusion": 42 },
  "results_url": "https://analysis-be.pathtodeal.com/v1/sessions/624fc5fd-…",
  "media_url": "https://analysis-be.pathtodeal.com/v1/sessions/624fc5fd-…/media",
  "media_expires_at": "2026-07-14T11:25:00Z",
  "sent_at": "2026-07-13T11:25:00Z"
}
```
- On failure: `"event": "analysis.failed"`, `"status": "failed"`, and an extra `"error"` field.
- **Delivery & retries:** up to **3 attempts** with exponential backoff (2s, 4s, 8s), 10s timeout each.
  Respond `2xx` to acknowledge. Every attempt is recorded (see §7).

### Getting the audio/video file itself

`media_url` / `media_expires_at` are only populated when the request that submitted
the session set `include_media_in_webhook=true` (§3) — otherwise both are `null`,
matching today's default behavior exactly.

- `media_url` points at `GET /v1/sessions/{id}/media`, authenticated the same way as
  every other `/v1` endpoint — send your API token: `Authorization: Bearer $TOKEN`.
  It streams the raw file back with its original `Content-Type` and supports HTTP
  `Range` requests.
- `media_expires_at` is `null` when the session also had `retain_media=true` (the
  file is kept long-term, governed by the server's retention sweep). Otherwise it's
  an ISO-8601 timestamp — fetch the file before then. After it passes, NEXUS deletes
  the buffered file and `GET .../media` returns `404`.
- The buffer window is `WEBHOOK_MEDIA_BUFFER_HOURS` (default **24h**, server-configured).
  This isn't about webhook delivery itself (that resolves in seconds — see retry/timeout
  above) — it's how long *you* have to actually go fetch the file after being notified,
  to account for consumers that don't process the webhook the instant it arrives (queue
  workers, batch jobs, temporarily-down receivers). Don't rely on it being long; treat it
  as a courtesy window, not guaranteed storage.
- Fetch it as soon as you receive the webhook — don't defer:
  ```bash
  curl -H "Authorization: Bearer $TOKEN" -o meeting.mp4 \
    "$BASE/v1/sessions/624fc5fd-…/media"
  ```

### Verify the signature (do this on every webhook!)
`X-Nexus-Signature` is `sha256=` + HMAC-SHA256 of the **raw request body** using your endpoint's
secret (for one-off `callback_url`, the server's global secret is used).

**Python**
```python
import hmac, hashlib

def verify(raw_body: bytes, signature_header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)   # timing-safe
```

**Node.js**
```js
const crypto = require("crypto");
function verify(rawBody, signatureHeader, secret) {
  const expected = "sha256=" +
    crypto.createHmac("sha256", secret).update(rawBody).digest("hex");
  return crypto.timingSafeEqual(Buffer.from(expected), Buffer.from(signatureHeader));
}
```
> Verify against the **raw bytes** of the body, before any JSON re-serialization.

### One-off callback (no registration)
```bash
curl -X POST "$BASE/v1/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@clip.mp4" \
  -F "callback_url=https://you.com/hooks/nexus"
```

---

## 7. Manage webhooks & view deliveries

| Action | Request |
|--------|---------|
| List endpoints (secret hidden) | `GET /v1/webhooks` |
| Pause / resume / change URL | `PATCH /v1/webhooks/{id}` body `{ "active": false }` / `{ "url": "…" }` |
| Rotate the signing secret | `POST /v1/webhooks/{id}/rotate-secret` → returns new secret **once** |
| Delete an endpoint | `DELETE /v1/webhooks/{id}` |
| Recent delivery attempts | `GET /v1/webhook-deliveries?limit=50` |

`GET /v1/webhook-deliveries` returns your sessions' delivery attempts:
```json
{
  "deliveries": [
    {
      "id": "…", "session_id": "624fc5fd-…", "event": "analysis.completed",
      "callback_url": "https://you.com/hooks/nexus",
      "attempts": 1, "delivered": true, "last_status": 200,
      "last_error": null, "created_at": "…", "delivered_at": "…"
    }
  ],
  "count": 1
}
```

---

## 8. Manage tokens

| Action | Request |
|--------|---------|
| List your tokens (masked) | `GET /v1/api-keys` |
| Revoke a token | `DELETE /v1/api-keys/{id}` |

A revoked or expired token stops authenticating immediately (`401`). In-flight analyses
already running are unaffected.

---

## 9. Rate limits

- Default **60 requests/minute** per token (set per token at creation).
- Exceeding it returns **`429 Too Many Requests`** with a `Retry-After` header (seconds until the
  window resets). Back off and retry.

---

## 10. Errors

| Code | Meaning | Common cause |
|------|---------|--------------|
| `400` | Bad request | Unsupported file type. |
| `401` | Unauthorized | Missing / invalid / revoked / expired token. |
| `403` | Forbidden | Role too low for the action. |
| `404` | Not found | Session/endpoint doesn't exist **or isn't yours**. |
| `413` | Payload too large | File over 300 MB (use the dashboard's chunked upload for larger). |
| `422` | Validation error | `callback_url` not public HTTPS (SSRF guard), or bad field. |
| `429` | Rate limited | Too many requests this minute — honour `Retry-After`. |
| `500` | Server error | Report it; if you see `relation "api_keys" does not exist`, DB migrations aren't applied on that server. |

Error bodies are JSON: `{ "detail": "…human-readable reason…" }`.

---

## 11. Full response formats (every endpoint)

Every response is JSON. Timestamps are ISO-8601 UTC strings. UUIDs are strings.
Fields that can be absent are shown as `null`.

### `POST /v1/api-keys` → 200
```json
{
  "id": "89a655cf-1ce8-4e28-b564-7b5f0147c592",
  "name": "CI pipeline",
  "key": "nxs_live_…",          // plaintext — ONLY returned here, once
  "key_prefix": "nxs_live_",
  "last4": "_E50",
  "rate_limit_per_min": 60,
  "expires_at": "2026-10-11T09:48:32Z",   // null = never expires
  "created_at": "2026-07-13T09:48:32Z"
}
```

### `GET /v1/api-keys` → 200
```json
{
  "api_keys": [
    {
      "id": "89a655cf-…",
      "name": "CI pipeline",
      "key_prefix": "nxs_live_",
      "key_last4": "_E50",
      "masked_key": "nxs_live_…_E50",
      "rate_limit_per_min": 60,
      "request_count": 42,          // total calls made with this token
      "role": null,                 // null = inherits your account role
      "last_used_at": "2026-07-13T09:49:12Z",
      "expires_at": null,
      "revoked_at": null,
      "active": true,               // false once revoked or expired
      "created_at": "2026-07-13T09:48:32Z"
    }
  ]
}
```

### `DELETE /v1/api-keys/{id}` → 200
```json
{ "id": "89a655cf-…", "revoked": true }
```

### `POST /v1/analyze` → 200
```json
{
  "session_id": "624fc5fd-02f8-454f-9400-518bc8e36d85",
  "status": "processing",
  "title": "Acme discovery call",
  "meeting_type": "sales_call",
  "retain_media": false
}
```

### `GET /v1/sessions/{id}` → 200
```json
{
  "session_id": "624fc5fd-…",
  "status": "completed",          // processing | completed | partial | failed
  "title": "Acme discovery call",
  "meeting_type": "sales_call",
  "duration_ms": 1267000,         // media length; null until known
  "speaker_count": 3,             // null until known
  "signal_count": 2690,
  "signals_by_agent": {           // counts per domain
    "voice": 2403, "language": 448, "conversation": 15, "video": 2690, "fusion": 42
  },
  "has_report": true,
  "created_at": "2026-07-13T…",
  "completed_at": "2026-07-13T…"  // null while processing
}
```

### `GET /v1/sessions/{id}/signals` → 200
```json
{
  "session_id": "624fc5fd-…",
  "count": 2,
  "signals": [
    {
      "id": 918273,                       // bigint, stable per signal
      "session_id": "624fc5fd-…",
      "speaker_id": "46f62cf7-…",         // UUID of the speaker (nullable)
      "speaker_label": "Speaker_1",       // human label (nullable)
      "agent": "voice",                   // voice|language|conversation|video|fusion
      "signal_type": "vocal_stress_score",
      "value": 0.72,                      // numeric magnitude (meaning depends on type)
      "value_text": "elevated_stress",    // categorical label
      "confidence": 0.61,                 // 0–0.85 (deception-related capped at 0.55)
      "window_start_ms": 45000,           // time window this signal covers
      "window_end_ms": 47000,
      "metadata": {                       // type-specific extras (free-form object)
        "face_box_area": 0.049,
        "face_centre_x": 0.48,
        "rule_id": "VOICE-STRESS-01"
      },
      "created_at": "2026-07-13T…"
    }
  ]
}
```

**Signal field reference**

| Field | Type | Notes |
|-------|------|-------|
| `id` | integer | Stable id of the signal. |
| `speaker_id` / `speaker_label` | string / string | Who the signal is about; `null` for session-level fusion signals. |
| `agent` | string | Which domain produced it. |
| `signal_type` | string | e.g. `vocal_stress_score`, `head_nod`, `presence_detected`, `sentiment_score`, … |
| `value` | number | Magnitude; interpret together with `value_text`. |
| `value_text` | string | Categorical label (e.g. `elevated_stress`). |
| `confidence` | number | `0`–`0.85` global cap; deception-related types capped at `0.55`. |
| `window_start_ms` / `window_end_ms` | integer | Millisecond window in the media. |
| `metadata` | object | Type-specific; may include `rule_id`, face geometry, etc. |

### `GET /v1/sessions/{id}/report` → 200 (404 if not generated yet)
```json
{
  "session_id": "624fc5fd-…",
  "report": {
    "id": "…",
    "session_id": "624fc5fd-…",
    "report_type": "sales_call",
    "narrative": "Executive summary text…",   // plain-text summary
    "content": { … }                          // structured report (keys below)
  }
}
```

`content` is a structured object whose keys **depend on `meeting_type`**. Common keys:

| Key | Appears for | Meaning |
|-----|-------------|---------|
| `executive_summary` | all | Top-line narrative. |
| `general_summary` | all | Longer summary. |
| `notes` | all | Analyst-style notes. |
| `key_moments` | all | Notable timestamped moments. |
| `speaker_analyses` | all | Per-speaker breakdown. |
| `recommendations` / `action_items` | all | Suggested next steps. |
| `cross_modal_insights` | all | Findings from fusing domains. |
| `entities` | all | Extracted names/orgs/values. |
| `signal_graph` / `graph_analytics` | all | Signal relationships. |
| `deal_assessment` / `objection_handling` | sales_call | Sales-specific analysis. |
| `contradiction_analysis` / `contamination_timeline` / `technique_analysis` / `risk_assessment` | interrogation_video | Forensic analysis. |
| `voice_text_correlations` | interrogation_video | Voice-vs-words incongruence. |

> Not every key is present in every report — only those the meeting type produces.

### `POST /v1/webhooks` → 200
```json
{
  "id": "13265ad4-…",
  "url": "https://you.com/hooks/nexus",
  "description": "Production receiver",
  "active": true,
  "secret": "whsec_…",                 // signing secret — ONLY returned here, once
  "created_at": "2026-07-13T…",
  "last_delivery_at": null
}
```

### `GET /v1/webhooks` → 200 (secret never included)
```json
{
  "webhooks": [
    {
      "id": "13265ad4-…",
      "url": "https://you.com/hooks/nexus",
      "description": "Production receiver",
      "active": true,
      "created_at": "2026-07-13T…",
      "last_delivery_at": "2026-07-13T11:25:02Z"   // null if never delivered
    }
  ]
}
```

### `PATCH /v1/webhooks/{id}` → 200
Returns the updated (masked) endpoint — same shape as one item in `GET /v1/webhooks`.

### `POST /v1/webhooks/{id}/rotate-secret` → 200
```json
{ "id": "13265ad4-…", "secret": "whsec_…" }   // new secret — shown once
```

### `DELETE /v1/webhooks/{id}` → 200
```json
{ "id": "13265ad4-…", "deleted": true }
```

### `GET /v1/webhook-deliveries` → 200
```json
{
  "count": 1,
  "deliveries": [
    {
      "id": "…",
      "session_id": "624fc5fd-…",
      "event": "analysis.completed",      // or analysis.failed
      "callback_url": "https://you.com/hooks/nexus",
      "attempts": 1,                      // 1–3
      "delivered": true,                  // true if a 2xx was received
      "last_status": 200,                 // last HTTP status from your endpoint (null if none)
      "last_error": null,                 // error text if delivery failed
      "created_at": "2026-07-13T…",
      "delivered_at": "2026-07-13T…"      // null if never delivered
    }
  ]
}
```

### Error responses (all endpoints)
```json
{ "detail": "Invalid or revoked API key" }
```
Plus `429` includes a `Retry-After` header (seconds).

### Webhook payload (POST'd to *your* endpoint)
See §6 — headers `X-Nexus-Event`, `X-Nexus-Signature`, `X-Nexus-Delivery`, and the
`analysis.completed` / `analysis.failed` JSON body.

---

## 12. Full endpoint reference

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/v1/api-keys` | JWT or token | Create a token (returns secret once) |
| `GET` | `/v1/api-keys` | JWT or token | List your tokens (masked) |
| `DELETE` | `/v1/api-keys/{id}` | JWT or token | Revoke a token |
| `POST` | `/v1/analyze` | token | Submit a file for analysis |
| `GET` | `/v1/sessions/{id}` | token | Session status + summary |
| `GET` | `/v1/sessions/{id}/signals` | token | Signals (filterable) |
| `GET` | `/v1/sessions/{id}/report` | token | Narrative report |
| `GET` | `/v1/sessions/{id}/media` | token | Raw audio/video file (only if buffered/retained — see §6) |
| `POST` | `/v1/webhooks` | JWT or token | Register a webhook endpoint |
| `GET` | `/v1/webhooks` | JWT or token | List endpoints (masked) |
| `PATCH` | `/v1/webhooks/{id}` | JWT or token | Update url/active/description |
| `DELETE` | `/v1/webhooks/{id}` | JWT or token | Delete an endpoint |
| `POST` | `/v1/webhooks/{id}/rotate-secret` | JWT or token | New signing secret (once) |
| `GET` | `/v1/webhook-deliveries` | JWT or token | Recent delivery attempts |

---

## 13. End-to-end example (audio, webhook-driven)

```bash
export BASE="https://analysis-be.pathtodeal.com"
export TOKEN="nxs_live_…"

# 1. Register a webhook once (do this a single time; save the secret)
curl -X POST "$BASE/v1/webhooks" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://you.com/hooks/nexus","description":"prod"}'
# → save "secret": "whsec_…"

# 2. Submit a recording
curl -X POST "$BASE/v1/analyze" -H "Authorization: Bearer $TOKEN" \
  -F "file=@call.mp3" -F "meeting_type=sales_call"
# → { "session_id": "…", "status": "processing" }

# 3. …minutes later NEXUS POSTs the signed 'analysis.completed' payload to your endpoint.
#    Verify X-Nexus-Signature, then GET results_url for the full breakdown.
```

---

## 14. Security checklist

- Store tokens and webhook secrets like passwords (env vars / a secret manager) — never commit them.
- **Always verify** `X-Nexus-Signature` on incoming webhooks with a timing-safe compare.
- Rotate a token/secret immediately if it may have leaked (`DELETE` the token / `rotate-secret`).
- Set an `expires_in_days` on tokens used by short-lived jobs or CI.
- Keep `retain_media=false` and `include_media_in_webhook=false` (the defaults) unless you
  specifically need the file to persist — it minimises stored data. Two things to be aware of on
  `/v1/analyze` specifically: `retain_media=true` on an **audio-only** file is kept
  **indefinitely** (not swept after a few days), and `include_media_in_webhook=true` buffers the
  raw file on disk for `WEBHOOK_MEDIA_BUFFER_HOURS` even when `retain_media=false`. Both mean the
  recording sits on disk for longer than the zero-retention default — factor that into any
  data-retention/compliance requirements before enabling them.

---

*NEXUS produces probabilistic behavioural indicators, never certainties. Signal confidence is
capped (max 0.85; deception-related signals max 0.55). Use results as decision support, not proof.*
