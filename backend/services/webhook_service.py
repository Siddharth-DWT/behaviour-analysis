# backend/services/webhook_service.py
"""
NEXUS Backend - Webhook delivery for programmatic API integrations.

On pipeline completion/failure a signed JSON payload is POSTed to the caller's
registered callback_url. Signature: HMAC-SHA256 of the raw body with
WEBHOOK_SIGNING_SECRET, sent as `X-Nexus-Signature: sha256=<hex>`. Consumers must
verify with a timing-safe comparison (hmac.compare_digest).

SSRF defense: validate_callback_url() rejects private / loopback / link-local /
reserved targets (blocks cloud metadata 169.254.169.254 and RFC-1918). It is
called both at submit time (POST /v1/analyze) and again immediately before each
delivery attempt (DNS-rebinding defense).
"""
import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import urllib.parse
import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import HTTPException

logger = logging.getLogger("nexus.backend.webhooks")

WEBHOOK_SIGNING_SECRET = os.getenv("WEBHOOK_SIGNING_SECRET", "")
WEBHOOK_MAX_ATTEMPTS = int(os.getenv("WEBHOOK_MAX_ATTEMPTS", "3"))
WEBHOOK_TIMEOUT_SECONDS = float(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "10"))


def validate_callback_url(url: str) -> None:
    """
    Raise HTTPException(422) if the URL is not http(s) or resolves to a
    private / loopback / link-local / reserved address (SSRF guard).
    Resolves ALL A/AAAA records — every one must be public.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(422, "callback_url must be http(s)")
    host = parsed.hostname
    if not host:
        raise HTTPException(422, "callback_url has no host")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise HTTPException(422, "callback_url host does not resolve")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise HTTPException(422, "callback_url resolves to a disallowed (non-public) address")


def _sign(body: bytes, secret: Optional[str] = None) -> str:
    key = (secret or WEBHOOK_SIGNING_SECRET).encode("utf-8")
    return hmac.new(key, body, hashlib.sha256).hexdigest()


async def deliver_webhook(
    *,
    session_id: str,
    event: str,
    payload: dict,
    callback_url: str,
    key_id: Optional[str],
    pool,
    secret: Optional[str] = None,
) -> None:
    """
    Sign + POST the payload with retries; record the attempt in webhook_deliveries.
    Signs with `secret` (a managed endpoint's whsec_…) when given, else the global
    WEBHOOK_SIGNING_SECRET. Never raises — webhook failures are non-fatal.
    """
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = _sign(body, secret)

    # Audit row first (delivered=false) so a crash mid-delivery is still recorded.
    delivery_id = None
    try:
        row = await pool.fetchrow(
            """
            INSERT INTO webhook_deliveries
                (session_id, key_id, callback_url, event, payload)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING id
            """,
            _uuid.UUID(session_id) if session_id else None,
            _uuid.UUID(key_id) if key_id else None,
            callback_url,
            event,
            json.dumps(payload),
        )
        delivery_id = row["id"] if row else None
    except Exception as exc:
        logger.warning("[%s] webhook audit insert failed (continuing): %s", session_id, exc)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "NEXUS-Webhook/1",
        "X-Nexus-Event": event,
        "X-Nexus-Signature": f"sha256={signature}",
        "X-Nexus-Delivery": str(delivery_id) if delivery_id else "",
    }

    last_status: Optional[int] = None
    last_error: Optional[str] = None
    delivered = False

    for attempt in range(1, WEBHOOK_MAX_ATTEMPTS + 1):
        try:
            validate_callback_url(callback_url)  # re-resolve each attempt (anti-rebind)
            async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
                resp = await client.post(callback_url, content=body, headers=headers)
            last_status = resp.status_code
            if 200 <= resp.status_code < 300:
                delivered = True
                break
        except HTTPException as exc:
            last_error = f"blocked: {exc.detail}"
            break  # SSRF block — do not retry
        except Exception as exc:
            last_error = str(exc)
        if attempt < WEBHOOK_MAX_ATTEMPTS:
            await asyncio.sleep(2 ** attempt)  # 2s, 4s, 8s …

    if delivery_id is not None:
        try:
            await pool.execute(
                """
                UPDATE webhook_deliveries
                SET attempts = $2, delivered = $3, last_status = $4,
                    last_error = $5, delivered_at = $6
                WHERE id = $1
                """,
                delivery_id,
                attempt,
                delivered,
                last_status,
                last_error,
                datetime.now(timezone.utc) if delivered else None,
            )
        except Exception as exc:
            logger.warning("[%s] webhook audit update failed: %s", session_id, exc)

    if delivered:
        logger.info("[%s] Webhook delivered (%s) → %s", session_id, event, callback_url)
    else:
        logger.warning(
            "[%s] Webhook delivery failed (%s) after %d attempts: status=%s error=%s",
            session_id, event, attempt, last_status, last_error,
        )
