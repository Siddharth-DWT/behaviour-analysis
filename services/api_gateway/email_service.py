"""
NEXUS API Gateway - Email Service
Sends transactional emails via ZeptoMail (Zoho) REST API.
"""
import os
import logging

import httpx

logger = logging.getLogger("nexus.gateway.email")

# ── Configuration ──
ZEPTOMAIL_TOKEN = os.getenv("ZEPTOMAIL_TOKEN", "")
ZEPTOMAIL_API_URL = os.getenv(
    "ZEPTOMAIL_API_URL", "https://api.zeptomail.com/v1.1/email"
)
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@yourdomain.com")
FROM_NAME = os.getenv("FROM_NAME", "NEXUS")
APP_URL = os.getenv("APP_URL", "http://localhost:3000")


def is_email_configured() -> bool:
    """Check if ZeptoMail credentials are configured."""
    return bool(ZEPTOMAIL_TOKEN)


async def send_email(
    to_email: str,
    to_name: str,
    subject: str,
    html_body: str,
    text_body: str = "",
) -> bool:
    """
    Send an email via ZeptoMail REST API.
    Returns True on success, False on failure.
    """
    if not is_email_configured():
        logger.warning("Email not configured — ZEPTOMAIL_TOKEN is empty")
        return False

    payload = {
        "from": {"address": FROM_EMAIL, "name": FROM_NAME},
        "to": [{"email_address": {"address": to_email, "name": to_name}}],
        "subject": subject,
        "htmlbody": html_body,
    }
    if text_body:
        payload["textbody"] = text_body

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": ZEPTOMAIL_TOKEN,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                ZEPTOMAIL_API_URL, json=payload, headers=headers
            )
            if resp.status_code in (200, 201):
                logger.info(f"Email sent to {to_email} — subject: {subject}")
                return True
            else:
                logger.error(
                    f"ZeptoMail API error {resp.status_code}: {resp.text}"
                )
                return False
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False


async def send_verification_email(
    to_email: str, to_name: str, verification_token: str
) -> bool:
    """Send an email-verification link to a newly registered user."""
    verify_url = f"{APP_URL}/verify-email?token={verification_token}"

    subject = "Verify your NEXUS account"

    html_body = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background-color:#0F1117;font-family:Arial,Helvetica,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0F1117;padding:40px 0;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background-color:#1A1D27;border:1px solid #2D3348;border-radius:8px;padding:40px;">
        <tr><td align="center" style="padding-bottom:24px;">
          <h1 style="color:#4F8BFF;font-size:28px;margin:0;">NEXUS</h1>
        </td></tr>
        <tr><td style="color:#E8ECF4;font-size:16px;line-height:1.6;">
          <p style="margin:0 0 16px;">Hi {to_name},</p>
          <p style="margin:0 0 24px;">Thanks for signing up for NEXUS. Please verify your email address by clicking the button below.</p>
        </td></tr>
        <tr><td align="center" style="padding:8px 0 24px;">
          <a href="{verify_url}" style="display:inline-block;background-color:#4F8BFF;color:#ffffff;text-decoration:none;font-size:16px;font-weight:bold;padding:14px 32px;border-radius:6px;">
            Verify Email Address
          </a>
        </td></tr>
        <tr><td style="color:#8B93A7;font-size:13px;line-height:1.5;">
          <p style="margin:0 0 8px;">If you didn't create a NEXUS account, you can safely ignore this email.</p>
          <p style="margin:0 0 8px;">This link expires in 24 hours.</p>
          <p style="margin:0;word-break:break-all;">Or copy this URL: <a href="{verify_url}" style="color:#4F8BFF;">{verify_url}</a></p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    text_body = (
        f"Hi {to_name},\n\n"
        f"Thanks for signing up for NEXUS. Please verify your email by visiting:\n"
        f"{verify_url}\n\n"
        f"This link expires in 24 hours.\n\n"
        f"If you didn't create a NEXUS account, you can safely ignore this email."
    )

    return await send_email(to_email, to_name, subject, html_body, text_body)


async def send_password_reset_email(
    to_email: str, to_name: str, reset_token: str
) -> bool:
    """Send a password-reset link (for future use)."""
    reset_url = f"{APP_URL}/auth/reset-password?token={reset_token}"

    subject = "Reset your NEXUS password"

    html_body = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background-color:#0F1117;font-family:Arial,Helvetica,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0F1117;padding:40px 0;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background-color:#1A1D27;border:1px solid #2D3348;border-radius:8px;padding:40px;">
        <tr><td align="center" style="padding-bottom:24px;">
          <h1 style="color:#4F8BFF;font-size:28px;margin:0;">NEXUS</h1>
        </td></tr>
        <tr><td style="color:#E8ECF4;font-size:16px;line-height:1.6;">
          <p style="margin:0 0 16px;">Hi {to_name},</p>
          <p style="margin:0 0 24px;">We received a request to reset your NEXUS password. Click the button below to choose a new password.</p>
        </td></tr>
        <tr><td align="center" style="padding:8px 0 24px;">
          <a href="{reset_url}" style="display:inline-block;background-color:#4F8BFF;color:#ffffff;text-decoration:none;font-size:16px;font-weight:bold;padding:14px 32px;border-radius:6px;">
            Reset Password
          </a>
        </td></tr>
        <tr><td style="color:#8B93A7;font-size:13px;line-height:1.5;">
          <p style="margin:0 0 8px;">If you didn't request a password reset, you can safely ignore this email.</p>
          <p style="margin:0 0 8px;">This link expires in 1 hour.</p>
          <p style="margin:0;word-break:break-all;">Or copy this URL: <a href="{reset_url}" style="color:#4F8BFF;">{reset_url}</a></p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    text_body = (
        f"Hi {to_name},\n\n"
        f"We received a request to reset your NEXUS password. Visit this link:\n"
        f"{reset_url}\n\n"
        f"This link expires in 1 hour.\n\n"
        f"If you didn't request a password reset, you can safely ignore this email."
    )

    return await send_email(to_email, to_name, subject, html_body, text_body)
