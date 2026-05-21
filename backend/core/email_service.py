# backend/core/email_service.py
"""
NEXUS Backend - Email Service
Sends transactional emails via ZeptoMail REST API.
Copied verbatim from services/api_gateway/email_service.py.
No import path changes required.
"""
import os
import logging

import httpx

logger = logging.getLogger("nexus.gateway.email")

# ── Configuration ──

ZEPTOMAIL_TOKEN = os.getenv("ZEPTOMAIL_TOKEN", "")
ZEPTOMAIL_API_URL = os.getenv(
    "ZEPTOMAIL_API_URL", "https://api.zeptomail.in/v1.1/email"
)
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@yourdomain.com")
FROM_NAME = os.getenv("FROM_NAME", "NEXUS")
APP_URL = os.getenv("APP_URL", "http://localhost:3000")


def is_email_configured() -> bool:
    """Return True if ZeptoMail credentials are present."""
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
    Returns True on success, False on failure (non-fatal).
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

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": ZEPTOMAIL_TOKEN,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(ZEPTOMAIL_API_URL, json=payload, headers=headers)

        if resp.status_code in (200, 201):
            logger.info(f"Verification email sent to {to_email}")
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
    """Send a branded NEXUS email-verification message."""
    verify_url = f"{APP_URL}/verify-email?token={verification_token}"
    subject = "Verify your NEXUS account"

    html_body = f"""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background-color:#0F1117;font-family:
  -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="background-color:#0F1117;padding:40px 20px;">
    <tr><td align="center">
      <table role="presentation" width="520" cellpadding="0" cellspacing="0"
             style="background-color:#1A1D27;border:1px solid #2D3348;
                    border-radius:12px;padding:40px;">
        <tr><td style="text-align:center;padding-bottom:24px;">
          <h1 style="margin:0;font-size:28px;font-weight:700;color:#E8ECF4;
                     letter-spacing:-0.5px;">NEXUS</h1>
          <p style="margin:4px 0 0;font-size:13px;color:#8B93A7;">
            Multi-Agent Behavioural Analysis</p>
        </td></tr>

        <tr><td style="padding-bottom:24px;">
          <h2 style="margin:0 0 12px;font-size:20px;color:#E8ECF4;">
            Verify your email address</h2>
          <p style="margin:0;font-size:15px;line-height:1.6;color:#8B93A7;">
            Hi {to_name},<br><br>
            Thanks for signing up for NEXUS. Please confirm your email address
            by clicking the button below.</p>
        </td></tr>

        <tr><td align="center" style="padding-bottom:24px;">
          <a href="{verify_url}"
             style="display:inline-block;padding:14px 32px;
                    background-color:#4F8BFF;color:#ffffff;font-size:15px;
                    font-weight:600;text-decoration:none;border-radius:8px;">
            Verify Email Address</a>
        </td></tr>

        <tr><td style="padding-bottom:24px;">
          <p style="margin:0;font-size:13px;line-height:1.6;color:#8B93A7;">
            This link will expire in <strong style="color:#E8ECF4;">24 hours</strong>.
            If you did not create a NEXUS account you can safely ignore this email.</p>
        </td></tr>

        <tr><td style="border-top:1px solid #2D3348;padding-top:20px;">
          <p style="margin:0;font-size:12px;color:#8B93A7;">
            If the button doesn't work, copy and paste this URL into your browser:</p>
          <p style="margin:8px 0 0;font-size:12px;word-break:break-all;">
            <a href="{verify_url}" style="color:#4F8BFF;">{verify_url}</a></p>
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
        f"If you did not create a NEXUS account, ignore this email.\n"
    )

    return await send_email(to_email, to_name, subject, html_body, text_body)


async def send_password_reset_email(
    to_email: str, to_name: str, reset_token: str
) -> bool:
    """Send a branded NEXUS password-reset message (for future use)."""
    reset_url = f"{APP_URL}/reset-password?token={reset_token}"
    subject = "Reset your NEXUS password"

    html_body = f"""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background-color:#0F1117;font-family:
  -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="background-color:#0F1117;padding:40px 20px;">
    <tr><td align="center">
      <table role="presentation" width="520" cellpadding="0" cellspacing="0"
             style="background-color:#1A1D27;border:1px solid #2D3348;
                    border-radius:12px;padding:40px;">
        <tr><td style="text-align:center;padding-bottom:24px;">
          <h1 style="margin:0;font-size:28px;font-weight:700;color:#E8ECF4;
                     letter-spacing:-0.5px;">NEXUS</h1>
          <p style="margin:4px 0 0;font-size:13px;color:#8B93A7;">
            Multi-Agent Behavioural Analysis</p>
        </td></tr>

        <tr><td style="padding-bottom:24px;">
          <h2 style="margin:0 0 12px;font-size:20px;color:#E8ECF4;">
            Reset your password</h2>
          <p style="margin:0;font-size:15px;line-height:1.6;color:#8B93A7;">
            Hi {to_name},<br><br>
            We received a request to reset the password for your NEXUS account.
            Click the button below to choose a new password.</p>
        </td></tr>

        <tr><td align="center" style="padding-bottom:24px;">
          <a href="{reset_url}"
             style="display:inline-block;padding:14px 32px;
                    background-color:#4F8BFF;color:#ffffff;font-size:15px;
                    font-weight:600;text-decoration:none;border-radius:8px;">
            Reset Password</a>
        </td></tr>

        <tr><td style="padding-bottom:24px;">
          <p style="margin:0;font-size:13px;line-height:1.6;color:#8B93A7;">
            This link will expire in <strong style="color:#E8ECF4;">1 hour</strong>.
            If you did not request a password reset you can safely ignore this email.</p>
        </td></tr>

        <tr><td style="border-top:1px solid #2D3348;padding-top:20px;">
          <p style="margin:0;font-size:12px;color:#8B93A7;">
            If the button doesn't work, copy and paste this URL into your browser:</p>
          <p style="margin:8px 0 0;font-size:12px;word-break:break-all;">
            <a href="{reset_url}" style="color:#4F8BFF;">{reset_url}</a></p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    text_body = (
        f"Hi {to_name},\n\n"
        f"We received a request to reset the password for your NEXUS account.\n"
        f"Visit this link to choose a new password:\n"
        f"{reset_url}\n\n"
        f"This link expires in 1 hour.\n\n"
        f"If you did not request a password reset, ignore this email.\n"
    )

    return await send_email(to_email, to_name, subject, html_body, text_body)


async def send_processing_complete_email(
    to_email: str,
    to_name: str,
    session_id: str,
    session_title: str,
    meeting_type: str,
    status: str,
    signal_counts: dict,
    duration_seconds: float = 0,
) -> bool:
    """
    Send a branded NEXUS notification when full behavioural analysis completes.
    Only called for run_behavioural=True sessions — NOT for lightweight
    transcription/diarization/entity-extraction-only sessions.
    """
    session_url = f"{APP_URL}/sessions/{session_id}"
    subject = f"Your NEXUS analysis is ready — {session_title}"

    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)
    duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    total_signals = sum(signal_counts.values())
    signal_breakdown = " · ".join(
        f"{agent.title()}: {count}"
        for agent, count in sorted(signal_counts.items())
        if count > 0
    )

    if status == "completed":
        status_color = "#10B981"
        status_label = "Analysis Complete"
        status_message = "Your session has been fully analysed and is ready for review."
    elif status == "partial":
        status_color = "#F59E0B"
        status_label = "Analysis Partial"
        status_message = (
            "Your session has been analysed but some agents encountered issues. "
            "Results are still available for review."
        )
    else:
        status_color = "#EF4444"
        status_label = "Analysis Failed"
        status_message = "There was an issue processing your session. Please try re-uploading."

    html_body = f"""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background-color:#0F1117;font-family:
  -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="background-color:#0F1117;padding:40px 20px;">
    <tr><td align="center">
      <table role="presentation" width="520" cellpadding="0" cellspacing="0"
             style="background-color:#1A1D27;border:1px solid #2D3348;
                    border-radius:12px;padding:40px;">

        <!-- Header -->
        <tr><td style="text-align:center;padding-bottom:24px;">
          <h1 style="margin:0;font-size:28px;font-weight:700;color:#E8ECF4;
                     letter-spacing:-0.5px;">NEXUS</h1>
          <p style="margin:4px 0 0;font-size:13px;color:#8B93A7;">
            Multi-Agent Behavioural Analysis</p>
        </td></tr>

        <!-- Status badge -->
        <tr><td align="center" style="padding-bottom:20px;">
          <span style="display:inline-block;padding:6px 16px;
                       background-color:{status_color}20;color:{status_color};
                       font-size:13px;font-weight:600;border-radius:20px;
                       border:1px solid {status_color}40;">
            &#9679; {status_label}
          </span>
        </td></tr>

        <!-- Title + message -->
        <tr><td style="padding-bottom:24px;">
          <h2 style="margin:0 0 12px;font-size:20px;color:#E8ECF4;">
            {session_title}</h2>
          <p style="margin:0;font-size:15px;line-height:1.6;color:#8B93A7;">
            Hi {to_name},<br><br>
            {status_message}</p>
        </td></tr>

        <!-- Stats row -->
        <tr><td style="padding-bottom:24px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
                 style="background-color:#0F1117;border-radius:8px;padding:16px;">
            <tr>
              <td style="text-align:center;padding:8px;">
                <p style="margin:0;font-size:22px;font-weight:700;color:#E8ECF4;">
                  {duration_str}</p>
                <p style="margin:4px 0 0;font-size:12px;color:#8B93A7;">Duration</p>
              </td>
              <td style="text-align:center;padding:8px;">
                <p style="margin:0;font-size:22px;font-weight:700;color:#E8ECF4;">
                  {total_signals}</p>
                <p style="margin:4px 0 0;font-size:12px;color:#8B93A7;">Signals</p>
              </td>
              <td style="text-align:center;padding:8px;">
                <p style="margin:0;font-size:22px;font-weight:700;color:#E8ECF4;">
                  {meeting_type.replace('_', ' ').title()}</p>
                <p style="margin:4px 0 0;font-size:12px;color:#8B93A7;">Type</p>
              </td>
            </tr>
          </table>
          <p style="margin:12px 0 0;font-size:12px;color:#8B93A7;text-align:center;">
            {signal_breakdown}</p>
        </td></tr>

        <!-- CTA button -->
        <tr><td align="center" style="padding-bottom:24px;">
          <a href="{session_url}"
             style="display:inline-block;padding:14px 32px;
                    background-color:#4F8BFF;color:#ffffff;font-size:15px;
                    font-weight:600;text-decoration:none;border-radius:8px;">
            View Results</a>
        </td></tr>

        <!-- Fallback URL -->
        <tr><td style="border-top:1px solid #2D3348;padding-top:20px;">
          <p style="margin:0;font-size:12px;color:#8B93A7;">
            If the button doesn't work, copy and paste this URL into your browser:</p>
          <p style="margin:8px 0 0;font-size:12px;word-break:break-all;">
            <a href="{session_url}" style="color:#4F8BFF;">{session_url}</a></p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

    text_body = (
        f"Hi {to_name},\n\n"
        f"{status_message}\n\n"
        f"Session: {session_title}\n"
        f"Duration: {duration_str}\n"
        f"Signals detected: {total_signals} ({signal_breakdown})\n\n"
        f"View your results here: {session_url}\n\n"
        f"— NEXUS\n"
    )

    return await send_email(to_email, to_name, subject, html_body, text_body)
