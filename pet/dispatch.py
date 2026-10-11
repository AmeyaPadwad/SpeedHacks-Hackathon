"""Module 5: dispatch connectors. Called only after the user approves a Draft in the overlay."""
import json
import os
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path

import requests

from .brain import Draft

CONTACTS_FILE = Path(os.environ.get("FLOWSTATE_CONTACTS") or Path.home() / ".flowstate_contacts.json")
TIMEOUT_S = 10
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class DispatchError(Exception):
    pass


def load_contacts(path: Path = CONTACTS_FILE) -> dict[str, dict]:
    """Optional name -> {"slack": user id, "discord": user id, "email": address} map, keyed case-insensitively."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k).strip().lower(): v for k, v in data.items() if isinstance(v, dict)}
    except (OSError, ValueError, AttributeError):
        return {}


def configured(platform: str) -> bool:
    return {
        "slack": bool(os.environ.get("SLACK_WEBHOOK_URL")),
        "discord": bool(os.environ.get("DISCORD_WEBHOOK_URL")),
        "email": bool(os.environ.get("SMTP_HOST")),
    }.get(platform, False)


def _post(url: str, payload: dict):
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise DispatchError(f"Network error: {type(e).__name__}") from e
    if r.status_code >= 300:
        raise DispatchError(f"Webhook returned HTTP {r.status_code}")


def send_slack(d: Draft, contact: dict):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        raise DispatchError("SLACK_WEBHOOK_URL is not set.")
    who = f"<@{contact['slack']}>" if contact.get("slack") else d.recipient
    _post(url, {"text": f"{who} {d.message}" if who else d.message})


def send_discord(d: Draft, contact: dict):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise DispatchError("DISCORD_WEBHOOK_URL is not set.")
    uid = contact.get("discord")
    who = f"<@{uid}>" if uid else d.recipient
    # Only ping the resolved recipient; never @everyone/@here or roles from model text.
    _post(url, {"content": f"{who} {d.message}" if who else d.message,
                "allowed_mentions": {"users": [str(uid)] if uid else []}})


def send_email(d: Draft, contact: dict):
    host = os.environ.get("SMTP_HOST")
    if not host:
        raise DispatchError("SMTP_HOST is not set.")
    to = contact.get("email") or d.recipient
    if not EMAIL_RE.match(to):
        raise DispatchError(f"No email address for {d.recipient!r}; add it to the contacts file.")
    msg = EmailMessage()
    msg["From"] = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER", "")
    msg["To"] = to
    msg["Subject"] = d.subject or "Quick note"
    msg.set_content(d.message)
    try:
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT") or 587), timeout=TIMEOUT_S) as s:
            s.starttls()
            if os.environ.get("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
            s.send_message(msg)
    except (OSError, smtplib.SMTPException) as e:
        raise DispatchError(f"Email failed: {type(e).__name__}") from e


SENDERS = {"slack": send_slack, "discord": send_discord, "email": send_email}


def dispatch(d: Draft, contacts: dict[str, dict] | None = None) -> str:
    """Send an approved draft; returns a short confirmation or raises DispatchError."""
    sender = SENDERS.get(d.platform)
    if sender is None:
        raise DispatchError(f"Nothing to send for platform {d.platform!r}.")
    contacts = load_contacts() if contacts is None else contacts
    sender(d, contacts.get(d.recipient.strip().lower(), {}) if d.recipient else {})
    if not d.recipient:
        return f"Sent to your configured {d.platform.capitalize()} channel."
    return f"Sent to {d.recipient} on {d.platform.capitalize()}."
