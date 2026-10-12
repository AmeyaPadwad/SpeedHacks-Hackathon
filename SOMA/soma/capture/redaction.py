"""Privacy filters applied before anything is captured, stored or shown.

* :func:`is_excluded` decides whether a foreground window may be captured at
  all (password managers, private browsing, banking, auth prompts, user list).
* :func:`redact_text` masks secrets in recognized text.
* :func:`redact_elements` applies it to OCR/accessibility elements.
* :func:`mask_regions` blacks out normalized boxes in a frame before any
  persistence; :func:`sensitive_boxes` finds the boxes whose text was redacted.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Iterable, Sequence

import numpy as np

from soma.config import CaptureConfig
from soma.types import OCRElement, WindowInfo

# ----------------------------------------------------------------- exclusion

#: Process-name fragments that are always excluded (case-insensitive substring).
BUILTIN_EXCLUDED_APPS: tuple[str, ...] = (
    # password managers
    "1password", "keepass", "keepassxc", "bitwarden", "lastpass", "dashlane",
    "enpass", "keeper", "roboform", "nordpass", "proton pass",
    # authentication prompts / secure desktop
    "credentialuibroker", "consent.exe", "logonui", "lockapp", "credwiz",
)

#: Window-title fragments that are always excluded (case-insensitive substring).
BUILTIN_EXCLUDED_TITLES: dict[str, tuple[str, ...]] = {
    "private-browsing": ("inprivate", "incognito", "private browsing", "private window"),
    "password-manager": ("1password", "bitwarden", "keepass", "lastpass", "dashlane",
                         "password manager"),
    "banking": ("online banking", "bank of america", "chase.com", "wells fargo",
                "paypal", "barclays", "hsbc", "credit union"),
    "auth-prompt": ("windows security", "user account control", "sign in", "log in",
                    "login", "two-factor", "2-step verification", "verification code",
                    "enter your password", "authenticator"),
}


def is_excluded(window: WindowInfo, cfg: CaptureConfig | None = None) -> tuple[bool, str]:
    """Return ``(True, reason)`` when ``window`` must never be captured.

    Matching is case-insensitive substring on the process name and title,
    combining built-in rules with ``cfg.excluded_apps`` and
    ``cfg.excluded_title_patterns``.
    """
    cfg = cfg or CaptureConfig()
    app = (window.app_name or "").lower()
    title = (window.title or "").lower()

    for frag in BUILTIN_EXCLUDED_APPS:
        if frag and frag in app:
            return True, f"excluded-app:{frag}"
    for frag in cfg.excluded_apps:
        f = (frag or "").strip().lower()
        if f and f in app:
            return True, f"user-excluded-app:{f}"
    for reason, frags in BUILTIN_EXCLUDED_TITLES.items():
        for frag in frags:
            if frag in title:
                return True, f"{reason}:{frag}"
    for frag in cfg.excluded_title_patterns:
        f = (frag or "").strip().lower()
        if f and f in title:
            return True, f"user-excluded-title:{f}"
    return False, ""


# ----------------------------------------------------------------- text redaction


def _mask(kind: str) -> str:
    return f"[REDACTED:{kind}]"


def luhn_valid(digits: str) -> bool:
    """Luhn checksum for a string of digits (credit-card validation)."""
    ds = [int(c) for c in digits if c.isdigit()]
    if len(ds) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_SECRET_KEYS = r"(?:password|passwd|pwd|passphrase|secret|client[_-]?secret|token|auth[_-]?token|" \
               r"access[_-]?token|api[_-]?key|apikey|access[_-]?key|private[_-]?key)"
_PAIR_RE = re.compile(
    rf"(?i)(\b{_SECRET_KEYS}\b[\"']?\s*[:=]\s*)(\"[^\"\n]*\"|'[^'\n]*'|[^\s,;&]+)")

# (kind, pattern) applied in order after key/value pairs.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}")),
    ("api-key", re.compile(r"\bsk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_\-]{16,}")),
    ("api-key", re.compile(r"\b(?:rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]
_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
_B64_RE = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{40,}={0,2}")


def _looks_random(tok: str) -> bool:
    """A long token containing upper, lower and digits is treated as a secret."""
    return (any(c.isdigit() for c in tok) and any(c.isupper() for c in tok)
            and any(c.islower() for c in tok))


def redact_text(text: str) -> str:
    """Mask secrets: API keys, AWS/GitHub/Slack tokens, JWTs, emails,
    Luhn-valid card numbers, long hex/base64 tokens and ``password=...`` pairs.
    """
    if not text:
        return text or ""
    out = _PAIR_RE.sub(lambda m: m.group(1) + _mask("secret"), text)
    for kind, pat in _PATTERNS:
        out = pat.sub(_mask(kind), out)
    out = _CARD_RE.sub(lambda m: _mask("card") if luhn_valid(m.group(0)) else m.group(0), out)
    out = _HEX_RE.sub(_mask("hex"), out)
    out = _B64_RE.sub(lambda m: _mask("token") if _looks_random(m.group(0)) else m.group(0), out)
    return out


def contains_secret(text: str) -> bool:
    """True when :func:`redact_text` would change ``text``."""
    return redact_text(text) != (text or "")


def redact_elements(elements: Iterable[OCRElement]) -> list[OCRElement]:
    """Copies of ``elements`` with secrets in ``text`` masked."""
    return [dataclasses.replace(e, text=redact_text(e.text)) for e in elements]


def sensitive_boxes(elements: Iterable[OCRElement]) -> list[tuple[float, float, float, float]]:
    """Normalized boxes of elements whose text contains a secret."""
    return [tuple(e.bbox_xyxy_normalized) for e in elements  # type: ignore[misc]
            if contains_secret(e.text) or "[REDACTED:" in e.text]


# ----------------------------------------------------------------- image masking


def mask_regions(image: np.ndarray,
                 boxes_normalized: Sequence[Sequence[float]],
                 fill: int = 0) -> np.ndarray:
    """Return a copy of ``image`` ([H,W,C] or [H,W]) with each normalized
    ``xyxy`` box filled with ``fill`` (black by default). Boxes are clipped to
    the image; the input is never modified.
    """
    out = np.array(image, copy=True)
    h, w = out.shape[:2]
    for box in boxes_normalized:
        x0, y0, x1, y1 = (float(v) for v in box)
        px0 = int(np.floor(max(0.0, min(x0, x1)) * w))
        py0 = int(np.floor(max(0.0, min(y0, y1)) * h))
        px1 = int(np.ceil(min(1.0, max(x0, x1)) * w))
        py1 = int(np.ceil(min(1.0, max(y0, y1)) * h))
        if px1 > px0 and py1 > py0:
            out[py0:py1, px0:px1] = fill
    return out
