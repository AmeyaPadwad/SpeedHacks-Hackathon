"""Privacy: window exclusion, secret redaction and region masking."""

from __future__ import annotations

import numpy as np
import pytest

from soma.capture.redaction import (
    contains_secret,
    is_excluded,
    luhn_valid,
    mask_regions,
    redact_elements,
    redact_text,
    sensitive_boxes,
)
from soma.config import CaptureConfig
from soma.types import OCRElement, WindowInfo


def win(app: str, title: str = "") -> WindowInfo:
    return WindowInfo(app_name=app, title=title)


# ----------------------------------------------------------------- exclusion

@pytest.mark.parametrize("app,title,reason_prefix", [
    ("1Password.exe", "Vault", "excluded-app"),
    ("KeePassXC.exe", "db.kdbx", "excluded-app"),
    ("Bitwarden.exe", "My vault", "excluded-app"),
    ("CredentialUIBroker.exe", "Windows Security", "excluded-app"),
    ("consent.exe", "User Account Control", "excluded-app"),
    ("msedge.exe", "New tab - [InPrivate] - Microsoft Edge", "private-browsing"),
    ("chrome.exe", "New Tab - Google Chrome (Incognito)", "private-browsing"),
    ("firefox.exe", "Mozilla Firefox Private Browsing", "private-browsing"),
    ("chrome.exe", "Chase Online Banking - Google Chrome", "banking"),
    ("chrome.exe", "PayPal: Summary", "banking"),
    ("chrome.exe", "Sign in to GitHub", "auth-prompt"),
    ("chrome.exe", "2-Step Verification", "auth-prompt"),
])
def test_builtin_exclusions(app, title, reason_prefix):
    excluded, reason = is_excluded(win(app, title), CaptureConfig(excluded_apps=[],
                                                                  excluded_title_patterns=[]))
    assert excluded
    assert reason.startswith(reason_prefix)


def test_user_exclusions_case_insensitive():
    cfg = CaptureConfig(excluded_apps=["SLACK"], excluded_title_patterns=["Payroll"])
    assert is_excluded(win("slack.exe", "general"), cfg)[0]
    assert is_excluded(win("EXCEL.EXE", "PAYROLL 2026.xlsx"), cfg) == (True, "user-excluded-title:payroll")


def test_regular_apps_not_excluded():
    cfg = CaptureConfig()
    for app, title in [("Code.exe", "main.py - SOMA - Visual Studio Code"),
                       ("WindowsTerminal.exe", "pytest"),
                       ("chrome.exe", "PyTorch documentation - Google Chrome")]:
        assert is_excluded(win(app, title), cfg) == (False, "")


def test_empty_window_not_excluded():
    assert is_excluded(win("", ""))[0] is False


# ----------------------------------------------------------------- text

SECRETS = {
    "openai": "sk-proj-abcdefghijklmnopqrstuvwx1234",
    "openai_plain": "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    "aws": "AKIAIOSFODNN7EXAMPLE",
    "github": "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
    "github_pat": "github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "slack": "xoxb-1234567890-abcdefghij",
    "email": "jane.doe@example.com",
    "card": "4111 1111 1111 1111",
    "card_dashes": "5500-0000-0000-0004",
    "hex": "a3f5c9e1b2d4f6a8c0e2b4d6f8a0c2e4",
    "b64": "Zm9vYmFyQmF6UXV4MTIzNDU2Nzg5MGFiY2RlZmdoaWprbG1ub3A",
}


@pytest.mark.parametrize("kind", sorted(SECRETS))
def test_each_secret_class_is_masked(kind):
    secret = SECRETS[kind]
    text = f"value: {secret} end"
    out = redact_text(text)
    assert secret not in out
    assert "[REDACTED:" in out
    assert out.startswith("value") and out.endswith("end")
    assert contains_secret(text)


@pytest.mark.parametrize("text,secret", [
    ("password=hunter2", "hunter2"),
    ("PASSWORD: 'correct horse battery'", "correct horse battery"),
    ('api_key = "abc123"', "abc123"),
    ("export ACCESS_TOKEN=xyz987&next=1", "xyz987"),
    ('{"client_secret": "s3cr3t"}', "s3cr3t"),
])
def test_key_value_pairs(text, secret):
    out = redact_text(text)
    assert secret not in out
    assert "[REDACTED:secret]" in out


def test_non_luhn_numbers_are_kept():
    text = "order 1234 5678 9012 3456 shipped"   # fails Luhn
    assert not luhn_valid("1234567890123456")
    assert redact_text(text) == text


def test_ordinary_text_is_untouched():
    for text in ["Traceback (most recent call last):",
                 'File "app.py", line 12, in <module>',
                 "FAILED tests/test_x.py::test_add - AssertionError",
                 "Run tests", "def password_strength(pw):", ""]:
        assert redact_text(text) == text


def test_redact_elements_and_sensitive_boxes():
    els = [OCRElement("hello", (0.0, 0.0, 0.1, 0.1)),
           OCRElement("token=abc", (0.2, 0.2, 0.4, 0.3))]
    out = redact_elements(els)
    assert out[0].text == "hello"
    assert "abc" not in out[1].text
    assert els[1].text == "token=abc"           # originals untouched
    assert sensitive_boxes(els) == [(0.2, 0.2, 0.4, 0.3)]
    assert sensitive_boxes(out) == [(0.2, 0.2, 0.4, 0.3)]


# ----------------------------------------------------------------- masking

def test_mask_regions_blacks_out_copy():
    img = np.full((100, 200, 3), 255, np.uint8)
    out = mask_regions(img, [(0.25, 0.5, 0.5, 1.0)])
    assert img.min() == 255                      # input untouched
    assert (out[50:100, 50:100] == 0).all()
    assert (out[:50] == 255).all()
    assert (out[:, 100:] == 255).all()


def test_mask_regions_clips_and_handles_degenerate_boxes():
    img = np.full((10, 10), 9, np.uint8)
    out = mask_regions(img, [(-1, -1, 0.5, 0.5), (0.6, 0.6, 0.6, 0.9), (2, 2, 3, 3)])
    assert (out[:5, :5] == 0).all()
    assert out.sum() == 9 * (100 - 25)
