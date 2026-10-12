"""Screen capture gates: consent, pause, exclusion, rate limit, change detection, persistence."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from soma.capture.screen import (
    SKIP_EXCLUDED,
    SKIP_NO_CONSENT,
    SKIP_PAUSED,
    SKIP_RATE_LIMITED,
    SKIP_UNCHANGED,
    ScreenCapturer,
    _clip_region,
)
from soma.capture.window import foreground_window, neutral_window, static_window
from soma.config import CaptureConfig
from soma.consent import ConsentManager, Scope
from soma.types import WindowInfo


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeGrab:
    """Returns a controllable image and counts calls."""

    def __init__(self, h: int = 120, w: int = 200) -> None:
        self.calls = 0
        self.image = np.full((h, w, 3), 200, np.uint8)

    def __call__(self, window: WindowInfo) -> np.ndarray:
        self.calls += 1
        return self.image.copy()


def make(granted: bool = True, app: str = "Code.exe", title: str = "main.py - VS Code",
         cfg: CaptureConfig | None = None):
    consent = ConsentManager()
    if granted:
        consent.grant(Scope.SCREEN_CAPTURE)
    grab, clock = FakeGrab(), FakeClock()
    cap = ScreenCapturer(consent, cfg or CaptureConfig(min_interval_s=1.0, change_threshold=0.01),
                         grab=grab, window_fn=static_window(app, title), clock=clock)
    return cap, consent, grab, clock


# ----------------------------------------------------------------- gates never grab

def test_no_consent_never_grabs():
    cap, _, grab, _ = make(granted=False)
    for _ in range(3):
        assert cap.capture() is None
        assert cap.capture(force=True) is None
    assert grab.calls == 0
    assert cap.last_skip_reason == SKIP_NO_CONSENT
    assert cap.stats.skipped_no_consent == 6


def test_paused_never_grabs_and_resume_works():
    cap, consent, grab, clock = make()
    consent.pause()
    assert cap.capture(force=True) is None
    assert cap.last_skip_reason == SKIP_PAUSED
    assert grab.calls == 0
    consent.resume()
    assert cap.capture() is not None
    assert grab.calls == 1


def test_window_fn_not_called_without_consent():
    consent = ConsentManager()
    calls = []

    def window_fn():
        calls.append(1)
        return WindowInfo("Code.exe", "x")

    cap = ScreenCapturer(consent, CaptureConfig(), grab=FakeGrab(), window_fn=window_fn)
    assert cap.capture() is None
    assert calls == []


@pytest.mark.parametrize("app,title", [
    ("1Password.exe", "Vault"),
    ("msedge.exe", "Bank - InPrivate"),
    ("chrome.exe", "Sign in - Google Accounts"),
    ("CredentialUIBroker.exe", "Windows Security"),
])
def test_excluded_window_never_grabs(app, title):
    cap, _, grab, _ = make(app=app, title=title)
    assert cap.capture(force=True) is None
    assert cap.last_skip_reason.startswith(SKIP_EXCLUDED)
    assert grab.calls == 0
    assert cap.stats.skipped_excluded == 1


def test_user_excluded_app():
    cap, _, grab, _ = make(app="Slack.exe", title="general",
                           cfg=CaptureConfig(excluded_apps=["slack"]))
    assert cap.capture() is None
    assert grab.calls == 0


def test_revoking_consent_stops_capture():
    cap, consent, grab, clock = make()
    assert cap.capture() is not None
    consent.revoke(Scope.SCREEN_CAPTURE)
    clock.t += 10
    assert cap.capture() is None
    assert grab.calls == 1


# ----------------------------------------------------------------- rate limit + change

def test_rate_limit():
    cap, _, grab, clock = make()
    f1 = cap.capture()
    assert f1 is not None and f1.version == 1
    clock.t += 0.5
    grab.image[:] = 0                       # changed, but too soon
    assert cap.capture() is None
    assert cap.last_skip_reason == SKIP_RATE_LIMITED
    assert grab.calls == 1                  # rate limit is checked before grabbing
    clock.t += 0.6
    f2 = cap.capture()
    assert f2 is not None and f2.version == 2


def test_change_detection_skips_identical_and_tiny_changes():
    cap, _, grab, clock = make()
    assert cap.capture() is not None
    clock.t += 2
    assert cap.capture() is None
    assert cap.last_skip_reason == SKIP_UNCHANGED
    # change ~0.1% of the screen: below threshold
    grab.image[0:2, 0:2] = 0
    clock.t += 2
    assert cap.capture() is None
    assert cap.stats.skipped_unchanged == 2
    # change a large block: meaningful
    grab.image[:60, :100] = 0
    clock.t += 2
    frame = cap.capture()
    assert frame is not None and frame.version == 2
    assert cap.stats.last_change_ratio > 0.2


def test_force_bypasses_rate_limit_and_change_only():
    cap, _, grab, clock = make()
    assert cap.capture() is not None
    assert cap.capture(force=True).version == 2
    assert cap.capture(force=True).version == 3


def test_window_switch_counts_as_change():
    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    grab, clock = FakeGrab(), FakeClock()
    windows = iter([WindowInfo("Code.exe", "a", window_id="1"),
                    WindowInfo("Code.exe", "a", window_id="1"),
                    WindowInfo("WindowsTerminal.exe", "b", window_id="2")])
    cap = ScreenCapturer(consent, CaptureConfig(), grab=grab, window_fn=lambda: next(windows),
                         clock=clock)
    assert cap.capture() is not None
    clock.t += 2
    assert cap.capture() is None              # same window, same pixels
    clock.t += 2
    assert cap.capture() is not None          # different window, same pixels


def test_versions_are_monotonic_and_frame_contents():
    cap, _, grab, clock = make(title="token=abc123 - notes")
    versions = []
    for i in range(5):
        grab.image[:] = (i * 40) % 256
        clock.t += 1.5
        f = cap.capture()
        assert f is not None
        versions.append(f.version)
        assert f.image.dtype == np.uint8 and f.image.shape == (120, 200, 3)
        assert f.size == (200, 120)
        assert "abc123" not in f.window.title
    assert versions == sorted(versions) and len(set(versions)) == 5
    assert cap.version == 5


def test_grab_error_is_a_skip():
    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)

    def bad(_w):
        raise OSError("display gone")

    cap = ScreenCapturer(consent, CaptureConfig(), grab=bad, window_fn=static_window("Code.exe"))
    assert cap.capture() is None
    assert cap.last_skip_reason.startswith("grab-error")
    assert cap.stats.errors == 1


# ----------------------------------------------------------------- persistence

def test_screenshots_not_persisted_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cap, _, grab, clock = make()
    for i in range(3):
        grab.image[:] = i * 50
        clock.t += 2
        frame = cap.capture()
        assert frame is not None
        assert cap.persist(frame, tmp_path) is None
    assert list(tmp_path.iterdir()) == []
    assert cap.stats.persisted == 0


def test_persist_requires_flag_and_dir_and_masks(tmp_path):
    from PIL import Image

    cap, _, grab, _ = make(cfg=CaptureConfig(persist_screenshots=True))
    frame = cap.capture()
    assert cap.persist(frame, None) is None
    path = cap.persist(frame, tmp_path / "shots", extra_masks=[(0.0, 0.0, 0.5, 0.5)])
    assert path is not None and path.parent == tmp_path / "shots"
    saved = np.asarray(Image.open(path))
    assert (saved[:60, :100] == 0).all()
    assert (saved[60:, 100:] == 200).all()
    assert (frame.image == 200).all()          # in-memory frame untouched


# ----------------------------------------------------------------- window helpers

def test_foreground_window_never_crashes():
    w = foreground_window()
    assert isinstance(w, WindowInfo)
    if sys.platform != "win32":
        assert w == neutral_window()


def test_clip_region():
    virtual = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    assert _clip_region((-7, -7, 1550, 926), virtual) == {"left": 0, "top": 0, "width": 1543, "height": 919}
    assert _clip_region((0, 0, 0, 0), virtual) is None
    assert _clip_region((5000, 0, 100, 100), virtual) is None


# ----------------------------------------------------------------- accessibility

from soma.capture.accessibility import SimpleNode, elements_from_tree, read_accessibility  # noqa: E402


def _tree() -> SimpleNode:
    return SimpleNode("Editor", "WindowControl", (100, 100, 1100, 900), [
        SimpleNode("Run tests", "ButtonControl", (150, 120, 250, 150)),
        SimpleNode("", "PaneControl", (100, 160, 1100, 900), [
            SimpleNode("api_key=abc123", "TextControl", (120, 200, 600, 220)),
            SimpleNode("Offscreen", "ButtonControl", (5000, 5000, 5100, 5050)),
        ]),
    ])


def test_accessibility_tree_flattened_with_roles():
    w = WindowInfo("Code.exe", "x", window_id="hwnd:1", bounds=(100, 100, 1000, 800))
    els = elements_from_tree(_tree(), w, timestamp_ms=7)
    by_text = {e.text: e for e in els}
    assert "Offscreen" not in by_text
    btn = by_text["Run tests"]
    assert btn.source == "accessibility" and btn.ui_role == "button"
    assert btn.bbox_xyxy_normalized == pytest.approx((0.05, 0.025, 0.15, 0.0625))
    assert btn.window_id == "hwnd:1" and btn.timestamp_ms == 7
    assert any(e.ui_role == "text" and "abc123" not in e.text for e in els)
    assert by_text["Editor"].ui_role == "window"


def test_accessibility_gates_and_unavailable_backend():
    w = WindowInfo("Code.exe", "x", bounds=(100, 100, 1000, 800))
    consent = ConsentManager()
    assert read_accessibility(w, consent, root_fn=_tree) == []          # no consent
    consent.grant(Scope.SCREEN_CAPTURE)
    assert len(read_accessibility(w, consent, root_fn=_tree)) == 3
    consent.pause()
    assert read_accessibility(w, consent, root_fn=_tree) == []
    assert read_accessibility(WindowInfo("1Password.exe", "v", bounds=(0, 0, 9, 9)),
                              root_fn=_tree) == []

    def boom():
        raise RuntimeError("COM failure")

    assert read_accessibility(w, root_fn=boom) == []
    assert isinstance(read_accessibility(w), list)                     # real backend or []


@pytest.mark.skipif(sys.platform != "win32", reason="real screen grab only exercised on Windows")
def test_real_mss_grab_smoke(tmp_path, monkeypatch):
    from soma.capture.screen import mss_grab

    monkeypatch.chdir(tmp_path)
    try:
        img = mss_grab(WindowInfo("x", "y", bounds=(0, 0, 0, 0)))
    except Exception as exc:  # headless CI / no display
        pytest.skip(f"no display: {exc}")
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert list(tmp_path.iterdir()) == []
