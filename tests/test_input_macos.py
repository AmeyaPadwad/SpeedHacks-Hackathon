import os
import sys
from types import SimpleNamespace
from pathlib import Path

from pet.inputs.macos import Hotkeys, capture_foreground, foreground_target


def native_fakes(monkeypatch, *, permission=True, pid=None):
    monkeypatch.setattr("pet.inputs.macos.sys.platform", "darwin")
    app = SimpleNamespace(localizedName=lambda: "Code", processIdentifier=lambda: pid or os.getpid() + 1)
    workspace = SimpleNamespace(sharedWorkspace=lambda: SimpleNamespace(frontmostApplication=lambda: app))
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSWorkspace=workspace))
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSMutableData=SimpleNamespace(data=lambda: bytearray(b"\x89PNG\r\n\x1a\nimage"))))
    captured = []
    q = SimpleNamespace(kCGWindowListOptionOnScreenOnly=1, kCGWindowListExcludeDesktopElements=2,
        kCGNullWindowID=0, kCGWindowOwnerPID="pid", kCGWindowLayer="layer", kCGWindowBounds="bounds",
        kCGWindowNumber="id", kCGWindowName="title", CGRectNull=None,
        kCGWindowListOptionIncludingWindow=4, kCGWindowImageBoundsIgnoreFraming=8,
        CGWindowListCopyWindowInfo=lambda *args: [
            {"pid": app.processIdentifier(), "layer": 1, "id": 3, "bounds": {"Width": 200}},
            {"pid": app.processIdentifier(), "layer": 0, "id": 7, "title": "auth.py", "bounds": {"Width": 200}}],
        CGPreflightScreenCaptureAccess=lambda: permission,
        CGWindowListCreateImage=lambda *args: (captured.append(args) or object()),
        CGImageDestinationCreateWithData=lambda *args: object(),
        CGImageDestinationAddImage=lambda *args: None,
        CGImageDestinationFinalize=lambda *args: True,
        CGPreflightListenEventAccess=lambda: permission)
    monkeypatch.setitem(sys.modules, "Quartz", q)
    return captured


def test_native_capture_targets_foreground_app_window_not_monitor(monkeypatch):
    captured = native_fakes(monkeypatch)
    result = capture_foreground()
    assert result.window.window_id == 7 and result.window.title == "auth.py"
    assert result.png.startswith(b"\x89PNG")
    assert captured[0][2] == 7 and not result.warning


def test_missing_permission_never_captures_and_hotkey_falls_back(monkeypatch):
    captured = native_fakes(monkeypatch, permission=False)
    assert "Screen Recording" in capture_foreground().warning
    assert not captured
    keys = Hotkeys()
    assert not keys.start(lambda: None, lambda: None)
    assert "Input Monitoring" in keys.error and "fully quit" in keys.error


def test_panel_foreground_is_not_misrepresented_as_ide(monkeypatch):
    captured = native_fakes(monkeypatch, pid=os.getpid())
    result = capture_foreground()
    assert "Pette is foreground" in result.warning and not result.png
    assert not captured


def test_hotkeys_map_configured_chord_and_escape_and_close(monkeypatch):
    native_fakes(monkeypatch)
    observed = {}

    class Listener:
        # Real pynput defaults to False until its listener thread starts.
        IS_TRUSTED = False

        def __init__(self, bindings):
            observed.update(bindings)

        def start(self):
            observed["started"] = True

        def stop(self):
            observed["stopped"] = True

        def join(self, timeout):
            pass

    monkeypatch.setitem(sys.modules, "pynput.keyboard", SimpleNamespace(GlobalHotKeys=Listener))
    monkeypatch.setenv("FLOWSTATE_HOTKEY", "ctrl+shift+space")
    keys = Hotkeys()
    trigger, cancel = lambda: None, lambda: None
    assert keys.start(trigger, cancel)
    assert observed["<ctrl>+<shift>+<space>"] == trigger and observed["<esc>"] == cancel
    keys.close()
    assert observed["started"] and observed["stopped"] and keys.listener is None


def test_invalid_hotkey_reports_configuration_error(monkeypatch):
    native_fakes(monkeypatch)

    def invalid_listener(bindings):
        raise ValueError("invalid chord")

    monkeypatch.setitem(sys.modules, "pynput.keyboard", SimpleNamespace(GlobalHotKeys=invalid_listener))
    keys = Hotkeys()
    assert not keys.start(lambda: None, lambda: None)
    assert "Invalid FLOWSTATE_HOTKEY" in keys.error and keys.listener is None


def test_listener_start_failure_is_not_reported_as_missing_permission(monkeypatch):
    native_fakes(monkeypatch)

    class Listener:
        def __init__(self, bindings):
            pass

        def start(self):
            raise RuntimeError("listener unavailable")

    monkeypatch.setitem(sys.modules, "pynput.keyboard", SimpleNamespace(GlobalHotKeys=Listener))
    keys = Hotkeys()
    assert not keys.start(lambda: None, lambda: None)
    assert "RuntimeError" in keys.error and "Input Monitoring" not in keys.error
    assert keys.listener is None


def focus_pette(monkeypatch):
    app = SimpleNamespace(localizedName=lambda: "Pette", processIdentifier=lambda: os.getpid())
    workspace = SimpleNamespace(sharedWorkspace=lambda: SimpleNamespace(frontmostApplication=lambda: app))
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSWorkspace=workspace))


def test_clicking_pette_captures_the_remembered_external_window(monkeypatch):
    captured = native_fakes(monkeypatch)
    previous = foreground_target()
    assert previous.window.app == "Code" and previous.window.window_id == 7
    focus_pette(monkeypatch)
    assert foreground_target() is None  # Never overwrite the last IDE with Pette.
    result = capture_foreground(previous)
    assert result.window.app == "Code" and result.window.window_id == 7
    assert result.png and captured[0][2] == 7 and not result.warning


def test_remembered_window_is_not_captured_after_it_closes(monkeypatch):
    captured = native_fakes(monkeypatch)
    previous = foreground_target()
    focus_pette(monkeypatch)
    sys.modules["Quartz"].CGWindowListCopyWindowInfo = lambda *args: []
    result = capture_foreground(previous)
    assert not result.png and not captured and "No capturable" in result.warning


def test_window_only_command_fallback_and_temp_cleanup(monkeypatch):
    native_fakes(monkeypatch)
    sys.modules["Quartz"].CGWindowListCreateImage = lambda *args: None
    calls = []

    def command(args, **kwargs):
        calls.append((args, kwargs))
        Path(args[-1]).write_bytes(b"\x89PNG\r\n\x1a\nimage")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("pet.inputs.macos.subprocess.run", command)
    result = capture_foreground()
    assert result.png and not result.warning
    assert calls[0][0][:-1] == ["/usr/sbin/screencapture", "-x", "-o", "-t", "png", "-l", "7"]
    assert calls[0][1]["timeout"] == 5
    assert not Path(calls[0][0][-1]).parent.exists()


def test_permission_denial_does_not_try_command_fallback(monkeypatch):
    native_fakes(monkeypatch, permission=False)
    monkeypatch.setattr("pet.inputs.macos.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No capture without access")))
    assert "Screen Recording" in capture_foreground().warning
