import os

import pytest

from pet.focus import DEFAULT_APPS, FocusGate, allowed_apps, classify
from pet.ui import Overlay

APPS = set(DEFAULT_APPS)
OWN = os.getpid()


@pytest.mark.parametrize("exe,expected", [
    (r"C:\Users\me\AppData\Local\Programs\Microsoft VS Code\Code.exe", True),
    (r"C:\Program Files\WindowsApps\Microsoft.WindowsTerminal\WindowsTerminal.exe", True),
    (r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe", True),
    (r"C:\Program Files\Git\usr\bin\mintty.exe", True),
    (r"C:\Program Files\Google\Chrome\Application\chrome.exe", False),
    (r"C:\Users\me\AppData\Local\Discord\Discord.exe", False),
])
def test_classify_apps(exe, expected):
    assert classify(exe, "SomeWindow", 1234, OWN, APPS) is expected


def test_neutral_cases_keep_state():
    assert classify(r"C:\Windows\explorer.exe", "Shell_TrayWnd", 1, OWN, APPS) is None   # taskbar / tray
    assert classify(r"C:\Python\python.exe", "Qt", OWN, OWN, APPS) is None               # Pette's own windows
    assert classify(None, "", 0, OWN, APPS) is None                                       # unknown


def test_env_override(monkeypatch):
    monkeypatch.setenv("FLOWSTATE_SHOW_IN", "Cursor.exe, code.exe")
    assert allowed_apps() == {"cursor.exe", "code.exe"}


def test_gate_hides_and_shows_overlay_and_blocks_hotkeys(qapp):
    o = Overlay(tray=False)
    o.show()
    fg = {"v": (r"C:\chrome.exe", "Chrome_WidgetWin_1", 10)}
    gate = FocusGate(o, info=lambda: fg["v"], apps=APPS)
    gate.enabled = True
    gate.poll()
    assert not o.context_allowed and o.isHidden()
    fg["v"] = (r"C:\Windows\explorer.exe", "Shell_TrayWnd", 11)   # clicking the tray: no change
    gate.poll()
    assert o.isHidden()
    fg["v"] = (r"C:\VS Code\Code.exe", "Chrome_WidgetWin_1", 12)
    gate.poll()
    assert o.context_allowed and o.isVisible()
    gate.set_enabled(False)
    fg["v"] = (r"C:\chrome.exe", "Chrome_WidgetWin_1", 10)
    gate.poll()
    assert o.isVisible()                                          # gate off: always shown


def test_look_now_is_ignored_outside_allowed_apps(qapp):
    from pet import soma
    from tests.test_soma import FakeClient
    o = Overlay(tray=False)
    client = FakeClient()
    s = soma.Suggester(o, client, spawn=lambda f: f())   # keep a reference: Qt drops a collected QObject
    o.set_context_allowed(False)
    o.look_requested.emit()
    assert client.calls == [] and o.bubble.proposal is None
    o.set_context_allowed(True)
    o.look_requested.emit()
    assert client.calls == [("trigger",)] and s.state == "looking"
