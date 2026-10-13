"""Gestures drive only the browser in front: tabs with Ctrl+(Shift+)Tab, touch-style wheel scrolling."""
import pytest

from db import Store
from pet import browser as br
from pet.ui import Overlay
from pet.vision import DRAG_END, DRAG_START, PINCH, POINT_LEFT, POINT_RIGHT
from pet.watcher import Watcher


class Fakes:
    def __init__(self, exe="C:/Program Files/Google/Chrome/Application/chrome.exe"):
        self.exe, self.keys, self.wheel = exe, [], []

    def control(self):
        return br.BrowserControl(foreground=lambda: (self.exe, "Chrome_WidgetWin_1", 7),
                                 send_keys=self.keys.append, wheel=self.wheel.append)


@pytest.mark.parametrize("exe,yes", [("chrome.exe", True), (r"C:\x\msedge.exe", True), ("firefox", True),
                                     ("Comet.EXE", True), ("Code.exe", False), ("WindowsTerminal.exe", False),
                                     (None, False), ("", False)])
def test_is_browser(exe, yes):
    assert br.is_browser(exe) is yes


def test_tabs_only_in_a_browser():
    f = Fakes()
    c = f.control()
    assert c.switch_tab(+1) and c.switch_tab(-1)
    assert f.keys == [br.NEXT_TAB, br.PREV_TAB]
    f.exe = "Code.exe"
    assert not c.switch_tab(+1) and f.keys == [br.NEXT_TAB, br.PREV_TAB]     # never types into the editor
    f.exe = "chrome.exe"
    c.enabled = False
    assert not c.switch_tab(+1)


def test_scroll_is_touch_style_and_keeps_fractions():
    f = Fakes()
    c = f.control()
    assert c.scroll(-0.02) == 0                       # 0.6 notch: carried, not sent
    assert c.scroll(-0.02) == -1                      # 1.2 notches total -> one down (hand up = page up)
    assert c.scroll(0.1) == 2 and f.wheel == [-1, 2]  # hand down -> scroll up (-0.2 carried over)
    f.exe = "Code.exe"
    assert c.scroll(1.0) == 0 and f.wheel == [-1, 2]  # not a browser: nothing, and the carry is dropped
    f.exe = "chrome.exe"
    assert c.scroll(0.02) == 0


def test_foreground_errors_mean_no_input():
    c = br.BrowserControl(foreground=lambda: 1 / 0, send_keys=lambda k: pytest.fail("typed"),
                          wheel=lambda n: pytest.fail("scrolled"))
    assert not c.switch_tab(1) and c.scroll(1.0) == 0


def test_watcher_routes_gestures_and_logs_facts(qapp, tmp_path):
    f = Fakes()
    store = Store(tmp_path / "m.sqlite3")
    o = Overlay(tray=False)
    w = Watcher(o, memory=store, browser=f.control())
    w.on_vision_event(POINT_RIGHT)
    w.on_vision_event(POINT_LEFT)
    assert f.keys == [br.NEXT_TAB, br.PREV_TAB]
    w.on_vision_event(DRAG_START)
    w.on_scroll(-0.1)
    w.on_vision_event(DRAG_END)
    assert f.wheel == [-3]
    w.on_vision_event(PINCH)                          # no card: nothing happens, not a tab switch
    assert f.keys == [br.NEXT_TAB, br.PREV_TAB]
    o.set_paused(True)
    w.on_vision_event(POINT_RIGHT)
    w.on_scroll(-1.0)
    assert f.keys == [br.NEXT_TAB, br.PREV_TAB] and f.wheel == [-3]          # paused: nothing
    kinds = [k for (k,) in store.query("SELECT kind FROM events ORDER BY rowid")]
    assert kinds == ["point_right", "point_left", "drag_start", "drag_end", "pinch"]
    store.close()


def test_watcher_without_browser_control_ignores_tab_gestures(qapp):
    o = Overlay(tray=False)
    w = Watcher(o)
    w.on_vision_event(POINT_LEFT)
    w.on_scroll(0.5)                                  # no crash, nothing to drive
