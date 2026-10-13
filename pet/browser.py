"""Hand gestures that drive the browser in front: point left / right switches tabs, pinch-and-drag scrolls.

Only when the foreground window is a browser (so a gesture never types into your editor or terminal), only
while Pette isn't paused, and only these inputs: Ctrl+Tab / Ctrl+Shift+Tab and the mouse wheel. Scrolling is
touch-style: move your pinched hand up and the page moves up with it. The wheel goes to the window under the
mouse pointer (Windows' rule), so keep the pointer over the page.
"""
import sys
from pathlib import Path

from . import focus

BROWSERS = {"chrome", "msedge", "firefox", "brave", "arc", "opera", "vivaldi", "comet", "chromium", "thorium",
            "zen", "librewolf", "waterfox", "floorp"}
NEXT_TAB, PREV_TAB = "ctrl+tab", "ctrl+shift+tab"
NOTCHES_PER_HEIGHT = 30   # moving the pinch the full camera height scrolls 30 wheel notches (~3,000 px)
WHEEL_DELTA = 120


def is_browser(exe: str | None) -> bool:
    name = Path(exe or "").name.lower()
    return (name[:-4] if name.endswith(".exe") else name) in BROWSERS


def _send_keys(combo: str):
    import keyboard
    keyboard.send(combo)


def _send_wheel(notches: int):
    """One wheel event of `notches` clicks (+ = scroll up), like a real mouse wheel."""
    if sys.platform != "win32":
        return
    import ctypes
    MOUSEEVENTF_WHEEL = 0x0800
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(notches * WHEEL_DELTA), 0)


class BrowserControl:
    """Collaborators are injectable: `foreground() -> (exe, class, pid)`, `send_keys(combo)`, `wheel(notches)`."""

    def __init__(self, *, foreground=focus.foreground_info, send_keys=_send_keys, wheel=_send_wheel):
        self._foreground, self._send_keys, self._wheel = foreground, send_keys, wheel
        self.enabled = True
        self._carry = 0.0     # fractional notches not sent yet

    def in_browser(self) -> bool:
        try:
            exe = self._foreground()[0]
        except Exception:
            return False
        return is_browser(exe)

    def switch_tab(self, step: int) -> bool:
        """+1 = next tab (Ctrl+Tab), -1 = previous tab (Ctrl+Shift+Tab). False if not in a browser."""
        if not self.enabled or step == 0 or not self.in_browser():
            return False
        self._send_keys(NEXT_TAB if step > 0 else PREV_TAB)
        return True

    def scroll(self, dy: float) -> int:
        """Touch-style scroll for a pinch that moved `dy` image heights (+ = down). Returns notches sent."""
        if not self.enabled or not self.in_browser():
            self._carry = 0.0
            return 0
        self._carry += dy * NOTCHES_PER_HEIGHT     # hand down -> page down with it -> wheel up (+)
        notches = int(self._carry)                 # whole notches only; keep the remainder
        if notches:
            self._carry -= notches
            self._wheel(notches)
        return notches

    def reset(self):
        self._carry = 0.0
