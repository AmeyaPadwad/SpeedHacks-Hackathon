"""Show Pette only while VS Code or a terminal is the foreground app (Windows).

Polls the foreground window's process name. Pette's own windows and the taskbar/tray are
"neutral" (no change), so clicking Pette or her tray menu doesn't hide her.
"""
import os
import sys

from PySide6.QtCore import QObject, QTimer, Signal

POLL_MS = 300
DEFAULT_APPS = (
    # VS Code
    "code.exe", "code - insiders.exe",
    # terminals
    "windowsterminal.exe", "openconsole.exe", "conhost.exe", "cmd.exe", "powershell.exe", "pwsh.exe",
    "wezterm-gui.exe", "alacritty.exe", "mintty.exe", "hyper.exe", "tabby.exe", "warp.exe",
)
NEUTRAL_CLASSES = {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "NotifyIconOverflowWindow",
                   "TopLevelWindowForOverflowXamlIsland", "Windows.UI.Core.CoreWindow", "XamlExplorerHostIslandWindow"}


def allowed_apps() -> set[str]:
    raw = os.environ.get("FLOWSTATE_SHOW_IN")
    return {a.strip().lower() for a in raw.split(",") if a.strip()} if raw else set(DEFAULT_APPS)


def classify(exe: str | None, window_class: str, pid: int, own_pid: int, apps: set[str]) -> bool | None:
    """True = show, False = hide, None = neutral (keep the current state)."""
    if exe is None or pid == own_pid or window_class in NEUTRAL_CLASSES:
        return None
    return os.path.basename(exe).lower() in apps


def foreground_info() -> tuple[str | None, str, int]:
    """(exe path, window class, pid) of the foreground window; (None, "", 0) if unknown."""
    if sys.platform != "win32":
        return None, "", 0
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None, "", 0
    cls = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, cls, 256)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None, cls.value, pid.value
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return (buf.value if ok else None), cls.value, pid.value
    finally:
        kernel32.CloseHandle(handle)


class FocusGate(QObject):
    """Drives Overlay.set_context_allowed() from the foreground app."""
    changed = Signal(bool)

    def __init__(self, overlay, *, info=foreground_info, apps: set[str] | None = None):
        super().__init__()
        self.overlay = overlay
        self._info = info
        self.apps = apps or allowed_apps()
        self.enabled = sys.platform == "win32"
        self._own_pid = os.getpid()
        self._timer = QTimer(self, interval=POLL_MS, timeout=self.poll)

    def start(self):
        self._timer.start()
        self.poll()

    def set_enabled(self, on: bool):
        self.enabled = on
        if not on:
            self.overlay.set_context_allowed(True)
        else:
            self.poll()

    def poll(self):
        if not self.enabled:
            return
        exe, cls, pid = self._info()
        verdict = classify(exe, cls, pid, self._own_pid, self.apps)
        if verdict is not None and verdict != self.overlay.context_allowed:
            self.overlay.set_context_allowed(verdict)
            self.changed.emit(verdict)
