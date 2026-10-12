"""Foreground window metadata (app name, title, bounds). No pixels, no content.

On Windows this uses ``ctypes`` against ``user32``/``kernel32``
(``GetForegroundWindow``, ``GetWindowTextW``, ``GetWindowRect``,
``QueryFullProcessImageNameW``). On other platforms, or if any call fails, it
returns a neutral :class:`WindowInfo` so callers never crash.
"""

from __future__ import annotations

import ntpath
import sys
from typing import Callable

from soma.types import WindowInfo

WindowFn = Callable[[], WindowInfo]

NEUTRAL_WINDOW = WindowInfo(app_name="", title="", window_id="", pid=0, bounds=(0, 0, 0, 0))

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def neutral_window() -> WindowInfo:
    """A fresh empty :class:`WindowInfo` (unknown app, no bounds)."""
    return WindowInfo(app_name="", title="", window_id="", pid=0, bounds=(0, 0, 0, 0))


def _win32_foreground() -> WindowInfo:  # pragma: no cover - exercised on Windows only
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return neutral_window()

    # Title
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(max(length, 0) + 1)
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW(hwnd, buf, len(buf))
    title = buf.value

    # Bounds
    rect = wintypes.RECT()
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    bounds = (0, 0, 0, 0)
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        bounds = (rect.left, rect.top, max(0, rect.right - rect.left), max(0, rect.bottom - rect.top))

    # Process image name
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    app_name = ""
    if pid.value:
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if handle:
            try:
                size = wintypes.DWORD(1024)
                path_buf = ctypes.create_unicode_buffer(size.value)
                kernel32.QueryFullProcessImageNameW.argtypes = [
                    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
                if kernel32.QueryFullProcessImageNameW(handle, 0, path_buf, ctypes.byref(size)):
                    app_name = ntpath.basename(path_buf.value)
            finally:
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle(handle)

    return WindowInfo(app_name=app_name, title=title, window_id=f"hwnd:{int(hwnd)}",
                      pid=int(pid.value), bounds=bounds)


def foreground_window() -> WindowInfo:
    """Metadata of the current foreground window; neutral when unavailable."""
    if sys.platform != "win32":
        return neutral_window()
    try:
        return _win32_foreground()
    except Exception:  # noqa: BLE001 - never crash the capture loop on OS quirks
        return neutral_window()


def static_window(app_name: str, title: str = "", bounds: tuple[int, int, int, int] = (0, 0, 0, 0),
                  window_id: str = "", pid: int = 0) -> WindowFn:
    """Return a ``window_fn`` that always reports the given window (tests, demos)."""
    info = WindowInfo(app_name=app_name, title=title, window_id=window_id or f"static:{app_name}",
                      pid=pid, bounds=bounds)
    return lambda: info
