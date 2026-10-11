"""Native one-window snapshot and a hotkey listener; optional imports stay lazy."""
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .packet import MAX_IMAGE_BYTES, WindowContext


@dataclass(frozen=True)
class WindowCapture:
    window: WindowContext = WindowContext()
    png: bytes | None = None
    warning: str = ""


@dataclass(frozen=True)
class WindowTarget:
    pid: int
    window: WindowContext


def foreground_target() -> WindowTarget | None:
    """Remember the IDE/browser while Pette has not yet taken focus."""
    if sys.platform != "darwin":
        return None
    try:
        import Quartz as Q
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None or app.processIdentifier() == os.getpid():
            return None
        windows = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements,
                                             Q.kCGNullWindowID)
        window = next((w for w in windows if w.get(Q.kCGWindowOwnerPID) == app.processIdentifier()
                       and w.get(Q.kCGWindowLayer) == 0
                       and w.get(Q.kCGWindowBounds, {}).get("Width", 0) > 1), None)
        if window is None:
            return None
        return WindowTarget(app.processIdentifier(), WindowContext(app.localizedName() or "",
            window.get(Q.kCGWindowName, "") or "", int(window[Q.kCGWindowNumber])))
    except Exception:
        return None


def _capture_window_png(Q, window_id):
    """Prefer Quartz; fall back to macOS's window-only capture command."""
    try:
        from Foundation import NSMutableData
        image = Q.CGWindowListCreateImage(Q.CGRectNull, Q.kCGWindowListOptionIncludingWindow,
                                        window_id, Q.kCGWindowImageBoundsIgnoreFraming)
        if image is not None:
            data = NSMutableData.data()
            destination = Q.CGImageDestinationCreateWithData(data, "public.png", 1, None)
            Q.CGImageDestinationAddImage(destination, image, None)
            if Q.CGImageDestinationFinalize(destination):
                return bytes(data)
    except Exception:
        pass
    with tempfile.TemporaryDirectory(prefix="flowstate-screen-") as folder:
        path = Path(folder) / "window.png"
        result = subprocess.run(["/usr/sbin/screencapture", "-x", "-o", "-t", "png", "-l", str(window_id), str(path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        if result.returncode != 0 or not path.is_file():
            raise RuntimeError("Window capture failed")
        with path.open("rb") as stream:
            return stream.read(MAX_IMAGE_BYTES + 1)


def capture_foreground(previous: WindowTarget | None = None) -> WindowCapture:
    if sys.platform != "darwin":
        return WindowCapture(warning="Foreground-window capture is implemented for macOS only.")
    context = WindowContext()
    try:
        import Quartz as Q
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return WindowCapture(warning="No foreground app is available. Switch to your IDE and capture again.")
        context = WindowContext(app.localizedName() or "", "", None)
        pid = app.processIdentifier()
        if app.processIdentifier() == os.getpid():
            if previous is None or previous.pid == os.getpid():
                return WindowCapture(context, warning="Pette is foreground and no previous IDE/browser window is known. Switch to your IDE/terminal and start a new capture with the hotkey.")
            pid, context = previous.pid, previous.window
        windows = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements,
                                             Q.kCGNullWindowID)
        window = next((w for w in windows if w.get(Q.kCGWindowOwnerPID) == pid
                       and w.get(Q.kCGWindowLayer) == 0
                       and (app.processIdentifier() != os.getpid() or w.get(Q.kCGWindowNumber) == context.window_id)
                       and w.get(Q.kCGWindowBounds, {}).get("Width", 0) > 1), None)
        if window is None:
            return WindowCapture(context, warning="No capturable foreground window. Continue with project files.")
        window_id = int(window[Q.kCGWindowNumber])
        context = WindowContext(context.app, window.get(Q.kCGWindowName, "") or "", window_id)
        if not Q.CGPreflightScreenCaptureAccess():
            return WindowCapture(context, warning="macOS has not granted Screen Recording to this process. Enable it for the app launching Python (VS Code, Terminal, or iTerm) in System Settings → Privacy & Security, then fully quit and reopen that app.")
        png = _capture_window_png(Q, window_id)
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            return WindowCapture(context, warning="Window capture returned invalid image data. Switch to your IDE and retry.")
        if len(png) > MAX_IMAGE_BYTES:
            return WindowCapture(context, warning="Window screenshot exceeds 4.9 MB; use file context instead.")
        return WindowCapture(context, png)
    except subprocess.TimeoutExpired:
        return WindowCapture(context, warning="Window capture timed out. Switch to your IDE and retry.")
    except Exception as exc:
        return WindowCapture(context, warning=f"Window capture failed ({type(exc).__name__}). Check Screen Recording access and retry from your IDE.")


class Hotkeys:
    def __init__(self):
        self.listener = None
        self.error = ""

    def start(self, trigger, cancel) -> bool:
        self.error = ""
        try:
            if sys.platform == "darwin":
                import Quartz
                if not Quartz.CGPreflightListenEventAccess():
                    self.error = (
                        "macOS has not granted Input Monitoring to this process. "
                        "Enable it for the app launching Python (VS Code, Terminal, or iTerm), "
                        "then fully quit and reopen that app."
                    )
                    return False
            from pynput.keyboard import GlobalHotKeys
            value = os.environ.get("FLOWSTATE_HOTKEY") or "ctrl+shift+space"
            parts = value.lower().split("+")
            hotkey = "+".join(f"<{p}>" if p in {"ctrl", "shift", "alt", "cmd", "space"} else p for p in parts)
            listener = GlobalHotKeys({hotkey: trigger, "<esc>": cancel})
            # On macOS, pynput initializes IS_TRUSTED to False and updates it
            # inside the listener thread. Checking it before start() prevents
            # even an authorized listener from ever starting.
            listener.start()
            self.listener = listener
            return True
        except ImportError:
            self.error = "Hotkey dependencies are missing. Run python -m pip install -r requirements-input.txt."
            return False
        except ValueError:
            self.error = "Invalid FLOWSTATE_HOTKEY. Use a chord such as ctrl+shift+space."
            return False
        except Exception as exc:
            self.error = f"Could not start the hotkey listener ({type(exc).__name__})."
            return False

    def close(self):
        if self.listener:
            self.listener.stop()
            self.listener.join(timeout=1)
            self.listener = None
