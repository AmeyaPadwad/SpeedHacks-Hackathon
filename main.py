"""Pette entry point: the UI for SOMA. Pette (sprite + glass bubble + camera orb), world-model panel, tray.

SOMA (SOMA/, started by run_pette_soma.sh) looks at the screen and the folders you index and suggests what
to do next. Pette shows each suggestion as a card; "Open" / "Copy" (or a pinch) does it, "Not now" puts it off.
Click Pette (or Ctrl+Shift+Space) to have SOMA look at the screen right now.
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

# Load before importing pet modules: they read settings at import time. Real env vars win over .env.
load_dotenv(Path(__file__).resolve().parent / ".env")

from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from db import Store, WorldModel  # noqa: E402
from pet import hotkeys, soma  # noqa: E402
from pet.browser import BrowserControl  # noqa: E402
from pet.focus import FocusGate  # noqa: E402
from pet.ui import Overlay  # noqa: E402
from pet.watcher import Watcher  # noqa: E402
from pet.worldview import WorldWindow  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pette: the on-screen face of SOMA")
    ap.add_argument("--soma", default=None, metavar="ENDPOINT_FILE",
                    help=f"SOMA endpoint file written by SOMA/scripts/serve.py (default ${soma.ENDPOINT_VAR})")
    ap.add_argument("--no-camera", action="store_true", help="don't start the webcam (the orb shows 'camera off')")
    ap.add_argument("--no-world", action="store_true", help="don't open the world-model window at launch")
    ap.add_argument("--focus-only", action="store_true", help="show Pette only over VS Code and terminals")
    ap.add_argument("--always-show", action="store_true", help=argparse.SUPPRESS)   # old flag; now the default
    return ap.parse_args(argv)


def build(args: argparse.Namespace) -> SimpleNamespace:
    """Create and wire every component without entering the event loop (tools/sandbox.py reuses this)."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = Overlay()
    memory = Store()                    # db/pette_cache.sqlite3: facts only, local, 30-day retention
    world = WorldModel(memory)
    suggester = soma.Suggester(overlay, connect=lambda: soma.SomaClient.from_endpoint(args.soma),
                               memory=memory, world=world)
    browser = BrowserControl()
    watcher = Watcher(overlay, memory=memory, world=world, browser=browser)
    app.aboutToQuit.connect(memory.close)
    worldwin = WorldWindow(overlay, world, suggester, watcher)
    overlay.world_requested.connect(worldwin.toggle)
    for window in (overlay, overlay.bar, worldwin):   # SOMA must never OCR Pette's own bubble or panel
        soma.exclude_from_capture(window)

    # --- tray: SOMA ---
    soma_action = overlay.add_menu_action(f"SOMA suggestions ({hotkeys.SOMA_HOTKEY.title()})",
                                          suggester.set_enabled, checkable=True, checked=True)
    if soma_action:
        suggester.enabled_changed.connect(soma_action.setChecked)

    def pick_folder():
        path = QFileDialog.getExistingDirectory(None, "Choose a folder for SOMA to learn from")
        suggester.add_folder(path)

    overlay.add_menu_action("Add a folder for SOMA…", pick_folder)
    folders_menu = overlay.add_menu("Indexed folders")

    def show_folders(folders: list):
        if folders_menu is None:
            return
        folders_menu.clear()
        if not folders:
            folders_menu.addAction("(none yet: add a folder)").setEnabled(False)
        for f in folders:
            name = Path(f["path"]).name or f["path"]
            folders_menu.addAction(f"Remove {name} ({f.get('memories', 0)} files)",
                                   lambda p=f["path"]: suggester.remove_folder(p))

    suggester.folders_changed.connect(show_folders)
    show_folders([])

    # --- tray: where Pette shows, camera, memory ---
    gate = FocusGate(overlay)
    overlay.add_menu_action("Show only in VS Code / Terminal", gate.set_enabled, checkable=True,
                            checked=args.focus_only)
    gate.enabled = args.focus_only
    camera_action = overlay.add_menu_action("Camera: you, hands, presence (local)", watcher.set_camera,
                                            checkable=True)
    overlay.add_menu_action(f"Hide / show camera picture ({hotkeys.CAMERA_HOTKEY.title()})",
                            overlay.toggle_camera_view)
    overlay.add_menu_action("Browser gestures: point = switch tab, pinch + drag = scroll",
                            lambda on: setattr(browser, "enabled", on), checkable=True, checked=True)
    overlay.add_menu_action("Remember my patterns", lambda on: setattr(memory, "enabled", on),
                            checkable=True, checked=True)

    def clear_memory():
        memory.clear()
        overlay.say("Pette's memory cleared (SOMA's indexed folders stay; remove them from the tray).",
                    title="Pette's memory", seconds=6)
        worldwin.refresh()

    overlay.add_menu_action("Clear Pette memory", clear_memory)

    def on_camera_status(text: str):
        if "unavailable" in text or "disconnected" in text:
            overlay.say(text, title="Camera", seconds=8)
            if camera_action:
                camera_action.setChecked(False)
            else:
                watcher.set_camera(False)

    watcher.camera_status.connect(on_camera_status)

    def on_pause(paused: bool):
        if paused:  # release the camera immediately; resuming never turns it back on by itself
            watcher.set_camera(False)
            if camera_action:
                camera_action.setChecked(False)

    overlay.pause_toggled.connect(on_pause)

    # Global hotkeys fire on the keyboard library's thread: they only emit signals (queued to the main thread).
    combos = {hotkeys.LOOK_HOTKEY: suggester.look_requested.emit,
              hotkeys.CAMERA_HOTKEY: overlay.camera_view_requested.emit,
              hotkeys.WORLD_HOTKEY: overlay.world_requested.emit,
              hotkeys.SOMA_HOTKEY: suggester.toggle_requested.emit,
              hotkeys.BUBBLE_HOTKEY: overlay.bubble_toggle_requested.emit}
    registered = [hotkeys.register(emit, combo) for combo, emit in combos.items()]   # register all, no short-circuit
    hotkey_ok = all(registered)
    app.aboutToQuit.connect(hotkeys.unregister_all)
    app.aboutToQuit.connect(watcher.stop)
    app.aboutToQuit.connect(suggester.stop)
    app.aboutToQuit.connect(overlay.bar.close)

    overlay.place_default()
    overlay.show()
    gate.start()
    suggester.start()
    if not (args.no_camera or os.environ.get("QT_QPA_PLATFORM") == "offscreen"):
        if camera_action:
            camera_action.setChecked(True)
        else:
            watcher.set_camera(True)
    if not args.no_world:
        worldwin.show_near()
    look = f"Click me{f' (or {hotkeys.LOOK_HOTKEY.title()})' if hotkey_ok else ''} and SOMA looks at your screen now."
    overlay.say(f"I'll suggest next steps as you work. {look} Add folders from the tray so SOMA knows your files.",
                title="Ready · SOMA", seconds=8)
    return SimpleNamespace(app=app, overlay=overlay, memory=memory, world=world, suggester=suggester, browser=browser,
                           watcher=watcher, gate=gate, camera_action=camera_action, worldwin=worldwin)


def main(argv=None) -> int:
    return build(parse_args(argv)).app.exec()


if __name__ == "__main__":
    sys.exit(main())
