"""Pette entry point: Pette overlay (bubble + camera orb + sprite), world-model window, tray, hotkey.

One workflow: click Pette (or press the hotkey) -> speak -> your words in the bubble -> Groq drafts ->
a short countdown with Cancel -> the message (with code + a summary of the changes) is dispatched.
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

# Load before importing pet modules: they read settings at import time. Real env vars win over .env.
load_dotenv(Path(__file__).resolve().parent / ".env")

from PySide6.QtWidgets import QApplication  # noqa: E402

from db import Store, WorldModel  # noqa: E402
from pet import eleven, llm, voice  # noqa: E402
from pet.brain import OfflineBrain  # noqa: E402
from pet.focus import FocusGate  # noqa: E402
from pet.pipeline import Pipeline  # noqa: E402
from pet.ui import Overlay  # noqa: E402
from pet.watcher import Watcher  # noqa: E402
from pet.worldview import WorldWindow  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pette voice agent")
    ap.add_argument("--no-voice", action="store_true", help="type commands instead of speaking them")
    ap.add_argument("--demo", action="store_true",
                    help="offline demo: typed commands and a keyword drafter (labeled DEMO); dispatch still runs")
    ap.add_argument("--no-camera", action="store_true", help="don't start the webcam (the orb shows 'camera off')")
    ap.add_argument("--camera", action="store_true", help=argparse.SUPPRESS)   # old flag; the camera is on by default
    ap.add_argument("--no-world", action="store_true", help="don't open the world-model window at launch")
    ap.add_argument("--always-show", action="store_true",
                    help="show Pette over every app, not only VS Code and terminals")
    return ap.parse_args(argv)


def make_listen(settings: "eleven.Settings | None"):
    """The listener for the one workflow: ElevenLabs (Scribe or live) when configured, else Groq Whisper."""
    if settings is None:
        return lambda on_level=None, register_cancel=None: voice.transcribe(voice.record_until_silence())
    if settings.mode == "scribe":
        return lambda on_level=None, register_cancel=None: eleven.scribe(voice.record_until_silence(),
                                                                         settings.api_key)

    def live(on_level=None, register_cancel=None):
        t = eleven.LiveTranscriber(settings, on_level=on_level)
        if register_cancel:
            register_cancel(t.cancel)
        return t.run()
    return live


def build(args: argparse.Namespace) -> SimpleNamespace:
    """Create and wire every component without entering the event loop (tools/sandbox.py reuses this)."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = Overlay()
    memory = Store()                    # db/pette_cache.sqlite3: facts only, local, 30-day retention
    world = WorldModel(memory)
    settings = None if (args.demo or args.no_voice) else eleven.Settings.from_env()
    pipeline = Pipeline(overlay, OfflineBrain() if args.demo else None, listen=make_listen(settings),
                        use_voice=not (args.no_voice or args.demo), memory=memory, world=world)
    watcher = Watcher(overlay, memory=memory, world=world)
    app.aboutToQuit.connect(memory.close)
    worldwin = WorldWindow(overlay, world, pipeline, watcher)
    overlay.world_requested.connect(worldwin.toggle)
    gate = FocusGate(overlay)
    overlay.add_menu_action("Show only in VS Code / Terminal", gate.set_enabled, checkable=True,
                            checked=not args.always_show)
    gate.enabled = gate.enabled and not args.always_show

    # --- tray: camera + memory ---
    camera_action = overlay.add_menu_action("Camera: you, hands, presence (local)", watcher.set_camera,
                                            checkable=True)
    overlay.add_menu_action("Remember my patterns", lambda on: setattr(memory, "enabled", on),
                            checkable=True, checked=True)

    def clear_memory():
        memory.clear()
        overlay.say("Memory cleared. Back to defaults.", title="Pette's memory", seconds=5)
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

    hotkey_ok = voice.register_hotkey(pipeline.hotkey_pressed.emit)
    app.aboutToQuit.connect(voice.unregister_hotkeys)
    app.aboutToQuit.connect(watcher.stop)

    overlay.place_default()
    overlay.show()
    gate.start()   # hides Pette right away if VS Code / a terminal isn't in front
    if not (args.no_camera or os.environ.get("QT_QPA_PLATFORM") == "offscreen"):
        if camera_action:
            camera_action.setChecked(True)
        else:
            watcher.set_camera(True)
    if not args.no_world:
        worldwin.show_near()
    how = f"Click me (or {voice.HOTKEY.title()}) and tell me what to send."
    if args.demo:
        overlay.say(f"Demo mode: type commands, keyword drafter. {how}", title=OfflineBrain.label, seconds=8)
    elif not llm.configured():
        overlay.say(f"{llm.KEY_VAR} isn't set, so I can't draft messages yet.", title="Setup needed", seconds=10)
    else:
        voice_name = "ElevenLabs" if settings else "Whisper"
        overlay.say(how if hotkey_ok else "Click me and tell me what to send.", title=f"Ready · {voice_name}",
                    seconds=7)
    return SimpleNamespace(app=app, overlay=overlay, memory=memory, world=world, pipeline=pipeline,
                           watcher=watcher, gate=gate, camera_action=camera_action, worldwin=worldwin,
                           settings=settings)


def main(argv=None) -> int:
    return build(parse_args(argv)).app.exec()


if __name__ == "__main__":
    sys.exit(main())
