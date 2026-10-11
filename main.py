"""FlowState entry point: tray + floating Pette overlay + global hotkeys + CV watcher."""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load before importing pet modules: they read settings at import time. Real env vars win over .env.
load_dotenv(Path(__file__).resolve().parent / ".env")

from PySide6.QtGui import QActionGroup  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from db import Store, WorldModel  # noqa: E402
from pet import agents, voice  # noqa: E402
from pet.analyst import LocalAnalyst  # noqa: E402
from pet.brain import OfflineBrain  # noqa: E402
from pet.focus import FocusGate  # noqa: E402
from pet.pipeline import Pipeline  # noqa: E402
from pet.ui import Overlay, ResultPanel  # noqa: E402
from pet.watcher import Watcher  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="FlowState voice agent")
    ap.add_argument("--no-voice", action="store_true", help="type commands instead of speaking them")
    ap.add_argument("--demo", action="store_true",
                    help="offline demo: typed commands, keyword drafter and local-only analysis (labeled DEMO)")
    ap.add_argument("--camera", action="store_true", help="turn on the webcam (local face/eyes/pinch) at launch")
    ap.add_argument("--always-show", action="store_true",
                    help="show Pette over every app, not only VS Code and terminals")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = Overlay()
    results = ResultPanel()
    memory = Store()                    # db/pette_cache.sqlite3: facts only, local, 30-day retention
    world = WorldModel(memory)
    pipeline = Pipeline(overlay, OfflineBrain() if args.demo else None,
                        analyst=LocalAnalyst() if args.demo else None,
                        show_result=results.show_result, use_voice=not (args.no_voice or args.demo),
                        memory=memory, world=world)
    watcher = Watcher(overlay, lambda: pipeline.busy, memory=memory, world=world)
    app.aboutToQuit.connect(memory.close)
    watcher.offer.connect(pipeline.offer_investigation)
    gate = FocusGate(overlay)
    overlay.add_menu_action("Show only in VS Code / Terminal", gate.set_enabled, checkable=True,
                            checked=not args.always_show)
    gate.enabled = gate.enabled and not args.always_show

    # --- tray: CV controls (camera off by default; screen watching off by default) ---
    camera_action = overlay.add_menu_action("Camera: face, eyes, pinch (local)", watcher.set_camera, checkable=True)
    overlay.add_menu_action("Proactive help on IDE errors", lambda on: setattr(watcher, "proactive", on),
                            checkable=True, checked=True)
    overlay.add_menu_action("Watch screen for errors (local CV)", lambda on: setattr(watcher, "watch_screen", on),
                            checkable=True)
    group = QActionGroup(overlay)
    current = pipeline.agent.name
    for a in agents.AGENTS.values():
        if a.available():
            act = overlay.add_menu_action(f"Coding agent: {a.label}", checkable=True, checked=a.name == current)
            if act:
                group.addAction(act)
                act.triggered.connect(lambda _=False, n=a.name: setattr(pipeline, "agent_name", n))

    # --- tray: memory / world model ---
    overlay.add_menu_action("Remember my patterns", lambda on: setattr(memory, "enabled", on),
                            checkable=True, checked=True)
    overlay.add_menu_action("What Pette learned", lambda: overlay.say(world.summary(), title="Pette's memory",
                                                                      seconds=15))

    def clear_memory():
        memory.clear()
        overlay.say("Memory cleared. Back to defaults.", title="Pette's memory", seconds=5)

    overlay.add_menu_action("Clear Pette memory", clear_memory)

    def on_camera_status(text: str):
        if "unavailable" in text or "disconnected" in text:
            overlay.say(text, title="Camera", seconds=8)
            if camera_action:
                camera_action.setChecked(False)
        elif text.startswith("Camera on"):
            overlay.notify("Pette camera", "Camera on. Frames are processed locally and never stored or uploaded.")

    watcher.camera_status.connect(on_camera_status)

    def on_pause(paused: bool):
        if paused:  # release the camera immediately; resuming never turns it back on by itself
            watcher.set_camera(False)
            if camera_action:
                camera_action.setChecked(False)

    overlay.pause_toggled.connect(on_pause)
    watcher.start()

    hotkey_ok = voice.register_hotkey(pipeline.hotkey_pressed.emit)
    voice.register_hotkey(pipeline.investigate_hotkey.emit, voice.INVESTIGATE_HOTKEY)
    app.aboutToQuit.connect(voice.unregister_hotkeys)
    app.aboutToQuit.connect(watcher.stop)

    overlay.place_default()
    overlay.show()
    gate.start()   # hides Pette right away if VS Code / a terminal isn't in front
    if args.camera and camera_action:
        camera_action.setChecked(True)
    keys = f"{voice.HOTKEY.title()} to ask, {voice.INVESTIGATE_HOTKEY.title()} to investigate your screen."
    if args.demo:
        overlay.say(f"Demo mode: no AI. {keys}", title=OfflineBrain.label, seconds=8)
    elif not os.environ.get("OPENAI_API_KEY"):
        overlay.say("OPENAI_API_KEY isn't set, so voice and drafting are off. Investigate still works locally.",
                    title="Setup needed", seconds=10)
    elif hotkey_ok:
        overlay.say(f"Press {keys}", seconds=7)
    else:
        overlay.say("Global hotkeys unavailable. Click me to ask, or use the tray to investigate.", seconds=6)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
