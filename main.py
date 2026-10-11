"""FlowState entry point: tray + floating Pette overlay + global hotkey pipeline."""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load before importing pet modules: they read settings at import time. Real env vars win over .env.
load_dotenv(Path(__file__).resolve().parent / ".env")

from PySide6.QtWidgets import QApplication  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser(description="FlowState voice agent")
    ap.add_argument("--no-voice", action="store_true", help="type commands instead of speaking them")
    ap.add_argument("--demo", action="store_true",
                    help="offline demo: typed commands, keyword drafter instead of GPT-4o (labeled DEMO)")
    ap.add_argument("--input-gathering", action="store_true", help="macOS coding input collector; emits reviewed context for Darwin")
    ap.add_argument("--project", type=Path, default=Path.cwd(), help="project directory for input gathering (default: launch directory)")
    args = ap.parse_args()
    if args.input_gathering:
        if args.demo or args.no_voice:
            ap.error("--input-gathering has its own Type task button; use it without --demo/--no-voice")
        if not args.project.expanduser().is_dir():
            ap.error("--project must point to an existing directory")

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    from pet.runtime import TerminalShutdown
    shutdown = TerminalShutdown(app)
    if args.input_gathering:
        from pet.inputs.ui import run_input_gathering
        return run_input_gathering(app, args.project)

    from pet import voice
    from pet.brain import OfflineBrain
    from pet.pipeline import Pipeline
    from pet.ui import Overlay

    overlay = Overlay()
    brain = OfflineBrain() if args.demo else None
    pipeline = Pipeline(overlay, brain, use_voice=not (args.no_voice or args.demo))

    hotkey_ok = voice.register_hotkey(pipeline.hotkey_pressed.emit)
    app.aboutToQuit.connect(voice.unregister_hotkeys)

    overlay.place_default()
    overlay.show()
    if args.demo:
        overlay.say(f"Demo mode: no AI. Press {voice.HOTKEY.title()} or click me, then type a command.",
                    title=OfflineBrain.label, seconds=8)
    elif not os.environ.get("OPENAI_API_KEY"):
        overlay.say("OPENAI_API_KEY isn't set, so I can't draft anything yet.", title="Setup needed", seconds=10)
    elif hotkey_ok:
        overlay.say(f"Press {voice.HOTKEY.title()} or click me, then speak.", seconds=6)
    else:
        overlay.say("Global hotkey unavailable. Click me to ask.", seconds=6)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
