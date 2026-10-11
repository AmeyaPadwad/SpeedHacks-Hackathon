"""FlowState entry point: tray + floating Pette overlay + global hotkey pipeline."""
import argparse
import os
import sys

from PySide6.QtWidgets import QApplication

from pet import voice
from pet.pipeline import Pipeline
from pet.ui import Overlay


def main() -> int:
    ap = argparse.ArgumentParser(description="FlowState voice agent")
    ap.add_argument("--no-voice", action="store_true", help="type commands instead of speaking them")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = Overlay()
    pipeline = Pipeline(overlay, use_voice=not args.no_voice)

    hotkey_ok = voice.register_hotkey(pipeline.hotkey_pressed.emit)
    app.aboutToQuit.connect(voice.unregister_hotkeys)

    overlay.place_default()
    overlay.show()
    if not os.environ.get("OPENAI_API_KEY"):
        overlay.say("OPENAI_API_KEY isn't set, so I can't draft anything yet.", title="Setup needed", seconds=10)
    elif hotkey_ok:
        overlay.say(f"Press {voice.HOTKEY.title()} or click me, then speak.", seconds=6)
    else:
        overlay.say("Global hotkey unavailable. Click me to ask.", seconds=6)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
