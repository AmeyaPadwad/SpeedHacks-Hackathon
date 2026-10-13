"""Global hotkeys. Callbacks fire on the keyboard library's thread, so they must only emit Qt signals."""
import os

LOOK_HOTKEY = os.environ.get("FLOWSTATE_HOTKEY") or "ctrl+shift+space"   # same as clicking Pette: look now
CAMERA_HOTKEY = os.environ.get("PETTE_CAMERA_HOTKEY") or "ctrl+alt+c"     # hide / show the camera orb
WORLD_HOTKEY = os.environ.get("PETTE_WORLD_HOTKEY") or "ctrl+alt+w"       # hide / show the world-model panel
SOMA_HOTKEY = os.environ.get("PETTE_SOMA_HOTKEY") or "ctrl+alt+s"         # SOMA suggestions on / off


def register(callback, combo: str) -> bool:
    """Call `callback` (on the keyboard library's thread) when `combo` is pressed. False if unavailable."""
    try:
        import keyboard
        keyboard.add_hotkey(combo, callback, suppress=False)
        return True
    except Exception:  # missing package, or OS refused the hook
        return False


def unregister_all():
    try:
        import keyboard
        keyboard.unhook_all_hotkeys()
    except Exception:
        pass
