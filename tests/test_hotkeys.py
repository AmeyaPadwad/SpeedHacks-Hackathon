"""Global hotkeys: hide/show the camera orb and the world-model panel without touching widgets off-thread."""
import threading

import pytest

from pet import voice
from pet.ui import Overlay


@pytest.fixture
def overlay(qapp):
    o = Overlay(tray=False)
    o.place_default()
    o.show()
    yield o
    o.close()


def test_camera_view_toggles_and_pette_stays_put(qapp, overlay):
    pet_bottom = overlay.geometry().bottom()
    overlay.camera_view_requested.emit()
    assert not overlay.camera_visible and overlay.orb.isHidden()
    assert overlay.geometry().bottom() == pet_bottom
    overlay.camera_view_requested.emit()
    assert overlay.camera_visible
    assert overlay.geometry().bottom() == pet_bottom


def test_hidden_camera_view_keeps_camera_state_and_status(qapp, overlay):
    overlay.set_camera_indicator(True)
    overlay.set_camera_visible(False)
    overlay.set_status("listening")
    assert overlay.orb.camera_on and overlay.orb.label() == "Listening"   # only the view is hidden
    overlay.set_camera_visible(False)                                      # idempotent
    assert not overlay.camera_visible


def test_hotkey_emitted_from_another_thread_is_queued_to_the_main_thread(qapp, overlay):
    t = threading.Thread(target=overlay.camera_view_requested.emit)
    t.start()
    t.join()
    assert overlay.camera_visible          # nothing touched yet: the slot is queued
    qapp.processEvents()
    assert not overlay.camera_visible


def test_build_registers_camera_and_world_hotkeys(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("FLOWSTATE_DB", str(tmp_path / "c.sqlite3"))
    registered = {}
    monkeypatch.setattr(voice, "register_hotkey",
                        lambda cb, combo=voice.HOTKEY: registered.setdefault(combo, cb) is cb)
    import main
    app = main.build(main.parse_args(["--demo", "--no-camera", "--no-world", "--always-show"]))
    try:
        assert {voice.HOTKEY, voice.CAMERA_HOTKEY, voice.WORLD_HOTKEY} <= set(registered)
        registered[voice.CAMERA_HOTKEY]()
        qapp.processEvents()
        assert not app.overlay.camera_visible
        registered[voice.WORLD_HOTKEY]()
        qapp.processEvents()
        assert app.worldwin.isVisible()
        registered[voice.WORLD_HOTKEY]()
        qapp.processEvents()
        assert not app.worldwin.isVisible()
    finally:
        app.worldwin.close()
        app.overlay.close()
        app.memory.close()
