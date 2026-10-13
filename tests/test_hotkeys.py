"""Global hotkeys: hide/show the camera orb and the world-model panel without touching widgets off-thread."""
import threading

import pytest

from pet import hotkeys
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
    overlay.set_status("looking")
    assert overlay.orb.camera_on and overlay.orb.label() == "Looking"   # only the view is hidden
    overlay.set_camera_visible(False)                                      # idempotent
    assert not overlay.camera_visible


def test_hotkey_emitted_from_another_thread_is_queued_to_the_main_thread(qapp, overlay):
    t = threading.Thread(target=overlay.camera_view_requested.emit)
    t.start()
    t.join()
    assert overlay.camera_visible          # nothing touched yet: the slot is queued
    qapp.processEvents()
    assert not overlay.camera_visible


def test_build_registers_all_hotkeys(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("FLOWSTATE_DB", str(tmp_path / "c.sqlite3"))
    monkeypatch.setenv("PETTE_SOMA_ENDPOINT", str(tmp_path / "none.json"))
    registered = {}
    monkeypatch.setattr(hotkeys, "register", lambda cb, combo: registered.setdefault(combo, cb) is cb)
    import main
    app = main.build(main.parse_args(["--no-camera", "--no-world"]))
    try:
        assert set(registered) == {hotkeys.LOOK_HOTKEY, hotkeys.CAMERA_HOTKEY, hotkeys.WORLD_HOTKEY,
                                   hotkeys.SOMA_HOTKEY}
        registered[hotkeys.CAMERA_HOTKEY]()
        qapp.processEvents()
        assert not app.overlay.camera_visible
        registered[hotkeys.WORLD_HOTKEY]()
        qapp.processEvents()
        assert app.worldwin.isVisible()
        registered[hotkeys.WORLD_HOTKEY]()
        qapp.processEvents()
        assert not app.worldwin.isVisible()
        registered[hotkeys.SOMA_HOTKEY]()
        qapp.processEvents()
        assert app.suggester.enabled is False
    finally:
        app.suggester.stop()
        app.worldwin.close()
        app.overlay.close()
        app.memory.close()
