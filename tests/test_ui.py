"""Pette without the sprite: the orb is her body (click = look now, drag = move, moods), icon drawn in code."""
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from pet.ui import Overlay, tray_icon


def mouse(kind, pos: QPoint):
    return QMouseEvent(kind, QPointF(pos), QPointF(pos), Qt.LeftButton,
                       Qt.LeftButton if kind != QMouseEvent.Type.MouseButtonRelease else Qt.NoButton,
                       Qt.NoModifier)


def test_click_on_orb_asks_to_look_now_but_a_drag_does_not(qapp):
    o = Overlay(tray=False)
    o.place_default()
    o.show()
    looks = []
    o.look_requested.connect(lambda: looks.append(1))
    orb = o.orb
    start = orb.mapToGlobal(orb.rect().center())
    orb.mousePressEvent(mouse(QMouseEvent.Type.MouseButtonPress, start))
    orb.mouseReleaseEvent(mouse(QMouseEvent.Type.MouseButtonRelease, start))
    assert looks == [1]
    before = o.pos()
    orb.mousePressEvent(mouse(QMouseEvent.Type.MouseButtonPress, start))
    orb.mouseMoveEvent(mouse(QMouseEvent.Type.MouseMove, start - QPoint(40, 30)))
    orb.mouseReleaseEvent(mouse(QMouseEvent.Type.MouseButtonRelease, start - QPoint(40, 30)))
    assert looks == [1] and o.pos() == before - QPoint(40, 30)       # moved, not a click
    o.set_paused(True)
    o.orb.clicked.emit()
    assert looks == [1]                                               # paused: clicks do nothing
    o.close()


def test_moods_follow_presence_and_pause(qapp):
    o = Overlay(tray=False)
    assert o.orb.mood == "idle"
    o.set_presence(away=True, eyes_closed=False)
    assert o.orb.mood == "away"
    o.set_presence(away=False, eyes_closed=True)
    assert o.orb.mood == "blink"
    o.set_paused(True)
    assert o.orb.mood == "sleeping"
    o.set_paused(False)
    o.set_presence(away=False, eyes_closed=False)
    assert o.orb.mood == "idle"
    o.orb.set_mood("dancing")                                         # unknown moods are ignored
    assert o.orb.mood == "idle"


def test_orb_paints_every_mood_and_status(qapp):
    o = Overlay(tray=False)
    for mood in ("idle", "away", "blink", "sleeping"):
        o.orb.set_mood(mood)
        for state in ("idle", "looking", "suggesting", "failed"):
            o.set_status(state, "SOMA offline")
            assert not o.orb.grab().isNull()


def test_tray_icon_is_drawn_without_image_files(qapp):
    icon = tray_icon()
    assert not icon.isNull() and not icon.pixmap(32, 32).isNull()


def test_speech_bubble_docks_bottom_centre_of_the_screen(qapp):
    from PySide6.QtGui import QGuiApplication
    o = Overlay(tray=False)
    o.place_default()
    o.show()
    o.say("Indexed 12 file(s) from notes.", title="SOMA memory", seconds=None)
    bar, a = o.bar, QGuiApplication.primaryScreen().availableGeometry()
    assert bar.isVisible() and o.bubble.isVisible() and o.bubble.tail
    if not bar.geometry().translated(a.center().x() - bar.geometry().center().x(), 0).intersects(o.geometry()):
        assert abs(bar.geometry().center().x() - a.center().x()) <= 1         # centred when there's room
    assert a.bottom() - bar.geometry().bottom() < 30                         # just above the taskbar
    assert not bar.geometry().intersects(o.geometry())                      # not stacked on the orb
    o.dismiss()
    assert not bar.isVisible()
    o.close()


def test_bubble_toggle_hides_it_and_suppresses_messages(qapp):
    from pet.ui import Proposal
    o = Overlay(tray=False)
    o.show()
    seen = []
    o.bubble_toggled.connect(seen.append)
    o.propose(Proposal("SOMA suggests · Next step", "Write a test"))
    o.bubble_toggle_requested.emit()
    assert o.bubble_enabled is False and seen == [False]
    assert o.bubble.proposal is None and not o.bar.isVisible()               # open card dropped
    o.say("hello", seconds=None)
    o.propose(Proposal("x", "y"))
    assert not o.bar.isVisible() and o.bubble.proposal is None
    o.set_bubble_enabled(True)
    o.say("back", seconds=None)
    assert o.bar.isVisible() and seen == [False, True]
    o.close()


def test_bubble_hidden_with_pette_outside_allowed_apps(qapp):
    o = Overlay(tray=False)
    o.show()
    o.say("hi", seconds=None)
    o.set_context_allowed(False)
    assert not o.bar.isVisible()
    o.set_context_allowed(True)
    assert o.bar.isVisible()
    o.close()
