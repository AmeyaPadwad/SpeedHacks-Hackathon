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
