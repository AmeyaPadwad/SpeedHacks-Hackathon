import pytest

from db import Store, WorldModel
from db.world_model import DEFAULT_BLINK_ON


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def mem(tmp_path):
    clock = Clock()
    store = Store(tmp_path / "cache.sqlite3", clock=clock)
    yield store, WorldModel(store, clock=clock), clock
    store.close()


def test_schema_has_no_content_columns(mem):
    store, _, _ = mem
    cols = {r[1] for t in ("events", "offers", "blink_baseline")
            for r in store.query(f"PRAGMA table_info({t})")}
    assert not cols & {"message", "text", "body", "subject", "image", "frame", "screenshot", "code",
                       "path", "file", "summary"}


def test_blink_threshold_calibrates_after_enough_samples(mem):
    store, world, _ = mem
    store.log_blink_baseline(0.3, 0.55, 100)
    assert world.blink_threshold() == DEFAULT_BLINK_ON             # not enough evidence yet
    store.log_blink_baseline(0.3, 0.55, 250)
    assert world.blink_threshold() == 0.7                          # 0.55 + 0.15
    store.clear()
    store.log_blink_baseline(0.1, 0.2, 400)
    assert world.blink_threshold() == 0.5                          # floor
    store.clear()
    store.log_blink_baseline(0.6, 0.8, 400)
    assert world.blink_threshold() == 0.85                         # ceiling


def test_presence_and_summary(mem):
    store, world, _ = mem
    store.log_event("face_away")
    store.log_event("away_ended", 12.0)
    store.log_event("away_ended", 30.0)
    store.log_event("eyes_closed")
    store.log_event("pinch")
    p = world.presence_today()
    assert p == {"away_episodes": 2, "median_away_s": 21.0, "eyes_closed": 1, "pinches": 1,
                 "tab_switches": 0, "scrolls": 0}
    for kind in ("point_left", "point_right", "point_right", "drag_start"):
        store.log_event(kind)
    assert world.presence_today()["tab_switches"] == 3 and world.presence_today()["scrolls"] == 1
    store.log_offer("soma:review_changes", "accepted")
    s = world.summary()
    assert "1 done" in s and "away 2x" in s and "(default)" in s


def test_disable_prune_and_clear(mem):
    store, _, clock = mem
    store.enabled = False
    store.log_event("pinch")
    assert store.counts()["events"] == 0                           # opted out: nothing written
    store.enabled = True
    store.log_event("pinch")
    clock.t += 31 * 86400
    store.log_event("pinch")
    store.prune()
    assert store.counts()["events"] == 1                           # 30-day retention
    store.clear()
    assert sum(store.counts().values()) == 0


def test_thread_safety(mem):
    import threading
    store, _, _ = mem
    threads = [threading.Thread(target=lambda: [store.log_event("pinch") for _ in range(50)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.counts()["events"] == 200


def test_soma_suggestion_outcomes_show_as_a_rule(mem):
    store, world, _ = mem
    assert world.soma_counts() == {"shown": 0, "accepted": 0, "dismissed": 0, "ignored": 0}
    assert world.snapshot()["rules"][0]["why"] == "no suggestions yet"
    for outcome in ("shown", "accepted", "shown", "dismissed", "shown", "ignored"):
        store.log_offer("soma:review_changes", outcome)
    store.log_offer("proactive", "accepted")                      # other offers don't count
    snap = world.snapshot()
    assert snap["soma"] == {"shown": 3, "accepted": 1, "dismissed": 1, "ignored": 1}
    assert snap["accept_rate"] == 0.5
    assert snap["rules"][0]["why"] == "3 shown: 1 done, 1 not now, 1 ignored"
    assert snap["kinds"] == [{"kind": "soma:review_changes", "label": "Review changes", "shown": 3,
                              "accepted": 1, "dismissed": 1, "ignored": 1}]
    assert snap["recent"][0]["kind"] == "Review changes"


def test_kinds_you_keep_dismissing_are_muted(mem):
    store, world, _ = mem
    for _ in range(3):
        store.log_offer("soma:write_tests", "dismissed")
    assert world.muted_kinds() == set()                            # not enough evidence yet
    store.log_offer("soma:write_tests", "ignored")                 # ignoring is never a "Not now"
    assert world.muted_kinds() == set()
    store.log_offer("soma:write_tests", "dismissed")
    assert world.muted_kinds() == {"soma:write_tests"}
    store.log_offer("soma:write_tests", "accepted")
    store.log_offer("soma:write_tests", "accepted")
    assert world.muted_kinds() == set()                            # 4 of 6 is below 80 %
    for _ in range(4):
        store.log_offer("soma:take_notes", "dismissed")
    snap = world.snapshot()
    assert snap["muted"] == ["Save notes"]
    assert any(r["then"] == "Pette doesn't show it any more" for r in snap["rules"])
    store.clear()
    assert world.muted_kinds() == set()


def test_old_messages_table_is_dropped(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE messages (ts REAL, recipient TEXT, platform TEXT, outcome TEXT)")
    db.execute("INSERT INTO messages VALUES (1, 'sarah', 'slack', 'sent')")
    db.commit()
    db.close()
    store = Store(path)
    assert store.query("SELECT name FROM sqlite_master WHERE name = 'messages'") == []
    store.close()
