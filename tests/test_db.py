import pytest

from db import Store, WorldModel
from db.world_model import DEFAULT_BLINK_ON, DEFAULT_COOLDOWN_S


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
    cols = {r[1] for t in ("events", "messages", "offers", "blink_baseline")
            for r in store.query(f"PRAGMA table_info({t})")}
    assert not cols & {"message", "text", "body", "subject", "image", "frame", "screenshot", "code"}


def test_preferred_platform_weights_sends(mem):
    store, world, _ = mem
    assert world.preferred_platform("Sarah") is None
    store.log_message("Sarah", "discord", "drafted")
    store.log_message("Sarah", "slack", "drafted")
    store.log_message("Sarah", "slack", "sent")
    store.log_message("Sarah", "discord", "cancelled")
    assert world.preferred_platform("sarah ") == "slack"          # case/space-insensitive
    assert world.known_contacts() == [("sarah", "slack")]
    assert world.send_rate() == 0.5


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


@pytest.mark.parametrize("accepted,declined,factor", [(0, 6, 3), (2, 4, 1.5), (3, 3, 1), (5, 1, 0.5)])
def test_offer_cooldown_adapts(mem, accepted, declined, factor):
    store, world, _ = mem
    for _ in range(accepted):
        store.log_offer("proactive", "accepted")
    for _ in range(declined):
        store.log_offer("proactive", "declined")
    store.log_offer("proactive", "shown")                          # 'shown' never counts
    assert world.offer_cooldown_s() == max(300, DEFAULT_COOLDOWN_S * factor)


def test_cooldown_default_with_little_data(mem):
    store, world, _ = mem
    store.log_offer("proactive", "declined")
    assert world.offer_cooldown_s() == DEFAULT_COOLDOWN_S


def test_presence_and_summary(mem):
    store, world, _ = mem
    store.log_event("face_away")
    store.log_event("away_ended", 12.0)
    store.log_event("away_ended", 30.0)
    store.log_event("eyes_closed")
    store.log_event("pinch")
    p = world.presence_today()
    assert p == {"away_episodes": 2, "median_away_s": 21.0, "eyes_closed": 1, "pinches": 1}
    store.log_message("Ameya", "discord", "sent")
    s = world.summary()
    assert "Ameya (discord)" in s and "away 2x" in s and "(default)" in s


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
    assert not any("SOMA" in r["when"] for r in world.snapshot()["rules"])
    for outcome in ("shown", "accepted", "shown", "dismissed", "shown", "ignored"):
        store.log_offer("soma:share_changes", outcome)
    store.log_offer("proactive", "accepted")                      # other offers don't count
    snap = world.snapshot()
    assert snap["soma"] == {"shown": 3, "accepted": 1, "dismissed": 1, "ignored": 1}
    rule = next(r for r in snap["rules"] if "SOMA" in r["when"])
    assert rule["why"] == "3 shown: 1 done, 1 not now, 1 ignored"
