// Port of tests/test_db.py: the fact log stores facts only; the world model's rules (muting, blink calibration,
// presence) are small and explainable.
import { FactStore, ROW_KEYS, TABLES } from "./store";
import { DEFAULT_BLINK_ON, WorldModel, kindLabel } from "./worldModel";

const DAY = 86_400_000;

function rig() {
  const clock = { t: 1_000_000_000 };
  const now = () => clock.t;
  const store = new FactStore(null, now);
  return { store, world: new WorldModel(store, now), clock };
}

let dbCount = 0;
const freshName = () => `pette-test-${++dbCount}-${Math.random()}`;

describe("FactStore", () => {
  it("stores facts only: no record has a content-like field", async () => {
    const store = await FactStore.open(freshName());
    store.logEvent("face_away");
    store.logEvent("away_ended", 12);
    store.logOffer("soma:review_changes", "shown");
    store.logOffer("soma:review_changes", "rejected");
    store.logBlinkBaseline(0.3, 0.5, 100);
    const stored = await store.stored();
    const banned = ["message", "text", "body", "subject", "image", "frame", "screenshot", "code", "path", "file",
      "summary", "title", "detail", "name"];
    for (const t of TABLES) {
      expect(stored[t].length).toBeGreaterThan(0);
      for (const rec of stored[t]) {
        const keys = Object.keys(rec);
        expect(keys.every((k) => (ROW_KEYS[t] as readonly string[]).includes(k))).toBe(true);
        expect(keys.some((k) => banned.includes(k))).toBe(false);
      }
    }
    expect(stored.offers.map((r) => r.outcome)).toEqual(["shown", "dismissed"]); // Not now is stored as "dismissed"
    store.close();
  });

  it("persists across reopen, and prunes rows older than 30 days", async () => {
    const name = freshName();
    const clock = { t: 5_000 * DAY };
    const a = await FactStore.open(name, () => clock.t);
    a.logEvent("pinch");
    clock.t += 31 * DAY;
    a.logEvent("pinch");
    await a.flush();
    a.close();
    const b = await FactStore.open(name, () => clock.t);
    expect(b.counts().events).toBe(1);
    expect((await b.stored()).events).toHaveLength(1);
    await b.clear();
    expect(Object.values(b.counts()).reduce((x, y) => x + y)).toBe(0);
    b.close();
  });

  it("writes nothing while disabled, and notifies subscribers", () => {
    const { store } = rig();
    let changes = 0;
    store.subscribe(() => changes++);
    store.enabled = false;
    store.logEvent("pinch");
    expect(store.counts().events).toBe(0);
    store.enabled = true;
    store.logEvent("pinch");
    expect(store.counts().events).toBe(1);
    expect(changes).toBe(1);
  });
});

describe("WorldModel", () => {
  it("calibrates the blink threshold after enough samples, within 0.5..0.85", async () => {
    const { store, world } = rig();
    store.logBlinkBaseline(0.3, 0.55, 100);
    expect(world.blinkThreshold()).toBe(DEFAULT_BLINK_ON);
    store.logBlinkBaseline(0.3, 0.55, 250);
    expect(world.blinkThreshold()).toBe(0.7);
    await store.clear();
    store.logBlinkBaseline(0.1, 0.2, 400);
    expect(world.blinkThreshold()).toBe(0.5);
    await store.clear();
    store.logBlinkBaseline(0.6, 0.8, 400);
    expect(world.blinkThreshold()).toBe(0.85);
  });

  it("summarises presence over the last 24 hours", () => {
    const { store, world, clock } = rig();
    store.logEvent("face_away");
    store.logEvent("away_ended", 12);
    store.logEvent("away_ended", 30);
    store.logEvent("eyes_closed");
    store.logEvent("pinch");
    expect(world.presenceToday()).toEqual({ awayEpisodes: 2, medianAwayS: 21, eyesClosed: 1, pinches: 1, steps: 0, scrolls: 0 });
    for (const k of ["point_left", "point_right", "point_right", "drag_start"]) store.logEvent(k);
    expect(world.presenceToday().steps).toBe(3);
    expect(world.presenceToday().scrolls).toBe(1);
    store.logOffer("soma:review_changes", "accepted");
    const s = world.summary();
    expect(s).toContain("1 done");
    expect(s).toContain("away 2x");
    expect(s).toContain("(default)");
    clock.t += 2 * DAY;
    expect(world.presenceToday().pinches).toBe(0);
  });

  it("shows suggestion outcomes as a rule", () => {
    const { store, world } = rig();
    expect(world.somaCounts()).toEqual({ shown: 0, accepted: 0, dismissed: 0, ignored: 0 });
    expect(world.snapshot().rules[0].why).toBe("no suggestions yet");
    for (const o of ["shown", "accepted", "shown", "rejected", "shown", "ignored"] as const) store.logOffer("soma:review_changes", o);
    store.logOffer("proactive", "accepted"); // other offers don't count
    const snap = world.snapshot();
    expect(snap.soma).toEqual({ shown: 3, accepted: 1, dismissed: 1, ignored: 1 });
    expect(snap.acceptRate).toBe(0.5);
    expect(snap.rules[0].why).toBe("3 shown: 1 done, 1 not now, 1 ignored");
    expect(snap.kinds).toEqual([{ kind: "soma:review_changes", label: "Review changes", shown: 3, accepted: 1, dismissed: 1, ignored: 1 }]);
    expect(snap.recent[0].kind).toBe("Review changes");
  });

  it("mutes kinds you keep putting off (Not now >= 80 % over >= 4 answers)", async () => {
    const { store, world } = rig();
    for (let i = 0; i < 3; i++) store.logOffer("soma:write_tests", "rejected");
    expect(world.mutedKinds().size).toBe(0);
    store.logOffer("soma:write_tests", "ignored"); // ignoring is never a "Not now"
    expect(world.mutedKinds().size).toBe(0);
    store.logOffer("soma:write_tests", "rejected");
    expect([...world.mutedKinds()]).toEqual(["soma:write_tests"]);
    store.logOffer("soma:write_tests", "accepted");
    store.logOffer("soma:write_tests", "accepted");
    expect(world.mutedKinds().size).toBe(0); // 4 of 6 is below 80 %
    for (let i = 0; i < 4; i++) store.logOffer("soma:take_notes", "dismissed");
    const snap = world.snapshot();
    expect(snap.muted).toEqual(["Save notes"]);
    expect(snap.rules.some((r) => r.then === "Pette doesn't show it any more")).toBe(true);
    await store.clear();
    expect(world.mutedKinds().size).toBe(0);
  });

  it("labels kinds", () => {
    expect(kindLabel("soma:open_relevant_file")).toBe("Open file");
    expect(kindLabel("soma:something_new")).toBe("Something new");
  });
});
