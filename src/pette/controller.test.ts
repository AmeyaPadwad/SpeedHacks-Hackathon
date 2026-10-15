// Port of tests/test_ui.py's Overlay rules: moods, the bubble toggle, single-use cards, history paging, nudge.
import type { Proposal } from "./contracts";
import { HISTORY_S, PetteController } from "./controller";
import { statusLabel } from "./status";

const card = (id: string, text = `text ${id}`): Proposal =>
  ({ id, title: `SOMA suggests · ${id}`, text, actionLabel: "Copy", cancelLabel: "Not now", glow: true });

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

it("click asks to look now, except while paused", () => {
  const c = new PetteController();
  const looks: number[] = [];
  c.events.on("lookRequested", () => looks.push(1));
  c.requestLook();
  c.setPaused(true);
  c.requestLook();
  expect(looks).toHaveLength(1);
});

it("moods follow presence and pause", () => {
  const c = new PetteController();
  expect(c.getState().mood).toBe("idle");
  c.setPresence(false, true);
  expect(c.getState().mood).toBe("blink");
  c.setPresence(true, false);
  expect(c.getState().mood).toBe("away");
  c.setPaused(true);
  expect(c.getState().mood).toBe("sleeping");
  c.setPaused(false);
  c.setPresence(false, false);
  expect(c.getState().mood).toBe("idle");
});

it("status labels", () => {
  expect(statusLabel("idle")).toBe("");
  expect(statusLabel("done")).toBe("Done ✓");
  expect(statusLabel("failed", "SOMA offline")).toBe("SOMA offline");
});

it("messages hide after their time; null keeps them", () => {
  const c = new PetteController();
  c.say("hi", { title: "T", seconds: 2 });
  expect(c.getState().bubble).toEqual({ kind: "message", title: "T", text: "hi" });
  vi.advanceTimersByTime(2000);
  expect(c.getState().bubble).toBeNull();
  c.say("stay", { seconds: null });
  vi.advanceTimersByTime(60_000);
  expect(c.getState().bubble?.kind).toBe("message");
});

it("the bubble toggle hides it and suppresses messages and cards", () => {
  const c = new PetteController();
  const toggles: boolean[] = [];
  c.events.on("bubbleToggled", (on) => toggles.push(on));
  c.say("hello");
  c.toggleBubble();
  expect(c.getState().bubble).toBeNull();
  c.say("nope");
  c.propose(card("a"));
  expect(c.getState().bubble).toBeNull();
  expect(c.proposal).toBeNull();
  expect(c.getState().notice?.text).toContain("Speech bubble off");
  c.toggleBubble();
  c.say("back");
  expect(c.getState().bubble?.kind).toBe("message");
  expect(toggles).toEqual([false, true]);
});

it("a card decision is single-use (buttons and pinch)", () => {
  const c = new PetteController();
  const approved: string[] = [];
  const cancelled: string[] = [];
  const pinched: string[] = [];
  c.events.on("approved", (p) => approved.push(p));
  c.events.on("cancelled", (p) => cancelled.push(p));
  c.events.on("pinchOnProposal", (p) => pinched.push(p));
  c.propose(card("a"));
  c.pinch();
  expect(pinched).toEqual(["a"]);
  c.approve();
  c.approve();
  c.cancel();
  expect(approved).toEqual(["a"]);
  expect(cancelled).toEqual([]);
  const b = c.getState().bubble;
  expect(b?.kind === "card" && b.live).toBe(false); // still shown, buttons disabled
  c.pinch();
  expect(pinched).toEqual(["a"]);
});

it("pausing drops an open card and pinch does nothing while paused", () => {
  const c = new PetteController();
  const pinched: string[] = [];
  c.events.on("pinchOnProposal", (p) => pinched.push(p));
  c.propose(card("a"));
  c.setPaused(true);
  expect(c.proposal).toBeNull();
  expect(c.getState().bubble).toBeNull();
  c.pinch();
  expect(pinched).toEqual([]);
});

it("away -> back re-titles an open card once", () => {
  const c = new PetteController();
  c.nudge(); // nothing up: ignored
  c.propose(card("a"));
  c.nudge();
  c.nudge();
  const b = c.getState().bubble;
  expect(b?.kind === "card" && b.title).toBe("While you were away · SOMA suggests · a");
});

it("point left / right pages through recent suggestions read-only", () => {
  const c = new PetteController();
  const pinched: string[] = [];
  c.events.on("pinchOnProposal", (p) => pinched.push(p));
  c.step(-1); // no history: nothing
  expect(c.getState().bubble).toBeNull();
  for (const id of ["a", "b", "c"]) {
    c.propose(card(id));
    c.consume();
    c.dismiss();
  }
  c.propose(card("d"));
  c.step(-1);
  expect(c.getState().bubble).toEqual({ kind: "history", title: "SOMA suggests · c", text: "text c", index: 3, total: 4 });
  c.pinch(); // ignored while paging
  expect(pinched).toEqual([]);
  c.step(-1);
  c.step(-1);
  c.step(-1); // clamps at the oldest
  expect(c.getState().bubble).toMatchObject({ kind: "history", index: 1 });
  c.step(1);
  c.step(1);
  c.step(1); // back to the live card
  const b = c.getState().bubble;
  expect(b?.kind === "card" && b.live).toBe(true);
  c.step(-1);
  vi.advanceTimersByTime(HISTORY_S * 1000); // a history page times out back to the live view
  expect(c.getState().bubble?.kind).toBe("card");
});

it("history keeps the last five", () => {
  const c = new PetteController();
  for (const id of ["1", "2", "3", "4", "5", "6", "7"]) c.propose(card(id));
  c.dismiss();
  c.step(-1);
  expect(c.getState().bubble).toMatchObject({ kind: "history", index: 5, total: 5, text: "text 7" });
  for (let i = 0; i < 6; i++) c.step(-1);
  expect(c.getState().bubble).toMatchObject({ index: 1, text: "text 3" });
});

it("frames reach the orb's listeners but not React state; gesture changes do", () => {
  const c = new PetteController();
  const frames: unknown[] = [];
  c.onFrame((f) => frames.push(f));
  let renders = 0;
  c.subscribe(() => renders++);
  c.setFrame({ bitmap: null, hands: [], gesture: "" });
  expect(renders).toBe(0);
  c.setFrame({ bitmap: null, hands: [], gesture: "pinch" });
  expect(renders).toBe(1);
  expect(c.getState().gesture).toBe("pinch");
  expect(frames).toHaveLength(3); // initial null + two frames
  const scrolls: number[] = [];
  c.events.on("scroll", (dy) => scrolls.push(dy));
  c.scroll(0.1);
  expect(scrolls).toEqual([0.1]);
});
