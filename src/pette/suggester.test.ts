// Port of tests/test_soma.py: cards only when free, nothing happens without "Do it", feedback and facts flow back.
import type { SuggestionPayload } from "@/soma/service";
import * as actions from "./actions";
import { PetteController } from "./controller";
import { FakeSoma } from "./fakeSoma";
import { LOOK_TIMEOUT_S, REPORT_AFTER, Suggester, kindOf, opener } from "./suggester";
import { FactStore } from "./world/store";
import { WorldModel } from "./world/worldModel";

const settle = async () => {
  for (let i = 0; i < 20; i++) await Promise.resolve();
};

function sug(id = "s1", over: Partial<SuggestionPayload> = {}): Partial<SuggestionPayload> {
  return { available: true, silent: false, valid: true, id, action: "SUGGEST_NEXT_STEP", template: "review_changes",
    text: 'Review your changes to "app.py" before you commit', evidence: ['editing "app.py"'],
    file: null, line: null, memorySource: null, ...over };
}

function rig() {
  const ui = new PetteController();
  const soma = new FakeSoma();
  const store = new FactStore();
  const world = new WorldModel(store);
  const opened: [string, number | null][] = [];
  const copied: string[] = [];
  const states: string[] = [];
  const r = {
    ui, soma, store, world, opened, copied, states,
    openFile: async (ref: string, line: number | null) => {
      opened.push([ref, line]);
      return `Opened ${ref}`;
    },
    s: null as unknown as Suggester,
    async deliver(...items: Partial<SuggestionPayload>[]) {
      for (const it of items) {
        soma.queue(it);
        await r.s.pollOnce();
      }
      await settle();
    },
    offers: () => store.offers().map((o) => [o.kind, o.outcome]),
    calls: () => soma.requests.filter((q) => q.call !== "readFile").map((q) => [q.call, ...q.args]),
    bubble: () => ui.getState().bubble,
    text: () => {
      const b = ui.getState().bubble;
      return b && b.kind !== "card" ? b.text : b?.proposal.text ?? "";
    },
  };
  r.s = new Suggester(ui, soma, {
    memory: store, world, showS: 2,
    openFile: (ref, line) => r.openFile(ref, line),
    copyText: async (t) => {
      copied.push(t);
      return "Copied. Paste it to your coding agent or notes.";
    },
  });
  r.s.events.on("status", (st) => states.push(st));
  return r;
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

it("shows a card, then Copy copies the text and reports accepted", async () => {
  const r = rig();
  await r.deliver(sug());
  const b = r.bubble();
  expect(b?.kind).toBe("card");
  if (b?.kind !== "card") return;
  expect(b.title).toBe("SOMA suggests · Next step");
  expect(b.proposal.actionLabel).toBe("Copy");
  expect(b.proposal.cancelLabel).toBe("Not now");
  expect(r.ui.getState().progress).toBe(1);
  expect(r.copied).toEqual([]);
  expect(r.ui.getState().status).toBe("suggesting");
  r.ui.approve();
  await settle();
  expect(r.copied).toEqual(['Review your changes to "app.py" before you commit']);
  expect(r.calls()).toContainEqual(["feedback", "s1", "accepted"]);
  expect(r.ui.getState().status).toBe("done");
  expect(r.ui.getState().progress).toBeNull(); // "Done" message has no time-left bar
  expect(r.offers()).toEqual([["soma:review_changes", "shown"], ["soma:review_changes", "accepted"]]);
});

it("opens the on-screen file at its line, and the file behind a memory (pinch = Do it)", async () => {
  const r = rig();
  await r.deliver(sug("s1", { template: "", action: "OPEN_RELEVANT_FILE", text: "Open calc.py at line 40", file: "f/calc.py", line: 40 }));
  const b = r.bubble();
  expect(b?.kind === "card" && b.proposal.actionLabel).toBe("Open");
  expect(b?.kind === "card" && b.title.endsWith("Open file")).toBe(true);
  r.ui.approve();
  await settle();
  expect(r.opened).toEqual([["f/calc.py", 40]]);
  await r.deliver(sug("s2", { template: "", action: "COMPARE_WITH_PRIOR_WORK", text: 'Compare with "retries.md"', memorySource: "f/retries.md" }));
  r.ui.pinch();
  await settle();
  expect(r.opened.at(-1)).toEqual(["f/retries.md", null]);
  expect(r.copied).toEqual([]);
  r.ui.pinch(); // single-use: a second pinch does nothing
  await settle();
  expect(r.opened).toHaveLength(2);
  expect(r.store.offers().filter((o) => o.outcome === "accepted").map((o) => o.kind))
    .toEqual(["soma:open_relevant_file", "soma:compare_with_prior_work"]);
});

it("reports a missing file instead of throwing", async () => {
  const r = rig();
  r.openFile = async () => {
    throw new actions.ActionError("calc.py isn't there any more.");
  };
  await r.deliver(sug("s1", { file: "f/calc.py" }));
  r.ui.approve();
  await settle();
  expect(r.text()).toContain("isn't there");
  expect(r.states.at(-1)).toBe("failed");
});

it("Not now rejects; an expired card is ignored (never a dismissal)", async () => {
  const r = rig();
  await r.deliver(sug("s1"));
  r.ui.cancel();
  await settle();
  expect(r.ui.proposal).toBeNull();
  expect(r.calls().at(-1)).toEqual(["feedback", "s1", "rejected"]);
  expect(r.states.at(-1)).toBe("dismissed");
  await r.deliver(sug("s2"));
  vi.advanceTimersByTime(1000);
  expect(r.ui.getState().progress).toBeCloseTo(0.5);
  vi.advanceTimersByTime(1000); // 2 s at 250 ms ticks
  await settle();
  expect(r.ui.proposal).toBeNull();
  expect(r.bubble()).toBeNull();
  expect(r.calls().at(-1)).toEqual(["feedback", "s2", "ignored"]);
  expect(r.store.offers().map((o) => o.outcome)).toEqual(["shown", "dismissed", "shown", "ignored"]);
});

it("shows a suggestion once, and not while busy, hidden or paused", async () => {
  const r = rig();
  await r.deliver(sug("s1"));
  r.ui.cancel();
  await r.deliver(sug("s1"));
  expect(r.ui.proposal).toBeNull(); // already handled
  await r.deliver(sug("s2", { silent: true }), sug("s3", { valid: false }), sug("s4", { text: "" }));
  expect(r.ui.proposal).toBeNull();
  r.ui.setContextAllowed(false);
  await r.deliver(sug("s5"));
  expect(r.ui.proposal).toBeNull();
  r.ui.setContextAllowed(true);
  r.ui.setPaused(true);
  await r.deliver(sug("s6"));
  expect(r.ui.proposal).toBeNull();
  expect(r.calls()).toContainEqual(["pause"]);
  r.ui.setPaused(false);
  await r.deliver(sug("s7"));
  expect(r.ui.proposal).not.toBeNull();
  expect(r.calls()).toContainEqual(["resume"]);
});

it("pausing with a card up drops it as ignored", async () => {
  const r = rig();
  await r.deliver(sug("s1"));
  r.ui.setPaused(true);
  await settle();
  expect(r.calls()).toContainEqual(["feedback", "s1", "ignored"]);
  expect(r.ui.getState().mood).toBe("sleeping");
});

it("skips a muted kind unless you asked", async () => {
  const r = rig();
  for (let i = 0; i < 4; i++) r.store.logOffer("soma:write_tests", "rejected");
  await r.deliver(sug("s1", { template: "write_tests", text: "Write a test for app.py" }));
  expect(r.ui.proposal).toBeNull();
  expect(r.calls().at(-1)).toEqual(["feedback", "s1", "ignored"]);
  r.soma.emptyLook = true;
  r.ui.requestLook(); // you asked: show it anyway
  await settle();
  await r.deliver(sug("s2", { template: "write_tests", text: "Write a test for app.py" }));
  expect(r.ui.proposal).not.toBeNull();
});

it("click looks now, then a card, nothing, or a failure", async () => {
  const r = rig();
  r.soma.emptyLook = true;
  r.ui.requestLook();
  await settle();
  expect(r.calls()).toContainEqual(["trigger"]);
  expect(r.states.at(-1)).toBe("looking");
  expect(r.text()).toContain("Looking at your screen");
  await r.deliver(sug("s1"));
  expect(r.ui.proposal).not.toBeNull();
  expect(r.states.at(-1)).toBe("suggesting");
  r.ui.cancel();
  r.ui.requestLook();
  await settle();
  vi.advanceTimersByTime(LOOK_TIMEOUT_S * 1000); // the look timed out
  expect(r.text()).toContain("Nothing worth suggesting");
  expect(r.states.at(-1)).toBe("idle");
  r.soma.triggerResult = false; // SOMA paused on its side / no screen shared
  r.ui.requestLook();
  await settle();
  expect(r.text()).toContain("paused");
  expect(r.states.at(-1)).toBe("failed");
});

it("look-now finds the queued card through the fake's trigger", async () => {
  const r = rig();
  r.ui.requestLook();
  await settle();
  await r.s.pollOnce();
  await settle();
  const b = r.bubble();
  expect(b?.kind === "card" && b.proposal.text).toContain("Review your changes");
});

it("toggling off pauses SOMA and clears the card", async () => {
  const r = rig();
  await r.deliver(sug());
  r.s.toggle();
  await settle();
  expect(r.s.enabled).toBe(false);
  expect(r.calls()).toContainEqual(["pause"]);
  expect(r.calls()).toContainEqual(["feedback", "s1", "ignored"]);
  await r.deliver(sug("s2"));
  expect(r.text()).toContain("Suggestions off");
  r.ui.requestLook();
  await settle();
  expect(r.calls()).not.toContainEqual(["trigger"]); // off means off
  r.s.toggle();
  await settle();
  expect(r.s.enabled).toBe(true);
  expect(r.calls().at(-1)).toEqual(["resume"]);
});

it("adds a folder, reports indexing done, and removes it", async () => {
  const r = rig();
  const seen: unknown[] = [];
  r.s.events.on("foldersChanged", (f) => seen.push(f));
  await r.s.addFolder();
  await settle();
  expect(r.calls()).toContainEqual(["addFolder", null]);
  expect(r.states).toContain("indexing");
  vi.advanceTimersByTime(1500); // the indexing poll sees it finish
  await settle();
  expect(r.text()).toContain("Indexed 2 file(s) from notes");
  expect(seen.at(-1)).toEqual([{ id: "folder_notes", name: "notes", memories: 2 }]);
  expect(r.states.at(-1)).toBe("idle");
  await r.s.removeFolder("folder_notes");
  await settle();
  expect(r.text()).toContain("forgot 2 file(s)");
  expect(seen.at(-1)).toEqual([]);
  r.soma.addFolder = async () => {
    throw new Error("Refusing to index an entire home directory or drive root.");
  };
  await r.s.addFolder();
  expect(r.text()).toContain("home directory");
  expect(r.states.at(-1)).toBe("failed");
});

it("stays quiet while SOMA starts, then reports offline once", async () => {
  const r = rig();
  r.soma.offline = true;
  const notices: string[] = [];
  r.ui.subscribe(() => {
    const n = r.ui.getState().notice;
    if (n && !notices.includes(n.text)) notices.push(n.text);
  });
  for (let i = 0; i < REPORT_AFTER - 1; i++) await r.s.pollOnce();
  expect(notices).toEqual([]);
  await r.s.pollOnce();
  await r.s.pollOnce();
  expect(notices).toHaveLength(1);
  expect(r.ui.getState().detail).toBe("SOMA offline");
});

it("cards wait while the bubble is off", async () => {
  const r = rig();
  r.ui.setBubbleEnabled(false);
  await r.deliver(sug("s1"));
  expect(r.ui.proposal).toBeNull();
  expect(r.store.counts().offers).toBe(0);
  r.ui.setBubbleEnabled(true);
  await r.deliver(sug("s2"));
  expect(r.ui.proposal).not.toBeNull();
});

it("polls every second once started, and stops", async () => {
  const r = rig();
  r.s.start();
  await settle();
  r.soma.queue(sug("s9"));
  await vi.advanceTimersByTimeAsync(1000);
  expect(r.ui.proposal).not.toBeNull();
  r.s.dispose();
  r.ui.cancel();
  expect(r.calls()).not.toContainEqual(["feedback", "s9", "rejected"]); // detached from the controller
});

it("kindOf and opener", () => {
  expect(kindOf(sug() as SuggestionPayload)).toBe("soma:review_changes");
  expect(kindOf(sug("x", { template: "", action: "EXPLAIN_ERROR" }) as SuggestionPayload)).toBe("soma:explain_error");
  expect(opener(sug() as SuggestionPayload)).toBeNull();
  expect(opener(sug("x", { file: "a.py", line: 3 }) as SuggestionPayload)).toEqual(["a.py", 3]);
  expect(opener(sug("x", { file: "a.py", line: "3" as unknown as number }) as SuggestionPayload)).toEqual(["a.py", null]);
  expect(opener(sug("x", { memorySource: "n.md" }) as SuggestionPayload)).toEqual(["n.md", null]);
});

it("a stop / start while a poll is in flight leaves one poll loop", async () => {
  const r = rig();
  let polls = 0;
  const original = r.soma.suggestion.bind(r.soma);
  r.soma.suggestion = () => {
    polls++;
    return original();
  };
  r.s.start();
  r.s.stop();
  r.s.start();
  await settle();
  polls = 0;
  await vi.advanceTimersByTimeAsync(3000);
  expect(polls).toBe(3);
  r.s.stop();
});
it("a card expires on wall-clock time even when a hidden tab delays the ticks", async () => {
  const r = rig();
  await r.deliver(sug("s1"));
  expect(r.ui.proposal).not.toBeNull();
  vi.setSystemTime(Date.now() + 3000); // the clock moves past showS while no tick has run
  vi.advanceTimersByTime(250);
  await settle();
  expect(r.ui.proposal).toBeNull();
  expect(r.calls().at(-1)).toEqual(["feedback", "s1", "ignored"]);
});
