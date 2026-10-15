import { act, render, screen } from "@testing-library/react";
import { FakeSoma } from "../fakeSoma";
import { createPette } from "../usePette";
import { FactStore } from "../world/store";
import { CHANNEL, type DashboardSnapshot, type PetteLive } from "./channel";
import { Dashboard, pushPoint } from "./Dashboard";
import { buildSnapshot } from "./publisher";

const live: PetteLive = {
  status: "idle", label: "", mood: "idle", camera: false, gesture: "", present: true, eyesClosed: false, paused: false,
  suggestions: true, bubble: true, gestures: true, screenShared: true, floating: false, cardUp: false,
};

it("builds a facts-only snapshot from Pette and SOMA", async () => {
  const soma = new FakeSoma();
  await soma.grantScreen();
  const pette = createPette(soma, new FactStore());
  const snap = await buildSnapshot(pette, live, true, 1000);
  expect(snap.at).toBe(1000);
  expect(snap.soma?.capturing).toBe(true);
  expect(snap.diagnostics?.app).toBe("code");
  expect(snap.diagnostics?.candidates.every((c) => Object.keys(c).sort().join() === "action,template")).toBe(true);
  expect(JSON.stringify(snap)).not.toMatch(/calc\.py|Review your changes/); // no file names, no suggestion text
});

it("adds a step-time point only when SOMA processed a new frame", () => {
  const snap = (processed: number, total: number) => ({ soma: { processed }, diagnostics: { timingsMs: { total } } }) as unknown as DashboardSnapshot;
  let s = pushPoint([], snap(1, 2000));
  s = pushPoint(s, snap(1, 2000));
  s = pushPoint(s, snap(2, 1500));
  expect(s).toEqual([{ steps: 1, totalMs: 2000 }, { steps: 2, totalMs: 1500 }]);
  expect(pushPoint(s, null)).toBe(s);
});

it("shows live numbers when a Pette tab publishes on the channel", async () => {
  render(<Dashboard />);
  expect(await screen.findByText("Waiting for Pette")).toBeTruthy();
  const soma = new FakeSoma();
  await soma.grantScreen();
  const data = await buildSnapshot(createPette(soma, new FactStore()), live, false);
  const ch = new BroadcastChannel(CHANNEL);
  await act(async () => {
    ch.postMessage({ type: "snapshot", data });
    await new Promise((r) => setTimeout(r, 50));
  });
  expect(screen.getByText("Live")).toBeTruthy();
  expect(screen.getAllByText("review_changes").length).toBeGreaterThan(0); // a candidate badge and its bar
  ch.close();
});
