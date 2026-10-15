// Port of tests/test_ui.py (render side): the orb, the glass bubble, the file viewer, the world panel, and the
// whole app in sandbox mode around a fake SOMA.
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "@/App";
import { WorldPanel, type LiveState } from "@/components/WorldPanel";
import { PetteController } from "../controller";
import { FakeSoma, SAMPLES } from "../fakeSoma";
import { FactStore } from "../world/store";
import { WorldModel } from "../world/worldModel";
import { FileViewer } from "./FileViewer";
import { ORB_BOX, Orb } from "./Orb";
import { BUBBLE_WIDTH, SpeechBubble, dockBubble } from "./SpeechBubble";

// jsdom has no PointerEvent: a MouseEvent carries button / clientX just the same.
if (typeof window.PointerEvent === "undefined") {
  (window as unknown as { PointerEvent: unknown }).PointerEvent = class extends MouseEvent {
    pointerId: number;
    constructor(type: string, init: PointerEventInit = {}) {
      super(type, init);
      this.pointerId = init.pointerId ?? 1;
    }
  };
}

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null); // jsdom has no canvas
});
afterEach(() => vi.restoreAllMocks());

const card = (id: string, text: string) => ({ id, title: "SOMA suggests · Next step", text, actionLabel: "Copy", cancelLabel: "Not now", glow: true });

function Bubble({ c }: { c: PetteController }) {
  const s = c.getState();
  return <SpeechBubble controller={c} bubble={s.bubble} progress={s.progress} />;
}

it("renders suggestion text as text, never HTML, and the buttons decide once", () => {
  const c = new PetteController();
  const approved: string[] = [];
  c.events.on("approved", (p) => approved.push(p));
  c.propose(card("a", '<img src=x onerror="alert(1)"> <b>bold</b>'));
  c.setProgress(0.5);
  const { container, rerender } = render(<Bubble c={c} />);
  expect(screen.getByText('<img src=x onerror="alert(1)"> <b>bold</b>')).toBeTruthy();
  expect(container.querySelector("img, b")).toBeNull();
  expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("50");
  fireEvent.click(screen.getByText("Copy"));
  rerender(<Bubble c={c} />);
  expect(approved).toEqual(["a"]);
  expect((screen.getByText("Copy") as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByText("Not now") as HTMLButtonElement).disabled).toBe(true);
});

it("shows history pages read-only with their index", () => {
  const c = new PetteController();
  c.propose(card("a", "first"));
  c.propose(card("b", "second"));
  c.dismiss();
  c.step(-1);
  render(<Bubble c={c} />);
  expect(screen.getByTestId("history-index").textContent).toBe("2 / 2");
  expect(screen.getByText("second")).toBeTruthy();
  expect(screen.queryByText("Copy")).toBeNull();
});

it("docks bottom-centre and slides left rather than cover Pette", () => {
  const view = { width: 1200, height: 800 };
  const size = { width: BUBBLE_WIDTH, height: 140 };
  const centre = dockBubble(view, size, null);
  expect(centre).toEqual({ left: 370, top: 800 - 140 - 18 });
  const pette = { x: 560, y: 640, width: ORB_BOX.width, height: ORB_BOX.height };
  const slid = dockBubble(view, size, pette);
  expect(slid.left + BUBBLE_WIDTH).toBeLessThanOrEqual(pette.x);
  expect(dockBubble(view, size, { x: 1000, y: 640, width: 144, height: 156 })).toEqual(centre); // not in the way
});

it("orb: click looks now, a drag moves her instead", () => {
  const c = new PetteController();
  const looks: number[] = [];
  c.events.on("lookRequested", () => looks.push(1));
  const moves: { x: number; y: number }[] = [];
  render(<Orb controller={c} status="failed" detail="SOMA offline" mood="idle" cameraOn={false} feedVisible gesture="pinch"
    position={{ x: 100, y: 100 }} onMove={(p) => moves.push(p)} />);
  expect(screen.getByTestId("orb-label").textContent).toBe("SOMA offline");
  expect(screen.getByText("pinch")).toBeTruthy();
  const orb = screen.getByRole("button");
  fireEvent.pointerDown(orb, { button: 0, clientX: 150, clientY: 150 });
  fireEvent.pointerUp(orb, { button: 0, clientX: 150, clientY: 150 });
  expect(looks).toHaveLength(1);
  fireEvent.pointerDown(orb, { button: 0, clientX: 150, clientY: 150 });
  fireEvent.pointerMove(orb, { clientX: 190, clientY: 170 });
  fireEvent.pointerUp(orb, { button: 0, clientX: 190, clientY: 170 });
  expect(looks).toHaveLength(1);
  expect(moves.at(-1)).toEqual({ x: 140, y: 120 });
  fireEvent.keyDown(orb, { key: "Enter" });
  expect(looks).toHaveLength(2);
});

it("file viewer highlights the line and closes on Escape", () => {
  const closed: number[] = [];
  const text = "a\nb\nc\nd";
  render(<FileViewer file={{ name: "calc.py", text, blob: new Blob([text]), line: 3 }} onClose={() => closed.push(1)} />);
  expect(screen.getByRole("dialog", { name: "File: calc.py" })).toBeTruthy();
  expect(document.querySelector('[data-line="3"]')?.className).toContain("bg-amber");
  fireEvent.keyDown(window, { key: "Escape" });
  expect(closed).toHaveLength(1);
});

it("world panel shows facts only: counts, kinds, folder names", () => {
  const store = new FactStore();
  store.logOffer("soma:review_changes", "shown");
  store.logOffer("soma:review_changes", "accepted");
  store.logEvent("pinch");
  const snap = new WorldModel(store).snapshot();
  const live: LiveState = { state: "idle", detail: "", label: "", camera: false, gesture: "", present: true, eyesClosed: false,
    paused: false, suggestions: true, screenShared: false, now: Date.now() };
  render(<WorldPanel snapshot={snap} live={live} folders={[{ id: "f1", name: "notes", memories: 12 }]} />);
  expect(screen.getAllByText("Review changes").length).toBeGreaterThan(0);
  expect(screen.getByText("notes")).toBeTruthy();
  expect(screen.getByText("12 files")).toBeTruthy();
  expect(screen.getByText("no screen shared")).toBeTruthy();
});

it("the app in sandbox mode: first run, then a SOMA card in the bubble, Not now goes back to SOMA", async () => {
  const soma = new FakeSoma();
  const store = new FactStore();
  render(<App soma={soma} store={store} sandbox />);
  expect(screen.getByText("Share a screen")).toBeTruthy();
  fireEvent.click(screen.getByText("Share a screen"));
  await waitFor(() => expect(screen.getByText("Ready · SOMA")).toBeTruthy());
  expect(soma.calls("grantScreen")).toHaveLength(1);
  expect(screen.getByTestId("world-panel")).toBeTruthy();
  const id = soma.queue(SAMPLES["Review changes"]);
  await waitFor(() => expect(screen.getByText('Review your changes to "calc.py" before you commit')).toBeTruthy(), { timeout: 3000 });
  await act(async () => {
    fireEvent.click(screen.getByText("Not now"));
  });
  await waitFor(() => expect(soma.calls("feedback").map((r) => r.args)).toContainEqual([id, "rejected"]));
  expect(store.offers().map((o) => o.outcome)).toEqual(["shown", "dismissed"]);
  expect(screen.getByTestId("sandbox-log").textContent).toContain("feedback");
});
