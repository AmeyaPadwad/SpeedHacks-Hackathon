/**
 * `?sandbox`: the real app plus a control panel around a scripted `FakeSoma` (port of tools/sandbox.py).
 * Queue an open-file, your-files or copy-only card, make "look now" find nothing, stop SOMA to see the
 * offline state, simulate webcam events and gestures, watch every request Pette makes, and tick off the
 * manual checklist (copied as Markdown).
 */
import { ChevronDown, ChevronUp, FlaskConical } from "lucide-react";
import { useEffect, useState } from "react";
import type { VisionEvent } from "./contracts";
import { type FakeRequest, FakeSoma, SAMPLES } from "./fakeSoma";
import type { Pette } from "./usePette";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** [area, scenario, how, expected] */
export const SCENARIOS: [string, string, string, string][] = [
  ["Launch", "Startup", "Open the page with ?sandbox.", "First-run screen; after Continue: Pette, the world panel, bubble “Ready · SOMA”."],
  ["Cards", "Open a file", "Panel: “Open file at line”.",
    "Glass card “SOMA suggests · Open file”, Open / Not now, a draining bar. Open → the file viewer at line 7; orb “Done ✓”."],
  ["Cards", "Your files", "Panel: “Your files (memory)”, then Open.", "The note opens in the viewer; nothing else runs."],
  ["Cards", "Copy", "Panel: “Review changes”, then Copy.", "The text is on the clipboard; bubble says Copied."],
  ["Cards", "Not now", "Queue a card, click Not now.", "Card closes, orb “Not now”, log shows feedback rejected."],
  ["Cards", "Ignore", "Queue a card, wait 15 s.", "Bar drains, card closes, log shows feedback ignored."],
  ["Cards", "Pinch", "Queue a card, panel: Pinch.", "Same as Open/Copy, once."],
  ["Cards", "Muted kind", "Answer Not now to “Write a test” 4 times, then queue it again.",
    "It no longer shows; world panel lists it as muted with the reason."],
  ["Look now", "Click Pette", "Click Pette (or Ctrl+Shift+Space).", "“Looking at your screen…”, log shows trigger; the queued (or next) card appears."],
  ["Look now", "Nothing found", "Tick “Look finds nothing”, click Pette, wait 12 s.", "“Nothing worth suggesting right now.”"],
  ["Folders", "Add", "Settings → Add.", "“Indexing …” then “Indexed N file(s)”; settings and the world panel list it."],
  ["Folders", "Remove", "Settings → the bin next to it.", "“SOMA forgot N file(s)”; the list empties."],
  ["Control", "Toggle", "Ctrl+Alt+S twice.", "“Suggestions off” (log: pause), then on (log: resume); no cards while off."],
  ["Control", "Pause", "Settings → Pause all, queue a card, unpause.", "No card while paused, camera off; log: pause/resume."],
  ["Control", "Offline", "Click “Stop fake SOMA”, wait ~20 s.", "A notice; orb “SOMA offline”."],
  ["Orb", "Hide camera", "Ctrl+Alt+C twice.", "The picture in the orb becomes “camera hidden” and returns; Pette stays."],
  ["Orb", "Click / drag", "Click the orb; then drag it somewhere else; reload.", "Click = look now; drag moves Pette (no look); she stays there after reload."],
  ["Bubble", "Bottom of the page", "Queue any card; drag Pette to the bottom centre.", "The glass bubble sits bottom-centre with a tail, and slides left rather than cover Pette."],
  ["Bubble", "Toggle", "Ctrl+Alt+B, queue a card, Ctrl+Alt+B again.", "Off: bubble gone, the card doesn't show (notice). On: the next card shows."],
  ["Gestures", "Earlier / later", "Answer three cards, then panel: Point left, Point right.", "The bubble pages through them read-only (“2 / 3”)."],
  ["Gestures", "Scroll", "Panel: Pinch-drag up / down.", "The world panel scrolls (touch-style)."],
  ["Gestures", "Real camera", "Settings → Camera: point left/right, pinch and move your hand.", "The orb chip shows “point left/right” / “pinch”."],
  ["Float", "Picture-in-Picture", "Settings → Float Pette (Chrome/Edge).", "A small always-on-top window with Pette and the bubble; cards work there."],
  ["World model", "Hide panel", "Ctrl+Alt+W twice.", "Panel hides and docks again."],
  ["World model", "Facts only", "Answer a few cards, open the panel.", "Counts per kind and folder names only; no suggestion text or file names."],
  ["Webcam", "Away / blink", "Face away / Face back with a card up; Eyes closed.", "“While you were away · …”; Pette blinks on eyes closed."],
];

const RESULTS = ["—", "Pass", "Fail", "Skip"] as const;

export function Sandbox({ soma, pette, simulate }: {
  soma: FakeSoma;
  pette: Pette;
  simulate: (ev: VisionEvent | { scroll: number }) => void;
}) {
  const { controller, suggester } = pette;
  const [open, setOpen] = useState(true);
  const [log, setLog] = useState<FakeRequest[]>(() => [...soma.requests]);
  const [emptyLook, setEmptyLook] = useState(soma.emptyLook);
  const [offline, setOffline] = useState(soma.offline);
  const [results, setResults] = useState<string[]>(() => SCENARIOS.map(() => "—"));
  const [copied, setCopied] = useState("");
  useEffect(() => soma.onRequest((r) => setLog((l) => [...l.slice(-199), r])), [soma]);

  const drag = (direction: -1 | 1) => {
    simulate("drag_start");
    for (let i = 0; i < 5; i++) simulate({ scroll: 0.04 * direction });
    simulate("drag_end");
  };
  const summary = `${results.filter((r) => r === "Pass").length} pass · ${results.filter((r) => r === "Fail").length} fail · `
    + `${results.filter((r) => r === "Skip").length} skip · ${results.filter((r) => r === "—").length} not run`;
  const copyReport = async () => {
    const lines = ["# Manual test report", "", `- Date: ${new Date().toISOString().slice(0, 16).replace("T", " ")}`,
      "- SOMA: fake (scripted)", `- Result: ${summary}`, "", "| Area | Scenario | Result |", "|---|---|---|",
      ...SCENARIOS.map(([area, name], i) => `| ${area} | ${name} | ${results[i]} |`)];
    try {
      await navigator.clipboard.writeText(lines.join("\n") + "\n");
      setCopied("Copied");
    } catch {
      setCopied("Clipboard blocked");
    }
  };

  const group = (title: string, buttons: [string, () => void][]) => (
    <div>
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">{title}</div>
      <div className="flex flex-wrap gap-1">
        {buttons.map(([label, fn]) => <Button key={label} size="sm" variant="outline" className="h-7 px-2 text-xs" onClick={fn}>{label}</Button>)}
      </div>
    </div>
  );

  return (
    <section aria-label="Sandbox" className={cn("fixed left-3 top-3 z-[58] w-[380px] max-w-[calc(100vw-24px)] rounded-xl border bg-white/95 shadow-xl backdrop-blur",
      open ? "max-h-[calc(100vh-24px)] overflow-y-auto" : "")}>
      <button type="button" className="flex w-full items-center gap-2 px-3 py-2 text-sm font-semibold" onClick={() => setOpen((o) => !o)}>
        <FlaskConical className="h-4 w-4 text-violet-500" /> Sandbox · fake SOMA
        <span className="ml-auto text-xs font-normal text-zinc-500">{suggester.state}</span>
        {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>
      {open && (
        <div className="space-y-3 px-3 pb-3">
          {group("Queue a SOMA suggestion", Object.entries(SAMPLES).map(([name, s]) => [name, () => void soma.queue(s)]))}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-1.5"><input type="checkbox" checked={emptyLook}
              onChange={(e) => { soma.emptyLook = e.target.checked; setEmptyLook(e.target.checked); }} /> Look finds nothing</label>
            <Button size="sm" variant={offline ? "default" : "outline"} className="h-7 px-2 text-xs"
              onClick={() => { soma.offline = !offline; setOffline(!offline); }}>{offline ? "Start fake SOMA" : "Stop fake SOMA"}</Button>
          </div>
          {group("Simulated webcam events", [
            ["Face away", () => simulate("face_away")], ["Face back", () => simulate("face_back")],
            ["Eyes closed", () => simulate("eyes_closed")], ["Eyes open", () => simulate("eyes_open")], ["Pinch", () => simulate("pinch")],
          ])}
          {group("Simulated gestures", [
            ["Point left", () => simulate("point_left")], ["Point right", () => simulate("point_right")],
            ["Pinch-drag up", () => drag(-1)], ["Pinch-drag down", () => drag(1)],
          ])}
          {group("Triggers", [
            ["Click Pette (look now)", () => controller.requestLook()], ["Speech bubble on/off", () => controller.toggleBubble()],
            ["Pause / resume", () => controller.setPaused(!controller.paused)], ["Suggestions on/off", () => suggester.toggle()],
            ["World panel", () => controller.toggleWorld()],
          ])}
          <div>
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Requests to SOMA</div>
            <ol className="max-h-32 overflow-y-auto rounded-md border bg-zinc-50 p-1.5 font-mono text-[11px]" data-testid="sandbox-log">
              {log.length ? log.slice().reverse().map((r, i) => (
                <li key={i}>{new Date(r.at).toLocaleTimeString()} {r.call} {r.args.map((a) => JSON.stringify(a)).join(" ")}</li>
              )) : <li className="text-zinc-400">Every request Pette makes to SOMA lands here.</li>}
            </ol>
          </div>
          <details>
            <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Checklist · {summary}</summary>
            <ul className="mt-1 space-y-1.5">
              {SCENARIOS.map(([area, name, how, expected], i) => (
                <li key={`${area}-${name}`} className="rounded-md border p-1.5 text-[11px]">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{area} · {name}</span>
                    <select aria-label={`${area} ${name} result`} className="ml-auto rounded border px-1 py-0.5" value={results[i]}
                      onChange={(e) => setResults((r) => r.map((x, j) => (j === i ? e.target.value : x)))}>
                      {RESULTS.map((o) => <option key={o}>{o}</option>)}
                    </select>
                  </div>
                  <div className="text-zinc-500">{how}</div>
                  <div>{expected}</div>
                </li>
              ))}
            </ul>
            <Button size="sm" variant="outline" className="mt-2 h-7 px-2 text-xs" onClick={() => void copyReport()}>
              Copy report (Markdown) {copied && `· ${copied}`}
            </Button>
          </details>
        </div>
      )}
    </section>
  );
}
