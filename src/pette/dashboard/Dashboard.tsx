/**
 * The `/debug` dashboard: every Pette + SOMA stat on one page, live from the Pette tab over the dashboard
 * channel. Open it in its own tab next to Pette. Without a Pette tab it still shows what the fact log knows
 * (read from IndexedDB) and says it is waiting.
 *
 * Facts and numbers only: status, counts, timings, kinds. Pinch-and-drag in the Pette tab scrolls this page.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Activity, Brain, Camera, Cpu, ExternalLink, Eye, FolderOpen, Gauge, Hand, MonitorUp, Radio, Sparkles } from "lucide-react";
import { WorldPanel } from "@/components/WorldPanel";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { statusColor, statusLabel } from "../status";
import { FactStore } from "../world/store";
import { WorldModel, type WorldSnapshot } from "../world/worldModel";
import { openChannel, type DashboardSnapshot } from "./channel";

/** No snapshot for this long = the Pette tab is closed (it publishes every second). */
export const STALE_MS = 4000;
const SERIES_LEN = 60;
const BAR = "#8b5cf6"; // one hue for every single-series bar
const SILENT = "#a1a1aa"; // NO_SUGGESTION and other "nothing" bars stay neutral

export interface StepPoint { steps: number; totalMs: number }

/** Append a step-time point when SOMA processed a new frame (keyed by step count). */
export function pushPoint(series: StepPoint[], snap: DashboardSnapshot | null, max = SERIES_LEN): StepPoint[] {
  const total = snap?.diagnostics?.timingsMs.total;
  const steps = snap?.soma?.processed ?? 0;
  if (!total || series.at(-1)?.steps === steps) return series;
  return [...series, { steps, totalMs: total }].slice(-max);
}

function useLocalWorld(enabled: boolean): WorldSnapshot | null {
  const [snap, setSnap] = useState<WorldSnapshot | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const load = async () => {
      try {
        const store = await FactStore.open();
        if (alive) setSnap(new WorldModel(store).snapshot());
        store.close();
      } catch {
        // no IndexedDB: the dashboard just waits for Pette
      }
    };
    void load();
    const t = setInterval(() => void load(), 5000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [enabled]);
  return snap;
}

export function Dashboard() {
  const [snap, setSnap] = useState<DashboardSnapshot | null>(null);
  const [series, setSeries] = useState<StepPoint[]>([]);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const ch = openChannel((m) => {
      if (m.type === "snapshot") {
        setSnap(m.data);
        setSeries((s) => pushPoint(s, m.data));
      } else if (m.type === "scroll") {
        window.scrollBy({ top: -m.dy * window.innerHeight * 1.5 });
      }
    });
    ch.post({ type: "hello" });
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      clearInterval(t);
      ch.close();
    };
  }, []);

  const live = !!snap && now - snap.at < STALE_MS;
  const localWorld = useLocalWorld(!live);
  const world = live ? snap!.world : localWorld;
  const d = live ? snap!.diagnostics : null;
  const soma = live ? snap!.soma : null;
  const p = live ? snap!.pette : null;

  return (
    <main className="min-h-full bg-gradient-to-br from-zinc-50 via-white to-violet-50/40 p-4 sm:p-6">
      <header className="mx-auto mb-5 flex max-w-7xl flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-[11px] font-medium uppercase tracking-[0.14em] text-zinc-500">Pette · SOMA</div>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-950">Dashboard</h1>
          <p className="text-sm text-zinc-500">Everything Pette and SOMA are doing, live from your Pette tab. Facts and numbers only.</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={live ? "success" : "outline"} className="gap-1.5">
            <Radio className="h-3 w-3" /> {live ? (snap!.sandbox ? "Live · sandbox" : "Live") : "Waiting for Pette"}
          </Badge>
          <a href="./" target="_blank" rel="noopener" className="inline-flex items-center gap-1 rounded-md border bg-white px-2.5 py-1 text-xs font-medium hover:bg-zinc-50">
            Open Pette <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </header>

      {!live && (
        <div className="mx-auto mb-5 max-w-7xl rounded-xl border border-dashed bg-white/70 p-4 text-sm text-zinc-600">
          No Pette tab is running in this browser. Open Pette (button above) and this page fills in within a second.
          The world model below comes from Pette's saved fact log.
        </div>
      )}

      <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-[1fr_380px]">
        <div className="space-y-4">
          <section aria-label="Headline numbers" className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
            <Stat icon={<Activity className="h-4 w-4" />} label="Pette" value={p ? p.label || statusLabel(p.status) || "Watching" : "—"}
              dot={p ? statusColor(p.status) : undefined} />
            <Stat icon={<MonitorUp className="h-4 w-4" />} label="Screen" value={p ? (p.screenShared ? "Shared" : "Not shared") : "—"}
              sub={soma ? (soma.paused ? "paused" : soma.ready ? "SOMA ready" : "SOMA loading") : undefined} />
            <Stat icon={<Gauge className="h-4 w-4" />} label="Steps" value={soma ? String(soma.steps) : "—"}
              sub={soma ? `${soma.processed} processed` : undefined} />
            <Stat icon={<Sparkles className="h-4 w-4" />} label="Suggestions" value={d ? String(d.spoke) : world ? String(world.soma.shown) : "—"}
              sub={world ? `${world.soma.accepted} done · ${world.soma.dismissed} not now` : undefined} />
            <Stat icon={<FolderOpen className="h-4 w-4" />} label="Memories" value={soma ? String(soma.memories) : "—"}
              sub={live ? `${snap!.folders.length} folder(s)` : undefined} />
            <Stat icon={<Cpu className="h-4 w-4" />} label="Tensors" value={d?.tf ? String(d.tf.tensors) : "—"}
              sub={d?.tf ? `${d.tf.backend} · ${(d.tf.bytes / 1e6).toFixed(1)} MB` : "model not loaded"} />
          </section>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm"><Brain className="h-4 w-4" /> What SOMA sees</CardTitle>
                <CardDescription>The last processed frame</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <KV k="App" v={d ? d.app || "unknown" : "—"} />
                <KV k="Surface" v={d?.surface || "—"} />
                <KV k="OCR lines" v={d ? String(d.ocrLines) : "—"} />
                <KV k="Frame version" v={soma ? String(soma.frameVersion) : "—"} />
                <KV k="Last skip" v={soma?.lastSkipReason ?? "none"} />
                <div>
                  <div className="mb-1 text-xs text-zinc-500">Candidates ({d?.candidates.length ?? 0})</div>
                  <div className="flex flex-wrap gap-1">
                    {d?.candidates.length
                      ? d.candidates.map((c, i) => <Badge key={i} variant="outline">{c.template || c.action.toLowerCase()}</Badge>)
                      : <span className="text-xs text-zinc-400">none: SOMA stays silent</span>}
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm"><Sparkles className="h-4 w-4" /> Last decision</CardTitle>
                <CardDescription>Probability per option, after the ranker</CardDescription>
              </CardHeader>
              <CardContent>
                <Bars rows={Object.entries(d?.distribution ?? {}).sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => ({ label: k === "NO_SUGGESTION" ? "stay silent" : k.split(":").at(-1)!.toLowerCase(), value: v,
                    color: k === "NO_SUGGESTION" ? SILENT : BAR, text: `${Math.round(v * 100)}%` }))}
                  max={1} empty="No decision yet" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm"><Gauge className="h-4 w-4" /> Step time</CardTitle>
                <CardDescription>Milliseconds per processed frame, last {SERIES_LEN}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Sparkline points={series} />
                <Bars rows={Object.entries(d?.timingsMs ?? {}).filter(([k]) => k !== "total")
                  .map(([k, v]) => ({ label: k, value: v, color: BAR, text: `${v} ms` }))}
                  max={Math.max(1, ...Object.entries(d?.timingsMs ?? {}).filter(([k]) => k !== "total").map(([, v]) => v))}
                  empty="No frame processed yet" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm"><Activity className="h-4 w-4" /> Why steps skipped</CardTitle>
                <CardDescription>Since SOMA started in the Pette tab</CardDescription>
              </CardHeader>
              <CardContent>
                <Bars rows={Object.entries(d?.skips ?? {}).sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => ({ label: k, value: v, color: SILENT, text: String(v) }))}
                  max={Math.max(1, ...Object.values(d?.skips ?? {}))} empty="No skips" />
              </CardContent>
            </Card>

            <Card className="md:col-span-2">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm"><Camera className="h-4 w-4" /> Pette</CardTitle>
                <CardDescription>Switches, camera and gestures right now</CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
                <Flag label="Floating over tabs" on={p?.floating} />
                <Flag label="Card up" on={p?.cardUp} />
                <Flag label="Suggestions" on={p?.suggestions} />
                <Flag label="Speech bubble" on={p?.bubble} />
                <Flag label="Paused" on={p?.paused} />
                <Flag label="Camera" on={p?.camera} />
                <Flag label="Gestures" on={p?.gestures} />
                <Flag label="You're here" on={p?.present} icon={<Eye className="h-3.5 w-3.5" />} />
                <KV k="Gesture" v={p?.gesture || "—"} icon={<Hand className="h-3.5 w-3.5" />} />
                <KV k="Mood" v={p?.mood ?? "—"} />
                <KV k="Eyes" v={p ? (p.eyesClosed ? "closed" : "open") : "—"} />
              </CardContent>
            </Card>
          </div>
        </div>

        <div className="lg:sticky lg:top-4 lg:h-[calc(100vh-2rem)]">
          {world ? (
            <WorldPanel snapshot={world} folders={live ? snap!.folders : []} className="h-[80vh] p-0 lg:h-full"
              live={{ state: p?.status ?? "idle", detail: "", label: p?.label ?? "", camera: !!p?.camera, gesture: p?.gesture ?? "",
                present: p?.present ?? true, eyesClosed: !!p?.eyesClosed, paused: !!p?.paused, suggestions: p?.suggestions ?? true,
                screenShared: !!p?.screenShared, now: world.updated }} />
          ) : (
            <Card className="p-4 text-sm text-zinc-500">Loading the world model…</Card>
          )}
        </div>
      </div>
    </main>
  );
}

function Stat({ icon, label, value, sub, dot }: { icon: ReactNode; label: string; value: string; sub?: string; dot?: string }) {
  return (
    <Card className="p-3">
      <div className="flex items-center gap-1.5 text-xs text-zinc-500">{icon}{label}</div>
      <div className="mt-1 flex items-center gap-1.5 text-lg font-semibold text-zinc-950">
        {dot && <span className="h-2 w-2 rounded-full" style={{ background: dot }} aria-hidden />}
        <span className="truncate">{value}</span>
      </div>
      {sub && <div className="truncate text-[11px] text-zinc-500">{sub}</div>}
    </Card>
  );
}

function KV({ k, v, icon }: { k: string; v: string; icon?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="flex items-center gap-1 text-xs text-zinc-500">{icon}{k}</span>
      <span className="truncate font-medium text-zinc-900">{v}</span>
    </div>
  );
}

function Flag({ label, on, icon }: { label: string; on: boolean | undefined; icon?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 rounded-md border px-2 py-1">
      <span className="flex items-center gap-1 text-xs text-zinc-600">{icon}{label}</span>
      <span className={cn("text-xs font-medium", on ? "text-emerald-700" : "text-zinc-400")}>{on === undefined ? "—" : on ? "on" : "off"}</span>
    </div>
  );
}

interface BarRow { label: string; value: number; color: string; text: string }

/** Single-series horizontal bars: one hue, value labels in ink, native tooltip per row. */
function Bars({ rows, max, empty }: { rows: BarRow[]; max: number; empty: string }) {
  if (!rows.length) return <p className="text-xs text-zinc-400">{empty}</p>;
  return (
    <ul className="space-y-1.5">
      {rows.map((r) => (
        <li key={r.label} title={`${r.label}: ${r.text}`} className="grid grid-cols-[96px_1fr_56px] items-center gap-2 text-xs">
          <span className="truncate text-zinc-600">{r.label}</span>
          <span className="h-2.5 rounded-full bg-zinc-100">
            <span className="block h-full rounded-full" style={{ width: `${Math.max(2, (r.value / max) * 100)}%`, background: r.color }} />
          </span>
          <span className="text-right tabular-nums text-zinc-900">{r.text}</span>
        </li>
      ))}
    </ul>
  );
}

/** Step time over the last frames: a 2px line, hover shows the nearest point. */
function Sparkline({ points }: { points: StepPoint[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const W = 320;
  const H = 64;
  const max = useMemo(() => Math.max(1, ...points.map((p) => p.totalMs)), [points]);
  if (points.length < 2) return <p className="text-xs text-zinc-400">Waiting for a few processed frames…</p>;
  const x = (i: number) => (i / (points.length - 1)) * (W - 8) + 4;
  const y = (v: number) => H - 6 - (v / max) * (H - 12);
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.totalMs).toFixed(1)}`).join(" ");
  const h = hover !== null ? points[hover] : null;
  return (
    <div className="relative">
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="h-16 w-full" role="img"
        aria-label={`Step time, latest ${points.at(-1)!.totalMs} ms, max ${max} ms`}
        onMouseMove={(e) => {
          const r = ref.current!.getBoundingClientRect();
          const i = Math.round(((e.clientX - r.left) / r.width) * (points.length - 1));
          setHover(Math.max(0, Math.min(points.length - 1, i)));
        }}
        onMouseLeave={() => setHover(null)}>
        <line x1="0" x2={W} y1={H - 6} y2={H - 6} stroke="#e4e4e7" strokeWidth="1" />
        <path d={d} fill="none" stroke={BAR} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        {hover !== null && <>
          <line x1={x(hover)} x2={x(hover)} y1="0" y2={H} stroke="#d4d4d8" strokeWidth="1" />
          <circle cx={x(hover)} cy={y(points[hover].totalMs)} r="4" fill={BAR} stroke="#fff" strokeWidth="2" />
        </>}
      </svg>
      <div className="mt-1 flex justify-between text-[11px] text-zinc-500">
        <span>{h ? `step ${h.steps}: ${h.totalMs} ms` : `latest ${points.at(-1)!.totalMs} ms`}</span>
        <span>max {max} ms</span>
      </div>
    </div>
  );
}
