import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, Camera, CameraOff, Eye, EyeOff, FolderOpen, Hand, Lightbulb, X } from "lucide-react";
import { BackgroundGradient } from "@/components/aceternity/background-gradient";
import { MovingBorder } from "@/components/aceternity/moving-border";
import { Spotlight } from "@/components/aceternity/spotlight";
import { Badge, type BadgeProps } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { ago, cn } from "@/lib/utils";

type Rule = { when: string; then: string; why: string };
type Kind = { kind: string; label: string; shown: number; accepted: number; dismissed: number; ignored: number };
type Folder = { name: string; path: string; memories: number };
type Live = { state: string; detail: string; label: string; camera: boolean; gesture: string; present: boolean;
  eyes_closed: boolean; paused: boolean; suggestions: boolean; now: number };
type Snapshot = {
  soma: { shown: number; accepted: number; dismissed: number; ignored: number }; accept_rate: number | null;
  kinds: Kind[]; muted: string[]; folders: Folder[];
  blink_threshold: number; blink_calibrated: boolean; baseline_samples: number;
  presence: { away_episodes: number; median_away_s: number | null; eyes_closed: number; pinches: number;
    tab_switches?: number; scrolls?: number };
  rules: Rule[]; recent: { ts: number; kind: string; outcome: string }[];
  events: { ts: number; kind: string }[]; updated: number; live: Live; error?: string;
};

declare global { interface Window { __pette?: (d: Snapshot) => void } }

const now = Date.now() / 1000;
const SAMPLE: Snapshot = {
  soma: { shown: 9, accepted: 4, dismissed: 3, ignored: 2 }, accept_rate: 4 / 7,
  kinds: [
    { kind: "soma:open_relevant_file", label: "Open file", shown: 4, accepted: 3, dismissed: 1, ignored: 0 },
    { kind: "soma:review_changes", label: "Review changes", shown: 3, accepted: 1, dismissed: 1, ignored: 1 },
    { kind: "soma:take_notes", label: "Save notes", shown: 2, accepted: 0, dismissed: 1, ignored: 1 },
  ],
  muted: [],
  folders: [{ name: "notes", path: "C:\\Users\\you\\notes", memories: 42 }, { name: "my-repo", path: "C:\\code\\my-repo", memories: 118 }],
  blink_threshold: 0.6, blink_calibrated: false, baseline_samples: 120,
  presence: { away_episodes: 2, median_away_s: 18, eyes_closed: 5, pinches: 3, tab_switches: 12, scrolls: 7 },
  rules: [
    { when: "SOMA sees a next step on your screen", then: "Pette shows it as a card; nothing happens until you press “Do it” or pinch", why: "9 shown: 4 done, 3 not now, 2 ignored" },
    { when: "Your eyes look closed on camera", then: "blink score above 0.6", why: "default until the camera has seen enough open-eye frames" },
  ],
  recent: [{ ts: now - 40, kind: "Open file", outcome: "accepted" }, { ts: now - 300, kind: "Save notes", outcome: "dismissed" }],
  events: [{ ts: now - 12, kind: "pinch" }, { ts: now - 90, kind: "face_back" }],
  updated: now,
  live: { state: "suggesting", detail: "", label: "Suggestion", camera: true, gesture: "", present: true, eyes_closed: false, paused: false, suggestions: true, now },
};

const STATE_STYLE: Record<string, { color: string; text: string }> = {
  idle: { color: "#a1a1aa", text: "Watching · click Pette to look now" },
  looking: { color: "#3b82f6", text: "Looking at your screen" },
  suggesting: { color: "#f59e0b", text: "Suggestion on screen" },
  done: { color: "#22c55e", text: "Done ✓" },
  dismissed: { color: "#71717a", text: "Not now" },
  indexing: { color: "#8b5cf6", text: "Indexing a folder" },
  failed: { color: "#ef4444", text: "SOMA offline" },
};
const OUTCOME: Record<string, { v: BadgeProps["variant"]; text: string }> = {
  accepted: { v: "success", text: "done" }, dismissed: { v: "secondary", text: "not now" }, ignored: { v: "outline", text: "ignored" },
};

export default function App() {
  const [d, setD] = useState<Snapshot>(SAMPLE);
  const [live, setLive] = useState(false);
  useEffect(() => {
    window.__pette = (data) => { setD(data); setLive(true); };
    return () => { window.__pette = undefined; };
  }, []);

  const st = STATE_STYLE[d.live?.state] ?? STATE_STYLE.idle;
  const busy = ["looking", "suggesting", "indexing"].includes(d.live?.state);
  const s = d.soma ?? { shown: 0, accepted: 0, dismissed: 0, ignored: 0 };
  const t = d.live?.now ?? d.updated;
  const label = d.live?.state === "failed" ? st.text : d.live?.state === "idle" || !d.live?.label ? st.text : d.live.label;

  return (
    <div className="h-full p-2">
      <Card className="relative flex h-full flex-col overflow-hidden rounded-2xl shadow-xl">
        {/* header */}
        <div className="relative overflow-hidden bg-zinc-950 px-4 pb-4 pt-3 text-zinc-50">
          <Spotlight className="-left-10 -top-24" fill="#a78bfa" />
          <div className="relative z-10 flex items-center justify-between">
            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.14em] text-zinc-400">Pette · SOMA</div>
              <div className="text-base font-semibold">World model</div>
            </div>
            <Button variant="ghost" size="icon" className="text-zinc-400 hover:bg-zinc-800 hover:text-zinc-50"
              onClick={() => (document.title = `close:${Date.now()}`)} aria-label="Close"><X className="h-4 w-4" /></Button>
          </div>
          <p className="relative z-10 mt-1 text-xs text-zinc-400">What SOMA suggested, what you did with it, and the folders it learns from.</p>
          {!live && <Badge variant="warning" className="relative z-10 mt-2">preview data</Badge>}
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto p-3">
          {/* live status */}
          <BackgroundGradient animate={busy} containerClassName="rounded-2xl" className="rounded-[14px] bg-white p-3">
            <div className="flex items-center justify-between">
              <MovingBorder active={busy} color={st.color}>
                <div className="flex items-center gap-2 px-3 py-1 text-xs font-semibold">
                  <span className="relative flex h-2 w-2">
                    {busy && <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75" style={{ background: st.color }} />}
                    <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: st.color }} />
                  </span>
                  <AnimatePresence mode="wait">
                    <motion.span key={label} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}>{label}</motion.span>
                  </AnimatePresence>
                </div>
              </MovingBorder>
              <div className="flex gap-1">
                {d.live?.paused && <Badge variant="secondary">paused</Badge>}
                {d.live && !d.live.suggestions && <Badge variant="secondary">suggestions off</Badge>}
              </div>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
              <Stat icon={d.live?.camera ? <Camera className="h-3.5 w-3.5" /> : <CameraOff className="h-3.5 w-3.5" />} label="Camera" value={d.live?.camera ? "on" : "off"} />
              <Stat icon={<Hand className="h-3.5 w-3.5" />} label="Hand" value={d.live?.gesture || "—"} />
              <Stat icon={d.live?.eyes_closed ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />} label="You"
                value={!d.live?.present ? "away" : d.live?.eyes_closed ? "eyes closed" : "here"} />
            </div>
          </BackgroundGradient>

          {/* suggestions */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Lightbulb className="h-4 w-4" /> Suggestions</CardTitle>
              <CardDescription>{d.accept_rate != null ? `You act on ${Math.round(d.accept_rate * 100)}% of the ones you answer` : "No answered suggestions yet"}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Progress value={(d.accept_rate ?? 0) * 100} indicator="bg-emerald-500" />
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline">{s.shown} shown</Badge>
                <Badge variant="success">{s.accepted} done</Badge>
                <Badge variant="secondary">{s.dismissed} not now</Badge>
                <Badge variant="outline">{s.ignored} ignored</Badge>
              </div>
              {d.kinds?.map((k) => {
                const total = Math.max(1, k.shown);
                const muted = d.muted?.includes(k.label);
                return (
                  <div key={k.kind} className="space-y-1">
                    <div className="flex items-center justify-between text-[13px]">
                      <span className={cn("font-medium", muted && "text-muted-foreground line-through")}>{k.label}</span>
                      <span className="text-[11px] text-muted-foreground">{muted ? "muted" : `${k.accepted}/${k.shown}`}</span>
                    </div>
                    <div className="flex h-1.5 overflow-hidden rounded-full bg-secondary">
                      <div className="bg-emerald-500" style={{ width: `${(k.accepted / total) * 100}%` }} />
                      <div className="bg-zinc-400" style={{ width: `${(k.dismissed / total) * 100}%` }} />
                    </div>
                  </div>
                );
              })}
            </CardContent>
          </Card>

          {/* folders */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><FolderOpen className="h-4 w-4" /> Your files</CardTitle>
              <CardDescription>Folders SOMA indexed for suggestions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-1.5">
              {d.folders?.length ? d.folders.map((f) => (
                <div key={f.path} className="flex items-center justify-between text-xs" title={f.path}>
                  <span className="truncate font-medium">{f.name}</span>
                  <Badge variant="info">{f.memories} files</Badge>
                </div>
              )) : <p className="text-xs text-muted-foreground">None yet. Tray → “Add a folder for SOMA…” so suggestions can point at your own files.</p>}
            </CardContent>
          </Card>

          {/* behaviour rules */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Activity className="h-4 w-4" /> Behaviour</CardTitle>
              <CardDescription>Rules derived from the db, applied right now</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {d.rules?.map((r, i) => (
                <motion.div key={r.when} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.04 }}
                  className="rounded-lg border bg-muted/40 p-2.5">
                  <div className="text-[11px] text-muted-foreground">When {r.when.charAt(0).toLowerCase() + r.when.slice(1)}</div>
                  <div className="text-[13px] font-medium">→ {r.then}</div>
                  <div className="mt-0.5 text-[11px] italic text-muted-foreground">because {r.why}</div>
                </motion.div>
              ))}
            </CardContent>
          </Card>

          {/* presence + calibration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Eye className="h-4 w-4" /> Last 24 hours</CardTitle>
              <CardDescription>Camera signals and gestures (event names only, no frames stored)</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2 text-[11px]">
              <Stat label="Away" value={`${d.presence?.away_episodes ?? 0}×${d.presence?.median_away_s ? ` · ~${d.presence.median_away_s}s` : ""}`} />
              <Stat label="Eyes closed" value={`${d.presence?.eyes_closed ?? 0}×`} />
              <Stat label="Pinches" value={`${d.presence?.pinches ?? 0}`} />
              <Stat label="Tab switches" value={`${d.presence?.tab_switches ?? 0}`} />
              <Stat label="Scrolls" value={`${d.presence?.scrolls ?? 0}`} />
              <Stat label="Blink threshold" value={`${d.blink_threshold}${d.blink_calibrated ? " ✓" : ""}`} />
            </CardContent>
          </Card>

          {/* recent */}
          <Card>
            <CardHeader>
              <CardTitle>Recent</CardTitle>
              <CardDescription>Which kind and what you did, never the text</CardDescription>
            </CardHeader>
            <CardContent className="space-y-1.5">
              {d.recent?.length ? d.recent.map((m, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="font-medium">{m.kind}</span>
                  <span className="flex items-center gap-1.5"><Badge variant={OUTCOME[m.outcome]?.v ?? "outline"}>{OUTCOME[m.outcome]?.text ?? m.outcome}</Badge>
                    <span className="w-14 text-right text-muted-foreground">{ago(m.ts, t)}</span></span>
                </div>
              )) : <p className="text-xs text-muted-foreground">Nothing yet.</p>}
              {d.events?.length > 0 && <Separator className="my-2" />}
              <div className="flex flex-wrap gap-1">
                {d.events?.map((e, i) => <Badge key={i} variant="secondary" className="font-normal">{e.kind.replace("_", " ")} · {ago(e.ts, t)}</Badge>)}
              </div>
            </CardContent>
          </Card>
          {d.error && <p className="text-xs text-destructive">World model unavailable ({d.error}).</p>}
        </div>
      </Card>
    </div>
  );
}

function Stat({ icon, label, value }: { icon?: React.ReactNode; label: string; value: string }) {
  return (
    <div className={cn("rounded-lg border bg-background px-2 py-1.5")}>
      <div className="flex items-center gap-1 text-muted-foreground">{icon}{label}</div>
      <div className="truncate text-[13px] font-semibold capitalize">{value}</div>
    </div>
  );
}
