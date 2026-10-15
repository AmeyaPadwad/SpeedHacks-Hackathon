/**
 * The world-model panel (port of web/src/App.tsx + pet/worldview.py): live status, suggestions shown / done /
 * not now per kind (and which are muted), the folders SOMA indexed, the explainable rules in effect, presence
 * over the last 24 hours, and recent kinds + outcomes.
 *
 * Facts only: it gets `WorldModel.snapshot()`, folder names and counts, and live flags. Never suggestion text,
 * file names or screen content. Pinch-and-drag scrolls it (`controller.events.on("scroll")`).
 */
import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, Camera, CameraOff, Eye, EyeOff, FolderOpen, Hand, Lightbulb, X } from "lucide-react";
import type { FolderInfo } from "@/soma/service";
import type { PetteStatus } from "@/pette/contracts";
import type { WorldSnapshot } from "@/pette/world/worldModel";
import { BackgroundGradient } from "@/components/aceternity/background-gradient";
import { MovingBorder } from "@/components/aceternity/moving-border";
import { Spotlight } from "@/components/aceternity/spotlight";
import { Badge, type BadgeProps } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { ago, cn } from "@/lib/utils";

/** Live state next to the learned facts. */
export interface LiveState {
  state: PetteStatus;
  detail: string;
  /** The orb's pill text. */
  label: string;
  camera: boolean;
  gesture: string;
  present: boolean;
  eyesClosed: boolean;
  paused: boolean;
  suggestions: boolean;
  screenShared: boolean;
  now: number;
}

const STATE_STYLE: Record<PetteStatus, { color: string; text: string }> = {
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

/** Touch-style: the panel follows your hand (hand down = content down). */
export const SCROLL_GAIN = 1.5;

export interface WorldPanelProps {
  snapshot: WorldSnapshot;
  live: LiveState;
  folders: FolderInfo[];
  onClose?: () => void;
  /** Pinch-and-drag: subscribe to scroll amounts (image heights, + = hand down). */
  onScrollGesture?: (fn: (dy: number) => void) => () => void;
  className?: string;
}

export function WorldPanel({ snapshot: d, live, folders, onClose, onScrollGesture, className }: WorldPanelProps) {
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => onScrollGesture?.((dy) => {
    const el = scroller.current;
    if (el) el.scrollTop -= dy * el.clientHeight * SCROLL_GAIN;
  }), [onScrollGesture]);

  const st = STATE_STYLE[live.state] ?? STATE_STYLE.idle;
  const busy = live.state === "looking" || live.state === "suggesting" || live.state === "indexing";
  const s = d.soma;
  const t = live.now;
  const label = live.state === "idle" || !live.label ? st.text : live.label;

  return (
    <aside aria-label="World model" data-testid="world-panel" className={cn("h-full p-2", className)}>
      <Card className="relative flex h-full flex-col overflow-hidden rounded-2xl shadow-xl">
        {/* header */}
        <div className="relative overflow-hidden bg-zinc-950 px-4 pb-4 pt-3 text-zinc-50">
          <Spotlight className="-left-10 -top-24" fill="#a78bfa" />
          <div className="relative z-10 flex items-center justify-between">
            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.14em] text-zinc-400">Pette · SOMA</div>
              <div className="text-base font-semibold">World model</div>
            </div>
            {onClose && (
              <Button variant="ghost" size="icon" className="text-zinc-400 hover:bg-zinc-800 hover:text-zinc-50" onClick={onClose}
                aria-label="Close the world panel"><X className="h-4 w-4" /></Button>
            )}
          </div>
          <p className="relative z-10 mt-1 text-xs text-zinc-400">What SOMA suggested, what you did with it, and the folders it learns from.</p>
        </div>

        <div ref={scroller} className="flex-1 space-y-3 overflow-y-auto p-3">
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
              <div className="flex flex-wrap justify-end gap-1">
                {!live.screenShared && <Badge variant="warning">no screen shared</Badge>}
                {live.paused && <Badge variant="secondary">paused</Badge>}
                {!live.suggestions && <Badge variant="secondary">suggestions off</Badge>}
              </div>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
              <Stat icon={live.camera ? <Camera className="h-3.5 w-3.5" /> : <CameraOff className="h-3.5 w-3.5" />} label="Camera" value={live.camera ? "on" : "off"} />
              <Stat icon={<Hand className="h-3.5 w-3.5" />} label="Hand" value={live.gesture || "—"} />
              <Stat icon={live.eyesClosed ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />} label="You"
                value={!live.present ? "away" : live.eyesClosed ? "eyes closed" : "here"} />
            </div>
          </BackgroundGradient>

          {/* suggestions */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Lightbulb className="h-4 w-4" /> Suggestions</CardTitle>
              <CardDescription>{d.acceptRate != null ? `You act on ${Math.round(d.acceptRate * 100)}% of the ones you answer` : "No answered suggestions yet"}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Progress value={(d.acceptRate ?? 0) * 100} indicator="bg-emerald-500" />
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline">{s.shown} shown</Badge>
                <Badge variant="success">{s.accepted} done</Badge>
                <Badge variant="secondary">{s.dismissed} not now</Badge>
                <Badge variant="outline">{s.ignored} ignored</Badge>
              </div>
              {d.kinds.map((k) => {
                const total = Math.max(1, k.shown);
                const muted = d.muted.includes(k.label);
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
              {folders.length ? folders.map((f) => (
                <div key={f.id} className="flex items-center justify-between text-xs">
                  <span className="truncate font-medium">{f.name}</span>
                  <Badge variant="info">{f.memories} files</Badge>
                </div>
              )) : <p className="text-xs text-muted-foreground">None yet. Settings → “Add a folder” so suggestions can point at your own files.</p>}
            </CardContent>
          </Card>

          {/* behaviour rules */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Activity className="h-4 w-4" /> Behaviour</CardTitle>
              <CardDescription>Rules derived from the fact log, applied right now</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {d.rules.map((r, i) => (
                <motion.div key={r.when} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.04 }}
                  className="rounded-lg border bg-muted/40 p-2.5">
                  <div className="text-[11px] text-muted-foreground">When {/^[A-Z][a-z]/.test(r.when) ? r.when.charAt(0).toLowerCase() + r.when.slice(1) : r.when}</div>
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
              <Stat label="Away" value={`${d.presence.awayEpisodes}×${d.presence.medianAwayS ? ` · ~${d.presence.medianAwayS}s` : ""}`} />
              <Stat label="Eyes closed" value={`${d.presence.eyesClosed}×`} />
              <Stat label="Pinches" value={`${d.presence.pinches}`} />
              <Stat label="Points" value={`${d.presence.steps}`} />
              <Stat label="Scrolls" value={`${d.presence.scrolls}`} />
              <Stat label="Blink threshold" value={`${d.blinkThreshold}${d.blinkCalibrated ? " ✓" : ""}`} />
            </CardContent>
          </Card>

          {/* recent */}
          <Card>
            <CardHeader>
              <CardTitle>Recent</CardTitle>
              <CardDescription>Which kind and what you did, never the text</CardDescription>
            </CardHeader>
            <CardContent className="space-y-1.5">
              {d.recent.length ? d.recent.map((m, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="font-medium">{m.kind}</span>
                  <span className="flex items-center gap-1.5"><Badge variant={OUTCOME[m.outcome]?.v ?? "outline"}>{OUTCOME[m.outcome]?.text ?? m.outcome}</Badge>
                    <span className="w-14 text-right text-muted-foreground">{ago(m.ts, t)}</span></span>
                </div>
              )) : <p className="text-xs text-muted-foreground">Nothing yet.</p>}
              {d.events.length > 0 && <Separator className="my-2" />}
              <div className="flex flex-wrap gap-1">
                {d.events.map((e, i) => <Badge key={i} variant="secondary" className="font-normal">{e.kind.replace(/_/g, " ")} · {ago(e.ts, t)}</Badge>)}
              </div>
            </CardContent>
          </Card>
        </div>
      </Card>
    </aside>
  );
}

function Stat({ icon, label, value }: { icon?: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-background px-2 py-1.5">
      <div className="flex items-center gap-1 text-muted-foreground">{icon}{label}</div>
      <div className="truncate text-[13px] font-semibold capitalize">{value}</div>
    </div>
  );
}
