import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, Camera, CameraOff, Eye, EyeOff, Hand, Mail, MessageSquare, Send, Timer, UserRound, X } from "lucide-react";
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
type Contact = { name: string; platform: string; sent: number; drafted: number; cancelled: number };
type Live = { state: string; detail: string; label: string; camera: boolean; gesture: string; present: boolean;
  eyes_closed: boolean; paused: boolean; now: number };
type Snapshot = {
  contacts: Contact[]; outcomes: Record<string, number>; send_rate: number | null; send_delay_s: number;
  blink_threshold: number; blink_calibrated: boolean; baseline_samples: number;
  presence: { away_episodes: number; median_away_s: number | null; eyes_closed: number; pinches: number };
  rules: Rule[]; recent: { ts: number; recipient: string; platform: string; outcome: string }[];
  events: { ts: number; kind: string }[]; updated: number; live: Live; error?: string;
};

declare global { interface Window { __pette?: (d: Snapshot) => void } }

const now = Date.now() / 1000;
const SAMPLE: Snapshot = {
  contacts: [{ name: "Sarah", platform: "slack", sent: 3, drafted: 4, cancelled: 1 }, { name: "Alex", platform: "discord", sent: 1, drafted: 2, cancelled: 0 }],
  outcomes: { drafted: 6, sent: 4, cancelled: 1, failed: 0 }, send_rate: 0.8, send_delay_s: 4,
  blink_threshold: 0.6, blink_calibrated: false, baseline_samples: 120,
  presence: { away_episodes: 2, median_away_s: 18, eyes_closed: 5, pinches: 3 },
  rules: [
    { when: "You say “message Sarah” without a platform", then: "Pette uses Slack", why: "3 sent, 4 drafted to Sarah there" },
    { when: "A draft is ready", then: "Pette waits 4s before sending (Cancel shows)", why: "default until 5 messages are decided" },
    { when: "Your eyes look closed on camera", then: "blink score above 0.6", why: "default until the camera has seen enough open-eye frames" },
  ],
  recent: [{ ts: now - 40, recipient: "Sarah", platform: "slack", outcome: "sent" }, { ts: now - 300, recipient: "Alex", platform: "discord", outcome: "cancelled" }],
  events: [{ ts: now - 12, kind: "pinch" }, { ts: now - 90, kind: "face_back" }],
  updated: now,
  live: { state: "listening", detail: "", label: "Listening", camera: true, gesture: "open hand", present: true, eyes_closed: false, paused: false, now },
};

const STATE_STYLE: Record<string, { badge: BadgeProps["variant"]; color: string; text: string }> = {
  idle: { badge: "secondary", color: "#a1a1aa", text: "Idle · click Pette to talk" },
  listening: { badge: "violet", color: "#8b5cf6", text: "Listening" },
  thinking: { badge: "info", color: "#3b82f6", text: "Drafting with Groq" },
  pending: { badge: "warning", color: "#f59e0b", text: "Sending soon" },
  sending: { badge: "info", color: "#3b82f6", text: "Sending" },
  sent: { badge: "success", color: "#22c55e", text: "Sent ✓" },
  cancelled: { badge: "secondary", color: "#71717a", text: "Not sent (cancelled)" },
  failed: { badge: "destructive", color: "#ef4444", text: "Not sent" },
};
const OUTCOME: Record<string, BadgeProps["variant"]> = { sent: "success", cancelled: "secondary", failed: "destructive", drafted: "outline" };
const PlatformIcon = ({ p }: { p: string }) => (p === "email" ? <Mail className="h-3 w-3" /> : <MessageSquare className="h-3 w-3" />);

export default function App() {
  const [d, setD] = useState<Snapshot>(SAMPLE);
  const [live, setLive] = useState(false);
  useEffect(() => {
    window.__pette = (data) => { setD(data); setLive(true); };
    return () => { window.__pette = undefined; };
  }, []);

  const st = STATE_STYLE[d.live?.state] ?? STATE_STYLE.idle;
  const busy = ["listening", "thinking", "pending", "sending"].includes(d.live?.state);
  const o = d.outcomes ?? {};
  const decided = (o.sent ?? 0) + (o.cancelled ?? 0);
  const t = d.live?.now ?? d.updated;

  return (
    <div className="h-full p-2">
      <Card className="relative flex h-full flex-col overflow-hidden rounded-2xl shadow-xl">
        {/* header */}
        <div className="relative overflow-hidden bg-zinc-950 px-4 pb-4 pt-3 text-zinc-50">
          <Spotlight className="-left-10 -top-24" fill="#a78bfa" />
          <div className="relative z-10 flex items-center justify-between">
            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.14em] text-zinc-400">Pette</div>
              <div className="text-base font-semibold">World model</div>
            </div>
            <Button variant="ghost" size="icon" className="text-zinc-400 hover:bg-zinc-800 hover:text-zinc-50"
              onClick={() => (document.title = `close:${Date.now()}`)} aria-label="Close"><X className="h-4 w-4" /></Button>
          </div>
          <p className="relative z-10 mt-1 text-xs text-zinc-400">What Pette has learned from your local history, and how it changes what she does.</p>
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
                    <motion.span key={d.live?.label || st.text} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}>
                      {d.live?.state === "pending" ? `Sending in ${d.live.detail}s` : d.live?.label || st.text}
                    </motion.span>
                  </AnimatePresence>
                </div>
              </MovingBorder>
              {d.live?.paused && <Badge variant="secondary">paused</Badge>}
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
              <Stat icon={d.live?.camera ? <Camera className="h-3.5 w-3.5" /> : <CameraOff className="h-3.5 w-3.5" />} label="Camera" value={d.live?.camera ? "on" : "off"} />
              <Stat icon={<Hand className="h-3.5 w-3.5" />} label="Hand" value={d.live?.gesture || "—"} />
              <Stat icon={d.live?.eyes_closed ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />} label="You"
                value={!d.live?.present ? "away" : d.live?.eyes_closed ? "eyes closed" : "here"} />
            </div>
          </BackgroundGradient>

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

          {/* sending */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Send className="h-4 w-4" /> Sending</CardTitle>
              <CardDescription>{decided ? `${Math.round((d.send_rate ?? 0) * 100)}% of decided messages sent` : "No decided messages yet"}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Progress value={(d.send_rate ?? 0) * 100} indicator="bg-emerald-500" />
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="success">{o.sent ?? 0} sent</Badge>
                <Badge variant="secondary">{o.cancelled ?? 0} cancelled</Badge>
                <Badge variant="destructive">{o.failed ?? 0} failed</Badge>
                <Badge variant="outline">{o.drafted ?? 0} drafted</Badge>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground"><Timer className="h-3.5 w-3.5" /> Countdown before dispatch: <span className="font-semibold text-foreground">{d.send_delay_s}s</span></div>
            </CardContent>
          </Card>

          {/* contacts */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><UserRound className="h-4 w-4" /> People</CardTitle>
              <CardDescription>Usual platform per person</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {d.contacts?.length ? d.contacts.map((c) => {
                const total = Math.max(1, c.sent + c.drafted + c.cancelled);
                return (
                  <div key={c.name} className="space-y-1">
                    <div className="flex items-center justify-between text-[13px]">
                      <span className="font-medium">{c.name}</span>
                      <Badge variant="outline" className="capitalize"><PlatformIcon p={c.platform} />{c.platform || "?"}</Badge>
                    </div>
                    <div className="flex h-1.5 overflow-hidden rounded-full bg-secondary">
                      <div className="bg-emerald-500" style={{ width: `${(c.sent / total) * 100}%` }} />
                      <div className="bg-zinc-400" style={{ width: `${(c.drafted / total) * 100}%` }} />
                      <div className="bg-rose-400" style={{ width: `${(c.cancelled / total) * 100}%` }} />
                    </div>
                  </div>
                );
              }) : <p className="text-xs text-muted-foreground">Send a message and Pette starts learning who you talk to where.</p>}
            </CardContent>
          </Card>

          {/* presence + calibration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Eye className="h-4 w-4" /> Last 24 hours</CardTitle>
              <CardDescription>Camera signals (event names only, no frames stored)</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2 text-[11px]">
              <Stat label="Away" value={`${d.presence?.away_episodes ?? 0}×${d.presence?.median_away_s ? ` · ~${d.presence.median_away_s}s` : ""}`} />
              <Stat label="Eyes closed" value={`${d.presence?.eyes_closed ?? 0}×`} />
              <Stat label="Pinches" value={`${d.presence?.pinches ?? 0}`} />
              <Stat label="Blink threshold" value={`${d.blink_threshold}${d.blink_calibrated ? " ✓" : ""}`} />
            </CardContent>
          </Card>

          {/* recent */}
          <Card>
            <CardHeader>
              <CardTitle>Recent</CardTitle>
              <CardDescription>Who and where, never what</CardDescription>
            </CardHeader>
            <CardContent className="space-y-1.5">
              {d.recent?.length ? d.recent.map((m, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1.5"><PlatformIcon p={m.platform} /><span className="font-medium">{m.recipient}</span>
                    <span className="capitalize text-muted-foreground">· {m.platform}</span></span>
                  <span className="flex items-center gap-1.5"><Badge variant={OUTCOME[m.outcome] ?? "outline"}>{m.outcome}</Badge>
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
