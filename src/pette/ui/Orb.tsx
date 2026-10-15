/**
 * Pette's body (port of pet/ui.py's Orb): you (the mirrored webcam preview, drawn here only and never stored),
 * your hand skeleton and gesture, a glowing status ring, and her mood (dozing while you're away or paused,
 * a lid when your eyes close). Click = look now; drag = move her.
 */
import { useEffect, useRef } from "react";
import type { Mood, PetteStatus, PreviewFrame, PreviewPoint } from "../contracts";
import type { PetteController } from "../controller";
import { ORB, statusColor, statusLabel } from "../status";
import { cn } from "@/lib/utils";

export const HAND_EDGES: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

/** The orb's box: ring + glow around the ORB circle, room for the pills. */
export const ORB_BOX = { width: ORB + 28, height: ORB + 40 };
const DRAG_PX = 3;

interface DrawState {
  status: PetteStatus;
  mood: Mood;
  cameraOn: boolean;
  feedVisible: boolean;
  frame: PreviewFrame | null;
  phase: number;
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Paint one orb frame (pure: everything comes from `s`). */
export function drawOrb(ctx: CanvasRenderingContext2D, s: DrawState, w: number, h: number): void {
  ctx.clearRect(0, 0, w, h);
  const dozing = s.mood === "away" || s.mood === "sleeping";
  const [r0, g0, b0] = hexToRgb(statusColor(s.status));
  const cx = w / 2, cy = ORB / 2 + 12, r = ORB / 2;
  ctx.save();
  ctx.globalAlpha = dozing ? 0.6 : 1;
  // glow
  [26, 16, 8].forEach((a, i) => {
    ctx.strokeStyle = `rgba(${r0},${g0},${b0},${a / 255})`;
    ctx.lineWidth = 6 + i * 5;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
    ctx.stroke();
  });
  // rotating conic ring (Aceternity "moving border")
  const start = (s.phase * Math.PI) / 180;
  if (typeof ctx.createConicGradient === "function") {
    const grad = ctx.createConicGradient(start, cx, cy);
    const light = `rgb(${Math.min(255, r0 * 1.5)},${Math.min(255, g0 * 1.5)},${Math.min(255, b0 * 1.5)})`;
    grad.addColorStop(0, `rgb(${r0},${g0},${b0})`);
    grad.addColorStop(0.35, `rgba(${r0},${g0},${b0},0.16)`);
    grad.addColorStop(0.6, light);
    grad.addColorStop(1, `rgb(${r0},${g0},${b0})`);
    ctx.strokeStyle = grad;
  } else {
    ctx.strokeStyle = `rgb(${r0},${g0},${b0})`;
  }
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  // inner: webcam or placeholder
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r - 3, 0, Math.PI * 2);
  ctx.clip();
  const ix = cx - r + 3, iy = cy - r + 3, is = 2 * r - 6;
  const label = (text: string) => {
    ctx.fillStyle = "#71717a";
    ctx.font = "10px 'Inter Variable', 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(text, cx, iy + is - 16);
  };
  if (!s.feedVisible) {
    ctx.fillStyle = "#ede9fe";
    ctx.fillRect(ix, iy, is, is);
    ctx.strokeStyle = "#8b5cf6";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(cx, cy - 6, 20, 11, 0, 0, Math.PI * 2); // a closed eye: camera hidden
    ctx.moveTo(cx - 24, cy + 8);
    ctx.lineTo(cx + 24, cy - 20);
    ctx.stroke();
    label("camera hidden");
  } else if (s.frame?.bitmap) {
    ctx.drawImage(s.frame.bitmap, ix, iy, is, is);
    drawHands(ctx, s.frame.hands, ix, iy, is);
  } else {
    ctx.fillStyle = "#f4f4f5";
    ctx.fillRect(ix, iy, is, is);
    ctx.strokeStyle = "#a1a1aa";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy - 10, 15, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(cx, cy + 32, 28, 22, 0, Math.PI, Math.PI * 2);
    ctx.stroke();
    label(s.cameraOn ? "starting…" : "camera off");
    if (s.frame?.hands.length) drawHands(ctx, s.frame.hands, ix, iy, is);
  }
  if (s.mood === "blink") { // your eyes are shut: a soft lid comes down over the orb
    const lid = ctx.createLinearGradient(0, iy, 0, iy + is);
    lid.addColorStop(0, "rgba(24,24,27,0.78)");
    lid.addColorStop(0.55, "rgba(24,24,27,0.59)");
    lid.addColorStop(0.56, "rgba(24,24,27,0)");
    ctx.fillStyle = lid;
    ctx.fillRect(ix, iy, is, is);
  }
  ctx.restore();
  ctx.restore();
  if (dozing) { // away or paused: Pette dozes
    const off = Math.floor(s.phase / 6) % 14;
    ctx.fillStyle = "#8b5cf6";
    ctx.font = "600 16px 'Inter Variable', 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("z", cx + r - 10, cy - r + 22 - off / 2);
    ctx.fillText("Z", cx + r + 2, cy - r + 10 - off / 2);
  }
}

function drawHands(ctx: CanvasRenderingContext2D, hands: PreviewPoint[][], x: number, y: number, size: number) {
  for (const hand of hands) {
    const pts = hand.map((p) => [x + p.x * size, y + p.y * size] as const);
    ctx.strokeStyle = "rgba(255,255,255,0.86)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (const [a, b] of HAND_EDGES) {
      if (a < pts.length && b < pts.length) {
        ctx.moveTo(...pts[a]);
        ctx.lineTo(...pts[b]);
      }
    }
    ctx.stroke();
    ctx.fillStyle = "#a78bfa";
    for (const [px, py] of pts) {
      ctx.beginPath();
      ctx.arc(px, py, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

export interface OrbProps {
  controller: PetteController;
  status: PetteStatus;
  detail: string;
  mood: Mood;
  cameraOn: boolean;
  feedVisible: boolean;
  gesture: string;
  /** Top-left in the viewport; omitted = laid out by the parent (the floating window). */
  position?: { x: number; y: number };
  onMove?: (p: { x: number; y: number }) => void;
  className?: string;
}

export function Orb({ controller, status, detail, mood, cameraOn, feedVisible, gesture, position, onMove, className }: OrbProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const draw = useRef<DrawState>({ status, mood, cameraOn, feedVisible, frame: null, phase: 0 });
  draw.current = { ...draw.current, status, mood, cameraOn, feedVisible };
  const press = useRef<{ dx: number; dy: number; x0: number; y0: number; moved: boolean } | null>(null);

  useEffect(() => controller.onFrame((f) => { draw.current.frame = f; }), [controller]);

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext?.("2d") ?? null;
    if (!el || !ctx) return; // no canvas (tests)
    const view = el.ownerDocument.defaultView ?? window;
    const dpr = view.devicePixelRatio || 1;
    el.width = ORB_BOX.width * dpr;
    el.height = ORB_BOX.height * dpr;
    ctx.scale(dpr, dpr);
    let raf = 0;
    const loop = () => {
      const s = draw.current;
      const speed = s.status === "looking" || s.status === "indexing" || s.status === "suggesting" ? 6 : 1.2;
      s.phase = (s.phase + speed / 2) % 360;
      drawOrb(ctx, s, ORB_BOX.width, ORB_BOX.height);
      raf = view.requestAnimationFrame(loop);
    };
    raf = view.requestAnimationFrame(loop);
    return () => view.cancelAnimationFrame(raf);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    try {
      (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    } catch {
      // no active pointer (synthetic or already released): clicking still works, dragging just isn't captured
    }
    press.current = { dx: e.clientX - (position?.x ?? 0), dy: e.clientY - (position?.y ?? 0), x0: e.clientX, y0: e.clientY, moved: false };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const p = press.current;
    if (!p || !onMove || !position) return;
    if (Math.abs(e.clientX - p.x0) + Math.abs(e.clientY - p.y0) > DRAG_PX) p.moved = true;
    if (p.moved) onMove({ x: e.clientX - p.dx, y: e.clientY - p.dy });
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const p = press.current;
    press.current = null;
    if (e.button === 0 && p && !p.moved) controller.requestLook();
  };

  const label = statusLabel(status, detail);
  const color = statusColor(status);
  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={`Pette${label ? ` · ${label}` : ""}. Click to look at the screen now; drag to move.`}
      title={"Click: look at my screen now · drag to move\nCamera: processed in this browser, never stored"}
      data-mood={mood}
      data-status={status}
      className={cn("pette-orb relative cursor-pointer select-none touch-none outline-none", position && "fixed z-40", className)}
      style={{ width: ORB_BOX.width, height: ORB_BOX.height, ...(position ? { left: position.x, top: position.y } : {}) }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          controller.requestLook();
        }
      }}
    >
      <canvas ref={canvas} className="pointer-events-none absolute inset-0" style={{ width: ORB_BOX.width, height: ORB_BOX.height }} />
      {status === "looking" && (
        <span aria-hidden className="pointer-events-none absolute left-1/2 h-8 w-8 -translate-x-1/2 -translate-y-1/2 animate-spin rounded-full border-2 border-blue-500/30 border-t-blue-500"
          style={{ top: ORB / 2 + 12 }} />
      )}
      {gesture && feedVisible && (
        <span className="pointer-events-none absolute left-1/2 -translate-x-1/2 rounded-full border border-zinc-200 bg-white/90 px-2 py-0.5 text-[11px] font-semibold text-zinc-950"
          style={{ top: 6 }}>{gesture}</span>
      )}
      {label && (
        <span data-testid="orb-label" className="pointer-events-none absolute left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-semibold text-white"
          style={{ top: ORB + 4, background: color, borderColor: color }}>{label}</span>
      )}
    </div>
  );
}
