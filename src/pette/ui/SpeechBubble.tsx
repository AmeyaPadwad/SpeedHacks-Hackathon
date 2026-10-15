/**
 * SOMA's speech bubble (port of pet/ui.py's Bubble + BubbleBar): frosted glass, docked bottom-centre, a tail
 * pointing down, title + text, and on a card the Open/Copy and "Not now" buttons over a draining 15 s bar.
 * It slides left rather than cover Pette. Point left / right (or the arrows) page through recent suggestions
 * as read-only history ("2 / 5").
 *
 * Suggestion text is untrusted: rendered as React text only (never HTML) and never turned into links.
 */
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useLayoutEffect, useRef, useState } from "react";
import type { BubbleView, PetteController } from "../controller";
import { GLASS } from "../status";
import { cn } from "@/lib/utils";

export const BUBBLE_WIDTH = 460;
/** Gap between the bubble and the bottom of the viewport. */
export const BAR_MARGIN = 18;
export const TAIL = 10;
const GUTTER = 16;

type Rect = { x: number; y: number; width: number; height: number };

const overlaps = (a: Rect, b: Rect) =>
  a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;

/**
 * Where the bubble goes: bottom-centre of the viewport; if that would cover Pette, slide it left of her
 * (never past the left gutter). Pure, for tests.
 */
export function dockBubble(view: { width: number; height: number }, size: { width: number; height: number }, orb?: Rect | null): { left: number; top: number } {
  const width = Math.min(size.width, view.width - 2 * GUTTER);
  const g: Rect = { x: (view.width - width) / 2, y: view.height - size.height - BAR_MARGIN, width, height: size.height };
  if (orb && overlaps(g, orb)) g.x = Math.max(GUTTER, orb.x - width - 8);
  return { left: Math.round(g.x), top: Math.round(g.y) };
}

export interface SpeechBubbleProps {
  controller: PetteController;
  bubble: BubbleView | null;
  progress: number | null;
  /** Pette's box, so the bubble can avoid it. Omitted in the floating window (laid out by the parent). */
  orb?: Rect | null;
  /** Docked = fixed to the viewport; false = flows inside its parent (the floating window). */
  docked?: boolean;
}

export function SpeechBubble({ controller, bubble, progress, orb, docked = true }: SpeechBubbleProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(120);
  const [view, setView] = useState(() => ({ width: window.innerWidth, height: window.innerHeight }));

  useLayoutEffect(() => {
    const el = ref.current;
    if (el && el.offsetHeight && el.offsetHeight !== height) setHeight(el.offsetHeight);
  });
  useLayoutEffect(() => {
    const on = () => setView({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);

  if (!bubble) return null;
  const pos = docked ? dockBubble(view, { width: BUBBLE_WIDTH, height }, orb) : null;
  const isCard = bubble.kind === "card";
  const glowing = isCard && bubble.live && bubble.proposal.glow;
  const title = bubble.title;
  const text = bubble.kind === "card" ? bubble.proposal.text : bubble.text;

  return (
    <motion.div
      ref={ref}
      role={isCard ? "alertdialog" : "status"}
      aria-live="polite"
      aria-label={title || "Pette"}
      data-testid="speech-bubble"
      data-kind={bubble.kind}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={cn("pette-bubble z-50", docked ? "fixed" : "relative mx-auto")}
      style={{ width: docked ? Math.min(BUBBLE_WIDTH, view.width - 32) : "100%", maxWidth: BUBBLE_WIDTH,
        ...(pos ? { left: pos.left, top: pos.top } : {}), paddingBottom: TAIL }}
    >
      <div className="relative rounded-[14px] p-[2px]" style={{ filter: "drop-shadow(0 8px 16px rgba(0,0,0,0.15))" }}>
        {/* moving gradient border while a card counts down */}
        {glowing && (
          <div className="absolute inset-0 overflow-hidden rounded-[14px]" aria-hidden>
            <motion.div className="absolute inset-[-150%]"
              style={{ background: "conic-gradient(from 0deg, #f59e0b, #ec4899, #8b5cf6, #3b82f6, #f59e0b)" }}
              animate={{ rotate: 360 }} transition={{ duration: 3, repeat: Infinity, ease: "linear" }} />
          </div>
        )}
        <div
          className="relative overflow-hidden rounded-[13px] px-4 pb-3.5 pt-3 backdrop-blur-xl backdrop-saturate-150"
          style={{
            background: `linear-gradient(to bottom, rgba(255,255,255,${GLASS.top}), rgba(250,250,252,${GLASS.bottom}))`,
            boxShadow: glowing ? undefined : `0 0 0 1px rgba(0,0,0,${GLASS.hairline}), inset 0 0 0 1px rgba(255,255,255,${GLASS.edge})`,
          }}
        >
          <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 h-12"
            style={{ background: `linear-gradient(to bottom, rgba(255,255,255,${GLASS.sheen}), rgba(255,255,255,0))` }} />
          <div className="relative flex items-center gap-2">
            {title && <div className="min-w-0 flex-1 truncate text-xs font-semibold tracking-[0.2px] text-zinc-700">{title}</div>}
            {bubble.kind === "history" && (
              <div className="ml-auto flex items-center gap-0.5 text-[11px] font-medium text-zinc-500">
                <button type="button" aria-label="Earlier suggestion" className="rounded p-0.5 hover:bg-white/60" onClick={() => controller.page(-1)}>
                  <ChevronLeft className="h-3.5 w-3.5" />
                </button>
                <span data-testid="history-index">{bubble.index} / {bubble.total}</span>
                <button type="button" aria-label="Later suggestion" className="rounded p-0.5 hover:bg-white/60" onClick={() => controller.page(1)}>
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </div>
            )}
          </div>
          <p className="relative mt-1.5 select-text whitespace-pre-wrap break-words text-sm text-zinc-950">{text}</p>
          {isCard && (
            <div className="relative mt-2.5 flex justify-end gap-2">
              <button type="button" disabled={!bubble.live} onClick={() => controller.cancel()}
                className="rounded-lg border border-white/70 bg-white/40 px-3 py-1.5 text-[13px] font-medium text-zinc-950 hover:bg-white/60 disabled:bg-zinc-100/40 disabled:text-zinc-400">
                {bubble.proposal.cancelLabel}
              </button>
              <button type="button" disabled={!bubble.live} onClick={() => controller.approve()}
                className="rounded-lg border border-zinc-900/90 bg-zinc-900/85 px-3 py-1.5 text-[13px] font-medium text-zinc-50 hover:bg-zinc-800/90 disabled:border-white/50 disabled:bg-zinc-100/40 disabled:text-zinc-400">
                {bubble.proposal.actionLabel}
              </button>
            </div>
          )}
          {isCard && progress !== null && (
            <div className="relative mt-2.5 h-[3px] w-full overflow-hidden rounded-full bg-white/50" role="progressbar"
              aria-label="Time left" aria-valuenow={Math.round(progress * 100)}>
              <div className="h-full rounded-full bg-zinc-900 transition-[width] duration-200 ease-linear" style={{ width: `${progress * 100}%` }} />
            </div>
          )}
        </div>
      </div>
      {/* tail pointing down, slightly left of centre */}
      <div aria-hidden className="absolute bottom-[3px] h-3.5 w-3.5 rotate-45 border-b border-r"
        style={{ left: "calc(50% - 25px)", background: `rgba(250,250,252,${GLASS.bottom})`, borderColor: `rgba(0,0,0,${GLASS.hairline})`,
          backdropFilter: "blur(24px)" }} />
    </motion.div>
  );
}
