/**
 * Pette's side of the dashboard link: every second (and on "hello") it gathers the live state, the world
 * model snapshot, SOMA's status and diagnostics, and posts them on the channel. Pinch-and-drag scroll is
 * forwarded so the gesture scrolls the dashboard.
 */
import { useEffect, useRef } from "react";
import { hasDiagnostics, type SomaService } from "@/soma/service";
import type { PetteController } from "../controller";
import type { Pette } from "../usePette";
import { PUBLISH_MS, openChannel, type DashboardSnapshot, type PetteLive } from "./channel";

export async function buildSnapshot(pette: Pette, live: PetteLive, sandbox: boolean, now = Date.now()): Promise<DashboardSnapshot> {
  const soma: SomaService = pette.soma;
  const [status, diagnostics] = await Promise.all([
    soma.status().catch(() => null),
    hasDiagnostics(soma) ? soma.diagnostics().catch(() => null) : Promise.resolve(null),
  ]);
  return { at: now, pette: live, world: pette.world.snapshot(), folders: [...pette.suggester.folders], soma: status, diagnostics, sandbox };
}

/** Publish while mounted. `live` is read through a ref so the interval never restarts on re-render. */
export function useDashboardPublisher(pette: Pette, live: PetteLive, sandbox: boolean, controller: PetteController): void {
  const liveRef = useRef(live);
  liveRef.current = live;
  useEffect(() => {
    let busy = false;
    const ch = openChannel((m) => {
      if (m.type === "hello") void publish();
    });
    async function publish() {
      if (busy) return;
      busy = true;
      try {
        ch.post({ type: "snapshot", data: await buildSnapshot(pette, liveRef.current, sandbox) });
      } catch {
        // the dashboard is a debug aid: never let it disturb Pette
      } finally {
        busy = false;
      }
    }
    const t = setInterval(() => void publish(), PUBLISH_MS);
    const off = controller.events.on("scroll", (dy) => ch.post({ type: "scroll", dy }));
    void publish();
    return () => {
      clearInterval(t);
      off();
      ch.close();
    };
  }, [pette, sandbox, controller]);
}
