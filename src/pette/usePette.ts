/**
 * Wiring (port of main.py's `build()`): one controller, world model and hub around a `SomaService` and a fact
 * log, plus the React hooks that read them. `tools/sandbox.py`'s rule carries over: the `?sandbox` mode uses
 * this same wiring around `FakeSoma`, so the sandbox stays the real app.
 */
import { useEffect, useState, useSyncExternalStore } from "react";
import type { FolderInfo, SomaService } from "@/soma/service";
import { PetteController, type PetteState } from "./controller";
import { Suggester, type SuggesterOptions } from "./suggester";
import type { FactStore } from "./world/store";
import { WorldModel, type WorldSnapshot } from "./world/worldModel";
import { ORB_BOX } from "./ui/Orb";

export interface Pette {
  soma: SomaService;
  store: FactStore;
  controller: PetteController;
  world: WorldModel;
  suggester: Suggester;
}

export function createPette(soma: SomaService, store: FactStore, opts: SuggesterOptions = {}): Pette {
  const controller = new PetteController();
  const world = new WorldModel(store);
  const suggester = new Suggester(controller, soma, { memory: store, world, ...opts });
  return { soma, store, controller, world, suggester };
}

export function usePetteState(c: PetteController): PetteState {
  return useSyncExternalStore(c.subscribe, c.getState);
}

/** The hub's switch and SOMA's folders. */
export function useSuggester(s: Suggester): { enabled: boolean; folders: FolderInfo[] } {
  const [enabled, setEnabled] = useState(s.enabled);
  const [folders, setFolders] = useState(s.folders);
  useEffect(() => {
    const offs = [s.events.on("enabledChanged", setEnabled), s.events.on("foldersChanged", (f) => setFolders([...f]))];
    return () => offs.forEach((off) => off());
  }, [s]);
  return { enabled, folders };
}

/** World panel refresh: on every new fact, and every 1.5 s for the "ago" times (pet/worldview.py's REFRESH_MS). */
export const REFRESH_MS = 1500;

export function useWorldSnapshot(world: WorldModel, store: FactStore, active = true): WorldSnapshot {
  const [snap, setSnap] = useState(() => world.snapshot());
  useEffect(() => {
    if (!active) return;
    const refresh = () => {
      try {
        setSnap(world.snapshot());
      } catch {
        // a storage hiccup must never take Pette down
      }
    };
    refresh();
    const off = store.subscribe(refresh);
    const t = setInterval(refresh, REFRESH_MS);
    return () => {
      off();
      clearInterval(t);
    };
  }, [world, store, active]);
  return snap;
}

// --- where Pette sits --------------------------------------------------------
export const ORB_POS_KEY = "pette.orb.position";
const MARGIN = 24;

export type Point = { x: number; y: number };

/** Keep the orb inside the viewport. */
export function clampOrb(p: Point, view = { width: window.innerWidth, height: window.innerHeight }): Point {
  return {
    x: Math.max(0, Math.min(p.x, view.width - ORB_BOX.width)),
    y: Math.max(0, Math.min(p.y, view.height - ORB_BOX.height)),
  };
}

const defaultOrb = (): Point => ({ x: window.innerWidth - ORB_BOX.width - MARGIN, y: window.innerHeight - ORB_BOX.height - MARGIN });

function readOrb(): Point | null {
  try {
    const v = JSON.parse(localStorage.getItem(ORB_POS_KEY) ?? "null");
    return v && Number.isFinite(v.x) && Number.isFinite(v.y) ? { x: v.x, y: v.y } : null;
  } catch {
    return null;
  }
}

/** The orb's position, remembered per browser (bottom-right by default), kept on screen when the window resizes. */
export function useOrbPosition(): [Point, (p: Point) => void] {
  const [pos, setPos] = useState<Point>(() => clampOrb(readOrb() ?? defaultOrb()));
  useEffect(() => {
    const onResize = () => setPos((p) => clampOrb(p));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const move = (p: Point) => {
    const c = clampOrb(p);
    setPos(c);
    try {
      localStorage.setItem(ORB_POS_KEY, JSON.stringify(c));
    } catch {
      // storage blocked: Pette just won't remember where she was
    }
  };
  return [pos, move];
}

// --- where the world panel docks (pet/worldview.py's dock) ---------------------
export const PANEL = { width: 380, height: 600, gap: 10 };

/** Left of Pette, bottom-aligned; flips to the right if there's no room; always on screen. */
export function dockPanel(orb: Point, view = { width: window.innerWidth, height: window.innerHeight }) {
  const width = Math.min(PANEL.width, view.width - 16);
  const height = Math.min(PANEL.height, view.height - 16);
  let x = orb.x - width - PANEL.gap;
  if (x < 8) x = Math.min(orb.x + ORB_BOX.width + PANEL.gap, view.width - width - 8);
  x = Math.max(8, x);
  const y = Math.max(8, Math.min(orb.y + ORB_BOX.height - height, view.height - height - 8));
  return { left: x, top: y, width, height };
}
