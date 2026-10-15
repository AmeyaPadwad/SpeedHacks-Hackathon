/**
 * Loads Pette's webcam (`src/pette/vision`: `CameraWorker`, `Watcher`) lazily: tfjs and the hand / face
 * models download only when the camera is first turned on.
 */
import type { WatcherTarget } from "./contracts";
import type { Watcher, WatcherOptions } from "./vision";

export type { Watcher, WatcherOptions };

/** A `Watcher` driving `target` (Pette's controller), or null when the module can't load. */
export async function loadWatcher(target: WatcherTarget, options: WatcherOptions): Promise<Watcher | null> {
  try {
    const mod = await import("./vision");
    return new mod.Watcher(target, options);
  } catch (e) {
    console.error("Pette: the webcam module failed to load", e);
    return null;
  }
}
