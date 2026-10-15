/**
 * A tiny typed event emitter (stands in for Qt signals). Listeners run synchronously, in order; a throwing
 * listener never stops the others.
 */
export class Emitter<Events extends { [K in keyof Events]: unknown[] }> {
  private listeners: { [K in keyof Events]?: Set<(...args: Events[K]) => void> } = {};

  /** Subscribe; returns the unsubscribe function. */
  on<K extends keyof Events>(event: K, fn: (...args: Events[K]) => void): () => void {
    (this.listeners[event] ??= new Set()).add(fn);
    return () => this.listeners[event]?.delete(fn);
  }

  emit<K extends keyof Events>(event: K, ...args: Events[K]): void {
    for (const fn of [...(this.listeners[event] ?? [])]) {
      try {
        fn(...args);
      } catch (e) {
        console.error(`Pette: a "${String(event)}" listener failed`, e);
      }
    }
  }
}
