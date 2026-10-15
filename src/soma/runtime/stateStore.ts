/**
 * Tiny key/value store for the runtime's own local state (the online preference ranker's arm counts).
 * A separate IndexedDB database ("soma-runtime") from the memory store ("soma"). Facts only: numbers and
 * candidate keys, never screen text, file contents or pixels.
 */
import { openDB } from "idb";

export interface KeyValueStore {
  get<T>(key: string): Promise<T | undefined>;
  set(key: string, value: unknown): Promise<void>;
}

export const RUNTIME_DB = "soma-runtime";

/** IndexedDB-backed store; falls back to memory when IndexedDB is unavailable. */
export async function openStateStore(dbName = RUNTIME_DB): Promise<KeyValueStore> {
  try {
    const db = await openDB(dbName, 1, {
      upgrade(d) {
        if (!d.objectStoreNames.contains("state")) d.createObjectStore("state");
      },
    });
    return {
      get: async <T>(key: string) => (await db.get("state", key)) as T | undefined,
      set: async (key, value) => {
        await db.put("state", value, key);
      },
    };
  } catch {
    return memoryStateStore();
  }
}

export function memoryStateStore(): KeyValueStore {
  const m = new Map<string, unknown>();
  return {
    get: async <T>(key: string) => m.get(key) as T | undefined,
    set: async (key, value) => {
      m.set(key, structuredClone(value));
    },
  };
}
