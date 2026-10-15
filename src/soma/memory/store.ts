/**
 * IndexedDB memory store (port of soma/memory/sqlite_store.py, via `idb`).
 *
 * Object stores:
 * - `folders`    {id, name, handle, addedAt}: folders the user picked (FileSystemDirectoryHandle).
 * - `items`      MemoryItem + `folderId`: a short summary (capped at `SUMMARY_CAP`) and metadata.
 * - `embeddings` {id, vec}: one unit vector per item.
 *
 * Raw file contents are never stored: files stay in the user's folder and are re-read through the
 * handle when the user opens one. `sourceUri` is `folderId/relative/path`.
 */
import { openDB, type DBSchema, type IDBPDatabase } from "idb";
import { newId, type MemoryItem } from "../types";
import { VectorIndex } from "./vectorIndex";

/** Maximum characters of `textSummary` ever written. */
export const SUMMARY_CAP = 500;
export const DB_VERSION = 1;

export interface FolderRecord {
  id: string;
  name: string;
  handle: FileSystemDirectoryHandle;
  addedAt: string;
}

type ItemRecord = MemoryItem & { folderId: string | null };

interface SomaMemoryDB extends DBSchema {
  folders: { key: string; value: FolderRecord };
  items: { key: string; value: ItemRecord; indexes: { folderId: string } };
  embeddings: { key: string; value: { id: string; vec: Float32Array } };
}

export const utcNowIso = (): string => new Date().toISOString();

/** Trim a summary to `cap` characters (with an ellipsis marker when cut). */
export const capSummary = (text: string | null | undefined, cap = SUMMARY_CAP): string => {
  const t = text ?? "";
  return t.length <= cap ? t : `${t.slice(0, cap - 1)}…`;
};

/** The folder id of a `folderId/relative/path` source URI, or null. */
export function folderOf(sourceUri: string | null | undefined): string | null {
  if (!sourceUri) return null;
  const i = sourceUri.indexOf("/");
  return i > 0 ? sourceUri.slice(0, i) : null;
}

function toItem(r: ItemRecord): MemoryItem {
  const { folderId: _drop, ...item } = r;
  void _drop;
  return item;
}

export class MemoryStore {
  /** Live handles by folder id: keeps permission state and identity within a session. */
  private readonly handles = new Map<string, FileSystemDirectoryHandle>();

  private constructor(readonly db: IDBPDatabase<SomaMemoryDB>) {}

  static async open(dbName = "soma"): Promise<MemoryStore> {
    const db = await openDB<SomaMemoryDB>(dbName, DB_VERSION, {
      upgrade(d) {
        if (!d.objectStoreNames.contains("folders")) d.createObjectStore("folders", { keyPath: "id" });
        if (!d.objectStoreNames.contains("items")) d.createObjectStore("items", { keyPath: "id" }).createIndex("folderId", "folderId");
        if (!d.objectStoreNames.contains("embeddings")) d.createObjectStore("embeddings", { keyPath: "id" });
      },
    });
    return new MemoryStore(db);
  }

  close(): void {
    this.db.close();
  }

  // ------------------------------------------------------------ folders

  /** Remember a picked folder. Picking the same folder again returns the existing record. */
  async addFolder(handle: FileSystemDirectoryHandle, name: string = handle.name): Promise<FolderRecord> {
    for (const f of await this.listFolders()) {
      try {
        if (f.handle && typeof f.handle.isSameEntry === "function" && (await f.handle.isSameEntry(handle))) {
          this.handles.set(f.id, handle);
          return { ...f, handle };
        }
      } catch {
        // a stale handle never matches
      }
    }
    const rec: FolderRecord = { id: newId("fld_"), name, handle, addedAt: utcNowIso() };
    await this.db.put("folders", rec);
    this.handles.set(rec.id, handle);
    return rec;
  }

  private withLiveHandle(f: FolderRecord): FolderRecord {
    const live = this.handles.get(f.id);
    return live ? { ...f, handle: live } : f;
  }

  async listFolders(): Promise<FolderRecord[]> {
    const all = await this.db.getAll("folders");
    return all.sort((a, b) => a.addedAt.localeCompare(b.addedAt)).map((f) => this.withLiveHandle(f));
  }

  async getFolder(id: string): Promise<FolderRecord | null> {
    const f = await this.db.get("folders", id);
    return f ? this.withLiveHandle(f) : null;
  }

  /** Forget the folder record only (see `deletion.forgetFolder` for its memories). */
  async removeFolder(id: string): Promise<boolean> {
    const had = (await this.db.getKey("folders", id)) !== undefined;
    await this.db.delete("folders", id);
    this.handles.delete(id);
    return had;
  }

  // ------------------------------------------------------------ items

  /**
   * Insert or replace an item (and its embedding when given). Fills timestamps like the Python store:
   * `createdAt` is kept from the prior record, `updatedAt` is now on replace. The summary is capped.
   */
  async putItem(item: MemoryItem, vec?: Float32Array | null): Promise<MemoryItem> {
    const now = utcNowIso();
    const tx = this.db.transaction(["items", "embeddings"], "readwrite");
    const items = tx.objectStore("items");
    const existing = await items.get(item.id);
    const stored: MemoryItem = {
      ...item,
      tags: [...item.tags],
      provenance: { ...item.provenance },
      createdAt: item.createdAt || existing?.createdAt || now,
      updatedAt: existing || !item.updatedAt ? now : item.updatedAt,
      textSummary: capSummary(item.textSummary),
    };
    await items.put({ ...stored, folderId: folderOf(stored.sourceUri) });
    if (vec) await tx.objectStore("embeddings").put({ id: item.id, vec: Float32Array.from(vec) });
    await tx.done;
    return stored;
  }

  async getItem(id: string): Promise<MemoryItem | null> {
    const r = await this.db.get("items", id);
    return r ? toItem(r) : null;
  }

  async allItems(): Promise<MemoryItem[]> {
    return (await this.db.getAll("items")).map(toItem);
  }

  async itemsForFolder(folderId: string): Promise<MemoryItem[]> {
    return (await this.db.getAllFromIndex("items", "folderId", folderId)).map(toItem);
  }

  async getEmbedding(id: string): Promise<Float32Array | null> {
    return (await this.db.get("embeddings", id))?.vec ?? null;
  }

  /** Delete items and their embeddings. Returns how many items existed. */
  async deleteItems(ids: Iterable<string>): Promise<number> {
    const tx = this.db.transaction(["items", "embeddings"], "readwrite");
    let n = 0;
    for (const id of ids) {
      if ((await tx.objectStore("items").getKey(id)) !== undefined) n += 1;
      await tx.objectStore("items").delete(id);
      await tx.objectStore("embeddings").delete(id);
    }
    await tx.done;
    return n;
  }

  countItems(): Promise<number> {
    return this.db.count("items");
  }

  /** Item counts per folder id (folders with no items are absent). */
  async countByFolder(): Promise<Record<string, number>> {
    const out: Record<string, number> = {};
    let cursor = await this.db.transaction("items").store.index("folderId").openKeyCursor();
    while (cursor) {
      const k = String(cursor.key);
      out[k] = (out[k] ?? 0) + 1;
      cursor = await cursor.continue();
    }
    return out;
  }

  /** Build a cosine index from every stored embedding (rows of the wrong size are skipped). */
  async loadIndex(dim = 256): Promise<VectorIndex> {
    const idx = new VectorIndex(dim);
    for (const { id, vec } of await this.db.getAll("embeddings")) if (vec?.length === dim) idx.add(id, vec);
    return idx;
  }
}
