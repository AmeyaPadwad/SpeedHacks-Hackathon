/**
 * Forget memories everywhere they live: the item, its embedding and its row in the live vector index
 * (port of soma/memory/deletion.py). Original user files are never touched.
 */
import type { MemoryStore } from "./store";
import type { VectorIndex } from "./vectorIndex";

/** Delete one memory. True if anything was removed. */
export async function forget(store: MemoryStore, index: VectorIndex, itemId: string): Promise<boolean> {
  const removedRow = (await store.deleteItems([itemId])) > 0;
  const removedVec = index.remove(itemId);
  return removedRow || removedVec;
}

/** Forget a folder and every memory from it. Returns the number of memories forgotten. */
export async function forgetFolder(store: MemoryStore, index: VectorIndex, folderId: string): Promise<number> {
  // drop the folder first so retrieval stops permitting its items even while deletion runs
  await store.removeFolder(folderId);
  const ids = (await store.itemsForFolder(folderId)).map((i) => i.id);
  await store.deleteItems(ids);
  ids.forEach((id) => index.remove(id));
  return ids.length;
}

/** Delete memories whose `expiresAt` has passed. Returns the count. */
export async function purgeExpired(store: MemoryStore, index: VectorIndex, now: number = Date.now()): Promise<number> {
  const ids = (await store.allItems())
    .filter((i) => i.expiresAt && !Number.isNaN(Date.parse(i.expiresAt)) && Date.parse(i.expiresAt) <= now)
    .map((i) => i.id);
  await store.deleteItems(ids);
  ids.forEach((id) => index.remove(id));
  return ids.length;
}

/** True when `itemId` is absent from the store, its embeddings and the live index. */
export async function verifyDeleted(store: MemoryStore, index: VectorIndex, itemId: string): Promise<boolean> {
  return (await store.getItem(itemId)) === null && (await store.getEmbedding(itemId)) === null && !index.has(itemId);
}
