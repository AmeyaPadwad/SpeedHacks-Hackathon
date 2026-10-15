import { describe, expect, it, vi } from "vitest";
import { DEFAULT_CONFIG, type MemoryConfig } from "../config";
import { SourceType, newId, type MemoryItem } from "../types";
import { authorizedFile, isAuthorizedRef, matchScreenPath, parseRef, requestPermission, resolveFile } from "./access";
import { forget, forgetFolder, purgeExpired, verifyDeleted } from "./deletion";
import { TextEmbedder, fitDim } from "./embedder";
import { FakeDir, FakeFile } from "./fakeFs";
import { codeSymbols, indexFolder, itemIdFor, redactForMemory } from "./ingestion";
import { MemoryRetriever, conflictKey, memoriesToArrays } from "./retrieval";
import { MemoryStore, SUMMARY_CAP } from "./store";
import { VectorIndex } from "./vectorIndex";

const DIM = 64;
const te = new TextEmbedder(DIM, { vocabSize: 4096 });
const cfg: MemoryConfig = { ...DEFAULT_CONFIG.memory };
const cos = (a: Float32Array, b: Float32Array) => a.reduce((s, v, i) => s + v * b[i], 0);
const openStore = () => MemoryStore.open(newId("test_"));

function item(id: string, over: Partial<MemoryItem> = {}): MemoryItem {
  return {
    id, sourceType: SourceType.PREFERENCE, userApproved: true, textSummary: id, sourceUri: null, tags: [],
    provenance: {}, createdAt: "", updatedAt: "", expiresAt: null, ...over,
  };
}

async function env(tree: Record<string, unknown> = {}) {
  const store = await openStore();
  const index = new VectorIndex(DIM);
  const dir = new FakeDir("proj", tree);
  const folder = await store.addFolder(dir.asHandle);
  const ret = new MemoryRetriever(store, index, te, cfg);
  return { store, index, dir, folder, ret, deps: { store, embedder: te, index, cfg } };
}

// ----------------------------------------------------------------- embedder / index

describe("TextEmbedder", () => {
  it("is deterministic and returns unit vectors", () => {
    const a = te.embed("deploy checklist for the release");
    expect(a).toHaveLength(DIM);
    expect(Math.hypot(...a)).toBeCloseTo(1, 5);
    expect(Array.from(new TextEmbedder(DIM, { vocabSize: 4096 }).embed("deploy checklist for the release"))).toEqual(Array.from(a));
    expect(Math.hypot(...te.embed(""))).toBeCloseTo(1, 5);
  });

  it("keeps lexical overlap as cosine similarity", () => {
    const q = te.embed("deploying the checklist");
    expect(cos(q, te.embed("deploy checklist"))).toBeGreaterThan(cos(q, te.embed("banana bread recipe")));
  });

  it("fitDim pads or truncates and renormalizes", () => {
    const v = fitDim([3, 4], 4);
    expect(Array.from(v)).toEqual([0.6, 0.8, 0, 0].map((x) => Math.fround(x)));
    expect(fitDim(te.embed("x"), 16)).toHaveLength(16);
    expect(Math.hypot(...fitDim(te.embed("x"), 16))).toBeCloseTo(1, 5);
  });
});

describe("VectorIndex", () => {
  it("searches, filters before scoring, removes and round-trips", () => {
    const idx = new VectorIndex(3);
    idx.add("a", [1, 0, 0]);
    idx.add("b", [0, 1, 0]);
    idx.add("c", [1, 1, 0]);
    expect(idx.search([1, 0, 0], 2).map((h) => h.id)).toEqual(["a", "c"]);
    expect(idx.search([1, 0, 0], 5, new Set(["b"])).map((h) => h.id)).toEqual(["b"]);
    expect(idx.search([1, 0, 0], 0)).toEqual([]);
    expect(() => idx.add("x", [1, 2])).toThrow();
    const back = VectorIndex.deserialize(structuredClone(idx.serialize()));
    expect(back.search([1, 0, 0], 3)).toEqual(idx.search([1, 0, 0], 3));
    expect(idx.remove("a")).toBe(true);
    expect(idx.remove("a")).toBe(false);
    expect(idx.size).toBe(2);
  });
});

// ----------------------------------------------------------------- store

describe("MemoryStore", () => {
  it("keeps folders with live handles and dedupes the same folder", async () => {
    const store = await openStore();
    const dir = new FakeDir("proj");
    const f = await store.addFolder(dir.asHandle);
    expect(f).toMatchObject({ name: "proj" });
    expect((await store.addFolder(dir.asHandle)).id).toBe(f.id);
    expect((await store.listFolders()).map((x) => x.id)).toEqual([f.id]);
    expect((await store.getFolder(f.id))?.handle).toBe(dir);
    expect(await store.removeFolder(f.id)).toBe(true);
    expect(await store.getFolder(f.id)).toBeNull();
  });

  it("round-trips items, caps summaries, fills timestamps, counts by folder", async () => {
    const store = await openStore();
    const vec = te.embed("hello");
    const saved = await store.putItem(item("m1", { sourceType: SourceType.DOCUMENT, sourceUri: "fld_a/x.md", textSummary: "x".repeat(5000), tags: ["md"] }), vec);
    expect(saved.textSummary.length).toBeLessThanOrEqual(SUMMARY_CAP);
    expect(saved.createdAt).toBeTruthy();
    await store.putItem(item("m2", { sourceUri: "fld_a/y.md" }), vec);
    await store.putItem(item("p1"), vec);
    const got = await store.getItem("m1");
    expect(got).toMatchObject({ id: "m1", tags: ["md"], sourceUri: "fld_a/x.md" });
    expect(got).not.toHaveProperty("folderId");
    expect(await store.countItems()).toBe(3);
    expect(await store.countByFolder()).toEqual({ fld_a: 2 });
    expect((await store.itemsForFolder("fld_a")).map((i) => i.id).sort()).toEqual(["m1", "m2"]);
    const idx = await store.loadIndex(DIM);
    expect(idx.size).toBe(3);
    expect(await store.deleteItems(["m1", "nope"])).toBe(1);
    expect(await store.getEmbedding("m1")).toBeNull();
    expect((await store.loadIndex(DIM)).size).toBe(2);
  });
});

// ----------------------------------------------------------------- ingestion

describe("indexFolder", () => {
  it("follows the walk rules", async () => {
    const big = new FakeFile("big.txt", "x", 1, cfg.maxFileBytes + 1);
    const { folder, dir, deps, store } = await env({
      "notes.md": "# Deploy\nThe deploy checklist.",
      "app.py": "def main():\n    pass\n",
      ".env": "SECRET=1",
      "binary.exe": "MZ",
      "big.txt": big,
      ".git": { "config": "x" },
      node_modules: { "lib.js": "x" },
      src: { "util.ts": "export function add(a: number, b: number) { return a + b }" },
    });
    const progress = vi.fn();
    const r = await indexFolder(dir.asHandle, folder.id, deps, progress);
    expect(r).toMatchObject({ files: 3, added: 3, capped: false });
    expect(r.skipped).toBe(3); // .env, binary.exe, big.txt
    expect(progress).toHaveBeenCalled();
    const uris = (await store.itemsForFolder(folder.id)).map((i) => i.sourceUri).sort();
    expect(uris).toEqual([`${folder.id}/app.py`, `${folder.id}/notes.md`, `${folder.id}/src/util.ts`]);
  });

  it("stores only a short redacted summary with code symbols", async () => {
    const secret = "sk-proj-abcdefghijklmnopqrstuvwx1234";
    const body = `def connect():\n    key = "${secret}"\n\nclass Client:\n    pass\n` + "filler text ".repeat(2000);
    const { folder, dir, deps, store } = await env({ "db.py": body });
    await indexFolder(dir.asHandle, folder.id, deps);
    const it = (await store.itemsForFolder(folder.id))[0];
    expect(it.textSummary.length).toBeLessThanOrEqual(SUMMARY_CAP);
    expect(it.textSummary.startsWith("db.py [connect, Client]: ")).toBe(true);
    expect(it.textSummary).not.toContain(secret);
    expect(JSON.stringify(it)).not.toContain("filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text filler text");
    expect(it).toMatchObject({ sourceType: SourceType.DOCUMENT, userApproved: true, tags: ["py"] });
  });

  it("re-indexing is a no-op until a file changes; vanished files are forgotten", async () => {
    const { folder, dir, deps, store, index } = await env({ "a.md": "alpha", "b.md": "beta" });
    await indexFolder(dir.asHandle, folder.id, deps);
    expect(await indexFolder(dir.asHandle, folder.id, deps)).toMatchObject({ unchanged: 2, added: 0, updated: 0 });
    const a = dir.children.get("a.md") as FakeFile;
    a.content = "alpha two";
    a.lastModified += 1000;
    dir.children.delete("b.md");
    expect(await indexFolder(dir.asHandle, folder.id, deps)).toMatchObject({ updated: 1, removed: 1, files: 1 });
    expect(await store.countItems()).toBe(1);
    expect(index.has(itemIdFor(`${folder.id}/b.md`))).toBe(false);
  });

  it("summarizes images by name and size and honours the per-folder cap", async () => {
    const { folder, dir, deps, store } = await env({ "shot.png": "\u0089PNG", "z.md": "z", "y.md": "y" });
    const r = await indexFolder(dir.asHandle, folder.id, { ...deps, cfg: { ...cfg, maxFilesPerFolder: 2 }, imageSize: async () => ({ width: 640, height: 480 }) });
    expect(r).toMatchObject({ files: 2, capped: true });
    const img = await store.getItem(itemIdFor(`${folder.id}/shot.png`));
    expect(img).toMatchObject({ sourceType: SourceType.IMAGE, textSummary: "Image shot.png (640x480, 1 KB)" });
  });

  it("helpers", () => {
    expect(codeSymbols("function a() {}\nexport const b = (x) => x\ninterface C {}\n# Title")).toEqual(["a", "b", "C", "Title"]);
    expect(redactForMemory("password: hunter2 and 4111 1111 1111 1111")).not.toMatch(/hunter2|4111/);
    expect(itemIdFor("f/a")).toBe(itemIdFor("f/a"));
    expect(itemIdFor("f/a")).not.toBe(itemIdFor("f/b"));
  });
});

// ----------------------------------------------------------------- retrieval

describe("MemoryRetriever", () => {
  it("ranks the relevant note first and returns embeddings", async () => {
    const { folder, dir, deps, ret } = await env({ "deploy.md": "deploy checklist: bump version, tag release", "bread.md": "banana bread recipe with walnuts" });
    await indexFolder(dir.asHandle, folder.id, deps);
    const res = await ret.retrieve("how do I deploy the release", 2);
    expect(res[0].item.sourceUri).toBe(`${folder.id}/deploy.md`);
    expect(res[0].embedding).toHaveLength(DIM);
    expect(res[0].score).toBeGreaterThan(res[1].score);
    const packed = memoriesToArrays(res, 4, DIM);
    expect(Array.from(packed.mask)).toEqual([0, 0, 1, 1]);
  });

  it("empty memory returns nothing", async () => {
    const { ret } = await env();
    expect(await ret.retrieve("anything")).toEqual([]);
  });

  it("recency breaks ties and the newest conflicting preference wins", async () => {
    const { store, index, ret } = await env();
    const now = Date.now();
    const iso = (msAgo: number) => new Date(now - msAgo).toISOString();
    const put = async (id: string, text: string, ago: number, tags: string[] = []) => {
      await store.putItem(item(id, { textSummary: text, tags, updatedAt: iso(ago) }), te.embed(text));
      index.add(id, te.embed(text));
    };
    await put("old", "use black formatter", 365 * 86_400_000);
    await put("new", "use black formatter", 3_600_000);
    expect((await ret.retrieve("formatter", 2, now)).map((r) => r.item.id)).toEqual(["new", "old"]);
    await put("tabs", "indent python with tabs", 10 * 86_400_000, ["conflict:indent"]);
    await put("spaces", "use four spaces for indentation", 86_400_000, ["conflict:indent"]);
    const ids = (await ret.retrieve("indent python with tabs", 5, now)).map((r) => r.item.id);
    expect(ids).toContain("spaces");
    expect(ids).not.toContain("tabs");
    expect(conflictKey((await store.getItem("spaces"))!)).toBe("indent");
  });

  it("filters unapproved, expired and removed-folder items before scoring", async () => {
    const { store, index, folder, ret } = await env();
    const add = async (id: string, over: Partial<MemoryItem>) => {
      await store.putItem(item(id, { textSummary: "deploy notes", ...over }), te.embed("deploy notes"));
      index.add(id, te.embed("deploy notes"));
    };
    await add("ok", { sourceType: SourceType.DOCUMENT, sourceUri: `${folder.id}/ok.md` });
    await add("unapproved", { userApproved: false });
    await add("expired", { expiresAt: new Date(Date.now() - 1000).toISOString() });
    await add("foreign", { sourceType: SourceType.DOCUMENT, sourceUri: "fld_gone/x.md" });
    await add("nouri", { sourceType: SourceType.DOCUMENT, sourceUri: null });
    const spy = vi.spyOn(index, "search");
    const res = await ret.retrieve("deploy notes", 10);
    expect(res.map((r) => r.item.id)).toEqual(["ok"]);
    expect([...(spy.mock.calls[0][2] as Set<string>)]).toEqual(["ok"]);
    expect(ret.permissionViolations).toBe(0);
    const noFeedback = new MemoryRetriever(store, index, te, cfg, { allowFeedback: () => false });
    await add("fb", { sourceType: SourceType.FEEDBACK });
    expect((await noFeedback.retrieve("deploy notes", 10)).map((r) => r.item.id)).not.toContain("fb");
  });
});

// ----------------------------------------------------------------- access

describe("access", () => {
  it("parses only safe references", () => {
    expect(parseRef("fld_1/src/a.py")).toEqual({ folderId: "fld_1", parts: ["src", "a.py"] });
    for (const bad of ["", "fld_1", "fld_1/", "/etc/passwd", "C:/x", "fld_1/../x", "fld_1/./x", "fld_1/a\\b", "fld_1//x", "fld_1/a\u0000", "http://x/y", "fld_1/C:x"]) {
      expect(parseRef(bad)).toBeNull();
    }
    expect(isAuthorizedRef("fld_1/a.py", new Set(["fld_1"]))).toBe(true);
    expect(isAuthorizedRef("fld_2/a.py", ["fld_1"])).toBe(false);
  });

  it("resolves files inside picked folders only", async () => {
    const { store, folder, dir } = await env({ src: { "a.py": "print(1)" } });
    const fh = await resolveFile(store, `${folder.id}/src/a.py`);
    expect(fh?.name).toBe("a.py");
    expect(await authorizedFile(store, `${folder.id}/src/a.py`)).toBe(true);
    expect(await resolveFile(store, `${folder.id}/src/missing.py`)).toBeNull();
    expect(await resolveFile(store, `${folder.id}/../a.py`)).toBeNull();
    expect(await resolveFile(store, "fld_unknown/src/a.py")).toBeNull();
    expect(await authorizedFile(store, "fld_unknown/src/a.py")).toBe(false);
    dir.permission = "prompt";
    expect(await resolveFile(store, `${folder.id}/src/a.py`)).toBeNull();
    dir.requestResult = "denied";
    expect(await requestPermission(dir.asHandle)).toBe(false);
    dir.requestResult = "granted";
    expect(await requestPermission(dir.asHandle)).toBe(true);
    expect(await resolveFile(store, `${folder.id}/src/a.py`)).not.toBeNull();
  });

  it("maps a path seen on screen to an indexed reference", () => {
    const uris = ["f1/src/main.py", "f1/tests/test_main.py", "f1/a/util.py", "f1/b/util.py"];
    expect(matchScreenPath(uris, "C:\\dev\\app\\src\\main.py")).toBe("f1/src/main.py");
    expect(matchScreenPath(uris, "tests/test_main.py")).toBe("f1/tests/test_main.py");
    expect(matchScreenPath(uris, "util.py")).toBeNull(); // ambiguous
    expect(matchScreenPath(uris, "a/util.py")).toBe("f1/a/util.py");
    expect(matchScreenPath(uris, "other.py")).toBeNull();
    expect(matchScreenPath(uris, "../src/main.py")).toBeNull();
  });
});

// ----------------------------------------------------------------- deletion

describe("deletion", () => {
  it("forgets a folder everywhere", async () => {
    const { folder, dir, deps, store, index, ret } = await env({ "a.md": "deploy", "b.md": "release" });
    await indexFolder(dir.asHandle, folder.id, deps);
    const ids = (await store.itemsForFolder(folder.id)).map((i) => i.id);
    expect(await forgetFolder(store, index, folder.id)).toBe(2);
    for (const id of ids) expect(await verifyDeleted(store, index, id)).toBe(true);
    expect(await store.getFolder(folder.id)).toBeNull();
    expect(await ret.retrieve("deploy")).toEqual([]);
  });

  it("forgets one item and purges expired ones", async () => {
    const store = await openStore();
    const index = new VectorIndex(DIM);
    for (const [id, exp] of [["a", null], ["b", new Date(Date.now() - 1).toISOString()]] as const) {
      await store.putItem(item(id, { expiresAt: exp }), te.embed(id));
      index.add(id, te.embed(id));
    }
    expect(await purgeExpired(store, index)).toBe(1);
    expect(await verifyDeleted(store, index, "b")).toBe(true);
    expect(await verifyDeleted(store, index, "a")).toBe(false);
    expect(await forget(store, index, "a")).toBe(true);
    expect(await forget(store, index, "a")).toBe(false);
    expect(await verifyDeleted(store, index, "a")).toBe(true);
  });
});
