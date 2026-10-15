/**
 * SOMA in the browser: `createSoma()` wires the real collaborators into a `SomaRuntime` + scheduler and
 * implements `SomaService` (port of SOMA/soma/runtime/factory.py, SOMA/soma/consent.py and the verbs of
 * SOMA/soma/api/local_api.py, now in-process).
 *
 * Consent, web edition:
 * - **Screen**: granted only through the browser's `getDisplayMedia` picker; the user can stop sharing from
 *   the browser at any time, which stops capture.
 * - **Folders**: only the directories the user picks (`showDirectoryPicker`); files outside them are never
 *   indexed, proposed or read.
 * - **Feedback storage**: always on and local (IndexedDB): the preference ranker's counts, no text.
 *
 * tfjs (webgl, falling back to cpu) and tesseract.js load lazily on the first screen grant. Nothing here
 * executes anything: it only returns suggestions and records feedback.
 */
import { ScreenCapturer } from "./capture/screen";
import { DEFAULT_CONFIG, type SomaConfig } from "./config";
import { OnlinePreferenceRanker, type PreferenceRankerState } from "./learning/preferenceRanker";
import { requestPermission, resolveFile } from "./memory/access";
import { forgetFolder } from "./memory/deletion";
import { TextEmbedder } from "./memory/embedder";
import { indexFolder, type IndexDeps } from "./memory/ingestion";
import { MemoryRetriever } from "./memory/retrieval";
import { MemoryStore } from "./memory/store";
import type { VectorIndex } from "./memory/vectorIndex";
import type { OcrEngine } from "./perception/ocr";
import type { Clock } from "./policy/safetyGate";
import type { CapturerLike, ModelLike } from "./runtime/deps";
import { suggestionPayload } from "./runtime/payload";
import { SomaRuntime, type StepResult } from "./runtime/pipeline";
import { InferenceScheduler, type Timers } from "./runtime/scheduler";
import { type KeyValueStore, openStateStore } from "./runtime/stateStore";
import type { FolderInfo, FoldersPayload, SomaService, SomaStatus, SuggestionPayload } from "./service";
import type { FeedbackOutcome } from "./types";

export type { SomaService } from "./service";

/** Where the model's weights live once trained / saved. */
export const MODEL_URL = "indexeddb://soma-model";
const RANKER_KEY = "preference-ranker";

/** A capturer the service can grant / revoke (`ScreenCapturer`). */
export interface GrantableCapturer extends CapturerLike {
  grant(): Promise<boolean>;
  revoke(): void;
  onEnded?: () => void;
}

export interface SomaDeps {
  store: MemoryStore;
  index: VectorIndex;
  embedder: TextEmbedder;
  capturer: GrantableCapturer;
  /** Built on the first grant (default: tesseract.js behind `FallbackOcr`). */
  createOcr: () => Promise<OcrEngine>;
  /** Built on the first grant (default: tfjs `webgl` -> `cpu`, seeded `SomaModel`, saved weights if any). */
  createModel: (cfg: SomaConfig) => Promise<ModelLike>;
  ranker: OnlinePreferenceRanker;
  state: KeyValueStore;
  /** Prompt for a folder; null when the user cancels. */
  pickDirectory: () => Promise<FileSystemDirectoryHandle | null>;
  clock: Clock;
  timers: Timers;
  /** For tests (image sizes during indexing). */
  imageSize: IndexDeps["imageSize"];
}

/** `SomaService` plus handles for the app shell and tests. */
export interface SomaHandle extends SomaService {
  readonly runtime: SomaRuntime;
  readonly scheduler: InferenceScheduler<StepResult>;
  /** Resolves when tfjs + OCR are loaded (after the first grant). */
  whenReady(): Promise<void>;
  /** Resolves when background indexing has finished. */
  whenIndexed(): Promise<void>;
  dispose(): Promise<void>;
}

async function defaultModel(cfg: SomaConfig): Promise<ModelLike> {
  const tf = await import("@tensorflow/tfjs");
  let ok = false;
  try {
    ok = await tf.setBackend("webgl");
  } catch {
    ok = false;
  }
  if (!ok) await tf.setBackend("cpu");
  await tf.ready();
  const { SomaModel } = await import("./models/somaModel");
  const model = new SomaModel(cfg.model, cfg.seed);
  try {
    await model.load(MODEL_URL);
  } catch {
    // no saved weights: keep the seeded, untrained model
  }
  return model;
}

async function defaultOcr(): Promise<OcrEngine> {
  const { FallbackOcr, TesseractOcr } = await import("./perception/ocr");
  return new FallbackOcr(new TesseractOcr());
}

async function defaultPick(): Promise<FileSystemDirectoryHandle | null> {
  const picker = (globalThis as unknown as { showDirectoryPicker?: (o?: object) => Promise<FileSystemDirectoryHandle> })
    .showDirectoryPicker;
  if (typeof picker !== "function") return null;
  try {
    return await picker({ mode: "read", id: "soma-folder" });
  } catch {
    return null; // cancelled
  }
}

const TEXT_EXTENSIONS = (cfg: SomaConfig): Set<string> =>
  new Set(cfg.memory.allowedExtensions.filter((e) => ![".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"].includes(e)));

/** Build SOMA. Every collaborator can be replaced through `deps` (tests pass fakes). */
export async function createSoma(opts: { cfg?: SomaConfig; deps?: Partial<SomaDeps> } = {}): Promise<SomaHandle> {
  const cfg = opts.cfg ?? DEFAULT_CONFIG;
  const d = opts.deps ?? {};
  const D = cfg.model.dModel;
  const store = d.store ?? (await MemoryStore.open());
  const index = d.index ?? (await store.loadIndex(D));
  const embedder = d.embedder ?? new TextEmbedder(D);
  const capturer = d.capturer ?? new ScreenCapturer(cfg.capture);
  const state = d.state ?? (await openStateStore());
  const ranker = d.ranker ?? new OnlinePreferenceRanker();
  try {
    const saved = await state.get<PreferenceRankerState>(RANKER_KEY);
    if (saved) ranker.loadStateDict(saved);
  } catch {
    // corrupt state: start fresh
  }
  const createOcr = d.createOcr ?? defaultOcr;
  const createModel = d.createModel ?? defaultModel;
  const pickDirectory = d.pickDirectory ?? defaultPick;

  const retriever = new MemoryRetriever(store, index, embedder, cfg.memory);
  const runtime = new SomaRuntime(cfg, {
    capturer, embedder, retriever, ranker, clock: d.clock,
    fileRefs: async () => (await store.allItems()).map((i) => i.sourceUri).filter((u): u is string => !!u),
  });
  runtime.onLearned = () => {
    void state.set(RANKER_KEY, ranker.stateDict()).catch(() => undefined);
  };
  runtime.setAllowedFolders((await store.listFolders()).map((f) => f.id));

  const scheduler = new InferenceScheduler(runtime, Math.max(0.25, cfg.capture.minIntervalS), { timers: d.timers });

  let warm: Promise<void> | null = null;
  const warmUp = (): Promise<void> => {
    warm ??= (async () => {
      const [ocr, model] = await Promise.all([createOcr(), createModel(cfg)]);
      runtime.ocr = ocr;
      runtime.model = model;
    })().catch((err) => {
      warm = null; // retry on the next grant
      throw err;
    });
    return warm;
  };

  let indexingCount = 0;
  let indexing: Promise<void> = Promise.resolve();
  let lastReport: FoldersPayload["lastReport"] = {};

  const refreshFolders = async (): Promise<void> => {
    runtime.setAllowedFolders((await store.listFolders()).map((f) => f.id));
  };

  const service: SomaHandle = {
    runtime,
    scheduler,

    async grantScreen(): Promise<boolean> {
      const ok = await capturer.grant();
      if (ok) {
        capturer.onEnded = () => undefined; // the runtime sees an inactive stream and stays silent
        warmUp().catch(() => undefined);
      }
      return ok;
    },

    revokeScreen(): void {
      capturer.revoke();
    },

    start(): void {
      scheduler.start();
    },

    stop(): void {
      scheduler.stop();
    },

    suggestion(): Promise<SuggestionPayload> {
      return suggestionPayload(runtime, (id) => store.getItem(id));
    },

    async status(): Promise<SomaStatus> {
      const last = runtime.lastResult;
      let memories = 0;
      try {
        memories = await store.countItems();
      } catch {
        memories = 0;
      }
      return {
        paused: runtime.paused,
        capturing: capturer.active,
        steps: runtime.steps,
        processed: runtime.processed,
        frameVersion: last?.frameVersion ?? 0,
        lastSkipReason: (runtime.lastSkipReason ?? "").split(":")[0] || null,
        memories,
        ready: runtime.ocr !== null && runtime.model !== null,
      };
    },

    async feedback(suggestionId: string, outcome: FeedbackOutcome): Promise<void> {
      await runtime.feedback(suggestionId, outcome);
    },

    async trigger(): Promise<boolean> {
      if (runtime.paused || !capturer.active) return false;
      return scheduler.trigger();
    },

    async pause(): Promise<void> {
      scheduler.pause();
    },

    async resume(): Promise<void> {
      scheduler.resume();
    },

    async folders(): Promise<FoldersPayload> {
      const [rows, counts] = await Promise.all([store.listFolders(), store.countByFolder()]);
      return {
        folders: rows.map((f) => ({ id: f.id, name: f.name, memories: counts[f.id] ?? 0 })),
        indexing: indexingCount > 0,
        lastReport: { ...lastReport },
      };
    },

    async addFolder(handle?: FileSystemDirectoryHandle): Promise<FolderInfo | null> {
      const dir = handle ?? (await pickDirectory());
      if (!dir || dir.kind !== "directory") return null;
      if (!(await requestPermission(dir))) return null;
      const folder = await store.addFolder(dir);
      await refreshFolders();
      indexingCount += 1;
      const job = indexFolder(dir, folder.id, { store, embedder, index, cfg: cfg.memory, imageSize: d.imageSize })
        .then((r) => {
          lastReport = { files: r.files, skipped: r.skipped };
        })
        .catch((err: unknown) => {
          lastReport = { error: (err as Error)?.name || "Error" }; // never the message (may hold paths)
        })
        .finally(() => {
          indexingCount -= 1;
          runtime.invalidateFiles();
        });
      indexing = Promise.all([indexing, job]).then(() => undefined);
      return { id: folder.id, name: folder.name, memories: 0 };
    },

    async removeFolder(folderId: string): Promise<number> {
      const forgotten = await forgetFolder(store, index, folderId);
      await refreshFolders();
      runtime.invalidateFiles();
      return forgotten;
    },

    async readFile(ref: string) {
      if (!runtime.authorizedFile(ref)) return null;
      const handle = await resolveFile(store, ref);
      if (!handle) return null;
      let file: File;
      try {
        file = await handle.getFile();
      } catch {
        return null;
      }
      const dot = file.name.lastIndexOf(".");
      const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
      const text = TEXT_EXTENSIONS(cfg).has(ext) && file.size <= cfg.memory.maxFileBytes ? await file.text() : null;
      return { name: file.name, text, blob: file };
    },

    whenReady: () => warmUp(),
    whenIndexed: () => indexing,

    async dispose(): Promise<void> {
      scheduler.stop();
      capturer.revoke();
      await runtime.ocr?.dispose().catch(() => undefined);
      const m = runtime.model as (ModelLike & { dispose?: () => void }) | null;
      m?.dispose?.();
    },
  };
  return service;
}
