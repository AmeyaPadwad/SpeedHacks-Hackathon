/**
 * A scripted `SomaService` (port of tools/sandbox.py's fake SOMA): used by the tests and by the `?sandbox`
 * dev mode. `queue(sample)` makes the next `suggestion()` return it (once); `emptyLook` makes "look now" find
 * nothing; `offline` makes every call fail, like a stopped SOMA. Every call is recorded in `requests`.
 *
 * It ships two tiny files (`sandbox/calc.py`, `sandbox/retries.md`) so "Open" has something to show.
 */
import type { FolderInfo, FoldersPayload, SomaService, SomaStatus, SuggestionPayload } from "@/soma/service";
import type { FeedbackOutcome } from "@/soma/types";

export type Sample = Pick<SuggestionPayload, "action" | "template" | "text" | "file" | "line" | "memorySource">;

export const SAMPLES: Record<string, Sample> = {
  "Open file at line": { action: "OPEN_RELEVANT_FILE", template: "", text: "Open calc.py at line 7", file: "sandbox/calc.py", line: 7 },
  "Your files (memory)": { action: "COMPARE_WITH_PRIOR_WORK", template: "", text: 'Compare with your earlier document "Retry notes"',
    memorySource: "sandbox/retries.md" },
  "Review changes": { action: "SUGGEST_NEXT_STEP", template: "review_changes", text: 'Review your changes to "calc.py" before you commit' },
  "Write a test": { action: "SUGGEST_NEXT_STEP", template: "write_tests", text: 'Write a test for the code in "calc.py"' },
  "Explain error": { action: "EXPLAIN_ERROR", template: "", text: 'Explain the error "ZeroDivisionError: division by zero"' },
};

export const SANDBOX_FILES: Record<string, string> = {
  "sandbox/calc.py": "def total(items):\n    count = 0\n    for i in items:\n        count += i\n    return count\n\n\n"
    + "print(total([]) / len([]))\n",
  "sandbox/retries.md": "# Retry notes\nBack off exponentially; cap at 30 s.\n",
};

export interface FakeRequest { at: number; call: string; args: unknown[] }

export class FakeSoma implements SomaService {
  current: SuggestionPayload = { available: true, silent: true };
  paused = false;
  capturing = false;
  /** "Look now" finds nothing. */
  emptyLook = false;
  /** Every call fails ("Stop fake SOMA"). */
  offline = false;
  /** What `trigger()` returns when not paused (false = "no screen shared"). */
  triggerResult = true;
  folderList: FolderInfo[] = [];
  files: Record<string, string> = { ...SANDBOX_FILES };
  readonly requests: FakeRequest[] = [];
  private pending: SuggestionPayload[] = [];
  private indexingLeft = 0;
  private lastReport: FoldersPayload["lastReport"] = {};
  private nextId = 0;
  private listeners = new Set<(r: FakeRequest) => void>();

  /** Called with every request (the sandbox's request log). */
  onRequest(fn: (r: FakeRequest) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private record(call: string, ...args: unknown[]) {
    const r = { at: Date.now(), call, args };
    this.requests.push(r);
    for (const fn of this.listeners) fn(r);
    if (this.offline) throw new Error("SOMA isn't running.");
  }

  /** Make the next `suggestion()` return `sample` (once). Returns its id. */
  queue(sample: Sample | Partial<SuggestionPayload>): string {
    const id = `fake_${++this.nextId}`;
    this.pending.push({ available: true, silent: false, valid: true, id, evidence: [], file: null, line: null,
      memorySource: null, ...sample });
    return id;
  }

  calls(name: string): FakeRequest[] {
    return this.requests.filter((r) => r.call === name);
  }

  // --- SomaService ------------------------------------------------------------
  async grantScreen() { this.record("grantScreen"); this.capturing = true; return true; }
  revokeScreen() { this.capturing = false; this.requests.push({ at: Date.now(), call: "revokeScreen", args: [] }); }
  start() { /* scripted: nothing to run */ }
  stop() { /* scripted: nothing to run */ }

  async suggestion(): Promise<SuggestionPayload> {
    if (this.offline) throw new Error("SOMA isn't running.");
    if (this.pending.length && !this.paused) this.current = this.pending.shift() as SuggestionPayload;
    return this.current;
  }

  async status(): Promise<SomaStatus> {
    if (this.offline) throw new Error("SOMA isn't running."); // a read: not logged
    return { paused: this.paused, capturing: this.capturing, steps: 0, processed: 0, frameVersion: 0, lastSkipReason: null,
      memories: this.folderList.reduce((a, f) => a + f.memories, 0), ready: true };
  }

  async feedback(suggestionId: string, outcome: FeedbackOutcome) { this.record("feedback", suggestionId, outcome); }

  async trigger() {
    this.record("trigger");
    if (this.paused || !this.triggerResult) return false;
    if (!this.emptyLook) this.queue(SAMPLES["Review changes"]);
    return true;
  }

  async pause() { this.record("pause"); this.paused = true; }
  async resume() { this.record("resume"); this.paused = false; }

  async folders(): Promise<FoldersPayload> {
    if (this.offline) throw new Error("SOMA isn't running.");
    const indexing = this.indexingLeft > 0;
    if (indexing) this.indexingLeft--;
    return { folders: this.folderList.map((f) => ({ ...f })), indexing, lastReport: this.lastReport };
  }

  /** Adds `handle`'s folder, or a made-up "notes" folder holding the sandbox files; indexing takes one poll. */
  async addFolder(handle?: FileSystemDirectoryHandle): Promise<FolderInfo | null> {
    this.record("addFolder", handle?.name ?? null);
    const name = handle?.name ?? "notes";
    const files = Object.keys(this.files).length;
    const info: FolderInfo = { id: `folder_${name}`, name, memories: files };
    this.folderList = [...this.folderList.filter((f) => f.id !== info.id), info];
    this.lastReport = { files, skipped: 0 };
    this.indexingLeft = 1;
    return info;
  }

  async removeFolder(folderId: string): Promise<number> {
    this.record("removeFolder", folderId);
    const gone = this.folderList.filter((f) => f.id === folderId);
    this.folderList = this.folderList.filter((f) => f.id !== folderId);
    return gone.reduce((a, f) => a + f.memories, 0);
  }

  async readFile(ref: string): ReturnType<SomaService["readFile"]> {
    this.record("readFile", ref);
    const text = this.files[ref];
    if (text === undefined) return null;
    const name = ref.split("/").pop() ?? ref;
    return { name, text, blob: new Blob([text], { type: "text/plain" }) };
  }
}
