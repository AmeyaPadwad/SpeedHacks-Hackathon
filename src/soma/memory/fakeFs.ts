/**
 * In-memory stand-ins for File System Access handles, for tests and demos only.
 *
 * Methods live on the prototypes (no own function properties), so the handles survive IndexedDB's
 * structured clone like real `FileSystemDirectoryHandle`s do.
 */

let nextUid = 1;

export class FakeFile {
  readonly kind = "file" as const;
  readonly uid = nextUid++;
  constructor(public name: string, public content: string, public lastModified = 1_700_000_000_000, public sizeOverride?: number) {}

  async getFile(): Promise<File> {
    const content = this.content;
    const size = this.sizeOverride ?? new TextEncoder().encode(content).length;
    return { name: this.name, size, lastModified: this.lastModified, type: "", text: async () => content } as unknown as File;
  }

  async isSameEntry(other: { uid?: number }): Promise<boolean> {
    return other?.uid === this.uid;
  }
}

export class FakeDir {
  readonly kind = "directory" as const;
  readonly uid = nextUid++;
  children = new Map<string, FakeDir | FakeFile>();
  permission: PermissionState = "granted";
  requestResult: PermissionState = "granted";

  constructor(public name: string, tree: Record<string, unknown> = {}) {
    for (const [k, v] of Object.entries(tree)) {
      this.children.set(k, typeof v === "string" ? new FakeFile(k, v) : v instanceof FakeFile || v instanceof FakeDir ? v : new FakeDir(k, v as Record<string, unknown>));
    }
  }

  async *entries(): AsyncGenerator<[string, FakeDir | FakeFile]> {
    for (const e of this.children) yield e;
  }

  async getDirectoryHandle(name: string): Promise<FakeDir> {
    const c = this.children.get(name);
    if (!(c instanceof FakeDir)) throw new DOMException("not found", "NotFoundError");
    return c;
  }

  async getFileHandle(name: string): Promise<FakeFile> {
    const c = this.children.get(name);
    if (!(c instanceof FakeFile)) throw new DOMException("not found", "NotFoundError");
    return c;
  }

  async isSameEntry(other: { uid?: number }): Promise<boolean> {
    return other?.uid === this.uid;
  }

  async queryPermission(): Promise<PermissionState> {
    return this.permission;
  }

  async requestPermission(): Promise<PermissionState> {
    this.permission = this.requestResult;
    return this.permission;
  }

  /** As the DOM type, for APIs that take a real handle. */
  get asHandle(): FileSystemDirectoryHandle {
    return this as unknown as FileSystemDirectoryHandle;
  }
}
