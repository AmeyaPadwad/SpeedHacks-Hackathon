/**
 * Pette's in-app file viewer: where "Open" shows a file SOMA pointed at. Text files show with line numbers and
 * the line scrolled into view and highlighted; images show from an object URL; anything else is offered as a
 * download of its bytes (application/octet-stream, so the browser never renders or runs it).
 * The content stays in memory while the viewer is open; nothing is stored.
 */
import { Download, FileText, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { downloadBlob, fileKind } from "../actions";
import type { OpenedFile } from "../controller";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** Show at most this many lines (big files stay responsive). */
export const MAX_LINES = 5000;

function useObjectUrl(blob: Blob | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!blob || typeof URL.createObjectURL !== "function") return;
    const u = URL.createObjectURL(blob);
    setUrl(u);
    return () => {
      URL.revokeObjectURL(u);
      setUrl(null);
    };
  }, [blob]);
  return url;
}

export function FileViewer({ file, onClose }: { file: OpenedFile; onClose: () => void }) {
  const kind = fileKind(file);
  const target = useRef<HTMLDivElement>(null);
  const lines = useMemo(() => (file.text ?? "").split(/\r?\n/).slice(0, MAX_LINES), [file.text]);
  const imageUrl = useObjectUrl(kind === "image" ? file.blob : null);
  const download = useMemo(() => (kind === "binary" ? downloadBlob(file.blob) : null), [kind, file.blob]);
  const downloadUrl = useObjectUrl(download);

  useEffect(() => {
    target.current?.scrollIntoView?.({ block: "center" });
  }, [file]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-zinc-950/30 p-4 backdrop-blur-sm" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-label={`File: ${file.name}`} data-testid="file-viewer"
        className="flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-xl border bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 border-b px-4 py-2.5">
          <FileText className="h-4 w-4 text-zinc-500" />
          <div className="min-w-0 flex-1 truncate text-sm font-semibold">{file.name}</div>
          {file.line && <span className="text-xs text-muted-foreground">line {file.line}</span>}
          <Button variant="ghost" size="icon" aria-label="Close" onClick={onClose}><X className="h-4 w-4" /></Button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          {kind === "text" && (
            <pre className="py-2 font-mono text-[12.5px] leading-5">
              {lines.map((l, i) => {
                const hit = file.line === i + 1;
                return (
                  <div key={i} ref={hit ? target : undefined} data-line={i + 1}
                    className={cn("flex px-2", hit && "bg-amber-200/70")}>
                    <span className="w-12 shrink-0 select-none pr-3 text-right text-zinc-400">{i + 1}</span>
                    <code className="whitespace-pre">{l || " "}</code>
                  </div>
                );
              })}
            </pre>
          )}
          {kind === "image" && (
            <div className="flex items-center justify-center bg-zinc-50 p-4">
              {imageUrl ? <img src={imageUrl} alt={file.name} className="max-h-[70vh] max-w-full object-contain" /> : null}
            </div>
          )}
          {kind === "binary" && (
            <div className="space-y-3 p-6 text-sm">
              <p>Pette can't show this kind of file. You can save a copy; it won't be opened or run.</p>
              {downloadUrl && (
                <a href={downloadUrl} download={file.name}
                  className="inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm font-medium hover:bg-accent">
                  <Download className="h-4 w-4" /> Download {file.name}
                </a>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
