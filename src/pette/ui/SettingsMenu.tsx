/**
 * Settings (replaces the desktop tray menu): screen sharing, pause, suggestions, speech bubble, camera and its
 * picture, gestures, folders (add / remove with memory counts), the floating window, the world panel, and
 * Pette's own memory. Folder names and counts only; never paths or file names.
 */
import { AnimatePresence, motion } from "framer-motion";
import { FolderPlus, MonitorUp, Settings, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { FolderInfo } from "@/soma/service";
import { HOTKEYS, comboLabel } from "../hotkeys";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

export interface SettingsState {
  screenShared: boolean;
  paused: boolean;
  suggestions: boolean;
  bubble: boolean;
  camera: boolean;
  cameraPicture: boolean;
  gestures: boolean;
  floating: boolean;
  floatSupported: boolean;
  world: boolean;
  remember: boolean;
  folders: FolderInfo[];
}

export interface SettingsActions {
  onShareScreen: () => void;
  onStopSharing: () => void;
  onLook: () => void;
  onPause: (on: boolean) => void;
  onSuggestions: (on: boolean) => void;
  onBubble: (on: boolean) => void;
  onCamera: (on: boolean) => void;
  onCameraPicture: (on: boolean) => void;
  onGestures: (on: boolean) => void;
  onAddFolder: () => void;
  onRemoveFolder: (id: string) => void;
  onFloat: (on: boolean) => void;
  onWorld: (on: boolean) => void;
  onRemember: (on: boolean) => void;
  onClearMemory: () => void;
}

function Row({ label, hint, checked, onChange, disabled }: { label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <label className={cn("flex items-center justify-between gap-3 py-1.5", disabled && "opacity-50")}>
      <span className="min-w-0">
        <span className="block text-[13px] font-medium">{label}</span>
        {hint && <span className="block text-[11px] text-muted-foreground">{hint}</span>}
      </span>
      <Switch checked={checked} onCheckedChange={onChange} disabled={disabled} aria-label={label} />
    </label>
  );
}

export function SettingsMenu(props: SettingsState & SettingsActions & { className?: string }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={box} className={cn("relative", props.className)}>
      <Button variant="outline" size="icon" className="h-9 w-9 rounded-full bg-white/80 backdrop-blur" aria-label="Pette settings"
        aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <Settings className="h-4 w-4" />
      </Button>
      <AnimatePresence>
        {open && (
          <motion.div role="menu" aria-label="Pette settings" initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
            className="absolute right-0 top-11 z-[55] max-h-[80vh] w-80 overflow-y-auto rounded-xl border bg-white/95 p-3 shadow-xl backdrop-blur">
            <div className="mb-1 flex items-center justify-between">
              <div className="text-sm font-semibold">Pette</div>
              <Button variant="ghost" size="icon" aria-label="Close settings" onClick={() => setOpen(false)}><X className="h-4 w-4" /></Button>
            </div>

            {props.screenShared ? (
              <Button variant="outline" size="sm" className="w-full" onClick={props.onStopSharing}>
                <MonitorUp className="h-4 w-4" /> Stop sharing the screen
              </Button>
            ) : (
              <Button size="sm" className="w-full" onClick={props.onShareScreen}>
                <MonitorUp className="h-4 w-4" /> Share a screen with SOMA
              </Button>
            )}
            <Button variant="ghost" size="sm" className="mt-1 w-full justify-start" onClick={() => { props.onLook(); setOpen(false); }}
              disabled={props.paused}>
              Look at my screen now <span className="ml-auto text-[11px] text-muted-foreground">{comboLabel(HOTKEYS.look)}</span>
            </Button>
            <Separator className="my-2" />

            <Row label="Pause all" hint="No cards, no camera, SOMA stops looking" checked={props.paused} onChange={props.onPause} />
            <Row label="SOMA suggestions" hint={comboLabel(HOTKEYS.suggestions)} checked={props.suggestions} onChange={props.onSuggestions} />
            <Row label="Speech bubble" hint={comboLabel(HOTKEYS.bubble)} checked={props.bubble} onChange={props.onBubble} />
            <Separator className="my-2" />

            <Row label="Camera" hint="You, hands, presence: processed in this browser, never stored" checked={props.camera}
              onChange={props.onCamera} disabled={props.paused} />
            <Row label="Show camera picture" hint={comboLabel(HOTKEYS.camera)} checked={props.cameraPicture} onChange={props.onCameraPicture} />
            <Row label="Gestures" hint="Pinch = Do it · point = earlier / later · pinch-drag = scroll" checked={props.gestures} onChange={props.onGestures} />
            <Separator className="my-2" />

            <div className="flex items-center justify-between py-1">
              <span className="text-[13px] font-medium">Folders SOMA learns from</span>
              <Button variant="outline" size="sm" onClick={props.onAddFolder}><FolderPlus className="h-4 w-4" /> Add</Button>
            </div>
            {props.folders.length ? (
              <ul className="space-y-1">
                {props.folders.map((f) => (
                  <li key={f.id} className="flex items-center justify-between rounded-md border px-2 py-1 text-xs">
                    <span className="min-w-0 truncate font-medium">{f.name}</span>
                    <span className="flex items-center gap-1 text-muted-foreground">
                      {f.memories} files
                      <Button variant="ghost" size="icon" aria-label={`Remove ${f.name}`} onClick={() => props.onRemoveFolder(f.id)}>
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </span>
                  </li>
                ))}
              </ul>
            ) : <p className="text-[11px] text-muted-foreground">None yet: add a folder so suggestions can point at your files.</p>}
            <Separator className="my-2" />

            <Row label="Float Pette over other apps" hint={props.floatSupported ? "Picture-in-Picture window" : "Needs Chrome or Edge"}
              checked={props.floating} onChange={props.onFloat} disabled={!props.floatSupported} />
            <Row label="World panel" hint={comboLabel(HOTKEYS.world)} checked={props.world} onChange={props.onWorld} />
            <Row label="Remember my patterns" hint="Facts only: kinds and outcomes, event names" checked={props.remember} onChange={props.onRemember} />
            <Button variant="ghost" size="sm" className="mt-1 w-full justify-start text-destructive" onClick={props.onClearMemory}>
              <Trash2 className="h-4 w-4" /> Clear Pette's memory
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
