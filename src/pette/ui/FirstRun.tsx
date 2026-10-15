/**
 * First run: what Pette and SOMA do, and the consent model. Sharing a screen is the opt-in for SOMA to look;
 * the camera is optional; nothing leaves the browser.
 */
import { Camera, Check, HardDrive, MonitorUp, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Spotlight } from "@/components/aceternity/spotlight";

export interface FirstRunProps {
  screenShared: boolean;
  cameraOn: boolean;
  onShareScreen: () => void;
  onCamera: (on: boolean) => void;
  onContinue: () => void;
}

export function FirstRun({ screenShared, cameraOn, onShareScreen, onCamera, onContinue }: FirstRunProps) {
  return (
    <div className="relative flex min-h-full items-center justify-center overflow-y-auto bg-zinc-950 p-4">
      <Spotlight className="-left-10 -top-24" fill="#a78bfa" />
      <Card className="relative z-10 w-full max-w-lg rounded-2xl shadow-2xl">
        <CardHeader className="p-6 pb-3">
          <div className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">Pette · SOMA</div>
          <CardTitle className="text-xl">Next steps, when you want them</CardTitle>
          <CardDescription className="text-sm">
            SOMA looks at a screen or window you share, reads it in this browser, and ranks what you might do next.
            Pette shows one suggestion at a time in a speech bubble. Nothing happens until you press Open / Copy, or pinch.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 p-6 pt-2">
          <ul className="space-y-2.5 text-[13px]">
            <li className="flex gap-2.5"><MonitorUp className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
              <span><b>Sharing a screen is your opt-in.</b> SOMA only sees what you share, keeps screenshots in memory, and stops when you stop sharing or pause.</span></li>
            <li className="flex gap-2.5"><Camera className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
              <span><b>The camera is optional.</b> It lets Pette notice when you step away and read gestures (pinch = Do it). Frames are drawn on her orb only, never stored.</span></li>
            <li className="flex gap-2.5"><HardDrive className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
              <span><b>Nothing leaves the browser.</b> No uploads, no cloud AI. Pette's memory holds facts only: which kind of suggestion and what you did with it.</span></li>
            <li className="flex gap-2.5"><ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
              <span><b>Nothing runs by itself.</b> Open shows a file from a folder you granted; Copy copies the text. That's all.</span></li>
          </ul>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Button className="flex-1" onClick={onShareScreen} disabled={screenShared}>
              {screenShared ? <Check className="h-4 w-4" /> : <MonitorUp className="h-4 w-4" />} {screenShared ? "Screen shared" : "Share a screen"}
            </Button>
            <Button variant="outline" className="flex-1" onClick={() => onCamera(!cameraOn)} aria-pressed={cameraOn}>
              {cameraOn ? <Check className="h-4 w-4" /> : <Camera className="h-4 w-4" />} {cameraOn ? "Camera on" : "Use the camera"}
            </Button>
          </div>
          <button type="button" className="w-full text-center text-xs text-muted-foreground underline-offset-2 hover:underline" onClick={onContinue}>
            {screenShared ? "Continue" : "Continue without sharing (you can share later from settings)"}
          </button>
        </CardContent>
      </Card>
    </div>
  );
}
