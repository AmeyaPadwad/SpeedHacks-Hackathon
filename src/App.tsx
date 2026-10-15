/**
 * Pette's shell (port of main.py): the first-run consent screen, then Pette (the orb), SOMA's speech bubble,
 * the world panel and the settings menu, all driven by one controller and hub around the `SomaService` it is
 * given. The webcam loads lazily the first time it is turned on.
 */
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { PictureInPicture2 } from "lucide-react";
import type { SomaService } from "@/soma/service";
import { WorldPanel } from "@/components/WorldPanel";
import { Button } from "@/components/ui/button";
import type { VisionEvent } from "@/pette/contracts";
import { FakeSoma } from "@/pette/fakeSoma";
import { HOTKEYS, comboLabel, useHotkeys } from "@/pette/hotkeys";
import { loadWatcher, type Watcher } from "@/pette/loadVision";
import { usePip } from "@/pette/pip";
import { Sandbox } from "@/pette/Sandbox";
import { statusLabel } from "@/pette/status";
import { FileViewer } from "@/pette/ui/FileViewer";
import { FirstRun } from "@/pette/ui/FirstRun";
import { Floating } from "@/pette/ui/Floating";
import { Notice } from "@/pette/ui/Notice";
import { ORB_BOX, Orb } from "@/pette/ui/Orb";
import { SettingsMenu } from "@/pette/ui/SettingsMenu";
import { SpeechBubble } from "@/pette/ui/SpeechBubble";
import { createPette, dockPanel, useOrbPosition, usePetteState, useSuggester, useWorldSnapshot } from "@/pette/usePette";
import type { FactStore } from "@/pette/world/store";

export interface AppProps {
  soma: SomaService;
  store: FactStore;
  /** Show the sandbox control panel (only with a `FakeSoma`). */
  sandbox?: boolean;
  /** Skip the first-run screen (tests, returning sandbox sessions). */
  skipIntro?: boolean;
}

/** Camera messages worth a bubble (the desktop app surfaced only these). */
const CAMERA_PROBLEM = /unavailable|disconnected|denied|blocked|not allowed|no camera|failed|error/i;

export default function App({ soma, store, sandbox = false, skipIntro = false }: AppProps) {
  const [pette] = useState(() => createPette(soma, store));
  const { controller, suggester, world } = pette;
  const st = usePetteState(controller);
  const { enabled, folders } = useSuggester(suggester);
  const snapshot = useWorldSnapshot(world, store, st.worldVisible);
  const [intro, setIntro] = useState(!skipIntro);
  const [screenShared, setScreenShared] = useState(false);
  const [remember, setRemember] = useState(store.enabled);
  const [orbPos, setOrbPos] = useOrbPosition();
  const pip = usePip();
  const watcher = useRef<Watcher | null>(null);
  const somaStarted = useRef(false);
  const greeted = useRef(false);

  // --- lifecycle -------------------------------------------------------------
  useEffect(() => {
    suggester.start();
    return () => suggester.stop();
  }, [suggester]);

  useEffect(() => () => {
    watcher.current?.stop();
    soma.stop();
  }, [soma]);

  // Screen sharing can end from the browser's own "Stop sharing" bar: follow SOMA's status.
  useEffect(() => {
    const t = setInterval(() => {
      soma.status().then((s) => setScreenShared(s.capturing), () => undefined);
    }, 2000);
    return () => clearInterval(t);
  }, [soma]);

  const greet = useCallback(() => {
    if (greeted.current) return;
    greeted.current = true;
    controller.say(`I'll suggest next steps as you work. Click me (or ${comboLabel(HOTKEYS.look)}) and SOMA looks at your screen now. `
      + "Add folders from settings so SOMA knows your files.", { title: "Ready · SOMA", seconds: 8 });
  }, [controller]);

  // --- webcam ----------------------------------------------------------------
  const ensureWatcher = useCallback(async () => {
    if (!watcher.current) {
      watcher.current = await loadWatcher(controller, {
        factLog: store,
        blinkOn: () => world.blinkThreshold(),
        onCameraChange: (on) => controller.setCameraOn(on),
        onStatus: (m) => {
          if (CAMERA_PROBLEM.test(m)) controller.say(m, { title: "Camera", seconds: 8 });
        },
      });
    }
    return watcher.current;
  }, [controller, store, world]);

  const setCamera = useCallback(async (on: boolean) => {
    if (on && controller.paused) return;
    if (!on) {
      watcher.current?.setCamera(false);
      controller.setCameraOn(false);
      return;
    }
    const w = await ensureWatcher();
    if (w) w.setCamera(true);
    else controller.say("The camera module couldn't load.", { title: "Camera", seconds: 8 });
  }, [controller, ensureWatcher]);

  // Pause releases the camera immediately; resuming never turns it back on by itself.
  useEffect(() => controller.events.on("pauseToggled", (p) => {
    if (p) void setCamera(false);
  }), [controller, setCamera]);

  // --- screen ----------------------------------------------------------------
  const shareScreen = useCallback(async () => {
    let ok = false;
    try {
      ok = await soma.grantScreen();
    } catch {
      ok = false;
    }
    setScreenShared(ok);
    if (!ok) {
      controller.say("No screen shared, so SOMA can't look yet. Share one from settings whenever you like.", { title: "SOMA", seconds: 6 });
      return;
    }
    if (!somaStarted.current) {
      somaStarted.current = true;
      soma.start();
    }
    setIntro(false);
    greet();
  }, [soma, controller, greet]);

  const stopSharing = useCallback(() => {
    soma.revokeScreen();
    setScreenShared(false);
    controller.say("Stopped sharing. SOMA isn't looking at anything now.", { title: "SOMA", seconds: 4 });
  }, [soma, controller]);

  // --- float -----------------------------------------------------------------
  const setFloating = useCallback(async (on: boolean) => {
    if (!on) return pip.close();
    if (!(await pip.open())) controller.say("This browser can't float Pette (try Chrome or Edge).", { title: "Pette", seconds: 6 });
  }, [pip, controller]);

  // --- hotkeys (in this tab and in the floating window) -----------------------
  useHotkeys({
    look: () => controller.requestLook(),
    suggestions: () => suggester.toggle(),
    bubble: () => controller.toggleBubble(),
    camera: () => controller.toggleCameraView(),
    world: () => controller.toggleWorld(),
  }, [window, pip.window]);

  const onScrollGesture = useCallback((fn: (dy: number) => void) => controller.events.on("scroll", fn), [controller]);

  // Sandbox: simulated webcam events go through the real watcher (logging, presence, pinch, paging).
  const simulate = useCallback(async (ev: VisionEvent | { scroll: number }) => {
    const w = await ensureWatcher();
    if (!w) return;
    if (typeof ev === "string") w.onVisionEvent(ev);
    else w.onScroll(ev.scroll);
  }, [ensureWatcher]);

  const leaveIntro = () => {
    setIntro(false);
    greet();
  };

  const settings = (
    <SettingsMenu
      screenShared={screenShared} paused={st.paused} suggestions={enabled} bubble={st.bubbleEnabled} camera={st.cameraOn}
      cameraPicture={st.feedVisible} gestures={st.gesturesEnabled} floating={!!pip.root} floatSupported={pip.supported}
      world={st.worldVisible} remember={remember} folders={folders}
      onShareScreen={() => void shareScreen()} onStopSharing={stopSharing} onLook={() => controller.requestLook()}
      onPause={(p) => controller.setPaused(p)} onSuggestions={(on) => suggester.setEnabled(on)}
      onBubble={(on) => controller.setBubbleEnabled(on)} onCamera={(on) => void setCamera(on)}
      onCameraPicture={(on) => controller.setCameraVisible(on)} onGestures={(on) => controller.setGesturesEnabled(on)}
      onAddFolder={() => void suggester.addFolder()} onRemoveFolder={(id) => void suggester.removeFolder(id)}
      onFloat={(on) => void setFloating(on)} onWorld={(on) => controller.setWorldVisible(on)}
      onRemember={(on) => {
        store.enabled = on;
        setRemember(on);
      }}
      onClearMemory={() => {
        void store.clear().then(() => controller.say("Pette's memory cleared (SOMA's folders stay; remove them in settings).",
          { title: "Pette's memory", seconds: 6 }));
      }}
    />
  );

  const fake = sandbox && soma instanceof FakeSoma ? soma : null;
  const panel = dockPanel(orbPos);
  const orbRect = { x: orbPos.x, y: orbPos.y, width: ORB_BOX.width, height: ORB_BOX.height };
  let body: ReactNode;
  if (intro) {
    body = <FirstRun screenShared={screenShared} cameraOn={st.cameraOn} onShareScreen={() => void shareScreen()}
      onCamera={(on) => void setCamera(on)} onContinue={leaveIntro} />;
  } else {
    body = (
      <>
        <div className="pointer-events-none fixed inset-x-0 top-6 mx-auto max-w-md px-4 text-center text-xs text-zinc-500">
          {screenShared
            ? "SOMA is watching the screen you shared. Keep this tab open; float Pette to keep her on top of other apps."
            : "No screen shared yet: open settings (top right) to share one."}
        </div>
        <div className="fixed right-4 top-4 z-[56]">{settings}</div>
        {st.worldVisible && (
          <div className="fixed z-30" style={panel}>
            <WorldPanel snapshot={snapshot} folders={folders} onClose={() => controller.setWorldVisible(false)} onScrollGesture={onScrollGesture}
              live={{ state: st.status, detail: st.detail, label: statusLabel(st.status, st.detail), camera: st.cameraOn, gesture: st.gesture,
                present: !st.away, eyesClosed: st.eyesClosed, paused: st.paused, suggestions: enabled, screenShared, now: snapshot.updated }} />
          </div>
        )}
        {pip.root ? (
          <div className="fixed z-40 flex flex-col items-center gap-2 rounded-xl border bg-white/90 p-3 text-xs shadow"
            style={{ left: orbPos.x - 40, top: orbPos.y + 20 }}>
            <span>Pette is floating</span>
            <Button size="sm" variant="outline" onClick={pip.close}><PictureInPicture2 className="h-4 w-4" /> Bring her back</Button>
          </div>
        ) : (
          <>
            <Orb controller={controller} status={st.status} detail={st.detail} mood={st.mood} cameraOn={st.cameraOn}
              feedVisible={st.feedVisible} gesture={st.gesture} position={orbPos} onMove={setOrbPos} />
            <SpeechBubble controller={controller} bubble={st.bubble} progress={st.progress} orb={orbRect} />
          </>
        )}
      </>
    );
  }

  return (
    <div className="relative h-full bg-gradient-to-br from-zinc-50 via-white to-violet-50/40">
      {body}
      {fake && <Sandbox soma={fake} pette={pette} simulate={simulate} />}
      <Notice notice={st.notice} />
      {st.file && <FileViewer file={st.file} onClose={() => controller.closeFile()} />}
      {pip.root && createPortal(<Floating controller={controller} state={st} />, pip.root)}
    </div>
  );
}
