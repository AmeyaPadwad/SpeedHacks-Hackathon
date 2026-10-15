/**
 * What the floating (Picture-in-Picture) window shows: Pette top-left with the bubble beside her. Anchored to
 * the top-left so she stays visible even when a browser opens the window larger than asked (Comet does) and
 * part of it is off-screen. It uses the same controller as the page, so clicks, pinches and cards behave the same.
 */
import type { PetteController, PetteState } from "../controller";
import { Orb } from "./Orb";
import { SpeechBubble } from "./SpeechBubble";

export function Floating({ controller, state }: { controller: PetteController; state: PetteState }) {
  return (
    <div className="flex h-full max-h-[340px] max-w-[540px] items-start gap-2 bg-transparent p-2">
      <div className="shrink-0">
        <Orb controller={controller} status={state.status} detail={state.detail} mood={state.mood} cameraOn={state.cameraOn}
          feedVisible={state.feedVisible} gesture={state.gesture} />
      </div>
      <div className="min-w-0 flex-1 overflow-y-auto">
        <SpeechBubble controller={controller} bubble={state.bubble} progress={state.progress} docked={false} />
      </div>
    </div>
  );
}
