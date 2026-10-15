/**
 * What the floating (Picture-in-Picture) window shows: the bubble above Pette, laid out in a column. It uses
 * the same controller as the page, so clicks, pinches and cards behave the same.
 */
import type { PetteController, PetteState } from "../controller";
import { Orb } from "./Orb";
import { SpeechBubble } from "./SpeechBubble";

export function Floating({ controller, state }: { controller: PetteController; state: PetteState }) {
  return (
    <div className="flex h-full flex-col items-center justify-end gap-1 bg-transparent p-2">
      <div className="w-full flex-1 overflow-y-auto">
        <div className="flex min-h-full items-end">
          <SpeechBubble controller={controller} bubble={state.bubble} progress={state.progress} docked={false} />
        </div>
      </div>
      <Orb controller={controller} status={state.status} detail={state.detail} mood={state.mood} cameraOn={state.cameraOn}
        feedVisible={state.feedVisible} gesture={state.gesture} />
    </div>
  );
}
