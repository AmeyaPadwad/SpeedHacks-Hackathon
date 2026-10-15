/** Pette's webcam: presence, eye closure and hand gestures (tfjs, local only). */
export {
  AWAY_AFTER_S, BACK_AFTER_S, BLINK_HOLD_S, BLINK_OFF, BLINK_ON, BaselineMeter, DRAG_SMOOTH, DRAG_START_DY,
  PINCH_COOLDOWN_S, PINCH_HOLD_S, PINCH_ON, PINCH_REARM, POINT_COOLDOWN_S, POINT_HOLD_S, VisionPolicy,
  gestureName, observation, pinchRatio, pinchY, pointing,
  type BaselineSummary, type Landmark, type Observation, type PointSide,
} from "./policy";
export {
  EAR_CLOSED, EAR_OPEN, EYE_KEYPOINTS, blinkScores, centerSquare, earToBlink, eyeAspectRatio, handLandmarks,
  previewPoints, type PixelKeypoint,
} from "./landmarks";
export {
  CameraWorker, MAX_FPS, PREVIEW_SIZE, loadTfjsDetectors,
  type CameraOptions, type Cancel, type Detectors, type FaceEstimator, type HandEstimator,
} from "./camera";
export { Watcher, type CameraLike, type WatcherOptions } from "./watcher";
