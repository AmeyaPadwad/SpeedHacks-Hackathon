/**
 * Save / load named weight tensors through `tf.io` (port of `checkpoint.py`: weights + config + version,
 * never code). `dest` is a tfjs URL such as `indexeddb://soma-model` or `localstorage://x`, or any
 * `tf.io.IOHandler` (tests use an in-memory one).
 */
import * as tf from "@tensorflow/tfjs";

/** Where weights go / come from: a tfjs URL or an IOHandler. */
export type WeightsTarget = string | tf.io.IOHandler;

/** Payload format stored in `modelTopology`; anything else is rejected on load. */
export interface WeightsMeta {
  format: string;
  [key: string]: unknown;
}

function saveHandler(dest: WeightsTarget): tf.io.IOHandler {
  if (typeof dest !== "string") return dest;
  const handlers = tf.io.getSaveHandlers(dest);
  if (!handlers.length) throw new Error(`no tf.io save handler for ${dest}`);
  return handlers[0];
}

function loadHandler(src: WeightsTarget): tf.io.IOHandler | null {
  if (typeof src !== "string") return src;
  const handlers = tf.io.getLoadHandlers(src);
  return handlers.length ? handlers[0] : null;
}

/** Write `tensors` and `meta` to `dest`. */
export async function saveWeights(dest: WeightsTarget, tensors: tf.NamedTensorMap, meta: WeightsMeta): Promise<void> {
  const handler = saveHandler(dest);
  if (!handler.save) throw new Error("handler cannot save");
  const { data, specs } = await tf.io.encodeWeights(tensors);
  await handler.save({
    modelTopology: meta as unknown as tf.io.ModelArtifacts["modelTopology"],
    weightSpecs: specs,
    weightData: data,
    format: meta.format,
  });
}

/**
 * Read weights from `src`. Returns null when nothing is stored there, the payload is malformed, or its
 * `format` is not `format`. The caller owns (and must dispose) the returned tensors.
 */
export async function loadWeights(
  src: WeightsTarget, format: string,
): Promise<{ tensors: tf.NamedTensorMap; meta: WeightsMeta } | null> {
  const handler = loadHandler(src);
  if (!handler?.load) return null;
  let art: tf.io.ModelArtifacts;
  try {
    art = await handler.load();
  } catch {
    return null; // absent (indexeddb throws "Cannot find model") or unreadable
  }
  const meta = art.modelTopology as unknown as WeightsMeta | undefined;
  if (!meta || meta.format !== format || !art.weightSpecs || !art.weightData) return null;
  const data = Array.isArray(art.weightData) ? concatBuffers(art.weightData) : art.weightData;
  return { tensors: tf.io.decodeWeights(data, art.weightSpecs), meta };
}

function concatBuffers(parts: ArrayBuffer[]): ArrayBuffer {
  const total = parts.reduce((n, p) => n + p.byteLength, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(new Uint8Array(p), o);
    o += p.byteLength;
  }
  return out.buffer;
}

/** In-memory IOHandler (save then load), handy for tests and for copying a model. */
export function memoryIO(): tf.io.IOHandler & { artifacts: tf.io.ModelArtifacts | null } {
  const io = {
    artifacts: null as tf.io.ModelArtifacts | null,
    async save(a: tf.io.ModelArtifacts): Promise<tf.io.SaveResult> {
      io.artifacts = a;
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
    },
    async load(): Promise<tf.io.ModelArtifacts> {
      if (!io.artifacts) throw new Error("nothing saved");
      return io.artifacts;
    },
  };
  return io;
}
