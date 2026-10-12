# SOMA runtime benchmarks

These are measured per-stage latencies on one machine. **They are not latency promises.** They
cover one observation-to-suggestion pass at batch size 1 with `torch.inference_mode()`, using
synthetic screens and randomly initialised weights. Weights do not change the cost.

## Machine (2026-10-12)

| | |
|---|---|
| CPU | AMD Ryzen 9 8940HX with Radeon Graphics, 32 logical CPUs |
| OS | Windows 11 (10.0.26200) |
| Python / torch / numpy | 3.12.12 / 2.14.1+cpu (16 intra-op threads) / 2.5.2 |
| CUDA | not available (CPU-only torch build) |
| Device used | CPU |

Other processes were running on the machine (other build agents' tests), so treat P95 values
as noisy.

## Reproduce

```powershell
.venv\Scripts\python -m scripts.benchmark --iters 50 --warmup 5 --rapidocr-iters 10 --torch-profiler --json runs/benchmark.json
```

`soma/runtime/profiling.py` times each stage with `time.perf_counter`. On CUDA it synchronises
after each model stage. `--torch-profiler` adds a `torch.profiler` operator table for the full
forward pass. `hardware_info()` records the facts in the table above.

Stages:

- `render`: renders a 1280x800 synthetic screen. This stands in for capture.
- `ocr_synthetic`: ground-truth OCR.
- `ocr_rapidocr`: RapidOCR (onnxruntime) on the same frame, 10 samples.
- `parse`: `parse_screen`.
- `preprocess`: letterbox to tensor.
- `featurize`: OCR to hashed token tensors.
- `visual_encode`, `text_encode`, `screen_fusion`, `temporal`, `memory_fusion`, `policy`: the
  `SomaModel` sub-modules called one at a time.
- `retrieval_N<n>`: cosine top-K over a `VectorIndex` of N memories, with every id in the
  permission allow-list. This is the worst case of the permission-filtered path.
- `candidates`: `generate_candidates`.
- `gate`: `SafetyGate.decide`.
- `model_forward`: the full `SomaModel` forward in one call.

## Results: 50 timed passes after 5 warm-up passes

### dev config (`configs/dev.yaml`)

d_model=256, 448 px / 32 px patches (196 visual tokens), L=256 OCR tokens, T=8, K=8, A=8, GRU,
9.70 M parameters.

| stage | n | P50 ms | P95 ms |
|---|---|---|---|
| render (synthetic capture) | 50 | 10.91 | 17.57 |
| ocr_synthetic | 50 | 0.17 | 0.40 |
| ocr_rapidocr | 10 | 577.65 | 1375.66 |
| parse | 50 | 0.94 | 2.72 |
| preprocess | 50 | 1.80 | 4.13 |
| featurize | 50 | 0.41 | 1.01 |
| visual_encode | 50 | 3.35 | 4.40 |
| text_encode | 50 | 3.25 | 3.84 |
| screen_fusion | 50 | 6.20 | 7.67 |
| temporal | 50 | 0.96 | 1.17 |
| retrieval N=1,000 | 50 | 0.55 | 0.71 |
| retrieval N=10,000 | 50 | 5.63 | 6.66 |
| memory_fusion | 50 | 0.84 | 1.08 |
| candidates | 50 | 0.26 | 0.63 |
| policy | 50 | 0.38 | 0.47 |
| gate | 50 | 0.10 | 0.13 |
| **model_forward (all modules)** | 50 | **14.48** | **18.03** |

### small config (`small_model_config()`)

d_model=64, 64 px / 16 px patches, L=32, T=4, K=4, A=4, GRU, 0.36 M parameters.

| stage | n | P50 ms | P95 ms |
|---|---|---|---|
| render (synthetic capture) | 50 | 12.56 | 19.76 |
| ocr_synthetic | 50 | 0.20 | 0.43 |
| ocr_rapidocr | 10 | 604.25 | 1343.29 |
| parse | 50 | 1.05 | 3.11 |
| preprocess | 50 | 1.92 | 2.93 |
| featurize | 50 | 0.22 | 0.38 |
| visual_encode | 50 | 0.66 | 2.11 |
| text_encode | 50 | 0.49 | 0.94 |
| screen_fusion | 50 | 0.54 | 0.89 |
| temporal | 50 | 0.53 | 0.78 |
| retrieval N=1,000 | 50 | 0.44 | 0.62 |
| retrieval N=10,000 | 50 | 4.28 | 5.80 |
| memory_fusion | 50 | 0.52 | 0.96 |
| candidates | 50 | 0.29 | 0.67 |
| policy | 50 | 0.29 | 0.61 |
| gate | 50 | 0.10 | 0.14 |
| **model_forward (all modules)** | 50 | **2.59** | **5.29** |

## Observations

- **OCR dominates the cost.** RapidOCR takes about 0.6 s at P50 and up to about 1.4 s at P95 on
  a 1280x800 frame on this CPU. That is about 40 times the dev model's full forward pass.
  Change-detection gating and rate limiting (`capture.min_interval_s`) matter more than any
  model optimisation. OCR on changed regions only is the obvious next step.
- **Dev model forward:** about 14.5 ms P50 on CPU. In the `torch.profiler` table, transformer
  encoder layers account for about 71% of forward CPU time (`aten::_transformer_encoder_layer_fwd`)
  and attention for about 41%.
  - Screen fusion is the largest single module, at 6.2 ms.
  - The visual and text encoders take about 3.3 ms each.
- **Retrieval** scales linearly with N under the brute-force numpy cosine search: about 0.55 ms
  for 1k memories and about 5.6 ms for 10k. The faiss backend (`backend="faiss"`) was not
  measured.
- **Gate and candidate generation** are negligible (under 0.7 ms P95).
- **Training throughput:** `scripts.train_ranker` with `configs/train.yaml` (d_model 128, 224 px)
  took 47.7 s on CPU for 2,000 items over 8 epochs. The small-config ablations take about 6 to
  9 s per variant (see `docs/EVALUATION.md`).

## Future work

- **Single-scale vs multi-scale visual features (M5).** Not implemented. `VisualEncoder` uses one
  patch size (DECISIONS D7). A cheap hook would be a second patch-embedding branch at half the
  patch size with tokens concatenated before fusion, behind a `ModelConfig` flag. The
  comparison needs screens with tiny controls, which the current synthetic generator does not
  produce. Without them the ablation would show nothing.
- GPU numbers: no CUDA device on this machine.
- Pretrained `torchvision_vit_b_16` backbone latency (`vision_backbone`): not measured, because
  it needs a weight download.
- Real desktop capture (`soma.capture.screen`) and the end-to-end runtime loop: profile once the
  runtime package lands, on real apps rather than synthetic renders.
- faiss retrieval backend, and retrieval through `Retriever` with SQLite permission filtering
  included.
