# SOMA offline evaluation (M4 / M5)

> **Every number on this page is SYNTHETIC.** The data comes from
> `soma.learning.synthetic_data`: non-sensitive, rule-generated GUI scenarios
> whose labels follow fixed templates. The numbers show that the training,
> evaluation, ablation and rollout machinery works and which components carry
> signal *in that generator*. They are **not** evidence of real-world quality
> or of benefit to real users. No consented human-labeled set exists yet
> (build plan section 12).

Measured on 2026-10-12, AMD Ryzen 9 8940HX, Windows 11, torch 2.14.1+cpu, CPU only
(see `docs/BENCHMARKS.md`).

## Reproduce

Run everything from `SOMA/` with the project venv. Checkpoints go to `checkpoints/` and
run logs go to `runs/`. Both folders are gitignored.

```powershell
# 1. Stage B training (configs/train.yaml: d_model=128, 224px, 8 epochs, seed 0)
.venv\Scripts\python -m scripts.train_ranker --config configs/train.yaml --n 2000 --out checkpoints/ranker.pt --seed 0
#    fast variant for CI / laptops
.venv\Scripts\python -m scripts.train_ranker --small --n 200 --epochs 2 --out checkpoints/ranker_small.pt

# 2. Offline evaluation on a fixed synthetic holdout (seed 1, different from training)
.venv\Scripts\python -m scripts.evaluate --checkpoint checkpoints/ranker.pt --n 500 --seed 1 --json runs/eval.json

# 3. M5 ablations (small config, one variable at a time)
.venv\Scripts\python -m scripts.ablations --json runs/ablations.json                 # seed 0, ~80 s
.venv\Scripts\python -m scripts.ablations --seeds 0 1 2 --json runs/ablations_3.json # ~4.5 min

# 4. Tests (train/eval scripts, ablations, M4 acceptance)
.venv\Scripts\python -m pytest tests/test_scripts_train_eval.py tests/test_ablations.py tests/test_m4_acceptance.py -q
```

Determinism: the data, the split (`seeded_split`), model init and loader shuffling are all
seeded. Re-running a command with the same seed reproduces the checkpoint bitwise on CPU
(`test_train_script_is_seeded`) and the same evaluation numbers. The single-seed ablation run
printed identical tables on two separate runs.

## 1. Training (`scripts.train_ranker`)

`configs/train.yaml`, 2,000 synthetic items (1,600 train / 400 val, 45.8% silence labels),
8 epochs, AdamW lr 3e-4, best validation epoch 4 (restored), 47.7 s on CPU.

| metric | train | val |
|---|---|---|
| top1 | 1.000 | 0.990 |
| top-2 | 1.000 | 0.998 |
| abstain precision | 1.000 | 0.995 |
| abstain recall | 1.000 | 0.984 |
| unwanted suggestion rate | 0.000 | 0.016 |
| missed suggestion rate | 0.000 | 0.005 |
| ECE | 0.001 | 0.006 |
| NLL | 0.001 | 0.033 |

The checkpoint `extra` records `data_kind: "synthetic"`, the seed, split sizes, label
distribution, train config and these metrics.

## 2. Evaluation vs baselines (`scripts.evaluate`)

Holdout: 500 items, seed 1 (training used seed 0), 42.2% silence labels.

| metric | full model | (1) no-memory, screen-only | (2) rules: type priority | rules: keyword (reference) |
|---|---|---|---|---|
| top1 | **0.994** | 0.854 | 0.130 | 1.000 |
| top-2 | 1.000 | 0.920 | 0.246 | 1.000 |
| abstain precision | 1.000 | 0.748 | 0.000 | 1.000 |
| abstain recall | 0.986 | 0.986 | 0.000 | 1.000 |
| unwanted suggestion rate | 0.014 | 0.014 | 1.000 | 0.000 |
| missed suggestion rate | 0.000 | 0.242 | 0.000 | 0.000 |
| ECE | 0.006 | 0.139 | 0.545 | 0.504 |
| NLL | 0.026 | 1.580 | 3.271 | 0.718 |

- **Baseline (1), no memory:** the same weights with every memory slot masked
  (`evaluate_no_memory`). The model misses every `prior_work` case that needs a retrieved
  memory, which is the missed-suggestion rate of 0.242.
- **Baseline (2), scripted type priority:** a screen-blind candidate-type heuristic. It always
  proposes the open candidate of the highest-priority type (error > file > next step > prior
  work > clarify), so it never stays silent: its unwanted rate is 1.0.
- **Keyword rules (reference only):** OCR-keyword rules that copy the synthetic generator's own
  templates. They score 1.000 here because the labels are rule-generated. This marks the
  ceiling of rule-generated data, not a fair baseline. With the small config's 512-bucket
  hashed vocabulary, hash collisions drop it to 0.794.

## 3. M5 ablations (`scripts.ablations`)

Small config (`d_model=64`, 64 px, 4 candidates). Every variant uses the same budget: 1,500
training items, 6 epochs, batch size 32, lr 1e-3, no early stopping, and a fixed eval set of
500 items (seed 1). Each variant changes one variable. The table shows mean +/- std over
training seeds 0, 1 and 2.

| group | variant | top1 | unwanted rate | ECE | NLL | prior top1 |
|---|---|---|---|---|---|---|
| memory | none | 0.937 +/- 0.045 | 0.072 +/- 0.059 | 0.054 +/- 0.038 | 0.333 +/- 0.236 | - |
| memory | retrieval_only | 0.998 +/- 0.002 | 0.005 +/- 0.004 | 0.002 +/- 0.001 | 0.003 +/- 0.002 | - |
| memory | gated | 0.998 +/- 0.003 | 0.005 +/- 0.007 | 0.002 +/- 0.002 | 0.008 +/- 0.010 | - |
| temporal | current_frame | 0.998 +/- 0.002 | 0.005 +/- 0.004 | 0.002 +/- 0.001 | 0.004 +/- 0.003 | - |
| temporal | temporal_context | 0.998 +/- 0.003 | 0.005 +/- 0.007 | 0.002 +/- 0.002 | 0.008 +/- 0.010 | - |
| ocr | visual_only | 0.921 +/- 0.050 | 0.066 +/- 0.043 | 0.040 +/- 0.014 | 0.201 +/- 0.109 | - |
| ocr | ocr_visual | 0.998 +/- 0.003 | 0.005 +/- 0.007 | 0.002 +/- 0.002 | 0.008 +/- 0.010 | - |
| temporal_kind | gru | 0.998 +/- 0.003 | 0.005 +/- 0.007 | 0.002 +/- 0.002 | 0.008 +/- 0.010 | - |
| temporal_kind | transformer | 0.996 +/- 0.003 | 0.006 +/- 0.006 | 0.003 +/- 0.002 | 0.018 +/- 0.016 | - |
| adaptation | base_no_adaptation | 0.319 +/- 0.001 | 0.750 +/- 0.002 | 0.679 +/- 0.001 | 9.119 +/- 0.491 | 0.998 +/- 0.003 |
| adaptation | frozen_base_online_ranker | 0.320 +/- 0.002 | 0.750 +/- 0.002 | 0.677 +/- 0.002 | 7.044 +/- 0.494 | 0.998 +/- 0.003 |
| adaptation | frozen_base_online_ranker_scaled | 0.689 +/- 0.112 | 0.260 +/- 0.082 | 0.109 +/- 0.049 | 0.727 +/- 0.207 | 0.998 +/- 0.003 |
| adaptation | finetuned_policy_head | 0.919 +/- 0.082 | 0.161 +/- 0.171 | 0.091 +/- 0.013 | 0.243 +/- 0.124 | 0.820 +/- 0.030 |

The variants are batch transforms in `soma/learning/ablations.py`, applied the same way at train
and eval time, so the model code does not change:

- **(a) Memory.** `none` masks every memory. `retrieval_only` masks memory in fusion: a
  `COMPARE_WITH_PRIOR_WORK` candidate exists only if retrieval returned something, and the mean
  memory embedding is added to that candidate. `gated` is the full gated cross-attention.
  - Memory matters: removing it costs about 6 top1 points.
  - On this data, retrieval-only ties gated attention. In the generator, memory's only role is
    "is there something to compare with", so a candidate filter captures all of it. The data
    cannot show any advantage of fused memory attention.
- **(b) Temporal.** `current_frame` masks all history. There is **no difference** between the
  two variants. This is expected and is a limitation: synthetic history is random noise and no
  label depends on it. The generator needs history-dependent scenarios before this ablation
  can say anything.
- **(c) OCR.** `visual_only` masks every OCR token and loses about 8 top1 points. The only
  visual cue is a weak colour band per scenario, so OCR carries most of the signal.
- **(d) GRU vs causal transformer.** These are equal within noise, which is consistent with (b):
  the temporal module has nothing to model here.
- **(e) Adaptation to a personal app.** The personal app's preferences are: silence instead of
  error explanations, and a clarifying question instead of a next step. Every variant starts
  from the same generic base. Personal data is 200 items. The online ranker gets *simulated
  explicit* feedback on the frozen base's own suggestions: a shown suggestion is accepted when
  it matches the label and rejected otherwise, and a correct alternative counts as accepted.
  - The **default ranker changes nothing**. Its bias is a clipped Beta log-odds of roughly
    log(n) magnitude, about 4.4 at most after 80 rejections. The base's logit margins are about
    16, so the bias cannot flip them. This is a deliberate safety property of the Stage C
    ranker, not a bug, but it means the default ranker cannot correct a confidently wrong base.
  - Scaling the log-odds by 4 (`scale=4, abstain_scale=4, max_bias=20`) raises personal top1
    from 0.32 to 0.69. It causes **zero forgetting**: prior top1 is unchanged, because biases
    are keyed per app.
  - Fine-tuning only the policy head on the personal data, with an equal amount of generic
    replay, reaches 0.92 personal top1. It forgets about 18 points of generic top1 (0.998 to
    0.820), because the head is shared across apps. With the M4 gate (`max_forgetting` 0.02 to
    0.05), `ControlledRollout` would **reject** this head-only update. The full-model rollout
    with 3:1 anchor replay in the M4 acceptance test passes the gate (section 4).

**Multi-scale vs single-scale visual features.** This ablation is not implemented. The visual
encoder has a single patch resolution (DECISIONS D7), and the synthetic images have no tiny
controls to test it on. See `docs/BENCHMARKS.md` (Future work).

## 4. M4 acceptance (`tests/test_m4_acceptance.py`)

Setup: the small config, a generic base trained on 600 items for 8 epochs, 160 personal feedback
examples, and a replay buffer of 600 generic anchors (ratio 3:1). Holdouts: 150 personal items
(seed 11) and 300 generic items (seed 12). Gates: improvement >= 0.05, forgetting <= 0.05,
unwanted-rate increase <= 0.05.

| model | personal top1 | generic (prior) top1 |
|---|---|---|
| fixed baseline (base, frozen) | 0.273 | 0.957 |
| scripted type-priority rule | 0.007 | - |
| deployed candidate (`ControlledRollout`) | **0.980** | 0.987 |

- The candidate is deployed. Its unwanted-rate change is within tolerance on both sets, and the
  decision JSON matches an independent re-evaluation.
- The deployed checkpoint also loads in `scripts.evaluate`, where it beats the scripted baseline.
- Sabotaged update (lr 5.0, clip 100): **rolled back** twice from the same base.
  - Both runs give identical decisions and metrics, and the live weights match the base
    bitwise, as does the saved `base.pt`.
  - After rollback, the restored model's evaluation is identical to the fixed baseline's.

## Test results

```
tests/test_scripts_train_eval.py  5 passed  (~8 s)
tests/test_ablations.py           8 passed  (~8 s)
tests/test_m4_acceptance.py       2 passed  (~10 s; ~25 s while another CPU job was running)
```
