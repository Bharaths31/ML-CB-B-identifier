# Cattle & Buffalo Breed Classifier — Context Database

> **Purpose**: This document is the single source of truth for any AI model, agent, or developer working on this project. It provides complete architectural, implementation, and operational context with near-zero token cost (read only the sections you need).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Directory Map](#3-directory-map)
4. [Model Architecture](#4-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Data Pipeline](#6-data-pipeline)
7. [Export & Deployment](#7-export--deployment)
8. [Configuration Reference](#8-configuration-reference)
9. [API Reference](#9-api-reference)
10. [Common Operations](#10-common-operations)
11. [Known Constraints & Gotchas](#11-known-constraints--gotchas)
12. [Changelog](#12-changelog)

---

## 1. Project Overview

| Field | Value |
|---|---|
| **Goal** | Classify images of Indian cattle (57 breeds) and buffalo (18 breeds) using a lightweight, mobile-deployable CNN |
| **Model** | EfficientNet-Lite{2,4} backbone + CBAM/SE attention + 3-head classifier (binary + cattle + buffalo) + training-only projection head for SupCon |
| **Stack** | Python 3.11+, PyTorch >= 2.1.0, Custom PyTorch inference GUI (test_model.py) |
| **Training** | 2-phase default: all-heads warmup → multi-task fine-tune (+ optional QAT phase 3). Augmentation/mixing **OFF by default** |
| **Deployment** | ONNX, INT8 (converter-side PTQ), float16, or portable self-contained folder |
| **Outputs** | Timestamped per run (`DD-MM-YYYY-HH-MM`) — never overwrites previous results |
| **Dataset** | Multi-source Kaggle datasets: `algsoch` & `atharvadarpude` merged into `data/raw/cattle/<breed>/*.jpg` + `data/raw/buffalo/<breed>/*.jpg` |

### Key Numbers

- **75 total breeds**: 57 cattle + 18 buffalo
- **Input size**: 260×260 RGB
- **Feature dim**: 1280 (from EfficientNet head)
- **Backbone params**: ~6M (lite2), ~13M (lite4)

---

## 2. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA FLOW                                │
│                                                              │
│  data/raw/{cattle,buffalo}/<breed>/*.jpg                     │
│        ↓                                                     │
│  data_pipeline.prepare_splits() → data/splits/*.csv          │
│        ↓                                                     │
│  CattleBuffaloDataset → DataLoader (CutMix/MixUp collate)   │
│        ↓                                                     │
│  ┌─ EfficientNet-Lite backbone (stages 0..6) ─┐             │
│  │   stem → [stage0..3] → CBAM → [stage4..6] → head        │
│  └──────────────────────────────────────────────┘            │
│        ↓ AdaptiveAvgPool2d(1) → flatten(1)                   │
│        ↓ (1280-dim pooled feature vector)                    │
│  ┌─────┼─────────┬──────────────┬───────────────────┐        │
│  ↓     ↓         ↓              ↓                   ↓        │
│ binary_head  cattle_head  buffalo_head   projection_head     │
│  (→2)        (→57)        (→18)          (→128, train-only)  │
│        ↓                                                     │
│  masked_loss: w_bin*CE_bin + w_cat*CE_cat + w_buf*CE_buf     │
│               (+ τ·log(prior) logit adjustment on breed CE)  │
│               (+ λ·SupCon(projection embedding))             │
│        ↓                                                     │
│  outputs/checkpoints/<backbone>_phase{1,2,3}_best.pt         │
│        ↓                                                     │
│  outputs/export/portable/<backbone>_*/  (self-contained)     │
└─────────────────────────────────────────────────────────────┘
```

### Inference Flow

```
Image → Resize(260) → CenterCrop(260) → ToTensor()
     → model.forward() → {binary, cattle, buffalo, features, embedding}
     → soft routing: p(species)·softmax(head) over all 75 breeds → top-k
```

`model.predict(x)` returns the soft-routed 75-class distribution. `test_model.py`
and both Flutter engines use the same `p(species)·softmax(head)` mixture instead
of a hard binary argmax, so a confident breed head can still win when the binary
head is ambiguous (removes two-stage routing error propagation).

---

## 3. Directory Map

```
Mini Project/
├── src/                        # Core ML package (run as python -m src.<module>)
│   ├── __init__.py
│   ├── config.py               # All hyperparameters & paths
│   ├── data_pipeline.py        # Dataset, splits, augmentation, DataLoaders
│   ├── model.py                # BreedClassifier (backbone + attention + heads + projection head)
│   ├── cbam.py                 # CBAM & SE attention modules
│   ├── efficientnet_lite.py    # EfficientNet-Lite{2,4} architecture
│   ├── train.py                # 2-phase training (logit adj, SupCon, EMA, adaptive weights)
│   ├── metrics.py              # evaluate_epoch() — per-head acc, F1, macro-F1, soft-routed top1, shot buckets
│   ├── run_utils.py            # run ids, timestamped paths, unique_path, find_latest_checkpoint, resolve_checkpoint
│   ├── evaluate.py             # Full evaluation with confusion matrices (timestamped outputs)
│   ├── export.py               # ONNX, INT8, float16, portable export
│   └── verify.py               # Quick architecture sanity check
├── colab/
│   ├── cattle_buffalo_trainer.py   # Colab training script (percent-format)
│   ├── cattle_buffalo_trainer.ipynb # Jupyter notebook (auto-generated)
│   ├── cattle_buffalo_tester.py    # Colab testing script for large-scale evaluation
│   ├── cattle_buffalo_tester.ipynb # Colab testing notebook (auto-generated)
│   ├── convert_to_notebook.py  # .py → .ipynb converter
│   └── README.md               # Colab setup instructions
├── data/
│   ├── raw/                    # Source images: raw/{cattle,buffalo}/<breed>/*.jpg
│   ├── splits/                 # Generated: train.csv, val.csv, test.csv, *_classes.json
│   └── dataset_inventory/      # Per-dataset JSON: breed, species, count, per-image resolution
├── outputs/
│   ├── checkpoints/            # Training checkpoints (*.pt)
│   ├── export/                 # ONNX/INT8/float16 exports
│   │   └── portable/           # Self-contained model bundles
│   └── metrics/                # Evaluation JSON + confusion matrix PNGs
├── scripts/                    # Colab archive creators, app asset prep
├── create_training_zip.py      # Creates lightweight standalone training package
├── create_test_eval_zip.py     # Creates zip of test split images for evaluation
├── local_train.py              # Fully automated local training pipeline (setup → train → export)
├── test_model.py               # Standalone PyTorch model testing GUI server (http://localhost:8501)
├── setup.sh                    # Shell script helper for environment setup
├── setup_venv.py               # Automated virtual environment setup script
├── .gitignore                  # Git ignore rules (includes outputs, venv, cache)
├── efficientnet_lite{2,4}.pth  # Pretrained ImageNet backbone weights
├── requirements.txt            # Python dependencies
```

---

## 4. Model Architecture

### BreedClassifier (`src/model.py`)

```python
class BreedClassifier(nn.Module):
    backbone: EfficientNetLite     # stem → 7 MBConv stages → head(1280ch)
    attention: CBAM | SEBlock      # inserted after stage 3 (88ch lite2 / 112ch lite4)
    avg_pool: AdaptiveAvgPool2d(1)
    binary_head: Linear(1280→256→2)
    cattle_head: Linear(1280→512) + BN + ReLU + Drop(0.3) + Linear(512→256) + BN + ReLU + Drop(0.2) + Linear(256→57)
    buffalo_head: Linear(1280→512) + BN + ReLU + Drop(0.3) + Linear(512→256) + BN + ReLU + Drop(0.2) + Linear(256→18)
    projection_head: Linear(1280→1280) + ReLU + Linear(1280→128)   # training-only (SupCon)
```

### Forward Path

1. `forward_features(x)`: backbone stages 0..3 → CBAM → stages 4..6 → head → pool → flatten
2. `forward(x)`: features → 3 parallel heads + projection head → dict{binary, cattle, buffalo, features, embedding}
3. `predict(x)` (`@torch.no_grad()`): returns the soft-routed combined 75-class distribution `[p(species=0)·softmax(cattle), p(species=1)·softmax(buffalo)]`

The `projection_head` and `embedding` output are consumed only by the
supervised-contrastive (SupCon) loss; they are never exported and are tolerated
as missing keys when loading legacy checkpoints (`src/export._load_model`).

### Freeze/Unfreeze Methods

- `freeze_backbone()`: freezes backbone + attention
- `freeze_all()` / `unfreeze_all()`: all parameters
- `backbone_eval()` / `backbone_train()`: BatchNorm mode control

### EfficientNet-Lite (`src/efficientnet_lite.py`)

- MBConv blocks with `relu6` activation (no SE in lite variant)
- `forward_until(x, stage_idx)` / `forward_from(x, stage_idx)`: split forward for attention insertion
- Weight loading: `load_backbone_weights()` — strict=False, allows missing `fc.*`

### Attention (`src/cbam.py`)

- **CBAM**: ChannelAttention(avg+max pool → MLP → sigmoid) → SpatialAttention(avg+max concat → conv → sigmoid)
- **SE**: AdaptiveAvgPool → FC reduce → ReLU → FC expand → Sigmoid

---

## 5. Training Pipeline

### Two-Phase Training (`src/train.py`) — QAT is opt-in

| Phase | What | Frozen | LR | Epochs | Loss Weights |
|---|---|---|---|---|---|
| 1 | All-heads warmup | backbone + attention | 3e-3 | 8 | bin=0.15, cat=0.50, buf=0.35 |
| 2 | Multi-task fine-tune + EMA (+ optional distillation) | nothing | 2e-4 (warmup+cosine) | 80 | bin=0.15→0.05, cat=0.50→0.55, buf=0.35→0.40 |
| 3 | QAT — **opt-in** via `--include-qat` | nothing | 5e-6 | 10 | bin=0.15, cat=0.50, buf=0.35 |

**Adaptive loss weights**: once `binary_acc ≥ BINARY_SATURATION_ACC` (0.95) the
weights automatically switch to `LOSS_WEIGHT_*_FINAL` (`0.05/0.55/0.40`) from the
next epoch, reallocating the loss budget from the saturated binary head to the
breed heads.

Default is 2 phases. Mobile INT8 comes from **converter-side PTQ**
(`src.export --mode tflite / onnx-int8`), not from PyTorch QAT. The optional
QAT phase uses per-tensor observers, starts from the best phase-2 EMA
checkpoint, and saves a timestamped `<backbone>_quantized_<runid>.pt` — but it
measurably degrades accuracy (≈ −9 pts on the recorded run), so PTQ is preferred.

### Knowledge Distillation (`--teacher <ckpt>`)

Train a bigger teacher first (e.g. `--backbone lite4`), then distill into the
student (`--teacher outputs/checkpoints/lite4_phase2_best_<runid>.pt`). Loss:
`(1-α)·masked_hard_CE + α·T²·masked_KL(teacher‖student)` with α=0.7, T=4.0
applied to all three heads. The teacher runs frozen in eval mode on the same
augmented batch; the student's architecture/size/latency are unchanged.

### EMA (phase 2)

`run_epoch` maintains an exponential moving average of the weights **and**
BatchNorm buffers (running mean/var EMA'd, `num_batches_tracked` copied).
The update runs **once per optimizer step** (not per micro-batch), with warm-up
`decay_t = min(EMA_DECAY, (1+t)/(10+t))` (`EMA_WARMUP=True`). At phase start it
prints steps/epoch, total optimizer steps and the EMA time constant, and warns
when the time constant exceeds `EMA_WARN_FRAC` (25%) of the phase. Every eval
logs **both** raw and EMA metrics and saves whichever scores better.

### Long-tail handling (exactly ONE mechanism)

- **Sampler** (default and only mechanism): effective-number-of-samples
  weighting (`SAMPLER_BETA=0.99`) — softens pure inverse-frequency oversampling
  of 5-image breeds. Counts are keyed on `(species, breed)` because `bargur`
  exists under both species.
- **Logit adjustment is OFF by default** (`LOGIT_ADJUST=False`). The sampler
  already rebalances every batch; enabling logit adjustment on top double-corrects
  and over-predicts rare breeds at inference (the 2026-09-21 regression). If
  enabled with `--logit-adjust`, the prior is computed from the effective
  **sampled** distribution (`LOGIT_ADJUST_PRIOR="sampled"`), never raw counts;
  startup prints the active mechanism and prior max/min ratio.
- **Rare-class mixing guard** (`RARE_CLASS_THRESHOLD=30`): breeds below 30 train
  images are excluded from CutMix/MixUp via a per-sample `keep` mask.
- **SupCon feature learning** (`CONTRASTIVE_WEIGHT=0.2`,
  `CONTRASTIVE_TEMPERATURE=0.1`): supervised-contrastive loss on the 128-d
  projection embedding; skipped on mixed batches (soft labels).
- Binary head: per-batch species re-weighting (`BALANCE_BINARY_HEAD=True`)
  neutralizes the ~57:18 breed-count species prior.

### Mixing & augmentation (opt-in; OFF by default)

Heavy stochastic augmentation measurably hurts fine-grained breed
identification on a long tail, so everything is opt-in per run:
- `MIX_ENABLED=False`; enable CutMix/MixUp with `--mix` (α=0.4/0.2, p=0.25).
- Mixing pairs **within the same species** (`MIX_SAME_SPECIES=True`,
  `_pairing_perm`) so binary labels stay one-hot and breed targets stay proper
  distributions.
- `MIX_OFF_LAST_FRAC=0.15` disables mixing for the last 15% of phase 2.
- `--flip`, `--color-jitter`, `--randaugment`, `--rrc` enable the individual
  train transforms; `--augment-all` enables everything incl. mixing. With all
  off, the train transform `_train_transform({})` equals `_eval_transform()`.

### SOTA Optimizations

- **Optimizer**: AdamW (weight_decay=1e-2) — decoupled weight decay for better generalization
- **Label smoothing**: 0.05 — prevents overconfident predictions
- **LR schedule**: Linear warmup (3 epochs) → Cosine annealing (Phase 2)
- **Gradient accumulation**: 2 steps (effective batch = 128 with batch_size=64)
- **CutMix/MixUp**: applied 50% of steps (α=1.0 / α=0.3), on GPU inside the loop

### CUDA Optimization

- `cudnn.benchmark = True` — auto-tune convolution algorithms
- `TF32` enabled for Ampere+ GPUs
- `torch.amp.autocast("cuda")` + `GradScaler` for mixed-precision
- `pin_memory=True` on all DataLoaders when CUDA available
- `persistent_workers=True` on DataLoaders
- `non_blocking=True` on `.to(device)` transfers
- `optimizer.zero_grad(set_to_none=True)` — faster than fill with zeros
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)`

### CPU Fallback

All CUDA optimizations gracefully skip on CPU. AMP scaler is `None`, cudnn settings are not touched.

### Loss Function

- `soft_ce`: soft cross-entropy supporting CutMix/MixUp label mixing, plus optional logit adjustment (`pred + τ·log_prior`)
- `masked_loss`: binary CE always active, cattle/buffalo CE only on matching species (masked by cattle_mask/buffalo_mask); adds the SupCon term
- `supervised_contrastive_loss`: SupCon (Khosla et al. 2020) over the projection embedding; anchors without a positive are ignored
- `masked_kd_loss`: `(1−α)·masked_loss + α·T²·masked_KL` + SupCon
- **Checkpoint selection** uses `BEST_METRIC="blended_score"` = `0.5·macro-F1 + 0.5·combined_top1_soft` (pure macro-F1 was too noisy with 1–2 val images per rare breed; pure top-1 is dominated by the ~10 large breeds). Each eval also logs `acc_fewshot`/`acc_mediumshot`/`acc_manyshot` and `pred_hist_entropy`.

### Smoke Test (`--smoke-test`)

Creates a mini-dataset of 5 images per breed using `prepare_smoke_splits()`:
- Samples from full dataset, not random batches
- Real training signal on actual breed images
- 1 epoch per phase
- Auto-exports portable model after training

### Half-Data Mode (`--half-data`)

Creates a reduced dataset using 50% of images per breed via `prepare_half_splits()`:
- Deterministic sampling (seed=42) for reproducibility
- Same 70/15/15 stratified split (with long-tail minimums) on the sampled subset
- Class maps include ALL breeds — model architecture stays identical to full training
- Ideal for faster iteration on local machines with limited VRAM

### Quarter-Data Mode (`--quarter-data`)

Creates a reduced dataset using 25% of images per breed via `prepare_quarter_splits()`:
- Deterministic sampling (seed=42) for reproducibility
- Same 70/15/15 stratified split (with long-tail minimums) on the sampled subset
- Class maps include ALL breeds — model architecture stays identical to full training
- Fastest local training mode (~4x speedup)

### Auto-Export (timestamped, non-overwriting)

After training completes, automatically creates a portable export in
`outputs/export/portable/<backbone>_<...>_<runid>/` containing:
- `model.pt` — checkpoint with state_dict
- `cattle_classes.json`, `buffalo_classes.json` — label maps
- `model_info.json` — architecture metadata + usage instructions

**All outputs are timestamped by default** (`TIMESTAMP_OUTPUTS=True`,
`RUN_ID_FORMAT="%d-%m-%Y-%H-%M"` → `DD-MM-YYYY-HH-MM`; colons are illegal on
Windows). Checkpoints are `<backbone>_phase{N}_best_<runid>.pt`; exports/metrics
carry the same run id. `src/run_utils.find_latest_checkpoint()` discovers the
newest checkpoint, so `src.export` / `src.evaluate` / `src.parity_check` /
`local_train.py` need no path argument. `unique_path()` guards explicit
`--run-tag` reuse so nothing is overwritten.

---

## 6. Data Pipeline

### Dataset Layout

```
data/raw/
├── cattle/
│   ├── amritmahal/  (N .jpg files)
│   ├── ayrshire/
│   └── ... (57 breeds)
└── buffalo/
    ├── alambadi/
    ├── banni/
    └── ... (18 breeds)
```

### Split Strategy

- **Full training**: 70/15/15 stratified per `(species, breed)` with long-tail minimums — every breed with ≥3 images gets ≥1 val and ≥1 test image (grouping is by species+breed because `bargur` exists under both)
- **Smoke test**: 5 images/breed → 60/20/20 split (tiny but real)

### Augmentation (flip + RRC ON by default)

- **Train (default)**: Resize(288) → RandomResizedCrop(260, scale=(0.8,1.0), ratio=(0.92,1.08)) → RandomHorizontalFlip → Normalize. Flip/RRC don't mix content between breeds, so they are on by default.
- **Train (opt-in)**: `--color-jitter` (hue capped at 0.02), `--randaugment`, `--augment-preset light`, `--breed-aug` (per-breed `BREED_AUG_POLICY`; coat-colour breeds skip colour jitter), `--pad-to-square` (resize long side + pad, keeps full-body side profiles).
- **`--no-augment`** forces every transform off; then the train transform equals eval.
- **Eval**: Resize(260) → CenterCrop(260) → ToTensor() → Normalize (`EVAL_MATCH_TRAIN_RESOLUTION=True` switches to Resize(288)+CenterCrop(260); `--pad-to-square` uses pad mode). Train/eval chains are printed at startup and a scale-mismatch warning is emitted when RRC is on but eval is not aligned.
- **Batch mixing (opt-in, `--mix`)**: 25% chance of CutMix(α=0.4) or MixUp(α=0.2); partners drawn **within the same species**; rare breeds kept unmixed; disabled for the last `MIX_OFF_LAST_FRAC` (15%) of phase 2. Gated on the explicit `training=True` flag (not the tqdm desc), and the eval log prints `mix=on (n/N)` only when mixing actually ran.
- **Caching**: `CACHE_IMAGES` stores raw JPEG bytes in RAM, preventing OOMs while bypassing disk I/O.

### Label Encoding

- Binary: one-hot [cattle=0, buffalo=1]
- Breed: one-hot over species-specific classes
- Masks: `cattle_mask=1.0` if cattle, else `buffalo_mask=1.0`

### Sampling

- `WeightedRandomSampler` with effective-number-of-samples weights
  (`SAMPLER_BETA=0.99`), keyed on `(species, breed)`
- `compute_class_priors(source="sampled")` returns the effective **sampled**
  log-prior (used only if `--logit-adjust` is enabled; absent classes get the
  smallest present prior, never NaN)

---

## 7. Export & Deployment

### Export Modes (`python -m src.export`)

All artifact names are suffixed with the run id (`<backbone>_<runid>_...`), so
repeated exports never overwrite. Labels (`labels_*.txt`) are deterministic.

| Mode | Output | Notes |
|---|---|---|
| `onnx` | `<backbone>_<runid>_fp32.onnx` | opset 13, static or dynamic batch; caller-normalized input (test_model.py compatible) |
| `tflite` | `<backbone>_<runid>_fp32.tflite` + `<backbone>_<runid>_int8.tflite` + `labels_*.txt` | ONNX → onnx2tf → TFLite; INT8 is full-integer PTQ with float32 [0,1] I/O, normalization baked in; auto-copies into `flutter_app/assets/models/` |
| `onnx-int8` | `<backbone>_<runid>_mobile_int8.onnx` + `labels_*.txt` | QDQ static quantization (per-channel weights) for ONNX Runtime Mobile; input [0,1], normalization baked in |
| `float16` | `<backbone>_<runid>_float16.pt` | TorchScript traced |
| `portable` | `portable/<backbone>_<tag>/` folder | Self-contained: model + labels + info (unique dir per run) |

Default checkpoint discovery uses `run_utils.find_latest_checkpoint()`; pass
`--checkpoint` to override and `--run-tag` to name the run.

`--mode int8` (x86 PTQ) was **removed** — it failed conversion
(`Unsupported qscheme: per_channel_affine`) and its artifacts were unusable
on Android. Mobile artifacts expect input RGB in [0,1] (the Flutter app's
`pixel/255` preprocessing is now exactly correct, zero app changes).
Always run the parity gate after exporting:
`python -m src.parity_check --checkpoint <pt> --tflite <tflite> --onnx-int8 <onnx>`.

### Portable Export Structure

```
outputs/export/portable/<backbone>_phase2_best/
├── model.pt                 # torch checkpoint {state_dict: ...}
├── cattle_classes.json      # {"amritmahal": 0, "ayrshire": 1, ...}
├── buffalo_classes.json     # {"alambadi": 0, "banni": 1, ...}
└── model_info.json          # {backbone, image_size, usage, exported_at}
```

---

## 8. Configuration Reference

### `src/config.py` — All Constants

| Constant | Value | Purpose |
|---|---|---|
| `IMAGE_SIZE` | 260 | Input image dimension |
| `NUM_CATTLE_BREEDS` | 57 | Expected cattle breed count |
| `NUM_BUFFALO_BREEDS` | 18 | Expected buffalo breed count |
| `NUM_BREEDS_TOTAL` | 75 | Total breed count across species |
| `CBAM_AFTER_STAGE` | 3 | Attention insertion point |
| `FEATURE_DIM` | 1280 | Backbone output dimension |
| `BINARY_DIM` | 256 | Binary head hidden dim |
| `BREED_DIM` | 512 | Breed head hidden dim |
| `DROPOUT` | 0.3 | Breed head dropout |
| `BATCH_SIZE` | 64 | Default batch size |
| `NUM_WORKERS` | 4 | DataLoader workers |
| `TRAIN/VAL/TEST_RATIO` | 0.70/0.15/0.15 | Split ratios (≥1 val/test per breed) |
| `PHASE{1,2,3}_EPOCHS` | 8/80/10 | Training epochs (phase 3 is opt-in) |
| `PHASE{1,2,3}_LR` | 3e-3/2e-4/5e-6 | Learning rates |
| `LOSS_WEIGHT_*` | 0.15/0.50/0.35 → 0.05/0.55/0.40 | Auto-switch after the binary head saturates |
| `WEIGHT_DECAY` | 1e-2 | AdamW weight decay |
| `LABEL_SMOOTHING` | 0.05 | Label smoothing factor |
| `WARMUP_EPOCHS` | 3 | Linear warmup epochs (phase 2) |
| `GRADIENT_ACCUMULATION_STEPS` | 2 | Grad accum steps |
| `KD_ALPHA` | 0.7 | Distillation blend: (1-α)·hard CE + α·T²·KL(teacher‖student) |
| `KD_TEMPERATURE` | 4.0 | Distillation temperature |
| `SAMPLER_BETA` | 0.99 | Effective-number-of-samples sampler β (softened) |
| `LOGIT_ADJUST` / `LOGIT_ADJUST_TAU` / `LOGIT_ADJUST_PRIOR` | False / 1.0 / `sampled` | Logit adjustment — **OFF by default** (sampler is the single mechanism); prior from the effective sampled distribution |
| `CONTRASTIVE_WEIGHT` / `CONTRASTIVE_TEMPERATURE` | 0.2 / 0.1 | SupCon loss on the projection embedding |
| `PROJECTION_DIM` | 128 | Projection-head output dim (training-only) |
| `RARE_CLASS_THRESHOLD` | 30 | Breeds below this many train images are excluded from CutMix/MixUp |
| `BINARY_SATURATION_ACC` | 0.95 | Switch to `LOSS_WEIGHT_*_FINAL` when reached |
| `LOSS_WEIGHT_BINARY_FINAL` / `LOSS_WEIGHT_CATTLE_FINAL` / `LOSS_WEIGHT_BUFFALO_FINAL` | 0.05 / 0.55 / 0.40 | Post-saturation loss weights |
| `BEST_METRIC` / `BEST_METRIC_MACRO_WEIGHT` / `BEST_METRIC_TOP1_WEIGHT` | `blended_score` / 0.5 / 0.5 | Checkpoint selection = 0.5·macro-F1 + 0.5·soft top-1 |
| `SHOT_FEW_MAX` / `SHOT_MEDIUM_MAX` | 30 / 100 | few/medium/many-shot diagnostic buckets |
| `BALANCE_BINARY_HEAD` | True | Per-batch species re-weighting of binary CE |
| `EMA_DECAY` / `EMA_WARMUP` / `EMA_WARN_FRAC` | 0.999 / True / 0.25 | EMA once per optimizer step, warm-up `(1+t)/(10+t)` |
| `SMOKE_SAMPLES_PER_BREED` | 5 | Images per breed in smoke test |
| `HALF_DATA_RATIO` | 0.5 | Fraction of images/breed for half-data mode |
| `QUARTER_DATA_RATIO` | 0.25 | Fraction of images/breed for quarter-data mode |
| `EVAL_EVERY_PHASE{1,2,3}` | 1 / 2 / 2 | Evaluation frequencies (epochs) |
| `CACHE_IMAGES` | False | RAM image caching toggle |
| `SPECIES_LABELS` | `{"cattle": 0, "buffalo": 1}` | Species integer mapping |
| `MIX_ENABLED` | False | CutMix/MixUp master switch (opt-in) |
| `CUTMIX_ALPHA` | 0.4 | CutMix beta distribution α (when enabled) |
| `MIXUP_ALPHA` | 0.2 | MixUp beta distribution α (when enabled) |
| `CUTMIX_MIXUP_PROB` | 0.25 | Per-step probability of CutMix or MixUp (when enabled) |
| `MIX_SAME_SPECIES` / `MIX_OFF_LAST_FRAC` | True / 0.15 | Pair within species; disable mixing for the last 15% of phase 2 |
| `AUG_HORIZONTAL_FLIP` / `AUG_RANDOM_RESIZED_CROP` | **True** | On by default (no content mixing between breeds) |
| `AUG_COLOR_JITTER` / `AUG_RANDAUGMENT` | False | Off by default (can erase breed identity) |
| `RRC_SCALE` / `RRC_RATIO` | (0.8,1.0) / (0.92,1.08) | Near-square crop preserves body proportions |
| `COLOR_JITTER_*` / `ALLOW_HUE` | 0.15/0.15/0.1/0.02 / False | Hue capped unless `--allow-hue` |
| `BREED_AUG_POLICY` / `COAT_COLOUR_BREEDS` | coat breeds skip jitter | Per-breed augmentation |
| `EVAL_PAD_TO_SQUARE` / `TRAIN_PAD_TO_SQUARE` | False | Pad-to-square (keeps full frame) |
| `CBAM_IDENTITY_INIT` | True | Attention identity at init (bug fix) |
| `TRAIT_WEIGHT` / `COSINE_HEAD` / `HARD_PAIRS` / `DEDUP_SPLITS` | 0.0 / False / None / False | New opt-in features (see Phase D/E) |
| `RANDAUGMENT_OPS` / `RANDAUGMENT_MAGNITUDE` | 2 / 5 | RandAugment settings (when enabled) |
| `TRAIN_RESIZE` / `EVAL_MATCH_TRAIN_RESOLUTION` | 288 / False | Crop-source scale; eval can match train resolution |
| `TIMESTAMP_OUTPUTS` / `RUN_ID_FORMAT` | True / `%d-%m-%Y-%H-%M` | Timestamped non-overwriting outputs (DD-MM-YYYY-HH-MM) |

### Path Constants

| Constant | Path |
|---|---|
| `RAW_DATA_DIR` | `data/raw/` |
| `SPLIT_DIR` | `data/splits/` |
| `CHECKPOINT_DIR` | `outputs/checkpoints/` |
| `EXPORT_DIR` | `outputs/export/` |
| `PORTABLE_EXPORT_DIR` | `outputs/export/portable/` |
| `TFLITE_APP_ASSETS_DIR` | `flutter_app/assets/models/` |
| `METRICS_DIR` | `outputs/metrics/` |

---

## 9. API Reference

### `python local_train.py` — Automated Pipeline

#### Data Mode (mutually exclusive)
| Flag | Description |
|---|---|
| `--half-data` | 50% of images per breed (seed=42) |
| `--quarter-data` | 25% of images per breed (seed=42) |
| `--smoke-test` | 5 imgs/breed, 1 epoch per phase |
| `--full-data` | All images (explicit default) |

#### Model Config
| Flag | Default | Description |
|---|---|---|
| `--backbone {lite2,lite4}` | `lite2` | Backbone architecture |
| `--attention {cbam,se}` | `cbam` | Attention module |

#### Training Overrides
| Flag | Default | Description |
|---|---|---|
| `--include-qat` | off | Enable Phase 3 QAT |
| `--phase1-epochs N` | 5 | Phase 1 epoch count |
| `--phase2-epochs N` | 40 | Phase 2 epoch count |
| `--phase3-epochs N` | 10 | Phase 3 epoch count |
| `--num-workers N` | 4 | DataLoader workers |

#### Skip Stages
| Flag | Description |
|---|---|
| `--skip-download` | Skip Kaggle download |
| `--skip-setup` | Skip venv creation |
| `--skip-verify` | Skip architecture verification |
| `--skip-export` | Skip multi-format export |

---

### `python -m src.train` — Core Training Engine

| Flag | Default | Description |
|---|---|---|
| `--backbone {lite2,lite4}` | `lite2` | Backbone architecture |
| `--weights PATH` | project root `.pth` | Custom pretrained weights |
| `--attention {cbam,se}` | `cbam` | Attention module |
| `--data PATH` | `data/raw/` | Raw image root |
| `--split-dir PATH` | `data/splits/` | CSV split directory |
| `--batch-size N` | `64` | Per-GPU batch size |
| `--num-workers N` | `4` | DataLoader workers |
| `--device DEVICE` | auto | Force `cuda` or `cpu` |
| `--mix` | off | Enable CutMix/MixUp (opt-in; same-species, α=0.4/0.2, p=0.25) |
| `--no-mix` | off | Force-disable CutMix/MixUp (default) |
| `--flip` / `--color-jitter` / `--randaugment` / `--rrc` | off | Enable individual train transforms |
| `--augment-all` | off | Enable flip + color-jitter + randaugment + rrc + mix |
| `--phase1-epochs N` | `8` | Phase 1 epochs |
| `--phase2-epochs N` | `80` | Phase 2 epochs |
| `--phase3-epochs N` | `10` | Phase 3 epochs (opt-in) |
| `--include-qat` | off | Enable QAT phase 3 (OFF by default) |
| `--skip-qat` | off | Kept for compat; QAT is already off by default |
| `--contrastive-weight F` | `0.2` | SupCon weight on the projection embedding (0 disables) |
| `--logit-adjust` | off | Enable logit adjustment **on top of** the sampler (double-corrects) |
| `--logit-adjust-prior {sampled,raw}` | `sampled` | Prior source when logit adjustment is enabled |
| `--no-logit-adjust` | off | Force-disable logit adjustment |
| `--rare-threshold N` | `30` | Breeds below this many train images are excluded from CutMix/MixUp |
| `--run-tag STR` | auto (DD-MM-YYYY-HH-MM) | Run id for timestamped outputs |
| `--teacher PATH` | off | Teacher checkpoint for distillation |
| `--teacher-backbone` | `lite4` | Teacher backbone |
| `--teacher-attention` | student's | Teacher attention type |
| `--smoke-test` | off* | 5 imgs/breed, 1 epoch |
| `--half-data` | off* | 50% images/breed |
| `--quarter-data` | off* | 25% images/breed |
| `--no-compile` | off | Disable `torch.compile` |
| `--seed N` | `42` | Random seed |
| `--export-dir PATH` | `outputs/export/portable/` | Export destination |
| `--no-export` | off | Skip auto-export |
| `--weight-decay F` | `1e-2` | AdamW weight decay |
| `--label-smoothing F` | `0.05` | Label smoothing |
| `--warmup-epochs N` | `3` | LR warmup epochs |
| `--grad-accum N` | `2` | Gradient accumulation steps |

*mutually exclusive group

---

### `python -m src.export`

| Flag | Default | Description |
|---|---|---|
| `--backbone {lite2,lite4}` | `lite2` | Which backbone to export |
| `--mode {onnx,onnx-int8,tflite,float16,portable,int8}` | `onnx` | Export format (`int8` prints removal guidance) |
| `--calibration-images N` | `500` | Train-split images for INT8 calibration |
| `--skip-app-assets` | off | Don't copy TFLite artifacts into `flutter_app/assets/models/` |
| `--checkpoint PATH` | auto | Specific `.pt` file |

### `python -m src.evaluate`

| Flag | Default | Description |
|---|---|---|
| `--backbone {lite2,lite4}` | `lite2` | Backbone to evaluate |
| `--checkpoint PATH` | auto | Specific `.pt` file |
| `--device DEVICE` | auto | Force `cuda` or `cpu` |

### `python test_model.py`

| Flag | Default | Description |
|---|---|---|
| `--port N` | `8501` | HTTP port |
| `--no-browser` | off | Don't auto-open browser |
| `--dev` | on | Developer mode: view full metadata, system logs, edit presenter settings |
| `--present` | off | Presenter mode: clean UI, hides technical details and export options |

---

## 10. Common Operations

### Automated Pipeline (Recommended)
```bash
python local_train.py                  # Full data
python local_train.py --half-data      # 50% data
python local_train.py --quarter-data   # 25% data
python local_train.py --smoke-test     # Smoke test (seconds)
```

### Model Testing GUI
```bash
python test_model.py                   # → http://localhost:8501
python test_model.py --port 9000       # Custom port
python test_model.py --no-browser      # Headless
```

### Manual Setup (Linux/macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch>=2.1.0 torchvision>=0.16.0 numpy pandas scikit-learn tqdm Pillow onnx
```

### Manual Setup (Windows)
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install torch>=2.1.0 torchvision>=0.16.0 numpy pandas scikit-learn tqdm Pillow onnx
```

### Verify Architecture
```bash
python -m src.verify
```

### Prepare Data Splits
```bash
python -m src.data_pipeline
```

### Train (Manual)
```bash
python -m src.train --backbone lite2 --skip-qat          # default: no augmentation
python -m src.train --half-data --skip-qat
python -m src.train --quarter-data --skip-qat --run-tag exp1
python -m src.train --smoke-test --skip-qat
python -m src.train --backbone lite2 --mix --flip        # opt-in augmentation
```

### Diagnostics (run on the GPU/dataset machine — read-only)
```bash
python scripts/audit_data.py --data data/raw --split-dir data/splits
python scripts/diagnose_model.py --checkpoint <pt> [--ema <pt>] --split test
python scripts/onnx_parity_10.py --checkpoint <pt> --onnx <fp32.onnx> --images "Testing data/**/*.jpg"
python scripts/test_fixes_cpu.py                          # CPU-only, no data needed
```

### Evaluate
```bash
python -m src.evaluate --backbone lite2
```

### Export
```bash
python -m src.export --mode portable --backbone lite2
python -m src.export --mode onnx --backbone lite2
python -m src.export --mode onnx-int8 --backbone lite2
python -m src.export --mode float16 --backbone lite2
```

### Create Training Zip
```bash
python create_training_zip.py
```

---

## 11. Known Constraints & Gotchas

1. **QAT + CUDA AMP conflict**: Phase 3 (QAT) disables AMP scaler because quantization observers don't support mixed precision. This is intentional — handled automatically.

12. **EMA must include BN buffers, once per optimizer step** (`src/train.py`): the phase-2 EMA updates parameters AND BatchNorm buffers; `num_batches_tracked` is hard-copied. It runs **once per optimizer step** (not per micro-batch) with warm-up `min(0.999,(1+t)/(10+t))`. Removing the sync or moving it back into the inner loop corrupts the EMA (stale BN / doubled effective decay).

13. **Mobile artifacts are batch-1 static** with input RGB in [0,1] (normalization baked in): the Flutter engine feeds `pixel/255` floats and reads outputs by index (0=binary, 1=cattle, 2=buffalo). Keep output names/order stable when editing `_MobileOutputs`.

14. **Preprocessing must match eval**: eval uses shortest-side `Resize(260)` + `CenterCrop(260)` + ImageNet normalize. `test_model.py` and the Flutter preprocessor match. Square `Resize((260,260))` distorts aspect ratio — it wrecked external-batch predictions before the 2026-09-23 fix.

15. **Exportable checkpoint loading**: `src/export._sanitize_state_dict` strips `_orig_mod.`/`module.` prefixes, QAT `fake_quant`/`activation_post_process`/fused-BN keys, and loads with `strict=False` (the training-only `projection_head.*` is tolerated). Genuinely incompatible heads (missing non-projection keys) still raise. Export the phase-2 EMA checkpoint for best accuracy.

16. **Logit adjustment is OFF by default**: the effective-number sampler already rebalances every batch; enabling `--logit-adjust` too double-corrects and over-predicts rare breeds at inference. If enabled, its prior is the effective *sampled* distribution (absent classes get the smallest present prior, never NaN).

17. **Outputs are timestamped**: checkpoints/exports/metrics carry a run id (`DD-MM-YYYY-HH-MM`, config `RUN_ID_FORMAT`). Colons are avoided because they are illegal in Windows filenames. `src/run_utils` provides `make_run_id`/`timestamped`/`unique_path`/`find_latest_checkpoint`; tools auto-discover the newest checkpoint.

18. **TFLite toolchain is optional**: `--mode tflite` needs `tensorflow` + `onnx2tf` (see requirements.txt); `--mode onnx-int8` needs only `onnxruntime`. TF is not installable on Python 3.14 — run TFLite conversion from the Windows venv (Python 3.13) or Colab.

2. **Backbone weights are included in the repo**: `efficientnet_lite{2,4}.pth` are tracked in git. No separate download required. Without them, backbone trains from scratch (significantly worse accuracy).

3. **Fixed head sizes**: Model heads are always sized for 57 cattle + 18 buffalo classes regardless of data mode. Unused class outputs receive no gradient. Architecture is identical across all modes.

4. **WeightedRandomSampler (training only, default on)**: Training oversamples rare breeds with effective-number weights (`SAMPLER_BETA=0.99`). Evaluation uses no sampling — test set reflects natural distribution. This is the **single** long-tail mechanism (logit adjustment is off by default).

5. **Soft cross-entropy**: supports fractional label vectors from CutMix/MixUp; hard one-hot labels work identically. Mixing is off by default and, when enabled, pairs within a species so labels stay well-formed.

6. **Portable export requires `BreedClassifier` class**: Saves `state_dict`, not TorchScript. Loading requires `from src.model import BreedClassifier`. For framework-free inference, use ONNX.

7. **Smoke test uses ALL 75 classes**: Even with 5 imgs/breed, class maps include all breeds. Architecture is identical to full training.

8. **`torch.compile` OOM on T4 GPU**: `mode="reduce-overhead"` uses CUDA Graphs, causing OOM on 15 GB T4. Default `torch.compile(model)` (no mode) avoids this.

9. **`torch.compile` crashes on Windows**: Triton not supported on Windows → `BackendCompilerFailed`. Auto-detected via `os.name == 'nt'`, falls back to eager. Use `--no-compile` to force.

10. **`data/splits/` regenerated each run**: CSV splits are always regenerated at the start of training. Do not manually edit files in `data/splits/`.

11. **Windows `curl` alias**: In PowerShell, `curl` is an alias for `Invoke-WebRequest`. Use `curl.exe` or the Python `requests` library. `local_train.py` uses `requests` internally.

---

## 14. Colab Training

### Notebook Location

- `colab/cattle_buffalo_trainer.ipynb` — main Colab notebook
- `colab/cattle_buffalo_trainer.py` — same content as percent-format script
- `colab/README.md` — setup instructions

### Project Setup Options (in Colab)

1. **GitHub clone** (recommended):
   ```bash
   !rm -rf /content/project   # force-refresh to pick up latest code
   !git clone https://github.com/Bharaths31/ML-CB-B-identifier /content/project
   ```
   > **Important**: Always `rm -rf /content/project` before cloning so that updated code (bug fixes, new features) is pulled correctly. Simply re-running the clone cell without deleting first silently keeps the old cached copy.
2. **Upload `colab_project.zip`**: created by `python scripts/create_colab_project_zip.py`
3. **Google Drive mount**: copy `colab_project.zip` from `My Drive/ML-CB-B-identifier/`

### Dataset Options (in Colab)

1. **Kaggle API / Direct Download** (`DATASET_MODE="both"` by default): downloads
   and merges **both** sources into `data/raw/` — the unified
   `algsoch/breed-cattle-buffalo` set *and* the atharvadarpude indigenous sets
   (`indian-cattle-image-dataset` + `indian-buffalo-dataset`) via
   `merge_into_species_dir()` with breed-name normalization
2. **Upload `archive.zip`**: created by `python scripts/create_colab_archive.py`
3. **Google Drive**: copy `archive.zip` from Drive

### T4 GPU Optimizations

| Setting | Value | Reason |
|---|---|---|
| `batch_size` | Auto | Dynamically detects VRAM (16 on 4GB GPUs up to 128 on 24GB GPUs) |
| `grad_accum` | Auto | Adjusts with batch size to maintain a constant effective batch of 128 |
| `num_workers` | 2 | Colab has 2 CPU cores |
| `prefetch_factor` | 4 | Keeps GPU fed |
| `pin_memory` | True | Faster CPU→GPU transfer |
| CPU Offloading | GPU MixUp/CutMix | Tensor slicing moved to GPU to unblock CPU |
| Memory Caching | Byte Caching | Caching JPEG bytes prevents RAM OOM while keeping fast IO |
| AMP | phases 1-2 only | Disabled for QAT phase 3 |
| Dataset location | `/content/data/raw/` | Local SSD, not Drive |

### Known Colab Gotchas

| Symptom | Root Cause | Fix |
|---|---|---|
| `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'` | PyTorch attribute is `total_memory`, not `total_mem` | **Fixed** in `src/train.py:49` and `colab/cattle_buffalo_trainer.py:29,900` |
| Re-running GitHub clone cell doesn't pick up new code | Colab re-uses cached `/content/project/` directory | Add `!rm -rf /content/project` before `git clone` in §1 |
| Runtime restarts wipe all files | Colab free tier has ephemeral storage | Re-run all cells from §0 top-to-bottom after any restart |

---

## 15. Android Deployment

### Mobile INT8 pipeline (TFLite + ONNX Runtime Mobile)

The Android path is **converter-side PTQ**, not PyTorch QAT:
1. `python -m src.export --mode tflite` — ONNX (static batch 1, normalization baked) → onnx2tf → TFLite FP32 → full-integer INT8 PTQ calibrated on ~500 real train images. Emits `model.tflite` + `labels_{binary,cattle,buffalo}.txt` into `flutter_app/assets/models/`.
2. `python -m src.export --mode onnx-int8` — QDQ static quantization (per-channel weights) for ONNX Runtime Mobile.
3. `python -m src.parity_check` — accuracy gate: INT8 must stay within 1 pt of fp32 (`--synthetic N` for artifact-only parity without data).

Input convention for both mobile runtimes: RGB float32 in **[0,1]** with
ImageNet normalization baked into the graph — the Flutter engine's existing
`pixel/255` preprocessing is exactly correct.

### QAT (optional recovery path)

`python -m src.train --include-qat` (OFF by default). Per-tensor observers,
starts from the best phase-2 EMA checkpoint, converts with
`torch.ao.quantization.convert()` to a timestamped
`<backbone>_quantized_<runid>.pt`. Only useful as an x86-side accuracy-recovery
tool if converter PTQ drops > 2 pt — the artifact is NOT a TFLite/ORT model.

### Model Sizes (measured on lite2, 2026-09-20)

| Artifact | Size |
|---|---|
| Portable model.pt (FP32 state_dict) | ~26 MB |
| ONNX FP32 | ~25.7 MB |
| ONNX INT8 (QDQ, mobile) | ~7.1 MB |
| TFLite INT8 | ~6–7 MB (expected) |

---

## 16. Changelog

### 2026-09-24 — Blocking bug fixes, flip+RRC default-on, dataset inventory

- **`src/run_utils.py` must be synced** (it was missing on the GPU machine →
  `ModuleNotFoundError`). `local_train.py` now preflights every required `src/`
  module and raises a clear "sync these files" error.
- **`SAMPLER_BETA` import** added to `src/train.py` (used but not imported).
- **Mixing gate fixed**: `run_epoch` used `desc.startswith("train")` (never
  matched `phase{n} ...`), so `--mix` did nothing. Now an explicit `training`
  flag + `mix_stats`; log prints `mix=on (n/N)` only when it ran.
- **`find_latest_checkpoint` fixed** to match timestamped names; new
  `resolve_checkpoint(path_or_tag)` lets `--checkpoint V3` find a tagged run.
- **Class-count fail-fast** in `prepare_*` (extra/missing names; prints shared
  species names like `bargur`).
- **Flip + mild RRC ON by default** (`AUG_HORIZONTAL_FLIP`,
  `AUG_RANDOM_RESIZED_CROP`); colour jitter / RandAugment / mix stay off;
  `--no-augment` forces all off. RRC ratio `(0.92,1.08)`, hue capped at 0.02.
- **Breed-aware augmentation** (`BREED_AUG_POLICY`, coat-colour breeds skip
  jitter) and **pad-to-square** (`--pad-to-square`).
- **CBAM/SE identity at init** (`CBAM_IDENTITY_INIT`).
- **Metrics**: species-aware `combined_top3/5` over the soft-routed 75-way
  scores; oracle variants renamed `*_oracle`; `val_min_per_breed` /
  `val_median_per_breed`; entropy comment fixed.
- **Dataset inventory** (`build_dataset_inventory` in `local_train.py` + Colab):
  `data/dataset_inventory/<dataset>.json` with breed, species, count and
  per-image resolution for both datasets and the merged tree.
- **Tests**: `scripts/test_master_cpu.py` (40 checks) + `scripts/test_fixes_cpu.py`.

### 2026-09-23 — Tail-bias regression fix: single imbalance mechanism, safe mixing, EMA, soft routing, timestamped outputs

Diagnosed from a 10-photo ONNX batch test (0/10 correct; top-5 dominated by rare
breeds; a Gir bull predicted as a rare breed) that regressed after the
2026-09-21 overhaul.

- **Double long-tail correction removed**: `LOGIT_ADJUST=False` by default — the
  effective-number sampler is the single mechanism. `--logit-adjust` re-enables
  it; its prior comes from the effective **sampled** distribution
  (`--logit-adjust-prior sampled|raw`). Startup prints the active mechanism and
  prior max/min ratio. Fixed a divide-by-zero in `_effective_num_weights` that
  produced NaN priors for classes absent from train.
- **Same-species mixing**: CutMix/MixUp pair within a species (`_pairing_perm`),
  so binary labels stay one-hot and breed targets stay proper distributions.
  Strength reduced (`CUTMIX_MIXUP_PROB` 0.5→0.25, α 1.0→0.4 / 0.3→0.2);
  `MIX_OFF_LAST_FRAC=0.15` disables mixing for the last 15% of phase 2.
- **Augmentation OFF by default**: flip / ColorJitter / RandAugment /
  RandomResizedCrop / mix are opt-in (`--mix --flip --color-jitter --randaugment
  --rrc --augment-all`). With all off, train transform == eval transform.
- **EMA fixed**: once per **optimizer** step (was twice per micro-batch), with
  warm-up `min(0.999,(1+t)/(10+t))`. Startup prints the time constant and warns
  if it exceeds 25% of phase-2 steps. Each eval logs raw AND EMA, saving the
  better one.
- **Checkpoint metric**: `BEST_METRIC="blended_score"` = 0.5·macro-F1 + 0.5·soft
  top-1; each eval logs few/medium/many-shot acc + predicted-histogram entropy.
- **Preprocessing**: `test_model.py` now uses the eval transform (shortest-side
  resize + CenterCrop) instead of square `Resize((260,260))`; added
  `EVAL_MATCH_TRAIN_RESOLUTION`.
- **Timestamped outputs**: checkpoints/exports/metrics carry a run id
  (`DD-MM-YYYY-HH-MM`, `TIMESTAMP_OUTPUTS=True`); `src/run_utils` provides
  `make_run_id`/`timestamped`/`unique_path`/`find_latest_checkpoint`. Tools
  auto-discover the newest checkpoint.
- **Tooling**: `scripts/audit_data.py`, `scripts/diagnose_model.py`,
  `scripts/onnx_parity_10.py`, `scripts/test_fixes_cpu.py`.

### 2026-09-21 — Long-tail accuracy overhaul: logit adjustment, SupCon features, soft routing, 70/15/15 splits

Motivation: phase-2 best val top-1 was 0.578 (binary 0.95, cattle 0.59,
buffalo 0.61) with 21+ indigenous breeds at ≤14 images versus `gir`=768, and the
old 85/10/5 split left many rare breeds with **zero** test images.

**Training (`src/train.py`, `src/config.py`):**
- Sampler `SAMPLER_BETA` 0.999 → **0.99**; `PHASE2_EPOCHS` 60 → **80**.
- New **logit adjustment** (`LOGIT_ADJUST`, τ=`LOGIT_ADJUST_TAU`=1.0) from smoothed
  train priors (`compute_class_priors`) — applied to breed logits in training only.
- New **SupCon feature loss** (`CONTRASTIVE_WEIGHT`=0.2, T=0.1) on a
  training-only projection head (`PROJECTION_DIM`=128); skipped on mixed batches.
- New **rare-class mixing guard** (`RARE_CLASS_THRESHOLD`=30): breeds below 30
  train images are restored unmixed in CutMix/MixUp (`_rare_keep_mask`).
- **Adaptive loss weights**: after `binary_acc ≥ 0.95` switch
  `0.15/0.50/0.35 → 0.05/0.55/0.40` (`LOSS_WEIGHT_*_FINAL`).
- `--no-mix` now genuinely disables mixing; new `--contrastive-weight`,
  `--no-logit-adjust`, `--rare-threshold` flags.
- Checkpoints selected on `BEST_METRIC="balanced_score"` (mean macro-F1).

**Data (`src/data_pipeline.py`):**
- Splits are **70/15/15** with long-tail minimums (≥1 val/test per breed with ≥3
  images); grouping fixed to key on `(species, breed)` (`bargur` exists in both).
- Sampler and priors count per `(species, breed)`.

**Model & metrics (`src/model.py`, `src/metrics.py`):**
- `BreedClassifier.predict()` returns the soft-routed 75-class distribution
  `p(species)·softmax(head)`; `evaluate_epoch` adds macro-F1, balanced accuracy
  and `combined_top1_soft`; `test_model.py` and both Flutter engines now use the
  same soft-routing mixture instead of a hard binary argmax.

**Export (`src/export.py`, `local_train.py`):**
- `_sanitize_state_dict` strips `_orig_mod.`/`module.` prefixes, QAT observer
  keys and fused-BN names, loads with `strict=False` (tolerates the
  training-only `projection_head.*`) — QAT/compiled checkpoints no longer crash
  ONNX/FP16/INT8 export.
- `local_train.py` prefers the phase-2 (EMA) checkpoint and exports INT8 via
  converter-side PTQ (`--mode onnx-int8`).

### 2026-09-20 — Accuracy/Efficiency overhaul: distillation, EMA fix, mobile exports

**Training (`src/train.py`):**
- Fixed a critical EMA bug: EMA now updates BatchNorm buffers (running mean/var) in addition to parameters; `num_batches_tracked` is hard-copied. Previously every phase-2 checkpoint exported stale BN stats.
- Added knowledge distillation: `--teacher <ckpt>` (+ `--teacher-backbone`, `--teacher-attention`). Loss = `(1-α)·masked_hard_CE + α·T²·masked_KL(teacher‖student)` on all three heads (α=0.7, T=4.0).
- QAT is now **opt-in** (`--include-qat`); default training is 2 phases. QAT uses per-tensor observers (fixes the `Unsupported qscheme: per_channel_affine` conversion failure) and starts from the best phase-2 EMA checkpoint.
- Binary CE is species-balanced per batch (`BALANCE_BINARY_HEAD=True`) to counter the 57:18 breed-count prior.

**Data (`src/data_pipeline.py`):**
- Sampler switched to effective-number-of-samples weighting (`SAMPLER_BETA=0.999`) — softens over-oversampling of tiny breeds.
- `CutMix/MixUp` probability raised 0.25 → 0.5 (`CUTMIX_MIXUP_PROB`).
- All `prepare_*_splits` now create the split directory if missing.

**Export (`src/export.py`):**
- New `--mode tflite`: ONNX → onnx2tf → TFLite FP32 + full-integer INT8 PTQ (float32 [0,1] I/O), emits `labels_*.txt`, auto-copies into `flutter_app/assets/models/`.
- New `--mode onnx-int8`: QDQ static quantization for ONNX Runtime Mobile (per-channel weights, calibrated on real train images).
- Mobile artifacts have ImageNet normalization baked into the graph — the Flutter app's `pixel/255` preprocessing is now exactly correct.
- Removed the broken x86 PTQ `--mode int8` path (kept as a guidance stub).

**Verification (`src/parity_check.py`, new):**
- `python -m src.parity_check` compares fp32 PyTorch vs TFLite/ONNX INT8 on val/test (same metrics as training) or `--synthetic N` for artifact-only parity. Gate: INT8 within 1 pt combined_top1 of fp32.

**Measured:** ONNX INT8 = 7.10 MB (from 25.65 MB FP32); fp32 ONNX exact-parity; INT8 max|Δlogit| ≈ 0.05–0.07 on random inputs; end-to-end mini training (2-phase, KD, QAT) verified on CPU.

### 2026-09-15 — Presenter Mode, Logging & Advanced Image Metadata

**Model Tester GUI (`test_model.py`):**
- Added `--dev` (default) and `--present` flag modes.
- Developer Mode (`--dev`): Advanced view showing image metadata (EXIF, size, proportion), cattle/buffalo JSON data, model specifications, and options to edit the presenter's view settings.
- Presenter Mode (`--present`): Clean, minimalist test page that hides detailed technical stats, diminishes confidence metrics, and removes the export option for a cleaner presentation.
- Presenter configurations (like branding, section toggles, and confidence modes) are saved and loaded persistently via `outputs/logs/presenter_config.json`.
- Comprehensive session logging captures all actions (start/stop, model/image selection, prediction results, reasoning) into a separate log file in `outputs/logs/`.


### 2026-09-14 — GUI Batch Processing & ODT Export

**GUI Modernization (`test_model.py`):**
- Added a tabbed interface separating single image testing from batch processing.
- Added batch analysis via drag-and-drop with real-time progress indicators and an aggregate dashboard.
- Added functionality to export single or batch predictions into professionally formatted OpenDocument Text (`.odt`) files via `odfpy`.
- Added ONNX Runtime support for discovering and running `.onnx` models (`onnxruntime`).

---

### 2026-09-14 — Knowledge Base Verification & Constant Alignment

**Knowledge Base & Documentation:**
- Audit and synchronization of knowledge base files (`.agents/knowledge/CONTEXT.md`, `.agents/knowledge/MODULE_REFERENCE.md`).
- Synchronized all constants in `CONTEXT.md` Section 8 (`NUM_BREEDS_TOTAL`, `HALF_DATA_RATIO`, `QUARTER_DATA_RATIO`, `EVAL_EVERY_PHASE1`, `EVAL_EVERY_PHASE2`, `EVAL_EVERY_PHASE3`, `CACHE_IMAGES`, `SPECIES_LABELS`) with `src/config.py`.
- Updated module line counts in `MODULE_REFERENCE.md` to reflect exact source code line lengths.
- Verified repository health and confirmed working tree is clean.

---

### 2026-09-13 — Colab Testing Notebook & Large-Scale Evaluation

**Testing & Evaluation:**
- Added `colab/cattle_buffalo_tester.py` and `colab/cattle_buffalo_tester.ipynb` for automated evaluation of exported models on Google Colab.
- Added comprehensive HTML report generation for single images and batch evaluations.
- Added Large-Scale Kaggle Evaluation mode to automatically download the dataset and test all images.
- Added `create_test_eval_zip.py` script to easily bundle test dataset splits for Colab.
- Updated documentation and knowledge base (`CONTEXT.md`, `README.md`, `docs/`) with testing workflow details.

---

### 2026-09-08 — Webapp & Memory Layer Removal & Architecture Streamlining

**Refactoring & Cleanup:**
- Completely removed the `webapp/` (FastAPI backend and HTML/JS frontend) and `memory/` (Mem0 AI vector context layer) directories.
- Removed unused dependencies (`fastapi`, `uvicorn`, `mem0ai`, etc.) from `requirements.txt`.
- Removed `docs/webapp.md` and `docs/memory-layer.md` documentation pages and updated `mkdocs.yml`.
- Standardized interactive inference testing exclusively around `test_model.py` (custom standalone PyTorch inference browser application).
- Cleaned up obsolete webapp skipping flags and references from `local_train.py` and `create_training_zip.py`.
- Completely updated knowledge base (`.agents/AGENTS.md`, `.agents/knowledge/CONTEXT.md`, `MODULE_REFERENCE.md`, `TRAINING_INTERNALS.md`), `README.md`, and `docs/` pages to unify instructions and eliminate all conflicting or deprecated references.

---

### 2026-09-08 — Fix: `torch.compile` on Windows

**Bug Fix:**
- Fixed `BackendCompilerFailed: Cannot find a working triton installation` error that crashed phase 2 training on Windows.
- Added OS detection in `src/train.py` to automatically disable `torch.compile` (fallback to eager mode) when running on Windows.

---

### 2026-09-07 — Local Training Automation & Half-Data Mode

### 2026-09-06 — Hotfix: Colab CPU Bottleneck & OOM Prevention

**Performance Fixes:**
- Implemented **Dynamic VRAM Auto-Scaling** in `train.py`. The script now detects physical GPU VRAM and automatically adjusts `batch_size` and `grad_accum` to support hardware ranging from 4GB local cards (e.g., RTX 3050) up to 24GB+ instances, while maintaining an effective batch size of 128.
- Switched `CattleBuffaloDataset` caching strategy from storing `PIL.Image` objects (which caused massive memory leaks and OOMs on Colab) to caching raw JPEG `bytes`. This completely skips slow disk I/O after the first epoch without exhausting system RAM.
- Moved `CutMix` and `MixUp` augmentations from the CPU-bound `mixed_collate` function to the GPU inside `run_epoch`. This offloads heavy tensor slicing to the CUDA device, significantly increasing training throughput and un-starving the GPU.

### 2026-09-07 — Fix: `torch.compile` OutOfMemoryError on GPU

**Bug Fix & Stability:**
- Removed `mode="reduce-overhead"` from `torch.compile(model)` in `src/train.py`, `colab/cattle_buffalo_trainer.py`, and `colab/cattle_buffalo_trainer.ipynb`.
- Root cause: `mode="reduce-overhead"` uses CUDA Graphs which pre-allocates substantial VRAM during backward pass, causing `OutOfMemoryError` on 15GB Tesla T4 GPUs during Phase 2 multi-task fine-tuning.

### 2026-09-07 — Unified Kaggle Dataset & Colab Trainer Update

**Dataset Pipeline & Colab Notebook:**
- Updated dataset download source to unified Kaggle dataset `algsoch/breed-cattle-buffalo` containing pre-structured `cattle/` (57 breeds) and `buffalo/` (18 breeds) subdirectories.
- Simplified Kaggle download logic in `colab/cattle_buffalo_trainer.py` to extract directly into `data/raw/`, eliminating redundant file moving operations and outdated inline comments.
- Regenerated `colab/cattle_buffalo_trainer.ipynb` from updated python script.
- Updated project documentation across `README.md`, `docs/`, and knowledge base.

---

### 2026-09-06 — Hotfix: CUDA `total_mem` AttributeError

**Bug Fix:**
- Fixed `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'` that crashed §6 Training on Colab T4
- Root cause: PyTorch uses `total_memory`, not `total_mem`
- Fixed in `src/train.py` (`setup_device()`) and both occurrences in `colab/cattle_buffalo_trainer.py`
- Regenerated `colab/cattle_buffalo_trainer.ipynb` from fixed `.py`

**Documentation:**
- Added Colab gotchas table to `CONTEXT.md` §14 covering: `total_mem` bug, GitHub clone cache issue, runtime restart behaviour
- Added `rm -rf /content/project` before `git clone` in §14 best practices to ensure latest code is always used

---

### 2026-09-06 — Colab + SOTA Hyperparameters + Android QAT

**Colab Training:**
- Created `colab/` directory with full training notebook
- 3 project setup options: GitHub clone, zip upload, Google Drive
- 3 dataset options: Kaggle API, archive upload, Google Drive
- Hyperparameter configuration cell with all tunable parameters
- Image prediction cell for testing with uploaded images
- Export & download: portable bundle + ONNX + INT8
- GPU memory monitor cell

**SOTA Hyperparameters:**
- Switched from Adam → AdamW (weight_decay=1e-2)
- Added label smoothing (0.1) to soft cross-entropy
- Added linear warmup scheduler (3 epochs) before cosine annealing
- Added gradient accumulation (2 steps, effective batch=128)
- Increased batch size 32 → 64
- Optimized split ratio 80/10/10 → 85/10/5
- Phase 2 epochs 30 → 40, LR 1e-4 → 2e-4
- Phase 1 LR 1e-3 → 3e-3
- Phase 3 LR 1e-5 → 5e-6
- Dropout 0.3 → 0.4

**Data Pipeline:**
- Train augmentation: added RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- Added prefetch_factor=4 to all DataLoaders

**Android Deployment:**
- QAT (Phase 3) enabled by default (not skipped)
- Auto INT8 conversion after QAT
- ONNX export in Colab notebook for mobile deployment

### 2026-09-05 — Major Update

**Training:**
- Added CUDA optimization: `cudnn.benchmark`, TF32, AMP (`torch.amp`), `GradScaler`
- Added gradient clipping (`max_norm=1.0`)
- Enabled `pin_memory`, `persistent_workers`, `non_blocking` transfers
- Smoke test now uses real mini-dataset (5 imgs/breed) instead of 2-batch limit
- Auto portable export after training completes
- Better tqdm progress bars throughout

**Export & Standalone Packaging:**
- Added `portable` mode: self-contained folder with model + labels + metadata
- Improved progress bars on INT8 calibration
- Created `create_training_zip.py` script to generate a clean, webapp-free training zip package
- Added `.gitignore` configured to track `memory/` while ignoring `.venv/`, `outputs/`, `data/splits/`, `*.zip`, cache files

**Webapp:**
- Fixed argument formatting bug (`--phase1_epochs` → `--phase1-epochs`)
- Fixed `jobStatusHTML` crash when metrics object has missing keys
- Added model cache auto-invalidation after training (mtime-based)
- Progress bars now show completion/error states
- Running job indicator with pulse animation in header
- Auto-refresh status, metrics, and exports after job completion
- Added portable export option in UI dropdown

**Config:**
- Added `PORTABLE_EXPORT_DIR`, `SMOKE_SAMPLES_PER_BREED` constants

