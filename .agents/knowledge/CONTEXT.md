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
8. [Webapp (FastAPI + Vanilla JS)](#8-webapp)
9. [Memory Layer (Mem0)](#9-memory-layer)
10. [Configuration Reference](#10-configuration-reference)
11. [API Reference](#11-api-reference)
12. [Common Operations](#12-common-operations)
13. [Known Constraints & Gotchas](#13-known-constraints--gotchas)
14. [Changelog](#14-changelog)

---

## 1. Project Overview

| Field | Value |
|---|---|
| **Goal** | Classify images of Indian cattle (57 breeds) and buffalo (18 breeds) using a lightweight, mobile-deployable CNN |
| **Model** | EfficientNet-Lite{2,4} backbone + CBAM/SE attention + 3-head classifier (binary + cattle + buffalo) |
| **Stack** | Python 3.11+, PyTorch >= 2.1.0, FastAPI, Vanilla JS frontend |
| **Training** | 3-phase: binary warmup → multi-task fine-tune → optional QAT |
| **Deployment** | ONNX, INT8, float16, or portable self-contained folder |
| **Dataset** | `data/raw/cattle/<breed>/*.jpg` + `data/raw/buffalo/<breed>/*.jpg` |

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
│        ↓ (1280-dim feature vector)                           │
│  ┌─────┼─────────┬──────────────┐                            │
│  ↓     ↓         ↓              ↓                            │
│ binary_head  cattle_head  buffalo_head  (feature passthrough)│
│  (→2)        (→57)        (→18)                              │
│        ↓                                                     │
│  masked_loss: w_bin*CE_bin + w_cat*CE_cat + w_buf*CE_buf     │
│        ↓                                                     │
│  outputs/checkpoints/<backbone>_phase{1,2,3}_best.pt         │
│        ↓                                                     │
│  outputs/export/portable/<backbone>_*/  (self-contained)     │
└─────────────────────────────────────────────────────────────┘
```

### Inference Flow

```
Image → Resize(260) → CenterCrop(260) → ToTensor()
     → model.forward() → {binary, cattle, buffalo, features}
     → argmax(binary) → select cattle/buffalo head → argmax → breed
```

---

## 3. Directory Map

```
Mini Project/
├── src/                        # Core ML package (run as python -m src.<module>)
│   ├── __init__.py
│   ├── config.py               # All hyperparameters & paths
│   ├── data_pipeline.py        # Dataset, splits, augmentation, DataLoaders
│   ├── model.py                # BreedClassifier (backbone + attention + heads)
│   ├── cbam.py                 # CBAM & SE attention modules
│   ├── efficientnet_lite.py    # EfficientNet-Lite{2,4} architecture
│   ├── train.py                # 3-phase training with AMP, auto-export
│   ├── metrics.py              # evaluate_epoch() — per-head accuracy + F1
│   ├── evaluate.py             # Full evaluation with confusion matrices
│   ├── export.py               # ONNX, INT8, float16, portable export
│   └── verify.py               # Quick architecture sanity check
├── colab/
│   ├── cattle_buffalo_trainer.py   # Colab training script (percent-format)
│   ├── cattle_buffalo_trainer.ipynb # Jupyter notebook (auto-generated)
│   ├── convert_to_notebook.py  # .py → .ipynb converter
│   └── README.md               # Colab setup instructions
├── webapp/
│   ├── server.py               # FastAPI backend (predict, train, evaluate, memory)
│   └── static/
│       ├── index.html          # Single-page app (tabs: predict/train/eval/memory/debug)
│       ├── app.js              # Frontend logic, polling, progress bars
│       └── style.css           # Dark theme, progress bars, pulse animations
├── memory/
│   ├── __init__.py             # Exports Mem0Layer
│   └── service.py              # Mem0-based context memory (store/recall/chat)
├── data/
│   ├── raw/                    # Source images: raw/{cattle,buffalo}/<breed>/*.jpg
│   └── splits/                 # Generated: train.csv, val.csv, test.csv, *_classes.json
├── outputs/
│   ├── checkpoints/            # Training checkpoints (*.pt)
│   ├── export/                 # ONNX/INT8/float16 exports
│   │   └── portable/           # Self-contained model bundles
│   ├── metrics/                # Evaluation JSON + confusion matrix PNGs
│   └── memory/                 # Mem0 ChromaDB storage
├── scripts/                    # Colab archive creators, app asset prep
├── create_training_zip.py      # Creates lightweight standalone training package (excludes webapp)
├── setup.sh                    # Shell script helper for environment setup
├── setup_venv.py               # Automated virtual environment setup script
├── .gitignore                  # Git ignore rules (includes outputs, venv, cache; tracks memory/)
├── efficientnet_lite{2,4}.pth  # Pretrained ImageNet backbone weights
├── requirements.txt            # Python dependencies
└── setup.sh / setup_venv.py    # Environment setup
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
    cattle_head: Linear(1280→512→57) + Dropout(0.3)
    buffalo_head: Linear(1280→512→18) + Dropout(0.3)
```

### Forward Path

1. `forward_features(x)`: backbone stages 0..3 → CBAM → stages 4..6 → head → pool → flatten
2. `forward(x)`: features → 3 parallel heads → dict{binary, cattle, buffalo, features}

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

### Three-Phase Training (`src/train.py`)

| Phase | What | Frozen | LR | Epochs | Loss Weights |
|---|---|---|---|---|---|
| 1 | Binary head warmup | backbone + attention + breed heads | 3e-3 | 5 | bin=1.0, cat=0.0, buf=0.0 |
| 2 | Multi-task fine-tune | nothing | 2e-4 (warmup+cosine) | 40 | bin=0.5, cat=0.25, buf=0.25 |
| 3 | QAT (for Android) | nothing | 5e-6 | 10 | bin=0.5, cat=0.25, buf=0.25 |

### SOTA Optimizations

- **Optimizer**: AdamW (weight_decay=1e-2) — decoupled weight decay for better generalization
- **Label smoothing**: 0.1 — prevents overconfident predictions
- **LR schedule**: Linear warmup (3 epochs) → Cosine annealing (Phase 2)
- **Gradient accumulation**: 2 steps (effective batch = 128 with batch_size=64)

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

- `soft_ce`: soft cross-entropy supporting CutMix/MixUp label mixing
- `masked_loss`: binary CE always active, cattle/buffalo CE only on matching species (masked by cattle_mask/buffalo_mask)

### Smoke Test (`--smoke-test`)

Creates a mini-dataset of 5 images per breed using `prepare_smoke_splits()`:
- Samples from full dataset, not random batches
- Real training signal on actual breed images
- 1 epoch per phase
- Auto-exports portable model after training

### Auto-Export

After training completes, automatically creates a portable export in `outputs/export/portable/` containing:
- `model.pt` — checkpoint with state_dict
- `cattle_classes.json`, `buffalo_classes.json` — label maps
- `model_info.json` — architecture metadata + usage instructions

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

- **Full training**: 85/10/5 stratified per breed (optimized for maximum training data)
- **Smoke test**: 5 images/breed → 60/20/20 split (tiny but real)

### Augmentation

- **Train**: Resize(288) → RandomResizedCrop(260, scale=0.8-1.0) → RandomHorizontalFlip → ColorJitter(0.2,0.2,0.2,0.1) → RandAugment(ops=2, mag=9) → ToTensor()
- **Eval**: Resize(260) → CenterCrop(260) → ToTensor()
- **Batch mixing**: 50% chance of CutMix(α=0.4) or MixUp(α=0.2) applied directly on the GPU during the training loop.
- **Caching**: `CACHE_IMAGES` stores raw JPEG bytes in RAM, preventing OOMs while bypassing disk I/O.

### Label Encoding

- Binary: one-hot [cattle=0, buffalo=1]
- Breed: one-hot over species-specific classes
- Masks: `cattle_mask=1.0` if cattle, else `buffalo_mask=1.0`

### Sampling

- `WeightedRandomSampler` balances breeds by inverse frequency

---

## 7. Export & Deployment

### Export Modes (`python -m src.export`)

| Mode | Output | Notes |
|---|---|---|
| `onnx` | `<backbone>_fp32.onnx` | opset 13, static or dynamic batch |
| `int8` | `<backbone>_int8.pt` + `_int8_traced.pt` | PTQ with 32-batch calibration |
| `float16` | `<backbone>_float16.pt` | TorchScript traced |
| `portable` | `portable/<backbone>_<tag>/` folder | Self-contained: model + labels + info |

### Portable Export Structure

```
outputs/export/portable/<backbone>_phase2_best/
├── model.pt                 # torch checkpoint {state_dict: ...}
├── cattle_classes.json      # {"amritmahal": 0, "ayrshire": 1, ...}
├── buffalo_classes.json     # {"alambadi": 0, "banni": 1, ...}
└── model_info.json          # {backbone, image_size, usage, exported_at}
```

---

## 8. Webapp

### Backend (`webapp/server.py`)

- **Framework**: FastAPI on uvicorn (port 8000)
- **ModelBox**: Thread-safe model loading with mtime-based cache invalidation
- **JobRunner**: Subprocess manager for training/eval/export with real-time log parsing
- **Auto-invalidation**: ModelBox cache cleared when training job completes

### Frontend (`webapp/static/`)

- Single-page app with 5 tabs: Predict, Train, Evaluate, Memory, Debug
- **Polling**: 1.2s interval, auto-starts when job detected, auto-stops on completion
- **Progress bars**: Parse tqdm output + `[train]` log lines for phase/epoch/batch progress
- **State chips**: GPU/CPU indicator, model status, running job indicator with pulse animation

### Key API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | System info, checkpoints, exports |
| POST | `/api/predict` | Image classification (multipart upload) |
| POST | `/api/train` | Start training subprocess |
| GET | `/api/job` | Poll job status + log tail |
| POST | `/api/job/stop` | Terminate running job |
| POST | `/api/evaluate` | Start evaluation subprocess |
| POST | `/api/export` | Start export subprocess |
| GET | `/api/metrics` | Saved evaluation metrics |
| GET | `/api/dataset` | Dataset breed counts |

---

## 9. Memory Layer

### Mem0Layer (`memory/service.py`)

- ChromaDB-backed vector store for context memories
- Optional LLM integration (via litellm) for extraction/deduplication
- Scoped by `user_id`, `agent_id`, `run_id`
- Token-efficient recall: ranks memories, prunes to fit budget
- Memory-aware chat: injects recalled context into LLM prompt

---

## 10. Configuration Reference

### `src/config.py` — All Constants

| Constant | Value | Purpose |
|---|---|---|
| `IMAGE_SIZE` | 260 | Input image dimension |
| `NUM_CATTLE_BREEDS` | 57 | Expected cattle breed count |
| `NUM_BUFFALO_BREEDS` | 18 | Expected buffalo breed count |
| `CBAM_AFTER_STAGE` | 3 | Attention insertion point |
| `FEATURE_DIM` | 1280 | Backbone output dimension |
| `BINARY_DIM` | 256 | Binary head hidden dim |
| `BREED_DIM` | 512 | Breed head hidden dim |
| `DROPOUT` | 0.4 | Breed head dropout |
| `BATCH_SIZE` | 64 | Default batch size |
| `NUM_WORKERS` | 4 | DataLoader workers |
| `TRAIN/VAL/TEST_RATIO` | 0.85/0.10/0.05 | Split ratios |
| `PHASE{1,2,3}_EPOCHS` | 5/40/10 | Training epochs |
| `PHASE{1,2,3}_LR` | 3e-3/2e-4/5e-6 | Learning rates |
| `LOSS_WEIGHT_*` | 0.5/0.25/0.25 | Multi-task loss weights |
| `WEIGHT_DECAY` | 1e-2 | AdamW weight decay |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing factor |
| `WARMUP_EPOCHS` | 3 | Linear warmup epochs (phase 2) |
| `GRADIENT_ACCUMULATION_STEPS` | 2 | Grad accum steps |
| `SMOKE_SAMPLES_PER_BREED` | 5 | Images per breed in smoke test |
| `CUTMIX_ALPHA` | 0.4 | CutMix beta distribution α |
| `MIXUP_ALPHA` | 0.2 | MixUp beta distribution α |
| `RANDAUGMENT_OPS` | 2 | RandAugment operations |
| `RANDAUGMENT_MAGNITUDE` | 9 | RandAugment intensity |

### Path Constants

| Constant | Path |
|---|---|
| `RAW_DATA_DIR` | `data/raw/` |
| `SPLIT_DIR` | `data/splits/` |
| `CHECKPOINT_DIR` | `outputs/checkpoints/` |
| `EXPORT_DIR` | `outputs/export/` |
| `PORTABLE_EXPORT_DIR` | `outputs/export/portable/` |
| `METRICS_DIR` | `outputs/metrics/` |

---

## 11. API Reference

### Training Arguments (`python -m src.train`)

```
--backbone {lite2,lite4}     Backbone architecture (default: lite2)
--weights PATH               Pretrained weights path
--attention {cbam,se}        Attention module (default: cbam)
--data PATH                  Raw data root
--split-dir PATH             Split CSV output directory
--batch-size N               Batch size (default: 32)
--num-workers N              DataLoader workers (default: 4)
--device DEVICE              Force device (auto-detects cuda/cpu)
--no-mix                     Disable CutMix/MixUp
--phase{1,2,3}-epochs N      Override epoch count
--skip-qat                   Skip phase 3 (QAT)
--smoke-test                 Use mini-dataset (5 imgs/breed, 1 epoch)
--seed N                     Random seed (default: 42)
--export-dir PATH            Portable export destination
--no-export                  Skip auto-export after training
```

### Webapp API Payload Formats

**POST /api/train**:
```json
{
  "backbone": "lite2",
  "smoke_test": true,
  "skip_qat": false,
  "phase1_epochs": 5,
  "phase2_epochs": 30,
  "phase3_epochs": 10,
  "num_workers": 4
}
```

**POST /api/export**:
```json
{"backbone": "lite2", "mode": "portable"}
```

---

## 12. Common Operations

### Setup
```bash
python setup_venv.py          # Create venv + install deps
source .venv/bin/activate
```

### Verify Architecture
```bash
python -m src.verify          # Check backbone loading + forward pass shapes
```

### Prepare Data Splits
```bash
python -m src.data_pipeline   # Scan raw/ → generate splits/*.csv
```

### Train (Full)
```bash
python -m src.train --backbone lite2
```

### Train (Smoke Test)
```bash
python -m src.train --smoke-test --skip-qat
```

### Evaluate
```bash
python -m src.evaluate --backbone lite2
```

### Export
```bash
python -m src.export --mode portable --backbone lite2
python -m src.export --mode onnx --backbone lite2
```

### Create Standalone Training Zip
```bash
python create_training_zip.py # Creates training_package.zip (excludes webapp/ and memory/)
```

### Run Webapp
```bash
python webapp/server.py       # → http://localhost:8000
```

---

## 13. Known Constraints & Gotchas

1. **QAT + CUDA AMP conflict**: Phase 3 (QAT) disables AMP scaler because quantization observers don't support mixed precision. This is intentional.

2. **Backbone weight files required**: `efficientnet_lite{2,4}.pth` must exist in project root for pretrained initialization. Without them, backbone trains from scratch (much worse accuracy).

3. **Class count mismatch**: If the dataset has fewer breeds than `NUM_CATTLE_BREEDS`/`NUM_BUFFALO_BREEDS`, the model head is still sized for 57/18 classes. Unused class outputs are never trained. This is by design for consistent model architecture.

4. **WeightedRandomSampler**: Training uses inverse-frequency sampling to balance breeds. This means rare breeds are over-sampled. For evaluation, no sampling is used.

5. **Soft cross-entropy**: Training uses soft labels (not hard argmax) because CutMix/MixUp produce fractional label vectors. This works with hard labels too (one-hot = special case of soft).

6. **Model cache invalidation**: The webapp's ModelBox now checks file mtime, so retraining automatically invalidates the cache on next prediction. No manual reload needed.

7. **Portable export is checkpoint-based**: The portable export saves `state_dict` (not TorchScript), so loading requires the `BreedClassifier` class definition. For framework-free deployment, use ONNX export instead.

8. **Smoke test uses ALL breed classes**: Even though only 5 images per breed are used, the class maps include ALL breeds from the full dataset. This ensures the model architecture is identical between smoke and full training.

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

1. **Kaggle API**: auto-downloads cattle + buffalo datasets
2. **Upload `archive.zip`**: created by `python scripts/create_colab_archive.py`
3. **Google Drive**: copy `archive.zip` from Drive

### T4 GPU Optimizations

| Setting | Value | Reason |
|---|---|---|
| `batch_size` | 64 | Maximizes T4 utilization (15 GB VRAM) |
| `grad_accum` | 2 | Effective batch = 128 |
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

### QAT Pipeline

Phase 3 (QAT) produces an INT8-ready model for mobile inference:
1. Conv-BN fusion: merges batch norm into convolutions
2. QAT training: inserts fake-quantize observers, fine-tunes with quantization noise
3. INT8 conversion: `torch.ao.quantization.convert()` produces true INT8 weights
4. Checkpoint: `<backbone>_quantized.pt`

### Export Formats for Android

| Format | File | Use Case |
|---|---|---|
| Portable | `portable/<backbone>_*/model.pt` | PyTorch Mobile / custom runtime |
| ONNX | `<backbone>_fp32.onnx` | ONNX Runtime Mobile, TFLite via converter |
| INT8 | `<backbone>_quantized.pt` | Smallest size, fastest inference |

### Model Sizes (approximate)

| Backbone | FP32 | INT8 (post-QAT) |
|---|---|---|
| lite2 | ~24 MB | ~6 MB |
| lite4 | ~50 MB | ~13 MB |

---

## 16. Changelog

### 2026-09-06 — Hotfix: Colab CPU Bottleneck & OOM Prevention

**Performance Fixes:**
- Switched `CattleBuffaloDataset` caching strategy from storing `PIL.Image` objects (which caused massive memory leaks and OOMs on Colab) to caching raw JPEG `bytes`. This completely skips slow disk I/O after the first epoch without exhausting system RAM.
- Moved `CutMix` and `MixUp` augmentations from the CPU-bound `mixed_collate` function to the GPU inside `run_epoch`. This offloads heavy tensor slicing to the CUDA device, significantly increasing training throughput and un-starving the GPU.

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

