# Cattle & Buffalo Breed Classifier

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/) [![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](https://github.com/Bharaths31/ML-CB-B-identifier)

A **lightweight, mobile-deployable image classifier** for **57 Indian cattle breeds** and **18 Indian buffalo breeds**, built on EfficientNet-Lite with CBAM/SE attention and a 3-head multi-task training pipeline.

| Field | Value |
|---|---|
| **Breeds** | 57 cattle + 18 buffalo = 75 total |
| **Backbone** | EfficientNet-Lite2 (~6M params) or Lite4 (~13M params) |
| **Input** | 260 × 260 RGB |
| **Training** | 3-phase: Binary warm-up → Multi-task fine-tune → Optional QAT |
| **Export** | ONNX, INT8, FP16, Portable bundle |
| **Dataset** | Kaggle: `algsoch/breed-cattle-buffalo` |

---

## Table of Contents

- [Architecture & Data Flow](#architecture--data-flow)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Windows](#windows)
  - [Linux / macOS](#linux--macos)
- [Kaggle API Setup](#kaggle-api-setup)
- [Automated Training (`local_train.py`)](#automated-training-local_trainpy)
  - [Data Modes](#data-modes)
  - [Complete Flag Reference](#complete-flag-reference-local_trainpy)
  - [Example Commands](#example-commands-local_trainpy)
- [Model Testing GUI (`test_model.py`)](#model-testing-gui-test_modelpy)
- [Google Colab Training](#google-colab-training)
- [SOTA Features](#sota-features)
- [Export & Android Deployment](#export--android-deployment)
- [Manual / Advanced Setup](#manual--advanced-setup)
  - [Manual Setup — Windows](#manual-setup--windows)
  - [Manual Setup — Linux / macOS](#manual-setup--linux--macos)
  - [Complete Flag Reference (`src.train`)](#complete-flag-reference-python--m-srctr​ain)
- [Known Constraints & Gotchas](#known-constraints--gotchas)
- [Changelog](#changelog)

---

## Architecture & Data Flow

```
data/raw/{cattle,buffalo}/<breed>/*.jpg
        ↓  data_pipeline.prepare_splits()
data/splits/{train,val,test}.csv
        ↓  CattleBuffaloDataset + DataLoader (CutMix/MixUp on GPU)
EfficientNet-Lite backbone (stages 0..6)
  stem → [stage 0..3] → CBAM/SE attention → [stage 4..6] → head
        ↓  AdaptiveAvgPool2d(1) → flatten → 1280-dim feature vector
   ┌────┴────────────┬────────────┐
binary_head (→2)  cattle_head (→57)  buffalo_head (→18)
        ↓  masked_loss: w_bin·CE + w_cat·CE + w_buf·CE
outputs/checkpoints/<backbone>_phase{1,2,3}_best.pt
        ↓  auto-export
outputs/export/portable/<backbone>_phase2_best/
```

---

## Directory Structure

```
ML-CB-B-identifier/
├── src/                        # Core ML package (run as python -m src.<module>)
│   ├── config.py               # All hyperparameters & paths (single source of truth)
│   ├── data_pipeline.py        # Dataset, splits, augmentation, DataLoaders
│   ├── model.py                # BreedClassifier (backbone + attention + heads)
│   ├── cbam.py                 # CBAM & SE attention modules
│   ├── efficientnet_lite.py    # EfficientNet-Lite{2,4} architecture
│   ├── train.py                # 3-phase training with AMP, auto-export
│   ├── metrics.py              # Per-head accuracy + F1
│   ├── evaluate.py             # Full evaluation + confusion matrices
│   ├── export.py               # ONNX, INT8, float16, portable export
│   └── verify.py               # Architecture sanity check
├── colab/
│   ├── cattle_buffalo_trainer.ipynb  # Colab GPU notebook
│   └── cattle_buffalo_trainer.py     # Same as percent-format script
├── data/
│   ├── raw/                    # Source images: raw/{cattle,buffalo}/<breed>/*.jpg
│   └── splits/                 # Generated: train.csv, val.csv, test.csv, *_classes.json
├── outputs/
│   ├── checkpoints/            # Training checkpoints (*.pt)
│   ├── export/                 # ONNX / INT8 / float16 exports
│   │   └── portable/           # Self-contained model bundles
│   └── metrics/                # Evaluation JSON + confusion matrix PNGs
├── scripts/                    # Utility scripts
├── local_train.py              # 🚀 Fully automated pipeline (setup → download → train → export)
├── test_model.py               # 🔬 Standalone web GUI for testing exported models
├── create_training_zip.py      # Lightweight training-only zip (no webapp, no memory)
├── setup_venv.py               # Automated venv creation & dependency install
├── setup.sh                    # Shell helper for quick env setup (Linux/macOS)
├── efficientnet_lite2.pth      # Pre-trained ImageNet backbone weights (included in repo)
├── efficientnet_lite4.pth      # Pre-trained ImageNet backbone weights (included in repo)
└── requirements.txt            # Python dependencies
```

> **Note:** The backbone weight files (`efficientnet_lite2.pth`, `efficientnet_lite4.pth`) are **already included in the repository**. No separate download is needed.

---

## Prerequisites

Before you begin, make sure you have the following installed:

| Requirement | Minimum | Recommended | Notes |
|---|---|---|---|
| **Python** | 3.9 | 3.11+ | Must be on `PATH` |
| **Git** | Any | Latest | To clone the repo |
| **GPU (CUDA)** | Optional | RTX 3050 4 GB+ | Falls back to CPU gracefully |
| **Kaggle account** | Required | — | Free — for dataset download |
| **Disk space** | ~8 GB | ~15 GB | For dataset + outputs |
| **RAM** | 8 GB | 16 GB+ | |

> **Windows users**: Some Python packages compile C extensions. Install **Microsoft C++ Build Tools** to avoid build errors:
> 1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
> 2. Run the installer and select the **"Desktop development with C++"** workload.
> 3. PyTorch installs via pre-built wheels and does **not** require this — it only matters for other packages.

---

## Installation

### Windows

Open **Command Prompt** or **PowerShell** as a regular user (not Administrator).

**Step 1 — Install Python 3.11+**
Download from https://www.python.org/downloads/ and run the installer.
- ✅ Check **"Add Python to PATH"** during installation.
- Verify: `python --version` should print `Python 3.11.x`

**Step 2 — Install Git**
Download from https://git-scm.com/download/win and install with default settings.
- Verify: `git --version`

**Step 3 — Clone the repository**
```cmd
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier
```

**Step 4 — Run the automated pipeline**

The `local_train.py` script handles **all remaining setup** automatically — virtual environment creation, dependency installation, Kaggle download, and training. Jump to [Automated Training](#automated-training-local_trainpy).

If you prefer manual setup, see [Manual Setup — Windows](#manual-setup--windows).

---

### Linux / macOS

Open a terminal.

**Step 1 — Install Python 3.11+**

*Ubuntu / Debian:*
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip -y
```

*macOS (with Homebrew):*
```bash
brew install python@3.11
```

Verify: `python3 --version`

**Step 2 — Install Git**

*Ubuntu / Debian:*
```bash
sudo apt install git -y
```

*macOS:*
```bash
brew install git
```

**Step 3 — Clone the repository**
```bash
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier
```

**Step 4 — Run the automated pipeline**

The `local_train.py` script handles all remaining setup automatically. Jump to [Automated Training](#automated-training-local_trainpy).

If you prefer manual setup, see [Manual Setup — Linux / macOS](#manual-setup--linux--macos).

---

## Kaggle API Setup

The dataset (`algsoch/breed-cattle-buffalo`) is hosted on Kaggle. You need a **free** Kaggle account and API key to download it.

**Getting your API key:**
1. Log in to [kaggle.com](https://www.kaggle.com)
2. Go to **Account Settings** → scroll to **API** section
3. Click **"Create New Token"** — this downloads `kaggle.json`
4. The file contains your `username` and `key`

**`local_train.py` handles this automatically:**
- On first run, it checks for `~/.kaggle/kaggle.json`
- If not found, it prompts you interactively for your username and key, then saves them
- On subsequent runs, existing credentials are reused silently

**Manual credential setup (optional):**

*Windows:*
```cmd
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

*Linux / macOS:*
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## Automated Training (`local_train.py`)

The **`local_train.py`** script is the recommended entry point. It runs a fully automated 8-stage pipeline:

| Stage | Name | What it does |
|---|---|---|
| §0 | **Prerequisites** | Checks Python ≥ 3.9, git, pip; Windows: checks MSVC build tools |
| §1 | **Environment Setup** | Creates `.venv`, installs training dependencies (skips webapp deps) |
| §2 | **Kaggle Download** | Prompts for API credentials if needed, downloads dataset |
| §3 | **Unzip & Organize** | Extracts images into `data/raw/cattle/` and `data/raw/buffalo/` |
| §4 | **Verify Architecture** | Runs `src.verify` to confirm backbone loading + forward pass |
| §5 | **Data Validation** | Validates dataset structure and breed counts |
| §6 | **Training** | Runs `src.train` with VRAM auto-scaling |
| §7 | **Export** | Exports model in 4 formats: portable, ONNX, INT8, float16 |

**Quick Start:**
```bash
# Fastest local test (25% data, ~4× speedup)
python local_train.py --quarter-data

# Balanced speed/accuracy (50% data)
python local_train.py --half-data

# Full training (all data, maximum accuracy)
python local_train.py
```

---

### Data Modes

These flags are **mutually exclusive** — pick exactly one (or none for full data):

| Flag | Data Used | Speed | Best For |
|---|---|---|---|
| *(none)* / `--full-data` | 100% of images | Slowest | Final production training |
| `--half-data` | 50% per breed (seed=42) | ~2× faster | Good local GPU training |
| `--quarter-data` | 25% per breed (seed=42) | ~4× faster | Quick local iteration |
| `--smoke-test` | 5 imgs/breed, 1 epoch | Seconds | CI / sanity check |

All subset modes use **deterministic sampling** (`seed=42`) and maintain the full 57/18 class map — the model architecture is identical across all modes.

---

### Complete Flag Reference (`local_train.py`)

```
python local_train.py [OPTIONS]
```

#### Data Mode (mutually exclusive, pick at most one)

| Flag | Description |
|---|---|
| `--half-data` | Use 50% of images per breed |
| `--quarter-data` | Use 25% of images per breed |
| `--smoke-test` | Use 5 images per breed, 1 epoch per phase |
| `--full-data` | Explicitly use all images (same as default) |

#### Model Configuration

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Backbone architecture (~6M vs ~13M params) |
| `--attention` | `cbam` \| `se` | `cbam` | Attention module type |

#### Training Overrides

| Flag | Type | Default | Description |
|---|---|---|---|
| `--include-qat` | flag | off | Enable Phase 3 QAT for INT8 Android deployment |
| `--phase1-epochs` | int | `5` | Override Phase 1 (binary warmup) epoch count |
| `--phase2-epochs` | int | `40` | Override Phase 2 (multi-task) epoch count |
| `--phase3-epochs` | int | `10` | Override Phase 3 (QAT) epoch count |
| `--num-workers` | int | `4` | DataLoader worker processes |

#### Skip Stages

| Flag | Description |
|---|---|
| `--skip-download` | Skip Kaggle dataset download (data already in `data/raw/`) |
| `--skip-setup` | Skip venv creation (dependencies already installed) |
| `--skip-verify` | Skip architecture verification step |
| `--skip-export` | Skip multi-format export after training |

---

### Example Commands (`local_train.py`)

```bash
# One-command full pipeline — quarter data, fastest local run
python local_train.py --quarter-data

# Half data with QAT (for Android INT8 deployment)
python local_train.py --half-data --include-qat

# Full training with lite4 backbone
python local_train.py --backbone lite4

# Re-run training only (data already downloaded, venv ready)
python local_train.py --half-data --skip-download --skip-setup

# Custom epoch counts
python local_train.py --half-data --phase2-epochs 20

# Skip export (just train, examine checkpoint manually)
python local_train.py --quarter-data --skip-export

# Verify pipeline works end-to-end in seconds
python local_train.py --smoke-test
```

---

## Model Testing GUI (`test_model.py`)

After training, use the standalone web GUI to visually test your models on individual images:

```bash
python test_model.py
```

Opens a browser at `http://localhost:8501` automatically. No virtual environment activation needed if you've already installed dependencies.

**Features:**
- Drag-and-drop or click-to-upload: PNG, JPG, JPEG, BMP, WebP
- **Auto-discovers** all trained checkpoints (Phase 1/2/3, INT8 quantized, portable bundles)
- Species classification: **Cattle** or **Buffalo** with confidence %
- **Top-5 breed predictions** with animated confidence bars
- Model metadata footer (which checkpoint, device info)

**Flags:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--port` | int | `8501` | HTTP port to serve on |
| `--no-browser` | flag | off | Don't auto-open the browser window |

```bash
# Default (auto-opens browser on port 8501)
python test_model.py

# Custom port
python test_model.py --port 9000

# Headless (no browser pop-up, useful on servers)
python test_model.py --no-browser
```

**What it needs:**
- At least one trained checkpoint in `outputs/checkpoints/` or `outputs/export/portable/`
- Class map JSONs in `data/splits/` (auto-generated by training)

---

## Google Colab Training

For GPU-accelerated training on **Google Colab Free (T4 GPU, 15 GB VRAM)**:

**Notebook location:** `colab/cattle_buffalo_trainer.ipynb`

### Colab Setup Workflow

**Step 1 — Load the project (choose one):**

Option A — GitHub clone (recommended, always gets latest code):
```bash
# In a Colab cell — always delete first to avoid stale cache
!rm -rf /content/project
!git clone https://github.com/Bharaths31/ML-CB-B-identifier /content/project
```

Option B — Upload `colab_project.zip` (generated by `python scripts/create_colab_project_zip.py`)

Option C — Google Drive: copy `colab_project.zip` from `My Drive/ML-CB-B-identifier/`

**Step 2 — Load the dataset (choose one):**
- Kaggle API: notebook auto-downloads `algsoch/breed-cattle-buffalo`
- Upload `archive.zip` (created by `python scripts/create_colab_archive.py`)
- Google Drive mount

**Step 3 — Run cells top to bottom.**

### Colab-Specific Optimizations

| Setting | Value | Reason |
|---|---|---|
| `batch_size` | Auto (64 on T4) | Maximizes 15 GB VRAM |
| `grad_accum` | Auto (2 on T4) | Effective batch = 128 |
| `num_workers` | 2 | Colab has 2 CPU cores |
| `prefetch_factor` | 4 | Keeps GPU fed |
| `pin_memory` | True | Faster CPU→GPU transfer |
| AMP | Phases 1–2 only | Disabled for QAT phase 3 |
| Dataset location | `/content/data/raw/` | Local SSD, not Drive |

### Known Colab Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| Re-running clone cell doesn't pick up new code | Colab caches `/content/project/` | Always `rm -rf /content/project` before `git clone` |
| Runtime restart wipes all files | Colab free tier uses ephemeral storage | Re-run all cells from §0 after any restart |

---

## SOTA Features

| Technique | Setting | Effect |
|---|---|---|
| **Optimizer** | AdamW, weight_decay=1e-2 | Decoupled weight decay, better generalization |
| **LR Schedule** | Linear warmup (3 epochs) → Cosine annealing | Stable convergence, avoids early overfitting |
| **Label Smoothing** | ε=0.1 | Prevents overconfident predictions |
| **Augmentation** | RandAugment(ops=2, mag=9) + ColorJitter + CutMix + MixUp | Diverse training signal, reduces overfitting |
| **Gradient Accumulation** | 2 steps → effective batch=128 | Stable gradients on small VRAM GPUs |
| **Mixed Precision (AMP)** | Phases 1–2 | ~2× faster training, lower VRAM usage |
| **Dynamic VRAM Scaling** | Auto batch_size + grad_accum | Adapts to 4 GB → 24 GB GPUs automatically |
| **Byte Caching** | Raw JPEG bytes in RAM | Prevents RAM OOM, eliminates repeated disk I/O |
| **GPU-side CutMix/MixUp** | Operations on CUDA device | Prevents CPU DataLoader bottleneck |
| **Split Ratio** | 85/10/5 stratified | Maximizes training data for rare breeds |

---

## Export & Android Deployment

`local_train.py` automatically exports 4 formats after training. To trigger exports manually:

```bash
# Self-contained portable bundle (Python inference, includes labels + metadata)
python -m src.export --mode portable --backbone lite2

# ONNX (cross-platform, ONNX Runtime Mobile, TFLite converter)
python -m src.export --mode onnx --backbone lite2

# INT8 quantized TorchScript (smallest size, fastest mobile CPU inference)
python -m src.export --mode int8 --backbone lite2

# FP16 TorchScript (mobile GPU inference)
python -m src.export --mode float16 --backbone lite2
```

**Export flag reference:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Which backbone's checkpoint to export |
| `--mode` | `onnx` \| `int8` \| `float16` \| `portable` | required | Export format |
| `--checkpoint` | path | auto (latest) | Specific `.pt` file to export from |

**Output locations:**

```
outputs/export/
├── lite2_fp32.onnx              # ONNX (cross-platform)
├── lite2_int8.pt                # INT8 TorchScript (mobile CPU)
├── lite2_float16.pt             # FP16 TorchScript (mobile GPU)
└── portable/
    └── lite2_phase2_best/
        ├── model.pt             # PyTorch checkpoint (state_dict)
        ├── cattle_classes.json  # {"amritmahal": 0, "ayrshire": 1, ...}
        ├── buffalo_classes.json # {"alambadi": 0, "banni": 1, ...}
        └── model_info.json      # backbone, image_size, usage, exported_at
```

**Model sizes:**

| Backbone | FP32 (ONNX) | INT8 (post-QAT) |
|---|---|---|
| lite2 | ~24 MB | ~6 MB |
| lite4 | ~50 MB | ~13 MB |

**Android workflow:**
1. Train with QAT: `python local_train.py --include-qat`
2. Use `lite2_quantized.pt` with **PyTorch Mobile Android SDK**, or `lite2_fp32.onnx` with **ONNX Runtime Mobile**

---

## Manual / Advanced Setup

Use this section if you want step-by-step control without `local_train.py`.

### Manual Setup — Windows

**Step 1 — Create virtual environment:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Step 2 — Install training dependencies:**
```cmd
pip install --upgrade pip
pip install torch>=2.1.0 torchvision>=0.16.0
pip install numpy pandas matplotlib scikit-learn tqdm Pillow requests onnx
```

**Step 3 — Download dataset:**
```cmd
curl -L -o breed-cattle-buffalo.zip https://www.kaggle.com/api/v1/datasets/download/algsoch/breed-cattle-buffalo
mkdir data\raw
tar -xf breed-cattle-buffalo.zip -C data\raw\
```
*(Windows 10/11 includes `tar`. If not available, use 7-Zip or WinZip.)*

**Step 4 — Prepare data splits:**
```cmd
python -m src.data_pipeline
```

**Step 5 — Verify architecture:**
```cmd
python -m src.verify
```

**Step 6 — Train:**
```cmd
python -m src.train --backbone lite2 --quarter-data --skip-qat
```

---

### Manual Setup — Linux / macOS

**Step 1 — Create virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Step 2 — Install training dependencies:**
```bash
pip install --upgrade pip
pip install torch>=2.1.0 torchvision>=0.16.0
pip install numpy pandas matplotlib scikit-learn tqdm Pillow requests onnx
```

Or install everything at once:
```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes webapp deps (`fastapi`, `uvicorn`, etc.) which are not needed for training-only use. Install selectively if preferred.

**Step 3 — Download dataset:**
```bash
mkdir -p data/raw
curl -L -o breed-cattle-buffalo.zip \
  https://www.kaggle.com/api/v1/datasets/download/algsoch/breed-cattle-buffalo
unzip -q breed-cattle-buffalo.zip -d data/raw/
```

**Step 4 — Prepare data splits:**
```bash
python -m src.data_pipeline
```

**Step 5 — Verify architecture:**
```bash
python -m src.verify
```

**Step 6 — Train:**
```bash
python -m src.train --backbone lite2 --quarter-data --skip-qat
```

---

### Complete Flag Reference (`python -m src.train`)

```
python -m src.train [OPTIONS]
```

> **Note:** This runs the core training engine directly. `local_train.py` wraps this with automated setup, download, and export stages.

#### Model Configuration

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Backbone architecture |
| `--weights` | path | project root `.pth` | Custom pretrained weights path |
| `--attention` | `cbam` \| `se` | `cbam` | Attention module type |

#### Data & I/O

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data` | path | `data/raw/` | Raw image root directory |
| `--split-dir` | path | `data/splits/` | CSV split output directory |
| `--export-dir` | path | `outputs/export/portable/` | Portable export destination |

#### Training Hyperparameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--batch-size` | int | `64` | Per-GPU batch size |
| `--num-workers` | int | `4` | DataLoader worker count |
| `--device` | str | auto | Force device: `cuda` or `cpu` |
| `--seed` | int | `42` | Global random seed |
| `--weight-decay` | float | `1e-2` | AdamW weight decay |
| `--label-smoothing` | float | `0.1` | Label smoothing factor |
| `--warmup-epochs` | int | `3` | Linear LR warmup epochs (Phase 2) |
| `--grad-accum` | int | `2` | Gradient accumulation steps |

#### Phase Control

| Flag | Type | Default | Description |
|---|---|---|---|
| `--phase1-epochs` | int | `5` | Phase 1 epoch count (binary warmup) |
| `--phase2-epochs` | int | `40` | Phase 2 epoch count (multi-task fine-tune) |
| `--phase3-epochs` | int | `10` | Phase 3 epoch count (QAT) |
| `--skip-qat` | flag | off | Skip Phase 3 (QAT) entirely |

#### Data Mode (mutually exclusive)

| Flag | Description |
|---|---|
| `--smoke-test` | 5 images/breed, 1 epoch per phase |
| `--half-data` | 50% of images per breed (seed=42) |
| `--quarter-data` | 25% of images per breed (seed=42) |
| *(none)* | Full dataset (default) |

#### Augmentation & Compilation

| Flag | Description |
|---|---|
| `--no-mix` | Disable CutMix/MixUp batch mixing |
| `--no-compile` | Disable `torch.compile` (automatically set on Windows) |
| `--no-export` | Skip automatic portable export after training |

#### Example `src.train` Commands

```bash
# Quick sanity check — full pipeline in seconds
python -m src.train --smoke-test --skip-qat

# Quarter-data training (recommended for local GPU)
python -m src.train --quarter-data --skip-qat

# Half-data with custom epochs
python -m src.train --half-data --phase2-epochs 20 --skip-qat

# Full training with lite4 backbone
python -m src.train --backbone lite4

# Full training with QAT for Android deployment
python -m src.train --backbone lite2

# Override hyperparameters explicitly
python -m src.train \
  --backbone lite4 \
  --batch-size 32 \
  --phase1-epochs 5 \
  --phase2-epochs 40 \
  --weight-decay 0.01 \
  --label-smoothing 0.1 \
  --device cuda

# Force CPU (debugging)
python -m src.train --quarter-data --device cpu --no-compile
```

---

## Known Constraints & Gotchas

1. **QAT + CUDA AMP conflict**: Phase 3 (QAT) disables AMP because quantization observers don't support mixed precision. This is intentional and handled automatically.

2. **Backbone weights included**: `efficientnet_lite{2,4}.pth` are tracked in the repository — no separate download needed.

3. **`torch.compile` crashes on Windows**: Windows lacks Triton support, causing `BackendCompilerFailed: Cannot find a working triton installation`. `src/train.py` automatically detects `os.name == 'nt'` and falls back to eager execution. Use `--no-compile` to force this manually.

4. **`torch.compile` OOM on T4 (Colab)**: `mode="reduce-overhead"` pre-allocates VRAM via CUDA Graphs, causing OOM on 15 GB T4. The default mode (no `reduce-overhead`) is used instead.

5. **Fixed head sizes**: Model heads remain sized for 57 cattle + 18 buffalo classes even on subset training. Unused outputs are never trained — architecture is identical across all modes.

6. **WeightedRandomSampler**: Training uses inverse-frequency sampling (rare breeds oversampled). Evaluation uses no sampling — test set reflects natural distribution.

7. **Portable export requires class definition**: The portable bundle saves `state_dict`, not TorchScript. Loading requires the `BreedClassifier` class from `src/model.py`. For framework-free deployment, use ONNX instead.

8. **Smoke test uses ALL 75 classes**: Even with only 5 images per breed, class maps include all breeds. Architecture is identical to full training.

9. **Windows build tools (optional)**: PyTorch installs via pre-built wheels and does not need MSVC. C++ Build Tools are only required if a package (e.g., an older `onnx` build) tries to compile C extensions.

---

## Changelog

**2026-09-08 — Model Tester GUI & Quarter-Data Mode**
- Added `test_model.py` — standalone web GUI for visual model testing (port 8501).
- Added `--quarter-data` flag to both `local_train.py` and `src/train.py` — trains on 25% of images/breed.
- Added `--full-data` explicit flag to `local_train.py` (same as default, makes intent clear).
- Added Windows Visual C++ Build Tools prerequisite detection to `local_train.py`.
- **Fix**: `src/train.py` auto-detects Windows and disables `torch.compile` to prevent `BackendCompilerFailed` errors.

**2026-09-07 — Automated Local Training Pipeline**
- Added `local_train.py` — fully automated 8-stage pipeline (prereqs → venv → download → unzip → verify → validate → train → export).
- Added `--half-data` flag for deterministic 50% subset training.

**2026-09-07 — Unified Kaggle Dataset & GPU OOM Fixes**
- Removed `mode="reduce-overhead"` from `torch.compile()` — prevented CUDA Graph OOM on T4.
- Switched to unified `algsoch/breed-cattle-buffalo` dataset (cattle + buffalo in one download).

**2026-09-06 — Colab & SOTA Hyperparameters**
- Added full Google Colab T4 GPU trainer (`colab/cattle_buffalo_trainer.ipynb`).
- SOTA hyperparameters: AdamW, label smoothing (0.1), warmup + cosine LR, RandAugment, gradient accumulation.
- Enabled QAT Phase 3 for INT8 Android deployment.
- Dynamic VRAM auto-scaling (batch_size + grad_accum adjusts to GPU).
- Raw JPEG byte caching — prevents RAM OOM on Colab.

**2026-09-05 — Major Update**
- CUDA optimizations: `cudnn.benchmark`, TF32, AMP, `GradScaler`, `pin_memory`, `persistent_workers`.
- Gradient clipping (`max_norm=1.0`).
- Smoke test uses real mini-dataset (5 imgs/breed) instead of batch limits.
- Auto portable export after training completes.
