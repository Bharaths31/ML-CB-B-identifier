# ML-CB-B-identifier
 # Cattle & Buffalo Breed Classifier

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture & Data Flow](#architecture--data-flow)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup & Virtual Environment](#setup--virtual-environment)
- [Data Preparation](#data-preparation)
- [Local Training (Automated)](#local-training-automated)
- [Google Colab Training Setup](#google-colab-training-setup)
- [SOTA Hyperparameter & Pipeline Suite](#sota-hyperparameter--pipeline-suite)
- [Quick Sanity Check (verify)](#quick-sanity-check-verify)
- [Training](#training)
  - [Full training](#full-training)
  - [Half‑data training](#half-data-training)
  - [Smoke‑test training](#smoke-test-training)
- [Evaluation](#evaluation)
- [Exporting the Model & Android Deployment](#exporting-the-model--android-deployment)
- [Running the FastAPI Webapp](#running-the-fastapi-webapp)
- [Memory Layer (Mem0)](#memory-layer-mem0)
- [Common Scripts & Commands](#common-scripts--commands)
- [Knowledge Base](#knowledge-base)
- [Known Constraints & Gotchas](#known-constraints--gotchas)
- [Changelog](#changelog)

---

## Project Overview
This repository implements a **lightweight, mobile‑deployable image classifier** for Indian cattle (57 breeds) and buffalo (18 breeds). The model uses **EfficientNet‑Lite** backbones (lite2 or lite4) with optional **CBAM/SE** attention modules and three parallel classification heads (binary, cattle, buffalo).

Key features:
- End‑to‑end training pipeline with three phases (binary warm‑up → multi‑task fine‑tune → optional QAT).
- Mixed‑precision (AMP) on CUDA, graceful CPU fallback.
- Automatic export to ONNX, INT8, FP16, or a **portable self‑contained bundle**.
- FastAPI + vanilla JavaScript front‑end for inference, training, evaluation, and a memory‑augmented chat using **Mem0**.

---

## Architecture & Data Flow
```
data/raw/{cattle,buffalo}/<breed>/*.jpg → data_pipeline.prepare_splits() → data/splits/*.csv
→ CattleBuffaloDataset → DataLoader (with CutMix / MixUp) → EfficientNet‑Lite backbone + attention
→ AdaptiveAvgPool2d → flatten → three heads (binary, cattle, buffalo) → predictions
```
The `ModelBox` caches the latest checkpoint and automatically invalidates when a training job finishes.

---

## Directory Structure
```
ML-CB-B-identifier/
├── src/                     # Core ML package (run as `python -m src.<module>`)
│   ├── __init__.py
│   ├── config.py           # Hyper‑parameters & path constants
│   ├── data_pipeline.py    # Dataset, splits, augmentations, DataLoaders
│   ├── model.py            # BreedClassifier definition
│   ├── cbam.py
│   ├── efficientnet_lite.py
│   ├── train.py            # 3‑phase training with AMP & auto‑export
│   ├── metrics.py          # Per‑head accuracy & F1
│   ├── evaluate.py         # Full evaluation with confusion matrices
│   ├── export.py           # ONNX / INT8 / FP16 / portable export
│   └── verify.py           # Quick sanity check (backbone loading & shapes)
├── webapp/                 # FastAPI backend + vanilla JS front‑end
│   ├── server.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── memory/                 # Mem0 vector‑store layer
│   ├── __init__.py
│   └── service.py
├── data/                  # Dataset folders (raw images & generated splits)
│   ├── raw/
│   └── splits/
├── outputs/               # Training checkpoints, exports, metrics, memory store
│   ├── checkpoints/
│   ├── export/
│   │   └── portable/
│   ├── metrics/
│   └── memory/
├── scripts/               # Utility scripts (e.g., colab archive creator)
├── local_train.py         # 🚀 Fully automated local pipeline (setup → download → train → export)
├── test_model.py          # 🔬 Standalone GUI for testing exported models on individual images
├── create_training_zip.py  # Generates a lightweight zip (excludes webapp & memory)
├── setup.sh               # Helper shell script for quick env setup
├── setup_venv.py          # Automated Python virtual‑env creation & dep install
├── requirements.txt       # Python dependencies
└── .gitignore
```
---

## Prerequisites
- **Python ≥ 3.9** (3.11+ recommended)
- **Git** (to clone the repo)
- **CUDA‑capable GPU** (optional – the code falls back to CPU; RTX 3050 4GB+ recommended for local training)
- **Kaggle account** (free — for dataset download; get API key at [kaggle.com/settings](https://www.kaggle.com/settings) → API)
- **Internet connection** (to download the pretrained EfficientNet‑Lite weights and dataset)

---

## Setup & Virtual Environment
The project ships an **automated venv helper** (`setup_venv.py`). It creates a `.venv` directory inside the project root, installs the dependencies from `requirements.txt`, and activates the environment.

```bash
# 1. Clone the repository (if not already done)
git clone <repo‑url>  # replace with actual URL
cd ML-CB-B-identifier

# 2. Create and activate the virtual environment
python setup_venv.py          # creates .venv and runs pip install -r requirements.txt
source .venv/bin/activate     # activate the venv (Linux/macOS)
# On Windows use: .venv\Scripts\activate
```
If you prefer a manual approach:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> **NOTE**: The `.gitignore` already excludes the `.venv` folder, so it won’t be committed.

---

## Data Preparation

### 1. Download the Dataset
Download the dataset archive (`breed-cattle-buffalo.zip`) from Kaggle using `curl`:

```bash
#!/bin/bash
curl -L -o ~/Downloads/breed-cattle-buffalo.zip \
  https://www.kaggle.com/api/v1/datasets/download/algsoch/breed-cattle-buffalo
```

### 2. Extract & Prepare Images for Training
Extract the downloaded archive into `data/raw/` in the project root:

```bash
# Create the target directory
mkdir -p data/raw

# Unzip the dataset into data/raw/
unzip -q ~/Downloads/breed-cattle-buffalo.zip -d data/raw/
```

Ensure the extracted raw images follow the expected species and breed directory layout:
```text
data/raw/
├── cattle/
│   ├── gir/
│   ├── sahiwal/
│   └── ... (57 cattle breeds)
└── buffalo/
    ├── murrah/
    ├── jafarabadi/
    └── ... (18 buffalo breeds)
```

### 3. Generate Train / Val / Test Splits
Run the data pipeline script to scan `data/raw/` and generate stratified split CSVs (`train.csv`, `val.csv`, `test.csv`) under `data/splits/`:

```bash
python -m src.data_pipeline   # scans data/raw/ and writes CSVs to data/splits/
```

> **Optimized Split Strategy**:
> - **Train Ratio**: `85%` (0.85) — allocated more training images to maximize accuracy across 75 fine-grained breeds.
> - **Val Ratio**: `10%` (0.10) — used for hyperparameter tuning and learning rate scheduling.
> - **Test Ratio**: `5%` (0.05) — preserved for final benchmark evaluation.

---

## Local Training (Automated)

The **`local_train.py`** script provides a **fully automated pipeline** that handles everything from environment setup to model export in a single command. It is designed for local machines with GPUs like the RTX 3050 (4GB VRAM).

### Quick Start (One Command)

```bash
# Clone the repo and run the automation script
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier

# Run fully automated pipeline (half-data for faster training)
python local_train.py --half-data
```

The script will:
1. ✅ Check prerequisites (Python, git, pip)
2. ✅ Create a virtual environment and install training dependencies
3. ✅ Prompt for your **Kaggle API credentials** (username + key) if not already configured
4. ✅ Download the `algsoch/breed-cattle-buffalo` dataset from Kaggle
5. ✅ Extract and organize images into `data/raw/cattle/` and `data/raw/buffalo/`
6. ✅ Verify model architecture (backbone loading + forward pass)
7. ✅ Train the model (3-phase pipeline with VRAM auto-scaling)
8. ✅ Export the model in 4 formats: portable, ONNX, INT8, float16

### Kaggle API Setup

When you run the script for the first time without existing credentials, it will prompt you:

```
  ┌─────────────────────────────────────────────┐
  │  Kaggle API credentials required             │
  │  Get your key at: kaggle.com/settings → API │
  └─────────────────────────────────────────────┘

  Enter your Kaggle username: your_username
  Enter your Kaggle API key:  your_api_key_here
```

Credentials are saved to `~/.kaggle/kaggle.json` and reused on subsequent runs.

### All `local_train.py` Flags

| Flag | Description |
|------|-------------|
| `--half-data` | Use **50% of images** per breed for faster training |
| `--quarter-data` | Use **25% of images** per breed for fastest local training |
| `--smoke-test` | Tiny dataset (5 imgs/breed), 1 epoch per phase — quick sanity check |
| `--full-data` | Use all images (default behavior) |
| `--backbone {lite2,lite4}` | Backbone architecture (default: `lite2`, ~6M params) |
| `--attention {cbam,se}` | Attention module (default: `cbam`) |
| `--include-qat` | Enable QAT Phase 3 for Android INT8 deployment (default: skipped) |
| `--phase1-epochs N` | Override Phase 1 epoch count |
| `--phase2-epochs N` | Override Phase 2 epoch count |
| `--phase3-epochs N` | Override Phase 3 epoch count |
| `--num-workers N` | DataLoader worker count |
| `--skip-download` | Skip Kaggle download (dataset already present in `data/raw/`) |
| `--skip-setup` | Skip venv creation (environment already configured) |
| `--skip-verify` | Skip architecture verification step |
| `--skip-export` | Skip multi-format export after training |

---

## Google Colab Training Setup

For GPU-accelerated training using **Google Colab Free T4 GPU** (15 GB VRAM, 2 CPU cores), use the dedicated notebook located in `colab/`:

- **Jupyter Notebook**: `colab/cattle_buffalo_trainer.ipynb`
- **Percent Script**: `colab/cattle_buffalo_trainer.py`

### Colab Workflow Overview
1. **Environment Setup (Flexible Setup Options)**:
   - **Option A (Recommended)**: Clone directly from the GitHub repository (`https://github.com/Bharaths31/ML-CB-B-identifier`).
   - **Option B**: Upload `colab_project.zip` generated via `python scripts/create_colab_project_zip.py`.
   - **Option C**: Mount Google Drive and link the project folder.
2. **Dataset Loading**:
   - **Option A (Recommended)**: Direct Kaggle dataset download via `curl` with auto-extraction into `data/raw/`.
   - **Option B**: Upload custom zip or load from Google Drive.
3. **Hyperparameter Customization**:
   - Interactively configure `BATCH_SIZE`, `PHASE1_LR`, `PHASE2_LR`, `PHASE3_LR`, `EPOCHS`, `WEIGHT_DECAY`, `LABEL_SMOOTHING`, and `WARMUP_EPOCHS` directly in dedicated notebook cells.
4. **Three-Phase Mobile-Optimized Training**:
   - Phase 1: Binary species warm-up (`Cattle vs Buffalo`).
   - Phase 2: Full multi-task breed fine-tuning (`AdamW + Cosine Scheduler + Label Smoothing + CutMix/MixUp`).
   - Phase 3: Quantization-Aware Training (QAT) for INT8 Android mobile deployment.
5. **Interactive Prediction & Evaluation**:
   - Predict species and breed on any uploaded test image using `evaluate_single_image()`.
   - Download trained model checkpoints, ONNX exports, or portable zip directly to local disk or Google Drive.

---

## SOTA Hyperparameter & Pipeline Suite

The ML pipeline is upgraded with State-of-the-Art (SOTA) computer vision techniques:

| Hyperparameter / Feature | Previous Value | New SOTA Value | Rationale |
|--------------------------|----------------|----------------|-----------|
| **Train / Val / Test Split** | `80 / 10 / 10` | `85 / 10 / 5` | Provides 5% more training images per breed to combat class imbalance |
| **Batch Size** | `32` | **Auto-Scaled (16 to 128)** | Dynamically adapts to local VRAM (e.g. 4GB RTX 3050 up to 24GB RTX 4090) |
| **Optimizer** | `Adam` | `AdamW` | Weight decay regularizes deep EfficientNet-Lite feature extractors |
| **Weight Decay** | `0.0` | `1e-2` (0.01) | Prevents overfitting on high-resolution fine-grained breed features |
| **Label Smoothing** | `0.0` | `0.1` | Prevents overconfidence on visually similar cattle/buffalo breeds |
| **LR Scheduler** | Step / Constant | **Linear Warmup (3 ep) + Cosine Annealing** | Prevents initial gradient shocks and ensures smooth convergence |
| **Gradient Accumulation** | `1` step | **Auto-Scaled** | Maintains an effective batch size of 128 regardless of VRAM constraints |
| **Dropout** | `0.3` | `0.4` | Enhanced regularization on dense classification heads |
| **Data Augmentation** | Standard Flip/Color | **RandAugment + GPU-Accelerated CutMix/MixUp** | SOTA regularization offloaded to GPU to prevent CPU bottlenecks |
| **Data Caching** | Disk reads or PIL Caching | **Raw JPEG Byte Caching** | Prevents RAM OOM errors while completely skipping slow disk IO after epoch 1 |
| **DataLoader Prefetching** | Standard | `prefetch_factor=4`, `pin_memory=True` | Eliminates CPU-GPU bottleneck on Colab T4 |

---

## Quick Sanity Check (verify)
Run the verification script to ensure the backbone loads correctly and the forward‑pass shapes match expectations:
```bash
python -m src.verify
```
A successful run prints the feature dimension (1280) and confirms the heads are correctly wired.

---

## Training

### Training Arguments Reference

```
python -m src.train [OPTIONS]

--backbone {lite2,lite4}     Backbone architecture (default: lite2)
--weights PATH               Pretrained weights path
--attention {cbam,se}        Attention module (default: cbam)
--data PATH                  Raw data root
--split-dir PATH             Split CSV output directory
--batch-size N               Batch size (auto-scaled by VRAM)
--num-workers N              DataLoader workers (default: 4)
--device DEVICE              Force device (auto-detects cuda/cpu)
--no-mix                     Disable CutMix/MixUp batch mixing
--phase1-epochs N            Override phase 1 epoch count (default: 5)
--phase2-epochs N            Override phase 2 epoch count (default: 40)
--phase3-epochs N            Override phase 3 epoch count (default: 10)
--skip-qat                   Skip phase 3 (QAT)
--smoke-test                 Use mini-dataset (5 imgs/breed, 1 epoch)
--half-data                  Use 50% of images per breed (faster training)
--quarter-data               Use 25% of images per breed (fastest local training)
--seed N                     Random seed (default: 42)
--export-dir PATH            Portable export destination
--no-export                  Skip auto-export after training
--weight-decay FLOAT         AdamW weight decay (default: 0.01)
--label-smoothing FLOAT      Label smoothing factor (default: 0.1)
--warmup-epochs N            Linear warmup epochs for phase 2 (default: 3)
--grad-accum N               Gradient accumulation steps (default: 2)
--no-compile                 Disable torch.compile
```

### Full training
```bash
python -m src.train --backbone lite2   # default hyper‑params (5/40/10 epochs per phase)
```
You can override any argument (e.g., `--batch-size 64`, `--device cuda`). All three phases run sequentially; after phase 2 the model is **auto‑exported** to a portable bundle.

### Half‑data training
Trains on **50% of images per breed** — useful for faster iteration on local machines with limited VRAM (e.g., RTX 3050 4GB):
```bash
python -m src.train --half-data --skip-qat
```
The `--half-data` flag:
- Deterministically samples 50% of images per breed (seed=42 for reproducibility)
- Applies the same 85/10/5 stratified split on the sampled subset
- Class maps still include ALL breeds — model architecture is identical to full training
- Mutually exclusive with `--smoke-test` and `--quarter-data`

### Quarter‑data training
Trains on **25% of images per breed** — the fastest mode for resource-constrained local machines:
```bash
python -m src.train --quarter-data --skip-qat
```
The `--quarter-data` flag:
- Deterministically samples 25% of images per breed (seed=42)
- Applies the same 85/10/5 stratified split on the sampled subset
- Class maps include ALL breeds — identical model architecture to full training
- Mutually exclusive with `--smoke-test` and `--half-data`

### Smoke‑test training
A fast sanity‑check that trains on 5 images per breed for a single epoch per phase:
```bash
python -m src.train --smoke-test --skip-qat
```
Ideal for CI pipelines or quick verification of the whole pipeline.

---

## Evaluation
After training (or on any checkpoint), evaluate the model and generate confusion‑matrix PNGs:
```bash
python -m src.evaluate --backbone lite2   # uses the latest checkpoint in outputs/checkpoints/
```
Metrics are stored in `outputs/metrics/` as JSON files and accompanying PNG visualisations.

---

## Exporting the Model & Android Deployment

The `export` module supports four export modes for mobile and edge deployment:

| Mode | Command | Result | Deployment Target |
|------|---------|--------|-------------------|
| `onnx` | `python -m src.export --mode onnx --backbone lite2` | `<backbone>_fp32.onnx` (opset 13) | Cross-platform / ONNX Runtime Mobile |
| `int8` | `python -m src.export --mode int8 --backbone lite2` | QAT/PTQ INT8 quantized checkpoint | Android NNAPI / Edge CPU |
| `float16` | `python -m src.export --mode float16 --backbone lite2` | TorchScript FP16 traced model | Mobile GPU (Vulkan/Metal) |
| `portable` | `python -m src.export --mode portable --backbone lite2` | Self-contained folder with `model.pt`, class maps, and `model_info.json` | Python / C++ embedded inference |

### Android Deployment Workflow
1. Run Phase 3 **Quantization-Aware Training (QAT)** via `python -m src.train --phase3-epochs 10` or via the Google Colab notebook cell.
2. Export the INT8 ONNX or TorchScript bundle using `python -m src.export --mode onnx` or `--mode int8`.
3. Load the INT8 model in Android using **ONNX Runtime for Android** or **PyTorch Mobile Android SDK** for real-time offline breed classification on mid-range smartphones.

---

## Running the FastAPI Webapp
Start the backend server (after activating the venv):
```bash
python webapp/server.py   # listens on http://localhost:8000
```
The single‑page front‑end (`webapp/static/index.html`) provides tabs for:
- **Predict** – upload an image and get the breed prediction.
- **Train** – launch a training job; progress bars are updated via polling (≈1.2 s interval).
- **Evaluate** – run evaluation on the current checkpoint.
- **Memory** – interact with the Mem0 vector store (store/retrieve chat context).
- **Debug** – view raw logs, system info, and current checkpoints.

The UI uses the `JobRunner` component to spawn subprocesses (`src.train`, `src.evaluate`, `src.export`) and parses their stdout in real‑time.

---

## Memory Layer (Mem0)
The optional memory service (`memory/service.py`) provides a **ChromaDB‑backed vector store** for LLM‑augmented context. It is scoped by `user_id`, `agent_id`, and `run_id`. The webapp can store/retrieve memories via the `/api/memory/*` endpoints (not listed in the API table but available in the code).

---

## Common Scripts & Commands
| Script / Notebook | Purpose |
|-------------------|---------|
| **`local_train.py`** | 🚀 **Fully automated pipeline**: venv → Kaggle download → unzip → train → export |
| **`test_model.py`** | 🔬 **Standalone GUI** for testing exported models on individual images |
| `setup_venv.py` | Creates `.venv` + installs `requirements.txt` |
| `setup.sh` | Convenience wrapper that calls `setup_venv.py` and prints usage |
| `scripts/create_colab_project_zip.py` | Generates lightweight `colab_project.zip` containing `src/`, `requirements.txt`, and pretrained weights |
| `colab/cattle_buffalo_trainer.ipynb` | Google Colab Jupyter Notebook optimized for T4 GPU free tier |
| `colab/cattle_buffalo_trainer.py` | Percent-script format of the Colab trainer notebook |
| `src/data_pipeline.py` | Scans `data/raw/` and writes 85/10/5 CSV splits |
| `src/verify.py` | Architecture sanity check |
| `src/train.py` | 3‑phase training pipeline (AdamW, Cosine LR, QAT, Label Smoothing) |
| `src/evaluate.py` | Computes per‑head accuracy, F1, confusion matrices, and single-image prediction |
| `src/export.py` | Export to ONNX / INT8 / FP16 / portable bundle |
| `webapp/server.py` | FastAPI backend with endpoints for prediction, training, evaluation, export, and memory |

### Workflow A: Automated (Recommended)
```bash
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier
python local_train.py --half-data    # One command does everything
```

### Workflow B: Manual Step-by-Step
```bash
# 1. Setup
python setup_venv.py && source .venv/bin/activate

# 2. Prepare data (download + extract + split)
mkdir -p data/raw
# Download from Kaggle (requires ~/.kaggle/kaggle.json)
python -m kaggle datasets download -d algsoch/breed-cattle-buffalo -p data/
unzip -q data/breed-cattle-buffalo.zip -d data/raw/
python -m src.data_pipeline

# 3. Verify architecture
python -m src.verify

# 4. Train (choose one)
python -m src.train --backbone lite2               # Full data
python -m src.train --half-data --skip-qat          # Half data (faster)
python -m src.train --smoke-test --skip-qat         # Smoke test (sanity)

# 5. Evaluate
python -m src.evaluate --backbone lite2

# 6. Export
python -m src.export --mode portable --backbone lite2
python -m src.export --mode onnx --backbone lite2

# 7. Serve webapp
python webapp/server.py   # → http://localhost:8000
```

---

## Knowledge Base
For complete architecture specifications, dataset schema, training flow, Google Colab workflow, and Android quantization details, refer to the project knowledge base:
- **Full Architecture & System Specification**: [`.agents/knowledge/CONTEXT.md`](file:///home/ragnarok/Documents/College/ML-CB-B-identifier/.agents/knowledge/CONTEXT.md)

---

## Known Constraints & Gotchas
- **QAT + CUDA AMP**: Phase 3 disables the AMP scaler because quantization observers aren’t compatible with mixed precision.
- **Backbone weight files** (`efficientnet_lite2.pth`, `efficientnet_lite4.pth`) must exist in the repo root; otherwise the backbone is trained from scratch.
- **Fixed head sizes**: Even if the dataset contains fewer breeds, the model heads remain sized for 57 cattle and 18 buffalo classes – unused outputs are never trained.
- **WeightedRandomSampler** balances breeds during training; evaluation uses the original validation split without sampling.
- **Portable export** stores a `state_dict`; loading requires the `BreedClassifier` class definition (i.e., the Python code). For framework‑free inference, export to ONNX instead.
- **Smoke‑test** still creates class‑maps for *all* breeds, ensuring the architecture matches full training.

---

## Changelog

**2026‑09‑08 – Model Tester GUI & Quarter-Data Mode**
- Added `test_model.py` — standalone GUI for testing exported models on individual images.
  - Upload PNG/JPG/JPEG/BMP/WebP images via drag-and-drop or file browser.
  - Select from all available model checkpoints (Phase 1/2/3, quantized, portable).
  - Returns species (Cattle/Buffalo) with confidence %, top-5 breed predictions with animated confidence bars.
  - Self-contained: runs at `http://localhost:8501`, auto-opens browser.
- Added `--quarter-data` flag — trains on 25% of images/breed for fastest local training.
- Added Windows Visual C++ Build Tools prerequisite check to `local_train.py`.
- All data modes are now a proper mutually-exclusive argparse group.


- Added `local_train.py` — fully automated pipeline: prerequisites → venv → Kaggle download → unzip → verify → train → multi-format export.
- Interactive Kaggle API credential input (prompts user for username + key, saves to `~/.kaggle/kaggle.json`).
- Added `--half-data` flag to `src/train.py` — trains on 50% of images per breed for faster local training.
- Added `prepare_half_splits()` to `src/data_pipeline.py` — deterministic 50% sampling with identical model architecture.
- Added comprehensive exception handling to `local_train.py` to prevent crashes during dataset download and environment setup.
- Updated documentation across README, docs/, and CONTEXT.md knowledge base.

**2026-09-07 – Fix: `torch.compile` OOM on GPU**
- Removed `mode="reduce-overhead"` from `torch.compile()` to prevent CUDA Graph OOM on T4.

**2026-09-07 – Unified Kaggle Dataset & Colab Update**
- Switched to unified `algsoch/breed-cattle-buffalo` Kaggle dataset.
- Simplified Colab download logic.

**2026-09-06 – Hotfix: Colab `total_memory` & Clone Fix**
- Fixed `AttributeError` for `total_mem` → `total_memory`.
- Enhanced Colab GitHub clone cell with `rm -rf` for fresh code.

**2026‑09‑06 – Google Colab & SOTA Mobile Training Suite**
- Added full Google Colab T4 GPU trainer.
- SOTA hyperparameters: AdamW, label smoothing, warmup + cosine LR, gradient accumulation, RandAugment.
- QAT (Quantization-Aware Training) for Android INT8 deployment.

**2026‑09‑05 – Major Update**
- CUDA optimizations, real smoke-test dataset, auto-export, webapp fixes.

---

*Happy coding! 🎉*

