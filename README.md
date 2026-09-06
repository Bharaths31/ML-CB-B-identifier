# ML-CB-B-identifier
 # Cattle & Buffalo Breed Classifier

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture & Data Flow](#architecture--data-flow)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup & Virtual Environment](#setup--virtual-environment)
- [Data Preparation](#data-preparation)
- [Google Colab Training Setup](#google-colab-training-setup)
- [SOTA Hyperparameter & Pipeline Suite](#sota-hyperparameter--pipeline-suite)
- [Quick Sanity Check (verify)](#quick-sanity-check-verify)
- [Training](#training)
  - [Full training](#full-training)
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
├── create_training_zip.py  # Generates a lightweight zip (excludes webapp & memory)
├── setup.sh               # Helper shell script for quick env setup
├── setup_venv.py          # Automated Python virtual‑env creation & dep install
├── requirements.txt       # Python dependencies
└── .gitignore
```
---

## Prerequisites
- **Python ≥ 3.11**
- **Git** (to clone the repo)
- **CUDA‑capable GPU** (optional – the code falls back to CPU automatically)
- **Internet connection** (to download the pretrained EfficientNet‑Lite weights)

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

## Google Colab Training Setup

For GPU-accelerated training using **Google Colab Free T4 GPU** (15 GB VRAM, 2 CPU cores), use the dedicated notebook located in `colab/`:

- **Jupyter Notebook**: [`colab/cattle_buffalo_trainer.ipynb`](file:///home/ragnarok/Documents/College/ML-CB-B-identifier/colab/cattle_buffalo_trainer.ipynb)
- **Percent Script**: [`colab/cattle_buffalo_trainer.py`](file:///home/ragnarok/Documents/College/ML-CB-B-identifier/colab/cattle_buffalo_trainer.py)

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
### Full training
```bash
python -m src.train --backbone lite2   # default hyper‑params (5/30/10 epochs per phase)
```
You can override any argument (e.g., `--batch-size 64`, `--device cuda`). All three phases run sequentially; after phase 2 the model is **auto‑exported** to a portable bundle.

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

Typical workflow:
1. **Setup** → `python setup_venv.py && source .venv/bin/activate`
2. **Prepare data** → `python -m src.data_pipeline`
3. **Verify** → `python -m src.verify`
4. **Train** → `python -m src.train` (or smoke‑test)
5. **Evaluate** → `python -m src.evaluate`
6. **Export** → `python -m src.export --mode portable`
7. **Serve** → `python webapp/server.py`

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
**2026-09-06 – Hotfix: Colab `total_memory` & Clone Fix**
- Fixed an `AttributeError` caused by using `total_mem` instead of PyTorch's `total_memory` in `setup_device()`.
- Enhanced Colab notebook's GitHub clone cell with robust `rm -rf` and error checking to ensure fresh code is pulled properly.
- Updated `CONTEXT.md` knowledge base with a Colab Gotchas table.

**2026‑09‑06 – Google Colab & SOTA Mobile Training Suite**
- Added full Google Colab T4 GPU trainer (`colab/cattle_buffalo_trainer.ipynb` and `colab/cattle_buffalo_trainer.py`).
- Added 3 project setup methods (GitHub public clone `https://github.com/Bharaths31/ML-CB-B-identifier`, zip upload, Google Drive).
- Updated dataset split ratio to `85% Train / 10% Val / 5% Test` for improved fine-grained breed accuracy.
- Integrated SOTA hyperparameters: `AdamW`, `weight_decay=1e-2`, `label_smoothing=0.1`, `warmup_epochs=3`, `CosineAnnealingLR`, `gradient_accumulation_steps=2`, `RandAugment`, `RandomResizedCrop`.
- Enabled QAT (Quantization-Aware Training) for mid-range Android smartphone INT8 ONNX/PyTorch deployment.
- Updated project requirements and comprehensive Knowledge Base (`.agents/knowledge/CONTEXT.md`).

**2026‑09‑05 – Major Update**
- Added CUDA optimizations (`cudnn.benchmark`, TF32, AMP, gradient clipping).
- Implemented real mini‑dataset for smoke‑test (5 images per breed).
- Auto‑portable export after training.
- Improved progress‑bar UI and model cache invalidation.
- Introduced `create_training_zip.py` for clean training packages.
- Updated `.gitignore` to ignore `.venv/`, `outputs/`, and data splits.
- Fixed argument naming bug in the webapp (`--phase1-epochs`).
- Added several UI/UX enhancements (pulse animation, auto‑refresh).

---

*Happy coding! 🎉*

