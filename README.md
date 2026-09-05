# ML-CB-B-identifier
 # Cattle & Buffalo Breed Classifier

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture & Data Flow](#architecture--data-flow)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup & Virtual Environment](#setup--virtual-environment)
- [Data Preparation](#data-preparation)
- [Quick Sanity Check (verify)](#quick-sanity-check-verify)
- [Training](#training)
  - [Full training](#full-training)
  - [Smoke‑test training](#smoke-test-training)
- [Evaluation](#evaluation)
- [Exporting the Model](#exporting-the-model)
- [Running the FastAPI Webapp](#running-the-fastapi-webapp)
- [Memory Layer (Mem0)](#memory-layer-mem0)
- [Common Scripts & Commands](#common-scripts--commands)
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

> **Note**: For a quick smoke-test, the training script can generate tiny splits automatically (`python -m src.train --smoke-test --skip-qat`).

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

## Exporting the Model
The `export` module supports four modes:
| Mode | Command | Result |
|------|---------|--------|
| `onnx` | `python -m src.export --mode onnx --backbone lite2` | `<backbone>_fp32.onnx` (opset 13) |
| `int8` | `python -m src.export --mode int8 --backbone lite2` | PTQ‑calibrated INT8 checkpoint |
| `float16` | `python -m src.export --mode float16 --backbone lite2` | TorchScript‑traced FP16 model |
| `portable` | `python -m src.export --mode portable --backbone lite2` | Self‑contained folder with `model.pt`, class‑maps, and `model_info.json` |

The portable bundle is ready for framework‑free deployment (e.g., embedded C++ inference) because it contains the raw `state_dict` and all label maps.

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
| Script | Purpose |
|--------|---------|
| `setup_venv.py` | Creates `.venv` + installs `requirements.txt` |
| `setup.sh` | Convenience wrapper that calls `setup_venv.py` and prints usage |
| `create_training_zip.py` | Packages a minimal training archive (`training_package.zip`) that excludes the webapp and memory layers – useful for distribution or cloud training |
| `src/data_pipeline.py` | Scans `data/raw/` and writes CSV splits |
| `src/verify.py` | Architecture sanity check |
| `src/train.py` | 3‑phase training pipeline (supports `--smoke-test`, `--skip-qat`, device selection) |
| `src/evaluate.py` | Computes per‑head accuracy, F1, and confusion matrices |
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

## Known Constraints & Gotchas
- **QAT + CUDA AMP**: Phase 3 disables the AMP scaler because quantization observers aren’t compatible with mixed precision.
- **Backbone weight files** (`efficientnet_lite2.pth`, `efficientnet_lite4.pth`) must exist in the repo root; otherwise the backbone is trained from scratch.
- **Fixed head sizes**: Even if the dataset contains fewer breeds, the model heads remain sized for 57 cattle and 18 buffalo classes – unused outputs are never trained.
- **WeightedRandomSampler** balances breeds during training; evaluation uses the original validation split without sampling.
- **Portable export** stores a `state_dict`; loading requires the `BreedClassifier` class definition (i.e., the Python code). For framework‑free inference, export to ONNX instead.
- **Smoke‑test** still creates class‑maps for *all* breeds, ensuring the architecture matches full training.

---

## Changelog
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

