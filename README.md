# ML-CB-B-identifier
# Cattle & Buffalo Breed Classifier

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture & Data Flow](#architecture--data-flow)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup & Automated Training](#setup--automated-training)
- [Google Colab Training Setup](#google-colab-training-setup)
- [SOTA Features](#sota-features)
- [Evaluation & Testing (GUI)](#evaluation--testing-gui)
- [Exporting & Android Deployment](#exporting--android-deployment)
- [Running the FastAPI Webapp](#running-the-fastapi-webapp)
- [Advanced / Manual Setup](#advanced--manual-setup)
- [Knowledge Base](#knowledge-base)
- [Known Constraints & Gotchas](#known-constraints--gotchas)
- [Changelog](#changelog)

---

## Project Overview
This repository implements a **lightweight, mobile‑deployable image classifier** for Indian cattle (57 breeds) and buffalo (18 breeds). The model uses **EfficientNet‑Lite** backbones (lite2 or lite4) with optional **CBAM/SE** attention modules and three parallel classification heads (binary, cattle, buffalo).

Key features:
- End‑to‑end automated training pipeline with three phases (binary warm‑up → multi‑task fine‑tune → optional QAT).
- Mixed‑precision (AMP) on CUDA, graceful CPU fallback.
- Automatic export to ONNX, INT8, FP16, or a **portable self‑contained bundle**.
- FastAPI + vanilla JavaScript front‑end for inference, training, evaluation, and a memory‑augmented chat using **Mem0**.
- Standalone GUI for testing exported models.

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
├── webapp/                  # FastAPI backend + vanilla JS front‑end
├── memory/                  # Mem0 vector‑store layer
├── data/                    # Dataset folders (raw images & generated splits)
├── outputs/                 # Training checkpoints, exports, metrics, memory store
├── scripts/                 # Utility scripts (e.g., colab archive creator)
├── local_train.py           # 🚀 Fully automated local pipeline (setup → download → train → export)
├── test_model.py            # 🔬 Standalone GUI for testing exported models
├── create_training_zip.py   # Generates a lightweight zip (excludes webapp & memory)
├── setup.sh                 # Helper shell script for quick env setup
├── setup_venv.py            # Automated Python virtual‑env creation & dep install
└── requirements.txt         # Python dependencies
```

---

## Prerequisites
- **Python ≥ 3.9** (3.11+ recommended)
- **Git** (to clone the repo)
- **CUDA‑capable GPU** (optional – falls back to CPU; RTX 3050 4GB+ recommended)
- **Kaggle account** (free — get API key at [kaggle.com/settings](https://www.kaggle.com/settings) → API)

---

## Setup & Automated Training

The **`local_train.py`** script is the **recommended** way to run the project. It handles everything from environment setup to downloading the dataset, verifying the architecture, training the model, and exporting the final formats.

### Quick Start (One Command)
```bash
# Clone the repo and run the automation script
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier

# Run fully automated pipeline (quarter-data for the fastest local test)
python local_train.py --quarter-data
```

The script will:
1. Check prerequisites (Python, git, pip, Windows build tools).
2. Create a virtual environment (`.venv`) and install dependencies.
3. Prompt for your Kaggle API credentials (saved to `~/.kaggle/kaggle.json`).
4. Download and extract the Kaggle dataset (`algsoch/breed-cattle-buffalo`).
5. Verify model architecture.
6. Train the model using VRAM auto-scaling.
7. Auto-export to portable, ONNX, INT8, and float16 formats.

### Data Modes (Mutually Exclusive)
You can control how much data is used for training depending on your hardware:

| Flag | Description |
|------|-------------|
| `--quarter-data` | **Recommended for quick iteration.** Uses 25% of images/breed. |
| `--half-data` | Uses 50% of images/breed. Good balance of speed and accuracy. |
| `--full-data` | Uses all images. Longest training time but highest accuracy. |
| `--smoke-test` | Tiny dataset (5 imgs/breed), 1 epoch per phase. For CI/sanity checks. |

*Other useful flags:*
- `--include-qat`: Enable Phase 3 (Quantization-Aware Training) for Android INT8.
- `--backbone {lite2,lite4}`: Choose backbone size (default: lite2).

---

## Google Colab Training Setup

For GPU-accelerated training using a **Google Colab Free T4 GPU** (15 GB VRAM), use the dedicated notebook:
- **Jupyter Notebook**: `colab/cattle_buffalo_trainer.ipynb`

### Workflow
1. **Clone the repo** in Colab (`!rm -rf /content/project && git clone ...`).
2. The notebook auto-downloads the dataset directly from Kaggle to Colab's fast local SSD.
3. Interactively configure hyperparameters in the notebook cells.
4. Train and evaluate the model, then download the portable bundle or ONNX/INT8 exports directly.

---

## SOTA Features
The pipeline utilizes modern State-of-the-Art computer vision techniques:
- **Optimizer**: `AdamW` with decoupled weight decay (`1e-2`).
- **Learning Rate**: Linear warmup (3 epochs) followed by Cosine Annealing.
- **Regularization**: Label smoothing (`0.1`), Dropout (`0.4`).
- **Augmentation**: GPU-accelerated CutMix/MixUp, RandAugment, ColorJitter.
- **Memory Optimization**: Raw JPEG byte caching (prevents RAM OOM on Colab), dynamic VRAM auto-scaling (batch size/grad accum adjusts to your hardware).
- **Split**: 85/10/5 stratified split to maximize training exposure for rare breeds.

---

## Evaluation & Testing (GUI)

### Model Tester GUI
To visually test your trained models on individual images, run the standalone GUI:
```bash
python test_model.py
```
This launches a Streamlit app at `http://localhost:8501`. You can:
- Drag-and-drop images (PNG/JPG/BMP/WebP).
- Select from any of your trained checkpoints (`phase1`, `phase2`, `quantized`, or `portable`).
- View the species (Cattle/Buffalo) and animated confidence bars for the top-5 breed predictions.

### Command-Line Evaluation
To generate metrics and confusion matrices on the test set:
```bash
python -m src.evaluate --backbone lite2
```

---

## Exporting & Android Deployment

If you want to manually trigger exports after training:
```bash
python -m src.export --mode onnx --backbone lite2
python -m src.export --mode int8 --backbone lite2
python -m src.export --mode portable --backbone lite2
```

### Android Workflow
1. Train with QAT: `python local_train.py --include-qat`
2. The resulting `lite2_quantized.pt` (or `lite2_int8.pt`) is true INT8 and can be deployed using the **PyTorch Mobile Android SDK** or converted via **ONNX Runtime Mobile**.

---

## Running the FastAPI Webapp

Start the backend server to access the full web dashboard:
```bash
python webapp/server.py   # listens on http://localhost:8000
```
The single‑page front‑end (`webapp/static/index.html`) provides tabs for Predict, Train, Evaluate, Memory (Mem0), and Debug.

---

## Advanced / Manual Setup

If you prefer to run the pipeline steps granularly without `local_train.py`:

**1. Virtual Environment:**
```bash
python setup_venv.py
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

**2. Manual Data Prep:**
```bash
mkdir -p data/raw data/splits
curl -L -o breed-cattle-buffalo.zip https://www.kaggle.com/api/v1/datasets/download/algsoch/breed-cattle-buffalo
unzip -q breed-cattle-buffalo.zip -d data/raw/
python -m src.data_pipeline
```

**3. Verification & Training:**
```bash
python -m src.verify
python -m src.train --quarter-data --skip-qat
```

---

## Knowledge Base
For complete architecture specifications, dataset schema, and Android quantization details, refer to the project knowledge base:
- **Full Architecture & System Specification**: [`.agents/knowledge/CONTEXT.md`](file:///home/ragnarok/Documents/College/ML-CB-B-identifier/.agents/knowledge/CONTEXT.md)

---

## Known Constraints & Gotchas
- **QAT + CUDA AMP**: Phase 3 disables the AMP scaler because quantization observers aren’t compatible with mixed precision.
- **Backbone weights**: `efficientnet_lite{2,4}.pth` must exist in the repo root; otherwise the backbone trains from scratch (handled automatically by `local_train.py`).
- **Fixed head sizes**: The model heads remain sized for 57 cattle and 18 buffalo classes even if you train on fewer. Unused outputs are ignored.
- **Windows compatibility**: PyTorch compilation (`torch.compile`) via Triton is not supported on Windows. The code automatically detects Windows and falls back to eager execution.

---

## Changelog

**2026‑09‑08 – Model Tester GUI & Quarter-Data Mode**
- Added `test_model.py` — standalone GUI for testing exported models on individual images.
- Added `--quarter-data` flag — trains on 25% of images/breed for fastest local training.
- Added Windows Visual C++ Build Tools prerequisite check to `local_train.py`.
- **Fix**: Added OS detection in `src/train.py` to automatically disable `torch.compile` (fallback to eager mode) on Windows to prevent `BackendCompilerFailed: Cannot find a working triton installation` errors.

**2026-09-07 – Automated Local Training Pipeline**
- Added `local_train.py` — fully automated pipeline.
- Added `--half-data` flag for deterministic 50% subset training.

**2026-09-07 – Unified Kaggle Dataset & GPU OOM Fixes**
- Removed `mode="reduce-overhead"` from `torch.compile()` to prevent CUDA Graph OOM on T4.
- Switched to unified `algsoch/breed-cattle-buffalo` Kaggle dataset.

**2026‑09‑06 – Colab & SOTA Mobile Training Suite**
- Added full Google Colab T4 GPU trainer.
- Implemented SOTA hyperparameters: AdamW, label smoothing, warmup + cosine LR, RandAugment.
- Enabled QAT (Quantization-Aware Training).
- Fixed `AttributeError` for `total_mem` → `total_memory`.

*Happy coding! 🎉*
