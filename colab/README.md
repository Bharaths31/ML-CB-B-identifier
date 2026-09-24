# Colab Training — Cattle & Buffalo Breed Classifier

This directory contains everything needed to train the model on Google Colab's free T4 GPU.

## Quick Start

1. Open `cattle_buffalo_trainer.py` as a notebook in Colab (or use the `.ipynb` version)
2. Follow the cells in order — they handle everything from setup to export

## What Files Do You Need?

### Option A: Clone from GitHub (Easiest)
The notebook can clone directly from the public repo. No file uploads needed — just run the GitHub clone cell.

### Option B: Upload `colab_project.zip`
Create it locally with:
```bash
python scripts/create_colab_project_zip.py
```

**Contents of `colab_project.zip`:**
```
src/__init__.py
src/config.py
src/run_utils.py
src/run_logger.py
src/data_pipeline.py
src/model.py
src/cbam.py
src/efficientnet_lite.py
src/traits.py
src/train.py
src/metrics.py
src/evaluate.py
src/export.py
src/parity_check.py
src/verify.py
requirements.txt
efficientnet_lite2.pth         (~24 MB)
efficientnet_lite4.pth         (~50 MB, optional)
```

### Option C: Google Drive
Upload `colab_project.zip` to `My Drive/ML-CB-B-identifier/colab_project.zip`.
The notebook will mount your Drive and copy it automatically.

## Dataset Options

| Method | How |
|--------|-----|
| **Kaggle API** | Set Kaggle credentials, auto-downloads and extracts datasets (`algsoch`, `atharvadarpude`, or `both`) into `data/raw/` |
| **Upload archive.zip** | Create locally with `python scripts/create_colab_archive.py`, then upload to Colab |
| **Google Drive** | Upload `archive.zip` to Drive, notebook copies it |

## After Training

The notebook exports:
- **Portable bundle** — `model.pt` + class maps + metadata (for inference anywhere)
- **FP32 ONNX** — for desktop evaluation and GUI tester
- **Mobile INT8 ONNX & TFLite** — converter-side PTQ (QDQ/INT8) optimized for mobile deployment
- **Parity verification** — validates exported artifacts against PyTorch reference

You can download the complete results archive directly or save it to Google Drive.

## T4 GPU Settings

The notebook is pre-configured for Colab's free T4 (15 GB VRAM):
- Dynamic VRAM auto-scaling (16 to 128 batch size, maintains 128 effective batch)
- Workers: 2 (Colab CPU limit)
- Mixed precision (AMP) for phases 1-2
- AMP disabled for optional QAT phase 3
- Dataset on local SSD (`/content/data/raw/`), not Drive
