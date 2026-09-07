# ML-CB-B-identifier
 # Cattle & Buffalo Breed Classifier

## Table of Contents
- [Project Overview](#1-project-overview)
- [Architecture & Data Flow](architecture.md)
- [Directory Structure](architecture.md#3-directory-map)
- [Model Architecture](model-architecture.md)
- [Data Preparation](data-pipeline.md)
- [Local Training (Automated)](local-training.md)
- [Google Colab Training Setup](colab-training.md)
- [Training Pipeline](training-pipeline.md)
- [Exporting the Model & Android Deployment](export-deployment.md)
- [Running the FastAPI Webapp](webapp.md)
- [Memory Layer (Mem0)](memory-layer.md)
- [Configuration Reference](config-reference.md)
- [API Reference](api-reference.md)
- [Common Operations](common-operations.md)
- [Known Constraints & Gotchas](constraints-gotchas.md)
- [Changelog](changelog.md)

---

# 1. Project Overview

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