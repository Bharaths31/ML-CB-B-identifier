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