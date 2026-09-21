# Cattle & Buffalo Breed Classifier

A **lightweight, mobile-deployable image classifier** for 57 Indian cattle breeds and 18 Indian buffalo breeds. Built on EfficientNet-Lite with CBAM/SE attention and a 3-head multi-task training pipeline.

| Field | Value |
|---|---|
| **Breeds** | 57 cattle + 18 buffalo = 75 total |
| **Backbone** | EfficientNet-Lite2 (~6M params) or Lite4 (~13M params) |
| **Input** | 260 × 260 RGB |
| **Training** | 2-phase: head warm-up → multi-task fine-tune (+ EMA, optional teacher distillation; QAT opt-in) |
| **Export** | TFLite INT8, ONNX Runtime INT8, ONNX, FP16, Portable bundle |
| **Dataset** | Kaggle: `algsoch/breed-cattle-buffalo` |

---

## Quick Start

```bash
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier

# Fastest local run (25% data, ~4× speedup)
python local_train.py --quarter-data

# Test the trained model visually
python test_model.py
```

---

## Documentation

| Page | Description |
|---|---|
| [Installation](installation.md) | Step-by-step setup for Windows and Linux/macOS |
| [Local Training (Automated)](local-training.md) | `local_train.py` — full flag reference |
| [Training Pipeline](training-pipeline.md) | 2-phase training, distillation, data modes, CUDA optimizations |
| [Model Tester GUI](model-tester.md) | `test_model.py` — visual testing tool |
| [Architecture](architecture.md) | Model architecture and data flow |
| [Data Pipeline](data-pipeline.md) | Dataset layout, augmentation, splits |
| [Export & Deployment](export-deployment.md) | TFLite INT8, ONNX INT8, Android deployment |
| [Google Colab Training](colab-training.md) | T4 GPU notebook setup |
| [API Reference](api-reference.md) | Complete flag reference for all CLI tools |
| [Configuration](config-reference.md) | `src/config.py` constants |
| [Constraints & Gotchas](constraints-gotchas.md) | Known issues and platform-specific notes |
| [Changelog](changelog.md) | Version history |