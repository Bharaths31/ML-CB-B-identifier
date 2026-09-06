# 2. Architecture & Data Flow

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

# 3. Directory Map

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