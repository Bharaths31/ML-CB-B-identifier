# Common Operations

Concise, copyable command reference for every workflow. All commands run from the **project root** (`ML-CB-B-identifier/`) with your virtual environment active.

---

## 1. Automated Pipeline (Recommended)

The fastest path from zero to a trained model. Handles everything automatically.

```bash
# Full pipeline — quarter data (fastest local run)
python local_train.py --quarter-data

# Full pipeline — half data (good balance)
python local_train.py --half-data

# Full pipeline — all data (maximum accuracy)
python local_train.py

# Verify the entire pipeline works end-to-end in seconds
python local_train.py --smoke-test

# Re-run training only (venv ready, data already downloaded)
python local_train.py --half-data --skip-setup --skip-download

# Opt-in QAT phase (recovery tool — mobile INT8 uses converter PTQ, see Export docs)
python local_train.py --include-qat
```

See [Local Training (Automated)](local-training.md) for the complete flag reference.

---

## 2. Model Testing GUI

Launch the visual model tester after training:

```bash
python test_model.py
```

Opens at `http://localhost:8501`. Drag-and-drop any cattle/buffalo image to see species + top-5 breed predictions.

```bash
# Custom port
python test_model.py --port 9000

# Headless (SSH / server)
python test_model.py --no-browser
```

---

## 3. Manual Step-by-Step Operations

For granular control, debugging, or custom experiments.

### 3.1 Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch>=2.1.0 torchvision>=0.16.0 numpy pandas matplotlib scikit-learn tqdm Pillow requests onnx
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch>=2.1.0 torchvision>=0.16.0 numpy pandas matplotlib scikit-learn tqdm Pillow requests onnx
```

Alternatively, use the helper script:
```bash
python setup_venv.py
```

### 3.2 Dataset Download

**Linux / macOS:**
```bash
mkdir -p data/raw
curl -L -o breed-cattle-buffalo.zip \
  https://www.kaggle.com/api/v1/datasets/download/algsoch/breed-cattle-buffalo
unzip -q breed-cattle-buffalo.zip -d data/raw/
```

**Windows (Command Prompt):**
```cmd
mkdir data\raw
curl -L -o breed-cattle-buffalo.zip https://www.kaggle.com/api/v1/datasets/download/algsoch/breed-cattle-buffalo
tar -xf breed-cattle-buffalo.zip -C data\raw\
```

### 3.3 Data Pipeline (Generate Splits)
```bash
python -m src.data_pipeline
```
Scans `data/raw/` → generates `data/splits/train.csv`, `val.csv`, `test.csv`, `cattle_classes.json`, `buffalo_classes.json`.

### 3.4 Architecture Verification
```bash
python -m src.verify
```
Confirms backbone weight loading and forward pass tensor shapes. Run this before training to catch config issues early.

### 3.5 Training

```bash
# Smoke test — 5 images/breed, 1 epoch per phase
python -m src.train --smoke-test

# Quarter-data training
python -m src.train --quarter-data

# Half-data training
python -m src.train --half-data

# Full training (2 phases: warmup + multi-task fine-tune with EMA)
python -m src.train --backbone lite2

# Teacher run (bigger backbone, used for distillation)
python -m src.train --backbone lite4

# Distill the lite4 teacher into the lite2 student (same size/latency)
python -m src.train --backbone lite2 \
  --teacher outputs/checkpoints/lite4_phase2_best_<runid>.pt

# Opt-in QAT phase (recovery tool; mobile INT8 uses converter PTQ)
python -m src.train --backbone lite2 --include-qat

# Custom hyperparameters
python -m src.train \
  --backbone lite4 \
  --batch-size 32 \
  --phase1-epochs 8 \
  --phase2-epochs 80 \
  --weight-decay 0.01 \
  --device cuda
```

See [API Reference](api-reference.md#python--m-srctrain--core-training-engine) for all `src.train` flags.

### 3.6 Evaluation
```bash
# Evaluate latest lite2 checkpoint on test.csv
python -m src.evaluate --backbone lite2

# Evaluate with specific checkpoint
python -m src.evaluate --backbone lite2 \
  --checkpoint outputs/checkpoints/lite2_phase2_best_<runid>.pt
```
Outputs per-class metrics to console and confusion matrix PNGs to `outputs/metrics/`.

### 3.7 Model Export
```bash
# TFLite INT8 + labels → ready for the Flutter app (flutter_app/assets/models/)
# Requires the optional toolchain: pip install tensorflow onnx2tf tf-keras onnx-graphsurgeon sng4onnx onnxsim
python -m src.export --mode tflite --backbone lite2

# ONNX Runtime Mobile INT8 (QDQ, calibrated on real train images)
python -m src.export --mode onnx-int8 --backbone lite2

# Self-contained portable bundle (desktop testing)
python -m src.export --mode portable --backbone lite2

# ONNX fp32 (cross-platform; caller-normalized input)
python -m src.export --mode onnx --backbone lite2

# FP16 TorchScript (mobile GPU)
python -m src.export --mode float16 --backbone lite2
```

### 3.8 Export Parity Gate
```bash
# Accuracy parity: fp32 PyTorch vs TFLite INT8 vs ONNX INT8 on the val split
python -m src.parity_check --backbone lite2 \
  --tflite outputs/export/lite2_int8.tflite \
  --onnx-int8 outputs/export/lite2_mobile_int8.onnx \
  --split val

# Artifact-only check (no dataset needed): random-input logit comparison
python -m src.parity_check --backbone lite2 --synthetic 16 \
  --onnx-int8 outputs/export/lite2_mobile_int8.onnx
```
INT8 artifacts must stay within 1 pt `combined_top1` of fp32 — otherwise fall back to the FP32 TFLite file. Reports land in `outputs/metrics/`.

---

## 4. Creating Distribution Packages

```bash
# Lightweight training-only zip
python create_training_zip.py

# Colab-ready project zip
python scripts/create_colab_project_zip.py
```