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

# Full training + QAT for Android INT8 deployment
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
python -m src.train --smoke-test --skip-qat

# Quarter-data training
python -m src.train --quarter-data --skip-qat

# Half-data training
python -m src.train --half-data --skip-qat

# Full training (all phases including QAT)
python -m src.train --backbone lite2

# Full training, skip QAT
python -m src.train --backbone lite2 --skip-qat

# Custom hyperparameters
python -m src.train \
  --backbone lite4 \
  --batch-size 32 \
  --phase1-epochs 5 \
  --phase2-epochs 30 \
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
  --checkpoint outputs/checkpoints/lite2_phase2_best.pt
```
Outputs per-class metrics to console and confusion matrix PNGs to `outputs/metrics/`.

### 3.7 Model Export
```bash
# Self-contained portable bundle
python -m src.export --mode portable --backbone lite2

# ONNX (cross-platform, framework-free inference)
python -m src.export --mode onnx --backbone lite2

# INT8 quantized TorchScript (smallest, fastest mobile CPU)
python -m src.export --mode int8 --backbone lite2

# FP16 TorchScript (mobile GPU)
python -m src.export --mode float16 --backbone lite2
```

---

## 4. Creating Distribution Packages

```bash
# Lightweight training-only zip
python create_training_zip.py

# Colab-ready project zip
python scripts/create_colab_project_zip.py
```