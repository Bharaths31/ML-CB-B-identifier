# 12. Training & Execution Operations

This section provides **copyable code snippets** for every phase of the execution pipeline. Modern workflows should use the **Automated Pipeline (`local_train.py`)**, while manual step-by-step commands are available for advanced users and custom debugging.

---

## 1. Automated Setup & Training Pipeline (Recommended)

The **`local_train.py`** script automates everything end-to-end: Python version checks, `.venv` creation, dependency installation, Kaggle dataset download & extraction, architecture verification, multi-phase model training, and multi-format export.

```bash
# Clone the repository
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier

# -------------------------------------------------------------
# Select your desired training mode:
# -------------------------------------------------------------

# Full training (all images, maximum accuracy)
python local_train.py

# Half-data training (50% images/breed, ~2x speedup)
python local_train.py --half-data

# Quarter-data training (25% images/breed, ~4x speedup, ideal for local testing)
python local_train.py --quarter-data

# Smoke test (5 images/breed, 1 epoch per phase, runs in seconds)
python local_train.py --smoke-test

# Include QAT (Quantization Aware Training) for INT8 mobile deployment
python local_train.py --half-data --include-qat
```

---

## 2. Interactive Model Testing GUI

Test exported PyTorch or ONNX models on individual images using the standalone web GUI.

```bash
# Launch the interactive GUI (opens at http://localhost:8501)
python test_model.py
```
*Features: Drag-and-drop image upload, checkpoint dropdown selector, top-5 breed prediction bar chart, species badge indicator.*

---

## 3. Advanced / Manual Step-by-Step Operations

For granular control, custom hyperparameter experiments, or modular execution, you can run each step manually.

### 3.1 Virtual Environment Setup
```bash
# Create .venv and install dependencies
python setup_venv.py

# Activate the virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 3.2 Data Preparation & CSV Splits
```bash
# Download dataset manually (if not using local_train.py)
curl -L -o breed-cattle-buffalo.zip https://www.kaggle.com/api/v1/datasets/download/algsoch/breed-cattle-buffalo
unzip -q breed-cattle-buffalo.zip -d data/raw/

# Process raw images into stratified train/val/test CSV splits
python -m src.data_pipeline
```

### 3.3 Architecture Verification
```bash
# Verify backbone weight loading and forward pass tensor shapes
python -m src.verify
```

### 3.4 Manual Model Training
```bash
# Quick sanity test (5 images per breed)
python -m src.train --smoke-test --skip-qat

# Train with 50% data subset
python -m src.train --half-data --skip-qat

# Train with 25% data subset
python -m src.train --quarter-data --skip-qat

# Full custom training run
python -m src.train \
  --backbone lite2 \
  --batch-size 64 \
  --phase1-epochs 5 \
  --phase2-epochs 40 \
  --phase1-lr 3e-3 \
  --phase2-lr 2e-4 \
  --device cuda
```

### 3.5 Model Evaluation
```bash
# Evaluate checkpoint on test.csv holdout set
python -m src.evaluate --backbone lite2
```
*Outputs per-class metrics and confusion matrix PNGs to `outputs/metrics/`.*

### 3.6 Multi-Format Model Export
```bash
# Export to ONNX (Cross-platform)
python -m src.export --mode onnx --backbone lite2

# Export to INT8 TorchScript (Quantized for mobile/edge)
python -m src.export --mode int8 --backbone lite2

# Export to FP16 TorchScript (Mobile GPUs)
python -m src.export --mode float16 --backbone lite2

# Create self-contained Portable Bundle
python -m src.export --mode portable --backbone lite2
```

---

## 4. Running the Web Application

Deploy the local FastAPI dashboard to interact with the model visually, monitor progress live, and test predictions.

```bash
# Start FastAPI backend (serves at http://localhost:8000)
python webapp/server.py
```

---

## 5. Creating Remote & Colab Packages

Generate lightweight zip archives stripped of unnecessary files for cloud or Colab execution.

```bash
# Create colab_project.zip (for Google Colab)
python scripts/create_colab_project_zip.py

# Create standalone training package (without webapp/ and memory/)
python create_training_zip.py
```