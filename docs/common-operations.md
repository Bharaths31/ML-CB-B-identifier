# 12. Training & Execution Operations

This section provides **highly detailed, copyable code snippets** for every phase of the execution pipeline, from environment setup to deployment. These commands are intended to be run from the root of the project (`ML-CB-B-identifier/`).

---

## 1. Environment & Setup

Before executing any ML code, you must prepare the virtual environment and ensure all dependencies are installed.

```bash
# Clone the repository
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier

# Use the automated setup script to create .venv and install dependencies
python setup_venv.py

# Activate the virtual environment (Linux/macOS)
source .venv/bin/activate
# Or on Windows:
# .venv\Scripts\activate
```

---

## 2. Data Preparation Pipeline

The raw images must be processed into stratified CSV splits (Train: 85%, Val: 10%, Test: 5%) before training can begin. Ensure your data is extracted to `data/raw/`.

```bash
# Ensure directories exist
mkdir -p data/raw data/splits

# (Optional) Download Kaggle dataset directly via curl
curl -L -o breed-cattle-buffalo.zip https://www.kaggle.com/api/v1/datasets/download/algsoch/breed-cattle-buffalo
unzip -q breed-cattle-buffalo.zip -d data/raw/

# Run the data pipeline to generate train.csv, val.csv, and test.csv
python -m src.data_pipeline
```
*Expected Output: Logs detailing the exact number of images processed and class imbalances handled, ending with CSVs generated in `data/splits/`.*

---

## 3. Architecture Verification

Always run a sanity check to ensure your system can load the EfficientNet-Lite backbone, establish the three classification heads (Binary, Cattle, Buffalo), and push data through the network without shape mismatches.

```bash
# Run the verification script
python -m src.verify
```
*Expected Output: Confirms feature dimension (1280 for lite2/lite4) and successful forward pass shapes `(batch_size, num_classes)`.*

---

## 4. Training the Model

The training system uses a sophisticated 3-phase pipeline (Binary Warmup -> Multi-task Finetune -> QAT). You can customize the run using heavily parameterized command-line arguments.

### Quick Sanity Check (Smoke Test)
Use this to ensure the entire pipeline (data loading, loss calculation, gradients, saving) works in seconds without waiting hours. It builds a mini-dataset of exactly 5 images per breed.
```bash
python -m src.train --smoke-test --skip-qat
```

### Standard Full Training (Default)
Trains `lite2` using SOTA hyperparameters, Mixed Precision (AMP) on CUDA, and auto-exports a portable bundle at the end.
```bash
python -m src.train --backbone lite2
```

### Advanced Custom Training
A highly parameterized example explicitly setting learning rates, epochs, and regularization.
```bash
python -m src.train \
  --backbone lite4 \
  --batch-size 64 \
  --phase1-epochs 5 \
  --phase2-epochs 40 \
  --phase3-epochs 10 \
  --phase1-lr 3e-3 \
  --phase2-lr 2e-4 \
  --weight-decay 0.01 \
  --label-smoothing 0.1 \
  --device cuda
```

---

## 5. Model Evaluation

Once trained, evaluate the model on the `test.csv` holdout set. This calculates F1 scores, accuracy, and generates confusion matrices.

```bash
# Evaluate the latest checkpoint
python -m src.evaluate --backbone lite2
```
*Expected Output: Per-class accuracy printed to the console, and confusion matrix PNGs generated in `outputs/metrics/`.*

---

## 6. Exporting for Deployment

Export your trained PyTorch `.pt` checkpoints into mobile or edge-friendly formats. The script auto-detects the latest checkpoint in `outputs/checkpoints/`.

```bash
# 1. Export to ONNX (Cross-platform, widely supported)
python -m src.export --mode onnx --backbone lite2

# 2. Export to TorchScript INT8 (Quantized for Android/Edge CPUs)
python -m src.export --mode int8 --backbone lite2

# 3. Export to TorchScript FP16 (Optimized for Mobile GPUs)
python -m src.export --mode float16 --backbone lite2

# 4. Create a Portable Bundle (Includes class maps & metadata for Python inference)
python -m src.export --mode portable --backbone lite2
```

---

## 7. Running the Web Application

Deploy the local dashboard to interact with the model visually, view logs, and trigger new runs.

```bash
# Start the FastAPI server on port 8000
python webapp/server.py
```
*Next Steps: Open `http://localhost:8000` in your browser.*

---

## 8. Creating a Colab/Remote Package

If you need to train on Google Colab or another remote GPU, use the packaging script. It strips out the webapp, memory layer, and Git history, leaving a lightweight zip containing only what is strictly necessary to train.

```bash
# Create training_package.zip
python scripts/create_colab_project_zip.py
```
*Next Steps: Upload `colab_project.zip` to Colab and unzip it.*