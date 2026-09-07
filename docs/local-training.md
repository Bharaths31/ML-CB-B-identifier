# Local Training (Automated Pipeline)

The **`local_train.py`** script provides a fully automated, end-to-end pipeline for training the Cattle & Buffalo Breed Classifier on your local machine. It handles everything from environment setup to multi-format model export in a single command.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier

# 2. Run the automated pipeline (half-data for faster training)
python local_train.py --half-data
```

That's it. The script will walk you through every step.

---

## What the Script Does

The pipeline runs 8 stages sequentially:

| Stage | Name | Description |
|-------|------|-------------|
| §0 | **Prerequisites** | Checks Python ≥ 3.9, git, pip availability |
| §1 | **Environment Setup** | Creates `.venv`, installs training dependencies (skips webapp deps) |
| §2 | **Kaggle Download** | Prompts for API credentials (if needed), downloads dataset |
| §3 | **Unzip & Organize** | Extracts images into `data/raw/cattle/` and `data/raw/buffalo/` |
| §4 | **Verify Architecture** | Runs `src.verify` to confirm backbone loading + forward pass |
| §5 | **Data Validation** | Validates dataset structure and breed counts |
| §6 | **Training** | Runs `src.train` with appropriate flags and VRAM auto-scaling |
| §7 | **Export** | Exports model in 4 formats: portable, ONNX, INT8, float16 |

Each stage prints clear `§N` banners and ✅/❌ status indicators.

---

## Kaggle API Credentials

The script automatically manages Kaggle credentials:

1. **First run**: If `~/.kaggle/kaggle.json` doesn't exist, the script prompts you interactively:
   ```
     ┌─────────────────────────────────────────────┐
     │  Kaggle API credentials required             │
     │  Get your key at: kaggle.com/settings → API │
     └─────────────────────────────────────────────┘

     Enter your Kaggle username: your_username
     Enter your Kaggle API key:  your_api_key_here
   ```

2. **Subsequent runs**: Existing credentials are reused automatically.

### Getting Your API Key

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section
3. Click **Create New Token** — this downloads `kaggle.json`
4. Copy the `username` and `key` values from that file

---

## Data Modes

The script supports three data modes (mutually exclusive):

### Full Data (Default)
```bash
python local_train.py
```
Uses **all images** for every breed. Produces the highest accuracy but takes the longest.

### Half Data (Recommended for Local)
```bash
python local_train.py --half-data
```
Uses **50% of images per breed** — a good balance of speed and accuracy for local training on GPUs with limited VRAM (e.g., RTX 3050 4GB).

- Deterministic sampling (seed=42) for reproducibility
- Same 85/10/5 stratified split applied to the subset
- Class maps include ALL breeds — model architecture is identical to full training
- Typically ~2× faster than full data

### Smoke Test
```bash
python local_train.py --smoke-test
```
Uses **5 images per breed**, 1 epoch per phase. Completes in seconds. Useful for verifying the entire pipeline works before committing to a long training run.

---

## VRAM Auto-Scaling

The script automatically detects your GPU's available VRAM and adjusts `batch_size` and `grad_accum` to maximize utilization while maintaining a constant effective batch size of 128:

| GPU VRAM | Example Cards | Batch Size | Grad Accum | Effective Batch |
|----------|--------------|-----------|------------|-----------------|
| < 6 GB | RTX 3050 4GB | 16 | 8 | 128 |
| 6–10 GB | RTX 3070 8GB | 32 | 4 | 128 |
| 10–16 GB | T4 15GB | 64 | 2 | 128 |
| 16+ GB | RTX 3090/4090 24GB | 128 | 1 | 128 |

---

## Complete Flag Reference

```
python local_train.py [OPTIONS]
```

### Data Mode (mutually exclusive)

| Flag | Description |
|------|-------------|
| `--half-data` | Use 50% of images per breed (faster training) |
| `--smoke-test` | Tiny dataset (5 imgs/breed), 1 epoch per phase |
| `--full-data` | Use all images (default) |

### Model Configuration

| Flag | Description |
|------|-------------|
| `--backbone {lite2,lite4}` | Backbone architecture (default: `lite2`, ~6M params) |
| `--attention {cbam,se}` | Attention module (default: `cbam`) |

### Training Overrides

| Flag | Description |
|------|-------------|
| `--include-qat` | Enable QAT Phase 3 for INT8 Android deployment (default: skipped) |
| `--phase1-epochs N` | Override Phase 1 epoch count (default: 5) |
| `--phase2-epochs N` | Override Phase 2 epoch count (default: 40) |
| `--phase3-epochs N` | Override Phase 3 epoch count (default: 10) |
| `--num-workers N` | DataLoader worker count (default: 4) |

### Skip Stages

| Flag | Description |
|------|-------------|
| `--skip-download` | Skip Kaggle download (dataset already in `data/raw/`) |
| `--skip-setup` | Skip venv creation (dependencies already installed) |
| `--skip-verify` | Skip architecture verification step |
| `--skip-export` | Skip multi-format export after training |

---

## Example Commands

```bash
# Full training with all images (QAT skipped for speed)
python local_train.py

# Quick training with 50% of images
python local_train.py --half-data

# Smoke test — verify everything works in seconds
python local_train.py --smoke-test

# Full training with QAT for Android deployment
python local_train.py --include-qat

# Re-run training (data already downloaded, venv ready)
python local_train.py --half-data --skip-download --skip-setup

# Use the larger backbone with custom phase 2 epochs
python local_train.py --backbone lite4 --phase2-epochs 30

# Quick half-data training without verification or export
python local_train.py --half-data --skip-verify --skip-export
```

---

## Output Files

After a successful run, you'll find:

### Checkpoints (`outputs/checkpoints/`)
```
lite2_phase1_best.pt    # Binary head warmup checkpoint
lite2_phase2_best.pt    # Multi-task fine-tune checkpoint (main model)
lite2_phase3_best.pt    # QAT checkpoint (if --include-qat)
lite2_quantized.pt      # INT8 converted model (if --include-qat)
```

### Exports (`outputs/export/`)
```
lite2_fp32.onnx                      # ONNX format (cross-platform)
lite2_int8.pt                        # INT8 quantized (Android/Edge)
lite2_float16.pt                     # FP16 TorchScript (mobile GPU)
portable/lite2_phase2_best/          # Self-contained bundle:
  ├── model.pt                       #   Model checkpoint
  ├── cattle_classes.json            #   Cattle breed label map
  ├── buffalo_classes.json           #   Buffalo breed label map
  └── model_info.json                #   Architecture metadata + usage
```

---

## Idempotent Re-runs

The script is designed for safe re-runs:

- **venv**: If `.venv/` already exists, it reuses it (still installs deps to ensure nothing is missing)
- **Dataset**: If `data/raw/cattle/` and `data/raw/buffalo/` exist with breed subdirectories, download is skipped
- **Kaggle creds**: If `~/.kaggle/kaggle.json` is valid, no prompt appears
- **Zip cleanup**: The downloaded zip is removed after extraction to save disk space

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `ModuleNotFoundError: No module named 'torch'` | Run without `--skip-setup` to install dependencies |
| `RuntimeError: Kaggle credentials are required` | Get your API key at [kaggle.com/settings](https://www.kaggle.com/settings) → API |
| `Dataset not found at data/raw/` | Remove `--skip-download` to auto-download from Kaggle |
| GPU OOM during training | Script auto-scales batch size; if still OOMing, try `--half-data` or `--smoke-test` |
| `backbone .pth not found` | Ensure `efficientnet_lite2.pth` and `efficientnet_lite4.pth` are in project root |

---
