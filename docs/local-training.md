# Local Training — Automated Pipeline

The **`local_train.py`** script is the recommended entry point for local training. It automates every step from environment setup to dataset download, model training, and multi-format export in a single command.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier

# 2. Run the pipeline — pick your preferred data mode:

# Fastest local iteration (25% data, ~4× speedup)
python local_train.py --quarter-data

# Good balance of speed and accuracy (50% data)
python local_train.py --half-data

# Full training (all data, highest accuracy)
python local_train.py

# Sanity check — completes in seconds
python local_train.py --smoke-test
```

That's it. The script handles all remaining steps automatically.

---

## What the Script Does

The pipeline runs 8 stages sequentially:

| Stage | Name | Description |
|---|---|---|
| §0 | **Prerequisites** | Checks Python ≥ 3.9, git, pip availability; Windows: checks MSVC build tools |
| §1 | **Environment Setup** | Creates `.venv`, installs training dependencies |
| §2 | **Kaggle Download** | Prompts for API credentials if needed, downloads `algsoch/breed-cattle-buffalo` |
| §3 | **Unzip & Organize** | Extracts images into `data/raw/cattle/` and `data/raw/buffalo/` |
| §4 | **Verify Architecture** | Runs `src.verify` — confirms backbone loading + forward pass shapes |
| §5 | **Data Validation** | Validates dataset structure and breed counts |
| §6 | **Training** | Runs `src.train` with VRAM auto-scaling and appropriate flags |
| §7 | **Export** | Exports in 4 formats: portable, ONNX, INT8, float16 |

Each stage prints a `§N —` banner with ✅/❌ status indicators.

---

## Kaggle API Credentials

The script automatically manages Kaggle credentials.

**First run** — if `~/.kaggle/kaggle.json` doesn't exist, you'll be prompted:
```
┌─────────────────────────────────────────────┐
│  Kaggle API credentials required             │
│  Get your key at: kaggle.com/settings → API │
└─────────────────────────────────────────────┘

Enter your Kaggle username: your_username
Enter your Kaggle API key:  your_api_key_here
```
Credentials are saved to `~/.kaggle/kaggle.json` for future runs.

**Subsequent runs** — existing credentials are reused automatically.

**Manual credential setup:**

*Windows:*
```cmd
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

*Linux / macOS:*
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Getting your API key: Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**. Copy `username` and `key` from the downloaded `kaggle.json`.

---

## Data Modes

The script supports four data modes (mutually exclusive — pick at most one):

| Flag | Data Used | Typical Speed | Best For |
|---|---|---|---|
| *(none)* / `--full-data` | 100% of images | Slowest | Final production training |
| `--half-data` | 50% per breed (seed=42) | ~2× faster | Good local GPU training |
| `--quarter-data` | 25% per breed (seed=42) | ~4× faster | Quick local iteration |
| `--smoke-test` | 5 images/breed, 1 epoch | Seconds | CI / end-to-end sanity check |

**Key properties of all subset modes:**
- Deterministic sampling (`seed=42`) — same images selected every run
- Same **85/10/5 stratified split** applied to the sampled subset
- Class maps include **ALL 75 breeds** — model architecture is identical to full training
- `--smoke-test` uses 60/20/20 split (tiny but real training signal)

### Full Data (Default)
```bash
python local_train.py
# or explicitly:
python local_train.py --full-data
```
Uses every image for every breed. Produces the highest accuracy but takes the longest.

### Half Data
```bash
python local_train.py --half-data
```
Uses 50% of images per breed. A good balance of speed and accuracy for GPUs with limited VRAM (e.g., RTX 3050 4 GB).

### Quarter Data
```bash
python local_train.py --quarter-data
```
Uses 25% of images per breed. ~4× faster than full data. Ideal for quick iteration and validation on very constrained local machines.

### Smoke Test
```bash
python local_train.py --smoke-test
```
Uses 5 images per breed, 1 epoch per phase. Completes in seconds. Use this to verify the entire pipeline works before committing to a long run.

---

## Windows Prerequisites

On Windows, the script automatically checks for **Visual C++ Build Tools** (MSVC), which some Python packages require to compile C extensions.

What it checks:
- `cl.exe` — MSVC compiler in `PATH`
- `vswhere.exe` — Visual Studio Build Tools installer registry

If missing, a warning is printed with the download link — but the script **does not fail**. PyTorch installs via pre-built wheels and does not require MSVC.

```
⚠️  WARNING: Missing Windows build dependencies:
    - cl.exe
    - Visual Studio / Build Tools installer (vswhere not found)

  Fix: Install 'Microsoft C++ Build Tools' (free):
    https://visualstudio.microsoft.com/visual-cpp-build-tools/
  Select workload: 'Desktop development with C++'
```

---

## VRAM Auto-Scaling

The script automatically detects your GPU's available VRAM and adjusts `batch_size` and `grad_accum` to maximize utilization while maintaining a constant effective batch size of 128:

| GPU VRAM | Example Cards | Batch Size | Grad Accum | Effective Batch |
|---|---|---|---|---|
| < 6 GB | RTX 3050 4 GB | 16 | 8 | 128 |
| 6–10 GB | RTX 3070 8 GB | 32 | 4 | 128 |
| 10–16 GB | T4 15 GB | 64 | 2 | 128 |
| 16+ GB | RTX 3090/4090 24 GB | 128 | 1 | 128 |

---

## Complete Flag Reference

```
python local_train.py [OPTIONS]
```

### Data Mode *(mutually exclusive)*

| Flag | Description |
|---|---|
| `--half-data` | Use 50% of images per breed |
| `--quarter-data` | Use 25% of images per breed |
| `--smoke-test` | Use 5 images per breed, 1 epoch per phase |
| `--full-data` | Use all images (explicit form of the default) |

### Model Configuration

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Backbone architecture |
| `--attention` | `cbam` \| `se` | `cbam` | Attention module type |

### Training Overrides

| Flag | Type | Default | Description |
|---|---|---|---|
| `--include-qat` | flag | off | Enable Phase 3 QAT for INT8 Android deployment |
| `--phase1-epochs` | int | `5` | Override Phase 1 epoch count |
| `--phase2-epochs` | int | `40` | Override Phase 2 epoch count |
| `--phase3-epochs` | int | `10` | Override Phase 3 epoch count |
| `--num-workers` | int | `4` | DataLoader worker count |

### Skip Stages

| Flag | Description |
|---|---|
| `--skip-download` | Skip Kaggle download (dataset already in `data/raw/`) |
| `--skip-setup` | Skip venv creation (dependencies already installed) |
| `--skip-verify` | Skip architecture verification step |
| `--skip-export` | Skip multi-format export after training |

---

## Example Commands

```bash
# Full pipeline — fastest local run (quarter data)
python local_train.py --quarter-data

# Full pipeline — balanced speed/accuracy (half data)
python local_train.py --half-data

# Full pipeline — maximum accuracy (all data)
python local_train.py

# Include QAT for Android INT8 deployment
python local_train.py --half-data --include-qat

# Use lite4 backbone (larger, more accurate, more VRAM)
python local_train.py --quarter-data --backbone lite4

# Use SE attention instead of CBAM
python local_train.py --quarter-data --attention se

# Custom epoch overrides
python local_train.py --half-data --phase2-epochs 20

# Re-run training only (data already downloaded, venv ready)
python local_train.py --half-data --skip-download --skip-setup

# Skip export (examine checkpoint manually)
python local_train.py --quarter-data --skip-export

# Verify the full pipeline end-to-end in seconds
python local_train.py --smoke-test

# Full training with QAT, using lite4 backbone
python local_train.py --backbone lite4 --include-qat
```

---

## Output Files

After a successful run, you'll find:

### Checkpoints (`outputs/checkpoints/`)
```
lite2_phase1_best.pt    # Binary head warmup checkpoint
lite2_phase2_best.pt    # Multi-task fine-tune checkpoint (primary model)
lite2_phase3_best.pt    # QAT checkpoint (if --include-qat)
lite2_quantized.pt      # INT8 converted model (if --include-qat)
```

### Exports (`outputs/export/`)
```
lite2_fp32.onnx                       # ONNX (cross-platform)
lite2_int8.pt                         # INT8 quantized TorchScript
lite2_float16.pt                      # FP16 TorchScript
portable/
└── lite2_phase2_best/
    ├── model.pt                      # PyTorch checkpoint (state_dict)
    ├── cattle_classes.json           # {"amritmahal": 0, "ayrshire": 1, ...}
    ├── buffalo_classes.json          # {"alambadi": 0, "banni": 1, ...}
    └── model_info.json               # backbone, image_size, usage, exported_at
```

---

## Idempotent Re-runs

The script is safe to re-run:

- **venv**: Reused if `.venv/` already exists (deps still installed to ensure nothing is missing)
- **Dataset**: If `data/raw/cattle/` and `data/raw/buffalo/` exist with breed subdirectories, download is skipped
- **Kaggle credentials**: If `~/.kaggle/kaggle.json` is valid, no prompt appears
- **Zip cleanup**: Downloaded zip is removed after extraction to save disk space

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Run without `--skip-setup` to install dependencies |
| `RuntimeError: Kaggle credentials are required` | Get your API key at [kaggle.com/settings](https://www.kaggle.com/settings) → API |
| `Dataset not found at data/raw/` | Remove `--skip-download` to auto-download from Kaggle |
| GPU OOM during training | Script auto-scales batch size; if still OOMing, try `--quarter-data` or `--smoke-test` |
| `BackendCompilerFailed: triton` (Windows) | Auto-handled — `torch.compile` is disabled on Windows automatically |
| `backbone .pth not found` | The `.pth` files are included in the repository — ensure your clone is complete |
