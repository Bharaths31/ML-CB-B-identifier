# API Reference

Complete command-line flag reference for all tools in the Cattle & Buffalo Breed Classifier project.

---

## `python local_train.py` — Automated Pipeline

The fully automated 8-stage pipeline: prerequisites → venv → dataset → verify → train → export.

```
python local_train.py [OPTIONS]
```

### Data Mode *(mutually exclusive — pick at most one)*

| Flag | Description |
|---|---|
| `--half-data` | Use 50% of images per breed (deterministic, seed=42) |
| `--quarter-data` | Use 25% of images per breed (deterministic, seed=42) |
| `--smoke-test` | Use 5 images per breed, 1 epoch per phase (seconds) |
| `--full-data` | Explicitly use all images (same as default, makes intent clear) |
| *(none)* | Use all images (default) |

All subset modes maintain the full 57-cattle / 18-buffalo class map. The model architecture is **identical** across all modes.

### Model Configuration

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Backbone architecture (~6M vs ~13M params) |
| `--attention` | `cbam` \| `se` | `cbam` | Attention module type |

### Training Overrides

| Flag | Type | Default | Description |
|---|---|---|---|
| `--include-qat` | flag | off | Enable Phase 3 (Quantization-Aware Training) for INT8 Android deployment |
| `--phase1-epochs` | int | `5` | Override Phase 1 epoch count (binary warmup) |
| `--phase2-epochs` | int | `40` | Override Phase 2 epoch count (multi-task fine-tune) |
| `--phase3-epochs` | int | `10` | Override Phase 3 epoch count (QAT) |
| `--num-workers` | int | `4` | DataLoader worker process count |

### Skip Stages

| Flag | Description |
|---|---|
| `--skip-download` | Skip Kaggle dataset download — data already in `data/raw/` |
| `--skip-setup` | Skip `.venv` creation and dependency install |
| `--skip-verify` | Skip architecture verification (`src.verify`) |
| `--skip-export` | Skip multi-format model export after training |

### Examples

```bash
# Fastest local run
python local_train.py --quarter-data

# Half data, include QAT for Android
python local_train.py --half-data --include-qat

# Full data, lite4 backbone, custom phase 2 epochs
python local_train.py --backbone lite4 --phase2-epochs 30

# Re-run training only (skip setup + download)
python local_train.py --half-data --skip-setup --skip-download

# Smoke test — verifies the full pipeline in seconds
python local_train.py --smoke-test
```

---

## `python -m src.train` — Core Training Engine

Runs the 3-phase training loop directly. `local_train.py` wraps this with automated setup and export stages.

```
python -m src.train [OPTIONS]
```

> Run from the **project root** directory (not inside `src/`).

### Model Configuration

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Backbone architecture |
| `--weights` | path | project root `.pth` | Custom pretrained `.pth` weights path |
| `--attention` | `cbam` \| `se` | `cbam` | Attention module type |

### Data & I/O

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data` | path | `data/raw/` | Raw image root directory |
| `--split-dir` | path | `data/splits/` | CSV split output directory |
| `--export-dir` | path | `outputs/export/portable/` | Portable export destination directory |

### Device & Workers

| Flag | Type | Default | Description |
|---|---|---|---|
| `--batch-size` | int | `64` | Per-GPU batch size (auto-scaled by `local_train.py`) |
| `--num-workers` | int | `4` | DataLoader worker processes |
| `--device` | str | auto | Force device: `cuda` or `cpu` (auto-detects by default) |
| `--seed` | int | `42` | Global random seed |

### Hyperparameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--weight-decay` | float | `1e-2` | AdamW weight decay |
| `--label-smoothing` | float | `0.1` | Label smoothing factor (prevents overconfident predictions) |
| `--warmup-epochs` | int | `3` | Linear LR warmup epochs (Phase 2 only) |
| `--grad-accum` | int | `2` | Gradient accumulation steps (effective_batch = batch_size × grad_accum) |

### Phase Control

| Flag | Type | Default | Description |
|---|---|---|---|
| `--phase1-epochs` | int | `5` | Phase 1 epochs (binary head warmup) |
| `--phase2-epochs` | int | `40` | Phase 2 epochs (multi-task fine-tune) |
| `--phase3-epochs` | int | `10` | Phase 3 epochs (QAT) |
| `--skip-qat` | flag | off | Skip Phase 3 entirely |

### Data Mode *(mutually exclusive)*

| Flag | Description |
|---|---|
| `--smoke-test` | 5 images/breed, 1 epoch per phase (real data, not random batches) |
| `--half-data` | 50% of images per breed, same stratified 85/10/5 split |
| `--quarter-data` | 25% of images per breed, same stratified 85/10/5 split |
| *(none)* | Full dataset (default) |

### Augmentation & Compilation

| Flag | Description |
|---|---|
| `--no-mix` | Disable CutMix/MixUp batch mixing |
| `--no-compile` | Disable `torch.compile` (auto-set on Windows; use to debug on Linux too) |
| `--no-export` | Skip automatic portable export after training completes |

### Examples

```bash
# Sanity check — full pipeline in seconds
python -m src.train --smoke-test --skip-qat

# Quarter-data training (recommended for local GPU)
python -m src.train --quarter-data --skip-qat

# Half-data with overridden epochs
python -m src.train --half-data --phase2-epochs 20 --skip-qat

# Full training with lite4 backbone
python -m src.train --backbone lite4

# Full training with all QAT phases
python -m src.train --backbone lite2

# Override hyperparameters explicitly
python -m src.train \
  --backbone lite4 \
  --batch-size 32 \
  --phase1-epochs 5 \
  --phase2-epochs 40 \
  --weight-decay 0.01 \
  --label-smoothing 0.1 \
  --device cuda

# CPU-only run (no torch.compile)
python -m src.train --quarter-data --device cpu --no-compile
```

---

## `python -m src.export` — Model Export

Exports a trained checkpoint to one of four formats.

```
python -m src.export [OPTIONS]
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Which backbone's checkpoint to load |
| `--mode` | `onnx` \| `int8` \| `float16` \| `portable` | **required** | Export format |
| `--checkpoint` | path | auto (latest in `outputs/checkpoints/`) | Specific `.pt` file to export |

### Export Modes

| Mode | Output File | Description |
|---|---|---|
| `portable` | `outputs/export/portable/<backbone>_<tag>/` | Self-contained folder: `model.pt` + class JSONs + metadata. Requires `BreedClassifier` class to load. |
| `onnx` | `outputs/export/<backbone>_fp32.onnx` | ONNX opset 13, framework-free inference |
| `int8` | `outputs/export/<backbone>_int8.pt` | PTQ INT8 TorchScript, 32-batch calibration |
| `float16` | `outputs/export/<backbone>_float16.pt` | FP16 TorchScript |

### Examples

```bash
# Export all 4 formats for lite2 (what local_train.py does automatically)
python -m src.export --mode portable --backbone lite2
python -m src.export --mode onnx --backbone lite2
python -m src.export --mode int8 --backbone lite2
python -m src.export --mode float16 --backbone lite2

# Export specific checkpoint file
python -m src.export --mode portable --backbone lite2 \
  --checkpoint outputs/checkpoints/lite2_phase2_best.pt
```

---

## `python -m src.evaluate` — Model Evaluation

Evaluates a trained checkpoint on the `test.csv` holdout set.

```
python -m src.evaluate [OPTIONS]
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Backbone architecture to evaluate |
| `--checkpoint` | path | auto (latest phase2 checkpoint) | Specific `.pt` file to evaluate |
| `--device` | str | auto | Force device: `cuda` or `cpu` |

### Output

- Per-class accuracy and F1 scores printed to console
- Confusion matrix PNGs saved to `outputs/metrics/`
- Summary JSON saved to `outputs/metrics/eval_<backbone>.json`

### Examples

```bash
# Evaluate latest lite2 checkpoint
python -m src.evaluate --backbone lite2

# Evaluate specific checkpoint
python -m src.evaluate --backbone lite2 \
  --checkpoint outputs/checkpoints/lite2_phase2_best.pt

# Force CPU evaluation
python -m src.evaluate --backbone lite2 --device cpu
```

---

## `python test_model.py` — Model Tester GUI

Standalone web GUI for visually testing exported models on individual images.

```
python test_model.py [OPTIONS]
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--port` | int | `8501` | HTTP port to serve the GUI on |
| `--no-browser` | flag | off | Don't automatically open the browser |

### Examples

```bash
# Launch with default settings (opens browser at http://localhost:8501)
python test_model.py

# Use a different port
python test_model.py --port 9000

# Headless server (no browser pop-up — useful for SSH sessions)
python test_model.py --no-browser
```

### Requirements

- At least one trained checkpoint in `outputs/checkpoints/` or `outputs/export/portable/`
- Class map files in `data/splits/` (`cattle_classes.json`, `buffalo_classes.json`) — generated automatically during training

---

## `python -m src.verify` — Architecture Verification

Quick sanity check: loads backbone weights, constructs the model, runs a forward pass, and validates output shapes.

```
python -m src.verify
```

No flags. Expected output:
```
[verify] backbone: lite2
[verify] feature dim: 1280 ✓
[verify] binary head output: (1, 2) ✓
[verify] cattle head output: (1, 57) ✓
[verify] buffalo head output: (1, 18) ✓
[verify] All checks passed.
```