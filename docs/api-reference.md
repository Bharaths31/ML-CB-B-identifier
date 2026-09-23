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
| `--include-qat` | flag | off | Enable the optional Phase 3 (QAT) — a recovery tool; mobile INT8 comes from converter PTQ ([Android Deployment](android-deployment.md)) |
| `--phase1-epochs` | int | `8` | Override Phase 1 epoch count (all-heads warmup) |
| `--phase2-epochs` | int | `80` | Override Phase 2 epoch count (multi-task fine-tune) |
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

# Full data, lite4 backbone (teacher run for distillation)
python local_train.py --backbone lite4

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
| `--phase1-epochs` | int | `8` | Phase 1 epochs (all-heads warmup) |
| `--phase2-epochs` | int | `80` | Phase 2 epochs (multi-task fine-tune + EMA) |
| `--phase3-epochs` | int | `10` | Phase 3 epochs (QAT, opt-in) |
| `--include-qat` | flag | off | Enable Phase 3 (QAT). Mobile INT8 comes from converter PTQ instead — see [Export & Deployment](export-deployment.md) |
| `--skip-qat` | flag | off | Kept for backwards compatibility; QAT is already off by default |

### Distillation

| Flag | Type | Default | Description |
|---|---|---|---|
| `--teacher` | path | off | Teacher checkpoint (`.pt`) for knowledge distillation, e.g. `outputs/checkpoints/lite4_phase2_best_<runid>.pt` |
| `--teacher-backbone` | `lite2` \| `lite4` | `lite4` | Teacher backbone architecture |
| `--teacher-attention` | `cbam` \| `se` | same as student | Teacher attention type |

Blend and temperature are configured in `src/config.py` (`KD_ALPHA=0.7`, `KD_TEMPERATURE=4.0`).

### Data Mode *(mutually exclusive)*

| Flag | Description |
|---|---|
| `--smoke-test` | 5 images/breed, 1 epoch per phase (real data, not random batches) |
| `--half-data` | 50% of images per breed, same 70/15/15 stratified split |
| `--quarter-data` | 25% of images per breed, same 70/15/15 stratified split |
| *(none)* | Full dataset (default) |

#### Augmentation (ALL OFF by default — opt-in per run)

| Flag | Description |
|---|---|
| `--mix` | Enable CutMix/MixUp (same-species pairing, α=0.4/0.2, p=0.25) |
| `--flip` | Enable RandomHorizontalFlip |
| `--color-jitter` | Enable ColorJitter |
| `--randaugment` | Enable RandAugment(ops=2, mag=5) |
| `--rrc` | Enable RandomResizedCrop(260, scale=0.8–1.0) |
| `--augment-all` | Enable flip + color-jitter + randaugment + rrc + mix |
| `--no-mix` | Force-disable mixing (default) |

#### Imbalance & Output

| Flag | Type | Default | Description |
|---|---|---|---|
| `--logit-adjust` | flag | off | Enable logit adjustment **on top of** the sampler (off by default; double-corrects) |
| `--logit-adjust-prior` | `sampled` \| `raw` | `sampled` | Prior source when logit adjustment is enabled |
| `--rare-threshold` | int | `30` | Breeds below this many train images are excluded from CutMix/MixUp |
| `--contrastive-weight` | float | `0.2` | SupCon weight (0 disables) |
| `--run-tag` | str | auto `DD-MM-YYYY-HH-MM` | Run id for timestamped outputs |

#### Compilation

| Flag | Description |
|---|---|
| `--no-compile` | Disable `torch.compile` (auto-set on Windows; use to debug on Linux too) |
| `--no-export` | Skip automatic portable export after training completes |

### Examples

```bash
# Sanity check — full pipeline in seconds
python -m src.train --smoke-test

# Quarter-data training (recommended for local GPU)
python -m src.train --quarter-data

# Half-data with overridden epochs
python -m src.train --half-data --phase2-epochs 20

# Full training with lite4 backbone (teacher run)
python -m src.train --backbone lite4

# Full training, distilled from the lite4 teacher
python -m src.train --backbone lite2 \
  --teacher outputs/checkpoints/lite4_phase2_best_<runid>.pt

# Opt-in QAT phase (recovery path; mobile INT8 uses converter PTQ)
python -m src.train --backbone lite2 --include-qat

# Override hyperparameters explicitly
python -m src.train \
  --backbone lite4 \
  --batch-size 32 \
  --phase1-epochs 8 \
  --phase2-epochs 80 \
  --weight-decay 0.01 \
  --label-smoothing 0.05 \
  --device cuda

# CPU-only run (no torch.compile)
python -m src.train --quarter-data --device cpu --no-compile
```

---

## `python -m src.export` — Model Export

Exports a trained checkpoint to desktop or mobile formats.

```
python -m src.export [OPTIONS]
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Which backbone's checkpoint to load |
| `--mode` | `onnx` \| `onnx-int8` \| `tflite` \| `float16` \| `portable` | **required** | Export format |
| `--checkpoint` | path | auto (`<backbone>_phase2_best_<runid>.pt`) | Specific `.pt` file to export |
| `--calibration-images` | int | `500` | Train images used for INT8 calibration (tflite / onnx-int8) |
| `--skip-app-assets` | flag | off | Don't copy TFLite artifacts into `flutter_app/assets/models/` |
| `--onnx-static-batch` | flag | off | Fix batch size 1 in the fp32 ONNX export |

### Export Modes

| Mode | Output File | Description |
|---|---|---|
| `portable` | `outputs/export/portable/<backbone>_<tag>/` | Self-contained folder: `model.pt` + class JSONs + metadata. Requires `BreedClassifier` class to load. |
| `onnx` | `outputs/export/<backbone>_fp32.onnx` | ONNX opset 13; caller-normalized input (test_model.py compatible) |
| `tflite` | `<backbone>_fp32.tflite` + `<backbone>_int8.tflite` + `labels_*.txt` | Full-integer INT8 PTQ via onnx2tf; input [0,1], normalization baked in; auto-copies into `flutter_app/assets/models/` |
| `onnx-int8` | `outputs/export/<backbone>_mobile_int8.onnx` | QDQ static INT8 for ONNX Runtime Mobile; per-channel weights, calibrated on real train images |
| `float16` | `outputs/export/<backbone>_float16.pt` | FP16 TorchScript |

> `--mode int8` (x86 PTQ) was removed — it failed conversion and its artifacts were unusable on Android. Running it prints guidance to use `tflite` / `onnx-int8`.

### Examples

```bash
# Mobile: TFLite INT8 + labels → flutter_app/assets/models/
python -m src.export --mode tflite --backbone lite2

# Mobile: ONNX Runtime Mobile INT8
python -m src.export --mode onnx-int8 --backbone lite2

# Desktop testing formats
python -m src.export --mode portable --backbone lite2
python -m src.export --mode onnx --backbone lite2
python -m src.export --mode float16 --backbone lite2

# Export specific checkpoint file
python -m src.export --mode tflite --backbone lite2 \
  --checkpoint outputs/checkpoints/lite2_phase2_best_<runid>.pt
```

---

## `python -m src.parity_check` — Export Parity Gate

Compares the fp32 PyTorch reference against the mobile artifacts on the same data — the accuracy gate every export must pass.

```
python -m src.parity_check [OPTIONS]
```

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | path | auto (`<backbone>_phase2_best_<runid>.pt`) | fp32 reference checkpoint |
| `--backbone` | `lite2` \| `lite4` | `lite2` | Reference backbone |
| `--tflite` | path | — | TFLite artifact (fp32 or INT8) |
| `--onnx-int8` | path | — | Quantized ONNX (mobile convention: input [0,1]) |
| `--onnx` | path | — | fp32 ONNX (caller-normalized convention) |
| `--split` | `val` \| `test` | `val` | Split to evaluate |
| `--limit` | int | `0` (all) | Max images per runtime |
| `--synthetic` | int | `0` | Skip the dataset; compare N random inputs (max \|Δlogit\|) |
| `--tolerance` | float | `0.01` | Allowed combined_top1 drop vs fp32 |
| `--out` | path | `outputs/metrics/<backbone>_parity*.json` | JSON report path |

### Output

- Accuracy table: `binary_acc`, `cattle_acc`, `buffalo_acc`, `combined_top1`, `combined_top3` per runtime
- Deltas vs `pytorch_fp32` and a **PASS / FAIL** verdict against the tolerance
- Synthetic mode: per-head max \|Δlogit\| (guidance < 0.05)

### Examples

```bash
# Full accuracy parity on the validation split
python -m src.parity_check --backbone lite2 \
  --tflite outputs/export/lite2_int8.tflite \
  --onnx-int8 outputs/export/lite2_mobile_int8.onnx \
  --split val

# Quick artifact-only check without any dataset
python -m src.parity_check --backbone lite2 --synthetic 16 \
  --onnx-int8 outputs/export/lite2_mobile_int8.onnx
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
  --checkpoint outputs/checkpoints/lite2_phase2_best_<runid>.pt

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
| `--dev` | flag | on | Developer mode: view full metadata, system logs, edit presenter settings |
| `--present` | flag | off | Presenter mode: clean UI, hides technical details and export options |

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