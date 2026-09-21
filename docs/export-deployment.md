# Export & Deployment

The classifier exports to **desktop testing formats** and **mobile deployment formats** for two Android runtimes (TFLite and ONNX Runtime Mobile). `local_train.py` exports automatically after training; manual export is fully supported.

---

## Export Command

```
python -m src.export [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Which backbone's checkpoint to export |
| `--mode` | `onnx` \| `onnx-int8` \| `tflite` \| `float16` \| `portable` | **required** | Export format |
| `--checkpoint` | path | auto (`outputs/checkpoints/<backbone>_phase2_best.pt`) | Specific `.pt` file to use |
| `--calibration-images` | int | `500` | Train-split images used for INT8 calibration |
| `--skip-app-assets` | flag | off | Don't copy TFLite artifacts into `flutter_app/assets/models/` |
| `--onnx-static-batch` | flag | off | Fix batch size 1 in the fp32 ONNX export |

---

## Export Formats

| Mode | Output | Use Case |
|---|---|---|
| `portable` | `outputs/export/portable/<backbone>_<tag>/` | Python inference with full metadata (test_model.py) |
| `onnx` | `outputs/export/<backbone>_fp32.onnx` | Framework-free desktop testing; caller normalizes input |
| `tflite` | `<backbone>_fp32.tflite` + `<backbone>_int8.tflite` + `labels_*.txt` | **Flutter app (tflite_flutter)** — full-integer INT8 PTQ |
| `onnx-int8` | `<backbone>_mobile_int8.onnx` + `labels_*.txt` | **ONNX Runtime Mobile** — QDQ static INT8 |
| `float16` | `outputs/export/<backbone>_float16.pt` | FP16 TorchScript |

### Input conventions (important!)

| Artifact | Input the caller must provide |
|---|---|
| `onnx` (fp32) | ImageNet-**normalized** tensor (mean/std applied by the caller) — matches `test_model.py` |
| `tflite` / `onnx-int8` (mobile) | RGB float32 in **[0, 1]** — ImageNet normalization is **baked into the graph** as constants |

The mobile convention means the Flutter engine's existing `pixel / 255.0` preprocessing is exactly correct with **zero app changes**.

### Portable Bundle

The portable format is a self-contained folder for Python-based inference:

```
outputs/export/portable/<backbone>_phase2_best/
├── model.pt             # torch.load() checkpoint containing state_dict
├── cattle_classes.json  # {"amritmahal": 0, "ayrshire": 1, ...}
├── buffalo_classes.json # {"alambadi": 0, "banni": 1, ...}
└── model_info.json      # backbone, image_size, exported_at, usage
```

> **Note:** Portable bundles save `state_dict` (not TorchScript). Loading requires the `BreedClassifier` class from `src/model.py`. For framework-free deployment use ONNX / TFLite instead.

### TFLite (full-integer INT8)

`--mode tflite` runs the chain **PyTorch → ONNX (static batch 1) → onnx2tf → TFLite FP32 → full-integer INT8 PTQ**:
- Calibrated on up to 500 real train-split images (`--calibration-images`)
- Per-channel weight quantization, INT8 activations; quantize/dequantize ops at the I/O boundary keep inputs as float32 [0,1]
- Emits `labels_binary.txt` / `labels_cattle.txt` / `labels_buffalo.txt` (line *i* = class *i*)
- Copies `model.tflite` + labels into `flutter_app/assets/models/` automatically (disable with `--skip-app-assets`)
- A TFLite **FP32** fallback is emitted alongside for A/B checks

Requires the optional toolchain (see `requirements.txt`):
```bash
pip install tensorflow onnx2tf tf-keras onnx-graphsurgeon sng4onnx onnxsim
```

### ONNX INT8 (ONNX Runtime Mobile)

`--mode onnx-int8` produces a **QDQ static-quantized** ONNX model: per-channel weights, INT8 activations, calibrated on the same real train images. Runs with `onnxruntime` (already a dependency) — no extra toolchain needed.

### Legacy `--mode int8` (removed)

The old x86 PTQ path was **removed**: it failed conversion (`Unsupported qscheme: per_channel_affine`) and its artifacts were not usable on Android. Running `--mode int8` prints this guidance.

---

## Manual Export Commands

```bash
# Mobile: TFLite INT8 + labels, ready for the Flutter app
python -m src.export --mode tflite --backbone lite2

# Mobile: ONNX Runtime Mobile INT8
python -m src.export --mode onnx-int8 --backbone lite2

# Desktop testing formats
python -m src.export --mode portable  --backbone lite2
python -m src.export --mode onnx      --backbone lite2
python -m src.export --mode float16   --backbone lite2

# Export from a specific checkpoint
python -m src.export --mode tflite --backbone lite2 \
  --checkpoint outputs/checkpoints/lite2_phase2_best.pt
```

---

## Parity Gate (always run after exporting)

`src.parity_check` verifies that the quantized artifacts match the fp32 PyTorch reference before you ship:

```bash
# Accuracy mode — same metrics as training validation, per runtime
python -m src.parity_check --backbone lite2 \
  --tflite outputs/export/lite2_int8.tflite \
  --onnx-int8 outputs/export/lite2_mobile_int8.onnx \
  --onnx outputs/export/lite2_fp32.onnx \
  --split val

# Artifact-only mode — no dataset needed, random-input logit comparison
python -m src.parity_check --backbone lite2 --synthetic 16 \
  --onnx-int8 outputs/export/lite2_mobile_int8.onnx
```

**Acceptance:** any INT8 artifact must stay within **1 pt** `combined_top1` of the fp32 reference. If it fails, fall back to the FP32 TFLite file or revisit calibration. Reports are saved to `outputs/metrics/<backbone>_parity*.json`.

---

## Model Sizes (measured, lite2)

| Artifact | Size |
|---|---|
| Portable `model.pt` (FP32 state_dict) | ~26 MB |
| ONNX FP32 | ~25.7 MB |
| ONNX INT8 (QDQ, mobile) | **~7.1 MB** |
| TFLite INT8 | ~6–7 MB (expected) |

---

## Android Deployment

### Mobile INT8 pipeline (recommended)

The Android path is **converter-side PTQ**, not PyTorch QAT:

```bash
# 1. Train (2 phases by default; QAT is opt-in and not needed for mobile INT8)
python -m src.train --backbone lite2 --teacher outputs/checkpoints/lite4_phase2_best.pt

# 2. Export both runtimes
python -m src.export --mode tflite --backbone lite2
python -m src.export --mode onnx-int8 --backbone lite2

# 3. Parity gate
python -m src.parity_check --backbone lite2 \
  --tflite outputs/export/lite2_int8.tflite \
  --onnx-int8 outputs/export/lite2_mobile_int8.onnx
```

Step 2 places the artifacts the Flutter app expects (`tflite_flutter` loads `assets/models/model.tflite` + `labels_*.txt`), with the input convention the app already uses (`pixel / 255.0`, outputs read by index: 0=binary, 1=cattle, 2=buffalo).

### Deployment Options

| Format | File | Runtime |
|---|---|---|
| TFLite INT8 | `<backbone>_int8.tflite` → `assets/models/model.tflite` | `tflite_flutter` (CPU XNNPACK / GPU delegate) |
| ONNX INT8 | `<backbone>_mobile_int8.onnx` | ONNX Runtime Mobile (`onnxruntime-android`) |
| TFLite FP32 | `<backbone>_fp32.tflite` | Fallback if INT8 parity fails |

### QAT (optional recovery path)

`python -m src.train --include-qat` (OFF by default) fine-tunes with fake-quantization and saves `<backbone>_quantized.pt`. This is an **x86-side recovery tool** if converter PTQ drops more than ~2 pts — the artifact is not a TFLite/ORT model. Quantized/QAT checkpoints also cannot be re-exported through `src.export` (fused module names); always export from the phase-2 EMA checkpoint.