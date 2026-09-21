# 15. Android Deployment

The Android pipeline targets **two runtimes** — TFLite (what the Flutter app uses today via `tflite_flutter`) and ONNX Runtime Mobile. Mobile INT8 comes from **converter-side PTQ**, not PyTorch QAT.

---

## Pipeline Overview

```
data/raw → src.train (2 phases, EMA, optional --teacher distillation)
        → outputs/checkpoints/<backbone>_phase2_best.pt   (EMA weights)
        → src.export --mode tflite | onnx-int8            (INT8 PTQ)
        → flutter_app/assets/models/{model.tflite, labels_*.txt}
        → src.parity_check                                (accuracy gate)
```

## Step 1 — Train

Default training is 2 phases (head warmup → multi-task fine-tune with EMA). For maximum accuracy, distill from a lite4 teacher:

```bash
# Teacher (once): lite4 + CBAM
python -m src.train --backbone lite4

# Student: same lite2 size/latency, teacher-quality logits
python -m src.train --backbone lite2 \
  --teacher outputs/checkpoints/lite4_phase2_best.pt
```

## Step 2 — Export

```bash
# TFLite INT8 — full-integer PTQ, calibrated on ~500 real train images.
# Copies model.tflite + labels into flutter_app/assets/models/ automatically.
python -m src.export --mode tflite --backbone lite2

# ONNX Runtime Mobile INT8 — QDQ static quantization, per-channel weights.
python -m src.export --mode onnx-int8 --backbone lite2
```

TFLite toolchain (only needed for `--mode tflite`):

```bash
pip install tensorflow onnx2tf tf-keras onnx-graphsurgeon sng4onnx onnxsim
```

## Step 3 — Parity Gate

```bash
python -m src.parity_check --backbone lite2 \
  --tflite outputs/export/lite2_int8.tflite \
  --onnx-int8 outputs/export/lite2_mobile_int8.onnx \
  --split val
```

**Acceptance:** INT8 within 1 pt `combined_top1` of fp32. If it fails, ship the FP32 TFLite fallback (`<backbone>_fp32.tflite`, emitted alongside) or revisit calibration.

## On-device Input Convention

Mobile artifacts have **ImageNet normalization baked into the graph**:

| | Value |
|---|---|
| Input | RGB float32 in **[0, 1]**, shape `[1, 3, 260, 260]` (static batch 1) |
| Preprocessing on device | `pixel / 255.0` — exactly what the Flutter engine already does |
| Outputs | `binary` `[1,2]`, `cattle` `[1,57]`, `buffalo` `[1,18]` logits, read by index 0/1/2 |
| Post-processing | softmax + argmax, species-gated breed head (as in the app) |

## Artifacts in the App

```
flutter_app/assets/models/
├── model.tflite            # INT8 compute, float32 [0,1] I/O
├── labels_binary.txt       # "cattle" (index 0) / "buffalo" (index 1)
├── labels_cattle.txt       # 57 lines, line i = class i
└── labels_buffalo.txt      # 18 lines, line i = class i
```

## Model Sizes (measured, lite2)

| Artifact | Size |
|---|---|
| ONNX FP32 (reference) | ~25.7 MB |
| ONNX INT8 (QDQ, mobile) | **~7.1 MB** |
| TFLite INT8 | ~6–7 MB (expected) |

## QAT — optional recovery path only

`python -m src.train --include-qat` (OFF by default) fine-tunes with **per-tensor** fake-quantization observers (the x86 per-channel default broke conversion), starting from the best phase-2 EMA checkpoint, and saves `<backbone>_quantized.pt`. Use it only if converter PTQ drops more than ~2 pts — the artifact is not a TFLite/ORT model, and quantized checkpoints cannot be re-exported through `src.export`.
