# Export & Deployment

The classifier can be exported to four formats for different deployment targets. `local_train.py` exports all four automatically after training. Manual export is also supported.

---

## Export Command

```
python -m src.export [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backbone` | `lite2` \| `lite4` | `lite2` | Which backbone's checkpoint to export |
| `--mode` | `onnx` \| `int8` \| `float16` \| `portable` | **required** | Export format |
| `--checkpoint` | path | auto (latest in `outputs/checkpoints/`) | Specific `.pt` file to use |

---

## Export Formats

| Mode | Output | Use Case |
|---|---|---|
| `portable` | `outputs/export/portable/<backbone>_<tag>/` | Python inference with full metadata |
| `onnx` | `outputs/export/<backbone>_fp32.onnx` | Framework-free: ONNX Runtime, TFLite converter |
| `int8` | `outputs/export/<backbone>_int8.pt` | Smallest size, fastest mobile CPU |
| `float16` | `outputs/export/<backbone>_float16.pt` | Mobile GPU inference |

### Portable Bundle

The portable format is a self-contained folder for Python-based inference:

```
outputs/export/portable/<backbone>_phase2_best/
├── model.pt             # torch.load() checkpoint containing state_dict
├── cattle_classes.json  # {"amritmahal": 0, "ayrshire": 1, ...}
├── buffalo_classes.json # {"alambadi": 0, "banni": 1, ...}
└── model_info.json      # backbone, image_size, exported_at, usage
```

> **Note:** Portable bundles save `state_dict` (not TorchScript). Loading requires the `BreedClassifier` class from `src/model.py`. For framework-free deployment use ONNX instead.

### ONNX

ONNX opset 13 export. Use with [ONNX Runtime](https://onnxruntime.ai/) or convert to TFLite via ONNX-TensorFlow converter.

### INT8 (Post-QAT)

Requires training with `--include-qat` (or running Phase 3). The INT8 model uses PyTorch's `torch.ao.quantization.convert()` after Conv-BN fusion and fake-quantize observer training.

---

## Manual Export Commands

```bash
# Export all 4 formats (what local_train.py does automatically)
python -m src.export --mode portable  --backbone lite2
python -m src.export --mode onnx      --backbone lite2
python -m src.export --mode int8      --backbone lite2
python -m src.export --mode float16   --backbone lite2

# Export from a specific checkpoint
python -m src.export --mode portable --backbone lite2 \
  --checkpoint outputs/checkpoints/lite2_phase2_best.pt
```

---

## Model Sizes

| Backbone | FP32 (ONNX / portable) | INT8 (post-QAT) |
|---|---|---|
| lite2 | ~24 MB | ~6 MB |
| lite4 | ~50 MB | ~13 MB |

---

## Android Deployment

### QAT Pipeline

Phase 3 (QAT) produces an INT8-ready model for mobile inference:

1. **Conv-BN fusion**: Merges batch normalization into preceding convolutions
2. **QAT training**: Inserts fake-quantize observers, fine-tunes with quantization noise for 10 epochs
3. **INT8 conversion**: `torch.ao.quantization.convert()` produces true INT8 weights
4. **Checkpoint**: `<backbone>_quantized.pt`

Enable QAT:
```bash
python local_train.py --include-qat       # Automated pipeline
python -m src.train --backbone lite2      # Manual (QAT is default unless --skip-qat)
```

### Deployment Options

| Format | File | Runtime |
|---|---|---|
| Portable | `portable/<backbone>_*/model.pt` | PyTorch Mobile / custom Python runtime |
| ONNX | `<backbone>_fp32.onnx` | ONNX Runtime Mobile, TFLite (via converter) |
| INT8 TorchScript | `<backbone>_quantized.pt` | PyTorch Mobile (smallest, fastest) |

### Android Workflow

```bash
# Step 1: Train with QAT
python local_train.py --include-qat

# Step 2: The INT8 model is at:
#   outputs/export/lite2_int8.pt
#   outputs/export/lite2_fp32.onnx   (for ONNX Runtime Mobile)

# Step 3: Integrate into Android app using:
#   - PyTorch Mobile: org.pytorch:pytorch_android
#   - ONNX Runtime: com.microsoft.onnxruntime:onnxruntime-android
```