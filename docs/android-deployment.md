# 15. Android Deployment

### QAT Pipeline

Phase 3 (QAT) produces an INT8-ready model for mobile inference:
1. Conv-BN fusion: merges batch norm into convolutions
2. QAT training: inserts fake-quantize observers, fine-tunes with quantization noise
3. INT8 conversion: `torch.ao.quantization.convert()` produces true INT8 weights
4. Checkpoint: `<backbone>_quantized.pt`

### Export Formats for Android

| Format | File | Use Case |
|---|---|---|
| Portable | `portable/<backbone>_*/model.pt` | PyTorch Mobile / custom runtime |
| ONNX | `<backbone>_fp32.onnx` | ONNX Runtime Mobile, TFLite via converter |
| INT8 | `<backbone>_quantized.pt` | Smallest size, fastest inference |

### Model Sizes (approximate)

| Backbone | FP32 | INT8 (post-QAT) |
|---|---|---|
| lite2 | ~24 MB | ~6 MB |
| lite4 | ~50 MB | ~13 MB |

---