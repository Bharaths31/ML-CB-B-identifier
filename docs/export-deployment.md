# 7. Export & Deployment

### Export Modes (`python -m src.export`)

| Mode | Output | Notes |
|---|---|---|
| `onnx` | `<backbone>_fp32.onnx` | opset 13, static or dynamic batch |
| `int8` | `<backbone>_int8.pt` + `_int8_traced.pt` | PTQ with 32-batch calibration |
| `float16` | `<backbone>_float16.pt` | TorchScript traced |
| `portable` | `portable/<backbone>_<tag>/` folder | Self-contained: model + labels + info |

### Portable Export Structure

```
outputs/export/portable/<backbone>_phase2_best/
├── model.pt                 # torch checkpoint {state_dict: ...}
├── cattle_classes.json      # {"amritmahal": 0, "ayrshire": 1, ...}
├── buffalo_classes.json     # {"alambadi": 0, "banni": 1, ...}
└── model_info.json          # {backbone, image_size, usage, exported_at}
```

---