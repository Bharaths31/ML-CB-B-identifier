# Self-Describing Models (Zero External JSON)

Every `.pt` and `.onnx` artifact is now the single source of truth for how to
run it. `test_model.py` and the Flask API no longer need
`cattle_classes.json`, `buffalo_classes.json`, `model_info.json` or
`outputs/ood/*.json` beside a model.

Two mechanisms work together:

1. **Runtime introspection** — `src/model_introspect.py` derives the contract
   from tensors / the ONNX graph, so files exported before this change work
   immediately.
2. **Embedded metadata** — `scripts/embed_metadata.py` (and `src/export.py`)
   permanently writes the contract into the file.

---

## 1. Runtime API

```python
from src.model_introspect import inspect_and_load

bundle = inspect_and_load("path/to/model.pt")   # or .onnx
# bundle.engine              -> "pt" | "onnx" | "torchscript"
# bundle.model_or_session    -> nn.Module or onnxruntime.InferenceSession
# bundle.transform           -> torchvision Compose for THIS model
# bundle.cattle_inv_map      -> {0: "amritmahal", ...}
# bundle.buffalo_inv_map     -> {0: "banni", ...}
# bundle.ood_config          -> embedded thresholds (or config defaults)
# bundle.image_size, bundle.normalization_baked, bundle.input_dtype
# bundle.spec_summary()      -> JSON-safe description for the Dev Console
```

Specialised entry points: `inspect_and_load_pt(path, device)`,
`inspect_and_load_onnx(path, use_cuda)`, and `describe_model_file(path)`.

### What is introspected

| Field | PyTorch source | ONNX source |
|---|---|---|
| Backbone (`lite2`/`lite4`) | `backbone.blocks.0.0._project_conv.weight` width (16 / 24) | metadata, else filename fallback |
| Attention (`cbam`/`se`) | presence of `attention.spatial_attention.conv.weight` | metadata |
| Head widths | `binary_head.0.weight`, `cattle_head.0.weight` | output shapes |
| Class counts | final `cattle_head.<fi>.weight` / `buffalo_head.<fi>.weight` | last dim of `cattle` / `buffalo` outputs |
| Cosine head | final head weight has no bias | metadata |
| Image size | metadata, default `IMAGE_SIZE` | `input.shape[2]` |
| Normalization | metadata, default caller-side | embedded `mean`/`std` initializers, leading `Sub → Div`, or `*mobile*` / `*int8*` filename |
| Class names | metadata → `EXPECTED_*_BREEDS` (count match) → local JSON → synthetic | same |
| OOD thresholds | `ckpt["model_meta"]["ood"]` | `metadata_props["ood"]` |

Class maps are accepted as `{name: idx}`, `{idx: name}`, a list of names, or a
JSON string. A map is only used when its length matches the head width, so a
50-class map can never be paired with a 42-class head.

### Normalization correctness

`bundle.transform` stops at `ToTensor()` when `normalization_baked=True`.
Mobile ONNX exports (`_MobileOutputs`) already contain `(x - mean) / std`, so
this prevents the historical **double-normalization** bug. Raw ONNX exports
(`_RawOutputs`) and PyTorch checkpoints get `Normalize(mean, std)` as before.

Inference code should always call `bundle.transform(img)` instead of a global
`TRANSFORM`.

---

## 2. Embedding metadata

```bash
# every model under outputs/
python scripts/embed_metadata.py --all

# one or more files
python scripts/embed_metadata.py --file outputs/checkpoints/my_model.pt
python scripts/embed_metadata.py --file outputs/export/my_model.onnx

# inspect without running inference
python scripts/embed_metadata.py --inspect outputs/export/my_model.onnx

# override auto-detection
python scripts/embed_metadata.py --file model.onnx \
  --cattle-classes data/splits/cattle_classes.json \
  --buffalo-classes data/splits/buffalo_classes.json \
  --ood-json outputs/ood/my_model.json \
  --baked-normalization
```

Behaviour:

- **`.pt`** — adds/updates `ckpt["model_meta"]` and preserves `phase`,
  `epoch`, `metrics`, optimizer state, etc. A raw `state_dict` is wrapped as
  `{"state_dict": ..., "model_meta": ...}`. TorchScript archives store the
  same JSON in an extra file (`model_meta.json`).
- **`.onnx`** — replaces duplicate keys in `metadata_props` and writes the
  contract as JSON strings (idempotent).
- Files are written atomically; weights are never modified.
- OOD thresholds are auto-discovered from the model's directory or
  `outputs/ood/`; otherwise `src/config.py` defaults are embedded.

`src/export.py` calls the embedders automatically for `--mode onnx`,
`--mode onnx-int8` (FP32 + INT8), and `--mode float16`, and the portable
bundle's `model.pt` is embedded too. `src/train.py` embeds metadata into every
saved checkpoint and re-embeds the calibrated OOD thresholds after training.

---

## 3. Metadata schema (`model_meta_version: "1.0"`)

```json
{
  "model_meta_version": "1.0",
  "backbone": "lite2",
  "attention": "cbam",
  "feature_dim": 1280,
  "binary_dim": 512,
  "breed_dim": 384,
  "cosine_head": false,
  "num_cattle": 50,
  "num_buffalo": 18,
  "image_size": 260,
  "eval_resize": 260,
  "pad_to_square": true,
  "imagenet_mean": [0.485, 0.456, 0.406],
  "imagenet_std": [0.229, 0.224, 0.225],
  "normalization_baked": false,
  "normalization_mode": "caller",
  "input_dtype": "tensor(float)",
  "cattle_classes": {"amritmahal": 0, "...": 49},
  "buffalo_classes": {"banni": 0, "...": 17},
  "ood": {
    "enabled": true,
    "energy_temperature": 1.0,
    "energy_threshold": -3.6719,
    "msp_threshold": 0.6023,
    "binary_confidence_floor": 0.6
  }
}
```

`pad_to_square` records whether the runtime should use `PadToSquare` (whole
animal) or `Resize + CenterCrop`. `normalization_mode` is the human-readable
form of `normalization_baked`.

---

## 4. Uploads

`POST /api/upload-model` (Flask and the standalone tester) now:

1. saves the file,
2. embeds default metadata when it is missing,
3. validates it with `inspect_and_load`, and
4. returns the introspected spec in the response.

A single `.pt`/`.onnx` upload is therefore usable immediately — no JSON
sidecars.

---

## 5. Migration checklist

```bash
# 1. Bake the contract (and current OOD calibration) into existing artifacts
python scripts/embed_metadata.py --all

# 2. Verify one file
python scripts/embed_metadata.py --inspect outputs/export/portable/<bundle>/model.pt

# 3. Run the app; JSON files may now be deleted
python test_model.py
```
