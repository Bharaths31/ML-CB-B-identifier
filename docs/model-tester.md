# Model Tester GUI

The **`test_model.py`** script provides a standalone web GUI for testing exported models on individual images. It auto-discovers all available checkpoints and lets you visually predict the species and breed of any cattle or buffalo image.

---

## Quick Start

```bash
# After training (or using an exported model):
python test_model.py
```

This opens a browser at `http://localhost:8501` with the GUI.

---

## Features

- **Drag & Drop / File Upload**: Supports PNG, JPG, JPEG, BMP, and WebP formats
- **Model Selector**: Auto-discovers all available checkpoints:
  - Phase 1/2/3 training checkpoints (`outputs/checkpoints/`)
  - INT8 quantized models (`lite2_quantized.pt`)
  - Portable export bundles (`outputs/export/portable/`)
- **Prediction Results**:
  - Species classification: **Cattle** or **Buffalo** with confidence %
  - **Top-5 breed predictions** with animated confidence bars
  - Model metadata footer (which model was used, device info)
- **Zero Config**: Works out of the box with any trained checkpoint

---

## Usage

```bash
# Auto-open browser at http://localhost:8501
python test_model.py

# Custom port
python test_model.py --port 9000

# Don't auto-open browser
python test_model.py --no-browser
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--port N` | Port to serve on (default: `8501`) |
| `--no-browser` | Don't auto-open the browser window |

---

## How It Works

```
┌─────────────┐     POST /api/predict     ┌──────────────────┐
│  Browser UI  │ ─────────────────────────▶│  test_model.py   │
│  (HTML/JS)   │                           │  HTTP Server     │
│              │◀───────────────────────── │                  │
│  Shows Top-5 │     JSON response         │  ModelManager    │
│  breed bars  │                           │  ├─ load model   │
└─────────────┘                            │  ├─ preprocess   │
                                           │  └─ predict      │
                                           └──────────────────┘
```

1. **ModelManager** scans `outputs/checkpoints/` and `outputs/export/portable/` on startup
2. User uploads an image and selects a model in the browser
3. Image is sent as `multipart/form-data` to `POST /api/predict`
4. The model runs inference and returns species + top-5 breeds as JSON
5. The browser renders animated confidence bars

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the HTML GUI |
| `GET` | `/api/models` | Returns available models + device info |
| `POST` | `/api/predict` | Accepts `image` (file) + `model` (name), returns prediction |

### Example `/api/predict` Response

```json
{
  "species": "Cattle",
  "species_confidence": 90.07,
  "top_breed": "Gir",
  "top_breed_confidence": 45.2,
  "top5_breeds": [
    {"breed": "Gir", "confidence": 45.2},
    {"breed": "Sahiwal", "confidence": 22.1},
    {"breed": "Red Sindhi", "confidence": 12.8},
    {"breed": "Tharparkar", "confidence": 8.4},
    {"breed": "Kankrej", "confidence": 5.1}
  ],
  "model_used": "lite2_phase2_best"
}
```

---

## Workflow

After training with `local_train.py`:

```bash
# 1. Train (any mode)
python local_train.py --quarter-data

# 2. Test exported models visually
python test_model.py

# 3. Upload any cattle/buffalo image in the browser
# 4. See species + breed prediction with confidence
```

---

## Requirements

- Trained model checkpoint(s) in `outputs/checkpoints/` or `outputs/export/portable/`
- Class maps in `data/splits/` (`cattle_classes.json`, `buffalo_classes.json`)
- Python packages: `torch`, `torchvision`, `Pillow` (installed by `local_train.py`)

---
