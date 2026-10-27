# Project Source of Truth — Cattle & Buffalo Breed Classifier

> **Status:** Agent-facing reference, verified against the current source tree on **2026-10-27**.
>
> **Scope:** Only mechanisms that are actually implemented and reachable in this repository are described as active. Legacy, dormant, stored-only, and explicitly unimplemented paths are labelled. If this document and the code disagree, **the code wins** — update this document after verifying the code.
>
> **Supersedes for agent decisions:** `README.md`, `docs/*.md`, `architecture_components.md`, `.agents/knowledge/CONTEXT.md`, and comments inside source files. Several of those are stale in specific, documented places (see §15.1).

---

## 0. How to use this document

| Label | Meaning |
|---|---|
| **ACTIVE** | Reachable from the current launcher/routes/training entry points. |
| **LEGACY** | Code exists but is not reachable from the current router or launcher. Do not describe it as a current user feature. |
| **STORED-ONLY** | A setting is persisted/editable but has no consumer in the active runtime. |
| **NOT IMPLEMENTED** | No code path exists. Do not assume it. |

The two production surfaces are:

1. **React web app** (`breed-classifier-react/`), routes `/` and `/dev`, served by Vite and proxying to the Flask API.
2. **Standalone model tester** (`test_model.py`), an independent HTTP GUI on port **8501**, with its own embedded HTML/JS. It is *not* the backend used by the React app.

The active backend for the React app is **`server/app.py` on port 5000**, not `test_model.py`. `breed-classifier-react/src/api.js` has a stale comment mentioning 8501; the Vite proxy targets 5000.

---

## 1. Project snapshot

| Field | Implemented value |
|---|---|
| Goal | Classify images of Indian cattle and buffalo breeds with a lightweight CNN suitable for CPU/mobile deployment. |
| Framework | Python 3.11+ (repo uses a `.venv`), PyTorch ≥ 2.1, torchvision. |
| Model | `BreedClassifier`: EfficientNet-Lite2/4 backbone + CBAM/SE attention + binary head + two breed heads + training-only projection/trait heads. |
| Configured classes | **50 cattle + 18 buffalo = 68** (`src/config.py`). Heads are dynamically sized from the class maps actually present, so a trained checkpoint can have other counts. |
| Input size | **260 × 260 RGB**, ImageNet normalization. |
| Feature dimension | **1280**. |
| Parameters (measured) | Lite2: 4.81M backbone + 0.98K attention + 1.94M heads = **8.55M total**; Lite4: 11.73M backbone + 1.67K attention + 1.94M heads = **15.47M total**. Default heads (50 cattle + 18 buffalo), binary_dim 512, breed_dim 384. |
| Training phases | Phase 1 warmup (8 epochs), Phase 2 multi-task fine-tune (80 epochs), optional Phase 3 QAT (10 epochs). |
| Serving | Flask 3 + Flask-SocketIO + Socket.IO, port 5000, `async_mode="threading"`. |
| Web frontend | React 18, Vite 5, React Router 7, `socket.io-client`, hand-written CSS. |
| Edge app | Flutter: TFLite on Android, TF.js on web. |
| Export formats | Portable checkpoint bundle, FP32 ONNX, FP16 TorchScript, ONNX INT8 QDQ, TFLite FP32/INT8. |
| OOD | Post-hoc energy + binary MSP + binary confidence floor. No retraining required. Per-model auto-resolution at startup and on model switch. |
| Explainability | CBAM spatial-attention mask + linear class-activation heatmap (zero-backward Grad-CAM projection), rendered as an interactive 260×260 overlay. |
| Confidence rubric | Five levels from joint soft-routed confidence C, with expert escalation for Levels 1–2. |
| Expert escalation | Thread-safe SQLite queue (`outputs/escalations.db`), veterinary review portal at `/expert`, and verified-image promotion into `data/raw/` + `outputs/active_learning_manifest.csv`. |
| Registration | Bharat Pashudhan (BPA) 12-digit ear-tag registration card with canvas-rendered QR, PNG and printable ODT. |
| Persistence | JSON files: `outputs/app_config.json`, `outputs/ood/*.json`, `outputs/logs/presenter_config.json`; SQLite `outputs/escalations.db`; CSV `outputs/active_learning_manifest.csv`; browser `localStorage` for session/theme/language/escalation outbox. |
| Auth | **None.** No user accounts, no API auth, no CSRF protection. |


### 1.1 Canonical launch commands

```bash
# Full app: Flask (5000) + React Vite (5173)
python run.py

# Backend only
python server/app.py --port 5000

# Frontend only
cd breed-classifier-react && npm run dev

# Training (automated wrapper, recommended)
python local_train.py --quarter-data

# Training (unified CLI wrapper)
python train_model.py --half-data

# Training (core engine directly)
python -m src.train --backbone lite2

# Independent model tester GUI
python test_model.py            # http://localhost:8501
python test_model.py --present  # presenter mode

# Evaluation / verification
python -m src.evaluate --backbone lite2
python -m src.verify
python -m src.parity_check --checkpoint <pt> --synthetic 8
```

### 1.2 Run/output naming (actual)

`make_run_id()` does **not** use the `RUN_ID_FORMAT` string by default. With no explicit `--run-tag` it increments `VERSION.txt` and produces:

```
v<N>_<MMDD-HHMM>_<4 random chars>      e.g. v30_1027-1432_a3f2
```

Artifacts are then suffixed with `_<run_id>`:

```
outputs/checkpoints/lite2_phase2_best_v30_1027-1432_a3f2.pt
outputs/export/lite2_v30_1027-1432_a3f2_fp32.onnx
outputs/export/portable/lite2_lite2_phase2_best_v30_.../
outputs/metrics/lite2_v30_1027-1432_a3f2_metrics.json
outputs/ood/<checkpoint-stem>.json
outputs/ood_thresholds.json
logs/<exec_id>/{manifest.json,config.json,run.log,events.jsonl,...}
```

`unique_path()` appends `_2`, `_3`, … if a name already exists, so explicit tags do not overwrite prior results.

---

## 2. System map

```text
PLAIN BROWSER (React, :5173)
  UserPage "/"                         DevPage "/dev"          ExpertPage "/expert"
     │                                    │                        │
     │ fetch /api/*                       │ fetch /api/*           │ fetch /api/*
     └──────────────┬─────────────────────┴────────┬───────────────┘
                    ▼                              ▼
        Vite dev proxy (:5173 -> :5000)
                    ▼
        Flask API (:5000, server/app.py)
          ├─ ModelManager (discover/load/cache checkpoints + count-matched class maps)
          ├─ inference.py (PIL → 260 transform → model → soft routing → OOD → CBAM/CAM)
          ├─ SharedConfigDB (outputs/app_config.json)
          ├─ OOD profile manager (outputs/ood/*.json, per-model auto binding)
          ├─ ODT export (odfpy)
          ├─ EscalationDB (outputs/escalations.db + active-learning manifest)
          └─ Socket.IO namespace "/ws"

        Artifacts read by Flask:
          outputs/checkpoints/*.pt
          outputs/export/*.onnx
          outputs/export/portable/<bundle>/{model.pt, cattle_classes.json, buffalo_classes.json, model_info.json}
          outputs/ood/*.json, outputs/ood_thresholds.json
          outputs/escalations.db

STANDALONE TESTER (:8501, test_model.py)
  embedded HTML/JS ──► its own HTTPServer + ModelManager
  independent of Flask and of the React app

FLUTTER APP
  assets/models/model.tflite + labels_*.txt
  or TF.js assets/models/model_web.json on web
  → on-device softmax + soft routing; no OOD, no session, no traits
```

---

## 3. Repository map

| Path | Role | Status |
|---|---|---|
| `src/config.py` | Single source of constants: geometry, losses, phases, paths, OOD, logging. | ACTIVE |
| `src/model.py` | Consolidated backbone, CBAM/SE, `BreedClassifier`, trait heads, `OODDetector`. | ACTIVE |
| `src/data_pipeline.py` | Dataset scan, splits, transforms, sampler, mixers, dataloaders. | ACTIVE |
| `src/train.py` | Training loop, losses, EMA, checkpointing, export hook, OOD calibration, CLI. | ACTIVE |
| `src/metrics.py` | `evaluate_epoch` used by `src.train` (autocast-aware). | ACTIVE |
| `src/evaluate.py` | Detailed val/test evaluation + confusion matrices. | ACTIVE |
| `src/export.py` | Portable/ONNX/FP16/ONNX-INT8/TFLite export and checkpoint loader. | ACTIVE |
| `src/parity_check.py` | fp32 vs ONNX/TFLite INT8 parity gate. | ACTIVE |
| `src/verify.py` | Backbone load + head-shape sanity check. | ACTIVE |
| `src/run_utils.py` | Run ids, timestamped paths, checkpoint discovery, execution logger. | ACTIVE |
| `src/ood_calibration.py` | Folder-based OOD calibration and breed-name matching. | ACTIVE |
| `src/cbam.py`, `src/efficientnet_lite.py`, `src/ood_detector.py`, `src/traits.py` | Compatibility re-export shims to `src/model.py`. | ACTIVE (shims) |
| `train_model.py` | High-level CLI wrapper around `src.train` + post-training export/OOD/config sync. | ACTIVE |
| `local_train.py` | Fully automated pipeline: env → dataset → verify → train → export → OOD. | ACTIVE |
| `run.py` | Starts Flask + Vite, health checks, opens browser. | ACTIVE |
| `server/app.py` | Flask app factory, blueprints, Socket.IO, `/api/health`. | ACTIVE |
| `server/model_manager.py` | Discovery, class-map loading, lazy loading, active-model switch, OOD recalibration. | ACTIVE |
| `server/services/inference.py` | Server-side preprocessing, inference, OOD, soft routing, display filtering. | ACTIVE |
| `server/shared_config.py` | Thread-safe JSON config DB with dot keys and notifications. | ACTIVE |
| `server/routes/*.py` | REST endpoints (predict/models/config/OOD/export/dev). | ACTIVE |
| `server/services/odt_export.py` | ODT report generation. | ACTIVE |
| `server/services/metadata.py` | Image/EXIF metadata extraction. | ACTIVE |
| `server/websocket/events.py` | `/ws` namespace with server status events; React client now connects to `/ws`. | ACTIVE |
| `server/services/escalation_db.py` | Thread-safe SQLite expert queue + active-learning promotion. | ACTIVE |
| `server/routes/escalations.py` | `POST/GET /api/escalations`, resolve endpoint, `/ws` events. | ACTIVE |
| `test_model.py` | Standalone HTTP tester/GUI, dev/present modes, its own ModelManager/API. | ACTIVE, independent |
| `breed-classifier-react/src/pages/UserPage.jsx` | Public user page. | ACTIVE |
| `breed-classifier-react/src/pages/DevPage.jsx` | Developer control plane. | ACTIVE |
| `breed-classifier-react/src/pages/ExpertPage.jsx` | Veterinary expert review portal (`/expert`). | ACTIVE |
| `breed-classifier-react/src/components/BpaRegistrationModal.jsx` | BPA ear-tag registration + QR health card. | ACTIVE |
| `breed-classifier-react/src/utils/escalationQueue.js` | Offline escalation outbox + GPS helper. | ACTIVE |
| `breed-classifier-react/src/utils/confidence.js` | Five-level rubric + presenter confidence formatting. | ACTIVE |
| `breed-classifier-react/src/components/*` | UI components. Some are legacy-only (see §9.16). | mixed |
| `breed-classifier-react/src/utils/imageOptimizer.js` | Client 260×260 zero-crop optimizer + opt-in LAB CLAHE. | ACTIVE |
| `breed-classifier-react/src/utils/cropAndQuality.js` | Crop math, quality gate, 260×260 report helper, share-card renderer. | ACTIVE |
| `breed-classifier-react/src/utils/sessionStore.js` | Local herd session + CSV/JSON/ODT shaping. | ACTIVE |
| `breed-classifier-react/src/data/breedTraits.js` | Curated 68-breed UI trait table (not a model output). | ACTIVE |

| `breed-classifier-react/src/i18n/*` | EN + Tamil translations, provider. | ACTIVE |
| `flutter_app/lib/*` | Flutter edge inference and UI. | ACTIVE (separate app) |
| `data/raw/{cattle,buffalo}/<breed>/*` | Training images. | data |
| `data/splits/*.csv`, `*_classes.json` | Generated split manifests and class maps. | generated |
| `data/breed_traits.json` | Training-only trait labels. | data |
| `data/dataset_inventory/*.json` | Per-dataset breed/resolution inventory. | generated |
| `colab/*` | Standalone Colab copies/converters; not the authoritative engine. | alternative |
| `scripts/*` | Calibration, audit, diagnosis, parity, packaging helpers. | utilities |
| `docs/*`, `README.md`, `architecture_components.md` | Human documentation, partly stale. | reference only |

---

## 4. Model definition

### 4.1 `BreedClassifier`

Source: `src/model.py`.

**Backbone** — a custom EfficientNet-Lite implementation:

| Arch | Stem | Stage output channels | Head channels |
|---|---|---|---|
| `lite2` | 32 | `[16, 24, 48, 88, 120, 208, 352]` | 1280 |
| `lite4` | 32 | `[24, 32, 56, 112, 160, 272, 448]` | 1280 |

- MBConv blocks use ReLU6 by default; `make_activation()` also supports SiLU/Swish and ReLU.
- Pretrained ImageNet weights are loaded from `efficientnet_lite2.pth` / `efficientnet_lite4.pth` at the project root. Missing files cause training from scratch with a printed warning.
- `load_backbone_weights()` rejects key mismatches except `fc.weight` / `fc.bias`.

**Attention**

- `CBAM_AFTER_STAGE = 3` by default (inserted after stage index 3).
- `build_attention("cbam" | "se", channels)`.
- Both CBAM and SE use identity initialization by default (`CBAM_IDENTITY_INIT=True`): the gate is `2*sigmoid(att)-1`, and the final conv is zero-initialized, so the module starts as an identity transform.

**Heads**

```
pooled 1280-d
├── binary_head : Linear(1280→512) + BN + ReLU + Dropout(0.4)
│                 Linear(512→256)  + BN + ReLU + Dropout(0.2)
│                 Linear(256→2)
├── cattle_head : Linear(1280→384) + BN + ReLU + Dropout(0.4)
│                 Linear(384→192)  + BN + ReLU + Dropout(0.2)
│                 Linear(192→C_cattle)        (or CosineHead)
├── buffalo_head: same structure → C_buffalo  (or CosineHead)
├── projection_head (training-only, never exported):
│                 Linear(1280→1280) + ReLU + Linear(1280→128)
└── trait_heads (only if --trait-weight > 0; training-only):
                  one 2-layer classifier per trait field
```

- `binary_dim` defaults to 512, `breed_dim` to 384, `PROJECTION_DIM=128`, `DROPOUT=0.4`.
- Cosine/ArcFace heads are optional (`--cosine-head`). `CosineHead.forward` returns `scale * cos(theta)` with **no** margin; the additive angular margin is applied only to target-class logits inside the loss during training.
- `forward()` returns a dict with `binary`, `cattle`, `buffalo`, `features`, `embedding`.
- `predict()` performs **soft routing**:
  `cat([P(cattle)*softmax(cattle), P(buffalo)*softmax(buffalo)])`.
  It never discards the non-argmax species head.
- `predict_safe()` returns a zero distribution plus OOD details when the detector rejects.

### 4.2 Dynamic head sizing / checkpoint loading

Implemented in `src/export._load_model`, `server/model_manager._load_model`, `test_model.ModelManager._load_model`, `scripts/calibrate_ood.py`:

- Accepts checkpoints shaped as:
  - `{"state_dict": ...}`
  - `{"model_state_dict": ...}`
  - a raw state dict
  - an already-instantiated `torch.nn.Module` / TorchScript module
- Infers `binary_dim`, `breed_dim`, cattle class count, and buffalo class count from the final layer weights.
- Detects cosine heads by checking that the final head layer has a weight but no bias.
- Strips `_orig_mod.` / `module.` prefixes and QAT/fake-quant/observer keys during export.
- Loads with `strict=False` and filters shape-mismatched keys in the tester/server. The exporter raises on genuinely missing non-projection keys.
- Training sizes heads from the class maps on disk, so dropped/absent breeds produce smaller heads end to end.

### 4.3 OOD detector

Source: `src/model.py::OODDetector`.

**Energy**

```
E(x) = -T * logsumexp(concat(binary_logits, cattle_logits, buffalo_logits) / T)
```

Higher energy = more out-of-distribution.

**MSP**

```
MSP(x) = max(softmax(binary_logits))
```

**Decision rule — any one trigger rejects**

```
is_ood = (energy > energy_threshold)
      OR (msp < msp_threshold)
      OR (msp < binary_confidence_floor)
```

Defaults in `src/config.py`:

| Parameter | Default |
|---|---|
| `OOD_ENABLED` | `True` |
| `OOD_PERCENTILE` | `95.0` |
| `OOD_ENERGY_TEMPERATURE` | `1.0` |
| `OOD_ENERGY_THRESHOLD` | `-3.2004` |
| `OOD_MSP_THRESHOLD` | `0.5863` |
| `OOD_BINARY_CONFIDENCE_FLOOR` | `0.60` |

**Calibration** (`OODDetector.calibrate`):

- Energy threshold = `quantile(energies, percentile/100)`.
- MSP threshold = `quantile(msps, (100-percentile)/100)`.
- Because the triggers are OR-ed, the actual in-distribution rejection rate can exceed `100 - percentile`.
- Saved profile includes temperature, thresholds, floor, percentile, sample count, timestamp, and energy/MSP mean/std/min/max.

**Profile resolution order** (`resolve_profile_path`):

1. Exact `outputs/ood/<clean_model_name>.json`
2. `outputs/ood/<clean_model_name>_ood.json`
3. `outputs/export/portable/<matching bundle>/ood_thresholds.json`
4. Substring match in `outputs/ood/*.json`
5. Global `outputs/ood_thresholds.json`
6. Config fallback values

`save()` writes both `outputs/ood/<name>.json` and the global `outputs/ood_thresholds.json`.

### 4.4 Export contracts

Source: `src/export.py`.

| Mode | Wrapper | Input convention | Outputs |
|---|---|---|---|
| `portable` | raw checkpoint | PyTorch `[B,3,260,260]`, caller normalizes | `model.pt`, class maps, `model_info.json`, optional `ood_thresholds.json` |
| `onnx` | `_RawOutputs` | caller applies ImageNet normalization | `input`, `binary`, `cattle`, `buffalo` (opset 13) |
| `float16` | `_RawOutputs` | caller applies ImageNet normalization | TorchScript FP16 |
| `onnx-int8` | `_MobileOutputs` | RGB `[0,1]`; normalization baked into graph | statically quantized QDQ ONNX (opset 17) |
| `tflite` | `_MobileOutputs` | RGB `[0,1]`; normalization baked in | FP32 and full-integer INT8 TFLite + label files |
| `int8` (x86) | — | — | **Removed**; prints an error instead. |

- All graphs emit only the three heads; projection/trait heads are excluded.
- `_write_label_files()` writes `labels_binary.txt`, `labels_cattle.txt`, `labels_buffalo.txt` ordered by class index.
- TFLite export copies `model.tflite`, `lite2_fp32.tflite`/`lite4_fp32.tflite`, and label files into `flutter_app/assets/models/` unless `--skip-app-assets`.
- The TFLite path requires `tensorflow`, `onnx2tf`, `tf-keras`, `onnx-graphsurgeon`, `sng4onnx`, `onnxsim`.
- **No TF.js `model_web.json` export exists** in this pipeline.

---

## 5. Input image handling (end-to-end)

This is the most important contract in the project. The model is only as good as the region and scale it receives.

### 5.1 Canonical numeric contract

Training and server evaluation use:

```text
PIL RGB
  → Resize(shortest_side = 260)     # torchvision Resize(int)
  → CenterCrop(260)
  → ToTensor()                      # [0,1], CHW
  → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

- Optional `EVAL_PAD_TO_SQUARE=True` / `--pad-to-square` replaces resize+center-crop with:
  `long side → 260`, centered on a 260×260 canvas filled with ImageNet mean.
- `TRAIN_RESIZE = 288`; eval uses 260 unless `EVAL_MATCH_TRAIN_RESOLUTION=True`.
- The React client sends an already **260×260 JPEG**. Therefore the server's `Resize(260)` and `CenterCrop(260)` are identity transforms; the region the user framed is the region the model evaluates.

### 5.2 Browser decode and 260×260 optimizer

Source: `breed-classifier-react/src/utils/imageOptimizer.js` and `hooks/useImageOptimizer.js`.

**Decode**

1. `createImageBitmap(blob, { imageOrientation: "from-image" })` where supported (EXIF orientation applied).
2. Fallback: `new Image()` with an object URL.
3. Decode failure → `"Could not decode image"`.

**Constants**

| Item | Value |
|---|---|
| Target | `260 × 260` |
| ImageNet mean RGB | `[124, 116, 104]` / `#7c7468` |
| Fit modes | `stretch`, `pad` |
| Pad styles | `imagenet_mean`, `ambient_blur`, `black` |
| Default upload fit | `stretch` + `imagenet_mean` |
| Allow upscale | always `false` from the UI |
| Output format | JPEG quality `0.9` |
| Output name | `zero_crop_260_<sanitized-original>.jpg` |

**Layout math**

- `stretch`: draw directly to 260×260; `scaleX=260/W`, `scaleY=260/H`; aspect ratio changes, no bars, no crop.
- `pad`:
  - `scaleFactor = min(1, 260 / max(W,H))` when `allowUpscale=false`.
  - `innerW = round(W*scaleFactor)`, `innerH = round(H*scaleFactor)`.
  - `padX = floor((260-innerW)/2)`, `padY = floor((260-innerH)/2)`.
  - Small images are **padded**, not enlarged, by default.
- Padding is drawn first:
  - `imagenet_mean`: solid `#7c7468` (≈0 after normalization).
  - `ambient_blur`: blurred/darkened copy of the image stretched behind the sharp image (`blur(12px) brightness(0.6)`).
  - `black`: `#000000`.
- Progressive halving (`stepDownResize`) avoids aliasing from large photos; each halving uses high-quality canvas smoothing.

**Metadata returned to the UI**

`originalWidth`, `originalHeight`, `originalAspectRatio`, `originalBytes`, `optimizedWidth/Height`, `scaledInnerWidth/Height`, `padX`, `padY`, `scaleFactor`, `scaleX/Y`, `upscaled`, `fitMode`, `padStyle`, `optimizedBytes`, `savingsPct`, `durationMs`, optional `manuallyCropped`.

**State lifecycle** (`useImageOptimizer`)

- `original` is the immutable user upload: `{file, dataUrl, name, bytes}`.
- `cropped` is the current manual crop source.
- `result` is the current 260×260 optimized file.
- Request IDs cancel stale async optimizations.
- `applyCrop(file)` runs the optimizer on the cropped file.
- `clearCrop()` drops the manual crop and re-optimizes the original upload (“downscale only”).
- `setFitMode(nextFit)` re-runs on `cropped.file || original.file`.
- `setClaheEnabled(bool)` re-runs with opt-in LAB CLAHE.
- `adoptResult(file, dataUrl, meta)` accepts an already-optimized camera result; no full-resolution original is retained in that path.
- `reset()` also resets fit to `stretch` + `imagenet_mean`.
- Blob object URLs are revoked via `releaseOptimizerResult()`.

### 5.3 Upload path (ACTIVE)

Source: `components/ImageUpload.jsx`, `pages/UserPage.jsx`.

- Drag/drop, click-to-browse, or keyboard `Enter`/`Space`.
- `accept="image/*"`; only `file.type.startsWith("image/")` is checked. No dimension or byte-size limit in the client.
- The card displays the **original** image only. The 260×260 model input is inspectable in the header **Advanced** panel.
- **Image preparation mode:** **Downscale only** (default; no manual crop, full image resized to 260×260) or **Frame manually** (opens the 1:1 cropper). Switching back to downscale uses `optimizer.clearCrop()`.
- Additional actions: **Remove Image**, **Choose Another**, and the primary **Identify Breed** / **Identify Framed Breed** button.
- The classify button is hidden after a result exists (`showClassify={!predictionResult}`).
- A new upload, crop, fit change, or downscale-reset clears the previous prediction and error.
- Every upload/crop is scored on the resulting 260×260 canvas (`analyzeDataUrlQuality` → `analyzeCanvasQuality`); the report is forwarded to `UserPage`.
- The file sent to `/api/predict` is the optimized 260×260 JPEG; the original filename is sent separately for display/log/export.

### 5.4 Camera path (ACTIVE)

Source: `components/CameraCapture.jsx`, `utils/cropAndQuality.js`.

**Stream**

- `getUserMedia({video:{facingMode:{ideal}, width:{ideal:1920}, height:{ideal:1080}}, audio:false})`.
- Falls back to `{video:true, audio:false}`.
- Front/rear flip only shown when `enumerateDevices()` finds more than one video input.
- Torch only shown when `track.getCapabilities().torch` is true; toggled with `applyConstraints({advanced:[{torch:...}]})`.
- Camera errors are mapped to permission / not-found / generic messages.

**Viewfinder**

- Full sensor frame with `object-fit: contain`; the guide corners are drawn around the actual video rectangle, not a crop.
- Responsive viewport targets a clean 16:10/16:9 frame (`min-height: 360px`, `max-height: 62vh`).
- Top HUD is a single flex row: compact quality pill on the left, `260×260 • Zero-Crop` badge on the right.
- Optional **AR anatomical guide** (`Cattle Guide`, `Buffalo Guide`, or `Minimal Grid (Off)`) shows a rounded safe frame, dorsal/chest reference axes, and four flip-aware zones: Head & Horns, Hump / Dorsal Ridge, Dewlap, Tail / Rump. Labels remain upright when direction flips.
- Bottom telemetry pills show independent lighting/framing/sharpness states from the quality report; a dark or empty frame now reports `Animal in Frame: No Subject`, never `Ready`.
- A 130×130 downscaled full-frame quality report runs every **250 ms**.

**Capture**

- Manual shutter captures the current video frame to a canvas.
- Front-camera capture is mirrored on the canvas so the saved image matches the mirrored preview.
- Burst ×3 captures the first frame plus two more at 180 ms intervals and keeps the frame with the **highest sharpness score**.
- Capture is blocked while a capture is already in progress.
- Flash overlay for 220 ms.

**Auto-capture**

- When enabled, three consecutive quality reports with `isReady === true` trigger capture (~750 ms).
- A countdown ring shows `readyStreak/3`.
- Streak resets on a non-ready frame and after capture.

**Review**

- Shows the optimized 260×260 preview, telemetry, fit toggles (Pad / Stretch), full quality badge, and metadata grid.
- Fit change re-optimizes the captured frame (or the manual crop if one exists).
- **Adjust framing** opens the shared `ImageCropper` on the original captured frame.
- Actions: Retake, Adjust, Use This Photo.
- On confirm, the parent receives:
  `(optimized.file, optimized.dataUrl, optimized.meta, qualityReport)`.
- `UserPage` adopts that result, stores the quality report, switches to the Upload tab, and shows the image without immediately classifying.

**Default camera fit:** `stretch`.

### 5.5 Interactive cropper (ACTIVE, optional)

Source: `components/ImageCropper.jsx`, `utils/cropAndQuality.js`.

- Square reticle is **78% of the stage's shorter side** (`stageSide = min(w,h)*0.78`).
- Drag to pan on both axes; pinch to zoom; mouse wheel zooms 1.08×/0.92×; range slider 1–3× (`maxZoom=3` prop, internal clamps allow up to 4).
- Reticle source rectangle:

```
scale = coverScale(source, viewport) * zoom
offsetX = (sourceW*scale - viewportW)/2
offsetY = (sourceH*scale - viewportH)/2
size = min(reticleSize/scale, min(sourceW, sourceH))
x = clamp((reticleX + offsetX - panX)/scale, 0, sourceW - size)
y = clamp((reticleY + offsetY - panY)/scale, 0, sourceH - size)
if mirrored: x = sourceW - x - size
```

- Live quality is computed on a 130×130 sample of the authoritative square, debounced 140 ms.
- Confirm exports a **520×520 JPEG at quality 0.92** (`EXPORT_SIZE=520`), named `crop_<base>_<timestamp>.jpg`.
- The optimizer then downsizes that 520 crop to 260.
- The cropper always reopens the **original** upload/capture, never a previous crop.
- The target label reads “Model sees this · 1:1 · 260×260” (the intermediate file is 520, then optimized to 260).

### 5.6 On-device quality gate

Source: `utils/cropAndQuality.js`.

**Metrics**

| Metric | Formula |
|---|---|
| Sharpness | `100 * (1 - exp(-max(0, laplacian_variance)/260))` on the grayscale interior, 4-neighbour Laplacian |
| Exposure | `100 - |mean_luma-128|*0.55 - darkPct*60 - brightPct*60`, clamped 0–100 |
| Framing | `centerEdge*4 - borderTouch*110`, clamped 0–100 |
| Overall | `0.45*sharpness + 0.35*exposure + 0.20*framing` |

Where:

- `darkPct` = fraction of pixels with luma `< 25`.
- `brightPct` = fraction with luma `> 245`.
- `centerEdge` = mean Sobel magnitude in the central 70% (`15%..85%` box).
- `borderTouch` = fraction of the outer 10% ring with Sobel magnitude `> 60`.
- `borderEdge` = mean Sobel magnitude in that outer ring.

**Thresholds / statuses**

| Flag | Condition | Notes |
|---|---|---|
| `TOO_BLURRY` | `sharpness < 85` | first flag wins |
| `TOO_DARK` | `meanLuma < 45` or `darkPct > 0.5` | |
| `OVEREXPOSED` | `meanLuma > 225` or `brightPct > 0.4` | |
| `NO_SUBJECT_OR_TOO_FAR` | `meanLuma < 35` **or** `darkPct > 0.65` **or** `meanLuma > 235` **or** `centerEdge < 8` | framing score capped at 15 |
| `PARTIALLY_OUT_OF_FRAME` | `borderTouch > 0.28` **and** `borderEdge >= centerEdge` | framing score capped at 55 |
| `READY` | no flags | `isReady=true` only in this state |

The report also returns independent UI states:

- `lightingState`: `good` / `too_dark` / `overexposed`
- `framingState`: `ready` only when there is a subject and no edge clipping; otherwise `no_subject` or `edge_clipped`
- `sharpnessState`: `optimal` only when sharpness ≥ 85 **and** the frame is bright enough; otherwise `blurry`

Status is the **first flag** in the order above. The UI maps each status to an icon, tone, short guidance, and a longer hint (EN/TA).

**Important:** this is a hand-written heuristic, not an animal detector. It can flag empty/textured backgrounds or miss a small animal.

### 5.7 Server-side image handling

Source: `server/services/inference.py`, `test_model.py`.

- Reads raw multipart bytes from the `image` field.
- `Image.open(io.BytesIO(bytes)).convert("RGB")`.
- No explicit Flask `MAX_CONTENT_LENGTH` is configured in this repo.
- No EXIF transpose is performed server-side; PNG/JPG/WEBP/BMP that PIL can decode are accepted.
- The same transform chain as §5.1.
- `GET /api/image-info` (and the tester equivalent) exposes dimensions, aspect ratio, megapixels, format, mode, channels, bit depth, file size, DPI, EXIF, and unique colors for small images.
- The original filename is only passed through to logs/responses/export; the server does not store uploaded images.

### 5.8 Flutter edge handling

Source: `flutter_app/lib/data/utils/image_preprocessor.dart`.

```text
decodeImage(bytes)
→ resize shortest side to 260
→ center crop 260×260
→ planar RGB float32, value/255
→ shape [1, 3, 260, 260] (NCHW)
```

- TFLite graph expects `[0,1]` and has ImageNet normalization baked in (from `_MobileOutputs`).
- Backend selection: `kIsWeb ? TfJsEngine() : TfliteEngine()`.
- Image picker constraints: `maxWidth:1600`, `maxHeight:1600`, `imageQuality:92`.
- The edge postprocessor does soft routing and reports top-3; **no OOD**, no quality gate, no session, no traits, no share card.

### 5.9 What the image pipeline does *not* do

- No server-side object detection, segmentation, animal localization, pose, or body-part crop.
- No server-side quality gate or rejection.
- No server-side upscaling.
- No automatic rotation of EXIF from arbitrary API clients (only the browser decode path applies EXIF).
- No storage of the uploaded file.
- No automatic image retention/deletion policy beyond normal temp memory.
- No client-side compression loop or WebP/AVIF output (JPEG only).

---

## 6. Training pipeline

### 6.1 Entry points and call chain

| Entry point | What it actually does |
|---|---|
| `python -m src.train` | Core engine. Parses all flags, prepares splits/loaders, trains phases, exports portable, calibrates OOD, logs. |
| `python train_model.py` | Preflight, maps its flags to `src.train`, runs it in a subprocess, then runs `src.export` for `portable,onnx,float16` by default, runs `scripts/calibrate_ood.py`, and optionally updates `outputs/app_config.json` with `--set-active`. |
| `python local_train.py` | Fully automated: prerequisites, venv/deps, Kaggle download, unzip/organize/inventory, `src.verify`, data validation, `src.train`, multi-format `src.export`, `scripts/calibrate_ood.py`, config sync, summary. |
| `colab/cattle_buffalo_*.py/.ipynb` | Standalone Colab copies. They import parts of `src` but are not the authoritative engine. |
| `python data_collection.py` | Optional Kaggle/NBAGR/web image collection helper; not part of `local_train`'s default call chain. |
| `scripts/calibrate_ood.py` | Standalone validation-set OOD calibration used by the wrappers. |

### 6.2 Dataset layout, split rules, class maps

Source: `src/data_pipeline.py`, `local_train.py`.

**Expected root**

```
data/raw/
├── cattle/<breed_slug>/*.{jpg,jpeg,png,bmp,webp}
└── buffalo/<breed_slug>/*.{jpg,jpeg,png,bmp,webp}
```

- Folder names are lowercased; only two aliases are applied in `src/config.py`:
  - `holstein-friesian → holstein_friesian`
  - `luit_(swamp) → luit`
- `local_train.merge_into_species_dir()` also normalizes spaces/hyphens and strips `_cattle`, `_buffalo`, `_breed`, `indian_` prefixes/suffixes.
- `breed` alone is not globally unique: `bargur` exists under both cattle and buffalo, so every count/split/sampler operation is keyed on `(species, breed)`.

**Split modes**

| Mode | Selection | Split |
|---|---|---|
| Full (`prepare_splits`) | all images | 70/15/15 stratified with long-tail minimums |
| Half (`--half-data`) | `max(2, floor(0.5*n))` per breed | 70/15/15 |
| Quarter (`--quarter-data`) | `max(2, floor(0.25*n))` per breed | 70/15/15 |
| Smoke (`--smoke-test`) | `min(5, n)` per breed | 60/20/20 |

Long-tail minimums for full/half/quarter:

- `n == 1` → train only.
- `n == 2` → 1 train, 1 test.
- `n >= 3` → at least 1 val and 1 test; if train would fall below 2, borrow from test then val.

Class maps:

- Full mode uses the surviving classes and can drop small breeds only if `DROP_SMALL_BREEDS=True` with `min_images`.
- Half/quarter/smoke always write class maps from **all** breeds so architecture stays consistent.
- `_validate_class_counts()` expects exactly 50 cattle + 18 buffalo unless fewer were deliberately dropped. A mismatch raises with the extra/missing names.
- The generated `data/splits/` files are the authoritative label order for a training run; `labels_*.txt` and portable class maps are copied from there.

**Optional dedup**

`--dedup-splits` computes a 64-bit dHash (8×8) and keeps near-duplicates (Hamming ≤ 4) inside the same split.

### 6.3 Transforms and augmentation — actual defaults

The docs often say “all augmentation off”. The code is more specific:

`src/config.py` defaults:

| Setting | Default |
|---|---|
| `AUG_HORIZONTAL_FLIP` | **True** |
| `AUG_RANDOM_RESIZED_CROP` | False |
| `AUG_COLOR_JITTER` | False |
| `AUG_RANDAUGMENT` | False |
| `MIX_ENABLED` | False |

`src.train` ORs config defaults with CLI flags; `--no-augment` forces all off and disables mixing.

Per-entry-point effective defaults:

| Entry point | Flip | RRC | Color jitter | RandAugment | Mix |
|---|---|---|---|---|---|
| `python -m src.train` (no flags) | ON | off | off | off | off |
| `train_model.py` (default `--augment-preset none`) | ON | off | off | off | off |
| `local_train.py` (default `--augment-preset light`) | ON | ON | ON | off | off |

`--augment-preset light` = flip + mild RRC + mild colour jitter. `--augment-all` = flip + jitter + RandAugment + RRC + mix.

Transform details when enabled:

- `RandomResizedCrop(260, scale=(0.8,1.0), ratio=(0.92,1.08))` after `Resize(288)`.
- `RandomHorizontalFlip()`.
- `ColorJitter(brightness=.15, contrast=.15, saturation=.10, hue=.02)`; `--allow-hue` raises the hue cap.
- `RandAugment(num_ops=2, magnitude=5)`.
- Per-breed override policy: coat-colour breeds skip colour jitter when `--breed-aug` is used:
  `gir, kankrej, red_sindhi, tharparkar, sahiwal, kapila, red_kandhari, shweta_kapila, konkan_kapila, gir_cattle`.

Without any enabled transform, the train transform is exactly the eval transform.

### 6.4 Loss, sampling, and auxiliary objectives

Source: `src/train.py`, `src/model.py`, `src/data_pipeline.py`.

**Sampler**

- `WeightedRandomSampler` with effective-number weights:
  `E_n = (1 - beta^n)/(1 - beta)`, `weight = 1/E_n`, `beta=0.99`.
- This is the single long-tail mechanism by default.

**Multi-task loss**

```
L = w_bin * CE_binary
  + w_cat * CE_cattle * cattle_mask
  + w_buf * CE_buffalo * buffalo_mask
  + (optional) lambda_supcon * SupCon(projection)
  + (optional) w_trait * mean_trait_CE
  + (optional) KD term
```

Default weights:

- Initial: `0.15 / 0.50 / 0.35`.
- After raw binary validation accuracy ≥ `0.97` in Phase 2: `0.10 / 0.52 / 0.38` from the next epoch.
- Label smoothing: `0.1` in `src/config.py`; `train_model.py` passes `0.05`; `local_train.py` does not override, so it uses `0.1`.
- Binary CE is species-balanced per batch (`BALANCE_BINARY_HEAD=True`) so cattle/buffalo contribute equal mass regardless of sampler mix.
- `soft_ce` supports soft CutMix/MixUp targets and optional logit adjustment.
- Logit adjustment is **off** by default and is optional (`--logit-adjust`); enabling it on top of the sampler double-corrects.

**SupCon**

- Uses the 128-d projection embedding, temperature `0.1`, weight `0.3` in config / `0.2` when `train_model.py` defaults.
- Skipped on mixed batches (soft labels).
- Optional `--hard-pairs` JSON up-weights confused-pair negatives by `2.0`.

**Traits**

- Training-only; enabled with `--trait-weight > 0` (default `0.15`).
- `data/breed_traits.json` fields: `hump, horn, coat, ear, dewlap, face, size`.
- Per-field cross-entropy, masked to classes with a `-1` target; averaged over fields that have at least one valid sample.
- Trait heads are attached as a submodule, so they are trained and saved but excluded from export.
- Trait heads are frozen in Phase 1 and trained in Phase 2.

**Knowledge distillation**

- Optional `--teacher <pt>` + `--teacher-backbone` + `--teacher-attention`.
- `L = (1-alpha)*masked_hard_CE + alpha*T^2*masked_KL(teacher||student)` across all three heads.
- `alpha=0.7`, `T=4.0`; teacher is frozen/eval and sees the same augmented batch.

**Cosine heads**

- Optional `--cosine-head`; scale `30.0`, margin `0.3`, margin ramps over the first 10 Phase-2 epochs.
- Margin applied to target logits only during training; exported graph is plain scaled cosine.

### 6.5 Phases, optimizer, EMA, scheduling

Source: `src/train.py`.

| Phase | Frozen | LR | Epochs | Scheduler | Eval cadence |
|---|---|---|---|---|---|
| 1 — all-heads warmup | Backbone + attention frozen; all heads train | `3e-3` | `8` | none | every epoch |
| 2 — multi-task fine-tune | nothing | `1e-4` base | `80` | linear warmup (`5` default) → cosine | every `3` epochs |
| 3 — QAT (opt-in) | nothing | `5e-6` | `10` | none | every `2` epochs |

Phase 2 parameter groups:

- backbone: `lr * 0.1`
- attention: `lr * 0.5`
- binary/cattle/buffalo heads: `lr`
- projection/trait heads: `lr`

Other behavior:

- AdamW, weight decay `1e-2`, gradient clipping `max_norm=1.0`.
- Gradient accumulation default `4` (`train_model.py` defaults to `2`).
- AMP + GradScaler on CUDA.
- `cudnn.benchmark=True`, TF32, channels-last, `pin_memory`, non-blocking transfers.
- `torch.compile` in Phase 2 on non-Windows, non-smoke runs, unless `--no-compile`.
- Auto batch scaling on CUDA:
  - `<6 GB → 16`, `<10 GB → 32`, `<16 GB → 64`, else `128`.
  - `grad_accum = max(1, original_batch*original_grad_accum / new_batch)`.
- EMA in Phase 2:
  - Deep copy of the model, updated once per **optimizer step**.
  - Decay `min(0.9995, (1+t)/(10+t))`.
  - Parameters **and floating buffers** (BN running stats) are EMA'd; integer buffers are copied.
- Validation runs on both raw and EMA weights; the better `blended_score` is checkpointed.
- Mixing (when enabled) is disabled for the last `15%` of Phase 2.
- Phase 3 starts from the best Phase-2 checkpoint, fuses Conv-BN, prepares per-tensor QAT observers, trains, converts to INT8, and saves `<backbone>_quantized_<runid>.pt`. **Export still uses Phase 2 by default** because QAT can degrade accuracy.

### 6.6 Metrics and checkpoint selection

Source: `src/metrics.py` (used by `src.train`), `src/evaluate.py`.

`evaluate_epoch` returns:

- `binary_acc`, `binary_f1`
- `cattle_acc`, `buffalo_acc`
- `cattle_macro_f1`, `buffalo_macro_f1`
- `cattle_balanced_acc`, `buffalo_balanced_acc`
- `combined_top1` (hard-routed legacy)
- `combined_top1_soft`, `combined_top3`, `combined_top5` (species-aware soft routing)
- `combined_top3_oracle`, `combined_top5_oracle` (true species head; legacy)
- `balanced_score = 0.5*(cattle_macro_f1 + buffalo_macro_f1)`
- `blended_score = 0.5*macro + 0.5*combined_top1_soft` ← checkpoint metric
- `acc_fewshot`, `acc_mediumshot`, `acc_manyshot` (train images per breed: `<30`, `30–100`, `>100`)
- `val_min_per_breed`, `val_median_per_breed`
- `pred_hist_entropy` (normalized predicted-class histogram entropy)

Checkpoint metadata saved:

```python
{
  "phase", "epoch", "val_top1", "best_metric", "source": "raw"|"ema",
  "metrics": {...}, "state_dict": {...}
}
```

### 6.7 Post-training exports and OOD calibration

- `src.train` always attempts a **portable** export unless `--no-export`.
- `src.train` auto-calibrates OOD on the validation loader unless `--no-ood-calibrate`; it saves `<checkpoint-stem>.json`, the run id JSON, and embeds thresholds into the portable bundle.
- `local_train.py` additionally exports ONNX FP32, ONNX INT8, and float16.
- `train_model.py` exports formats from `--export-formats` (default `portable,onnx,float16`).
- `scripts/calibrate_ood.py` recalibrates from `data/splits/val.csv` and updates the global fallback.
- `--set-active` writes the checkpoint stem into `outputs/app_config.json` (and selected OOD profile in the `local_train` path).

### 6.8 Execution logs

- `sitecustomize.py` auto-initializes the run logger for non-trivial Python invocations when the project root is on `sys.path`.
- Each execution gets `logs/<exec_id>/` with `manifest.json`, `config.json`, `run.log`, and category JSONL files such as `events.jsonl`, `training.jsonl`, `data.jsonl`, `test.jsonl`, `export.jsonl`, `actions.jsonl`.
- Disable with `RUN_LOG_DISABLE=1` or `LOG_TO_FILE=False`.
- `outputs/logs/` holds server session logs and `presenter_config.json` for the standalone tester.

### 6.9 Training data helper scripts

| Script | Purpose |
|---|---|
| `local_train.py --dataset-mode atharvadarpude` (default) | Kaggle `atharvadarpude/indian-cattle-image-dataset` + `atharvadarpude/indian-buffalo-dataset`. |
| `--dataset-mode algsoch` | Kaggle `algsoch/breed-cattle-buffalo`. |
| `--dataset-mode both` | All three. |
| `data_collection.py` | Independent requests-based collector for Kaggle/NBAGR/web images. |
| `scripts/audit_data.py` | Duplicate/near-duplicate/name audit. |
| `scripts/diagnose_model.py` | Few/medium/many-shot recall + histograms. |
| `scripts/mine_confusions.py` | Confusion pairs for `--hard-pairs`. |
| `scripts/populate_breed_traits.py` | Generates trait values via Gemini API when a key is supplied. |
| `scripts/make_trait_template.py` | Writes an empty trait template. |
| `scripts/onnx_parity_10.py` | 10-image ONNX vs PyTorch logits/soft-routing check. |
| `scripts/prepare_app_assets.py` | Builds app label/index assets from class maps. |
| `create_training_zip.py`, `create_test_eval_zip.py`, `create_colab_*` | Packaging helpers. |

---

## 7. Model testing and evaluation

### 7.1 `test_model.py` — standalone tester (ACTIVE but independent)

Source: `test_model.py` (~2,970 lines).

- Starts a minimal `http.server` on **port 8501**.
- Serves one embedded HTML page generated by `build_html(mode)` and a JSON API.
- `--dev` (default) / `--present`; `--present` strips raw arrays and timing fields from predictions, disables ODT export, and changes the embedded UI/presenter config.
- Auto-opens the browser unless `--no-browser`.
- Uses its own `ModelManager`, which:
  - discovers `.pt` / `.onnx` under `outputs/` (skips `logs`/`metrics`),
  - names portable bundles `portable/<dir>`,
  - selects class maps from the **newest portable export** by `model_info.json.exported_at` (falling back to filesystem mtime), then `data/splits/`,
  - caches models lazily,
  - infers head dimensions and cosine heads,
  - filters QAT/observer and shape-mismatched tensors,
  - caches OOD detectors per model and per custom profile,
  - rejects OOD calibration for ONNX (needs a PyTorch `.pt`).

**Endpoints**

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Embedded tester UI |
| GET | `/api/models` | Discovered models + device |
| GET | `/api/dev-info` | Active cattle/buffalo class maps + union of known class-map names for the expert portal |
| GET | `/api/model-spec?model=` | Params/components/ONNX I/O |
| GET | `/api/presenter-config` | Presenter settings |
| GET | `/api/ood-profiles` | Available OOD profiles |
| GET | `/api/ood-data-dirs` | Suggested calibration folders |
| GET | `/api/ood-info?model=` | Resolved detector info |
| GET | `/api/dataset-info` | Inventory/splits/class-map sizes |
| GET | `/api/logs` | Current session log text |
| POST | `/api/predict` | Predict (`image`, `model`, `ood_profile`, `filename`) |
| POST | `/api/upload-model` | Save `.pt`/`.onnx` into `outputs/checkpoints/uploaded_models/` |
| POST | `/api/upload-ood` | Save a JSON OOD profile |
| POST | `/api/export-odt` | ODT report; 403 in presenter mode |
| POST | `/api/image-info` | Image/EXIF metadata |
| POST | `/api/presenter-config` | Save presenter settings |
| POST | `/api/recalibrate-ood` and `/api/ood/recalibrate` | Validation-set or folder-based OOD recalibration |

**Prediction differences vs Flask**

- Always returns a fixed top-5 (no shared-config filtering).
- OOD detector is resolved per model / profile (`auto` resolves to the model's JSON).
- Returns `ood_profile`, `ood_thresholds`, `_raw_binary_probs`, `_raw_breed_probs_top10`.
- In `--present`, raw arrays and timings are removed; export is disabled.
- No shared config DB, no Socket.IO, no React assets, no active-model switch.

**Presenter config** (`outputs/logs/presenter_config.json`) controls titles, confidence display modes (`scaled`, `clamped`, `badge_only`, `hide`, `percentage`), floor, hide toggles, top-5 count, and batch visibility. It is separate from the React app's `outputs/app_config.json`.

### 7.2 `src/evaluate.py`

- Loads a checkpoint via `src.export._load_model`.
- Runs `evaluate_epoch` on validation and test, logs metrics, writes:
  - `*_cattle_cm.csv` / `*_buffalo_cm.csv`
  - `*_cattle_cm.png` / `*_buffalo_cm.png`
  - `*_metrics.json` (val/test + `test_binary_f1_ok`)
- Confusion matrices are sized from the surviving class maps.

### 7.3 `src/parity_check.py`

- Compares PyTorch FP32, ONNX FP32, ONNX INT8, and TFLite INT8 on the same images.
- Accepts `[0,1]` inputs and applies the correct convention per runtime.
- Accuracy mode needs `data/splits`; `--synthetic N` runs random tensors without data.
- Acceptance guidance in the module: an INT8 artifact should stay within **1.0 point** of fp32 `combined_top1`.

### 7.4 `src/verify.py`

- Loads both backbone weight files.
- Runs a forward pass and checks `binary=(1,2)`, `cattle=(1, NUM_CATTLE_BREEDS)`, `buffalo=(1, NUM_BUFFALO_BREEDS)`, `features=(1,1280)`.
- Note: it uses configured counts (50/18), not dynamic class maps.

### 7.5 Manual CPU/smoke test suites

| File | What it checks |
|---|---|
| `scripts/test_master_cpu.py` | Transform defaults, pad behavior, class-count validation, mixing, contrastive, trait integration, logger, checkpoint naming. |
| `scripts/test_fixes_cpu.py` | Augmentation defaults, same-species mixing, class counts, SupCon, logit adjustment, run-id/path behavior. |
| `breed-classifier-react/src/utils/__tests__/*` | **42** Node tests: optimizer layout, crop math, quality metrics (incl. pitch-black no-subject), session store, trait comparison, five-level confidence, CLAHE, 68-breed coverage. |


Run frontend tests with `npm test` inside `breed-classifier-react/`.

---

## 8. Serving and inference runtime (Flask)

### 8.1 Launcher

`run.py`:

- Starts `server/app.py` on port 5000 unless already in use.
- Starts `npm run dev -- --port 5173` unless already in use.
- Waits for `/api/health` and `/`.
- Opens `http://localhost:5173/`; `--open-dev` also opens `/dev`.
- `--backend-only`, `--frontend-only`, `--no-browser`.
- Graceful SIGINT/SIGTERM shutdown.
- Uses `.venv/bin/python` if present.

`server/app.py`:

- `CORS(app, resources={r"/api/*": {"origins": "*"}})`.
- `SocketIO(..., cors_allowed_origins="*", async_mode="threading")`.
- Registers `/ws` namespace.
- Binds `0.0.0.0`.

### 8.2 `ModelManager` (Flask)

Source: `server/model_manager.py`.

**Discovery**

- Walks `outputs/`, skipping paths containing `logs` or `metrics`.
- Finds `.pt` and `.onnx`.
- Portable `model.pt` → display name `portable/<bundle-dir>`.
- Other files → filename without extension; ONNX gets `" (ONNX)"`.
- Files in top-level `v*`/`202*` subfolders get a `[subdir]` prefix.
- Backbone guessed from the filename (`lite2` substring else `lite4` if present else `lite2`).

**Class maps**

- Sorts `outputs/export/portable/*` newest-first by `model_info.json.exported_at`, falling back to filesystem `mtime` (same rule as `test_model.py`).
- On model load (and on active-model switch) inspects the checkpoint's `cattle_head`/`buffalo_head`/ONNX output widths and binds the portable class map whose lengths match those head outputs.
- Falls back to `data/splits/*_classes.json` only if no portable bundle matches, and logs a warning when the fallback counts differ.
- Exposes `manager.class_counts(name)` for `/api/model-spec` and `/api/dev-info`.

**Active model**

- `outputs/app_config.json` → `active_model` if it exists in the discovered set.
- Otherwise first discovered name containing `"portable"`.
- Otherwise first discovered model.
- `set_active_model()` writes the config, rebinds the count-matched class maps and (when `ood.selected_profile == "auto"`) rebinds the OOD detector, then emits `model_switched` and `ood_status_changed` on `/ws`.


**Loading**

- ONNX: `onnxruntime.InferenceSession` with CUDA provider when available, else CPU.
- PyTorch: `torch.load(..., weights_only=False)`; falls back to `torch.jit.load`.
- Handles module, `state_dict`, `model_state_dict`, raw dict.
- Infers dimensions/cosine heads; filters keys with `strict=False`.
- Caches the loaded model in `info["model"]`.
- Emits `model_loaded` on `/ws` with load time and parameter count.

**OOD**

- Single `manager.ood_detector`, rebound whenever the active model changes.
- Startup: if `ood.selected_profile != "auto"`, load that JSON; if `"auto"`, resolve the active model's per-model profile through `OODDetector.resolve_profile_path(active_model_name)` (not the global fallback unless no per-model file exists).
- `set_active_model()` rebinds the detector for `"auto"` and emits `ood_status_changed` on `/ws`.
- `/api/ood/select` reinitializes it (auto → active model profile).
- `/api/ood/toggle` sets it or `None` (auto → active model profile).
- `/api/ood/recalibrate` re-binds it to the recalibrated model when OOD is enabled.


### 8.3 Request flow

```text
POST /api/predict (multipart: image, model, filename)
  → request.files["image"].read()
  → emit prediction_started on /ws
  → resolve model name (empty → active model)
  → PIL open + RGB
  → Resize(260) + CenterCrop(260) + ToTensor + ImageNet normalize
  → load model lazily
  → PyTorch or ONNX forward (PyTorch also captures CBAM mask + final feature map)
  → OOD detector (if enabled)
  → softmax all heads
  → combined = [P(cattle)*P(cattle_breed), P(buffalo)*P(buffalo_breed)]
  → top-k + display filtering from shared config
  → confidence_level (1..5) from top combined probability
  → CBAM spatial mask + linear class-activation heatmap → base64 PNG data URLs
  → build result + raw diagnostics
  → log prediction to execution log
  → emit prediction_progress / prediction_complete on /ws
  → JSON response
```

Progress events are emitted on namespace `/ws`; the React client now connects to the same namespace.

### 8.4 Prediction response contract

Actual fields produced by `server/services/inference.py`:

| Field | Type / range | Meaning |
|---|---|---|
| `species` / `predicted_species` | `"Cattle"` / `"Buffalo"` / `"Unknown"` | Species of the top combined breed, or Unknown when OOD. |
| `species_confidence` | 0–100 | Binary probability of the winning species; 0 when OOD. |
| `species_prob` | 0–1 | Max binary softmax probability, regardless of OOD. |
| `top_breed` / `predicted_breed` | string | Display name with underscores replaced and title-cased. |
| `confidence` | 0–1 | Top breed combined probability; 0 when OOD. |
| `confidence_level` | 1–5 | Joint confidence rubric level (1 = OOD/very low, 5 = very high). |
| `confidence_band` | string | `very_low`, `low`, `moderate`, `high`, `very_high`. |
| `confidence_action` | string | Translation key for the prescribed action. |
| `top_breed_confidence` | 0–100 | Same as `confidence` in percent. |
| `top5_breeds` | `[{breed, confidence}]` | Percent values; filtered by config. OOD → one placeholder `"Not a recognized animal"` with 0. |
| `top_k` | `[{breed, prob}]` | 0–1 values; same list as `top5_breeds`. |
| `model_used` / `model_name` | string | Selected model name. |
| `inference_time_ms` | float | Forward-pass time. |
| `total_time_ms` | float | Decode + preprocess + inference + postprocess. |
| `routing` | `"soft"` | Fixed string. |
| `ood` / `is_ood` | bool | OOD decision. |
| `ood_score` | float or null | Energy value when OOD; null otherwise. |
| `ood_details` | object | `energy`, `energy_threshold`, `msp`, `msp_threshold`, `binary_confidence_floor`, `ood_by_energy`, `ood_by_msp`, `ood_by_binary_floor`, `source_file`, `profile_name`. |
| `attention_map_b64` | string or null | RGBA PNG data URL of the winning-head class-activation heatmap (PyTorch only). |
| `spatial_attention_b64` | string or null | RGBA PNG data URL of the CBAM 7×7 spatial attention mask. |
| `attention_meta` | object | `source`, `feature_grid`, `cbam_grid`, `winning_class_index`, `winning_species`, `dominant_quadrant`, `quadrant_scores`, `peak_px`. `null`/unavailable for ONNX. |
| `_raw_binary_probs` | `[2 floats]` | Raw binary softmax. |
| `_raw_breed_probs_top10` | list | Top-10 raw combined classes with global index/species/breed/prob. |
| `filename` | string | Added by the route from the form. |

**Display filtering**

Read from shared config at prediction time:

```
fetch_k = min(max(max_breeds, 5), total_classes)
candidates = top-k of combined
filtered = candidates with confidence >= min_breed_confidence_pct
if len(filtered) < min_breeds: filtered = candidates[:min_breeds]
top5 = filtered[:max_breeds]
```

`max_breeds` and `top_k` are synchronized by the config DB. The React result card applies the same min/max filter again.

### 8.5 OOD runtime behavior

- If OOD is disabled or `manager.ood_detector` is `None`, no OOD check runs.
- If OOD is triggered:
  - breed classification is **not** withheld at the model level — inference still runs;
  - the response replaces breed output with `Unknown` + placeholder;
  - the React card shows a safety abstention and does not display the ranked list.
- The React tier logic is computed client-side:
  - `tier 3` if `is_ood`
  - `tier 1` if `topProb >= 0.72` and `top1-top2 margin >= 0.12`
  - `tier 2` otherwise

### 8.6 Shared config DB

Source: `server/shared_config.py`, path `outputs/app_config.json`.

- Defaults are deep-merged over loaded JSON, so missing new keys are filled.
- Writes are atomic (`*.tmp` + `os.replace`) and guarded by a lock.
- Dot notation: `db.get("ood.enabled")`, `db.set("display.top_k", 5)`.
- `display.max_breeds` and `display.top_k` are kept equal.
- `config_changed` is emitted on `/ws` after updates/reset.
- `version`, `active_model`, `ood`, `display`, `dataset`, `inference` sections.

Default config:

```json
{
  "version": 1,
  "active_model": null,
  "ood": {
    "enabled": true,
    "selected_profile": "auto",
    "energy_threshold": -5.0,
    "msp_threshold": 0.5,
    "temperature": 1.0
  },
  "display": {
    "confidence_mode": "percentage",
    "confidence_floor_pct": 80.0,
    "top_k": 3,
    "min_breeds": 1,
    "max_breeds": 5,
    "min_breed_confidence_pct": 5.0,
    "show_species_confidence": true,
    "show_species_badge": true,
    "show_escalation_warning": true,
    "show_technical_details": false,
    "theme": "dark"
  },
  "dataset": {
    "min_images_per_breed": 15,
    "max_images_per_breed": 500
  },
  "inference": {
    "device_override": null,
    "batch_size": 16
  }
}
```

**Which keys actually affect the active runtime**

| Key | Effect |
|---|---|
| `active_model` | Selects the model in `ModelManager`. |
| `ood.enabled` | Read on every prediction; toggles detection. |
| `ood.selected_profile` | Used at startup, profile selection, and toggle. |
| `ood.energy_threshold`, `ood.msp_threshold`, `ood.temperature` | Used by the active detector; threshold endpoint mutates the live detector too. |
| `display.min_breed_confidence_pct` | Server filtering + client filtering. |
| `display.min_breeds` | Server guarantees at least this many candidates. |
| `display.max_breeds` / `display.top_k` | Server fetch/slice + client slice. |
| `display.show_technical_details` | Initial expanded state of the advanced diagnostics. |
| `display.confidence_mode` | Live: percentage / scaled / clamped / badge_only / hide. |
| `display.confidence_floor_pct` | Live floor for scaled/clamped modes. |
| `display.show_species_confidence` | Live species-probability pill. |
| `display.show_species_badge` | Live species badge toggle. |
| `display.show_escalation_warning` | Live: hides Level 1–2 vet escalation warning/button when false. |
| `display.theme` | **STORED-ONLY**; the React theme is browser-local (`breed-classifier-theme`). |
| `dataset.min_images_per_breed` | **STORED-ONLY**; training reads `src/config.py` / CLI, not this file. |
| `dataset.max_images_per_breed` | **STORED-ONLY**. |
| `inference.device_override` | **STORED-ONLY**; device is auto-selected. |
| `inference.batch_size` | **STORED-ONLY**; prediction is one image per request. |

### 8.7 REST endpoints (Flask, 29 route rules)

| Method | Path | Behavior |
|---|---|---|
| GET | `/api/health` | status, device, model count, uptime |
| GET | `/api/models` | model list + device + active model |
| GET | `/api/model-spec?model=` | parameter/component/ONNX I/O spec |
| GET | `/api/dev-info` | active cattle/buffalo class maps + known-breed union for expert review |
| GET | `/api/logs` | current server session log text |
| GET | `/api/config` | full shared config |
| GET | `/api/config/<path:key>` | one dot-notation subtree |
| PATCH/POST | `/api/config` | update by `{key,value}` or object patch |
| POST | `/api/config/reset` | reset to defaults |
| GET | `/api/presenter-config` | legacy presenter config |
| POST | `/api/presenter-config` | save legacy presenter config |
| GET | `/api/ood/profiles` | global + `outputs/ood/*.json` profiles |
| GET | `/api/ood/active` | OOD config + resolved detector status |
| GET | `/api/ood/data-dirs` | suggested calibration folders |
| POST | `/api/ood/select` | select `"auto"` or a profile filename |
| POST | `/api/ood/toggle` | enable/disable |
| POST | `/api/ood/thresholds` | update energy/MSP/temperature live |
| POST | `/api/ood/upload` | validate + save JSON profile |
| POST | `/api/ood/recalibrate` | folder/validation recalibration |
| POST | `/api/predict` | multipart prediction |
| POST | `/api/select-model` | set active model |
| POST | `/api/upload-model` | save `.pt`/`.onnx` under `uploaded_models` |
| POST | `/api/export-odt` | ODT report bytes (supports optional BPA registration block) |
| POST | `/api/image-info` | image/EXIF metadata |
| GET | `/api/escalations?status=all\|pending\|verified` | Expert queue + summary counts |
| POST | `/api/escalations` | JSON or multipart escalation; emits `escalation_created` on `/ws` |
| GET | `/api/escalations/<id>` | Single escalation |
| POST | `/api/escalations/<id>/resolve` | Vet verdict + optional active-learning promotion |
| POST | `/api/bpa/qr` | QR module matrix for the BPA canvas card |

The model-spec endpoint computes class counts and head widths from the loaded checkpoint/ONNX outputs and the bound class maps; it no longer returns hardcoded 57/18 architecture strings.

### 8.8 WebSocket events

Server namespace: `/ws`.

Events emitted:

- `server_status` on connect and `request_status` (models count, device, uptime, active model, full config)
- `prediction_started`
- `prediction_progress` (`preprocessing`, `inference`, `postprocessing`)
- `prediction_complete`
- `prediction_error`
- `model_loaded`
- `model_switched`
- `config_changed`
- `ood_status_changed`
- `escalation_created`
- `escalation_resolved`

React client:

- `useSocket()` constructs `io('/ws', { path: '/socket.io' })` — the same namespace the server emits on.
- It listens for `server_status`, `prediction_*`, `model_loaded`, `model_switched`, `ood_status_changed`, and escalation events, and emits `request_status`.
- `useSharedConfig()` listens for `config_changed` and `server_status` and updates the shared config live.

**Namespace:** server and client both use `/ws`. Realtime status, prediction progress, model switches, OOD changes, config changes and escalation events are delivered without a page reload (verified with the Flask-SocketIO test client).


### 8.9 ODT export

Source: `server/services/odt_export.py`, `test_model.py::generate_odt_report`.

- Requires `odfpy`.
- `mode="single"` wraps one dict; any other mode expects a list.
- Per result it writes: filename, species + species confidence, predicted breed + confidence, and a Top-5 table.
- `mode="batch"` adds a batch summary table plus total/cattle/buffalo and average top-breed confidence.
- The React herd drawer sends `mode="batch"` with reversed session entries.
- The Dev config panel’s “Export ODT Report” now reads `breed-herd-session-v1` and exports the real session entries (or a summary shell only when the session is empty).
- The single-prediction React card has **PNG share/download only**, not ODT.
- When a result includes a `bpa` object, the ODT gains a Bharat Pashudhan registration block.


---

## 9. React user-facing application

### 9.1 Routes and active files

`src/main.jsx` → `I18nProvider` → `router.jsx`.

| Route | Component |
|---|---|
| `/` | `pages/UserPage.jsx` |
| `/dev` | `pages/DevPage.jsx` |
| `/expert` | `pages/ExpertPage.jsx` |

`App.jsx`, `SingleImage.jsx`, `Batch.jsx`, `ResultsPanel.jsx`, `PredictionProgress.jsx`, `ModelSelect.jsx`, and `DevTools.jsx` exist but are **not imported by the active router** (see §9.16).

`?mode=camera` is read by `UserPage` and starts in the camera tab. This is the kiosk/QR deep link; DevPage links it as “Camera Test”.

### 9.2 Page structure

Header:

- Brand + title/subtitle (i18n).
- Connection status badge.
- **Advanced** button.
- Active model chip from `config.active_model` or socket status; fallback “Default Model”.
- Herd session button with count.
- Language toggle (EN / தமிழ்).
- Theme dropdown (5 themes).
- “Developer Settings” link to `/dev`.

Main:

- Tabs: Upload Photo / Live Camera.
- Upload card or camera.
- Prediction progress line if a `predictionEvent` progress arrives.
- Error banner.
- Result card below when a prediction exists.

Footer:

- Localized footer text.

### 9.3 Upload UX

- Drag/drop, click, keyboard.
- Preview is the **original** upload.
- Actions: Adjust framing, Remove, Choose another.
- Primary classify button: “Identify Breed” or “Identify Framed Breed”.
- While optimizing: button spinner “Optimizing to 260×260…”.
- While analyzing: “Analyzing with Vision Model…”.
- The prediction is cleared whenever the optimized input file changes.
- The file name shown is the original name, not `zero_crop_260_*`.
- Format support copy: JPEG/PNG/WEBP.

### 9.4 Camera UX

Detailed in §5.4. User-visible features:

- Full-frame live preview.
- Quality pill with score/status.
- Auto-capture countdown after 3 ready checks.
- Burst ×3 best-sharpness capture.
- Torch and camera flip when supported.
- Review screen with optimized 260×260 preview, telemetry, Pad/Stretch toggle, quality badge, metadata (original, model input, scaled, padding, payload, time).
- Retake / Adjust framing / Use This Photo.
- Manual framing via the shared cropper.
- Toggleable SVG anatomical silhouette (Cattle / Buffalo / Off), flip direction, animated Hump / Head & Horns / Dewlap landmark pills.
- Three live status pills: Lighting (Good / Too Dark / Overexposed), Animal in Frame (Ready / Too Far / Edge Clipped), Sharpness (Optimal / Blurry).
- Opt-in LAB CLAHE shadow normalization in the camera review screen (and Advanced panel).
- On use, the camera closes and the upload tab shows the chosen image.

### 9.5 Advanced panel

Opened from the header. It is a modal drawer with:

- **View model input ⇄ View original** toggle.
- 260×260 preview with badges.
- Fit modes:
  - Stretch 260×260
  - Pad · ImageNet mean
  - Pad · Ambient blur
  - Pad · Black
- Opt-in **Normalize Harsh Shadows (CLAHE)** toggle (default off preserves inference parity).
- **Detailed prediction diagnostics** when a result exists: five-level action/escalation warning, X-Ray, Top-5, trait comparison, quality subscores, OOD triggers, technical details, BPA registration and Vet escalation actions.
- Metadata grid: original dimensions, model input, scaled inner size, padding, payload, optimization time.
- File name.
- Optimizer errors.
- **Advanced batch tester**.

Changing fit mode re-optimizes the current source and invalidates the displayed prediction (via the `optimizer.result.file` effect).

### 9.6 Result card

Source: `components/ResultCard.jsx`.

**Tier banner**

| Tier | Condition | UI |
|---|---|---|
| 1 High confidence | not OOD; top probability ≥ 0.72; margin ≥ 0.12 | green, positive message |
| 2 Look-alike | not OOD; otherwise | amber, trait comparison open by default, retake action |
| 3 Safety abstention | OOD | red, exact trigger list, ranked list withheld, retake |

**Five-level confidence rubric**

| Level | Condition | Badge | Prescribed action |
|---|---|---|---|
| 5 | `C ≥ 0.85`, not OOD | Dark green | Auto-populated for BPA registration (single-tap confirm) |
| 4 | `0.75 ≤ C < 0.85`, not OOD | Light green | High reliability — review Top-2 alternative |
| 3 | `0.60 ≤ C < 0.75`, not OOD | Amber | Compare Top-3 morphological traits |
| 2 | `0.40 ≤ C < 0.60`, not OOD | Orange | Block auto-registration; recapture or queue vet review |
| 1 | `C < 0.40` or OOD | Red | Reject/OOD; route to vet queue or retake |

The compact main card shows the level pill (only when `display.show_escalation_warning=true`) and prediction result. The prescribed action, escalation warning/button, BPA registration, X-Ray, trait comparison, quality subscores, OOD trigger list and technical diagnostics live in the header **Advanced** drawer (`AdvancedResultDetails.jsx`).

**Compact main card**

- Species pill (or “OUT-OF-DISTRIBUTION”), confidence pill, optional level pill, species confidence and quality chip.
- Predicted breed title (or OOD title).
- Confidence bar with 0/50/100 scale (hidden in `hide`/`badge_only` display modes).
- Top 3 breed candidates only.
- Buttons: **Advanced details**, Download/Share Result Card (PNG), Retake with Guided Coach, Classify Another Photo.
- OOD safety advisory remains visible inline because it is essential.

**Advanced result details (header Advanced drawer)**

- Full five-level banner with prescribed action and escalation trigger warning.
- **Escalate to Vet Expert** button (hidden when `display.show_escalation_warning=false`).
- BPA registration for Levels 3–5.
- AI Vision X-Ray (interactive CBAM + class-activation overlay).
- Full Top-5 ranked list, look-alike/trait comparison, quality subscores, OOD trigger diagnostics and technical details.

`display.show_escalation_warning=false` hides the level/escalation warning in both the compact card and Advanced details, leaving only prediction results.

**AI Vision X-Ray**

- Interactive toggle + 0–100% opacity slider over the 260×260 model input inside Advanced details.
- Overlay uses `attention_map_b64` (winning-head class activation) and optionally `spatial_attention_b64` (CBAM mask).
- Legend from *Low focus (background suppressed)* to *High focus (horns / hump / dewlap)*; peak quadrant and feature/CBAM grid sizes are shown.
- ONNX artifacts display “attention map unavailable” because internal hooks are not available.

**BPA registration**

- Levels 3–5 show **Register in BPA (Bharat Pashudhan)** inside Advanced details.
- Modal generates a 12-digit numeric ear-tag UID (custom entry allowed), auto-fills species/breed/confidence/Top-3/FLW/timestamp/GPS, captures owner/village/age/sex/lactation/vaccination, saves to the herd session, and downloads a PNG health card with a scannable QR (matrix rendered on canvas) or a printable ODT.

### 9.7 “Why this breed?” trait comparison

Source: `components/TraitComparison.jsx`, `data/breedTraits.js`.

- **Presentation-only lookup**, not model explainability.
- **68 configured breeds** × 8 fields: `hump`, `horn`, `coat`, `ear`, `dewlap`, `face`, `size`, `origin` (plus `keyDistinguisher`); the table also includes a few legacy alias/extra entries.
- `keyDistinguisher` is shown for the top breed.
- The comparison table highlights rows whose values differ among the top 3.
- Breed names are normalized to snake_case; alias `bhadwari → bhadawari`.
- Missing breed → “No curated trait note for this breed yet.”
- Disclaimer tells the user to confirm critical decisions with a breed expert.

### 9.8 Quality badge

Source: `components/QualityBadge.jsx`.

- Full panel: overall score ring, sharpness/lighting/framing bars, status icon, hint.
- Compact pill: icon + score, used in the camera guidance and result header.
- Status meta:

| Status | Icon | Tone |
|---|---|---|
| `READY` | ✓ | green |
| `TOO_BLURRY` | ◌ | amber |
| `TOO_DARK` | ☾ | amber |
| `OVEREXPOSED` | ☀ | amber |
| `NO_SUBJECT_OR_TOO_FAR` | ⊕ | amber |
| `PARTIALLY_OUT_OF_FRAME` | ⌗ | amber |

### 9.9 Herd session

Source: `components/HerdSessionDrawer.jsx`, `utils/sessionStore.js`.

- Every **confirmed** prediction (after a successful `/api/predict`) appends an entry.
- Entries are newest-first.
- Stored in `localStorage` key `breed-herd-session-v1`.
- Limit: **60** entries.
- Thumbnail: 160×160 JPEG quality 0.7, generated from the current preview.
- If quota is exceeded, it retries with half the entries and no thumbnails, then gives up silently.
- Entry fields: `id`, `timestamp`, `filename`, `species`, `speciesConfidence`, `breed`, `confidence` (0–1), `isOod`, `oodScore`, `model`, `top5`, `qualityScore`, `qualityStatus`, `thumbnail`.
- OOD entries carry zero confidence and are **excluded** from the average confidence.
- Average quality only includes entries with a non-null score.
- Stats: total, cattle, buffalo, flagged/OOD, average confidence, average quality.
- Exports:
  - **CSV** (client-side): `timestamp, filename, species, species_confidence, breed, confidence, is_ood, ood_score, quality_score, model`.
  - **JSON** (client-side): `{exportedAt, count, animals}`.
  - **ODT** (server): session entries reversed to chronological order, mapped to the server result schema, `mode="batch"`.
- Clear session asks for confirmation.
- Session data never leaves the browser except when the user exports ODT (which posts the result payload to the server).

### 9.10 Shareable PNG result card

Source: `utils/cropAndQuality.js::renderShareCard`, `shareOrDownloadCanvas`.

- Canvas size **1080×1350**.
- Contains brand header, photo (cover crop), species badge, breed, confidence, quality score, top-3 bars, morphology (hump/horn/dewlap), model name, filename, timestamp.
- Uses the current preview image (original upload, or 260 camera preview).
- If `navigator.canShare({files})` works, invokes the Web Share sheet; otherwise downloads `breed_<breed>_<timestamp>.png`.
- Abort from the share sheet is treated as “cancelled”.

### 9.11 Internationalization

Source: `i18n/translations.js`, `I18nProvider.jsx`, `LanguageToggle.jsx`.

- Languages: **English (`en`)** and **Tamil (`ta`)**.
- Persisted in `localStorage` key `breed-classifier-lang`.
- `t(key, vars)` interpolates `{name}` tokens.
- Missing Tamil key falls back to English.
- Sets `<html lang="ta-IN">` or `"en"`.
- No external i18n service.

### 9.12 Themes

Source: `components/ThemeToggle.jsx`, `styles.css`.

| ID | Name |
|---|---|
| `dark` | Midnight Dark |
| `light` | Clean Light |
| `emerald` | Pasture Green |
| `sunset` | Warm Sunset |
| `cyber` | Cyber Neon |

- Applied via `data-theme` on `<html>`.
- Persisted in `localStorage` key `breed-classifier-theme`.
- Browser-local; not synced to `display.theme`.

### 9.13 Batch tester (inside Advanced panel)

Source: `components/AdvancedBatchTester.jsx`.

- Multi-select `image/*`.
- Runs sequentially, using the currently selected model and the current Advanced fit.
- For each file: optimize to 260 → POST `/api/predict`.
- Shows per-file error rows, breed, confidence, OOD badge.
- Summary: valid/total, cattle, buffalo, OOD count, average top confidence, total original vs optimized bytes.
- “Add results to herd session” appends entries with `quality=null`, `thumbnail=null`.

### 9.14 Status, error, accessibility, privacy behavior

- Backend-unreachable banner appears only in the **legacy** `App.jsx` view; the active `UserPage` shows prediction errors in its own error banner.
- The active header “Connected” badge reflects Socket.IO connection only.
- Upload dropzone is keyboard-activatable (`Enter` / `Space`).
- Modals use `role="dialog"`, `aria-modal`, labels, and Escape close (Advanced).
- Buttons have titles/aria labels in several places.
- Images are not uploaded until the user clicks Identify.
- No server-side image storage.
- Herd session is local-only.
- No analytics, tracking, service worker, or offline cache.

### 9.15 Config consumed by the user UI

- Active model chip: `active_model`.
- Ranked list: `display.min_breed_confidence_pct`, `display.max_breeds` / `top_k`.
- OOD evaluation row: `ood.enabled`.
- Diagnostics initial state: `display.show_technical_details`.
- Confidence presentation: `display.confidence_mode` (`percentage`, `scaled`, `clamped`, `badge_only`, `hide`), `display.confidence_floor_pct` (scaled/clamped floor).
- Species display: `display.show_species_confidence`, `display.show_species_badge`.
- Everything else in `display` / `dataset` / `inference` is not consumed by the active user UI.

### 9.16 Legacy / unreachable React code

| File | Why it is legacy |
|---|---|
| `src/App.jsx` | Not imported by `main.jsx`; would provide Single Image / Batch / Dev Tools tabs. |
| `components/SingleImage.jsx` | Only used by `App.jsx`; uses `pad` default and has demo-sample generation. |
| `components/Batch.jsx` | Only used by `App.jsx`; separate batch flow with ODT export. |
| `components/ResultsPanel.jsx` | Only used by `SingleImage.jsx`. |
| `components/PredictionProgress.jsx` | Only used by `SingleImage`/`Batch`; active UserPage uses a simple progress line. |
| `components/ModelSelect.jsx` | Only used by `SingleImage`/`Batch`; active user page has no model picker. |
| `components/DevTools.jsx` | Only used by `App.jsx`; contains model spec, class maps, image metadata, presenter config, session logs. |

Do not describe these as current user features. The APIs they call (`/api/model-spec`, `/api/dev-info`, `/api/image-info`, `/api/presenter-config`) still exist.

---

## 10. Developer control plane (`/dev`)

Source: `pages/DevPage.jsx`, `components/DevModelPanel.jsx`, `DevOODPanel.jsx`, `DevConfigPanel.jsx`.

Header:

- Link back to `/`.
- Title “Developer Control Plane”.
- Connection dot.
- Device / model count / uptime pills.
- “Camera Test” link to `/?mode=camera`.
- Theme toggle.
- No authentication.

### 10.1 Model Checkpoints

- Lists models from `GET /api/models`.
- Shows active model badge, backbone, size.
- **Inspect** loads `/api/model-spec`:
  - checkpoint name, format/type, total parameters, input resolution, weight size, path;
  - ONNX input/output nodes;
  - for PyTorch: total/trainable/frozen, per-component parameter table, `model_info.json`, hardcoded architecture summary.
- **Activate** calls `/api/select-model`, which updates the shared config.
- **Upload** accepts `.pt` / `.onnx` (client-side filter only) and posts to `/api/upload-model`.
- Uploaded files go to `outputs/checkpoints/uploaded_models/` and are discovered immediately.
- There is no delete/unload action in the UI.

### 10.2 OOD Safety Profiles

- Master enable/disable toggle.
- Profile select:
  - `auto`
  - global `ood_thresholds.json`
  - every JSON in `outputs/ood/`
- Upload a calibration JSON (validated as JSON server-side).
- Runtime detector status: initialized, profile name, source file.
- Live threshold sliders:
  - energy `-15..0`
  - MSP `0..1`
  - temperature `0.1..3.0`
- “Apply & Persist Thresholds” updates config and the active detector.
- Manual folder recalibration:
  - model select
  - folder path with datalist of suggested project folders
  - percentile
  - max images
  - runs `/api/ood/recalibrate`
  - shows samples, matched, unmatched, energy, MSP, percentile, breed coverage chips, unmatched names, saved path.
- Recalibration requires a PyTorch `.pt` model; ONNX returns an error.

### 10.3 Shared Database (config)

- Quick range presets: Top 1 Strict, Standard 1–3, Extended 2–5, All up to 10.
- Min/max candidate breeds, min confidence cutoff.
- Confidence display mode dropdown (**live**: percentage/scaled/clamped/badge_only/hide).
- Confidence floor slider (**live** for scaled/clamped modes).
- Show species confidence toggle (**live**).
- Show species badge toggle (**live**).
- Show vet escalation warning toggle (**live**; off shows only prediction results).
- Always show technical details toggle (**live**).
- Dataset min images slider (**stored only** for training).
- Raw JSON snapshot of the shared config.
- Export ODT report (empty result shell).
- Reset all settings to defaults.

### 10.4 System Logs

- `GET /api/logs` returns the entire current `outputs/logs/server_session_*.log` text.
- Refresh button; no live streaming or tail limit.

---

## 11. Flutter edge app

Source: `flutter_app/lib/`.

| File | Role |
|---|---|
| `main.dart` | Picks TF.js on web, TFLite elsewhere; creates `InferenceController`; Material 3 green theme. |
| `domain/services/i_model_service.dart` | `initialize()`, `predict(Uint8List)`, `dispose()`. |
| `domain/entities/prediction_result.dart` | `BreedScore`, `PredictionResult`. |
| `data/utils/image_preprocessor.dart` | Decode → shortest-side resize 260 → center crop → planar RGB /255. |
| `data/ml_engines/android_tflite_engine.dart` | Loads `assets/models/model.tflite` + label files; runs three outputs; softmax + soft routing; top-3. |
| `data/ml_engines/web_tfjs_engine.dart` | Loads `assets/models/model_web.json` through global `tf.loadGraphModel`; same soft routing. |
| `presentation/state/inference_controller.dart` | Busy/error/result state. |
| `presentation/screens/camera_screen.dart` | Capture/Gallery via `image_picker` (`maxWidth/maxHeight=1600`, quality 92), result card with species, breed, confidences, latency, top-3. |

Current asset state:

- `flutter_app/assets/models/labels_binary.txt`, `labels_cattle.txt`, `labels_buffalo.txt` exist.
- `model.tflite` and `model_web.json` are **not present** in the checked-in assets; run `src.export --mode tflite` to populate the TFLite artifact. No export path generates `model_web.json`.
- No OOD, quality, session, traits, share card, i18n, or themes in the Flutter app.

---

## 12. Data formats and artifacts

### 12.1 Split manifests

`data/splits/train.csv`, `val.csv`, `test.csv` columns:

```
path, species, breed, binary_label
```

- `binary_label`: 0 cattle, 1 buffalo.
- Paths are absolute in practice.

`data/splits/cattle_classes.json` / `buffalo_classes.json`:

```json
{ "breed_slug": 0, "another_breed": 1 }
```

Class index order is the head output order.

### 12.2 Training traits

`data/breed_traits.json`:

```json
{
  "fields": ["hump", "horn", "coat", "ear", "dewlap", "face", "size"],
  "breeds": {
    "cattle": { "gir": { "hump": "...", "...": "..." } },
    "buffalo": { "murrah": { "...": "..." } }
  }
}
```

Used only for auxiliary training heads. The UI uses a separate hardcoded table in `breedTraits.js`.

### 12.3 OOD profile JSON

Typical fields:

```json
{
  "energy_temperature": 1.0,
  "energy_threshold": -3.8443,
  "msp_threshold": 0.8688,
  "binary_confidence_floor": 0.6,
  "percentile": 95.0,
  "calibrated_on_samples": 1234,
  "calibrated_at": "2026-...",
  "energy_stats": {"mean": ..., "std": ..., "min": ..., "max": ...},
  "msp_stats": {"mean": ..., "std": ..., "min": ..., "max": ...},
  "model_name": "...",
  "data_dir": "...",
  "matched_images": ...,
  "unmatched_images": ...,
  "unknown_breeds": [...],
  "breed_counts": {...}
}
```

### 12.4 Portable bundle

`outputs/export/portable/<name>/`:

```
model.pt
cattle_classes.json
buffalo_classes.json
model_info.json
ood_thresholds.json      # when calibration ran
```

`model_info.json` contains counts from **config constants**, not the checkpoint's dynamic dimensions, so it can be wrong for dropped-breed models. `src.train.create_portable_export` hardcodes 57/18; `src.export.create_portable_export` uses 50/18.

### 12.5 Browser localStorage keys

| Key | Content |
|---|---|
| `breed-herd-session-v1` | Herd session entries (including optional `bpa` registration metadata) |
| `breed-classifier-lang` | `"en"` / `"ta"` |
| `breed-classifier-theme` | one of the five theme IDs |
| `breed-escalation-outbox-v1` | Offline veterinary escalation queue |
| `breed-bpa-tag-v1` | Reserved BPA tag helper key |
| `breed-dev-mode` | Legacy `App.jsx` dev toggle only |

### 12.6 Expert escalation database

`outputs/escalations.db` (SQLite, created lazily):

- `escalations` row: id, created_at, filename, image_b64, gps_lat/lng, location_label, flw_id, model_used, predicted_species/breed, confidence, confidence_level, is_ood, ood_details_json, top3_json, quality_json, status, expert_species/breed, expert_notes, reviewed_at, attention_map_b64.
- Verified + `promote_to_dataset=true` writes `data/raw/<species>/<breed>/escalated_<id>.jpg` and appends `outputs/active_learning_manifest.csv`.


---

## 13. Runbook

### 13.1 Train

```bash
# Fastest end-to-end local run
python local_train.py --quarter-data

# Balanced
python local_train.py --half-data

# Full data, then activate in the web app
python local_train.py --full-data --set-active

# Skip already-done environment/download
python local_train.py --skip-setup --skip-download

# Core engine only
python -m src.train --backbone lite2 --quarter-data --no-compile

# Smoke test
python -m src.train --smoke-test --no-compile
```

### 13.2 Export

```bash
python -m src.export --mode portable --backbone lite2
python -m src.export --mode onnx --backbone lite2
python -m src.export --mode float16 --backbone lite2
python -m src.export --mode onnx-int8 --backbone lite2
python -m src.export --mode tflite --backbone lite2
```

### 13.3 Evaluate / verify / parity

```bash
python -m src.evaluate --backbone lite2
python -m src.verify
python -m src.parity_check --checkpoint <pt> --synthetic 8
python scripts/calibrate_ood.py --backbone lite2 --percentile 95
```

### 13.4 Serve

```bash
python run.py
# User:       http://localhost:5173/
# Camera:     http://localhost:5173/?mode=camera
# Dev:        http://localhost:5173/dev
# API health: http://localhost:5000/api/health
```

### 13.5 Test the standalone tester

```bash
python test_model.py
python test_model.py --port 8502 --no-browser
python test_model.py --present
```

### 13.6 Frontend tests/build

```bash
cd breed-classifier-react
npm test
npm run build
```

### 13.7 Manual OOD recalibration

1. Train/export a PyTorch `.pt` checkpoint.
2. Put breed-named images in a folder (flat `Deoni_01.jpg` or nested `cattle/deoni/…`).
3. Open `/dev` → OOD Safety Profiles.
4. Pick the model and folder, set percentile (default 95) and max images (default 500).
5. Recalibrate. The profile is written to `outputs/ood/<checkpoint-stem>.json`; the global fallback is also updated.
6. The endpoint sets `ood.selected_profile = "auto"` and rebinds the detector to the model.

Matching rules:

- Normalizes file stem, tokens, and parent folders to snake_case.
- Applies spelling aliases (e.g. `bhadawari/bhadwari`, `hallikaru/hallikar`, `amruthamahal/amritmahal`).
- Uses a species hint from `cattle`/`buffalo` in the path when a name exists in both maps.
- Fuzzy matches with cutoff 0.88.
- Skips unreadable images and reports them; unmatched names are reported, never used.

---

## 14. Invariants an agent must preserve

1. **Model input is exactly 260×260 RGB, ImageNet-normalized.** Server, tester, and Flutter all use shortest-side resize + center crop (or equivalent client preprocessing).
2. **Class index order is authoritative from the class-map JSON.** Never enumerate breeds alphabetically at inference.
3. **Soft routing is required for parity with training/eval.** Do not replace it with a hard binary argmax.
4. **The binary head indexes species as 0=cattle, 1=buffalo** (`SPECIES_LABELS`).
5. **OOD is post-hoc and OR-based.** Changing one threshold affects the combined behavior.
6. **Phase 2 is the default export checkpoint.** QAT is a recovery path and may degrade accuracy.
7. **EMA must include BN buffers.** Removing buffer EMA silently degrades validation/export.
8. **Do not enable logit adjustment together with the effective-number sampler** unless deliberately changing the long-tail policy.
9. **Client-side 260×260 input makes the server crop an identity.** Preserve that property when changing either side.
10. **Use `(species, breed)` as the key** for counts/splits/sampling; `bargur` exists in both species.
11. **Do not trust hardcoded 57/18 architecture labels** in docs/UI/server spec strings; read checkpoint shapes and class maps.
12. **No auth exists.** Treat `/dev`, model upload, config mutation, OOD mutation, and log access as trusted-network-only.

---

## 15. Known constraints, gotchas, and drift

### 15.1 Documentation drift (verified)

| Claim in docs/README | Actual code |
|---|---|
| “57 cattle” in several places | `NUM_CATTLE_BREEDS=50`; checkpoint v26 has 50 cattle. |
| “75 breeds” in old Colab/README text | 68 total (50+18) configured. |
| Phase 2 LR `2e-4` (`docs/training-pipeline.md`) | `PHASE2_LR=1e-4`. |
| EMA `0.999` | `EMA_DECAY=0.9995`. |
| Final loss weights `0.05/0.55/0.40` | `0.10/0.52/0.38`. |
| “All augmentation off by default” | `AUG_HORIZONTAL_FLIP=True` in config; `local_train.py` defaults to the `light` preset. |
| “Smoke test uses 75 classes” | Uses the class maps from all breeds present, currently 50/18 in a complete dataset. |
| `api.js` comment “backend port 8501” | Vite proxy and active backend are **5000**; 8501 is the standalone tester. |
| `docs/architecture.md` lists separate `cbam.py` / `efficientnet_lite.py` implementations | They are compatibility re-export shims; implementation is in `src/model.py`. |
| `docs/training-pipeline.md` says `src/metrics.py` was merged into `train.py` | `src/train.py` imports `evaluate_epoch` from `src/metrics.py`; there is also an unused duplicate at the bottom of `train.py`. |

### 15.2 Class-map / checkpoint pairing — **RESOLVED 2026-10-27**

- `ModelManager` now sorts portable bundles newest-first by `model_info.json.exported_at`, falling back to filesystem `mtime` (same rule as `test_model.py`).
- On load/switch it reads `cattle_head`/`buffalo_head` output widths (or ONNX output shapes) and binds only a class map with matching lengths.
- `data/splits/*_classes.json` is only a fallback, and a count mismatch is logged.
- On the checked-in tree this correctly binds the 50-cattle/18-buffalo v26 bundle instead of the older 17-buffalo map.

### 15.3 OOD profile pairing — **RESOLVED 2026-10-27**

- Flask `"auto"` now resolves `OODDetector.resolve_profile_path(active_model_name)` at startup and inside `set_active_model()`.
- Active-model switching rebinds the detector and emits `ood_status_changed` on `/ws`.
- `/api/ood/select` and `/api/ood/toggle` also resolve auto against the active model.

### 15.4 WebSocket namespace — **RESOLVED 2026-10-27**

- Server events are on `/ws`; `useSocket()` now also connects to `/ws`.
- `server_status`, `prediction_*`, `model_switched`, `config_changed`, `ood_status_changed`, `escalation_created`, and `escalation_resolved` are delivered live.
- Verified with a Flask-SocketIO test client on namespace `/ws`.

### 15.5 Config fields that are editable but inert

Still stored-only (no active consumer): `display.theme`, `dataset.min_images_per_breed`, `dataset.max_images_per_breed`, `inference.device_override`, `inference.batch_size`.

Now live in the React result card:

- `display.confidence_mode`
- `display.confidence_floor_pct`
- `display.show_species_confidence`
- `display.show_species_badge`

### 15.6 Quality-state nuance — **RESOLVED 2026-10-27**

- Uploads and crops now score the resulting 260×260 model input with `analyzeDataUrlQuality` / `analyzeCanvasQuality`.
- `ImageUpload` and `ImageCropper` forward a quality report; `UserPage` recomputes it from the optimized data URL and clears stale quality immediately when the image changes.
- Quality remains a heuristic, not animal detection.

### 15.7 Server/route validation gaps

- Still: no auth, no CSRF, CORS `*` for `/api/*`, server binds `0.0.0.0`, no explicit upload-size limit.
- `/api/upload-model` now applies `werkzeug.utils.secure_filename` and strictly allows only `.pt` / `.onnx`.
- `/api/presenter-config` still accepts arbitrary JSON.
- `/api/ood/upload` validates JSON and uses `secure_filename`.

### 15.8 Model/class metadata strings — **RESOLVED 2026-10-27**

- `/api/model-spec` derives cattle/buffalo counts, head widths, and architecture descriptions from the loaded checkpoint/ONNX outputs and active class maps.
- `model_info.json` may still carry historical counts; the API no longer trusts it for display counts.
- `test_model.py` retains its own embedded strings; changes there are out of scope for the Flask app.

### 15.9 Training data state vs config

- `data/raw` and `data/splits` are mutable working data. A complete dataset is expected to have 50 cattle + 18 buffalo, but the checked-in working tree can be a partial subset.
- Full training with a class-count mismatch raises in `_validate_class_counts` unless small breeds were deliberately dropped.
- Do not infer the model's class count from folder counts; read the checkpoint or class map.

### 15.10 Tester vs web app

- `test_model.py` uses its own manager and API; changes to Flask routes do not affect it.
- It always returns top-5 and does not apply shared display filters.
- Presenter mode is separate from the React app and has its own config file.
- ODT export is disabled in presenter mode.

### 15.11 Edge assets

- Flutter TFLite assets are not checked in (`model.tflite` absent).
- TF.js expects `model_web.json`, which no export command generates.
- Edge has no OOD/quality/session features.

### 15.12 ODT empty report — **RESOLVED 2026-10-27**

`DevConfigPanel` now reads `breed-herd-session-v1`, maps the entries and exports `mode="batch"` (or a summary shell only when there is genuinely no session data). The single-result BPA card can also request an ODT with a registration block.

### 15.13 `RUN_ID_FORMAT` unused

Although `src/config.py` defines `RUN_ID_FORMAT`, `make_run_id()` uses the `VERSION.txt` counter plus `MMDD-HHMM` plus a random suffix unless an explicit `--run-tag` is supplied.

---

## 16. Explicitly not implemented

Do not describe these as features:

- User accounts, login, roles, or multi-tenant access.
- Server-side prediction history/database; the herd session is browser-local.
- Cloud sync of sessions, themes, or language.
- Authentication/authorization on `/dev` or any API.
- Live/streaming server logs (the UI fetches the whole log file on demand).
- Server-side image storage, retention, or audit trail of uploaded images.
- Object detection, segmentation, animal bounding boxes, pose, or body-part localization.
- Automatic breed measurement, weight estimation, or health diagnosis.
- OOD calibration for ONNX models (explicitly rejected).
- x86 PyTorch INT8 export (removed; uses ONNX/TFLite converter-side PTQ instead).
- TF.js web model export (`model_web.json`).
- Languages other than English and Tamil.
- PWA/offline mode/service worker.
- Model deletion/unloading from the dev UI.
- Per-user model defaults.
- Server-side authentication or role enforcement for the expert portal.
- Persisting expert escalations to a remote/cloud database (the SQLite file is local to the deployment).


---

## 17. Fast file index

| Question | File |
|---|---|
| Where are all constants? | `src/config.py` |
| Where is the model? | `src/model.py` |
| Where is the transform? | `src/data_pipeline.py` (`_train_transform`, `_eval_transform`, `_PadToSquare`) |
| Where is the training loop/loss? | `src/train.py` |
| Where are metrics? | `src/metrics.py` (training), `src/evaluate.py` (full eval) |
| Where is export? | `src/export.py` |
| Where is checkpoint loading? | `src/export._load_model`, `server/model_manager._load_model`, `test_model.ModelManager._load_model` |
| Where is server preprocessing/inference? | `server/services/inference.py` |
| Where is soft routing? | `src/model.BreedClassifier.predict`, `server/services/inference.py`, both Flutter engines |
| Where is OOD math? | `src/model.OODDetector` |
| Where is OOD folder calibration? | `src/ood_calibration.py` |
| Where is the shared config? | `server/shared_config.py`, `outputs/app_config.json` |
| Where is the user page? | `breed-classifier-react/src/pages/UserPage.jsx` |
| Where is the camera? | `breed-classifier-react/src/components/CameraCapture.jsx` |
| Where is the 260 optimizer? | `breed-classifier-react/src/utils/imageOptimizer.js` |
| Where is the quality gate/crop math? | `breed-classifier-react/src/utils/cropAndQuality.js` |
| Where is the result card? | `breed-classifier-react/src/components/ResultCard.jsx` |
| Where is the 5-level rubric? | `breed-classifier-react/src/utils/confidence.js`, `server/services/inference.py` |
| Where is the expert queue? | `server/services/escalation_db.py`, `server/routes/escalations.py`, `pages/ExpertPage.jsx` |
| Where is the offline outbox? | `breed-classifier-react/src/utils/escalationQueue.js` |
| Where is the attention X-Ray? | `server/services/inference.py`, `ResultCard.jsx` |
| Where is BPA registration? | `components/BpaRegistrationModal.jsx`, `server/routes/dev.py` (`/api/bpa/qr`) |
| Where is CLAHE? | `breed-classifier-react/src/utils/imageOptimizer.js` |
| Where is the herd session? | `breed-classifier-react/src/utils/sessionStore.js`, `HerdSessionDrawer.jsx` |
| Where are traits? | `breed-classifier-react/src/data/breedTraits.js` |
| Where is i18n? | `breed-classifier-react/src/i18n/` |
| Where is the dev portal? | `breed-classifier-react/src/pages/DevPage.jsx` + `Dev*Panel.jsx` |
| Where is the standalone tester? | `test_model.py` |
| Where is the Flutter preprocessor? | `flutter_app/lib/data/utils/image_preprocessor.dart` |

---

## 18. Feature release — 2026-10-27

The following report-promised features are now **ACTIVE** and should be described as current:

1. **Realtime namespace parity** — React and Flask both use `/ws`; live prediction, config, model, OOD and escalation events.
2. **Count-matched model state** — class maps are bound to the checkpoint's head widths; `/api/model-spec` reports dynamic counts.
3. **Per-model OOD binding** — `"auto"` resolves the active model's profile at startup and on every model switch.
4. **Five-level confidence rubric** — backend `confidence_level`/`confidence_band`/`confidence_action` + EN/TA UI actions.
5. **Expert escalation portal** — `/expert` with pending/verified/rejected metrics, attention heatmap, Top-3 bars, OOD telemetry, quality subscores, vet notes, and active-learning promotion.
6. **Offline escalation outbox** — `breed-escalation-outbox-v1`, header badge, one-click retry and auto-sync on `/ws` reconnect.
7. **AI Vision X-Ray** — interactive CBAM spatial mask + winning-head class-activation overlay with opacity/legend.
8. **68-breed morphological coverage** — all configured cattle/buffalo breeds have eight curated display fields.
9. **Guided camera capture** — clean flip-aware AR anatomical HUD (Head & Horns, Hump/Dorsal Ridge, Dewlap, Tail/Rump), 16:10 viewport, compact top HUD and independent lighting/framing/sharpness telemetry.
10. **Opt-in LAB CLAHE** — pure-JS shadow normalization on the 260×260 canvas; off by default to preserve inference parity.
11. **BPA registration** — 12-digit ear-tag UID, owner/animal form, herd-session save, scannable QR PNG card and printable ODT.
12. **Upload hardening + real ODT** — `secure_filename` + `.pt/.onnx` validation; Dev config ODT exports the actual herd session.
13. **Compact result / advanced details** — the main view shows only essential prediction results; X-Ray, traits, quality, diagnostics, BPA and vet escalation live in the Advanced drawer. Dev can hide the vet warning with `display.show_escalation_warning=false`.
14. **Upload preparation choice** — Downscale only (no crop) or Frame manually (1:1 cropper), with a one-click reset back to the original image.

### Verification snapshot (2026-10-27)

- `npm test` in `breed-classifier-react/`: **42/42 pass** (includes the five-level rubric, 260×260 CLAHE dimensions, 68/68 breed coverage, and pitch-black camera framing rejection).
- `npm run build`: passes.
- `python -m src.verify`: **VERIFICATION PASSED** (lite2 + lite4, 50 cattle / 18 buffalo heads, 1280-d features).
- Flask smoke: `/api/health`, `/api/models`, `/api/predict`, `/api/escalations`, `/api/bpa/qr` → HTTP 200 (escalation creation → 201).
- Socket.IO test client connected to `/ws` and received `server_status` on connect and after `request_status`.
