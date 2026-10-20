# BreedID — Cattle & Buffalo Breed Identification System
## Project Technical Report (architecture, implementation & non-ML feature inventory)

| Field | Value |
|---|---|
| Project | `ML-CB-B-identifier` — BreedID Livestock Vision |
| Version | v29 (`VERSION.txt`) |
| Report scope | Full repository: ML core (`src/`), serving layer (`server/`, `test_model.py`), React web app (`breed-classifier-react/`), Flutter edge app (`flutter_app/`), automation (`local_train.py`, `scripts/`), documentation (`docs/`, `ADR.md`) |
| Deployment targets | Web browser (desktop + mobile), Flask API, on-device Android (TFLite), browser inference (TF.js), Colab |
| Headline task | 50 cattle + 18 buffalo breed classification from a single 260×260 RGB image, with out-of-distribution (OOD) rejection |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem, Goals and Non-Goals](#2-problem-goals-and-non-goals)
3. [System Architecture](#3-system-architecture)
4. [Repository Map](#4-repository-map)
5. [ML Core](#5-ml-core)
6. [Serving Layer](#6-serving-layer)
7. [React Frontend](#7-react-frontend)
8. [Non-ML Feature Inventory](#8-non-ml-feature-inventory)
9. [Edge, Mobile and Colab Workflows](#9-edge-mobile-and-colab-workflows)
10. [Data, Governance and Observability](#10-data-governance-and-observability)
11. [Quality, Verification and Evidence](#11-quality-verification-and-evidence)
12. [Performance and Resource Profile](#12-performance-and-resource-profile)
13. [Known Limitations and Risks](#13-known-limitations-and-risks)
14. [Roadmap](#14-roadmap)
15. [Appendices](#15-appendices)

---

## 1. Executive Summary

**What it is.** BreedID is an end-to-end system for identifying Indian cattle and buffalo breeds from a photograph. It is not a model-only project: it is a *complete product surface* around a compact CNN — a training factory, an export/quantisation pipeline, a dual-backend serving layer, a React application with a camera-and-placement workflow, an edge app, and an operations/observability plane.

**Problem solved.** Field staff, veterinarians and livestock officers need a quick, offline-capable answer to "what breed is this animal?". A generic classifier fails on near-identical indigenous breeds, on long-tailed data, and on photos that are blurry, mis-framed or simply not cattle at all. The system addresses all three failure modes explicitly:

| Failure mode | System response |
|---|---|
| Look-alike breeds (Sahiwal vs Red Sindhi, Murrah vs Nili Ravi) | Shared backbone with species-specific heads, soft species routing, supervised-contrastive feature loss, and a curated trait comparison table shown in the UI |
| Long-tailed breed distribution | Effective-number sampler, per-rare-class mix guards, optional logit adjustment, breed-aware augmentation policy, dynamic head sizing |
| Wrong subject / bad photo | Post-hoc energy + MSP OOD detector with per-model calibration, plus an on-device image-quality gate (sharpness, exposure, framing) that coaches the user before inference |

**Scale of the codebase.**

| Area | Size | Files |
|---|---|---|
| ML core (`src/`) | 6,799 LOC | 16 modules |
| Serving (`server/` + `test_model.py`) | 5,010 LOC | 16 modules |
| React web app | 8,477 LOC JS/JSX + 4,242 LOC CSS | 39 files (20 components, 3 hooks, 3 utils, 3 test specs) |
| Automation (`local_train.py`, `scripts/`) | 3,761 LOC | 15 entry points |
| Documentation (`docs/`, `README.md`, `ADR.md`) | ~15.6k words | 18 mkdocs pages + ADR |
| Data | 11,199 images / 59 breeds inventoried | Kaggle sources merged locally |

**Status.** All subsystems are implemented, wired and verified: `npm test` 30/30, production build clean, Python modules compile, and browser-level end-to-end runs cover upload → frame → classify → export → session log, full-frame camera capture, manual OOD recalibration from a breed-named folder, and batch inference with the selected configuration.

---

## 2. Problem, Goals and Non-Goals

### 2.1 Goals

1. **Accuracy on a hard, imbalanced label set** — 50 cattle + 18 buffalo breeds, many visually adjacent, sourced from noisy Kaggle repositories.
2. **Edge-ready footprint** — a classifier small enough to run on a low-end Android phone and to be served as INT8 TFLite / ONNX.
3. **Zero-frust capture** — a non-technical user must be able to take a photo that the model can actually evaluate, without reading documentation.
4. **Operational safety** — the system must refuse to guess on a non-livestock or badly captured image, and it must be possible to re-tune that decision per deployed model without retraining.
5. **Reproducibility and auditability** — every run, prediction, export and threshold change is traceable.

### 2.2 Non-goals (deliberate)

| Non-goal | Rationale |
|---|---|
| Server-side object detection | A detection stage would add a second model and latency; ADR-001 replaced it with guided capture + a single classifier |
| Cloud dependency at inference | The core path runs on-device / on a local server; only optional tooling calls external APIs (e.g. trait auto-population via Gemini) |
| Fine-grained "instance" identification | Breed-level classification only; individual animal ID is out of scope |
| Image editing | The client normalises and frames images; it does not retouch them |

---

## 3. System Architecture

### 3.1 Layer map

```
┌───────────────────────────────────────────────────────────────────────────┐
│  CLIENTS                                                                  │
│  ┌────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐  │
│  │ React web app      │  │ Flutter edge app    │  │ Colab notebooks    │  │
│  │ (Vite + React 18)  │  │ (TFLite / TF.js)    │  │ (train + evaluate) │  │
│  └─────────┬──────────┘  └──────────┬──────────┘  └─────────┬──────────┘  │
│            │ REST + Socket.IO         │ on-device                │           │
└────────────┼──────────────────────────┼──────────────────────────┼───────────┘
             ▼                          ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  SERVING                                                                   │
│  ┌────────────────────────────┐      ┌───────────────────────────────┐     │
│  │ Flask API (server/)        │      │ test_model.py (stdlib HTTP +  │     │
│  │ 24 REST endpoints,         │      │ built-in GUI, dev/present)    │     │
│  │ Socket.IO events,          │      │ 16 endpoints                  │     │
│  │ shared config DB           │      │                               │     │
│  └────────────┬───────────────┘      └──────────────┬────────────────┘     │
│               └───────────────┬──────────────────────┘                      │
│                               ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  SHARED CORE  (src/)                                                  │  │
│  │  model.py · data_pipeline.py · train.py · export.py · metrics.py      │  │
│  │  run_utils.py · ood_calibration.py · parity_check.py · traits.py      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data flow — training

```
Kaggle sources ──► local_train.py §3 download ──► unzip/organize
      │                        (breed alias normalisation)
      ▼
data/raw/{cattle,buffalo}/<breed>/*.jpg
      ▼  prepare_splits()  ── 70/15/15 stratified, ≥1 val/test per breed,
      │                        optional dHash group-aware dedup
      ▼
data/splits/{train,val,test}.csv + {cattle,buffalo}_classes.json
      ▼  get_dataloaders()  ── effective-number sampler, breed-aware augmentation,
      │                        GPU-side CutMix/MixUp, colour-jitter policy per breed
      ▼
src/train.py  ── phase 1 (frozen backbone, head warm-up)
      │            phase 2 (full fine-tune, warmup+cosine, EMA, optional KD/SupCon)
      │            phase 3 (QAT, opt-in)
      ▼
outputs/checkpoints/<backbone>_phaseN_best_<run>.pt
      ▼  src/export.py  ── portable · ONNX fp32 · ONNX INT8(QDQ) · FP16 · TFLite(+INT8)
      ▼
outputs/export/**  ──► parity_check.py gate ──► outputs/ood/<model>.json
```

### 3.3 Data flow — inference (web)

```
file / camera ──► React ImageUpload ──► useImageOptimizer
                      │  decode (EXIF-upright) ─► 260×260 canvas
                      │  fit: stretch (default) | pad(mean | ambient-blur | black)
                      ▼
                  original kept for re-framing; optimized File used for inference
                      ▼
                  POST /api/predict  (multipart: image, model, original filename)
                      ▼
        Flask / test_model.py  ── PIL decode ── Resize(260)─CenterCrop(260)
                      │                          ── Normalize(ImageNet)
                      ▼
                 BreedClassifier (PyTorch or ONNX Runtime)
                      ▼
        soft routing  p(species) × softmax(head)   ──►  OOD gate (energy/MSP/floor)
                      ▼
        { top_k, ood_details, thresholds, latency, filename }
                      ▼
        React ResultCard  ── tier badge · trait matrix · quality badge · share PNG
                      ▼
        Herd session log (localStorage) ──► CSV / JSON / ODT export
```

### 3.4 Key architectural decisions (ADR-001, `ADR.md`)

| Decision | Alternatives rejected | Consequence |
|---|---|---|
| Single shared backbone, 3 heads (binary + cattle + buffalo) | Two-stage detect→classify; YOLOv8-classify | One forward pass, one export, but hard-attention to per-species heads |
| Guided capture instead of detection | YOLO detector for framing | No second model, no detector latency; user must frame the animal |
| Soft species routing `p(species)·softmax(head)` | Hard binary argmax | Removes ~5% routing error propagation; needs `top_k` blending in UI |
| Post-hoc energy OOD (no retraining) | Retrain with outlier exposure | Swappable per deployed model via JSON profile; requires calibration data |
| 3-phase curriculum | Single-phase training | Stable convergence; phase 1 is cheap and gives a usable checkpoint early |
| Mobile graph bakes ImageNet normalization | Runtime normalization in the app | Avoids float/uint8 mismatch bugs; parity gates still compare against PyTorch |

---

## 4. Repository Map

```
ML-CB-B-identifier/
├── src/                     ML core (6,799 LOC)
│   ├── config.py            single source of truth for all hyper-parameters
│   ├── model.py             EfficientNet-Lite, CBAM/SE, BreedClassifier, OODDetector, traits
│   ├── data_pipeline.py     splits, dedup, augmentation, sampler, datasets
│   ├── train.py             3-phase training loop, metrics, EMA, KD, SupCon
│   ├── metrics.py           macro-F1, blended score, shot buckets, confusion
│   ├── export.py            portable/ONNX/FP16/INT8/TFLite exporters
│   ├── evaluate.py          per-head evaluation + JSON/PNG metric artefacts
│   ├── parity_check.py      PyTorch ↔ ONNX ↔ TFLite parity gate
│   ├── ood_calibration.py   breed-named folder scanning + threshold calibration
│   ├── traits.py            auxiliary trait heads (training only)
│   ├── run_utils.py         run ids, timestamped paths, JSONL event logging
│   ├── verify.py            architecture sanity check
│   └── cbam.py / efficientnet_lite.py / ood_detector.py  (thin legacy shims)
├── server/                  Flask API (2,040 LOC)
│   ├── app.py               app factory, blueprint registration, Socket.IO
│   ├── model_manager.py     discovery, loading, prediction, OOD recalibration
│   ├── shared_config.py     thread-safe file-backed settings DB (outputs/app_config.json)
│   ├── routes/              models, predict, ood, config, shared_config, export, dev
│   ├── services/            inference, metadata, ODT report generation
│   ├── websocket/events.py  live status broadcast
│   └── utils/logger.py      session logger
├── test_model.py            standalone stdlib HTTP server + built-in GUI (2,970 LOC)
├── local_train.py           8-stage end-to-end training orchestrator (1,507 LOC)
├── train_model.py           manual training entry point
├── run.py                   lightweight inference runner
├── scripts/                 13 operational tools (2,254 LOC)
├── breed-classifier-react/   React web application (Vite)
│   └── src/
│       ├── pages/           UserPage (public), DevPage (control plane)
│       ├── components/      20 components (see §7.3)
│       ├── hooks/           useSocket, useSharedConfig, useImageOptimizer
│       ├── utils/           cropAndQuality, imageOptimizer, sessionStore
│       ├── data/            breedTraits.js (44 curated breeds)
│       ├── i18n/            I18nProvider + translations (EN / தமிழ்)
│       └── styles.css       4,242-line themeable design system
├── flutter_app/             Android/web edge app (TFLite + TF.js engines)
├── colab/                   Colab trainer + tester notebooks
├── docs/                    18 mkdocs pages
├── scripts/ + ADR.md        tooling and architecture decision record
└── outputs/                 checkpoints, exports, OOD profiles, metrics, logs
```

---

## 5. ML Core

### 5.1 Data pipeline (`src/data_pipeline.py`, 1,150 LOC)

| Capability | Implementation |
|---|---|
| Split creation | `_stratified_split()` — 70/15/15 per breed, ≥1 validation and ≥1 test image per breed, deterministic seed |
| Duplicate control | `_dhash()` perceptual hash + `_groups_by_hash()`; `--dedup-splits` keeps near-duplicates (Hamming ≤ 4) in the same split to stop inflated validation scores |
| Fast mode splits | `prepare_half_splits()`, `prepare_quarter_splits()`, `prepare_smoke_splits()` for 50% / 25% / tiny data budgets |
| Class maps | `_load_class_maps()` writes `cattle_classes.json` / `buffalo_classes.json`; head sizes are derived from the surviving classes (dynamic head sizing) |
| Long-tail sampler | `_make_weighted_sampler()` implements effective-number-of-samples weighting (β = 0.99) |
| Augmentation | `_train_transform()` (flip, RRC, colour-jitter, RandAugment), `_eval_transform()` (Resize 260 → CenterCrop 260 → Normalize), `_PadToSquare` with ImageNet-mean fill |
| Breed-aware policy | `resolve_breed_augment()` disables colour jitter for colour-defined breeds so coat identity is preserved |
| Mixing | `cutmix()` / `mixup()` executed on GPU, same-species pairing, rare-class guard, mixing switched off for the last 15% of phase 2 |
| Metadata | `describe_transform()` prints the exact transform chain at start-up for auditability |
| Priors & counts | `compute_class_priors()`, `compute_class_counts()`, `compute_rare_classes()` for imbalance handling and mix guards |

### 5.2 Model (`src/model.py`, 919 LOC)

```
Input 260×260×3
  └── EfficientNetLite (lite2: 4.81M backbone params | lite4: 11.73M)
        └── CBAM (identity-initialised) or SE, inserted after stage 3
              └── pooled feature (1280-d)
                    ├── binary_head      → 2      (cattle / buffalo)
                    ├── cattle_head      → N_cattle
                    ├── buffalo_head    → N_buffalo
                    └── projection_head → 128    (SupCon auxiliary, never exported)
```

* **Soft species routing** — the combined distribution is `cat(p_cattle·softmax(cattle), p_buffalo·softmax(buffalo))`; no hard species gate.
* **Optional cosine/ArcFace head** (`--cosine-head`) with scale 30 and a 10-epoch margin ramp; export stays margin-free and auto-detects cosine checkpoints.
* **Trait classifier** (`src/traits.py`) — per-field masked auxiliary heads (hump, horn, coat, ear, dewlap, face, size) over pooled features; training-only, excluded from export.
* **OOD detector** (`src/model.py:613`) — free energy over concatenated heads, MSP over the binary head, and a binary-confidence floor; flags a sample OOD if any of the three trips. Supports per-model JSON profiles, global fallback, and calibration helpers (`calibrate`, `save`, `list_available_profiles`).

### 5.3 Training curriculum (`src/train.py`, 1,742 LOC)

| Phase | Epochs (default) | LR | What happens |
|---|---|---|---|
| 1 — head warm-up | 8 | 3e-3 | Backbone frozen, all heads train, `backbone_eval()` |
| 2 — multi-task fine-tune | 80 | 1e-4 | Everything unfrozen, warm-up + cosine schedule, EMA, optional KD/SupCon/logit-adjust, binary-head saturation re-weighting |
| 3 — QAT (opt-in) | 10 | 5e-6 | Starts from the best phase-2 weights, prepares for quantisation |

**Techniques in the box:** AdamW (wd 1e-2), label smoothing 0.1, gradient accumulation (VRAM-scaled), AMP, EMA (decay 0.9995, warm-up `(1+t)/(10+t)`), effective-number sampler, supervised contrastive loss (λ 0.3) with optional confusion-driven hard-pair boosting, knowledge distillation (α 0.7, T 4.0), same-species CutMix/MixUp with rare-class guard, logit adjustment (off by default to avoid double-correcting with the sampler), mix-off tail, deterministic seeding, and timestamped run ids.

### 5.4 Evaluation and metrics (`src/metrics.py`, `src/evaluate.py`)

* **Checkpoint selection metric** `blended_score = 0.5 · mean(cattle_macro_f1, buffalo_macro_f1) + 0.5 · combined_top1_soft` — deliberately avoids selecting on raw top-1 or macro-F1 alone.
* Per-head accuracy, macro-F1, balanced score, per-shot-bucket accuracy (1-shot / few / medium), confusion matrices, true/predicted histograms.
* `scripts/diagnose_model.py` — full diagnostic report for one or two checkpoints (raw vs EMA) including the top-20 confusion pairs.
* `scripts/mine_confusions.py` — emits `outputs/metrics/confusion_pairs.json`, consumable by `--hard-pairs` during training.

### 5.5 OOD detection and calibration

| Capability | Location |
|---|---|
| Detector + thresholds + per-model profile resolution | `src/model.py` (`OODDetector`) |
| Automatic calibration after training | `scripts/calibrate_ood.py`, `local_train.py` §7.5, `OOD_AUTO_CALIBRATE` |
| **Manual folder calibration** (new) | `src/ood_calibration.py` — walks a folder, matches breed names to the model's class maps (alias + fuzzy fallback), calibrates on matched images only, reports coverage |
| CLI calibration on the validation split | `scripts/calibrate_ood.py` |
| Runtime tuning / profile management | `server/routes/ood_api.py`, `test_model.py` `/api/ood*`, React `DevOODPanel` |

`src/ood_calibration.py` details: supports flat (`Deoni_01.jpg`) and nested (`cattle/deoni/…`) layouts, normalises tokens (`Baragur→bargur`, `Malenadu_Gidda→malnad_gidda`, `Muthamahal→amritmahal`, `Hallikaru→hallikar`, `Bhadwari→bhadwari`, `Poda_Thurpu→poda_thirupu`, `Bhelai↔belahi`, `Marathwadi→marathwada`, `Gangatiri→gangatari`, `Ghumsari↔ghumusari`), verifies every file is readable, excludes unmatched/unreadable files from calibration, and returns thresholds, sample counts, per-breed coverage, unmatched names and the saved profile path.

### 5.6 Export and deployment (`src/export.py`, 656 LOC)

| Format | Artifact | Size (observed) |
|---|---|---|
| Portable bundle | `model.pt` + class maps + `model_info.json` | 42 MB (lite2 fp32) |
| ONNX fp32 (desktop) | `*_fp32.onnx` | 26–28 MB |
| ONNX fp32 (mobile graph, normalization baked in) | `*_mobile_fp32.onnx` | 26–28 MB |
| ONNX INT8 QDQ (ORT Mobile) | `*_mobile_int8.onnx` | **7.2–7.6 MB** |
| PyTorch fp16 | `*_float16.pt` | 17–18 MB |
| TFLite fp32 / INT8 PTQ | `model.tflite` | INT8 for Flutter assets |

`src/parity_check.py` is the release gate: it compares PyTorch fp32 against ONNX INT8 and TFLite INT8 on the same images and fails if the combined top-1 drops more than a configured tolerance (default 1 percentage point). `scripts/onnx_parity_10.py` performs a stricter 10-image logit-level check.

---

## 6. Serving Layer

### 6.1 Two backends, one contract

| Aspect | `server/` (Flask) | `test_model.py` (stdlib HTTP + GUI) |
|---|---|---|
| Purpose | Production-style API for the React app | Zero-dependency tester with a built-in GUI |
| Endpoints | 24 | 17 |
| Async events | Socket.IO (`/ws` namespace) | none |
| Settings | Shared JSON DB (`outputs/app_config.json`) with live broadcast | `presenter_config.json` file |
| Modes | — | `--dev` (tools) / `--present` (clean demo) |
| Ports | 5000 | 8501 |

### 6.2 API surface

**Flask (`server/`)**

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness + model count + device |
| POST | `/api/predict` | Multipart inference (`image`, `model`, `filename`) → species, top-k, OOD details, thresholds, latency |
| GET | `/api/models` | Discovered model registry with sizes/backbones |
| POST | `/api/select-model` | Switch the active model globally |
| GET | `/api/model-spec` | Introspection: parameter counts, per-layer breakdown, ONNX I/O shapes |
| POST | `/api/upload-model` | Register a new checkpoint at runtime |
| GET | `/api/ood/profiles` | Enumerate calibration profiles with thresholds and metadata |
| GET | `/api/ood/active` | Active profile + resolved detector status |
| POST | `/api/ood/select` | Choose `auto` or a specific profile |
| POST | `/api/ood/toggle` | Enable/disable OOD filtering |
| POST | `/api/ood/thresholds` | Live threshold tuning (energy, MSP, temperature) |
| POST | `/api/ood/upload` | Upload a custom calibration JSON |
| GET | `/api/ood/data-dirs` | Suggested calibration folders |
| POST | `/api/ood/recalibrate` | **Manual folder-based recalibration for the selected model** |
| GET/POST | `/api/config`, PATCH `/api/config`, POST `/api/config/reset`, GET `/api/config/<key>` | Shared settings DB (dot-notation) |
| GET/POST | `/api/presenter-config` | Presenter view customisation |
| POST | `/api/export-odt` | ODT report from single/batch results |
| GET | `/api/dev-info` | Class maps |
| POST | `/api/image-info` | EXIF/dimension metadata extractor |
| GET | `/api/logs` | Session log tail |

**`test_model.py`** exposes the same prediction/export/upload semantics with the legacy names (`/api/ood-profiles`, `/api/ood-info`, `/api/dataset-info`, `/api/recalibrate-ood` — kept as an alias of `/api/ood/recalibrate` — plus `/api/ood-data-dirs`).

### 6.3 Runtime services

* **Model registry & loader** (`server/model_manager.py`, `test_model.py::ModelManager`) — discovers `.pt`/`.onnx`/portable exports, infers architecture dimensions from checkpoint tensors, supports cosine heads, caches loaded models, and reports spec metadata.
* **Shared configuration DB** (`server/shared_config.py`) — thread-safe, file-backed, dot-notation, atomic writes, change notifications. Groups: `active_model`, `ood.*`, `display.*`, `dataset.*`, `inference.*`.
* **Real-time events** (`server/websocket/events.py`) — broadcasts `server_status`, `config_changed`, `ood_status_changed`, `prediction_started`, `prediction_complete`, `prediction_error`, `model_switched`, `model_loaded` to every connected client.
* **ODT report service** (`server/services/odt_export.py`) — builds a styled OpenDocument report from single or batch results, including per-image top-5 tables and a batch summary.
* **Session logging** (`src/run_utils.py`, `server/utils/logger.py`) — one folder per execution with `manifest.json` (argv, cwd, git commit, environment, exit code), `config.json` (full hyper-parameter snapshot), `run.log`, and `events.jsonl` / `actions.jsonl` / `training.jsonl` / `data.jsonl` / `test.jsonl` / `export.jsonl`.

---

## 7. React Frontend

### 7.1 Stack and structure

| Aspect | Choice |
|---|---|
| Build | Vite 5 (`breed-classifier-react/vite.config.js`), dev proxy `/api` and `/socket.io` → backend |
| UI | React 18 (function components + hooks only), React Router 7 |
| Realtime | `socket.io-client` |
| Styling | Single hand-written design system, `styles.css` (4,242 LOC), CSS custom properties, 5 themes |
| State | Local component state + 3 custom hooks; no global store |
| i18n | Custom lightweight provider (`I18nProvider` + `translations.js`), EN + தமிழ், persisted |
| Tests | `node --test` (no test framework dependency), 30 tests across 3 spec files |
| Routes | `/` → `UserPage` (public app), `/dev` → `DevPage` (developer control plane) |

### 7.2 State architecture

```
UserPage
 ├── useImageOptimizer()      original image · optimized 260×260 result · fit config · crop state
 ├── useSharedConfig(socket)  server settings (model, display, OOD) with live updates
 ├── useSocket()              connection state + prediction events
 └── local state              mode (upload|camera), quality report, prediction result, herd session, panels
```

`useImageOptimizer` (`src/hooks/useImageOptimizer.js`) is the single owner of the upload lifecycle: it keeps the **original** file separate from the **optimized 260×260** result, so re-framing always starts from the original and the visible image is never replaced by the model input.

### 7.3 Component inventory

| Component | Responsibility |
|---|---|
| `UserPage` | Public page: tabs (upload / camera), preview, explicit classify action, result card, herd session, Advanced panel |
| `DevPage` | Control plane: model, OOD, config and log tabs |
| `ImageUpload` | Minimal upload surface — shows the **original** image, framing/remove/choose actions, explicit **Identify / Identify Framed Breed** button |
| `CameraCapture` | Full-frame live camera, quality gate, auto-capture, burst-3, torch, Pad/Stretch review, optional manual framing, hands off an optimized 260×260 file |
| `ImageCropper` | Shared interactive 1:1 cropper: drag/pinch/wheel pan on both axes, zoom 1–3×, live quality, square export |
| `AdvancedPanel` | Header-level panel: model-input vs original preview, `ORIGINAL/MODEL INPUT/SCALED/PADDING/PAYLOAD/TIME`, fit modes, batch tester |
| `AdvancedBatchTester` | Multi-image run with the selected model + fit config, stats, payload saved, optional session append |
| `ResultCard` | 3-tier confidence/OOD banner, ranked breeds, trait comparison, quality badge, share PNG, technical diagnostics |
| `TraitComparison` | Shared “Why this breed?” table with differing-row highlighting |
| `QualityBadge` | 0–100 sharpness / lighting / framing breakdown and status pill |
| `HerdSessionDrawer` | Session log: totals, cattle/buffalo, flagged, average confidence/quality, CSV/JSON/ODT export |
| `DevModelPanel` | Registry, active-model switch, model spec viewer, checkpoint upload |
| `DevOODPanel` | OOD lab: toggle, profile select/upload, live threshold sliders, **manual folder recalibration** |
| `DevConfigPanel` | Shared settings editor |
| `Batch` / `SingleImage` / `ResultsPanel` / `ModelSelect` / `PredictionProgress` / `ThemeToggle` / `LanguageToggle` / `DevTools` / `App` | Legacy/complementary surfaces |

### 7.4 Client-side image pipeline (`src/utils/`)

| Utility | Purpose |
|---|---|
| `imageOptimizer.js` | 260×260 zero-crop optimizer: EXIF-upright decode, progressive halving, letterbox (ImageNet mean / ambient blur / black) and stretch modes, full metadata contract, object-URL lifecycle management |
| `cropAndQuality.js` | Authoritative crop math (cover/zoom/pan/mirror), on-device quality gate (Laplacian sharpness, exposure histogram, Sobel framing), share-card renderer |
| `sessionStore.js` | Herd session persistence, statistics, CSV/JSON/ODT mapping, thumbnail generation, quota-safe writes |

### 7.5 Design system

* 5 themes (`dark`, `light`, `emerald`, `sunset`, `cyber`) defined purely as CSS custom properties, applied by a single `data-theme` attribute; persisted in `localStorage`.
* Reusable primitives: cards, tabs, pills, sliders, toggles, drawers, tables, progress, toasts.
* Accessibility: `aria-label`/`aria-pressed`/`aria-expanded`, `role="status"` live regions for quality and batch progress, `role="dialog"` + `aria-modal` for overlays, keyboard-reachable dropzone and controls, non-colour-only status (icon + text + border), responsive layouts down to 360 px.

---

## 8. Non-ML Feature Inventory

Everything in this section is product/engineering surface built **around** the model. It is what turns a checkpoint into a usable product.

### 8.1 Capture, framing and image preparation

| # | Feature | Where | Why it matters |
|---|---|---|---|
| F1 | **Zero-crop 260×260 client optimizer** — the client converts every input to exactly the model input size, so the server's `Resize→CenterCrop` becomes an identity pass-through and 25–44% of a 4:3/16:9 animal is never truncated | `src/utils/imageOptimizer.js` | Removes an entire class of silent data loss |
| F2 | **Four fit modes** — stretch (default), ImageNet-mean pad, ambient-blur pad, black pad | Advanced panel / camera review | Different ground truths: no distortion vs full content |
| F3 | **EXIF-upright decoding** with `createImageBitmap({imageOrientation:'from-image'})` and an `Image` fallback | `imageOptimizer.js` | Portrait phone photos are not sideways in production |
| F4 | **Progressive multi-step downscaling** (halving then one high-quality draw) | `imageOptimizer.js` | No aliasing on horns/coats from 12 MP sources |
| F5 | **Full-frame camera viewfinder** (`object-fit: contain`) with a guide framing the whole sensor frame | `CameraCapture.jsx` | The user sees 100% of what the camera captures |
| F6 | **Authoritative framing cropper** with pan on **both axes**, 1–3× zoom, wheel/pinch, rule-of-thirds grid | `ImageCropper.jsx` | Framing is a first-class, optional step — and always re-applied to the original |
| F7 | **Live quality gate** — Laplacian sharpness, exposure histogram, Sobel framing heuristics, 4 Hz analysis | `cropAndQuality.js` | Catches blur/darkness/partial-subject *before* inference |
| F8 | **Auto-capture when stable** (3 consecutive READY checks ≈ 750 ms) with a countdown ring | `CameraCapture.jsx` | Removes shake and mis-taps |
| F9 | **Burst ×3 sharpest-frame selection** | `CameraCapture.jsx` | Best-of-3 without user effort |
| F10 | **Torch / flash control** via capability detection, front/rear toggle, permission and no-device states | `CameraCapture.jsx` | Works on real phones, degrades gracefully |
| F11 | **Crop on the original, never on a previous crop** | `useImageOptimizer.js` | Repeated re-framing cannot compound degradation |

### 8.2 Confidence, explainability and safety

| # | Feature | Where |
|---|---|---|
| F12 | **3-tier confidence model** — verified (green), ambiguous/look-alike (amber), safety abstention (red) with margin-aware thresholds (≥0.72 confidence and ≥0.12 top-1/top-2 margin) | `ResultCard.jsx` |
| F13 | **Exact OOD trigger reporting** — which of energy / MSP / binary floor tripped, with values and thresholds | `ResultCard.jsx`, `ood_details` |
| F14 | **“Why this breed?” trait matrix** — 44 curated breeds × 8 fields (hump, horn, coat, ear, dewlap, face, size, origin) with differing-row highlighting | `src/data/breedTraits.js`, `TraitComparison.jsx` |
| F15 | **Look-alike explanation** with a key-distinguisher sentence per breed | `breedTraits.js` |
| F16 | **Quality badge carried into the result** (sharpness/lighting/framing, 0–100) | `ResultCard.jsx` |
| F17 | **Actionable abstention** — one-click retake with the guided coach instead of a dead end | `ResultCard.jsx` |

### 8.3 Session, reporting and provenance

| # | Feature | Where |
|---|---|---|
| F18 | **Herd session log** persisted in `localStorage` (thumbnail, timestamp, model, species, breed, confidence, OOD, quality) | `src/utils/sessionStore.js` |
| F19 | **Session statistics** — totals, cattle/buffalo split, flagged count, average confidence and quality | `HerdSessionDrawer.jsx` |
| F20 | **Session exports** — CSV, JSON and ODT (via the existing report service) | `HerdSessionDrawer.jsx` |
| F21 | **Original filename provenance** carried through prediction, exports, share card, diagnostics and server logs | end-to-end |
| F22 | **Shareable PNG result card** rendered on canvas (photo, species badge, top-3 bars, traits, quality, model, filename, timestamp) with native share and download fallback | `cropAndQuality.js` |
| F23 | **ODT report generation** for single and batch runs | `server/services/odt_export.py` |

### 8.4 Developer / operator control plane

| # | Feature | Where |
|---|---|---|
| F24 | **Model registry** — discovery, active-model switching, checkpoint upload, per-model spec (params, layer breakdown, ONNX I/O) | `DevModelPanel.jsx` |
| F25 | **OOD laboratory** — profile list, enable/disable, live threshold tuning, JSON upload, and **manual folder recalibration with coverage reporting** | `DevOODPanel.jsx` |
| F26 | **Shared configuration database** with dot-notation, atomic writes and live broadcast to all clients | `server/shared_config.py` |
| F27 | **Real-time collaboration** — model switch, config change, OOD status and prediction events over Socket.IO | `server/websocket/events.py` |
| F28 | **Execution logs viewer** — list, summary, metrics, event queries and **diff between two runs** | `scripts/view_logs.py` |
| F29 | **Image metadata analyzer** (EXIF, dimensions, colour space) | `/api/image-info` |
| F30 | **Dataset inventory viewer** (breed/species/count/resolution per source) | `local_train.py`, `docs/` |
| F31 | **Presenter mode** — a cleaned, configurable UI for demonstrations | `test_model.py --present` |
| F32 | **Batch tester inside the Advanced panel** using the selected model and fit configuration | `AdvancedBatchTester.jsx` |

### 8.5 Product ergonomics

| # | Feature | Where |
|---|---|---|
| F33 | **Minimal default view** — the user sees the original photo; every optimizer control is one click away in the Advanced panel | `UserPage.jsx` |
| F34 | **Explicit classify action** that changes to “Identify Framed Breed” after framing, and reappears whenever the input changes | `ImageUpload.jsx` |
| F35 | **Stale-result protection** — a new upload, crop or fit change clears the previous prediction | `UserPage.jsx` |
| F36 | **5 themes** and **EN / தமிழ்** localisation with persisted choice | `ThemeToggle.jsx`, `LanguageToggle.jsx` |
| F37 | **Camera deep link** `/?mode=camera` for kiosks and QR launches | `UserPage.jsx` |
| F38 | **Offline-capable edge app** with the same preprocessing contract on device | `flutter_app/` |

---

## 9. Edge, Mobile and Colab Workflows

| Target | Implementation | Notes |
|---|---|---|
| Android | `flutter_app/` — TFLite interpreter engine, camera screen, inference controller, domain/entity layering | Assets prepared by `scripts/prepare_app_assets.py`; normalization baked into the TFLite graph |
| Browser (TF.js) | `flutter_app/lib/data/ml_engines/web_tfjs_engine.dart` | Same preprocessing as the server |
| Colab training | `colab/cattle_buffalo_trainer.py` + notebook | Full-data training on hosted GPUs; archive helpers for dataset/project upload |
| Colab testing | `colab/cattle_buffalo_tester.py` + notebook | Notebook-based evaluation |
| Local automation | `local_train.py` — 8 stages: prerequisites → venv → dataset download → organize → verify → splits → train → export → OOD calibration → summary | `--quarter-data` / `--half-data` / `--smoke-test` budgets |
| Deployment docs | `docs/android-deployment.md`, `docs/export-deployment.md`, `docs/colab-*.md` | 18 mkdocs pages |

---

## 10. Data, Governance and Observability

### 10.1 Data provenance and quality

| Asset | Content |
|---|---|
| Sources | Kaggle `algsoch/breed-cattle-buffalo`, `atharvadarpude/indian-cattle-image-dataset`, `atharvadarpude/indian-buffalo-dataset` |
| Merged inventory | **59 breeds / 11,199 images** (8,531 cattle, 2,668 buffalo), 0 unreadable |
| Class maps | 50 cattle + 18 buffalo configured; head sizes derived from the surviving classes |
| Breed alias map | `BREED_ALIASES` in `src/config.py` + `BREED_ALIASES` in `local_train.py` normalise Kaggle spelling drift |
| `scripts/audit_data.py` | Report-only audit: counts, fuzzy breed-name collisions, cross-species duplicates, near-duplicate images across splits, corrupt files — never deletes |
| `scripts/diagnose_model.py` | Per-checkpoint diagnostic report including per-shot-bucket recall and top confusion pairs |
| `scripts/mine_confusions.py` | Emits confusion pairs for hard-pair training |
| `scripts/populate_breed_traits.py` | Auto-populates `data/breed_traits.json` with an LLM, then a human review |
| `data/breed_traits.json` | 75 breed entries × 7 canonical training fields (+ display-only origin/distinguisher) |

### 10.2 Observability

| Signal | Mechanism |
|---|---|
| Run identity | `run_id` (auto `DD-MM-YYYY-HH-MM` or `--run-tag`), `exec_id` (`YYYYmmdd-HHMMSS-xxxx`) |
| Immutable artefacts | Timestamped checkpoints/exports/metrics — repeated runs never overwrite |
| Full environment capture | `manifest.json` (argv, cwd, Python, **git commit**, env, exit code) + `config.json` (every config constant) |
| Structured events | `events.jsonl`, `actions.jsonl`, `training.jsonl` (per-epoch losses and metrics), `data.jsonl`, `test.jsonl` (one record per prediction, with original filename), `export.jsonl` (artefact sizes, parity verdicts) |
| Log diffing | `scripts/view_logs.py diff <exec_a> <exec_b>` |
| Dataset inventory | JSON per source with breed/species/count/resolution |

---

## 11. Quality, Verification and Evidence

### 11.1 Automated checks

| Suite | Scope | Result |
|---|---|---|
| `breed-classifier-react` unit tests (`node --test`) | crop math (cover, zoom-centre invariance, pan clamp, mirror complement, shortest-side bound), quality thresholds (flat, checkerboard, dark, overexposed), optimizer layout (1920×1080→260×146 padY 57; 1080×1920→146×260 padX 57; identity; no-upscale; stretch; aspect ratios), session store (entry mapping, stats, CSV escaping, ODT mapping, trait aliases/comparison) | **30/30 pass** |
| Production build | Vite bundle of 95 modules | **success** — 415 kB JS (131 kB gzip), 72 kB CSS |
| Python compile | `test_model.py`, `src/ood_calibration.py`, `server/model_manager.py`, `server/routes/ood_api.py`, `server/services/inference.py`, `server/routes/predict.py` | **pass** |
| Inline GUI JS | `test_model.py` generated HTML script extracted and checked with `node --check` | **pass** |
| CPU synthetic suites | `scripts/test_fixes_cpu.py`, `scripts/test_master_cpu.py` | available without GPU/dataset |
| Parity gate | `src/parity_check.py`, `scripts/onnx_parity_10.py` | PyTorch ↔ ONNX-INT8 ↔ TFLite-INT8 within tolerance |

### 11.2 Browser-level end-to-end evidence

| Scenario | Verified outcome |
|---|---|
| Upload → optimize | Original `1280×853` shown; model input `260×260`; metadata `SCALED 260×173`, `PADDING 0,43`, payload, timing |
| Advanced panel | Model-input/original toggle, metadata grid, 4 fit modes, batch tester all functional |
| Re-frame | Crop source `1280px` before **and** after a crop (always the original) |
| Classify | Result card + tier badge; session entry `Deoni_01.jpg`; diagnostics show `FILE Deoni_01.jpg` |
| Re-frame after a result | Previous result cleared; “Identify Framed Breed” returns |
| Camera | Full-frame contain view, guide badge, quality pill, Pad/Stretch review, handoff of a 260×260 file |
| Batch (selected config) | 2/2 images, cattle/buffalo/OOD split, average confidence, `889.9 KB → 56.5 KB` |
| Manual OOD recalibration (React) | 16 models listed, 67 candidate folders, result `SAMPLES 6 / MATCHED 6 / UNMATCHED 0`, breed chips, saved profile path |
| Manual OOD recalibration (API) | `energy −3.8443`, `msp 0.8688`, profile written to `outputs/ood/<model>.json`; the next prediction used the new thresholds |
| Logging | JSONL: `{"event":"prediction","filename":"Deoni_01.jpg",...}`; batch row `Hallikaru_02.jpg` |

### 11.3 Last recorded training runs (historical logs, `Test results/`)

| Run | Dataset | Phase 2 best val top-1 |
|---|---|---|
| lite2, full data, 40 phase-2 epochs | 7,374 train / 867 val | 0.5779 |
| lite4, 40 phase-2 epochs | same corpus | 0.6344 |

Phase-1 head warm-up reached 0.8143 binary accuracy on lite2. QAT (phase 3) historically degraded accuracy and is therefore opt-in and still an open engineering item (§13).

---

## 12. Performance and Resource Profile

| Metric | lite2 | lite4 |
|---|---|---|
| Backbone parameters | 4,811,072 | 11,725,568 |
| Total trainable parameters | 6,490,719 | 13,405,903 |
| Checkpoint size (fp32) | 42 MB | — |
| ONNX fp32 export | 26–28 MB | — |
| ONNX INT8 (QDQ) export | 7.2–7.6 MB | — |
| Client upload payload (260×260 JPEG) | 19 KB (pad) / 29 KB (stretch) from a 1280×853 source — −94% to −96% | — |
| Server CPU inference (observed) | ~140–250 ms/image including decode | — |
| Client optimization time (observed) | ~80–220 ms for a 1280×853 source, progressive | — |

---

## 13. Known Limitations and Risks

| # | Limitation | Impact | Mitigation / status |
|---|---|---|---|
| L1 | QAT (phase 3) historically reduced accuracy (loss spike, top-1 drop) | Quantisation-aware training not production-ready | Phase 3 is opt-in; mobile INT8 comes from converter PTQ; parity gates protect releases |
| L2 | Manual OOD calibration supports PyTorch `.pt` checkpoints only | ONNX models cannot be recalibrated from a folder | Clear error returned; ONNX path can be added by porting the calibration loop to ORT outputs |
| L3 | Letterbox padding is a small distribution shift vs the centre-cropped training distribution | Possible accuracy delta vs the manual crop path | Pad vs Stretch is one click apart; validate on the val split before a field default |
| L4 | The committed `data/splits` are small (174/58/58) relative to the 11,199-image corpus | Quick runs are not representative | `local_train.py` regenerates splits from `data/raw`; full runs are logged in `Test results/` |
| L5 | Herd session is browser-local | No multi-device sync | CSV/JSON/ODT export covers sharing; server sync is a roadmap item |
| L6 | Tier thresholds are constants in the component | Not runtime-tunable | Deliberate: thresholds that drift silently are harder to defend in review |
| L7 | No automated test suite for the Python services in CI | Regressions found manually | CPU synthetic suites exist (`scripts/test_*_cpu.py`); CI currently publishes docs only |
| L8 | Trained artefacts in `outputs/` are large (42 MB checkpoints) | Repository weight | Excluded from version control via `.gitignore`; distributable bundles are zipped on demand |
| L9 | Ambient-blur padding uses `ctx.filter` | Unsupported in some browsers → plain stretched background | Cosmetic only; sharp image is always drawn on top |

---

## 14. Roadmap

| Horizon | Item | Value |
|---|---|---|
| Next | ONNX-capable OOD calibration (ORT logits path) | Parity between `.pt` and deployed ONNX artefacts | 
| Next | Automated CI for the Python services and a React component test layer | Faster, safer iteration |
| Next | Side-by-side look-alike photo comparison (top-2 crops with difference callouts) | Faster human adjudication of ambiguous cases |
| Next | Configurable tier thresholds in the shared settings DB | Field tuning without redeploying |
| Medium | Server-side session sync for multi-inspector deployments | Herd workflow across devices |
| Medium | Model A/B comparison in the Advanced panel (same input, two checkpoints) | Evidence for model promotion decisions |
| Medium | Guided capture hints derived from the predicted breed | “For a Gir, photograph the side profile” |
| Long | Offline field mode with a service worker and queued sync | Field use without connectivity |
| Long | ON-device OOD for the Flutter app (same thresholds, embedded profile) | Consistent safety behaviour on mobile |

---

## 15. Appendices

### 15.1 Reproduction quick reference

```bash
# 1. Environment
python setup_venv.py && source .venv/bin/activate && pip install -r requirements.txt

# 2. Full local pipeline (download → organize → verify → split → train → export → OOD)
python local_train.py --skip-setup --skip-download
python local_train.py --quarter-data          # fast iteration
python local_train.py --smoke-test            # seconds-long sanity check

# 3. Manual stages
python scripts/audit_data.py --data data/raw --split-dir data/splits
python scripts/diagnose_model.py --checkpoint outputs/checkpoints/lite2_phase2_best_*.pt --split test
python scripts/calibrate_ood.py --checkpoint outputs/checkpoints/lite2_phase2_best_*.pt --percentile 95
python scripts/view_logs.py summary latest

# 4. Serving
python server/app.py --port 5000              # Flask API + Socket.IO
python test_model.py --port 8501 --no-browser # standalone tester + GUI

# 5. Web app
cd breed-classifier-react && npm install && npm run dev   # http://localhost:5173
npm test && npm run build
```

### 15.2 Key CLI surfaces

| Command | Purpose |
|---|---|
| `local_train.py` | 8-stage pipeline; `--quarter-data`, `--half-data`, `--smoke-test`, `--include-qat`, `--backbone lite4`, `--run-tag` |
| `python -m src.train` | Training; `--phase1/2/3-epochs`, `--teacher`, `--cosine-head`, `--hard-pairs`, `--dedup-splits`, `--logit-adjust`, `--min-images-per-breed`, `--trait-weight` |
| `python -m src.export` | `--mode portable | onnx | onnx-int8 | int8 | float16 | tflite`, `--calibration-images` |
| `python -m src.evaluate` | Per-head metrics, JSON artefacts, confusion PNG |
| `python -m src.parity_check` | Release gate across runtimes |

### 15.3 Glossary

| Term | Meaning |
|---|---|
| Breed head / species head | Per-species classification head sharing one backbone |
| Soft routing | `p(species) × softmax(head)` instead of a hard species gate |
| Blended score | `0.5·macro-F1 + 0.5·soft top-1`; the checkpoint-selection metric |
| Energy score | Free energy of the concatenated logits; high energy ⇒ OOD |
| MSP | Maximum softmax probability of the binary (species) head |
| Parity gate | Runtime-equivalence check across PyTorch/ONNX/TFLite |
| Portable bundle | Self-contained `model.pt` + class maps + `model_info.json` |
| Zero-crop | Producing the model input without discarding any part of the source image |

### 15.4 File → responsibility index (selected)

| Path | Responsibility |
|---|---|
| `src/config.py` | Every hyper-parameter and path; single source of truth |
| `src/model.py` | Backbone, attention, heads, traits, OOD detector |
| `src/data_pipeline.py` | Splits, dedup, augmentation, sampler, datasets |
| `src/train.py` | Three-phase training loop and all training-time techniques |
| `src/metrics.py` | Metric definitions incl. blended score and shot buckets |
| `src/export.py` | Multi-format exporters with normalization baked for mobile |
| `src/ood_calibration.py` | Breed-name aware folder scanning and calibration |
| `src/run_utils.py` | Run ids, timestamped paths, JSONL event logging |
| `server/model_manager.py` | Registry, loader, prediction, recalibration |
| `server/shared_config.py` | Live settings database |
| `server/routes/ood_api.py` | OOD profiles, thresholds, manual recalibration |
| `server/services/odt_export.py` | ODT report generation |
| `test_model.py` | Standalone tester server and GUI (dev/present modes) |
| `breed-classifier-react/src/utils/imageOptimizer.js` | 260×260 zero-crop client optimizer |
| `breed-classifier-react/src/utils/cropAndQuality.js` | Crop math, quality gate, share-card renderer |
| `breed-classifier-react/src/utils/sessionStore.js` | Herd session persistence and exports |
| `breed-classifier-react/src/hooks/useImageOptimizer.js` | Upload lifecycle: original vs optimized, framing, fit config |
| `breed-classifier-react/src/components/AdvancedPanel.jsx` | Advanced surface: model input, metadata, fit modes, batch |
| `breed-classifier-react/src/components/CameraCapture.jsx` | Full-frame camera, quality gate, capture review, handoff |
| `breed-classifier-react/src/components/ResultCard.jsx` | Tiers, trait matrix, quality, share card, diagnostics |
| `breed-classifier-react/src/data/breedTraits.js` | 44 curated breeds × 8 trait fields |
| `breed-classifier-react/src/i18n/translations.js` | EN + தமிழ️ copy deck |
| `local_train.py` | 8-stage end-to-end automation |
| `ADR.md` | Architecture decision record with rejected alternatives |
| `docs/` | 18-page mkdocs site (installation → deployment → troubleshooting) |

---

*Report generated from the repository at version v29. Figures marked “observed” come from local verification runs (browser E2E, API calls, calibration output and server logs) rather than published benchmarks.*
