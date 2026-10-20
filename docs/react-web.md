# Web Application (React)

The React app in `breed-classifier-react/` is the primary human-facing surface:
a public capture-and-classify page plus a developer control plane. It talks to
the same Flask API (`server/`, port 5000) that the Colab and `test_model.py`
workflows use.

| Aspect | Detail |
|---|---|
| Stack | React 18 (hooks only), Vite 5, React Router 7, `socket.io-client`, hand-written CSS design system |
| Routes | `/` → public app (`UserPage`), `/dev` → developer control plane (`DevPage`) |
| Styling | 5 themes (`dark`, `light`, `emerald`, `sunset`, `cyber`) via CSS custom properties, persisted |
| Languages | English + தமிழ் (`src/i18n/`), persisted, no external service |
| Tests | `npm test` → `node --test`, 30 tests across 3 spec files (no test framework dependency) |
| Build | `npm run build` → static bundle served by any web server; dev proxy forwards `/api` and `/socket.io` to the backend |

---

## 1. Quick start

```bash
# terminal 1 — backend
python server/app.py --port 5000

# terminal 2 — web app
cd breed-classifier-react
npm install
npm run dev            # http://localhost:5173
```

Other useful commands:

```bash
npm test               # unit tests (crop math, optimizer layout, session store)
npm run build          # production bundle
npm run preview        # serve the built bundle
```

---

## 2. The capture → classify path

```
file / camera
  └─ ImageUpload
       ├─ keeps the ORIGINAL image (never replaced by the model input)
       ├─ optional framing via ImageCropper (1:1, pan on both axes, 1–3× zoom)
       └─ useImageOptimizer → 260×260 zero-crop file
            ├─ stretch (default)  | pad · ImageNet mean | pad · ambient blur | pad · black
            ├─ EXIF-upright decode, progressive halving
            └─ meta: original size, inner scaled size, padding, payload, time
                 ↓
        POST /api/predict  (image, model, original filename)
                 ↓
        ResultCard
          ├─ 3-tier confidence banner (verified / look-alike / abstention)
          ├─ ranked breeds + top-1/top-2 margin
          ├─ “Why this breed?” trait matrix (44 breeds × 8 fields)
          ├─ capture-quality badge
          ├─ shareable PNG result card
          └─ technical diagnostics (model, OOD thresholds, file name, latency)
                 ↓
        Herd session log (localStorage) → CSV / JSON / ODT export
```

### Why the client optimises

The server evaluates with `Resize(260) → CenterCrop(260)`. A 16:9 upload would
lose ~44% of the long axis. Because the client already sends a 260×260 square,
both server steps become identity transforms and the animal is never cropped.
The pad colour is the ImageNet mean `rgb(124,116,104)`, which normalises to
approximately zero.

---

## 3. User surface

### 3.1 Upload

* The card shows **only the original image**.
* Actions: **Adjust framing**, **Remove**, **Choose another**, and the primary
  **Identify Breed** button — which becomes **Identify Framed Breed** after a
  crop, so the classify step is always explicit.
* Applying a new upload, crop or fit change clears the previous prediction
  (a stale result is never shown next to a new input).

### 3.2 Live camera

* Full-frame viewfinder (`object-fit: contain`) with a guide framing the whole
  sensor frame and the badge *“Full Frame Active • Auto-Optimized to 260×260
  (Zero Crop)”*.
* On-device quality gate (sharpness, exposure, framing) drives a guidance pill.
* Auto-capture after 3 stable checks (~750 ms) with a countdown ring.
* Burst ×3 keeps the sharpest frame; torch is exposed when the device supports it.
* The review screen shows the 260×260 result, the telemetry
  (`1920×1080 → 260×260 • 0% Cropped • 22 KB`), Pad/Strict switch and an
  optional manual framing pass.

### 3.3 Result card

| Tier | Condition | Presentation |
|---|---|---|
| 1 — verified | not OOD, confidence ≥ 0.72, top-1/top-2 margin ≥ 0.12 | green badge, “top match well separated” |
| 2 — look-alike | not OOD, otherwise | amber badge + trait comparison + retake action |
| 3 — abstention | OOD | red badge, the exact trigger (energy / MSP / binary floor) and a retake action |

### 3.4 Herd session

Every confirmed classification is appended to a local session with a thumbnail,
timestamp, model, species, breed, confidence, OOD flag and capture quality. The
drawer shows totals, the cattle/buffalo split, flagged count and averages, and
exports **CSV**, **JSON** and **ODT**.

---

## 4. Advanced panel (header, next to Connected)

Everything technical is one click away:

| Block | Contents |
|---|---|
| Preview | *View model input* ⇄ *View original* |
| Metadata | `ORIGINAL`, `MODEL INPUT`, `SCALED`, `PADDING`, `PAYLOAD`, `TIME`, original file name |
| Fit mode | `Stretch 260×260` (default), `Pad · ImageNet mean`, `Pad · Ambient blur`, `Pad · Black` |
| Batch test | Multi-image run with the **selected model and fit configuration**, per-image results, payload saved, optional append to the herd session |

---

## 5. Developer control plane (`/dev`)

| Tab | Purpose |
|---|---|
| Models | Registry, active-model switch, per-model spec (parameters, layer breakdown, ONNX I/O), checkpoint upload |
| OOD Safety | Enable/disable, profile select/upload, live threshold sliders, **manual recalibration from a breed-named folder** |
| Config | Shared settings database (display, dataset, inference, OOD) |
| Logs | Live server log tail |

### Manual OOD recalibration

1. Pick the model that is deployed.
2. Pick a folder of breed-named images — flat (`Deoni_01.jpg`) or nested
   (`cattle/deoni/…`). The picker is pre-filled with the folders found in the
   repository.
3. Choose the percentile (default 95) and the maximum number of images.
4. Recalibrate. The response reports the new thresholds, the number of samples,
   matched/unmatched/unreadable counts and the per-breed coverage, and the
   profile is written to `outputs/ood/<model>.json`.

Unmatched names are reported instead of being used, so a mis-named folder is
visible rather than silently skewing the thresholds. Calibration requires a
PyTorch `.pt` checkpoint (ONNX models return a clear error).

---

## 6. Module map

| Path | Responsibility |
|---|---|
| `src/pages/UserPage.jsx` | Public page: tabs, preview, explicit classify action, result, session, Advanced panel |
| `src/pages/DevPage.jsx` | Control plane shell |
| `src/components/ImageUpload.jsx` | Minimal upload surface (original image + actions) |
| `src/components/CameraCapture.jsx` | Full-frame camera, quality gate, capture review, handoff |
| `src/components/ImageCropper.jsx` | Shared 1:1 framing cropper (both-axis pan, zoom, quality) |
| `src/components/AdvancedPanel.jsx` | Model input, metadata, fit modes, batch entry point |
| `src/components/AdvancedBatchTester.jsx` | Batch inference with the selected configuration |
| `src/components/ResultCard.jsx` | Tiers, trait matrix, quality badge, share card, diagnostics |
| `src/components/TraitComparison.jsx` | Shared look-alike trait table |
| `src/components/QualityBadge.jsx` | 0–100 sharpness / lighting / framing breakdown |
| `src/components/HerdSessionDrawer.jsx` | Session log, statistics, CSV/JSON/ODT export |
| `src/components/DevOODPanel.jsx` | OOD laboratory incl. manual folder recalibration |
| `src/components/DevModelPanel.jsx` | Model registry and spec viewer |
| `src/components/ThemeToggle.jsx`, `LanguageToggle.jsx` | 5 themes, EN/தமிழ் |
| `src/hooks/useImageOptimizer.js` | Original vs optimized state, fit config, crop lifecycle |
| `src/hooks/useSharedConfig.js`, `useSocket.js` | Server settings and realtime events |
| `src/utils/imageOptimizer.js` | 260×260 zero-crop optimizer |
| `src/utils/cropAndQuality.js` | Crop math, quality gate, share-card renderer |
| `src/utils/sessionStore.js` | Herd session persistence and exports |
| `src/data/breedTraits.js` | 44 curated breeds × 8 trait fields |
| `src/i18n/` | Translations and the provider |
| `src/styles.css` | Themeable design system |

---

## 7. Deployment notes

* The built bundle (`dist/`) is static — serve it from any web server and point
  it at the API host (same-origin `/api` or a reverse proxy).
* The dev proxy forwards `/api` and `/socket.io` to `http://localhost:5000`.
* For an offline kiosk, open `/?mode=camera` to start straight in the camera.
* The app never uploads an image the user did not submit, and the herd session
  never leaves the browser unless the user exports it.
