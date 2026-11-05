# Development History & Replication Guide

**Project:** BreedID Livestock Vision — cattle/buffalo breed classifier with a
Flask backend and a standalone Capacitor Android app.

**Last updated:** 2026-11-05  
**Built from:** `7aab043`  
**Current Android artifact:** `artifacts/android/breedid-android-debug.apk`  
**Artifact SHA-256:** `D8601292AF20756DF4695CDE51FB8A54F007CB630BF0FC73ABC029455C974259`

---

## 1. Project at a glance

| Area | Implementation |
|---|---|
| Task | 68-class breed recognition: 50 cattle + 18 buffalo, plus binary species routing |
| Backbone | EfficientNet-Lite2 (default) / Lite4, CBAM or SE attention |
| Heads | Binary species, 50-class cattle, 18-class buffalo, training-only projection head |
| Input | 260 × 260 RGB, pad-to-square, ImageNet normalization unless baked into the ONNX graph |
| OOD | Energy + MSP + binary-confidence floor, calibrated per model |
| Explainability | Optional X-ray/heatmap, breed traits, confidence bands |
| Backend | Flask + Socket.IO, `/api/*` REST, `/ws` namespace, port 5000 |
| Web | React 18 + Vite 5, farmer app and developer console; separate vet portal on 5174 |
| Android | Capacitor 8 WebView, on-device ONNX Runtime Web, IndexedDB persistence |
| Localization | English, Tamil, Hindi — 621 keys per language |
| Themes | Emerald Obsidian, Meadow Daylight, Earthy Harvest, Aqua Pasture |
| Setup | Smart Python venv + Node/npm + Capacitor sync + optional APK build |

---

## 2. Chronological implementation history

### 2.1 Model and metadata

- Built the EfficientNet-Lite + CBAM/SE multi-head classifier with soft species
  routing.
- Added the self-describing model contract: class maps, input resolution,
  normalization mode and OOD thresholds embedded in `.pt` and `.onnx`.
- Added `src/model_introspect.py`, `scripts/embed_metadata.py`, and
  `docs/model-metadata.md`.
- Exported ONNX FP32 / INT8 / FP16 and portable bundles.
- Added the developer model tools: inspect, embed and model-specific test via
  `POST /api/models/{inspect,embed,test}`.

### 2.2 OOD and guardrails

- Added post-hoc OOD calibration (energy, MSP, binary floor) and per-model JSON
  profiles, later embedded into model metadata.
- Added four offline guardrail layers:
  1. photorealism / cartoon / screenshot rejection,
  2. human / selfie / other-animal gate (OpenCV Haar + optional CLIP),
  3. feature-entropy / logit sanity check,
  4. logit OOD check with trilingual rejection messages.
- Added CLIP-based zero-shot gender/age attributes with graceful fallback.
- Added `docs/input-guardrails.md`, `docs/attributes.md`, and
  `docs/image-quality.md`.

### 2.3 Web and vet portal

- Built the farmer web app with upload/camera, guided framing, quality report,
  result tabs, breed encyclopedia, farmer advisory, escalation queue and ODT
  exports.
- Added the developer console (`/dev`) and the standalone vet portal.
- Added vet-controlled user visibility, shared config, escalation database and
  offline outbox.
- Added 4 themes and trilingual UI on the website.

### 2.4 Capacitor Android app

- Wrapped the React bundle in Capacitor 8 without a React Native/Flutter
  migration.
- Added `AndroidUserPage` as a single portal: no `/dev`, no login.
- Added on-device ONNX Runtime Web inference with WASM files in `public/ort/`.
- Added IndexedDB stores for models, OOD profiles and settings.
- Added the bundled `default_model.onnx` with embedded calibrated OOD.
- Added model upload (`onnx` runs; `.pt` is visible but disabled until export).
- Added OOD profile upload, thresholds and profile selection.
- Added the guided camera with cattle/buffalo silhouette HUD, lighting/framing/
  sharpness pills, torch, flip, auto-capture and burst.
- Added recent scans (last 5), share/clipboard, haptics and wake-lock.

### 2.5 Early Android polish

- Gender and age made read-only; manual override pills removed.
- Gallery uploads decoded once with `createImageBitmap`; preview uses object
  URLs, removing the double base64/FileReader delay.
- “Preparing image…” feedback added.
- Result simplified to four cards; top-3 bars filtered to the predicted species.
- OOD banner removed from the result; embedded thresholds take precedence.

### 2.6 Stitch-driven UI rebuild

- Imported the Stitch design system/screens and rebuilt the Android visual layer:
  - two-line `BreedID / LIVESTOCK VISION` header,
  - language chip, online pill, hamburger,
  - stacked gallery/camera ingest cards,
  - ONNX telemetry card,
  - 3-column recent-scans ledger,
  - Identification Dossier result screen,
  - full-screen Settings page.
- Added a shared inline SVG icon set (`AndroidIcons.jsx`).
- Replaced malformed/emoji brand and action icons.

### 2.7 Localization pass

- Added a full `android.*` translation namespace in English, Tamil and Hindi.
- Wired dashboard, dossier, settings and status messages through `t()`.
- Dynamic species, breed, gender and age use `tDynamic` / `tBreed`.
- Language is stored in localStorage and IndexedDB settings and applies
  immediately.
- Fixed a double-encoded UTF-8 corruption in `AndroidUserPage.jsx`.

### 2.8 Result display controls

- Added:
  - **Show prediction view** — off hides the top-N spectrum and shows only the
    main breed result.
  - **Show confidence percentages** — hides percentage text while keeping bars.
  - **Predicted breeds to show** — 1–5, or **Main result only** when the
    prediction view is off.
- Added `showPredictionView`, `showConfidence`, `topBreedsCount` to
  `DEFAULT_SETTINGS`.
- Added the matching localized strings and settings UI.

### 2.9 Audio removal

- Removed the audio-help banner, speech state, speech handlers, audio icons and
  audio CSS from the Android app.
- Breed Tips remain text-only.

### 2.10 Online mode and backend fallback

- Online mode now mirrors the web API contract:
  `Settings URL → VITE_BACKEND_URL → last successful backend → current web origin`.
- `/api/health` is polled every 15 seconds while online mode is enabled.
- Header and Settings show **Connected / Reconnecting / Offline / Not
  configured**.
- Breed Tips are allowed only while the backend is connected.
- A successful health check caches the URL in
  `localStorage["breedid_last_backend_url"]`.
- Clearing the API field reuses that cached backend, so tips keep working without
  re-entering the URL.
- No machine IP is hardcoded. A build-time public URL can be supplied with
  `VITE_BACKEND_URL`.

### 2.11 Setup and venv

- `requirements.txt` now includes the guardrail/CLIP stack:
  `opencv-python-headless>=4.9,<5` and `transformers>=4.40`.
- `setup_venv.py` installs Python plus Node dependencies, builds the web bundle
  when needed, runs `npx cap sync android`, and can compile the APK with
  `--android-build`.
- New flags: `--no-frontend`, `--no-android`, `--android-build`.
- Multi-constraint requirements (`>=4.9,<5`) and OpenCV/OpenCV-headless conflict
  handling are supported.
- `setup.sh`, `setup.ps1` and `setup.bat` forward the flags and document Node 22+,
  JDK 21 and Android SDK prerequisites.
- `scripts/status_report.py` now honours multi-constraint requirements.

### 2.12 Object/monitor guardrail and biometric age/gender (2026-11-05)

- Added `src/utils/imageGuardrails.js`:
  - Stage A Sobel axis-edge/bezel detection, contiguous straight-edge spans,
    screen glow/blue-cast and 16×16 flat-block wall checks.
  - Stage B 68-class entropy, top-logit Z-score peak, binary confidence, energy
    and MSP checks.
  - Rejections return no species/breed and a localized reason.
- Calibrated gate thresholds against the shipped `lite2_v27` model because the
  original brief values rejected every valid test cattle image.
  Final: `EDGE_SPAN_MAX 0.62`, entropy `0.76`, `Z_peak 2.0`, `Z_max 1.6`,
  binary floor `0.72`, energy floor `-3.2004`.
- Fixed on-device MSP to use the binary species head, matching the Python
  `OODDetector.compute_msp`; energy now uses all three heads.
- Added `src/services/biometricAgeGender.js`: juvenile/heavy breed logit mass,
  torso fill, crown/horn texture, compactness, draught/dairy breed prior and
  neck/hull ratio → age/gender with 65–94% confidence.
- Added rejection UI and excluded rejected scans from Recent Scans.
- Verified in the browser harness: 4 cattle images accepted; app screenshot
  (monitor UI) and goat image rejected. Device was disconnected during the final
  APK build, so install verification is pending.

---

## 3. Current repository map

| Path | Responsibility |
|---|---|
| `src/model.py` | Backbone, attention, multi-head classifier, OOD detector |
| `src/preprocess.py` | Pad-to-square + inference transforms |
| `src/train.py` | Two-phase training, multi-task loss, EMA, distillation |
| `src/evaluate.py` | Per-class confusion matrices and metrics |
| `src/input_guardrails.py` | Four-layer image validation |
| `src/model_introspect.py` | Self-describing model contract |
| `src/export.py` | ONNX/TFLite/INT8/FP16/portable exports |
| `server/` | Flask app, routes, model manager, Socket.IO events |
| `breed-classifier-react/` | Farmer web app + Capacitor Android shell |
| `breed-classifier-react/src/pages/AndroidUserPage.jsx` | Native single portal |
| `breed-classifier-react/src/components/SettingsModal.jsx` | Full-screen settings |
| `breed-classifier-react/src/components/CameraCapture.jsx` | Guided camera |
| `breed-classifier-react/src/services/onDeviceInference.js` | ORT Web, IndexedDB, OOD |
| `breed-classifier-react/src/services/imagePrep.js` | Crop/resize pipeline |
| `breed-classifier-react/src/services/apiBase.js` | Public/fallback API base |
| `breed-classifier-react/src/hooks/useBackendHealth.js` | `/api/health` polling |
| `breed-classifier-react/src/i18n/translations.js` | en/ta/hi copy |
| `vet-portal/` | Separate vet review app |
| `artifacts/android/` | Tracked APK artifact + checksums |

---

## 4. Fresh-machine setup

### Prerequisites

- Python 3.10+ (3.14 tested)
- Node.js 22+ and npm (Capacitor 8)
- JDK 21
- Android SDK platform/build-tools, `adb`
- Git

### One-command setup

Windows:

```powershell
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Linux/macOS:

```bash
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier
bash setup.sh
```

What the installer does:

1. Reuses or creates `.venv`.
2. Installs only missing/outdated Python packages.
3. Selects CPU/GPU PyTorch wheels.
4. Installs `breed-classifier-react` and `vet-portal` Node dependencies.
5. Builds the web bundle when needed and runs `npx cap sync android`.
6. Optionally builds the APK with `--android-build`.
7. Runs `python -m src.verify` unless `--no-verify`.
8. Prints the status report.

Useful flags:

```text
--check          report only, change nothing
--cpu / --gpu    force PyTorch backend
--cuda-tag TAG   cu118 / cu121 / cu124 / cpu
--reinstall      repair PyTorch + refresh requirements
--no-frontend    Python only
--no-android     skip Capacitor sync
--android-build  compile the debug APK
--no-verify      skip src.verify
--report         run scripts/status_report.py
```

### Android build environment

```powershell
$env:ANDROID_HOME="$env:LOCALAPPDATA\Android\Sdk"
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$env:JAVA_HOME="C:\Program Files\Java\jdk-21"
```

Build/install:

```powershell
cd breed-classifier-react
npm run build
npx cap sync android
cd android
.\gradlew.bat assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

### Public internet backend

The Android app cannot use the PC’s LAN IP as a permanent backend. Use one of:

1. **Build-time public URL**

   ```powershell
   $env:VITE_BACKEND_URL="https://your-backend.example.com"
   cd breed-classifier-react
   npm run build
   npx cap sync android
   ```

2. **Runtime Settings URL**
   `Settings → Online Mode → Backend URL`.

3. **Cached fallback**
   With the field empty, the app reuses the last URL that successfully passed
   `/api/health`. A successful URL is cached automatically.

On the web build, an empty URL falls back to the current origin, matching the PC
website `/api` proxy behaviour.

---

## 5. Verification checklist

### Python

```powershell
.\.venv\Scripts\python.exe setup_venv.py --check
.\.venv\Scripts\python.exe -m src.verify
.\.venv\Scripts\python.exe -m tests.run_vv_suite --skip-benchmark
```

Expected:

- setup check: `Requirements already satisfied`
- `src.verify`: `VERIFICATION PASSED`
- V&V: `PASS 28, FAIL 0, WARN 2` (external-dataset OOD rate and skipped benchmark)

### Client

```powershell
cd breed-classifier-react
npm run build
npm test
node src\verify_ui_i18n.mjs
```

Expected:

- build succeeds
- tests: 57 pass, 0 fail
- i18n: 621/621 keys in en/ta/hi, all script/dynamic checks pass

### Device flow

1. Install the APK.
2. Grant camera permission.
3. Home: confirm recent scans, model name, engine status.
4. Upload a gallery image; confirm Original vs Model input 260×260.
5. Try Pad, Center and Manual crop.
6. Analyze; confirm Species, Breed (+ top-N), Gender, Age.
7. Toggle **Show prediction view** and **Show confidence percentages** in
   Settings → Advanced → Result Display.
8. Switch English / தமிழ் / हिन्दी; confirm labels and breed names.
9. Open Breed Tips with Online Mode connected.
10. Camera: silhouette HUD, status pills, shutter, torch/flip, review.

### Transfer test assets

```powershell
adb shell mkdir -p /sdcard/BreedClassifierTransfer/mutated_models
adb shell mkdir -p "/sdcard/BreedClassifierTransfer/test_data"
adb push outputs\mutated_models\onnx /sdcard/BreedClassifierTransfer/mutated_models/
adb push outputs\mutated_models\pt /sdcard/BreedClassifierTransfer/mutated_models/
adb push outputs\mutated_models\manifest.json /sdcard/BreedClassifierTransfer/mutated_models/
adb push "test data ceck" /sdcard/BreedClassifierTransfer/test_data/
```

---

## 6. Known constraints

- On-device inference runs **ONNX only**; `.pt` files are listed but disabled.
- The cached backend fallback is convenient but will fail if the PC’s IP/port
  changes. Enter a new URL or use a public `VITE_BACKEND_URL` for durable use.
- OOD is computed on-device but the result UI intentionally does not show a large
  rejection banner.
- CLIP weights download on first use unless cached; CLIP is optional.
- OpenCV is pinned `<5` because OpenCV 5 no longer bundles the Haar cascades.
- The Android WebView requires a modern Chrome/System WebView for ORT WASM and
  `createImageBitmap`.
- The old emoji assets are gone; use `AndroidIcons.jsx` for new UI icons.

---

## 7. Testing devices

| Device | Serial | OS | Notes |
|---|---|---|---|
| iQOO I2302 | `10BE9S223P001KK` | Android 16, 1080×2400 @440 dpi | Final APK installed and smoke-tested |

---

## 8. Artifacts

| File | Purpose |
|---|---|
| `artifacts/android/breedid-android-debug.apk` | Tracked debug APK |
| `artifacts/android/SHA256SUMS.txt` | Integrity checksum |
| `artifacts/android/build-info.json` | App ID, commit, build time, size, hash |
| `artifacts/android/README.md` | Install and verification instructions |

Install the tracked artifact:

```powershell
adb install -r artifacts\android\breedid-android-debug.apk
```

Verify:

```powershell
Get-FileHash artifacts\android\breedid-android-debug.apk -Algorithm SHA256
```
