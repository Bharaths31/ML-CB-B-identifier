# Android build (Capacitor)

The existing React + Vite app is wrapped by Capacitor; no React Native/Flutter
migration. The native build uses a **single user portal** (`AndroidUserPage`)
with a Settings drawer. `/dev` is not routed on Android.

## Prerequisites

- Node 22+ (Capacitor 8 requirement; the web build alone needs Node 18+)
- JDK 21
- Android SDK (platform 36+, build-tools 36+)
- A device with USB debugging, or an emulator

The repository setup script does the Node install and Capacitor sync for you:

```bash
bash setup.sh                 # or setup.ps1 / setup.bat on Windows
bash setup.sh --android-build # also compile the debug APK (JDK + SDK required)
```

Set the SDK/JDK for a terminal:

```powershell
$env:ANDROID_HOME="$env:LOCALAPPDATA\Android\Sdk"
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$env:JAVA_HOME="C:\Program Files\Java\jdk-21"
```

## Build & install

```bash
cd breed-classifier-react
npm install
npm run build
npx cap sync android

# Windows
cd android
gradlew.bat assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

# or from breed-classifier-react/
npm run android:assemble
```

APK: `android/app/build/outputs/apk/debug/app-debug.apk`.

Open the project in Android Studio with `npm run cap:open`.

## On-device model rules

| Upload | Behaviour |
|---|---|
| `.onnx` | Validated with ONNX Runtime Web and saved to IndexedDB. Appears in **Settings → Advanced → Active model** and runs offline. |
| `.pt` / `.pth` | Saved and shown in the selector with **(needs ONNX export)**, but disabled. Export it to ONNX in the desktop developer console first. |

The bundled default model is `public/models/default_model.onnx` (FP32,
caller-side ImageNet normalization). It is copied into IndexedDB on first run.

The model selector auto-scrolls into view after an upload, and the settings
message confirms the saved filename/size. Uploaded models persist across app
restarts.

## Bundled assets

- `public/models/default_model.onnx` – default on-device model
- `public/ort/ort-wasm-simd-threaded.{wasm,mjs}` – ONNX Runtime Web WASM
- `vite.config.js` copies the ORT WASM files into `dist/ort` at build time

## Runtime notes

- **Camera and uploads:** the Android app uses the native file/camera input
  (`capture="environment"`) for Take Photo and a gallery file input for Upload.
  Both paths decode once and downscale directly to the model's `260×260` canvas;
  no silhouette overlay, foreground mask, contour extraction or custom Android
  camera constraint/HUD is used.
- **QoL:** haptic feedback on capture/analyze/recent, screen wake-lock while
  inference is active, native share/clipboard for results, and a persisted
  **Recent scans** strip (last 5, stored locally) on the home page.
- **Result cards:** species, breed (+ Top-3), gender and age are read-only.
  Gender and age are always the model/heuristic output and cannot be
  overridden from the result screen.
- **Dossier layout:** the result is presented as an identification dossier with
  a record ID, inference-latency strip, analysed image, species card, breed card
  with rank/confidence/top-3 bars, and read-only gender/age cards. The top-3
  spectrum is filtered to the predicted species.
- **Home:** two-line brand header, online status, a header language chip
  (`EN / த / हि`), stacked gallery/camera ingest cards and a recent-scans ledger.
  The **Active ONNX model** panel was removed. A **Developer mode** switch in
  Advanced Settings controls whether model/OOD diagnostics are visible.
- **Simple mode (default):** with Developer mode off, the home page shows only
  the title, two source buttons and recent scans (when present); the result shows
  only species, breed, gender, age and actions. ONNX model name, WASM/SIMD,
  latency, rank, model-match badge, inference strip, Softmax/top-N spectrum,
  percentages, scientific names and method text are hidden.
- **Online mode:** matches the website’s connection model. It uses the same
  `/api` contract, resolving the base from the Settings URL, then
  `VITE_BACKEND_URL`, then the **last backend URL that passed `/api/health`**,
  then the current web origin. A successful check is cached, so clearing the
  Settings field reuses the previous working backend. When enabled, the app polls
  `GET /api/health`, shows **Connected / Reconnecting / Offline**, and only
  enables Breed Tips while the backend is reachable. The check repeats every 15
  seconds and follows backend URL changes. No machine IP is hardcoded.
- **Gallery uploads:** decoded once with `createImageBitmap` (EXIF-aware,
  off-main-thread) and previewed via an object URL, so large photos do not
  round-trip through base64 twice before the preview appears.
- **Settings:** a full-screen page opened from the header hamburger (☰). It has
  a dedicated **Select Language** section (English / தமிழ் / हिन्दी), theme
  cards, Online Mode + backend URL, and an **Advanced settings** section with a
  **Developer mode** switch. When Developer mode is off, model, OOD, result
  display and diagnostic controls are hidden; turning it on reveals them.
  **Save & Return** closes the page; all choices are persisted immediately.
- **Result display:** Advanced Settings includes *Show prediction view*,
  *Show confidence percentages* and *Predicted breeds to show* (1–5, default 3).
  Turning off the prediction view hides the entire top-N spectrum and shows only
  the main breed result; the count selector then reads **Main result only**.
- **Localisation:** dashboard, dossier, settings and status messages use the
  shared website `t()` dictionary (English / Tamil / Hindi), including species,
  gender, age and breed names via `tDynamic`.
- **Icons:** a shared inline SVG set (`AndroidIcons.jsx`) replaces the mixed
  emoji marks for the brand, gallery, camera, lock, trash, language, theme,
  online, advanced, gender, age and result actions.
- **Guardrail:** every on-device prediction passes a multi-stage gate
  (`src/utils/imageGuardrails.js`):
  - Human gate: `FaceDetector` when available, then a YCbCr fallback using
    central skin ratio, 5×5 skin smoothness, bilateral symmetry, dark hair cap
    and eye/brow density; close-up selfies return `human_or_selfie`.
  - Stage A: center-focused (180×180) Sobel straight-edge/bezel and flat-block
    checks, so barn roofs, fences and sky at the frame border do not reject a
    valid animal. A high-threshold global fallback still catches full-screen
    monitors/walls.
  - Stage B: 68-class normalized entropy (≤0.86), top-3 (≥0.20) / top-5 (≥0.30)
    breed mass, binary-confidence floor, energy and MSP. Top-k mass replaces the
    old single-class `Z_peak`/`Z_max` checks so visually similar breeds are not
    rejected when probability splits across the top classes.
  - Rejected images return no species/breed and show a localized
    “No cattle or buffalo detected” card; rejected scans are not saved to recent.
- **Age/gender:** `src/services/biometricAgeGender.js` fuses juvenile/heavy breed
  logit mass, foreground torso fill, crown/horn texture, and draught/dairy breed
  priors to produce read-only age and gender estimates with 65–94% confidence.
  On-device MSP uses the binary species head, matching the Python OOD detector.
- **OOD calibration:** a model with embedded `model_meta.ood` (the
  `*__ood.onnx` mutant or the bundled default) applies its own energy/MSP
  thresholds automatically. A non-OOD model is accepted but the OOD gate is
  disabled until you upload a matching OOD `.json` in Settings.
- **Online mode:** only Breed Tips use the network. The Android image pipeline is
  fully local; `server/app.py` inference is not called by the app.
- Preprocessing: `260×260`, `pixel/255`, then ImageNet mean/std unless the model
  name contains `mobile`/`int8` (baked normalization).
- Breed Tips use the public backend only when Online Mode is ON, the device has
  internet, **and** the backend `/api/health` check succeeds. The base URL comes
  from Settings, `VITE_BACKEND_URL`, or the current web origin, in that order.
  Otherwise a localized offline/backend notice is shown.
- Build the Android bundle against a public backend with:
  `VITE_BACKEND_URL=https://your-backend.example.com npm run build` (PowerShell:
  `$env:VITE_BACKEND_URL="https://your-backend.example.com"; npm run build`).
  The same URL can be entered at runtime in **Settings → Online Mode**.
- Android cleartext traffic is enabled for local development, but production
  online mode is expected to use an HTTPS URL.
