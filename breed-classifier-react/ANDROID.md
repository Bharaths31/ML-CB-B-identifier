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

- **Guided camera:** the camera opens the live `getUserMedia` preview with the
  cattle/buffalo silhouette HUD (head, hump/dorsal ridge, dewlap, tail), live
  lighting/framing/sharpness pills, torch, facing, auto-capture and burst.
  Android mode hides the manual crop/fit controls; the captured 260×260 frame
  flows straight to analysis.
- **QoL:** haptic feedback on capture/analyze/recent, screen wake-lock while the
  camera or inference is active, native share/clipboard for results, and a
  persisted **Recent scans** strip (last 5, stored locally) on the home page.
- **Result cards:** species, breed (+ Top-3), gender and age are read-only.
  Gender and age are always the model/heuristic output and cannot be
  overridden from the result screen.
- **Dossier layout:** the result is presented as an identification dossier with
  a record ID, inference-latency strip, analysed image, species card, breed card
  with rank/confidence/top-3 bars, and read-only gender/age cards. The top-3
  spectrum is filtered to the predicted species.
- **Home:** two-line brand header, online status, a header language chip
  (`EN / த / हि`), audio-help button (speech synthesis in the selected language),
  stacked gallery/camera ingest cards, ONNX telemetry, and a 3-column recent
  scans ledger with confidence badges and scientific names.
- **Gallery uploads:** decoded once with `createImageBitmap` (EXIF-aware,
  off-main-thread) and previewed via an object URL, so large photos do not
  round-trip through base64 twice before the preview appears.
- **Settings:** a full-screen page opened from the header hamburger (☰). It has
  a dedicated **Select Language** section (English / தமிழ் / हिन्दी), theme
  cards, Online Mode + backend URL, and a collapsible **Advanced AI Model**
  section for ONNX/OOD. Gallery downscale mode lives there too.
  **Save & Return** closes the page; all choices are persisted immediately.
- **OOD calibration:** a model with embedded `model_meta.ood` (the
  `*__ood.onnx` mutant or the bundled default) applies its own energy/MSP
  thresholds automatically. A non-OOD model is accepted but the OOD gate is
  disabled until you upload a matching OOD `.json` in Settings.
- Preprocessing: `260×260`, `pixel/255`, then ImageNet mean/std unless the model
  name contains `mobile`/`int8` (baked normalization).
- Breed Tips use the configured backend URL only when Online Mode is ON **and**
  the device has internet. Otherwise a localized offline notice is shown.
- Android cleartext traffic is enabled for local Flask/LAN backends.
