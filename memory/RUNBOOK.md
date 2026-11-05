# Runbook

Windows + PowerShell environment. If `npm.ps1` is blocked by execution policy,
wrap commands with `cmd /c` as shown.

## Repository

```powershell
cd C:\Users\shiva304\Documents\ML-CB-B-identifier
git status -sb
```

## App: install dependencies and run checks

```powershell
cd breed-classifier-react
cmd /c "npm install"
cmd /c "npm run build"
cmd /c "npm test"
node src\verify_ui_i18n.mjs
```

Expected: build passes, tests pass, i18n reports 643/643 for en/ta/hi.

## App: rebuild the tracked Android APK

```powershell
cd C:\Users\shiva304\Documents\ML-CB-B-identifier\breed-classifier-react
cmd /c "npm run build"
cmd /c "npx cap sync android"
cd android
$env:ANDROID_HOME='C:\Users\shiva304\AppData\Local\Android\Sdk'
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$env:JAVA_HOME='C:\Program Files\Java\jdk-21'
cmd /c "gradlew.bat assembleDebug"

$apk = Get-Item app\build\outputs\apk\debug\app-debug.apk
Copy-Item $apk.FullName C:\Users\shiva304\Documents\ML-CB-B-identifier\artifacts\android\breedid-android-debug.apk -Force
$tracked = Get-Item C:\Users\shiva304\Documents\ML-CB-B-identifier\artifacts\android\breedid-android-debug.apk
"bytes=$($tracked.Length) modified=$($tracked.LastWriteTime.ToString('yyyy-MM-ddTHH:mm:sszzz'))"
(Get-FileHash $tracked.FullName -Algorithm SHA256).Hash
```

Important: run `npx cap sync android` before `assembleDebug`, otherwise the APK
can be "up to date" with the previous web bundle.

Then update:

- `artifacts/android/build-info.json`
- `artifacts/android/SHA256SUMS.txt`
- `artifacts/android/README.md`
- `docs/development-history.md` artifact hash
- `memory/CURRENT_STATE.md` APK block

## Device: install and verify

```powershell
$adb = "C:\Users\shiva304\AppData\Local\Android\Sdk\platform-tools\adb.exe"
& $adb devices -l
& $adb install -r C:\Users\shiva304\Documents\ML-CB-B-identifier\artifacts\android\breedid-android-debug.apk
& $adb shell input keyevent 3
& $adb shell am force-stop com.agritech.breedclassifier
& $adb shell am start -n com.agritech.breedclassifier/.MainActivity
```

If the screen is asleep or the shade is stuck:

```powershell
& $adb shell input keyevent 26
& $adb shell input swipe 540 2000 540 500 300
& $adb shell input keyevent 3
& $adb shell cmd statusbar collapse
```

Device physical resolution is `1080x2400`; screenshots are often `900x2000`.
Scale taps by 1.2 when reading screenshot coordinates.

## Advisory API: tests and local server

```powershell
cd C:\Users\shiva304\Documents\ML-CB-B-identifier\advisory-api
cmd /c "npm test"
cmd /c "npm run dev"
# http://localhost:3001/api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=en
```

Regenerate the compact 68-breed data:

```powershell
cmd /c "npm run build:data"
```

## Vercel: deploy and inspect

Preferred deploy path is Git: commit and push to `main`. Vercel builds from the
GitHub repo with Root Directory `advisory-api`.

```powershell
cd C:\Users\shiva304\Documents\ML-CB-B-identifier
git push
cd advisory-api
cmd /c "vercel ls advisory-api"
```

Check project settings:

```powershell
cmd /c "vercel project inspect advisory-api"
cmd /c "vercel project protection advisory-api --json"
```

The correct state is:

- Root Directory: `advisory-api`
- `ssoProtection`: `null`

Set them if needed:

```powershell
cmd /c "vercel project update advisory-api --root-directory advisory-api --yes"
cmd /c "vercel project protection disable advisory-api --sso"
```

Check the live endpoints:

```powershell
$root = "https://advisory-api-bharaths31s-projects.vercel.app/"
$api  = "https://advisory-api-bharaths31s-projects.vercel.app/api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=en"
Invoke-WebRequest -UseBasicParsing $root
Invoke-WebRequest -UseBasicParsing $api
```

Expected: `200 text/html` and `200 application/json`.

### Deploy pitfalls

- Do not run `vercel deploy` from inside `advisory-api` while the project Root
  Directory is also `advisory-api`; it can report that the root directory does
  not exist.
- Running `vercel deploy` from the repository root uploads the whole repo,
  including large artifacts, and may abort. Prefer the Git integration.
- The short `advisory-api-rho.vercel.app` alias can be transient. The canonical
  team URL above is the app default.
- `vercel link` creates a root `.env.local` with an OIDC token. It is ignored; do
  not commit it.

## Guardrail verification harness

The guardrail was verified in the browser against prepared `260×260` canvases.
The test assets were copied under `public/test/` temporarily and removed before
commits. A future run needs:

1. Start the Vite dev server.
2. Serve a cattle image, a human selfie, a goat image and a monitor screenshot.
3. Import `prepareImage` and `runOnDeviceInference`.
4. Assert cattle pass and human/monitor reject.

Key functions:

- `checkHumanSelfie(canvas)` — async; FaceDetector + YCbCr fallback.
- `geometryGate(canvas)` — center-focused Stage A.
- `evaluateImageGuardrails(...)` — async human + geometry + logits.
- `verifyDynamicPredictions({ canvases, settings, model })` — two-canvas
  uniqueness/latency smoke test.

## Commit hygiene

- Never commit `.env.local`, `.vercel/`, tokens or device serials.
- Refresh the APK and its metadata in the same change set.
- Keep `memory/CURRENT_STATE.md` current after deployment or APK refreshes.
