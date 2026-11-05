# Android artifact

Tracked debug APK for the BreedID Livestock Vision Capacitor app.

| Field | Value |
|---|---|
| App ID | `com.agritech.breedclassifier` |
| Version | `1.0` (`versionCode 1`) |
| Min / target SDK | 24 / 36 |
| Built from | `f5f0022` |
| File | `breedid-android-debug.apk` |
| Size | 34,310,561 bytes |
| SHA-256 | `076B2A3A34683095A295A5697D2282BCE81C263663442E7C109A07FA2E8AB816` |

## Install

```powershell
adb install -r artifacts\android\breedid-android-debug.apk
```

Or use the stable project path:

```powershell
adb install -r breed-classifier-react\android\app\build\outputs\apk\debug\app-debug.apk
```

## Verify

```powershell
Get-FileHash artifacts\android\breedid-android-debug.apk -Algorithm SHA256
Get-Content artifacts\android\SHA256SUMS.txt
```

Expected hash:

```text
076B2A3A34683095A295A5697D2282BCE81C263663442E7C109A07FA2E8AB816
```

## Rebuild

```powershell
cd breed-classifier-react
npm run build
npx cap sync android
cd android
.\gradlew.bat assembleDebug
```

The build output is `android/app/build/outputs/apk/debug/app-debug.apk`. Copy it
over this tracked artifact after verifying the change, then update
`build-info.json` and `SHA256SUMS.txt`.

## Scope

This is a debug build. Inference runs fully on-device; the network is only used
for optional Breed Tips via the configured public API base or the cached
last-successful backend.
