# Progress

## 2026-11-05

### 1. On-device guardrail and dynamic age/gender

- Added Stage A whole-frame geometry checks and Stage B logit checks in
  `breed-classifier-react/src/utils/imageGuardrails.js`.
- Added `dynamicAgeGender.js`; removed the old foreground-mask biometric
  estimator.
- Integrated guardrails into `onDeviceInference.js`; rejected results contain no
  species/breed and are not saved to Recent Scans.
- Added rejection UI, localized reasons and retry action.
- Verified with real cattle, goat and monitor images.

### 2. Direct 260×260 input, native camera, simple developer mode

- Removed the Android `getUserMedia` camera modal and silhouette path; Android now
  uses native file inputs with `capture="environment"`.
- Both gallery and camera decode once and draw directly to `260×260`.
- Removed the Active ONNX model panel from Home.
- Added Developer Mode in Advanced Settings (default off).
- Simple mode hides ONNX/WASM, latency, rank/model-match, top-N spectrum,
  percentages, scientific names, method text, online status and diagnostics.
- Removed silhouette HUD/state from the shared web `CameraCapture`.
- Verified simple mode on device and screenshot.

### 3. Center-focused guardrail, selfie blocker, top-k OOD

- Stage A geometry moved to a central 180×180 region, with only high-threshold
  global fallbacks. This prevents barn roofs, fences and sky at the border from
  rejecting valid animals.
- Added `checkHumanSelfie()`:
  - browser `FaceDetector` when available;
  - YCbCr fallback using central skin ratio, 5×5 skin luma sigma, bilateral
    symmetry, dark hair cap and eye/brow density.
- Replaced single-class `Z_peak`/`Z_max` Stage B rejection with top-3 and top-5
  breed-mass gates plus a relaxed entropy ceiling.
- Verification: 34 cattle images accepted, human selfie and monitor rejected.
- Known trade-off: the goat test image is now accepted.

### 4. Free Vercel advisory API

- Added `advisory-api/`:
  - `api/advisory.js` serverless handler with CORS, validation and caching;
  - `data/breed_knowledge.json` with 50 cattle + 18 buffalo;
  - `lib/i18n.js` for en/ta/hi advisory copy and card logic;
  - `lib/knowledge.js` for lookup/aliases;
  - local dev server and 7 node:test checks.
- Dynamic rules: female adult lactation, male adult draught, calf colostrum and
  vaccination, young growth, general fallback.
- Added `breed-classifier-react/src/services/onlineAdvisory.js` with 5-second
  timeout, offline fallback and non-throwing results.
- Added the live Breed Tips modal using the Vercel payload.

### 5. Vercel deployment and 404 fixes

- First `vercel deploy --temporary` returned a working URL.
- Authenticated CLI later identified as `bharaths31`.
- Root URL returned 404 because there was no `/` page.
- API returned 404 after authenticated deploys because the Vercel project was
  building the repository root, not `advisory-api`.
- Set project Root Directory to `advisory-api`.
- Added `public/index.html` with a live advisory form.
- Disabled Vercel SSO deployment protection.
- Redeployed via Git; root and API now return 200.
- Switched the app default to the canonical team URL.
- Removed the legacy Online Mode/backend-health gate from Breed Tips; only
  internet connectivity is required.

### 6. Final APK and device verification

- Rebuilt, synced and installed the APK.
- Verified simple home, result screen and live Breed Tips on the iQOO device.
- Refreshed the tracked APK and metadata.

## Commit map (newest first)

```text
edc4590 chore: refresh APK for online Breed Tips fix
4c60daa fix(android): open online breed tips without legacy online-mode gate
8639a01 chore: refresh APK for canonical advisory URL
ce30ccb fix(api): serve public landing page and canonical advisory URL
36f8b35 feat(api): add live advisory landing page
c863137 chore: refresh APK for live Vercel advisory build
f5f0022 feat(android): wire live Vercel advisory into Breed Tips
7c23ad0 chore(api): expose utility_type in breed payloads
45fc2b9 chore: refresh tracked APK for guardrail and advisory build
d9c8b77 feat(api): free Vercel breed advisory API and Android client
07f0b8b feat(android): center-focused guardrail, selfie blocker, top-k OOD
210259f feat(android): reject human close-ups, simplify farmer UI, direct 260 input
```

## Screenshots

```text
outputs/screenshots/stitch_rebuild/29_simple_device_final.png
outputs/screenshots/stitch_rebuild/31_result.png
outputs/screenshots/stitch_rebuild/32_breed_tips_live.png
outputs/screenshots/stitch_rebuild/34_breed_tips_summary.png
outputs/screenshots/stitch_rebuild/35_breed_tips_fixed.png
outputs/screenshots/stitch_rebuild/36_breed_tips_canonical.png
outputs/screenshots/stitch_rebuild/37_breed_tips_final.png
outputs/screenshots/stitch_rebuild/38_vercel_landing.png
```
