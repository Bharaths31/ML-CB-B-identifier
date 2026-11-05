# Decision Log

Each entry records the decision, the context, the rationale and the consequence.
This is a summary of engineering reasoning, not a raw chain-of-thought.

---

## D-001 — Use center-focused geometry instead of global filters

**Date:** 2026-11-05
**Status:** Accepted

**Context:** Global straight-edge and flat-patch checks rejected valid cattle when
the frame included a barn roof, fence, sky or other border clutter.

**Decision:** Run the man-made/screen checks on the central 180×180 area
(`CENTER_BOX`). Keep only high-threshold global fallbacks for obvious full-frame
monitors/walls.

**Why:** A screen or monitor fills the center; background structures usually sit
at the border. Center focus preserves the safety check while removing false
rejections from farm backgrounds.

**Consequence:** A monitor at the edge of the frame could pass Stage A, but Stage
B logits and the human gate still apply. The central region is the primary
decision area.

---

## D-002 — Replace single-class Z-peak rejection with top-k breed mass

**Date:** 2026-11-05
**Status:** Accepted

**Context:** Visually similar breeds split probability across the top 2–3
classes. The old `Z_peak`/`Z_max` single-class checks rejected many real cattle
even when the top-k mass was meaningful.

**Decision:** Use `top3_mass >= 0.20` and `top5_mass >= 0.30`, with a relaxed
entropy ceiling of `0.86`. Keep binary confidence, energy and MSP floors.

**Why:** Top-k mass measures whether the model has a coherent breed region even
when the single winner is not dominant. It matches how the app displays a Top-3
result.

**Consequence:** 34 cattle test images pass, including the `0.4688` flat-patch
cases. The goat test image also passes; this is a documented open item.

---

## D-003 — Keep binary MSP, energy and binary-confidence floors

**Date:** 2026-11-05
**Status:** Accepted

**Context:** The Python OOD detector and the on-device harness both use the binary
species head for MSP. Energy uses all three heads.

**Decision:** Keep those floors even after relaxing entropy/peak checks.

**Why:** They catch species-level uncertainty and out-of-distribution logits
without penalizing breed-level splits.

**Consequence:** On-device MSP matches the Python contract; OOD remains
configurable through embedded metadata or uploaded JSON.

---

## D-004 — Human selfie gate with FaceDetector plus YCbCr fallback

**Date:** 2026-11-05
**Status:** Accepted

**Context:** A selfie bypassed the monitor filter and was confidently classified
as Hallikar. `FaceDetector` is not available in the Electron harness or the
Android WebView used here.

**Decision:** Try `FaceDetector` when present; otherwise use a pure-JS fallback on
the `260×260` pixels:
- central 150×150 skin ratio;
- median 5×5 skin luma sigma;
- bilateral skin symmetry over a ±24 px axis search;
- dark hair cap ratio;
- eye/brow density.

**Why:** Human portraits have a large, smooth, bilaterally symmetric skin region
with dark facial features. Brown/red cattle coats are less symmetric and have
more fur texture or a side-profile orientation. The fallback keeps the app fully
offline.

**Consequence:** The tested selfie is rejected with `human_or_selfie`. The gate
is heuristic, not a trained face detector; a future model could improve it.

---

## D-005 — Keep age and gender read-only

**Date:** 2026-11-05
**Status:** Accepted

**Context:** A later request mentioned a one-click override pill, but the earlier
product requirement said gender/age must remain read-only.

**Decision:** Do not add the override pill. Continue computing age/gender
dynamically from current pixels and logits.

**Why:** The explicit earlier requirement wins until the user changes it.

**Consequence:** Result cards stay read-only. The dynamic estimator lives in
`dynamicAgeGender.js` and has no breed-lookup fallback.

---

## D-006 — Native Android camera input, direct 260×260 downscale

**Date:** 2026-11-05
**Status:** Accepted

**Context:** The custom `getUserMedia` camera modal and silhouette HUD added
complexity and did not match the native Android experience.

**Decision:** Use `<input type="file" accept="image/*" capture="environment">` for
Take Photo and the standard file input for gallery. Decode once and draw directly
to `260×260`.

**Why:** Fewer moving parts, native camera UI, consistent model input size and no
custom HUD/segmentation path.

**Consequence:** The shared web `CameraCapture` still exists but its silhouette
overlay was removed. Android no longer opens the custom camera modal.

---

## D-007 — Developer Mode hides technical UI by default

**Date:** 2026-11-05
**Status:** Accepted

**Context:** The farmer-facing screen showed ONNX/WASM, model name, latency,
Softmax/top-N, rank badges and OOD controls.

**Decision:** Add a Developer Mode switch (default off). Simple mode shows only
the title, two source buttons, recent scans (when present), species, breed,
gender, age and actions.

**Why:** The primary user needs a clear identification result, not model
telemetry. Diagnostics remain available for developers.

**Consequence:** Model/OOD/display controls, percentages, scientific names,
method text, online status and the inference strip are hidden in simple mode.

---

## D-008 — Use a static Vercel function and static JSON

**Date:** 2026-11-05
**Status:** Accepted

**Context:** The advisory data is read-only and small enough to bundle. The app
needs a free, low-latency, no-auth endpoint for rural users.

**Decision:** Build a zero-dependency CommonJS Vercel function plus a static
68-breed JSON file. No database, no API key, no server-side model.

**Why:** Free tier, instant cold starts, easy CORS and no operational burden.

**Consequence:** Breed data changes require a rebuild/redeploy. `npm run
build:data` regenerates the compact JSON from the project's curated knowledge
base.

---

## D-009 — Set Vercel Root Directory to `advisory-api`

**Date:** 2026-11-05
**Status:** Accepted

**Context:** Authenticated Vercel deploys cloned the repository root and produced
an empty build, so `/api/advisory` returned 404.

**Decision:** Set the Vercel project Root Directory to `advisory-api`.

**Why:** The serverless function and its data live in that subfolder; the repo
root contains large training/artifact files not meant for deployment.

**Consequence:** Git pushes build only `advisory-api`. CLI deploys from inside the
folder can double-nest the path, so future deploys should use the Git integration
or a root-linked/prebuilt flow.

---

## D-010 — Disable Vercel SSO protection

**Date:** 2026-11-05
**Status:** Accepted

**Context:** The project had `ssoProtection` set to `all_except_custom_domains`,
so public URLs could redirect to Vercel Authentication.

**Decision:** Disable SSO protection for this project.

**Why:** The Android app and browser landing page are public, no-auth endpoints.

**Consequence:** The API and landing page are publicly reachable. If the project
later needs preview protection, enable it only for preview deployments.

---

## D-011 — Add a static landing page at `/`

**Date:** 2026-11-05
**Status:** Accepted

**Context:** The base deployment URL returned 404 because only `/api/advisory`
existed.

**Decision:** Add `advisory-api/public/index.html` with a live form that calls the
API.

**Why:** The deployment base URL should be useful and should not look broken.
It also gives a quick manual test surface.

**Consequence:** `/` returns HTML; `/api/advisory` remains the JSON endpoint.

---

## D-012 — Use the canonical team URL in the Android client

**Date:** 2026-11-05
**Status:** Accepted

**Context:** The short `advisory-api-rho.vercel.app` alias stopped working after a
bad deploy and later recovered. The team-scoped project URL is the canonical
production alias.

**Decision:** Default to
`https://advisory-api-bharaths31s-projects.vercel.app/api/advisory`.

**Why:** It is the project production URL shown by `vercel project ls` and is
less likely to be confused with a temporary alias.

**Consequence:** The app hardcodes the canonical URL but still accepts an explicit
`apiUrl` argument or `VITE_ADVISORY_API_URL`.

---

## D-013 — Breed Tips only needs internet, not legacy Online Mode

**Date:** 2026-11-05
**Status:** Accepted

**Context:** `openTips` required `settings.onlineMode` and a healthy Python
backend. The new Vercel API is independent of that backend, and `onlineMode`
defaults to false.

**Decision:** Gate Breed Tips only on `navigator.onLine`.

**Why:** The farmer should get tips on any internet connection without discovering
a separate settings toggle. The advisory client already handles offline/timeout.

**Consequence:** The legacy Online Mode setting remains for other backend/health
behavior but no longer blocks Breed Tips.

---

## D-014 — Track the debug APK in git

**Date:** 2026-11-05
**Status:** Accepted

**Context:** The team wants a runnable artifact alongside the source.

**Decision:** Keep `artifacts/android/breedid-android-debug.apk` and its metadata
in git, refreshed after meaningful app changes.

**Why:** Reproducible handoff and easy install without a local Android build.

**Consequence:** Binary commits are larger. Refresh `build-info.json`,
`SHA256SUMS.txt`, `README.md` and the history hash together.

---

## D-015 — Install the Vercel plugin for agent guidance

**Date:** 2026-11-05
**Status:** Accepted

**Context:** The Vercel onboarding playbook recommends the plugin for OpenCode.

**Decision:** Install `github:vercel/vercel-plugin` globally and use its skills.

**Why:** Keeps future deployment work aligned with current Vercel CLI/platform
guidance.

**Consequence:** The plugin is installed. MCP still needs a user OAuth click via
`/mcps`; it is not an automated step.
