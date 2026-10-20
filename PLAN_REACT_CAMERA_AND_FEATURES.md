# React Camera & Product Features — Architecture + Implementation Plan

Status: **implemented** (plan + code + tests in this change)
Scope: `breed-classifier-react/` and its Python backend contracts (`server/`, `src/`, `test_model.py`).
Out of scope: Flutter/Dart.

---

## 1. Executive Summary

- The previous camera reticle was **cosmetic**: capture drew the full video frame (`HEAD:breed-classifier-react/src/components/CameraCapture.jsx:139`) while the model receives a centre crop (`src/data_pipeline.py:689`).
- The previous handoff was **broken**: `onCapture(capturedBlob)` passed one argument (`HEAD:…/CameraCapture.jsx:160`); `UserPage` expected `(file, dataUrl)` (`HEAD:…/UserPage.jsx:29`) and immediately unmounted the camera (`HEAD:…/UserPage.jsx:149`).
- Uploaded photos were sent **unframed**, so the server's `Resize(260) -> CenterCrop(260)` silently removed 25–44% of a 4:3/16:9 animal.
- Fix: an **authoritative 1:1 square crop**. The square the user sees is the exact pixel region exported and sent to `/api/predict`; because the image is already square, the server centre crop is a no-op.
- Implemented shared crop math and tests in `src/utils/cropAndQuality.js`; interactive cropper in `src/components/ImageCropper.jsx`.
- Implemented an on-device quality gate (Laplacian sharpness, exposure histogram, Sobel framing) with non-colour-only guidance.
- Upgraded live camera with 1:1 mask, zoom/pan, auto-capture, burst-3 best frame, torch support, and freeze-frame re-crop.
- Implemented 6 product features: guided quality coach, 3-tier confidence/OOD abstention, “Why this breed?” trait matrix, herd session log with CSV/JSON/ODT export, shareable PNG result card, EN/TA i18n.
- The user-specified coordinate formula omitted `zoom` from the centring offset; the implementation computes the offset at `coverScale × zoom` so zoom is anchored at the viewport centre, matching CSS `transform: scale()`.
- Verification: `npm test` → 20/20 geometry, quality, session, and trait tests pass; `npm run build` → production bundle succeeds; headless Chromium smoke tests cover upload → crop → prediction, camera → shutter → re-crop, tier 1/2/3, traits, session persistence, and EN/TA switching.

---

## 2. Current-State Audit

| Area | What existed | Bug / gap | Evidence (pre-change) |
|---|---|---|---|
| Camera capture region | Full-frame canvas capture | Reticle did not affect the crop | `HEAD:breed-classifier-react/src/components/CameraCapture.jsx:122-139` |
| Camera preview geometry | `object-fit: contain`, rectangular reticle `inset: 30px` | Reticle floated over letterbox bars; not 1:1 | `HEAD:…/styles.css:1410`, `HEAD:…/styles.css:1444` |
| Camera handoff | `onCapture(capturedBlob)`; parent expects `(file, dataUrl)` | Blob not wrapped in `File`; preview `undefined`; camera unmounted | `HEAD:…/CameraCapture.jsx:160`, `HEAD:…/UserPage.jsx:29-34`, `HEAD:…/UserPage.jsx:149` |
| Upload path | Raw file + `<img>` preview | No crop/placement; server centre-crops and truncates the animal | `HEAD:…/ImageUpload.jsx:39`, `server/services/inference.py:12-17` |
| Quality guidance | None | User gets no feedback until an expensive inference call | — |
| Confidence UX | Binary OOD banner + confidence pill | No ambiguity tier, no margin check, no actionable next step | `HEAD:…/ResultCard.jsx:36-58` |
| Explainability | None in the user UI | Reviewer cannot see why the top breed won | — |
| Batch/session history | `Batch.jsx` only (separate upload flow) | No persistent session, no per-animal triage, no export | `HEAD:…/Batch.jsx` |
| i18n | Hard-coded English strings | No language toggle, no externalised copy | `HEAD:…/UserPage.jsx` |
| Backend contract | `Resize(260) -> CenterCrop(260)`; ImageNet normalization | Correct, but only lossless for square inputs | `src/data_pipeline.py:689-704`, `server/services/inference.py:12-17` |
| OOD contract | Energy + MSP + binary floor | Rich `ood_details` already returned but not surfaced | `src/model.py:880-918`, `server/services/inference.py:176-185` |
| Traits | `data/breed_traits.json` (7 training fields) | No frontend display / comparison; no `origin` / `key_distinguisher` for the UI | `data/breed_traits.json`, `src/config.py:159` |
| ODT export | `/api/export-odt` + `exportOdt()` | Not wired to the session log | `server/routes/export.py:10`, `breed-classifier-react/src/api.js:144` |

---

## 3. Decision Record — Crop vs Overlay

### Option A — Decorative overlay (status quo)
Reticle is a hint; capture sends the full frame.
- ✅ No new math.
- ❌ Guidance is a lie: the model sees a different region.
- ❌ Horns/tail still truncated; users blame the model.
- **Rejected.**

### Option B — Authoritative 1:1 target crop (chosen)
The square target defines the exact source rectangle. The client exports a square JPEG (`520×520`), the server centre-crops to `260×260` losslessly, and the TFLite/ONNX graphs receive the same framing.
- ✅ Overlay and model input are provably identical.
- ✅ Removes the silent long-axis truncation.
- ✅ Enables on-device quality scoring on the exact model region.
- ⚠️ Requires correct coordinate math for cover, zoom, pan, and front-camera mirroring.
- **Chosen.**

### Option C — Full-frame letterbox (`pad-to-square`)
Send the whole image with padding, matching `EVAL_PAD_TO_SQUARE`.
- ✅ No truncation.
- ❌ Changes the evaluation distribution vs the trained transform; accuracy risk.
- ❌ Wastes resolution on background.
- **Rejected for the default path** (kept as a backend option).

### Coordinate math (implemented)

Let `s = max(Wv / Wi, Hv / Hi)` (cover), `z = zoom`, and `S = s · z`.

```
Ox = (Wi · S − Wv) / 2
Oy = (Hi · S − Hv) / 2

Li = min(Lv / S, min(Wi, Hi))

xi = clamp((xv + Ox − Δx) / S, 0, Wi − Li)
yi = clamp((yv + Oy − Δy) / S, 0, Hi − Li)

mirrored front camera: xi' = Wi − xi − Li
```

**Correction to the original brief:** the offset must be evaluated at `S = s·z`, not at `s`. Using `(Wi·s − Wv)/2` anchors zoom at the source origin and the crop drifts toward the top-left as `z` increases. The implemented equation (and its test “zoom shrinks the crop around the same centre”) uses the zoomed offset. CSS applies `scale(z)` about the viewport centre, so this is the matching inverse.

---

## 4. Capture UX

### State machine

```
IDLE
  └─ open ─────────────► REQUESTING_PERMISSION
                            ├─ denied ─────► ERROR_PERMISSION ──retry──┐
                            ├─ no device ───► ERROR_NO_DEVICE ──retry──┤
                            └─ granted ─────► LIVE ◄──────────────────┘
LIVE
  ├─ quality check (4 Hz) ─► LIVE (READY / BLURRY / DARK / FAR / EDGE)
  ├─ auto-capture + READY×3 ► CAPTURING
  └─ shutter / burst ──────► CAPTURING
CAPTURING
  └─ frame(s) frozen ──────► REVIEW
REVIEW
  ├─ drag / pinch / wheel ─► REVIEW (re-crop, quality recomputed)
  ├─ cancel ───────────────► LIVE
  └─ confirm ──────────────► EXPORT_SQUARE ─► CONFIRMED (parent)
CONFIRMED
  └─ parent shows square preview + “Identify Breed”
```

### ASCII wireframes

**Live viewfinder**

```
┌──────────────────────────────────────────────┐
│  ● Live Camera                            ✕  │
├──────────────────────────────────────────────┤
│  [✓ Framing optimal — hold steady      82]   │
│        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░          │
│        ░┌────────────────────────┐░          │
│        ░│  ┼───────┼───────┼    │░          │
│        ░│  │       │       │    │░          │
│        ░│  ┼───────┼───────┼    │░          │
│        ░│  │   animal side-on  │░          │
│        ░│  └────────────────────┘░          │
│        ░░░ Model sees this · 1:1 ░░          │
│  [ Zoom ————————●———— 1.4× ]                 │
├──────────────────────────────────────────────┤
│ [🔄 Flip] [💡 Torch] [⏱ Auto] [⧉ Burst ×3]   │
│              ( ◉ )            Cancel         │
└──────────────────────────────────────────────┘
```

**Review / re-crop**

```
┌──────────────────────────────────────────────┐
│  Review & Frame                           ✕  │
├──────────────────────────────────────────────┤
│  Drag to move · pinch/scroll to zoom         │
│  ┌──────────────────────────────┐            │
│  │        ┼──────┼──────┼       │            │
│  │        │      │      │       │            │
│  │        ┼──────┼──────┼       │            │
│  │        │  framed animal     │            │
│  │        └────────────────────┘            │
│  │  [ Zoom ————●———— 2.1× ] [Reset]          │
│  └──────────────────────────────┘            │
│  Quality 84/100  Sharp 88 · Light 79 · Frame 86│
│  [ Cancel ]            [ ✓ Use This Photo ]  │
└──────────────────────────────────────────────┘
```

**Result card**

```
[✓ High confidence]  top match well separated
🐄 CATTLE · 91% confidence · quality 84
Gir
──────────────────────────────────────────────
#1 Gir 91% ████████████████████████
#2 Red Sindhi 5% ██
#3 Sahiwal 2% █
▸ Why this breed? · Look-alike comparison
  ┌──────────┬────────────┬──────────────┐
  │ Trait    │ Gir        │ Red Sindhi   │
  │ Hump     │ Large      │ Prominent    │
  │ Horn     │ Long curve │ Short thick  │
  │ Dewlap   │ Large      │ Moderate     │
  └──────────┴────────────┴──────────────┘
[ Download / Share PNG ] [ Retake ] [ Another ]
```

---

## 5. Worked Coordinate Examples

Stage: modal `688 × 516` CSS px, target side `Lv = 0.78 × 516 = 402.5`, `xv = 142.75`, `yv = 56.75`.

### Landscape 1920×1080, zoom 1, no pan

```
s  = max(688/1920, 516/1080) = 0.47778
S  = 0.47778
Ox = (1920 × 0.47778 − 688)/2 = 114.67
Oy = (1080 × 0.47778 − 516)/2 = 0

Li = 402.5 / 0.47778 = 842.4
xi = (142.75 + 114.67)/0.47778 = 538.8
yi = (56.75 + 0)/0.47778     = 118.8
→ crop (539, 119, 842, 842) → 520×520 JPEG → server 260×260
```

### Portrait 1080×1920, zoom 1, no pan

```
s  = max(688/1080, 516/1920) = 0.63704
S  = 0.63704
Ox = (1080 × 0.63704 − 688)/2 = 0
Oy = (1920 × 0.63704 − 516)/2 = 353.56

Li = 402.5 / 0.63704 = 631.9
xi = (142.75 + 0)/0.63704 = 224.1
yi = (56.75 + 353.56)/0.63704 = 644.0
→ crop (224, 644, 632, 632)
```

### Same landscape, zoom 2

`S = 0.95556`, `Ox = (1920×0.95556−688)/2 = 573.33`, `Oy = (1080×0.95556−516)/2 = 258`, `Li = 421.2`, `xi = (142.75+573.33)/0.95556 = 749.6`, `yi = (56.75+258)/0.95556 = 329.4`. The crop centre stays at `(960, 540)`, which is the viewport centre mapped into the source.

### Front camera

`xi' = 1920 − 538.8 − 842.4 = 538.8` for the centred landscape case (symmetric). For an off-centre pan, the mirrored x is the complement, matching `scaleX(-1)`.

---

## 6. Feature Portfolio (14 candidates)

Scoring: 1 (low) … 5 (high). “Effort inv.” and “Risk inv.” are inverted so higher is better.

| # | Feature | Value | Demo | Effort inv. | Risk inv. | Zero-ML | Offline | Total | Phase |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Guided capture + quality gate | 5 | 5 | 3 | 4 | 5 | 5 | **27** | 1 ✅ |
| 2 | 3-tier confidence + OOD abstention | 5 | 5 | 4 | 4 | 5 | 5 | **28** | 1 ✅ |
| 3 | “Why this breed?” + top-3 trait matrix (`TraitComparison.jsx`) | 5 | 5 | 3 | 4 | 5 | 5 | **27** | 1 ✅ |
| 4 | Herd session log + CSV/JSON/ODT | 5 | 4 | 4 | 4 | 5 | 5 | **27** | 1 ✅ |
| 5 | Shareable PNG result card | 4 | 5 | 4 | 4 | 5 | 5 | **27** | 1 ✅ |
| 6 | EN / தமிழ் i18n | 4 | 4 | 4 | 4 | 5 | 5 | **26** | 1 ✅ |
| 7 | Auto-capture when stable | 4 | 4 | 4 | 3 | 5 | 5 | **25** | 1 ✅ |
| 8 | Burst-3 sharpest frame | 4 | 4 | 3 | 4 | 5 | 5 | **25** | 1 ✅ |
| 9 | Torch control (supported devices) | 3 | 3 | 4 | 3 | 5 | 5 | **23** | 1 ✅ |
| 10 | Offline field mode + sync queue | 5 | 4 | 1 | 2 | 4 | 5 | **21** | 3 |
| 11 | Voice-guided capture | 4 | 4 | 2 | 3 | 5 | 5 | **23** | 3 |
| 12 | Side-by-side look-alike photos | 4 | 5 | 2 | 3 | 4 | 4 | **22** | 2 |
| 13 | Consent-based data flywheel | 3 | 2 | 2 | 1 | 3 | 3 | **14** | defer |
| 14 | Model A/B compare in demo mode | 3 | 5 | 4 | 4 | 5 | 4 | **25** | 2 |

### Phase 1 — demo-critical (shipped)
Features 1–9. All run without retraining and without a backend change; the only server contract used is the existing multipart `/api/predict` plus `/api/export-odt`.

### Phase 2 — reviewer differentiators (1–2 weeks)
- **#12 side-by-side photo comparison**: reuse the trait matrix but show the actual top-2 crops side by side with difference callouts.
- **#14 model A/B**: `DevPage` already lists models; add a user-page toggle that runs the same crop through two checkpoints and shows latency/confidence deltas.

### Phase 3 — field deployment (3+ weeks)
- **#10 offline queue**: IndexedDB for images + predictions, service-worker sync, conflict policy.
- **#11 voice guidance**: Web Speech API, language-matched prompts, graceful fallback.

### Rejected as gimmicks
- **AR floating breed label** — adds no diagnostic value, high tracking cost, poor low-light behaviour.
- **Gamified points/streaks** — distracts from a veterinary/field tool.
- **Social feed / leaderboard** — privacy and moderation burden.
- **Blockchain provenance** — no buyer, no problem it solves here.
- **Chatbot wrapper** — would invent explanations the model cannot support.

---

### Trait data ownership

`data/breed_traits.json` remains the training source of truth: `TRAIT_FIELDS` stays at the seven categorical fields consumed by `src/traits.py`. `origin` and `key_distinguisher` were added to the JSON as **display-only metadata** (ignored by training), and the same strings are exposed through `src/data/breedTraits.js` for the React panel. This keeps the ML contract stable while giving reviewers the explanatory fields the UI needs. The frontend module is the curated display layer; the JSON is the ML layer.

### Scope boundaries (deliberate, not gaps)

- `Batch.jsx` intentionally keeps its fast multi-file flow: it does not open a per-image cropper. Batch images still use the server `CenterCrop(260)`, and the batch panel now tags OOD rows and can append recognised animals to the herd session. A per-image batch cropper is a Phase 2 item if field users need it.
- `ResultsPanel.jsx` is the legacy single-image surface used by `SingleImage.jsx`. It now shares `TraitComparison` and the 3-tier banner so it does not diverge from `ResultCard`.
- `data/breed_traits.json` keeps the 7 training fields; the added `origin` / `key_distinguisher` keys are ignored by `src/traits.py`.

## 7. Implementation Backlog

| ID | Task | Files | Estimate | Definition of done |
|---|---|---|---|---|
| A1 | Pure crop math + mirror/clamp | `src/utils/cropAndQuality.js` | 0.5 d | Unit tests for portrait, landscape, zoom, pan, mirror, clamp ✅ |
| A2 | Quality gate (blur/exposure/framing) | `src/utils/cropAndQuality.js` | 0.5 d | Flat/black/white/checkerboard tests ✅ |
| A3 | Shared interactive cropper | `src/components/ImageCropper.jsx` | 1 d | Drag/pinch/wheel, quality recompute, square export ✅ |
| A4 | Live camera rewrite | `src/components/CameraCapture.jsx` | 1.5 d | 1:1 mask, guidance, auto, burst, torch, review ✅ |
| A5 | Upload cropper | `src/components/ImageUpload.jsx` | 0.5 d | Adjust framing before inference ✅ |
| A6 | Handoff fix | `src/pages/UserPage.jsx` | 0.25 d | `(File, dataUrl, quality)` reaches preview ✅ |
| B1 | 3-tier result + OOD trigger | `src/components/ResultCard.jsx` | 0.75 d | Tier 1/2/3 with actionable retake ✅ |
| B2 | Trait matrix | `src/data/breedTraits.js`, `ResultCard.jsx`, `TraitComparison.jsx` | 1 d | 44 breeds, 8 fields, differs highlight ✅ |
| B3 | Session log | `src/utils/sessionStore.js`, `HerdSessionDrawer.jsx` | 1 d | Persist, stats, CSV/JSON/ODT ✅ |
| B4 | Share PNG | `cropAndQuality.js`, `ResultCard.jsx` | 0.5 d | Canvas card + native share fallback ✅ |
| B5 | i18n | `src/i18n/*`, `LanguageToggle.jsx` | 0.75 d | EN/TA toggle persisted ✅ |
| B6 | CSS system | `src/styles.css` | 0.75 d | All 5 themes + mobile ✅ |
| C1 | Tests + build | `__tests__`, `package.json` | 0.25 d | 20/20 pass, `npm run build` ✅ |

### Test matrix

| Layer | Cases |
|---|---|
| Unit | cover scale; landscape/portrait centre; zoom centre invariance; pan clamp; mirror complement; crop ≤ shortest side; flat/blur; checkerboard; dark/overexposed; session CSV/JSON/ODT mapping; trait aliases/comparison |
| Integration | upload → crop → confirm → predict; camera → capture → re-crop → confirm; ODT export from session |
| Devices | Android Chrome (front/rear, torch); iOS Safari (playsInline, permission); desktop Firefox/Chrome (webcam, wheel zoom) |
| Failure | permission denied; no device; camera busy; offline; storage quota exceeded |
| A11y | keyboard dropzone; slider labels; `role=status` live quality; non-colour status icons; 200% zoom |
| i18n | EN/TA switch without losing preview; long Tamil strings do not clip |

---

## 8. 5-Minute Reviewer Demo Script

1. **0:00 — Framing problem.** Upload a 16:9 photo. Show the 1:1 adjuster and the dimmed surround: “This square is exactly what the model receives; previously the server centre-cropped 44% of the animal.”
2. **0:45 — Quality gate.** Drag into a dark/blurred region and show the amber pill plus the sharpness/lighting/framing bars.
3. **1:15 — Live camera.** Open camera, show the 1:1 mask, zoom, auto-capture ring, burst-3. Confirm a capture and show the frozen re-crop.
4. **2:00 — Prediction tiers.** Run inference. Show tier 1 (green) for a clear Gir; switch to a look-alike image to show tier 2 (amber) and the trait matrix.
5. **2:45 — OOD.** Show a dog/car image. Tier 3 lists the exact trigger (energy / MSP / binary floor) and offers “Retake with Guided Coach”.
6. **3:15 — Session.** Open the Herd drawer: totals, cattle/buffalo split, flagged count, average confidence/quality. Export CSV, JSON, ODT.
7. **3:45 — Share card.** Download the branded PNG with crop, top-3 bars, traits, quality, model provenance.
8. **4:15 — i18n.** Toggle `EN | தமிழ்`; the whole flow switches without reload.
9. **4:30 — Close.** Emphasise: no retraining, no new backend endpoint, authoritative crop math is unit-tested, and the UI degrades gracefully when the camera or storage is unavailable.

---

## 9. Risks & Mitigations

| Risk | Trigger | Mitigation |
|---|---|---|
| iOS Safari camera quirks | `playsInline`/constraint rejection | fallback to `{video:true}`; file-upload path always available |
| Overlay/crop drift | CSS transform mismatch | single source of truth `computeSourceRect`; unit tests; visual corner check |
| Front-camera mirror | user-facing capture | frame is mirrored once at capture; crop math tested |
| Quality gate false “READY” | textured background | conservative thresholds; user can always override |
| localStorage quota | many high-res thumbnails | 160px JPEG thumbnails, 60-entry cap, lean retry |
| ODT payload mismatch | schema drift | `sessionToOdtResults` maps to the exact keys consumed by `odt_export.py` |
| Copy overflow (Tamil) | longer strings | flex/wrap layouts, no fixed-height text containers |
| Share API unavailable | desktop browsers | `navigator.canShare` guard + download fallback |

---

## 10. Deferred Product Decisions (defaults shipped)

None of these block the implementation; each has a shipped default so there is no unresolved code path.

1. **Session scope** — device-local in `localStorage`. Server sync is a Phase 2 decision; the export path (CSV/JSON/ODT) already covers sharing.
2. **Export size** — `520×520` client crop; the server downsamples to `260×260`. This preserves review quality without changing the model contract.
3. **Tier thresholds** — `0.72` confidence and `0.12` margin are constants in `ResultCard.jsx`. Moving them to the shared config DB is a future configuration task, not a missing feature.
4. **Thumbnail consent** — thumbnails stay in the browser and are never uploaded; the session can be cleared in one click. A consent screen is only needed if server-side sync is added.
5. **Retake behaviour** — “Retake with Coach” opens the camera without force-enabling auto-capture, so the user keeps manual control.
