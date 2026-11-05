# Guardrails

Source: `breed-classifier-react/src/utils/imageGuardrails.js`
Integration: `breed-classifier-react/src/services/onDeviceInference.js`

All analysis runs on the prepared `260×260` canvas. There is no silhouette,
foreground mask, contour extraction or bounding-box segmentation.

## Pipeline

```text
260×260 canvas
  → checkHumanSelfie()        (FaceDetector or YCbCr fallback)
  → geometryGate()            (center-focused Stage A)
  → evaluateImageGuardrails() (Stage B logits)
  → accept or reject
```

`evaluateImageGuardrails` is `async` because of the optional FaceDetector call.

## Human selfie gate

### Tier 1 — FaceDetector

If `'FaceDetector' in window`:

```js
const detector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 3 });
const faces = await detector.detect(canvas);
```

Reject if any face bounding box area is `>= 3.5%` of the frame
(`FACE_AREA_MIN = 0.035`). Errors are swallowed and fall through to Tier 2.

**Observed:** FaceDetector was unavailable in the Electron harness and the
Android WebView used for verification, so Tier 2 is the effective path.

### Tier 2 — YCbCr fallback

Central portrait zone: `x 45..215`, `y 25..215`.

Skin candidate:

```text
Y > 60
80 ≤ Cb ≤ 125
136 ≤ Cr ≤ 172
R > G > B
|R − G| ≥ 12
```

Features:

- central 150×150 skin ratio;
- median 5×5 local luma sigma over skin pixels;
- bilateral skin symmetry, searching shifts `-24..+24` in steps of 4 px around
  the image center;
- dark hair cap ratio in `y 25..100`;
- eye/brow density in `y 60..140`, `x 70..190`;
- `darkCut = min(85, medianSkinLuma × 0.62)`.

Reject as `human_or_selfie` when all hold:

```text
centerSkinRatio > 0.62
2.2 ≤ medianSkinSigma ≤ 14.5
bestSkinSymmetry > 0.82
darkCapRatio > 0.02
eyeBrowDensity > 0.05
```

Observed selfie values:

```text
centerSkinRatio 0.800
medianSkinSigma 4.94
skinSymmetry    0.8895
darkCapRatio    0.1549
eyeBrowDensity  0.1270
```

The goat image was not rejected by this gate because its eye/brow density was
only `0.0297`, below `0.05`.

## Stage A — center-focused geometry

Center box: `x 40..220`, `y 40..220` (180×180 inner area).

For each region:

- Sobel gradient; `GRADIENT_THRESHOLD = 48`.
- Straight-edge ratio: gradient angle within ±4° of 0°, 90° or 180°.
- Flat blocks: 16×16 blocks with luma sigma `< 3.0`.

Constants:

```text
CENTER_STRAIGHT_EDGE_MAX = 0.30
CENTER_FLAT_PATCH_MAX    = 0.30
CENTER_FLAT_EDGE_MIN     = 6000
GLOBAL_STRAIGHT_EDGE_MAX = 0.55
GLOBAL_FLAT_PATCH_MAX    = 0.70
GLOBAL_FLAT_EDGE_MIN     = 3000
```

Reject `manmade_object_or_screen` if:

```text
centerStraight > 0.30
OR (centerFlat > 0.30 AND centerStrongEdges < 6000)
OR globalStraight > 0.55
OR (globalFlat > 0.70 AND globalStrongEdges < 3000)
```

Why center focus: a monitor fills the center, while barn roofs/fences/sky usually
sit at the border. The old global checks rejected valid farm images.

Observed center metrics:

| Image | Straight | Flat | Edges | Result |
|---|---:|---:|---:|---|
| Monitor screenshot | 0.374 | 0.281 | 9,381 | Geometry reject |
| Cattle range | 0.102–0.189 | 0–0.182 | 8,283–23,210 | Pass |

## Stage B — logit gate

Inputs from `onDeviceInference.js`:

- combined 68-class probabilities;
- active species logits (cattle or buffalo);
- binary species probabilities;
- energy across all three heads;
- binary MSP;
- thresholds from embedded OOD metadata, selected JSON profile, or defaults.

Constants:

```text
ENTROPY_MAX      = 0.86
TOP3_MASS_MIN    = 0.20
TOP5_MASS_MIN    = 0.30
BINARY_FLOOR_MIN = 0.60
MSP_FLOOR_MIN    = 0.5863
ENERGY_FLOOR_MIN = -3.2004
```

Rejection order:

1. `binaryConfidence < binaryCutoff` → `ood_low_binary_confidence`
2. `energy > energyCutoff` → `ood_energy`
3. `msp < mspCutoff` → `ood_msp`
4. `entropy > 0.86` → `ood_entropy`
5. `top3Mass < 0.20 || top5Mass < 0.30` → `ood_low_topk_mass`

`Z_peak`/`Z_max` are still reported in details but no longer reject. This was
the key false-rejection fix for similar breeds.

## Calibration set results

Test set: 34 real cattle images + goat + human selfie + monitor screenshot.

| Category | Result |
|---|---|
| 34 cattle | Accepted |
| Flat-patch `0.4688` cattle cases | Accepted |
| Human selfie | Rejected `human_or_selfie` |
| Monitor screenshot | Rejected `manmade_object_or_screen` |
| Goat | Accepted (known trade-off) |

The three `0.4688` flat-patch cattle images had strong edges around 20k, so the
adaptive center check did not reject them.

## Known trade-off

The top-k relaxation that fixed cattle false rejections also accepts the goat test
image. At the time, the model’s logits did not cleanly separate that goat from
cattle:
- goat entropy `0.7836` (below the new `0.86` ceiling);
- goat top-3 mass `0.3363` (above `0.20`);
- goat top-5 mass `0.4810` (above `0.30`);
- goat binary confidence `0.9613`.

The selfie gate does not catch it because eye/brow density is too low. A future
fix needs an animal-class check, not a tighter cattle threshold, or it will
reintroduce cattle false rejections.

## Rejection reason codes

```text
human_or_selfie
manmade_object_or_screen
ood_low_binary_confidence
ood_energy
ood_msp
ood_entropy
ood_low_topk_mass
```

Rejected results contain no species/breed and are not saved to Recent Scans.
