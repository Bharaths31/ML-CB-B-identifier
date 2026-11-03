# Pre-Inference Image Quality Gate & OOD Explainer

Field photos fail in predictable ways — motion blur, dark barns, blown-out
sunlight, tiny uploads, extreme panoramas — and a dropped pixel or two is often
all a farmer needs to know to retake the shot. `src/image_quality.py` detects
these conditions **before** the model runs, in a few milliseconds, with no
model, no GPU and no extra dependencies (Pillow + NumPy).

It also converts the OOD detector's three technical thresholds into one plain
sentence the farmer can act on, in English, Tamil and Hindi.

---

## What is measured

Analysis happens on a grayscale thumbnail (long side 256 px), which makes the
metrics resolution-independent and cheap.

| Check | Metric | Flag |
|---|---|---|
| Sharpness | Variance of a 4-neighbour Laplacian | `is_blurry` when `< QUALITY_BLUR_THRESHOLD` (45) |
| Exposure | Mean luminance + shadow/highlight clipping | `lighting_issue = "too_dark"` (`mean < 40`) or `"overexposed"` (`mean > 225`) |
| Resolution | Smaller original dimension | `resolution_issue` when `< 180 px` |
| Framing | Long:short side ratio (EXIF orientation respected) | `aspect_issue` when `> 2.5` |
| Clipping | Fraction of pixels in the darkest/brightest buckets | `clipping_issue` at `>= 55%` |

Thresholds live in `src/config.py` (`QUALITY_*`) so they can be tuned against a
real validation set without touching code.

**Calibration note.** On the 256 px thumbnail, sharp scenes measure ~50–170
lap-var; Gaussian blur r=2 ≈ 85, r=3 ≈ 44, r=6 ≈ 7. The 45 threshold catches
visible motion blur while letting low-texture backgrounds (dry grass, plain
sky) through.

---

## Report shape

```json
{
  "ok": false,
  "score": 40,
  "action": "retake",
  "issues": ["blurry", "too_dark"],
  "is_blurry": true,
  "sharpness_score": 6.3,
  "blur_threshold": 45.0,
  "lighting_issue": "too_dark",
  "mean_luminance": 22.9,
  "contrast": 8.1,
  "shadow_clip_pct": 61.2,
  "highlight_clip_pct": 0.0,
  "clipping_issue": true,
  "width": 1200,
  "height": 900,
  "min_dimension": 900,
  "aspect_ratio": 1.333,
  "resolution_issue": false,
  "aspect_issue": false,
  "tips": [
    {"en": "Photo is blurry. Hold the phone steady …",
     "ta": "படம் மங்கலாக உள்ளது …",
     "hi": "फ़ोटो धुंधली है …"}
  ],
  "tips_by_language": {"en": ["…"], "ta": ["…"], "hi": ["…"]},
  "summary": {"en": "…", "ta": "…", "hi": "…"},
  "latency_ms": 3.6
}
```

The gate is **advisory**, not a blocker: a blurry photo still receives a
best-effort prediction plus the tips, because a farmer in a field is better
served by an answer and a retake suggestion than by an error page. UIs can
choose to require `action == "proceed"` before submitting.

Measured latency: **3.6 ms** from JPEG bytes (includes decode) and **1.8 ms**
when the already-decoded image is reused — well inside the 15 ms budget.

---

## OOD explainer

`explain_ood_for_user(is_ood, ood_details, quality_report)` maps the detector's
three thresholds onto two farmer-facing failure modes:

| Condition | `code` | English message |
|---|---|---|
| `msp < binary_confidence_floor` | `not_animal` | "No cow or buffalo clearly detected in this photo. Please upload a clear side-view photo of cattle or buffalo." |
| `energy > energy_threshold` or `msp < msp_threshold` | `unclear_features` | "Animal detected, but breed features (horns, hump, coat) are partially hidden or unclear. Try capturing the full side profile from head to tail." |
| not OOD, quality issues | `poor_quality` | "The prediction below is the best available, but the photo quality limits accuracy…" |
| not OOD, clean photo | `ok` | "The photo is sharp enough and well exposed for breed analysis." |

Every message is trilingual. Raw `energy`, `msp` and thresholds are kept under
`technical` for developer views only — they never appear in the headline.
Quality tips are merged into the OOD tips and de-duplicated, so the farmer gets
one actionable list.

---

## API integration

Both prediction services call the gate after decoding and attach two keys:

```python
quality = analyze_image_quality(img)                  # PIL image, ~2 ms
...
ood_explanation = explain_ood_for_user(is_ood, ood_details, quality)
result["quality"] = quality
result["ood_explanation"] = ood_explanation
```

- `server/services/inference.py` → Flask `POST /api/predict`
- `test_model.py :: ModelManager.predict()` → standalone tester + its `/api/predict`

The rest of the response is unchanged; existing clients can ignore the new
keys. The React app renders them in the Advanced drawer (`PhotoGuidance`), and
the main result card now uses the friendly OOD message instead of the generic
one.

---

## Configuring

```python
# src/config.py
QUALITY_ENABLED = True
QUALITY_ANALYSIS_SIZE = 256
QUALITY_BLUR_THRESHOLD = 45.0
QUALITY_DARK_MEAN = 40.0
QUALITY_BRIGHT_MEAN = 225.0
QUALITY_MIN_DIM = 180
QUALITY_MAX_ASPECT = 2.5
QUALITY_SHADOW_CLIP = 0.55
QUALITY_HIGHLIGHT_CLIP = 0.55
```

---

## Verifying

```bash
python scripts/verify_image_quality.py     # flags, tips, latency, OOD branches
python scripts/verify_attributes.py        # /api/predict integration
```

## Limitations

- Blur is a global heuristic: a sharp animal against a defocused background can
  score lower than expected. The cost of a false positive is a tip, not a
  blocked prediction.
- Very low-contrast (dull light) photos naturally measure low lap-var and may
  report both `blurry` and `too_dark`; the lighting tip is the more useful one.
- Thresholds were calibrated on synthetic scenes and should be spot-checked
  against a sample of real field photos before tightening the gate.
- Exposure/brightness uses the whole frame, so a bright sky can mask a dark
  animal; the metric is deliberately simple and fast.
