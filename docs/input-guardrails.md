# 4-Layer Input Guardrails

A fully offline, no-retraining safety layer that runs before/after the breed
classifier. It rejects drawings, screenshots, selfies, other animals and
unrelated scenes, and returns a **specific trilingual explanation** instead of a
generic "out of distribution" banner.

Implementation: `src/input_guardrails.py`  
Config: `GUARDRAIL_*` in `src/config.py`

---

## Pipeline

| Stage | Check | Rejects when |
|---|---|---|
| **1** | Photorealism / drawing detector | >38% of 8×8 pixel blocks are near-flat (`std < 1.8`), a near-white/black canvas has sharp edges and low saturation, or the 4×4×4 colour histogram is too sparse (`entropy < 1.65` or top-3 bins > 75%) |
| **2** | Human / selfie / semantic gate | a Haar face/profile face covers ≥4% of the image; the extreme skin-tone fallback fires; or CLIP's best category is human / art / other animal / object (or `P(valid) < 0.55`) |
| **3** | Feature + entropy gate | soft-routed 68-breed entropy > 0.78; top-1 < 0.08; or the 1280-d feature L2 norm is outside `[5, 120]` |
| **4** | Logit OOD | the existing `OODDetector` energy / MSP / binary-floor check fires |

Any stage can reject. Stages 1–2 run **before** the network, so a cartoon never
spends inference time. Stages 3–4 run on the model tensors.

### Offline behaviour

- **OpenCV is optional.** If `cv2` is missing, the Haar face check is skipped.
- **CLIP is optional.** It reuses the process-wide
  `attributes_zeroshot.get_estimator()` and never downloads at import time. If
  weights are not cached/available, the semantic gate is skipped.
- The skin-tone fallback is deliberately extreme (`>0.92` overall and centre)
  because brown buffalo coats otherwise look "skin-like" when no face/CLIP
  detector is available.

### Calibration note

The specification's Stage-3 top-1 floor of `0.18` rejects many valid buffalo
images on this checkpoint (validation top-1 ≈ 0.35; real images commonly sit at
0.15–0.25). The shipped default is **0.08**, which still rejects extreme
uncertainty while leaving the energy/MSP stage to handle the rest. Override
`GUARDRAIL_TOP1_MIN` if you retrain a stronger classifier.

---

## Rejection codes

`reason_code` is one of:

| Code | Meaning |
|---|---|
| `cartoon_or_digital_graphic` | Clip-art, cartoon, screenshot, flat digital graphic |
| `sketch_or_line_drawing` | White/black canvas with line art |
| `synthetic_illustration` | Too few colours / posterised illustration |
| `human_or_selfie` | Face, portrait or selfie |
| `other_animal` | Dog, goat, sheep, horse, bird, etc. |
| `unrelated_object` | Vehicle, building, document, empty scenery |
| `uncertain_animal` | Animal-like but entropy/top-1/feature norm is abnormal |
| `out_of_distribution` | Energy/MSP/binary-floor OOD signal |

The API response contains:

```json
"rejection_info": {
  "rejected": true,
  "reason_code": "human_or_selfie",
  "stage": "Stage 2: Human / Semantic Check",
  "title":       { "en": "...", "ta": "...", "hi": "..." },
  "message":     { "en": "...", "ta": "...", "hi": "..." },
  "instruction": { "en": "...", "ta": "...", "hi": "..." }
}
```

`verdict_to_ood_explanation()` also populates the legacy
`ood_explanation` shape (`code`, `severity`, `title`, `message`,
`tips_by_language`) so older UIs keep working.

---

## Integration

| Surface | Behaviour |
|---|---|
| `server/services/inference.py` | Stage 1–2 before inference; Stage 3–4 after. Early rejection returns `rejection_info`, `is_ood=true`, `guardrail`, and zero confidence. |
| `src/model.py` | `BreedClassifier.predict_guarded(x, pil_img, ...)` returns `(out, verdict)`. |
| `test_model.py` | Same pre/post flow; `rejections` are shown in the OOD banner using the localized title/message. Schema tests can pass `guardrails=False`. |
| `server/app.py` | `/api/health` reports guardrail availability; `/api/guardrails/status` (dev route) reports OpenCV/CLIP state and thresholds. |
| React | `PhotoGuidance.jsx` prefers `rejection_info` over the legacy OOD copy. |

### Disable / tune

```python
# src/config.py
GUARDRAIL_ENABLED = False          # master switch
GUARDRAIL_CLIP_ENABLED = False     # never try CLIP
GUARDRAIL_TOP1_MIN = 0.08
GUARDRAIL_ENTROPY_MAX = 0.78
```

---

## Tests

```bash
python -m unittest tests.test_input_guardrails -v
```

The tests cover sketch/cartoon/solid rejection, real-photo acceptance,
trilingual copy, confident-model acceptance, uniform-entropy rejection,
logit-OOD rejection and feature-norm outliers.
