# Post-hoc Attributes — Gender, Age Group & Morphological Traits

Five outputs from one existing checkpoint, with **no retraining**:

| # | Output | Source | Confidence |
|---|--------|--------|------------|
| 1 | Species (Cattle / Buffalo) | `BreedClassifier.binary_head` | `species_confidence` |
| 2 | Breed Top-1 / Top-5 | `cattle_head` / `buffalo_head` (soft routing) | `top_breed_confidence`, `top5_breeds[].confidence` |
| 3 | Gender | CLIP zero-shot, else morphological heuristic | `gender.confidence` |
| 4 | Age group | CLIP zero-shot, else morphological heuristic | `age_group.confidence` |
| 5 | Traits (`hump`, `horn`, `coat`, `ear`, `dewlap`, `face`, `size`) | `trait_heads` already inside the checkpoint | `traits_detail[].confidence` |

The checkpoint is `lite2_phase2_best_v26_1012-2332_f9ce.pt` (or any checkpoint
that contains `trait_heads.*` tensors). ONNX exports keep working unchanged.

---

## 1. Whole-animal preprocessing

Training-eval used shortest-side resize + `CenterCrop(260)`. On a 4:3 phone
photo that discards ~25% of the width — the strip that carries horns, udder and
tail. Inference now uses `PadToSquare(260)` (long side → 260, short side padded
with the ImageNet mean, which becomes ≈0 after normalization), so the whole
body profile reaches the network.

Implemented once in `src/preprocess.py` and used by both entry points
(`test_model.py`, `server/services/inference.py`); `src/data_pipeline.py`
re-exports the same class as `_PadToSquare` so training and inference cannot
drift.

```bash
# default: keep the whole frame
python run.py

# legacy parity with the exported ONNX/Flutter preprocessor
BREEDID_PAD_TO_SQUARE=0 python run.py
```

Toggle: `src/config.py → INFERENCE_PAD_TO_SQUARE` (env override above).

> Expect slightly different — not necessarily worse — probabilities on crops
> where the animal already filled the centre: the network now also sees the
> padded border. Compare both settings with `scripts/verify_attributes.py` and
> your own validation images before shipping.

---

## 2. Trait heads (already trained, now exposed)

`src/train.py` attaches a `TraitClassifier` only while `--trait-weight > 0`, and
`src/export.py` excludes it from the exported graph — but the trained tensors
remain in the `.pt` file:

```
trait_heads.heads.{hump,horn,coat,ear,dewlap,face,size}.0.{weight,bias}   # Linear(1280, 256)
trait_heads.heads.{...}.3.{weight,bias}                                    # Linear(256, N_field)
```

On load, `src.model.attach_trait_heads()`:

1. reads each field's `in_dim`, `hidden_dim` and `out_dim` **from the
   checkpoint tensors** (no hardcoded widths),
2. rebuilds a `TraitClassifier` with exactly those shapes,
3. loads every `trait_heads.*` tensor,
4. stores the module as `model.trait_heads` and keeps the vocab/metadata in
   `model.trait_heads.trait_meta`.

During `predict()`, the pooled 1280-d feature vector is passed to those heads
(`src.model.predict_traits`), and `src.attributes_zeroshot.decode_traits()`
turns the argmax indices into display strings using
`data/breed_traits.json` — the same sorted vocabulary `build_trait_vocab()`
produced at training time.

**Vocabulary drift guard.** If `data/breed_traits.json` gains/loses values so a
field's vocabulary length no longer matches the checkpoint's `out_dim`, that
field falls back to opaque `class_<i>` labels rather than silently mapping an
index to the wrong trait. `traits_meta.source` reports
`breed_traits.json` / `partial_fallback` / `fallback`.

**Missing JSON.** `TRAIT_GENERIC_LABELS` in `src/attributes_zeroshot.py`
provides ordered human labels (Hump: *Absent/Small/Medium/Large/…*, Dewlap:
*Minimal/Moderate/Large & Pendulous*, Size: *Small/Compact → Large*).

**ONNX** models do not expose pooled features, so `traits` is `{}` and
`traits_meta.reason = "onnx"`; gender/age still work from the heuristic.

---

## 3. Gender & age estimation (`src/attributes_zeroshot.py`)

Runs **after** the breed decision and **only when `is_ood == False`**.

### Engine A — CLIP zero-shot (primary)

- Lazy-loads `openai/clip-vit-base-patch32` through `transformers`
  (`CLIPProcessor`, `CLIPModel`), FP16 on CUDA and FP32 on CPU.
- Looks in the local HF cache first (`local_files_only=True`), then optionally
  downloads. Every failure is captured: no package, no cache, no network, bad
  weights → the heuristic engine is used and `attributes_meta.clip.error`
  explains why.
- Concurrent requests never queue behind a download: while the model is
  loading, other requests use the heuristic immediately.
- Prompt-conditioned on the predicted species (`cattle` / `buffalo`):

  | Field | Labels | Prompt |
  |---|---|---|
  | gender | Female | "a side profile photo of a female dairy {species} cow with an udder, slender neck, and feminine face" |
  | gender | Male | "a side profile photo of a muscular male {species} bull or ox with a heavy neck, large hump, and broad head" |
  | age | Calf (< 1 yr) | "a photo of a small young baby calf with short legs, small head, and no horn development" |
  | age | Young (1–3 yrs) | "a photo of a medium-sized young heifer or sub-adult {species} with developing horns and lean body" |
  | age | Adult (> 3 yrs) | "a photo of a fully grown mature adult {species} with full body mass and developed horns" |

- Probabilities are `softmax(logits_per_image / temperature)`; temperature is
  `ATTRIBUTES_CLIP_TEMPERATURE` (default `1.0`).

### Engine B — morphological heuristic (fallback / Fast Mode)

Uses only signals already computed:

- **Trait activations** — hump size, horn development, dewlap, body size.
- **Subject geometry** — foreground bounding box from border-colour
  segmentation (`subject_stats`): a compact, tall silhouette nudges toward
  calf/young.
- **Breed prior** — dairy breeds (`DAIRY_BREEDS`: Gir, Sahiwal, Murrah,
  Mehsana, …) start from an ≈85% adult-female prior, matching commercial dairy
  imagery.

Scores are additive log-odds → softmax, then **shrunk toward uncertainty**
(confidence ceiling ≈ 88% gender / 82% age) because this is a prior, not a
measurement. The payload is marked `method: "morphological_heuristic"` and
`requires_confirmation: true` so the UI can ask the farmer to confirm with one
tap.

---

## 4. Response schema

`POST /api/predict` (both `test_model.py` and `server/app.py`) adds:

```json
{
  "species": "Cattle",
  "species_confidence": 98.4,
  "top_breed": "Gir",
  "top_breed_confidence": 91.2,
  "top5_breeds": [{"breed": "Gir", "confidence": 91.2}],
  "gender": {
    "label": "Female (Cow)",
    "short_label": "Female",
    "confidence": 86.5,
    "method": "clip_zeroshot",
    "note": "Visual estimation from body/udder profile"
  },
  "age_group": {
    "label": "Adult (> 3 yrs)",
    "short_label": "Adult",
    "confidence": 89.1,
    "method": "clip_zeroshot",
    "note": "Visual stage estimate; verify via dentition (teeth) for exact age"
  },
  "traits": {
    "hump": "Large / Prominent",
    "horn": "Curved (Half-moon)",
    "ear": "Long & Pendulous",
    "dewlap": "Large & Pendulous",
    "size": "Medium-Large"
  },
  "ood": false
}
```

Additional keys (safe to ignore, useful in developer views):

| Key | Meaning |
|---|---|
| `traits_detail.<field>.{index,token,confidence,top3}` | raw head output |
| `traits_meta` | which fields loaded, vocab source, mismatches |
| `attributes_meta` | engine used, CLIP state/error, heuristic inputs |
| `attributes_time_ms` | time spent after the breed forward pass |
| `gender.distribution`, `age_group.distribution` | full softmax |
| `gender.key`, `age_group.key` | machine-readable `female/male`, `calf/young/adult` |

OOD inputs return `gender.method = "skipped"`, `age_group.method = "skipped"`
and empty `traits`.

---

## 5. Configuration

`src/config.py` / environment:

| Setting | Default | Effect |
|---|---|---|
| `INFERENCE_PAD_TO_SQUARE` | `True` | `BREEDID_PAD_TO_SQUARE=0` restores center-crop |
| `ATTRIBUTES_ENABLED` | `True` | master switch |
| `ATTRIBUTES_CLIP_ENABLED` | `True` | never try CLIP (offline installs) |
| `ATTRIBUTES_FAST_MODE` | `False` | skip CLIP, use the heuristic |
| `ATTRIBUTES_CLIP_MODEL_ID` | `openai/clip-vit-base-patch32` | any CLIP checkpoint |
| `ATTRIBUTES_CLIP_TEMPERATURE` | `1.0` | softmax temperature |
| `ATTRIBUTES_CLIP_ALLOW_DOWNLOAD` | `True` | `False` = cache-only |

Server-side these can be toggled at runtime through the shared config DB
(`outputs/app_config.json`, `server/shared_config.py`):

```
attributes.enabled       -> ATTRIBUTES_ENABLED
attributes.clip_enabled  -> ATTRIBUTES_CLIP_ENABLED
attributes.fast_mode     -> ATTRIBUTES_FAST_MODE
```

### Installing CLIP (optional)

```bash
.venv\Scripts\pip install transformers
# CPU-only torch is already installed; CLIP runs fine on it.
```

The first non-OOD prediction downloads ~600 MB of weights. Call
`AttributeEstimator().preload()` at start-up if you prefer that cost outside a
request.

---

## 6. Verifying

```bash
python scripts/verify_attributes.py            # synthetic 4:3 frame
python scripts/verify_attributes.py --image data/samples/gir.jpg
python scripts/verify_attributes.py --skip-onnx
```

Checks: checkpoint load, trait-head shapes, all five response blocks, ONNX
regression, and OOD gating.

---

## 7. Limitations (read before quoting a farmer)

- **Gender/age were not trained.** They are CLIP guesses or transparent priors.
  Treat them as triage hints, not veterinary facts; the API says so via
  `method` / `requires_confirmation`.
- **Udder visibility** is the main gender cue and is often hidden by the animal's
  stance; the heuristic uses hump/dewlap/size proxies instead.
- **Age** is a visual stage estimate. Dentition (teeth) remains the only
  reliable field method.
- **Traits** come from auxiliary heads trained on breed-level labels; they
  describe breed-typical morphology, not this individual's exact measurements.
- **Pad-to-square** changes the input distribution relative to the exported
  ONNX preprocessor; keep `BREEDID_PAD_TO_SQUARE=0` where bit-exact parity
  matters (e.g. ONNX parity tests).
