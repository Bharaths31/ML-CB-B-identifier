# Breed Knowledge Base & Farmer Advisory

Every prediction is enriched with practical, offline guidance for farmers and
the general public — no retraining, no extra model, no network call.

```
breed prediction ──► src/breed_knowledge.py ──► advisory (gender × age)
                     data/breed_knowledge.json
```

Coverage: **all 68 classifier breeds** (50 cattle + 18 buffalo) plus the
crossbred/alias classes `holstein_friesian` and `jersey`.

---

## What each breed entry contains

| Field | Example |
|---|---|
| `display_name` | `"Gir"` |
| `local_names` | `{"en": "Gir", "hi": "गीर", "ta": "கிர்"}` |
| `origin_region` | `"Saurashtra — Junagadh, Amreli, Bhavnagar and Porbandar, Gujarat"` |
| `utility_type` | `"Milch (Dairy)"`, `"Draught (Work)"`, `"Dual-Purpose"`, `"Ornamental / Dwarf Dairy"` |
| `milk_yield` | daily litres, lactation kg, fat %, lactation length, β-casein note (numeric ranges included for income maths) |
| `physical_markers` | 3 bullets for visual verification, derived from `data/breed_traits.json` |
| `feeding_and_care` | green/dry fodder, concentrate, mineral mixture, water (kg or g/day) |
| `climate_and_disease_resistance` | heat tolerance, tick/parasite resistance, disease notes, housing, smallholder suitability |
| `estimated_market_price_inr` | `"₹45,000 - ₹100,000"` (numeric range included) |
| `govt_schemes` | RGM, NLM, NADCP, KCC-AH, DIDF, ICAR conservation support |
| `data_confidence` | `"curated"` or `"regional_default"` |

**Data policy.** Curated entries follow NBAGR/ICAR breed descriptions and
standard Indian feeding norms (2–2.5 % body-weight dry matter, 0.4 kg
concentrate per litre of milk above 5 L). Rare breeds without reliable
published production data use the utility template and are explicitly flagged
`regional_default` — the UI tells the farmer so. Prices, yields and vaccination
timings vary by farm and district, and every advisory ends with a disclaimer.

---

## Advisory engine

`get_farmer_advisory(species, breed_slug, gender, age_group)` builds a context
matrix. It is pure dictionary assembly (~0.1 ms), so the UI can re-render on a
toggle without touching the image model.

| Context | Emphasis |
|---|---|
| **Female + Adult** | Lactation management, expected yield/fat, concentrate top-up (0.4 kg/L above 5 L), calcium + mineral mixture, mastitis prevention, drying-off, calving interval, milk income estimate |
| **Female + Young** | Heifer growth ration (400–600 g/day gain), target breeding weight, age at first calving by breed type, estrous/heat signs |
| **Male + Adult** | Milk is hidden (not the relevant metric); draught work capacity, safe load, yoke/hoof care, breeding-bull and semen value |
| **Male + Young** | Bull-calf vs draught-ox decision, castration timing, growth target, temperament handling |
| **Calf (any gender)** | Colostrum in the first 1–2 h, 10 % body-weight milk, calf starter, deworming on day 10/30/60, FMD/HS/BQ calendar, brucellosis for female calves |
| **Unspecified** | General breed briefing when the model's gender/age estimate is unavailable |

Returned shape:

```json
{
  "breed": {"species": "Cattle", "slug": "gir", "display_name": "Gir",
            "local_names": {"hi": "गीर", "ta": "கிர்"},
            "origin_region": "...", "utility_type": "Milch (Dairy)",
            "data_confidence": "curated"},
  "context": {"gender": "Female", "age_group": "Adult"},
  "headline": "Gir adult female — protect lactation ...",
  "priority_actions": ["...", "..."],
  "nutrition": {"green_fodder": "22 - 30 kg/day", "...": "..."},
  "health": {"vaccination": [], "watch_for": [], "housing": "..."},
  "production": {"milk": {...}, "draught": null},
  "breeding": {"heat_cycle": "...", "service_window": "..."},
  "economics": {"market_price": "₹45,000 - ₹100,000", "schemes": []},
  "physical_markers": ["...", "...", "..."],
  "schemes": ["..."],
  "warnings": ["..."],
  "disclaimer": "..."
}
```

---

## REST API

### `GET /api/breed-knowledge/<species>/<breed>`

```bash
curl http://localhost:5000/api/breed-knowledge/Cattle/Gir
curl http://localhost:5000/api/breed-knowledge/Buffalo/Nili-Ravi   # alias-tolerant
```

Returns `{species, breed, knowledge, advisory, advisory_options, latency_ms}`.
`404` for an unknown breed/species pair, `200` otherwise.

### `POST /api/advisory-recalculate`

```bash
curl -X POST http://localhost:5000/api/advisory-recalculate \
  -H "Content-Type: application/json" \
  -d '{"species":"Buffalo","breed":"murrah","gender":"Female","age_group":"Adult"}'
```

Returns `{advisory, context, latency_ms}`. Measured ~0.07 ms per call (200
calls averaged), well inside the < 10 ms UI budget. Both endpoints exist in the
Flask server (`server/routes/knowledge_api.py`, registered in `server/app.py`)
and in the standalone tester (`test_model.py`).

### Attached to predictions

`POST /api/predict` now includes a `knowledge` block computed from the predicted
breed and the estimated/confirmed gender and age:

```json
{
  "top_breed": "Gir",
  "gender": {"short_label": "Female", "...": "..."},
  "age_group": {"short_label": "Adult", "...": "..."},
  "knowledge": {"headline": "Gir adult female — protect lactation ...", "...": "..."}
}
```

OOD inputs return `{"skipped": true, "reason": "out_of_distribution"}` instead
of misleading advice. If the advisory ever fails it returns an `error` key and
the prediction itself still succeeds.

---

## Generated data file

`data/breed_knowledge.json` is created automatically on first use:

```bash
python -m src.breed_knowledge --regenerate    # rewrite all 68 + 2 crossbred entries
python -m src.breed_knowledge --stats         # coverage counts
python -m src.breed_knowledge --show gir
python -m src.breed_knowledge --show gir --advisory Female Adult
```

The loader only adds *missing* breeds, so hand-edited numbers survive a
restart. `--regenerate` overwrites everything from the module.

**Current coverage:** 59 curated, 11 regional defaults.

---

## Verifying

```bash
python scripts/verify_breed_knowledge.py     # all entries + all 8 contexts + latency
python scripts/verify_attributes.py          # prediction-level integration
```

## Limitations

- Figures are typical ranges for the breed, not this animal's measurement.
- Market prices are indicative 2024–25 farm-gate ranges and move with locality,
  lactation stage, fat % and festival demand.
- Vaccination schedules must follow the local State Animal Husbandry
  Department; NADCP timing can differ by district.
- `regional_default` breeds (e.g. Bhelai, Ghumsari, Marathwada) have thin
  published data and are flagged as estimates.
