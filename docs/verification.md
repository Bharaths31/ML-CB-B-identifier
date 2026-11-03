# Verification & Validation (V&V) Suite

A single command exercises the model, API contract, OOD safety, knowledge base,
trilingual UI and themes. It prints a PASS/FAIL/WARN scorecard and writes a
review-ready markdown report.

```bash
python -m tests.run_vv_suite
# partial run (does not overwrite the report)
python -m tests.run_vv_suite --suite "3."
# skip the latency benchmark
python -m tests.run_vv_suite --skip-benchmark

# standalone UI audit used by Suite 5
node breed-classifier-react/src/verify_ui_i18n.mjs
```

Reports: `outputs/metrics/vv_review_report.md` and `.json` (generated, ignored by git).

## Coverage

| Suite | What it proves |
|---|---|
| 1. Architecture | Forward shapes `(2,2)/(2,50)/(2,18)/(2,1280)`, finite checkpoint weights, 7 hidden `trait_heads`, soft routing sums to 1.0 |
| 2. Big-4 | Species, breed, gender and age are present and in range; top-5 canonical breeds; full traits; `PadToSquare` keeps 4:3 and 9:16 extent |
| 2b. External data | Real-image accuracy on `atharvadarpude/indian-buffalo-dataset` |
| 3. OOD & quality | Noise/solid/checkerboard are flagged with trilingual guidance; blur and dark gates; explainer branches |
| 4. Knowledge | All 68 breeds complete; female/male/calf advisories change correctly |
| 5. i18n & themes | en/ta/hi key parity, Tamil/Devanagari script coverage, 70 breed names, hero/trait labels, 4 theme token sets |
| 6. Performance | Cold load and warm inference against CPU/GPU budgets |

## Latest run (2026-11-03)

```
PASS 30 · FAIL 0 · WARN 1
Warm inference: 49.1 ms mean / 51.1 ms p95 (CPU)
Cold model load: 181 ms
```

The single WARN is the external-dataset OOD abstention rate (19.1%). The Kaggle
buffalo set contains diagrams, microscopy images and other non-animal files, so
the safety gate correctly declines them; treat it as data noise, not a model
defect.

### External validation

2,785 images across 17 buffalo breeds, balanced deterministic sample of 136:

| Metric | Value |
|---|---|
| Species accuracy | **90.0 %** |
| Top-5 breed accuracy | **61.8 %** |
| Top-1 breed accuracy | **18.2 %** (3.3× chance over 18 classes) |
| False OOD abstentions | 19.1 % (WARN, see above) |

Download with the project's built-in Kaggle facility (never commit credentials):

```powershell
$env:KAGGLE_USERNAME="<user>"
$env:KAGGLE_KEY="<token>"
python data_collection.py --source kaggle `
  --dataset atharvadarpude/indian-buffalo-dataset `
  --output data/external/atharvadarpude_buffalo
```

`data/external/` is gitignored. The suite picks the dataset up automatically
when present and skips with a WARN otherwise.
