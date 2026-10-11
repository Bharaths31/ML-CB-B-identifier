# Implementation Plan — 2026-10-06

Status of the current request ("analyze/consolidate, docs, training tweaks, OOD,
GUI tools, naming") and the exact remaining steps. **Pause point: implement the
"Remaining work" section next.**

---

## A. Completed this session (verified)

| # | Item | Where | Status |
|---|------|-------|--------|
| A1 | `src/` consolidated 16 → 10 files (model/train/run_utils merges) | `src/` | ✅ |
| A2 | Section maps + references in every module; inline banners | `src/*.py` | ✅ |
| A3 | **No random crops** by default (`AUG_RANDOM_RESIZED_CROP=False`) | `src/config.py`, `src/data_pipeline.py` | ✅ |
| A4 | **Drop small breeds** (`MIN_IMAGES_PER_BREED=15`, `DROP_SMALL_BREEDS=True`) + `dropped_breeds.json` | `src/config.py`, `src/data_pipeline.py` | ✅ |
| A5 | **Dynamic head sizing** end-to-end (dataset one-hot, train model, export loader, evaluate, traits) | `src/data_pipeline.py`, `src/train.py`, `src/export.py`, `src/evaluate.py`, `src/model.py` | ✅ |
| A6 | **Automatic per-model OOD** (`OODDetector.calibrate/save`, `OOD_DIR=outputs/ood/<name>.json`) | `src/model.py` (§6) | ✅ |
| A7 | Train auto-calibrates OOD at end of run (aliased by run id + checkpoint stem) | `src/train.py` | ✅ |
| A8 | **test_model GUI tools**: OOD linkage, recalibrate, view dataset (endpoints + UI) | `test_model.py` | ✅ (HTML builds; needs runtime check with data) |
| A9 | **Unique run-id modifier** on top of `Vx` (`v7_1006-1530_a3f2`) | `src/run_utils.py` | ✅ |
| A10 | CPU tests updated & passing (61 + 36) | `scripts/test_*.py` | ✅ |

Config knobs added: `OOD_ENABLED/OOD_AUTO_CALIBRATE/OOD_PERCENTILE/OOD_DIR`,
`DROP_SMALL_BREEDS/MIN_IMAGES_PER_BREED/DROPPED_BREEDS_FILE`,
`AUG_RANDOM_RESIZED_CROP=False`.
New CLI: `--min-images-per-breed`, `--no-ood-calibrate`.

---

## B. Remaining work (implement next)

### B1. Docs — README + detailed guide  *(requested)*
1. **`README.md`** — update:
   - Directory structure (already 10 `src/` files; verify).
   - Defaults section: random crops OFF; small breeds dropped; OOD auto-calibrated.
   - Training flags table: `--min-images-per-breed`, `--no-ood-calibrate`, `--rrc`.
   - New "OOD (per-model)" and "Small-breed policy" subsections.
   - Model naming: `v<N>_MMDD-HHMM_xxxx` examples.
   - test-model GUI tools (OOD linkage / recalibrate / dataset view).
2. **`docs/PROJECT_GUIDE.md` (new)** — the "completely understand the project" doc:
   - Overview, goals, 75-class architecture, model param budget, I/O contract.
   - Input: path layout, transforms (train/eval), normalization, pad-to-square.
   - Output: 3 heads (binary 2 / cattle C / buffalo B), soft routing (75-way),
     confidence semantics, OOD score/thresholds.
   - End-to-end pipeline diagram + every `python -m src.*` entry point.
   - Training: phases, losses (CE/logit-adj/SupCon/KD/traits), EMA, sampler,
     adaptive weights, checkpoint metric, dynamic head sizing.
   - Data integrity: splits 70/15/15, dHash dedup, dropped breeds, inventory.
   - OOD: energy + MSP, calibration, per-model JSON, GUI recalibration.
   - Export/deploy: portable/ONNX/TFLite/INT8, parity gate, Flutter.
   - Troubleshooting + FAQ + glossary.

### B2. Scripts alignment (dynamic sizing + per-model OOD)
- `scripts/calibrate_ood.py` → use `src.export._load_model` and
  `OODDetector.calibrate/save` (write `outputs/ood/<model>.json`).
- `scripts/mine_confusions.py`, `scripts/diagnose_model.py` → use `_load_model`
  so dropped-breed checkpoints load (they currently build `BreedClassifier`
  with default class counts).
- `scripts/make_trait_template.py` unaffected.

### B3. CPU tests to add
- `make_run_id()` returns unique modifier (`v<N>_MMDD-HHMM_xxxx`) and does not
  collide across calls (careful: it mutates `VERSION.txt` — test with a temp
  `PROJECT_ROOT` or assert format only).
- `OODDetector.calibrate → save → reload` round-trip (per-model JSON).
- Drop-small-breeds: `prepare_splits(min_images=N)` drops and writes JSON;
  dynamic class maps; `_validate_class_counts(allow_fewer=True)`.
- `export._load_model` infers `num_cattle/num_buffalo` from a reduced-head ckpt.
- Default train transform has no RRC; RRC ratio only when enabled.

### B4. Doc/knowledge sync
- `.agents/knowledge/CONTEXT.md`: config rows (OOD_DIR/AUTO/PERCENTILE,
  DROP_SMALL_BREEDS/MIN_IMAGES_PER_BREED), directory map (`outputs/ood/`),
  changelog entry.
- `.agents/knowledge/MODULE_REFERENCE.md`: OODDetector per-model API + new
  config; note dynamic head sizing.
- `docs/changelog.md`: 2026-10-06 entry covering the above.
- Regenerate graphify: `graphify update . --force` + drop self-loops.

### B5. Runtime verification on the GPU machine (cannot do here)
```bash
python -m src.train --quarter-data --run-tag smoke_v?   # writes outputs/ood/<run>.json
python scripts/diagnose_model.py --checkpoint <pt> --split test
# GUI: python test_model.py  -> Dev Tools -> OOD Calibration & Dataset
#   "OOD Linkage" shows the per-model JSON; "Recalibrate OOD" rewrites it;
#   "View Dataset" shows split/counts/dropped breeds.
```

### B6. Optional / decisions
- Whether dropped breeds should also be **excluded from the exported label
  files** (they already are, since class maps only contain kept breeds).
- Whether to keep the legacy `outputs/ood_thresholds.json` fallback (kept).
- Add `--ood-percentile` CLI to train (currently config-only).

---

## C. Acceptance criteria
- `py_compile` clean; `pyflakes` no undefined names.
- `scripts/test_master_cpu.py` + `scripts/test_fixes_cpu.py` all pass.
- `python -m src.verify` passes; all `python -m src.<module> --help` work.
- `build_html()` contains the OOD tools and renders.
- README + `docs/PROJECT_GUIDE.md` describe current behaviour and defaults.

## D. Files touched so far
`src/config.py`, `src/model.py`, `src/train.py`, `src/data_pipeline.py`,
`src/run_utils.py`, `src/export.py`, `src/evaluate.py`, `test_model.py`,
`scripts/test_master_cpu.py`, `scripts/test_fixes_cpu.py`.
