# 16. Changelog

### 2026-09-24 — Breed trait heads, group-aware splits, hard negatives, tooling

- **Breed trait auxiliary heads (D2, opt-in `--trait-weight`, suggested 0.1)**:
  new `src/traits.py` + `scripts/make_trait_template.py` generating
  `data/breed_traits.json` (75 breeds × `hump/horn/coat/ear/dewlap/face/size`,
  all empty for you to fill). Trait classifiers run on the pooled 1280-d
  features, masked loss ignores empty traits, and per-trait val accuracy is
  logged each eval. Training-only; excluded from export (verified).
- **Group-aware splits (E1, `--dedup-splits`)**: dHash (no external deps) groups
  near-duplicates (Hamming ≤ `DEDUP_HAMMING=4`) within each breed so twins never
  cross train/val/test; hashes cached in `data/splits/hashes.csv`; long-tail
  minimums preserved.
- **Val-noise warning (E3)**: `prepare_splits` prints min/median val images per
  breed and warns when any breed has < `VAL_MIN_WARN` (2).
- **SupCon hard negatives (D4)**: `supervised_contrastive_loss` up-weights
  confused-pair negatives (`CONTRASTIVE_HARD_NEG_WEIGHT=2.0`); `--hard-pairs`
  loads `confusion_pairs.json` produced by the new `scripts/mine_confusions.py`.
- **Per-group LR logging (F4)**: `train_phase` prints + logs each optimizer
  group's LR and parameter count at phase start.
- **Tooling (G2/G4)**: `scripts/mine_confusions.py` (top-30 confused breed pairs
  as global ids), `scripts/run_ablations.sh` / `.ps1` (quarter-data sweep
  R0→R7, `--dry-run` supported), plus `scripts/view_logs.py`.
- **Cosine/ArcFace breed heads (D3, `--cosine-head`)**: `CosineHead` replaces the
  final Linear on cattle/buffalo heads. Forward is margin-free scaled cosine
  (`scale=30`), so inference/export are unchanged; the additive angular margin
  (`COSINE_MARGIN=0.3`) is ramped over the first `COSINE_MARGIN_RAMP_EPOCHS=10`
  phase-2 epochs and applied to the target class **inside the loss**. `export`
  and `test_model.py` auto-detect cosine checkpoints (final head layer has no
  bias) and build the matching model.
- **Tests**: `scripts/test_master_cpu.py` now 61 checks (traits, dedup, hard
  negatives, cosine head, logger, mixing, EMA, metrics, transforms, CBAM identity).
- **Still pending**: the hard-pair **batch sampler** (D5); `--hard-pairs` only
  up-weights SupCon negatives for now.

### 2026-09-24 — Unified execution logger (`logs/<exec_id>/`) + `ema_decay` fix

- **Fixed `NameError: ema_decay`** in `train_phase` (the EMA-diagnostics block
  referenced `ema_decay` without declaring it — crashed phase 2). Added the
  parameter and forwarded it to `run_epoch`. Verified on CPU with an EMA model;
  `pyflakes` now reports **no undefined names** across `src/`, `local_train.py`,
  `test_model.py`.
- **New `src/run_logger.py`**: per-execution logger writing
  `logs/<exec_id>/{manifest.json,config.json,run.log,events.jsonl,actions.jsonl,
  training.jsonl,data.jsonl,test.jsonl,export.jsonl}`. `exec_id` =
  `YYYYmmdd-HHMMSS-<4hex>`, override with `--exec-id` / `RUN_EXEC_ID`.
- **Auto-start**: `sitecustomize.py` initialises the logger for any Python run
  inside the project (use `PYTHONPATH=.`), and every entry point
  (`src.train`, `local_train.py`, `test_model.py`, `src.export`,
  `src.evaluate`, `src/parity_check`, `src.verify`) calls `init_run_logger()`.
  Stdout/stderr are tee'd into `run.log` (tqdm `\r` frames collapsed).
- **Instrumentation**: training logs the plan + per-epoch raw/EMA metrics +
  checkpoint saves; `local_train.py` logs stages, subprocess commands and the
  dataset inventory; `data_pipeline` logs split summaries; `test_model.py` logs
  one record per prediction; export/evaluate/parity log artifacts and verdicts.
- **New `scripts/view_logs.py`**: `list` / `summary` / `metrics` / `events` /
  `diff` over the log folders (read-only).
- `logs/` added to `.gitignore`.

### 2026-09-24 — Blocking bug fixes, flip+RRC on by default, dataset inventory

- **Missing `src/run_utils.py` crash fixed**: the file is now present and
  `local_train.py` preflights every required `src/` module, raising a clear
  "sync these files" error instead of `ModuleNotFoundError`.
- **`SAMPLER_BETA` import** added to `src/train.py` (it was used in `main()`
  but never imported → would crash).
- **Mixing gate fixed**: `run_epoch` gated mixing on `desc.startswith("train")`,
  which never matched `phase{n} e{i}/{N}`, so `--mix` was dead code. Replaced
  with an explicit `training=True` flag and a `mix_stats` counter; the eval log
  now prints `mix=on (n/N batches)` only when mixing actually ran.
- **`find_latest_checkpoint` pattern fixed** to match timestamped names
  (`lite2_phase2_best_V3.pt`), and `resolve_checkpoint(path_or_tag)` added so
  `--run-tag V3` works with `--checkpoint` in export/evaluate/parity_check.
- **Class-count fail-fast**: `prepare_*` now raises with explicit extra/missing
  breed names and prints names shared across species (`bargur`).
- **Flip + mild RRC ON by default** (`AUG_HORIZONTAL_FLIP`,
  `AUG_RANDOM_RESIZED_CROP`) — they don't mix content between breeds. Colour
  jitter / RandAugment / mix stay off. `--no-augment` forces all off.
- **RRC ratio tightened** to `(0.92, 1.08)` and **hue capped at 0.02**
  (`--allow-hue` to raise) so body proportions and coat colour survive.
- **Breed-aware augmentation**: `BREED_AUG_POLICY` / `COAT_COLOUR_BREEDS`
  (coat-colour breeds skip colour jitter), applied per sample in the Dataset.
- **Pad-to-square** (`--pad-to-square`): resize long side + pad to square,
  keeping full-body side profiles.
- **CBAM/SE identity at init** (`CBAM_IDENTITY_INIT`): zero-init final layers
  with `x*(1+gate)`, so attention no longer distorts pretrained features during
  phase-1 warmup.
- **Metrics**: `combined_top3/5` are now species-aware over the soft-routed
  75-way scores; the old true-species versions are `combined_top3_oracle` /
  `combined_top5_oracle`. Added `val_min_per_breed` / `val_median_per_breed`
  and fixed the `pred_hist_entropy` comment.
- **Dataset inventory**: `build_dataset_inventory()` in `local_train.py` and the
  Colab downloader writes `data/dataset_inventory/<dataset>.json` with breed,
  species (cattle/buffalo), count, and per-image resolution for both datasets
  and the merged tree.
- **Tests**: `scripts/test_master_cpu.py` (40 CPU-only synthetic checks).

### 2026-09-23 — Tail-bias regression fix: single imbalance mechanism, safe mixing, EMA, soft routing, timestamped outputs

Diagnosed from a 10-photo ONNX batch test (0/10 correct; top-5 dominated by rare
breeds; a Gir bull predicted as a rare breed) that regressed after the
2026-09-21 overhaul.

**Imbalance (`src/config.py`, `src/data_pipeline.py`, `src/train.py`):**
- `LOGIT_ADJUST` is now **False by default** — the effective-number sampler is
  the single long-tail mechanism. Enabling both double-corrects and over-predicts
  rare breeds at inference. `--logit-adjust` re-enables it; its prior is computed
  from the effective **sampled** distribution (`--logit-adjust-prior sampled|raw`),
  never raw counts. Startup prints the active mechanism and prior max/min ratio.
- Fixed `_effective_num_weights` divide-by-zero for classes absent from train
  (previously produced NaN logit-adjustment priors).

**Mixing (`src/config.py`, `src/data_pipeline.py`, `src/train.py`):**
- CutMix/MixUp now pair **within the same species** (`_pairing_perm`), so binary
  labels stay one-hot and breed targets stay proper distributions.
- Strength reduced: `CUTMIX_MIXUP_PROB` 0.5→0.25, `CUTMIX_ALPHA` 1.0→0.4,
  `MIXUP_ALPHA` 0.3→0.2. Rare-class guard kept.
- `MIX_OFF_LAST_FRAC=0.15` disables mixing for the last 15% of phase 2.

**Augmentation OFF by default (`src/config.py`, `src/data_pipeline.py`):**
- flip / ColorJitter / RandAugment / RandomResizedCrop / mixing are opt-in via
  `--mix --flip --color-jitter --randaugment --rrc --augment-all`. With all off,
  the train transform equals the eval transform.

**EMA (`src/train.py`):**
- Updated once per **optimizer** step (was twice per micro-batch with
  `grad_accum=2`), with warm-up `decay_t = min(0.999,(1+t)/(10+t))`.
- Startup prints steps/epoch, total optimizer steps and the EMA time constant,
  warning if it exceeds 25% of phase-2 steps. Every eval logs BOTH raw and EMA
  metrics and saves whichever scores better.

**Checkpoint selection (`src/config.py`, `src/metrics.py`):**
- `BEST_METRIC="blended_score"` = 0.5·macro-F1 + 0.5·soft-routed top-1.
- Each eval also logs few/medium/many-shot accuracy and predicted-histogram
  entropy.

**Preprocessing (`test_model.py`, `src/config.py`):**
- `test_model.py` now uses the eval transform (shortest-side resize +
  CenterCrop) instead of a square `Resize((260,260))`. New
  `EVAL_MATCH_TRAIN_RESOLUTION` to test `Resize(288)+CenterCrop(260)` at eval.

**Timestamped outputs (`src/run_utils.py`, new):**
- Checkpoints/exports/metrics carry a `run id` (`DD-MM-YYYY-HH-MM` or `--run-tag`)
  and never overwrite previous runs. `src.export` / `src.evaluate` /
  `src.parity_check` / `local_train.py` auto-discover the newest checkpoint.

**Tooling (run-later):** `scripts/audit_data.py`, `scripts/diagnose_model.py`,
`scripts/onnx_parity_10.py`, `scripts/test_fixes_cpu.py`.

---

### 2026-09-21 — Long-Tail Accuracy Overhaul: Logit Adjustment, Feature Metric Learning, Soft Routing, 70/15/15 Splits

Motivated by a graph-assisted gap analysis: phase-2 best val top-1 was 0.578
(binary 0.95, cattle 0.59, buffalo 0.61), with 21+ indigenous cattle breeds
carrying ≤14 images against `gir`=768, and the old 85/10/5 split leaving many
rare breeds with **zero** test images.

**Config (`src/config.py`):**
- `SAMPLER_BETA` 0.999 → **0.99** (softer effective-number oversampling).
- Split ratios **70/15/15** (was 85/10/5); `PHASE2_EPOCHS` 60 → **80**.
- New: `LOGIT_ADJUST`/`LOGIT_ADJUST_TAU`, `CONTRASTIVE_WEIGHT`/`CONTRASTIVE_TEMPERATURE`,
  `RARE_CLASS_THRESHOLD`, `PROJECTION_DIM`, `BEST_METRIC="balanced_score"`,
  and saturated-binary loss weights (`*_FINAL`, `BINARY_SATURATION_ACC`).

**Imbalance (`src/data_pipeline.py`, `src/train.py`):**
- **Logit adjustment** (Menon et al., ICLR 2021): `tau·log(prior)` added to breed
  logits during training only, from smoothed train priors (`compute_class_priors`).
- **Long-tail split minimums**: every `(species, breed)` with ≥3 images now gets
  ≥1 val and ≥1 test image; fixed grouping to key on `(species, breed)` because
  `bargur` exists under both species.
- **Rare-class mixing guard**: breeds below `RARE_CLASS_THRESHOLD` (30 train
  images) are excluded from CutMix/MixUp (`compute_rare_classes` + `keep` mask).
- **Adaptive loss weights**: once `binary_acc ≥ 0.95`, weights switch from
  `0.15/0.50/0.35` to `0.05/0.55/0.40` to reallocate budget to the breed heads.
- `--no-mix` now actually disables mixing (previously it only swapped the collate fn).

**Feature learning (`src/model.py`, `src/train.py`):**
- New auxiliary **projection head** + **supervised contrastive (SupCon)** loss on
  the previously-unused pooled features (`CONTRASTIVE_WEIGHT=0.2`); skipped on
  mixed batches where labels are soft. Adds a dedicated optimizer param group.

**Metrics & selection (`src/metrics.py`):**
- `evaluate_epoch` now reports per-head **macro-F1**, **balanced accuracy**, and
  **soft-routed combined top-1** (`p(species)·softmax(head)`).
- Checkpoints are selected on `balanced_score = ½(cattle_macro_f1 + buffalo_macro_f1)`
  instead of combined top-1, which was dominated by ~10 large breeds.

**Soft routing:** `BreedClassifier.predict()`, `test_model.py`, and both Flutter
engines (`android_tflite_engine.dart`, `web_tfjs_engine.dart`) now mix
`p(species)·softmax(head)` across all 75 breeds instead of hard binary argmax,
removing two-stage routing error propagation.

**Export (`src/export.py`, `local_train.py`):**
- `_sanitize_state_dict` strips `_orig_mod.`/`module.` prefixes and QAT
  `fake_quant`/`activation_post_process`/fused-BN keys, so phase-3/QAT and
  compiled checkpoints no longer crash ONNX/FP16/INT8 export.
- `local_train.py` prefers the phase-2 (EMA) checkpoint and exports INT8 via the
  converter-side PTQ path (`--mode onnx-int8`), not the removed `--mode int8`.

---

### 2026-09-20 — Accuracy/Efficiency Overhaul: Distillation, EMA Fix, Mobile INT8 Exports

**Training (`src/train.py`):**
- **Fixed a critical EMA bug**: the EMA now updates BatchNorm buffers (running mean/var) in addition to parameters; `num_batches_tracked` is hard-copied. Previously every phase-2 checkpoint exported stale BN stats, silently degrading validation accuracy and the deployed model.
- **Knowledge distillation**: new `--teacher`, `--teacher-backbone`, `--teacher-attention` flags. Loss = `(1-α)·masked_hard_CE + α·T²·masked_KL(teacher‖student)` on all three heads (α=0.7, T=4.0). Train a lite4 teacher, distill into the unchanged lite2 student — teacher accuracy at zero on-device cost.
- **QAT is now opt-in** (`--include-qat`); default training is a clean 2-phase run. QAT uses **per-tensor observers** (fixes the `Unsupported qscheme: per_channel_affine` conversion failure) and starts from the **best phase-2 EMA checkpoint** instead of final-epoch weights.
- **Binary-head species balancing**: per-batch re-weighting (`BALANCE_BINARY_HEAD=True`) neutralizes the 57:18 breed-count species prior.

**Data (`src/data_pipeline.py`):**
- Sampler switched to **effective-number-of-samples** weighting (`SAMPLER_BETA=0.999`) — softens over-oversampling of 5-image breeds.
- CutMix/MixUp probability raised 0.25 → 0.5 (`CUTMIX_MIXUP_PROB`).
- All `prepare_*_splits()` create the split directory if missing.

**Export (`src/export.py`):**
- New `--mode tflite`: ONNX → onnx2tf → TFLite FP32 + **full-integer INT8 PTQ** (float32 [0,1] I/O), emits `labels_*.txt`, auto-copies into `flutter_app/assets/models/`.
- New `--mode onnx-int8`: QDQ static quantization for ONNX Runtime Mobile (per-channel weights, calibrated on real train images).
- **ImageNet normalization is baked into mobile graphs** — the Flutter app's `pixel/255` preprocessing is now exactly correct with zero app changes.
- Removed the broken x86 PTQ `--mode int8` path (guidance stub remains).

**Verification (`src/parity_check.py`, new):**
- `python -m src.parity_check` compares fp32 PyTorch vs TFLite/ONNX INT8 on val/test (training-equivalent metrics) or `--synthetic N` for artifact-only parity. Gate: INT8 within 1 pt `combined_top1` of fp32.

**Measured:** ONNX INT8 = 7.10 MB (from 25.65 MB FP32); fp32 ONNX exact parity; INT8 max |Δlogit| ≈ 0.05–0.07 on random inputs; end-to-end smoke training (2-phase, distillation, QAT→INT8) verified on CPU.

---

### 2026-09-15 — Architecture Improvements, EMA, & ImageNet Normalization Fix

**Architecture:**
- Deepened `cattle_head` and `buffalo_head` with an extra hidden layer (`BREED_DIM // 2`).
- Integrated `BatchNorm1d` into both breed classification heads for better convergence and to prevent covariate shift.
- Adjusted dropout values for the new layers (`0.3` for first, `0.2` for second).

**Training Pipeline:**
- **Phase 1 (Warmup):** Now trains all three heads (binary, cattle, buffalo) to build robust initial representations, instead of just the binary head.
- **Differential Learning Rates (Phase 2):** Applied fine-grained LR scaling (backbone: 0.1x, attention: 0.5x, heads: 1.0x).
- **EMA:** Integrated Exponential Moving Average (EMA) with a decay of 0.999 for model weights during Phase 2 to drastically improve evaluation stability and generalization.
- **Bug Fix:** Fixed critical bug where the training pipeline lacked `transforms.Normalize()` using ImageNet statistics, aligning it properly with inference logic.
- Reconfigured default portable export to securely capture the best weights from Phase 2 instead of Phase 3, avoiding the massive accuracy drop previously caused by aggressive INT8 QAT, while maintaining a very lightweight model footprint (~27.3 MB) ready for Android deployment.

**Metrics:**
- Added vectorized **Top-5 combined accuracy** tracking to `evaluate_epoch()` alongside Top-1 and Top-3.

---

### 2026-09-15 — Presenter Mode, Logging & Advanced Image Metadata

**Model Tester GUI (`test_model.py`):**
- Added `--dev` (default) and `--present` flag modes.
- Developer Mode (`--dev`): Advanced view showing image metadata (EXIF, size, proportion), cattle/buffalo JSON data, model specifications, and options to edit the presenter's view settings.
- Presenter Mode (`--present`): Clean, minimalist test page that hides detailed technical stats, diminishes confidence metrics, and removes the export option for a cleaner presentation.
- Presenter configurations (like branding, section toggles, and confidence modes) are saved and loaded persistently via `outputs/logs/presenter_config.json`.
- Comprehensive session logging captures all actions (start/stop, model/image selection, prediction results, reasoning) into a separate log file in `outputs/logs/`.

---

### 2026-09-13 — Colab Testing Notebook & Large-Scale Evaluation

**Testing & Evaluation:**
- Added `colab/cattle_buffalo_tester.py` and `colab/cattle_buffalo_tester.ipynb` for automated evaluation of exported models on Google Colab.
- Added comprehensive HTML report generation for single images and batch evaluations.
- Added Large-Scale Kaggle Evaluation mode to automatically download the dataset and test all images.
- Added `create_test_eval_zip.py` script to easily bundle test dataset splits for Colab.
- Updated documentation and knowledge base (`CONTEXT.md`, `README.md`, `docs/`) with testing workflow details.

---

### 2026-09-08 — Fix: `torch.compile` on Windows

**Bug Fix:**
- Fixed `BackendCompilerFailed: Cannot find a working triton installation` error that crashed phase 2 training on Windows.
- Added OS detection in `src/train.py` to automatically disable `torch.compile` (fallback to eager mode) when running on Windows.

---
### 2026-09-07 — Unified Kaggle Dataset & Colab Trainer Update

**Dataset Pipeline & Colab Notebook:**
- Updated dataset download source to unified Kaggle dataset `algsoch/breed-cattle-buffalo` containing pre-structured `cattle/` (57 breeds) and `buffalo/` (18 breeds) subdirectories.
- Simplified Kaggle download logic in `colab/cattle_buffalo_trainer.py` to extract directly into `data/raw/`, eliminating redundant file moving operations and outdated inline comments.
- Regenerated `colab/cattle_buffalo_trainer.ipynb` from updated python script.
- Updated project documentation across `README.md`, `docs/`, and knowledge base.

---

### 2026-09-06 — Hotfix: CUDA `total_mem` AttributeError

**Bug Fix:**
- Fixed `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'` that crashed §6 Training on Colab T4
- Root cause: PyTorch uses `total_memory`, not `total_mem`
- Fixed in `src/train.py` (`setup_device()`) and both occurrences in `colab/cattle_buffalo_trainer.py`
- Regenerated `colab/cattle_buffalo_trainer.ipynb` from fixed `.py`

**Documentation:**
- Added Colab gotchas table to `CONTEXT.md` §14 covering: `total_mem` bug, GitHub clone cache issue, runtime restart behaviour
- Added `rm -rf /content/project` before `git clone` in §14 best practices to ensure latest code is always used

---

### 2026-09-06 — Colab + SOTA Hyperparameters + Android QAT

**Colab Training:**
- Created `colab/` directory with full training notebook
- 3 project setup options: GitHub clone, zip upload, Google Drive
- 3 dataset options: Kaggle API, archive upload, Google Drive
- Hyperparameter configuration cell with all tunable parameters
- Image prediction cell for testing with uploaded images
- Export & download: portable bundle + ONNX + INT8
- GPU memory monitor cell

**SOTA Hyperparameters:**
- Switched from Adam → AdamW (weight_decay=1e-2)
- Added label smoothing (0.1) to soft cross-entropy
- Added linear warmup scheduler (3 epochs) before cosine annealing
- Added gradient accumulation (2 steps, effective batch=128)
- Increased batch size 32 → 64
- Optimized split ratio 80/10/10 → 85/10/5
- Phase 2 epochs 30 → 40, LR 1e-4 → 2e-4
- Phase 1 LR 1e-3 → 3e-3
- Phase 3 LR 1e-5 → 5e-6
- Dropout 0.3 → 0.4

**Data Pipeline:**
- Train augmentation: added RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- Added prefetch_factor=4 to all DataLoaders

**Android Deployment:**
- QAT (Phase 3) enabled by default (not skipped)
- Auto INT8 conversion after QAT
- ONNX export in Colab notebook for mobile deployment

### 2026-09-05 — Major Update

**Training:**
- Added CUDA optimization: `cudnn.benchmark`, TF32, AMP (`torch.amp`), `GradScaler`
- Added gradient clipping (`max_norm=1.0`)
- Enabled `pin_memory`, `persistent_workers`, `non_blocking` transfers
- Smoke test now uses real mini-dataset (5 imgs/breed) instead of 2-batch limit
- Auto portable export after training completes
- Better tqdm progress bars throughout

**Export & Standalone Packaging:**
- Added `portable` mode: self-contained folder with model + labels + metadata
- Improved progress bars on INT8 calibration
- Created `create_training_zip.py` script to generate a clean, standalone training zip package
- Added `.gitignore` configured to track `memory/` while ignoring `.venv/`, `outputs/`, `data/splits/`, `*.zip`, cache files


**Config:**
- Added `PORTABLE_EXPORT_DIR`, `SMOKE_SAMPLES_PER_BREED` constants