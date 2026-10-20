# Module Reference — Source Code Cross-Reference

> Quick-reference for navigating the (consolidated) `src/` package. Read this to
> trace a call or understand module dependencies. `src/config.py` is the single
> source of truth for all constants.

---

## Consolidation note (2026-10-06)

`src/` was consolidated from 16 files to **10** to reduce fragmentation. The
former modules were **merged** into cohesive homes (old import paths are gone):

| Former module | Now lives in | Section |
|---|---|---|
| `efficientnet_lite.py` | `model.py` | §1 Backbone |
| `cbam.py` | `model.py` | §2 Attention |
| `model.py` (heads) | `model.py` | §3/§4 Cosine head + BreedClassifier |
| `traits.py` | `model.py` | §5 Trait aux heads |
| `ood_detector.py` | `model.py` | §6 OOD detector |
| `metrics.py` | `train.py` | §7 Evaluation metrics |
| `run_logger.py` | `run_utils.py` | §2 Execution logger |

`config.py` stays separate (change behaviour there or via the matching CLI flag).

---

## Dependency Graph

```
config.py ← (imported by everything; constants + path creation)
    ↓
run_utils.py  (run ids / timestamped paths + execution logger)
    ↓
data_pipeline.py  (discovery, splits, transforms, mixing, sampler, loaders)
    ↓
model.py  (§1 backbone, §2 attention, §3/4 heads, §5 traits, §6 OOD)
    ↓
train.py  (losses, EMA, loop, CLI + §7 evaluate_epoch)
    ↓
export.py ← (model, run_utils)        evaluate.py ← (data_pipeline, train, export)
    ↓                                      ↓
parity_check.py ← (export, data_pipeline)   verify.py ← (model)

ood_calibration.py ← (model, config)     # manual folder-based OOD calibration
    ↓
server/model_manager.py  (registry, loader, predict, recalibrate_ood)
    ↓
server/routes/ood_api.py  (/api/ood/*)  →  React DevOODPanel
```

## Function Index

### `src/config.py`
Constants only, no functions. Side effect: creates output/log directories on import.

### `src/run_utils.py`
- §1 Naming/discovery: `make_run_id`, `sanitize_run_id`, `timestamped`,
  `timestamped_dir`, `unique_path`, `find_latest`, `find_latest_checkpoint`,
  `resolve_checkpoint`
- §2 Logging: `RunLogger`, `init_run_logger`, `get_logger`, `log_event`,
  `log_metrics`, `finish_run`

### `src/data_pipeline.py`
- §1 Validation/splits: `_validate_class_counts`, `_dhash`, `_hamming`,
  `compute_hashes`, `_groups_by_hash`, `_stratified_split`, `_stratified_split_dedup`
- §2 Discovery: `_collect_rows`
- §3 Split builders: `prepare_splits`, `prepare_half_splits`,
  `prepare_quarter_splits`, `prepare_smoke_splits`
- §4 Transforms: `_PadToSquare`, `resolve_breed_augment`, `_color_jitter`,
  `_train_transform`, `_eval_transform`, `describe_transform`
- §5 Dataset: `CattleBuffaloDataset`, `mixed_collate`
- §6 Mixing: `_rand_bbox`, `cutmix`, `mixup`
- §7 Sampling/priors: `_make_weighted_sampler`, `compute_class_priors`,
  `compute_class_counts`, `compute_rare_classes`, `_effective_num_weights`
- §8 Loaders: `get_dataloaders`

### `src/model.py`
- §1 Backbone: `activation_fn`, `MBConvBlock`, `EfficientNetLite`,
  `load_backbone_weights`
- §2 Attention: `ChannelAttention`, `SpatialAttention`, `CBAM`, `SEBlock`,
  `build_attention`, `_zero_init_last`
- §3/§4 Heads: `CosineHead`, `_make_breed_head`, `BreedClassifier`
  (`.forward`, `.forward_features`, `.predict`, `.predict_safe`, freeze helpers)
- §5 Traits: `load_trait_spec`, `build_trait_vocab`, `breed_trait_targets`,
  `TraitClassifier`, `evaluate_traits`
- §6 OOD: `OODDetector`

### `src/train.py`
- §1–§2 Device + losses: `setup_device`, `soft_ce`, `_apply_cosine_margin`,
  `supervised_contrastive_loss`, `_combined_class_ids`, `_contrastive_term`,
  `masked_loss`, `masked_kd_loss`, `_compute_loss`
- §3 Training core: `_rare_keep_mask`, `_mix_off_epoch`, `_ema_decay_at`,
  `_apply_ema`, `run_epoch`, `_build_warmup_cosine_scheduler`, `_fmt_metrics`,
  `train_phase`
- §4–§6: `create_portable_export`, `setup_qat`, `main`
- §7 Metrics (former `metrics.py`): `_macro_scores`, `evaluate_epoch`

### `src/export.py`
- §1 Wrappers: `_RawOutputs`, `_MobileOutputs`
- §2 Helpers: `_size`, `_sanitize_state_dict`, `_load_model`, `_write_label_files`,
  `_copy_to_app_assets`, `_calibration_images`
- §3 Quantization: `export_onnx_int8`, `export_tflite`
- §4 `create_portable_export`
- §5 `main`

### `src/evaluate.py`
- `full_evaluation`, `_save_confusion`, `main`

### `src/parity_check.py`
- `make_torch_runner`, `make_onnx_runner`, `make_tflite_runner`, `accumulate`,
  `finalize`, `synthetic_parity`, `main`

### `src/verify.py`
- `check_backbone`, `check_full_model`, `main`

### `src/ood_calibration.py` (2026-10-20)
- Breed naming: `normalize_breed_key`, `_candidate_keys`, `_match_in_maps`,
  `derive_breed`
- Folder scan: `scan_breed_images` (recursive walk, readability check, coverage)
- Calibration: `calibrate_from_folder` (matched images only, percentile driven)
- Paths: `resolve_data_dir`, `list_calibration_dirs`
- Dataset: `BreedImageDataset` (yields `(tensor, class_id)` for
  `OODDetector.calibrate`)

### `src/traits.py`
- `load_trait_spec`, `build_trait_vocab`, `breed_trait_targets`,
  `TraitClassifier` (re-exported helpers live in `src/model.py`)

### React application (`breed-classifier-react/src/`)
- `utils/imageOptimizer.js` — `computeOptimizerLayout`, `stepDownResize`,
  `optimizeSourceTo260`, `optimizeImageTo260`, `releaseOptimizerResult`,
  `aspectRatio`, `formatBytes`
- `utils/cropAndQuality.js` — `computeSourceRect`, `clampPan`, `analyzeImageData`,
  `analyzeCanvas`, `drawCropToCanvas`, `analyzeSourceCrop`, `makeCroppedFile`,
  `renderShareCard`, `shareOrDownloadCanvas`, `fileToDataUrl`, `loadImage`
- `utils/sessionStore.js` — `readSession`, `writeSession`, `addSessionEntry`,
  `makeSessionEntry`, `sessionStats`, `sessionToCsv`, `sessionToJson`,
  `sessionToOdtResults`, `thumbnailFromDataUrl`, `downloadTextFile`
- `hooks/` — `useImageOptimizer`, `useSharedConfig`, `useSocket`
- `components/` — 20 components: user surface (`UserPage`, `ImageUpload`,
  `CameraCapture`, `ImageCropper`, `ResultCard`, `TraitComparison`,
  `QualityBadge`, `HerdSessionDrawer`, `AdvancedPanel`,
  `AdvancedBatchTester`, `LanguageToggle`, `ThemeToggle`) and dev surface
  (`DevPage`, `DevModelPanel`, `DevOODPanel`, `DevConfigPanel`, `DevTools`,
  `Batch`, `SingleImage`, `ResultsPanel`, `ModelSelect`,
  `PredictionProgress`, `App`)

### External entry points
- `test_model.py` — stdlib HTTP tester + built-in GUI, 17 endpoints, `--dev` /
  `--present` modes (imports `src.model.OODDetector`, `src.ood_calibration`).
- `local_train.py` — 8-stage automated pipeline (wraps `python -m src.train`).
- `server/` — Flask serving layer: `model_manager.py` (registry, loader,
  `recalibrate_ood`), `shared_config.py` (settings DB), `routes/ood_api.py`
  (`/api/ood/recalibrate`, `/api/ood/data-dirs`), `services/odt_export.py`.
- `breed-classifier-react/` — React 18 + Vite app (public page + dev control
  plane) talking to the Flask API over the Vite proxy.
- `flutter_app/` — edge client (TFLite / TF.js engines) using the same
  preprocessing contract.
- `scripts/` — audit/diagnose/mine-confusions/calibrate-ood/parity/trait
  template + population/log viewer/ablations/app-asset tools.
