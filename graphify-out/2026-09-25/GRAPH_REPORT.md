# Graph Report - ML-CB-B-identifier  (2026-09-25)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 878 nodes · 1706 edges · 71 communities (50 shown, 21 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 59 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `caf562dd`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- BreedClassifier
- cattle_buffalo_tester.py
- local_train.py
- audit_data.py
- export.py
- cattle_buffalo_trainer.py
- cbam.py
- test_model.py
- Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)
- web_tfjs_engine.dart
- data_collection.py
- android_tflite_engine.dart
- os
- OODDetector
- test_master_cpu.py
- ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition
- CattleBuffaloDataset
- setup_venv.py
- prepare_splits
- main
- parity_check.py
- run_logger.py
- camera_screen.dart
- prediction_result.dart
- masked_loss
- 20260925-011758-29c2/manifest.json
- RunLogger
- inference_controller.dart
- view_logs.py
- data_pipeline.py
- evaluate.py
- 20260925-151550-e085/manifest.json
- sys
- main.dart
- init_run_logger
- SessionLogger
- build_dataset_inventory
- i_model_service.dart
- _DS
- labels_binary.txt — cattle/buffalo species labels
- _groups_by_hash
- _Tee
- image_preprocessor.dart
- get_dataloaders
- _apply_ema
- TraitClassifier
- Local Training Run Log
- Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like)
- Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle
- ci workflow
- compute_hashes
- get_model_spec
- mkdocs.yml — Material docs site config
- run
- setup.sh
- Project Rules (AGENTS.md)
- Config Centralization Convention
- CUDA/CPU Graceful Support
- Real Mini-Dataset Smoke Test Rule
- EMA (Exponential Moving Average)
- Masked Loss Computation
- CBAM (Convolutional Block Attention Module)
- EfficientNet-Lite2
- Knowledge Distillation
- Phase 1: All-heads Warmup
- Phase 2: Multi-task Fine-tune
- Phase 3: Quantization Aware Training
- Training Pipeline Documentation
- requirements.txt — Python dependencies

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 32 edges
2. `main()` - 28 edges
3. `init_run_logger()` - 22 edges
4. `get_dataloaders()` - 18 edges
5. `log_event()` - 17 edges
6. `main()` - 15 edges
7. `find_latest_checkpoint()` - 14 edges
8. `prepare_splits()` - 14 edges
9. `resolve_checkpoint()` - 14 edges
10. `ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition` - 14 edges

## Surprising Connections (you probably didn't know these)
- `ModelManager` --uses--> `BreedClassifier`  [INFERRED]
  test_model.py → src/model.py
- `stage_export()` --calls--> `find_latest_checkpoint()`  [INFERRED]
  local_train.py → src/run_utils.py
- `_log_event()` --calls--> `log_event()`  [INFERRED]
  local_train.py → src/run_logger.py
- `Best-effort structured logging (never raises if the logger is absent).` --rationale_for--> `_log_event()`  [EXTRACTED]
  src/data_pipeline.py → local_train.py
- `main()` --calls--> `init_run_logger()`  [INFERRED]
  local_train.py → src/run_logger.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Breed Classifier Model Tester GUI Feature Set** — test_results_screenshot_20260914_213516_image_upload_panel, test_results_screenshot_20260914_213516_prediction_results_panel, test_results_screenshot_20260914_213516_select_model_dropdown, test_results_screenshot_20260914_213516_analyze_breed_button, test_results_screenshot_20260914_213516_lite2_fp32_onnx_model [EXTRACTED 0.95]
- **Label asset trio consumed by the Flutter inference engine** — flutter_app_assets_models_labels_binary, flutter_app_assets_models_labels_cattle, flutter_app_assets_models_labels_buffalo, concept_flutter_app [EXTRACTED 1.00]
- **QAT Failure and PTQ Pivot** — test_results_training_result_phase3_degradation, test_results_training_result_qat_export_mismatch, test_results_training_result_int8_conversion_failure [EXTRACTED 1.00]
- **Cattle/Buffalo breed classifier test image set (Testing data, chunk 6 of 6)** — testing_data_amruthamahal_01_amruthamahal, testing_data_baragur_02_baragur, testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hallikaru_02_hallikaru, testing_data_hariana_01_hariana, testing_data_malenadu_gidda_01_malenadu_gidda, testing_data_nagori_01_nagori, testing_data_raghav_gir_bull_at_hyderabad_raghav_gir_bull [EXTRACTED 1.00]
- **Sequential Training Phases** — concept_phase_1_warmup, concept_phase_2_finetune, concept_phase_3_qat [EXTRACTED 1.00]
- **Colab T4 training setup flow** — colab_cattle_buffalo_trainer, kaggle_dataset, t4_gpu, scripts_create_colab_project_zip [INFERRED 0.75]
- **Karnataka draught-type grey/white zebu cattle (Amruthamahal/Hallikar-type)** — testing_data_amruthamahal_01_amruthamahal, testing_data_hallikaru_02_hallikaru [INFERRED 0.75]
- **White/light-coated humped zebu breeds (visually similar coat colour)** — testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hariana_01_hariana, testing_data_nagori_01_nagori [INFERRED 0.75]
- **Cattle Breed Prediction Outputs Across Test Screenshots** — test_results_screenshot_20260913_135102_umblachery_prediction, test_results_screenshot_20260914_213516_amritmahal_prediction, test_results_screenshot_20260914_215842_amritmahal_prediction, test_results_screenshot_20260914_215909_kenkatha_prediction [INFERRED 0.85]

## Communities (71 total, 21 thin omitted)

### Community 0 - "BreedClassifier"
Cohesion: 0.07
Nodes (24): detailed_eval(), load_model(), main(), print_report(), no_grad, Diagnose a trained checkpoint on a split (val or test). Reports, for one or two…, shot_bucket(), main() (+16 more)

### Community 1 - "cattle_buffalo_tester.py"
Cohesion: 0.05
Nodes (31): base64, compute_macro_f1(), find_breed_folders(), img_to_base64(), per_breed_report(), plot_confusion_matrix(), predict_single(), preprocess_image() (+23 more)

### Community 2 - "local_train.py"
Cohesion: 0.14
Nodes (33): _banner(), build_dataset_inventory(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb(), _install_torch(), _log_event() (+25 more)

### Community 3 - "audit_data.py"
Cohesion: 0.07
Nodes (26): Automatic Mixed Precision (AMP), collections, Create a zip file of test-split images for Colab batch evaluation. Reads…, create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), csv (+18 more)

### Community 4 - "export.py"
Cohesion: 0.11
Nodes (23): _calibration_images(), _copy_to_app_assets(), create_portable_export(), export_onnx_int8(), __init__(), export_tflite(), representative_dataset(), main() (+15 more)

### Community 5 - "cattle_buffalo_trainer.py"
Cohesion: 0.10
Nodes (23): download_kaggle_dataset(), Download and unzip a Kaggle dataset into dest_dir., math, matplotlib, matplotlib_pyplot, numpy, random, evaluate_epoch() (+15 more)

### Community 6 - "cbam.py"
Cohesion: 0.13
Nodes (17): BreedClassifier, CBAM attention, Documentation Index, Model Architecture, Model Tester GUI, EfficientNet-Lite backbone, ModelManager, 3-head multi-task learning (+9 more)

### Community 7 - "test_model.py"
Cohesion: 0.11
Nodes (24): logging, pil_exiftags, struct, build_html(), _build_js(), extract_image_metadata(), generate_odt_report(), load_presenter_config() (+16 more)

### Community 8 - "Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 9 - "web_tfjs_engine.dart"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 10 - "data_collection.py"
Cohesion: 0.16
Nodes (21): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file() (+13 more)

### Community 11 - "android_tflite_engine.dart"
Cohesion: 0.08
Nodes (23): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+15 more)

### Community 12 - "os"
Cohesion: 0.17
Nodes (17): glob, os, pil, Mine the most-confused breed pairs from a checkpoint's validation set. Writes…, expand_images(), main(), Parity check: PyTorch checkpoint vs fp32 ONNX on a small image list. For each…, soft_top5() (+9 more)

### Community 13 - "OODDetector"
Cohesion: 0.11
Nodes (13): OODDetector, E(x) = -T * log(Σ exp(f_i / T)) across all logit heads., Max softmax probability from binary head (cattle-vs-buffalo)., Returns (is_ood: bool, ood_score: float, details: dict). A sample is considered…, Post-hoc energy-based out-of-distribution detector. Computes an OOD score from…, ModelManager, no_grad, Discovers, loads, and caches models for prediction. (+5 more)

### Community 14 - "test_master_cpu.py"
Cohesion: 0.13
Nodes (14): copy, importlib, json, pandas, CPU-only synthetic tests for the 2026-09-23 fixes. No dataset/GPU needed. Run:…, _M, CPU-only synthetic tests for the master-task fixes. No GPU/dataset needed. Run:…, breed_trait_targets() (+6 more)

### Community 15 - "ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition"
Cohesion: 0.16
Nodes (20): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+12 more)

### Community 16 - "CattleBuffaloDataset"
Cohesion: 0.12
Nodes (11): Dataset, main(), CattleBuffaloDataset, _color_jitter(), _eval_transform(), _PadToSquare, Resize the LONG side to ``size`` and pad the short side to a square. A 4:3…, Merge per-breed augmentation overrides on top of the base flags. (+3 more)

### Community 17 - "setup_venv.py"
Cohesion: 0.23
Nodes (17): load_index(), main(), write_labels(), check_python(), clean_broken_torch(), create_venv(), detect_nvidia_gpu(), install_requirements() (+9 more)

### Community 18 - "prepare_splits"
Cohesion: 0.14
Nodes (14): _collect_rows(), _log_event(), prepare_half_splits(), prepare_quarter_splits(), prepare_smoke_splits(), prepare_splits(), Create a dataset using a fraction of images per breed for faster training. Uses…, Best-effort structured logging (never raises if the logger is absent). (+6 more)

### Community 19 - "main"
Cohesion: 0.15
Nodes (17): compute_class_counts(), compute_class_priors(), compute_rare_classes(), _count_per_class(), describe_transform(), _effective_num_weights(), _load_class_maps(), Human-readable list of a Compose's ops (for startup logging). (+9 more)

### Community 20 - "parity_check.py"
Cohesion: 0.19
Nodes (14): accumulate(), evaluate_on_split(), finalize(), main(), make_onnx_runner(), make_tflite_runner(), make_torch_runner(), run() (+6 more)

### Community 21 - "run_logger.py"
Cohesion: 0.14
Nodes (16): atexit, platform, finish_run(), get_logger(), _git_commit(), log_event(), log_metrics(), Unified per-execution logging for every python entry point. Each process gets… (+8 more)

### Community 22 - "camera_screen.dart"
Cohesion: 0.14
Nodes (16): ChangeNotifier, ../../domain/entities/prediction_result.dart, PredictionResult, CattleBreedApp, build, _buildBody, CameraScreen, _pickAndPredict (+8 more)

### Community 23 - "prediction_result.dart"
Cohesion: 0.12
Nodes (16): breed, breedConfidence, breedIndex, BreedScore, confidence, index, label, latencyMs (+8 more)

### Community 24 - "masked_loss"
Cohesion: 0.14
Nodes (16): _combined_class_ids(), _compute_loss(), _contrastive_term(), masked_kd_loss(), masked_loss(), SupCon (Khosla et al., 2020) over the auxiliary projection embedding. Pulls…, Global class id: cattle 0..C-1, buffalo C..C+B-1., Multi-task loss with species-masked breed CE and logit adjustment. The binary… (+8 more)

### Community 25 - "20260925-011758-29c2/manifest.json"
Cohesion: 0.12
Nodes (16): argv, cwd, duration_s, ended_at, env, RUN_EXEC_ID, exec_id, executable (+8 more)

### Community 26 - "RunLogger"
Cohesion: 0.30
Nodes (3): _dumps(), _json_default(), RunLogger

### Community 27 - "inference_controller.dart"
Cohesion: 0.15
Nodes (12): bool get, _busy, dispose, _error, initialize, _initialized, predictImage, _result (+4 more)

### Community 28 - "view_logs.py"
Cohesion: 0.38
Nodes (12): all_execs(), cmd_diff(), cmd_events(), cmd_list(), cmd_metrics(), cmd_summary(), _final(), load_json() (+4 more)

### Community 29 - "data_pipeline.py"
Cohesion: 0.23
Nodes (12): cutmix(), mixup(), _pairing_perm(), _rand_bbox(), Permutation that pairs each sample with another of the SAME species. A global…, Return `mixed` where keep is False and `original` where keep is True. `keep` is…, _restore_unmixed(), _rare_keep_mask() (+4 more)

### Community 30 - "evaluate.py"
Cohesion: 0.23
Nodes (11): full_evaluation(), main(), no_grad, _save_confusion(), _load_model(), make_run_id(), Make a run id safe to embed in a filename. Replaces illegal characters with…, Return a sanitized explicit run tag, or a fresh timestamp. Uses config… (+3 more)

### Community 31 - "20260925-151550-e085/manifest.json"
Cohesion: 0.15
Nodes (12): argv, cwd, env, exec_id, executable, git_commit, hostname, module (+4 more)

### Community 32 - "sys"
Cohesion: 0.21
Nodes (10): argparse, build_template(), main(), Generate the ``data/breed_traits.json`` template for the trait aux-heads. Lists…, call_gemini(), main(), Populate breed_traits.json using Gemini 2.5 Flash API. Usage: python…, shutil (+2 more)

### Community 33 - "main.dart"
Cohesion: 0.17
Nodes (11): data/ml_engines/android_tflite_engine.dart, data/ml_engines/web_tfjs_engine.dart, ../../domain/services/i_model_service.dart, build, controller, initialize, main, service (+3 more)

### Community 34 - "init_run_logger"
Cohesion: 0.24
Nodes (8): _bootstrap(), _looks_like_project(), Automatic execution-logger bootstrap. Python imports ``sitecustomize`` at…, init_run_logger(), _install_excepthook(), make_exec_id(), Initialise (once) and return the process RunLogger. Idempotent: a second call…, ``YYYYmmdd-HHMMSS-<4 hex>`` — unique per process, filesystem-safe.

### Community 35 - "SessionLogger"
Cohesion: 0.22
Nodes (3): Read the current log file contents., File + stdout logger for the entire GUI session., SessionLogger

### Community 36 - "build_dataset_inventory"
Cohesion: 0.22
Nodes (8): build_dataset_inventory(), find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace., Write a JSON inventory of the dataset to…, Auto-detect breed folders inside source_base and copy/merge them into…

### Community 37 - "i_model_service.dart"
Cohesion: 0.22
Nodes (8): dart:typed_data, ../entities/prediction_result.dart, TfliteEngine, TfJsEngine, dispose, IModelService, initialize, predict

### Community 38 - "_DS"
Cohesion: 0.22
Nodes (4): _DS, _FakeModel, run_once(), synth_labels()

### Community 39 - "labels_binary.txt — cattle/buffalo species labels"
Cohesion: 0.39
Nodes (8): 18 Indian buffalo breeds, 57 Indian cattle breeds, Flutter app (cattle_breed_app), labels_binary.txt — cattle/buffalo species labels, labels_buffalo.txt — 18 buffalo breed labels, labels_cattle.txt — 57 cattle breed labels, flutter_app/pubspec.yaml — cattle_breed_app manifest, flutter_app/web/index.html — Flutter web entry

### Community 40 - "_groups_by_hash"
Cohesion: 0.29
Nodes (5): _groups_by_hash(), _hamming(), Union-find groups of (index, hash) within `hamming` distance., Like ``_stratified_split`` but near-duplicate groups stay in one split.…, _stratified_split_dedup()

### Community 42 - "image_preprocessor.dart"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 43 - "get_dataloaders"
Cohesion: 0.40
Nodes (5): get_dataloaders(), _make_weighted_sampler(), mixed_collate(), Effective-number-of-samples class balancing. Per-class sampling weight = 1 /…, Build train/val/test DataLoaders. Args: pin_memory: If None, auto-detect (True…

### Community 44 - "_apply_ema"
Cohesion: 0.40
Nodes (5): _apply_ema(), _ema_decay_at(), EMA decay for optimizer step ``step`` (1-based), with warm-up. ``decay_t =…, One EMA update: parameters AND floating buffers (BN running stats)., _ema_update()

### Community 46 - "Local Training Run Log"
Cohesion: 0.50
Nodes (5): Local Training Run Log, GPU VRAM Auto-Scaling, INT8 Conversion per_channel_affine Failure, QAT Phase Degradation, QAT Checkpoint Export Mismatch

### Community 47 - "Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like)"
Cohesion: 0.40
Nodes (5): Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like), Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 48 - "Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle"
Cohesion: 0.60
Nodes (5): Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle, Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle, Gaolao_02 — Gaolao cattle: white body with darker grey head/neck and grey hump, short horizontal horns, rope tether; zebu cattle, Hariana_01 — Hariana cattle: white/light grey coat with darker grey head and muzzle, prominent hump, short horns; zebu cattle, Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle

### Community 49 - "ci workflow"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 50 - "compute_hashes"
Cohesion: 0.50
Nodes (4): compute_hashes(), _dhash(), Return {path: dhash} for every row, caching results in a CSV., 64-bit difference hash (no external deps). None if unreadable.

### Community 51 - "get_model_spec"
Cohesion: 0.50
Nodes (4): get_model_spec(), _human_number(), Extract detailed model architecture and parameter info., Format large numbers: 6234567 → '6.23M'.

## Knowledge Gaps
- **136 isolated node(s):** `_binaryLabels`, `_buffaloLabels`, `_cattleLabels`, `dispose`, `_exp` (+131 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 404 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BreedClassifier` connect `BreedClassifier` to `export.py`, `cattle_buffalo_trainer.py`, `test_model.py`, `os`, `OODDetector`, `test_master_cpu.py`, `CattleBuffaloDataset`, `main`, `evaluate.py`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `init_run_logger()` connect `init_run_logger` to `BreedClassifier`, `local_train.py`, `export.py`, `cattle_buffalo_trainer.py`, `test_model.py`, `_Tee`, `main`, `parity_check.py`, `run_logger.py`, `RunLogger`, `evaluate.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `RunLogger` connect `RunLogger` to `init_run_logger`, `run_logger.py`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `BreedClassifier` (e.g. with `EfficientNetLite` and `ModelManager`) actually correct?**
  _`BreedClassifier` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `main()` (e.g. with `breed_trait_targets()` and `build_trait_vocab()`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_binaryLabels`, `_buffaloLabels`, `_cattleLabels` to the rest of the system?**
  _136 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `BreedClassifier` be split into smaller, more focused modules?**
  _Cohesion score 0.06560283687943262 - nodes in this community are weakly interconnected._