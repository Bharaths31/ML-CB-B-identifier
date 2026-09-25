# Graph Report - ML-CB-B-identifier  (2026-09-25)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1013 nodes · 2024 edges · 75 communities (55 shown, 20 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 60 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a5c82b64`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DevTools.jsx
- data_collection.py
- run_logger.py
- verify.py
- cattle_buffalo_tester.py
- local_train.py
- parity_check.py
- train.py
- src/export.py
- test_model.py
- Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)
- web_tfjs_engine.dart
- app.py
- test_master_cpu.py
- android_tflite_engine.dart
- ModelManager
- cattle_buffalo_trainer.py
- BreedClassifier
- ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition
- src/config.py
- setup_venv.py
- data_pipeline.py
- CattleBuffaloDataset
- SessionLogger
- camera_screen.dart
- prediction_result.dart
- 20260925-011758-29c2/manifest.json
- main
- os
- prepare_half_splits
- inference_controller.dart
- view_logs.py
- 20260925-151550-e085/manifest.json
- main.dart
- run_utils.py
- audit_data.py
- .__init__
- build_dataset_inventory
- i_model_service.dart
- _DS
- get_manager
- OODDetector
- labels_binary.txt — cattle/buffalo species labels
- dev.py
- config_api.py
- train_phase
- _groups_by_hash
- .forward
- image_preprocessor.dart
- routes/export.py
- Local Training Run Log
- Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like)
- Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle
- ci workflow
- compute_hashes
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
- no_grad

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 31 edges
2. `main()` - 28 edges
3. `get_logger()` - 21 edges
4. `init_run_logger()` - 21 edges
5. `get_dataloaders()` - 18 edges
6. `log_event()` - 17 edges
7. `find_latest_checkpoint()` - 15 edges
8. `main()` - 15 edges
9. `prepare_splits()` - 14 edges
10. `resolve_checkpoint()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `predict_image()` --calls--> `log_event()`  [INFERRED]
  server/services/inference.py → src/run_logger.py
- `_run()` --calls--> `get_logger()`  [INFERRED]
  local_train.py → server/utils/logger.py
- `main()` --calls--> `init_run_logger()`  [INFERRED]
  test_model.py → src/run_logger.py
- `main()` --calls--> `log_event()`  [INFERRED]
  test_model.py → src/run_logger.py
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

## Communities (75 total, 20 thin omitted)

### Community 0 - "DevTools.jsx"
Cohesion: 0.05
Nodes (49): dependencies, react, react-dom, socket.io-client, devDependencies, vite, @vitejs/plugin-react, name (+41 more)

### Community 1 - "data_collection.py"
Cohesion: 0.08
Nodes (34): collections, Create a zip file of test-split images for Colab batch evaluation. Reads…, create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), csv, _absolute() (+26 more)

### Community 2 - "run_logger.py"
Cohesion: 0.08
Nodes (22): atexit, logging, platform, _bootstrap(), _looks_like_project(), Automatic execution-logger bootstrap. Python imports ``sitecustomize`` at…, _dumps(), get_logger() (+14 more)

### Community 3 - "verify.py"
Cohesion: 0.09
Nodes (25): Automatic Mixed Precision (AMP), BreedClassifier, CBAM attention, Colab Testing & Evaluation, Colab Training, Documentation Index, Model Architecture, Model Tester GUI (+17 more)

### Community 4 - "cattle_buffalo_tester.py"
Cohesion: 0.06
Nodes (28): compute_macro_f1(), find_breed_folders(), img_to_base64(), per_breed_report(), plot_confusion_matrix(), predict_single(), preprocess_image(), Walk to find leaf dirs with images (breed folders). (+20 more)

### Community 5 - "local_train.py"
Cohesion: 0.14
Nodes (33): _banner(), build_dataset_inventory(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb(), _install_torch(), _log_event() (+25 more)

### Community 6 - "parity_check.py"
Cohesion: 0.09
Nodes (25): detailed_eval(), main(), print_report(), no_grad, shot_bucket(), get_dataloaders(), _log_event(), _make_weighted_sampler() (+17 more)

### Community 7 - "train.py"
Cohesion: 0.10
Nodes (27): math, _apply_cosine_margin(), _apply_ema(), _combined_class_ids(), _compute_loss(), _contrastive_term(), _ema_decay_at(), masked_kd_loss() (+19 more)

### Community 8 - "src/export.py"
Cohesion: 0.11
Nodes (23): _calibration_images(), _copy_to_app_assets(), create_portable_export(), export_onnx_int8(), __init__(), export_tflite(), representative_dataset(), main() (+15 more)

### Community 9 - "test_model.py"
Cohesion: 0.11
Nodes (25): base64, datetime, struct, build_html(), _build_js(), extract_image_metadata(), generate_odt_report(), get_model_spec() (+17 more)

### Community 10 - "Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 11 - "web_tfjs_engine.dart"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 12 - "app.py"
Cohesion: 0.18
Nodes (15): flask, flask_cors, flask_socketio, Namespace, create_app(), Flask API server for the Breed Classifier React frontend. python server/app.py…, init_manager(), predict() (+7 more)

### Community 13 - "test_master_cpu.py"
Cohesion: 0.11
Nodes (16): copy, importlib, json, pandas, random, CPU-only synthetic tests for the 2026-09-23 fixes. No dataset/GPU needed. Run:…, _M, CPU-only synthetic tests for the master-task fixes. No GPU/dataset needed. Run:… (+8 more)

### Community 14 - "android_tflite_engine.dart"
Cohesion: 0.08
Nodes (23): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+15 more)

### Community 15 - "ModelManager"
Cohesion: 0.13
Nodes (10): no_grad, ModelManager, main(), ModelManager, Format large numbers: 6234567 → '6.23M'., Discovers, loads, and caches models for prediction., Load breed label maps, prioritizing portable exports to avoid mismatch., Find all available checkpoints. (+2 more)

### Community 16 - "cattle_buffalo_trainer.py"
Cohesion: 0.15
Nodes (16): download_kaggle_dataset(), Download and unzip a Kaggle dataset into dest_dir., matplotlib, matplotlib_pyplot, full_evaluation(), main(), no_grad, _save_confusion() (+8 more)

### Community 17 - "BreedClassifier"
Cohesion: 0.15
Nodes (13): numpy, load_model(), Diagnose a trained checkpoint on a split (val or test). Reports, for one or two…, main(), expand_images(), main(), Parity check: PyTorch checkpoint vs fp32 ONNX on a small image list. For each…, soft_top5() (+5 more)

### Community 18 - "ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition"
Cohesion: 0.16
Nodes (20): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+12 more)

### Community 19 - "src/config.py"
Cohesion: 0.19
Nodes (11): argparse, Mine the most-confused breed pairs from a checkpoint's validation set. Writes…, build_attention(), CBAM, SEBlock, _macro_scores(), Macro-F1 and macro-recall from a confusion matrix (true x pred). Classes with…, torch (+3 more)

### Community 20 - "setup_venv.py"
Cohesion: 0.20
Nodes (17): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, re, check_python(), clean_broken_torch(), create_venv(), detect_nvidia_gpu() (+9 more)

### Community 21 - "data_pipeline.py"
Cohesion: 0.18
Nodes (19): compute_class_counts(), compute_class_priors(), compute_rare_classes(), _count_per_class(), cutmix(), _effective_num_weights(), _load_class_maps(), mixup() (+11 more)

### Community 22 - "CattleBuffaloDataset"
Cohesion: 0.12
Nodes (10): Dataset, CattleBuffaloDataset, _color_jitter(), _eval_transform(), _PadToSquare, Resize the LONG side to ``size`` and pad the short side to a square. A 4:3…, Merge per-breed augmentation overrides on top of the base flags., Build the training transform. Augmentation is OFF by default (see ``AUG_*`` /… (+2 more)

### Community 23 - "SessionLogger"
Cohesion: 0.12
Nodes (5): File + stdout logger for the entire server session., SessionLogger, Read the current log file contents., File + stdout logger for the entire GUI session., SessionLogger

### Community 24 - "camera_screen.dart"
Cohesion: 0.14
Nodes (16): ChangeNotifier, ../../domain/entities/prediction_result.dart, PredictionResult, CattleBreedApp, build, _buildBody, CameraScreen, _pickAndPredict (+8 more)

### Community 25 - "prediction_result.dart"
Cohesion: 0.12
Nodes (16): breed, breedConfidence, breedIndex, BreedScore, confidence, index, label, latencyMs (+8 more)

### Community 26 - "20260925-011758-29c2/manifest.json"
Cohesion: 0.12
Nodes (16): argv, cwd, duration_s, ended_at, env, RUN_EXEC_ID, exec_id, executable (+8 more)

### Community 27 - "main"
Cohesion: 0.13
Nodes (12): describe_transform(), Human-readable list of a Compose's ops (for startup logging)., finish_run(), _build_warmup_cosine_scheduler(), main(), Create a linear-warmup + cosine-annealing LR schedule., Pick the best available device and enable CUDA optimizations., setup_device() (+4 more)

### Community 28 - "os"
Cohesion: 0.18
Nodes (10): os, pil, build_template(), main(), Generate the ``data/breed_traits.json`` template for the trait aux-heads. Lists…, load_index(), main(), write_labels() (+2 more)

### Community 29 - "prepare_half_splits"
Cohesion: 0.16
Nodes (11): _collect_rows(), prepare_half_splits(), prepare_quarter_splits(), prepare_smoke_splits(), Create a dataset using a fraction of images per breed for faster training. Uses…, Create a dataset using 25% of images per breed for very fast training.…, Per-breed train/val/test index lists with hard minimums. A plain round() split…, Create a tiny dataset for smoke tests by sampling a few images per breed. This… (+3 more)

### Community 30 - "inference_controller.dart"
Cohesion: 0.15
Nodes (12): bool get, _busy, dispose, _error, initialize, _initialized, predictImage, _result (+4 more)

### Community 31 - "view_logs.py"
Cohesion: 0.38
Nodes (12): all_execs(), cmd_diff(), cmd_events(), cmd_list(), cmd_metrics(), cmd_summary(), _final(), load_json() (+4 more)

### Community 32 - "20260925-151550-e085/manifest.json"
Cohesion: 0.15
Nodes (12): argv, cwd, env, exec_id, executable, git_commit, hostname, module (+4 more)

### Community 33 - "main.dart"
Cohesion: 0.17
Nodes (11): data/ml_engines/android_tflite_engine.dart, data/ml_engines/web_tfjs_engine.dart, ../../domain/services/i_model_service.dart, build, controller, initialize, main, service (+3 more)

### Community 34 - "run_utils.py"
Cohesion: 0.20
Nodes (11): glob, main(), find_latest(), Run-scoped output helpers. Every training/export/evaluation run writes to a…, Resolve a ``--checkpoint`` value (path, run-tag, or None) to a file. * an…, Make a run id safe to embed in a filename. Replaces illegal characters with…, Append ``_<run_id>`` to a directory path (no extension expected)., Newest file matching ``pattern`` in ``directory`` (by mtime), or None. (+3 more)

### Community 35 - "audit_data.py"
Cohesion: 0.29
Nodes (9): difflib, hashlib, ahash(), hamming(), list_breeds(), main(), md5(), Audit data/raw before a training run. REPORT ONLY — never deletes anything.… (+1 more)

### Community 36 - ".__init__"
Cohesion: 0.27
Nodes (4): ChannelAttention, Zero the last Conv2d weight in a Sequential so its output is 0., SpatialAttention, _zero_init_last()

### Community 37 - "build_dataset_inventory"
Cohesion: 0.22
Nodes (8): build_dataset_inventory(), find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace., Write a JSON inventory of the dataset to…, Auto-detect breed folders inside source_base and copy/merge them into…

### Community 38 - "i_model_service.dart"
Cohesion: 0.22
Nodes (8): dart:typed_data, ../entities/prediction_result.dart, TfliteEngine, TfJsEngine, dispose, IModelService, initialize, predict

### Community 39 - "_DS"
Cohesion: 0.22
Nodes (4): _DS, _FakeModel, run_once(), synth_labels()

### Community 40 - "get_manager"
Cohesion: 0.39
Nodes (7): get_manager(), get_model_spec(), list_models(), model_spec(), route, upload_model(), Extract detailed model architecture and parameter info.

### Community 41 - "OODDetector"
Cohesion: 0.28
Nodes (5): OODDetector, E(x) = -T * log(Σ exp(f_i / T)) across all logit heads., Max softmax probability from binary head (cattle-vs-buffalo)., Returns (is_ood: bool, ood_score: float, details: dict). A sample is considered…, Post-hoc energy-based out-of-distribution detector. Computes an OOD score from…

### Community 42 - "labels_binary.txt — cattle/buffalo species labels"
Cohesion: 0.39
Nodes (8): 18 Indian buffalo breeds, 57 Indian cattle breeds, Flutter app (cattle_breed_app), labels_binary.txt — cattle/buffalo species labels, labels_buffalo.txt — 18 buffalo breed labels, labels_cattle.txt — 57 cattle breed labels, flutter_app/pubspec.yaml — cattle_breed_app manifest, flutter_app/web/index.html — Flutter web entry

### Community 43 - "dev.py"
Cohesion: 0.39
Nodes (6): pil_exiftags, dev_info(), image_info(), logs(), route, extract_image_metadata()

### Community 44 - "config_api.py"
Cohesion: 0.39
Nodes (7): get_config(), load_presenter_config(), route, save_presenter_config(), update_config(), Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON.

### Community 45 - "train_phase"
Cohesion: 0.25
Nodes (8): _fmt_metrics(), _mix_off_epoch(), Last epoch that still mixes; mixing is OFF for epochs after this.…, Train a single phase with AdamW, optional AMP, gradient accumulation, and label…, train_phase(), evaluate_traits(), no_grad, Per-field accuracy over a loader (soft/hard-agnostic; uses features).

### Community 46 - "_groups_by_hash"
Cohesion: 0.29
Nodes (5): _groups_by_hash(), _hamming(), Union-find groups of (index, hash) within `hamming` distance., Like ``_stratified_split`` but near-duplicate groups stay in one split.…, _stratified_split_dedup()

### Community 47 - ".forward"
Cohesion: 0.33
Nodes (3): no_grad, Soft species routing: never let a hard binary argmax discard the other head.…, Like predict(), but returns zero-confidence result for OOD inputs.

### Community 48 - "image_preprocessor.dart"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 49 - "routes/export.py"
Cohesion: 0.53
Nodes (4): io, export_odt(), route, generate_odt_report()

### Community 50 - "Local Training Run Log"
Cohesion: 0.50
Nodes (5): Local Training Run Log, GPU VRAM Auto-Scaling, INT8 Conversion per_channel_affine Failure, QAT Phase Degradation, QAT Checkpoint Export Mismatch

### Community 51 - "Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like)"
Cohesion: 0.40
Nodes (5): Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like), Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 52 - "Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle"
Cohesion: 0.60
Nodes (5): Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle, Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle, Gaolao_02 — Gaolao cattle: white body with darker grey head/neck and grey hump, short horizontal horns, rope tether; zebu cattle, Hariana_01 — Hariana cattle: white/light grey coat with darker grey head and muzzle, prominent hump, short horns; zebu cattle, Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle

### Community 53 - "ci workflow"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 54 - "compute_hashes"
Cohesion: 0.50
Nodes (4): compute_hashes(), _dhash(), Return {path: dhash} for every row, caching results in a CSV., 64-bit difference hash (no external deps). None if unreadable.

## Knowledge Gaps
- **150 isolated node(s):** `react`, `react-dom`, `socket.io-client`, `vite`, `@vitejs/plugin-react` (+145 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 423 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BreedClassifier` connect `BreedClassifier` to `run_utils.py`, `verify.py`, `src/export.py`, `test_master_cpu.py`, `ModelManager`, `cattle_buffalo_trainer.py`, `.forward`, `src/config.py`, `main`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `get_logger()` connect `app.py` to `local_train.py`, `dev.py`, `config_api.py`, `ModelManager`, `routes/export.py`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Why does `TraitClassifier` connect `main` to `test_master_cpu.py`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `main()` (e.g. with `breed_trait_targets()` and `build_trait_vocab()`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `react`, `react-dom`, `socket.io-client` to the rest of the system?**
  _150 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DevTools.jsx` be split into smaller, more focused modules?**
  _Cohesion score 0.05443371378402107 - nodes in this community are weakly interconnected._
- **Should `data_collection.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07610993657505286 - nodes in this community are weakly interconnected._