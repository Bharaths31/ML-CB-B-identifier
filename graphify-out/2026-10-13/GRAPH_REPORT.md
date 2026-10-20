# Graph Report - ML-CB-B-identifier  (2026-10-06)

## Corpus Check
- 114 files · ~296,242 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 40 file(s) not represented in the graph (top: (none) 8, .pt 8, .jsonl 8)

## Summary
- 1162 nodes · 2210 edges · 97 communities (76 shown, 21 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 80 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9dbf3978`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DevTools.jsx
- data_collection.py
- sitecustomize.py
- model.py
- cattle_buffalo_tester.py
- local_train.py
- parity_check.py
- train.py
- src/export.py
- run_server
- Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)
- web_tfjs_engine.dart
- app.py
- cattle_buffalo_trainer.py
- android_tflite_engine.dart
- ModelManager
- log_event
- BreedClassifier
- ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition
- calibrate_ood.py
- setup_venv.py
- data_pipeline.py
- CattleBuffaloDataset
- SessionLogger
- camera_screen.dart
- prediction_result.dart
- 20260925-011758-29c2/manifest.json
- `python -m src.train` — Core Training Engine
- json
- prepare_half_splits
- inference_controller.dart
- view_logs.py
- 20260925-151550-e085/manifest.json
- main.dart
- run_utils.py
- os
- RunLogger
- build_dataset_inventory
- i_model_service.dart
- _DS
- get_manager
- test_model.py
- labels_binary.txt — cattle/buffalo species labels
- dev.py
- config_api.py
- create_training_zip.py
- _groups_by_hash
- ModelManager
- image_preprocessor.dart
- pil
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
- Documentation Index
- requirements.txt — Python dependencies
- Colab Training — Cattle & Buffalo Breed Classifier
- SessionLogger
- compute_class_priors
- 3. Manual Step-by-Step Operations
- Export & Deployment
- predict_single
- 15. Android Deployment
- _train_transform
- _Tee
- Export Formats
- Colab Training
- 🐄 Breed Classifier — React Frontend
- Common Operations
- timestamped
- architecture.md
- 10. Configuration Reference
- Q: how the model utlize the efficientne lite 2, cbam ,ood
- Q: how does CNN works here
- compute_macro_f1
- plot_confusion_matrix
- Model Tester GUI
- find_breed_folders
- show_misclassified

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 38 edges
2. `main()` - 28 edges
3. `init_run_logger()` - 25 edges
4. `log_event()` - 21 edges
5. `get_logger()` - 20 edges
6. `get_dataloaders()` - 19 edges
7. `find_latest_checkpoint()` - 17 edges
8. `prepare_splits()` - 16 edges
9. `16. Changelog` - 16 edges
10. `main()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Export Modes` --references--> `BreedClassifier`  [INFERRED]
  docs/api-reference.md → src/model.py
- `Portable Bundle` --references--> `BreedClassifier`  [INFERRED]
  docs/export-deployment.md → src/model.py
- `2026-09-06 — Hotfix: CUDA `total_mem` AttributeError` --references--> `setup_device()`  [INFERRED]
  docs/changelog.md → src/train.py
- `2026-09-15 — Architecture Improvements, EMA, & ImageNet Normalization Fix` --references--> `evaluate_epoch()`  [INFERRED]
  docs/changelog.md → src/train.py
- `2026-09-24 — Breed trait heads, group-aware splits, hard negatives, tooling` --references--> `prepare_splits()`  [INFERRED]
  docs/changelog.md → src/data_pipeline.py

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

## Communities (97 total, 21 thin omitted)

### Community 0 - "DevTools.jsx"
Cohesion: 0.05
Nodes (49): dependencies, react, react-dom, socket.io-client, devDependencies, vite, @vitejs/plugin-react, name (+41 more)

### Community 1 - "data_collection.py"
Cohesion: 0.25
Nodes (17): _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file(), download_kaggle_dataset(), download_web_images(), _locate() (+9 more)

### Community 2 - "sitecustomize.py"
Cohesion: 0.67
Nodes (3): _bootstrap(), _looks_like_project(), Automatic execution-logger bootstrap. Python imports ``sitecustomize`` at…

### Community 3 - "model.py"
Cohesion: 0.07
Nodes (26): BreedClassifier, CBAM attention, 2026-09-24 — Breed trait heads, group-aware splits, hard negatives, tooling, Model Architecture, EfficientNet-Lite backbone, 3-head multi-task learning, SE attention, build_attention() (+18 more)

### Community 4 - "cattle_buffalo_tester.py"
Cohesion: 0.11
Nodes (10): img_to_base64(), per_breed_report(), Print and plot per-breed accuracy., Convert an image file to base64 data URI for embedding in HTML., getpass, google_colab, ipython_display, onnxruntime (+2 more)

### Community 5 - "local_train.py"
Cohesion: 0.13
Nodes (35): _banner(), build_dataset_inventory(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb(), _install_torch(), _log_event() (+27 more)

### Community 6 - "parity_check.py"
Cohesion: 0.19
Nodes (14): get_dataloaders(), Build train/val/test DataLoaders. Args: pin_memory: If None, auto-detect (True…, accumulate(), evaluate_on_split(), finalize(), main(), make_onnx_runner(), make_tflite_runner() (+6 more)

### Community 7 - "train.py"
Cohesion: 0.07
Nodes (39): math, shutil, describe_transform(), Human-readable list of a Compose's ops (for startup logging)., breed_trait_targets(), build_trait_vocab(), evaluate_traits(), load_trait_spec() (+31 more)

### Community 8 - "src/export.py"
Cohesion: 0.11
Nodes (22): _calibration_images(), _copy_to_app_assets(), create_portable_export(), export_onnx_int8(), __init__(), export_tflite(), representative_dataset(), main() (+14 more)

### Community 9 - "run_server"
Cohesion: 0.16
Nodes (16): build_html(), extract_image_metadata(), generate_odt_report(), load_presenter_config(), Extract detailed metadata from raw image bytes., Run a minimal HTTP server with the GUI and prediction API., Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON. (+8 more)

### Community 10 - "Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 11 - "web_tfjs_engine.dart"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 12 - "app.py"
Cohesion: 0.18
Nodes (15): flask, flask_cors, flask_socketio, Namespace, create_app(), Flask API server for the Breed Classifier React frontend. python server/app.py…, init_manager(), predict() (+7 more)

### Community 13 - "cattle_buffalo_trainer.py"
Cohesion: 0.11
Nodes (20): download_kaggle_dataset(), Download and unzip a Kaggle dataset into dest_dir., copy, importlib, matplotlib, matplotlib_pyplot, numpy, pandas (+12 more)

### Community 14 - "android_tflite_engine.dart"
Cohesion: 0.08
Nodes (23): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+15 more)

### Community 15 - "ModelManager"
Cohesion: 0.16
Nodes (9): main(), ModelManager, no_grad, Discovers, loads, and caches models for prediction., Load breed label maps, prioritizing portable exports to avoid mismatch., Find all available checkpoints., Return list of available model names with metadata., Load a model checkpoint into memory. (+1 more)

### Community 16 - "log_event"
Cohesion: 0.13
Nodes (16): full_evaluation(), main(), no_grad, _save_confusion(), get_logger(), log_event(), log_metrics(), evaluate_epoch() (+8 more)

### Community 17 - "BreedClassifier"
Cohesion: 0.05
Nodes (38): QAT — optional recovery path only, 16. Changelog, 2026-09-05 — Major Update, 2026-09-06 — Colab + SOTA Hyperparameters + Android QAT, 2026-09-06 — Hotfix: CUDA `total_mem` AttributeError, 2026-09-07 — Unified Kaggle Dataset & Colab Trainer Update, 2026-09-08 — Fix: `torch.compile` on Windows, 2026-09-13 — Colab Testing Notebook & Large-Scale Evaluation (+30 more)

### Community 18 - "ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition"
Cohesion: 0.25
Nodes (15): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+7 more)

### Community 19 - "calibrate_ood.py"
Cohesion: 0.17
Nodes (11): main(), _eval_transform(), _load_class_maps(), _log_event(), prepare_splits(), Best-effort structured logging (never raises if the logger is absent)., OODDetector, Post-hoc energy-based out-of-distribution detector. Computes an OOD score from… (+3 more)

### Community 20 - "setup_venv.py"
Cohesion: 0.31
Nodes (14): platform, check_python(), clean_broken_torch(), create_venv(), detect_nvidia_gpu(), install_requirements(), log(), main() (+6 more)

### Community 21 - "data_pipeline.py"
Cohesion: 0.22
Nodes (12): cutmix(), _make_weighted_sampler(), mixed_collate(), mixup(), _pairing_perm(), _rand_bbox(), Data pipeline: discovery, splits, transforms, mixing, sampling, priors. SECTION…, Permutation that pairs each sample with another of the SAME species. A global… (+4 more)

### Community 22 - "CattleBuffaloDataset"
Cohesion: 0.22
Nodes (4): Dataset, CattleBuffaloDataset, Merge per-breed augmentation overrides on top of the base flags., resolve_breed_augment()

### Community 23 - "SessionLogger"
Cohesion: 0.22
Nodes (3): Read the current log file contents., File + stdout logger for the entire GUI session., SessionLogger

### Community 24 - "camera_screen.dart"
Cohesion: 0.14
Nodes (16): ChangeNotifier, ../../domain/entities/prediction_result.dart, PredictionResult, CattleBreedApp, build, _buildBody, CameraScreen, _pickAndPredict (+8 more)

### Community 25 - "prediction_result.dart"
Cohesion: 0.12
Nodes (16): breed, breedConfidence, breedIndex, BreedScore, confidence, index, label, latencyMs (+8 more)

### Community 26 - "20260925-011758-29c2/manifest.json"
Cohesion: 0.12
Nodes (16): argv, cwd, duration_s, ended_at, env, RUN_EXEC_ID, exec_id, executable (+8 more)

### Community 27 - "`python -m src.train` — Core Training Engine"
Cohesion: 0.06
Nodes (36): API Reference, Augmentation (ALL OFF by default — opt-in per run), Compilation, Data & I/O, Data Mode *(mutually exclusive)*, Data Mode *(mutually exclusive — pick at most one)*, Device & Workers, Distillation (+28 more)

### Community 28 - "json"
Cohesion: 0.16
Nodes (11): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, json, re, call_gemini(), main(), Populate breed_traits.json using Gemini 2.5 Flash API. Usage: python… (+3 more)

### Community 29 - "prepare_half_splits"
Cohesion: 0.16
Nodes (11): _collect_rows(), prepare_half_splits(), prepare_quarter_splits(), prepare_smoke_splits(), Create a dataset using a fraction of images per breed for faster training. Uses…, Create a dataset using 25% of images per breed for very fast training.…, Create a tiny dataset for smoke tests by sampling a few images per breed. This…, Per-breed train/val/test index lists with hard minimums. A plain round() split… (+3 more)

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
Cohesion: 0.11
Nodes (21): atexit, glob, find_latest(), init_run_logger(), _install_excepthook(), make_exec_id(), make_run_id(), Run infrastructure: run ids / timestamped paths / checkpoint discovery +… (+13 more)

### Community 35 - "os"
Cohesion: 0.11
Nodes (21): argparse, collections, Create a zip file of test-split images for Colab batch evaluation. Reads…, csv, difflib, hashlib, os, ahash() (+13 more)

### Community 36 - "RunLogger"
Cohesion: 0.08
Nodes (22): 2026-09-24 — Blocking bug fixes, flip+RRC on by default, dataset inventory, 2026-09-24 — Unified execution logger (`logs/<exec_id>/`) + `ema_decay` fix, 6. Data Pipeline, Augmentation (OFF by default — opt-in per run), Dataset Layout, Label Encoding, Performance Optimizations, Sampling (+14 more)

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
Nodes (7): get_manager(), get_model_spec(), list_models(), model_spec(), route, Extract detailed model architecture and parameter info., upload_model()

### Community 41 - "test_model.py"
Cohesion: 0.14
Nodes (14): base64, datetime, logging, pil_exiftags, struct, _build_js(), get_model_spec(), _human_number() (+6 more)

### Community 42 - "labels_binary.txt — cattle/buffalo species labels"
Cohesion: 0.39
Nodes (8): 18 Indian buffalo breeds, 57 Indian cattle breeds, Flutter app (cattle_breed_app), labels_binary.txt — cattle/buffalo species labels, labels_buffalo.txt — 18 buffalo breed labels, labels_cattle.txt — 57 cattle breed labels, flutter_app/pubspec.yaml — cattle_breed_app manifest, flutter_app/web/index.html — Flutter web entry

### Community 43 - "dev.py"
Cohesion: 0.39
Nodes (6): dev_info(), image_info(), logs(), route, extract_image_metadata(), Extract detailed metadata from raw image bytes.

### Community 44 - "config_api.py"
Cohesion: 0.39
Nodes (7): get_config(), load_presenter_config(), route, Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON., save_presenter_config(), update_config()

### Community 45 - "create_training_zip.py"
Cohesion: 0.18
Nodes (9): create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), Path, pathlib, Create archive.zip for Colab upload from local organized dataset. Usage: python…, Create colab_project.zip containing only the files needed for Colab training.… (+1 more)

### Community 46 - "_groups_by_hash"
Cohesion: 0.29
Nodes (5): _groups_by_hash(), _hamming(), Union-find groups of (index, hash) within `hamming` distance., Like ``_stratified_split`` but near-duplicate groups stay in one split.…, _stratified_split_dedup()

### Community 47 - "ModelManager"
Cohesion: 0.19
Nodes (7): ModelManager, Load a model checkpoint into memory., Discovers, loads, and caches models for prediction., Load breed label maps, prioritizing portable exports to avoid mismatch., Find all available checkpoints., Return list of available model names with metadata., Format large numbers: 6234567 → '6.23M'.

### Community 48 - "image_preprocessor.dart"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 49 - "pil"
Cohesion: 0.27
Nodes (6): io, pil, export_odt(), route, generate_odt_report(), Generate an ODT report from prediction results. Args: results: single result…

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
Nodes (4): compute_hashes(), _dhash(), 64-bit difference hash (no external deps). None if unreadable., Return {path: dhash} for every row, caching results in a CSV.

### Community 72 - "Documentation Index"
Cohesion: 0.23
Nodes (7): Documentation Index, Installation & Setup, Pretrained backbone weights included in repo, Kaggle API credentials setup, Virtual environment setup, Local Training Documentation, Training Pipeline Documentation

### Community 74 - "Colab Training — Cattle & Buffalo Breed Classifier"
Cohesion: 0.20
Nodes (9): After Training, Colab Training — Cattle & Buffalo Breed Classifier, Dataset Options, Option A: Clone from GitHub (Easiest), Option B: Upload `colab_project.zip`, Option C: Google Drive, Quick Start, T4 GPU Settings (+1 more)

### Community 75 - "SessionLogger"
Cohesion: 0.22
Nodes (3): File + stdout logger for the entire server session., Read the current log file contents., SessionLogger

### Community 76 - "compute_class_priors"
Cohesion: 0.25
Nodes (9): 2026-09-23 — Tail-bias regression fix: single imbalance mechanism, safe mixing, EMA, soft routing, timestamped outputs, compute_class_priors(), compute_rare_classes(), _count_per_class(), _effective_num_weights(), Boolean per-class masks marking breeds below `threshold` train images., Per-class effective-number weights 1/E_n (same as the sampler). Classes with…, Smoothed log class priors for logit adjustment. IMPORTANT: the effective-number… (+1 more)

### Community 77 - "3. Manual Step-by-Step Operations"
Cohesion: 0.22
Nodes (9): 3.1 Virtual Environment, 3.2 Dataset Download, 3.3 Data Pipeline (Generate Splits), 3.4 Architecture Verification, 3.5 Training, 3.6 Evaluation, 3.7 Model Export, 3.8 Export Parity Gate (+1 more)

### Community 78 - "Export & Deployment"
Cohesion: 0.22
Nodes (9): Android Deployment, Deployment Options, Export Command, Export & Deployment, Manual Export Commands, Mobile INT8 pipeline (recommended), Model Sizes (measured, lite2), Parity Gate (always run after exporting) (+1 more)

### Community 79 - "predict_single"
Cohesion: 0.32
Nodes (8): predict_single(), preprocess_image(), Preprocess a PIL image for ONNX inference. Matches the evaluation transform…, Numerically stable softmax., Run full inference pipeline on a single PIL image. Returns: dict with species,…, softmax(), Image, ndarray

### Community 80 - "15. Android Deployment"
Cohesion: 0.25
Nodes (8): 15. Android Deployment, Artifacts in the App, Model Sizes (measured, lite2), On-device Input Convention, Pipeline Overview, Step 1 — Train, Step 2 — Export, Step 3 — Parity Gate

### Community 81 - "_train_transform"
Cohesion: 0.25
Nodes (5): _color_jitter(), _PadToSquare, Resize the LONG side to ``size`` and pad the short side to a square. A 4:3…, Build the training transform. Augmentation is OFF by default (see ``AUG_*`` /…, _train_transform()

### Community 83 - "Export Formats"
Cohesion: 0.33
Nodes (6): Export Formats, Input conventions (important!), Legacy `--mode int8` (removed), ONNX INT8 (ONNX Runtime Mobile), Portable Bundle, TFLite (full-integer INT8)

### Community 84 - "Colab Training"
Cohesion: 0.40
Nodes (5): Automatic Mixed Precision (AMP), Colab Testing & Evaluation, Colab Training, Kaggle algsoch/breed-cattle-buffalo dataset, Colab T4 GPU

### Community 85 - "🐄 Breed Classifier — React Frontend"
Cohesion: 0.40
Nodes (4): 🐄 Breed Classifier — React Frontend, Camera capture, Features, Setup

### Community 86 - "Common Operations"
Cohesion: 0.40
Nodes (4): 1. Automated Pipeline (Recommended), 2. Model Testing GUI, 4. Creating Distribution Packages, Common Operations

### Community 87 - "timestamped"
Cohesion: 0.40
Nodes (5): Known Constraints & Gotchas, Insert ``_<run_id>`` before the extension.…, timestamped(), Soft cross-entropy with optional label smoothing and logit adjustment. When…, soft_ce()

### Community 88 - "architecture.md"
Cohesion: 0.50
Nodes (3): 2. Architecture & Data Flow, 3. Directory Map, Inference Flow

### Community 89 - "10. Configuration Reference"
Cohesion: 0.50
Nodes (3): 10. Configuration Reference, Path Constants, `src/config.py` — All Constants

### Community 90 - "Q: how the model utlize the efficientne lite 2, cbam ,ood"
Cohesion: 0.50
Nodes (3): Answer, Q: how the model utlize the efficientne lite 2, cbam ,ood, Source Nodes

### Community 91 - "Q: how does CNN works here"
Cohesion: 0.50
Nodes (3): Answer, Q: how does CNN works here, Source Nodes

### Community 92 - "compute_macro_f1"
Cohesion: 0.67
Nodes (3): compute_macro_f1(), Compute macro-averaged F1 across all breeds., Compute macro-averaged F1 across all breeds.

### Community 93 - "plot_confusion_matrix"
Cohesion: 0.67
Nodes (3): plot_confusion_matrix(), Plot a confusion matrix heatmap for a single species., Plot a confusion matrix heatmap for a single species.

### Community 94 - "Model Tester GUI"
Cohesion: 0.67
Nodes (3): Model Tester GUI, ModelManager, Presenter configuration

## Knowledge Gaps
- **235 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+230 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 544 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BreedClassifier` connect `BreedClassifier` to `os`, `model.py`, `train.py`, `src/export.py`, `test_model.py`, `app.py`, `cattle_buffalo_trainer.py`, `ModelManager`, `ModelManager`, `calibrate_ood.py`, `Export Formats`, `timestamped`, ``python -m src.train` — Core Training Engine`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `Documentation Index` connect `Documentation Index` to `model.py`, `RunLogger`, `Colab Training`, `architecture.md`, `10. Configuration Reference`, `Model Tester GUI`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `init_run_logger()` connect `run_utils.py` to `sitecustomize.py`, `model.py`, `RunLogger`, `local_train.py`, `parity_check.py`, `train.py`, `src/export.py`, `test_model.py`, `cattle_buffalo_trainer.py`, `ModelManager`, `log_event`, `_Tee`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `BreedClassifier` (e.g. with `Export Modes` and `Known Constraints & Gotchas`) actually correct?**
  _`BreedClassifier` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _235 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DevTools.jsx` be split into smaller, more focused modules?**
  _Cohesion score 0.05443371378402107 - nodes in this community are weakly interconnected._
- **Should `model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0730804810360777 - nodes in this community are weakly interconnected._