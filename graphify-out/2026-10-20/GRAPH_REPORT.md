# Graph Report - ML-CB-B-identifier  (2026-10-20)

## Corpus Check
- 153 files · ~472,612 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 39 file(s) not represented in the graph (top: (none) 8, .pt 8, .jsonl 8)

## Summary
- 1538 nodes · 3154 edges · 105 communities (79 shown, 26 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 100 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9081e31a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- package.json
- data_collection.py
- get_shared_config
- verify.py
- cattle_buffalo_tester.py
- local_train.py
- SharedConfigDB
- imageOptimizer.js
- main
- test_model.py
- Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)
- web_tfjs_engine.dart
- app.py
- useI18n
- android_tflite_engine.dart
- init_run_logger
- UserPage.jsx
- RunLogger
- ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition
- cropAndQuality.js
- train.py
- data_pipeline.py
- ModelManager
- I18nProvider.jsx
- camera_screen.dart
- prediction_result.dart
- 20260925-011758-29c2/manifest.json
- `python -m src.train` — Core Training Engine
- run.py
- React Camera & Product Features — Architecture + Implementation Plan
- inference_controller.dart
- view_logs.py
- 20260925-151550-e085/manifest.json
- main.dart
- test_master_cpu.py
- api.js
- route
- train_model.py
- i_model_service.dart
- _DS
- get_manager
- ood_calibration.py
- labels_binary.txt — cattle/buffalo species labels
- dev.py
- config_api.py
- get_logger
- traits.py
- ModelManager
- image_preprocessor.dart
- cattle_buffalo_trainer.py
- Local Training Run Log
- Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)
- Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle
- ci workflow
- Export Formats
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
- 6. Data Pipeline
- 3. Manual Step-by-Step Operations
- Export & Deployment
- model.py
- 15. Android Deployment
- BreedClassifier
- audit_data.py
- build_dataset_inventory
- calibrate_from_folder
- 🐄 Breed Classifier — React Frontend
- BreedImageDataset
- _RawOutputs
- architecture.md
- 10. Configuration Reference
- Q: how the model utlize the efficientne lite 2, cbam ,ood
- Q: how does CNN works here
- detailed_eval
- Batch.jsx
- BreedClassifier
- _apply_ema
- .list_available_profiles
- TraitClassifier
- Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle
- Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle
- _Tee
- OODDetector
- 16. Changelog
- CattleBuffaloDataset
- prepare_app_assets.py

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 39 edges
2. `main()` - 32 edges
3. `OODDetector` - 29 edges
4. `get_logger()` - 28 edges
5. `useI18n()` - 26 edges
6. `react` - 25 edges
7. `get_shared_config()` - 25 edges
8. `init_run_logger()` - 25 edges
9. `get_dataloaders()` - 23 edges
10. `log_event()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `2. Current-State Audit` --references--> `exportOdt()`  [INFERRED]
  PLAN_REACT_CAMERA_AND_FEATURES.md → breed-classifier-react/src/api.js
- `Phase 2 — reviewer differentiators (1–2 weeks)` --references--> `DevPage()`  [INFERRED]
  PLAN_REACT_CAMERA_AND_FEATURES.md → breed-classifier-react/src/pages/DevPage.jsx
- `1. Executive Summary` --references--> `UserPage()`  [INFERRED]
  PLAN_REACT_CAMERA_AND_FEATURES.md → breed-classifier-react/src/pages/UserPage.jsx
- `QAT — optional recovery path only` --references--> `_sanitize_state_dict()`  [INFERRED]
  docs/android-deployment.md → src/export.py
- `Export Modes` --references--> `BreedClassifier`  [INFERRED]
  docs/api-reference.md → src/model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Breed Classifier Model Tester GUI Feature Set** — test_results_screenshot_20260914_213516_image_upload_panel, test_results_screenshot_20260914_213516_prediction_results_panel, test_results_screenshot_20260914_213516_select_model_dropdown, test_results_screenshot_20260914_213516_analyze_breed_button, test_results_screenshot_20260914_213516_lite2_fp32_onnx_model [EXTRACTED 0.95]
- **Label asset trio consumed by the Flutter inference engine** — flutter_app_assets_models_labels_binary, flutter_app_assets_models_labels_cattle, flutter_app_assets_models_labels_buffalo, concept_flutter_app [EXTRACTED 1.00]
- **QAT Failure and PTQ Pivot** — test_results_training_result_phase3_degradation, test_results_training_result_qat_export_mismatch, test_results_training_result_int8_conversion_failure [EXTRACTED 1.00]
- **Sequential Training Phases** — concept_phase_1_warmup, concept_phase_2_finetune, concept_phase_3_qat [EXTRACTED 1.00]
- **Colab T4 training setup flow** — colab_cattle_buffalo_trainer, kaggle_dataset, t4_gpu, scripts_create_colab_project_zip [INFERRED 0.75]
- **Cattle Breed Prediction Outputs Across Test Screenshots** — test_results_screenshot_20260913_135102_umblachery_prediction, test_results_screenshot_20260914_213516_amritmahal_prediction, test_results_screenshot_20260914_215842_amritmahal_prediction, test_results_screenshot_20260914_215909_kenkatha_prediction [INFERRED 0.85]

## Communities (105 total, 26 thin omitted)

### Community 0 - "package.json"
Cohesion: 0.10
Nodes (20): dependencies, react, react-dom, react-router-dom, socket.io-client, devDependencies, vite, @vitejs/plugin-react (+12 more)

### Community 1 - "data_collection.py"
Cohesion: 0.09
Nodes (30): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), _absolute() (+22 more)

### Community 2 - "get_shared_config"
Cohesion: 0.31
Nodes (10): get_config(), get_config_key(), route, Update configuration values. Supports either: 1. {"key": "ood.enabled",…, Reset configuration to initial default schema., Retrieve a single key or sub-tree, e.g. /api/config/ood/enabled., Retrieve the entire current application configuration., reset_config() (+2 more)

### Community 3 - "verify.py"
Cohesion: 0.05
Nodes (27): Compatibility forwarding module for CBAM and SE attention mechanisms. All…, Compatibility forwarding module for EfficientNet-Lite backbone. All backbone…, build_attention(), CBAM, ChannelAttention, CosineHead, EfficientNetLite, evaluate_traits() (+19 more)

### Community 4 - "cattle_buffalo_tester.py"
Cohesion: 0.05
Nodes (34): Automatic Mixed Precision (AMP), compute_macro_f1(), find_breed_folders(), img_to_base64(), per_breed_report(), plot_confusion_matrix(), predict_single(), preprocess_image() (+26 more)

### Community 5 - "local_train.py"
Cohesion: 0.12
Nodes (39): _banner(), build_dataset_inventory(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb(), _find_project_root(), _install_torch() (+31 more)

### Community 6 - "SharedConfigDB"
Cohesion: 0.13
Nodes (14): Any, _deep_merge(), Return a copy of the entire config tree., Get a value by dot-notation key path, e.g. 'ood.enabled'., Set a value by dot-notation key path and optionally emit a change event., Update multiple keys from a dictionary and save atomically., Reset config to default values and save., Recursively merge source into destination. (+6 more)

### Community 7 - "imageOptimizer.js"
Cohesion: 0.14
Nodes (23): aspectRatio(), canvasToBlob(), COMMON_RATIOS, computeOptimizerLayout(), createCanvas(), decodeImage(), decodeWithCreateImageBitmap(), decodeWithImage() (+15 more)

### Community 8 - "main"
Cohesion: 0.12
Nodes (19): _calibration_images(), _copy_to_app_assets(), create_portable_export(), export_onnx_int8(), __init__(), export_tflite(), representative_dataset(), main() (+11 more)

### Community 9 - "test_model.py"
Cohesion: 0.05
Nodes (41): base64, A. Completed this session (verified), B1. Docs — README + detailed guide  *(requested)*, B4. Doc/knowledge sync, B5. Runtime verification on the GPU machine (cannot do here), B6. Optional / decisions, B. Remaining work (implement next), C. Acceptance criteria (+33 more)

### Community 10 - "Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 11 - "web_tfjs_engine.dart"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 12 - "app.py"
Cohesion: 0.30
Nodes (8): flask_cors, flask_socketio, create_app(), Flask API server for the Breed Classifier React frontend. python server/app.py…, init_manager(), init_logger(), time, werkzeug_utils

### Community 13 - "useI18n"
Cohesion: 0.14
Nodes (23): ImageUpload(), QualityBadge(), QualityPill(), STATUS_META, statusMeta(), statusToHint(), toneClass(), normalizeTop() (+15 more)

### Community 14 - "android_tflite_engine.dart"
Cohesion: 0.08
Nodes (23): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+15 more)

### Community 15 - "init_run_logger"
Cohesion: 0.24
Nodes (8): _bootstrap(), _looks_like_project(), Automatic execution-logger bootstrap. Python imports ``sitecustomize`` at…, init_run_logger(), _install_excepthook(), make_exec_id(), ``YYYYmmdd-HHMMSS-<4 hex>`` — unique per process, filesystem-safe., Initialise (once) and return the process RunLogger. Idempotent: a second call…

### Community 16 - "UserPage.jsx"
Cohesion: 0.19
Nodes (21): predict(), AdvancedBatchTester(), AdvancedPanel(), FIT_OPTIONS, HerdSessionDrawer(), UserPage(), addSessionEntry(), clearSession() (+13 more)

### Community 17 - "RunLogger"
Cohesion: 0.27
Nodes (4): _dumps(), _git_commit(), _json_default(), RunLogger

### Community 18 - "ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition"
Cohesion: 0.25
Nodes (15): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+7 more)

### Community 19 - "cropAndQuality.js"
Cohesion: 0.15
Nodes (26): ImageCropper(), stageSide(), analyzeCanvas(), analyzeImageData(), analyzeSourceCrop(), canvasToBlob(), centerSquareRect(), clamp() (+18 more)

### Community 20 - "train.py"
Cohesion: 0.05
Nodes (53): 2026-09-24 — Blocking bug fixes, flip+RRC on by default, dataset inventory, 2026-09-24 — Breed trait heads, group-aware splits, hard negatives, tooling, 2026-09-24 — Unified execution logger (`logs/<exec_id>/`) + `ema_decay` fix, math, describe_transform(), Human-readable list of a Compose's ops (for startup logging)., evaluate_epoch(), _macro_scores() (+45 more)

### Community 21 - "data_pipeline.py"
Cohesion: 0.06
Nodes (49): 2026-09-21 — Long-Tail Accuracy Overhaul: Logit Adjustment, Feature Metric Learning, Soft Routing, 70/15/15 Splits, main(), _collect_rows(), _color_jitter(), compute_class_priors(), compute_hashes(), compute_rare_classes(), _count_per_class() (+41 more)

### Community 22 - "ModelManager"
Cohesion: 0.11
Nodes (11): ModelManager, no_grad, Discovers, loads, and caches models for prediction., Resolve and cache an OOD detector for a given model or custom profile., Load breed label maps, prioritizing the NEWEST portable export., Find all available checkpoints., Return list of available model names with metadata., Load a model checkpoint into memory. (+3 more)

### Community 23 - "I18nProvider.jsx"
Cohesion: 0.18
Nodes (13): LanguageToggle(), I18nContext, I18nProvider(), getStoredLanguage(), LANGUAGES, STORAGE_KEY, translate(), TRANSLATIONS (+5 more)

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

### Community 28 - "run.py"
Cohesion: 0.15
Nodes (13): get_python_executable(), is_port_in_use(), main(), Single-command launcher for the Cattle & Buffalo Breed Classifier. Starts both…, Check if a network port is already bound., Resolve virtual environment python if present., Poll an HTTP endpoint until it answers 200 OK or timeout expires., wait_for_service() (+5 more)

### Community 29 - "React Camera & Product Features — Architecture + Implementation Plan"
Cohesion: 0.06
Nodes (32): 10. Deferred Product Decisions (defaults shipped), 11. Zero-Crop 260×260 Optimizer (added), 12. Minimal User View, Header Advanced Panel, and Manual OOD Recalibration, 1. Executive Summary, 2. Current-State Audit, 3. Decision Record — Crop vs Overlay, 4. Capture UX (optional manual framing), 5. Worked Coordinate Examples (+24 more)

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

### Community 34 - "test_master_cpu.py"
Cohesion: 0.12
Nodes (12): copy, importlib, pandas, pil, random, CPU-only synthetic tests for the 2026-09-23 fixes. No dataset/GPU needed. Run:…, # NOTE: make_run_id() with no args auto-increments VERSION.txt (v5, v6, ...),, _M (+4 more)

### Community 35 - "api.js"
Cohesion: 0.10
Nodes (37): getActiveOOD(), getDevInfo(), getImageInfo(), getLogs(), getModels(), getModelSpec(), getOODDataDirs(), getOODProfiles() (+29 more)

### Community 36 - "route"
Cohesion: 0.11
Nodes (19): get_active_ood(), list_data_dirs(), list_ood_profiles(), route, Get active OOD configuration and currently resolved detector status., List all available OOD calibration profiles with metadata., Select active OOD profile ('auto' or specific filename)., Toggle OOD detection on or off. (+11 more)

### Community 37 - "train_model.py"
Cohesion: 0.12
Nodes (29): platform, check_python(), clean_broken_torch(), create_venv(), detect_nvidia_gpu(), install_requirements(), log(), main() (+21 more)

### Community 38 - "i_model_service.dart"
Cohesion: 0.22
Nodes (8): dart:typed_data, ../entities/prediction_result.dart, TfliteEngine, TfJsEngine, dispose, IModelService, initialize, predict

### Community 39 - "_DS"
Cohesion: 0.22
Nodes (4): _DS, _FakeModel, run_once(), synth_labels()

### Community 40 - "get_manager"
Cohesion: 0.38
Nodes (8): get_manager(), get_model_spec(), list_models(), model_spec(), route, Extract detailed model architecture and parameter info., select_model(), upload_model()

### Community 41 - "ood_calibration.py"
Cohesion: 0.18
Nodes (15): dataclasses, difflib, BreedImage, _candidate_keys(), derive_breed(), _match_in_maps(), normalize_breed_key(), Manual, folder-based OOD calibration shared by test_model.py and the Flask… (+7 more)

### Community 42 - "labels_binary.txt — cattle/buffalo species labels"
Cohesion: 0.39
Nodes (8): 18 Indian buffalo breeds, 57 Indian cattle breeds, Flutter app (cattle_breed_app), labels_binary.txt — cattle/buffalo species labels, labels_buffalo.txt — 18 buffalo breed labels, labels_cattle.txt — 57 cattle breed labels, flutter_app/pubspec.yaml — cattle_breed_app manifest, flutter_app/web/index.html — Flutter web entry

### Community 43 - "dev.py"
Cohesion: 0.19
Nodes (12): flask, io, dev_info(), image_info(), logs(), route, export_odt(), route (+4 more)

### Community 44 - "config_api.py"
Cohesion: 0.39
Nodes (7): get_config(), load_presenter_config(), route, Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON., save_presenter_config(), update_config()

### Community 45 - "get_logger"
Cohesion: 0.27
Nodes (7): Namespace, predict(), route, predict_image(), Run prediction on raw image bytes. Returns result dict., get_logger(), SystemEventsNamespace

### Community 46 - "traits.py"
Cohesion: 0.18
Nodes (9): breed_trait_targets(), build_trait_vocab(), evaluate_traits(), no_grad, Breed trait auxiliary heads (training-only). Traits are free-form string fields…, Per-field accuracy over a loader (soft/hard-agnostic; uses features)., {field: {value: index}} for fields that have at least one filled value., Per-field LongTensor[75]: global class id -> trait index, or -1 to ignore.… (+1 more)

### Community 47 - "ModelManager"
Cohesion: 0.13
Nodes (9): ModelManager, Return list of available model names with metadata., Format large numbers: 6234567 → '6.23M'., Discovers, loads, and caches models for prediction., Load a model checkpoint into memory., Return currently configured active model name or sensible default., Set the active model across the entire application., Load breed label maps, prioritizing portable exports to avoid mismatch. (+1 more)

### Community 48 - "image_preprocessor.dart"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 49 - "cattle_buffalo_trainer.py"
Cohesion: 0.08
Nodes (39): download_kaggle_dataset(), Download and unzip a Kaggle dataset into dest_dir., Known Constraints & Gotchas, B2. Scripts alignment (dynamic sizing + per-model OOD), B3. CPU tests to add, matplotlib, matplotlib_pyplot, numpy (+31 more)

### Community 50 - "Local Training Run Log"
Cohesion: 0.50
Nodes (5): Local Training Run Log, GPU VRAM Auto-Scaling, INT8 Conversion per_channel_affine Failure, QAT Phase Degradation, QAT Checkpoint Export Mismatch

### Community 51 - "Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)"
Cohesion: 0.67
Nodes (3): Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 53 - "ci workflow"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 54 - "Export Formats"
Cohesion: 0.33
Nodes (6): Export Formats, Input conventions (important!), Legacy `--mode int8` (removed), ONNX INT8 (ONNX Runtime Mobile), Portable Bundle, TFLite (full-integer INT8)

### Community 72 - "Documentation Index"
Cohesion: 0.21
Nodes (8): Documentation Index, Installation & Setup, Pretrained backbone weights included in repo, Kaggle API credentials setup, Virtual environment setup, Local Training Documentation, Training Pipeline Documentation, 3-head multi-task learning

### Community 74 - "Colab Training — Cattle & Buffalo Breed Classifier"
Cohesion: 0.20
Nodes (9): After Training, Colab Training — Cattle & Buffalo Breed Classifier, Dataset Options, Option A: Clone from GitHub (Easiest), Option B: Upload `colab_project.zip`, Option C: Google Drive, Quick Start, T4 GPU Settings (+1 more)

### Community 75 - "SessionLogger"
Cohesion: 0.22
Nodes (3): File + stdout logger for the entire server session., Read the current log file contents., SessionLogger

### Community 76 - "6. Data Pipeline"
Cohesion: 0.25
Nodes (7): 6. Data Pipeline, Augmentation (OFF by default — opt-in per run), Dataset Layout, Label Encoding, Performance Optimizations, Sampling, Split Strategy

### Community 77 - "3. Manual Step-by-Step Operations"
Cohesion: 0.15
Nodes (13): 1. Automated Pipeline (Recommended), 2. Model Testing GUI, 3.1 Virtual Environment, 3.2 Dataset Download, 3.3 Data Pipeline (Generate Splits), 3.4 Architecture Verification, 3.5 Training, 3.6 Evaluation (+5 more)

### Community 78 - "Export & Deployment"
Cohesion: 0.20
Nodes (9): Android Deployment, Deployment Options, Export Command, Export & Deployment, Manual Export Commands, Mobile INT8 pipeline (recommended), Model Sizes (measured, lite2), Parity Gate (always run after exporting) (+1 more)

### Community 79 - "model.py"
Cohesion: 0.08
Nodes (46): argparse, atexit, collections, Create a zip file of test-split images for Colab batch evaluation. Reads…, csv, glob, json, os (+38 more)

### Community 80 - "15. Android Deployment"
Cohesion: 0.22
Nodes (9): 15. Android Deployment, Artifacts in the App, Model Sizes (measured, lite2), On-device Input Convention, Pipeline Overview, QAT — optional recovery path only, Step 1 — Train, Step 2 — Export (+1 more)

### Community 82 - "audit_data.py"
Cohesion: 0.33
Nodes (8): hashlib, ahash(), hamming(), list_breeds(), main(), md5(), Audit data/raw before a training run. REPORT ONLY — never deletes anything.…, Average hash (perceptual) as a bit array; None if unreadable.

### Community 83 - "build_dataset_inventory"
Cohesion: 0.22
Nodes (8): build_dataset_inventory(), find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace., Write a JSON inventory of the dataset to…, Auto-detect breed folders inside source_base and copy/merge them into…

### Community 84 - "calibrate_from_folder"
Cohesion: 0.23
Nodes (8): Recalibrate OOD thresholds for the selected model. With ``data_dir`` the folder…, Calibrate energy and MSP thresholds on an in-distribution DataLoader., Save calibrated thresholds to JSON with appropriate naming., calibrate_from_folder(), Resolve a user-supplied folder path against the project root., Run the selected model over breed-named images and return thresholds + coverage., resolve_data_dir(), Calibrate OOD thresholds for one model and save outputs/ood/<name>.json. When…

### Community 85 - "🐄 Breed Classifier — React Frontend"
Cohesion: 0.40
Nodes (4): 🐄 Breed Classifier — React Frontend, Camera capture, Features, Setup

### Community 86 - "BreedImageDataset"
Cohesion: 0.33
Nodes (3): BreedImageDataset, Dataset, Yields (tensor, class_id) so OODDetector.calibrate can consume it.

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

### Community 92 - "detailed_eval"
Cohesion: 0.40
Nodes (3): detailed_eval(), no_grad, shot_bucket()

### Community 93 - "Batch.jsx"
Cohesion: 0.19
Nodes (19): downloadBlob(), exportOdt(), Batch(), analyzeFullFrame(), CameraCapture(), sleep(), DevConfigPanel(), ModelSelect() (+11 more)

### Community 94 - "BreedClassifier"
Cohesion: 0.39
Nodes (8): BreedClassifier, CBAM attention, Model Architecture, Model Tester GUI, EfficientNet-Lite backbone, ModelManager, Presenter configuration, SE attention

### Community 95 - "_apply_ema"
Cohesion: 0.40
Nodes (5): _apply_ema(), _ema_decay_at(), EMA decay for optimizer step ``step`` (1-based), with warm-up. ``decay_t =…, One EMA update: parameters AND floating buffers (BN running stats)., _ema_update()

### Community 102 - "OODDetector"
Cohesion: 0.18
Nodes (8): OODDetector, Post-hoc energy-based out-of-distribution detector. Computes an OOD score from…, Strip directory prefixes, file extensions, and UI tags from model name., Find the matching OOD calibration JSON for a given model., E(x) = -T * log(Σ exp(f_i / T)) across all logit heads., Max softmax probability from binary head (cattle-vs-buffalo)., Returns (is_ood: bool, ood_score: float, details: dict). A sample is considered…, Compatibility forwarding module for OOD detector. All OOD detector definitions…

### Community 104 - "16. Changelog"
Cohesion: 0.12
Nodes (16): 16. Changelog, 2026-09-05 — Major Update, 2026-09-06 — Colab + SOTA Hyperparameters + Android QAT, 2026-09-06 — Hotfix: CUDA `total_mem` AttributeError, 2026-09-07 — Unified Kaggle Dataset & Colab Trainer Update, 2026-09-08 — Fix: `torch.compile` on Windows, 2026-09-13 — Colab Testing Notebook & Large-Scale Evaluation, 2026-09-15 — Architecture Improvements, EMA, & ImageNet Normalization Fix (+8 more)

### Community 106 - "CattleBuffaloDataset"
Cohesion: 0.22
Nodes (4): CattleBuffaloDataset, Dataset, Merge per-breed augmentation overrides on top of the base flags., resolve_breed_augment()

### Community 110 - "prepare_app_assets.py"
Cohesion: 0.83
Nodes (3): load_index(), main(), write_labels()

## Knowledge Gaps
- **283 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+278 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 678 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_PadToSquare` connect `React Camera & Product Features — Architecture + Implementation Plan` to `data_pipeline.py`?**
  _High betweenness centrality (0.243) - this node is a cross-community bridge._
- **Why does `11. Zero-Crop 260×260 Optimizer (added)` connect `React Camera & Product Features — Architecture + Implementation Plan` to `cropAndQuality.js`, `useI18n`?**
  _High betweenness centrality (0.241) - this node is a cross-community bridge._
- **Why does `ResultCard()` connect `useI18n` to `UserPage.jsx`, `React Camera & Product Features — Architecture + Implementation Plan`, `cropAndQuality.js`, `Batch.jsx`?**
  _High betweenness centrality (0.164) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `BreedClassifier` (e.g. with `Export Modes` and `Known Constraints & Gotchas`) actually correct?**
  _`BreedClassifier` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `main()` (e.g. with `OODDetector` and `_ln()`) actually correct?**
  _`main()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `OODDetector` (e.g. with `main()` and `ModelManager`) actually correct?**
  _`OODDetector` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _283 weakly-connected nodes found - possible documentation gaps or missing edges._