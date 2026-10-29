# Graph Report - ML-CB-B-identifier  (2026-10-29)

## Corpus Check
- 170 files · ~519,182 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 40 file(s) not represented in the graph (top: (none) 8, .pt 8, .jsonl 8)

## Summary
- 1906 nodes · 3979 edges · 130 communities (106 shown, 24 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 225 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `51b2a202`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- package.json
- data_collection.py
- get_logger
- model.py
- cattle_buffalo_tester.py
- local_train.py
- SharedConfigDB
- imageOptimizer.js
- src/export.py
- escalation_db.py
- Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)
- web_tfjs_engine.dart
- model_manager.py
- sessionStore.js
- android_tflite_engine.dart
- os
- react
- setup_venv.py
- ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition
- cropAndQuality.js
- train.py
- data_pipeline.py
- ModelManager
- I18nProvider.jsx
- camera_screen.dart
- prediction_result.dart
- 20260925-011758-29c2/manifest.json
- API Reference
- run.py
- React Camera & Product Features — Architecture + Implementation Plan
- inference_controller.dart
- view_logs.py
- 20260925-151550-e085/manifest.json
- main.dart
- test_master_cpu.py
- api.js
- get_dataloaders
- train_model.py
- i_model_service.dart
- cattle_buffalo_trainer.py
- get_manager
- ood_calibration.py
- labels_binary.txt — cattle/buffalo species labels
- breedTraits.test.js
- config_api.py
- prepare_splits
- 6. Training pipeline
- ModelManager
- image_preprocessor.dart
- BreedClassifier
- Local Training Run Log
- Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)
- Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle
- ci workflow
- inference.py
- BreedID — Cattle & Buffalo Breed Identification System
- run
- setup.sh script
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
- UserPage.jsx
- 3. Manual Step-by-Step Operations
- Export & Deployment
- 7.3 Component inventory
- 15. Android Deployment
- _combined_class_ids
- DevPage.jsx
- build_dataset_inventory
- OODDetector
- 🐄 Breed Classifier — React Frontend
- 16. Changelog
- DevTools.jsx
- architecture.md
- 10. Configuration Reference
- Q: how the model utlize the efficientne lite 2, cbam ,ood
- Q: how does CNN works here
- TraitClassifier
- 15. Known constraints, gotchas, and drift
- BreedClassifier
- compute_class_counts
- 5. ML Core
- TraitClassifier
- Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle
- Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle
- _Tee
- run_utils.py
- ExpertPage.jsx
- RunLogger
- Web Application (React)
- 9. React user-facing application
- 8. Non-ML Feature Inventory
- 15. Appendices
- run_server
- AdvancedResultDetails.jsx
- audit_data.py
- calibrate_from_folder
- 6. Data Pipeline
- Project Source of Truth — Cattle & Buffalo Breed Classifier
- 13. Runbook
- ._load_model
- SessionLogger
- 5. Input image handling (end-to-end)
- 12. Data formats and artifacts
- routes/export.py
- BreedImageDataset
- AGENTS.md
- 10. Developer control plane (`/dev`)
- 3. System Architecture
- 7. React Frontend
- evaluate_epoch
- _apply_ema
- 6. Serving Layer
- list_calibration_dirs

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 42 edges
2. `get_logger()` - 35 edges
3. `OODDetector` - 32 edges
4. `main()` - 32 edges
5. `useI18n()` - 30 edges
6. `UserPage()` - 29 edges
7. `react` - 28 edges
8. `get_shared_config()` - 25 edges
9. `init_run_logger()` - 25 edges
10. `7.3 Component inventory` - 24 edges

## Surprising Connections (you probably didn't know these)
- `2. Current-State Audit` --references--> `exportOdt()`  [INFERRED]
  PLAN_REACT_CAMERA_AND_FEATURES.md → breed-classifier-react/src/api.js
- `15.12 ODT empty report — **RESOLVED 2026-10-27**` --references--> `DevConfigPanel()`  [INFERRED]
  PROJECT_SOURCE_OF_TRUTH.md → breed-classifier-react/src/components/DevConfigPanel.jsx
- `7.2 State architecture` --references--> `useImageOptimizer()`  [INFERRED]
  REPORTS.md → breed-classifier-react/src/hooks/useImageOptimizer.js
- `15.4 WebSocket namespace — **RESOLVED 2026-10-27**` --references--> `useSocket()`  [INFERRED]
  PROJECT_SOURCE_OF_TRUTH.md → breed-classifier-react/src/hooks/useSocket.js
- `Phase 2 — reviewer differentiators (1–2 weeks)` --references--> `DevPage()`  [INFERRED]
  PLAN_REACT_CAMERA_AND_FEATURES.md → breed-classifier-react/src/pages/DevPage.jsx

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Breed Classifier Model Tester GUI Feature Set** — test_results_screenshot_20260914_213516_image_upload_panel, test_results_screenshot_20260914_213516_prediction_results_panel, test_results_screenshot_20260914_213516_select_model_dropdown, test_results_screenshot_20260914_213516_analyze_breed_button, test_results_screenshot_20260914_213516_lite2_fp32_onnx_model [EXTRACTED 0.95]
- **Label asset trio consumed by the Flutter inference engine** — flutter_app_assets_models_labels_binary, flutter_app_assets_models_labels_cattle, flutter_app_assets_models_labels_buffalo, concept_flutter_app [EXTRACTED 1.00]
- **QAT Failure and PTQ Pivot** — test_results_training_result_phase3_degradation, test_results_training_result_qat_export_mismatch, test_results_training_result_int8_conversion_failure [EXTRACTED 1.00]
- **Sequential Training Phases** — concept_phase_1_warmup, concept_phase_2_finetune, concept_phase_3_qat [EXTRACTED 1.00]
- **Colab T4 training setup flow** — colab_cattle_buffalo_trainer, kaggle_dataset, t4_gpu, scripts_create_colab_project_zip [INFERRED 0.75]
- **Cattle Breed Prediction Outputs Across Test Screenshots** — test_results_screenshot_20260913_135102_umblachery_prediction, test_results_screenshot_20260914_213516_amritmahal_prediction, test_results_screenshot_20260914_215842_amritmahal_prediction, test_results_screenshot_20260914_215909_kenkatha_prediction [INFERRED 0.85]

## Communities (130 total, 24 thin omitted)

### Community 0 - "package.json"
Cohesion: 0.10
Nodes (20): dependencies, react, react-dom, react-router-dom, socket.io-client, devDependencies, vite, @vitejs/plugin-react (+12 more)

### Community 1 - "data_collection.py"
Cohesion: 0.25
Nodes (17): _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file(), download_kaggle_dataset(), download_web_images(), _locate() (+9 more)

### Community 2 - "get_logger"
Cohesion: 0.10
Nodes (28): Namespace, get_active_ood(), list_ood_profiles(), route, Get active OOD configuration and currently resolved detector status., List all available OOD calibration profiles with metadata., Select active OOD profile ('auto' or specific filename)., Toggle OOD detection on or off. (+20 more)

### Community 3 - "model.py"
Cohesion: 0.08
Nodes (22): 4.1 `BreedClassifier`, Compatibility forwarding module for CBAM and SE attention mechanisms. All…, Compatibility forwarding module for EfficientNet-Lite backbone. All backbone…, build_attention(), CBAM, ChannelAttention, CosineHead, EfficientNetLite (+14 more)

### Community 4 - "cattle_buffalo_tester.py"
Cohesion: 0.05
Nodes (33): Automatic Mixed Precision (AMP), compute_macro_f1(), find_breed_folders(), img_to_base64(), per_breed_report(), plot_confusion_matrix(), predict_single(), preprocess_image() (+25 more)

### Community 5 - "local_train.py"
Cohesion: 0.08
Nodes (48): create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), _banner(), build_dataset_inventory(), _check_windows_build_tools(), _detect_nvidia_gpu() (+40 more)

### Community 6 - "SharedConfigDB"
Cohesion: 0.13
Nodes (14): Any, _deep_merge(), Return a copy of the entire config tree., Get a value by dot-notation key path, e.g. 'ood.enabled'., Set a value by dot-notation key path and optionally emit a change event., Update multiple keys from a dictionary and save atomically., Reset config to default values and save., Recursively merge source into destination. (+6 more)

### Community 7 - "imageOptimizer.js"
Cohesion: 0.11
Nodes (31): applyCLAHE(), applyClaheToImageData(), aspectRatio(), buildClaheMaps(), canvasToBlob(), COMMON_RATIOS, computeOptimizerLayout(), createCanvas() (+23 more)

### Community 8 - "src/export.py"
Cohesion: 0.08
Nodes (32): Known Constraints & Gotchas, 12.4 Portable bundle, 1.1 Canonical launch commands, 1.2 Run/output naming (actual), 1. Project snapshot, 4.4 Export contracts, _calibration_images(), _copy_to_app_assets() (+24 more)

### Community 9 - "escalation_db.py"
Cohesion: 0.10
Nodes (28): base64, csv, _coerce_payload(), create_escalation(), get_escalation(), list_escalations(), _payload_from_request(), route (+20 more)

### Community 10 - "Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 11 - "web_tfjs_engine.dart"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 12 - "model_manager.py"
Cohesion: 0.18
Nodes (15): flask, flask_cors, flask_socketio, create_app(), Flask API server for the Breed Classifier React frontend. python server/app.py…, init_manager(), bpa_qr(), dev_info() (+7 more)

### Community 13 - "sessionStore.js"
Cohesion: 0.24
Nodes (14): HerdSessionDrawer(), addSessionEntry(), clearSession(), csvCell(), downloadTextFile(), hasStorage(), SESSION_KEY, SESSION_LIMIT (+6 more)

### Community 14 - "android_tflite_engine.dart"
Cohesion: 0.08
Nodes (23): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+15 more)

### Community 15 - "os"
Cohesion: 0.12
Nodes (17): Create a zip file of test-split images for Colab batch evaluation. Reads…, os, build_template(), main(), Generate the ``data/breed_traits.json`` template for the trait aux-heads. Lists…, call_gemini(), main(), Populate breed_traits.json using Gemini 2.5 Flash API. Usage: python… (+9 more)

### Community 16 - "react"
Cohesion: 0.19
Nodes (22): downloadBlob(), exportOdt(), predict(), AdvancedBatchTester(), AdvancedPanel(), FIT_OPTIONS, Batch(), DevConfigPanel() (+14 more)

### Community 17 - "setup_venv.py"
Cohesion: 0.10
Nodes (45): importlib_metadata, RuntimeError, git_report(), main(), model_report(), newest_file(), path_label(), platform_python() (+37 more)

### Community 18 - "ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition"
Cohesion: 0.25
Nodes (15): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+7 more)

### Community 19 - "cropAndQuality.js"
Cohesion: 0.10
Nodes (41): analyzeFullFrame(), CameraCapture(), sleep(), ImageCropper(), stageSide(), ImageUpload(), QualityBadge(), QualityPill() (+33 more)

### Community 20 - "train.py"
Cohesion: 0.11
Nodes (22): 2026-09-06 — Hotfix: CUDA `total_mem` AttributeError, math, random, describe_transform(), Human-readable list of a Compose's ops (for startup logging)., breed_trait_targets(), build_trait_vocab(), load_trait_spec() (+14 more)

### Community 21 - "data_pipeline.py"
Cohesion: 0.12
Nodes (20): 2026-09-23 — Tail-bias regression fix: single imbalance mechanism, safe mixing, EMA, soft routing, timestamped outputs, compute_hashes(), cutmix(), _dhash(), _effective_num_weights(), _groups_by_hash(), _hamming(), mixup() (+12 more)

### Community 22 - "ModelManager"
Cohesion: 0.16
Nodes (7): ModelManager, Discovers, loads, and caches models for prediction., Load breed label maps, prioritizing the NEWEST portable export., Find all available checkpoints., Return list of available model names with metadata., Return (and cache) the OODDetector linked to THIS model's JSON., Dataset overview: inventory (if any), split counts and class-map sizes.

### Community 23 - "I18nProvider.jsx"
Cohesion: 0.33
Nodes (8): LanguageToggle(), I18nContext, I18nProvider(), getStoredLanguage(), LANGUAGES, STORAGE_KEY, translate(), TRANSLATIONS

### Community 24 - "camera_screen.dart"
Cohesion: 0.14
Nodes (16): ChangeNotifier, ../../domain/entities/prediction_result.dart, PredictionResult, CattleBreedApp, build, _buildBody, CameraScreen, _pickAndPredict (+8 more)

### Community 25 - "prediction_result.dart"
Cohesion: 0.12
Nodes (16): breed, breedConfidence, breedIndex, BreedScore, confidence, index, label, latencyMs (+8 more)

### Community 26 - "20260925-011758-29c2/manifest.json"
Cohesion: 0.12
Nodes (16): argv, cwd, duration_s, ended_at, env, RUN_EXEC_ID, exec_id, executable (+8 more)

### Community 27 - "API Reference"
Cohesion: 0.05
Nodes (39): API Reference, Augmentation (ALL OFF by default — opt-in per run), Compilation, Data & I/O, Data Mode *(mutually exclusive)*, Data Mode *(mutually exclusive — pick at most one)*, Device & Workers, Distillation (+31 more)

### Community 28 - "run.py"
Cohesion: 0.13
Nodes (15): get_npm_executable(), get_python_executable(), is_port_in_use(), main(), Single-command launcher for the Cattle & Buffalo Breed Classifier. Starts both…, Check if a network port is already bound., Resolve virtual environment python if present., Resolve npm, including the Windows .cmd shim. (+7 more)

### Community 29 - "React Camera & Product Features — Architecture + Implementation Plan"
Cohesion: 0.07
Nodes (29): 10. Deferred Product Decisions (defaults shipped), 12. Minimal User View, Header Advanced Panel, and Manual OOD Recalibration, 1. Executive Summary, 2. Current-State Audit, 3. Decision Record — Crop vs Overlay, 4. Capture UX (optional manual framing), 5. Worked Coordinate Examples, 6. Feature Portfolio (14 candidates) (+21 more)

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
Cohesion: 0.07
Nodes (21): copy, importlib, pandas, CPU-only synthetic tests for the 2026-09-23 fixes. No dataset/GPU needed. Run:…, # NOTE: make_run_id() with no args auto-increments VERSION.txt (v5, v6, ...),, _DS, _FakeModel, _M (+13 more)

### Community 35 - "api.js"
Cohesion: 0.27
Nodes (18): getActiveOOD(), getModels(), getModelSpec(), getOODDataDirs(), getOODProfiles(), getPresenterConfig(), readJsonResponse(), recalibrateOOD() (+10 more)

### Community 36 - "get_dataloaders"
Cohesion: 0.11
Nodes (18): 11. Zero-Crop 260×260 Optimizer (added), 17. Fast file index, 5.1 Data pipeline (`src/data_pipeline.py`, 1,150 LOC), CattleBuffaloDataset, _color_jitter(), _eval_transform(), get_dataloaders(), _make_weighted_sampler() (+10 more)

### Community 37 - "train_model.py"
Cohesion: 0.18
Nodes (16): platform, _banner(), _file_size_mb(), _find_project_root(), main(), preflight_check(), _python_exe(), Return venv python if present, else sys.executable. (+8 more)

### Community 38 - "i_model_service.dart"
Cohesion: 0.22
Nodes (8): dart:typed_data, ../entities/prediction_result.dart, TfliteEngine, TfJsEngine, dispose, IModelService, initialize, predict

### Community 39 - "cattle_buffalo_trainer.py"
Cohesion: 0.09
Nodes (35): download_kaggle_dataset(), Download and unzip a Kaggle dataset into dest_dir., B3. CPU tests to add, matplotlib, matplotlib_pyplot, full_evaluation(), main(), no_grad (+27 more)

### Community 40 - "get_manager"
Cohesion: 0.24
Nodes (13): get_manager(), head_out_features(), Best-effort output width of a breed head (Linear or CosineHead)., get_model_spec(), _head_dims(), list_models(), model_spec(), route (+5 more)

### Community 41 - "ood_calibration.py"
Cohesion: 0.18
Nodes (15): dataclasses, difflib, BreedImage, _candidate_keys(), derive_breed(), _match_in_maps(), normalize_breed_key(), Manual, folder-based OOD calibration shared by test_model.py and the Flask… (+7 more)

### Community 42 - "labels_binary.txt — cattle/buffalo species labels"
Cohesion: 0.39
Nodes (8): 18 Indian buffalo breeds, 57 Indian cattle breeds, Flutter app (cattle_breed_app), labels_binary.txt — cattle/buffalo species labels, labels_buffalo.txt — 18 buffalo breed labels, labels_cattle.txt — 57 cattle breed labels, flutter_app/pubspec.yaml — cattle_breed_app manifest, flutter_app/web/index.html — Flutter web entry

### Community 43 - "breedTraits.test.js"
Cohesion: 0.15
Nodes (17): TraitComparison(), ALIASES, BREED_TRAITS, breedKey(), compareTraits(), getBreedTraits(), TRAIT_FIELDS, buffalo (+9 more)

### Community 44 - "config_api.py"
Cohesion: 0.39
Nodes (7): get_config(), load_presenter_config(), route, Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON., save_presenter_config(), update_config()

### Community 45 - "prepare_splits"
Cohesion: 0.13
Nodes (15): 6.2 Dataset layout, split rules, class maps, _collect_rows(), _log_event(), prepare_half_splits(), prepare_quarter_splits(), prepare_smoke_splits(), prepare_splits(), Create a dataset using a fraction of images per breed for faster training. Uses… (+7 more)

### Community 46 - "6. Training pipeline"
Cohesion: 0.25
Nodes (8): 6.1 Entry points and call chain, 6.3 Transforms and augmentation — actual defaults, 6.5 Phases, optimizer, EMA, scheduling, 6.6 Metrics and checkpoint selection, 6.7 Post-training exports and OOD calibration, 6.8 Execution logs, 6.9 Training data helper scripts, 6. Training pipeline

### Community 47 - "ModelManager"
Cohesion: 0.09
Nodes (21): ModelManager, _last_weight(), _layer_num(), portable_bundle_sort_key(), Return currently configured active model name or sensible default., Set the active model across the entire application., Load the newest complete class-map pair (constructor default)., Read head widths from a checkpoint/ONNX file without keeping it. (+13 more)

### Community 48 - "image_preprocessor.dart"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 49 - "BreedClassifier"
Cohesion: 0.09
Nodes (25): argparse, collections, B2. Scripts alignment (dynamic sizing + per-model OOD), glob, numpy, detailed_eval(), load_model(), main() (+17 more)

### Community 50 - "Local Training Run Log"
Cohesion: 0.50
Nodes (5): Local Training Run Log, GPU VRAM Auto-Scaling, INT8 Conversion per_channel_affine Failure, QAT Phase Degradation, QAT Checkpoint Export Mismatch

### Community 51 - "Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)"
Cohesion: 0.67
Nodes (3): Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 53 - "ci workflow"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 54 - "inference.py"
Cohesion: 0.13
Nodes (20): 18. Feature release — 2026-10-27, 8.4 Prediction response contract, Verification snapshot (2026-10-27), predict(), route, _build_xray_payload(), _colorize(), confidence_band() (+12 more)

### Community 55 - "BreedID — Cattle & Buffalo Breed Identification System"
Cohesion: 0.11
Nodes (19): 10.1 Data provenance and quality, 10.2 Observability, 10. Data, Governance and Observability, 11.1 Automated checks, 11.2 Browser-level end-to-end evidence, 11.3 Last recorded training runs (historical logs, `Test results/`), 11. Quality, Verification and Evidence, 12. Performance and Resource Profile (+11 more)

### Community 72 - "Documentation Index"
Cohesion: 0.16
Nodes (8): Documentation Index, Installation & Setup, Pretrained backbone weights included in repo, Kaggle API credentials setup, Virtual environment setup, Local Training Documentation, Training Pipeline Documentation, mkdocs.yml — Material docs site config

### Community 74 - "Colab Training — Cattle & Buffalo Breed Classifier"
Cohesion: 0.20
Nodes (9): After Training, Colab Training — Cattle & Buffalo Breed Classifier, Dataset Options, Option A: Clone from GitHub (Easiest), Option B: Upload `colab_project.zip`, Option C: Google Drive, Quick Start, T4 GPU Settings (+1 more)

### Community 75 - "SessionLogger"
Cohesion: 0.22
Nodes (3): File + stdout logger for the entire server session., Read the current log file contents., SessionLogger

### Community 76 - "UserPage.jsx"
Cohesion: 0.29
Nodes (16): createEscalation(), UserPage(), buildEscalationPayload(), dequeue(), enqueueEscalation(), getBrowserGps(), isPermanentEscalationError(), notify() (+8 more)

### Community 77 - "3. Manual Step-by-Step Operations"
Cohesion: 0.13
Nodes (15): 1. Automated Pipeline (Recommended), 2. Model Testing GUI, 3.1 Virtual Environment, 3.2 Dataset Download, 3.3 Data Pipeline (Generate Splits), 3.4 Architecture Verification, 3.5 Training, 3.6 Evaluation (+7 more)

### Community 78 - "Export & Deployment"
Cohesion: 0.13
Nodes (15): Android Deployment, Deployment Options, Export Command, Export & Deployment, Export Formats, Input conventions (important!), Legacy `--mode int8` (removed), Manual Export Commands (+7 more)

### Community 79 - "7.3 Component inventory"
Cohesion: 0.17
Nodes (14): App(), DevTools(), persistTheme(), readStoredTheme(), THEMES, ThemeToggle(), useSocket(), 8.1 Launcher (+6 more)

### Community 80 - "15. Android Deployment"
Cohesion: 0.22
Nodes (9): 15. Android Deployment, Artifacts in the App, Model Sizes (measured, lite2), On-device Input Convention, Pipeline Overview, QAT — optional recovery path only, Step 1 — Train, Step 2 — Export (+1 more)

### Community 81 - "_combined_class_ids"
Cohesion: 0.16
Nodes (14): 6.4 Loss, sampling, and auxiliary objectives, _apply_cosine_margin(), _combined_class_ids(), _compute_loss(), _contrastive_term(), masked_kd_loss(), masked_loss(), Soft cross-entropy with optional label smoothing and logit adjustment. When… (+6 more)

### Community 82 - "DevPage.jsx"
Cohesion: 0.23
Nodes (10): getSharedConfig(), resetSharedConfig(), updateSharedConfig(), useSharedConfig(), DevPage(), AppRouter(), router, breed_classifier_react_src_styles (+2 more)

### Community 83 - "build_dataset_inventory"
Cohesion: 0.22
Nodes (8): build_dataset_inventory(), find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace., Write a JSON inventory of the dataset to…, Auto-detect breed folders inside source_base and copy/merge them into…

### Community 84 - "OODDetector"
Cohesion: 0.16
Nodes (9): 8.5 OOD runtime behavior, OODDetector, Post-hoc energy-based out-of-distribution detector. Computes an OOD score from…, Strip directory prefixes, file extensions, and UI tags from model name., Find the matching OOD calibration JSON for a given model., E(x) = -T * log(Σ exp(f_i / T)) across all logit heads., Max softmax probability from binary head (cattle-vs-buffalo)., Returns (is_ood: bool, ood_score: float, details: dict). A sample is considered… (+1 more)

### Community 85 - "🐄 Breed Classifier — React Frontend"
Cohesion: 0.40
Nodes (4): 🐄 Breed Classifier — React Frontend, Camera capture, Features, Setup

### Community 86 - "16. Changelog"
Cohesion: 0.08
Nodes (28): 16. Changelog, 2026-09-05 — Major Update, 2026-09-06 — Colab + SOTA Hyperparameters + Android QAT, 2026-09-07 — Unified Kaggle Dataset & Colab Trainer Update, 2026-09-08 — Fix: `torch.compile` on Windows, 2026-09-13 — Colab Testing Notebook & Large-Scale Evaluation, 2026-09-15 — Architecture Improvements, EMA, & ImageNet Normalization Fix, 2026-09-15 — Presenter Mode, Logging & Advanced Image Metadata (+20 more)

### Community 87 - "DevTools.jsx"
Cohesion: 0.21
Nodes (9): getDevInfo(), getImageInfo(), getLogs(), ClassMaps(), DEFAULT_CFG, human(), ImageMeta(), ModelSpec() (+1 more)

### Community 88 - "architecture.md"
Cohesion: 0.25
Nodes (7): 2. Architecture & Data Flow, 3. Directory Map, Client Capture Flow (React, zero-crop), Inference Flow, Model Tester GUI, ModelManager, Presenter configuration

### Community 89 - "10. Configuration Reference"
Cohesion: 0.50
Nodes (3): 10. Configuration Reference, Path Constants, `src/config.py` — All Constants

### Community 90 - "Q: how the model utlize the efficientne lite 2, cbam ,ood"
Cohesion: 0.50
Nodes (3): Answer, Q: how the model utlize the efficientne lite 2, cbam ,ood, Source Nodes

### Community 91 - "Q: how does CNN works here"
Cohesion: 0.50
Nodes (3): Answer, Q: how does CNN works here, Source Nodes

### Community 92 - "TraitClassifier"
Cohesion: 0.20
Nodes (8): 2026-09-21 — Long-Tail Accuracy Overhaul: Logit Adjustment, Feature Metric Learning, Soft Routing, 70/15/15 Splits, evaluate_traits(), no_grad, Soft species routing: never let a hard binary argmax discard the other head.…, Like predict(), but returns zero-confidence result for OOD inputs., Masked multi-field CE. Returns (mean_loss, {field: acc})., Per-field accuracy over a loader (soft/hard-agnostic; uses features)., TraitClassifier

### Community 93 - "15. Known constraints, gotchas, and drift"
Cohesion: 0.15
Nodes (13): 15.10 Tester vs web app, 15.11 Edge assets, 15.12 ODT empty report — **RESOLVED 2026-10-27**, 15.13 `RUN_ID_FORMAT` unused, 15.1 Documentation drift (verified), 15.2 Class-map / checkpoint pairing — **RESOLVED 2026-10-27**, 15.3 OOD profile pairing — **RESOLVED 2026-10-27**, 15.4 WebSocket namespace — **RESOLVED 2026-10-27** (+5 more)

### Community 94 - "BreedClassifier"
Cohesion: 0.60
Nodes (6): BreedClassifier, CBAM attention, Model Architecture, EfficientNet-Lite backbone, 3-head multi-task learning, SE attention

### Community 95 - "compute_class_counts"
Cohesion: 0.25
Nodes (10): main(), compute_class_counts(), compute_class_priors(), compute_rare_classes(), _count_per_class(), _load_class_maps(), Smoothed log class priors for logit adjustment. IMPORTANT: the effective-number…, Raw train image counts per breed, for shot-bucket diagnostics. (+2 more)

### Community 96 - "5. ML Core"
Cohesion: 0.17
Nodes (8): 5.2 Model (`src/model.py`, 919 LOC), 5.3 Training curriculum (`src/train.py`, 1,742 LOC), 5.4 Evaluation and metrics (`src/metrics.py`, `src/evaluate.py`), 5.5 OOD detection and calibration, 5.6 Export and deployment (`src/export.py`, 656 LOC), 5. ML Core, Enumerate all available OOD calibration files and their metrics., List all available OOD calibration files.

### Community 101 - "run_utils.py"
Cohesion: 0.08
Nodes (25): atexit, py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, datetime, json, logging, pil_exiftags (+17 more)

### Community 102 - "ExpertPage.jsx"
Cohesion: 0.36
Nodes (9): getEscalations(), resolveEscalation(), breedOptions(), EscalationCard(), ExpertPage(), FILTERS, metricCards(), parseJson() (+1 more)

### Community 103 - "RunLogger"
Cohesion: 0.15
Nodes (12): 7.1 `test_model.py` — standalone tester (ACTIVE but independent), 7.2 `src/evaluate.py`, 7.3 `src/parity_check.py`, 7.4 `src/verify.py`, 7.5 Manual CPU/smoke test suites, 7. Model testing and evaluation, 8.2 `ModelManager` (Flask), logs() (+4 more)

### Community 104 - "Web Application (React)"
Cohesion: 0.14
Nodes (14): 1. Quick start, 2. The capture → classify path, 3.1 Upload, 3.2 Live camera, 3.3 Result card, 3.4 Herd session, 3. User surface, 4. Advanced panel (header, next to Connected) (+6 more)

### Community 105 - "9. React user-facing application"
Cohesion: 0.12
Nodes (17): 9.10 Shareable PNG result card, 9.11 Internationalization, 9.12 Themes, 9.13 Batch tester (inside Advanced panel), 9.14 Status, error, accessibility, privacy behavior, 9.15 Config consumed by the user UI, 9.16 Legacy / unreachable React code, 9.1 Routes and active files (+9 more)

### Community 106 - "8. Non-ML Feature Inventory"
Cohesion: 0.33
Nodes (6): 8.1 Capture, framing and image preparation, 8.2 Confidence, explainability and safety, 8.3 Session, reporting and provenance, 8.4 Developer / operator control plane, 8.5 Product ergonomics, 8. Non-ML Feature Inventory

### Community 107 - "15. Appendices"
Cohesion: 0.40
Nodes (5): 15.1 Reproduction quick reference, 15.2 Key CLI surfaces, 15.3 Glossary, 15.4 File → responsibility index (selected), 15. Appendices

### Community 108 - "run_server"
Cohesion: 0.07
Nodes (32): A. Completed this session (verified), B1. Docs — README + detailed guide  *(requested)*, B4. Doc/knowledge sync, B5. Runtime verification on the GPU machine (cannot do here), B6. Optional / decisions, B. Remaining work (implement next), C. Acceptance criteria, D. Files touched so far (+24 more)

### Community 109 - "AdvancedResultDetails.jsx"
Cohesion: 0.17
Nodes (22): getBpaQrMatrix(), AdvancedResultDetails(), BPA_TAG_KEY, BpaRegistrationModal(), generateBpaTag(), LACTATION, SEXES, VACCINATION (+14 more)

### Community 110 - "audit_data.py"
Cohesion: 0.23
Nodes (9): hashlib, pil, ahash(), hamming(), list_breeds(), main(), md5(), Audit data/raw before a training run. REPORT ONLY — never deletes anything.… (+1 more)

### Community 111 - "calibrate_from_folder"
Cohesion: 0.23
Nodes (8): Recalibrate OOD thresholds for the selected model. With ``data_dir`` the folder…, Calibrate energy and MSP thresholds on an in-distribution DataLoader., Save calibrated thresholds to JSON with appropriate naming., calibrate_from_folder(), Resolve a user-supplied folder path against the project root., Run the selected model over breed-named images and return thresholds + coverage., resolve_data_dir(), Calibrate OOD thresholds for one model and save outputs/ood/<name>.json. When…

### Community 112 - "6. Data Pipeline"
Cohesion: 0.25
Nodes (7): 6. Data Pipeline, Augmentation (OFF by default — opt-in per run), Dataset Layout, Label Encoding, Performance Optimizations, Sampling, Split Strategy

### Community 113 - "Project Source of Truth — Cattle & Buffalo Breed Classifier"
Cohesion: 0.25
Nodes (7): 0. How to use this document, 11. Flutter edge app, 14. Invariants an agent must preserve, 16. Explicitly not implemented, 2. System map, 3. Repository map, Project Source of Truth — Cattle & Buffalo Breed Classifier

### Community 114 - "13. Runbook"
Cohesion: 0.22
Nodes (9): 13.0 Smart setup / status, 13.1 Train, 13.2 Export, 13.3 Evaluate / verify / parity, 13.4 Serve, 13.5 Test the standalone tester, 13.6 Frontend tests/build, 13.7 Manual OOD recalibration (+1 more)

### Community 115 - "._load_model"
Cohesion: 0.20
Nodes (7): 4.2 Dynamic head sizing / checkpoint loading, 4.3 OOD detector, 4. Model definition, no_grad, Resolve and cache an OOD detector for a given model or custom profile., Load a model checkpoint into memory., Run prediction on raw image bytes. Returns result dict.

### Community 116 - "SessionLogger"
Cohesion: 0.22
Nodes (3): File + stdout logger for the entire GUI session., Read the current log file contents., SessionLogger

### Community 117 - "5. Input image handling (end-to-end)"
Cohesion: 0.25
Nodes (8): 5.1 Canonical numeric contract, 5.4 Camera path (ACTIVE), 5.5 Interactive cropper (ACTIVE, optional), 5.6 On-device quality gate, 5.7 Server-side image handling, 5.8 Flutter edge handling, 5.9 What the image pipeline does *not* do, 5. Input image handling (end-to-end)

### Community 118 - "12. Data formats and artifacts"
Cohesion: 0.33
Nodes (6): 12.1 Split manifests, 12.2 Training traits, 12.3 OOD profile JSON, 12.5 Browser localStorage keys, 12.6 Expert escalation database, 12. Data formats and artifacts

### Community 120 - "routes/export.py"
Cohesion: 0.43
Nodes (5): io, export_odt(), route, generate_odt_report(), Generate an ODT report from prediction results. Args: results: single result…

### Community 121 - "BreedImageDataset"
Cohesion: 0.33
Nodes (3): BreedImageDataset, Dataset, Yields (tensor, class_id) so OODDetector.calibrate can consume it.

### Community 123 - "10. Developer control plane (`/dev`)"
Cohesion: 0.40
Nodes (5): 10.1 Model Checkpoints, 10.2 OOD Safety Profiles, 10.3 Shared Database (config), 10.4 System Logs, 10. Developer control plane (`/dev`)

### Community 124 - "3. System Architecture"
Cohesion: 0.40
Nodes (5): 3.1 Layer map, 3.2 Data flow — training, 3.3 Data flow — inference (web), 3.4 Key architectural decisions (ADR-001, `ADR.md`), 3. System Architecture

### Community 125 - "7. React Frontend"
Cohesion: 0.40
Nodes (5): 7.1 Stack and structure, 7.2 State architecture, 7.4 Client-side image pipeline (`src/utils/`), 7.5 Design system, 7. React Frontend

### Community 126 - "evaluate_epoch"
Cohesion: 0.40
Nodes (4): evaluate_epoch(), _macro_scores(), no_grad, Macro-F1 and macro-recall from a confusion matrix (true x pred). Classes with…

### Community 127 - "_apply_ema"
Cohesion: 0.40
Nodes (5): _apply_ema(), _ema_decay_at(), EMA decay for optimizer step ``step`` (1-based), with warm-up. ``decay_t =…, One EMA update: parameters AND floating buffers (BN running stats)., _ema_update()

### Community 128 - "6. Serving Layer"
Cohesion: 0.50
Nodes (4): 6.1 Two backends, one contract, 6.2 API surface, 6.3 Runtime services, 6. Serving Layer

### Community 129 - "list_calibration_dirs"
Cohesion: 0.50
Nodes (4): list_data_dirs(), Suggested folders for manual breed-named calibration., list_calibration_dirs(), Suggested calibration folders for the React/GUI pickers.

## Knowledge Gaps
- **405 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+400 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 831 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **24 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `11. Zero-Crop 260×260 Optimizer (added)` connect `get_dataloaders` to `React Camera & Product Features — Architecture + Implementation Plan`, `cropAndQuality.js`, `AdvancedResultDetails.jsx`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `_PadToSquare` connect `get_dataloaders` to `data_pipeline.py`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `OODDetector` connect `OODDetector` to `5. ML Core`, `test_master_cpu.py`, `get_logger`, `model.py`, `run_utils.py`, `ood_calibration.py`, `model_manager.py`, `ModelManager`, `calibrate_from_folder`, `Project Source of Truth — Cattle & Buffalo Breed Classifier`, `._load_model`, `train.py`, `ModelManager`, `compute_class_counts`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `BreedClassifier` (e.g. with `Export Modes` and `Known Constraints & Gotchas`) actually correct?**
  _`BreedClassifier` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `OODDetector` (e.g. with `3. Repository map` and `5.5 OOD detection and calibration`) actually correct?**
  _`OODDetector` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _405 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `package.json` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._