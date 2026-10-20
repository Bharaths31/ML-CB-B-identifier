# Graph Report - ML-CB-B-identifier  (2026-10-20)

## Corpus Check
- 147 files · ~283,406 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 39 file(s) not represented in the graph (top: (none) 8, .pt 8, .jsonl 8)

## Summary
- 1468 nodes · 2946 edges · 111 communities (86 shown, 25 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 93 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f8807f36`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- package.json
- data_collection.py
- select_profile
- cbam.py
- cattle_buffalo_tester.py
- local_train.py
- SharedConfigDB
- train.py
- main
- run_server
- Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)
- web_tfjs_engine.dart
- os
- init_run_logger
- android_tflite_engine.dart
- parity_check.py
- UserPage.jsx
- RunLogger
- ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition
- cropAndQuality.js
- _combined_class_ids
- data_pipeline.py
- OODDetector
- shutil
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
- evaluate_epoch
- DevTools.jsx
- create_test_eval_zip.py
- train_model.py
- i_model_service.dart
- _DS
- get_manager
- test_model.py
- labels_binary.txt — cattle/buffalo species labels
- dev.py
- config_api.py
- DevPage.jsx
- train_phase
- ModelManager
- image_preprocessor.dart
- cattle_buffalo_trainer.py
- Local Training Run Log
- Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)
- Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle
- ci workflow
- predict_single
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
- BreedClassifier
- 15. Android Deployment
- prepare_splits
- audit_data.py
- Export Formats
- get_logger
- 🐄 Breed Classifier — React Frontend
- api.js
- run_utils.py
- architecture.md
- 10. Configuration Reference
- Q: how the model utlize the efficientne lite 2, cbam ,ood
- Q: how does CNN works here
- model.py
- react
- BreedClassifier
- plot_confusion_matrix
- .list_available_profiles
- TraitClassifier
- Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle
- Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle
- _Tee
- TraitClassifier
- verify.py
- get_shared_config
- 16. Changelog
- SessionLogger
- CattleBuffaloDataset
- CosineHead
- .__init__
- useSharedConfig.js
- subprocess

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 39 edges
2. `main()` - 32 edges
3. `get_logger()` - 27 edges
4. `OODDetector` - 26 edges
5. `init_run_logger()` - 25 edges
6. `get_shared_config()` - 23 edges
7. `react` - 22 edges
8. `useI18n()` - 22 edges
9. `get_dataloaders()` - 21 edges
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

## Communities (111 total, 25 thin omitted)

### Community 0 - "package.json"
Cohesion: 0.10
Nodes (20): dependencies, react, react-dom, react-router-dom, socket.io-client, devDependencies, vite, @vitejs/plugin-react (+12 more)

### Community 1 - "data_collection.py"
Cohesion: 0.25
Nodes (17): _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file(), download_kaggle_dataset(), download_web_images(), _locate() (+9 more)

### Community 2 - "select_profile"
Cohesion: 0.15
Nodes (13): get_active_ood(), list_ood_profiles(), route, Select active OOD profile ('auto' or specific filename)., Toggle OOD detection on or off., List all available OOD calibration profiles with metadata., Update custom OOD thresholds., Upload a new OOD JSON configuration profile. (+5 more)

### Community 3 - "cbam.py"
Cohesion: 0.15
Nodes (8): Compatibility forwarding module for CBAM and SE attention mechanisms. All…, build_attention(), CBAM, ChannelAttention, Zero the last Conv2d weight in a Sequential so its output is 0., SEBlock, SpatialAttention, _zero_init_last()

### Community 4 - "cattle_buffalo_tester.py"
Cohesion: 0.08
Nodes (18): base64, compute_macro_f1(), find_breed_folders(), img_to_base64(), per_breed_report(), Walk to find leaf dirs with images (breed folders)., Compute macro-averaged F1 across all breeds., Compute macro-averaged F1 across all breeds. (+10 more)

### Community 5 - "local_train.py"
Cohesion: 0.12
Nodes (39): _banner(), build_dataset_inventory(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb(), _find_project_root(), _install_torch() (+31 more)

### Community 6 - "SharedConfigDB"
Cohesion: 0.13
Nodes (14): Any, _deep_merge(), Return a copy of the entire config tree., Get a value by dot-notation key path, e.g. 'ood.enabled'., Set a value by dot-notation key path and optionally emit a change event., Update multiple keys from a dictionary and save atomically., Reset config to default values and save., Recursively merge source into destination. (+6 more)

### Community 7 - "train.py"
Cohesion: 0.08
Nodes (27): 2026-09-06 — Hotfix: CUDA `total_mem` AttributeError, math, _color_jitter(), describe_transform(), _eval_transform(), _PadToSquare, Resize the LONG side to ``size`` and pad the short side to a square. A 4:3…, Build the training transform. Augmentation is OFF by default (see ``AUG_*`` /… (+19 more)

### Community 8 - "main"
Cohesion: 0.10
Nodes (21): _calibration_images(), _copy_to_app_assets(), create_portable_export(), export_onnx_int8(), __init__(), export_tflite(), representative_dataset(), main() (+13 more)

### Community 9 - "run_server"
Cohesion: 0.08
Nodes (28): A. Completed this session (verified), B1. Docs — README + detailed guide  *(requested)*, B2. Scripts alignment (dynamic sizing + per-model OOD), B4. Doc/knowledge sync, B5. Runtime verification on the GPU machine (cannot do here), B6. Optional / decisions, B. Remaining work (implement next), C. Acceptance criteria (+20 more)

### Community 10 - "Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 11 - "web_tfjs_engine.dart"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 12 - "os"
Cohesion: 0.16
Nodes (15): flask, flask_cors, io, os, pil, create_app(), Flask API server for the Breed Classifier React frontend. python server/app.py…, init_manager() (+7 more)

### Community 13 - "init_run_logger"
Cohesion: 0.16
Nodes (13): _bootstrap(), _looks_like_project(), Automatic execution-logger bootstrap. Python imports ``sitecustomize`` at…, _log_event(), Best-effort structured logging (never raises if the logger is absent)., get_logger(), init_run_logger(), _install_excepthook() (+5 more)

### Community 14 - "android_tflite_engine.dart"
Cohesion: 0.08
Nodes (23): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+15 more)

### Community 15 - "parity_check.py"
Cohesion: 0.15
Nodes (17): get_dataloaders(), _make_weighted_sampler(), mixed_collate(), Build train/val/test DataLoaders. Args: pin_memory: If None, auto-detect (True…, Effective-number-of-samples class balancing. Per-class sampling weight = 1 /…, accumulate(), evaluate_on_split(), finalize() (+9 more)

### Community 16 - "UserPage.jsx"
Cohesion: 0.21
Nodes (19): predict(), HerdSessionDrawer(), UserPage(), addSessionEntry(), clearSession(), csvCell(), downloadTextFile(), hasStorage() (+11 more)

### Community 17 - "RunLogger"
Cohesion: 0.27
Nodes (4): _dumps(), _git_commit(), _json_default(), RunLogger

### Community 18 - "ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition"
Cohesion: 0.25
Nodes (15): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+7 more)

### Community 19 - "cropAndQuality.js"
Cohesion: 0.07
Nodes (65): CameraCapture(), qualityFromFrame(), sleep(), stageSide(), ImageCropper(), stageSide(), ImageUpload(), LanguageToggle() (+57 more)

### Community 20 - "_combined_class_ids"
Cohesion: 0.12
Nodes (18): _apply_cosine_margin(), _combined_class_ids(), _compute_loss(), _contrastive_term(), masked_kd_loss(), masked_loss(), Soft cross-entropy with optional label smoothing and logit adjustment. When…, Add the additive angular margin to the TARGET class logits. Only applied to… (+10 more)

### Community 21 - "data_pipeline.py"
Cohesion: 0.09
Nodes (31): 2026-09-23 — Tail-bias regression fix: single imbalance mechanism, safe mixing, EMA, soft routing, timestamped outputs, random, compute_class_counts(), compute_class_priors(), compute_hashes(), compute_rare_classes(), _count_per_class(), cutmix() (+23 more)

### Community 22 - "OODDetector"
Cohesion: 0.06
Nodes (23): main(), OODDetector, Post-hoc energy-based out-of-distribution detector. Computes an OOD score from…, Strip directory prefixes, file extensions, and UI tags from model name., Find the matching OOD calibration JSON for a given model., Calibrate energy and MSP thresholds on an in-distribution DataLoader., Save calibrated thresholds to JSON with appropriate naming., E(x) = -T * log(Σ exp(f_i / T)) across all logit heads. (+15 more)

### Community 23 - "shutil"
Cohesion: 0.16
Nodes (10): create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), Path, pathlib, Create archive.zip for Colab upload from local organized dataset. Usage: python…, Create colab_project.zip containing only the files needed for Colab training.… (+2 more)

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
Cohesion: 0.07
Nodes (28): 10. Deferred Product Decisions (defaults shipped), 1. Executive Summary, 2. Current-State Audit, 3. Decision Record — Crop vs Overlay, 4. Capture UX, 5. Worked Coordinate Examples, 6. Feature Portfolio (14 candidates), 7. Implementation Backlog (+20 more)

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

### Community 34 - "evaluate_epoch"
Cohesion: 0.40
Nodes (4): evaluate_epoch(), _macro_scores(), no_grad, Macro-F1 and macro-recall from a confusion matrix (true x pred). Classes with…

### Community 35 - "DevTools.jsx"
Cohesion: 0.17
Nodes (12): getDevInfo(), getImageInfo(), getLogs(), getPresenterConfig(), savePresenterConfig(), ClassMaps(), DEFAULT_CFG, human() (+4 more)

### Community 36 - "create_test_eval_zip.py"
Cohesion: 0.20
Nodes (8): Automatic Mixed Precision (AMP), collections, Create a zip file of test-split images for Colab batch evaluation. Reads…, csv, Colab Testing & Evaluation, Colab Training, Kaggle algsoch/breed-cattle-buffalo dataset, Colab T4 GPU

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

### Community 41 - "test_model.py"
Cohesion: 0.09
Nodes (23): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, datetime, json, pil_exiftags, re, requests (+15 more)

### Community 42 - "labels_binary.txt — cattle/buffalo species labels"
Cohesion: 0.39
Nodes (8): 18 Indian buffalo breeds, 57 Indian cattle breeds, Flutter app (cattle_breed_app), labels_binary.txt — cattle/buffalo species labels, labels_buffalo.txt — 18 buffalo breed labels, labels_cattle.txt — 57 cattle breed labels, flutter_app/pubspec.yaml — cattle_breed_app manifest, flutter_app/web/index.html — Flutter web entry

### Community 43 - "dev.py"
Cohesion: 0.43
Nodes (6): dev_info(), image_info(), logs(), route, extract_image_metadata(), Extract detailed metadata from raw image bytes.

### Community 44 - "config_api.py"
Cohesion: 0.39
Nodes (7): get_config(), load_presenter_config(), route, Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON., save_presenter_config(), update_config()

### Community 45 - "DevPage.jsx"
Cohesion: 0.19
Nodes (11): App(), DevTools(), THEMES, ThemeToggle(), useSocket(), DevPage(), AppRouter(), router (+3 more)

### Community 46 - "train_phase"
Cohesion: 0.13
Nodes (16): 2026-09-24 — Blocking bug fixes, flip+RRC on by default, dataset inventory, 2026-09-24 — Unified execution logger (`logs/<exec_id>/`) + `ema_decay` fix, _apply_ema(), _ema_decay_at(), _fmt_metrics(), _mix_off_epoch(), _rare_keep_mask(), Per-sample bool: True = this sample's breed is rare (do not mix). (+8 more)

### Community 47 - "ModelManager"
Cohesion: 0.13
Nodes (9): ModelManager, Return list of available model names with metadata., Format large numbers: 6234567 → '6.23M'., Load a model checkpoint into memory., Discovers, loads, and caches models for prediction., Return currently configured active model name or sensible default., Set the active model across the entire application., Load breed label maps, prioritizing portable exports to avoid mismatch. (+1 more)

### Community 48 - "image_preprocessor.dart"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 49 - "cattle_buffalo_trainer.py"
Cohesion: 0.08
Nodes (32): build_dataset_inventory(), download_kaggle_dataset(), find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Download and unzip a Kaggle dataset into dest_dir., Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace. (+24 more)

### Community 50 - "Local Training Run Log"
Cohesion: 0.50
Nodes (5): Local Training Run Log, GPU VRAM Auto-Scaling, INT8 Conversion per_channel_affine Failure, QAT Phase Degradation, QAT Checkpoint Export Mismatch

### Community 51 - "Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)"
Cohesion: 0.67
Nodes (3): Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 53 - "ci workflow"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 54 - "predict_single"
Cohesion: 0.32
Nodes (8): predict_single(), preprocess_image(), Preprocess a PIL image for ONNX inference. Matches the evaluation transform…, Numerically stable softmax., Run full inference pipeline on a single PIL image. Returns: dict with species,…, softmax(), Image, ndarray

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

### Community 79 - "BreedClassifier"
Cohesion: 0.11
Nodes (16): detailed_eval(), load_model(), main(), print_report(), no_grad, Diagnose a trained checkpoint on a split (val or test). Reports, for one or two…, shot_bucket(), main() (+8 more)

### Community 80 - "15. Android Deployment"
Cohesion: 0.22
Nodes (9): 15. Android Deployment, Artifacts in the App, Model Sizes (measured, lite2), On-device Input Convention, Pipeline Overview, QAT — optional recovery path only, Step 1 — Train, Step 2 — Export (+1 more)

### Community 81 - "prepare_splits"
Cohesion: 0.16
Nodes (12): _collect_rows(), prepare_half_splits(), prepare_quarter_splits(), prepare_smoke_splits(), prepare_splits(), Create a dataset using a fraction of images per breed for faster training. Uses…, Create a dataset using 25% of images per breed for very fast training.…, Create a tiny dataset for smoke tests by sampling a few images per breed. This… (+4 more)

### Community 82 - "audit_data.py"
Cohesion: 0.29
Nodes (9): difflib, hashlib, ahash(), hamming(), list_breeds(), main(), md5(), Audit data/raw before a training run. REPORT ONLY — never deletes anything.… (+1 more)

### Community 83 - "Export Formats"
Cohesion: 0.33
Nodes (6): Export Formats, Input conventions (important!), Legacy `--mode int8` (removed), ONNX INT8 (ONNX Runtime Mobile), Portable Bundle, TFLite (full-integer INT8)

### Community 84 - "get_logger"
Cohesion: 0.23
Nodes (8): flask_socketio, Namespace, predict(), route, predict_image(), Run prediction on raw image bytes. Returns result dict., get_logger(), SystemEventsNamespace

### Community 85 - "🐄 Breed Classifier — React Frontend"
Cohesion: 0.40
Nodes (4): 🐄 Breed Classifier — React Frontend, Camera capture, Features, Setup

### Community 86 - "api.js"
Cohesion: 0.32
Nodes (12): getActiveOOD(), getModels(), getModelSpec(), getOODProfiles(), selectModel(), selectOODProfile(), toggleOOD(), updateOODThresholds() (+4 more)

### Community 87 - "run_utils.py"
Cohesion: 0.16
Nodes (13): atexit, logging, find_latest(), Run infrastructure: run ids / timestamped paths / checkpoint discovery +…, Newest file matching ``pattern`` in ``directory`` (by mtime), or None., Resolve a ``--checkpoint`` value (path, run-tag, or None) to a file. * an…, Make a run id safe to embed in a filename. Replaces illegal characters with…, Append ``_<run_id>`` to a directory path (no extension expected). (+5 more)

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

### Community 92 - "model.py"
Cohesion: 0.10
Nodes (24): argparse, copy, glob, importlib, numpy, pandas, Mine the most-confused breed pairs from a checkpoint's validation set. Writes…, Parity check: PyTorch checkpoint vs fp32 ONNX on a small image list. For each… (+16 more)

### Community 93 - "react"
Cohesion: 0.37
Nodes (8): downloadBlob(), exportOdt(), Batch(), DevConfigPanel(), ModelSelect(), PredictionProgress(), SingleImage(), react

### Community 94 - "BreedClassifier"
Cohesion: 0.39
Nodes (8): BreedClassifier, CBAM attention, Model Architecture, Model Tester GUI, EfficientNet-Lite backbone, ModelManager, Presenter configuration, SE attention

### Community 95 - "plot_confusion_matrix"
Cohesion: 0.67
Nodes (3): plot_confusion_matrix(), Plot a confusion matrix heatmap for a single species., Plot a confusion matrix heatmap for a single species.

### Community 101 - "TraitClassifier"
Cohesion: 0.18
Nodes (8): 2026-09-21 — Long-Tail Accuracy Overhaul: Logit Adjustment, Feature Metric Learning, Soft Routing, 70/15/15 Splits, evaluate_traits(), no_grad, Soft species routing: never let a hard binary argmax discard the other head.…, Like predict(), but returns zero-confidence result for OOD inputs., Masked multi-field CE. Returns (mean_loss, {field: acc})., Per-field accuracy over a loader (soft/hard-agnostic; uses features)., TraitClassifier

### Community 102 - "verify.py"
Cohesion: 0.27
Nodes (7): Compatibility forwarding module for EfficientNet-Lite backbone. All backbone…, EfficientNetLite, load_backbone_weights(), check_backbone(), check_full_model(), main(), Architecture sanity check: load backbone weights and run a forward pass.…

### Community 103 - "get_shared_config"
Cohesion: 0.31
Nodes (10): get_config(), get_config_key(), route, Update configuration values. Supports either: 1. {"key": "ood.enabled",…, Reset configuration to initial default schema., Retrieve a single key or sub-tree, e.g. /api/config/ood/enabled., Retrieve the entire current application configuration., reset_config() (+2 more)

### Community 104 - "16. Changelog"
Cohesion: 0.20
Nodes (10): 16. Changelog, 2026-09-05 — Major Update, 2026-09-06 — Colab + SOTA Hyperparameters + Android QAT, 2026-09-07 — Unified Kaggle Dataset & Colab Trainer Update, 2026-09-08 — Fix: `torch.compile` on Windows, 2026-09-13 — Colab Testing Notebook & Large-Scale Evaluation, 2026-09-15 — Architecture Improvements, EMA, & ImageNet Normalization Fix, 2026-09-15 — Presenter Mode, Logging & Advanced Image Metadata (+2 more)

### Community 105 - "SessionLogger"
Cohesion: 0.22
Nodes (3): File + stdout logger for the entire GUI session., Read the current log file contents., SessionLogger

### Community 106 - "CattleBuffaloDataset"
Cohesion: 0.25
Nodes (4): Dataset, CattleBuffaloDataset, Merge per-breed augmentation overrides on top of the base flags., resolve_breed_augment()

### Community 107 - "CosineHead"
Cohesion: 0.29
Nodes (4): 2026-09-24 — Breed trait heads, group-aware splits, hard negatives, tooling, CosineHead, _make_breed_head(), Normalised-feature scaled-cosine classifier (ArcFace-style). Forward returns…

### Community 109 - "useSharedConfig.js"
Cohesion: 0.70
Nodes (4): getSharedConfig(), resetSharedConfig(), updateSharedConfig(), useSharedConfig()

### Community 110 - "subprocess"
Cohesion: 0.60
Nodes (4): load_index(), main(), write_labels(), subprocess

## Knowledge Gaps
- **275 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+270 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 655 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BreedClassifier` connect `BreedClassifier` to `TraitClassifier`, `verify.py`, `train.py`, `run_server`, `test_model.py`, `CosineHead`, `os`, `ModelManager`, `cattle_buffalo_trainer.py`, `Export Formats`, `OODDetector`, ``python -m src.train` — Core Training Engine`, `model.py`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `Documentation Index` connect `Documentation Index` to `create_test_eval_zip.py`, `6. Data Pipeline`, `Export & Deployment`, `architecture.md`, `10. Configuration Reference`, `BreedClassifier`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `Export Modes` connect ``python -m src.train` — Core Training Engine` to `BreedClassifier`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `BreedClassifier` (e.g. with `Export Modes` and `Known Constraints & Gotchas`) actually correct?**
  _`BreedClassifier` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `main()` (e.g. with `OODDetector` and `_ln()`) actually correct?**
  _`main()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _275 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `package.json` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._