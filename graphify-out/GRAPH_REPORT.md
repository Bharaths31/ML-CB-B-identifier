# Graph Report - ML-CB-B-identifier  (2026-09-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 944 nodes · 1953 edges · 51 communities (46 shown, 5 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 89 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4ce9e092`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- export.py
- Documentation Index
- cattle_buffalo_tester.py
- cbam.py
- BreedClassifier
- setup_venv.py
- local_train.py
- run_logger.py
- CONTEXT.md — Context Database
- cattle_buffalo_trainer.py
- test_master_cpu.py
- data_pipeline.py
- Android Deployment
- train.py
- Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)
- web_tfjs_engine.dart
- android_tflite_engine.dart
- prepare_splits
- cutmix
- data_collection.py
- camera_screen.dart
- prediction_result.dart
- 20260924-110247-395b/manifest.json
- ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition
- inference_controller.dart
- image_preprocessor.dart
- view_logs.py
- 20260924-210203-b312/manifest.json
- main.dart
- ModelManager
- test_model.py
- CattleBuffaloDataset
- audit_data.py
- SessionLogger
- build_dataset_inventory
- _DS
- run_server
- ._load_model
- build_html
- _groups_by_hash
- Colab README
- evaluate_epoch
- TraitClassifier
- Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like)
- Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle
- ci workflow
- TfliteEngine
- run
- setup.sh
- src

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 45 edges
2. `main()` - 28 edges
3. `CONTEXT.md — Context Database` - 25 edges
4. `init_run_logger()` - 24 edges
5. `log_event()` - 19 edges
6. `get_dataloaders()` - 19 edges
7. `Colab README` - 16 edges
8. `main()` - 15 edges
9. `Documentation Index` - 15 edges
10. `make_run_id()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Binary-head species balancing` --semantically_similar_to--> `masked_loss()`  [INFERRED] [semantically similar]
  docs/changelog.md → src/train.py
- `masked_loss()` --semantically_similar_to--> `Masked per-head loss`  [INFERRED] [semantically similar]
  src/train.py → ADR.md
- `Portable export bundle` --references--> `BreedClassifier`  [EXTRACTED]
  docs/export-deployment.md → src/model.py
- `BreedClassifier` --implements--> `Species Routing Inference`  [INFERRED]
  src/model.py → docs/architecture.md
- `BreedClassifier` --shares_data_with--> `FEATURE_DIM`  [INFERRED]
  src/model.py → docs/config-reference.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Breed Classifier Model Tester GUI Feature Set** — test_results_screenshot_20260914_213516_image_upload_panel, test_results_screenshot_20260914_213516_prediction_results_panel, test_results_screenshot_20260914_213516_select_model_dropdown, test_results_screenshot_20260914_213516_analyze_breed_button, test_results_screenshot_20260914_213516_lite2_fp32_onnx_model [EXTRACTED 0.95]
- **Architecture Training & Inference Flow** — src_model_breedclassifier, docs_architecture_cbam, docs_architecture_masked_loss, docs_architecture_species_routing [EXTRACTED 1.00]
- **Accuracy/efficiency overhaul (distillation + EMA + long-tail)** — docs_changelog_knowledge_distillation, docs_changelog_ema_bug, docs_changelog_effective_number_sampler, docs_changelog_binary_species_balancing [EXTRACTED 1.00]
- **Label asset trio consumed by the Flutter inference engine** — flutter_app_assets_models_labels_binary, flutter_app_assets_models_labels_cattle, flutter_app_assets_models_labels_buffalo, concept_flutter_app [EXTRACTED 1.00]
- **Mobile INT8 export and parity verification** — docs_export_deployment_tflite, docs_export_deployment_onnx_int8, docs_export_deployment_parity_gate, docs_export_deployment_imagenet_normalization [EXTRACTED 1.00]
- **Multi-task inference flow: features -> binary/cattle/buffalo heads** — concept_breedclassifier, concept_three_head_classifier, concept_masked_loss, concept_mobile_input_convention [EXTRACTED 1.00]
- **Mobile INT8 export and deployment pipeline** — src_train, src_export, src_parity_check, tflite_int8, onnx_runtime_mobile_int8, flutter_app [EXTRACTED 1.00]
- **QAT Failure and PTQ Pivot** — test_results_training_result_phase3_degradation, test_results_training_result_qat_export_mismatch, test_results_training_result_int8_conversion_failure, docs_api_reference_qat [EXTRACTED 1.00]
- **Cattle/Buffalo breed classifier test image set (Testing data, chunk 6 of 6)** — testing_data_amruthamahal_01_amruthamahal, testing_data_baragur_02_baragur, testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hallikaru_02_hallikaru, testing_data_hariana_01_hariana, testing_data_malenadu_gidda_01_malenadu_gidda, testing_data_nagori_01_nagori, testing_data_raghav_gir_bull_at_hyderabad_raghav_gir_bull [EXTRACTED 1.00]
- **Local training pipeline (data -> train -> eval -> export)** — local_train, src_data_pipeline, src_train, src_evaluate, src_export [EXTRACTED 1.00]
- **Colab T4 training setup flow** — colab_cattle_buffalo_trainer, colab_readme, kaggle_dataset, t4_gpu, scripts_create_colab_project_zip [INFERRED 0.75]
- **Shared-backbone hierarchical training pipeline** — adr_hierarchical_heads, src_train_masked_loss, docs_training_pipeline_qat, docs_training_pipeline_knowledge_distillation [INFERRED 0.75]
- **White/light-coated humped zebu breeds (visually similar coat colour)** — testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hariana_01_hariana, testing_data_nagori_01_nagori [INFERRED 0.75]
- **Export & Parity Gate Pipeline** — docs_api_reference_local_train, docs_api_reference_src_export, docs_api_reference_parity_check, docs_api_reference_qat [INFERRED 0.85]
- **Cattle Breed Prediction Outputs Across Test Screenshots** — test_results_screenshot_20260913_135102_umblachery_prediction, test_results_screenshot_20260914_213516_amritmahal_prediction, test_results_screenshot_20260914_215842_amritmahal_prediction, test_results_screenshot_20260914_215909_kenkatha_prediction [INFERRED 0.85]
- **Mobile INT8 deployment pipeline: export -> parity gate -> Flutter app** — concept_tflite_export, concept_onnx_int8, concept_parity_gate, concept_flutter_app, concept_android_deployment [INFERRED 0.85]

## Communities (51 total, 5 thin omitted)

### Community 0 - "export.py"
Cohesion: 0.05
Nodes (62): Module Reference — Source Code Cross-Reference, Module dependency graph, glob, _bootstrap(), _looks_like_project(), Automatic execution-logger bootstrap. Python imports ``sitecustomize`` at…, full_evaluation(), main() (+54 more)

### Community 1 - "Documentation Index"
Cohesion: 0.06
Nodes (55): Training Internals & Data Flow — Deep Reference, Evaluation metrics (combined_top1/3/5), Image to tensor pipeline, Label encoding (binary/cattle/buffalo + masks), Loss computation, QAT internals (fuse conv-bn, per-tensor observers), Smoke test data flow, Training loop detail (+47 more)

### Community 2 - "cattle_buffalo_tester.py"
Cohesion: 0.05
Nodes (33): compute_macro_f1(), find_breed_folders(), img_to_base64(), per_breed_report(), plot_confusion_matrix(), predict_single(), preprocess_image(), Walk to find leaf dirs with images (breed folders). (+25 more)

### Community 3 - "cbam.py"
Cohesion: 0.07
Nodes (23): CBAM attention, Model Architecture, EfficientNet-Lite backbone, efficientnet_lite2.pth weights, efficientnet_lite4.pth weights, SE attention, build_attention(), CBAM (+15 more)

### Community 4 - "BreedClassifier"
Cohesion: 0.06
Nodes (30): Project Rules (AGENTS.md), Config Centralization Convention, Architecture & Data Flow, CBAM Attention, Masked Multi-Task Loss, Species Routing Inference, Configuration Reference, CBAM_AFTER_STAGE (+22 more)

### Community 5 - "setup_venv.py"
Cohesion: 0.08
Nodes (31): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, Create a zip file of test-split images for Colab batch evaluation. Reads…, create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude() (+23 more)

### Community 6 - "local_train.py"
Cohesion: 0.13
Nodes (35): _banner(), build_dataset_inventory(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb(), _install_torch(), _log_event() (+27 more)

### Community 7 - "run_logger.py"
Cohesion: 0.09
Nodes (15): atexit, logging, platform, _dumps(), _git_commit(), _install_excepthook(), _json_default(), make_exec_id() (+7 more)

### Community 8 - "CONTEXT.md — Context Database"
Cohesion: 0.17
Nodes (31): CONTEXT.md — Context Database, Android / mobile INT8 deployment path, BreedClassifier (backbone + attention + 3 heads), 18 Indian buffalo breeds, 57 Indian cattle breeds, CBAM attention module, CutMix / MixUp batch mixing on GPU, Effective-number-of-samples WeightedRandomSampler (beta=0.999) (+23 more)

### Community 9 - "cattle_buffalo_trainer.py"
Cohesion: 0.16
Nodes (19): argparse, download_kaggle_dataset(), Download and unzip a Kaggle dataset into dest_dir., collections, json, matplotlib, matplotlib_pyplot, numpy (+11 more)

### Community 10 - "test_master_cpu.py"
Cohesion: 0.10
Nodes (19): copy, importlib, pandas, pil, random, CPU-only synthetic tests for the 2026-09-23 fixes. No dataset/GPU needed. Run:…, _M, CPU-only synthetic tests for the master-task fixes. No GPU/dataset needed. Run:… (+11 more)

### Community 11 - "data_pipeline.py"
Cohesion: 0.13
Nodes (23): _color_jitter(), compute_class_counts(), compute_class_priors(), compute_rare_classes(), _count_per_class(), _effective_num_weights(), _eval_transform(), get_dataloaders() (+15 more)

### Community 12 - "Android Deployment"
Cohesion: 0.13
Nodes (26): CUDA/CPU Graceful Support, Real Mini-Dataset Smoke Test Rule, Knowledge distillation (teacher-student), Android Deployment, API Reference, Knowledge Distillation, local_train.py Pipeline, src.parity_check Export Parity Gate (+18 more)

### Community 13 - "train.py"
Cohesion: 0.12
Nodes (25): math, describe_transform(), Human-readable list of a Compose's ops (for startup logging)., finish_run(), log_metrics(), Return ``path`` if free, else ``path`` with ``_2``, ``_3`` ... appended. Guards…, unique_path(), _build_warmup_cosine_scheduler() (+17 more)

### Community 14 - "Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 15 - "web_tfjs_engine.dart"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 16 - "android_tflite_engine.dart"
Cohesion: 0.08
Nodes (24): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+16 more)

### Community 17 - "prepare_splits"
Cohesion: 0.11
Nodes (17): _collect_rows(), compute_hashes(), _dhash(), _log_event(), prepare_half_splits(), prepare_quarter_splits(), prepare_smoke_splits(), prepare_splits() (+9 more)

### Community 18 - "cutmix"
Cohesion: 0.12
Nodes (20): Byte caching, Data Pipeline, RandAugment, cutmix(), mixup(), _pairing_perm(), _rand_bbox(), Permutation that pairs each sample with another of the SAME species. A global… (+12 more)

### Community 19 - "data_collection.py"
Cohesion: 0.25
Nodes (17): _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file(), download_kaggle_dataset(), download_web_images(), _locate() (+9 more)

### Community 20 - "camera_screen.dart"
Cohesion: 0.14
Nodes (16): ChangeNotifier, ../../domain/entities/prediction_result.dart, PredictionResult, CattleBreedApp, build, _buildBody, CameraScreen, _pickAndPredict (+8 more)

### Community 21 - "prediction_result.dart"
Cohesion: 0.12
Nodes (16): breed, breedConfidence, breedIndex, BreedScore, confidence, index, label, latencyMs (+8 more)

### Community 22 - "20260924-110247-395b/manifest.json"
Cohesion: 0.12
Nodes (16): argv, cwd, duration_s, ended_at, env, RUN_EXEC_ID, exec_id, executable (+8 more)

### Community 23 - "ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition"
Cohesion: 0.25
Nodes (15): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+7 more)

### Community 24 - "inference_controller.dart"
Cohesion: 0.15
Nodes (12): bool get, _busy, dispose, _error, initialize, _initialized, predictImage, _result (+4 more)

### Community 25 - "image_preprocessor.dart"
Cohesion: 0.15
Nodes (11): dart:typed_data, ../entities/prediction_result.dart, ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, dispose, IModelService (+3 more)

### Community 26 - "view_logs.py"
Cohesion: 0.38
Nodes (12): all_execs(), cmd_diff(), cmd_events(), cmd_list(), cmd_metrics(), cmd_summary(), _final(), load_json() (+4 more)

### Community 27 - "20260924-210203-b312/manifest.json"
Cohesion: 0.15
Nodes (12): argv, cwd, env, exec_id, executable, git_commit, hostname, module (+4 more)

### Community 28 - "main.dart"
Cohesion: 0.17
Nodes (11): data/ml_engines/android_tflite_engine.dart, data/ml_engines/web_tfjs_engine.dart, ../../domain/services/i_model_service.dart, build, controller, initialize, main, service (+3 more)

### Community 29 - "ModelManager"
Cohesion: 0.22
Nodes (7): Model Tester GUI, Presenter configuration, ModelManager, Discovers, loads, and caches models for prediction., Load breed label maps, prioritizing portable exports to avoid mismatch., Find all available checkpoints., Return list of available model names with metadata.

### Community 30 - "test_model.py"
Cohesion: 0.20
Nodes (9): base64, datetime, pil_exiftags, struct, generate_odt_report(), 🐮 Breed Classifier — Model Tester GUI ========================================…, Generate an ODT report from prediction results. Args: results: single result…, threading (+1 more)

### Community 31 - "CattleBuffaloDataset"
Cohesion: 0.22
Nodes (4): Dataset, CattleBuffaloDataset, Merge per-breed augmentation overrides on top of the base flags., resolve_breed_augment()

### Community 32 - "audit_data.py"
Cohesion: 0.29
Nodes (9): difflib, hashlib, ahash(), hamming(), list_breeds(), main(), md5(), Audit data/raw before a training run. REPORT ONLY — never deletes anything.… (+1 more)

### Community 33 - "SessionLogger"
Cohesion: 0.22
Nodes (3): Read the current log file contents., File + stdout logger for the entire GUI session., SessionLogger

### Community 34 - "build_dataset_inventory"
Cohesion: 0.22
Nodes (8): build_dataset_inventory(), find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace., Write a JSON inventory of the dataset to…, Auto-detect breed folders inside source_base and copy/merge them into…

### Community 35 - "_DS"
Cohesion: 0.22
Nodes (4): _DS, _FakeModel, run_once(), synth_labels()

### Community 36 - "run_server"
Cohesion: 0.31
Nodes (8): extract_image_metadata(), Extract detailed metadata from raw image bytes., Run a minimal HTTP server with the GUI and prediction API., run_server(), do_GET(), do_POST(), _parse_multipart(), _respond()

### Community 37 - "._load_model"
Cohesion: 0.22
Nodes (7): get_model_spec(), _human_number(), no_grad, Extract detailed model architecture and parameter info., Format large numbers: 6234567 → '6.23M'., Load a model checkpoint into memory., Run prediction on raw image bytes. Returns result dict.

### Community 38 - "build_html"
Cohesion: 0.25
Nodes (8): build_html(), _build_js(), load_presenter_config(), Build the JavaScript block — plain string, no f-string escaping issues., Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON., Return the complete single-page GUI HTML. Mode: 'dev' or 'present'., save_presenter_config()

### Community 39 - "_groups_by_hash"
Cohesion: 0.29
Nodes (5): _groups_by_hash(), _hamming(), Union-find groups of (index, hash) within `hamming` distance., Like ``_stratified_split`` but near-duplicate groups stay in one split.…, _stratified_split_dedup()

### Community 40 - "Colab README"
Cohesion: 0.70
Nodes (5): Automatic Mixed Precision (AMP), Colab README, Colab Training, Kaggle algsoch/breed-cattle-buffalo dataset, Colab T4 GPU

### Community 41 - "evaluate_epoch"
Cohesion: 0.40
Nodes (4): evaluate_epoch(), _macro_scores(), no_grad, Macro-F1 and macro-recall from a confusion matrix (true x pred). Classes with…

### Community 43 - "Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like)"
Cohesion: 0.40
Nodes (5): Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like), Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 44 - "Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle"
Cohesion: 0.60
Nodes (5): Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle, Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle, Gaolao_02 — Gaolao cattle: white body with darker grey head/neck and grey hump, short horizontal horns, rope tether; zebu cattle, Hariana_01 — Hariana cattle: white/light grey coat with darker grey head and muzzle, prominent hump, short horns; zebu cattle, Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle

### Community 45 - "ci workflow"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 46 - "TfliteEngine"
Cohesion: 0.67
Nodes (3): TfliteEngine, TfJsEngine, IModelService

## Knowledge Gaps
- **128 isolated node(s):** `_binaryLabels`, `_buffaloLabels`, `_cattleLabels`, `dispose`, `_execute` (+123 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 388 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BreedClassifier` connect `BreedClassifier` to `export.py`, `Documentation Index`, `cbam.py`, `._load_model`, `cattle_buffalo_trainer.py`, `test_master_cpu.py`, `Android Deployment`, `train.py`, `ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition`, `ModelManager`, `test_model.py`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Why does `init_run_logger()` connect `export.py` to `cbam.py`, `local_train.py`, `run_logger.py`, `cattle_buffalo_trainer.py`, `train.py`, `test_model.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `Common Operations` connect `setup_venv.py` to `export.py`, `Documentation Index`, `cbam.py`, `local_train.py`, `cattle_buffalo_trainer.py`, `data_pipeline.py`, `Android Deployment`, `train.py`, `test_model.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `BreedClassifier` (e.g. with `Species Routing Inference` and `CBAM_AFTER_STAGE`) actually correct?**
  _`BreedClassifier` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `CONTEXT.md — Context Database` (e.g. with `README.md` and `mkdocs.yml — Material docs site config`) actually correct?**
  _`CONTEXT.md — Context Database` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_binaryLabels`, `_buffaloLabels`, `_cattleLabels` to the rest of the system?**
  _128 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `export.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05009009009009009 - nodes in this community are weakly interconnected._