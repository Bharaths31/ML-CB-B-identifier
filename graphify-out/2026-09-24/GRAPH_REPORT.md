# Graph Report - ML-CB-B-identifier  (2026-09-24)

## Corpus Check
- 70 files · ~271,525 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 23 file(s) not represented in the graph (top: (none) 8, .pt 6, .ipynb 3)

## Summary
- 812 nodes · 1655 edges · 44 communities (40 shown, 4 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `02b3bb3e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- data_pipeline.py
- BreedClassifier
- export.py
- ModelManager
- Training Pipeline
- camera_screen.dart
- CONTEXT.md — Context Database
- local_train.py
- cattle_buffalo_tester.py
- Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)
- os
- setup_venv.py
- web_tfjs_engine.dart
- android_tflite_engine.dart
- data_collection.py
- prediction_result.dart
- ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition
- i_model_service.dart
- image_preprocessor.dart
- train.py
- Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like)
- Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle
- ci workflow
- audit_data.py
- _apply_ema
- setup.sh
- evaluate.py
- test_model.py
- inference_controller.dart
- main.dart
- diagnose_model.py
- test_master_cpu.py
- run_server
- SessionLogger
- model.py
- parity_check.py
- build_dataset_inventory
- ExportWrapper
- cbam.py
- .__init__
- Colab README
- predict_single
- _RawOutputs

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 44 edges
2. `CONTEXT.md — Context Database` - 25 edges
3. `main()` - 21 edges
4. `get_dataloaders()` - 17 edges
5. `Colab README` - 16 edges
6. `Documentation Index` - 15 edges
7. `ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition` - 14 edges
8. `Training Pipeline` - 14 edges
9. `main()` - 13 edges
10. `make_run_id()` - 13 edges

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

## Communities (44 total, 4 thin omitted)

### Community 0 - "data_pipeline.py"
Cohesion: 0.05
Nodes (56): Byte caching, Dataset, Data Pipeline, RandAugment, CattleBuffaloDataset, _collect_rows(), _color_jitter(), compute_class_counts() (+48 more)

### Community 1 - "BreedClassifier"
Cohesion: 0.06
Nodes (43): Project Rules (AGENTS.md), Config Centralization Convention, CUDA/CPU Graceful Support, Real Mini-Dataset Smoke Test Rule, Knowledge distillation (teacher-student), Android Deployment, API Reference, Knowledge Distillation (+35 more)

### Community 2 - "export.py"
Cohesion: 0.13
Nodes (22): Module Reference — Source Code Cross-Reference, Module dependency graph, Common Operations, _calibration_images(), _copy_to_app_assets(), create_portable_export(), export_onnx_int8(), __init__() (+14 more)

### Community 3 - "ModelManager"
Cohesion: 0.15
Nodes (10): Model Tester GUI, Presenter configuration, ModelManager, no_grad, Discovers, loads, and caches models for prediction., Load breed label maps from data/splits/., Find all available checkpoints., Return list of available model names with metadata. (+2 more)

### Community 4 - "Training Pipeline"
Cohesion: 0.06
Nodes (51): Training Internals & Data Flow — Deep Reference, Evaluation metrics (combined_top1/3/5), Image to tensor pipeline, Label encoding (binary/cattle/buffalo + masks), Loss computation, QAT internals (fuse conv-bn, per-tensor observers), Smoke test data flow, Training loop detail (+43 more)

### Community 5 - "camera_screen.dart"
Cohesion: 0.14
Nodes (16): ChangeNotifier, ../../domain/entities/prediction_result.dart, PredictionResult, CattleBreedApp, build, _buildBody, CameraScreen, _pickAndPredict (+8 more)

### Community 6 - "CONTEXT.md — Context Database"
Cohesion: 0.17
Nodes (31): CONTEXT.md — Context Database, Android / mobile INT8 deployment path, BreedClassifier (backbone + attention + 3 heads), 18 Indian buffalo breeds, 57 Indian cattle breeds, CBAM attention module, CutMix / MixUp batch mixing on GPU, Effective-number-of-samples WeightedRandomSampler (beta=0.999) (+23 more)

### Community 7 - "local_train.py"
Cohesion: 0.14
Nodes (32): _banner(), build_dataset_inventory(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb(), _install_torch(), main() (+24 more)

### Community 8 - "cattle_buffalo_tester.py"
Cohesion: 0.07
Nodes (22): compute_macro_f1(), find_breed_folders(), img_to_base64(), per_breed_report(), plot_confusion_matrix(), Walk to find leaf dirs with images (breed folders)., Compute macro-averaged F1 across all breeds., Plot a confusion matrix heatmap for a single species. (+14 more)

### Community 9 - "Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 10 - "os"
Cohesion: 0.13
Nodes (20): glob, numpy, os, expand_images(), main(), Parity check: PyTorch checkpoint vs fp32 ONNX on a small image list. For each…, soft_top5(), find_latest() (+12 more)

### Community 11 - "setup_venv.py"
Cohesion: 0.14
Nodes (23): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, platform, re, load_index(), main(), write_labels() (+15 more)

### Community 12 - "web_tfjs_engine.dart"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 13 - "android_tflite_engine.dart"
Cohesion: 0.08
Nodes (23): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+15 more)

### Community 14 - "data_collection.py"
Cohesion: 0.22
Nodes (18): _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file(), download_kaggle_dataset(), download_web_images(), _locate() (+10 more)

### Community 15 - "prediction_result.dart"
Cohesion: 0.12
Nodes (16): breed, breedConfidence, breedIndex, BreedScore, confidence, index, label, latencyMs (+8 more)

### Community 16 - "ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition"
Cohesion: 0.25
Nodes (15): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+7 more)

### Community 17 - "i_model_service.dart"
Cohesion: 0.22
Nodes (8): dart:typed_data, ../entities/prediction_result.dart, TfliteEngine, TfJsEngine, dispose, IModelService, initialize, predict

### Community 18 - "image_preprocessor.dart"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 19 - "train.py"
Cohesion: 0.11
Nodes (25): download_kaggle_dataset(), Download and unzip a Kaggle dataset into dest_dir., math, random, _macro_scores(), Macro-F1 and macro-recall from a confusion matrix (true x pred). Classes with…, _build_warmup_cosine_scheduler(), create_portable_export() (+17 more)

### Community 20 - "Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like)"
Cohesion: 0.40
Nodes (5): Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like), Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 21 - "Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle"
Cohesion: 0.60
Nodes (5): Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle, Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle, Gaolao_02 — Gaolao cattle: white body with darker grey head/neck and grey hump, short horizontal horns, rope tether; zebu cattle, Hariana_01 — Hariana cattle: white/light grey coat with darker grey head and muzzle, prominent hump, short horns; zebu cattle, Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle

### Community 22 - "ci workflow"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 23 - "audit_data.py"
Cohesion: 0.08
Nodes (23): collections, Create a zip file of test-split images for Colab batch evaluation. Reads…, create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), csv, difflib (+15 more)

### Community 24 - "_apply_ema"
Cohesion: 0.40
Nodes (5): _apply_ema(), _ema_decay_at(), EMA decay for optimizer step ``step`` (1-based), with warm-up. ``decay_t =…, One EMA update: parameters AND floating buffers (BN running stats)., _ema_update()

### Community 26 - "evaluate.py"
Cohesion: 0.19
Nodes (12): full_evaluation(), main(), no_grad, _save_confusion(), Make a checkpoint loadable into a plain float BreedClassifier. Phase-3 QAT…, _sanitize_state_dict(), evaluate_epoch(), no_grad (+4 more)

### Community 28 - "test_model.py"
Cohesion: 0.10
Nodes (18): base64, datetime, io, logging, pil, pil_exiftags, struct, _build_js() (+10 more)

### Community 29 - "inference_controller.dart"
Cohesion: 0.15
Nodes (12): bool get, _busy, dispose, _error, initialize, _initialized, predictImage, _result (+4 more)

### Community 30 - "main.dart"
Cohesion: 0.17
Nodes (11): data/ml_engines/android_tflite_engine.dart, data/ml_engines/web_tfjs_engine.dart, ../../domain/services/i_model_service.dart, build, controller, initialize, main, service (+3 more)

### Community 31 - "diagnose_model.py"
Cohesion: 0.23
Nodes (9): argparse, json, detailed_eval(), load_model(), main(), print_report(), no_grad, Diagnose a trained checkpoint on a split (val or test). Reports, for one or two… (+1 more)

### Community 32 - "test_master_cpu.py"
Cohesion: 0.10
Nodes (14): copy, importlib, pandas, CPU-only synthetic tests for the 2026-09-23 fixes. No dataset/GPU needed. Run:…, _DS, _FakeModel, _M, CPU-only synthetic tests for the master-task fixes. No GPU/dataset needed. Run:… (+6 more)

### Community 33 - "run_server"
Cohesion: 0.17
Nodes (15): build_html(), extract_image_metadata(), load_presenter_config(), main(), Extract detailed metadata from raw image bytes., Run a minimal HTTP server with the GUI and prediction API., Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON. (+7 more)

### Community 34 - "SessionLogger"
Cohesion: 0.22
Nodes (3): Read the current log file contents., File + stdout logger for the entire GUI session., SessionLogger

### Community 35 - "model.py"
Cohesion: 0.22
Nodes (9): EfficientNetLite, load_backbone_weights(), make_activation(), MBConvBlock, check_backbone(), check_full_model(), main(), torch (+1 more)

### Community 36 - "parity_check.py"
Cohesion: 0.21
Nodes (13): _load_model(), accumulate(), evaluate_on_split(), finalize(), main(), make_onnx_runner(), make_tflite_runner(), make_torch_runner() (+5 more)

### Community 37 - "build_dataset_inventory"
Cohesion: 0.22
Nodes (8): build_dataset_inventory(), find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace., Write a JSON inventory of one dataset to ``<out_dir>/<dataset_name>.json``. For…, Auto-detect breed folders inside source_base and copy/merge them into…

### Community 39 - "cbam.py"
Cohesion: 0.31
Nodes (6): CBAM attention, Model Architecture, SE attention, build_attention(), CBAM, SEBlock

### Community 40 - ".__init__"
Cohesion: 0.27
Nodes (4): ChannelAttention, Zero the last Conv2d weight in a Sequential so its output is 0., SpatialAttention, _zero_init_last()

### Community 41 - "Colab README"
Cohesion: 0.39
Nodes (8): Automatic Mixed Precision (AMP), Colab README, Colab Training, EfficientNet-Lite backbone, efficientnet_lite2.pth weights, efficientnet_lite4.pth weights, Kaggle algsoch/breed-cattle-buffalo dataset, Colab T4 GPU

### Community 42 - "predict_single"
Cohesion: 0.32
Nodes (8): predict_single(), preprocess_image(), Preprocess a PIL image for ONNX inference. Matches the evaluation transform…, Numerically stable softmax., Run full inference pipeline on a single PIL image. Returns: dict with species,…, softmax(), Image, ndarray

## Knowledge Gaps
- **100 isolated node(s):** `_ScoredBreed`, `_inputSize`, `_preprocessor`, `_interpreter`, `_binaryLabels` (+95 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 324 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BreedClassifier` connect `BreedClassifier` to `test_master_cpu.py`, `data_pipeline.py`, `export.py`, `model.py`, `Training Pipeline`, `parity_check.py`, `ModelManager`, `cbam.py`, `Colab README`, `os`, `ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition`, `train.py`, `evaluate.py`, `test_model.py`, `diagnose_model.py`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `Common Operations` connect `export.py` to `data_pipeline.py`, `BreedClassifier`, `model.py`, `Training Pipeline`, `parity_check.py`, `local_train.py`, `setup_venv.py`, `train.py`, `audit_data.py`, `evaluate.py`, `test_model.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `Training Pipeline` connect `Training Pipeline` to `BreedClassifier`, `train.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `BreedClassifier` (e.g. with `Species Routing Inference` and `CBAM_AFTER_STAGE`) actually correct?**
  _`BreedClassifier` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `CONTEXT.md — Context Database` (e.g. with `README.md` and `mkdocs.yml — Material docs site config`) actually correct?**
  _`CONTEXT.md — Context Database` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_ScoredBreed`, `_inputSize`, `_preprocessor` to the rest of the system?**
  _100 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `data_pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05029838022165388 - nodes in this community are weakly interconnected._