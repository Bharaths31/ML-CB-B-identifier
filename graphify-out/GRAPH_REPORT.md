# Graph Report - ML-CB-B-identifier  (2026-09-21)

## Corpus Check
- 88 files · ~253,544 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 23 file(s) not represented in the graph (top: (none) 8, .pt 6, .ipynb 3)

## Summary
- 688 nodes · 1304 edges · 45 communities (40 shown, 5 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 83 edges (avg confidence: 0.78)
- Token cost: 2,400 input · 1,900 output

## Community Hubs (Navigation)
- Model & Attention Architecture
- Export & Parity Pipeline
- Local Training Automation
- Training Internals & Docs
- Project Context Overview
- Config & API Reference
- Colab Trainer & Evaluation
- Model Tester GUI Screenshots
- Data Pipeline & Augmentation
- Flutter Web TF.js Engine
- Data Collection & Scraping
- Flutter Android TFLite Engine
- Data Splits & Training Setup
- Colab Tester Reporting
- Flutter Camera UI
- Prediction Result Model
- Architecture Decision Record
- Training Constraints & Techniques
- Environment Setup (venv)
- Model Manager Inference
- Model Tester GUI Core
- Flutter Inference Controller
- Notebook & Asset Scripts
- Flutter App Entry
- Model Tester HTTP Server
- Session Logger
- Flutter Model Service Interface
- Loss Functions & Distillation
- ONNX Singleton Inference
- Presenter Mode Config
- Colab Test Zip Tooling
- Flutter Image Preprocessor
- Augmentation Concepts
- Byte Cache Test
- Training Loop Epochs
- Cattle Breed Test Photos (1)
- Cattle Breed Test Photos (2)
- Docs CI/CD Workflows
- Macro F1 Metric
- Confusion Matrix Plot
- Export Wrapper Model
- Breed Folder Discovery
- Per-Breed Report
- Shell Setup Script

## God Nodes (most connected - your core abstractions)
1. `CONTEXT.md — Context Database` - 25 edges
2. `BreedClassifier` - 22 edges
3. `Colab README` - 16 edges
4. `get_dataloaders()` - 15 edges
5. `ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition` - 14 edges
6. `Training Pipeline` - 13 edges
7. `main()` - 12 edges
8. `_banner()` - 11 edges
9. `Common Operations` - 11 edges
10. `Android Deployment` - 11 edges

## Surprising Connections (you probably didn't know these)
- `QAT phase 3 (opt-in)` --semantically_similar_to--> `Quantization-Aware Training (QAT)`  [INFERRED] [semantically similar]
  docs/training-pipeline.md → ADR.md
- `masked_loss` --semantically_similar_to--> `Masked per-head loss`  [INFERRED] [semantically similar]
  docs/training-pipeline.md → ADR.md
- `QAT internals (fuse conv-bn, per-tensor observers)` --semantically_similar_to--> `QAT phase 3 (opt-in)`  [INFERRED] [semantically similar]
  .agents/knowledge/TRAINING_INTERNALS.md → docs/training-pipeline.md
- `ModelManager` --uses--> `BreedClassifier`  [INFERRED]
  test_model.py → src/model.py
- `mkdocs.yml — Material docs site config` --conceptually_related_to--> `CONTEXT.md — Context Database`  [INFERRED]
  mkdocs.yml → .agents/knowledge/CONTEXT.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Architecture Training & Inference Flow** — docs_architecture_breedclassifier, docs_architecture_cbam, docs_architecture_masked_loss, docs_architecture_species_routing [EXTRACTED 1.00]
- **Export & Parity Gate Pipeline** — docs_api_reference_local_train, docs_api_reference_src_export, docs_api_reference_parity_check, docs_api_reference_qat [INFERRED 0.85]
- **QAT Failure and PTQ Pivot** — test_results_training_result_phase3_degradation, test_results_training_result_qat_export_mismatch, test_results_training_result_int8_conversion_failure, docs_api_reference_qat [EXTRACTED 1.00]
- **Multi-task inference flow: features -> binary/cattle/buffalo heads** — concept_breedclassifier, concept_three_head_classifier, concept_masked_loss, concept_mobile_input_convention [EXTRACTED 1.00]
- **Mobile INT8 deployment pipeline: export -> parity gate -> Flutter app** — concept_tflite_export, concept_onnx_int8, concept_parity_gate, concept_flutter_app, concept_android_deployment [INFERRED 0.85]
- **Label asset trio consumed by the Flutter inference engine** — flutter_app_assets_models_labels_binary, flutter_app_assets_models_labels_cattle, flutter_app_assets_models_labels_buffalo, concept_flutter_app [EXTRACTED 1.00]
- **Shared-backbone hierarchical training pipeline** — adr_hierarchical_heads, docs_training_pipeline_masked_loss, docs_training_pipeline_qat, docs_training_pipeline_knowledge_distillation [INFERRED 0.75]
- **Mobile INT8 export and parity verification** — docs_export_deployment_tflite, docs_export_deployment_onnx_int8, docs_export_deployment_parity_gate, docs_export_deployment_imagenet_normalization [EXTRACTED 1.00]
- **Accuracy/efficiency overhaul (distillation + EMA + long-tail)** — docs_changelog_knowledge_distillation, docs_changelog_ema_bug, docs_changelog_effective_number_sampler, docs_changelog_binary_species_balancing [EXTRACTED 1.00]
- **Local training pipeline (data -> train -> eval -> export)** — local_train, src_data_pipeline, src_train, src_evaluate, src_export [EXTRACTED 1.00]
- **Mobile INT8 export and deployment pipeline** — src_train, src_export, src_parity_check, tflite_int8, onnx_runtime_mobile_int8, flutter_app [EXTRACTED 1.00]
- **Colab T4 training setup flow** — colab_cattle_buffalo_trainer, colab_readme, kaggle_dataset, t4_gpu, scripts_create_colab_project_zip [INFERRED 0.75]
- **Breed Classifier Model Tester GUI Feature Set** — test_results_screenshot_20260914_213516_image_upload_panel, test_results_screenshot_20260914_213516_prediction_results_panel, test_results_screenshot_20260914_213516_select_model_dropdown, test_results_screenshot_20260914_213516_analyze_breed_button, test_results_screenshot_20260914_213516_lite2_fp32_onnx_model [EXTRACTED 0.95]
- **Cattle Breed Prediction Outputs Across Test Screenshots** — test_results_screenshot_20260913_135102_umblachery_prediction, test_results_screenshot_20260914_213516_amritmahal_prediction, test_results_screenshot_20260914_215842_amritmahal_prediction, test_results_screenshot_20260914_215909_kenkatha_prediction [INFERRED 0.85]
- **Cattle/Buffalo breed classifier test image set (Testing data, chunk 6 of 6)** — testing_data_amruthamahal_01_amruthamahal, testing_data_baragur_02_baragur, testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hallikaru_02_hallikaru, testing_data_hariana_01_hariana, testing_data_malenadu_gidda_01_malenadu_gidda, testing_data_nagori_01_nagori, testing_data_raghav_gir_bull_at_hyderabad_raghav_gir_bull [EXTRACTED 1.00]
- **White/light-coated humped zebu breeds (visually similar coat colour)** — testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hariana_01_hariana, testing_data_nagori_01_nagori [INFERRED 0.75]
- **Karnataka draught-type grey/white zebu cattle (Amruthamahal/Hallikar-type)** — testing_data_amruthamahal_01_amruthamahal, testing_data_hallikaru_02_hallikaru [INFERRED 0.75]

## Communities (45 total, 5 thin omitted)

### Community 0 - "Model & Attention Architecture"
Cohesion: 0.07
Nodes (30): BreedClassifier, CBAM attention, Colab README, Colab Training, Documentation Index, Model Architecture, Model Tester GUI, EfficientNet-Lite backbone (+22 more)

### Community 1 - "Export & Parity Pipeline"
Cohesion: 0.07
Nodes (38): Module Reference — Source Code Cross-Reference, BreedClassifier, Module dependency graph, Portable export bundle, _calibration_images(), _copy_to_app_assets(), create_portable_export(), export_onnx_int8() (+30 more)

### Community 2 - "Local Training Automation"
Cohesion: 0.09
Nodes (42): argparse, create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), Common Operations, _banner(), _check_windows_build_tools() (+34 more)

### Community 3 - "Training Internals & Docs"
Cohesion: 0.08
Nodes (41): Training Internals & Data Flow — Deep Reference, Evaluation metrics (combined_top1/3/5), Image to tensor pipeline, Label encoding (binary/cattle/buffalo + masks), Loss computation, QAT internals (fuse conv-bn, per-tensor observers), Smoke test data flow, Training loop detail (+33 more)

### Community 4 - "Project Context Overview"
Cohesion: 0.17
Nodes (31): CONTEXT.md — Context Database, Android / mobile INT8 deployment path, BreedClassifier (backbone + attention + 3 heads), 18 Indian buffalo breeds, 57 Indian cattle breeds, CBAM attention module, CutMix / MixUp batch mixing on GPU, Effective-number-of-samples WeightedRandomSampler (beta=0.999) (+23 more)

### Community 5 - "Config & API Reference"
Cohesion: 0.11
Nodes (28): Project Rules (AGENTS.md), Config Centralization Convention, CUDA/CPU Graceful Support, Real Mini-Dataset Smoke Test Rule, API Reference, Knowledge Distillation, local_train.py Pipeline, src.parity_check Export Parity Gate (+20 more)

### Community 6 - "Colab Trainer & Evaluation"
Cohesion: 0.11
Nodes (21): download_kaggle_dataset(), find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Download and unzip a Kaggle dataset into dest_dir., Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace., Auto-detect breed folders inside source_base and copy/merge them into… (+13 more)

### Community 7 - "Model Tester GUI Screenshots"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 8 - "Data Pipeline & Augmentation"
Cohesion: 0.13
Nodes (16): Dataset, pandas, random, CattleBuffaloDataset, cutmix(), _eval_transform(), get_dataloaders(), _make_weighted_sampler() (+8 more)

### Community 9 - "Flutter Web TF.js Engine"
Cohesion: 0.10
Nodes (20): dart:js_interop, external JSObject get, _argmax, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute (+12 more)

### Community 10 - "Data Collection & Scraping"
Cohesion: 0.21
Nodes (19): _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file(), download_kaggle_dataset(), download_web_images(), _locate() (+11 more)

### Community 11 - "Flutter Android TFLite Engine"
Cohesion: 0.10
Nodes (19): _argmax, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, initialize, _inputSize (+11 more)

### Community 12 - "Data Splits & Training Setup"
Cohesion: 0.12
Nodes (16): _collect_rows(), prepare_half_splits(), prepare_quarter_splits(), prepare_smoke_splits(), prepare_splits(), Create a dataset using a fraction of images per breed for faster training. Uses…, Create a dataset using 25% of images per breed for very fast training.…, Create a tiny dataset for smoke tests by sampling a few images per breed. This… (+8 more)

### Community 13 - "Colab Tester Reporting"
Cohesion: 0.11
Nodes (11): base64, img_to_base64(), Display a gallery of misclassified images., Convert an image file to base64 data URI for embedding in HTML., show_misclassified(), getpass, google_colab, ipython_display (+3 more)

### Community 14 - "Flutter Camera UI"
Cohesion: 0.14
Nodes (16): ChangeNotifier, ../../domain/entities/prediction_result.dart, PredictionResult, CattleBreedApp, build, _buildBody, CameraScreen, _pickAndPredict (+8 more)

### Community 15 - "Prediction Result Model"
Cohesion: 0.12
Nodes (16): breed, breedConfidence, breedIndex, BreedScore, confidence, index, label, latencyMs (+8 more)

### Community 16 - "Architecture Decision Record"
Cohesion: 0.25
Nodes (15): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+7 more)

### Community 17 - "Training Constraints & Techniques"
Cohesion: 0.28
Nodes (14): Automatic Mixed Precision (AMP), Knowledge distillation (teacher-student), Android Deployment, Known Constraints & Gotchas, EMA (Exponential Moving Average), Flutter app, math, ONNX Runtime Mobile INT8 (QDQ) (+6 more)

### Community 18 - "Environment Setup (venv)"
Cohesion: 0.31
Nodes (14): platform, check_python(), clean_broken_torch(), create_venv(), detect_nvidia_gpu(), install_requirements(), log(), main() (+6 more)

### Community 19 - "Model Manager Inference"
Cohesion: 0.16
Nodes (9): main(), ModelManager, no_grad, Discovers, loads, and caches models for prediction., Load breed label maps from data/splits/., Find all available checkpoints., Return list of available model names with metadata., Load a model checkpoint into memory. (+1 more)

### Community 20 - "Model Tester GUI Core"
Cohesion: 0.15
Nodes (13): datetime, glob, logging, pil_exiftags, struct, get_model_spec(), _human_number(), 🐮 Breed Classifier — Model Tester GUI ========================================… (+5 more)

### Community 21 - "Flutter Inference Controller"
Cohesion: 0.15
Nodes (12): bool get, _busy, dispose, _error, initialize, _initialized, predictImage, _result (+4 more)

### Community 22 - "Notebook & Asset Scripts"
Cohesion: 0.19
Nodes (10): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, json, re, load_index(), main(), write_labels() (+2 more)

### Community 23 - "Flutter App Entry"
Cohesion: 0.17
Nodes (11): data/ml_engines/android_tflite_engine.dart, data/ml_engines/web_tfjs_engine.dart, ../../domain/services/i_model_service.dart, build, controller, initialize, main, service (+3 more)

### Community 24 - "Model Tester HTTP Server"
Cohesion: 0.24
Nodes (10): extract_image_metadata(), generate_odt_report(), Extract detailed metadata from raw image bytes., Run a minimal HTTP server with the GUI and prediction API., Generate an ODT report from prediction results. Args: results: single result…, run_server(), do_GET(), do_POST() (+2 more)

### Community 25 - "Session Logger"
Cohesion: 0.22
Nodes (3): Read the current log file contents., File + stdout logger for the entire GUI session., SessionLogger

### Community 26 - "Flutter Model Service Interface"
Cohesion: 0.22
Nodes (8): dart:typed_data, ../entities/prediction_result.dart, TfliteEngine, TfJsEngine, dispose, IModelService, initialize, predict

### Community 27 - "Loss Functions & Distillation"
Cohesion: 0.25
Nodes (8): _compute_loss(), masked_kd_loss(), masked_loss(), Multi-task loss blended with teacher distillation (Hinton et al.). total = (1 -…, Forward pass + loss, with optional knowledge distillation., Soft cross-entropy with optional label smoothing. When label_smoothing > 0,…, Multi-task loss with species-masked breed CE. The binary term is optionally re-…, soft_ce()

### Community 28 - "ONNX Singleton Inference"
Cohesion: 0.32
Nodes (8): predict_single(), preprocess_image(), Preprocess a PIL image for ONNX inference. Matches the evaluation transform…, Numerically stable softmax., Run full inference pipeline on a single PIL image. Returns: dict with species,…, softmax(), Image, ndarray

### Community 29 - "Presenter Mode Config"
Cohesion: 0.25
Nodes (8): build_html(), _build_js(), load_presenter_config(), Build the JavaScript block — plain string, no f-string escaping issues., Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON., Return the complete single-page GUI HTML. Mode: 'dev' or 'present'., save_presenter_config()

### Community 30 - "Colab Test Zip Tooling"
Cohesion: 0.29
Nodes (5): collections, Create a zip file of test-split images for Colab batch evaluation. Reads…, csv, Colab Testing & Evaluation, Kaggle algsoch/breed-cattle-buffalo dataset

### Community 31 - "Flutter Image Preprocessor"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 32 - "Augmentation Concepts"
Cohesion: 0.50
Nodes (5): Byte caching, CutMix, Data Pipeline, MixUp, RandAugment

### Community 33 - "Byte Cache Test"
Cohesion: 0.40
Nodes (3): io, os, pil

### Community 34 - "Training Loop Epochs"
Cohesion: 0.40
Nodes (5): mixup(), Run one training epoch with optional AMP, gradient accumulation, label…, Train a single phase with AdamW, optional AMP, gradient accumulation, and label…, run_epoch(), train_phase()

### Community 35 - "Cattle Breed Test Photos (1)"
Cohesion: 0.40
Nodes (5): Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like), Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 36 - "Cattle Breed Test Photos (2)"
Cohesion: 0.60
Nodes (5): Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle, Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle, Gaolao_02 — Gaolao cattle: white body with darker grey head/neck and grey hump, short horizontal horns, rope tether; zebu cattle, Hariana_01 — Hariana cattle: white/light grey coat with darker grey head and muzzle, prominent hump, short horns; zebu cattle, Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle

### Community 37 - "Docs CI/CD Workflows"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 38 - "Macro F1 Metric"
Cohesion: 0.67
Nodes (3): compute_macro_f1(), Compute macro-averaged F1 across all breeds., Compute macro-averaged F1 across all breeds.

### Community 39 - "Confusion Matrix Plot"
Cohesion: 0.67
Nodes (3): plot_confusion_matrix(), Plot a confusion matrix heatmap for a single species., Plot a confusion matrix heatmap for a single species.

## Knowledge Gaps
- **92 isolated node(s):** `_inputSize`, `_preprocessor`, `_interpreter`, `_binaryLabels`, `_cattleLabels` (+87 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 261 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Training Pipeline` connect `Training Internals & Docs` to `Training Constraints & Techniques`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `BreedClassifier` connect `Model & Attention Architecture` to `Export & Parity Pipeline`, `Colab Trainer & Evaluation`, `Data Splits & Training Setup`, `Training Constraints & Techniques`, `Model Manager Inference`, `Model Tester GUI Core`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Why does `BreedClassifier` connect `Export & Parity Pipeline` to `Architecture Decision Record`, `Training Constraints & Techniques`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `CONTEXT.md — Context Database` (e.g. with `README.md` and `mkdocs.yml — Material docs site config`) actually correct?**
  _`CONTEXT.md — Context Database` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `BreedClassifier` (e.g. with `EfficientNetLite` and `ModelManager`) actually correct?**
  _`BreedClassifier` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_inputSize`, `_preprocessor`, `_interpreter` to the rest of the system?**
  _92 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Model & Attention Architecture` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._