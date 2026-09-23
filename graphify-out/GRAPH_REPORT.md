# Graph Report - ML-CB-B-identifier  (2026-09-23)

## Corpus Check
- 69 files · ~266,805 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 23 file(s) not represented in the graph (top: (none) 8, .pt 6, .ipynb 3)

## Summary
- 772 nodes · 1558 edges · 39 communities (36 shown, 3 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `768f0eb2`
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
- API Reference
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
- onnx_parity_10.py
- run_epoch
- setup.sh
- evaluate.py
- test_model.py
- inference_controller.dart
- main.dart
- diagnose_model.py
- test_fixes_cpu.py
- run_server
- SessionLogger
- ._load_model
- CattleBuffaloDataset
- merge_into_species_dir
- ExportWrapper

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 44 edges
2. `CONTEXT.md — Context Database` - 25 edges
3. `main()` - 18 edges
4. `get_dataloaders()` - 17 edges
5. `Colab README` - 16 edges
6. `find_latest_checkpoint()` - 15 edges
7. `Documentation Index` - 15 edges
8. `ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition` - 14 edges
9. `Training Pipeline` - 14 edges
10. `main()` - 13 edges

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

## Communities (39 total, 3 thin omitted)

### Community 0 - "data_pipeline.py"
Cohesion: 0.11
Nodes (28): _collect_rows(), compute_class_counts(), compute_class_priors(), compute_rare_classes(), _count_per_class(), _effective_num_weights(), get_dataloaders(), _load_class_maps() (+20 more)

### Community 1 - "BreedClassifier"
Cohesion: 0.05
Nodes (40): Project Rules (AGENTS.md), Config Centralization Convention, Automatic Mixed Precision (AMP), CBAM attention, Colab README, Architecture & Data Flow, CBAM Attention, Masked Multi-Task Loss (+32 more)

### Community 2 - "export.py"
Cohesion: 0.06
Nodes (47): Module Reference — Source Code Cross-Reference, Module dependency graph, Knowledge distillation (teacher-student), Android Deployment, Known Constraints & Gotchas, EMA (Exponential Moving Average), Flutter app, ONNX Runtime Mobile INT8 (QDQ) (+39 more)

### Community 3 - "ModelManager"
Cohesion: 0.20
Nodes (8): Model Tester GUI, Presenter configuration, main(), ModelManager, Discovers, loads, and caches models for prediction., Load breed label maps from data/splits/., Find all available checkpoints., Return list of available model names with metadata.

### Community 4 - "Training Pipeline"
Cohesion: 0.05
Nodes (57): Training Internals & Data Flow — Deep Reference, Evaluation metrics (combined_top1/3/5), Image to tensor pipeline, Label encoding (binary/cattle/buffalo + masks), Loss computation, QAT internals (fuse conv-bn, per-tensor observers), Smoke test data flow, Training loop detail (+49 more)

### Community 5 - "camera_screen.dart"
Cohesion: 0.14
Nodes (16): ChangeNotifier, ../../domain/entities/prediction_result.dart, PredictionResult, CattleBreedApp, build, _buildBody, CameraScreen, _pickAndPredict (+8 more)

### Community 6 - "CONTEXT.md — Context Database"
Cohesion: 0.17
Nodes (31): CONTEXT.md — Context Database, Android / mobile INT8 deployment path, BreedClassifier (backbone + attention + 3 heads), 18 Indian buffalo breeds, 57 Indian cattle breeds, CBAM attention module, CutMix / MixUp batch mixing on GPU, Effective-number-of-samples WeightedRandomSampler (beta=0.999) (+23 more)

### Community 7 - "local_train.py"
Cohesion: 0.09
Nodes (37): Create a zip file of test-split images for Colab batch evaluation. Reads…, csv, Colab Testing & Evaluation, _banner(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb() (+29 more)

### Community 8 - "cattle_buffalo_tester.py"
Cohesion: 0.06
Nodes (29): compute_macro_f1(), find_breed_folders(), img_to_base64(), per_breed_report(), plot_confusion_matrix(), predict_single(), preprocess_image(), Walk to find leaf dirs with images (breed folders). (+21 more)

### Community 9 - "Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35)"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 10 - "API Reference"
Cohesion: 0.23
Nodes (15): CUDA/CPU Graceful Support, Real Mini-Dataset Smoke Test Rule, API Reference, Knowledge Distillation, local_train.py Pipeline, src.parity_check Export Parity Gate, Phase 3 QAT, src.export Model Export (+7 more)

### Community 11 - "setup_venv.py"
Cohesion: 0.21
Nodes (18): platform, load_index(), main(), write_labels(), check_python(), clean_broken_torch(), create_venv(), detect_nvidia_gpu() (+10 more)

### Community 12 - "web_tfjs_engine.dart"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 13 - "android_tflite_engine.dart"
Cohesion: 0.08
Nodes (23): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+15 more)

### Community 14 - "data_collection.py"
Cohesion: 0.15
Nodes (22): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file() (+14 more)

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
Cohesion: 0.13
Nodes (23): download_kaggle_dataset(), Download and unzip a Kaggle dataset into dest_dir., math, random, _build_warmup_cosine_scheduler(), _fmt_metrics(), _mix_off_epoch(), Last epoch that still mixes; mixing is OFF for epochs after this.… (+15 more)

### Community 20 - "Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like)"
Cohesion: 0.40
Nodes (5): Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like), Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 21 - "Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle"
Cohesion: 0.60
Nodes (5): Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle, Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle, Gaolao_02 — Gaolao cattle: white body with darker grey head/neck and grey hump, short horizontal horns, rope tether; zebu cattle, Hariana_01 — Hariana cattle: white/light grey coat with darker grey head and muzzle, prominent hump, short horns; zebu cattle, Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle

### Community 22 - "ci workflow"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 23 - "onnx_parity_10.py"
Cohesion: 0.13
Nodes (20): argparse, collections, difflib, hashlib, json, numpy, pil, ahash() (+12 more)

### Community 24 - "run_epoch"
Cohesion: 0.12
Nodes (20): Byte caching, Data Pipeline, RandAugment, cutmix(), mixup(), _pairing_perm(), _rand_bbox(), Permutation that pairs each sample with another of the SAME species. A global… (+12 more)

### Community 26 - "evaluate.py"
Cohesion: 0.14
Nodes (16): matplotlib, matplotlib_pyplot, full_evaluation(), main(), no_grad, _save_confusion(), Make a checkpoint loadable into a plain float BreedClassifier. Phase-3 QAT…, _sanitize_state_dict() (+8 more)

### Community 28 - "test_model.py"
Cohesion: 0.13
Nodes (14): base64, datetime, logging, pil_exiftags, struct, _build_js(), extract_image_metadata(), generate_odt_report() (+6 more)

### Community 29 - "inference_controller.dart"
Cohesion: 0.15
Nodes (12): bool get, _busy, dispose, _error, initialize, _initialized, predictImage, _result (+4 more)

### Community 30 - "main.dart"
Cohesion: 0.17
Nodes (11): data/ml_engines/android_tflite_engine.dart, data/ml_engines/web_tfjs_engine.dart, ../../domain/services/i_model_service.dart, build, controller, initialize, main, service (+3 more)

### Community 31 - "diagnose_model.py"
Cohesion: 0.13
Nodes (17): glob, os, detailed_eval(), load_model(), main(), print_report(), no_grad, Diagnose a trained checkpoint on a split (val or test). Reports, for one or two… (+9 more)

### Community 32 - "test_fixes_cpu.py"
Cohesion: 0.17
Nodes (9): copy, pandas, CPU-only synthetic tests for the 2026-09-23 fixes. No dataset/GPU needed. Run:…, src, Return ``path`` if free, else ``path`` with ``_2``, ``_3`` ... appended. Guards…, unique_path(), create_portable_export(), Bundle the trained model + labels + metadata into a portable folder. (+1 more)

### Community 33 - "run_server"
Cohesion: 0.23
Nodes (12): build_html(), load_presenter_config(), Run a minimal HTTP server with the GUI and prediction API., Load presenter config from JSON, creating defaults if missing., Save presenter config to JSON., Return the complete single-page GUI HTML. Mode: 'dev' or 'present'., run_server(), do_GET() (+4 more)

### Community 34 - "SessionLogger"
Cohesion: 0.22
Nodes (3): Read the current log file contents., File + stdout logger for the entire GUI session., SessionLogger

### Community 35 - "._load_model"
Cohesion: 0.22
Nodes (7): get_model_spec(), _human_number(), no_grad, Extract detailed model architecture and parameter info., Format large numbers: 6234567 → '6.23M'., Load a model checkpoint into memory., Run prediction on raw image bytes. Returns result dict.

### Community 36 - "CattleBuffaloDataset"
Cohesion: 0.20
Nodes (5): Dataset, CattleBuffaloDataset, _eval_transform(), Build the training transform. Augmentation is OFF by default (see ``AUG_*`` /…, _train_transform()

### Community 37 - "merge_into_species_dir"
Cohesion: 0.33
Nodes (6): find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace., Auto-detect breed folders inside source_base and copy/merge them into…

## Knowledge Gaps
- **100 isolated node(s):** `_ScoredBreed`, `_inputSize`, `_preprocessor`, `_interpreter`, `_binaryLabels` (+95 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 306 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BreedClassifier` connect `BreedClassifier` to `test_fixes_cpu.py`, `data_pipeline.py`, `export.py`, `ModelManager`, `Training Pipeline`, `._load_model`, `API Reference`, `ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition`, `train.py`, `onnx_parity_10.py`, `evaluate.py`, `test_model.py`, `diagnose_model.py`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `Common Operations` connect `Training Pipeline` to `data_pipeline.py`, `BreedClassifier`, `export.py`, `local_train.py`, `API Reference`, `setup_venv.py`, `train.py`, `evaluate.py`, `test_model.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `Training Pipeline` connect `Training Pipeline` to `BreedClassifier`, `train.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `BreedClassifier` (e.g. with `Species Routing Inference` and `CBAM_AFTER_STAGE`) actually correct?**
  _`BreedClassifier` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `CONTEXT.md — Context Database` (e.g. with `README.md` and `mkdocs.yml — Material docs site config`) actually correct?**
  _`CONTEXT.md — Context Database` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_ScoredBreed`, `_inputSize`, `_preprocessor` to the rest of the system?**
  _100 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `data_pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11174242424242424 - nodes in this community are weakly interconnected._