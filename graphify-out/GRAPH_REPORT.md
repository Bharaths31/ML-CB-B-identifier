# Graph Report - ML-CB-B-identifier  (2026-09-21)

## Corpus Check
- 88 files · ~253,544 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 23 file(s) not represented in the graph (top: (none) 8, .pt 6, .ipynb 3)

## Summary
- 632 nodes · 1115 edges · 28 communities (24 shown, 4 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.78)
- Token cost: 68,244 input · 53,695 output

## Community Hubs (Navigation)
- Data Pipeline & Training
- Model & Attention Core
- Export & Mobile Deployment
- Model Tester GUI
- Training Internals & Docs
- Flutter App UI & Controllers
- Project Context Overview
- Local Training Automation
- Colab Evaluation & Metrics
- Model Tester Screenshots
- Config & API Reference
- Environment & Packaging Scripts
- Flutter Web TF.js Engine
- Flutter Android TFLite Engine
- Data Collection & Splits
- Prediction Result Model
- Architecture Decision Record
- Flutter Model Service Interface
- Flutter Image Preprocessor
- Notebook Conversion
- Cattle Breed Test Photos (1)
- Cattle Breed Test Photos (2)
- Docs CI/CD Workflows
- App Asset Preparation
- Colab Archive Tooling
- Shell Setup Script

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 38 edges
2. `CONTEXT.md — Context Database` - 25 edges
3. `Colab README` - 16 edges
4. `get_dataloaders()` - 15 edges
5. `ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition` - 14 edges
6. `Training Pipeline` - 13 edges
7. `masked_loss()` - 12 edges
8. `main()` - 12 edges
9. `_banner()` - 11 edges
10. `ModelManager` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Binary-head species balancing` --semantically_similar_to--> `masked_loss()`  [INFERRED] [semantically similar]
  docs/changelog.md → src/train.py
- `masked_loss()` --semantically_similar_to--> `Masked per-head loss`  [INFERRED] [semantically similar]
  src/train.py → ADR.md
- `BreedClassifier` --shares_data_with--> `FEATURE_DIM`  [INFERRED]
  src/model.py → docs/config-reference.md
- `Portable export bundle` --references--> `BreedClassifier`  [EXTRACTED]
  docs/export-deployment.md → src/model.py
- `BreedClassifier` --implements--> `Species Routing Inference`  [INFERRED]
  src/model.py → docs/architecture.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Multi-task inference flow: features -> binary/cattle/buffalo heads** — concept_breedclassifier, concept_three_head_classifier, concept_masked_loss, concept_mobile_input_convention [EXTRACTED 1.00]
- **Mobile INT8 deployment pipeline: export -> parity gate -> Flutter app** — concept_tflite_export, concept_onnx_int8, concept_parity_gate, concept_flutter_app, concept_android_deployment [INFERRED 0.85]
- **QAT Failure and PTQ Pivot** — test_results_training_result_phase3_degradation, test_results_training_result_qat_export_mismatch, test_results_training_result_int8_conversion_failure, docs_api_reference_qat [EXTRACTED 1.00]
- **Mobile INT8 export and deployment pipeline** — src_train, src_export, src_parity_check, tflite_int8, onnx_runtime_mobile_int8, flutter_app [EXTRACTED 1.00]
- **Export & Parity Gate Pipeline** — docs_api_reference_local_train, docs_api_reference_src_export, docs_api_reference_parity_check, docs_api_reference_qat [INFERRED 0.85]
- **Architecture Training & Inference Flow** — src_model_breedclassifier, docs_architecture_cbam, docs_architecture_masked_loss, docs_architecture_species_routing [EXTRACTED 1.00]
- **Accuracy/efficiency overhaul (distillation + EMA + long-tail)** — docs_changelog_knowledge_distillation, docs_changelog_ema_bug, docs_changelog_effective_number_sampler, docs_changelog_binary_species_balancing [EXTRACTED 1.00]
- **Colab T4 training setup flow** — colab_cattle_buffalo_trainer, colab_readme, kaggle_dataset, t4_gpu, scripts_create_colab_project_zip [INFERRED 0.75]
- **Local training pipeline (data -> train -> eval -> export)** — local_train, src_data_pipeline, src_train, src_evaluate, src_export [EXTRACTED 1.00]
- **Mobile INT8 export and parity verification** — docs_export_deployment_tflite, docs_export_deployment_onnx_int8, docs_export_deployment_parity_gate, docs_export_deployment_imagenet_normalization [EXTRACTED 1.00]
- **Shared-backbone hierarchical training pipeline** — adr_hierarchical_heads, src_train_masked_loss, docs_training_pipeline_qat, docs_training_pipeline_knowledge_distillation [INFERRED 0.75]
- **Label asset trio consumed by the Flutter inference engine** — flutter_app_assets_models_labels_binary, flutter_app_assets_models_labels_cattle, flutter_app_assets_models_labels_buffalo, concept_flutter_app [EXTRACTED 1.00]
- **Breed Classifier Model Tester GUI Feature Set** — test_results_screenshot_20260914_213516_image_upload_panel, test_results_screenshot_20260914_213516_prediction_results_panel, test_results_screenshot_20260914_213516_select_model_dropdown, test_results_screenshot_20260914_213516_analyze_breed_button, test_results_screenshot_20260914_213516_lite2_fp32_onnx_model [EXTRACTED 0.95]
- **Cattle Breed Prediction Outputs Across Test Screenshots** — test_results_screenshot_20260913_135102_umblachery_prediction, test_results_screenshot_20260914_213516_amritmahal_prediction, test_results_screenshot_20260914_215842_amritmahal_prediction, test_results_screenshot_20260914_215909_kenkatha_prediction [INFERRED 0.85]
- **Cattle/Buffalo breed classifier test image set (Testing data, chunk 6 of 6)** — testing_data_amruthamahal_01_amruthamahal, testing_data_baragur_02_baragur, testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hallikaru_02_hallikaru, testing_data_hariana_01_hariana, testing_data_malenadu_gidda_01_malenadu_gidda, testing_data_nagori_01_nagori, testing_data_raghav_gir_bull_at_hyderabad_raghav_gir_bull [EXTRACTED 1.00]
- **White/light-coated humped zebu breeds (visually similar coat colour)** — testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hariana_01_hariana, testing_data_nagori_01_nagori [INFERRED 0.75]

## Communities (28 total, 4 thin omitted)

### Community 0 - "Data Pipeline & Training"
Cohesion: 0.05
Nodes (54): Byte caching, download_kaggle_dataset(), ExportWrapper, find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Download and unzip a Kaggle dataset into dest_dir., Find all leaf directories containing images (breed folders). (+46 more)

### Community 1 - "Model & Attention Core"
Cohesion: 0.06
Nodes (35): Automatic Mixed Precision (AMP), CBAM attention, Colab README, Create a zip file of test-split images for Colab batch evaluation. Reads…, CBAM Attention, Species Routing Inference, Colab Testing & Evaluation, Colab Training (+27 more)

### Community 2 - "Export & Mobile Deployment"
Cohesion: 0.07
Nodes (42): Module Reference — Source Code Cross-Reference, Module dependency graph, Android Deployment, Flutter app, ONNX Runtime Mobile INT8 (QDQ), Export parity gate, Quantization-Aware Training (QAT), _calibration_images() (+34 more)

### Community 3 - "Model Tester GUI"
Cohesion: 0.06
Nodes (37): Model Tester GUI, Presenter configuration, build_html(), _build_js(), extract_image_metadata(), generate_odt_report(), get_model_spec(), _human_number() (+29 more)

### Community 4 - "Training Internals & Docs"
Cohesion: 0.07
Nodes (46): Training Internals & Data Flow — Deep Reference, Evaluation metrics (combined_top1/3/5), Image to tensor pipeline, Label encoding (binary/cattle/buffalo + masks), Loss computation, QAT internals (fuse conv-bn, per-tensor observers), Smoke test data flow, Training loop detail (+38 more)

### Community 5 - "Flutter App UI & Controllers"
Cohesion: 0.06
Nodes (39): bool get, ChangeNotifier, data/ml_engines/android_tflite_engine.dart, data/ml_engines/web_tfjs_engine.dart, ../../domain/entities/prediction_result.dart, ../../domain/services/i_model_service.dart, PredictionResult, build (+31 more)

### Community 6 - "Project Context Overview"
Cohesion: 0.17
Nodes (31): CONTEXT.md — Context Database, Android / mobile INT8 deployment path, BreedClassifier (backbone + attention + 3 heads), 18 Indian buffalo breeds, 57 Indian cattle breeds, CBAM attention module, CutMix / MixUp batch mixing on GPU, Effective-number-of-samples WeightedRandomSampler (beta=0.999) (+23 more)

### Community 7 - "Local Training Automation"
Cohesion: 0.16
Nodes (30): _banner(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb(), _install_torch(), main(), merge_into_species_dir() (+22 more)

### Community 8 - "Colab Evaluation & Metrics"
Cohesion: 0.08
Nodes (22): compute_macro_f1(), find_breed_folders(), img_to_base64(), per_breed_report(), plot_confusion_matrix(), predict_single(), preprocess_image(), Walk to find leaf dirs with images (breed folders). (+14 more)

### Community 9 - "Model Tester Screenshots"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 10 - "Config & API Reference"
Cohesion: 0.12
Nodes (24): Project Rules (AGENTS.md), Config Centralization Convention, CUDA/CPU Graceful Support, Real Mini-Dataset Smoke Test Rule, API Reference, Knowledge Distillation, local_train.py Pipeline, src.parity_check Export Parity Gate (+16 more)

### Community 11 - "Environment & Packaging Scripts"
Cohesion: 0.16
Nodes (20): create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), Common Operations, Path, Create colab_project.zip containing only the files needed for Colab training.…, check_python() (+12 more)

### Community 12 - "Flutter Web TF.js Engine"
Cohesion: 0.10
Nodes (20): dart:js_interop, external JSObject get, _argmax, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute (+12 more)

### Community 13 - "Flutter Android TFLite Engine"
Cohesion: 0.10
Nodes (19): _argmax, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, initialize, _inputSize (+11 more)

### Community 14 - "Data Collection & Splits"
Cohesion: 0.28
Nodes (15): _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file(), download_kaggle_dataset(), download_web_images(), _locate() (+7 more)

### Community 15 - "Prediction Result Model"
Cohesion: 0.12
Nodes (16): breed, breedConfidence, breedIndex, BreedScore, confidence, index, label, latencyMs (+8 more)

### Community 16 - "Architecture Decision Record"
Cohesion: 0.25
Nodes (15): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+7 more)

### Community 17 - "Flutter Model Service Interface"
Cohesion: 0.22
Nodes (8): dart:typed_data, ../entities/prediction_result.dart, TfliteEngine, TfJsEngine, dispose, IModelService, initialize, predict

### Community 18 - "Flutter Image Preprocessor"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 19 - "Notebook Conversion"
Cohesion: 0.40
Nodes (3): py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…

### Community 20 - "Cattle Breed Test Photos (1)"
Cohesion: 0.40
Nodes (5): Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like), Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 21 - "Cattle Breed Test Photos (2)"
Cohesion: 0.60
Nodes (5): Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle, Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle, Gaolao_02 — Gaolao cattle: white body with darker grey head/neck and grey hump, short horizontal horns, rope tether; zebu cattle, Hariana_01 — Hariana cattle: white/light grey coat with darker grey head and muzzle, prominent hump, short horns; zebu cattle, Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle

### Community 22 - "Docs CI/CD Workflows"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 23 - "App Asset Preparation"
Cohesion: 0.83
Nodes (3): load_index(), main(), write_labels()

## Knowledge Gaps
- **92 isolated node(s):** `_inputSize`, `_preprocessor`, `_interpreter`, `_binaryLabels`, `_cattleLabels` (+87 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 245 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BreedClassifier` connect `Model & Attention Core` to `Data Pipeline & Training`, `Export & Mobile Deployment`, `Model Tester GUI`, `Training Internals & Docs`, `Config & API Reference`, `Architecture Decision Record`?**
  _High betweenness centrality (0.143) - this node is a cross-community bridge._
- **Why does `Common Operations` connect `Environment & Packaging Scripts` to `Data Pipeline & Training`, `Model & Attention Core`, `Export & Mobile Deployment`, `Model Tester GUI`, `Local Training Automation`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `Training Pipeline` connect `Training Internals & Docs` to `Data Pipeline & Training`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `BreedClassifier` (e.g. with `Species Routing Inference` and `CBAM_AFTER_STAGE`) actually correct?**
  _`BreedClassifier` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `CONTEXT.md — Context Database` (e.g. with `README.md` and `mkdocs.yml — Material docs site config`) actually correct?**
  _`CONTEXT.md — Context Database` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_inputSize`, `_preprocessor`, `_interpreter` to the rest of the system?**
  _92 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Data Pipeline & Training` be split into smaller, more focused modules?**
  _Cohesion score 0.05093167701863354 - nodes in this community are weakly interconnected._