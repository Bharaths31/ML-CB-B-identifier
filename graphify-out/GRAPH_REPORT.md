# Graph Report - ML-CB-B-identifier  (2026-09-25)

## Corpus Check
- 117 files · ~361,610 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 34 file(s) not represented in the graph (top: (none) 8, .jsonl 8, .pt 6)

## Summary
- 877 nodes · 1741 edges · 75 communities (55 shown, 20 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 49 edges (avg confidence: 0.79)
- Token cost: 73,270 input · 4,263 output

## Community Hubs (Navigation)
- Train Local Module
- Src Model Module
- Src Run Module
- Src Train Module
- Scripts Test Module
- Results Test Module
- Flutter App Module
- Flutter Tflite Module
- Data Src Module
- Src Evaluate Module
- Src Efficientnet Module
- Model Test Module
- Colab Cattle Module
- Data Collection Module
- Run Src Module
- Export Src Module
- Adr Docs Module
- Venv Setup Module
- Src Parity Module
- Flutter App Module
- Flutter App Module
- Windows Data Module
- Scripts Colab Module
- Cbam Src Module
- Src Train Module
- Create Zip Module
- Src Data Module
- Flutter App Module
- Scripts View Module
- Src Model Module
- Windows Data Module
- Test Cache Module
- Flutter Main Module
- Model Scripts Module
- Src Data Module
- Create Test Module
- Scripts Audit Module
- Cbam Src Module
- Test Model Module
- Colab Cattle Module
- Flutter App Module
- Scripts Test Module
- Logger Run Module
- Src Data Module
- Src Train Module
- Colab Cattle Module
- Flutter App Module
- Image Flutter Module
- Scripts Onnx Module
- Src Export Module
- Src Traits Module
- Test Results Module
- Testing Data Module
- Testing Data Module
- Github Workflows Module
- Agents Knowledge Module
- Colab Cattle Module
- Colab Cattle Module
- Run Scripts Module
- Setup Entry Module
- Agents Module
- Agents Config Module
- Agents Device Module
- Agents Smoke Module
- Concept Knowledge Module
- Concept Phase Module
- Concept Phase Module
- Concept Phase Module
- Docs Training Module
- Requirements Module
- Windows Data Module
- Windows Data Module
- Windows Data Module

## God Nodes (most connected - your core abstractions)
1. `BreedClassifier` - 33 edges
2. `main()` - 28 edges
3. `init_run_logger()` - 24 edges
4. `get_dataloaders()` - 19 edges
5. `log_event()` - 19 edges
6. `prepare_splits()` - 15 edges
7. `main()` - 15 edges
8. `find_latest_checkpoint()` - 15 edges
9. `make_run_id()` - 14 edges
10. `resolve_checkpoint()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `ModelManager` --uses--> `BreedClassifier`  [INFERRED]
  test_model.py → src/model.py
- `_log_event()` --calls--> `log_event()`  [EXTRACTED]
  local_train.py → src/run_logger.py
- `_run()` --calls--> `get_logger()`  [EXTRACTED]
  local_train.py → src/run_logger.py
- `stage_export()` --calls--> `find_latest_checkpoint()`  [EXTRACTED]
  local_train.py → src/run_utils.py
- `main()` --calls--> `init_run_logger()`  [EXTRACTED]
  local_train.py → src/run_logger.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **QAT Failure and PTQ Pivot** — test_results_training_result_phase3_degradation, test_results_training_result_qat_export_mismatch, test_results_training_result_int8_conversion_failure [EXTRACTED 1.00]
- **Colab T4 training setup flow** — colab_cattle_buffalo_trainer, kaggle_dataset, t4_gpu, scripts_create_colab_project_zip [INFERRED 0.75]
- **Label asset trio consumed by the Flutter inference engine** — flutter_app_assets_models_labels_binary, flutter_app_assets_models_labels_cattle, flutter_app_assets_models_labels_buffalo, concept_flutter_app [EXTRACTED 1.00]
- **Breed Classifier Model Tester GUI Feature Set** — test_results_screenshot_20260914_213516_image_upload_panel, test_results_screenshot_20260914_213516_prediction_results_panel, test_results_screenshot_20260914_213516_select_model_dropdown, test_results_screenshot_20260914_213516_analyze_breed_button, test_results_screenshot_20260914_213516_lite2_fp32_onnx_model [EXTRACTED 0.95]
- **Cattle Breed Prediction Outputs Across Test Screenshots** — test_results_screenshot_20260913_135102_umblachery_prediction, test_results_screenshot_20260914_213516_amritmahal_prediction, test_results_screenshot_20260914_215842_amritmahal_prediction, test_results_screenshot_20260914_215909_kenkatha_prediction [INFERRED 0.85]
- **Cattle/Buffalo breed classifier test image set (Testing data, chunk 6 of 6)** — testing_data_amruthamahal_01_amruthamahal, testing_data_baragur_02_baragur, testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hallikaru_02_hallikaru, testing_data_hariana_01_hariana, testing_data_malenadu_gidda_01_malenadu_gidda, testing_data_nagori_01_nagori, testing_data_raghav_gir_bull_at_hyderabad_raghav_gir_bull [EXTRACTED 1.00]
- **Karnataka draught-type grey/white zebu cattle (Amruthamahal/Hallikar-type)** — testing_data_amruthamahal_01_amruthamahal, testing_data_hallikaru_02_hallikaru [INFERRED 0.75]
- **White/light-coated humped zebu breeds (visually similar coat colour)** — testing_data_deoni_01_deoni, testing_data_gangatiri_02_gangatiri, testing_data_gaolao_02_gaolao, testing_data_hariana_01_hariana, testing_data_nagori_01_nagori [INFERRED 0.75]
- **Sequential Training Phases** — concept_phase_1_warmup, concept_phase_2_finetune, concept_phase_3_qat [EXTRACTED 1.00]
- **Classification Label Maps** — windows_data_labels_binary, windows_data_labels_cattle, windows_data_labels_buffalo [EXTRACTED 1.00]

## Communities (75 total, 20 thin omitted)

### Community 0 - "Train Local Module"
Cohesion: 0.13
Nodes (34): _banner(), build_dataset_inventory(), _check_windows_build_tools(), _detect_nvidia_gpu(), _elapsed(), _file_size_mb(), _install_torch(), _log_event() (+26 more)

### Community 1 - "Src Model Module"
Cohesion: 0.07
Nodes (18): Dataset, main(), CattleBuffaloDataset, Merge per-breed augmentation overrides on top of the base flags., resolve_breed_augment(), OODDetector, E(x) = -T * log(Σ exp(f_i / T)) across all logit heads., Max softmax probability from binary head (cattle-vs-buffalo). (+10 more)

### Community 2 - "Src Run Module"
Cohesion: 0.09
Nodes (15): atexit, logging, platform, _dumps(), _git_commit(), _install_excepthook(), _json_default(), make_exec_id() (+7 more)

### Community 3 - "Src Train Module"
Cohesion: 0.11
Nodes (26): math, _apply_cosine_margin(), _combined_class_ids(), _compute_loss(), _contrastive_term(), _fmt_metrics(), masked_kd_loss(), masked_loss() (+18 more)

### Community 4 - "Scripts Test Module"
Cohesion: 0.15
Nodes (14): copy, importlib, json, os, pandas, Mine the most-confused breed pairs from a checkpoint's validation set. Writes…, CPU-only synthetic tests for the 2026-09-23 fixes. No dataset/GPU needed. Run:…, _M (+6 more)

### Community 5 - "Results Test Module"
Cohesion: 0.10
Nodes (26): Terminal CLI Breed Classifier Output (Screenshot 2026-09-13 13:51), Image Upload Field (Browse / umblachery_0086.png), Inference Latency Metric (20.1 ms), Species Classification Output (Cattle 95.5% / Buffalo 4.5%), Top-5 Predictions Bar Chart (Umblachery, Pulikulam, Kankrej, Kangayam, Kenkatha), Top Breed Prediction: Umblachery (45.3%), Breed Classifier -- Model Tester Web GUI (Screenshot 2026-09-14 21:35), Prediction: Amritmahal (40.9% Breed Confidence), Species Cattle (75.0%) (+18 more)

### Community 6 - "Flutter App Module"
Cohesion: 0.08
Nodes (24): dart:js_interop, external JSObject get, _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _execute, _exp (+16 more)

### Community 7 - "Flutter Tflite Module"
Cohesion: 0.08
Nodes (23): _binaryLabels, _buffaloLabels, _cattleLabels, dispose, _exp, index, initialize, _inputSize (+15 more)

### Community 8 - "Data Src Module"
Cohesion: 0.11
Nodes (21): random, compute_hashes(), cutmix(), _dhash(), _groups_by_hash(), _hamming(), _make_weighted_sampler(), mixed_collate() (+13 more)

### Community 9 - "Src Evaluate Module"
Cohesion: 0.14
Nodes (18): download_kaggle_dataset(), Download and unzip a Kaggle dataset into dest_dir., matplotlib, matplotlib_pyplot, numpy, get_dataloaders(), Build train/val/test DataLoaders. Args: pin_memory: If None, auto-detect (True…, full_evaluation() (+10 more)

### Community 10 - "Src Efficientnet Module"
Cohesion: 0.16
Nodes (11): EfficientNetLite, load_backbone_weights(), make_activation(), MBConvBlock, CosineHead, _make_breed_head(), Normalised-feature scaled-cosine classifier (ArcFace-style). Forward returns…, check_backbone() (+3 more)

### Community 11 - "Model Test Module"
Cohesion: 0.11
Nodes (22): build_html(), _build_js(), extract_image_metadata(), generate_odt_report(), get_model_spec(), _human_number(), load_presenter_config(), Build the JavaScript block — plain string, no f-string escaping issues. (+14 more)

### Community 12 - "Colab Cattle Module"
Cohesion: 0.09
Nodes (14): find_breed_folders(), img_to_base64(), per_breed_report(), Walk to find leaf dirs with images (breed folders)., Print and plot per-breed accuracy., Display a gallery of misclassified images., Convert an image file to base64 data URI for embedding in HTML., show_misclassified() (+6 more)

### Community 13 - "Data Collection Module"
Cohesion: 0.19
Nodes (20): _absolute(), breed_dir(), _breed_from_name(), create_dataset_manifest(), download_file(), download_kaggle_dataset(), download_web_images(), _locate() (+12 more)

### Community 14 - "Run Src Module"
Cohesion: 0.15
Nodes (18): glob, main(), find_latest(), find_latest_checkpoint(), make_run_id(), Run-scoped output helpers. Every training/export/evaluation run writes to a…, Make a run id safe to embed in a filename. Replaces illegal characters with…, Return a sanitized explicit run tag, or a fresh timestamp. Uses config… (+10 more)

### Community 15 - "Export Src Module"
Cohesion: 0.12
Nodes (19): _calibration_images(), _copy_to_app_assets(), create_portable_export(), export_onnx_int8(), __init__(), export_tflite(), representative_dataset(), main() (+11 more)

### Community 16 - "Adr Docs Module"
Cohesion: 0.16
Nodes (20): ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition, Bharat Pashudhan App (BPA), CBAM attention module, EfficientNet-Lite2 backbone, Guided image capture overlay, Binary head + species-specific sub-heads, Masked per-head loss, Shared-backbone multi-head hierarchical architecture decision (+12 more)

### Community 17 - "Venv Setup Module"
Cohesion: 0.23
Nodes (17): load_index(), main(), write_labels(), check_python(), clean_broken_torch(), create_venv(), detect_nvidia_gpu(), install_requirements() (+9 more)

### Community 18 - "Src Parity Module"
Cohesion: 0.19
Nodes (14): accumulate(), evaluate_on_split(), finalize(), main(), make_onnx_runner(), make_tflite_runner(), make_torch_runner(), run() (+6 more)

### Community 19 - "Flutter App Module"
Cohesion: 0.14
Nodes (16): ChangeNotifier, ../../domain/entities/prediction_result.dart, PredictionResult, CattleBreedApp, build, _buildBody, CameraScreen, _pickAndPredict (+8 more)

### Community 20 - "Flutter App Module"
Cohesion: 0.12
Nodes (16): breed, breedConfidence, breedIndex, BreedScore, confidence, index, label, latencyMs (+8 more)

### Community 21 - "Windows Data Module"
Cohesion: 0.12
Nodes (16): argv, cwd, duration_s, ended_at, env, RUN_EXEC_ID, exec_id, executable (+8 more)

### Community 22 - "Scripts Colab Module"
Cohesion: 0.15
Nodes (12): argparse, py_to_ipynb(), Convert a percent-format .py script to .ipynb., Convert cattle_buffalo_trainer.py to .ipynb notebook format. Parses the `# %%`…, re, build_template(), main(), Generate the ``data/breed_traits.json`` template for the trait aux-heads. Lists… (+4 more)

### Community 23 - "Cbam Src Module"
Cohesion: 0.21
Nodes (13): BreedClassifier, CBAM attention, Documentation Index, Model Architecture, Model Tester GUI, EfficientNet-Lite backbone, ModelManager, 3-head multi-task learning (+5 more)

### Community 24 - "Src Train Module"
Cohesion: 0.12
Nodes (16): describe_transform(), Human-readable list of a Compose's ops (for startup logging)., finish_run(), _build_warmup_cosine_scheduler(), create_portable_export(), main(), Create a linear-warmup + cosine-annealing LR schedule., Pick the best available device and enable CUDA optimizations. (+8 more)

### Community 25 - "Create Zip Module"
Cohesion: 0.16
Nodes (10): create_training_zip(), Script to create a lightweight standalone training zip package. Excludes…, Check if relative path matches any exclusion rule., should_exclude(), Path, pathlib, Create archive.zip for Colab upload from local organized dataset. Usage: python…, Create colab_project.zip containing only the files needed for Colab training.… (+2 more)

### Community 26 - "Src Data Module"
Cohesion: 0.16
Nodes (11): _collect_rows(), prepare_half_splits(), prepare_quarter_splits(), prepare_smoke_splits(), Create a dataset using a fraction of images per breed for faster training. Uses…, Create a dataset using 25% of images per breed for very fast training.…, Per-breed train/val/test index lists with hard minimums. A plain round() split…, Create a tiny dataset for smoke tests by sampling a few images per breed. This… (+3 more)

### Community 27 - "Flutter App Module"
Cohesion: 0.15
Nodes (12): bool get, _busy, dispose, _error, initialize, _initialized, predictImage, _result (+4 more)

### Community 28 - "Scripts View Module"
Cohesion: 0.38
Nodes (12): all_execs(), cmd_diff(), cmd_events(), cmd_list(), cmd_metrics(), cmd_summary(), _final(), load_json() (+4 more)

### Community 29 - "Src Model Module"
Cohesion: 0.21
Nodes (4): BreedClassifier, no_grad, Soft species routing: never let a hard binary argmax discard the other head.…, Like predict(), but returns zero-confidence result for OOD inputs.

### Community 30 - "Windows Data Module"
Cohesion: 0.15
Nodes (12): argv, cwd, env, exec_id, executable, git_commit, hostname, module (+4 more)

### Community 31 - "Test Cache Module"
Cohesion: 0.18
Nodes (9): base64, datetime, io, pil, pil_exiftags, struct, 🐮 Breed Classifier — Model Tester GUI ========================================…, threading (+1 more)

### Community 32 - "Flutter Main Module"
Cohesion: 0.17
Nodes (11): data/ml_engines/android_tflite_engine.dart, data/ml_engines/web_tfjs_engine.dart, ../../domain/services/i_model_service.dart, build, controller, initialize, main, service (+3 more)

### Community 33 - "Model Scripts Module"
Cohesion: 0.24
Nodes (9): detailed_eval(), load_model(), main(), print_report(), no_grad, Diagnose a trained checkpoint on a split (val or test). Reports, for one or two…, shot_bucket(), Make a checkpoint loadable into a plain float BreedClassifier. Phase-3 QAT… (+1 more)

### Community 34 - "Src Data Module"
Cohesion: 0.25
Nodes (11): compute_class_counts(), compute_class_priors(), compute_rare_classes(), _count_per_class(), _effective_num_weights(), _load_class_maps(), Per-class effective-number weights 1/E_n (same as the sampler). Classes with…, Smoothed log class priors for logit adjustment. IMPORTANT: the effective-number… (+3 more)

### Community 35 - "Create Test Module"
Cohesion: 0.20
Nodes (8): Automatic Mixed Precision (AMP), collections, Create a zip file of test-split images for Colab batch evaluation. Reads…, csv, Colab Testing & Evaluation, Colab Training, Kaggle algsoch/breed-cattle-buffalo dataset, Colab T4 GPU

### Community 36 - "Scripts Audit Module"
Cohesion: 0.29
Nodes (9): difflib, hashlib, ahash(), hamming(), list_breeds(), main(), md5(), Audit data/raw before a training run. REPORT ONLY — never deletes anything.… (+1 more)

### Community 37 - "Cbam Src Module"
Cohesion: 0.27
Nodes (4): ChannelAttention, Zero the last Conv2d weight in a Sequential so its output is 0., SpatialAttention, _zero_init_last()

### Community 38 - "Test Model Module"
Cohesion: 0.22
Nodes (3): Read the current log file contents., File + stdout logger for the entire GUI session., SessionLogger

### Community 39 - "Colab Cattle Module"
Cohesion: 0.22
Nodes (8): build_dataset_inventory(), find_breed_dirs(), merge_into_species_dir(), normalize_breed_name(), Find all leaf directories containing images (breed folders)., Normalize breed folder names: lowercase, underscores, strip whitespace., Write a JSON inventory of the dataset to…, Auto-detect breed folders inside source_base and copy/merge them into…

### Community 40 - "Flutter App Module"
Cohesion: 0.22
Nodes (8): dart:typed_data, ../entities/prediction_result.dart, TfliteEngine, TfJsEngine, dispose, IModelService, initialize, predict

### Community 41 - "Scripts Test Module"
Cohesion: 0.22
Nodes (4): _DS, _FakeModel, run_once(), synth_labels()

### Community 42 - "Logger Run Module"
Cohesion: 0.31
Nodes (8): _bootstrap(), _looks_like_project(), Automatic execution-logger bootstrap. Python imports ``sitecustomize`` at…, get_logger(), init_run_logger(), log_event(), Initialise (once) and return the process RunLogger. Idempotent: a second call…, main()

### Community 43 - "Src Data Module"
Cohesion: 0.25
Nodes (6): _color_jitter(), _eval_transform(), _PadToSquare, Resize the LONG side to ``size`` and pad the short side to a square. A 4:3…, Build the training transform. Augmentation is OFF by default (see ``AUG_*`` /…, _train_transform()

### Community 44 - "Src Train Module"
Cohesion: 0.22
Nodes (9): _apply_ema(), _ema_decay_at(), _rare_keep_mask(), Per-sample bool: True = this sample's breed is rare (do not mix)., EMA decay for optimizer step ``step`` (1-based), with warm-up. ``decay_t =…, One EMA update: parameters AND floating buffers (BN running stats)., Run one training epoch with optional AMP, gradient accumulation, label…, run_epoch() (+1 more)

### Community 45 - "Colab Cattle Module"
Cohesion: 0.32
Nodes (8): predict_single(), preprocess_image(), Preprocess a PIL image for ONNX inference. Matches the evaluation transform…, Numerically stable softmax., Run full inference pipeline on a single PIL image. Returns: dict with species,…, softmax(), Image, ndarray

### Community 46 - "Flutter App Module"
Cohesion: 0.39
Nodes (8): 18 Indian buffalo breeds, 57 Indian cattle breeds, Flutter app (cattle_breed_app), labels_binary.txt — cattle/buffalo species labels, labels_buffalo.txt — 18 buffalo breed labels, labels_cattle.txt — 57 cattle breed labels, flutter_app/pubspec.yaml — cattle_breed_app manifest, flutter_app/web/index.html — Flutter web entry

### Community 47 - "Image Flutter Module"
Cohesion: 0.33
Nodes (5): ImagePreprocessor, preprocess, _resizeThenCenterCrop, size, package:image/image.dart

### Community 48 - "Scripts Onnx Module"
Cohesion: 0.47
Nodes (5): expand_images(), main(), Parity check: PyTorch checkpoint vs fp32 ONNX on a small image list. For each…, soft_top5(), torchvision

### Community 51 - "Test Results Module"
Cohesion: 0.50
Nodes (5): Local Training Run Log, GPU VRAM Auto-Scaling, INT8 Conversion per_channel_affine Failure, QAT Phase Degradation, QAT Checkpoint Export Mismatch

### Community 52 - "Testing Data Module"
Cohesion: 0.40
Nodes (5): Amruthamahal_01 — Amruthamahal (Amritmahal) cattle: dark grey-to-black coat with lighter grey body, heavy muscular neck/hump, long thick horns curving upward and outward, rope halter at head; large zebu-type frame (species/build partly buffalo-like), Baragur_02 — Baragur cattle: reddish-brown coat with extensive irregular white patches (patchy piebald), long horns sweeping outward, loose dewlap, rope tether; humped zebu cattle, Hallikaru_02 — Hallikar cattle: light grey/silver-white slender body, long lyre-shaped horns curving backward, prominent hump, rope tether; draught zebu cattle, Malenadu_Gidda_01 — Malenadu Gidda cattle: solid reddish-brown/dun coat, compact small-bodied frame, short curved horns, modest hump; dwarf zebu cattle, Raghav_Gir_bull_at_Hyderabad — Gir bull: reddish-brown coat mottled with white speckling, very prominent hump, long pendulous droopy ears, curved horns, large frame; zebu cattle (Gir)

### Community 53 - "Testing Data Module"
Cohesion: 0.60
Nodes (5): Deoni_01 — Deoni cattle: predominantly white/light grey coat with black mottling and spots on face and legs, black muzzle, long horns curving upward/inward; zebu cattle, Gangatiri_02 — Gangatiri cattle: white/off-white coat, prominent hump, short upward-curving horns, clean slender legs, rope halter; zebu cattle, Gaolao_02 — Gaolao cattle: white body with darker grey head/neck and grey hump, short horizontal horns, rope tether; zebu cattle, Hariana_01 — Hariana cattle: white/light grey coat with darker grey head and muzzle, prominent hump, short horns; zebu cattle, Nagori_01 — Nagori cattle: white/off-white coat, long horns, deep body, standing in a barn stall with rope halter; zebu cattle

### Community 54 - "Github Workflows Module"
Cohesion: 0.83
Nodes (4): GitHub Pages, ci workflow, Deploy MkDocs to GitHub Pages workflow, MkDocs Material

### Community 56 - "Colab Cattle Module"
Cohesion: 0.67
Nodes (3): compute_macro_f1(), Compute macro-averaged F1 across all breeds., Compute macro-averaged F1 across all breeds.

### Community 57 - "Colab Cattle Module"
Cohesion: 0.67
Nodes (3): plot_confusion_matrix(), Plot a confusion matrix heatmap for a single species., Plot a confusion matrix heatmap for a single species.

## Knowledge Gaps
- **135 isolated node(s):** `_ScoredBreed`, `_inputSize`, `_preprocessor`, `_interpreter`, `_binaryLabels` (+130 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 403 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BreedClassifier` connect `Src Model Module` to `Src Model Module`, `Model Scripts Module`, `Src Train Module`, `Scripts Test Module`, `Src Evaluate Module`, `Src Efficientnet Module`, `Run Src Module`, `Scripts Onnx Module`, `Src Train Module`, `Test Cache Module`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Why does `init_run_logger()` connect `Logger Run Module` to `Train Local Module`, `Src Run Module`, `Src Train Module`, `Src Evaluate Module`, `Src Efficientnet Module`, `Run Src Module`, `Export Src Module`, `Src Parity Module`, `Src Train Module`, `Test Cache Module`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `log_event()` connect `Logger Run Module` to `Train Local Module`, `Src Model Module`, `Src Run Module`, `Src Train Module`, `Data Src Module`, `Src Evaluate Module`, `Data Collection Module`, `Run Src Module`, `Export Src Module`, `Src Parity Module`, `Src Train Module`, `Test Cache Module`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `BreedClassifier` (e.g. with `EfficientNetLite` and `ModelManager`) actually correct?**
  _`BreedClassifier` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_ScoredBreed`, `_inputSize`, `_preprocessor` to the rest of the system?**
  _135 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Train Local Module` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._
- **Should `Src Model Module` be split into smaller, more focused modules?**
  _Cohesion score 0.07130124777183601 - nodes in this community are weakly interconnected._