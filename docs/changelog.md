# 16. Changelog

### 2026-09-20 — Accuracy/Efficiency Overhaul: Distillation, EMA Fix, Mobile INT8 Exports

**Training (`src/train.py`):**
- **Fixed a critical EMA bug**: the EMA now updates BatchNorm buffers (running mean/var) in addition to parameters; `num_batches_tracked` is hard-copied. Previously every phase-2 checkpoint exported stale BN stats, silently degrading validation accuracy and the deployed model.
- **Knowledge distillation**: new `--teacher`, `--teacher-backbone`, `--teacher-attention` flags. Loss = `(1-α)·masked_hard_CE + α·T²·masked_KL(teacher‖student)` on all three heads (α=0.7, T=4.0). Train a lite4 teacher, distill into the unchanged lite2 student — teacher accuracy at zero on-device cost.
- **QAT is now opt-in** (`--include-qat`); default training is a clean 2-phase run. QAT uses **per-tensor observers** (fixes the `Unsupported qscheme: per_channel_affine` conversion failure) and starts from the **best phase-2 EMA checkpoint** instead of final-epoch weights.
- **Binary-head species balancing**: per-batch re-weighting (`BALANCE_BINARY_HEAD=True`) neutralizes the 57:18 breed-count species prior.

**Data (`src/data_pipeline.py`):**
- Sampler switched to **effective-number-of-samples** weighting (`SAMPLER_BETA=0.999`) — softens over-oversampling of 5-image breeds.
- CutMix/MixUp probability raised 0.25 → 0.5 (`CUTMIX_MIXUP_PROB`).
- All `prepare_*_splits()` create the split directory if missing.

**Export (`src/export.py`):**
- New `--mode tflite`: ONNX → onnx2tf → TFLite FP32 + **full-integer INT8 PTQ** (float32 [0,1] I/O), emits `labels_*.txt`, auto-copies into `flutter_app/assets/models/`.
- New `--mode onnx-int8`: QDQ static quantization for ONNX Runtime Mobile (per-channel weights, calibrated on real train images).
- **ImageNet normalization is baked into mobile graphs** — the Flutter app's `pixel/255` preprocessing is now exactly correct with zero app changes.
- Removed the broken x86 PTQ `--mode int8` path (guidance stub remains).

**Verification (`src/parity_check.py`, new):**
- `python -m src.parity_check` compares fp32 PyTorch vs TFLite/ONNX INT8 on val/test (training-equivalent metrics) or `--synthetic N` for artifact-only parity. Gate: INT8 within 1 pt `combined_top1` of fp32.

**Measured:** ONNX INT8 = 7.10 MB (from 25.65 MB FP32); fp32 ONNX exact parity; INT8 max |Δlogit| ≈ 0.05–0.07 on random inputs; end-to-end smoke training (2-phase, distillation, QAT→INT8) verified on CPU.

---

### 2026-09-15 — Architecture Improvements, EMA, & ImageNet Normalization Fix

**Architecture:**
- Deepened `cattle_head` and `buffalo_head` with an extra hidden layer (`BREED_DIM // 2`).
- Integrated `BatchNorm1d` into both breed classification heads for better convergence and to prevent covariate shift.
- Adjusted dropout values for the new layers (`0.3` for first, `0.2` for second).

**Training Pipeline:**
- **Phase 1 (Warmup):** Now trains all three heads (binary, cattle, buffalo) to build robust initial representations, instead of just the binary head.
- **Differential Learning Rates (Phase 2):** Applied fine-grained LR scaling (backbone: 0.1x, attention: 0.5x, heads: 1.0x).
- **EMA:** Integrated Exponential Moving Average (EMA) with a decay of 0.999 for model weights during Phase 2 to drastically improve evaluation stability and generalization.
- **Bug Fix:** Fixed critical bug where the training pipeline lacked `transforms.Normalize()` using ImageNet statistics, aligning it properly with inference logic.
- Reconfigured default portable export to securely capture the best weights from Phase 2 instead of Phase 3, avoiding the massive accuracy drop previously caused by aggressive INT8 QAT, while maintaining a very lightweight model footprint (~27.3 MB) ready for Android deployment.

**Metrics:**
- Added vectorized **Top-5 combined accuracy** tracking to `evaluate_epoch()` alongside Top-1 and Top-3.

---

### 2026-09-15 — Presenter Mode, Logging & Advanced Image Metadata

**Model Tester GUI (`test_model.py`):**
- Added `--dev` (default) and `--present` flag modes.
- Developer Mode (`--dev`): Advanced view showing image metadata (EXIF, size, proportion), cattle/buffalo JSON data, model specifications, and options to edit the presenter's view settings.
- Presenter Mode (`--present`): Clean, minimalist test page that hides detailed technical stats, diminishes confidence metrics, and removes the export option for a cleaner presentation.
- Presenter configurations (like branding, section toggles, and confidence modes) are saved and loaded persistently via `outputs/logs/presenter_config.json`.
- Comprehensive session logging captures all actions (start/stop, model/image selection, prediction results, reasoning) into a separate log file in `outputs/logs/`.

---

### 2026-09-13 — Colab Testing Notebook & Large-Scale Evaluation

**Testing & Evaluation:**
- Added `colab/cattle_buffalo_tester.py` and `colab/cattle_buffalo_tester.ipynb` for automated evaluation of exported models on Google Colab.
- Added comprehensive HTML report generation for single images and batch evaluations.
- Added Large-Scale Kaggle Evaluation mode to automatically download the dataset and test all images.
- Added `create_test_eval_zip.py` script to easily bundle test dataset splits for Colab.
- Updated documentation and knowledge base (`CONTEXT.md`, `README.md`, `docs/`) with testing workflow details.

---

### 2026-09-08 — Fix: `torch.compile` on Windows

**Bug Fix:**
- Fixed `BackendCompilerFailed: Cannot find a working triton installation` error that crashed phase 2 training on Windows.
- Added OS detection in `src/train.py` to automatically disable `torch.compile` (fallback to eager mode) when running on Windows.

---
### 2026-09-07 — Unified Kaggle Dataset & Colab Trainer Update

**Dataset Pipeline & Colab Notebook:**
- Updated dataset download source to unified Kaggle dataset `algsoch/breed-cattle-buffalo` containing pre-structured `cattle/` (57 breeds) and `buffalo/` (18 breeds) subdirectories.
- Simplified Kaggle download logic in `colab/cattle_buffalo_trainer.py` to extract directly into `data/raw/`, eliminating redundant file moving operations and outdated inline comments.
- Regenerated `colab/cattle_buffalo_trainer.ipynb` from updated python script.
- Updated project documentation across `README.md`, `docs/`, and knowledge base.

---

### 2026-09-06 — Hotfix: CUDA `total_mem` AttributeError

**Bug Fix:**
- Fixed `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'` that crashed §6 Training on Colab T4
- Root cause: PyTorch uses `total_memory`, not `total_mem`
- Fixed in `src/train.py` (`setup_device()`) and both occurrences in `colab/cattle_buffalo_trainer.py`
- Regenerated `colab/cattle_buffalo_trainer.ipynb` from fixed `.py`

**Documentation:**
- Added Colab gotchas table to `CONTEXT.md` §14 covering: `total_mem` bug, GitHub clone cache issue, runtime restart behaviour
- Added `rm -rf /content/project` before `git clone` in §14 best practices to ensure latest code is always used

---

### 2026-09-06 — Colab + SOTA Hyperparameters + Android QAT

**Colab Training:**
- Created `colab/` directory with full training notebook
- 3 project setup options: GitHub clone, zip upload, Google Drive
- 3 dataset options: Kaggle API, archive upload, Google Drive
- Hyperparameter configuration cell with all tunable parameters
- Image prediction cell for testing with uploaded images
- Export & download: portable bundle + ONNX + INT8
- GPU memory monitor cell

**SOTA Hyperparameters:**
- Switched from Adam → AdamW (weight_decay=1e-2)
- Added label smoothing (0.1) to soft cross-entropy
- Added linear warmup scheduler (3 epochs) before cosine annealing
- Added gradient accumulation (2 steps, effective batch=128)
- Increased batch size 32 → 64
- Optimized split ratio 80/10/10 → 85/10/5
- Phase 2 epochs 30 → 40, LR 1e-4 → 2e-4
- Phase 1 LR 1e-3 → 3e-3
- Phase 3 LR 1e-5 → 5e-6
- Dropout 0.3 → 0.4

**Data Pipeline:**
- Train augmentation: added RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- Added prefetch_factor=4 to all DataLoaders

**Android Deployment:**
- QAT (Phase 3) enabled by default (not skipped)
- Auto INT8 conversion after QAT
- ONNX export in Colab notebook for mobile deployment

### 2026-09-05 — Major Update

**Training:**
- Added CUDA optimization: `cudnn.benchmark`, TF32, AMP (`torch.amp`), `GradScaler`
- Added gradient clipping (`max_norm=1.0`)
- Enabled `pin_memory`, `persistent_workers`, `non_blocking` transfers
- Smoke test now uses real mini-dataset (5 imgs/breed) instead of 2-batch limit
- Auto portable export after training completes
- Better tqdm progress bars throughout

**Export & Standalone Packaging:**
- Added `portable` mode: self-contained folder with model + labels + metadata
- Improved progress bars on INT8 calibration
- Created `create_training_zip.py` script to generate a clean, standalone training zip package
- Added `.gitignore` configured to track `memory/` while ignoring `.venv/`, `outputs/`, `data/splits/`, `*.zip`, cache files


**Config:**
- Added `PORTABLE_EXPORT_DIR`, `SMOKE_SAMPLES_PER_BREED` constants