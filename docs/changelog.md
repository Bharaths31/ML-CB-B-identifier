# 16. Changelog

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
- Created `create_training_zip.py` script to generate a clean, webapp-free training zip package
- Added `.gitignore` configured to track `memory/` while ignoring `.venv/`, `outputs/`, `data/splits/`, `*.zip`, cache files

**Webapp:**
- Fixed argument formatting bug (`--phase1_epochs` → `--phase1-epochs`)
- Fixed `jobStatusHTML` crash when metrics object has missing keys
- Added model cache auto-invalidation after training (mtime-based)
- Progress bars now show completion/error states
- Running job indicator with pulse animation in header
- Auto-refresh status, metrics, and exports after job completion
- Added portable export option in UI dropdown

**Config:**
- Added `PORTABLE_EXPORT_DIR`, `SMOKE_SAMPLES_PER_BREED` constants