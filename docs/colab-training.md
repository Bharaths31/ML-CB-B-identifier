# 14. Colab Training

### Notebook Location

- `colab/cattle_buffalo_trainer.ipynb` — main Colab notebook
- `colab/cattle_buffalo_trainer.py` — same content as percent-format script
- `colab/README.md` — setup instructions

### Project Setup Options (in Colab)

1. **GitHub clone** (recommended):
   ```bash
   !rm -rf /content/project   # force-refresh to pick up latest code
   !git clone https://github.com/Bharaths31/ML-CB-B-identifier /content/project
   ```
   > **Important**: Always `rm -rf /content/project` before cloning so that updated code (bug fixes, new features) is pulled correctly. Simply re-running the clone cell without deleting first silently keeps the old cached copy.
2. **Upload `colab_project.zip`**: created by `python scripts/create_colab_project_zip.py`
3. **Google Drive mount**: copy `colab_project.zip` from `My Drive/ML-CB-B-identifier/`

### Dataset Options (in Colab)

1. **Kaggle API / Direct Download**: auto-downloads unified `algsoch/breed-cattle-buffalo` dataset containing both cattle and buffalo breeds
2. **Upload `archive.zip`**: created by `python scripts/create_colab_archive.py`
3. **Google Drive**: copy `archive.zip` from Drive

### T4 GPU Optimizations

| Setting | Value | Reason |
|---|---|---|
| `batch_size` | 64 | Maximizes T4 utilization (15 GB VRAM) |
| `grad_accum` | 2 | Effective batch = 128 |
| `num_workers` | 2 | Colab has 2 CPU cores |
| `prefetch_factor` | 4 | Keeps GPU fed |
| `pin_memory` | True | Faster CPU→GPU transfer |
| AMP | phases 1-2 only | Disabled for QAT phase 3 |
| Dataset location | `/content/data/raw/` | Local SSD, not Drive |

### Known Colab Gotchas

| Symptom | Root Cause | Fix |
|---|---|---|
| `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'` | PyTorch attribute is `total_memory`, not `total_mem` | **Fixed** in `src/train.py:49` and `colab/cattle_buffalo_trainer.py:29,900` |
| Re-running GitHub clone cell doesn't pick up new code | Colab re-uses cached `/content/project/` directory | Add `!rm -rf /content/project` before `git clone` in §1 |
| Runtime restarts wipe all files | Colab free tier has ephemeral storage | Re-run all cells from §0 top-to-bottom after any restart |

---