# 10. Configuration Reference

### `src/config.py` — All Constants

| Constant | Value | Purpose |
|---|---|---|
| `IMAGE_SIZE` | 260 | Input image dimension |
| `NUM_CATTLE_BREEDS` | 57 | Expected cattle breed count |
| `NUM_BUFFALO_BREEDS` | 18 | Expected buffalo breed count |
| `CBAM_AFTER_STAGE` | 3 | Attention insertion point |
| `FEATURE_DIM` | 1280 | Backbone output dimension |
| `BINARY_DIM` | 256 | Binary head hidden dim |
| `BREED_DIM` | 512 | Breed head hidden dim |
| `DROPOUT` | 0.4 | Breed head dropout |
| `BATCH_SIZE` | 64 | Default batch size |
| `NUM_WORKERS` | 4 | DataLoader workers |
| `TRAIN/VAL/TEST_RATIO` | 0.85/0.10/0.05 | Split ratios |
| `PHASE{1,2,3}_EPOCHS` | 5/40/10 | Training epochs |
| `PHASE{1,2,3}_LR` | 3e-3/2e-4/5e-6 | Learning rates |
| `LOSS_WEIGHT_*` | 0.5/0.25/0.25 | Multi-task loss weights |
| `WEIGHT_DECAY` | 1e-2 | AdamW weight decay |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing factor |
| `WARMUP_EPOCHS` | 3 | Linear warmup epochs (phase 2) |
| `GRADIENT_ACCUMULATION_STEPS` | 2 | Grad accum steps |
| `SMOKE_SAMPLES_PER_BREED` | 5 | Images per breed in smoke test |
| `CUTMIX_ALPHA` | 0.4 | CutMix beta distribution α |
| `MIXUP_ALPHA` | 0.2 | MixUp beta distribution α |
| `RANDAUGMENT_OPS` | 2 | RandAugment operations |
| `RANDAUGMENT_MAGNITUDE` | 9 | RandAugment intensity |

### Path Constants

| Constant | Path |
|---|---|
| `RAW_DATA_DIR` | `data/raw/` |
| `SPLIT_DIR` | `data/splits/` |
| `CHECKPOINT_DIR` | `outputs/checkpoints/` |
| `EXPORT_DIR` | `outputs/export/` |
| `PORTABLE_EXPORT_DIR` | `outputs/export/portable/` |
| `METRICS_DIR` | `outputs/metrics/` |

---