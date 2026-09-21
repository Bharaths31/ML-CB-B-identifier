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
| `DROPOUT` | 0.3 | Breed head dropout |
| `BATCH_SIZE` | 64 | Default batch size |
| `NUM_WORKERS` | 4 | DataLoader workers |
| `TRAIN/VAL/TEST_RATIO` | 0.85/0.10/0.05 | Split ratios |
| `PHASE{1,2,3}_EPOCHS` | 8/60/10 | Training epochs (phase 3 opt-in via `--include-qat`) |
| `PHASE{1,2,3}_LR` | 3e-3/2e-4/5e-6 | Learning rates |
| `LOSS_WEIGHT_*` | 0.15/0.50/0.35 | Multi-task loss weights |
| `WEIGHT_DECAY` | 1e-2 | AdamW weight decay |
| `LABEL_SMOOTHING` | 0.05 | Label smoothing factor |
| `WARMUP_EPOCHS` | 3 | Linear warmup epochs (phase 2) |
| `GRADIENT_ACCUMULATION_STEPS` | 2 | Grad accum steps |
| `KD_ALPHA` | 0.7 | Distillation blend: (1-α)·hard CE + α·T²·KL(teacher‖student) |
| `KD_TEMPERATURE` | 4.0 | Distillation softmax temperature |
| `SAMPLER_BETA` | 0.999 | Effective-number-of-samples sampler β |
| `BALANCE_BINARY_HEAD` | True | Per-batch species re-weighting of binary CE |
| `EMA_DECAY` | 0.999 | Phase-2 weight EMA decay (parameters AND BN buffers) |
| `SMOKE_SAMPLES_PER_BREED` | 5 | Images per breed in smoke test |
| `CUTMIX_ALPHA` | 1.0 | CutMix beta distribution α |
| `MIXUP_ALPHA` | 0.3 | MixUp beta distribution α |
| `CUTMIX_MIXUP_PROB` | 0.5 | Per-step probability of CutMix or MixUp |
| `RANDAUGMENT_OPS` | 2 | RandAugment operations |
| `RANDAUGMENT_MAGNITUDE` | 5 | RandAugment intensity |

### Path Constants

| Constant | Path |
|---|---|
| `RAW_DATA_DIR` | `data/raw/` |
| `SPLIT_DIR` | `data/splits/` |
| `CHECKPOINT_DIR` | `outputs/checkpoints/` |
| `EXPORT_DIR` | `outputs/export/` |
| `PORTABLE_EXPORT_DIR` | `outputs/export/portable/` |
| `METRICS_DIR` | `outputs/metrics/` |
| `TFLITE_APP_ASSETS_DIR` | `flutter_app/assets/models/` |

---