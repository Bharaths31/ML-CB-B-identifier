# 6. Data Pipeline

### Dataset Layout

```
data/raw/
├── cattle/
│   ├── amritmahal/  (N .jpg files)
│   ├── ayrshire/
│   └── ... (57 breeds)
└── buffalo/
    ├── alambadi/
    ├── banni/
    └── ... (18 breeds)
```

### Split Strategy

- **Full training**: 70/15/15 stratified per `(species, breed)` with long-tail minimums (≥1 val and ≥1 test image for every breed with ≥3 images)
- **Smoke test**: 5 images/breed → 60/20/20 split (tiny but real)

### Augmentation (OFF by default — opt-in per run)

- **Train (no flags)**: Resize(260) → CenterCrop(260) → ToTensor() — identical to eval
- **Train (opt-in)**: Resize(288) → RandomResizedCrop(260, scale=0.8-1.0) → RandomHorizontalFlip → ColorJitter(0.2,0.2,0.2,0.1) → RandAugment(ops=2, mag=5) → ToTensor(), gated by `--rrc` / `--flip` / `--color-jitter` / `--randaugment` (or `--augment-all`)
- **Eval**: Resize(260) → CenterCrop(260) → ToTensor() (`EVAL_MATCH_TRAIN_RESOLUTION=True` switches to Resize(288)+CenterCrop(260))
- **Batch mixing (opt-in, `--mix`)**: 25% chance of CutMix(α=0.4) or MixUp(α=0.2) on the GPU, pairing partners within the same species; disabled for the last 15% of phase 2.

### Performance Optimizations

- **Byte Caching**: `CACHE_IMAGES` stores raw JPEG bytes in RAM instead of decoded `PIL.Image` objects. This prevents RAM Out-Of-Memory (OOM) errors while bypassing slow disk I/O after the first epoch.
- **GPU-Accelerated MixUp/CutMix**: Tensor slicing for data mixing is offloaded to the CUDA device in `run_epoch`, preventing the CPU `DataLoader` from becoming a bottleneck on platforms like Colab.

### Label Encoding

- Binary: one-hot [cattle=0, buffalo=1]
- Breed: one-hot over species-specific classes
- Masks: `cattle_mask=1.0` if cattle, else `buffalo_mask=1.0`

### Sampling

- `WeightedRandomSampler` balances breeds by inverse frequency

---