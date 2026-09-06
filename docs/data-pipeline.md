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

- **Full training**: 85/10/5 stratified per breed (optimized for maximum training data)
- **Smoke test**: 5 images/breed → 60/20/20 split (tiny but real)

### Augmentation

- **Train**: Resize(288) → RandomResizedCrop(260, scale=0.8-1.0) → RandomHorizontalFlip → ColorJitter(0.2,0.2,0.2,0.1) → RandAugment(ops=2, mag=9) → ToTensor()
- **Eval**: Resize(260) → CenterCrop(260) → ToTensor()
- **Batch mixing**: 50% chance of CutMix(α=0.4) or MixUp(α=0.2) applied directly on the GPU during the training loop.

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