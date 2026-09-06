# 5. Training Pipeline

### Three-Phase Training (`src/train.py`)

| Phase | What | Frozen | LR | Epochs | Loss Weights |
|---|---|---|---|---|---|
| 1 | Binary head warmup | backbone + attention + breed heads | 3e-3 | 5 | bin=1.0, cat=0.0, buf=0.0 |
| 2 | Multi-task fine-tune | nothing | 2e-4 (warmup+cosine) | 40 | bin=0.5, cat=0.25, buf=0.25 |
| 3 | QAT (for Android) | nothing | 5e-6 | 10 | bin=0.5, cat=0.25, buf=0.25 |

### SOTA Optimizations

- **Optimizer**: AdamW (weight_decay=1e-2) — decoupled weight decay for better generalization
- **Label smoothing**: 0.1 — prevents overconfident predictions
- **LR schedule**: Linear warmup (3 epochs) → Cosine annealing (Phase 2)
- **Gradient accumulation**: 2 steps (effective batch = 128 with batch_size=64)

### CUDA Optimization

- `cudnn.benchmark = True` — auto-tune convolution algorithms
- `TF32` enabled for Ampere+ GPUs
- `torch.amp.autocast("cuda")` + `GradScaler` for mixed-precision
- `pin_memory=True` on all DataLoaders when CUDA available
- `persistent_workers=True` on DataLoaders
- `non_blocking=True` on `.to(device)` transfers
- `optimizer.zero_grad(set_to_none=True)` — faster than fill with zeros
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)`
- **GPU-Accelerated Augmentation**: CutMix and MixUp operations are executed on the CUDA device to relieve CPU DataLoader bottlenecks.

### CPU Fallback

All CUDA optimizations gracefully skip on CPU. AMP scaler is `None`, cudnn settings are not touched.

### Loss Function

- `soft_ce`: soft cross-entropy supporting CutMix/MixUp label mixing
- `masked_loss`: binary CE always active, cattle/buffalo CE only on matching species (masked by cattle_mask/buffalo_mask)

### Smoke Test (`--smoke-test`)

Creates a mini-dataset of 5 images per breed using `prepare_smoke_splits()`:
- Samples from full dataset, not random batches
- Real training signal on actual breed images
- 1 epoch per phase
- Auto-exports portable model after training

### Auto-Export

After training completes, automatically creates a portable export in `outputs/export/portable/` containing:
- `model.pt` — checkpoint with state_dict
- `cattle_classes.json`, `buffalo_classes.json` — label maps
- `model_info.json` — architecture metadata + usage instructions

---