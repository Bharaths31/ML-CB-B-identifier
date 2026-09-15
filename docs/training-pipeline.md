# Training Pipeline

The model is trained in **three sequential phases**, each with distinct objectives, frozen layers, and hyperparameters. The entire pipeline is orchestrated by `src/train.py` and invoked automatically by `local_train.py`.

---

## Three-Phase Training

| Phase | Name | Frozen Layers | Learning Rate | Epochs | Loss Weights |
|---|---|---|---|---|---|
| **Phase 1** | Binary head warmup | Backbone + attention + breed heads | 3e-3 | 5 | bin=1.0, cat=0.0, buf=0.0 |
| **Phase 2** | Multi-task fine-tune | Nothing (all trainable) | 2e-4 (warmup + cosine) | 40 | bin=0.5, cat=0.25, buf=0.25 |
| **Phase 3** | QAT (optional) | Nothing | 5e-6 | 10 | bin=0.5, cat=0.25, buf=0.25 |

**Phase 1** trains only the binary (cattle/buffalo) head while keeping the backbone frozen. This anchors the feature extractor before full fine-tuning.

**Phase 2** unfreezes everything and jointly optimizes all three heads (binary, cattle, buffalo) with a masked loss — cattle head loss is only computed on cattle images, buffalo head loss only on buffalo images.

**Phase 3** (optional, `--include-qat` / without `--skip-qat`) performs Quantization-Aware Training for INT8 Android deployment.

---

## SOTA Optimizations

| Technique | Value | Effect |
|---|---|---|
| **Optimizer** | AdamW, weight_decay=1e-2 | Decoupled weight decay, better generalization |
| **LR schedule** | Linear warmup (3 epochs) → Cosine annealing | Stable Phase 2 convergence |
| **Label smoothing** | ε=0.1 | Prevents overconfident predictions |
| **Gradient accumulation** | 2 steps | Effective batch=128 even on small GPUs |
| **Gradient clipping** | max_norm=1.0 | Prevents gradient explosions |

---

## CUDA Optimizations

Automatically applied when a CUDA device is detected:

- `cudnn.benchmark = True` — auto-tunes convolution algorithms for your hardware
- `TF32` enabled for Ampere+ GPUs (A100, RTX 30xx/40xx) — huge speedup, negligible precision loss
- `torch.amp.autocast("cuda")` + `GradScaler` — mixed-precision training (phases 1–2)
- `pin_memory=True` on all DataLoaders
- `persistent_workers=True` on DataLoaders
- `non_blocking=True` on `.to(device)` transfers
- `optimizer.zero_grad(set_to_none=True)` — faster than filling with zeros
- GPU-accelerated CutMix and MixUp (tensor slicing on CUDA, not CPU)
- `torch.compile(model)` — compilation enabled on Linux (auto-disabled on Windows)

---

## CPU Fallback

All CUDA optimizations gracefully skip when running on CPU. AMP scaler is `None`, `cudnn` settings are not touched, `torch.compile` is skipped. CPU training is significantly slower but fully functional.

---

## Loss Functions

**`soft_ce`**: Soft cross-entropy that supports fractional label vectors from CutMix/MixUp:
```
loss = −(target_smooth × log_softmax(pred)).sum(dim=1)
where target_smooth = (1 - ε) × target + ε / num_classes
```

**`masked_loss`**: Combines three heads with species masking:
```
L = w_bin × CE(binary) + w_cat × CE(cattle) × cattle_mask + w_buf × CE(buffalo) × buffalo_mask
```
The cattle/buffalo head losses are only computed on images of the corresponding species. This prevents the buffalo head from receiving gradient signal on cattle images and vice versa.

---

## Data Modes

### Smoke Test (`--smoke-test`)
5 images per breed, 1 epoch per phase. Uses `prepare_smoke_splits()`:
- Samples real images from the full dataset (not artificial batches)
- 60/20/20 train/val/test split
- Full 75-class map maintained
- Runs in seconds — use for CI or pre-training verification

### Half-Data Mode (`--half-data`)
50% of images per breed via `prepare_half_splits()`:
- Deterministic sampling (seed=42) — same images every run
- Standard 85/10/5 stratified split applied to the sampled subset
- Full 75-class map maintained — architecture identical to full training
- ~2× faster than full training

### Quarter-Data Mode (`--quarter-data`)
25% of images per breed via `prepare_quarter_splits()`:
- Deterministic sampling (seed=42) — same images every run
- Standard 85/10/5 stratified split applied to the sampled subset
- Full 75-class map maintained — architecture identical to full training
- ~4× faster than full training — ideal for resource-constrained local machines

All data modes are **mutually exclusive**: `--smoke-test`, `--half-data`, `--quarter-data`.

---

## Automated Local Pipeline

For a fully automated experience that handles all stages from setup to export:

```bash
# Quarter data — fastest local run
python local_train.py --quarter-data

# Half data — balanced
python local_train.py --half-data

# Full training + QAT
python local_train.py --include-qat
```

See [Local Training (Automated)](local-training.md) for all flags.

---

## Auto-Export

After training completes, the pipeline automatically creates a portable export in `outputs/export/portable/` containing:
- `model.pt` — PyTorch checkpoint (state_dict)
- `cattle_classes.json` — cattle breed label map
- `buffalo_classes.json` — buffalo breed label map
- `model_info.json` — architecture metadata + usage instructions

`local_train.py` additionally exports ONNX, INT8, and float16 formats.