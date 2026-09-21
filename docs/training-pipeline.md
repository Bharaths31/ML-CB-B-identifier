# Training Pipeline

The model is trained in **two sequential phases by default** (QAT is opt-in), each with distinct objectives, frozen layers, and hyperparameters. The entire pipeline is orchestrated by `src/train.py` and invoked automatically by `local_train.py`.

---

## Two-Phase Training (QAT opt-in)

| Phase | Name | Frozen Layers | Learning Rate | Epochs | Loss Weights |
|---|---|---|---|---|---|
| **Phase 1** | All-heads warmup | Backbone + attention | 3e-3 | 8 | bin=0.15, cat=0.5, buf=0.35 |
| **Phase 2** | Multi-task fine-tune | Differential LR (Backbone 0.1x) | 2e-4 (warmup + cosine) + EMA | 60 | bin=0.15, cat=0.5, buf=0.35 |
| **Phase 3** | QAT (opt-in, `--include-qat`) | Nothing | 5e-6 | 10 | bin=0.15, cat=0.5, buf=0.35 |

**Phase 1** trains all three heads (binary, cattle, buffalo) while keeping the backbone frozen. This anchors the new classification layers to the pre-trained feature extractor before full fine-tuning.

**Phase 2** unfreezes everything and jointly optimizes the entire network using differential learning rates (backbone 0.1x, attention 0.5x, heads 1.0x). It also employs an Exponential Moving Average (EMA) model to stabilize weights across iterations. The masked loss ensures the cattle head only trains on cattle images, and the buffalo head only on buffalo images.

**Phase 3** is **opt-in** (`--include-qat`). Android INT8 now comes from **converter-side PTQ** (`src/export --mode tflite / onnx-int8`), so QAT is only a recovery tool when converter PTQ drops more than ~2 pts. It uses per-tensor quantization observers (converts cleanly), starts from the **best phase-2 EMA checkpoint**, and saves `<backbone>_quantized.pt`.

---

## Knowledge Distillation (--teacher)

Train a bigger teacher once, then distill into the student at **zero extra on-device cost** — the student keeps the lite2 architecture, size, and latency:

```bash
# Step 1 — train the teacher (lite4 + CBAM)
python -m src.train --backbone lite4

# Step 2 — distill into the lite2 student
python -m src.train --backbone lite2 \
  --teacher outputs/checkpoints/lite4_phase2_best.pt \
  --teacher-backbone lite4
```

The distillation loss blends hard-label supervision with teacher soft targets on **all three heads** (binary, cattle, buffalo):

```
L = (1 − α) · masked_hard_CE  +  α · T² · masked_KL(teacher ‖ student)
```

Defaults: `KD_ALPHA = 0.7`, `KD_TEMPERATURE = 4.0` (see `src/config.py`). The teacher runs frozen (eval mode) on the same augmented batch, so CutMix/MixUp images are distilled too. With an identical teacher the KL term is exactly zero; with `--teacher` omitted the run is a plain supervised run.

---

## EMA (Exponential Moving Average)

Phase 2 maintains an EMA of the weights with `EMA_DECAY = 0.999`. Validation, checkpoint selection, and the exported model all use the EMA weights.

> **Important:** the EMA tracks **parameters and BatchNorm buffers** (running mean/var are EMA'd, `num_batches_tracked` is hard-copied). Removing the buffer sync produces checkpoints with stale BN statistics that silently degrade both validation accuracy and every model exported from them.

---

## SOTA Optimizations

| Technique | Value | Effect |
|---|---|---|
| **Optimizer** | AdamW, weight_decay=1e-2 | Decoupled weight decay, better generalization |
| **LR schedule** | Linear warmup (3 epochs) → Cosine annealing | Stable Phase 2 convergence |
| **Label smoothing** | ε=0.05 | Prevents overconfident predictions |
| **Gradient accumulation** | 2 steps | Effective batch=128 even on small GPUs |
| **Gradient clipping** | max_norm=1.0 | Prevents gradient explosions |
| **EMA** | decay=0.999 (params + BN buffers) | Stabilizes validation & checkpoint selection |
| **Effective-number sampler** | `SAMPLER_BETA=0.99` | Balances rare breeds without over-oversampling 5-image breeds |
| **Logit adjustment** | `LOGIT_ADJUST_TAU=1.0` | Adds τ·log(prior) to breed logits during training |
| **SupCon features** | `CONTRASTIVE_WEIGHT=0.2` | Separates visually near-identical breeds in embedding space |
| **Binary species balancing** | `BALANCE_BINARY_HEAD=True` | Neutralizes the 57:18 breed-count species prior |
| **Batch mixing** | CutMix(α=1.0) / MixUp(α=0.3), 50% of steps | Strong regularization on GPU |
| **Distillation** | α=0.7, T=4.0 (optional `--teacher`) | Teacher accuracy in the student at zero device cost |

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
loss = −(target_smooth × log_softmax(pred + τ·log_prior)).sum(dim=1)
where target_smooth = (1 - ε) × target + ε / num_classes
```
When `LOGIT_ADJUST=True`, the smoothed **log class prior** (from the training split) is added to the logits with weight `LOGIT_ADJUST_TAU`. This compensates for the long tail without forcing a near-uniform sampler; the shift is absorbed into the learned biases, so inference stays raw.

**`masked_loss`**: Combines three heads with species masking:
```
L = w_bin × CE(binary) + w_cat × CE(cattle) × cattle_mask + w_buf × CE(buffalo) × buffalo_mask
    + λ_supcon × SupCon(projection_head(features))
```
The cattle/buffalo head losses are only computed on images of the corresponding species. This prevents the buffalo head from receiving gradient signal on cattle images and vice versa.

The binary term is additionally **species-balanced per batch** (`BALANCE_BINARY_HEAD=True`): because the per-breed sampler leaves a ~57:18 species prior, each species' binary-CE mass is renormalized to 0.5 with adaptive sample weights (this is an identity on perfectly balanced batches).

The auxiliary **supervised-contrastive term** (`CONTRASTIVE_WEIGHT`) uses the previously-unused pooled 1280-d features through a projection head; it is skipped on CutMix/MixUp batches where labels are soft. Once `binary_acc ≥ BINARY_SATURATION_ACC`, the loss weights automatically rebalance from `0.15/0.50/0.35` to `0.05/0.55/0.40` (see `LOSS_WEIGHT_*_FINAL`).

**`masked_kd_loss`**: the distillation blend described above — `(1−α)·masked_loss + α·T²·masked_KL`, temperature-scaled on all three heads with the same species masking, plus the SupCon term.

**Checkpoint selection**: `BEST_METRIC="balanced_score"` = mean of cattle/buffalo **macro-F1**. Combined top-1 is dominated by the ~10 large breeds and hides regressions on the 30+ rare indigenous breeds.

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
- Stratified split (70/15/15) with long-tail minimums applied to the sampled subset
- Full 75-class map maintained — architecture identical to full training
- ~2× faster than full training

### Quarter-Data Mode (`--quarter-data`)
25% of images per breed via `prepare_quarter_splits()`:
- Deterministic sampling (seed=42) — same images every run
- Stratified split (70/15/15) with long-tail minimums applied to the sampled subset
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

# Opt-in QAT phase (recovery tool; mobile INT8 uses converter PTQ)
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