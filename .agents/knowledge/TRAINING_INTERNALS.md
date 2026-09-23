# Training Internals & Data Flow — Deep Reference

> Read this when you need to understand the exact data transformations, loss computation, or training loop mechanics.

---

## 1. Image → Tensor Pipeline

### Training Path
```
Raw JPEG/PNG (any size)
  → PIL.Image.open().convert("RGB")
  → Resize(260)              # Shortest side → 260, maintains aspect ratio
  → CenterCrop(260)          # 260×260 crop
  → RandAugment(ops=2, mag=9)  # 2 random augmentation ops at intensity 9
  → ToTensor()               # [0,255] uint8 → [0.0,1.0] float32, HWC→CHW
  → [3, 260, 260] tensor
```

### Evaluation Path
```
Raw image → Resize(260) → CenterCrop(260) → ToTensor() → [3, 260, 260]
```

### Batch Mixing (Training Only)
```
50% chance: apply one of:
  - CutMix(α=0.4): paste rectangular patch from shuffled sample
  - MixUp(α=0.2): linear interpolation with shuffled sample
Both mix labels proportionally (soft labels).
```

---

## 2. Label Encoding

For a cattle breed image (e.g., "sahiwal", cattle_classes["sahiwal"] = 42):
```python
{
  "binary":       [1.0, 0.0],           # one-hot: cattle=0
  "cattle":       [0,...,1.0,...,0],     # one-hot at index 42 (57-dim)
  "buffalo":      [0,...,0],             # all zeros (18-dim)
  "cattle_mask":  1.0,                  # active
  "buffalo_mask": 0.0,                  # inactive
}
```

After CutMix/MixUp, labels become fractional (e.g., `binary = [0.7, 0.3]`).

---

## 3. Loss Computation

```python
# soft_ce: supports fractional labels from CutMix/MixUp + OPTIONAL logit adjustment
soft_ce(pred, target, logit_prior=None, tau=0.0) = -(target * log_softmax(pred + tau*log_prior)).sum(dim=1)

# masked_loss: species-conditional breed loss + SupCon
total = w_binary * balanced_mean(soft_ce(binary_out, binary_label))
      + w_cattle * sum(soft_ce(cattle_out, cattle_label) * cattle_mask) / sum(cattle_mask)
      + w_buffalo * sum(soft_ce(buffalo_out, buffalo_label) * buffalo_mask) / sum(buffalo_mask)
      + contrastive_weight * SupCon(projection_embedding, global_class_ids)   # skipped when mixed
```

Phase 1 weights: `(0.15, 0.5, 0.35)` — all heads train (backbone frozen, no SupCon)
Phase 2 weights: start `(0.15, 0.5, 0.35)`, auto-switch to `(0.05, 0.55, 0.40)` once `binary_acc ≥ 0.95` — differential LR + EMA + SupCon
Phase 3 weights: `(0.15, 0.5, 0.35)` — no SupCon

**Logit adjustment is OFF by default** (`LOGIT_ADJUST=False`): the effective-number
sampler already rebalances every batch, so adding `tau*log_prior` too double-corrects
and over-predicts rare breeds at inference. Enable with `--logit-adjust`; the prior
then comes from the effective **sampled** distribution (`--logit-adjust-prior sampled`).

**Binary species balancing** (`BALANCE_BINARY_HEAD=True`): the per-breed
weighted sampler leaves a ~57:18 species prior, so per batch each species'
binary-CE mass is renormalized to 0.5 via sample weights
`w_cat·(1-p_buf) + w_buf·p_buf` (adaptive; identity on balanced batches).

**Mixing is OFF by default** (`--mix` to enable). When on, CutMix/MixUp pair
partners **within the same species** (`_pairing_perm`), so binary labels stay
one-hot and breed targets stay proper distributions. Mixing is disabled for the
last `MIX_OFF_LAST_FRAC` (15%) of phase 2 via `_mix_off_epoch`.

**Distillation** (`masked_kd_loss`, active when `--teacher` is passed):

```python
total = (1 - kd_alpha) * masked_loss                # hard labels, α=0.7
      + kd_alpha * T² * [ w_bin·KL(τ_bin‖s_bin).mean()
                        + w_cat·masked_KL(cattle)   # T=4.0
                        + w_buf·masked_KL(buffalo) ]
```

Teacher runs frozen (eval mode) on the same augmented batch; gradients flow
only through the student. With `kd_alpha=0` it reduces exactly to
`masked_loss`; with an identical teacher the KL term vanishes exactly.

---

## 4. Training Loop Detail

```python
for epoch in range(epochs):
    # -- Training --
    model.train()  # or backbone_eval() in phase 1
    for images, labels in train_loader:
        images → device (non_blocking)
        labels → device (non_blocking)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss = _compute_loss(model, images, labels, ..., teacher_model)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _compute_loss(model, images, labels, ..., teacher_model)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if ema_model is not None:            # phase 2 — see §3/§4 notes
            with torch.no_grad():            # EMA once per OPTIMIZER step
                ...                          # parameters AND BN buffers;
                                             # num_batches_tracked copied;
                                             # decay = min(0.999,(1+t)/(10+t))

    scheduler.step()  # if cosine annealing (phase 2)

    # -- Validation (BOTH raw and EMA, every eval) --
    model.eval()
    raw_metrics = evaluate_epoch(model, val_loader, device, train_counts=...)
    ema_metrics = evaluate_epoch(ema_model, val_loader, device, ...) if ema_model else None
    chosen, src = (ema_metrics, "ema") if (ema_metrics and ema_metrics[best_key] >= raw_metrics[best_key]) \
                   else (raw_metrics, "raw")

    # -- Checkpoint (save whichever of raw/EMA scores better) --
    if chosen[best_key] >= best:
        best = chosen[best_key]
        torch.save({phase, epoch, val_top1, best_metric, source, metrics, state_dict},
                   checkpoint_path)   # checkpoint_path carries the run id
```

---

## 5. Evaluation Metrics

`evaluate_epoch()` computes:

| Metric | Formula |
|---|---|
| `binary_acc` | correct_binary / total |
| `binary_f1` | 2·TP / (2·TP + FP + FN) — for buffalo class |
| `cattle_acc` | correct among cattle-masked samples |
| `buffalo_acc` | correct among buffalo-masked samples |
| `combined_top1` | hard routing: species argmax + breed both correct |
| `combined_top1_soft` | soft routing: argmax of `p(species)·softmax(head)` over 75 classes |
| `combined_top3` / `top5` | species correct AND true breed in top-3/5 |
| `cattle_macro_f1` / `buffalo_macro_f1` | macro-F1 (zero-support classes excluded) |
| `balanced_score` | 0.5·(cattle_macro_f1 + buffalo_macro_f1) |
| `blended_score` | 0.5·macro-F1 + 0.5·`combined_top1_soft` (**checkpoint metric**) |
| `acc_fewshot` / `acc_mediumshot` / `acc_manyshot` | soft-routed acc bucketed by true class train count (<30 / 30–100 / >100) |
| `pred_hist_entropy` | normalised entropy of the predicted-class histogram |

---

## 6. QAT (Quantization-Aware Training) — opt-in via `--include-qat`

Default training is 2 phases; mobile INT8 comes from converter-side PTQ
(`src/export --mode tflite / onnx-int8`). The optional QAT phase:
1. Loads the **best phase-2 EMA checkpoint** first (not final-epoch weights)
2. Fuses conv-bn pairs: stem, head, all MBConv depthwise/project/expand
3. Applies `prepare_qat()` with **per-tensor** observers (`QConfig(default_observer, default_weight_observer)`) — the x86 per-channel default broke `convert()` with `Unsupported qscheme: per_channel_affine`
4. Trains with fake quantization observers
5. After training: `convert()` → INT8 → `<backbone>_quantized.pt` (x86 artifact, not TFLite/ORT)
6. AMP is **disabled** during QAT (observers don't support fp16)

---

## 7. Smoke Test Data Flow

```
prepare_smoke_splits():
  1. Scan all breeds (same as full)
  2. Sample min(5, available) images per breed
  3. Split 60/20/20 (ensure ≥1 per split)
  4. Write same CSV format as full splits
  5. Write class maps from ALL breeds (not just sampled)

Result: ~375 images total (75 breeds × 5), but model has full 57+18 class heads
```

---

## 8. CLI Progress & Monitoring

```
python -m src.train / local_train.py
  ↓ stdout (unbuffered tqdm progress bars)
  ↓ Phase 1/2/3 Epoch Loops:
      - Epoch progress: tqdm batch progress bar with iteration speed
      - Metrics reporting: [train] phase X epoch Y/Z: loss=... | val binary_acc=...
  ↓ Model Checkpointing:
      - Saves best model checkpoints dynamically to outputs/checkpoints/
  ↓ Auto Portable Export:
      - Converts best model into outputs/export/portable/<backbone>_phase2_best/
  ↓ Completion:
      - Summary table printed to console
```
