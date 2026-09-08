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
# soft_ce: supports fractional labels from CutMix/MixUp
soft_ce(pred, target) = -(target * log_softmax(pred)).sum(dim=1)

# masked_loss: species-conditional breed loss
total = w_binary * mean(soft_ce(binary_out, binary_label))
      + w_cattle * sum(soft_ce(cattle_out, cattle_label) * cattle_mask) / sum(cattle_mask)
      + w_buffalo * sum(soft_ce(buffalo_out, buffalo_label) * buffalo_mask) / sum(buffalo_mask)
```

Phase 1 weights: `(1.0, 0.0, 0.0)` — only binary head trains
Phase 2/3 weights: `(0.5, 0.25, 0.25)` — all heads train

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
                out = model(images)
                loss = masked_loss(out, labels, ...)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = masked_loss(out, labels, ...)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    scheduler.step()  # if cosine annealing (phase 2)

    # -- Validation --
    model.eval()
    with torch.no_grad():
        metrics = evaluate_epoch(model, val_loader, device)

    # -- Checkpoint --
    if metrics[best_key] >= best:
        best = metrics[best_key]
        torch.save({phase, epoch, val_top1, state_dict}, checkpoint_path)
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
| `combined_top1` | species + breed both correct |
| `combined_top3` | species correct AND true breed in top-3 |

---

## 6. QAT (Quantization-Aware Training)

Phase 3 (optional):
1. Fuse conv-bn pairs: stem, head, all MBConv depthwise/project/expand
2. Apply `prepare_qat()` with `x86` qconfig
3. Train with fake quantization observers
4. After training: `convert()` → INT8 model
5. AMP is **disabled** during QAT (observers don't support fp16)

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
