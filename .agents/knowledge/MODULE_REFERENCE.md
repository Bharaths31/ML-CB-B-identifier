# Module Reference — Source Code Cross-Reference

> Quick-reference for navigating between modules. Read this when you need to trace a function call or understand module dependencies.

---

## Dependency Graph

```
config.py ← (imported by everything)
    ↓
efficientnet_lite.py ← cbam.py
    ↓                    ↓
    model.py ←───────────┘
    ↓
data_pipeline.py
    ↓
metrics.py
    ↓
train.py ← (uses model, data_pipeline, metrics)
    ↓
evaluate.py ← (uses model, data_pipeline, metrics)
    ↓
export.py ← (uses model, data_pipeline)
    ↓
verify.py ← (uses model, efficientnet_lite)
```

## Function Index

### `src/config.py`
- Constants only, no functions. Side-effect: creates output directories on import.

### `src/efficientnet_lite.py`
- `make_activation(name)` → nn.Module — returns relu6/silu/relu
- `MBConvBlock(cin, cout, kernel, stride, expand, activation)` — mobile inverted bottleneck
- `EfficientNetLite(arch, activation)` — full backbone
  - `.forward_until(x, stage_index)` — stem + stages[0..stage_index]
  - `.forward_from(x, stage_index)` — stages[stage_index..end] + head
  - `.forward(x)` — full forward
- `load_backbone_weights(backbone, path)` → int — loads pretrained, returns count

### `src/cbam.py`
- `ChannelAttention(channels, reduction)` — avg+max pool → shared MLP
- `SpatialAttention(kernel_size)` — avg+max concat → conv
- `CBAM(channels, reduction, kernel_size)` — channel → spatial
- `SEBlock(channels, reduction)` — squeeze-excitation
- `build_attention(kind, channels)` → nn.Module — factory function

### `src/model.py`
- `BreedClassifier(backbone, num_cattle, num_buffalo, cbam_stage, attention, activation, pretrained_path, dropout)` — main model
  - `.forward_features(x)` → [B, 1280] pooled features
  - `.forward(x)` → dict{binary, cattle, buffalo, features}
  - `.freeze_backbone()` / `.freeze_all()` / `.unfreeze_all()`
  - `.backbone_eval()` / `.backbone_train()`

### `src/data_pipeline.py`
- `prepare_splits(data_root, split_dir)` → dict|None — 85/10/5 stratified splits
- `prepare_smoke_splits(data_root, split_dir, samples_per_breed)` → dict|None — mini-dataset
- `prepare_half_splits(data_root, split_dir)` → dict|None — 50% dataset subset
- `prepare_quarter_splits(data_root, split_dir)` → dict|None — 25% dataset subset
- `get_dataloaders(split_dir, batch_size, num_workers, pin_memory)` → tuple|None
- `CattleBuffaloDataset(manifest, cattle_classes, buffalo_classes, transform)` — PyTorch Dataset
- `cutmix(images, labels, alpha)` / `mixup(images, labels, alpha)` — batch augmentation
- `mixed_collate(batch)` — collate_fn with random CutMix/MixUp

### `src/train.py`
- `setup_device(requested)` → (device, use_amp) — CUDA setup with optimizations & auto VRAM scaling
- `soft_ce(pred, target)` — soft cross-entropy for mixed labels
- `masked_loss(out, labels, w_binary, w_cattle, w_buffalo)` → (total, ce_b, ce_c, ce_buf)
- `run_epoch(model, loader, optimizer, device, loss_weights, scaler, ...)` → loss tuple
- `train_phase(model, loader, val_loader, device, phase, epochs, lr, ...)` → best_acc
- `create_portable_export(checkpoint_path, backbone, split_dir, export_dir)` → out_dir
- `setup_qat(model, device)` → bool — prepare QAT with module fusion
- `main()` — CLI entry point

### `src/metrics.py`
- `evaluate_epoch(model, loader, device, max_batches)` → dict{binary_acc, binary_f1, cattle_acc, buffalo_acc, combined_top1, combined_top3}

### `src/evaluate.py`
- `full_evaluation(model, loader, device, ...)` → dict with confusion matrices
- `main()` — CLI entry point

### `src/export.py`
- `create_portable_export(checkpoint_path, backbone, split_dir, export_dir)` → out_dir
- `main()` — CLI entry point (onnx/int8/float16/portable modes)

### `src/verify.py`
- `check_backbone(name)` — load weights + print stats
- `check_full_model(name)` → bool — forward pass shape check
- `main()` — CLI entry point

### `test_model.py`
- Standalone PyTorch testing GUI server (HTTP server at `http://localhost:8501`). Loads portable exports/checkpoints and renders interactive breed predictions with top-5 confidence bars.

---

## File Size Reference (for token budgeting)

| File | Lines | Purpose |
|---|---|---|
| config.py | 67 | Constants |
| efficientnet_lite.py | 143 | Backbone |
| cbam.py | 69 | Attention |
| model.py | 80 | Classifier |
| data_pipeline.py | ~290 | Data + augmentation |
| train.py | ~275 | Training pipeline |
| metrics.py | 69 | Evaluation metrics |
| evaluate.py | 156 | Full evaluation |
| export.py | ~195 | Export modes |
| verify.py | 69 | Sanity check |
| test_model.py | ~250 | Standalone Model Testing GUI |
