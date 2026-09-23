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
parity_check.py ← (uses model, data_pipeline, export)
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
  - `.forward(x)` → dict{binary, cattle, buffalo, features, embedding}
  - `.predict(x)` → [B, 75] soft-routed distribution `p(species)·softmax(head)`
  - `.projection_head` — training-only Linear→ReLU→Linear (→128) for the SupCon loss; never exported
  - `.freeze_backbone()` / `.freeze_all()` / `.unfreeze_all()`
  - `.backbone_eval()` / `.backbone_train()`

### `src/data_pipeline.py`
- `_stratified_split(df, rng)` — per-`(species, breed)` train/val/test with long-tail minimums
- `prepare_splits(data_root, split_dir)` → dict|None — 70/15/15 stratified splits
- `prepare_smoke_splits(data_root, split_dir, samples_per_breed)` → dict|None — mini-dataset
- `prepare_half_splits(data_root, split_dir)` → dict|None — 50% dataset subset
- `prepare_quarter_splits(data_root, split_dir)` → dict|None — 25% dataset subset
- `_train_transform(augment=None)` — train transform; **no-aug == eval transform**; opt-in blocks (rrc/flip/color/randaugment)
- `_eval_transform()` — Resize(260)+CenterCrop(260)+Normalize (Resize(288) if `EVAL_MATCH_TRAIN_RESOLUTION`)
- `compute_class_priors(split_dir, source="sampled"|"raw")` → log-prior tensors (absent classes get the smallest present prior)
- `compute_class_counts(split_dir)` → {cattle, buffalo} raw train counts (shot buckets)
- `compute_rare_classes(split_dir, threshold)` → {cattle, buffalo} bool tensors
- `_effective_num_weights(counts, beta)` → weights (0 for zero-count classes; no divide-by-zero)
- `get_dataloaders(split_dir, batch_size, num_workers, pin_memory, augment=None)` → tuple|None
- `CattleBuffaloDataset(manifest, cattle_classes, buffalo_classes, transform)` — PyTorch Dataset
- `_pairing_perm(labels, same_species=True)` — mixing partner permutation (within-species)
- `cutmix(images, labels, alpha, keep, same_species)` / `mixup(...)` — batch augmentation with rare-class `keep` mask + species pairing
- `_make_weighted_sampler(df, beta)` — effective-number sampler keyed on `(species, breed)`
- `mixed_collate(batch)` — collate_fn

### `src/train.py`
- `setup_device(requested)` → (device, use_amp) — CUDA setup with optimizations & auto VRAM scaling
- `soft_ce(pred, target, label_smoothing, logit_prior, tau)` — soft CE + optional logit adjustment
- `supervised_contrastive_loss(embedding, class_ids, temperature)` — SupCon on the projection embedding
- `_combined_class_ids(labels)` / `_rare_keep_mask(labels, rare_masks)` — loss helpers
- `_mix_off_epoch(epochs, mix_off_frac)` — last epoch that mixes
- `_ema_decay_at(step, ema_decay, warmup)` / `_apply_ema(ema_model, model, decay)` — EMA schedule + params/BN-buffer update
- `masked_loss(...)` / `masked_kd_loss(...)` → (total, ce_b, ce_c, ce_buf)
- `_compute_loss(model, images, labels, ...)` → hard CE or KD (+ SupCon)
- `run_epoch(..., mix_prob, same_species, ema_warmup, ema_state, rare_masks, logit_priors, contrastive_weight)` → loss tuple; EMA once per optimizer step
- `train_phase(..., best_key=BEST_METRIC, mix_off_frac, same_species, train_counts, ...)` → best; dual raw/EMA eval, saves the better one
- `_fmt_metrics(tag, m)` — one-line eval summary
- `create_portable_export(...)` → unique out_dir
- `setup_qat(model, device)` → bool — fuse conv-bn + per-tensor QAT observers
- `main()` — CLI entry point (`--mix/--flip/--color-jitter/--randaugment/--rrc/--augment-all`, `--logit-adjust`, `--logit-adjust-prior`, `--rare-threshold`, `--contrastive-weight`, `--run-tag`, `--teacher`, `--include-qat`)

### `src/run_utils.py`
- `make_run_id(run_tag=None, fmt=None)` — `DD-MM-YYYY-HH-MM` (config `RUN_ID_FORMAT`; `TIMESTAMP_OUTPUTS`)
- `timestamped(path, run_id)` / `timestamped_dir(path, run_id)` — insert the run id into a filename/dir
- `unique_path(path)` — `_2`, `_3`, ... if the path exists (never overwrite)
- `find_latest(directory, pattern)` / `find_latest_checkpoint(dir, backbone, phase="phase2")`

### `src/metrics.py`
- `_macro_scores(cm)` — macro-F1 + macro-recall (zero-support classes excluded)
- `evaluate_epoch(model, loader, device, max_batches, train_counts, few_max, medium_max)` → dict{binary/cattle/buffalo acc, macro_f1, balanced_acc, combined_top1/3/5, combined_top1_soft, blended_score, acc_{few,medium,many}shot, pred_hist_entropy}

### `src/evaluate.py`
- `full_evaluation(model, loader, device, ...)` → dict with confusion matrices
- `main()` — CLI entry point (auto-discovers newest checkpoint; timestamped outputs; `--run-tag`)

### `src/export.py`
- `_sanitize_state_dict(state)` — strip `_orig_mod./module.` prefixes + QAT/fused keys
- `_load_model(checkpoint_path, backbone, attention)` — sanitized load, `strict=False`, explicit missing/unexpected report, raises on missing non-projection keys
- `_RawOutputs(model)` — export wrapper, caller-normalized input (test_model.py convention)
- `_MobileOutputs(model)` — export wrapper, input [0,1] with ImageNet normalization baked in
- `_write_label_files(split_dir, out_dir)` — labels_binary/cattle/buffalo.txt
- `_calibration_images(split_dir, limit)` — [0,1] float32 calibration batches from train.csv
- `export_onnx_int8(model, onnx_fp32, out_path, split_dir)` — QDQ static quantization (ORT Mobile)
- `export_tflite(model, backbone, out_dir, split_dir, ..., stem=None)` — ONNX → onnx2tf → TFLite FP32 + INT8 PTQ (timestamped `stem`)
- `create_portable_export(...)` → unique out_dir
- `main()` — CLI entry point (onnx/onnx-int8/tflite/float16/portable; `--run-tag`; auto-discovers newest checkpoint)

### `src/parity_check.py`
- `make_torch_runner(model, device)` / `make_onnx_runner(path, mobile)` / `make_tflite_runner(path)` — unified [0,1]-input runners
- `accumulate(metrics, logits, labels)` / `finalize(metrics)` — training-equivalent metrics
- `synthetic_parity(runnings, n)` — max |Δlogit| vs fp32 on random inputs
- `main()` — CLI: accuracy mode (val/test) or `--synthetic N`; timestamped report (`--run-tag`)

### `scripts/` (run on the GPU/dataset machine; read-only)
- `audit_data.py` — per-breed counts, fuzzy breed-name collisions, `bargur` cross-species dupes, split duplicates, corrupt files
- `diagnose_model.py` — detailed val/test report incl. shot buckets + confusion pairs (raw vs EMA)
- `onnx_parity_10.py` — PyTorch vs fp32 ONNX top-5 + |Δlogit| assertion
- `test_fixes_cpu.py` — CPU-only synthetic unit tests (no data/GPU)

### `src/verify.py`
- `check_backbone(name)` — load weights + print stats
- `check_full_model(name)` → bool — forward pass shape check
- `main()` — CLI entry point

### `test_model.py`
- Standalone PyTorch/ONNX testing GUI server (HTTP server at `http://localhost:8501`). Loads portable exports, PyTorch checkpoints, and `.onnx` models. Renders interactive breed predictions with top-5 confidence bars, supports batch drag-and-drop processing, and ODT report export.

---

## File Size Reference (for token budgeting)

| File | Lines | Purpose |
|---|---|---|
| config.py | 80 | Constants |
| efficientnet_lite.py | 142 | Backbone |
| cbam.py | 68 | Attention |
| model.py | 79 | Classifier |
| data_pipeline.py | 547 | Data + augmentation |
| train.py | 553 | Training pipeline |
| metrics.py | 73 | Evaluation metrics |
| evaluate.py | 155 | Full evaluation |
| export.py | 203 | Export modes |
| parity_check.py | ~330 | fp32 vs mobile artifact parity gate |
| verify.py | 68 | Sanity check |
| test_model.py | 1766 | Standalone Model Testing GUI (Batch/ONNX/ODT) |
| local_train.py | 983 | Automated training workflow script |
