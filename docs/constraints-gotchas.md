# Known Constraints & Gotchas

---

1. **QAT + CUDA AMP conflict**
   Phase 3 (QAT, opt-in via `--include-qat`) disables the AMP (`GradScaler`) because quantization observers are not compatible with mixed precision. This is intentional and handled automatically — AMP is active for Phases 1–2, disabled for Phase 3.

1b. **EMA must include BatchNorm buffers, once per optimizer step** (`src/train.py`)
   The phase-2 EMA updates parameters **and** BN running stats (`num_batches_tracked` is hard-copied), **once per optimizer step** (not per micro-batch), with warm-up `decay_t = min(0.999, (1+t)/(10+t))`. Removing the buffer sync corrupts every phase-2 checkpoint with stale BN statistics while validation still looks plausible — keep the sync and the per-step placement.

1c. **Mobile artifacts are static batch-1 with [0,1] input**
   TFLite/ONNX-INT8 exports bake ImageNet normalization into the graph and expect RGB float32 in [0,1] (the Flutter app's `pixel/255`). Keep output names/order stable (`binary`, `cattle`, `buffalo`) when editing `_MobileOutputs` in `src/export.py`.

1d. **QAT/compiled checkpoints are re-exportable via the sanitizer**
   `src/export._sanitize_state_dict` strips `_orig_mod.`/`module.` prefixes and QAT observer/fused keys, then loads with `strict=False` (training-only `projection_head.*` tolerated); it raises only on genuinely missing non-projection keys. Export the phase-2 EMA checkpoint for best accuracy.

1e. **TFLite toolchain is optional and Python-version sensitive**
   `--mode tflite` needs `tensorflow` + `onnx2tf` (see requirements.txt). `--mode onnx-int8` needs only `onnxruntime`. TensorFlow is not installable on Python 3.14 — run TFLite conversion from a Python ≤3.13 venv (the Windows venv is 3.13) or Colab.

1f. **Preprocessing must match eval**
   Eval uses shortest-side `Resize(260)` + `CenterCrop(260)` + ImageNet normalize; `test_model.py` and the Flutter preprocessor match. Square `Resize((260,260))` distorts aspect ratio and wrecked external-batch predictions before the 2026-09-23 fix.

1g. **Outputs are timestamped per run**
   Checkpoints/exports/metrics carry a run id (`DD-MM-YYYY-HH-MM`, config `RUN_ID_FORMAT`). Colons are illegal in Windows filenames, hence `-`. `src/run_utils` provides `make_run_id`/`timestamped`/`unique_path`/`find_latest_checkpoint`; tools auto-discover the newest checkpoint.

2. **Backbone weights are included in the repository**
   `efficientnet_lite2.pth` and `efficientnet_lite4.pth` are tracked in git. No separate download is required. Without them, the backbone would train from scratch (significantly worse accuracy). `local_train.py` verifies their presence at startup.

3. **`torch.compile` crashes on Windows**
   Windows lacks native Triton support, causing `BackendCompilerFailed: Cannot find a working triton installation` when `torch.compile` is used. `src/train.py` automatically detects `os.name == 'nt'` and falls back to eager execution. Use `--no-compile` to force this behavior manually on any platform.

4. **`torch.compile(mode="reduce-overhead")` OOM on T4 (Colab)**
   `reduce-overhead` mode uses CUDA Graphs, which pre-allocates significant VRAM during the backwards pass. On 15 GB T4 GPUs, this causes `OutOfMemoryError` during Phase 2. The default `torch.compile(model)` (no mode argument) does not use CUDA Graphs and avoids the OOM.

5. **Fixed head sizes regardless of data mode**
   Model heads are always sized for **57 cattle + 18 buffalo = 75 classes**, even during smoke test or subset training. Unused class outputs simply receive no gradient from those images. This is by design — the model architecture is identical across all data modes.

6. **`WeightedRandomSampler` is the single long-tail mechanism (default on)**
   Training oversamples rare breeds using **effective-number-of-samples** weighting (`SAMPLER_BETA=0.99`). **Logit adjustment is OFF by default** (`LOGIT_ADJUST=False`) because the sampler already rebalances every batch — enabling both double-corrects and over-predicts rare breeds at inference. If enabled with `--logit-adjust`, its prior comes from the effective *sampled* distribution, and absent classes get the smallest present prior (never NaN). Breeds below `RARE_CLASS_THRESHOLD` train images are excluded from CutMix/MixUp. Evaluation uses no sampling — the test set reflects the natural distribution. The binary head is additionally species-balanced per batch (`BALANCE_BINARY_HEAD=True`).

7. **Soft cross-entropy (not hard labels)**
   Training supports soft label vectors because CutMix/MixUp produce fractional labels (e.g., 60% breed A / 40% breed B). Hard one-hot labels are a special case and work identically with `soft_ce`. Mixing is **off by default** and, when enabled, pairs within a species so labels stay well-formed.

8. **Portable export requires `BreedClassifier` class**
   The portable bundle saves `state_dict` (not TorchScript). Loading requires importing `BreedClassifier` from `src/model.py`. For completely framework-free inference, use the ONNX export instead.

9. **Smoke test uses ALL 75 breed classes**
   Even with only 5 images per breed, the class maps include all breeds from the full dataset. Architecture is identical to full training. This ensures smoke test results are a valid proxy for the real training configuration.

10. **Dataset zip cleanup**
    `local_train.py` deletes the downloaded `breed-cattle-buffalo.zip` after extraction to save disk space. If extraction fails partway, delete the zip and the incomplete `data/raw/` directory, then re-run.

11. **`data/splits/` is regenerated on every training run**
    The CSV splits are regenerated from scratch at the start of each `src.train` run. This ensures the correct stratified split for the selected data mode. Do not manually edit files in `data/splits/`.

12. **Windows: `curl` syntax differs**
    On Windows PowerShell, the built-in `curl` is an alias for `Invoke-WebRequest`. Use `curl.exe` explicitly, or use `Invoke-WebRequest` syntax. `local_train.py` uses the `requests` Python library internally to avoid this issue.