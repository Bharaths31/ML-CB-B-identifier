# Known Constraints & Gotchas

---

1. **QAT + CUDA AMP conflict**
   Phase 3 (QAT) disables the AMP (`GradScaler`) because quantization observers are not compatible with mixed precision. This is intentional and handled automatically — AMP is active for Phases 1–2, disabled for Phase 3.

2. **Backbone weights are included in the repository**
   `efficientnet_lite2.pth` and `efficientnet_lite4.pth` are tracked in git. No separate download is required. Without them, the backbone would train from scratch (significantly worse accuracy). `local_train.py` verifies their presence at startup.

3. **`torch.compile` crashes on Windows**
   Windows lacks native Triton support, causing `BackendCompilerFailed: Cannot find a working triton installation` when `torch.compile` is used. `src/train.py` automatically detects `os.name == 'nt'` and falls back to eager execution. Use `--no-compile` to force this behavior manually on any platform.

4. **`torch.compile(mode="reduce-overhead")` OOM on T4 (Colab)**
   `reduce-overhead` mode uses CUDA Graphs, which pre-allocates significant VRAM during the backwards pass. On 15 GB T4 GPUs, this causes `OutOfMemoryError` during Phase 2. The default `torch.compile(model)` (no mode argument) does not use CUDA Graphs and avoids the OOM.

5. **Fixed head sizes regardless of data mode**
   Model heads are always sized for **57 cattle + 18 buffalo = 75 classes**, even during smoke test or subset training. Unused class outputs simply receive no gradient from those images. This is by design — the model architecture is identical across all data modes.

6. **`WeightedRandomSampler` during training only**
   Training uses inverse-frequency sampling to oversample rare breeds. Evaluation (`src.evaluate`) uses no sampling — the test set reflects the natural distribution of the dataset.

7. **Soft cross-entropy (not hard labels)**
   Training uses soft label vectors because CutMix/MixUp produce fractional labels (e.g., 60% breed A / 40% breed B). Hard one-hot labels are a special case of soft labels and work identically with `soft_ce`.

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