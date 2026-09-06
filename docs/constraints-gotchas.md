# 13. Known Constraints & Gotchas

1. **QAT + CUDA AMP conflict**: Phase 3 (QAT) disables AMP scaler because quantization observers don't support mixed precision. This is intentional.

2. **Backbone weight files required**: `efficientnet_lite{2,4}.pth` must exist in project root for pretrained initialization. Without them, backbone trains from scratch (much worse accuracy).

3. **Class count mismatch**: If the dataset has fewer breeds than `NUM_CATTLE_BREEDS`/`NUM_BUFFALO_BREEDS`, the model head is still sized for 57/18 classes. Unused class outputs are never trained. This is by design for consistent model architecture.

4. **WeightedRandomSampler**: Training uses inverse-frequency sampling to balance breeds. This means rare breeds are over-sampled. For evaluation, no sampling is used.

5. **Soft cross-entropy**: Training uses soft labels (not hard argmax) because CutMix/MixUp produce fractional label vectors. This works with hard labels too (one-hot = special case of soft).

6. **Model cache invalidation**: The webapp's ModelBox now checks file mtime, so retraining automatically invalidates the cache on next prediction. No manual reload needed.

7. **Portable export is checkpoint-based**: The portable export saves `state_dict` (not TorchScript), so loading requires the `BreedClassifier` class definition. For framework-free deployment, use ONNX export instead.

8. **Smoke test uses ALL breed classes**: Even though only 5 images per breed are used, the class maps include ALL breeds from the full dataset. This ensures the model architecture is identical between smoke and full training.

---