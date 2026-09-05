# ADR-001: ML Architecture for Cattle & Buffalo Breed Recognition

**Status:** Proposed  
**Date:** 2026-08-09  
**Deciders:** Development team, project mentor  
**Related to:** SIH 2025 — Problem Statement 25004 (Ministry of Fisheries, Animal Husbandry & Dairying)

---

## Context

India has 50+ indigenous cattle and buffalo breeds with heavily overlapping phenotypic traits. The Bharat Pashudhan App (BPA) requires FLWs to accurately register breed data in offline rural environments. The reference implementation (SIH 2025 submission) achieves only 57.71% test accuracy against a 85% target. Analysis identifies five structural failure modes:

| ID | Root cause | Current manifestation |
|----|------------|----------------------|
| F1 | Backbone capacity insufficient for 60-class FGVC | MobileNetV2 lacks compound scaling; cannot separate subtle inter-breed features |
| F2 | Post-training INT8 quantization | 10× compression strips floating-point precision at decision boundaries |
| F3 | Two-stage error propagation | YOLO-Nano bad crops irreversibly corrupt classifier input |
| F4 | Data scarcity (~192 images/class) | Insufficient samples for supervised generalization across varied field conditions |
| F5 | No domain adaptation | Background clutter, backlighting, and unusual poses not handled during training |

The deployment context is offline Android devices (mid-range, 3–4 GB RAM) in areas with no reliable connectivity.

---

## Decision

Replace the two-stage YOLO-Nano + MobileNetV2 pipeline with a **shared-backbone multi-head hierarchical architecture**:

- A single **EfficientNet-Lite2** backbone (ImageNet pretrained) shared across all classification heads
- A **CBAM attention module** after backbone stage 4 for discriminative anatomical localization
- A **binary head** (cattle vs. buffalo) gating the prediction path
- Two **species-specific sub-heads** (35 cattle breed classes, 25 buffalo breed classes)
- **Quantization-Aware Training (QAT)** replacing post-training INT8 quantization
- A **guided image capture overlay** in the Android app eliminating the need for an explicit detection stage

---

## Options Considered

### Option A: Incremental two-stage improvement

Preserve the two-stage pipeline; upgrade each component independently.

| Dimension | Assessment |
|-----------|------------|
| Accuracy | Medium — est. 65–70% test accuracy |
| Model size | ~10 MB (YOLOv8n 3.2 MB + MobileNetV3-L 5.5 MB) |
| Complexity | Low — familiar paradigm, incremental changes |
| Error propagation | Still present — YOLO crop quality still gates classifier |
| Team familiarity | High |

**Pros:** Minimal codebase changes. YOLOv8n has significantly better mAP than YOLO-Nano on occluded/crowded scenes.

**Cons:** Does not fix F1 (backbone capacity), F2 (quantization method), or F3 (two-stage error propagation). Two separate models to version and deploy. Combined size triples despite a modest accuracy gain. Fails to reach the 85% target.

---

### Option B: Shared-backbone multi-head hierarchical pipeline *(Recommended)*

One EfficientNet-Lite2 backbone shared across a binary head and two species-specific sub-heads. Guided image capture replaces the detection stage.

| Dimension | Assessment |
|-----------|------------|
| Accuracy | High — est. 83–87% test accuracy |
| Model size | ~2 MB total INT8 |
| Complexity | Medium — multi-task training loop required |
| Error propagation | Eliminated — single model, no intermediate crop |
| Team familiarity | Medium — TF2 multi-output Model API is well-documented |

**Pros:** Fixes all five root causes simultaneously. Sharing the backbone rather than running two separate models makes this the smallest option (see Trade-off Analysis). CBAM provides implicit localization without a detection stage. Hierarchical split reduces per-head class count from 60 to 35/25, directly lowering decision complexity.

**Cons:** Dataset requires a one-time binary cattle/buffalo labeling pass. Multi-task training loop is more complex. CBAM INT8 TFLite export requires custom op delegation or an SE-block fallback (tracked in Action Item 5).

---

### Option C: YOLOv8-Classify unified detection + classification

Use YOLOv8s-cls, which handles detection and classification in a single model pass.

| Dimension | Assessment |
|-----------|------------|
| Accuracy | Medium — est. 68–74% test accuracy |
| Model size | ~12 MB |
| Complexity | Low — Ultralytics API handles training and export |
| Error propagation | Reduced (single-pass model) |
| Team familiarity | High |

**Pros:** Simplest implementation; one model artifact to deploy.

**Cons:** YOLO's classification head operates on stride-8/16 feature maps, which loses fine-grained spatial detail critical for FGVC with 35+ visually similar classes. No native hierarchical output. Largest model of the three options despite the lowest accuracy.

---

## Trade-off Analysis

**Option A vs B:**
Option A patches symptoms but does not resolve root causes. Upgrading MobileNetV2 → MobileNetV3-L adds capacity but not compound scaling; F1 remains. Two separate INT8 models sum to ~10 MB, larger than the current system. Option B trades a more complex training loop for a 15–20 percentage-point accuracy gain and a net model size reduction.

**Option A vs C:**
Option C simplifies the pipeline but uses a backbone not optimized for FGVC. Both options leave test accuracy below the 85% target.

**Option B cost efficiency:**
Two independent EfficientNet-Lite2 classifiers (one per species) would cost ~12 MB. Sharing the backbone collapses this to ~2 MB because the heavy Conv/MBConv layers are computed once — only the lightweight Dense classification heads are species-specific. This uniquely delivers more accuracy at less storage than any alternative.

**CBAM TFLite caveat:**
CBAM's channel-attention sigmoid is not natively quantizable in all TFLite backends. Mitigation options in priority order: (1) use `tfmot` to wrap CBAM with a custom quantization config, (2) replace channel-attention sigmoid with a Squeeze-and-Excitation (SE) block using a width-1 conv (equivalent expressiveness, fully TFLite-compatible), (3) keep CBAM in float16 for the first export and quantize in a follow-up sprint. Tracked in Action Item 5.

---

## Technical Specification

### Model architecture

```
Input (260×260×3)
└── EfficientNet-Lite2 backbone (ImageNet pretrained)
    ├── MBConv stages 0–3  (frozen during warmup)
    └── MBConv stage 4 + CBAM attention
        └── GlobalAveragePooling2D → shared_features [1280-dim]
            ├── Binary head:   Dense(256, ReLU) → Dense(2, softmax)
            ├── Cattle head:   Dense(512, ReLU) → Dropout(0.3) → Dense(35, softmax)
            └── Buffalo head:  Dense(512, ReLU) → Dropout(0.3) → Dense(25, softmax)
```

Input resolution 260×260 is the compound-scaled optimum for EfficientNet-Lite2.
CBAM position after stage 4 preserves semantic-level feature maps (14×14 spatial) while still being early enough to influence final pooling.

### Training protocol

**Phase 1 — Backbone warmup** (5 epochs, LR = 1e-3):
Freeze all backbone weights. Train binary head only on cattle/buffalo binary labels. Ensures stable gradient flow into the shared feature space before multi-task loss is introduced.

**Phase 2 — Multi-task fine-tuning** (30 epochs, LR = 1e-4, cosine decay):
Unfreeze all layers. Train all three heads simultaneously with masked per-head loss:

```
L_total = 0.50 × CE(y_binary, ŷ_binary)
        + 0.25 × CE(y_cattle,  ŷ_cattle)  × cattle_mask
        + 0.25 × CE(y_buffalo, ŷ_buffalo) × buffalo_mask

cattle_mask  = 1 for cattle images,  0 for buffalo images
buffalo_mask = 1 for buffalo images, 0 for cattle images
```

This ensures buffalo images never penalize the cattle head and vice versa. Loss weights (0.50 / 0.25 / 0.25) should be validated on the held-out split and tuned if binary head accuracy diverges.

**Phase 3 — Quantization-aware training** (10 epochs, LR = 1e-5):
Insert fake-quantization nodes at all Conv/Dense layers using `tfmot.quantization.keras.quantize_model()`. MBConv stages 0–3 and the Dense heads are quantized to INT8. Stage 4 + CBAM is kept at float16 for the first release to avoid the delegation issue; INT8 migration deferred to follow-up sprint (Action Item 5).

### Data pipeline

| Stage | Implementation |
|-------|----------------|
| Input resolution | 260×260 |
| Training augmentation | RandAugment(N=2, M=9) → CutMix(α=0.4) → MixUp(α=0.2) |
| Class balancing | `WeightedRandomSampler` with inverse-frequency class weights |
| Validation augmentation | Center crop + normalize only |
| Split | 80 / 10 / 10, stratified by breed class |
| Binary labels | Derived from existing breed-class labels; cattle breeds → 0, buffalo breeds → 1 |

### Inference path (on-device)

```
features = backbone(input_image)         # single forward pass
binary   = binary_head(features)         # cattle=0, buffalo=1

if binary == CATTLE:
    top3 = cattle_head(features).topk(3)
else:
    top3 = buffalo_head(features).topk(3)

return top3  # (breed_name, confidence) × 3
```

Only two of the three heads activate per image. The inactive head's Dense layers are skipped at inference, keeping latency close to a single-head model.

### Deployment

| Component | Spec |
|-----------|------|
| Runtime | TFLite Interpreter + NNAPI delegate |
| Fallback | CPU inference (~90 ms on Redmi 10 / Poco M3 class) |
| Model size | ~2 MB INT8 |
| Preprocessing | Bilinear resize to 260×260, normalize to [0, 1] |
| Output to app | Top-3 (breed, confidence) from active sub-head + binary prediction |

---

## Performance Projections

| Metric | Existing | Option A | **Option B (Rec.)** | Option C |
|--------|----------|----------|---------------------|----------|
| Test accuracy | 57.71% | ~65–70% | **~83–87%** | ~68–74% |
| Validation accuracy | 64.72% | ~70–75% | **~85–89%** | ~72–78% |
| Model size | 2.92 MB | ~10 MB | **~2 MB** | ~12 MB |
| Inference (mid-range Android) | ~80 ms | ~100 ms | **~70 ms** | ~120 ms |
| Two-stage error propagation | Yes | Reduced | **No** | No |

Accuracy projections are estimates based on published benchmarks for EfficientNet-Lite2 on analogous FGVC datasets (CUB-200, Stanford Dogs) under similar data constraints. Actual results depend on label quality and augmentation tuning.

---

## Consequences

**What becomes easier:**
- Single TFLite artifact to deploy, version, sign, and rollback on the BPA Android app
- Adding new breed classes only requires retraining the relevant sub-head; backbone stays frozen
- Shared backbone transfers domain-general cattle features to rare-breed sub-heads that would otherwise lack sufficient training data
- 2 MB model enables reliable offline use on sub-4 GB RAM devices common in rural areas

**What becomes harder:**
- Training pipeline is more complex (multi-output model, per-head loss masking, three-phase schedule)
- Dataset requires a one-time binary labeling pass (cattle / buffalo) for all 11,560 images
- CBAM TFLite INT8 export requires delegation handling or SE-block substitution
- Guided capture overlay adds ~2 weeks of Android development to eliminate the detection stage dependency

**What we will need to revisit:**
- If rare-breed accuracy (classes with < 100 samples) stays below 70% post-augmentation, add a prototypical-network fallback head for those classes
- Monitor binary head F1 independently in production; if it falls below 95%, introduce a confidence threshold that routes uncertain predictions to the expert review dashboard before BPA logging
- Evaluate on-device fine-tuning (federated learning) for regional breed variants post-deployment if annotated data from field devices becomes available

---

## Action Items

1. [ ] Annotate all 11,560 images with binary cattle/buffalo label — cross-reference NBAGR breed database; cattle breeds → label 0, buffalo breeds → label 1
2. [ ] Implement `EfficientNetLite2 + CBAM` in TF2 using the multi-output `Model(inputs, outputs=[binary_out, cattle_out, buffalo_out])` API
3. [ ] Build three-phase training loop: warmup → multi-task fine-tune → QAT; implement class-balanced `WeightedRandomSampler` and cosine LR schedule
4. [ ] Integrate `tfmot.quantization.keras.quantize_model()` for Phase 3; define custom quantization config for CBAM layers
5. [ ] Resolve CBAM TFLite INT8 delegation: test `tfmot` custom op config first; fall back to SE-block if needed; profile final INT8 model on Redmi 10 with NNAPI enabled
6. [ ] Benchmark INT8 model on at least two representative FLW device classes (e.g., Redmi 10 and a sub-3 GB device); verify <150 ms latency on fallback CPU path
7. [ ] Add guided framing overlay to Android camera screen — centered bounding box guide with "Centre the animal" text prompt; no network call triggered until user confirms frame
8. [ ] Update Android inference wrapper to pass Top-3 (breed, confidence) results from the active sub-head to the BPA result screen
9. [ ] Update Flask expert dashboard schema to receive per-head confidence vectors and binary prediction; deprecate flat 60-dim confidence vector
10. [ ] Set up per-class confusion matrix logging in evaluation script; automate per-head accuracy, precision, and recall reporting for post-training analysis and field monitoring
