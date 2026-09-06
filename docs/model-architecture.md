# 4. Model Architecture

### BreedClassifier (`src/model.py`)

```python
class BreedClassifier(nn.Module):
    backbone: EfficientNetLite     # stem → 7 MBConv stages → head(1280ch)
    attention: CBAM | SEBlock      # inserted after stage 3 (88ch lite2 / 112ch lite4)
    avg_pool: AdaptiveAvgPool2d(1)
    binary_head: Linear(1280→256→2)
    cattle_head: Linear(1280→512→57) + Dropout(0.3)
    buffalo_head: Linear(1280→512→18) + Dropout(0.3)
```

### Forward Path

1. `forward_features(x)`: backbone stages 0..3 → CBAM → stages 4..6 → head → pool → flatten
2. `forward(x)`: features → 3 parallel heads → dict{binary, cattle, buffalo, features}

### Freeze/Unfreeze Methods

- `freeze_backbone()`: freezes backbone + attention
- `freeze_all()` / `unfreeze_all()`: all parameters
- `backbone_eval()` / `backbone_train()`: BatchNorm mode control

### EfficientNet-Lite (`src/efficientnet_lite.py`)

- MBConv blocks with `relu6` activation (no SE in lite variant)
- `forward_until(x, stage_idx)` / `forward_from(x, stage_idx)`: split forward for attention insertion
- Weight loading: `load_backbone_weights()` — strict=False, allows missing `fc.*`

### Attention (`src/cbam.py`)

- **CBAM**: ChannelAttention(avg+max pool → MLP → sigmoid) → SpatialAttention(avg+max concat → conv → sigmoid)
- **SE**: AdaptiveAvgPool → FC reduce → ReLU → FC expand → Sigmoid

---