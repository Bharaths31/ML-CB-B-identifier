import sys

import torch

from .config import (BACKBONE_WEIGHTS, CBAM_AFTER_STAGE, IMAGE_SIZE,
                     NUM_BUFFALO_BREEDS, NUM_CATTLE_BREEDS)
from .efficientnet_lite import EfficientNetLite, load_backbone_weights
from .model import BreedClassifier


def check_backbone(name):
    path = BACKBONE_WEIGHTS[name]
    backbone = EfficientNetLite(arch=name)
    n = load_backbone_weights(backbone, path)
    params = sum(p.numel() for p in backbone.parameters())
    print(f"  [{name}] loaded {n} tensors from {path}, "
          f"{params:,} backbone params")
    return backbone


def check_full_model(name):
    path = BACKBONE_WEIGHTS[name]
    model = BreedClassifier(backbone=name, pretrained_path=path)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        feats = model.backbone.forward_until(x, CBAM_AFTER_STAGE)
        h, w = feats.shape[2], feats.shape[3]
        out = model(x)
    ok = (out["binary"].shape == (1, 2)
          and out["cattle"].shape == (1, NUM_CATTLE_BREEDS)
          and out["buffalo"].shape == (1, NUM_BUFFALO_BREEDS)
          and out["features"].shape == (1, 1280))
    print(f"  [{name}] CBAM stage feature map {h}x{w} "
          f"(expect ~{IMAGE_SIZE // 16}x{IMAGE_SIZE // 16})")
    print(f"  [{name}] heads -> binary{tuple(out['binary'].shape)} "
          f"cattle{tuple(out['cattle'].shape)} buffalo{tuple(out['buffalo'].shape)} "
          f"features{tuple(out['features'].shape)} -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    try:
        import torchvision
        print(f"torchvision: {torchvision.__version__}")
    except ImportError:
        print("torchvision: MISSING")
        return 1

    all_ok = True
    for name in ("lite2", "lite4"):
        try:
            check_backbone(name)
            all_ok &= check_full_model(name)
        except Exception as exc:
            print(f"  [{name}] FAILED: {exc}")
            all_ok = False

    if all_ok:
        print("\nVERIFICATION PASSED")
        return 0
    print("\nVERIFICATION FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())