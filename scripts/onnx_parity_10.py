#!/usr/bin/env python
"""Parity check: PyTorch checkpoint vs fp32 ONNX on a small image list.

For each image, preprocess exactly as training eval (shortest-side resize to
260 + CenterCrop(260) + ImageNet normalize), run both the PyTorch model and the
`--mode onnx` (caller-normalized) ONNX graph, and assert:

  * identical soft-routed top-5
  * max |dlogit| < --tol (default 1e-3)

Run on the GPU/dataset machine:

    python -m src.export --mode onnx --checkpoint <pt> --run-tag <runid>
    python scripts/onnx_parity_10.py \
        --checkpoint outputs/checkpoints/lite2_phase2_best_<runid>.pt \
        --onnx outputs/export/lite2_<runid>_fp32.onnx \
        --images "Testing data/**/*.jpg"

Read-only.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (EVAL_MATCH_TRAIN_RESOLUTION, IMAGENET_MEAN,
                        IMAGENET_STD, IMAGE_SIZE, TRAIN_RESIZE)
from src.model import BreedClassifier
from src.run_utils import find_latest_checkpoint
from src.config import CHECKPOINT_DIR

_EVAL_RESIZE = TRAIN_RESIZE if EVAL_MATCH_TRAIN_RESOLUTION else IMAGE_SIZE
TRANSFORM = transforms.Compose([
    transforms.Resize(_EVAL_RESIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def expand_images(spec):
    paths = []
    for item in spec:
        if os.path.isdir(item):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                paths += glob.glob(os.path.join(item, "**", ext), recursive=True)
        else:
            paths += glob.glob(item, recursive=True)
    return sorted(set(paths))


def soft_top5(binary, cattle, buffalo):
    ps = F.softmax(torch.as_tensor(binary), 0)
    pc = F.softmax(torch.as_tensor(cattle), 0)
    pb = F.softmax(torch.as_tensor(buffalo), 0)
    combined = torch.cat([ps[0] * pc, ps[1] * pb])
    return combined.topk(min(5, combined.numel())).indices.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--onnx", required=True, help="fp32 ONNX from --mode onnx")
    ap.add_argument("--images", nargs="+", required=True,
                    help="image files, globs, or directories")
    ap.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    ap.add_argument("--attention", choices=["cbam", "se"], default="cbam")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import onnxruntime as ort
    from src.export import _sanitize_state_dict

    ckpt_path = args.checkpoint or find_latest_checkpoint(CHECKPOINT_DIR, args.backbone)
    if not ckpt_path:
        raise SystemExit("no checkpoint found; pass --checkpoint")

    model = BreedClassifier(backbone=args.backbone, attention=args.attention)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(_sanitize_state_dict(state), strict=False)
    model.eval()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    paths = expand_images(args.images)
    if not paths:
        raise SystemExit("no images matched")
    print(f"checkpoint: {ckpt_path}\nonnx: {args.onnx}\nimages: {len(paths)}")

    n_ok = 0
    for path in paths:
        img = Image.open(path).convert("RGB")
        x = TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        torch_logits = {k: out[k][0].numpy().astype(np.float64)
                        for k in ("binary", "cattle", "buffalo")}
        ort_out = dict(zip(out_names, sess.run(None, {input_name: x.numpy()})))
        onnx_logits = {k: np.asarray(ort_out.get(k, ort_out[list(ort_out)[i]]),
                                     dtype=np.float64).reshape(-1)
                       for i, k in enumerate(("binary", "cattle", "buffalo"))}

        dmax = max(np.abs(torch_logits[k] - onnx_logits[k]).max()
                   for k in torch_logits)
        t5 = soft_top5(torch_logits["binary"], torch_logits["cattle"],
                       torch_logits["buffalo"])
        o5 = soft_top5(onnx_logits["binary"], onnx_logits["cattle"],
                       onnx_logits["buffalo"])
        ok = (t5 == o5) and (dmax < args.tol)
        n_ok += int(ok)
        print(f"  [{'OK ' if ok else 'FAIL'}] {os.path.basename(path):40s} "
              f"max|dlogit|={dmax:.2e}  top5_match={t5 == o5}")

    print(f"\nparity: {n_ok}/{len(paths)} passed (tol={args.tol})")
    if n_ok != len(paths):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
