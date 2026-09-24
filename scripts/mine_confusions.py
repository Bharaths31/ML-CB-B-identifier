#!/usr/bin/env python
"""Mine the most-confused breed pairs from a checkpoint's validation set.

Writes ``outputs/metrics/confusion_pairs.json`` with the top-N confused pairs as
GLOBAL class ids (cattle 0..56, buffalo 57..74) so ``src.train --hard-pairs``
can up-weight them in the SupCon loss.

    python scripts/mine_confusions.py --checkpoint <pt> --split val --top 30

Requires the dataset. Read-only.
"""

import argparse
import json
import os
import sys
from collections import Counter

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import CHECKPOINT_DIR, METRICS_DIR, NUM_CATTLE_BREEDS, SPLIT_DIR
from src.data_pipeline import get_dataloaders
from src.export import _sanitize_state_dict
from src.model import BreedClassifier
from src.run_utils import find_latest_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    ap.add_argument("--attention", choices=["cbam", "se"], default="cbam")
    ap.add_argument("--split-dir", default=SPLIT_DIR)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(METRICS_DIR, "confusion_pairs.json"))
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(split_dir=args.split_dir, batch_size=args.batch_size,
                              num_workers=args.num_workers)
    if loaders is None:
        raise SystemExit("no splits; run python -m src.data_pipeline first")
    _, val_loader, test_loader = loaders
    loader = val_loader if args.split == "val" else test_loader

    with open(os.path.join(args.split_dir, "cattle_classes.json")) as f:
        cattle = json.load(f)
    with open(os.path.join(args.split_dir, "buffalo_classes.json")) as f:
        buffalo = json.load(f)
    cattle_inv = {v: k for k, v in cattle.items()}
    buffalo_inv = {v: k for k, v in buffalo.items()}

    ckpt = args.checkpoint or find_latest_checkpoint(CHECKPOINT_DIR, args.backbone)
    if not ckpt:
        raise SystemExit("no checkpoint found; pass --checkpoint")
    model = BreedClassifier(backbone=args.backbone, attention=args.attention)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(_sanitize_state_dict(state), strict=False)
    model.to(device).eval()

    def name(g):
        return (cattle_inv.get(g, f"cattle_{g}") if g < NUM_CATTLE_BREEDS
                else buffalo_inv.get(g - NUM_CATTLE_BREEDS, f"buffalo_{g}"))

    conf = Counter()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            out = model(images)
            cmask = labels["cattle_mask"] > 0.5
            bmask = labels["buffalo_mask"] > 0.5
            if cmask.any():
                pred = out["cattle"][cmask].argmax(1)
                true = labels["cattle"].argmax(1)[cmask]
                for t, p in zip(true.tolist(), pred.tolist()):
                    if t != p:
                        conf[(t, p)] += 1
            if bmask.any():
                pred = out["buffalo"][bmask].argmax(1)
                true = labels["buffalo"].argmax(1)[bmask]
                for t, p in zip(true.tolist(), pred.tolist()):
                    if t != p:
                        conf[(NUM_CATTLE_BREEDS + t, NUM_CATTLE_BREEDS + p)] += 1

    pairs = [{"a": t, "b": p, "a_name": name(t), "b_name": name(p), "count": c}
             for (t, p), c in conf.most_common(args.top)]
    report = {"checkpoint": ckpt, "split": args.split, "top": args.top,
              "pairs": [[p["a"], p["b"]] for p in pairs], "detail": pairs}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[confusions] {len(pairs)} pairs -> {args.out}")
    for p in pairs[:20]:
        print(f"  {p['a_name']:>22s} -> {p['b_name']:<22s} x{p['count']}")
    print("\nUse with:  python -m src.train --hard-pairs "
          f"{args.out}")


if __name__ == "__main__":
    main()
