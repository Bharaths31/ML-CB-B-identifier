#!/usr/bin/env python
"""Diagnose a trained checkpoint on a split (val or test).

Reports, for one or two checkpoints (e.g. raw vs EMA):

  * binary / cattle / buffalo accuracy
  * combined top-1 / top-3 / top-5 (hard routing) and top-1 (soft routing)
  * per-class recall grouped by TRAIN image count (few / medium / many shot)
  * predicted-class and true-class histograms
  * top-N most frequent confusion pairs

Run this on the GPU/dataset machine, e.g.:

    python scripts/diagnose_model.py --checkpoint outputs/checkpoints/lite2_phase2_best_<runid>.pt
    python scripts/diagnose_model.py --checkpoint a.pt --ema b.pt --split test

This script is read-only: it never writes into the dataset.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (CHECKPOINT_DIR, IMAGE_SIZE, SHOT_FEW_MAX,
                        SHOT_MEDIUM_MAX, SPLIT_DIR)
from src.data_pipeline import get_dataloaders
from src.model import BreedClassifier
from src.run_utils import find_latest_checkpoint


def load_model(checkpoint_path, backbone, attention, device):
    from src.export import _sanitize_state_dict
    model = BreedClassifier(backbone=backbone, attention=attention)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, _ = model.load_state_dict(_sanitize_state_dict(state), strict=False)
    missing = [k for k in missing if not k.startswith("projection_head.")]
    if missing:
        raise SystemExit(f"checkpoint missing required keys: {missing[:10]}")
    model.to(device).eval()
    meta = {}
    if isinstance(ckpt, dict):
        meta = {"epoch": ckpt.get("epoch"), "source": ckpt.get("source"),
                "best_metric": ckpt.get("best_metric"),
                "metrics": ckpt.get("metrics")}
    return model, meta


def shot_bucket(count, few_max, medium_max):
    if count < few_max:
        return "few"
    if count <= medium_max:
        return "medium"
    return "many"


@torch.no_grad()
def detailed_eval(model, loader, device, train_counts, cattle_inv, buffalo_inv,
                  few_max, medium_max, top_confusions=20):
    n_cattle = len(cattle_inv)
    n_buffalo = len(buffalo_inv)
    n_global = n_cattle + n_buffalo
    shot_counts = torch.cat([train_counts["cattle"], train_counts["buffalo"]])

    counts = Counter()
    correct = Counter()
    pred_hist = Counter()
    true_hist = Counter()
    confusions = Counter()
    bucket_tot = Counter()
    bucket_hit = Counter()

    top1 = top3 = top5 = soft_top1 = 0
    n = 0
    bin_correct = 0
    cat_correct = cat_total = 0
    buf_correct = buf_total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        out = model(images)
        bin_true = labels["binary"].argmax(1)
        cat_true = labels["cattle"].argmax(1)
        buf_true = labels["buffalo"].argmax(1)
        cmask = labels["cattle_mask"] > 0.5
        bmask = labels["buffalo_mask"] > 0.5
        bs = images.size(0)
        n += bs

        bin_pred = out["binary"].argmax(1)
        bin_correct += (bin_pred == bin_true).sum().item()
        cat_pred = out["cattle"].argmax(1)
        buf_pred = out["buffalo"].argmax(1)
        cat_correct += (cat_pred[cmask] == cat_true[cmask]).sum().item()
        cat_total += int(cmask.sum())
        buf_correct += (buf_pred[bmask] == buf_true[bmask]).sum().item()
        buf_total += int(bmask.sum())

        is_cat = bin_true == 0
        # hard-routing top1
        hit = ((bin_pred == 0) & is_cat & (cat_pred == cat_true)) | \
              ((bin_pred == 1) & ~is_cat & (buf_pred == buf_true))
        top1 += hit.sum().item()
        # top-k within the TRUE species head
        c3 = out["cattle"].topk(min(3, n_cattle), 1).indices
        b3 = out["buffalo"].topk(min(3, n_buffalo), 1).indices
        c5 = out["cattle"].topk(min(5, n_cattle), 1).indices
        b5 = out["buffalo"].topk(min(5, n_buffalo), 1).indices
        top3 += torch.where(is_cat,
                            (c3 == cat_true.unsqueeze(1)).any(1),
                            (b3 == buf_true.unsqueeze(1)).any(1)).sum().item()
        top5 += torch.where(is_cat,
                            (c5 == cat_true.unsqueeze(1)).any(1),
                            (b5 == buf_true.unsqueeze(1)).any(1)).sum().item()
        # soft routing
        ps = torch.softmax(out["binary"], 1)
        soft = torch.cat([ps[:, 0:1] * torch.softmax(out["cattle"], 1),
                          ps[:, 1:2] * torch.softmax(out["buffalo"], 1)], 1)
        soft_pred = soft.argmax(1)
        true_global = torch.where(is_cat, cat_true, n_cattle + buf_true)
        soft_top1 += (soft_pred == true_global).sum().item()

        for t, p in zip(true_global.tolist(), soft_pred.tolist()):
            true_hist[t] += 1
            pred_hist[p] += 1
            counts[t] += 1
            if t == p:
                correct[t] += 1
            else:
                confusions[(t, p)] += 1
            b = shot_bucket(int(shot_counts[t]), few_max, medium_max)
            bucket_tot[b] += 1
            if t == p:
                bucket_hit[b] += 1

    def label(g):
        if g < n_cattle:
            return cattle_inv.get(g, f"cattle_{g}")
        return buffalo_inv.get(g - n_cattle, f"buffalo_{g - n_cattle}")

    def safe(a, b):
        return a / b if b else 0.0

    per_class = {
        label(g): {"train_n": int(shot_counts[g]), "val_n": counts[g],
                   "recall": safe(correct[g], counts[g])}
        for g in range(n_global) if counts[g] > 0
    }
    top_conf = [
        {"true": label(t), "pred": label(p), "count": c}
        for (t, p), c in confusions.most_common(top_confusions)
    ]
    return {
        "n": n,
        "binary_acc": safe(bin_correct, n),
        "cattle_acc": safe(cat_correct, cat_total),
        "buffalo_acc": safe(buf_correct, buf_total),
        "combined_top1": safe(top1, n),
        "combined_top3": safe(top3, n),
        "combined_top5": safe(top5, n),
        "combined_top1_soft": safe(soft_top1, n),
        "bucket_acc": {b: safe(bucket_hit[b], bucket_tot[b])
                       for b in ("few", "medium", "many")},
        "bucket_n": {b: bucket_tot[b] for b in ("few", "medium", "many")},
        "true_hist": {label(g): true_hist[g] for g in range(n_global) if true_hist[g]},
        "pred_hist": {label(g): pred_hist[g] for g in range(n_global) if pred_hist[g]},
        "top_confusions": top_conf,
        "per_class": per_class,
    }


def print_report(tag, rep):
    print(f"\n{'='*66}\n{tag}\n{'='*66}")
    print(f"  n={rep['n']}  binary={rep['binary_acc']:.4f}  "
          f"cattle={rep['cattle_acc']:.4f}  buffalo={rep['buffalo_acc']:.4f}")
    print(f"  combined top1={rep['combined_top1']:.4f}  "
          f"top3={rep['combined_top3']:.4f}  top5={rep['combined_top5']:.4f}  "
          f"top1_soft={rep['combined_top1_soft']:.4f}")
    print("  shot buckets: " + "  ".join(
        f"{b}={rep['bucket_acc'][b]:.3f} (n={rep['bucket_n'][b]})"
        for b in ("few", "medium", "many")))
    print("  top confusion pairs (true -> pred):")
    for c in rep["top_confusions"][:20]:
        print(f"    {c['true']:>22s} -> {c['pred']:<22s} x{c['count']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint .pt (default: newest phase2 for --backbone)")
    ap.add_argument("--ema", default=None,
                    help="optional second checkpoint (e.g. EMA) to compare")
    ap.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    ap.add_argument("--attention", choices=["cbam", "se"], default="cbam")
    ap.add_argument("--split-dir", default=SPLIT_DIR)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(split_dir=args.split_dir,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    if loaders is None:
        raise SystemExit("no splits found; run python -m src.data_pipeline first")
    _, val_loader, test_loader = loaders
    loader = val_loader if args.split == "val" else test_loader

    import json as _json
    with open(os.path.join(args.split_dir, "cattle_classes.json")) as f:
        cattle_classes = _json.load(f)
    with open(os.path.join(args.split_dir, "buffalo_classes.json")) as f:
        buffalo_classes = _json.load(f)
    cattle_inv = {v: k for k, v in cattle_classes.items()}
    buffalo_inv = {v: k for k, v in buffalo_classes.items()}

    from src.data_pipeline import compute_class_counts
    train_counts = compute_class_counts(args.split_dir)

    ckpt = args.checkpoint or find_latest_checkpoint(CHECKPOINT_DIR, args.backbone)
    if not ckpt:
        raise SystemExit("no checkpoint found; pass --checkpoint")

    reports = {}
    model, meta = load_model(ckpt, args.backbone, args.attention, device)
    print(f"checkpoint: {ckpt}\nmeta: {meta}")
    rep = detailed_eval(model, loader, device, train_counts, cattle_inv,
                        buffalo_inv, SHOT_FEW_MAX, SHOT_MEDIUM_MAX)
    print_report(f"RAW  [{os.path.basename(ckpt)}] on {args.split}", rep)
    reports["raw"] = {"checkpoint": ckpt, "meta": meta, "report": rep}

    if args.ema:
        ema_model, ema_meta = load_model(args.ema, args.backbone,
                                         args.attention, device)
        rep_e = detailed_eval(ema_model, loader, device, train_counts, cattle_inv,
                              buffalo_inv, SHOT_FEW_MAX, SHOT_MEDIUM_MAX)
        print_report(f"EMA  [{os.path.basename(args.ema)}] on {args.split}", rep_e)
        reports["ema"] = {"checkpoint": args.ema, "meta": ema_meta,
                          "report": rep_e}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"\n[json] report -> {args.out}")


if __name__ == "__main__":
    main()
