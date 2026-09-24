import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .config import (CHECKPOINT_DIR, METRICS_DIR, NUM_BUFFALO_BREEDS,
                     NUM_CATTLE_BREEDS, SPLIT_DIR)
from .data_pipeline import compute_class_counts, get_dataloaders
from .model import BreedClassifier
from .run_utils import (find_latest_checkpoint, make_run_id,
                        resolve_checkpoint, timestamped)


@torch.no_grad()
def full_evaluation(model, loader, device, num_cattle=NUM_CATTLE_BREEDS,
                    num_buffalo=NUM_BUFFALO_BREEDS):
    model.eval()
    cattle_cm = torch.zeros(num_cattle, num_cattle, dtype=torch.long)
    buffalo_cm = torch.zeros(num_buffalo, num_buffalo, dtype=torch.long)
    cattle_total = torch.zeros(num_cattle, dtype=torch.long)
    buffalo_total = torch.zeros(num_buffalo, dtype=torch.long)
    for images, labels in tqdm(loader, desc="eval detail", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        out = model(images)
        bin_true = labels["binary"].argmax(1)
        cattle_true = labels["cattle"].argmax(1)
        buffalo_true = labels["buffalo"].argmax(1)
        cmask = labels["cattle_mask"] > 0.5
        bmask = labels["buffalo_mask"] > 0.5
        if cmask.any():
            pred = out["cattle"][cmask].argmax(1)
            true = cattle_true[cmask]
            for t, p in zip(true.tolist(), pred.tolist()):
                cattle_cm[t, p] += 1
                cattle_total[t] += 1
        if bmask.any():
            pred = out["buffalo"][bmask].argmax(1)
            true = buffalo_true[bmask]
            for t, p in zip(true.tolist(), pred.tolist()):
                buffalo_cm[t, p] += 1
                buffalo_total[t] += 1

    def per_class(cm, total):
        n_classes = cm.size(0)
        precision = []
        recall = []
        for c in range(n_classes):
            tp = cm[c, c].item()
            col = cm[:, c].sum().item()
            row = cm[c, :].sum().item()
            precision.append(tp / col if col else 0.0)
            recall.append(tp / row if row else 0.0)
        return {
            "precision": precision,
            "recall": recall,
            "counts": total.tolist(),
        }

    return {
        "cattle_cm": cattle_cm.numpy(),
        "buffalo_cm": buffalo_cm.numpy(),
        "cattle_total": cattle_total.numpy(),
        "buffalo_total": buffalo_total.numpy(),
        "cattle_per_class": per_class(cattle_cm, cattle_total),
        "buffalo_per_class": per_class(buffalo_cm, buffalo_total),
    }


def _save_confusion(png_path, cm, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Per-head model evaluation")
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    parser.add_argument("--attention", choices=["cbam", "se"], default="cbam")
    parser.add_argument("--checkpoint", default=None,
                        help="trained model checkpoint (.pt)")
    parser.add_argument("--split-dir", default=SPLIT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=METRICS_DIR)
    parser.add_argument("--run-tag", default=None,
                        help="run id for timestamped metric files "
                             "(default: current time DD-MM-YYYY-HH-MM)")
    args = parser.parse_args()

    from .run_logger import init_run_logger, log_event, log_metrics
    init_run_logger(module="src.evaluate")
    log_event("cli_args", category="actions", **vars(args))

    run_id = make_run_id(args.run_tag)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(split_dir=args.split_dir,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    if loaders is None:
        print("[evaluate] no splits found. Run: python -m src.data_pipeline")
        return 1
    _, val_loader, test_loader = loaders

    checkpoint_path = resolve_checkpoint(args.checkpoint, args.backbone,
                                         CHECKPOINT_DIR)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[evaluate] no checkpoint found for '{args.checkpoint}' under "
              f"{CHECKPOINT_DIR}; pass --checkpoint PATH or --run-tag")
        return 1

    from .export import _load_model
    model = _load_model(checkpoint_path, args.backbone, args.attention)
    model.to(device)
    print(f"[evaluate] loaded {checkpoint_path} (run id {run_id})")

    from .metrics import evaluate_epoch

    train_counts = compute_class_counts(args.split_dir)
    val_metrics = evaluate_epoch(model, val_loader, device,
                                 train_counts=train_counts)
    test_metrics = evaluate_epoch(model, test_loader, device,
                                  train_counts=train_counts)
    full = full_evaluation(model, test_loader, device)

    log_metrics(val_metrics, tag="val", category="test", checkpoint=checkpoint_path)
    log_metrics(test_metrics, tag="test", category="test", checkpoint=checkpoint_path)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = timestamped(os.path.join(args.out_dir, args.backbone), run_id)
    np.savetxt(prefix + "_cattle_cm.csv", full["cattle_cm"], delimiter=",", fmt="%d")
    np.savetxt(prefix + "_buffalo_cm.csv", full["buffalo_cm"], delimiter=",", fmt="%d")
    _save_confusion(prefix + "_cattle_cm.png", full["cattle_cm"],
                    "Cattle confusion matrix")
    _save_confusion(prefix + "_buffalo_cm.png", full["buffalo_cm"],
                    "Buffalo confusion matrix")

    report = {
        "backbone": args.backbone,
        "run_id": run_id,
        "checkpoint": checkpoint_path,
        "val": val_metrics,
        "test": test_metrics,
        "test_binary_f1_ok": test_metrics["binary_f1"] >= 0.95,
    }
    with open(prefix + "_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("Validation:")
    for k, v in val_metrics.items():
        print(f"  {k:16s} {v:.4f}")
    print("Test:")
    for k, v in test_metrics.items():
        print(f"  {k:16s} {v:.4f}")
    print(f"[evaluate] metrics saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())