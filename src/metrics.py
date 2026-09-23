import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import (BEST_METRIC_MACRO_WEIGHT, BEST_METRIC_TOP1_WEIGHT,
                     SHOT_FEW_MAX, SHOT_MEDIUM_MAX)


def _macro_scores(cm):
    """Macro-F1 and macro-recall from a confusion matrix (true x pred).

    Classes with zero support are excluded so that breeds missing from the
    validation split do not dilute the average.
    """
    support = cm.sum(dim=1)
    tp = cm.diag()
    pred_pos = cm.sum(dim=0)
    denom = support + pred_pos
    f1 = torch.where(denom > 0, 2.0 * tp / denom.clamp(min=1),
                     torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support.clamp(min=1),
                         torch.zeros_like(tp))
    present = support > 0
    if present.any():
        macro_f1 = f1[present].mean().item()
        balanced_acc = recall[present].mean().item()
    else:
        macro_f1 = balanced_acc = 0.0
    return macro_f1, balanced_acc


@torch.no_grad()
def evaluate_epoch(model, loader, device, max_batches=None, train_counts=None,
                   few_max=SHOT_FEW_MAX, medium_max=SHOT_MEDIUM_MAX):
    model.eval()
    n_binary = correct_binary = 0
    binary_tp = binary_fp = binary_fn = 0
    n_cattle = correct_cattle = 0
    n_buffalo = correct_buffalo = 0
    n_combined = correct_combined = 0
    n_combined_soft = correct_combined_soft = 0
    n_top3 = correct_top3 = 0
    n_top5 = correct_top5 = 0

    n_cattle_classes = n_buffalo_classes = 0
    cattle_cm = buffalo_cm = None

    # Shot-bucket accuracy (soft-routed) + predicted-class histogram.
    bucket_stats = {"few": [0, 0], "medium": [0, 0], "many": [0, 0]}
    pred_hist = None
    shot_counts = None

    total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    for step, (images, labels) in enumerate(
            tqdm(loader, desc="eval validation", total=total, leave=False,
                 bar_format="{l_bar}{bar:30}{r_bar}")):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
        out = model(images)
        bin_pred = out["binary"].argmax(1)
        bin_true = labels["binary"].argmax(1)
        cattle_pred = out["cattle"].argmax(1)
        cattle_true = labels["cattle"].argmax(1)
        buffalo_pred = out["buffalo"].argmax(1)
        buffalo_true = labels["buffalo"].argmax(1)
        cmask = labels["cattle_mask"] > 0.5
        bmask = labels["buffalo_mask"] > 0.5

        if cattle_cm is None:
            n_cattle_classes = out["cattle"].size(1)
            n_buffalo_classes = out["buffalo"].size(1)
            cattle_cm = torch.zeros(n_cattle_classes, n_cattle_classes,
                                    dtype=torch.long, device=device)
            buffalo_cm = torch.zeros(n_buffalo_classes, n_buffalo_classes,
                                     dtype=torch.long, device=device)

        bs = images.size(0)
        n_binary += bs
        correct_binary += (bin_pred == bin_true).sum().item()
        binary_tp += ((bin_pred == 1) & (bin_true == 1)).sum().item()
        binary_fp += ((bin_pred == 1) & (bin_true == 0)).sum().item()
        binary_fn += ((bin_pred == 0) & (bin_true == 1)).sum().item()

        n_cattle += cmask.sum().item()
        correct_cattle += (cattle_pred[cmask] == cattle_true[cmask]).sum().item()
        n_buffalo += bmask.sum().item()
        correct_buffalo += (buffalo_pred[bmask] == buffalo_true[bmask]).sum().item()

        # --- Per-class confusion (for macro-F1 / balanced accuracy) ---
        if cmask.any():
            ct, cp = cattle_true[cmask], cattle_pred[cmask]
            for t, p in zip(ct.tolist(), cp.tolist()):
                cattle_cm[t, p] += 1
        if bmask.any():
            bt, bp = buffalo_true[bmask], buffalo_pred[bmask]
            for t, p in zip(bt.tolist(), bp.tolist()):
                buffalo_cm[t, p] += 1

        # --- Hard-routing combined top-1/3/5 (legacy metric) ---
        cattle_hit = (bin_pred == 0) & (bin_true == 0) & (cattle_pred == cattle_true)
        buffalo_hit = (bin_pred == 1) & (bin_true == 1) & (buffalo_pred == buffalo_true)
        correct_combined += (cattle_hit | buffalo_hit).sum().item()
        n_combined += bs

        cattle_top3 = out["cattle"].topk(3, dim=1).indices
        buffalo_top3 = out["buffalo"].topk(3, dim=1).indices
        cattle_in_top3 = (cattle_top3 == cattle_true.unsqueeze(1)).any(dim=1)
        buffalo_in_top3 = (buffalo_top3 == buffalo_true.unsqueeze(1)).any(dim=1)

        cattle_top5 = out["cattle"].topk(5, dim=1).indices
        buffalo_top5 = out["buffalo"].topk(5, dim=1).indices
        cattle_in_top5 = (cattle_top5 == cattle_true.unsqueeze(1)).any(dim=1)
        buffalo_in_top5 = (buffalo_top5 == buffalo_true.unsqueeze(1)).any(dim=1)

        is_cattle = (bin_true == 0)
        correct_top3 += torch.where(is_cattle, cattle_in_top3,
                                    buffalo_in_top3).sum().item()
        n_top3 += bs
        correct_top5 += torch.where(is_cattle, cattle_in_top5,
                                    buffalo_in_top5).sum().item()
        n_top5 += bs

        # --- Soft-routing combined top-1 ---
        # p(species) * softmax(breed head): a confident breed head can still
        # win when the binary head is ambiguous, removing the hard two-stage
        # error propagation.
        p_species = F.softmax(out["binary"], dim=1)
        p_cattle = F.softmax(out["cattle"], dim=1)
        p_buffalo = F.softmax(out["buffalo"], dim=1)
        soft_scores = torch.cat([p_species[:, 0:1] * p_cattle,
                                 p_species[:, 1:2] * p_buffalo], dim=1)
        soft_pred = soft_scores.argmax(1)
        true_global = torch.where(is_cattle, cattle_true,
                                  n_cattle_classes + buffalo_true)
        correct_combined_soft += (soft_pred == true_global).sum().item()
        n_combined_soft += bs

        # --- Shot-bucket accuracy + predicted-class histogram ---
        n_global = n_cattle_classes + n_buffalo_classes
        if pred_hist is None:
            pred_hist = torch.zeros(n_global, dtype=torch.long, device=device)
        pred_hist += torch.bincount(soft_pred, minlength=n_global)[:n_global]
        if train_counts is not None:
            if shot_counts is None:
                shot_counts = torch.cat([
                    train_counts["cattle"].to(device),
                    train_counts["buffalo"].to(device)])
            sample_counts = shot_counts[true_global]
            hit = (soft_pred == true_global)
            for name, lo, hi in (("few", 0, few_max),
                                 ("medium", few_max, medium_max),
                                 ("many", medium_max, float("inf"))):
                m = (sample_counts >= lo) & (sample_counts < hi)
                bucket_stats[name][0] += hit[m].sum().item()
                bucket_stats[name][1] += m.sum().item()

    def acc(c, n):
        return c / n if n else 0.0

    if cattle_cm is None:
        cattle_cm = torch.zeros(0, 0, dtype=torch.long)
        buffalo_cm = torch.zeros(0, 0, dtype=torch.long)
    cattle_macro_f1, cattle_balanced = _macro_scores(cattle_cm.cpu())
    buffalo_macro_f1, buffalo_balanced = _macro_scores(buffalo_cm.cpu())

    denom = 2 * binary_tp + binary_fp + binary_fn
    metrics = {
        "binary_acc": acc(correct_binary, n_binary),
        "binary_f1": (2 * binary_tp / denom) if denom else 0.0,
        "cattle_acc": acc(correct_cattle, n_cattle),
        "buffalo_acc": acc(correct_buffalo, n_buffalo),
        "cattle_macro_f1": cattle_macro_f1,
        "buffalo_macro_f1": buffalo_macro_f1,
        "cattle_balanced_acc": cattle_balanced,
        "buffalo_balanced_acc": buffalo_balanced,
        "combined_top1": acc(correct_combined, n_combined),
        "combined_top3": acc(correct_top3, n_top3),
        "combined_top5": acc(correct_top5, n_top5),
        "combined_top1_soft": acc(correct_combined_soft, n_combined_soft),
    }
    metrics["balanced_score"] = 0.5 * (cattle_macro_f1 + buffalo_macro_f1)
    macro = 0.5 * (cattle_macro_f1 + buffalo_macro_f1)
    metrics["blended_score"] = (BEST_METRIC_MACRO_WEIGHT * macro
                                + BEST_METRIC_TOP1_WEIGHT
                                * metrics["combined_top1_soft"])

    # Per-shot-bucket accuracy (soft-routed).
    for name, (c, n) in bucket_stats.items():
        metrics[f"acc_{name}shot"] = acc(c, n)
        metrics[f"n_{name}shot"] = n

    # Entropy of the predicted-class histogram, normalised to [0,1] by log(K).
    # High entropy = diffuse/uncertain predictions; low = confident but
    # possibly collapsed onto a few classes.
    if pred_hist is not None and pred_hist.sum() > 0:
        p = pred_hist.float() / pred_hist.sum()
        ent = -(p * torch.log(p.clamp(min=1e-12))).sum().item()
        metrics["pred_hist_entropy"] = ent / math.log(pred_hist.numel())
    else:
        metrics["pred_hist_entropy"] = 0.0
    return metrics
