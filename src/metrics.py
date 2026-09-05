import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_epoch(model, loader, device, max_batches=None):
    model.eval()
    n_binary = correct_binary = 0
    binary_tp = binary_fp = binary_fn = 0
    n_cattle = correct_cattle = 0
    n_buffalo = correct_buffalo = 0
    n_combined = correct_combined = 0
    n_top3 = correct_top3 = 0

    total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    for step, (images, labels) in enumerate(
            tqdm(loader, desc="eval validation", total=total, leave=False,
                 bar_format="{l_bar}{bar:30}{r_bar}")):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        out = model(images)
        bin_pred = out["binary"].argmax(1)
        bin_true = labels["binary"].argmax(1)
        cattle_pred = out["cattle"].argmax(1)
        cattle_true = labels["cattle"].argmax(1)
        buffalo_pred = out["buffalo"].argmax(1)
        buffalo_true = labels["buffalo"].argmax(1)
        cmask = labels["cattle_mask"] > 0.5
        bmask = labels["buffalo_mask"] > 0.5

        n_binary += images.size(0)
        correct_binary += (bin_pred == bin_true).sum().item()
        binary_tp += ((bin_pred == 1) & (bin_true == 1)).sum().item()
        binary_fp += ((bin_pred == 1) & (bin_true == 0)).sum().item()
        binary_fn += ((bin_pred == 0) & (bin_true == 1)).sum().item()

        n_cattle += cmask.sum().item()
        correct_cattle += (cattle_pred[cmask] == cattle_true[cmask]).sum().item()
        n_buffalo += bmask.sum().item()
        correct_buffalo += (buffalo_pred[bmask] == buffalo_true[bmask]).sum().item()

        for i in range(images.size(0)):
            if bin_pred[i] == 0:
                ok = (cattle_pred[i] == cattle_true[i]) and (bin_true[i] == 0)
            else:
                ok = (buffalo_pred[i] == buffalo_true[i]) and (bin_true[i] == 1)
            correct_combined += int(ok)
            n_combined += 1
            if bin_true[i] == 0:
                hit = cattle_true[i] in out["cattle"][i].topk(3).indices
            else:
                hit = buffalo_true[i] in out["buffalo"][i].topk(3).indices
            correct_top3 += int(hit)
            n_top3 += 1

    def acc(c, n):
        return c / n if n else 0.0

    denom = 2 * binary_tp + binary_fp + binary_fn
    return {
        "binary_acc": acc(correct_binary, n_binary),
        "binary_f1": (2 * binary_tp / denom) if denom else 0.0,
        "cattle_acc": acc(correct_cattle, n_cattle),
        "buffalo_acc": acc(correct_buffalo, n_buffalo),
        "combined_top1": acc(correct_combined, n_combined),
        "combined_top3": acc(correct_top3, n_top3),
    }