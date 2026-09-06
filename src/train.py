import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from tqdm import tqdm

from .config import (BACKBONE_WEIGHTS, BATCH_SIZE, CHECKPOINT_DIR,
                     EVAL_EVERY_PHASE1, EVAL_EVERY_PHASE2, EVAL_EVERY_PHASE3,
                     GRADIENT_ACCUMULATION_STEPS, LABEL_SMOOTHING,
                     LOSS_WEIGHT_BINARY, LOSS_WEIGHT_BUFFALO, LOSS_WEIGHT_CATTLE,
                     NUM_WORKERS, PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS,
                     PHASE2_LR, PHASE3_EPOCHS, PHASE3_LR, PORTABLE_EXPORT_DIR,
                     RAW_DATA_DIR, SEED, SPLIT_DIR, WARMUP_EPOCHS, WEIGHT_DECAY)
from .data_pipeline import get_dataloaders, prepare_smoke_splits, prepare_splits
from .metrics import evaluate_epoch
from .model import BreedClassifier


# ---------------------------------------------------------------------------
# Device setup helpers
# ---------------------------------------------------------------------------

def setup_device(requested):
    """Pick the best available device and enable CUDA optimizations."""
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_amp = False
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Enable TF32 for Ampere+ GPUs (huge speedup, negligible precision loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        use_amp = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[train] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"[train] CUDA optimizations: cudnn.benchmark=True, "
              f"TF32=True, AMP=True")
    else:
        print(f"[train] running on CPU (no CUDA available)")
        print(f"[train] tip: training will be significantly slower without GPU")

    print(f"[train] device: {device}")
    return device, use_amp


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def soft_ce(pred, target, label_smoothing=0.0):
    """Soft cross-entropy with optional label smoothing.

    When label_smoothing > 0, target distribution is smoothed:
        target_smooth = (1 - ε) * target + ε / num_classes
    This prevents overconfident predictions and improves generalization.
    """
    if label_smoothing > 0.0:
        n_classes = pred.size(1)
        target = (1.0 - label_smoothing) * target + label_smoothing / n_classes
    return -(target * F.log_softmax(pred, dim=1)).sum(dim=1)


def masked_loss(out, labels, w_binary, w_cattle, w_buffalo,
                label_smoothing=0.0):
    ce_binary = soft_ce(out["binary"], labels["binary"],
                        label_smoothing).mean()
    ce_cattle = (soft_ce(out["cattle"], labels["cattle"],
                         label_smoothing) * labels["cattle_mask"])
    ce_buffalo = (soft_ce(out["buffalo"], labels["buffalo"],
                          label_smoothing) * labels["buffalo_mask"])
    denom_c = labels["cattle_mask"].sum().clamp(min=1.0)
    denom_b = labels["buffalo_mask"].sum().clamp(min=1.0)
    ce_cattle = ce_cattle.sum() / denom_c
    ce_buffalo = ce_buffalo.sum() / denom_b
    total = (w_binary * ce_binary + w_cattle * ce_cattle + w_buffalo * ce_buffalo)
    return total, ce_binary, ce_cattle, ce_buffalo


# ---------------------------------------------------------------------------
# Training core
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, loss_weights, scaler=None,
              max_batches=None, set_train=None, desc="train",
              grad_accum_steps=1, label_smoothing=0.0):
    """Run one training epoch with optional AMP, gradient accumulation,
    and label smoothing."""
    set_train = set_train or (lambda m: m.train())
    set_train(model)
    running = []
    total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    use_amp = scaler is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=desc, total=total, leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(images)
                loss, ce_b, ce_c, ce_buf = masked_loss(
                    out, labels, *loss_weights,
                    label_smoothing=label_smoothing)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            out = model(images)
            loss, ce_b, ce_c, ce_buf = masked_loss(
                out, labels, *loss_weights,
                label_smoothing=label_smoothing)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running.append((loss.item() * grad_accum_steps, ce_b.item(),
                        ce_c.item(), ce_buf.item()))
        pbar.set_postfix(loss=f"{running[-1][0]:.4f}", refresh=False)

    pbar.close()
    if not running:
        return 0.0, 0.0, 0.0, 0.0
    return tuple(sum(x[i] for x in running) / len(running) for i in range(4))


def _build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Create a linear-warmup + cosine-annealing LR schedule."""
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs)

    warmup_sched = LambdaLR(
        optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs))
    return SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs])


def train_phase(model, loader, val_loader, device, phase, epochs, lr,
                loss_weights, scheduler_factory, checkpoint_path,
                scaler=None, max_batches=None, best_key="combined_top1",
                set_train=None, weight_decay=0.0,
                grad_accum_steps=1, label_smoothing=0.0, eval_every=1):
    """Train a single phase with AdamW, optional AMP, gradient accumulation,
    and label smoothing."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = scheduler_factory(optimizer) if scheduler_factory else None
    best = 0.0

    phase_pbar = tqdm(range(1, epochs + 1), desc=f"Phase {phase}",
                      unit="epoch", leave=True,
                      bar_format="{l_bar}{bar:20}{r_bar}")
    for epoch in phase_pbar:
        loss, ce_b, ce_c, ce_buf = run_epoch(
            model, loader, optimizer, device, loss_weights, scaler,
            max_batches, set_train=set_train,
            desc=f"phase{phase} e{epoch}/{epochs}",
            grad_accum_steps=grad_accum_steps,
            label_smoothing=label_smoothing)
        if scheduler is not None:
            scheduler.step()

        if epoch % eval_every == 0 or epoch == epochs:
            metrics = evaluate_epoch(model, val_loader, device,
                                     max_batches=max_batches)
            acc = metrics[best_key]
            tag = f"phase{phase} epoch {epoch}/{epochs}"
            print(f"[train] {tag}: loss={loss:.4f} ce_b={ce_b:.4f} ce_c={ce_c:.4f} "
                  f"ce_buf={ce_buf:.4f} | val binary={metrics['binary_acc']:.4f} "
                  f"cattle={metrics['cattle_acc']:.4f} buffalo={metrics['buffalo_acc']:.4f} "
                  f"top1={metrics['combined_top1']:.4f}", flush=True)
            if acc >= best:
                best = acc
                sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
                torch.save({"phase": phase, "epoch": epoch, "val_top1": acc,
                            "state_dict": sd}, checkpoint_path)
            phase_pbar.set_postfix(best=f"{best:.4f}", val=f"{acc:.4f}",
                                   refresh=False)
        else:
            tag = f"phase{phase} epoch {epoch}/{epochs}"
            print(f"[train] {tag}: loss={loss:.4f} ce_b={ce_b:.4f} ce_c={ce_c:.4f} ce_buf={ce_buf:.4f} | val=skipped", flush=True)
            phase_pbar.set_postfix(loss=f"{loss:.4f}", refresh=False)

    phase_pbar.close()
    print(f"[train] phase{phase} best val top1: {best:.4f} -> {checkpoint_path}")
    return best


# ---------------------------------------------------------------------------
# Portable export — self-contained model folder
# ---------------------------------------------------------------------------

def create_portable_export(checkpoint_path, backbone, split_dir, export_dir):
    """Bundle the trained model + labels + metadata into a portable folder."""
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = os.path.join(export_dir, f"{backbone}_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # Copy checkpoint
    dst_ckpt = os.path.join(out_dir, "model.pt")
    shutil.copy2(checkpoint_path, dst_ckpt)

    # Copy class maps
    for name in ("cattle_classes.json", "buffalo_classes.json"):
        src = os.path.join(split_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))

    # Write model info
    info = {
        "backbone": backbone,
        "checkpoint": os.path.basename(checkpoint_path),
        "image_size": 260,
        "num_cattle_breeds": 57,
        "num_buffalo_breeds": 18,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "usage": (
            "Load with: model = BreedClassifier(backbone=info['backbone']); "
            "ckpt = torch.load('model.pt', map_location='cpu'); "
            "model.load_state_dict(ckpt['state_dict'])"
        ),
    }
    with open(os.path.join(out_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"[export] portable model -> {out_dir}")
    print(f"[export]   model.pt ({os.path.getsize(dst_ckpt) / 1e6:.1f} MB)")
    print(f"[export]   model_info.json + class maps")
    return out_dir


# ---------------------------------------------------------------------------
# QAT setup
# ---------------------------------------------------------------------------

def setup_qat(model, device):
    model.train()
    try:
        import torch.ao.quantization as qat
        names = [["backbone.stem.0", "backbone.stem.1"],
                 ["backbone.head.0", "backbone.head.1"]]
        for s, stage in enumerate(model.backbone.blocks):
            for r, _ in enumerate(stage):
                base = f"backbone.blocks.{s}.{r}"
                names.append([f"{base}._depthwise_conv", f"{base}._bn1"])
                names.append([f"{base}._project_conv", f"{base}._bn2"])
                if hasattr(stage[r], "_expand_conv"):
                    names.append([f"{base}._expand_conv", f"{base}._bn0"])
        qat.fuse_modules_qat(model, names, inplace=True)
        print(f"[train] QAT: fused {len(names)} conv-bn pairs")
    except Exception as exc:
        print(f"[train] QAT: module fusion skipped ({exc})")
    try:
        import torch.ao.quantization as qat
        model.qconfig = qat.get_default_qat_qconfig("x86")
        qat.prepare_qat(model, inplace=True)
        print("[train] QAT: prepare_qat applied")
        return True
    except Exception as exc:
        print(f"[train] QAT: prepare_qat failed, fine-tuning fp32 ({exc})")
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Three-phase training: warmup -> multi-task -> QAT")
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    parser.add_argument("--weights", default=None,
                        help="pretrained .pth (default: project checkpoint)")
    parser.add_argument("--attention", choices=["cbam", "se"], default="cbam")
    parser.add_argument("--data", default=RAW_DATA_DIR)
    parser.add_argument("--split-dir", default=SPLIT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-mix", action="store_true",
                        help="disable CutMix/MixUp batch mixing")
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--phase3-epochs", type=int, default=None)
    parser.add_argument("--skip-qat", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--export-dir", default=PORTABLE_EXPORT_DIR,
                        help="directory for portable model export")
    parser.add_argument("--no-export", action="store_true",
                        help="skip automatic portable export after training")
    # --- SOTA hyperparameter args ---
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="AdamW weight decay (default: 1e-2)")
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING,
                        help="label smoothing factor (default: 0.1)")
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS,
                        help="linear warmup epochs for phase 2 (default: 3)")
    parser.add_argument("--grad-accum", type=int,
                        default=GRADIENT_ACCUMULATION_STEPS,
                        help="gradient accumulation steps (default: 2)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device, use_amp = setup_device(args.device)

    # --- Data preparation ---
    print()
    if args.smoke_test:
        print("=" * 60)
        print("  SMOKE TEST MODE — mini-dataset, 1 epoch per phase")
        print("=" * 60)
        summary = prepare_smoke_splits(data_root=args.data,
                                       split_dir=args.split_dir)
    else:
        summary = prepare_splits(data_root=args.data, split_dir=args.split_dir)

    if summary is None:
        return 1

    loaders = get_dataloaders(split_dir=args.split_dir,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    if loaders is None:
        print("[train] failed to build dataloaders")
        return 1
    if args.no_mix:
        loaders[0].collate_fn = lambda batch: (
            torch.stack([b[0] for b in batch]),
            {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]})
    train_loader, val_loader, _ = loaders

    # --- Model ---
    weights = args.weights or BACKBONE_WEIGHTS[args.backbone]
    model = BreedClassifier(backbone=args.backbone, attention=args.attention,
                            pretrained_path=weights if os.path.exists(weights) else None)
    if not os.path.exists(weights):
        print(f"[train] WARNING: {weights} not found, training backbone from scratch")
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] params: trainable={trainable:,} total={total:,}")

    # --- AMP scaler ---
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if scaler:
        print(f"[train] mixed-precision: GradScaler enabled")

    # --- Phase configuration ---
    max_batches = None  # No artificial limit — smoke test uses small dataset
    base = os.path.join(CHECKPOINT_DIR, args.backbone)
    phase1 = args.phase1_epochs or (1 if args.smoke_test else PHASE1_EPOCHS)
    phase2 = args.phase2_epochs or (1 if args.smoke_test else PHASE2_EPOCHS)
    phase3 = args.phase3_epochs or (1 if args.smoke_test else PHASE3_EPOCHS)

    total_phases = 2 if args.skip_qat else 3
    print(f"\n{'=' * 60}")
    print(f"  TRAINING PLAN: {total_phases} phases, "
          f"epochs={phase1}/{phase2}" +
          (f"/{phase3}" if not args.skip_qat else ""))
    print(f"  Dataset: {summary.get('train', '?')} train / "
          f"{summary.get('val', '?')} val images")
    print(f"  Batch size: {args.batch_size}, Workers: {args.num_workers}")
    print(f"  Optimizer: AdamW (wd={args.weight_decay}), "
          f"Label smoothing: {args.label_smoothing}")
    print(f"  Gradient accumulation: {args.grad_accum} steps "
          f"(effective batch={args.batch_size * args.grad_accum})")
    if args.warmup_epochs > 0:
        print(f"  Warmup: {args.warmup_epochs} epochs (phase 2)")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # --- Phase 1: Binary head warmup ---
    model.freeze_all()
    for p in model.binary_head.parameters():
        p.requires_grad = True
    model.backbone_eval()
    print(f"[train] phase 1: backbone frozen, binary head only, "
          f"lr={PHASE1_LR:.0e}, {phase1} epochs")
    train_phase(model, train_loader, val_loader, device, 1, phase1, PHASE1_LR,
                (1.0, 0.0, 0.0), None, f"{base}_phase1_best.pt", scaler,
                max_batches, best_key="binary_acc",
                set_train=lambda m: (m.train(), m.backbone_eval()),
                weight_decay=args.weight_decay,
                grad_accum_steps=args.grad_accum,
                label_smoothing=args.label_smoothing,
                eval_every=1 if args.smoke_test else EVAL_EVERY_PHASE1)

    # --- Phase 2: Full multi-task fine-tuning ---
    model.unfreeze_all()
    model.train()
    
    compiled_model = model
    if not args.no_compile and hasattr(torch, "compile") and not args.smoke_test:
        print("[train] compiling model for phase 2 (this may take a minute)...")
        compiled_model = torch.compile(model, mode="reduce-overhead")

    warmup_ep = min(args.warmup_epochs, phase2 - 1) if not args.smoke_test else 0
    print(f"\n[train] phase 2: multi-task fine-tune, lr={PHASE2_LR:.0e} "
          f"(warmup={warmup_ep}ep + cosine), "
          f"{phase2} epochs, weights "
          f"({LOSS_WEIGHT_BINARY}/{LOSS_WEIGHT_CATTLE}/{LOSS_WEIGHT_BUFFALO})")
    train_phase(compiled_model, train_loader, val_loader, device, 2, phase2, PHASE2_LR,
                (LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO),
                lambda opt: _build_warmup_cosine_scheduler(
                    opt, warmup_ep, phase2),
                f"{base}_phase2_best.pt", scaler, max_batches,
                weight_decay=args.weight_decay,
                grad_accum_steps=args.grad_accum,
                label_smoothing=args.label_smoothing,
                eval_every=1 if args.smoke_test else EVAL_EVERY_PHASE2)

    # --- Phase 3: QAT (optional) ---
    best_checkpoint = f"{base}_phase2_best.pt"
    if not args.skip_qat:
        model.unfreeze_all()
        # QAT must run on CPU or without AMP
        qat_ok = setup_qat(model, device)
        print(f"\n[train] phase 3: quantization-aware training, "
              f"lr={PHASE3_LR:.0e}, {phase3} epochs")
        train_phase(model, train_loader, val_loader, device, 3, phase3, PHASE3_LR,
                    (LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO),
                    None, f"{base}_phase3_best.pt", None, max_batches,
                    weight_decay=args.weight_decay,
                    grad_accum_steps=args.grad_accum,
                    label_smoothing=args.label_smoothing,
                    eval_every=1 if args.smoke_test else EVAL_EVERY_PHASE3)
        if qat_ok:
            try:
                import torch.ao.quantization as qat
                model.eval()
                qat.convert(model, inplace=True)
                torch.save({"state_dict": model.state_dict()}, f"{base}_quantized.pt")
                print(f"[train] converted to INT8, saved {base}_quantized.pt")
            except Exception as exc:
                print(f"[train] INT8 conversion failed ({exc})")
        best_checkpoint = f"{base}_phase3_best.pt"

    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE in {elapsed_str}")
    print(f"  Checkpoints saved under {CHECKPOINT_DIR}")
    print(f"{'=' * 60}")

    # --- Portable export ---
    if not args.no_export and os.path.exists(best_checkpoint):
        print(f"\n[train] creating portable export...")
        create_portable_export(best_checkpoint, args.backbone,
                               args.split_dir, args.export_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())