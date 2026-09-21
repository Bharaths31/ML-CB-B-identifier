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

from .config import (BACKBONE_WEIGHTS, BALANCE_BINARY_HEAD, BATCH_SIZE,
                     CHECKPOINT_DIR, EMA_DECAY, EVAL_EVERY_PHASE1,
                     EVAL_EVERY_PHASE2, EVAL_EVERY_PHASE3, KD_ALPHA,
                     KD_TEMPERATURE, GRADIENT_ACCUMULATION_STEPS,
                     LABEL_SMOOTHING, LOSS_WEIGHT_BINARY, LOSS_WEIGHT_BUFFALO,
                     LOSS_WEIGHT_CATTLE, NUM_WORKERS, PHASE1_EPOCHS, PHASE1_LR,
                     PHASE2_EPOCHS, PHASE2_LR, PHASE3_EPOCHS, PHASE3_LR,
                     PORTABLE_EXPORT_DIR, RAW_DATA_DIR, SEED, SPLIT_DIR,
                     WARMUP_EPOCHS, WEIGHT_DECAY, BACKBONE_LR_MULT,
                     CUTMIX_MIXUP_PROB)
from .data_pipeline import (get_dataloaders, prepare_half_splits,
                            prepare_quarter_splits, prepare_smoke_splits,
                            prepare_splits)
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
    """Multi-task loss with species-masked breed CE.

    The binary term is optionally re-weighted per batch so cattle and
    buffalo contribute equally regardless of how the weighted sampler mixed
    the batch (the per-breed sampler leaves a ~57:18 species prior).
    """
    ce_binary = soft_ce(out["binary"], labels["binary"],
                        label_smoothing)
    if BALANCE_BINARY_HEAD:
        p_buffalo = labels["binary"][:, 1].clamp(0.0, 1.0)
        mean_buf = p_buffalo.mean().clamp(min=1e-4)
        mean_cat = (1.0 - p_buffalo).mean().clamp(min=1e-4)
        # Each species' CE mass is normalised to 0.5 of the batch total.
        w_cat = 0.5 / mean_cat
        w_buf = 0.5 / mean_buf
        sample_w = w_cat * (1.0 - p_buffalo) + w_buf * p_buffalo
        ce_binary = (ce_binary * sample_w).sum() / sample_w.sum()
    else:
        ce_binary = ce_binary.mean()
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


def masked_kd_loss(out, teacher_out, labels, w_binary, w_cattle, w_buffalo,
                   kd_alpha=KD_ALPHA, kd_temp=KD_TEMPERATURE,
                   label_smoothing=0.0):
    """Multi-task loss blended with teacher distillation (Hinton et al.).

    total = (1 - kd_alpha) * masked hard-label CE
          + kd_alpha * T^2 * masked KL(teacher || student)

    The teacher runs frozen on the same augmented batch, so the student
    inherits the teacher's fine-grained discrimination without any extra
    on-device cost. Works with CutMix/MixUp batches (the hard-label part
    handles the soft targets; the teacher sees the same mixed images).
    """
    hard_total, ce_b, ce_c, ce_buf = masked_loss(
        out, labels, w_binary, w_cattle, w_buffalo,
        label_smoothing=label_smoothing)
    t = kd_temp

    def _kl(student_logits, teacher_logits):
        return F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.log_softmax(teacher_logits / t, dim=1),
            reduction="none", log_target=True).sum(dim=1)

    kl_binary = _kl(out["binary"], teacher_out["binary"])
    kl_cattle = _kl(out["cattle"], teacher_out["cattle"])
    kl_buffalo = _kl(out["buffalo"], teacher_out["buffalo"])

    denom_c = labels["cattle_mask"].sum().clamp(min=1.0)
    denom_b = labels["buffalo_mask"].sum().clamp(min=1.0)
    kd = w_binary * kl_binary.mean() \
        + w_cattle * (kl_cattle * labels["cattle_mask"]).sum() / denom_c \
        + w_buffalo * (kl_buffalo * labels["buffalo_mask"]).sum() / denom_b
    kd = kd * (t * t)

    total = (1.0 - kd_alpha) * hard_total + kd_alpha * kd
    return total, ce_b, ce_c, ce_buf


def _compute_loss(model, images, labels, loss_weights, label_smoothing,
                  teacher_model=None, kd_alpha=KD_ALPHA,
                  kd_temp=KD_TEMPERATURE):
    """Forward pass + loss, with optional knowledge distillation."""
    out = model(images)
    if teacher_model is not None:
        teacher_out = teacher_model(images)
        return masked_kd_loss(out, teacher_out, labels, *loss_weights,
                              kd_alpha=kd_alpha, kd_temp=kd_temp,
                              label_smoothing=label_smoothing)
    return masked_loss(out, labels, *loss_weights,
                       label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# Training core
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, loss_weights, scaler=None,
              max_batches=None, set_train=None, desc="train",
              grad_accum_steps=1, label_smoothing=0.0, ema_model=None,
              ema_decay=EMA_DECAY, teacher_model=None, kd_alpha=KD_ALPHA,
              kd_temp=KD_TEMPERATURE):
    """Run one training epoch with optional AMP, gradient accumulation,
    label smoothing, EMA (parameters *and* BatchNorm buffers), and teacher
    distillation."""
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

        if desc.startswith("train") and len(images) > 1 and random.random() < CUTMIX_MIXUP_PROB:
            from .data_pipeline import cutmix, mixup
            if random.random() < 0.5:
                images, labels = cutmix(images, labels)
            else:
                images, labels = mixup(images, labels)

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss, ce_b, ce_c, ce_buf = _compute_loss(
                    model, images, labels, loss_weights, label_smoothing,
                    teacher_model=teacher_model, kd_alpha=kd_alpha,
                    kd_temp=kd_temp)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss, ce_b, ce_c, ce_buf = _compute_loss(
                model, images, labels, loss_weights, label_smoothing,
                teacher_model=teacher_model, kd_alpha=kd_alpha,
                kd_temp=kd_temp)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if ema_model is not None:
            with torch.no_grad():
                # torch.compile wraps the module in OptimizedModule; EMA the
                # original module's tensors (same iteration order either way).
                src = getattr(model, "_orig_mod", model)
                for ema_param, param in zip(ema_model.parameters(),
                                            src.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data,
                                                        alpha=1 - ema_decay)
                # BatchNorm running stats live in buffers and must track the
                # EMA too — a stale-BN EMA silently corrupts both validation
                # metrics and every checkpoint exported from it.
                for ema_buf, buf in zip(ema_model.buffers(), src.buffers()):
                    if buf.dtype.is_floating_point:
                        ema_buf.data.mul_(ema_decay).add_(buf.data,
                                                          alpha=1 - ema_decay)
                    else:
                        ema_buf.data.copy_(buf.data)

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
                grad_accum_steps=1, label_smoothing=0.0, eval_every=1,
                ema_model=None, teacher_model=None, kd_alpha=KD_ALPHA,
                kd_temp=KD_TEMPERATURE):
    """Train a single phase with AdamW, optional AMP, gradient accumulation,
    and label smoothing."""
    raw_model = getattr(model, '_orig_mod', model)
    if hasattr(raw_model, 'backbone') and phase == 2:
        param_groups = [
            {"params": raw_model.backbone.parameters(), "lr": lr * BACKBONE_LR_MULT},
            {"params": raw_model.attention.parameters(), "lr": lr * 0.5},
            {"params": raw_model.binary_head.parameters(), "lr": lr},
            {"params": raw_model.cattle_head.parameters(), "lr": lr},
            {"params": raw_model.buffalo_head.parameters(), "lr": lr},
        ]
        for group in param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
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
            label_smoothing=label_smoothing, ema_model=ema_model,
            teacher_model=teacher_model, kd_alpha=kd_alpha, kd_temp=kd_temp)
        if scheduler is not None:
            scheduler.step()

        if epoch % eval_every == 0 or epoch == epochs:
            metrics = evaluate_epoch(ema_model if ema_model else model, val_loader, device,
                                     max_batches=max_batches)
            acc = metrics[best_key]
            tag = f"phase{phase} epoch {epoch}/{epochs}"
            print(f"[train] {tag}: loss={loss:.4f} ce_b={ce_b:.4f} ce_c={ce_c:.4f} "
                  f"ce_buf={ce_buf:.4f} | val binary={metrics['binary_acc']:.4f} "
                  f"cattle={metrics['cattle_acc']:.4f} buffalo={metrics['buffalo_acc']:.4f} "
                  f"top1={metrics['combined_top1']:.4f}", flush=True)
            if acc >= best:
                best = acc
                eval_mdl = ema_model if ema_model else model
                sd = eval_mdl._orig_mod.state_dict() if hasattr(eval_mdl, "_orig_mod") else eval_mdl.state_dict()
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
        # Per-tensor observers (MinMax) convert cleanly everywhere; the x86
        # default per-channel-affine scheme broke torch.ao conversion on this
        # architecture ("Unsupported qscheme: per_channel_affine").
        model.qconfig = qat.QConfig(
            activation=qat.default_observer,
            weight=qat.default_weight_observer,
        )
        qat.prepare_qat(model, inplace=True)
        print("[train] QAT: prepare_qat applied (per-tensor observers)")
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
    parser.add_argument("--include-qat", action="store_true",
                        help="enable the optional QAT phase 3 (OFF by default: "
                             "mobile INT8 now comes from converter-side PTQ, "
                             "see src/export.py --mode tflite / onnx-int8)")
    parser.add_argument("--skip-qat", action="store_true",
                        help="kept for backwards compatibility; QAT is "
                             "already off by default")
    parser.add_argument("--teacher", default=None,
                        help="teacher checkpoint (.pt) for knowledge "
                             "distillation, e.g. outputs/checkpoints/"
                             "lite4_phase2_best.pt")
    parser.add_argument("--teacher-backbone", choices=["lite2", "lite4"],
                        default="lite4")
    parser.add_argument("--teacher-attention", choices=["cbam", "se"],
                        default=None,
                        help="teacher attention (default: same as student)")
    # Data mode (mutually exclusive)
    data_mode_group = parser.add_mutually_exclusive_group()
    data_mode_group.add_argument("--smoke-test", action="store_true")
    data_mode_group.add_argument("--half-data", action="store_true",
                        help="use 50%% of images per breed for faster training")
    data_mode_group.add_argument("--quarter-data", action="store_true",
                        help="use 25%% of images per breed for very fast training")
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

    if args.smoke_test and args.half_data:
        parser.error("--smoke-test and --half-data are mutually exclusive")
    if getattr(args, 'quarter_data', False) and args.smoke_test:
        parser.error("--smoke-test and --quarter-data are mutually exclusive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device, use_amp = setup_device(args.device)

    # --- Auto-scale batch size for VRAM constraints ---
    if device.type == "cuda":
        total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"[train] Detected GPU with {total_memory_gb:.1f} GB VRAM")
        
        target_effective_batch = args.batch_size * args.grad_accum
        
        if total_memory_gb < 6.0:
            args.batch_size = 16  # Fits on 4GB cards like RTX 3050
        elif total_memory_gb < 10.0:
            args.batch_size = 32  # Fits on 8GB cards like RTX 3070
        elif total_memory_gb < 16.0:
            args.batch_size = 64  # Fits on 12-15GB cards like T4
        else:
            args.batch_size = 128 # 24GB cards like RTX 3090/4090
            
        args.grad_accum = max(1, target_effective_batch // args.batch_size)
        print(f"[train] Auto-scaled: batch_size={args.batch_size}, grad_accum={args.grad_accum}")


    # --- Data preparation ---
    print()
    if args.smoke_test:
        print("=" * 60)
        print("  SMOKE TEST MODE — mini-dataset, 1 epoch per phase")
        print("=" * 60)
        summary = prepare_smoke_splits(data_root=args.data,
                                       split_dir=args.split_dir)
    elif args.half_data:
        print("=" * 60)
        print("  HALF-DATA MODE — using 50% of images per breed")
        print("=" * 60)
        summary = prepare_half_splits(data_root=args.data,
                                      split_dir=args.split_dir)
    elif getattr(args, 'quarter_data', False):
        print("=" * 60)
        print("  QUARTER-DATA MODE — using 25% of images per breed")
        print("=" * 60)
        summary = prepare_quarter_splits(data_root=args.data,
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

    # --- Teacher (knowledge distillation) ---
    teacher_model = None
    if args.teacher:
        if not os.path.exists(args.teacher):
            print(f"[train] teacher checkpoint not found: {args.teacher}")
            return 1
        teacher_attention = args.teacher_attention or args.attention
        teacher_model = BreedClassifier(backbone=args.teacher_backbone,
                                        attention=teacher_attention)
        tckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
        teacher_model.load_state_dict(
            tckpt["state_dict"] if isinstance(tckpt, dict) and "state_dict" in tckpt
            else tckpt)
        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print(f"[train] KD: teacher {args.teacher_backbone}+{teacher_attention} "
              f"loaded from {args.teacher} "
              f"(alpha={KD_ALPHA}, T={KD_TEMPERATURE})")

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

    run_qat = args.include_qat and not args.skip_qat
    total_phases = 3 if run_qat else 2
    print(f"\n{'=' * 60}")
    print(f"  TRAINING PLAN: {total_phases} phases, "
          f"epochs={phase1}/{phase2}" +
          (f"/{phase3}" if run_qat else ""))
    print(f"  Dataset: {summary.get('train', '?')} train / "
          f"{summary.get('val', '?')} val images")
    print(f"  Batch size: {args.batch_size}, Workers: {args.num_workers}")
    print(f"  Optimizer: AdamW (wd={args.weight_decay}), "
          f"Label smoothing: {args.label_smoothing}")
    print(f"  Gradient accumulation: {args.grad_accum} steps "
          f"(effective batch={args.batch_size * args.grad_accum})")
    if args.warmup_epochs > 0:
        print(f"  Warmup: {args.warmup_epochs} epochs (phase 2)")
    if teacher_model is not None:
        print(f"  Distillation: alpha={KD_ALPHA}, T={KD_TEMPERATURE} "
              f"(teacher={args.teacher_backbone})")
    if run_qat:
        print("  QAT: enabled (phase 3)")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # --- Phase 1: All heads warmup ---
    model.freeze_all()
    for p in model.binary_head.parameters():
        p.requires_grad = True
    for p in model.cattle_head.parameters():
        p.requires_grad = True
    for p in model.buffalo_head.parameters():
        p.requires_grad = True
    model.backbone_eval()
    print(f"[train] phase 1: backbone frozen, all heads warm up, "
          f"lr={PHASE1_LR:.0e}, {phase1} epochs")
    train_phase(model, train_loader, val_loader, device, 1, phase1, PHASE1_LR,
                (LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO), None, f"{base}_phase1_best.pt", scaler,
                max_batches, best_key="combined_top1",
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
        if os.name == 'nt':
            print("[train] Windows detected: disabling torch.compile (falling back to eager)")
        else:
            print("[train] compiling model for phase 2 (this may take a minute)...")
            compiled_model = torch.compile(model)

    warmup_ep = min(args.warmup_epochs, phase2 - 1) if not args.smoke_test else 0
    print(f"\n[train] phase 2: multi-task fine-tune, lr={PHASE2_LR:.0e} "
          f"(warmup={warmup_ep}ep + cosine), "
          f"{phase2} epochs, weights "
          f"({LOSS_WEIGHT_BINARY}/{LOSS_WEIGHT_CATTLE}/{LOSS_WEIGHT_BUFFALO})")
          
    import copy
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False
        
    train_phase(compiled_model, train_loader, val_loader, device, 2, phase2, PHASE2_LR,
                (LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO),
                lambda opt: _build_warmup_cosine_scheduler(
                    opt, warmup_ep, phase2),
                f"{base}_phase2_best.pt", scaler, max_batches,
                weight_decay=args.weight_decay,
                grad_accum_steps=args.grad_accum,
                label_smoothing=args.label_smoothing,
                eval_every=1 if args.smoke_test else EVAL_EVERY_PHASE2,
                ema_model=ema_model, teacher_model=teacher_model)

    # --- Phase 3: QAT (opt-in; mobile INT8 is produced by converter PTQ) ---
    best_checkpoint = f"{base}_phase2_best.pt"
    if run_qat:
        # Start from the best EMA phase-2 weights rather than the final-epoch
        # weights — quantization noise is much easier to recover from there.
        if os.path.exists(best_checkpoint):
            bckpt = torch.load(best_checkpoint, map_location=device,
                               weights_only=False)
            model.load_state_dict(bckpt["state_dict"])
            print(f"[train] QAT: starting from best phase-2 checkpoint "
                  f"(val_top1={bckpt.get('val_top1', float('nan')):.4f})")
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
                    eval_every=1 if args.smoke_test else EVAL_EVERY_PHASE3,
                    teacher_model=teacher_model)
        if qat_ok:
            try:
                import torch.ao.quantization as qat
                model.eval()
                qat.convert(model, inplace=True)
                torch.save({"state_dict": model.state_dict(),
                            "quantized": True}, f"{base}_quantized.pt")
                print(f"[train] converted to INT8, saved {base}_quantized.pt")
            except Exception as exc:
                print(f"[train] INT8 conversion failed ({exc})")
        best_checkpoint = f"{base}_phase2_best.pt"  # Export phase 2 by default to preserve accuracy

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