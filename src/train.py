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

from .config import (AUG_COLOR_JITTER, AUG_HORIZONTAL_FLIP, AUG_RANDAUGMENT,
                     AUG_RANDOM_RESIZED_CROP,
                     BACKBONE_WEIGHTS, BALANCE_BINARY_HEAD, BEST_METRIC,
                     BINARY_SATURATION_ACC, BATCH_SIZE, CHECKPOINT_DIR,
                     CONTRASTIVE_HARD_NEG_WEIGHT, CONTRASTIVE_TEMPERATURE,
                     CONTRASTIVE_WEIGHT, COSINE_HEAD, COSINE_MARGIN,
                     COSINE_MARGIN_RAMP_EPOCHS, EMA_DECAY,
                     EMA_WARMUP, EMA_WARN_FRAC,
                     EVAL_EVERY_PHASE1, EVAL_EVERY_PHASE2, EVAL_EVERY_PHASE3,
                     EVAL_MATCH_TRAIN_RESOLUTION,
                     KD_ALPHA, KD_TEMPERATURE, LOGIT_ADJUST,
                     LOGIT_ADJUST_PRIOR, LOGIT_ADJUST_TAU,
                     GRADIENT_ACCUMULATION_STEPS, LABEL_SMOOTHING,
                     LOSS_WEIGHT_BINARY, LOSS_WEIGHT_BINARY_FINAL,
                     LOSS_WEIGHT_BUFFALO, LOSS_WEIGHT_BUFFALO_FINAL,
                     LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_CATTLE_FINAL, NUM_WORKERS,
                     MIX_OFF_LAST_FRAC, MIX_SAME_SPECIES,
                     PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR,
                     PHASE3_EPOCHS, PHASE3_LR, PORTABLE_EXPORT_DIR, RAW_DATA_DIR,
                     FEATURE_DIM, RARE_CLASS_THRESHOLD, SAMPLER_BETA, SEED,
                     SHOT_FEW_MAX, SHOT_MEDIUM_MAX, SPLIT_DIR, TRAIT_FILE,
                     TRAIT_WEIGHT, WARMUP_EPOCHS,
                     WEIGHT_DECAY, BACKBONE_LR_MULT, CUTMIX_MIXUP_PROB)
from .data_pipeline import (compute_class_counts, compute_class_priors,
                            compute_rare_classes, get_dataloaders,
                            prepare_half_splits, prepare_quarter_splits,
                            prepare_smoke_splits, prepare_splits)
from .metrics import evaluate_epoch
from .model import BreedClassifier
from .run_logger import finish_run, init_run_logger, log_event, log_metrics
from .run_utils import make_run_id, timestamped, unique_path


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

def soft_ce(pred, target, label_smoothing=0.0, logit_prior=None, tau=0.0):
    """Soft cross-entropy with optional label smoothing and logit adjustment.

    When label_smoothing > 0, target distribution is smoothed:
        target_smooth = (1 - ε) * target + ε / num_classes
    This prevents overconfident predictions and improves generalization.

    When logit_prior (log class prior) is given, ``tau * log_prior`` is added
    to the logits (Menon et al., ICLR 2021). Frequent classes must then be much
    more confident to win, which compensates for the long tail without forcing
    a near-uniform sampler; the shift is absorbed into the learned biases so
    inference stays raw.
    """
    if logit_prior is not None and tau:
        pred = pred + tau * logit_prior.to(pred.device).unsqueeze(0)
    if label_smoothing > 0.0:
        n_classes = pred.size(1)
        target = (1.0 - label_smoothing) * target + label_smoothing / n_classes
    return -(target * F.log_softmax(pred, dim=1)).sum(dim=1)


# Cosine/ArcFace head state (set by main()/train_phase; read by masked_loss).
_COSINE_SCALE = None
_COSINE_MARGIN = 0.0


def _apply_cosine_margin(logits, target, margin, scale):
    """Add the additive angular margin to the TARGET class logits.

    Only applied to (near-)one-hot targets; mixed/soft targets are left as-is
    (mixing is off by default). Forward/export never apply the margin.
    """
    if margin <= 0 or scale is None:
        return logits
    cos = (logits / scale).clamp(-1 + 1e-7, 1 - 1e-7)
    sin = torch.sqrt(1.0 - cos * cos)
    idx = target.argmax(1)
    hard = target.max(1).values > 0.9
    if not hard.any():
        return logits
    m = torch.as_tensor(float(margin), device=logits.device)
    new_cos = cos * torch.cos(m) - sin * torch.sin(m)
    out = logits.clone()
    rows = torch.arange(logits.size(0), device=logits.device)[hard]
    out[rows, idx[hard]] = scale * new_cos[rows, idx[hard]]
    return out


def supervised_contrastive_loss(embedding, class_ids, temperature=0.1,
                                hard_pairs=None, hard_neg_weight=1.0):
    """SupCon (Khosla et al., 2020) over the auxiliary projection embedding.

    Pulls same-breed embeddings together and pushes others apart. Anchors with
    no positive in the batch are ignored. ``hard_pairs`` (a set of frozenset of
    global class ids) up-weights those negatives by ``hard_neg_weight`` so the
    model is pushed hardest on the breeds it confuses.
    """
    if embedding.size(0) < 2:
        return embedding.new_zeros(())
    z = F.normalize(embedding, dim=1)
    sim = (z @ z.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

    self_mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    pos_mask = (class_ids.unsqueeze(0) == class_ids.unsqueeze(1)) & ~self_mask
    logits_mask = ~self_mask
    exp_sim = torch.exp(sim) * logits_mask
    if hard_pairs and hard_neg_weight != 1.0:
        ci = class_ids.unsqueeze(0)
        cj = class_ids.unsqueeze(1)
        hard = torch.zeros_like(exp_sim, dtype=torch.bool)
        for pair in hard_pairs:
            a, b = tuple(pair)
            hard |= ((ci == a) & (cj == b)) | ((ci == b) & (cj == a))
        hard &= logits_mask
        exp_sim = torch.where(hard, exp_sim * hard_neg_weight, exp_sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return z.new_zeros(())
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)
    return -mean_log_prob_pos[valid].mean()


def _combined_class_ids(labels):
    """Global class id: cattle 0..C-1, buffalo C..C+B-1."""
    cattle_idx = labels["cattle"].argmax(1)
    buffalo_idx = labels["buffalo"].argmax(1)
    offset = labels["cattle"].size(1)
    is_cattle = labels["cattle_mask"] > 0.5
    return torch.where(is_cattle, cattle_idx, offset + buffalo_idx)


# Confusion-driven hard pairs (set by main() from --hard-pairs); read by the
# contrastive term. Module-level to avoid threading through every loss call.
_HARD_PAIRS = None
_HARD_NEG_WEIGHT = CONTRASTIVE_HARD_NEG_WEIGHT


def _contrastive_term(out, labels, weight, temperature, mixed):
    if weight <= 0.0 or mixed or "embedding" not in out:
        return out["binary"].new_zeros(())
    return weight * supervised_contrastive_loss(
        out["embedding"], _combined_class_ids(labels), temperature,
        hard_pairs=_HARD_PAIRS, hard_neg_weight=_HARD_NEG_WEIGHT)


def masked_loss(out, labels, w_binary, w_cattle, w_buffalo,
                label_smoothing=0.0, logit_priors=None, adjust_tau=0.0,
                contrastive_weight=0.0, contrastive_temp=CONTRASTIVE_TEMPERATURE,
                mixed=False):
    """Multi-task loss with species-masked breed CE and logit adjustment.

    The binary term is optionally re-weighted per batch so cattle and
    buffalo contribute equally regardless of how the weighted sampler mixed
    the batch (the per-breed sampler leaves a ~57:18 species prior).
    """
    cattle_prior = buffalo_prior = None
    if logit_priors is not None:
        cattle_prior = logit_priors.get("cattle")
        buffalo_prior = logit_priors.get("buffalo")

    ce_binary = soft_ce(out["binary"], labels["binary"], label_smoothing)
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
    cattle_logits = out["cattle"]
    buffalo_logits = out["buffalo"]
    if _COSINE_SCALE is not None and _COSINE_MARGIN > 0:
        cattle_logits = _apply_cosine_margin(cattle_logits, labels["cattle"],
                                             _COSINE_MARGIN, _COSINE_SCALE)
        buffalo_logits = _apply_cosine_margin(buffalo_logits, labels["buffalo"],
                                              _COSINE_MARGIN, _COSINE_SCALE)
    ce_cattle = (soft_ce(cattle_logits, labels["cattle"], label_smoothing,
                         cattle_prior, adjust_tau) * labels["cattle_mask"])
    ce_buffalo = (soft_ce(buffalo_logits, labels["buffalo"], label_smoothing,
                          buffalo_prior, adjust_tau) * labels["buffalo_mask"])
    denom_c = labels["cattle_mask"].sum().clamp(min=1.0)
    denom_b = labels["buffalo_mask"].sum().clamp(min=1.0)
    ce_cattle = ce_cattle.sum() / denom_c
    ce_buffalo = ce_buffalo.sum() / denom_b
    total = (w_binary * ce_binary + w_cattle * ce_cattle + w_buffalo * ce_buffalo)
    total = total + _contrastive_term(out, labels, contrastive_weight,
                                      contrastive_temp, mixed)
    return total, ce_binary, ce_cattle, ce_buffalo


def masked_kd_loss(out, teacher_out, labels, w_binary, w_cattle, w_buffalo,
                   kd_alpha=KD_ALPHA, kd_temp=KD_TEMPERATURE,
                   label_smoothing=0.0, logit_priors=None, adjust_tau=0.0,
                   contrastive_weight=0.0,
                   contrastive_temp=CONTRASTIVE_TEMPERATURE, mixed=False):
    """Multi-task loss blended with teacher distillation (Hinton et al.).

    total = (1 - kd_alpha) * masked hard-label CE
          + kd_alpha * T^2 * masked KL(teacher || student)
          + contrastive_weight * SupCon(student embeddings)

    The teacher runs frozen on the same augmented batch, so the student
    inherits the teacher's fine-grained discrimination without any extra
    on-device cost. Works with CutMix/MixUp batches (the hard-label part
    handles the soft targets; the teacher sees the same mixed images).
    """
    hard_total, ce_b, ce_c, ce_buf = masked_loss(
        out, labels, w_binary, w_cattle, w_buffalo,
        label_smoothing=label_smoothing, logit_priors=logit_priors,
        adjust_tau=adjust_tau, contrastive_weight=0.0,
        contrastive_temp=contrastive_temp, mixed=mixed)
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
    total = total + _contrastive_term(out, labels, contrastive_weight,
                                      contrastive_temp, mixed)
    return total, ce_b, ce_c, ce_buf


def _compute_loss(model, images, labels, loss_weights, label_smoothing,
                  teacher_model=None, kd_alpha=KD_ALPHA,
                  kd_temp=KD_TEMPERATURE, logit_priors=None, adjust_tau=0.0,
                  contrastive_weight=0.0,
                  contrastive_temp=CONTRASTIVE_TEMPERATURE, mixed=False,
                  trait_module=None, trait_targets=None, trait_weight=0.0,
                  trait_stats=None):
    """Forward pass + loss, with optional knowledge distillation + trait heads."""
    out = model(images)
    if teacher_model is not None:
        teacher_out = teacher_model(images)
        result = masked_kd_loss(out, teacher_out, labels, *loss_weights,
                                kd_alpha=kd_alpha, kd_temp=kd_temp,
                                label_smoothing=label_smoothing,
                                logit_priors=logit_priors, adjust_tau=adjust_tau,
                                contrastive_weight=contrastive_weight,
                                contrastive_temp=contrastive_temp, mixed=mixed)
    else:
        result = masked_loss(out, labels, *loss_weights,
                             label_smoothing=label_smoothing,
                             logit_priors=logit_priors, adjust_tau=adjust_tau,
                             contrastive_weight=contrastive_weight,
                             contrastive_temp=contrastive_temp, mixed=mixed)
    if trait_module is not None and trait_weight > 0 and "features" in out:
        cls = _combined_class_ids(labels)
        tloss, taccs = trait_module.loss(out["features"], cls, trait_targets,
                                         label_smoothing=label_smoothing)
        total, ce_b, ce_c, ce_buf = result
        result = (total + trait_weight * tloss, ce_b, ce_c, ce_buf)
        if trait_stats is not None:
            trait_stats.update(taccs)
    return result


# ---------------------------------------------------------------------------
# Training core
# ---------------------------------------------------------------------------

def _rare_keep_mask(labels, rare_masks):
    """Per-sample bool: True = this sample's breed is rare (do not mix)."""
    if rare_masks is None:
        return None
    cattle_rare = rare_masks.get("cattle")
    buffalo_rare = rare_masks.get("buffalo")
    if cattle_rare is None or buffalo_rare is None:
        return None
    cattle_idx = labels["cattle"].argmax(1)
    buffalo_idx = labels["buffalo"].argmax(1)
    is_cattle = labels["cattle_mask"] > 0.5
    return torch.where(is_cattle, cattle_rare[cattle_idx], buffalo_rare[buffalo_idx])


def _mix_off_epoch(epochs, mix_off_frac):
    """Last epoch that still mixes; mixing is OFF for epochs after this.

    ``mix_off_frac`` of the phase (rounded) has no mixing at the end.
    """
    if mix_off_frac and mix_off_frac > 0:
        return max(1, epochs - int(round(epochs * mix_off_frac)))
    return epochs


def _ema_decay_at(step, ema_decay, warmup=EMA_WARMUP):
    """EMA decay for optimizer step ``step`` (1-based), with warm-up.

    ``decay_t = min(EMA_DECAY, (1+t)/(10+t))`` so early steps track the raw
    weights closely and the EMA only reaches full decay after many steps.
    """
    if warmup:
        return min(ema_decay, (1.0 + step) / (10.0 + step))
    return ema_decay


def _apply_ema(ema_model, model, decay):
    """One EMA update: parameters AND floating buffers (BN running stats)."""
    with torch.no_grad():
        # torch.compile wraps the module in OptimizedModule; EMA the original
        # module's tensors (same iteration order either way).
        src = getattr(model, "_orig_mod", model)
        for ema_param, param in zip(ema_model.parameters(), src.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
        for ema_buf, buf in zip(ema_model.buffers(), src.buffers()):
            if buf.dtype.is_floating_point:
                ema_buf.data.mul_(decay).add_(buf.data, alpha=1 - decay)
            else:
                ema_buf.data.copy_(buf.data)


def run_epoch(model, loader, optimizer, device, loss_weights, scaler=None,
              max_batches=None, set_train=None, desc="train",
              training=False, mix_stats=None,
              grad_accum_steps=1, label_smoothing=0.0, ema_model=None,
              ema_decay=EMA_DECAY, ema_warmup=EMA_WARMUP, ema_state=None,
              teacher_model=None, kd_alpha=KD_ALPHA,
              kd_temp=KD_TEMPERATURE, mix_prob=CUTMIX_MIXUP_PROB,
              same_species=MIX_SAME_SPECIES,
              rare_masks=None, logit_priors=None, adjust_tau=0.0,
              contrastive_weight=0.0,
              contrastive_temp=CONTRASTIVE_TEMPERATURE,
              trait_module=None, trait_targets=None, trait_weight=0.0,
              trait_stats=None):
    """Run one training epoch with optional AMP, gradient accumulation,
    label smoothing, logit adjustment, contrastive features, EMA (parameters
    *and* BatchNorm buffers, once per optimizer step), and teacher distillation.

    ``training`` must be True only for actual training epochs; batch mixing is
    gated on it (never on the tqdm description string). ``mix_stats`` (a dict)
    is updated with batch counts so callers can report whether mixing actually
    ran.
    """
    set_train = set_train or (lambda m: m.train())
    set_train(model)
    running = []
    total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    use_amp = scaler is not None and device.type == "cuda"
    ema_state = ema_state if ema_state is not None else {"step": 0}

    def _ema_update():
        """One EMA update per OPTIMIZER step (called after optimizer.step)."""
        if ema_model is None:
            return
        ema_state["step"] += 1
        decay = _ema_decay_at(ema_state["step"], ema_decay, ema_warmup)
        _apply_ema(ema_model, model, decay)

    pbar = tqdm(loader, desc=desc, total=total, leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

        if mix_stats is not None:
            mix_stats["batches"] = mix_stats.get("batches", 0) + 1

        mixed = False
        if training and mix_prob > 0 and len(images) > 1 \
                and random.random() < mix_prob:
            from .data_pipeline import cutmix, mixup
            keep = _rare_keep_mask(labels, rare_masks)
            if random.random() < 0.5:
                images, labels = cutmix(images, labels, keep=keep,
                                        same_species=same_species)
            else:
                images, labels = mixup(images, labels, keep=keep,
                                       same_species=same_species)
            mixed = True
            if mix_stats is not None:
                mix_stats["mixed_batches"] = mix_stats.get("mixed_batches", 0) + 1

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss, ce_b, ce_c, ce_buf = _compute_loss(
                    model, images, labels, loss_weights, label_smoothing,
                    teacher_model=teacher_model, kd_alpha=kd_alpha,
                    kd_temp=kd_temp, logit_priors=logit_priors,
                    adjust_tau=adjust_tau, contrastive_weight=contrastive_weight,
                    contrastive_temp=contrastive_temp, mixed=mixed,
                    trait_module=trait_module, trait_targets=trait_targets,
                    trait_weight=trait_weight, trait_stats=trait_stats)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                _ema_update()
        else:
            loss, ce_b, ce_c, ce_buf = _compute_loss(
                model, images, labels, loss_weights, label_smoothing,
                teacher_model=teacher_model, kd_alpha=kd_alpha,
                kd_temp=kd_temp, logit_priors=logit_priors,
                adjust_tau=adjust_tau, contrastive_weight=contrastive_weight,
                contrastive_temp=contrastive_temp, mixed=mixed,
                trait_module=trait_module, trait_targets=trait_targets,
                trait_weight=trait_weight, trait_stats=trait_stats)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _ema_update()

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

    def get_warmup_lambda(group_idx):
        group_name = optimizer.param_groups[group_idx].get("name", "")
        if "head" in group_name:
            return lambda epoch: 1.0  # Heads are already warm from Phase 1
        return lambda epoch: (epoch + 1) / warmup_epochs

    warmup_sched = LambdaLR(
        optimizer, lr_lambda=[get_warmup_lambda(i) for i in range(len(optimizer.param_groups))])
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs))
    return SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs])


def _fmt_metrics(tag, m):
    return (f"{tag}: bin={m['binary_acc']:.3f} cat={m['cattle_acc']:.3f} "
            f"buf={m['buffalo_acc']:.3f} top1={m['combined_top1']:.3f} "
            f"top1s={m['combined_top1_soft']:.3f} "
            f"mF1={0.5*(m['cattle_macro_f1']+m['buffalo_macro_f1']):.3f} "
            f"blend={m.get('blended_score', 0.0):.3f} "
            f"few/med/many="
            f"{m.get('acc_fewshot', 0.0):.2f}/"
            f"{m.get('acc_mediumshot', 0.0):.2f}/"
            f"{m.get('acc_manyshot', 0.0):.2f} "
            f"ent={m.get('pred_hist_entropy', 0.0):.3f}")


def train_phase(model, loader, val_loader, device, phase, epochs, lr,
                loss_weights, scheduler_factory, checkpoint_path,
                scaler=None, max_batches=None, best_key=BEST_METRIC,
                set_train=None, weight_decay=0.0,
                grad_accum_steps=1, label_smoothing=0.0, eval_every=1,
                ema_model=None, teacher_model=None, kd_alpha=KD_ALPHA,
                kd_temp=KD_TEMPERATURE, loss_weights_final=None,
                binary_sat_acc=None, mix_prob=CUTMIX_MIXUP_PROB,
                mix_off_frac=MIX_OFF_LAST_FRAC,
                same_species=MIX_SAME_SPECIES, ema_decay=EMA_DECAY,
                ema_warmup=EMA_WARMUP,
                train_counts=None, few_max=SHOT_FEW_MAX,
                medium_max=SHOT_MEDIUM_MAX,
                rare_masks=None, logit_priors=None, adjust_tau=0.0,
                contrastive_weight=0.0,
                contrastive_temp=CONTRASTIVE_TEMPERATURE,
                trait_module=None, trait_targets=None, trait_weight=0.0):
    """Train a single phase with AdamW, optional AMP, gradient accumulation,
    and label smoothing."""
    raw_model = getattr(model, '_orig_mod', model)
    if hasattr(raw_model, 'backbone') and phase == 2:
        param_groups = [
            {"name": "backbone", "params": raw_model.backbone.parameters(),
             "lr": lr * BACKBONE_LR_MULT},
            {"name": "attention", "params": raw_model.attention.parameters(),
             "lr": lr * 0.5},
            {"name": "binary_head", "params": raw_model.binary_head.parameters(),
             "lr": lr},
            {"name": "cattle_head", "params": raw_model.cattle_head.parameters(),
             "lr": lr},
            {"name": "buffalo_head", "params": raw_model.buffalo_head.parameters(),
             "lr": lr},
        ]
        if hasattr(raw_model, "projection_head"):
            param_groups.append(
                {"name": "projection_head",
                 "params": raw_model.projection_head.parameters(), "lr": lr})
        if hasattr(raw_model, "trait_heads"):
            param_groups.append(
                {"name": "trait_heads",
                 "params": raw_model.trait_heads.parameters(), "lr": lr})
        for group in param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)

    scheduler = scheduler_factory(optimizer) if scheduler_factory else None

    # --- log the per-group learning rates at phase start ---
    try:
        groups_info = [{"name": g.get("name", f"group{i}"), "lr": g["lr"],
                        "params": sum(p.numel() for p in g["params"])}
                       for i, g in enumerate(optimizer.param_groups)]
        print(f"[train] phase {phase} param groups: " + ", ".join(
            f"{g['name']}(lr={g['lr']:.1e}, n={g['params']:,})"
            for g in groups_info))
        log_event("param_groups", category="training", phase=phase,
                  groups=groups_info)
    except Exception:
        pass
    best = -1.0
    current_weights = tuple(loss_weights)

    # Mixing is disabled for the last `mix_off_frac` of the phase.
    mix_off_epoch = _mix_off_epoch(epochs, mix_off_frac)

    # --- EMA diagnostics (steps are OPTIMIZER steps now) ---
    ema_state = {"step": 0}
    if ema_model is not None:
        steps_per_epoch = max(1, len(loader))
        opt_steps_per_epoch = max(1, math.ceil(steps_per_epoch / grad_accum_steps))
        total_opt_steps = opt_steps_per_epoch * epochs
        time_const = (1.0 / (1.0 - ema_decay)) if ema_decay < 1.0 else float("inf")
        frac = (time_const / total_opt_steps) if total_opt_steps else 0.0
        print(f"[train] EMA: {steps_per_epoch} batches/epoch, "
              f"{opt_steps_per_epoch} optimizer steps/epoch, "
              f"{total_opt_steps} total; decay={ema_decay}, warmup={ema_warmup}, "
              f"time-constant≈{time_const:.0f} opt steps ({frac*100:.1f}% of phase)",
              flush=True)
        if frac > EMA_WARN_FRAC:
            print(f"[train] WARNING: EMA time constant is {frac*100:.1f}% of "
                  f"phase {phase} steps (> {EMA_WARN_FRAC*100:.0f}%); the EMA "
                  f"checkpoint will lag. Consider lowering EMA_DECAY or "
                  f"lengthening the phase.", flush=True)

    phase_pbar = tqdm(range(1, epochs + 1), desc=f"Phase {phase}",
                      unit="epoch", leave=True,
                      bar_format="{l_bar}{bar:20}{r_bar}")
    global _COSINE_MARGIN
    for epoch in phase_pbar:
        eff_mix_prob = mix_prob if epoch <= mix_off_epoch else 0.0
        # ArcFace margin ramps over the first N phase-2 epochs, 0 elsewhere.
        if _COSINE_SCALE is not None:
            if phase == 2:
                ramp = max(1, COSINE_MARGIN_RAMP_EPOCHS)
                _COSINE_MARGIN = COSINE_MARGIN * min(1.0, epoch / ramp)
            else:
                _COSINE_MARGIN = 0.0
        mix_stats = {"batches": 0, "mixed_batches": 0}
        trait_stats = {}
        loss, ce_b, ce_c, ce_buf = run_epoch(
            model, loader, optimizer, device, current_weights, scaler,
            max_batches, set_train=set_train,
            desc=f"phase{phase} e{epoch}/{epochs}", training=True,
            mix_stats=mix_stats,
            grad_accum_steps=grad_accum_steps,
            label_smoothing=label_smoothing, ema_model=ema_model,
            ema_decay=ema_decay, ema_warmup=ema_warmup, ema_state=ema_state,
            teacher_model=teacher_model, kd_alpha=kd_alpha, kd_temp=kd_temp,
            mix_prob=eff_mix_prob, same_species=same_species,
            rare_masks=rare_masks,
            logit_priors=logit_priors, adjust_tau=adjust_tau,
            contrastive_weight=contrastive_weight,
            contrastive_temp=contrastive_temp,
            trait_module=trait_module, trait_targets=trait_targets,
            trait_weight=trait_weight, trait_stats=trait_stats)
        if scheduler is not None:
            scheduler.step()

        if epoch % eval_every == 0 or epoch == epochs:
            raw_metrics = evaluate_epoch(
                model, val_loader, device, max_batches=max_batches,
                train_counts=train_counts, few_max=few_max,
                medium_max=medium_max)
            ema_metrics = None
            if ema_model is not None:
                ema_metrics = evaluate_epoch(
                    ema_model, val_loader, device, max_batches=max_batches,
                    train_counts=train_counts, few_max=few_max,
                    medium_max=medium_max)

            log_metrics(raw_metrics, epoch=epoch, tag="raw", phase=phase,
                        mix_prob=eff_mix_prob, loss=loss, ce_b=ce_b, ce_c=ce_c,
                        ce_buf=ce_buf, ema_step=ema_state.get("step", 0))
            if ema_metrics is not None:
                log_metrics(ema_metrics, epoch=epoch, tag="ema", phase=phase)

            if trait_module is not None and trait_weight > 0:
                from .traits import evaluate_traits
                val_traits = evaluate_traits(model, trait_module, val_loader,
                                             trait_targets, device, max_batches)
                log_event("trait_accuracy", category="training", epoch=epoch,
                          phase=phase,
                          train={k: v for k, v in trait_stats.items()},
                          val=val_traits)
                shown = ", ".join(f"{k}={v:.3f}" for k, v in val_traits.items()
                                  if v is not None)
                print(f"[train] phase{phase} epoch {epoch}/{epochs}: "
                      f"trait val acc: {shown}", flush=True)

            raw_acc = raw_metrics.get(best_key, 0.0)
            ema_acc = ema_metrics.get(best_key, 0.0) if ema_metrics else -1.0
            if ema_metrics is not None and ema_acc >= raw_acc:
                chosen, chosen_acc, chosen_src = ema_metrics, ema_acc, "ema"
            else:
                chosen, chosen_acc, chosen_src = raw_metrics, raw_acc, "raw"

            tag = f"phase{phase} epoch {epoch}/{epochs}"
            mix_note = (f"mix=on ({mix_stats['mixed_batches']}/"
                        f"{mix_stats['batches']} batches)"
                        if mix_stats["mixed_batches"] > 0 else "mix=OFF")
            print(f"[train] {tag}: loss={loss:.4f} ce_b={ce_b:.4f} "
                  f"ce_c={ce_c:.4f} ce_buf={ce_buf:.4f} "
                  f"{mix_note} | "
                  f"{_fmt_metrics('raw', raw_metrics)}", flush=True)
            if ema_metrics is not None:
                print(f"[train] {tag}: {_fmt_metrics('ema', ema_metrics)} "
                      f"-> best={chosen_src} ({best_key}={chosen_acc:.4f})",
                      flush=True)
            else:
                print(f"[train] {tag}: best={chosen_src} "
                      f"({best_key}={chosen_acc:.4f})", flush=True)

            if chosen_acc >= best:
                best = chosen_acc
                eval_mdl = ema_model if chosen_src == "ema" else model
                sd = eval_mdl._orig_mod.state_dict() if hasattr(eval_mdl, "_orig_mod") else eval_mdl.state_dict()
                torch.save({"phase": phase, "epoch": epoch,
                            "val_top1": chosen_acc, "best_metric": best_key,
                            "source": chosen_src, "metrics": chosen,
                            "state_dict": sd}, checkpoint_path)
                log_event("checkpoint_saved", category="training",
                          path=checkpoint_path, phase=phase, epoch=epoch,
                          source=chosen_src, best_metric=best_key, best=chosen_acc)
            # Once the binary head saturates, stop spending loss budget on it
            # and reallocate to the breed heads from the next epoch onward.
            if (loss_weights_final is not None and binary_sat_acc is not None
                    and raw_metrics["binary_acc"] >= binary_sat_acc
                    and tuple(current_weights) != tuple(loss_weights_final)):
                current_weights = tuple(loss_weights_final)
                print(f"[train] binary head saturated "
                      f"({raw_metrics['binary_acc']:.4f} >= {binary_sat_acc}); "
                      f"loss weights -> {current_weights}", flush=True)
            phase_pbar.set_postfix(best=f"{best:.4f}", val=f"{chosen_acc:.4f}",
                                   refresh=False)
        else:
            tag = f"phase{phase} epoch {epoch}/{epochs}"
            print(f"[train] {tag}: loss={loss:.4f} ce_b={ce_b:.4f} ce_c={ce_c:.4f} ce_buf={ce_buf:.4f} | val=skipped", flush=True)
            phase_pbar.set_postfix(loss=f"{loss:.4f}", refresh=False)

    phase_pbar.close()
    print(f"[train] phase{phase} best val {best_key}: {best:.4f} -> {checkpoint_path}")
    return best


# ---------------------------------------------------------------------------
# Portable export — self-contained model folder
# ---------------------------------------------------------------------------

def create_portable_export(checkpoint_path, backbone, split_dir, export_dir):
    """Bundle the trained model + labels + metadata into a portable folder."""
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = unique_path(os.path.join(export_dir, f"{backbone}_{tag}"))
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
        description="Two-phase training: all-heads warmup -> multi-task "
                    "fine-tune (optional QAT phase 3 via --include-qat)")
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    parser.add_argument("--weights", default=None,
                        help="pretrained .pth (default: project checkpoint)")
    parser.add_argument("--attention", choices=["cbam", "se"], default="cbam")
    parser.add_argument("--data", default=RAW_DATA_DIR)
    parser.add_argument("--split-dir", default=SPLIT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-augment", action="store_true",
                        help="force ALL augmentation off (flip + rrc included)")
    parser.add_argument("--mix", action="store_true",
                        help="enable CutMix/MixUp batch mixing (OFF by default)")
    parser.add_argument("--no-mix", action="store_true",
                        help="force-disable CutMix/MixUp (default; kept for compat)")
    parser.add_argument("--flip", action="store_true",
                        help="enable RandomHorizontalFlip (OFF by default)")
    parser.add_argument("--color-jitter", action="store_true",
                        help="enable ColorJitter (OFF by default)")
    parser.add_argument("--randaugment", action="store_true",
                        help="enable RandAugment (OFF by default)")
    parser.add_argument("--rrc", action="store_true",
                        help="enable RandomResizedCrop (OFF by default)")
    parser.add_argument("--augment-all", action="store_true",
                        help="enable flip + color-jitter + randaugment + rrc + mix")
    parser.add_argument("--augment-preset", choices=["none", "light"],
                        default="none",
                        help="'light' = flip + mild RRC + mild colour jitter "
                             "(no RandAugment, no mix). Default: none")
    parser.add_argument("--breed-aug", action="store_true",
                        help="apply per-breed augmentation overrides "
                             "(BREED_AUG_POLICY; coat-colour breeds skip jitter)")
    parser.add_argument("--allow-hue", action="store_true",
                        help="allow the stronger ColorJitter hue cap (> 0.02)")
    parser.add_argument("--pad-to-square", action="store_true",
                        help="resize the long side and pad to square for train "
                             "AND eval (keeps full-body side profiles)")
    parser.add_argument("--run-tag", default=None,
                        help="run id used to timestamp outputs "
                             "(default: current time DD-MM-YYYY-HH-MM)")
    parser.add_argument("--exec-id", default=None,
                        help="execution-log folder name under logs/ "
                             "(default: auto YYYYmmdd-HHMMSS-xxxx)")
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
                        help="label smoothing factor (default: 0.05)")
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS,
                        help="linear warmup epochs for phase 2 (default: 3)")
    parser.add_argument("--contrastive-weight", type=float,
                        default=CONTRASTIVE_WEIGHT,
                        help="auxiliary SupCon loss weight on pooled features "
                             "(0 disables; default: 0.2)")
    parser.add_argument("--trait-weight", type=float, default=TRAIT_WEIGHT,
                        help="auxiliary breed-trait head loss weight "
                             "(0 disables; suggested 0.1)")
    parser.add_argument("--trait-file", default=TRAIT_FILE,
                        help="breed traits JSON (see scripts/make_trait_template.py)")
    parser.add_argument("--hard-pairs", default=None,
                        help="confusion_pairs.json (scripts/mine_confusions.py); "
                             "up-weights confused-pair SupCon negatives")
    parser.add_argument("--cosine-head", action="store_true",
                        help="normalised scaled-cosine breed heads (ArcFace); "
                             "margin is applied only in the loss, export stays "
                             "plain cosine logits")
    parser.add_argument("--logit-adjust", action="store_true",
                        help="enable logit adjustment (OFF by default: the "
                             "effective-number sampler is the single long-tail "
                             "mechanism; enabling both double-corrects and "
                             "over-predicts rare breeds)")
    parser.add_argument("--logit-adjust-prior", choices=["sampled", "raw"],
                        default=LOGIT_ADJUST_PRIOR,
                        help="prior for logit adjustment: 'sampled' uses the "
                             "effective sampled distribution (default), 'raw' "
                             "uses raw train counts (old double-correcting "
                             "behaviour)")
    parser.add_argument("--no-logit-adjust", action="store_true",
                        help="force-disable logit adjustment (kept for compat)")
    parser.add_argument("--rare-threshold", type=int, default=RARE_CLASS_THRESHOLD,
                        help="breeds below this many train images are excluded "
                             "from CutMix/MixUp (default: 30)")
    parser.add_argument("--dedup-splits", action="store_true",
                        help="group-aware splits: near-duplicate images "
                             "(dHash Hamming<=4) never cross train/val/test")
    parser.add_argument("--grad-accum", type=int,
                        default=GRADIENT_ACCUMULATION_STEPS,
                        help="gradient accumulation steps (default: 2)")
    args = parser.parse_args()

    # --- execution logger (per-run folder under logs/<exec_id>/) ---
    init_run_logger(exec_id=getattr(args, "exec_id", None), module="src.train")
    log_event("cli_args", category="actions", **vars(args))

    # --- confusion-driven hard pairs (optional) ---
    global _HARD_PAIRS
    if args.hard_pairs:
        try:
            with open(args.hard_pairs) as f:
                hp = json.load(f)
            pairs = hp.get("pairs", hp) if isinstance(hp, dict) else hp
            _HARD_PAIRS = {frozenset(int(x) for x in p) for p in pairs}
            print(f"[train] hard pairs: {len(_HARD_PAIRS)} confused pairs "
                  f"from {args.hard_pairs} (SupCon hard-neg weight="
                  f"{_HARD_NEG_WEIGHT})")
        except Exception as exc:
            print(f"[train] could not load --hard-pairs ({exc}); ignoring")

    if args.smoke_test and args.half_data:
        parser.error("--smoke-test and --half-data are mutually exclusive")
    if getattr(args, 'quarter_data', False) and args.smoke_test:
        parser.error("--smoke-test and --quarter-data are mutually exclusive")

    # --- Run id + augmentation switches (all OFF unless flagged) ---
    run_id = make_run_id(args.run_tag)
    if args.augment_all:
        args.mix = True
    light = args.augment_preset == "light"
    # Config defaults (flip + rrc on) are OR-ed with the CLI flags; --no-augment
    # forces everything off.
    augment = {
        "horizontal_flip": (AUG_HORIZONTAL_FLIP or args.flip
                            or args.augment_all or light),
        "color_jitter": (AUG_COLOR_JITTER or args.color_jitter
                         or args.augment_all or light),
        "randaugment": AUG_RANDAUGMENT or args.randaugment or args.augment_all,
        "random_resized_crop": (AUG_RANDOM_RESIZED_CROP or args.rrc
                                or args.augment_all or light),
    }
    if args.no_augment:
        augment = {k: False for k in augment}
        args.mix = False
    pad = bool(args.pad_to_square)
    print(f"[train] run id: {run_id}")
    print(f"[train] augmentation: " +
          (", ".join(k for k, v in augment.items() if v) or "NONE (default)")
          + (f" (preset={args.augment_preset})" if light else "")
          + (", breed-aware" if args.breed_aug else "")
          + (", pad-to-square" if pad else ""))

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
        summary = prepare_splits(data_root=args.data, split_dir=args.split_dir,
                                 dedup=args.dedup_splits)

    if summary is None:
        return 1

    loaders = get_dataloaders(split_dir=args.split_dir,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"),
                              augment=augment,
                              breed_augment=args.breed_aug, pad=pad)
    if loaders is None:
        print("[train] failed to build dataloaders")
        return 1
    train_loader, val_loader, _ = loaders

    # --- Log the exact transform chain (C6) ---
    from .data_pipeline import describe_transform, _eval_transform, _train_transform
    print(f"[train] train transform: "
          f"{describe_transform(_train_transform(augment, pad=pad))}")
    print(f"[train] eval  transform: "
          f"{describe_transform(_eval_transform(pad=pad))}")
    if augment.get("random_resized_crop") and not pad \
            and not EVAL_MATCH_TRAIN_RESOLUTION:
        print("[train] WARNING: RRC is on (train Resize(288)) but eval uses "
              "Resize(260); set EVAL_MATCH_TRAIN_RESOLUTION=True or "
              "--pad-to-square to align scales.")

    # --- Imbalance handling: EXACTLY ONE long-tail mechanism ---
    # The effective-number sampler (used in get_dataloaders) already rebalances
    # every batch. Logit adjustment is OFF by default; enabling it on top of the
    # sampler double-corrects and over-predicts rare breeds at inference.
    logit_on = (args.logit_adjust or LOGIT_ADJUST) and not args.no_logit_adjust
    logit_priors = None
    adjust_tau = 0.0
    print(f"[train] imbalance mechanism: effective-number sampler "
          f"(SAMPLER_BETA={SAMPLER_BETA})" +
          ("" if not logit_on else " + logit adjustment"))
    if logit_on:
        logit_priors = compute_class_priors(args.split_dir,
                                            source=args.logit_adjust_prior)
        if logit_priors is not None:
            adjust_tau = LOGIT_ADJUST_TAU
            logit_priors = {k: v.to(device) for k, v in logit_priors.items()}
            ratios = []
            for species, lp in logit_priors.items():
                p = torch.exp(lp)
                ratio = (p.max() / p.min()).item()
                ratios.append(f"{species}={ratio:.1f}x")
            print(f"[train] logit adjustment ON: tau={adjust_tau}, "
                  f"prior={args.logit_adjust_prior}, "
                  f"max/min prior ratio {' '.join(ratios)}")
        else:
            print("[train] logit adjustment requested but priors unavailable; "
                  "continuing without it")
    else:
        print("[train] logit adjustment OFF (recommended: sampler is the "
              "single long-tail mechanism)")
    rare_masks = compute_rare_classes(args.split_dir, args.rare_threshold)
    if rare_masks is not None:
        rare_masks = {k: v.to(device) for k, v in rare_masks.items()}
        n_rare = sum(int(v.sum()) for v in rare_masks.values())
        print(f"[train] rare breeds (<{args.rare_threshold} train imgs): {n_rare} "
              f"(excluded from CutMix/MixUp)")
    train_counts = compute_class_counts(args.split_dir)
    # CutMix/MixUp are OFF by default and only run when --mix is passed.
    mix_prob = CUTMIX_MIXUP_PROB if (args.mix and not args.no_mix) else 0.0
    contrastive_weight = args.contrastive_weight
    print(f"[train] mixing: p={mix_prob}"
          + ("" if mix_prob > 0 else " (OFF by default; enable with --mix)")
          + f", same_species={MIX_SAME_SPECIES}, "
          f"off_last_frac={MIX_OFF_LAST_FRAC}")

    # --- Model ---
    weights = args.weights or BACKBONE_WEIGHTS[args.backbone]
    model = BreedClassifier(backbone=args.backbone, attention=args.attention,
                            pretrained_path=weights if os.path.exists(weights) else None,
                            cosine_head=args.cosine_head)
    if not os.path.exists(weights):
        print(f"[train] WARNING: {weights} not found, training backbone from scratch")
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    global _COSINE_SCALE
    if args.cosine_head:
        _COSINE_SCALE = model.cosine_scale
        print(f"[train] cosine/ArcFace breed heads ON (scale={model.cosine_scale}, "
              f"margin={COSINE_MARGIN}, ramp={COSINE_MARGIN_RAMP_EPOCHS} epochs)")

    # --- Teacher (knowledge distillation) ---
    teacher_model = None
    if args.teacher:
        if not os.path.exists(args.teacher):
            print(f"[train] teacher checkpoint not found: {args.teacher}")
            return 1
        teacher_attention = args.teacher_attention or args.attention
        tckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
        t_state = tckpt["state_dict"] if isinstance(tckpt, dict) and "state_dict" in tckpt else tckpt
        
        # Detect binary_dim from teacher checkpoint
        t_binary_dim = t_state["binary_head.0.weight"].shape[0] if "binary_head.0.weight" in t_state else 512
        
        teacher_model = BreedClassifier(backbone=args.teacher_backbone,
                                        attention=teacher_attention,
                                        binary_dim=t_binary_dim)
        teacher_model.load_state_dict(t_state)
        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print(f"[train] KD: teacher {args.teacher_backbone}+{teacher_attention} "
              f"loaded from {args.teacher} "
              f"(alpha={KD_ALPHA}, T={KD_TEMPERATURE})")

    # --- Breed trait auxiliary heads (training-only; excluded from export) ---
    trait_module = None
    trait_targets = None
    if args.trait_weight and args.trait_weight > 0:
        from .traits import (TraitClassifier, breed_trait_targets,
                             build_trait_vocab, load_trait_spec)
        spec = load_trait_spec(args.trait_file)
        vocab = build_trait_vocab(spec) if spec else {}
        if not vocab:
            print(f"[train] trait heads requested but no filled values found in "
                  f"{args.trait_file}; run scripts/make_trait_template.py and "
                  f"fill it in. Continuing without trait heads.")
        else:
            with open(os.path.join(args.split_dir, "cattle_classes.json")) as f:
                cattle_classes = json.load(f)
            with open(os.path.join(args.split_dir, "buffalo_classes.json")) as f:
                buffalo_classes = json.load(f)
            trait_targets = breed_trait_targets(spec, vocab, cattle_classes,
                                                buffalo_classes)
            trait_module = TraitClassifier(FEATURE_DIM, vocab)
            model.trait_heads = trait_module  # registers submodule (in state_dict)
            model.to(device)
            print(f"[train] trait heads ON (weight={args.trait_weight}): "
                  + ", ".join(f"{k}({len(v)})" for k, v in vocab.items()))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] params: trainable={trainable:,} total={total:,}")

    # --- AMP scaler ---
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if scaler:
        print(f"[train] mixed-precision: GradScaler enabled")

    # --- Phase configuration (timestamped, never overwrites previous runs) ---
    max_batches = None  # No artificial limit — smoke test uses small dataset
    base = os.path.join(CHECKPOINT_DIR, args.backbone)
    ckpt_p1 = unique_path(timestamped(f"{base}_phase1_best.pt", run_id))
    ckpt_p2 = unique_path(timestamped(f"{base}_phase2_best.pt", run_id))
    ckpt_p3 = unique_path(timestamped(f"{base}_phase3_best.pt", run_id))
    ckpt_quant = unique_path(timestamped(f"{base}_quantized.pt", run_id))
    phase1 = args.phase1_epochs or (1 if args.smoke_test else PHASE1_EPOCHS)
    phase2 = args.phase2_epochs or (1 if args.smoke_test else PHASE2_EPOCHS)
    phase3 = args.phase3_epochs or (1 if args.smoke_test else PHASE3_EPOCHS)

    run_qat = args.include_qat and not args.skip_qat
    total_phases = 3 if run_qat else 2
    print(f"\n{'=' * 60}")
    print(f"  TRAINING PLAN: {total_phases} phases, "
          f"epochs={phase1}/{phase2}" +
          (f"/{phase3}" if run_qat else ""))
    print(f"  Run id: {run_id}")
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

    log_event("train_plan", category="training", run_id=run_id,
              backbone=args.backbone, attention=args.attention,
              phase1=phase1, phase2=phase2, phase3=phase3, run_qat=run_qat,
              batch_size=args.batch_size, grad_accum=args.grad_accum,
              augment=augment, pad=pad, mix_prob=mix_prob,
              logit_adjust=logit_on, contrastive_weight=contrastive_weight,
              loss_weights=[LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE,
                            LOSS_WEIGHT_BUFFALO],
              train_images=summary.get("train"), val_images=summary.get("val"),
              test_images=summary.get("test"))

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
                (LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO), None, ckpt_p1, scaler,
                max_batches, best_key=BEST_METRIC,
                set_train=lambda m: (m.train(), m.backbone_eval()),
                weight_decay=args.weight_decay,
                grad_accum_steps=args.grad_accum,
                label_smoothing=args.label_smoothing,
                eval_every=1 if args.smoke_test else EVAL_EVERY_PHASE1,
                mix_prob=mix_prob, mix_off_frac=MIX_OFF_LAST_FRAC,
                same_species=MIX_SAME_SPECIES, train_counts=train_counts,
                rare_masks=rare_masks,
                logit_priors=logit_priors, adjust_tau=adjust_tau,
                contrastive_weight=0.0,
                trait_module=trait_module, trait_targets=trait_targets,
                trait_weight=args.trait_weight)

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
          f"({LOSS_WEIGHT_BINARY}/{LOSS_WEIGHT_CATTLE}/{LOSS_WEIGHT_BUFFALO}) "
          f"-> saturated "
          f"({LOSS_WEIGHT_BINARY_FINAL}/{LOSS_WEIGHT_CATTLE_FINAL}/"
          f"{LOSS_WEIGHT_BUFFALO_FINAL})")
    if contrastive_weight:
        print(f"[train] feature learning: SupCon weight={contrastive_weight}, "
              f"T={CONTRASTIVE_TEMPERATURE}")

    import copy
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    train_phase(compiled_model, train_loader, val_loader, device, 2, phase2, PHASE2_LR,
                (LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO),
                lambda opt: _build_warmup_cosine_scheduler(
                    opt, warmup_ep, phase2),
                ckpt_p2, scaler, max_batches,
                weight_decay=args.weight_decay,
                grad_accum_steps=args.grad_accum,
                label_smoothing=args.label_smoothing,
                eval_every=1 if args.smoke_test else EVAL_EVERY_PHASE2,
                ema_model=ema_model, teacher_model=teacher_model,
                loss_weights_final=(LOSS_WEIGHT_BINARY_FINAL,
                                    LOSS_WEIGHT_CATTLE_FINAL,
                                    LOSS_WEIGHT_BUFFALO_FINAL),
                binary_sat_acc=BINARY_SATURATION_ACC,
                mix_prob=mix_prob, mix_off_frac=MIX_OFF_LAST_FRAC,
                same_species=MIX_SAME_SPECIES, train_counts=train_counts,
                rare_masks=rare_masks,
                logit_priors=logit_priors, adjust_tau=adjust_tau,
                contrastive_weight=contrastive_weight,
                contrastive_temp=CONTRASTIVE_TEMPERATURE,
                trait_module=trait_module, trait_targets=trait_targets,
                trait_weight=args.trait_weight)

    # --- Phase 3: QAT (opt-in; mobile INT8 is produced by converter PTQ) ---
    best_checkpoint = ckpt_p2
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
                    None, ckpt_p3, None, max_batches,
                    weight_decay=args.weight_decay,
                    grad_accum_steps=args.grad_accum,
                    label_smoothing=args.label_smoothing,
                    eval_every=1 if args.smoke_test else EVAL_EVERY_PHASE3,
                    teacher_model=teacher_model,
                    mix_prob=mix_prob, mix_off_frac=MIX_OFF_LAST_FRAC,
                    same_species=MIX_SAME_SPECIES, train_counts=train_counts,
                    rare_masks=rare_masks,
                    logit_priors=logit_priors, adjust_tau=adjust_tau,
                    contrastive_weight=0.0,
                    trait_module=trait_module, trait_targets=trait_targets,
                    trait_weight=args.trait_weight)
        if qat_ok:
            try:
                import torch.ao.quantization as qat
                model.eval()
                qat.convert(model, inplace=True)
                torch.save({"state_dict": model.state_dict(),
                            "quantized": True}, ckpt_quant)
                print(f"[train] converted to INT8, saved {ckpt_quant}")
            except Exception as exc:
                print(f"[train] INT8 conversion failed ({exc})")
        best_checkpoint = ckpt_p2  # Export phase 2 by default to preserve accuracy

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
        log_event("portable_export", category="export", path=best_checkpoint)

    finish_run(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())