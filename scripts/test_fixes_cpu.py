#!/usr/bin/env python
"""CPU-only synthetic tests for the 2026-09-23 fixes. No dataset/GPU needed.

Run:  python scripts/test_fixes_cpu.py
"""

import copy
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import src.config as C
from src import data_pipeline as dp
from src import train as T
from src.export import _load_model
from src.model import BreedClassifier

PASS = []
FAIL = []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


def make_labels(species, cat_idx, buf_idx, n_cattle=C.NUM_CATTLE_BREEDS,
                n_buffalo=C.NUM_BUFFALO_BREEDS):
    B = len(species)
    binary = F.one_hot(torch.tensor(species), 2).float()
    cmask = torch.tensor([1.0 if s == 0 else 0.0 for s in species])
    bmask = torch.tensor([1.0 if s == 1 else 0.0 for s in species])
    # Match the dataset: only the active species' breed head is non-zero.
    cattle = F.one_hot(torch.tensor(cat_idx), n_cattle).float() * cmask[:, None]
    buffalo = F.one_hot(torch.tensor(buf_idx), n_buffalo).float() * bmask[:, None]
    return {"binary": binary, "cattle": cattle, "buffalo": buffalo,
            "cattle_mask": cmask, "buffalo_mask": bmask}


# ---------------------------------------------------------------------------
print("\n[1] Defaults: augmentation + logit adjustment OFF")
check("LOGIT_ADJUST is False", C.LOGIT_ADJUST is False)
check("MIX_ENABLED is False", C.MIX_ENABLED is False)
# New defaults: flip + mild RRC ON (they don't mix content between breeds);
# colour jitter / RandAugment / mix stay OFF.
check("flip + rrc default ON",
      C.AUG_HORIZONTAL_FLIP and C.AUG_RANDOM_RESIZED_CROP)
check("color-jitter + randaugment default OFF",
      not C.AUG_COLOR_JITTER and not C.AUG_RANDAUGMENT)
check("mixing strength 0.25/0.4/0.2",
      (C.CUTMIX_MIXUP_PROB, C.CUTMIX_ALPHA, C.MIXUP_ALPHA) == (0.25, 0.4, 0.2))
check("BEST_METRIC is blended_score", C.BEST_METRIC == "blended_score")

# ---------------------------------------------------------------------------
print("\n[2] No-aug train transform == eval transform")
img = Image.fromarray((np.random.rand(300, 240, 3) * 255).astype(np.uint8))
NO_AUG = {"horizontal_flip": False, "random_resized_crop": False,
          "color_jitter": False, "randaugment": False}
t_train = dp._train_transform(NO_AUG)
t_eval = dp._eval_transform()
check("explicit no-aug train == eval tensor",
      torch.allclose(t_train(img), t_eval(img)))
t_aug = dp._train_transform({"horizontal_flip": True})
check("flip-enabled transform builds", t_aug is not None)
# Default (config) transform now includes RRC with the tight aspect ratio.
dt = dp.describe_transform(dp._train_transform({}))
check("default train transform has flip + RRC", "RandomHorizontalFlip" in dt
      and "RandomResizedCrop" in dt)
check("RRC ratio is near-square (0.92, 1.08)", "0.92" in dt and "1.08" in dt)

# ---------------------------------------------------------------------------
print("\n[3] Same-species mixing keeps binary one-hot + masks integral")
species = [0, 0, 0, 1, 1, 1]
labels = make_labels(species, [0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2])
imgs = torch.randn(6, 3, 16, 16)
ok_bin = ok_cat = ok_buf = ok_mask = True
for _ in range(50):
    for fn in (dp.cutmix, dp.mixup):
        li = {k: v.clone() for k, v in labels.items()}
        _, lo = fn(imgs.clone(), li, keep=None, same_species=True)
        b = lo["binary"]
        ok_bin &= bool(torch.allclose(b.sum(1), torch.ones(6)) and
                       (b.max(1).values > 0.999).all())
        ok_mask &= bool(torch.allclose(lo["cattle_mask"] + lo["buffalo_mask"],
                                       torch.ones(6)))
        # cattle samples: cattle target sums to 1, buffalo sums to 0
        cat = b.argmax(1) == 0
        ok_cat &= bool(torch.allclose(lo["cattle"][cat].sum(1),
                                      torch.ones(int(cat.sum()))))
        ok_buf &= bool(torch.allclose(lo["buffalo"][cat].sum(1),
                                      torch.zeros(int(cat.sum()))))
check("binary labels stay one-hot", ok_bin)
check("masks stay 0/1", ok_mask)
check("cattle target sums to 1 for cattle", ok_cat)
check("buffalo target is 0 for cattle", ok_buf)

# explicit pairing check: no cattle<->buffalo pair
sp = labels["binary"].argmax(1)
perm = dp._pairing_perm(labels, same_species=True)
check("pairing is within-species",
      bool((sp[perm] == sp).all()))

# ---------------------------------------------------------------------------
print("\n[4] Sampled priors flatten the long tail vs raw counts")
tmp = tempfile.mkdtemp(prefix="prior_test_")
counts = {"gir": 100, "sahiwal": 50, "rare_a": 10, "rare_b": 5}
rows = []
for b, n in counts.items():
    for i in range(n):
        rows.append({"path": f"/x/{b}/{i}.jpg", "species": "cattle",
                     "breed": b, "binary_label": 0})
import pandas as pd
pd.DataFrame(rows).to_csv(os.path.join(tmp, "train.csv"), index=False)
cat_classes = {b: i for i, b in enumerate(sorted(counts))}
json.dump(cat_classes, open(os.path.join(tmp, "cattle_classes.json"), "w"))
json.dump({}, open(os.path.join(tmp, "buffalo_classes.json"), "w"))

# pad class maps to expected size is unnecessary: _count_per_class uses map keys.
prior_sampled = dp.compute_class_priors(tmp, source="sampled")["cattle"]
prior_raw = dp.compute_class_priors(tmp, source="raw")["cattle"]
idx = [cat_classes[b] for b in counts]
ps = torch.exp(prior_sampled)[idx]
pr = torch.exp(prior_raw)[idx]
ratio_sampled = (ps.max() / ps.min()).item()
ratio_raw = (pr.max() / pr.min()).item()
check("sampled prior ratio < raw prior ratio",
      ratio_sampled < ratio_raw,
      f"sampled={ratio_sampled:.2f} raw={ratio_raw:.2f}")
check("sampled priors finite for absent classes (no NaN/inf)",
      bool(torch.isfinite(prior_sampled).all()),
      f"finite={int(torch.isfinite(prior_sampled).sum())}/{prior_sampled.numel()}")

# ---------------------------------------------------------------------------
print("\n[5] EMA warm-up schedule + BN-buffer sync")
check("decay at step 0 = 0.1", abs(T._ema_decay_at(0, 0.999, True) - 0.1) < 1e-9)
d = [T._ema_decay_at(t, 0.999, True) for t in (0, 10, 100, 1000, 100000)]
check("decay monotonic + capped", all(d[i] <= d[i+1] for i in range(len(d)-1))
      and d[-1] <= 0.999)
check("no-warmup returns full decay",
      T._ema_decay_at(0, 0.999, False) == 0.999)

tiny = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
ema = copy.deepcopy(tiny)
for p in ema.parameters():
    p.data.zero_()
init_mean = ema[1].running_mean.clone()
tiny.train()
tiny(torch.randn(16, 4) * 3.0 + 1.0)
T._apply_ema(ema, tiny, 0.5)
lin_ok = torch.allclose(ema[0].weight, 0.5 * tiny[0].weight)
bn_ok = torch.allclose(ema[1].running_mean,
                       0.5 * init_mean + 0.5 * tiny[1].running_mean)
check("EMA params = decay*old + (1-decay)*new", lin_ok)
check("EMA BN running stats synced", bn_ok)

# ---------------------------------------------------------------------------
print("\n[6] Mixing turns off in the last MIX_OFF_LAST_FRAC of epochs")
mo = T._mix_off_epoch(80, 0.15)
check("80 epochs, 0.15 -> mix off after epoch 68", mo == 68)
check("last 15% has no mixing", all(not (e <= mo) for e in range(69, 81)))
check("frac=0 -> mixing all epochs", T._mix_off_epoch(80, 0.0) == 80)

# ---------------------------------------------------------------------------
print("\n[7] Export _load_model: missing-key behaviour")
sd = BreedClassifier("lite2").state_dict()
bad = {k: v for k, v in sd.items() if k != "binary_head.0.weight"}
p_bad = os.path.join(tmp, "bad.pt")
torch.save({"state_dict": bad}, p_bad)
raised = False
try:
    _load_model(p_bad, "lite2", "cbam")
except RuntimeError:
    raised = True
check("raises on missing required key", raised)

only_proj_missing = {k: v for k, v in sd.items()
                     if not k.startswith("projection_head.")}
p_proj = os.path.join(tmp, "proj.pt")
torch.save({"state_dict": only_proj_missing}, p_proj)
ok = True
try:
    _load_model(p_proj, "lite2", "cbam")
except Exception:
    ok = False
check("tolerates missing projection_head", ok)

# ---------------------------------------------------------------------------
print("\n[8] masked_loss finite + backward with mixing/SupCon (logit adj off)")
# Duplicate breeds so SupCon has at least one positive per anchor.
labels8 = make_labels([0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 0, 0, 1, 1],
                      [0, 0, 1, 1, 0, 0, 1, 1])
imgs8 = torch.randn(8, 3, 16, 16)
labels_mixed = {k: v.clone() for k, v in labels8.items()}
dp.mixup(imgs8.clone(), labels_mixed, same_species=True)
B = 8
out = {
    "binary": torch.randn(B, 2, requires_grad=True),
    "cattle": torch.randn(B, C.NUM_CATTLE_BREEDS, requires_grad=True),
    "buffalo": torch.randn(B, C.NUM_BUFFALO_BREEDS, requires_grad=True),
    "embedding": torch.randn(B, C.PROJECTION_DIM, requires_grad=True),
}
total, cb, cc, cbuf = T.masked_loss(
    out, labels_mixed, 0.15, 0.50, 0.35, label_smoothing=0.05,
    logit_priors=None, adjust_tau=0.0,
    contrastive_weight=0.2, contrastive_temp=0.1, mixed=False)
finite = torch.isfinite(total) and torch.isfinite(cb) and torch.isfinite(cc) and torch.isfinite(cbuf)
total.backward()
grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in out.values())
check("masked_loss finite (logit adj OFF, SupCon ON)", bool(finite))
check("all gradients (incl. embedding) finite", bool(grads_ok))

# SupCon skipped on mixed batches still finite
out2 = {k: v.detach().clone().requires_grad_(True) for k, v in out.items()}
tot2, *_ = T.masked_loss(out2, labels_mixed, 0.15, 0.50, 0.35,
                         contrastive_weight=0.2, mixed=True)
check("masked_loss finite when mixed=True", bool(torch.isfinite(tot2)))

# logit-adjustment path (only used when explicitly enabled) stays finite
priors = {"cattle": torch.log(torch.ones(C.NUM_CATTLE_BREEDS) / C.NUM_CATTLE_BREEDS),
          "buffalo": torch.log(torch.ones(C.NUM_BUFFALO_BREEDS) / C.NUM_BUFFALO_BREEDS)}
tot3, *_ = T.masked_loss(out2, labels_mixed, 0.15, 0.50, 0.35,
                         logit_priors=priors, adjust_tau=1.0,
                         contrastive_weight=0.0, mixed=True)
check("masked_loss finite with logit adjustment", bool(torch.isfinite(tot3)))

# ---------------------------------------------------------------------------
print("\n[9] Timestamped outputs never overwrite")
import re as _re
from src.run_utils import make_run_id, timestamped, unique_path
rid = make_run_id()
check("run id format DD-MM-YYYY-HH-MM",
      bool(_re.fullmatch(r"\d{2}-\d{2}-\d{4}-\d{2}-\d{2}", rid)), rid)
p1 = timestamped("outputs/checkpoints/lite2_phase2_best.pt", "01-01-2026-00-00")
p2 = timestamped("outputs/checkpoints/lite2_phase2_best.pt", "02-01-2026-00-00")
check("timestamped paths differ", p1 != p2 and "01-01-2026-00-00" in p1)

# unique_path never overwrites an existing artifact
import tempfile as _tf
d = _tf.mkdtemp(prefix="unique_")
f = os.path.join(d, "model.pt")
open(f, "w").close()
f2 = unique_path(f)
check("unique_path avoids overwrite", f2 != f and f2.endswith("_2.pt"))

# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL CPU TESTS PASSED")
