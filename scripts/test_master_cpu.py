#!/usr/bin/env python
"""CPU-only synthetic tests for the master-task fixes. No GPU/dataset needed.

Run:  python scripts/test_master_cpu.py
"""

import copy
import json
import os
import subprocess
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
from src.cbam import CBAM, SEBlock
from src.run_utils import (make_run_id, resolve_checkpoint, sanitize_run_id,
                           timestamped, unique_path)

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


def synth_labels(species, cat, buf):
    B = len(species)
    binary = F.one_hot(torch.tensor(species), 2).float()
    cmask = torch.tensor([1.0 if s == 0 else 0.0 for s in species])
    bmask = torch.tensor([1.0 if s == 1 else 0.0 for s in species])
    cattle = F.one_hot(torch.tensor(cat), C.NUM_CATTLE_BREEDS).float() * cmask[:, None]
    buffalo = F.one_hot(torch.tensor(buf), C.NUM_BUFFALO_BREEDS).float() * bmask[:, None]
    return {"binary": binary, "cattle": cattle, "buffalo": buffalo,
            "cattle_mask": cmask, "buffalo_mask": bmask}


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.size(0)
        return {"binary": torch.zeros(B, 2) + self.p,
                "cattle": torch.zeros(B, C.NUM_CATTLE_BREEDS) + self.p,
                "buffalo": torch.zeros(B, C.NUM_BUFFALO_BREEDS) + self.p,
                "embedding": torch.zeros(B, C.PROJECTION_DIM) + self.p}


class _DS(torch.utils.data.Dataset):
    def __init__(self, n=8):
        self.imgs = torch.randn(n, 3, 32, 32)
        self.species = [i % 2 for i in range(n)]
        self.labels = synth_labels(self.species,
                                   [i % C.NUM_CATTLE_BREEDS for i in range(n)],
                                   [i % C.NUM_BUFFALO_BREEDS for i in range(n)])

    def __len__(self):
        return len(self.species)

    def __getitem__(self, i):
        return self.imgs[i], {k: v[i] for k, v in self.labels.items()}


# ---------------------------------------------------------------------------
print("\n[A1] run_utils: sanitized run ids, timestamped paths, unique_path")
check("run id sanitized (illegal chars -> '-')",
      sanitize_run_id('V3:run/1?') == "V3-run-1", sanitize_run_id('V3:run/1?'))
check("run id strips trailing dots/spaces", sanitize_run_id("exp1. ") == "exp1")
check("make_run_id keeps a clean tag", make_run_id("V3") == "V3")
p = timestamped("outputs/checkpoints/lite2_phase2_best.pt", "V3")
check("timestamped inserts run id", p.endswith("lite2_phase2_best_V3.pt"), p)
_tmp = tempfile.mkdtemp()
f = os.path.join(_tmp, "x.pt"); open(f, "w").close()
check("unique_path avoids overwrite", unique_path(f).endswith("x_2.pt"))

# ---------------------------------------------------------------------------
print("\n[A2] all modules import + every CLI parses --help")
mods = ["config", "run_utils", "data_pipeline", "model", "cbam",
        "efficientnet_lite", "train", "metrics", "export", "evaluate",
        "parity_check", "verify"]
import importlib
imp_ok = True
for m in mods:
    try:
        importlib.import_module(f"src.{m}")
    except Exception as exc:  # noqa: BLE001
        imp_ok = False
        print(f"     import failed: src.{m}: {exc}")
check("import all src modules", imp_ok)
for cli in ("src.train", "src.export", "src.evaluate", "src.parity_check"):
    r = subprocess.run([sys.executable, "-m", cli, "--help"],
                       capture_output=True, text=True, cwd=PROJECT_ROOT)
    check(f"{cli} --help", r.returncode == 0)

# ---------------------------------------------------------------------------
print("\n[A3] mixing gate uses `training`, not the desc string")
loader = torch.utils.data.DataLoader(_DS(8), batch_size=4)


def run_once(training, mix_prob):
    model = _FakeModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = {"batches": 0, "mixed_batches": 0}
    T.run_epoch(model, loader, opt, torch.device("cpu"),
                (0.15, 0.5, 0.35), desc="phase2 e1/80", training=training,
                mix_prob=mix_prob, mix_stats=stats)
    return stats


s_on = run_once(True, 1.0)
s_off = run_once(True, 0.0)
s_eval = run_once(False, 1.0)
check("mix_prob=1.0 + training -> mixing ran", s_on["mixed_batches"] > 0,
      str(s_on))
check("mix_prob=0.0 -> no mixing", s_off["mixed_batches"] == 0, str(s_off))
check("training=False -> no mixing even at mix_prob=1.0",
      s_eval["mixed_batches"] == 0, str(s_eval))

# ---------------------------------------------------------------------------
print("\n[A4] resolve_checkpoint (path, run-tag, newest)")
cdir = tempfile.mkdtemp()
for name in ("lite2_phase2_best_old.pt", "lite2_phase2_best_V3.pt"):
    open(os.path.join(cdir, name), "w").close()
os.utime(os.path.join(cdir, "lite2_phase2_best_old.pt"), (10**9, 10**9))
os.utime(os.path.join(cdir, "lite2_phase2_best_V3.pt"), (2 * 10**9, 2 * 10**9))
check("resolve by run-tag V3",
      os.path.basename(resolve_checkpoint("V3", "lite2", cdir))
      == "lite2_phase2_best_V3.pt")
check("resolve explicit path",
      resolve_checkpoint(os.path.join(cdir, "lite2_phase2_best_old.pt"),
                         "lite2", cdir).endswith("old.pt"))
check("resolve None -> newest",
      os.path.basename(resolve_checkpoint(None, "lite2", cdir))
      == "lite2_phase2_best_V3.pt")
check("resolve missing tag -> None",
      resolve_checkpoint("nope", "lite2", cdir) is None)

# ---------------------------------------------------------------------------
print("\n[A5] class-count fail-fast + shared-species names")
try:
    dp._validate_class_counts(["gir", "bargur"], ["bargur"])
    check("raises on wrong counts", False)
except ValueError as exc:
    check("raises on wrong counts", "breed-count mismatch" in str(exc))
# Correct counts must not raise (monkeypatch expected lists to match).
old_c, old_b = C.EXPECTED_CATTLE_BREEDS, C.EXPECTED_BUFFALO_BREEDS
try:
    dp._validate_class_counts([f"c{i}" for i in range(C.NUM_CATTLE_BREEDS)],
                              [f"b{i}" for i in range(C.NUM_BUFFALO_BREEDS)])
    check("no raise when counts match", True)
except Exception:  # noqa: BLE001
    check("no raise when counts match", False)

# ---------------------------------------------------------------------------
print("\n[B2] same-species mixing keeps targets well-formed")
labels = synth_labels([0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 0, 0, 1, 1],
                      [0, 0, 1, 1, 0, 0, 1, 1])
imgs = torch.randn(8, 3, 16, 16)
ok = True
for _ in range(40):
    for fn in (dp.cutmix, dp.mixup):
        li = {k: v.clone() for k, v in labels.items()}
        _, lo = fn(imgs.clone(), li, same_species=True)
        cat = lo["binary"].argmax(1) == 0
        ok &= bool(torch.allclose(lo["binary"].sum(1), torch.ones(8)))
        ok &= bool(torch.allclose(lo["cattle_mask"] + lo["buffalo_mask"], torch.ones(8)))
        ok &= bool(torch.allclose(lo["cattle"][cat].sum(1), torch.ones(int(cat.sum()))))
        ok &= bool(torch.allclose(lo["buffalo"][cat].sum(1), torch.zeros(int(cat.sum()))))
check("binary one-hot + masks integral + targets sum to 1", ok)

# ---------------------------------------------------------------------------
print("\n[C] transforms: RRC ratio, pad-to-square, breed-aware selection")
dt = dp.describe_transform(dp._train_transform({}))
check("default train has flip + RRC", "RandomHorizontalFlip" in dt and "RandomResizedCrop" in dt)
check("RRC ratio bounded (0.92, 1.08)", "0.92" in dt and "1.08" in dt)
wide = Image.new("RGB", (400, 300), (200, 100, 50))
padded = dp._eval_transform(pad=True)(wide)
check("pad-to-square output is square 260", tuple(padded.shape) == (3, 260, 260),
      str(tuple(padded.shape)))
# A 4:3 image padded on the short side keeps full width: the left/right border
# columns should NOT be the pad fill (0 after Normalize) for a non-uniform image.
col = padded[:, :, 0].abs().sum().item()
check("pad keeps full width (no horizontal crop)", col > 1e-3, f"left-col L1={col:.3f}")
no_aug = {"horizontal_flip": False, "random_resized_crop": False,
          "color_jitter": False, "randaugment": False}
check("explicit no-aug train == eval",
      torch.allclose(dp._train_transform(no_aug)(wide), dp._eval_transform()(wide)))
check("coat-colour breed skips color jitter",
      dp.resolve_breed_augment("gir", {"color_jitter": True})["color_jitter"] is False)
check("non-coat breed keeps color jitter",
      dp.resolve_breed_augment("gir_x", {"color_jitter": True})["color_jitter"] is True)

# ---------------------------------------------------------------------------
print("\n[D1] CBAM / SE are identity at init")
for mod in (CBAM(88), SEBlock(88)):
    mod.eval()
    x = torch.randn(2, 88, 17, 17)
    with torch.no_grad():
        y = mod(x)
    check(f"{type(mod).__name__} identity at init", torch.allclose(y, x, atol=1e-6))

# ---------------------------------------------------------------------------
print("\n[F] EMA schedule + metrics species-aware top-k")
check("EMA decay at step 0 = 0.1", abs(T._ema_decay_at(0, 0.999, True) - 0.1) < 1e-9)
d = [T._ema_decay_at(t, 0.999, True) for t in (0, 10, 100, 1000, 100000)]
check("EMA decay monotonic + capped", all(d[i] <= d[i + 1] for i in range(len(d) - 1)) and d[-1] <= 0.999)
from src.metrics import evaluate_epoch


class _M(nn.Module):
    def forward(self, x):
        B = x.size(0)
        return {"binary": torch.randn(B, 2),
                "cattle": torch.randn(B, C.NUM_CATTLE_BREEDS),
                "buffalo": torch.randn(B, C.NUM_BUFFALO_BREEDS),
                "embedding": torch.randn(B, C.PROJECTION_DIM)}


m = evaluate_epoch(_M(), torch.utils.data.DataLoader(_DS(8), batch_size=4), "cpu")
for k in ("combined_top3", "combined_top5", "combined_top3_oracle",
          "combined_top5_oracle", "blended_score"):
    check(f"metrics has {k}", k in m, "" if k in m else str(sorted(m)))

# ---------------------------------------------------------------------------
print("\n[Inventory] per-dataset breed/species/count/resolution JSON")
import local_train as lt
root = tempfile.mkdtemp(prefix="inv_")
for rel, size in (("cattle/gir/a1.jpg", (640, 480)),
                  ("cattle/gir/a2.jpg", (640, 480)),
                  ("buffalo/nagpuri/b1.jpg", (800, 600))):
    fp = os.path.join(root, rel)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(fp)
out = tempfile.mkdtemp(prefix="invout_")
inv = lt.build_dataset_inventory(root, "synthetic", out_dir=out)
by = {e["breed"]: e for e in inv["breeds"]}
check("inventory json written", os.path.exists(os.path.join(out, "synthetic.json")))
check("inventory species + counts + resolutions",
      by["gir"]["species"] == "cattle" and by["gir"]["count"] == 2
      and by["gir"]["resolutions"] == {"640x480": 2}
      and by["nagpuri"]["species"] == "buffalo")
check("inventory has per-image resolution",
      len(by["gir"]["images"]) == 2 and by["gir"]["images"][0]["width"] == 640)
check("local_train preflight lists run_utils.py",
      "run_utils.py" in lt.REQUIRED_SRC_MODULES)

# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL MASTER CPU TESTS PASSED")
