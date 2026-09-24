#!/usr/bin/env python
"""CPU-only synthetic tests for the master-task fixes. No GPU/dataset needed.

Run:  python scripts/test_master_cpu.py
"""

import copy
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# Keep the test run itself out of logs/ (logger is tested in a subprocess).
os.environ.setdefault("RUN_LOG_DISABLE", "1")

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
                "features": torch.zeros(B, C.FEATURE_DIM) + self.p,
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
print("\n[D2] breed trait heads: vocab, targets, masked loss, train integration")
from src.traits import (TraitClassifier, breed_trait_targets, build_trait_vocab)
spec = {"fields": ["hump", "coat"],
        "breeds": {"cattle": {"gir": {"hump": "prominent", "coat": "red"},
                              "sahiwal": {"hump": "moderate", "coat": ""}},
                   "buffalo": {"murrah": {"hump": "", "coat": "black"}}}}
vocab = build_trait_vocab(spec)
check("trait vocab built from non-empty values",
      vocab == {"hump": {"moderate": 0, "prominent": 1},
                "coat": {"black": 0, "red": 1}}, str(vocab))
cat_c = {"gir": 0, "sahiwal": 1}
buf_c = {"murrah": 0}
tg = breed_trait_targets(spec, vocab, cat_c, buf_c)
check("empty trait -> -1 (ignored) and buffalo offset correct",
      tg["coat"][1] == -1 and tg["hump"][C.NUM_CATTLE_BREEDS] == -1
      and tg["coat"][C.NUM_CATTLE_BREEDS] == 0)
tm = TraitClassifier(C.FEATURE_DIM, vocab)
tloss, taccs = tm.loss(torch.randn(4, C.FEATURE_DIM),
                       torch.tensor([0, 1, 0, C.NUM_CATTLE_BREEDS]), tg)
check("trait masked loss finite", bool(torch.isfinite(tloss)))

# integration: train_phase runs with trait heads attached
import copy as _copy
tm_model = _FakeModel()
tm_model.trait_heads = TraitClassifier(C.FEATURE_DIM, vocab)
tm_ema = _copy.deepcopy(tm_model); tm_ema.eval()
_ = T.train_phase(tm_model, loader, torch.utils.data.DataLoader(_DS(8), batch_size=4),
                  torch.device("cpu"), 2, 1, 1e-3, (0.15, 0.5, 0.35),
                  lambda o: None, os.path.join(tempfile.mkdtemp(), "ck.pt"),
                  best_key="blended_score", ema_model=tm_ema, mix_prob=0.0,
                  trait_module=tm_model.trait_heads, trait_targets=tg,
                  trait_weight=0.1)
check("train_phase runs with trait heads", True)

print("\n[D4] SupCon hard-negative weighting")
from src.train import supervised_contrastive_loss
_z = torch.randn(6, 64)
_ids = torch.tensor([0, 0, 1, 1, 2, 2])
_base = supervised_contrastive_loss(_z, _ids, 0.1)
_hard = supervised_contrastive_loss(_z, _ids, 0.1,
                                    hard_pairs={frozenset({0, 1})},
                                    hard_neg_weight=4.0)
check("hard-neg SupCon finite", bool(torch.isfinite(_base) and torch.isfinite(_hard)))
check("hard-neg changes the loss", not torch.isclose(_base, _hard))

print("\n[E1] group-aware (dHash) splits keep near-duplicates together")
_inv = tempfile.mkdtemp(prefix="dedup_")
for _n, _s in (("a.jpg", 1), ("b.jpg", 1), ("c.jpg", 1), ("d.jpg", 1),
               ("e.jpg", 50), ("f.jpg", 51), ("g.jpg", 52), ("h.jpg", 53)):
    Image.fromarray((np.random.RandomState(_s).rand(40, 40, 3) * 255)
                    .astype("uint8")).save(os.path.join(_inv, _n))
_dpaths = [os.path.join(_inv, n) for n in
           ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg", "g.jpg", "h.jpg")]
_hashes = {p: dp._dhash(p) for p in _dpaths}
check("dHash: identical images distance 0",
      dp._hamming(_hashes[_dpaths[0]], _hashes[_dpaths[1]]) == 0)
check("dHash: distinct images distance > 4",
      dp._hamming(_hashes[_dpaths[0]], _hashes[_dpaths[4]]) > 4)
_ddf = pd.DataFrame({"path": _dpaths, "species": ["cattle"] * 8,
                     "breed": ["gir"] * 8, "binary_label": [0] * 8})
_dtr, _dva, _dte = dp._stratified_split_dedup(_ddf, random.Random(0), _hashes, 4)
_assign = {i: ("train" if i in _dtr else "val" if i in _dva else "test")
           for i in range(8)}
check("near-duplicate group stays in ONE split",
      len({_assign[i] for i in range(4)}) == 1, str(_assign))
check("dedup split keeps val/test non-empty",
      len(_dva) >= 1 and len(_dte) >= 1)

print("\n[D3] cosine/ArcFace breed heads")
from src.model import BreedClassifier as _BC, CosineHead as _CosineHead
from src.train import _apply_cosine_margin
from src.export import _load_model
_cm = _BC("lite2", cosine_head=True).eval()
check("cosine head is the final breed-head layer",
      isinstance(_cm.cattle_head[-1], _CosineHead))
with torch.no_grad():
    _co = _cm(torch.randn(2, 3, 260, 260))
check("margin-free forward bounded by scale",
      float(_co["cattle"].abs().max()) <= _cm.cosine_scale + 1e-3)
_lg = torch.randn(4, 5)
_tgt = F.one_hot(torch.tensor([0, 1, 2, 3]), 5).float()
_mg = _apply_cosine_margin(_lg, _tgt, 0.3, 30.0)
check("margin changes target logits", not torch.allclose(_lg, _mg))
check("margin output finite", bool(torch.isfinite(_mg).all()))
_cp = os.path.join(tempfile.mkdtemp(), "cos.pt")
torch.save({"state_dict": _cm.state_dict()}, _cp)
check("export auto-detects cosine head",
      _load_model(_cp, "lite2", "cbam").cosine_head is True)
_lin = _BC("lite2", cosine_head=False)
_lp = os.path.join(tempfile.mkdtemp(), "lin.pt")
torch.save({"state_dict": _lin.state_dict()}, _lp)
check("export keeps linear head when not cosine",
      _load_model(_lp, "lite2", "cbam").cosine_head is False)

print("\n[Logger] per-execution folder, manifest, config, events, metrics")
tmp_logs = tempfile.mkdtemp(prefix="logs_")
code = (
    "import sys; sys.path.insert(0, %r)\n"
    "from src.run_logger import init_run_logger, log_event, log_metrics, finish_run\n"
    "lg = init_run_logger(exec_id='TESTEXEC', module='unit', log_root=%r, capture_stdio=False)\n"
    "log_event('hello', category='actions', x=1)\n"
    "log_metrics({'blended_score':0.5,'combined_top1_soft':0.4}, epoch=1, tag='raw')\n"
    "finish_run(0)\n"
    "print(lg.dir)\n" % (PROJECT_ROOT, tmp_logs)
)
env = {k: v for k, v in os.environ.items() if k != "RUN_LOG_DISABLE"}
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
folder = (r.stdout.strip().splitlines() or [""])[-1]
check("logger creates exec folder", os.path.isdir(folder), folder or r.stderr[-200:])
check("manifest.json written", os.path.exists(os.path.join(folder, "manifest.json")))
check("config.json written", os.path.exists(os.path.join(folder, "config.json")))
check("events.jsonl contains the event",
      os.path.exists(os.path.join(folder, "events.jsonl"))
      and '"hello"' in open(os.path.join(folder, "events.jsonl")).read())
check("training.jsonl contains metrics",
      os.path.exists(os.path.join(folder, "training.jsonl"))
      and "blended_score" in open(os.path.join(folder, "training.jsonl")).read())

print(f"\n{'='*60}\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL MASTER CPU TESTS PASSED")
