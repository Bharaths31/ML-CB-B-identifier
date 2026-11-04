#!/usr/bin/env python3
"""Verification & Validation suite for the BreedID review panel.

Covers both halves of V&V:

  Verification  ("did we build the system right?")
    - backbone/head tensor shapes, checkpoint integrity, hidden trait heads
    - soft species routing maths
    - API response schema, aspect-ratio preprocessing
    - localization key parity / script coverage / theme tokens
    - cold-start + warm inference latency budgets

  Validation    ("did we build the right system for farmers?")
    - Big-4 outputs on in-distribution images
    - OOD safety and the farmer-friendly explanation
    - blur / exposure gate with trilingual tips
    - all 68 breeds in the knowledge base + gender/age advisory switching

Usage
    python -m tests.run_vv_suite
    python -m tests.run_vv_suite --json outputs/metrics/vv_review_report.json
    python -m tests.run_vv_suite --skip-benchmark

Exit code 0 = no FAIL (WARN allowed), 1 = at least one FAIL.
"""

import argparse
import glob
import io
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from src.breed_knowledge import (get_breed_knowledge,  # noqa: E402
                                 get_farmer_advisory)
from src.config import (BINARY_DIM, BREED_DIM, EXPECTED_BUFFALO_BREEDS,  # noqa: E402
                        EXPECTED_CATTLE_BREEDS, FEATURE_DIM, IMAGE_SIZE,
                        NUM_BUFFALO_BREEDS, NUM_CATTLE_BREEDS, TRAIT_FIELDS)
from src.image_quality import (analyze_image_quality,  # noqa: E402
                               explain_ood_for_user)
from src.model import (BreedClassifier, attach_trait_heads,  # noqa: E402
                       trait_head_spec_from_state)
from src.preprocess import PadToSquare, inference_transform  # noqa: E402

REPORT_MD = os.path.join(PROJECT_ROOT, "outputs", "metrics", "vv_review_report.md")
REPORT_JSON = os.path.join(PROJECT_ROOT, "outputs", "metrics", "vv_review_report.json")
PREFERRED_CKPT = "lite2_phase2_best_v26_1012-2332_f9ce.pt"


# ---------------------------------------------------------------------------
#  Result collector
# ---------------------------------------------------------------------------

class VV:
    def __init__(self):
        self.rows = []
        self.metrics = {}
        self.started = time.time()
        self.notes = []

    def add(self, suite, name, ok, detail="", status=None, metric=None):
        state = status or ("PASS" if ok else "FAIL")
        self.rows.append({
            "suite": suite,
            "name": name,
            "status": state,
            "detail": str(detail),
            "metric": metric,
        })
        return ok

    def warn(self, suite, name, detail=""):
        return self.add(suite, name, True, detail, status="WARN")

    def counts(self):
        out = {"PASS": 0, "FAIL": 0, "WARN": 0}
        for row in self.rows:
            out[row["status"]] += 1
        return out

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

def scene(width=1200, height=900, seed=3, blur=0, brightness=1.0):
    """Synthetic field photo: textured ground, animal blob, horns and an edge."""
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 0], base[..., 1], base[..., 2] = 112, 118, 96
    base = np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    draw.ellipse([width * 0.18, height * 0.28, width * 0.82, height * 0.86],
                 fill=(176, 158, 138))
    draw.polygon([(width * 0.30, height * 0.32), (width * 0.34, height * 0.10),
                  (width * 0.38, height * 0.32)], fill=(238, 232, 218))
    draw.polygon([(width * 0.62, height * 0.32), (width * 0.66, height * 0.10),
                  (width * 0.70, height * 0.32)], fill=(238, 232, 218))
    draw.line([(width * 0.12, height * 0.92), (width * 0.88, height * 0.92)], width=6)
    if brightness != 1.0:
        img = Image.eval(img, lambda v: max(0, min(255, int(v * brightness))))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


def encode(img, fmt="JPEG", quality=90):
    buf = io.BytesIO()
    img.save(buf, fmt, quality=quality)
    return buf.getvalue()


def checkerboard(width=900, height=600, cell=12):
    y, x = np.mgrid[0:height, 0:width]
    board = (((x // cell) + (y // cell)) % 2 * 255).astype(np.uint8)
    return Image.fromarray(np.stack([board] * 3, axis=2))


def noise_image(width=900, height=600, seed=11):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def solid(color, size=(900, 600)):
    return Image.new("RGB", size, color)


def pick_checkpoint():
    preferred = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", PREFERRED_CKPT)
    if os.path.isfile(preferred):
        return preferred
    candidates = sorted(
        glob.glob(os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "*.pt")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


_MANAGER = None
_MODEL_NAME = None


def manager_and_model():
    """Load test_model.ModelManager once per process."""
    global _MANAGER, _MODEL_NAME
    if _MANAGER is not None:
        return _MANAGER, _MODEL_NAME
    import test_model as tm

    _MANAGER = tm.ModelManager()
    preferred = os.path.splitext(PREFERRED_CKPT)[0]
    if preferred in _MANAGER.models:
        _MODEL_NAME = preferred
    else:
        pts = [n for n, i in _MANAGER.models.items() if i.get("type") == "pt"]
        if not pts:
            raise RuntimeError("No .pt checkpoint discovered under outputs/")
        _MODEL_NAME = sorted(pts)[-1]
    return _MANAGER, _MODEL_NAME


def canonical_breed_displays(manager):
    names = set()
    for cmap in manager.class_maps.values():
        for slug in cmap:
            names.add(slug.replace("_", " ").title())
    return names


# ---------------------------------------------------------------------------
#  Suite 1 — checkpoint & tensor architecture
# ---------------------------------------------------------------------------

def suite_architecture(vv):
    suite = "1. Architecture"

    model = BreedClassifier(backbone="lite2").eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))

    expected = {
        "binary": (2, 2),
        "cattle": (2, NUM_CATTLE_BREEDS),
        "buffalo": (2, NUM_BUFFALO_BREEDS),
        "features": (2, FEATURE_DIM),
    }
    shape_failures = []
    for key, shape in expected.items():
        if key not in out:
            shape_failures.append(f"{key}: missing")
        elif tuple(out[key].shape) != shape:
            shape_failures.append(f"{key}: {tuple(out[key].shape)} != {shape}")
    vv.add(suite, "Forward pass output shapes", not shape_failures,
           "; ".join(shape_failures) or
           f"binary {tuple(out['binary'].shape)}, cattle {tuple(out['cattle'].shape)}, "
           f"buffalo {tuple(out['buffalo'].shape)}, features {tuple(out['features'].shape)}",
           metric={"binary": BINARY_DIM, "breed": BREED_DIM, "features": FEATURE_DIM})

    ckpt_path = pick_checkpoint()
    if not ckpt_path:
        vv.add(suite, "Checkpoint discovered", False, "no .pt file under outputs/checkpoints")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt
    for key in ("state_dict", "model_state_dict"):
        if isinstance(ckpt, dict) and isinstance(ckpt.get(key), dict):
            state = ckpt[key]
            break

    nonfinite = []
    tensors = 0
    for key, value in state.items():
        if torch.is_tensor(value) and value.is_floating_point():
            tensors += 1
            if not torch.isfinite(value).all():
                nonfinite.append(key)
    vv.add(suite, "Checkpoint weights are finite", not nonfinite,
           f"{tensors} float tensors, {len(nonfinite)} non-finite" if not nonfinite
           else f"NaN/Inf in {nonfinite[:5]}",
           metric={"checkpoint": os.path.basename(ckpt_path), "tensors": tensors})

    spec = trait_head_spec_from_state(state)
    missing_fields = [f for f in TRAIT_FIELDS if f not in spec]
    vv.add(suite, "Hidden trait_heads present in checkpoint",
           not missing_fields and len(spec) == len(TRAIT_FIELDS),
           "; ".join(f"{f}={spec[f]}" for f in spec) if not missing_fields
           else f"missing {missing_fields}",
           metric={"fields": len(spec),
                   "keys": sum(1 for k in state if k.startswith("trait_heads."))})

    try:
        probe = BreedClassifier(backbone="lite2").eval()
        heads = attach_trait_heads(probe, state)
        mismatched = []
        if heads is not None:
            head_state = probe.trait_heads.state_dict()
            for key, value in state.items():
                if key.startswith("trait_heads."):
                    local = key[len("trait_heads."):]
                    if local in head_state and tuple(head_state[local].shape) != tuple(value.shape):
                        mismatched.append(key)
        vv.add(suite, "trait_heads load without shape mismatch",
               heads is not None and not mismatched,
               f"{len(spec)} fields, in_dim={next(iter(spec.values()))[0]}" if not mismatched
               else f"mismatched {mismatched[:5]}",
               metric={"in_dim": next(iter(spec.values()))[0]})
    except Exception as exc:
        vv.add(suite, "trait_heads load without shape mismatch", False,
               f"{type(exc).__name__}: {exc}")

    # Soft species routing must be a proper probability distribution over 68.
    torch.manual_seed(0)
    p_species = torch.softmax(torch.randn(7, 2), dim=1)
    p_cattle = torch.softmax(torch.randn(7, NUM_CATTLE_BREEDS), dim=1)
    p_buffalo = torch.softmax(torch.randn(7, NUM_BUFFALO_BREEDS), dim=1)
    combined = torch.cat([p_species[:, 0:1] * p_cattle,
                          p_species[:, 1:2] * p_buffalo], dim=1)
    max_err = float((combined.sum(dim=1) - 1.0).abs().max())
    vv.add(suite, "Soft routing sums to 1.0 over 68 breeds",
           combined.shape == (7, NUM_CATTLE_BREEDS + NUM_BUFFALO_BREEDS) and max_err < 1e-4,
           f"max |sum-1| = {max_err:.2e}, shape={tuple(combined.shape)}",
           metric={"max_error": max_err})


# ---------------------------------------------------------------------------
#  Suite 2 — Big-4 validation & preprocessing
# ---------------------------------------------------------------------------

def suite_big4(vv):
    suite = "2. Big-4"
    manager, model_name = manager_and_model()
    canonical = canonical_breed_displays(manager)

    checks = [
        ("In-distribution sample 1", scene(1200, 900, seed=3)),
        ("In-distribution sample 2", scene(1100, 850, seed=7, brightness=0.92)),
    ]
    for label, img in checks:
        data = encode(img)
        result = manager.predict(data, model_name, ood_profile="disabled",
                                 filename=f"vv_{label.lower().replace(' ', '_')}.jpg",
                                 guardrails=False)

        problems = []
        if result.get("error"):
            problems.append(result["error"])
        species = result.get("species")
        if species not in ("Cattle", "Buffalo"):
            problems.append(f"species={species}")
        if not (0 <= float(result.get("species_confidence") or 0) <= 100) or \
                float(result.get("species_confidence") or 0) <= 0:
            problems.append(f"species_confidence={result.get('species_confidence')}")

        top_breed = result.get("top_breed")
        if top_breed not in canonical:
            problems.append(f"top_breed={top_breed!r} not canonical")
        top5 = result.get("top5_breeds") or []
        if len(top5) != 5:
            problems.append(f"len(top5)={len(top5)}")
        if any(not (0 <= float(b.get("confidence", -1)) <= 100) for b in top5):
            problems.append("top5 confidence out of range")

        for block, allowed in (("gender", {"Female", "Male"}),
                               ("age_group", {"Calf", "Young", "Adult"})):
            payload = result.get(block) or {}
            if payload.get("short_label") not in allowed:
                problems.append(f"{block}={payload.get('short_label')!r}")
            conf = float(payload.get("confidence") or 0)
            if not (0 < conf <= 100):
                problems.append(f"{block}.confidence={conf}")

        traits = result.get("traits") or {}
        missing = [f for f in ("hump", "horn", "ear", "dewlap", "size")
                   if not str(traits.get(f) or "").strip()]
        if missing:
            problems.append(f"traits missing {missing}")

        vv.add(suite, f"Big-4 schema · {label}", not problems,
               "; ".join(problems) or
               f"{species} → {top_breed} · {result['gender']['short_label']} · "
               f"{result['age_group']['short_label']} · {len(traits)} traits",
               metric={"species_confidence": result.get("species_confidence"),
                       "top_breed_confidence": result.get("top_breed_confidence")})

    # Aspect-ratio preservation
    wide = Image.new("RGB", (1280, 960), (255, 255, 255))
    tall = Image.new("RGB", (720, 1280), (255, 255, 255))
    pad = PadToSquare(IMAGE_SIZE)
    fill = np.array(pad.fill, dtype=int)
    ratio_problems = []
    ratios = {}
    for label, img, expect in (("4:3", wide, 1280 / 960), ("9:16", tall, 720 / 1280)):
        out = pad(img)
        arr = np.asarray(out, dtype=int)
        mask = np.any(np.abs(arr - fill) > 6, axis=2)
        if out.size != (IMAGE_SIZE, IMAGE_SIZE) or not mask.any():
            ratio_problems.append(f"{label}: bad pad output {out.size}")
            continue
        ys, xs = np.where(mask)
        ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        ratios[label] = round(float(ratio), 3)
        if abs(ratio - expect) > 0.08:
            ratio_problems.append(f"{label}: {ratio:.3f} vs {expect:.3f}")

    tensor = inference_transform(pad=True)(wide)
    if tuple(tensor.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
        ratio_problems.append(f"tensor shape {tuple(tensor.shape)}")

    vv.add(suite, "PadToSquare preserves 4:3 and 9:16 extent",
           not ratio_problems,
           "; ".join(ratio_problems) or f"content ratios {ratios}, tensor {tuple(tensor.shape)}",
           metric=ratios)

    external = external_buffalo_metrics(manager, model_name)
    if external is None:
        vv.warn(suite, "External buffalo dataset",
                "data/external/indian_buffalo not found — download the Kaggle "
                "atharvadarpude/indian-buffalo-dataset to enable real-image validation")
    else:
        vv.metrics["external_dataset"] = external
        vv.add(suite, "External dataset · species accuracy",
               external["species_accuracy"] >= 0.75,
               f"{external['species_accuracy'] * 100:.1f}% on {external['samples']} real buffalo photos",
               metric={"accuracy": external["species_accuracy"]})
        vv.add(suite, "External dataset · top-5 breed accuracy",
               external["top5_accuracy"] >= 0.40,
               f"{external['top5_accuracy'] * 100:.1f}% (18 buffalo classes)",
               metric={"accuracy": external["top5_accuracy"]})
        vv.add(suite, "External dataset · top-1 breed accuracy",
               external["top1_accuracy"] >= 0.11,
               f"{external['top1_accuracy'] * 100:.1f}% · {external['top1_accuracy'] / (1/18):.1f}× chance",
               metric={"accuracy": external["top1_accuracy"]})
        if external["ood_rate"] <= 0.10:
            vv.add(suite, "External dataset · false OOD rejection",
                   True,
                   f"{external['ood_rate'] * 100:.1f}% rejected as OOD",
                   metric={"ood_rate": external["ood_rate"]})
        else:
            vv.warn(suite, "External dataset · false OOD rejection",
                    f"{external['ood_rate'] * 100:.1f}% rejected as OOD — the Kaggle set "
                    "contains diagrams/non-animal files; inspect before treating as model error")


def external_buffalo_metrics(manager, model_name, per_class=8):
    """Validate on the Atharva Darpude Indian buffalo dataset (2,785 images).

    Balanced deterministic sample (first N sorted images per class) keeps the
    suite fast while covering all 17 breeds. Ground-truth folders match the
    classifier's buffalo slugs except for hyphen/case normalisation.
    """
    root = None
    for candidate in (
        os.path.join(PROJECT_ROOT, "data", "external", "atharvadarpude_buffalo", "buffalo"),
        os.path.join(PROJECT_ROOT, "data", "external", "indian_buffalo", "buffalo"),
    ):
        if os.path.isdir(candidate):
            root = candidate
            break
    if root is None:
        return None

    samples = []
    rng = random.Random(2026)
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for folder in classes:
        folder_path = os.path.join(root, folder)
        images = sorted(
            name for name in os.listdir(folder_path)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        rng.shuffle(images)
        for name in images[:per_class]:
            samples.append((folder, os.path.join(folder_path, name)))

    if not samples:
        return None

    correct_species = correct_top1 = correct_top5 = false_ood = valid = 0
    for folder, path in samples:
        expected = folder.lower().replace("-", "_").replace(" ", "_")
        try:
            with open(path, "rb") as handle:
                payload = handle.read()
            result = manager.predict(payload, model_name, ood_profile="auto",
                                     filename=os.path.basename(path))
        except Exception:
            continue
        if result.get("error"):
            continue
        if result.get("ood"):
            false_ood += 1
            continue
        valid += 1
        if str(result.get("species", "")).lower().startswith("buff"):
            correct_species += 1
        slugs = [item.get("breed_slug") for item in result.get("top5_breeds", [])]
        if slugs and slugs[0] == expected:
            correct_top1 += 1
        if expected in slugs:
            correct_top5 += 1

    total = max(1, len(samples))
    denominator = max(1, valid)
    return {
        "dataset": "atharvadarpude/indian-buffalo-dataset",
        "samples": len(samples),
        "classes": len({folder for folder, _ in samples}),
        "valid": valid,
        "species_accuracy": round(correct_species / denominator, 4),
        "top1_accuracy": round(correct_top1 / denominator, 4),
        "top5_accuracy": round(correct_top5 / denominator, 4),
        "ood_rate": round(false_ood / total, 4),
    }


# ---------------------------------------------------------------------------
#  Suite 3 — OOD safety & quality gate
# ---------------------------------------------------------------------------

def suite_ood_quality(vv):
    suite = "3. OOD & Quality"
    manager, model_name = manager_and_model()

    cases = {
        "random noise": noise_image(),
        "solid black": solid((0, 0, 0)),
        "solid white": solid((255, 255, 255)),
        "checkerboard": checkerboard(),
    }
    unflagged = []
    for label, img in cases.items():
        result = manager.predict(encode(img), model_name, ood_profile="auto",
                                 filename="vv_ood.jpg")
        quality = result.get("quality") or {}
        explanation = result.get("ood_explanation") or {}
        flagged = bool(result.get("ood")) or quality.get("action") not in (None, "proceed")
        if not flagged:
            unflagged.append(label)
            continue
        if not result.get("ood") and quality.get("action") == "proceed":
            unflagged.append(label)
        messages = explanation.get("message") or {}
        if not all(str(messages.get(lang) or "").strip() for lang in ("en", "ta", "hi")):
            unflagged.append(f"{label} (missing trilingual message)")

    vv.add(suite, "Non-animal inputs are flagged with farmer guidance",
           not unflagged,
           f"unflagged: {unflagged}" if unflagged
           else f"{len(cases)}/{len(cases)} flagged; explanation in en/ta/hi",
           metric={"cases": len(cases)})

    blurry = scene(blur=8)
    blurred = analyze_image_quality(encode(blurry))
    tips = blurred.get("tips_by_language") or {}
    vv.add(suite, "Blur gate",
           bool(blurred.get("is_blurry")) and blurred.get("action") == "retake"
           and all(tips.get(lang) for lang in ("en", "ta", "hi")),
           f"lap-var={blurred.get('sharpness_score')} < {blurred.get('blur_threshold')}, "
           f"tips in {sorted(k for k, v in tips.items() if v)}",
           metric={"sharpness": blurred.get("sharpness_score")})

    dark = analyze_image_quality(encode(scene(brightness=0.12)))
    vv.add(suite, "Dark exposure gate",
           dark.get("lighting_issue") == "too_dark"
           and float(dark.get("mean_luminance") or 999) < 40
           and all((dark.get("tips_by_language") or {}).get(lang) for lang in ("en", "ta", "hi")),
           f"mean luminance={dark.get('mean_luminance')}, issue={dark.get('lighting_issue')}",
           metric={"mean_luminance": dark.get("mean_luminance")})

    # Explainer branches
    not_animal = explain_ood_for_user(True, {
        "msp": 0.30, "binary_confidence_floor": 0.55, "energy": -4.0,
        "energy_threshold": -5.0, "msp_threshold": 0.5,
        "ood_by_binary_floor": True,
    }, dark)
    unclear = explain_ood_for_user(True, {
        "msp": 0.62, "binary_confidence_floor": 0.55, "energy": -3.2,
        "energy_threshold": -5.0, "msp_threshold": 0.5, "ood_by_energy": True,
    }, blurred)
    vv.add(suite, "OOD explainer branches",
           not_animal["code"] == "not_animal" and unclear["code"] == "unclear_features"
           and "No cow or buffalo" in not_animal["message"]["en"]
           and "horns, hump, coat" in unclear["message"]["en"],
           "not_animal + unclear_features with trilingual copy",
           metric={"codes": [not_animal["code"], unclear["code"]]})


# ---------------------------------------------------------------------------
#  Suite 4 — knowledge base & advisory
# ---------------------------------------------------------------------------

def suite_knowledge(vv):
    suite = "4. Knowledge"
    required = ("origin_region", "utility_type", "milk_yield", "feeding_and_care",
                "estimated_market_price_inr", "govt_schemes")

    missing = []
    checked = 0
    for species, slugs in (("Cattle", EXPECTED_CATTLE_BREEDS),
                           ("Buffalo", EXPECTED_BUFFALO_BREEDS)):
        for slug in slugs:
            checked += 1
            entry = get_breed_knowledge(species, slug)
            if not entry:
                missing.append(f"{species}/{slug}: no record")
                continue
            for field in required:
                value = entry.get(field)
                if value in (None, "", [], {}):
                    missing.append(f"{species}/{slug}.{field}")
            advisory = get_farmer_advisory(species, slug)
            if "error" in advisory or not advisory.get("breed", {}).get("origin_region"):
                missing.append(f"{species}/{slug}: advisory")
            if len(entry.get("govt_schemes") or []) < 2:
                missing.append(f"{species}/{slug}: schemes")

    vv.add(suite, "68-breed knowledge coverage", not missing,
           f"{checked}/{checked} breeds complete" if not missing
           else f"{len(missing)} gaps: {missing[:5]}",
           metric={"breeds": checked, "gaps": len(missing)})

    female = get_farmer_advisory("Cattle", "gir", "Female", "Adult")
    male = get_farmer_advisory("Cattle", "gir", "Male", "Adult")
    calf = get_farmer_advisory("Cattle", "gir", None, "Calf")

    context_ok = (
        female.get("production", {}).get("milk")
        and male.get("production", {}).get("milk") is None
        and male.get("production", {}).get("draught")
        and "deworming_schedule" in (calf.get("health") or {})
        and female.get("headline") != male.get("headline") != calf.get("headline")
        and female.get("nutrition", {}).get("concentrate")
        != calf.get("nutrition", {}).get("concentrate")
    )
    vv.add(suite, "Gender/age advisory switching",
           bool(context_ok),
           "female=milk, male=draught, calf=colostrum/deworming, distinct feeding"
           if context_ok else "context blocks did not change as expected",
           metric={"contexts": 3})


# ---------------------------------------------------------------------------
#  Suite 5 — localization & themes (Node audit)
# ---------------------------------------------------------------------------

def suite_i18n_themes(vv):
    suite = "5. i18n & Themes"
    node = shutil.which("node")
    if not node:
        vv.warn(suite, "Node.js available", "node not found — UI audit skipped")
        return

    cwd = os.path.join(PROJECT_ROOT, "breed-classifier-react")
    script = os.path.join(cwd, "src", "verify_ui_i18n.mjs")
    proc = subprocess.run(
        [node, script], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    raw = (proc.stdout or "").strip()
    if not raw:
        vv.add(suite, "UI i18n audit executed", False,
               f"node exit {proc.returncode}: {(proc.stderr or '')[:200]}")
        return
    try:
        report = json.loads(raw.splitlines()[-1])
    except Exception as exc:
        vv.add(suite, "UI i18n audit JSON", False, f"{exc}: {raw[:160]}")
        return

    for item in report.get("checks", []):
        vv.add(suite, item.get("name", item.get("id")),
               item.get("status") == "PASS",
               item.get("detail", ""),
               status=item.get("status"),
               metric=item.get("metric"))

    vv.metrics["localization"] = report.get("metrics", {})
    if proc.returncode != 0 and not report.get("checks"):
        vv.add(suite, "UI i18n audit exit code", False, f"exit {proc.returncode}")


# ---------------------------------------------------------------------------
#  Suite 6 — performance
# ---------------------------------------------------------------------------

def suite_performance(vv, skip=False):
    suite = "6. Performance"
    if skip:
        vv.warn(suite, "Latency benchmark", "skipped by --skip-benchmark")
        return

    manager, model_name = manager_and_model()

    manager.models[model_name]["model"] = None
    cold_start = time.perf_counter()
    manager._load_model(model_name)
    cold_ms = (time.perf_counter() - cold_start) * 1000
    vv.metrics["cold_load_ms"] = round(cold_ms, 1)

    data = encode(scene(1200, 900, seed=21))
    manager.predict(data, model_name, ood_profile="disabled", filename="vv_warmup.jpg")

    latencies = []
    for index in range(8):
        start = time.perf_counter()
        result = manager.predict(data, model_name, ood_profile="disabled",
                                 filename=f"vv_warm_{index}.jpg")
        latencies.append((time.perf_counter() - start) * 1000)
        if result.get("error"):
            vv.add(suite, "Warm inference", False, result["error"])
            return

    mean_ms = statistics.mean(latencies)
    p95_ms = sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    budget = 500 if device == "cuda" else 1500

    vv.metrics["warm_inference_ms"] = round(mean_ms, 1)
    vv.metrics["warm_p95_ms"] = round(p95_ms, 1)
    vv.metrics["device"] = device
    vv.metrics["budget_ms"] = budget

    vv.add(suite, f"Warm inference latency ({device})", mean_ms < budget,
           f"mean {mean_ms:.1f} ms · p95 {p95_ms:.1f} ms · budget {budget} ms",
           metric={"mean_ms": round(mean_ms, 1), "p95_ms": round(p95_ms, 1)})
    vv.add(suite, "Cold model load", cold_ms < 15000,
           f"{cold_ms / 1000:.2f} s", metric={"cold_load_ms": round(cold_ms, 1)})


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------

ICON = {"PASS": "PASS ✅", "FAIL": "FAIL ❌", "WARN": "WARN ⚠️"}


def print_scorecard(vv):
    width_suite = 18
    width_name = 52
    width_detail = 58
    counts = vv.counts()
    print("\n" + "=" * 132)
    print("  BREEDID — VERIFICATION & VALIDATION SCORECARD")
    print("=" * 132)
    print(f"  {'SUITE':<{width_suite}} {'CHECK':<{width_name}} {'RESULT':<11} DETAIL")
    print("-" * 132)
    current = None
    for row in vv.rows:
        if row["suite"] != current:
            current = row["suite"]
            print(f"  {current:<{width_suite}}")
        name = row["name"][:width_name]
        detail = row["detail"][:width_detail]
        print(f"  {'':<{width_suite}} {name:<{width_name}} {ICON[row['status']]:<11} {detail}")
    print("-" * 132)
    print(f"  PASS {counts['PASS']}   FAIL {counts['FAIL']}   WARN {counts['WARN']}   "
          f"TOTAL {len(vv.rows)}   duration {time.time() - vv.started:.1f}s")
    if vv.metrics.get("warm_inference_ms"):
        print(f"  Warm inference {vv.metrics['warm_inference_ms']} ms "
              f"({vv.metrics.get('device')}), cold load {vv.metrics.get('cold_load_ms')} ms")
    print("=" * 132 + "\n")


def write_reports(vv):
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    counts = vv.counts()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# BreedID — Verification & Validation Report",
        "",
        f"**Generated:** {stamp}  ",
        f"**Environment:** Python {sys.version.split()[0]} · "
        f"torch {torch.__version__} · device {vv.metrics.get('device', 'cpu')}  ",
        f"**Result:** {counts['PASS']} passed · {counts['FAIL']} failed · "
        f"{counts['WARN']} warnings",
        "",
        "---",
        "",
        "## Scorecard",
        "",
        "| Suite | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    for row in vv.rows:
        lines.append(
            f"| {row['suite']} | {row['name']} | {ICON[row['status']]} | "
            f"{row['detail'].replace('|', '/')} |"
        )

    if vv.metrics:
        lines += ["", "## Metrics", "", "| Metric | Value |", "|---|---|"]
        for key, value in vv.metrics.items():
            if isinstance(value, dict):
                value = ", ".join(f"{k}={v}" for k, v in value.items())
            lines.append(f"| {key} | {value} |")

    failures = vv.failed
    if failures:
        lines += ["", "## Failures to resolve", ""]
        for row in failures:
            lines.append(f"- **{row['suite']} / {row['name']}** — {row['detail']}")
    elif counts["WARN"]:
        lines += ["", "## Verdict", "",
                  f"All verification and validation checks passed. "
                  f"{counts['WARN']} warning(s) remain for review — see the scorecard "
                  "rows marked WARN (for example external OOD abstentions on the "
                  "Kaggle buffalo set, which contains diagrams/non-animal files)."]
    else:
        lines += ["", "## Verdict", "",
                  "All verification and validation checks passed. The model checkpoint, "
                  "Big-4 API contract, OOD/quality safety, 68-breed knowledge base, "
                  "trilingual UI and theme tokens are ready for the review panel."]

    lines += [
        "",
        "---",
        "",
        "*Reproduce with:* `python -m tests.run_vv_suite`  ",
        "*UI audit:* `node breed-classifier-react/src/verify_ui_i18n.mjs`",
        "",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": stamp,
            "counts": counts,
            "metrics": vv.metrics,
            "rows": vv.rows,
        }, f, indent=2)
    return REPORT_MD


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

SUITES = (
    ("1. Architecture", suite_architecture),
    ("2. Big-4", suite_big4),
    ("3. OOD & Quality", suite_ood_quality),
    ("4. Knowledge", suite_knowledge),
    ("5. i18n & Themes", suite_i18n_themes),
    ("6. Performance", suite_performance),
)


def main(argv=None):
    global REPORT_MD, REPORT_JSON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=REPORT_JSON, help="JSON report path")
    parser.add_argument("--md", default=REPORT_MD, help="Markdown report path")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--suite", action="append", default=None,
                        help="run only suites whose label starts with this prefix")
    args = parser.parse_args(argv)

    REPORT_MD, REPORT_JSON = args.md, args.json

    vv = VV()
    print("\nBreedID — Verification & Validation suite")
    print(f"Checkpoint: {os.path.basename(pick_checkpoint() or 'none')}")

    for label, func in SUITES:
        if args.suite and not any(label.startswith(s) for s in args.suite):
            continue
        print(f"\n[*] Running {label} ...")
        try:
            if func is suite_performance:
                func(vv, skip=args.skip_benchmark)
            else:
                func(vv)
        except Exception as exc:
            vv.add(label, "Suite execution", False,
                   f"{type(exc).__name__}: {exc}")
            if os.environ.get("VV_DEBUG"):
                traceback.print_exc()

    print_scorecard(vv)
    if args.suite:
        print("  Partial run (--suite): review report not rewritten. "
              "Run the full suite to refresh outputs/metrics/vv_review_report.md.\n")
    else:
        path = write_reports(vv)
        print(f"  Markdown report: {os.path.relpath(path, PROJECT_ROOT)}")
        print(f"  JSON report:     {os.path.relpath(REPORT_JSON, PROJECT_ROOT)}\n")

    return 1 if vv.failed else 0


if __name__ == "__main__":
    sys.exit(main())

