#!/usr/bin/env python3
"""Smoke-test the post-hoc attribute pipeline (traits + gender + age).

Runs entirely offline against the newest PyTorch checkpoint -- no retraining,
no dataset, no CLIP download required.

    python scripts/verify_attributes.py
    python scripts/verify_attributes.py --image path/to/cow.jpg
    python scripts/verify_attributes.py --model lite2_phase2_best_v26_1012-2332_f9ce
    python scripts/verify_attributes.py --skip-onnx

Checks
  1. The checkpoint still loads through ModelManager._load_model().
  2. Trait heads are rebuilt from the state_dict and the vocab aligns with
     data/breed_traits.json (or cleanly falls back if it cannot).
  3. predict() returns the five API blocks with sane ranges.
  4. ONNX models keep working and simply omit traits.
  5. OOD images skip gender/age instead of guessing.

Exit code 0 = all checks passed.
"""

import argparse
import io
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from PIL import Image  # noqa: E402

from src.config import TRAIT_FIELDS  # noqa: E402

TRAIT_LABEL_FIELDS = ("hump", "horn", "ear", "dewlap", "size")

failures = []


def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        failures.append(message)
    return condition


def synthetic_image(width=1200, height=900):
    """A 4:3 'photo' so the pad-to-square path is exercised."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (118, 108, 94)).save(buf, "JPEG")
    return buf.getvalue()


def pick_model(manager, explicit=None):
    if explicit:
        return explicit
    preferred = "lite2_phase2_best_v26_1012-2332_f9ce"
    if preferred in manager.models:
        return preferred
    candidates = [n for n, i in manager.models.items() if i.get("type") == "pt"]
    if not candidates:
        raise SystemExit("No .pt checkpoint found under outputs/.")
    return sorted(candidates)[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="model name as listed by test_model.py")
    parser.add_argument("--image", help="image to use instead of a synthetic 4:3 frame")
    parser.add_argument("--skip-onnx", action="store_true",
                        help="skip the ONNX regression check")
    args = parser.parse_args()

    import test_model as tm

    print("\nBreedID — attribute pipeline verification")
    print("=" * 62)

    manager = tm.ModelManager()
    name = pick_model(manager, args.model)
    print(f"  model: {name}")

    image_bytes = open(args.image, "rb").read() if args.image else synthetic_image()

    # --- 1/2. load + trait heads ----------------------------------------
    print("\n[1] Checkpoint load & trait heads")
    model = manager._load_model(name)
    heads = getattr(model, "trait_heads", None)
    check(heads is not None, "trait_heads attached from state_dict")
    if heads is not None:
        meta = heads.trait_meta
        check(sorted(meta["fields"]) == sorted(TRAIT_FIELDS),
              f"all {len(TRAIT_FIELDS)} trait fields present: {sorted(meta['fields'])}")
        check(meta["in_dim"] == 1280, f"expected 1280-d pooled input (got {meta['in_dim']})")
        check(meta["source"] in ("breed_traits.json", "partial_fallback", "fallback"),
              f"vocab source = {meta['source']}")

    # --- 3. in-distribution prediction ----------------------------------
    print("\n[2] Enriched prediction (OOD disabled to force attributes)")
    result = manager.predict(image_bytes, name, ood_profile="disabled",
                             filename="verify.jpg")
    check("error" not in result, f"prediction returned without error ({result.get('error')})")

    for key in ("species", "species_confidence", "top_breed",
                "top_breed_confidence", "top5_breeds", "gender", "age_group",
                "traits", "knowledge", "quality", "ood_explanation", "ood"):
        check(key in result, f"response contains '{key}'")

    quality = result.get("quality") or {}
    check({"is_blurry", "lighting_issue", "resolution_issue", "tips"} <= set(quality),
          "quality report exposes the required fields")
    check(quality.get("latency_ms", 999) < 15,
          f"quality gate ran in {quality.get('latency_ms')} ms (< 15 ms)")
    check(result.get("ood_explanation", {}).get("code") in
          ("ok", "poor_quality"),
          f"ood_explanation.code = {result.get('ood_explanation', {}).get('code')}")

    knowledge = result.get("knowledge") or {}
    check(bool(knowledge.get("breed", {}).get("slug")),
          f"knowledge names the breed ({knowledge.get('breed', {}).get('slug')})")
    check(bool(knowledge.get("priority_actions")),
          "knowledge has priority actions")

    check(len(result.get("top5_breeds", [])) >= 1, "top5_breeds is populated")

    gender = result.get("gender") or {}
    age = result.get("age_group") or {}
    check(gender.get("method") in ("clip_zeroshot", "morphological_heuristic"),
          f"gender.method = {gender.get('method')}")
    check(age.get("method") in ("clip_zeroshot", "morphological_heuristic"),
          f"age_group.method = {age.get('method')}")
    check(0 <= float(gender.get("confidence", -1)) <= 100,
          f"gender confidence in 0..100 ({gender.get('confidence')})")
    check(0 <= float(age.get("confidence", -1)) <= 100,
          f"age confidence in 0..100 ({age.get('confidence')})")
    check(gender.get("short_label") in ("Female", "Male"),
          f"gender short_label = {gender.get('short_label')}")
    check(age.get("short_label") in ("Calf", "Young", "Adult"),
          f"age short_label = {age.get('short_label')}")

    traits = result.get("traits") or {}
    check(all(field in traits for field in TRAIT_LABEL_FIELDS),
          f"traits cover {TRAIT_LABEL_FIELDS}")
    check(all(isinstance(v, str) and v for v in traits.values()),
          "trait labels are non-empty strings")

    print("\n  gender : {label} ({confidence}%, {method})".format(**gender))
    print("  age    : {label} ({confidence}%, {method})".format(**age))
    print("  traits : " + json.dumps(traits))

    # --- 4. ONNX regression ---------------------------------------------
    if not args.skip_onnx:
        print("\n[3] ONNX regression (traits unavailable, no crash)")
        onnx_names = [n for n, i in manager.models.items() if i.get("type") == "onnx"]
        if onnx_names:
            onnx_name = onnx_names[0]
            onnx_result = manager.predict(image_bytes, onnx_name,
                                          ood_profile="disabled", filename="verify.jpg")
            check("error" not in onnx_result,
                  f"ONNX prediction works ({onnx_name})")
            check(onnx_result.get("traits") == {}, "ONNX omits traits")
            check(onnx_result.get("traits_meta", {}).get("reason") == "onnx",
                  "ONNX reports traits_meta.reason = onnx")
        else:
            print("  [SKIP] no ONNX models discovered")

    # --- 5. OOD gating ---------------------------------------------------
    print("\n[4] OOD gating")
    ood_result = manager.predict(image_bytes, name, ood_profile="auto",
                                 filename="verify.jpg")
    if ood_result.get("ood"):
        check(ood_result.get("gender", {}).get("method") == "skipped",
              "OOD input skips gender estimation")
        check(ood_result.get("age_group", {}).get("method") == "skipped",
              "OOD input skips age estimation")
        check(ood_result.get("knowledge", {}).get("skipped") is True,
              "OOD input skips the farmer advisory")
        check(ood_result.get("ood_explanation", {}).get("code") in
              ("not_animal", "unclear_features"),
              f"OOD explanation code = {ood_result.get('ood_explanation', {}).get('code')}")
        check(bool(ood_result.get("ood_explanation", {}).get("message", {}).get("ta")),
              "OOD explanation is trilingual")
    else:
        print("  [SKIP] synthetic frame was not flagged OOD by this profile")

    print("\n" + "=" * 62)
    if failures:
        print(f"  {len(failures)} check(s) FAILED")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
