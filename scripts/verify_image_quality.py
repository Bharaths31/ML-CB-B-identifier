#!/usr/bin/env python3
"""Verify the pre-inference quality gate and the OOD explainer.

Generates synthetic field-photo failure modes (sharp, blurry, dark, blown-out,
tiny, panoramic, unreadable) and asserts the flags, trilingual copy and the
< 15 ms budget. Also exercises both OOD branches with crafted detector details.

    python scripts/verify_image_quality.py
"""

import io
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.config import QUALITY_BLUR_THRESHOLD  # noqa: E402
from src.image_quality import (analyze_image_quality,  # noqa: E402
                               explain_ood_for_user)

failures = []


def check(condition, message):
    print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)
    return condition


def scene(width=1200, height=900, seed=3):
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 0], base[..., 1], base[..., 2] = 110, 120, 95
    base = np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    draw.ellipse([width * 0.2, height * 0.25, width * 0.8, height * 0.85],
                 fill=(180, 160, 140))
    draw.polygon([(width * 0.3, height * 0.3), (width * 0.34, height * 0.1),
                  (width * 0.38, height * 0.3)], fill=(240, 235, 220))
    draw.line([(width * 0.15, height * 0.9), (width * 0.85, height * 0.9)], width=6)
    return img


def encode(img, fmt="JPEG", quality=90):
    buf = io.BytesIO()
    img.save(buf, fmt, quality=quality)
    return buf.getvalue()


def main():
    print("\nBreedID — image quality gate verification")
    print("=" * 62)

    sharp = scene()
    cases = {
        "sharp": encode(sharp),
        "blurry": encode(sharp.filter(ImageFilter.GaussianBlur(6))),
        "dark": encode(Image.eval(sharp, lambda v: int(v * 0.18))),
        "overexposed": encode(Image.eval(sharp, lambda v: min(255, int(v * 1.7 + 60)))),
        "low_resolution": encode(sharp.resize((140, 110))),
        "panoramic": encode(sharp.resize((3200, 800))),
        "unreadable": b"this is not an image",
    }

    print("\n[1] Flags")
    reports = {name: analyze_image_quality(data) for name, data in cases.items()}

    check(not reports["sharp"]["is_blurry"],
          f"sharp image is not flagged blurry (lap-var {reports['sharp']['sharpness_score']})")
    check(reports["sharp"]["ok"] and reports["sharp"]["action"] == "proceed",
          "sharp image passes the gate")
    check(reports["blurry"]["is_blurry"], "heavily blurred image is flagged blurry")
    check(reports["blurry"]["action"] == "retake", "blurred image recommends retake")
    check(reports["dark"]["lighting_issue"] == "too_dark",
          "dark image flagged too_dark")
    check(reports["overexposed"]["lighting_issue"] == "overexposed",
          "overexposed image flagged overexposed")
    check(reports["low_resolution"]["resolution_issue"] and
          reports["low_resolution"]["min_dimension"] < 180,
          "tiny image flagged low_resolution")
    check(reports["panoramic"]["aspect_issue"], "3200x800 flagged extreme_aspect")
    check(reports["unreadable"]["issues"] == ["unreadable"],
          "unreadable bytes return an issue instead of raising")

    print("\n[2] Trilingual tips")
    blurry = reports["blurry"]
    check(len(blurry["tips"]) >= 1, "failed checks produce at least one tip")
    check(all(set(tip) == {"en", "ta", "hi"} for tip in blurry["tips"]),
          "every tip has en/ta/hi")
    check(bool(blurry["tips_by_language"]["ta"]) and bool(blurry["tips_by_language"]["hi"]),
          "tips_by_language exposes Tamil and Hindi")
    check(0 <= blurry["score"] <= 100, f"score is 0-100 ({blurry['score']})")
    check(reports["sharp"]["tips"] == [], "clean photo carries no tips")

    print("\n[3] Latency (< 15 ms)")
    # bytes path (includes JPEG decode; the API path reuses the decoded image)
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        analyze_image_quality(cases["sharp"])
        timings.append((time.perf_counter() - started) * 1000)
    avg_bytes = sum(timings) / len(timings)

    pil = Image.open(io.BytesIO(cases["sharp"])).convert("RGB")
    started = time.perf_counter()
    for _ in range(20):
        analyze_image_quality(pil)
    avg_pil = (time.perf_counter() - started) / 20 * 1000

    check(avg_bytes < 15.0, f"bytes path averages {avg_bytes:.2f} ms")
    check(avg_pil < 15.0, f"decoded-PIL path averages {avg_pil:.2f} ms")

    print("\n[4] OOD explainer")
    base = {"msp": 0.3, "binary_confidence_floor": 0.55, "energy": -4.0,
            "energy_threshold": -5.0, "msp_threshold": 0.5,
            "ood_by_binary_floor": True}
    not_animal = explain_ood_for_user(True, base, reports["blurry"])
    check(not_animal["code"] == "not_animal",
          "low species confidence -> not_animal")
    check("No cow or buffalo" in not_animal["message"]["en"],
          "not_animal message matches the farmer-facing copy")
    check(bool(not_animal["message"]["ta"]) and bool(not_animal["message"]["hi"]),
          "not_animal message is trilingual")

    weak = {"msp": 0.62, "binary_confidence_floor": 0.55, "energy": -3.2,
            "energy_threshold": -5.0, "msp_threshold": 0.5, "ood_by_energy": True}
    unclear = explain_ood_for_user(True, weak, reports["sharp"])
    check(unclear["code"] == "unclear_features",
          "high energy -> unclear_features")
    check("horns, hump, coat" in unclear["message"]["en"],
          "unclear_features mentions the missing breed features")
    check(any("side" in tip["en"].lower() or "profile" in tip["en"].lower()
              for tip in unclear["tips"]),
          "unclear_features suggests a side profile")

    ok = explain_ood_for_user(False, weak, reports["sharp"])
    check(ok["code"] == "ok", "in-distribution input returns code=ok")

    poor = explain_ood_for_user(False, weak, reports["blurry"])
    check(poor["code"] == "poor_quality",
          "low-quality in-distribution input is surfaced as poor_quality")

    check(set(not_animal["technical"]) >= {"energy", "msp", "energy_threshold"},
          "raw detector values stay under technical")

    print("\n" + "=" * 62)
    print(f"  blur threshold: {QUALITY_BLUR_THRESHOLD} (lap-var, 256 px grayscale)")
    if failures:
        print(f"  {len(failures)} check(s) FAILED")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
