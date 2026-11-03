#!/usr/bin/env python3
"""Verify the farmer knowledge base: coverage, schema and advisory logic.

    python scripts/verify_breed_knowledge.py
    python scripts/verify_breed_knowledge.py --json

Runs fully offline (no model, no dataset). Checks every configured breed and
every gender × age context, then times the recalculation path against the
< 10 ms UI budget.
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.breed_knowledge import (CATTLE_BREEDS, BUFFALO_BREEDS,
                                 EXTRA_CATTLE_BREEDS, get_breed_knowledge,
                                 get_farmer_advisory, knowledge_stats,
                                 load_knowledge)

REQUIRED_FIELDS = (
    "display_name", "local_names", "origin_region", "utility_type",
    "milk_yield", "physical_markers", "feeding_and_care",
    "climate_and_disease_resistance", "estimated_market_price_inr",
    "govt_schemes", "data_confidence",
)
VALID_UTILITY = {"Milch (Dairy)", "Draught (Work)", "Dual-Purpose",
                 "Ornamental / Dwarf Dairy"}
CONTEXTS = [(g, a) for g in ("Female", "Male", None)
            for a in ("Calf", "Young", "Adult", None)]

failures = []


def check(condition, message):
    print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)
    return condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the stats as JSON")
    args = parser.parse_args()

    print("\nBreedID — farmer knowledge base verification")
    print("=" * 62)

    kb = load_knowledge()
    cattle = kb.get("Cattle", {})
    buffalo = kb.get("Buffalo", {})
    print(f"  entries: {len(cattle)} cattle + {len(buffalo)} buffalo")

    missing_cattle = [b for b in CATTLE_BREEDS if b not in cattle]
    missing_buffalo = [b for b in BUFFALO_BREEDS if b not in buffalo]
    check(not missing_cattle, f"all {len(CATTLE_BREEDS)} cattle breeds present")
    check(not missing_buffalo, f"all {len(BUFFALO_BREEDS)} buffalo breeds present")
    check(all(b in cattle for b in EXTRA_CATTLE_BREEDS),
          f"crossbred classes present: {list(EXTRA_CATTLE_BREEDS)}")

    # --- schema for every entry -----------------------------------------
    print("\n[1] Entry schema")
    bad_fields, bad_utility, bad_markers = [], [], []
    for species, section in (("Cattle", cattle), ("Buffalo", buffalo)):
        for slug, entry in section.items():
            for field in REQUIRED_FIELDS:
                if field not in entry:
                    bad_fields.append(f"{species}/{slug}.{field}")
            if entry.get("utility_type") not in VALID_UTILITY:
                bad_utility.append(f"{species}/{slug}={entry.get('utility_type')}")
            markers = entry.get("physical_markers") or []
            if len(markers) != 3 or not all(isinstance(m, str) and m for m in markers):
                bad_markers.append(f"{species}/{slug}")

    check(not bad_fields, f"all entries have the {len(REQUIRED_FIELDS)} required fields"
          + (f" (missing: {bad_fields[:3]})" if bad_fields else ""))
    check(not bad_utility, "utility_type uses the four sanctioned labels"
          + (f" (bad: {bad_utility[:3]})" if bad_utility else ""))
    check(not bad_markers, "every entry has exactly 3 non-empty physical markers"
          + (f" (bad: {bad_markers[:3]})" if bad_markers else ""))

    # --- advisory matrix -------------------------------------------------
    print("\n[2] Advisory matrix (gender × age)")
    errors = []
    for species, section in (("Cattle", cattle), ("Buffalo", buffalo)):
        for slug in section:
            for gender, age in CONTEXTS:
                adv = get_farmer_advisory(species, slug, gender, age)
                if "error" in adv or not adv.get("headline") or not adv.get("priority_actions"):
                    errors.append(f"{species}/{slug}/{gender}/{age}")
                    continue
                milk = adv["production"]["milk"]
                draught = adv["production"]["draught"]
                if gender == "Male" and age == "Adult" and milk is not None:
                    errors.append(f"male adult still shows milk: {species}/{slug}")
                if gender == "Female" and age == "Adult" and not milk:
                    errors.append(f"female adult missing milk: {species}/{slug}")
                if age == "Calf" and "deworming_schedule" not in adv["health"]:
                    errors.append(f"calf missing deworming: {species}/{slug}")
                if not adv.get("disclaimer"):
                    errors.append(f"missing disclaimer: {species}/{slug}")

    check(not errors, f"all {len(cattle) + len(buffalo)} breeds × {len(CONTEXTS)} contexts build"
          + (f" (errors: {errors[:3]})" if errors else ""))

    # --- aliases ---------------------------------------------------------
    print("\n[3] Alias & lookup handling")
    check(get_breed_knowledge("Buffalo", "Nili Ravi")["slug"] == "nili_ravi",
          "space-separated alias resolves (Nili Ravi)")
    bhad = get_breed_knowledge("Buffalo", "bhadawari")
    check(bhad is not None and bhad["slug"] == "bhadwari",
          "alternate spelling resolves (bhadawari -> bhadwari)")
    check(get_breed_knowledge("Cattle", "hf_cross")["slug"] == "holstein_friesian",
          "crossbreed alias resolves (hf_cross -> holstein_friesian)")
    check(get_breed_knowledge("Cattle", "not_a_breed") is None,
          "unknown breed returns None")

    # --- latency ---------------------------------------------------------
    print("\n[4] Recalculation latency (UI budget < 10 ms)")
    samples = 500
    started = time.perf_counter()
    for _ in range(samples):
        get_farmer_advisory("Buffalo", "murrah", "Female", "Adult")
    avg_ms = (time.perf_counter() - started) / samples * 1000
    check(avg_ms < 10.0, f"{samples} advisories averaged {avg_ms:.3f} ms each")

    stats = knowledge_stats()
    print("\n" + "=" * 62)
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"  curated: {stats['curated']} · "
              f"regional defaults: {stats['regional_default']} · "
              f"file: {os.path.relpath(stats['file'], PROJECT_ROOT)}")
    if failures:
        print(f"  {len(failures)} check(s) FAILED")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
