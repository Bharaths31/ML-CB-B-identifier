#!/usr/bin/env python
"""Generate the ``data/breed_traits.json`` template for the trait aux-heads.

Lists every breed from ``<split-dir>/{cattle,buffalo}_classes.json`` with EMPTY
trait fields for you to fill in. Values are free-form strings; the trait head
builds one vocabulary per field from the distinct non-empty values it finds, so
you can use any wording (e.g. hump: "prominent" / "moderate" / "absent").

    python scripts/make_trait_template.py                 # write template
    python scripts/make_trait_template.py --force         # overwrite
    python scripts/make_trait_template.py --split-dir data/splits --out data/breed_traits.json

An existing file is never overwritten unless --force is given (so you don't lose
work). This script does not invent any trait values.
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.config import SPLIT_DIR, TRAIT_FIELDS, TRAIT_FILE  # noqa: E402


def build_template(split_dir):
    breeds = {}
    for species in ("cattle", "buffalo"):
        path = os.path.join(split_dir, f"{species}_classes.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found (run prepare_splits first)")
            breeds[species] = {}
            continue
        with open(path) as f:
            classes = json.load(f)
        breeds[species] = {name: {field: "" for field in TRAIT_FIELDS}
                           for name, _ in sorted(classes.items(), key=lambda kv: kv[1])}
    return {
        "_comment": ("Fill in trait values per breed (free-form strings). "
                     "Leave a field empty ('') to ignore it for that breed. "
                     "Empty values are excluded from the masked trait loss."),
        "fields": list(TRAIT_FIELDS),
        "breeds": breeds,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default=SPLIT_DIR)
    ap.add_argument("--out", default=TRAIT_FILE)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing file (may lose filled-in values)")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"[traits] {args.out} already exists — not overwriting "
              f"(use --force). It is safe to keep your filled-in values.")
        return 0

    template = build_template(args.split_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(template, f, indent=2)
    n = sum(len(v) for v in template["breeds"].values())
    print(f"[traits] wrote {args.out} with {n} breeds and fields "
          f"{template['fields']} (all values empty).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
