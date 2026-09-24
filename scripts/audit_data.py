#!/usr/bin/env python
"""Audit data/raw before a training run. REPORT ONLY — never deletes anything.

Checks:
  * per-species, per-breed image counts
  * near-duplicate breed folder names (fuzzy match) that should probably be one
    class, e.g. amritmahal/amruthamahal, hallikar/hallikaru,
    malnad_gidda/malenadu_gidda, gangatiri variants
  * "bargur" (and other names) that exist under BOTH species: counts,
    cross-species duplicate content hashes, sample file paths
  * exact duplicates (md5) and near-duplicates (average-hash Hamming distance)
    across the train/val/test splits, if splits exist
  * unreadable / corrupt image files

Run on the dataset machine:

    python scripts/audit_data.py --data data/raw --split-dir data/splits
    python scripts/audit_data.py --out outputs/audit_<runid>.json
"""

import argparse
import difflib
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_DATA_DIR, SPLIT_DIR  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_breeds(data_root):
    out = {}
    for species in ("cattle", "buffalo"):
        d = os.path.join(data_root, species)
        if not os.path.isdir(d):
            continue
        out[species] = {}
        for breed in sorted(os.listdir(d)):
            bd = os.path.join(d, breed)
            if not os.path.isdir(bd):
                continue
            files = [os.path.join(bd, f) for f in sorted(os.listdir(bd))
                     if f.lower().endswith(IMAGE_EXTS)]
            out[species][breed] = files
    return out


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def ahash(path, size=8):
    """Average hash (perceptual) as a bit array; None if unreadable."""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size, size))
            arr = np.asarray(im, dtype=np.float32)
        return (arr > arr.mean()).flatten()
    except Exception:
        return None


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=RAW_DATA_DIR)
    ap.add_argument("--split-dir", default=SPLIT_DIR)
    ap.add_argument("--name-ratio", type=float, default=0.85,
                    help="fuzzy name similarity threshold (0-1)")
    ap.add_argument("--phash-max-dist", type=int, default=4,
                    help="average-hash Hamming distance for near-duplicates")
    ap.add_argument("--max-phash", type=int, default=0,
                    help="cap images hashed per breed for phash (0 = all)")
    ap.add_argument("--out", default=None, help="JSON report path")
    args = ap.parse_args()

    breeds = list_breeds(args.data)
    if not breeds:
        raise SystemExit(f"no data/raw under {args.data}")

    report = {"counts": {}, "corrupt": [], "fuzzy_name_pairs": [],
              "cross_species_names": {}, "split_duplicates": {},
              "near_duplicates_within_species": []}

    # 1) counts + corrupt files
    for species, by_breed in breeds.items():
        report["counts"][species] = {}
        for breed, files in by_breed.items():
            bad = []
            for f in files:
                try:
                    with Image.open(f) as im:
                        im.verify()
                except Exception as exc:
                    bad.append({"path": f, "error": str(exc)})
            report["counts"][species][breed] = len(files)
            report["corrupt"].extend(bad)
            print(f"{species:8s} {breed:24s} {len(files):5d}"
                  + (f"  !! {len(bad)} corrupt" if bad else ""))

    # 2) fuzzy near-duplicate breed names within each species
    for species, by_breed in breeds.items():
        names = sorted(by_breed)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ratio = difflib.SequenceMatcher(None, names[i], names[j]).ratio()
                if ratio >= args.name_ratio:
                    report["fuzzy_name_pairs"].append({
                        "species": species, "a": names[i], "b": names[j],
                        "ratio": round(ratio, 3),
                        "a_n": len(by_breed[names[i]]),
                        "b_n": len(by_breed[names[j]]),
                    })
    if report["fuzzy_name_pairs"]:
        print("\n[!] fuzzy breed-name pairs (likely the same breed):")
        for p in report["fuzzy_name_pairs"]:
            print(f"    {p['species']}: {p['a']} ({p['a_n']}) <-> "
                  f"{p['b']} ({p['b_n']})  ratio={p['ratio']}")

    # 3) names present under BOTH species (e.g. bargur / baragur)
    cattle_names = set(breeds.get("cattle", {}))
    buffalo_names = set(breeds.get("buffalo", {}))
    shared = cattle_names & buffalo_names
    for name in sorted(shared):
        cfiles = breeds["cattle"][name]
        bfiles = breeds["buffalo"][name]
        chash = defaultdict(list)
        bhash = defaultdict(list)
        for f in cfiles:
            h = md5(f)
            if h:
                chash[h].append(f)
        for f in bfiles:
            h = md5(f)
            if h:
                bhash[h].append(f)
        dup_hashes = sorted(set(chash) & set(bhash))
        report["cross_species_names"][name] = {
            "cattle_n": len(cfiles), "buffalo_n": len(bfiles),
            "cattle_samples": cfiles[:3], "buffalo_samples": bfiles[:3],
            "duplicate_hashes": len(dup_hashes),
            "duplicate_examples": [
                {"cattle": chash[h][0], "buffalo": bhash[h][0]}
                for h in dup_hashes[:5]],
        }
    if shared:
        print("\n[!] breed names under BOTH species:")
        for name, info in report["cross_species_names"].items():
            print(f"    {name}: cattle={info['cattle_n']} buffalo={info['buffalo_n']} "
                  f"cross-species duplicate images={info['duplicate_hashes']}")
            for ex in info["duplicate_examples"]:
                print(f"        dup: {ex['cattle']}  ==  {ex['buffalo']}")

    # 4) duplicates across splits (if splits exist)
    split_files = {}
    for split in ("train", "val", "test"):
        p = os.path.join(args.split_dir, f"{split}.csv")
        if os.path.exists(p):
            import pandas as pd
            df = pd.read_csv(p)
            split_files[split] = list(df["path"]) if "path" in df else []
    if split_files:
        hashes = {s: defaultdict(list) for s in split_files}
        for split, files in split_files.items():
            for f in files:
                h = md5(f)
                if h:
                    hashes[split][h].append(f)
        names = list(split_files)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                dup = sorted(set(hashes[a]) & set(hashes[b]))
                if dup:
                    report["split_duplicates"][f"{a}__{b}"] = {
                        "n_hashes": len(dup),
                        "examples": [{"a": hashes[a][h][0], "b": hashes[b][h][0]}
                                     for h in dup[:5]],
                    }
        if report["split_duplicates"]:
            print("\n[!] exact-duplicate images leaked across splits:")
            for k, v in report["split_duplicates"].items():
                print(f"    {k}: {v['n_hashes']} shared hashes")

    # 5) near-duplicates within a species (average hash)
    for species, by_breed in breeds.items():
        seen = []
        for breed, files in by_breed.items():
            files = files if args.max_phash <= 0 else files[:args.max_phash]
            hashed = []
            for f in files:
                h = ahash(f)
                if h is not None:
                    hashed.append((f, h))
            for fi, hi in hashed:
                for fj, hj in seen:
                    if hamming(hi, hj) <= args.phash_max_dist:
                        report["near_duplicates_within_species"].append(
                            {"species": species, "a": fi, "b": fj,
                             "dist": hamming(hi, hj)})
            seen.extend(hashed)
    n_near = len(report["near_duplicates_within_species"])
    if n_near:
        print(f"\n[!] {n_near} near-duplicate image pairs within species "
              f"(showing up to 10):")
        for d in report["near_duplicates_within_species"][:10]:
            print(f"    {d['species']} dist={d['dist']}: {d['a']} ~ {d['b']}")

    # summary
    total = sum(sum(v.values()) for v in report["counts"].values())
    print(f"\n[audit] total images={total}, breeds="
          f"{ {s: len(v) for s, v in report['counts'].items()} }, "
          f"corrupt={len(report['corrupt'])}, "
          f"fuzzy_pairs={len(report['fuzzy_name_pairs'])}, "
          f"shared_names={len(shared)}")
    print("[audit] REPORT ONLY — nothing was modified or deleted.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[audit] report -> {args.out}")


if __name__ == "__main__":
    main()
