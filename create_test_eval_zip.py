#!/usr/bin/env python3
"""Create a zip file of test-split images for Colab batch evaluation.

Reads data/splits/test.csv, copies the referenced images into a
    test_images/{cattle,buffalo}/<breed>/*.jpg
structure, and produces test_eval_images.zip ready for upload to
colab/cattle_buffalo_tester.ipynb §6.
"""

import csv
import os
import shutil
import sys
import zipfile
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")
STAGING_DIR = os.path.join(PROJECT_ROOT, "test_images")
OUTPUT_ZIP = os.path.join(PROJECT_ROOT, "test_eval_images.zip")


def main():
    # --- 1. Read test.csv ---
    if not os.path.exists(TEST_CSV):
        print(f"❌ {TEST_CSV} not found. Run training first to generate splits.")
        return 1

    with open(TEST_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"📄 Read {len(rows)} entries from test.csv")

    # --- 2. Copy images into staging directory ---
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    copied = 0
    skipped = 0
    species_counts = Counter()
    breed_counts = Counter()

    for row in rows:
        src_path = row["path"]
        species = row["species"]
        breed = row["breed"]

        if not os.path.exists(src_path):
            print(f"  ⚠️  Missing: {src_path}")
            skipped += 1
            continue

        dest_dir = os.path.join(STAGING_DIR, species, breed)
        os.makedirs(dest_dir, exist_ok=True)

        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(src_path, dest_path)

        copied += 1
        species_counts[species] += 1
        breed_counts[(species, breed)] += 1

    print(f"\n✅ Copied {copied} images ({skipped} skipped)")
    for sp in sorted(species_counts):
        breed_cnt = len([b for (s, b) in breed_counts if s == sp])
        print(f"   {sp.title()}: {species_counts[sp]} images across {breed_cnt} breeds")

    # --- 3. Create zip ---
    if copied == 0:
        print("❌ No images copied — nothing to zip.")
        return 1

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(STAGING_DIR):
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, PROJECT_ROOT)
                zf.write(full_path, arcname)

    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"\n📦 Created: {OUTPUT_ZIP}")
    print(f"   Size: {zip_size_mb:.1f} MB")
    print(f"   Images: {copied}")

    # --- 4. Cleanup staging ---
    shutil.rmtree(STAGING_DIR)
    print(f"🧹 Cleaned up staging directory")

    print(f"\n🚀 Upload test_eval_images.zip to Colab §6 for batch evaluation!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
