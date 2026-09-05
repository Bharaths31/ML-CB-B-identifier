#!/usr/bin/env python3
"""
Create archive.zip for Colab upload from local organized dataset.
Usage: python scripts/create_colab_archive.py
Outputs: archive.zip in project root (ready for Colab upload)
"""
import zipfile
import pathlib
import shutil

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
ARCHIVE_OUT = PROJECT_ROOT / "archive.zip"


def main():
    if not DATA_RAW.exists():
        print(f"❌  Data directory not found: {DATA_RAW}")
        return 1

    cattle_dir = DATA_RAW / "cattle"
    buffalo_dir = DATA_RAW / "buffalo"

    if not cattle_dir.exists() and not buffalo_dir.exists():
        print(f"❌  No cattle/ or buffalo/ subdirectories in {DATA_RAW}")
        print("   Run: python data_collection.py --source organize --input <your_raw_images> --output data/raw")
        return 1

    print(f"📦  Creating archive.zip from {DATA_RAW}...")

    if ARCHIVE_OUT.exists():
        ARCHIVE_OUT.unlink()

    with zipfile.ZipFile(ARCHIVE_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add cattle images
        if cattle_dir.exists():
            for img_path in cattle_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    arcname = img_path.relative_to(DATA_RAW)
                    zf.write(img_path, arcname)

        # Add buffalo images
        if buffalo_dir.exists():
            for img_path in buffalo_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    arcname = img_path.relative_to(DATA_RAW)
                    zf.write(img_path, arcname)

    size_mb = ARCHIVE_OUT.stat().st_size / (1024 * 1024)
    print(f"✅  Created {ARCHIVE_OUT} ({size_mb:.1f} MB)")
    print(f"   Upload this file to Colab via Files panel, then run Option B cell")
    return 0


if __name__ == "__main__":
    exit(main())