#!/usr/bin/env python3
"""
Create colab_project.zip containing only the files needed for Colab training.
Usage: python scripts/create_colab_project_zip.py [--include-lite4]
Outputs: colab_project.zip in project root
"""
import argparse
import zipfile
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
ARCHIVE_OUT = PROJECT_ROOT / "colab_project.zip"

# Core files always included
CORE_INCLUDES = [
    "src/__init__.py",
    "src/config.py",
    "src/data_pipeline.py",
    "src/model.py",
    "src/cbam.py",
    "src/efficientnet_lite.py",
    "src/train.py",
    "src/metrics.py",
    "src/evaluate.py",
    "src/export.py",
    "src/verify.py",
    "requirements.txt",
    "efficientnet_lite2.pth",
]


def main():
    parser = argparse.ArgumentParser(
        description="Create colab_project.zip for Colab training")
    parser.add_argument("--include-lite4", action="store_true",
                        help="include efficientnet_lite4.pth (~50 MB)")
    parser.add_argument("--output", default=str(ARCHIVE_OUT),
                        help="output zip path")
    args = parser.parse_args()

    includes = list(CORE_INCLUDES)
    if args.include_lite4:
        includes.append("efficientnet_lite4.pth")

    out_path = pathlib.Path(args.output)
    if out_path.exists():
        out_path.unlink()

    print(f"📦  Creating {out_path.name}...")
    file_count = 0
    total_bytes = 0

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in includes:
            src = PROJECT_ROOT / item
            if not src.exists():
                print(f"  ⚠️  Skipping missing: {item}")
                continue
            zf.write(src, item)
            size_mb = src.stat().st_size / (1024 * 1024)
            total_bytes += src.stat().st_size
            file_count += 1
            print(f"  ✓ {item} ({size_mb:.2f} MB)")

    zip_size = out_path.stat().st_size / (1024 * 1024)
    total_mb = total_bytes / (1024 * 1024)

    print(f"\n✅ Created {out_path}")
    print(f"   Files:            {file_count}")
    print(f"   Uncompressed:     {total_mb:.1f} MB")
    print(f"   Compressed:       {zip_size:.1f} MB")
    print(f"\n🚀 Upload to Colab or Google Drive, then run the notebook")
    return 0


if __name__ == "__main__":
    exit(main())