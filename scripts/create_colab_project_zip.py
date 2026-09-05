#!/usr/bin/env python3
"""
Create colab_project.zip containing all source code needed for Colab training.
Usage: python scripts/create_colab_project_zip.py
Outputs: colab_project.zip in project root
"""
import zipfile
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
ARCHIVE_OUT = PROJECT_ROOT / "colab_project.zip"

# Files/dirs to include
INCLUDE = [
    "src",
    "scripts",
    "data_collection.py",
    "efficientnet_lite2.pth",
    "efficientnet_lite4.pth",
    "requirements.txt",
    "setup.sh",
    "setup_venv.py",
]

# Files/dirs to exclude
EXCLUDE = [
    "__pycache__",
    ".pyc",
    ".git",
    ".venv",
    "data",
    "outputs",
    "flutter_app",
    "webapp",
    "memory",
    "archive.zip",
    "colab_project.zip",
    "*.ipynb",
    ".opencode",
    "ADR.md",
]


def should_include(path: pathlib.Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT)
    # Check exclude patterns
    for pattern in EXCLUDE:
        if pattern in str(rel) or path.name == pattern:
            return False
    return True


def main():
    print(f"📦  Creating colab_project.zip from {PROJECT_ROOT}...")

    if ARCHIVE_OUT.exists():
        ARCHIVE_OUT.unlink()

    with zipfile.ZipFile(ARCHIVE_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            src = PROJECT_ROOT / item
            if not src.exists():
                print(f"  ⚠️  Skipping missing: {item}")
                continue

            if src.is_file():
                if should_include(src):
                    zf.write(src, item)
                    print(f"  ✓ {item}")
            else:
                for file_path in src.rglob("*"):
                    if file_path.is_file() and should_include(file_path):
                        arcname = file_path.relative_to(PROJECT_ROOT)
                        zf.write(file_path, str(arcname))
                        print(f"  ✓ {arcname}")

    size_mb = ARCHIVE_OUT.stat().st_size / (1024 * 1024)
    print(f"✅  Created {ARCHIVE_OUT} ({size_mb:.1f} MB)")
    print(f"   Upload this to Colab, then run: !unzip -q /content/colab_project.zip -d /content/colab_project")
    return 0


if __name__ == "__main__":
    exit(main())