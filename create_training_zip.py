#!/usr/bin/env python3
"""
Script to create a lightweight standalone training zip package.
Excludes webapp, memory, virtual environments, outputs, and cache files.
"""

import os
import zipfile
from pathlib import Path

# Files and directories to explicitly include for model training
INCLUDES = [
    "src",
    "data/raw",
    "requirements.txt",
    "setup.sh",
    "setup_venv.py",
    "efficientnet_lite2.pth",
    "efficientnet_lite4.pth",
    ".agents",
]

# Patterns to strictly exclude
EXCLUDES = {
    "flutter_app",
    "flutter_app",
    ".venv",
    "venv",
    "__pycache__",
    "outputs",
    "data/splits",
    ".ipynb_checkpoints",
    ".opencode",
}


def should_exclude(rel_path: Path) -> bool:
    """Check if relative path matches any exclusion rule."""
    parts = rel_path.parts
    for part in parts:
        if part in EXCLUDES or part.endswith(".pyc") or part.endswith(".zip"):
            return True
    return False


def create_training_zip(output_zip: str = "training_package.zip"):
    project_root = Path(__file__).parent.resolve()
    target_zip = project_root / output_zip

    print(f"📦 Packaging standalone training bundle into '{output_zip}'...")
    file_count = 0
    total_bytes = 0

    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in INCLUDES:
            item_path = project_root / item
            if not item_path.exists():
                print(f"⚠️ Warning: '{item}' does not exist, skipping.")
                continue

            if item_path.is_file():
                rel_path = item_path.relative_to(project_root)
                if not should_exclude(rel_path):
                    zipf.write(item_path, arcname=str(rel_path))
                    file_count += 1
                    total_bytes += item_path.stat().st_size
                    print(f"  + Added file: {rel_path}")
            elif item_path.is_dir():
                for root, dirs, files in os.walk(item_path):
                    # Filter out excluded subdirectories in-place
                    dirs[:] = [d for d in dirs if d not in EXCLUDES]
                    
                    for file in files:
                        file_full = Path(root) / file
                        rel_path = file_full.relative_to(project_root)
                        if not should_exclude(rel_path):
                            zipf.write(file_full, arcname=str(rel_path))
                            file_count += 1
                            total_bytes += file_full.stat().st_size

    size_mb = total_bytes / (1024 * 1024)
    zip_size_mb = target_zip.stat().st_size / (1024 * 1024)

    print("\n✅ Training package created successfully!")
    print(f"   Destination : {target_zip}")
    print(f"   Files included: {file_count}")
    print(f"   Uncompressed size: {size_mb:.2f} MB")
    print(f"   Compressed size  : {zip_size_mb:.2f} MB")
    print("\n🚀 To train on a new machine:")
    print("   1. Extract the zip: unzip training_package.zip")
    print("   2. Install requirements: pip install -r requirements.txt")
    print("   3. Run training: python -m src.train")


if __name__ == "__main__":
    create_training_zip()
