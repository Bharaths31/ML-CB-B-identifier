"""Manual, folder-based OOD calibration shared by test_model.py and the Flask backend.

The user selects a model and a folder of images named by breed. This module:
  1. Walks the folder (nested ``<breed>/`` folders or flat ``Breed_01.jpg`` files).
  2. Matches each image name to the model's cattle/buffalo class maps.
  3. Runs the selected model over the matched in-distribution images.
  4. Calibrates energy/MSP thresholds and returns breed coverage statistics.

Only matched breed images are used for calibration; unmatched files are reported
back to the UI so the folder can be fixed rather than silently polluting the
thresholds.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from difflib import get_close_matches

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import (
    OOD_BINARY_CONFIDENCE_FLOOR,
    OOD_ENERGY_TEMPERATURE,
    OOD_PERCENTILE,
    PROJECT_ROOT,
)
from src.model import OODDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "graphify-out",
    "outputs",
    "logs",
    ".ipynb_checkpoints",
}

# Spelling variants seen in Kaggle folders / hand-named test images. The
# matcher tries every variant against the model's actual class maps, so it
# works whether the export uses "bhadwari" or "bhadawari", etc.
ALIAS_GROUPS = {
    "baragur": ("baragur", "bargur"),
    "bargur": ("bargur", "baragur"),
    "amruthamahal": ("amruthamahal", "amritmahal"),
    "amritmahal": ("amritmahal", "amruthamahal"),
    "amruthmahal": ("amruthmahal", "amritmahal"),
    "gangatiri": ("gangatiri", "gangatari"),
    "gangatari": ("gangatari", "gangatiri"),
    "ghumusari": ("ghumusari", "ghumsari"),
    "ghumsari": ("ghumsari", "ghumusari"),
    "hallikaru": ("hallikaru", "hallikar"),
    "hallikar": ("hallikar", "hallikaru"),
    "malenadu_gidda": ("malenadu_gidda", "malnad_gidda"),
    "malnad_gidda": ("malnad_gidda", "malenadu_gidda"),
    "poda_thurpu": ("poda_thurpu", "poda_thirupu"),
    "poda_thirupu": ("poda_thirupu", "poda_thurpu"),
    "belahi": ("belahi", "bhelai"),
    "bhelai": ("bhelai", "belahi"),
    "khillar": ("khillar", "khillari"),
    "khillari": ("khillari", "khillar"),
    "bhadawari": ("bhadawari", "bhadwari"),
    "bhadwari": ("bhadwari", "bhadawari"),
    "marathwadi": ("marathwadi", "marathwada"),
    "marathwada": ("marathwada", "marathwadi"),
    "nili-ravi": ("nili_ravi",),
    "nili_ravi": ("nili_ravi",),
    "red-sindhi": ("red_sindhi",),
    "red_sindhi": ("red_sindhi",),
}

COMMON_DIRS = (
    "data/raw",
    "data/raw/cattle",
    "data/raw/buffalo",
    "Testing data",
    "test data ceck",
    "windows data",
    "Sudhipti-Outputs",
    "data",
)


def normalize_breed_key(value: str) -> str:
    """Turn 'Amruthamahal_01.jpg' / 'Gir (2)' / 'gir-bull' into a class-map key."""
    if not value:
        return ""
    text = os.path.splitext(os.path.basename(str(value)))[0].strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[()\[\].]", " ", text)
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = re.sub(r"_(copy|final|edited|new|img|image)$", "", text)
    text = re.sub(r"_\d{1,4}$", "", text)
    text = re.sub(r"^\d{1,4}_", "", text)
    return text


def _candidate_keys(value: str):
    key = normalize_breed_key(value)
    if not key:
        return []
    variants = [key]
    variants.extend(ALIAS_GROUPS.get(key, ()))
    # De-duplicate while preserving order.
    seen = set()
    return [v for v in variants if not (v in seen or seen.add(v))]


def _species_hint(relative_path: str) -> str | None:
    low = relative_path.lower()
    if "buffalo" in low:
        return "buffalo"
    if "cattle" in low:
        return "cattle"
    return None


def _match_in_maps(key: str, class_maps: dict, hint: str | None):
    """Return (species, breed, class_id) for one normalized key, or None."""
    hits = []
    for species in ("cattle", "buffalo"):
        cmap = class_maps.get(species) or {}
        if key in cmap:
            hits.append((species, key, cmap[key]))

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # "bargur" exists in both maps in some exports.
        if hint in ("cattle", "buffalo"):
            for hit in hits:
                if hit[0] == hint:
                    return hit
        return ("both", hits[0][1], None)

    # Spelling fallback for hand-named files.
    fuzzy = []
    for species in ("cattle", "buffalo"):
        cmap = class_maps.get(species) or {}
        close = get_close_matches(key, list(cmap.keys()), n=1, cutoff=0.88)
        if close:
            fuzzy.append((species, close[0], cmap[close[0]]))
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        if hint in ("cattle", "buffalo"):
            for hit in fuzzy:
                if hit[0] == hint:
                    return hit
        return ("both", fuzzy[0][1], None)
    return None


def derive_breed(path: str, data_root: str, class_maps: dict):
    """
    Resolve an image path to (species, breed, class_id).

    Tries the full stem, individual stem tokens, then parent folder names
    (nearest first). This handles both ``cattle/gir/img1.jpg`` and
    ``Testing data/Deoni_01.jpg`` layouts.
    """
    try:
        rel = os.path.relpath(path, data_root)
    except ValueError:
        rel = os.path.basename(path)
    parts = [p for p in rel.split(os.sep) if p not in ("", ".")]
    hint = _species_hint(rel)

    stem = os.path.splitext(parts[-1])[0] if parts else ""
    candidates = []
    candidates.extend(_candidate_keys(stem))
    for token in re.split(r"[_\-\s]+", stem):
        if token:
            candidates.extend(_candidate_keys(token))
    for folder in reversed(parts[:-1]):
        candidates.extend(_candidate_keys(folder))

    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        match = _match_in_maps(key, class_maps, hint)
        if match:
            return match
    return None


@dataclass
class BreedImage:
    path: str
    species: str
    breed: str
    class_id: int | None


class BreedImageDataset(Dataset):
    """Yields (tensor, class_id) so OODDetector.calibrate can consume it."""

    def __init__(self, items, transform):
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        try:
            with Image.open(item.path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Could not read calibration image {item.path}: {exc}") from exc
        label = item.class_id if item.class_id is not None else -1
        return tensor, label


def resolve_data_dir(spec: str | None) -> str | None:
    """Resolve a user-supplied folder path against the project root."""
    if not spec or not str(spec).strip():
        return None
    text = os.path.expanduser(str(spec).strip())
    path = text if os.path.isabs(text) else os.path.join(PROJECT_ROOT, text)
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ValueError(f"Calibration folder not found: {path}")
    return path


def scan_breed_images(data_dir: str, class_maps: dict, max_images: int | None = 500):
    """Walk a folder and resolve breed-named images to class-map entries."""
    if not os.path.isdir(data_dir):
        raise ValueError(f"Calibration folder not found: {data_dir}")

    limit = int(max_images) if max_images else None
    if limit is not None and limit <= 0:
        limit = None

    paths = []
    truncated = False
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = sorted(
            d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")
        )
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            if limit is not None and len(paths) >= limit:
                truncated = True
                break
            paths.append(os.path.join(root, name))
        if truncated:
            break

    items = []
    breed_counts = Counter()
    unknown = Counter()
    unknown_examples = {}
    unreadable = []

    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            unreadable.append(path)
            continue

        match = derive_breed(path, data_dir, class_maps)
        if not match:
            key = normalize_breed_key(path) or "unnamed"
            unknown[key] += 1
            unknown_examples.setdefault(key, os.path.basename(path))
            continue

        species, breed, class_id = match
        items.append(BreedImage(path=path, species=species, breed=breed, class_id=class_id))
        breed_counts[f"{species}:{breed}"] += 1

    unknown_list = [
        {"name": name, "count": count, "example": unknown_examples.get(name)}
        for name, count in unknown.most_common(30)
    ]

    return {
        "data_dir": data_dir,
        "items": items,
        "scanned_images": len(paths),
        "matched_images": len(items),
        # Unreadable files are reported separately; do not double-count them.
        "unmatched_images": max(0, len(paths) - len(items) - len(unreadable)),
        "unreadable_images": len(unreadable),
        "unreadable_files": unreadable[:20],
        "unknown_breeds": unknown_list,
        "breed_counts": dict(sorted(breed_counts.items())),
        "truncated": truncated,
    }


def calibrate_from_folder(
    model,
    data_dir: str,
    class_maps: dict,
    device,
    percentile: float = OOD_PERCENTILE,
    max_images: int | None = 500,
    batch_size: int = 32,
    temperature: float = OOD_ENERGY_TEMPERATURE,
    binary_floor: float = OOD_BINARY_CONFIDENCE_FLOOR,
):
    """Run the selected model over breed-named images and return thresholds + coverage."""
    from src.data_pipeline import _eval_transform

    scan = scan_breed_images(data_dir, class_maps, max_images=max_images)
    if not scan["items"]:
        detail = ", ".join(
            f"{entry['name']} ({entry['count']})" for entry in scan["unknown_breeds"][:8]
        )
        raise ValueError(
            "No breed-labelled images matched the model class maps in "
            f"{data_dir}. Unmatched examples: {detail or 'none'}"
        )

    dataset = BreedImageDataset(scan["items"], _eval_transform())
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(device) if isinstance(device, str) else device
    thresholds = OODDetector.calibrate(
        model,
        loader,
        device,
        percentile=float(percentile),
        temperature=float(temperature),
        binary_floor=float(binary_floor),
    )

    n_samples = int(thresholds.get("calibrated_on_samples") or len(scan["items"]))
    thresholds.update(
        {
            "data_dir": data_dir,
            "n_samples": n_samples,
            "scanned_images": scan["scanned_images"],
            "matched_images": scan["matched_images"],
            "unmatched_images": scan["unmatched_images"],
            "unreadable_images": scan["unreadable_images"],
            "unreadable_files": scan["unreadable_files"],
            "unknown_breeds": scan["unknown_breeds"],
            "breed_counts": scan["breed_counts"],
            "truncated": scan["truncated"],
            "energy_mean": thresholds.get("energy_stats", {}).get("mean"),
            "msp_mean": thresholds.get("msp_stats", {}).get("mean"),
        }
    )
    return thresholds


def list_calibration_dirs():
    """Suggested calibration folders for the React/GUI pickers."""
    dirs = []
    seen = set()
    for rel in COMMON_DIRS:
        path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isdir(path):
            continue
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        dirs.append(
            {
                "label": rel,
                "path": rel.replace(os.sep, "/"),
                "abs_path": abs_path,
            }
        )
    # Breed subfolders under data/raw are convenient shortcuts.
    for species in ("cattle", "buffalo"):
        base = os.path.join(PROJECT_ROOT, "data", "raw", species)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base))[:80]:
            child = os.path.join(base, name)
            if not os.path.isdir(child):
                continue
            abs_path = os.path.abspath(child)
            if abs_path in seen:
                continue
            seen.add(abs_path)
            rel = os.path.relpath(child, PROJECT_ROOT)
            dirs.append(
                {
                    "label": rel.replace(os.sep, "/"),
                    "path": rel.replace(os.sep, "/"),
                    "abs_path": abs_path,
                }
            )
    return dirs
