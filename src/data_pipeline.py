import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from .config import (ALLOW_HUE, AUG_COLOR_JITTER, AUG_HORIZONTAL_FLIP,
                     AUG_RANDAUGMENT, AUG_RANDOM_RESIZED_CROP, BREED_AUG_POLICY,
                     CACHE_IMAGES, COLOR_JITTER_BRIGHTNESS,
                     COLOR_JITTER_CONTRAST, COLOR_JITTER_HUE,
                     COLOR_JITTER_SATURATION, CUTMIX_ALPHA, DEDUP_HAMMING,
                     DEDUP_SPLITS, EVAL_MATCH_TRAIN_RESOLUTION,
                     EVAL_PAD_TO_SQUARE, HASH_CACHE_NAME,
                     EXPECTED_BUFFALO_BREEDS, EXPECTED_CATTLE_BREEDS,
                     HALF_DATA_RATIO, IMAGE_SIZE, MIX_SAME_SPECIES, MIXUP_ALPHA,
                     NUM_BUFFALO_BREEDS, NUM_CATTLE_BREEDS, QUARTER_DATA_RATIO,
                     RANDAUGMENT_MAGNITUDE, RANDAUGMENT_OPS, RARE_CLASS_THRESHOLD,
                     RAW_DATA_DIR, RRC_RATIO, RRC_SCALE, SAMPLER_BETA,
                     SMOKE_SAMPLES_PER_BREED, SPLIT_DIR, TEST_RATIO,
                     TRAIN_PAD_TO_SQUARE, TRAIN_RESIZE, VAL_MIN_WARN, VAL_RATIO,
                     IMAGENET_MEAN, IMAGENET_STD, CUTMIX_MIXUP_PROB)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _log_event(event, **fields):
    """Best-effort structured logging (never raises if the logger is absent)."""
    try:
        from .run_logger import log_event
        return log_event(event, **fields)
    except Exception:
        return None


def _stratified_split(df, rng):
    """Per-breed train/val/test index lists with hard minimums.

    A plain round() split starves the long tail: a 10-image breed under an
    85/10/5 split gets 9/1/0, so the test set never sees it. We guarantee at
    least one val and one test image whenever a breed has >= 3 images, and at
    least two training images, by borrowing from the training pool.
    """
    train_rows, val_rows, test_rows = [], [], []
    groups = df.groupby(["species", "breed"])
    for (_, _), group in tqdm(groups, desc="splitting breeds", leave=False,
                              unit="breed"):
        idxs = list(group.index)
        rng.shuffle(idxs)
        n = len(idxs)
        if n <= 0:
            continue
        if n == 1:
            train_rows.extend(idxs)
            continue
        if n == 2:
            # One train, one test; no val (avoid any image appearing twice).
            train_rows.extend(idxs[:1])
            test_rows.extend(idxs[1:])
            continue
        n_val = max(1, int(round(n * VAL_RATIO)))
        n_test = max(1, int(round(n * TEST_RATIO)))
        n_train = n - n_val - n_test
        if n_train < 2:
            # Reserve one more image for train, trimming the test pool first.
            deficit = 2 - n_train
            take_test = min(deficit, n_test - 1)
            n_test -= take_test
            deficit -= take_test
            n_train = n - n_val - n_test
            if deficit > 0:
                n_val = max(1, n_val - deficit)
                n_train = n - n_val - n_test
        train_rows.extend(idxs[:n_train])
        val_rows.extend(idxs[n_train:n_train + n_val])
        test_rows.extend(idxs[n_train + n_val:])
    return train_rows, val_rows, test_rows


def _dhash(path, hash_size=8):
    """64-bit difference hash (no external deps). None if unreadable."""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.int16)
        bits = arr[:, 1:] > arr[:, :-1]
        val = 0
        for b in bits.flatten():
            val = (val << 1) | int(b)
        return val
    except Exception:
        return None


def _hamming(a, b):
    return bin(a ^ b).count("1") if (a is not None and b is not None) else 64


def compute_hashes(df, cache_path):
    """Return {path: dhash} for every row, caching results in a CSV."""
    hashes = {}
    if os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path)
            for p, h in zip(cached["path"], cached["hash"]):
                hashes[p] = int(h)
        except Exception:
            hashes = {}
    missing = [p for p in df["path"] if p not in hashes]
    if missing:
        for p in tqdm(missing, desc="hashing images", leave=False, unit="img"):
            h = _dhash(p)
            if h is not None:
                hashes[p] = h
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            pd.DataFrame({"path": list(hashes), "hash": list(hashes.values())}) \
                .to_csv(cache_path, index=False)
        except Exception:
            pass
    return hashes


def _groups_by_hash(items, hamming):
    """Union-find groups of (index, hash) within `hamming` distance."""
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        hi = items[i][1]
        if hi is None:
            continue
        for j in range(i + 1, len(items)):
            hj = items[j][1]
            if hj is None:
                continue
            if _hamming(hi, hj) <= hamming:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(items[i][0])
    return list(groups.values())


def _stratified_split_dedup(df, rng, hashes, hamming=DEDUP_HAMMING):
    """Like ``_stratified_split`` but near-duplicate groups stay in one split.

    Duplicates are grouped WITHIN each (species, breed) so a near-duplicate
    image can never appear in a different split than its twin (which would
    inflate validation/test accuracy). Long-tail minimums are preserved.
    """
    train_rows, val_rows, test_rows = [], [], []
    for (_, _), group in tqdm(df.groupby(["species", "breed"]),
                              desc="splitting (dedup)", leave=False, unit="breed"):
        idxs = list(group.index)
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_rows.extend(idxs)
            continue
        if n == 2:
            train_rows.append(idxs[0])
            test_rows.append(idxs[1])
            continue
        n_val = max(1, int(round(n * VAL_RATIO)))
        n_test = max(1, int(round(n * TEST_RATIO)))
        n_train = n - n_val - n_test
        if n_train < 2:
            deficit = 2 - n_train
            take = min(deficit, n_test - 1)
            n_test -= take
            deficit -= take
            n_train = n - n_val - n_test
            if deficit > 0:
                n_val = max(1, n_val - deficit)
                n_train = n - n_val - n_test

        items = [(idx, hashes.get(df.at[idx, "path"])) for idx in idxs]
        groups = _groups_by_hash(items, hamming)
        groups.sort(key=len, reverse=True)

        cap = [n_train, n_val, n_test]
        cur = [0, 0, 0]
        buckets = [[], [], []]      # buckets of groups
        for g in groups:
            k = max(range(3), key=lambda s: cap[s] - cur[s])
            buckets[k].append(g)
            cur[k] += len(g)

        def move_smallest(src, dst):
            if not buckets[src]:
                return
            gi = min(range(len(buckets[src])), key=lambda i: len(buckets[src][i]))
            g = buckets[src].pop(gi)
            buckets[dst].append(g)
            cur[src] -= len(g)
            cur[dst] += len(g)

        if cur[1] == 0:
            move_smallest(0, 1)
        if cur[2] == 0:
            move_smallest(0, 2)

        for k, out in ((0, train_rows), (1, val_rows), (2, test_rows)):
            for g in buckets[k]:
                out.extend(g)
    return train_rows, val_rows, test_rows


def _collect_rows(data_root):
    rows = []
    for species, binary in (("cattle", 0), ("buffalo", 1)):
        species_dir = None
        for root, dirs, _ in os.walk(data_root):
            if species in dirs:
                species_dir = os.path.join(root, species)
                break
                
        if not species_dir or not os.path.isdir(species_dir):
            print(f"[data] missing directory: {species}")
            continue
        breeds = sorted(os.listdir(species_dir))
        for breed in tqdm(breeds, desc=f"scanning {species}", leave=False,
                          unit="breed"):
            breed_dir = os.path.join(species_dir, breed)
            if not os.path.isdir(breed_dir):
                continue
            names = sorted(os.listdir(breed_dir))
            for name in names:
                if name.lower().endswith(IMAGE_EXTS):
                    rows.append((os.path.join(breed_dir, name), species, breed, binary))
    return rows


def prepare_splits(data_root=RAW_DATA_DIR, split_dir=SPLIT_DIR, dedup=None):
    os.makedirs(split_dir, exist_ok=True)
    rows = _collect_rows(data_root)
    if not rows:
        print(f"[data] no images found under {data_root}")
        print("[data] expected layout: data/raw/cattle/<breed>/*.jpg and "
              "data/raw/buffalo/<breed>/*.jpg")
        return None

    df = pd.DataFrame(rows, columns=["path", "species", "breed", "binary_label"])
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    rng = random.Random(42)
    if dedup is None:
        dedup = DEDUP_SPLITS
    if dedup:
        print("[data] group-aware (dedup) splits enabled — near-duplicates "
              "stay within one split")
        hashes = compute_hashes(df, os.path.join(split_dir, HASH_CACHE_NAME))
        train_rows, val_rows, test_rows = _stratified_split_dedup(df, rng, hashes)
    else:
        train_rows, val_rows, test_rows = _stratified_split(df, rng)

    def save(name, idxs):
        out = os.path.join(split_dir, f"{name}.csv")
        df.loc[idxs].to_csv(out, index=False)
        return len(idxs)

    n_train = save("train", train_rows)
    n_val = save("val", val_rows)
    n_test = save("test", test_rows)

    cattle_breeds = sorted(df.loc[df.species == "cattle", "breed"].unique())
    buffalo_breeds = sorted(df.loc[df.species == "buffalo", "breed"].unique())
    cattle_classes = {b: i for i, b in enumerate(cattle_breeds)}
    buffalo_classes = {b: i for i, b in enumerate(buffalo_breeds)}
    with open(os.path.join(split_dir, "cattle_classes.json"), "w") as f:
        json.dump(cattle_classes, f, indent=2)
    with open(os.path.join(split_dir, "buffalo_classes.json"), "w") as f:
        json.dump(buffalo_classes, f, indent=2)

    summary = {
        "images": len(df),
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "cattle_breeds": len(cattle_breeds),
        "buffalo_breeds": len(buffalo_breeds),
    }
    print(f"[data] images={len(df)} train={n_train} val={n_val} test={n_test} "
          f"cattle_breeds={len(cattle_breeds)} buffalo_breeds={len(buffalo_breeds)}")
    _validate_class_counts(cattle_breeds, buffalo_breeds)

    # Validation-noise check: a breed with < VAL_MIN_WARN val images makes
    # checkpoint selection noisy.
    val_counts = df.loc[val_rows].groupby(["species", "breed"])["path"].count()
    if len(val_counts):
        low = val_counts[val_counts < VAL_MIN_WARN]
        print(f"[data] val images/breed: min={int(val_counts.min())} "
              f"median={val_counts.median():.0f} "
              f"(breeds<{VAL_MIN_WARN}: {len(low)})")
        if len(low):
            print(f"[data] WARNING: {len(low)} breed(s) have <{VAL_MIN_WARN} val "
                  f"images — validation/selection will be noisy for them")
        summary["val_min_per_breed"] = int(val_counts.min())
        summary["val_median_per_breed"] = float(val_counts.median())

    _log_event("splits_ready", category="data", mode="full", dedup=bool(dedup),
               **summary,
               shared_names=sorted(set(cattle_breeds) & set(buffalo_breeds)))
    return summary


def prepare_half_splits(data_root=RAW_DATA_DIR, split_dir=SPLIT_DIR,
                        ratio=HALF_DATA_RATIO):
    """Create a dataset using a fraction of images per breed for faster training.

    Uses the same 70/15/15 stratified split (with long-tail minimums) as full
    training but on a randomly-sampled subset. Class maps include ALL breeds so
    the model architecture stays identical.
    """
    os.makedirs(split_dir, exist_ok=True)
    rows = _collect_rows(data_root)
    if not rows:
        print(f"[data] no images found under {data_root}")
        return None

    df = pd.DataFrame(rows, columns=["path", "species", "breed", "binary_label"])
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    rng = random.Random(42)
    sampled_rows = []
    breeds = sorted(df["breed"].unique())
    for breed in tqdm(breeds, desc="sampling half-data", leave=False,
                      unit="breed"):
        group = df[df["breed"] == breed]
        idxs = list(group.index)
        rng.shuffle(idxs)
        pick = max(2, int(len(idxs) * ratio))  # at least 2 images per breed
        sampled_rows.extend(idxs[:pick])

    half_df = df.loc[sampled_rows].reset_index(drop=True)

    # Apply the same stratified split with long-tail minimums
    train_rows, val_rows, test_rows = _stratified_split(half_df, rng)

    def save(name, idxs):
        out = os.path.join(split_dir, f"{name}.csv")
        half_df.loc[idxs].to_csv(out, index=False)
        return len(idxs)

    n_train = save("train", train_rows)
    n_val = save("val", val_rows)
    n_test = save("test", test_rows)

    # Class maps from ALL breeds (full dataset) for consistent model heads
    cattle_breeds = sorted(df.loc[df.species == "cattle", "breed"].unique())
    buffalo_breeds = sorted(df.loc[df.species == "buffalo", "breed"].unique())
    cattle_classes = {b: i for i, b in enumerate(cattle_breeds)}
    buffalo_classes = {b: i for i, b in enumerate(buffalo_breeds)}
    with open(os.path.join(split_dir, "cattle_classes.json"), "w") as f:
        json.dump(cattle_classes, f, indent=2)
    with open(os.path.join(split_dir, "buffalo_classes.json"), "w") as f:
        json.dump(buffalo_classes, f, indent=2)

    summary = {
        "images": len(half_df),
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "cattle_breeds": len(cattle_breeds),
        "buffalo_breeds": len(buffalo_breeds),
        "half_data": True,
        "ratio": ratio,
    }
    print(f"[data] half-data: {len(half_df)} images ({ratio*100:.0f}%/breed) "
          f"train={n_train} val={n_val} test={n_test}")
    _validate_class_counts(cattle_breeds, buffalo_breeds)
    return summary


def prepare_quarter_splits(data_root=RAW_DATA_DIR, split_dir=SPLIT_DIR,
                           ratio=QUARTER_DATA_RATIO):
    """Create a dataset using 25% of images per breed for very fast training.

    Identical to prepare_half_splits but uses 25% instead of 50%.  Class maps
    still include ALL breeds so model architecture stays identical to full training.
    """
    os.makedirs(split_dir, exist_ok=True)
    rows = _collect_rows(data_root)
    if not rows:
        print(f"[data] no images found under {data_root}")
        return None

    df = pd.DataFrame(rows, columns=["path", "species", "breed", "binary_label"])
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    rng = random.Random(42)
    sampled_rows = []
    breeds = sorted(df["breed"].unique())
    for breed in tqdm(breeds, desc="sampling quarter-data", leave=False,
                      unit="breed"):
        group = df[df["breed"] == breed]
        idxs = list(group.index)
        rng.shuffle(idxs)
        pick = max(2, int(len(idxs) * ratio))  # at least 2 images per breed
        sampled_rows.extend(idxs[:pick])

    quarter_df = df.loc[sampled_rows].reset_index(drop=True)

    # Apply the same stratified split with long-tail minimums
    train_rows, val_rows, test_rows = _stratified_split(quarter_df, rng)

    def save(name, idxs):
        out = os.path.join(split_dir, f"{name}.csv")
        quarter_df.loc[idxs].to_csv(out, index=False)
        return len(idxs)

    n_train = save("train", train_rows)
    n_val   = save("val",   val_rows)
    n_test  = save("test",  test_rows)

    # Class maps from ALL breeds (full dataset) for consistent model heads
    cattle_breeds  = sorted(df.loc[df.species == "cattle",  "breed"].unique())
    buffalo_breeds = sorted(df.loc[df.species == "buffalo", "breed"].unique())
    cattle_classes  = {b: i for i, b in enumerate(cattle_breeds)}
    buffalo_classes = {b: i for i, b in enumerate(buffalo_breeds)}
    with open(os.path.join(split_dir, "cattle_classes.json"),  "w") as f:
        json.dump(cattle_classes, f, indent=2)
    with open(os.path.join(split_dir, "buffalo_classes.json"), "w") as f:
        json.dump(buffalo_classes, f, indent=2)

    summary = {
        "images":         len(quarter_df),
        "train":          n_train,
        "val":            n_val,
        "test":           n_test,
        "cattle_breeds":  len(cattle_breeds),
        "buffalo_breeds": len(buffalo_breeds),
        "quarter_data":   True,
        "ratio":          ratio,
    }
    print(f"[data] quarter-data: {len(quarter_df)} images ({ratio*100:.0f}%/breed) "
          f"train={n_train} val={n_val} test={n_test}")
    _validate_class_counts(cattle_breeds, buffalo_breeds)
    return summary


def prepare_smoke_splits(data_root=RAW_DATA_DIR, split_dir=SPLIT_DIR,
                         samples_per_breed=SMOKE_SAMPLES_PER_BREED):
    """Create a tiny dataset for smoke tests by sampling a few images per breed.

    This produces real training signal (unlike 2-batch from full set) while
    being fast enough for CI / quick sanity checks.
    """
    os.makedirs(split_dir, exist_ok=True)
    rows = _collect_rows(data_root)
    if not rows:
        print(f"[smoke] no images found under {data_root}")
        return None

    df = pd.DataFrame(rows, columns=["path", "species", "breed", "binary_label"])
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    rng = random.Random(42)
    sampled_rows = []
    breeds = sorted(df["breed"].unique())
    for breed in tqdm(breeds, desc="sampling smoke data", leave=False,
                      unit="breed"):
        group = df[df["breed"] == breed]
        idxs = list(group.index)
        rng.shuffle(idxs)
        pick = min(samples_per_breed, len(idxs))
        sampled_rows.extend(idxs[:pick])

    smoke_df = df.loc[sampled_rows].reset_index(drop=True)

    # Split sampled data: 60% train, 20% val, 20% test (small set needs
    # bigger val/test ratio to have at least 1 image per split per breed)
    train_rows, val_rows, test_rows = [], [], []
    for breed in smoke_df["breed"].unique():
        group = smoke_df[smoke_df["breed"] == breed]
        idxs = list(group.index)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(round(n * 0.6)))
        n_val = max(1, int(round(n * 0.2)))
        for i, idx in enumerate(idxs):
            if i < n_train:
                train_rows.append(idx)
            elif i < n_train + n_val:
                val_rows.append(idx)
            else:
                test_rows.append(idx)
    # Ensure test has at least something
    if not test_rows and val_rows:
        test_rows = val_rows[:1]

    def save(name, idxs):
        out = os.path.join(split_dir, f"{name}.csv")
        smoke_df.loc[idxs].to_csv(out, index=False)
        return len(idxs)

    n_train = save("train", train_rows)
    n_val = save("val", val_rows)
    n_test = save("test", test_rows)

    # Still need class maps from ALL breeds so model heads have correct size
    all_df = df
    cattle_breeds = sorted(all_df.loc[all_df.species == "cattle", "breed"].unique())
    buffalo_breeds = sorted(all_df.loc[all_df.species == "buffalo", "breed"].unique())
    cattle_classes = {b: i for i, b in enumerate(cattle_breeds)}
    buffalo_classes = {b: i for i, b in enumerate(buffalo_breeds)}
    with open(os.path.join(split_dir, "cattle_classes.json"), "w") as f:
        json.dump(cattle_classes, f, indent=2)
    with open(os.path.join(split_dir, "buffalo_classes.json"), "w") as f:
        json.dump(buffalo_classes, f, indent=2)

    summary = {
        "images": len(smoke_df),
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "cattle_breeds": len(cattle_breeds),
        "buffalo_breeds": len(buffalo_breeds),
        "smoke": True,
        "samples_per_breed": samples_per_breed,
    }
    print(f"[smoke] mini-dataset: {len(smoke_df)} images "
          f"({samples_per_breed}/breed) train={n_train} val={n_val} test={n_test}")
    return summary


class _PadToSquare:
    """Resize the LONG side to ``size`` and pad the short side to a square.

    A 4:3 photo loses ~25% of its width to CenterCrop; padding keeps the whole
    body. Fill is the ImageNet mean in 8-bit, which becomes ~0 after Normalize.
    """

    def __init__(self, size, fill=None):
        self.size = size
        self.fill = fill or tuple(int(round(m * 255)) for m in IMAGENET_MEAN)

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

    def __repr__(self):
        return f"PadToSquare({self.size}, fill={self.fill})"


def resolve_breed_augment(breed, base_augment):
    """Merge per-breed augmentation overrides on top of the base flags."""
    policy = dict(base_augment)
    policy.update(BREED_AUG_POLICY.get(breed, {}))
    return policy


def _color_jitter(allow_hue=False):
    hue = max(COLOR_JITTER_HUE, 0.1) if allow_hue else COLOR_JITTER_HUE
    return transforms.ColorJitter(brightness=COLOR_JITTER_BRIGHTNESS,
                                  contrast=COLOR_JITTER_CONTRAST,
                                  saturation=COLOR_JITTER_SATURATION, hue=hue)


def _train_transform(augment=None, pad=None, allow_hue=False):
    """Build the training transform.

    Augmentation is OFF by default (see ``AUG_*`` / ``MIX_ENABLED`` in config)
    because heavy stochastic augmentation hurts fine-grained breed
    identification on a long tail. When no augmentation block is enabled the
    train transform is identical to the eval transform.
    """
    augment = augment or {}
    if pad is None:
        pad = TRAIN_PAD_TO_SQUARE
    rrc = augment.get("random_resized_crop", AUG_RANDOM_RESIZED_CROP)
    flip = augment.get("horizontal_flip", AUG_HORIZONTAL_FLIP)
    color_jitter = augment.get("color_jitter", AUG_COLOR_JITTER)
    randaugment = augment.get("randaugment", AUG_RANDAUGMENT)

    if not (rrc or flip or color_jitter or randaugment):
        return _eval_transform(pad=pad)

    blocks = []
    if pad:
        blocks.append(_PadToSquare(IMAGE_SIZE))
    elif rrc:
        blocks += [transforms.Resize(TRAIN_RESIZE),  # 288px for scale variation
                   transforms.RandomResizedCrop(IMAGE_SIZE, scale=RRC_SCALE,
                                                ratio=RRC_RATIO)]
    else:
        blocks += [transforms.Resize(IMAGE_SIZE),
                   transforms.CenterCrop(IMAGE_SIZE)]
    if flip:
        blocks.append(transforms.RandomHorizontalFlip())
    if color_jitter:
        blocks.append(_color_jitter(allow_hue))
    if randaugment:
        blocks.append(transforms.RandAugment(num_ops=RANDAUGMENT_OPS,
                                             magnitude=RANDAUGMENT_MAGNITUDE))
    blocks += [transforms.ToTensor(),
               transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(blocks)


def _eval_transform(pad=None):
    # Default matches the Flutter preprocessor and _MobileOutputs exactly:
    # shortest-side resize to 260, then center-crop 260.
    # EVAL_MATCH_TRAIN_RESOLUTION=True instead resizes to 288 then center-crops
    # 260 (same object scale the training crop is drawn from) — compare on the
    # GPU machine before changing the default. pad=True keeps the whole frame.
    if pad is None:
        pad = EVAL_PAD_TO_SQUARE
    if pad:
        blocks = [_PadToSquare(IMAGE_SIZE)]
    else:
        resize = TRAIN_RESIZE if EVAL_MATCH_TRAIN_RESOLUTION else IMAGE_SIZE
        blocks = [transforms.Resize(resize), transforms.CenterCrop(IMAGE_SIZE)]
    blocks += [transforms.ToTensor(),
               transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(blocks)


def describe_transform(transform):
    """Human-readable list of a Compose's ops (for startup logging)."""
    if transform is None:
        return "none"
    ops = getattr(transform, "transforms", [transform])
    return " -> ".join(repr(op) for op in ops)


class CattleBuffaloDataset(Dataset):
    def __init__(self, manifest, cattle_classes, buffalo_classes, transform=None,
                 cache_images=False, augment=None, breed_augment=False, pad=None,
                 allow_hue=False):
        self.manifest = manifest.reset_index(drop=True)
        self.cattle_classes = cattle_classes
        self.buffalo_classes = buffalo_classes
        self.transform = transform or _eval_transform(pad=pad)
        self._use_cache = cache_images
        self._image_cache = {}
        # Per-breed augmentation: when `augment` is non-empty the transform is
        # selected per sample by breed (coat-colour breeds skip colour jitter).
        self._augment = dict(augment or {})
        self._breed_augment = breed_augment
        self._pad = pad
        self._allow_hue = allow_hue
        self._transform_cache = {}

    def _transform_for(self, breed):
        if not self._augment:
            return self.transform
        aug = (resolve_breed_augment(breed, self._augment)
               if self._breed_augment else dict(self._augment))
        key = (tuple(sorted((k, bool(v)) for k, v in aug.items())),
               bool(self._pad), bool(self._allow_hue))
        tf = self._transform_cache.get(key)
        if tf is None:
            tf = _train_transform(aug, pad=self._pad, allow_hue=self._allow_hue)
            self._transform_cache[key] = tf
        return tf

    def enable_cache(self):
        self._use_cache = True

    def disable_cache(self):
        self._use_cache = False
        self._image_cache.clear()

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        if self._use_cache and idx in self._image_cache:
            img_bytes = self._image_cache[idx]
            import io
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            with open(row["path"], "rb") as f:
                img_bytes = f.read()
            if self._use_cache:
                self._image_cache[idx] = img_bytes
            import io
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        transform = self._transform_for(row["breed"])
        if transform is not None:
            image = transform(image)
        is_cattle = row["binary_label"] == 0
        if is_cattle:
            cattle_onehot = torch.zeros(NUM_CATTLE_BREEDS)
            cattle_onehot[self.cattle_classes[row["breed"]]] = 1.0
            buffalo_onehot = torch.zeros(NUM_BUFFALO_BREEDS)
            cattle_mask = torch.tensor(1.0)
            buffalo_mask = torch.tensor(0.0)
        else:
            cattle_onehot = torch.zeros(NUM_CATTLE_BREEDS)
            buffalo_onehot = torch.zeros(NUM_BUFFALO_BREEDS)
            buffalo_onehot[self.buffalo_classes[row["breed"]]] = 1.0
            cattle_mask = torch.tensor(0.0)
            buffalo_mask = torch.tensor(1.0)
        binary_onehot = torch.zeros(2)
        binary_onehot[row["binary_label"]] = 1.0
        labels = {
            "binary": binary_onehot,
            "cattle": cattle_onehot,
            "buffalo": buffalo_onehot,
            "cattle_mask": cattle_mask,
            "buffalo_mask": buffalo_mask,
        }
        return image, labels


def _rand_bbox(size, lam):
    _, _, h, w = size
    cut_h = int(h * np.sqrt(1.0 - lam))
    cut_w = int(w * np.sqrt(1.0 - lam))
    cx = random.randint(0, h)
    cy = random.randint(0, w)
    bbx1 = max(0, cx - cut_h // 2)
    bby1 = max(0, cy - cut_w // 2)
    bbx2 = min(h, cx + cut_h // 2)
    bby2 = min(w, cy + cut_w // 2)
    return bbx1, bby1, bbx2, bby2


def _pairing_perm(labels, same_species=True):
    """Permutation that pairs each sample with another of the SAME species.

    A global randperm can pair a cattle image with a buffalo image, which makes
    the binary target fractional and leaves each breed target summing to λ<1
    instead of 1. Pairing within species keeps the binary label one-hot and
    every active breed target a proper distribution. Samples of a species with
    only one member map to themselves (no mixing).
    """
    batch = labels["binary"].size(0)
    perm = torch.arange(batch, device=labels["binary"].device)
    if not same_species:
        return torch.randperm(batch, device=labels["binary"].device)
    species = labels["binary"].argmax(1)
    for s in (0, 1):
        idx = torch.nonzero(species == s, as_tuple=False).flatten()
        if idx.numel() > 1:
            perm[idx] = idx[torch.randperm(idx.numel(),
                                           device=labels["binary"].device)]
    return perm


def _restore_unmixed(mixed, original, keep):
    """Return `mixed` where keep is False and `original` where keep is True.

    `keep` is a (B,) bool mask of samples that must NOT be mixed (rare
    breeds); broadcasts over any trailing label/feature dims.
    """
    if keep is None:
        return mixed
    view = keep.view(-1, *([1] * (mixed.dim() - 1)))
    return torch.where(view, original, mixed)


def cutmix(images, labels, alpha=CUTMIX_ALPHA, keep=None,
           same_species=MIX_SAME_SPECIES):
    original_images = images.clone()
    original_labels = {k: v.clone() for k, v in labels.items()}
    lam = np.random.beta(alpha, alpha)
    perm = _pairing_perm(labels, same_species)
    x1, y1, x2, y2 = _rand_bbox(images.size(), lam)
    images[:, :, x1:x2, y1:y2] = images[perm, :, x1:x2, y1:y2]
    area = (x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3))
    lam = 1.0 - area
    labels = {k: lam * labels[k] + (1.0 - lam) * labels[k][perm] for k in labels}
    images = _restore_unmixed(images, original_images, keep)
    labels = {k: _restore_unmixed(v, original_labels[k], keep)
              for k, v in labels.items()}
    return images, labels


def mixup(images, labels, alpha=MIXUP_ALPHA, keep=None,
          same_species=MIX_SAME_SPECIES):
    original_images = images.clone()
    original_labels = {k: v.clone() for k, v in labels.items()}
    lam = np.random.beta(alpha, alpha)
    perm = _pairing_perm(labels, same_species)
    images = lam * images + (1.0 - lam) * images[perm]
    labels = {k: lam * labels[k] + (1.0 - lam) * labels[k][perm] for k in labels}
    images = _restore_unmixed(images, original_images, keep)
    labels = {k: _restore_unmixed(v, original_labels[k], keep)
              for k, v in labels.items()}
    return images, labels


def mixed_collate(batch):
    images = torch.stack([b[0] for b in batch])
    labels = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    return images.to(memory_format=torch.channels_last), labels


def _make_weighted_sampler(df, beta=SAMPLER_BETA):
    """Effective-number-of-samples class balancing.

    Per-class sampling weight = 1 / E_n where E_n = (1 - beta^n) / (1 - beta).
    Pure inverse-frequency weighting gives a breed with 5 images the same
    relative boost over a breed with 500 images as one with 50 — it
    over-oversamples tiny classes into memorization noise. The effective
    number formulation saturates for large n (at 1/(1-beta)), softening the
    boost for common breeds while rare breeds keep approximately 1/n.

    Counts are keyed on (species, breed) because a handful of breed names
    (e.g. "bargur") exist under both cattle and buffalo.
    """
    counts = (df.groupby(["species", "breed"])["path"].count()
              .rename("n").reset_index())
    merged = df.merge(counts, on=["species", "breed"], how="left")
    eff_num = (1.0 - beta ** merged["n"]) / (1.0 - beta)
    weights = (1.0 / eff_num).to_numpy(dtype="float64").copy()
    return WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def _read_csv(split_dir, name):
    path = os.path.join(split_dir, f"{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_class_maps(split_dir):
    with open(os.path.join(split_dir, "cattle_classes.json")) as f:
        cattle_classes = json.load(f)
    with open(os.path.join(split_dir, "buffalo_classes.json")) as f:
        buffalo_classes = json.load(f)
    return cattle_classes, buffalo_classes


def _validate_class_counts(cattle_breeds, buffalo_breeds):
    """Fail fast when breed counts differ from the model's head sizes.

    A mismatch would otherwise surface much later as an index error when the
    dataset builds a one-hot vector. Also prints any breed name that appears
    under BOTH species (e.g. ``bargur``).
    """
    shared = sorted(set(cattle_breeds) & set(buffalo_breeds))
    if shared:
        print(f"[data] NOTE: breed name(s) under BOTH species: {shared} "
              f"(kept as separate classes, split per (species, breed))")
    errors = []
    for species, found, expected_n, expected_list in (
            ("cattle", cattle_breeds, NUM_CATTLE_BREEDS, EXPECTED_CATTLE_BREEDS),
            ("buffalo", buffalo_breeds, NUM_BUFFALO_BREEDS, EXPECTED_BUFFALO_BREEDS)):
        if len(found) == expected_n:
            continue
        msg = f"{species}: found {len(found)} breeds, expected {expected_n}"
        if expected_list:
            extra = sorted(set(found) - set(expected_list))
            missing = sorted(set(expected_list) - set(found))
            if extra:
                msg += f"; extra={extra}"
            if missing:
                msg += f"; missing={missing}"
        else:
            msg += f"; found={sorted(found)}"
        errors.append(msg)
    if errors:
        raise ValueError(
            "[data] breed-count mismatch (model heads are sized "
            f"{NUM_CATTLE_BREEDS} cattle + {NUM_BUFFALO_BREEDS} buffalo):\n  "
            + "\n  ".join(errors)
            + "\n  Fix the dataset folders, or set EXPECTED_*_BREEDS in "
              "src/config.py, or update NUM_*_BREEDS to match.")


def _count_per_class(df, class_map, num_classes, species=None):
    counts = np.zeros(num_classes, dtype=np.float64)
    if species is not None:
        df = df[df["species"] == species]
    by_breed = df.groupby("breed")["path"].count()
    for breed, idx in class_map.items():
        if breed in by_breed.index:
            counts[idx] = float(by_breed[breed])
    return counts


def _effective_num_weights(counts, beta=SAMPLER_BETA):
    """Per-class effective-number weights 1/E_n (same as the sampler).

    Classes with zero samples get weight 0 (never sampled) — the effective
    number E_0 = (1-beta^0)/(1-beta) = 0, so the raw reciprocal would be inf.
    """
    counts = np.asarray(counts, dtype=np.float64)
    eff_num = (1.0 - beta ** counts) / (1.0 - beta)
    eff_num = np.maximum(eff_num, 1e-12)
    weights = 1.0 / eff_num
    return weights * (counts > 0)


def compute_class_priors(split_dir=SPLIT_DIR, source="sampled"):
    """Smoothed log class priors for logit adjustment.

    IMPORTANT: the effective-number sampler already rebalances every batch, so
    applying logit adjustment on top of it double-corrects and over-predicts
    rare breeds at inference. Logit adjustment is OFF by default; if it is
    enabled, `source="sampled"` (the default here) uses the *effective sampled*
    distribution ``count_c * sampler_weight_c`` normalised, which is the
    distribution the model actually sees — not the raw image counts.

    source="raw" reproduces the old (double-correcting) behaviour.
    Returns {"cattle": Tensor[C], "buffalo": Tensor[B]} of log priors.
    """
    train_df = _read_csv(split_dir, "train")
    if train_df is None:
        return None
    cattle_classes, buffalo_classes = _load_class_maps(split_dir)
    priors = {}
    for species, class_map, num_classes in (
            ("cattle", cattle_classes, NUM_CATTLE_BREEDS),
            ("buffalo", buffalo_classes, NUM_BUFFALO_BREEDS)):
        counts = _count_per_class(train_df, class_map, num_classes,
                                  species=species)
        if source == "sampled":
            sampled = counts * _effective_num_weights(counts)
            present = counts > 0
            if present.any():
                prior = np.zeros_like(sampled)
                prior[present] = sampled[present] / sampled[present].sum()
                # Classes absent from train get the smallest present prior so
                # logit adjustment never boosts (or NaNs) an untrained class.
                prior[~present] = prior[present].min()
            else:
                prior = np.ones_like(sampled) / num_classes
        else:
            smoothed = counts + 1.0
            prior = smoothed / smoothed.sum()
        priors[species] = torch.log(torch.as_tensor(prior, dtype=torch.float32))
    return priors


def compute_class_counts(split_dir=SPLIT_DIR):
    """Raw train image counts per breed, for shot-bucket diagnostics."""
    train_df = _read_csv(split_dir, "train")
    if train_df is None:
        return None
    cattle_classes, buffalo_classes = _load_class_maps(split_dir)
    out = {}
    for species, class_map, num_classes in (
            ("cattle", cattle_classes, NUM_CATTLE_BREEDS),
            ("buffalo", buffalo_classes, NUM_BUFFALO_BREEDS)):
        counts = _count_per_class(train_df, class_map, num_classes,
                                  species=species)
        out[species] = torch.as_tensor(counts, dtype=torch.float32)
    return out


def compute_rare_classes(split_dir=SPLIT_DIR, threshold=RARE_CLASS_THRESHOLD):
    """Boolean per-class masks marking breeds below `threshold` train images."""
    train_df = _read_csv(split_dir, "train")
    if train_df is None:
        return None
    cattle_classes, buffalo_classes = _load_class_maps(split_dir)
    rare = {}
    for species, class_map, num_classes in (
            ("cattle", cattle_classes, NUM_CATTLE_BREEDS),
            ("buffalo", buffalo_classes, NUM_BUFFALO_BREEDS)):
        counts = _count_per_class(train_df, class_map, num_classes,
                                  species=species)
        rare[species] = torch.as_tensor(counts < threshold, dtype=torch.bool)
    return rare


def get_dataloaders(split_dir=SPLIT_DIR, batch_size=32, num_workers=4,
                    pin_memory=None, augment=None, breed_augment=False, pad=None,
                    allow_hue=False):
    """Build train/val/test DataLoaders.

    Args:
        pin_memory: If None, auto-detect (True when CUDA is available).
        augment: Optional dict of augmentation switches for the TRAIN split
            (see ``_train_transform``). Default = config defaults (all off).
        breed_augment: Apply per-breed overrides (``BREED_AUG_POLICY``).
        pad: Pad-to-square for all splits (None = config defaults).
    """
    train_df = _read_csv(split_dir, "train")
    val_df = _read_csv(split_dir, "val")
    test_df = _read_csv(split_dir, "test")
    if train_df is None:
        return None
    with open(os.path.join(split_dir, "cattle_classes.json")) as f:
        cattle_classes = json.load(f)
    with open(os.path.join(split_dir, "buffalo_classes.json")) as f:
        buffalo_classes = json.load(f)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds = CattleBuffaloDataset(
        train_df, cattle_classes, buffalo_classes,
        transform=_train_transform(augment, pad=pad), cache_images=CACHE_IMAGES,
        augment=augment, breed_augment=breed_augment, pad=pad)
    val_ds = CattleBuffaloDataset(val_df, cattle_classes, buffalo_classes,
                                  transform=_eval_transform(pad=pad),
                                  cache_images=CACHE_IMAGES)
    test_ds = CattleBuffaloDataset(test_df, cattle_classes, buffalo_classes,
                                   transform=_eval_transform(pad=pad),
                                   cache_images=CACHE_IMAGES)

    prefetch = 4 if num_workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=_make_weighted_sampler(train_df),
        num_workers=num_workers, collate_fn=mixed_collate, drop_last=True,
        pin_memory=pin_memory, persistent_workers=num_workers > 0,
        prefetch_factor=prefetch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=num_workers > 0,
                            prefetch_factor=prefetch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=num_workers > 0,
                             prefetch_factor=prefetch)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare stratified 70/15/15 splits")
    parser.add_argument("--data", default=RAW_DATA_DIR, help="raw data root")
    parser.add_argument("--split-dir", default=SPLIT_DIR)
    args = parser.parse_args()
    prepare_splits(data_root=args.data, split_dir=args.split_dir)