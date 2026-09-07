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

from .config import (CACHE_IMAGES, CUTMIX_ALPHA, HALF_DATA_RATIO, IMAGE_SIZE,
                     MIXUP_ALPHA, NUM_BUFFALO_BREEDS,
                     NUM_CATTLE_BREEDS, RANDAUGMENT_MAGNITUDE, RANDAUGMENT_OPS,
                     RAW_DATA_DIR, SMOKE_SAMPLES_PER_BREED, SPLIT_DIR,
                     TEST_RATIO, TRAIN_RATIO, VAL_RATIO)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def prepare_splits(data_root=RAW_DATA_DIR, split_dir=SPLIT_DIR):
    rows = _collect_rows(data_root)
    if not rows:
        print(f"[data] no images found under {data_root}")
        print("[data] expected layout: data/raw/cattle/<breed>/*.jpg and "
              "data/raw/buffalo/<breed>/*.jpg")
        return None

    df = pd.DataFrame(rows, columns=["path", "species", "breed", "binary_label"])
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    rng = random.Random(42)
    train_rows, val_rows, test_rows = [], [], []
    breeds = sorted(df["breed"].unique())
    for breed in tqdm(breeds, desc="splitting breeds", leave=False,
                      unit="breed"):
        group = df[df["breed"] == breed]
        idxs = list(group.index)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        for i, idx in enumerate(idxs):
            if i < n_train:
                train_rows.append(idx)
            elif i < n_train + n_val:
                val_rows.append(idx)
            else:
                test_rows.append(idx)

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
    if len(cattle_breeds) != NUM_CATTLE_BREEDS or len(buffalo_breeds) != NUM_BUFFALO_BREEDS:
        print(f"[data] WARNING: expected {NUM_CATTLE_BREEDS} cattle and "
              f"{NUM_BUFFALO_BREEDS} buffalo breeds, found {len(cattle_breeds)} "
              f"and {len(buffalo_breeds)}")
    return summary


def prepare_half_splits(data_root=RAW_DATA_DIR, split_dir=SPLIT_DIR,
                        ratio=HALF_DATA_RATIO):
    """Create a dataset using a fraction of images per breed for faster training.

    Uses the same 85/10/5 stratified split as full training but on a
    randomly-sampled subset. Class maps include ALL breeds so the model
    architecture stays identical.
    """
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

    # Apply the same 85/10/5 stratified split
    train_rows, val_rows, test_rows = [], [], []
    for breed in tqdm(sorted(half_df["breed"].unique()), desc="splitting breeds",
                      leave=False, unit="breed"):
        group = half_df[half_df["breed"] == breed]
        idxs = list(group.index)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(round(n * TRAIN_RATIO)))
        n_val = max(1, int(round(n * VAL_RATIO)))
        for i, idx in enumerate(idxs):
            if i < n_train:
                train_rows.append(idx)
            elif i < n_train + n_val:
                val_rows.append(idx)
            else:
                test_rows.append(idx)
    if not test_rows and val_rows:
        test_rows = val_rows[:1]

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
    return summary


def prepare_smoke_splits(data_root=RAW_DATA_DIR, split_dir=SPLIT_DIR,
                         samples_per_breed=SMOKE_SAMPLES_PER_BREED):
    """Create a tiny dataset for smoke tests by sampling a few images per breed.

    This produces real training signal (unlike 2-batch from full set) while
    being fast enough for CI / quick sanity checks.
    """
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


def _train_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 28),  # 288px for scale variation
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.RandAugment(num_ops=RANDAUGMENT_OPS,
                               magnitude=RANDAUGMENT_MAGNITUDE),
        transforms.ToTensor(),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])


class CattleBuffaloDataset(Dataset):
    def __init__(self, manifest, cattle_classes, buffalo_classes, transform=None, cache_images=False):
        self.manifest = manifest.reset_index(drop=True)
        self.cattle_classes = cattle_classes
        self.buffalo_classes = buffalo_classes
        self.transform = transform or _eval_transform()
        self._use_cache = cache_images
        self._image_cache = {}

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

        if self.transform is not None:
            image = self.transform(image)
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


def cutmix(images, labels, alpha=CUTMIX_ALPHA):
    batch = images.size(0)
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(batch, device=images.device)
    x1, y1, x2, y2 = _rand_bbox(images.size(), lam)
    images[:, :, x1:x2, y1:y2] = images[perm, :, x1:x2, y1:y2]
    area = (x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3))
    lam = 1.0 - area
    labels = {k: lam * labels[k] + (1.0 - lam) * labels[k][perm] for k in labels}
    return images, labels


def mixup(images, labels, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)
    images = lam * images + (1.0 - lam) * images[perm]
    labels = {k: lam * labels[k] + (1.0 - lam) * labels[k][perm] for k in labels}
    return images, labels


def mixed_collate(batch):
    images = torch.stack([b[0] for b in batch])
    labels = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    return images.to(memory_format=torch.channels_last), labels


def _make_weighted_sampler(df):
    counts = df.groupby("breed")["path"].count()
    weights = df["breed"].map(lambda b: 1.0 / counts[b]).to_numpy(
        dtype="float64").copy()
    return WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def _read_csv(split_dir, name):
    path = os.path.join(split_dir, f"{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_dataloaders(split_dir=SPLIT_DIR, batch_size=32, num_workers=4,
                    pin_memory=None):
    """Build train/val/test DataLoaders.

    Args:
        pin_memory: If None, auto-detect (True when CUDA is available).
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

    train_ds = CattleBuffaloDataset(train_df, cattle_classes, buffalo_classes,
                                    transform=_train_transform(), cache_images=CACHE_IMAGES)
    val_ds = CattleBuffaloDataset(val_df, cattle_classes, buffalo_classes,
                                  transform=_eval_transform(), cache_images=CACHE_IMAGES)
    test_ds = CattleBuffaloDataset(test_df, cattle_classes, buffalo_classes,
                                   transform=_eval_transform(), cache_images=CACHE_IMAGES)

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
    parser = argparse.ArgumentParser(description="Prepare stratified 80/10/10 splits")
    parser.add_argument("--data", default=RAW_DATA_DIR, help="raw data root")
    parser.add_argument("--split-dir", default=SPLIT_DIR)
    args = parser.parse_args()
    prepare_splits(data_root=args.data, split_dir=args.split_dir)