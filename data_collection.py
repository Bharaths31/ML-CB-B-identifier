import argparse
import html
import json
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image
from tqdm import tqdm

CATTLE_BREEDS = [
    "Gir", "Sahiwal", "Red Sindhi", "Tharparkar", "Rathi",
    "Hallikar", "Amritmahal", "Khillari", "Kangayam", "Bargur",
    "Hariana", "Kankrej", "Ongole", "Deoni", "Krishna Valley",
    "Punganur", "Vechur", "Malnad Gidda",
    "Jersey Cross", "HF Cross",
]

BUFFALO_BREEDS = [
    "Murrah", "Jaffrabadi", "Nili-Ravi", "Banni",
    "Pandharpuri", "Mehsana", "Surti", "Nagpuri",
    "Toda", "Bhadawari",
]

SPECIES_BREEDS = {"cattle": CATTLE_BREEDS, "buffalo": BUFFALO_BREEDS}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def species_of(breed):
    return "cattle" if breed in CATTLE_BREEDS else "buffalo"


def breed_dir(root, breed):
    return os.path.join(root, species_of(breed), slug(breed))


def _absolute(base, src):
    if src.startswith("http"):
        return src
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("/"):
        return base + src
    return src


def download_file(url, dest):
    resp = requests.get(url, headers=HEADERS, timeout=20, stream=True)
    if resp.status_code != 200:
        return False
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(65536):
            if chunk:
                f.write(chunk)
    try:
        with Image.open(dest) as im:
            im.load()
    except Exception:
        os.remove(dest)
        return False
    return True


def download_kaggle_dataset(dataset, output_dir):
    try:
        import kaggle
    except ImportError:
        print("kaggle package not installed; run: pip install kaggle")
        print("then set KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json")
        return
    print(f"downloading kaggle dataset: {dataset}")
    kaggle.api.dataset_download_files(dataset, path=output_dir, unzip=True)
    print("downloaded; organizing into cattle/buffalo breed folders ...")
    organize_dataset(output_dir, output_dir)


def scrape_nbagr_images(output_dir, breeds=None):
    base = "https://nbagr.icar.gov.in"
    for breed in breeds or (CATTLE_BREEDS + BUFFALO_BREEDS):
        dest = breed_dir(output_dir, breed)
        os.makedirs(dest, exist_ok=True)
        url = f"{base}/breed/{slug(breed)}/"
        print(f"\nfetching {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"  no breed page ({resp.status_code})")
                continue
            srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', resp.text, re.I)
            srcs = [_absolute(base, s) for s in srcs]
            srcs = [s for s in srcs if slug(breed) in s.lower() or "breed" in s.lower()]
            got = 0
            for s in dict.fromkeys(srcs):
                try:
                    if download_file(s, os.path.join(dest, f"{slug(breed)}_{got}.jpg")):
                        got += 1
                        print(f"  [{got}] {s[:90]}")
                    time.sleep(0.5)
                except requests.RequestException:
                    pass
            print(f"  downloaded {got} images -> {dest}")
        except requests.RequestException as e:
            print(f"  error: {e}")


def download_web_images(breed, output_dir, limit=100):
    dest = breed_dir(output_dir, breed)
    os.makedirs(dest, exist_ok=True)
    species = species_of(breed)
    query = f"{breed} {species} breed"
    first = 0
    fetched = 0
    print(f"\ndownloading up to {limit} images for {breed} -> {dest}")
    while fetched < limit:
        url = (f"https://www.bing.com/images/async?q={quote(query)}"
               f"&first={first}&count=35&adlt=off")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
        except requests.RequestException as e:
            print(f"  error: {e}")
            break
        if resp.status_code != 200:
            print(f"  search returned {resp.status_code}")
            break
        urls = []
        for ma in re.findall(r'm="([^"]+)"', resp.text):
            try:
                meta = json.loads(html.unescape(ma))
            except Exception:
                continue
            if meta.get("murl"):
                urls.append(meta["murl"])
        if not urls:
            break
        for u in dict.fromkeys(urls):
            if fetched >= limit:
                break
            target = os.path.join(dest, f"{species}_{slug(breed)}_{fetched}.jpg")
            try:
                ok = download_file(u, target)
            except requests.RequestException:
                ok = False
            if ok:
                fetched += 1
                print(f"  [{fetched}/{limit}] {os.path.basename(target)} <- {u[:80]}")
            time.sleep(0.4)
        first += 35
    print(f"  done: {fetched} images in {dest}")


def print_manual_instructions(breed, output_dir):
    dest = breed_dir(output_dir, breed)
    os.makedirs(dest, exist_ok=True)
    species = species_of(breed)
    print("=" * 60)
    print(f"manual collection: {breed} ({species})")
    print(f"target directory: {dest}")
    print(f"search: https://www.google.com/search?q={quote(f'{breed} {species} breed')}&tbm=isch")
    print("save clear side-profile images into that folder")
    print("=" * 60)


def _breed_from_name(name):
    n = name.lower()
    n = re.sub(r"^(cattle|buffalo)[_-]", "", n)
    n = re.sub(r"\s*\(.*?\)", "", n)
    n = re.sub(r"[-_.\s]+", "_", n).strip("_")
    return n or None


def _locate(image, raw_dir):
    rel = image.relative_to(raw_dir).parts
    species = None
    for part in rel:
        low = part.lower()
        if low in ("cattle", "buffalo"):
            species = low
            break
        if re.match(r"^(cattle|buffalo)[_-]", low):
            species = re.match(r"^(cattle|buffalo)", low).group(1)
            break
    if species is None:
        return None, None
    breed = None
    for part in reversed(rel[:-1]):
        low = part.lower()
        if low in ("cattle", "buffalo"):
            continue
        candidate = _breed_from_name(part)
        if candidate:
            breed = candidate
            break
    return species, breed


def organize_dataset(raw_dir, output_dir):
    files = [p for p in Path(raw_dir).rglob("*")
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"found {len(files)} images under {raw_dir}")
    moved = 0
    for p in tqdm(files, desc="organizing"):
        species, breed = _locate(p, raw_dir)
        if not (species and breed):
            continue
        dest_dir = os.path.join(output_dir, species, breed)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, p.name)
        if not os.path.exists(dest):
            shutil.move(str(p), dest)
            moved += 1
            print(f"  {p.name} -> {species}/{breed}/")
    print(f"organized {moved} images")


def create_dataset_manifest(data_dir, output_file=None):
    if output_file is None:
        output_file = os.path.join(data_dir, "manifest.json")
    manifest = {"data_dir": data_dir, "species": {}, "total_images": 0}
    for species in ("cattle", "buffalo"):
        manifest["species"][species] = {}
        sdir = os.path.join(data_dir, species)
        if not os.path.isdir(sdir):
            continue
        for breed in sorted(os.listdir(sdir)):
            bdir = os.path.join(sdir, breed)
            if not os.path.isdir(bdir):
                continue
            imgs = [p for p in Path(bdir).iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            manifest["species"][species][breed] = {
                "count": len(imgs),
                "images": [str(p.relative_to(data_dir)) for p in imgs],
            }
            manifest["total_images"] += len(imgs)
    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"created manifest: {output_file}")
    print(f"total images: {manifest['total_images']}")
    print(f"cattle breeds: {len(manifest['species']['cattle'])}")
    print(f"buffalo breeds: {len(manifest['species']['buffalo'])}")
    return manifest


def main():
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "raw")
    parser = argparse.ArgumentParser(
        description="Collect cattle/buffalo breed images into "
                    "data/raw/cattle/<breed>/ and data/raw/buffalo/<breed>/")
    parser.add_argument("--source",
                        choices=["kaggle", "nbagr", "web", "google",
                                 "organize", "manifest", "split"],
                        required=True)
    parser.add_argument("--output", default=default_out)
    parser.add_argument("--breed", help="single breed to process")
    parser.add_argument("--dataset", help="kaggle dataset name")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--input", help="dir for --source organize")
    parser.add_argument("--manifest", help="manifest output path")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, "cattle"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "buffalo"), exist_ok=True)

    if args.source == "kaggle":
        if not args.dataset:
            print("use --dataset, e.g. saurabhsharma/cattle-breed-classification")
            return
        download_kaggle_dataset(args.dataset, args.output)
    elif args.source == "nbagr":
        scrape_nbagr_images(args.output, [args.breed] if args.breed else None)
    elif args.source == "web":
        for breed in ([args.breed] if args.breed else CATTLE_BREEDS + BUFFALO_BREEDS):
            download_web_images(breed, args.output, args.limit)
    elif args.source == "google":
        for breed in ([args.breed] if args.breed else CATTLE_BREEDS + BUFFALO_BREEDS):
            print_manual_instructions(breed, args.output)
    elif args.source == "organize":
        organize_dataset(args.input or args.output, args.output)
    elif args.source == "manifest":
        create_dataset_manifest(args.output, args.manifest)
    elif args.source == "split":
        from src.data_pipeline import prepare_splits
        prepare_splits(args.output)

    print("\nnext step: python -m src.data_pipeline  (writes data/splits/*.csv)")


if __name__ == "__main__":
    main()
