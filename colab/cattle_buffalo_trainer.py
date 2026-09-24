# %% [markdown]
# # 🐄 Cattle & Buffalo Breed Classifier — Colab Training
#
# **State-of-the-art multi-task training pipeline** optimized for **Google Colab (T4 / V100 / A100 GPU)**.
#
# | Phase | Description | Status | AMP |
# |---|---|---|---|
# | **1** | All-heads warmup (frozen backbone + attention; binary + cattle + buffalo heads) | Default | ✅ |
# | **2** | Full multi-task fine-tuning (AdamW + warmup + cosine, EMA, SupCon, adaptive loss weights) | Default | ✅ |
# | **3** | QAT — Quantization-Aware Training (opt-in recovery path; mobile INT8 comes from converter PTQ) | Opt-in | ❌ |
#
# **Key Features & Enhancements:**
# - **Long-Tail Handling**: Single mechanism — effective-number-of-samples weighting (`SAMPLER_BETA=0.99`) keyed on `(species, breed)`.
# - **Feature Representation**: Supervised Contrastive (SupCon) feature learning on an auxiliary 128-d projection head (`CONTRASTIVE_WEIGHT=0.2`).
# - **Dynamic VRAM Scaling**: Auto-scales batch size (16 to 128) and gradient accumulation to fit any GPU VRAM while maintaining an effective batch of 128.
# - **Stabilized EMA**: Parameter AND BatchNorm buffer moving averages updated once per optimizer step with warmup.
# - **Adaptive Loss Weights**: Reallocates loss budget from the binary head to breed heads upon binary saturation (`binary_acc >= 0.95`).
# - **Same-Species Mixing**: Opt-in CutMix/MixUp pairs images within the same species to preserve valid label distributions.
# - **Soft Routing**: Inference uses `p(species) · softmax(head)` mixture over all 75 breeds (57 cattle + 18 buffalo), preventing two-stage error cascades.
# - **Deployment Options**: Self-contained portable folder, FP32 ONNX, Mobile QDQ INT8 ONNX, and TFLite.

# %% [markdown]
# ---
# ## §0 — GPU Check & Environment Setup

# %%
# ============================================================
#  GPU VERIFICATION & CUDA SETUP
# ============================================================
import os
import shutil
import subprocess
import sys
import torch

def check_gpu():
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Go to: Runtime → Change runtime type → T4 GPU (or better)")
        return False
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ GPU: {gpu} ({mem:.1f} GB VRAM)")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda}")
    return True

assert check_gpu(), "GPU required for training!"

# %%
# ============================================================
#  INSTALL DEPENDENCIES
# ============================================================
# Install required dependencies
!pip install -q pandas tqdm matplotlib scikit-learn Pillow numpy onnx onnxruntime

# %% [markdown]
# ---
# ## §1 — Project Setup (Source Code)
#
# Choose **ONE** of the three options below:
# - **Option A**: Clone fresh from GitHub (recommended — public repo)
# - **Option B**: Upload `colab_project.zip` manually
# - **Option C**: Mount Google Drive and copy from Drive

# %%
# ============================================================
#  OPTION A: CLONE FROM GITHUB (Recommended)
# ============================================================
GITHUB_REPO = "https://github.com/Bharaths31/ML-CB-B-identifier"
PROJECT_DIR = "/content/project"

# Step 1: Move to /content FIRST to avoid corrupting shell cwd
os.chdir("/content")

# Step 2: Remove old clone if it exists (force-refresh to pick up latest code)
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
    print(f"🗑️  Removed old clone at {PROJECT_DIR}")

# Step 3: Clone fresh
result = subprocess.run(
    ["git", "clone", GITHUB_REPO, PROJECT_DIR],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"❌ Clone failed:\n{result.stderr}")
    raise RuntimeError("git clone failed — check the repo URL and your internet connection")
print(f"✅ Cloned repo to {PROJECT_DIR}")

# Step 4: Add to Python path so `import src` works
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Step 5: Verify src/ structure and required modules
required_modules = [
    "config.py", "run_utils.py", "run_logger.py", "data_pipeline.py",
    "model.py", "cbam.py", "efficientnet_lite.py", "traits.py",
    "train.py", "metrics.py", "evaluate.py", "export.py", "parity_check.py"
]
for mod in required_modules:
    assert os.path.exists(f"{PROJECT_DIR}/src/{mod}"), f"❌ src/{mod} not found in {PROJECT_DIR}"
print(f"✅ All core source modules verified at {PROJECT_DIR}/src/")

# %%
# ============================================================
#  OPTION B: UPLOAD colab_project.zip MANUALLY
#  (Skip if you used Option A or C)
# ============================================================
# Uncomment the lines below to use this option:

# from google.colab import files
# print("📤 Upload colab_project.zip...")
# uploaded = files.upload()
# !mkdir -p /content/project
# !unzip -qo /content/colab_project.zip -d /content/project
# PROJECT_DIR = "/content/project"
# print(f"✅ Extracted to {PROJECT_DIR}")

# %%
# ============================================================
#  OPTION C: MOUNT GOOGLE DRIVE
#  (Skip if you used Option A or B)
# ============================================================
# Uncomment the lines below to use this option:

# from google.colab import drive
# drive.mount("/content/drive")
# DRIVE_ZIP = "/content/drive/MyDrive/ML-CB-B-identifier/colab_project.zip"
# PROJECT_DIR = "/content/project"
# !mkdir -p {PROJECT_DIR}
# !unzip -qo {DRIVE_ZIP} -d {PROJECT_DIR}
# print(f"✅ Extracted from Drive to {PROJECT_DIR}")

# %%
# ============================================================
#  ADD PROJECT TO PYTHON PATH  (shared — runs after A, B, or C)
# ============================================================
PROJECT_DIR = PROJECT_DIR if 'PROJECT_DIR' in dir() else "/content/project"

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

if os.path.exists(PROJECT_DIR):
    os.chdir(PROJECT_DIR)
    print(f"✅ Working directory: {os.getcwd()}")
    print(f"✅ Python path includes: {PROJECT_DIR}")
else:
    raise RuntimeError(f"❌ {PROJECT_DIR} does not exist — run one of the setup options above first")

# Verify imports work
try:
    from src.config import IMAGE_SIZE, BATCH_SIZE, NUM_CATTLE_BREEDS, NUM_BUFFALO_BREEDS
    print(f"✅ src.config imported (IMAGE_SIZE={IMAGE_SIZE}, BATCH_SIZE={BATCH_SIZE}, "
          f"Breeds: {NUM_CATTLE_BREEDS} cattle + {NUM_BUFFALO_BREEDS} buffalo = {NUM_CATTLE_BREEDS + NUM_BUFFALO_BREEDS} total)")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("   Make sure you ran one of the setup options above")

# %% [markdown]
# ---
# ## §2 — Dataset Acquisition
#
# Choose **ONE** of three dataset modes:
# - **`both`** (Recommended): Download ALL datasets and merge — maximum data for 57 cattle + 18 buffalo breeds
# - **`algsoch`**: Original unified dataset (`algsoch/breed-cattle-buffalo`)
# - **`atharvadarpude`**: Separate datasets (`atharvadarpude/indian-cattle-image-dataset` + `atharvadarpude/indian-buffalo-dataset`)
#
# Then fill in your Kaggle API credentials below.

# %%
# ============================================================
#  CONFIGURATION — Pick your dataset mode & Kaggle credentials
# ============================================================
DATASET_MODE = "both"  # ← "both", "algsoch", or "atharvadarpude"

# Kaggle credentials (required for API download)
KAGGLE_USERNAME = ""   # ← Fill in your Kaggle username
KAGGLE_KEY = ""        # ← Fill in your Kaggle API key

# Dataset slugs (do not change)
SLUG_ALGSOCH         = "algsoch/breed-cattle-buffalo"
SLUG_ATHARVA_CATTLE  = "atharvadarpude/indian-cattle-image-dataset"
SLUG_ATHARVA_BUFFALO = "atharvadarpude/indian-buffalo-dataset"

DATA_RAW = f"{PROJECT_DIR}/data/raw"

# ============================================================
#  KAGGLE API DOWNLOAD & DATASET MERGE
# ============================================================
import glob
import json

def download_kaggle_dataset(slug, dest_dir):
    """Download and unzip a Kaggle dataset into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    ret = os.system(f"kaggle datasets download -d {slug} -p {dest_dir} --unzip --force")
    if ret != 0:
        raise RuntimeError(f"❌ Failed to download {slug} (exit code {ret})")
    print(f"   ✅ Downloaded: {slug} → {dest_dir}")
    return dest_dir

def find_breed_dirs(base_dir):
    """Find all leaf directories containing images (breed folders)."""
    breed_dirs = []
    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for root, dirs, files in os.walk(base_dir):
        img_files = [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTS]
        if img_files and not dirs:  # leaf dir with images
            breed_dirs.append(root)
    return breed_dirs

# Spelling-variant merge map (empty by default; confirm with audit_data.py first)
BREED_ALIASES = {}

def normalize_breed_name(name):
    """Normalize breed folder names: lowercase, underscores, strip whitespace."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")

def build_dataset_inventory(source_base, dataset_name, out_dir=None,
                            source_hint=None, include_images=True):
    """Write a JSON inventory of the dataset to data/dataset_inventory/<dataset_name>.json."""
    import time as _time
    try:
        from PIL import Image
    except ImportError:
        Image = None

    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    out_dir = out_dir or f"{PROJECT_DIR}/data/dataset_inventory"
    os.makedirs(out_dir, exist_ok=True)

    def _species_from_path(rel_parts):
        for p in rel_parts:
            pl = p.lower()
            if pl in ("cattle", "cow", "cows"):
                return "cattle"
            if pl in ("buffalo", "buff", "buffaloes"):
                return "buffalo"
        return source_hint or "unknown"

    breeds, errors = [], []
    for root, dirs, files in os.walk(source_base):
        imgs = sorted(f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTS)
        if not imgs:
            continue
        rel = os.path.relpath(root, source_base)
        rel_parts = [] if rel == "." else rel.split(os.sep)
        species = _species_from_path(rel_parts)
        breed = normalize_breed_name(os.path.basename(root))
        entry = {
            "breed": breed, "species": species,
            "source_folder": rel.replace(os.sep, "/"), "count": 0,
            "resolutions": {}, "images": [] if include_images else None
        }
        for fname in imgs:
            fpath = os.path.join(root, fname)
            w = h = None
            if Image is not None:
                try:
                    with Image.open(fpath) as im:
                        w, h = im.size
                except Exception as exc:
                    errors.append({"file": fpath, "error": str(exc)})
                    continue
            entry["count"] += 1
            key = f"{w}x{h}" if w and h else "unknown"
            entry["resolutions"][key] = entry["resolutions"].get(key, 0) + 1
            if include_images:
                entry["images"].append({
                    "file": os.path.relpath(fpath, source_base).replace(os.sep, "/"),
                    "width": w, "height": h
                })
        breeds.append(entry)

    breeds.sort(key=lambda e: (e["species"], e["breed"]))
    by_species = {}
    for e in breeds:
        by_species[e["species"]] = by_species.get(e["species"], 0) + e["count"]

    inventory = {
        "dataset": dataset_name,
        "source_dir": os.path.abspath(source_base),
        "source_hint": source_hint,
        "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "totals": {
            "breeds": len(breeds),
            "images": sum(e["count"] for e in breeds),
            "by_species": by_species,
            "unreadable": len(errors)
        },
        "breeds": breeds,
        "errors": errors,
    }
    out_path = os.path.join(out_dir, f"{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"   📋 Inventory: {dataset_name} -> {out_path} "
          f"({inventory['totals']['breeds']} breeds, {inventory['totals']['images']} images"
          + (f", {len(errors)} unreadable" if errors else "") + ")")
    return inventory

def merge_into_species_dir(source_base, target_species_dir, species_hint=None):
    """Auto-detect breed folders inside source_base and copy/merge them into target_species_dir."""
    os.makedirs(target_species_dir, exist_ok=True)
    copied = 0
    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    species_sub = None
    if species_hint:
        for d in os.listdir(source_base):
            if d.lower() == species_hint.lower():
                species_sub = os.path.join(source_base, d)
                break

    scan_root = species_sub if species_sub else source_base
    breed_dirs = find_breed_dirs(scan_root)
    if not breed_dirs:
        for subdir in os.listdir(scan_root):
            subpath = os.path.join(scan_root, subdir)
            if os.path.isdir(subpath):
                breed_dirs.extend(find_breed_dirs(subpath))

    for breed_path in breed_dirs:
        breed_name = normalize_breed_name(os.path.basename(breed_path))
        breed_name = BREED_ALIASES.get(breed_name, breed_name)
        target_breed_dir = os.path.join(target_species_dir, breed_name)
        os.makedirs(target_breed_dir, exist_ok=True)

        for fname in os.listdir(breed_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTS:
                src_file = os.path.join(breed_path, fname)
                dst_file = os.path.join(target_breed_dir, fname)
                if os.path.exists(dst_file):
                    base, ext_ = os.path.splitext(fname)
                    dst_file = os.path.join(target_breed_dir, f"{base}_dup{ext_}")
                shutil.copy2(src_file, dst_file)
                copied += 1

    return copied

if KAGGLE_USERNAME and KAGGLE_KEY:
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    subprocess.run(["pip", "install", "-q", "kaggle"])

    if os.path.exists(DATA_RAW):
        shutil.rmtree(DATA_RAW)
    os.makedirs(DATA_RAW, exist_ok=True)

    cattle_dir = os.path.join(DATA_RAW, "cattle")
    buffalo_dir = os.path.join(DATA_RAW, "buffalo")
    TMP_DL = "/content/_kaggle_downloads"

    if DATASET_MODE in ("algsoch", "both"):
        print("\n📥 Downloading algsoch unified dataset...")
        dl_path = download_kaggle_dataset(SLUG_ALGSOCH, f"{TMP_DL}/algsoch")
        build_dataset_inventory(dl_path, "algsoch")
        n_cat = merge_into_species_dir(dl_path, cattle_dir, species_hint="cattle")
        n_buf = merge_into_species_dir(dl_path, buffalo_dir, species_hint="buffalo")
        print(f"      Merged: {n_cat} cattle, {n_buf} buffalo images")

    if DATASET_MODE in ("atharvadarpude", "both"):
        print("\n📥 Downloading atharvadarpude cattle dataset...")
        dl_path = download_kaggle_dataset(SLUG_ATHARVA_CATTLE, f"{TMP_DL}/atharva_cattle")
        build_dataset_inventory(dl_path, "atharvadarpude_cattle", source_hint="cattle")
        n_cat = merge_into_species_dir(dl_path, cattle_dir)
        print(f"      Merged: {n_cat} cattle images")

        print("\n📥 Downloading atharvadarpude buffalo dataset...")
        dl_path = download_kaggle_dataset(SLUG_ATHARVA_BUFFALO, f"{TMP_DL}/atharva_buffalo")
        build_dataset_inventory(dl_path, "atharvadarpude_buffalo", source_hint="buffalo")
        n_buf = merge_into_species_dir(dl_path, buffalo_dir)
        print(f"      Merged: {n_buf} buffalo images")

    # Inventory the final merged tree
    build_dataset_inventory(DATA_RAW, "merged")

    if os.path.exists(TMP_DL):
        shutil.rmtree(TMP_DL)

    print(f"\n✅ All datasets downloaded and merged into {DATA_RAW}")
    print(f"   Mode: {DATASET_MODE}")
else:
    print("⚠️  Kaggle credentials not set — fill in KAGGLE_USERNAME and KAGGLE_KEY above or upload archive.zip.")

# %%
# ============================================================
#  VERIFY DATASET STRUCTURE & CLASS COUNTS
# ============================================================
cattle_dir = os.path.join(DATA_RAW, "cattle")
buffalo_dir = os.path.join(DATA_RAW, "buffalo")

cattle_breeds = sorted([b for b in os.listdir(cattle_dir)
                        if os.path.isdir(os.path.join(cattle_dir, b))]) if os.path.isdir(cattle_dir) else []
buffalo_breeds = sorted([b for b in os.listdir(buffalo_dir)
                         if os.path.isdir(os.path.join(buffalo_dir, b))]) if os.path.isdir(buffalo_dir) else []

print(f"📊 Dataset Summary (mode: {DATASET_MODE}):")
print(f"   Cattle breeds found:  {len(cattle_breeds)} / 57 expected")
print(f"   Buffalo breeds found: {len(buffalo_breeds)} / 18 expected")

total_cat_imgs = sum(len([f for f in os.listdir(os.path.join(cattle_dir, b))
                          if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
                     for b in cattle_breeds)
total_buf_imgs = sum(len([f for f in os.listdir(os.path.join(buffalo_dir, b))
                          if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
                     for b in buffalo_breeds)

print(f"   Total cattle images:  {total_cat_imgs:,}")
print(f"   Total buffalo images: {total_buf_imgs:,}")
print(f"   Total images:         {total_cat_imgs + total_buf_imgs:,}")

assert len(cattle_breeds) > 0 or len(buffalo_breeds) > 0, "❌ No breed folders found! Check dataset setup."
print("✅ Dataset verified!")

# %% [markdown]
# ---
# ## §3 — Hyperparameter Configuration
#
# All parameters are aligned with the state-of-the-art configuration in `src/config.py`.

# %%
# ============================================================
#  ⚙️  HYPERPARAMETER CONFIGURATION
# ============================================================

# --- Architecture ---
BACKBONE = "lite2"              # "lite2" (~6M params, fast mobile) or "lite4" (~13M params)
ATTENTION = "cbam"              # "cbam" (CBAM attention) or "se" (Squeeze-and-Excitation)

# --- Data Mode ---
# "full": 100% of images (production training)
# "half": 50% of images per breed (quick iteration)
# "quarter": 25% of images per breed (fast iteration)
# "smoke": 5 images per breed, 1 epoch per phase (instant end-to-end integration test)
DATA_MODE = "full"

# --- Training & Hardware ---
BATCH_SIZE = 64                 # Auto-scaled dynamically if VRAM < 16GB
GRAD_ACCUM = 2                  # Auto-scaled to keep effective batch = 128
NUM_WORKERS = 2                 # Colab 2 CPU cores
SEED = 42

# --- Phase Epochs & Learning Rates ---
PHASE1_EPOCHS = 8               # All-heads warmup (frozen backbone + attention)
PHASE1_LR = 3e-3
PHASE2_EPOCHS = 80              # Multi-task fine-tuning (warmup + cosine decay)
PHASE2_LR = 2e-4
WARMUP_EPOCHS = 3               # Linear LR warmup epochs for phase 2

# --- Optional Phase 3: QAT ---
# Mobile INT8 is normally produced via converter-side PTQ (see §9). QAT is opt-in.
INCLUDE_QAT = False
PHASE3_EPOCHS = 10
PHASE3_LR = 5e-6

# --- Regularization & Optimizer ---
WEIGHT_DECAY = 1e-2             # AdamW decoupled weight decay
LABEL_SMOOTHING = 0.05          # Label smoothing for generalization

# --- Loss Weights & Dynamic Adaptation ---
# Loss = w_bin*CE_bin + w_cat*CE_cat + w_buf*CE_buf + w_supcon*SupCon
LOSS_WEIGHT_BINARY = 0.15
LOSS_WEIGHT_CATTLE = 0.50
LOSS_WEIGHT_BUFFALO = 0.35

# Automatically switch to final weights once binary_acc >= 0.95
BINARY_SATURATION_ACC = 0.95
LOSS_WEIGHT_BINARY_FINAL = 0.05
LOSS_WEIGHT_CATTLE_FINAL = 0.55
LOSS_WEIGHT_BUFFALO_FINAL = 0.40

# --- Long-Tail Handling (Single Mechanism) ---
SAMPLER_BETA = 0.99             # Effective-number-of-samples weighting (softened oversampling)
LOGIT_ADJUST = False            # OFF by default: sampler already rebalances every batch
LOGIT_ADJUST_TAU = 1.0
LOGIT_ADJUST_PRIOR = "sampled"  # Effective sampled prior if logit adjustment is turned on
RARE_CLASS_THRESHOLD = 30       # Breeds with <30 train images excluded from CutMix/MixUp

# --- Feature Learning (SupCon) ---
CONTRASTIVE_WEIGHT = 0.2        # Supervised Contrastive loss on auxiliary 128-d projection head
CONTRASTIVE_TEMPERATURE = 0.1

# --- Augmentation & Batch Mixing ---
AUG_HORIZONTAL_FLIP = True      # ON by default (safe spatial invariance)
AUG_RANDOM_RESIZED_CROP = True  # ON by default (mild crop: scale=(0.8, 1.0), ratio=(0.92, 1.08))
AUG_COLOR_JITTER = False        # OFF by default (coat colour is crucial for breed identity)
AUG_RANDAUGMENT = False         # OFF by default
PAD_TO_SQUARE = False           # Resize long side + pad (preserves full-body side profiles)
BREED_AUG = False               # Breed-aware policies (e.g. coat-colour breeds skip jitter)
MIX_ENABLED = False             # CutMix/MixUp batch mixing (opt-in; OFF by default)
CUTMIX_ALPHA = 0.4
MIXUP_ALPHA = 0.2
CUTMIX_MIXUP_PROB = 0.25
MIX_SAME_SPECIES = True         # Pair samples within the same species so labels stay well-formed
MIX_OFF_LAST_FRAC = 0.15        # Disable mixing for the last 15% of phase 2

# --- Checkpoint Selection & EMA ---
BEST_METRIC = "blended_score"   # 0.5 * macro-F1 + 0.5 * soft top-1
EMA_DECAY = 0.999
EMA_WARMUP = True

# --- Split Configuration ---
TRAIN_RATIO = 0.70              # 70/15/15 stratified per (species, breed)
VAL_RATIO = 0.15                # Guarantees >=1 val image per breed with >=3 images
TEST_RATIO = 0.15               # Guarantees >=1 test image per breed with >=3 images
DEDUP_SPLITS = False            # Group-aware splits: dHash near-duplicates stay in one split

# --- Output Naming ---
RUN_TAG = None                  # Custom tag or None for timestamp DD-MM-YYYY-HH-MM

print("=" * 60)
print("  HYPERPARAMETER SUMMARY")
print("=" * 60)
print(f"  Backbone:       EfficientNet-{BACKBONE} + {ATTENTION.upper()} attention")
print(f"  Data Mode:      {DATA_MODE.upper()}")
print(f"  Epochs:         Phase 1: {PHASE1_EPOCHS} ep | Phase 2: {PHASE2_EPOCHS} ep" +
      (f" | Phase 3 QAT: {PHASE3_EPOCHS} ep" if INCLUDE_QAT else " | QAT: OFF (PTQ preferred)"))
print(f"  Batch size:     {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
print(f"  Optimizer:      AdamW (lr_p1={PHASE1_LR}, lr_p2={PHASE2_LR}, wd={WEIGHT_DECAY})")
print(f"  Loss weights:   Initial ({LOSS_WEIGHT_BINARY}/{LOSS_WEIGHT_CATTLE}/{LOSS_WEIGHT_BUFFALO}) "
      f"→ Saturated ({LOSS_WEIGHT_BINARY_FINAL}/{LOSS_WEIGHT_CATTLE_FINAL}/{LOSS_WEIGHT_BUFFALO_FINAL})")
print(f"  Long-tail:      WeightedRandomSampler (beta={SAMPLER_BETA})" +
      (" + logit_adjust" if LOGIT_ADJUST else " (single mechanism)"))
print(f"  SupCon Loss:    weight={CONTRASTIVE_WEIGHT}, temp={CONTRASTIVE_TEMPERATURE}")
print(f"  Augmentation:   flip={AUG_HORIZONTAL_FLIP}, rrc={AUG_RANDOM_RESIZED_CROP}, "
      f"color_jitter={AUG_COLOR_JITTER}, mix={MIX_ENABLED}")
print(f"  Checkpoint:     Best metric = {BEST_METRIC} (EMA enabled, decay={EMA_DECAY})")
print("=" * 60)

# %% [markdown]
# ---
# ## §4 — Data Split & Preprocessing

# %%
# ============================================================
#  PREPARE DATA SPLITS & CLASS DISTRIBUTION
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src import config as cfg
from src.data_pipeline import (
    prepare_splits, prepare_half_splits, prepare_quarter_splits, prepare_smoke_splits
)

# Apply configuration
cfg.TRAIN_RATIO = TRAIN_RATIO
cfg.VAL_RATIO = VAL_RATIO
cfg.TEST_RATIO = TEST_RATIO
cfg.BATCH_SIZE = BATCH_SIZE
cfg.LABEL_SMOOTHING = LABEL_SMOOTHING
cfg.SAMPLER_BETA = SAMPLER_BETA
cfg.CONTRASTIVE_WEIGHT = CONTRASTIVE_WEIGHT
cfg.CONTRASTIVE_TEMPERATURE = CONTRASTIVE_TEMPERATURE

split_dir = f"{PROJECT_DIR}/data/splits"
os.makedirs(split_dir, exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/outputs/checkpoints", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/outputs/export/portable", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/outputs/metrics", exist_ok=True)

print(f"[splits] Creating splits for mode: '{DATA_MODE}'...")
if DATA_MODE == "smoke":
    summary = prepare_smoke_splits(data_root=DATA_RAW, split_dir=split_dir)
elif DATA_MODE == "half":
    summary = prepare_half_splits(data_root=DATA_RAW, split_dir=split_dir)
elif DATA_MODE == "quarter":
    summary = prepare_quarter_splits(data_root=DATA_RAW, split_dir=split_dir)
else:
    summary = prepare_splits(data_root=DATA_RAW, split_dir=split_dir, dedup=DEDUP_SPLITS)

if summary is None:
    raise RuntimeError("❌ Failed to create data splits. Check §2 dataset setup.")

print(f"\n✅ Splits created successfully:")
print(f"   Train images: {summary.get('train', '?'):,}")
print(f"   Val images:   {summary.get('val', '?'):,}")
print(f"   Test images:  {summary.get('test', '?'):,}")

# Visualize class distribution
train_df = pd.read_csv(f"{split_dir}/train.csv")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, species in zip(axes, ["cattle", "buffalo"]):
    sp_df = train_df[train_df["species"] == species]
    counts = sp_df["breed"].value_counts().sort_index()
    if len(counts) > 0:
        counts.plot(kind="barh", ax=ax, color="#2E7D32" if species == "cattle" else "#1565C0")
        ax.set_title(f"{species.title()} Breed Distribution (Train: {len(sp_df):,} images, {len(counts)} breeds)")
        ax.set_xlabel("Image Count")
        ax.tick_params(axis='y', labelsize=8)

plt.tight_layout()
dist_plot_path = f"{PROJECT_DIR}/outputs/class_distribution.png"
plt.savefig(dist_plot_path, dpi=120)
plt.show()
print(f"✅ Class distribution saved to {dist_plot_path}")

# %% [markdown]
# ---
# ## §5 — Architecture Verification

# %%
# ============================================================
#  VERIFY MODEL ARCHITECTURE & SOFT ROUTING
# ============================================================
from src.model import BreedClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = f"{PROJECT_DIR}/efficientnet_{BACKBONE}.pth"
has_weights = os.path.exists(weights_path)

model = BreedClassifier(
    backbone=BACKBONE,
    attention=ATTENTION,
    pretrained_path=weights_path if has_weights else None
)
model.eval()

# Forward pass verification
with torch.no_grad():
    x = torch.randn(2, 3, 260, 260)
    out = model(x)
    preds = model.predict(x)

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model Architecture Verified:")
print(f"   Backbone:            EfficientNet-{BACKBONE}")
print(f"   Attention:           {ATTENTION.upper()}")
print(f"   Pretrained Weights:  {'✅ ' + weights_path if has_weights else '⚠️ None (random init)'}")
print(f"   Binary Head:         {tuple(out['binary'].shape)} → 2 species")
print(f"   Cattle Head:         {tuple(out['cattle'].shape)} → 57 cattle breeds")
print(f"   Buffalo Head:        {tuple(out['buffalo'].shape)} → 18 buffalo breeds")
print(f"   Pooled Features:     {tuple(out['features'].shape)} → 1280-dim")
print(f"   Projection Head:     {tuple(out['embedding'].shape)} → 128-dim (SupCon training-only)")
print(f"   Soft Routing Output: {tuple(preds.shape)} → 75-class combined distribution")
print(f"   Total Parameters:    {total_params:,}")
print(f"   Trainable:           {trainable:,}")
print(f"   FP32 Model Size:     ~{total_params * 4 / 1e6:.1f} MB")

del model, x, out, preds
torch.cuda.empty_cache()

# %% [markdown]
# ---
# ## §6 — Full Training Pipeline
#
# Sequential multi-phase execution with:
# - **Dynamic VRAM auto-scaling**
# - **EMA on parameters AND BatchNorm buffers**
# - **SupCon auxiliary representation learning**
# - **Adaptive loss weights switching on binary saturation**

# %%
# ============================================================
#  🚀 FULL MULTI-PHASE TRAINING
# ============================================================
import copy
import random
import time
import numpy as np
import torch
from src.data_pipeline import (
    get_dataloaders, compute_class_counts, compute_class_priors, compute_rare_classes
)
from src.model import BreedClassifier
from src.train import (
    setup_device, train_phase, setup_qat, create_portable_export,
    _build_warmup_cosine_scheduler
)
from src.run_utils import make_run_id, timestamped, unique_path
from src.run_logger import init_run_logger

# Initialize run logger and run ID
run_id = make_run_id(RUN_TAG)
init_run_logger(exec_id=run_id, module="colab_train")

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Setup device & CUDA optimizations
device, use_amp = setup_device(None)

# Dynamic VRAM Auto-Scaling
actual_batch_size = BATCH_SIZE
actual_grad_accum = GRAD_ACCUM
if device.type == "cuda":
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    target_eff_batch = BATCH_SIZE * GRAD_ACCUM
    if vram_gb < 6.0:
        actual_batch_size = 16
    elif vram_gb < 10.0:
        actual_batch_size = 32
    elif vram_gb < 16.0:
        actual_batch_size = 64
    else:
        actual_batch_size = 128
    actual_grad_accum = max(1, target_eff_batch // actual_batch_size)
    print(f"[train] VRAM detected: {vram_gb:.1f} GB → Auto-scaled batch_size={actual_batch_size}, "
          f"grad_accum={actual_grad_accum} (effective batch = {actual_batch_size * actual_grad_accum})")

# Augmentation dictionary
augment = {
    "horizontal_flip": AUG_HORIZONTAL_FLIP,
    "random_resized_crop": AUG_RANDOM_RESIZED_CROP,
    "color_jitter": AUG_COLOR_JITTER,
    "randaugment": AUG_RANDAUGMENT,
}

# Create DataLoaders
loaders = get_dataloaders(
    split_dir=split_dir,
    batch_size=actual_batch_size,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    augment=augment,
    breed_augment=BREED_AUG,
    pad=PAD_TO_SQUARE
)
assert loaders is not None, "❌ Failed to create dataloaders"
train_loader, val_loader, test_loader = loaders

# Long-tail & auxiliary diagnostics
train_counts = compute_class_counts(split_dir)
rare_masks = compute_rare_classes(split_dir, RARE_CLASS_THRESHOLD)
if rare_masks is not None:
    rare_masks = {k: v.to(device) for k, v in rare_masks.items()}

logit_priors = None
adjust_tau = 0.0
if LOGIT_ADJUST:
    logit_priors = compute_class_priors(split_dir, source=LOGIT_ADJUST_PRIOR)
    if logit_priors is not None:
        adjust_tau = LOGIT_ADJUST_TAU
        logit_priors = {k: v.to(device) for k, v in logit_priors.items()}

mix_prob = CUTMIX_MIXUP_PROB if MIX_ENABLED else 0.0

# Initialize Model
model = BreedClassifier(
    backbone=BACKBONE,
    attention=ATTENTION,
    pretrained_path=weights_path if os.path.exists(weights_path) else None
)
model.to(device)
model = model.to(memory_format=torch.channels_last)

scaler = torch.amp.GradScaler("cuda") if use_amp else None
ckpt_dir = f"{PROJECT_DIR}/outputs/checkpoints"
base_ckpt = os.path.join(ckpt_dir, BACKBONE)

ckpt_p1 = unique_path(timestamped(f"{base_ckpt}_phase1_best.pt", run_id))
ckpt_p2 = unique_path(timestamped(f"{base_ckpt}_phase2_best.pt", run_id))
ckpt_p3 = unique_path(timestamped(f"{base_ckpt}_phase3_best.pt", run_id))
ckpt_quant = unique_path(timestamped(f"{base_ckpt}_quantized.pt", run_id))

phase1_epochs = 1 if DATA_MODE == "smoke" else PHASE1_EPOCHS
phase2_epochs = 1 if DATA_MODE == "smoke" else PHASE2_EPOCHS
phase3_epochs = 1 if DATA_MODE == "smoke" else PHASE3_EPOCHS

total_phases = 3 if INCLUDE_QAT else 2
print(f"\n{'=' * 60}")
print(f"  STARTING TRAINING: {total_phases} Phases | Run ID: {run_id}")
print(f"  Phase 1: {phase1_epochs} epochs | Phase 2: {phase2_epochs} epochs" +
      (f" | Phase 3: {phase3_epochs} epochs" if INCLUDE_QAT else ""))
print(f"  Effective Batch Size: {actual_batch_size * actual_grad_accum} ({actual_batch_size} × {actual_grad_accum})")
print(f"{'=' * 60}\n")

start_time = time.time()

# ─── Phase 1: All-Heads Warmup ───
print("━" * 60)
print(f"  PHASE 1: All-Heads Warmup ({phase1_epochs} epochs, lr={PHASE1_LR:.0e})")
print("━" * 60)
model.freeze_all()
for p in model.binary_head.parameters():
    p.requires_grad = True
for p in model.cattle_head.parameters():
    p.requires_grad = True
for p in model.buffalo_head.parameters():
    p.requires_grad = True
model.backbone_eval()

train_phase(
    model, train_loader, val_loader, device,
    phase=1, epochs=phase1_epochs, lr=PHASE1_LR,
    loss_weights=(LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO),
    scheduler_factory=None,
    checkpoint_path=ckpt_p1,
    scaler=scaler,
    best_key=BEST_METRIC,
    set_train=lambda m: (m.train(), m.backbone_eval()),
    weight_decay=WEIGHT_DECAY,
    grad_accum_steps=actual_grad_accum,
    label_smoothing=LABEL_SMOOTHING,
    eval_every=1,
    mix_prob=mix_prob,
    mix_off_frac=MIX_OFF_LAST_FRAC,
    same_species=MIX_SAME_SPECIES,
    train_counts=train_counts,
    rare_masks=rare_masks,
    logit_priors=logit_priors,
    adjust_tau=adjust_tau,
    contrastive_weight=0.0
)

# ─── Phase 2: Full Multi-Task Fine-Tuning ───
print(f"\n{'━' * 60}")
print(f"  PHASE 2: Multi-Task Fine-Tuning ({phase2_epochs} epochs, lr={PHASE2_LR:.0e})")
print(f"{'━' * 60}")
model.unfreeze_all()
model.train()

compiled_model = model
if hasattr(torch, "compile") and DATA_MODE != "smoke" and os.name != 'nt':
    try:
        print("[train] Compiling model for Phase 2 via torch.compile...")
        compiled_model = torch.compile(model)
    except Exception as e:
        print(f"[train] torch.compile skipped: {e}")
        compiled_model = model

warmup_ep = min(WARMUP_EPOCHS, phase2_epochs - 1) if DATA_MODE != "smoke" else 0

# EMA Model setup (weights + BatchNorm buffers)
ema_model = copy.deepcopy(model)
ema_model.eval()
for p in ema_model.parameters():
    p.requires_grad = False

train_phase(
    compiled_model, train_loader, val_loader, device,
    phase=2, epochs=phase2_epochs, lr=PHASE2_LR,
    loss_weights=(LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO),
    scheduler_factory=lambda opt: _build_warmup_cosine_scheduler(opt, warmup_ep, phase2_epochs),
    checkpoint_path=ckpt_p2,
    scaler=scaler,
    best_key=BEST_METRIC,
    weight_decay=WEIGHT_DECAY,
    grad_accum_steps=actual_grad_accum,
    label_smoothing=LABEL_SMOOTHING,
    eval_every=1 if DATA_MODE == "smoke" else 2,
    ema_model=ema_model,
    ema_decay=EMA_DECAY,
    ema_warmup=EMA_WARMUP,
    loss_weights_final=(LOSS_WEIGHT_BINARY_FINAL, LOSS_WEIGHT_CATTLE_FINAL, LOSS_WEIGHT_BUFFALO_FINAL),
    binary_sat_acc=BINARY_SATURATION_ACC,
    mix_prob=mix_prob,
    mix_off_frac=MIX_OFF_LAST_FRAC,
    same_species=MIX_SAME_SPECIES,
    train_counts=train_counts,
    rare_masks=rare_masks,
    logit_priors=logit_priors,
    adjust_tau=adjust_tau,
    contrastive_weight=CONTRASTIVE_WEIGHT,
    contrastive_temp=CONTRASTIVE_TEMPERATURE
)

best_checkpoint = ckpt_p2

# ─── Phase 3: QAT (Opt-in) ───
if INCLUDE_QAT:
    print(f"\n{'━' * 60}")
    print(f"  PHASE 3: Quantization-Aware Training (QAT, {phase3_epochs} epochs)")
    print(f"{'━' * 60}")
    if os.path.exists(ckpt_p2):
        bckpt = torch.load(ckpt_p2, map_location=device, weights_only=False)
        model.load_state_dict(bckpt["state_dict"])
        print(f"[train] QAT: Loaded best Phase 2 weights ({bckpt.get('best_metric', 'val')}={bckpt.get('val_top1', float('nan')):.4f})")

    model.unfreeze_all()
    qat_ok = setup_qat(model, device)

    train_phase(
        model, train_loader, val_loader, device,
        phase=3, epochs=phase3_epochs, lr=PHASE3_LR,
        loss_weights=(LOSS_WEIGHT_BINARY_FINAL, LOSS_WEIGHT_CATTLE_FINAL, LOSS_WEIGHT_BUFFALO_FINAL),
        scheduler_factory=None,
        checkpoint_path=ckpt_p3,
        scaler=None,  # AMP disabled for QAT
        best_key=BEST_METRIC,
        weight_decay=WEIGHT_DECAY,
        grad_accum_steps=actual_grad_accum,
        label_smoothing=LABEL_SMOOTHING,
        eval_every=1 if DATA_MODE == "smoke" else 2,
        mix_prob=mix_prob,
        mix_off_frac=MIX_OFF_LAST_FRAC,
        same_species=MIX_SAME_SPECIES,
        train_counts=train_counts,
        rare_masks=rare_masks,
        logit_priors=logit_priors,
        adjust_tau=adjust_tau,
        contrastive_weight=0.0
    )

    if qat_ok:
        try:
            import torch.ao.quantization as qat_lib
            model.eval()
            qat_lib.convert(model, inplace=True)
            torch.save({"state_dict": model.state_dict(), "quantized": True}, ckpt_quant)
            print(f"✅ QAT INT8 model converted: {ckpt_quant}")
        except Exception as exc:
            print(f"⚠️  QAT conversion failed: {exc}")

elapsed = time.time() - start_time
print(f"\n{'=' * 60}")
print(f"  ✅ TRAINING COMPLETED in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
print(f"  Best Checkpoint: {best_checkpoint}")
print(f"{'=' * 60}")

if torch.cuda.is_available():
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Peak GPU memory allocated: {peak_vram:.2f} GB")

# %% [markdown]
# ---
# ## §7 — Full Evaluation
#
# Evaluates the best trained model on the unseen test split and calculates:
# - Species accuracy, cattle & buffalo breed accuracies
# - Soft-routed Top-1, Top-3, Top-5 accuracy
# - Macro-F1 scores & Blended Score
# - Confusion matrices and few/medium/many-shot performance

# %%
# ============================================================
#  FULL MODEL EVALUATION
# ============================================================
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.evaluate import full_evaluation
from src.metrics import evaluate_epoch

print(f"[eval] Loading best checkpoint: {best_checkpoint}")
eval_model = BreedClassifier(backbone=BACKBONE, attention=ATTENTION)
ckpt = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# Filter out training-only auxiliary heads if needed
missing, unexpected = eval_model.load_state_dict(sd, strict=False)
if missing:
    print(f"[eval] Missing keys tolerated: {missing}")
eval_model.to(device)
eval_model.eval()

# Metrics on Validation and Test sets
val_metrics = evaluate_epoch(eval_model, val_loader, device, train_counts=train_counts)
test_metrics = evaluate_epoch(eval_model, test_loader, device, train_counts=train_counts)

print("\n" + "=" * 50)
print("  EVALUATION METRICS SUMMARY")
print("=" * 50)
print(f"  Binary Accuracy:          Val: {val_metrics['binary_acc']:.1%} | Test: {test_metrics['binary_acc']:.1%}")
print(f"  Cattle Breed Accuracy:    Val: {val_metrics['cattle_acc']:.1%} | Test: {test_metrics['cattle_acc']:.1%}")
print(f"  Buffalo Breed Accuracy:   Val: {val_metrics['buffalo_acc']:.1%} | Test: {test_metrics['buffalo_acc']:.1%}")
print(f"  Combined Soft Top-1:      Val: {val_metrics['combined_top1_soft']:.1%} | Test: {test_metrics['combined_top1_soft']:.1%}")
print(f"  Combined Soft Top-3:      Val: {val_metrics.get('combined_top3_soft', 0.0):.1%} | Test: {test_metrics.get('combined_top3_soft', 0.0):.1%}")
print(f"  Combined Soft Top-5:      Val: {val_metrics.get('combined_top5_soft', 0.0):.1%} | Test: {test_metrics.get('combined_top5_soft', 0.0):.1%}")
print(f"  Blended Score:            Val: {val_metrics.get('blended_score', 0.0):.3f} | Test: {test_metrics.get('blended_score', 0.0):.3f}")
print(f"  Few-Shot Accuracy:        Test: {test_metrics.get('acc_fewshot', 0.0):.1%}")
print(f"  Medium-Shot Accuracy:     Test: {test_metrics.get('acc_mediumshot', 0.0):.1%}")
print(f"  Many-Shot Accuracy:       Test: {test_metrics.get('acc_manyshot', 0.0):.1%}")
print("=" * 50 + "\n")

# Confusion matrices on Test split
metrics_dir = f"{PROJECT_DIR}/outputs/metrics"
os.makedirs(metrics_dir, exist_ok=True)
full_eval = full_evaluation(eval_model, test_loader, device)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, key, title in zip(axes, ["cattle_cm", "buffalo_cm"], ["Cattle", "Buffalo"]):
    cm = full_eval[key]
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(f"{title} Confusion Matrix (Test Split)", fontsize=13)
    ax.set_xlabel("Predicted Index")
    ax.set_ylabel("True Index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
cm_plot_path = f"{metrics_dir}/{BACKBONE}_{run_id}_confusion_matrices.png"
plt.savefig(cm_plot_path, dpi=130)
plt.show()

# Save JSON report
eval_report = {
    "run_id": run_id,
    "backbone": BACKBONE,
    "attention": ATTENTION,
    "checkpoint": best_checkpoint,
    "val_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in val_metrics.items()},
    "test_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in test_metrics.items()},
}
metrics_json_path = f"{metrics_dir}/{BACKBONE}_{run_id}_metrics.json"
with open(metrics_json_path, "w") as f:
    json.dump(eval_report, f, indent=2)

print(f"✅ Evaluation plots and JSON saved under {metrics_dir}")

del eval_model
torch.cuda.empty_cache()

# %% [markdown]
# ---
# ## §8 — Single Image Prediction (Soft Routing)
#
# Upload an image of a cattle or buffalo to test interactive inference using **soft routing**.

# %%
# ============================================================
#  🖼️  IMAGE PREDICTION WITH SOFT ROUTING
# ============================================================
from google.colab import files
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

print("📤 Upload an image of a cow or buffalo:")
uploaded = files.upload()

if uploaded:
    img_name = list(uploaded.keys())[0]
    img = Image.open(img_name).convert("RGB")

    pred_model = BreedClassifier(backbone=BACKBONE, attention=ATTENTION)
    bckpt = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    pred_model.load_state_dict(bckpt["state_dict"], strict=False)
    pred_model.to(device)
    pred_model.eval()

    # Preprocessing: shortest-side 260 + center crop + ImageNet normalization
    eval_transform = transforms.Compose([
        transforms.Resize(260),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = eval_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = pred_model(tensor)
        # Soft-routed 75-class combined distribution:
        # [p(cattle)*softmax(cattle), p(buffalo)*softmax(buffalo)]
        combined_probs = pred_model.predict(tensor)[0].cpu()

    # Class mappings
    with open(f"{split_dir}/cattle_classes.json") as f:
        cattle_classes = json.load(f)
    with open(f"{split_dir}/buffalo_classes.json") as f:
        buffalo_classes = json.load(f)

    cattle_names = sorted(cattle_classes.keys(), key=lambda k: cattle_classes[k])
    buffalo_names = sorted(buffalo_classes.keys(), key=lambda k: buffalo_classes[k])
    all_names = cattle_names + buffalo_names

    species_probs = F.softmax(out["binary"], dim=1)[0].cpu()
    species_idx = species_probs.argmax().item()
    species_name = "Cattle" if species_idx == 0 else "Buffalo"
    species_conf = species_probs[species_idx].item()

    # Top-5 overall across all 75 breeds
    top5_vals, top5_idxs = combined_probs.topk(5)

    print("\n" + "=" * 55)
    print(f"  Primary Species: {species_name.upper()} ({species_conf:.1%} confidence)")
    print("  Top-5 Breed Predictions (Soft Routing):")
    print("-" * 55)
    for i, (p, idx) in enumerate(zip(top5_vals, top5_idxs)):
        idx_val = idx.item()
        breed_label = all_names[idx_val]
        sp_tag = "🐄 Cattle" if idx_val < 57 else "🐃 Buffalo"
        bar = "█" * int(p.item() * 30)
        print(f"   {i+1}. [{sp_tag}] {breed_label:22s} {p.item():.1%}  {bar}")
    print("=" * 55)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(f"Predicted: {species_name} — {all_names[top5_idxs[0].item()]}\n"
                 f"Confidence: {top5_vals[0].item():.1%}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    del pred_model
    torch.cuda.empty_cache()

# %% [markdown]
# ---
# ## §9 — Multi-Format Export
#
# Automatically exports the trained model into:
# 1. **Self-Contained Portable Folder** (`outputs/export/portable/<backbone>_*/`)
# 2. **FP32 ONNX Model** (opset 13, caller-normalized, compatible with desktop `test_model.py`)
# 3. **Mobile INT8 ONNX Model** (opset 17, QDQ static quantization with baked-in normalization)
# 4. **TFLite INT8 / FP32** (optional full-integer mobile quantization)
# 5. **Parity Check** (verifies that exported INT8/ONNX stays within 1% top-1 of PyTorch)

# %%
# ============================================================
#  📦 MULTI-FORMAT EXPORT & PARITY GATE
# ============================================================
import torch.nn as nn
from src.export import (
    create_portable_export, export_onnx_int8, _load_model,
    _RawOutputs, _MobileOutputs, _write_label_files
)

export_dir = f"{PROJECT_DIR}/outputs/export"
portable_export_dir = f"{export_dir}/portable"
os.makedirs(export_dir, exist_ok=True)

# 1. Portable Bundle
print("📦 [1/4] Creating portable self-contained bundle...")
portable_dir = create_portable_export(best_checkpoint, BACKBONE, split_dir, portable_export_dir)
print(f"✅ Portable bundle saved: {portable_dir}")

# 2. FP32 Desktop ONNX (opset 13)
print("\n📦 [2/4] Creating FP32 ONNX model (caller-normalized)...")
export_model = _load_model(best_checkpoint, BACKBONE, ATTENTION)
export_model.eval()

raw_wrapper = _RawOutputs(export_model).eval()
onnx_fp32_path = f"{export_dir}/{BACKBONE}_{run_id}_fp32.onnx"
dummy_input = torch.randn(1, 3, 260, 260)

try:
    torch.onnx.export(
        raw_wrapper, dummy_input, onnx_fp32_path,
        input_names=["input"],
        output_names=["binary", "cattle", "buffalo"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "binary": {0: "batch"},
                      "cattle": {0: "batch"}, "buffalo": {0: "batch"}},
        dynamo=False,
    )
    sz_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    print(f"✅ FP32 ONNX exported: {onnx_fp32_path} ({sz_mb:.1f} MB)")
except Exception as exc:
    print(f"⚠️  FP32 ONNX export failed: {exc}")

# 3. Mobile QDQ INT8 ONNX (opset 17, normalization baked in)
print("\n📦 [3/4] Creating Mobile INT8 ONNX model (QDQ quantization)...")
mobile_wrapper = _MobileOutputs(export_model).eval()
mobile_fp32_onnx = f"{export_dir}/{BACKBONE}_{run_id}_mobile_fp32.onnx"
mobile_int8_onnx = f"{export_dir}/{BACKBONE}_{run_id}_mobile_int8.onnx"

try:
    torch.onnx.export(
        mobile_wrapper, dummy_input, mobile_fp32_onnx,
        input_names=["input"],
        output_names=["binary", "cattle", "buffalo"],
        opset_version=17,
        dynamic_axes=None, # static batch 1 for mobile
        dynamo=False,
    )
    # Quantize using train-split calibration images
    export_onnx_int8(export_model, mobile_fp32_onnx, mobile_int8_onnx, split_dir, calibration_images=200)
    int8_sz_mb = os.path.getsize(mobile_int8_onnx) / (1024 * 1024)
    print(f"✅ Mobile INT8 ONNX exported: {mobile_int8_onnx} ({int8_sz_mb:.1f} MB)")
except Exception as exc:
    print(f"⚠️  Mobile INT8 ONNX quantization skipped or failed: {exc}")

# Write label files for mobile deployment
labels = _write_label_files(split_dir, export_dir)
print(f"✅ Mobile label text files generated: {', '.join(labels)}")

# 4. Parity Check Verification
print("\n🔍 [4/4] Running Parity Check (PyTorch vs ONNX)...")
try:
    from src.parity_check import make_torch_runner, make_onnx_runner, compare_predictions
    t_run, _ = make_torch_runner(export_model, device)
    if os.path.exists(onnx_fp32_path):
        o_run, _ = make_onnx_runner(onnx_fp32_path, mobile=False)
        # Synthetic sanity comparison
        test_x = torch.rand(4, 3, 260, 260)
        t_out = t_run(test_x)
        o_out = o_run(test_x)
        max_diff = max((t - torch.from_numpy(o)).abs().max().item() for t, o in zip(t_out, o_out))
        print(f"✅ Parity check (fp32 ONNX vs PyTorch): Max absolute logit diff = {max_diff:.4f}")
except Exception as exc:
    print(f"⚠️  Parity check skipped: {exc}")

del export_model, raw_wrapper, mobile_wrapper
torch.cuda.empty_cache()

# %% [markdown]
# ---
# ## §10 — Zip & Download Results
#
# Collects all checkpoints, portable exports, ONNX models, metrics, and labels into a single archive for download.

# %%
# ============================================================
#  📥 PACKAGE & DOWNLOAD TRAINING ARTIFACTS
# ============================================================
import shutil
from google.colab import files

package_dir = f"/content/{BACKBONE}_{run_id}_results"
os.makedirs(package_dir, exist_ok=True)

# 1. Portable Bundle
if os.path.exists(portable_dir):
    shutil.copytree(portable_dir, f"{package_dir}/portable", dirs_exist_ok=True)

# 2. Checkpoints
ckpt_target_dir = f"{package_dir}/checkpoints"
os.makedirs(ckpt_target_dir, exist_ok=True)
for pt_file in glob.glob(f"{ckpt_dir}/*{run_id}*.pt"):
    shutil.copy2(pt_file, ckpt_target_dir)

# 3. ONNX & Mobile artifacts
for onnx_f in glob.glob(f"{export_dir}/*{run_id}*.*"):
    shutil.copy2(onnx_f, package_dir)

for lbl_f in glob.glob(f"{export_dir}/labels_*.txt"):
    shutil.copy2(lbl_f, package_dir)

# 4. Metrics & Plots
metrics_target_dir = f"{package_dir}/metrics"
os.makedirs(metrics_target_dir, exist_ok=True)
for m_f in glob.glob(f"{metrics_dir}/*{run_id}*.*"):
    shutil.copy2(m_f, metrics_target_dir)

if os.path.exists(f"{PROJECT_DIR}/outputs/class_distribution.png"):
    shutil.copy2(f"{PROJECT_DIR}/outputs/class_distribution.png", metrics_target_dir)

# Create zip
zip_base = f"/content/{BACKBONE}_{run_id}_complete_package"
shutil.make_archive(zip_base, "zip", package_dir)
final_zip_path = f"{zip_base}.zip"
final_zip_size = os.path.getsize(final_zip_path) / (1024 * 1024)

print(f"\n✅ All artifacts packaged successfully:")
print(f"   Archive: {final_zip_path} ({final_zip_size:.1f} MB)")

# Trigger download
files.download(final_zip_path)

# %%
# ============================================================
#  💾 SAVE TO GOOGLE DRIVE (Optional)
# ============================================================
# Uncomment to automatically copy results to your Google Drive:

# from google.colab import drive
# drive.mount("/content/drive")
# DRIVE_BACKUP = f"/content/drive/MyDrive/ML-CB-B-identifier/runs/{run_id}"
# os.makedirs(DRIVE_BACKUP, exist_ok=True)
# shutil.copy2(final_zip_path, DRIVE_BACKUP)
# print(f"✅ Saved results archive to Google Drive: {DRIVE_BACKUP}")

# %% [markdown]
# ---
# ## §11 — GPU Memory Monitor (Optional)
#
# Run this cell at any point during or after training to inspect VRAM usage.

# %%
# ============================================================
#  GPU MEMORY MONITOR
# ============================================================
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"📊 GPU Memory Stats ({torch.cuda.get_device_name(0)}):")
    print(f"   Currently Allocated: {allocated:.2f} GB")
    print(f"   Reserved / Cached:   {reserved:.2f} GB")
    print(f"   Peak Allocated:      {max_allocated:.2f} GB")
    print(f"   Total VRAM:          {total:.1f} GB")
    print(f"   Free VRAM:           {total - reserved:.2f} GB")
else:
    print("No CUDA GPU available.")
