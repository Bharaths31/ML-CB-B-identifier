# %% [markdown]
# # 🐄 Cattle & Buffalo Breed Classifier — Colab Training
#
# **3-Phase Training Pipeline** optimized for **T4 GPU** (15 GB VRAM)
#
# | Phase | Description | AMP |
# |-------|-------------|-----|
# | 1 | Binary head warmup (frozen backbone) | ✅ |
# | 2 | Full multi-task fine-tuning (AdamW + warmup + cosine) | ✅ |
# | 3 | QAT — Quantization-Aware Training (for Android) | ❌ |
#
# **Target deployment**: Mid-range Android phone (INT8 quantized model)

# %% [markdown]
# ---
# ## §0 — GPU Check & Environment Setup

# %%
# ============================================================
#  GPU VERIFICATION & CUDA SETUP
# ============================================================
import torch, subprocess, sys, os

def check_gpu():
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Go to: Runtime → Change runtime type → T4 GPU")
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
# Colab already has torch/torchvision; install only what's missing
!pip install -q pandas tqdm matplotlib scikit-learn Pillow numpy onnx

# %% [markdown]
# ---
# ## §1 — Project Setup (Source Code)
#
# Choose **ONE** of the three options below:
# - **Option A**: Clone from GitHub (easiest — public repo)
# - **Option B**: Upload `colab_project.zip` manually
# - **Option C**: Mount Google Drive and copy from Drive

# %%
# ============================================================
#  OPTION A: CLONE FROM GITHUB (Recommended)
# ============================================================
import shutil
GITHUB_REPO = "https://github.com/Bharaths31/ML-CB-B-identifier"
PROJECT_DIR = "/content/project"

# Step 1: Move to /content FIRST to avoid corrupting shell cwd
os.chdir("/content")

# Step 2: Remove old clone if it exists (force-refresh to pick up latest code)
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
    print(f"🗑️  Removed old clone at {PROJECT_DIR}")

# Step 3: Clone fresh
import subprocess
result = subprocess.run(
    ["git", "clone", GITHUB_REPO, PROJECT_DIR],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"❌ Clone failed:\n{result.stderr}")
    raise RuntimeError("git clone failed — check the repo URL and your internet connection")
print(f"✅ Cloned repo to {PROJECT_DIR}")

# Step 4: Add to Python path so `import src` works
import sys
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Step 5: Verify src/ structure
assert os.path.exists(f"{PROJECT_DIR}/src/config.py"), \
    f"❌ src/config.py not found in {PROJECT_DIR} — clone may have nested the repo incorrectly"
print(f"✅ Source code ready at {PROJECT_DIR}/src/")

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
import sys
# Ensure PROJECT_DIR is defined (in case user jumped straight here)
PROJECT_DIR = PROJECT_DIR if 'PROJECT_DIR' in dir() else "/content/project"

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# chdir to PROJECT_DIR so relative imports work, but only if it exists
if os.path.exists(PROJECT_DIR):
    os.chdir(PROJECT_DIR)
    print(f"✅ Working directory: {os.getcwd()}")
    print(f"✅ Python path includes: {PROJECT_DIR}")
else:
    raise RuntimeError(f"❌ {PROJECT_DIR} does not exist — run one of the setup options above first")

# Verify imports work
try:
    from src.config import IMAGE_SIZE, BATCH_SIZE
    print(f"✅ src.config imported (IMAGE_SIZE={IMAGE_SIZE}, BATCH_SIZE={BATCH_SIZE})")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("   Make sure you ran one of the setup options above")

# %% [markdown]
# ---
# ## §2 — Dataset Acquisition
#
# Choose **ONE** of three options:
# - **Option A**: Download from Kaggle API
# - **Option B**: Upload `archive.zip` manually
# - **Option C**: Copy from Google Drive

# %%
# ============================================================
#  OPTION A: KAGGLE API DOWNLOAD (Recommended)
# ============================================================
# Set your Kaggle credentials below:
KAGGLE_USERNAME = ""  # ← Fill in your Kaggle username
KAGGLE_KEY = ""       # ← Fill in your Kaggle API key

# Kaggle dataset slugs
CATTLE_DATASET = "atharvadarpude/indian-cattle-image-dataset"
BUFFALO_DATASET = "atharvadarpude/indian-buffalo-dataset"

DATA_RAW = f"{PROJECT_DIR}/data/raw"

if KAGGLE_USERNAME and KAGGLE_KEY:
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    !pip install -q kaggle

    # Download cattle dataset
    !mkdir -p /content/kaggle_data
    !kaggle datasets download -d {CATTLE_DATASET} -p /content/kaggle_data --unzip
    print("✅ Cattle dataset downloaded")

    # Download buffalo dataset
    !kaggle datasets download -d {BUFFALO_DATASET} -p /content/kaggle_data --unzip
    print("✅ Buffalo dataset downloaded")

    # Organize into expected structure: data/raw/cattle/<breed>/ & data/raw/buffalo/<breed>/
    !mkdir -p {DATA_RAW}/cattle {DATA_RAW}/buffalo

    # Move files — adjust paths based on actual dataset structure
    import shutil, glob
    for species in ["cattle", "buffalo"]:
        src_dirs = glob.glob(f"/content/kaggle_data/**/{species}/**", recursive=True)
        for src_dir in src_dirs:
            if os.path.isdir(src_dir):
                breed = os.path.basename(src_dir)
                dst = f"{DATA_RAW}/{species}/{breed}"
                if not os.path.exists(dst) and breed not in (species, ""):
                    shutil.copytree(src_dir, dst, dirs_exist_ok=True)

    print(f"✅ Dataset organized at {DATA_RAW}")
else:
    print("⚠️  Kaggle credentials not set — skip this cell or fill in above")
    print("   Alternatively, use Option B (upload) or Option C (Drive)")

# %%
# ============================================================
#  OPTION B: UPLOAD archive.zip MANUALLY
#  (Skip if you used Option A or C)
# ============================================================
# Create archive.zip locally: python scripts/create_colab_archive.py
# Then upload it here.

# Uncomment to use:
# from google.colab import files
# DATA_RAW = f"{PROJECT_DIR}/data/raw"
# print("📤 Upload archive.zip (containing cattle/ and buffalo/ folders)...")
# uploaded = files.upload()
# !mkdir -p {DATA_RAW}
# !unzip -qo /content/archive.zip -d {DATA_RAW}
# print(f"✅ Dataset extracted to {DATA_RAW}")

# %%
# ============================================================
#  OPTION C: COPY FROM GOOGLE DRIVE
#  (Skip if you used Option A or B)
# ============================================================
# Uncomment to use:
# from google.colab import drive
# drive.mount("/content/drive")
# DATA_RAW = f"{PROJECT_DIR}/data/raw"
# DRIVE_ARCHIVE = "/content/drive/MyDrive/ML-CB-B-identifier/archive.zip"
# !mkdir -p {DATA_RAW}
# !unzip -qo {DRIVE_ARCHIVE} -d {DATA_RAW}
# print(f"✅ Dataset extracted from Drive to {DATA_RAW}")

# %%
# ============================================================
#  VERIFY DATASET
# ============================================================
DATA_RAW = f"{PROJECT_DIR}/data/raw"
cattle_dir = f"{DATA_RAW}/cattle"
buffalo_dir = f"{DATA_RAW}/buffalo"

cattle_breeds = sorted(os.listdir(cattle_dir)) if os.path.exists(cattle_dir) else []
buffalo_breeds = sorted(os.listdir(buffalo_dir)) if os.path.exists(buffalo_dir) else []

cattle_breeds = [b for b in cattle_breeds if os.path.isdir(f"{cattle_dir}/{b}")]
buffalo_breeds = [b for b in buffalo_breeds if os.path.isdir(f"{buffalo_dir}/{b}")]

print(f"📊 Dataset Summary:")
print(f"   Cattle breeds:  {len(cattle_breeds)}")
print(f"   Buffalo breeds: {len(buffalo_breeds)}")

# Count images
total_imgs = 0
for b in cattle_breeds:
    total_imgs += len([f for f in os.listdir(f"{cattle_dir}/{b}")
                       if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
for b in buffalo_breeds:
    total_imgs += len([f for f in os.listdir(f"{buffalo_dir}/{b}")
                       if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
print(f"   Total images:   {total_imgs}")

assert len(cattle_breeds) > 0 or len(buffalo_breeds) > 0, \
    "❌ No breed folders found! Check dataset extraction."
print("✅ Dataset ready!")

# %% [markdown]
# ---
# ## §3 — Hyperparameter Configuration
#
# **Modify these values before training.** All training parameters are in this cell.

# %%
# ============================================================
#  ⚙️  HYPERPARAMETER CONFIGURATION
#  Modify these values to tune training behavior
# ============================================================

# --- Model Architecture ---
BACKBONE = "lite2"           # "lite2" (~6M params, faster) or "lite4" (~13M, more accurate)
ATTENTION = "cbam"           # "cbam" (CBAM attention) or "se" (Squeeze-Excite)

# --- Training ---
BATCH_SIZE = 64              # T4 can handle 64 for lite2 @ 260px (reduce to 32 if OOM)
GRAD_ACCUM = 2               # Gradient accumulation steps (effective batch = BATCH_SIZE * GRAD_ACCUM)
NUM_WORKERS = 2              # Colab has 2 CPU cores
SEED = 42                    # Reproducibility

# --- Phase 1: Binary head warmup ---
PHASE1_EPOCHS = 5
PHASE1_LR = 3e-3             # Higher LR for head-only training

# --- Phase 2: Full multi-task fine-tuning ---
PHASE2_EPOCHS = 40           # Main training phase
PHASE2_LR = 2e-4             # AdamW learning rate
WARMUP_EPOCHS = 3            # Linear warmup before cosine decay

# --- Phase 3: QAT (Quantization-Aware Training) ---
PHASE3_EPOCHS = 10           # QAT fine-tuning
PHASE3_LR = 5e-6             # Very low LR for quantization stability
SKIP_QAT = False             # ← Set to True to skip QAT (NOT recommended for Android)

# --- Optimizer (AdamW) ---
WEIGHT_DECAY = 1e-2          # L2 regularization (decoupled)
LABEL_SMOOTHING = 0.1        # Prevents overconfident predictions

# --- Data Split ---
TRAIN_RATIO = 0.85           # 85% train
VAL_RATIO = 0.10             # 10% validation
TEST_RATIO = 0.05            # 5% test

# --- Augmentation ---
RANDAUGMENT_OPS = 2          # Number of augmentation operations
RANDAUGMENT_MAG = 9          # Augmentation magnitude
CUTMIX_ALPHA = 0.4           # CutMix beta distribution α
MIXUP_ALPHA = 0.2            # MixUp beta distribution α

# --- Loss weights ---
LOSS_WEIGHT_BINARY = 0.50
LOSS_WEIGHT_CATTLE = 0.25
LOSS_WEIGHT_BUFFALO = 0.25

print("=" * 60)
print("  HYPERPARAMETER CONFIGURATION")
print("=" * 60)
print(f"  Backbone:   {BACKBONE} + {ATTENTION} attention")
print(f"  Batch size: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
print(f"  Phases:     {PHASE1_EPOCHS}/{PHASE2_EPOCHS}/{PHASE3_EPOCHS} epochs")
print(f"  LRs:        {PHASE1_LR}/{PHASE2_LR}/{PHASE3_LR}")
print(f"  Warmup:     {WARMUP_EPOCHS} epochs (phase 2)")
print(f"  Optimizer:  AdamW (wd={WEIGHT_DECAY})")
print(f"  Smoothing:  {LABEL_SMOOTHING}")
print(f"  Split:      {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
print(f"  QAT:        {'ENABLED' if not SKIP_QAT else 'DISABLED'}")
print("=" * 60)

# %% [markdown]
# ---
# ## §4 — Data Split & Preprocessing

# %%
# ============================================================
#  PREPARE DATA SPLITS & DISPLAY STATISTICS
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data_pipeline import prepare_splits
from src import config as cfg

# Override config with notebook hyperparameters
cfg.TRAIN_RATIO = TRAIN_RATIO
cfg.VAL_RATIO = VAL_RATIO
cfg.TEST_RATIO = TEST_RATIO
cfg.BATCH_SIZE = BATCH_SIZE
cfg.RANDAUGMENT_OPS = RANDAUGMENT_OPS
cfg.RANDAUGMENT_MAGNITUDE = RANDAUGMENT_MAG
cfg.CUTMIX_ALPHA = CUTMIX_ALPHA
cfg.MIXUP_ALPHA = MIXUP_ALPHA

# Ensure output directories exist
os.makedirs(f"{PROJECT_DIR}/data/splits", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/outputs/checkpoints", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/outputs/export/portable", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/outputs/metrics", exist_ok=True)

summary = prepare_splits(
    data_root=f"{PROJECT_DIR}/data/raw",
    split_dir=f"{PROJECT_DIR}/data/splits"
)

if summary is None:
    raise RuntimeError("❌ No data found! Check dataset setup (§2)")

print(f"\n✅ Splits created: train={summary['train']} val={summary['val']} "
      f"test={summary['test']}")

# %%
# ============================================================
#  VISUALIZE CLASS DISTRIBUTION
# ============================================================
import pandas as pd
import json

split_dir = f"{PROJECT_DIR}/data/splits"
train_df = pd.read_csv(f"{split_dir}/train.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, species in zip(axes, ["cattle", "buffalo"]):
    sp_df = train_df[train_df["species"] == species]
    counts = sp_df["breed"].value_counts().sort_index()
    if len(counts) > 0:
        counts.plot(kind="barh", ax=ax, color="#4CAF50" if species == "cattle" else "#2196F3")
        ax.set_title(f"{species.title()} Breed Distribution (Train)")
        ax.set_xlabel("Image Count")

plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/outputs/class_distribution.png", dpi=100)
plt.show()
print("✅ Class distribution saved")

# %% [markdown]
# ---
# ## §5 — Architecture Verification

# %%
# ============================================================
#  VERIFY MODEL ARCHITECTURE
# ============================================================
from src.model import BreedClassifier
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if pretrained weights exist
weights_path = f"{PROJECT_DIR}/efficientnet_{BACKBONE}.pth"
has_weights = os.path.exists(weights_path)

model = BreedClassifier(
    backbone=BACKBONE,
    attention=ATTENTION,
    pretrained_path=weights_path if has_weights else None
)
model.eval()

# Forward pass test
with torch.no_grad():
    x = torch.randn(1, 3, 260, 260)
    out = model(x)

print(f"✅ Model Architecture Verified:")
print(f"   Backbone:       EfficientNet-{BACKBONE}")
print(f"   Attention:      {ATTENTION.upper()}")
print(f"   Pretrained:     {'✅ ' + weights_path if has_weights else '❌ Training from scratch'}")
print(f"   Binary head:    {tuple(out['binary'].shape)} → 2 classes")
print(f"   Cattle head:    {tuple(out['cattle'].shape)} → {out['cattle'].shape[1]} breeds")
print(f"   Buffalo head:   {tuple(out['buffalo'].shape)} → {out['buffalo'].shape[1]} breeds")
print(f"   Feature dim:    {out['features'].shape[1]}")

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total params:   {total_params:,}")
print(f"   Trainable:      {trainable:,}")
print(f"   Model size:     ~{total_params * 4 / 1e6:.1f} MB (FP32)")

del model, x, out
torch.cuda.empty_cache()

# %% [markdown]
# ---
# ## §6 — Training (3-Phase Pipeline)
#
# **This is the main training cell.** It runs all 3 phases sequentially.
# Runtime: ~2-4 hours on T4 (depends on dataset size).

# %%
# ============================================================
#  🚀 FULL 3-PHASE TRAINING
# ============================================================
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from tqdm import tqdm
from src.model import BreedClassifier
from src.data_pipeline import get_dataloaders
from src.metrics import evaluate_epoch
from src.train import (setup_device, soft_ce, masked_loss, run_epoch,
                       train_phase, setup_qat, create_portable_export,
                       _build_warmup_cosine_scheduler)
from src import config as cfg

# --- Apply hyperparameters to config ---
cfg.BATCH_SIZE = BATCH_SIZE
cfg.WEIGHT_DECAY = WEIGHT_DECAY
cfg.LABEL_SMOOTHING = LABEL_SMOOTHING
cfg.WARMUP_EPOCHS = WARMUP_EPOCHS
cfg.GRADIENT_ACCUMULATION_STEPS = GRAD_ACCUM

# --- Seed everything ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- Device setup ---
device, use_amp = setup_device(None)

# --- Data loaders ---
split_dir = f"{PROJECT_DIR}/data/splits"
loaders = get_dataloaders(
    split_dir=split_dir,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
assert loaders is not None, "❌ Failed to create dataloaders"
train_loader, val_loader, test_loader = loaders

# --- Model ---
weights_path = f"{PROJECT_DIR}/efficientnet_{BACKBONE}.pth"
model = BreedClassifier(
    backbone=BACKBONE,
    attention=ATTENTION,
    pretrained_path=weights_path if os.path.exists(weights_path) else None
)
if not os.path.exists(weights_path):
    print(f"⚠️  {weights_path} not found — training backbone from scratch")
model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"[train] params: trainable={trainable:,} total={total_params:,}")

# --- AMP scaler ---
scaler = torch.amp.GradScaler("cuda") if use_amp else None
if scaler:
    print("[train] mixed-precision: GradScaler enabled")

# --- Checkpoint paths ---
ckpt_dir = f"{PROJECT_DIR}/outputs/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
base = f"{ckpt_dir}/{BACKBONE}"

total_phases = 2 if SKIP_QAT else 3
print(f"\n{'=' * 60}")
print(f"  TRAINING PLAN: {total_phases} phases")
print(f"  Epochs: {PHASE1_EPOCHS}/{PHASE2_EPOCHS}" +
      (f"/{PHASE3_EPOCHS}" if not SKIP_QAT else ""))
print(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
print(f"  Optimizer: AdamW (wd={WEIGHT_DECAY})")
print(f"  Label smoothing: {LABEL_SMOOTHING}")
print(f"{'=' * 60}\n")

start_time = time.time()
history = {"phase": [], "epoch": [], "loss": [], "val_top1": []}

# ─── Phase 1: Binary head warmup ───
print("━" * 60)
print("  PHASE 1: Binary Head Warmup")
print("━" * 60)
model.freeze_all()
for p in model.binary_head.parameters():
    p.requires_grad = True
model.backbone_eval()

train_phase(
    model, train_loader, val_loader, device,
    phase=1, epochs=PHASE1_EPOCHS, lr=PHASE1_LR,
    loss_weights=(1.0, 0.0, 0.0),
    scheduler_factory=None,
    checkpoint_path=f"{base}_phase1_best.pt",
    scaler=scaler,
    best_key="binary_acc",
    set_train=lambda m: (m.train(), m.backbone_eval()),
    weight_decay=WEIGHT_DECAY,
    grad_accum_steps=GRAD_ACCUM,
    label_smoothing=LABEL_SMOOTHING,
)

# ─── Phase 2: Full multi-task fine-tuning ───
print(f"\n{'━' * 60}")
print("  PHASE 2: Multi-Task Fine-Tuning")
print(f"━" * 60)
model.unfreeze_all()
model.train()

warmup_ep = min(WARMUP_EPOCHS, PHASE2_EPOCHS - 1)
print(f"[train] LR schedule: {warmup_ep}ep warmup → cosine decay")

train_phase(
    model, train_loader, val_loader, device,
    phase=2, epochs=PHASE2_EPOCHS, lr=PHASE2_LR,
    loss_weights=(LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO),
    scheduler_factory=lambda opt: _build_warmup_cosine_scheduler(
        opt, warmup_ep, PHASE2_EPOCHS),
    checkpoint_path=f"{base}_phase2_best.pt",
    scaler=scaler,
    weight_decay=WEIGHT_DECAY,
    grad_accum_steps=GRAD_ACCUM,
    label_smoothing=LABEL_SMOOTHING,
)

# ─── Phase 3: QAT ───
best_checkpoint = f"{base}_phase2_best.pt"
if not SKIP_QAT:
    print(f"\n{'━' * 60}")
    print("  PHASE 3: Quantization-Aware Training (QAT)")
    print(f"━" * 60)
    model.unfreeze_all()
    qat_ok = setup_qat(model, device)

    # AMP MUST be disabled for QAT (quantization observers don't support mixed precision)
    train_phase(
        model, train_loader, val_loader, device,
        phase=3, epochs=PHASE3_EPOCHS, lr=PHASE3_LR,
        loss_weights=(LOSS_WEIGHT_BINARY, LOSS_WEIGHT_CATTLE, LOSS_WEIGHT_BUFFALO),
        scheduler_factory=None,
        checkpoint_path=f"{base}_phase3_best.pt",
        scaler=None,  # No AMP for QAT
        weight_decay=WEIGHT_DECAY,
        grad_accum_steps=GRAD_ACCUM,
        label_smoothing=LABEL_SMOOTHING,
    )

    if qat_ok:
        try:
            import torch.ao.quantization as qat_lib
            model.eval()
            qat_lib.convert(model, inplace=True)
            torch.save({"state_dict": model.state_dict()},
                        f"{base}_quantized.pt")
            print(f"✅ INT8 model saved: {base}_quantized.pt")
        except Exception as exc:
            print(f"⚠️  INT8 conversion failed ({exc})")

    best_checkpoint = f"{base}_phase3_best.pt"

elapsed = time.time() - start_time
elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
print(f"\n{'=' * 60}")
print(f"  ✅ TRAINING COMPLETE in {elapsed_str}")
print(f"  Checkpoints: {ckpt_dir}")
print(f"{'=' * 60}")

# Show VRAM usage
if torch.cuda.is_available():
    allocated = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Peak GPU memory: {allocated:.2f} GB")

# %% [markdown]
# ---
# ## §7 — Evaluation

# %%
# ============================================================
#  FULL MODEL EVALUATION
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from src.evaluate import full_evaluation
from src.metrics import evaluate_epoch

# Load best checkpoint
ckpt_path = best_checkpoint
if not os.path.exists(ckpt_path):
    # Fallback to phase 2
    ckpt_path = f"{ckpt_dir}/{BACKBONE}_phase2_best.pt"

print(f"[eval] Loading checkpoint: {ckpt_path}")
eval_model = BreedClassifier(backbone=BACKBONE, attention=ATTENTION)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
eval_model.load_state_dict(ckpt["state_dict"])
eval_model.to(device)
eval_model.eval()

# Evaluate on val and test sets
val_metrics = evaluate_epoch(eval_model, val_loader, device)
test_metrics = evaluate_epoch(eval_model, test_loader, device)

print("\n📊 Validation Metrics:")
for k, v in val_metrics.items():
    print(f"  {k:20s} {v:.4f}")

print("\n📊 Test Metrics:")
for k, v in test_metrics.items():
    print(f"  {k:20s} {v:.4f}")

# Confusion matrices
metrics_dir = f"{PROJECT_DIR}/outputs/metrics"
os.makedirs(metrics_dir, exist_ok=True)
full = full_evaluation(eval_model, test_loader, device)

# Save confusion matrices
import numpy as np
np.savetxt(f"{metrics_dir}/{BACKBONE}_cattle_cm.csv",
           full["cattle_cm"], delimiter=",", fmt="%d")
np.savetxt(f"{metrics_dir}/{BACKBONE}_buffalo_cm.csv",
           full["buffalo_cm"], delimiter=",", fmt="%d")

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, key, title in zip(axes,
                           ["cattle_cm", "buffalo_cm"],
                           ["Cattle", "Buffalo"]):
    im = ax.imshow(full[key], cmap="Blues")
    ax.set_title(f"{title} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{metrics_dir}/{BACKBONE}_confusion_matrices.png", dpi=120)
plt.show()

# Save metrics JSON
report = {
    "backbone": BACKBONE,
    "val": val_metrics,
    "test": test_metrics,
}
with open(f"{metrics_dir}/{BACKBONE}_metrics.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\n✅ Metrics saved to {metrics_dir}")

del eval_model
torch.cuda.empty_cache()

# %% [markdown]
# ---
# ## §8 — Predict with Image
#
# Upload an image and get the breed prediction.

# %%
# ============================================================
#  🖼️  IMAGE PREDICTION
# ============================================================
from google.colab import files
from PIL import Image
from torchvision import transforms

print("📤 Upload an image of a cattle or buffalo:")
uploaded = files.upload()

if uploaded:
    img_name = list(uploaded.keys())[0]
    img = Image.open(img_name).convert("RGB")

    # Load model for prediction
    pred_model = BreedClassifier(backbone=BACKBONE, attention=ATTENTION)
    ckpt = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    pred_model.load_state_dict(ckpt["state_dict"])
    pred_model.to(device)
    pred_model.eval()

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize(260),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        out = pred_model(tensor)

    binary_pred = out["binary"].argmax(1).item()
    species = "cattle" if binary_pred == 0 else "buffalo"
    binary_conf = F.softmax(out["binary"], dim=1)[0]

    # Load class maps
    with open(f"{PROJECT_DIR}/data/splits/cattle_classes.json") as f:
        cattle_classes = json.load(f)
    with open(f"{PROJECT_DIR}/data/splits/buffalo_classes.json") as f:
        buffalo_classes = json.load(f)

    if species == "cattle":
        probs = F.softmax(out["cattle"], dim=1)[0]
        idx_to_breed = {v: k for k, v in cattle_classes.items()}
    else:
        probs = F.softmax(out["buffalo"], dim=1)[0]
        idx_to_breed = {v: k for k, v in buffalo_classes.items()}

    top5_vals, top5_idxs = probs.topk(5)

    print(f"\n{'=' * 50}")
    print(f"  Species: {species.upper()} ({binary_conf[binary_pred]:.1%} confidence)")
    print(f"  Top-5 Breed Predictions:")
    for i, (val, idx) in enumerate(zip(top5_vals, top5_idxs)):
        breed = idx_to_breed.get(idx.item(), f"class_{idx.item()}")
        bar = "█" * int(val.item() * 30)
        print(f"    {i+1}. {breed:25s} {val.item():.1%}  {bar}")
    print(f"{'=' * 50}")

    # Show image with prediction
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(f"Predicted: {species} — {idx_to_breed.get(top5_idxs[0].item(), '?')}\n"
                 f"Confidence: {top5_vals[0].item():.1%}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    del pred_model
    torch.cuda.empty_cache()

# %% [markdown]
# ---
# ## §9 — Export & Download
#
# Export the trained model for deployment and download to your local machine.

# %%
# ============================================================
#  📦 EXPORT: PORTABLE BUNDLE + ONNX
# ============================================================
from src.train import create_portable_export
from src.export import create_portable_export as export_portable

export_dir = f"{PROJECT_DIR}/outputs/export/portable"
split_dir = f"{PROJECT_DIR}/data/splits"

# Portable export
print("Creating portable export...")
portable_dir = create_portable_export(
    best_checkpoint, BACKBONE, split_dir, export_dir)
print(f"✅ Portable bundle: {portable_dir}")

# ONNX export (for Android deployment)
print("\nCreating ONNX export...")
onnx_model = BreedClassifier(backbone=BACKBONE, attention=ATTENTION)
ckpt = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
onnx_model.load_state_dict(ckpt["state_dict"])
onnx_model.eval()

import torch.nn as nn
class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return out["binary"], out["cattle"], out["buffalo"]

wrapper = ExportWrapper(onnx_model).eval()
dummy = torch.randn(1, 3, 260, 260)
onnx_path = f"{PROJECT_DIR}/outputs/export/{BACKBONE}_fp32.onnx"
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

try:
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["input"],
        output_names=["binary", "cattle", "buffalo"],
        opset_version=13,
        dynamic_axes={
            "input": {0: "batch"},
            "binary": {0: "batch"},
            "cattle": {0: "batch"},
            "buffalo": {0: "batch"},
        },
        dynamo=False,
    )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✅ ONNX: {onnx_path} ({onnx_size:.1f} MB)")
except Exception as exc:
    print(f"⚠️  ONNX export failed: {exc}")

del onnx_model, wrapper
torch.cuda.empty_cache()

# %%
# ============================================================
#  📥 ZIP & DOWNLOAD RESULTS
# ============================================================
import shutil

results_zip = f"/content/{BACKBONE}_trained_model"

# Collect all outputs
export_collect = f"/content/export_package"
os.makedirs(export_collect, exist_ok=True)

# Copy portable bundle
if os.path.exists(portable_dir):
    shutil.copytree(portable_dir, f"{export_collect}/portable",
                    dirs_exist_ok=True)

# Copy ONNX
if os.path.exists(onnx_path):
    shutil.copy2(onnx_path, export_collect)

# Copy quantized model if exists
quantized_path = f"{ckpt_dir}/{BACKBONE}_quantized.pt"
if os.path.exists(quantized_path):
    shutil.copy2(quantized_path, export_collect)

# Copy metrics
metrics_src = f"{PROJECT_DIR}/outputs/metrics"
if os.path.exists(metrics_src):
    shutil.copytree(metrics_src, f"{export_collect}/metrics",
                    dirs_exist_ok=True)

# Create zip
shutil.make_archive(results_zip, "zip", export_collect)
zip_path = f"{results_zip}.zip"
zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"\n✅ Results packaged: {zip_path} ({zip_size:.1f} MB)")

# Download
from google.colab import files
files.download(zip_path)

# %%
# ============================================================
#  💾 SAVE TO GOOGLE DRIVE (Optional)
# ============================================================
# Uncomment to save results to Google Drive:

# from google.colab import drive
# drive.mount("/content/drive")
# DRIVE_OUTPUT = "/content/drive/MyDrive/ML-CB-B-identifier/trained_models"
# os.makedirs(DRIVE_OUTPUT, exist_ok=True)
# shutil.copy2(zip_path, DRIVE_OUTPUT)
# print(f"✅ Saved to Google Drive: {DRIVE_OUTPUT}")

# %% [markdown]
# ---
# ## 📊 GPU Memory Monitor (Optional)
#
# Run this cell at any time to check VRAM usage.

# %%
# ============================================================
#  GPU MEMORY MONITOR
# ============================================================
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Memory:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Peak:      {max_allocated:.2f} GB")
    print(f"  Total:     {total:.1f} GB")
    print(f"  Free:      {total - reserved:.2f} GB")
else:
    print("No GPU available")
