import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_SIZE = 260
NUM_CATTLE_BREEDS = 57
NUM_BUFFALO_BREEDS = 18
NUM_BREEDS_TOTAL = 75

CBAM_AFTER_STAGE = 3

BACKBONE_WEIGHTS = {
    "lite2": os.path.join(PROJECT_ROOT, "efficientnet_lite2.pth"),
    "lite4": os.path.join(PROJECT_ROOT, "efficientnet_lite4.pth"),
}

BACKBONE_CHANNELS = {
    "lite2": [16, 24, 48, 88, 120, 208, 352],
    "lite4": [24, 32, 56, 112, 160, 272, 448],
}

FEATURE_DIM = 1280
BINARY_DIM = 256
BREED_DIM = 512
DROPOUT = 0.4

RANDAUGMENT_OPS = 2
RANDAUGMENT_MAGNITUDE = 9
CUTMIX_ALPHA = 0.4
MIXUP_ALPHA = 0.2

BATCH_SIZE = 64
NUM_WORKERS = 4

TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05

LOSS_WEIGHT_BINARY = 0.50
LOSS_WEIGHT_CATTLE = 0.25
LOSS_WEIGHT_BUFFALO = 0.25

PHASE1_EPOCHS = 5
PHASE1_LR = 3e-3

PHASE2_EPOCHS = 40
PHASE2_LR = 2e-4

PHASE3_EPOCHS = 10
PHASE3_LR = 5e-6

# --- SOTA additions ---
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 2

SEED = 42

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
EXPORT_DIR = os.path.join(OUTPUT_DIR, "export")
PORTABLE_EXPORT_DIR = os.path.join(OUTPUT_DIR, "export", "portable")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

SMOKE_SAMPLES_PER_BREED = 5

# --- Training speed optimizations ---
EVAL_EVERY_PHASE1 = 1   # Evaluate every epoch (short phase, keep all checks)
EVAL_EVERY_PHASE2 = 5   # Evaluate every 5 epochs in the long fine-tune phase
EVAL_EVERY_PHASE3 = 2   # Evaluate every 2 epochs in QAT
CACHE_IMAGES = False     # Cache decoded PIL images in RAM after first epoch

SPECIES_LABELS = {"cattle": 0, "buffalo": 1}

for _d in (SPLIT_DIR, CHECKPOINT_DIR, EXPORT_DIR, PORTABLE_EXPORT_DIR, METRICS_DIR):
    os.makedirs(_d, exist_ok=True)