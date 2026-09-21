import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_SIZE = 260
NUM_CATTLE_BREEDS = 57
NUM_BUFFALO_BREEDS = 18
NUM_BREEDS_TOTAL = 75

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BACKBONE_LR_MULT = 0.1

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
PROJECTION_DIM = 128
DROPOUT = 0.3

RANDAUGMENT_OPS = 2
RANDAUGMENT_MAGNITUDE = 5
CUTMIX_ALPHA = 1.0
MIXUP_ALPHA = 0.3
CUTMIX_MIXUP_PROB = 0.5
# Breeds with fewer than this many images are never mixed (CutMix/MixUp);
# mixing 10-image breeds with other breeds destroys the little signal they
# carry and makes the rare-breed signature unrecoverable.
RARE_CLASS_THRESHOLD = 30

# --- Fine-grained feature learning (auxiliary supervised contrastive loss) ---
# The 1280-d pooled feature vector was previously unused. A projection head +
# SupCon loss explicitly pulls same-breed embeddings together, which is what
# separates visually near-identical indigenous breeds (Hariana/Sahiwal/etc.).
CONTRASTIVE_WEIGHT = 0.2
CONTRASTIVE_TEMPERATURE = 0.1

# --- Logit adjustment for class imbalance (Menon et al., ICLR 2021) ---
# tau * log(prior) is added to the breed logits during training only; the
# prior shift is absorbed into the learned biases so inference stays raw.
LOGIT_ADJUST = True
LOGIT_ADJUST_TAU = 1.0

# --- Knowledge distillation (teacher -> student, e.g. lite4 -> lite2) ---
KD_ALPHA = 0.7      # blend: (1-alpha)*hard CE + alpha*T^2*KL(teacher||student)
KD_TEMPERATURE = 4.0

# --- Class imbalance ---
SAMPLER_BETA = 0.99           # effective-number-of-samples sampler beta
BALANCE_BINARY_HEAD = True    # per-batch species re-weighting of binary CE

# --- Exponential moving average of weights (phase 2) ---
EMA_DECAY = 0.999

BATCH_SIZE = 64
NUM_WORKERS = 4

# 70/15/15 gives rare breeds at least one (usually two) val/test image, so
# macro metrics and checkpoint selection can actually see the long tail.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LOSS_WEIGHT_BINARY = 0.15
LOSS_WEIGHT_CATTLE = 0.50
LOSS_WEIGHT_BUFFALO = 0.35

# Once the binary head saturates (>= BINARY_SATURATION_ACC) it no longer needs
# 15% of the loss budget; the breath heads are the bottleneck. These weights
# kick in automatically from the next epoch onward.
LOSS_WEIGHT_BINARY_FINAL = 0.05
LOSS_WEIGHT_CATTLE_FINAL = 0.55
LOSS_WEIGHT_BUFFALO_FINAL = 0.40
BINARY_SATURATION_ACC = 0.95

PHASE1_EPOCHS = 8
PHASE1_LR = 3e-3

PHASE2_EPOCHS = 80
PHASE2_LR = 2e-4

PHASE3_EPOCHS = 10
PHASE3_LR = 5e-6

# Checkpoint selection metric: macro-averaged F1 over both breed heads.
# Combined top-1 is dominated by the ~10 large breeds; macro-F1 is the only
# metric that rewards progress on the 30+ rare indigenous breeds.
BEST_METRIC = "balanced_score"

# --- SOTA additions ---
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 2

TFLITE_APP_ASSETS_DIR = os.path.join(
    PROJECT_ROOT, "flutter_app", "assets", "models")

SEED = 42

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
EXPORT_DIR = os.path.join(OUTPUT_DIR, "export")
PORTABLE_EXPORT_DIR = os.path.join(OUTPUT_DIR, "export", "portable")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

SMOKE_SAMPLES_PER_BREED = 5
HALF_DATA_RATIO = 0.5    # Fraction of images per breed for --half-data mode
QUARTER_DATA_RATIO = 0.25  # Fraction of images per breed for --quarter-data mode

# --- Training speed optimizations ---
EVAL_EVERY_PHASE1 = 1   # Evaluate every epoch (short phase, keep all checks)
EVAL_EVERY_PHASE2 = 2   # Evaluate every 2 epochs in the long fine-tune phase
EVAL_EVERY_PHASE3 = 2   # Evaluate every 2 epochs in QAT
CACHE_IMAGES = False     # Cache decoded PIL images in RAM after first epoch

SPECIES_LABELS = {"cattle": 0, "buffalo": 1}

for _d in (SPLIT_DIR, CHECKPOINT_DIR, EXPORT_DIR, PORTABLE_EXPORT_DIR, METRICS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)