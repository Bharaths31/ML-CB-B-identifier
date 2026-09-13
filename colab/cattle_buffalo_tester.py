# %% [markdown]
# # 🐄 Cattle & Buffalo Breed Classifier — Model Tester
#
# **Test your trained ONNX model on Google Colab**
#
# This notebook lets you:
# 1. Upload your exported ONNX model (`.onnx`)
# 2. Upload class mapping JSONs (`cattle_classes.json`, `buffalo_classes.json`)
# 3. **Single Image Prediction** — upload any image and get species + breed + top-5
# 4. **Batch Evaluation** — upload a folder of labelled images and get accuracy, F1, confusion matrices
#
# ---
#
# ## Model Details
#
# | Property | Value |
# |---|---|
# | Architecture | EfficientNet-Lite{2,4} + CBAM/SE + 3-head classifier |
# | Input | 260×260 RGB |
# | Outputs | `binary` (2), `cattle` (57), `buffalo` (18) |
# | ONNX opset | 13 |
# | Inference | Binary head → species → breed head → top-K breed |

# %% [markdown]
# ## §0 — Install Dependencies

# %%
!pip install -q onnxruntime-gpu Pillow matplotlib seaborn numpy

# %% [markdown]
# ## §1 — Core Imports & Configuration

# %%
import io
import json
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

try:
    import onnxruntime as ort
    print(f"✅ ONNX Runtime {ort.__version__} loaded")
    providers = ort.get_available_providers()
    print(f"   Available providers: {providers}")
except ImportError:
    raise ImportError("onnxruntime not found. Run: pip install onnxruntime")

# --- Configuration ---
IMAGE_SIZE = 260
NUM_CATTLE_BREEDS = 57
NUM_BUFFALO_BREEDS = 18
SPECIES_LABELS = {0: "Cattle", 1: "Buffalo"}

print(f"📐 Input size: {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"🐄 Cattle breeds: {NUM_CATTLE_BREEDS}")
print(f"🐃 Buffalo breeds: {NUM_BUFFALO_BREEDS}")

# %% [markdown]
# ## §2 — Upload ONNX Model & Class Maps
#
# Upload the following files from your `outputs/export/` directory:
#
# | File | Source | Required? |
# |---|---|---|
# | `*_fp32.onnx` | `outputs/export/lite2_fp32.onnx` | ✅ Yes |
# | `cattle_classes.json` | `data/splits/cattle_classes.json` or portable bundle | ✅ Yes |
# | `buffalo_classes.json` | `data/splits/buffalo_classes.json` or portable bundle | ✅ Yes |
#
# > **Tip**: All three files are also in `outputs/export/portable/<backbone>_*/` if you have a portable bundle.

# %%
from google.colab import files

print("📁 Upload your ONNX model file (.onnx):")
uploaded_model = files.upload()

onnx_filename = None
for fname in uploaded_model:
    if fname.endswith(".onnx"):
        onnx_filename = fname
        size_mb = len(uploaded_model[fname]) / (1024 * 1024)
        print(f"✅ Model: {fname} ({size_mb:.1f} MB)")
        break

if onnx_filename is None:
    raise FileNotFoundError("❌ No .onnx file uploaded. Please re-run this cell.")

# %%
print("📁 Upload cattle_classes.json and buffalo_classes.json:")
print("   (You can also upload them one at a time — re-run this cell for the second file)")
uploaded_maps = files.upload()

cattle_classes = None
buffalo_classes = None

for fname, content in uploaded_maps.items():
    try:
        data = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"⚠️  Skipped {fname} (not valid JSON)")
        continue

    # Must be a dict of {string: int} to be a class map
    if not isinstance(data, dict) or not all(isinstance(v, int) for v in data.values()):
        print(f"⚠️  Skipped {fname} (not a class map — expected {{breed_name: index}})")
        continue

    # Detect by filename first, then fall back to entry count
    if "cattle" in fname.lower():
        cattle_classes = data
        print(f"✅ Loaded cattle classes from {fname} ({len(data)} breeds)")
    elif "buffalo" in fname.lower():
        buffalo_classes = data
        print(f"✅ Loaded buffalo classes from {fname} ({len(data)} breeds)")
    elif len(data) == 57:
        cattle_classes = data
        print(f"✅ Auto-detected {fname} as cattle classes ({len(data)} breeds)")
    elif len(data) == 18:
        buffalo_classes = data
        print(f"✅ Auto-detected {fname} as buffalo classes ({len(data)} breeds)")
    else:
        print(f"⚠️  Skipped {fname} ({len(data)} entries — expected 57 for cattle or 18 for buffalo)")

if cattle_classes is None or buffalo_classes is None:
    missing = []
    if cattle_classes is None:
        missing.append("cattle_classes.json (57 breeds)")
    if buffalo_classes is None:
        missing.append("buffalo_classes.json (18 breeds)")
    raise FileNotFoundError(
        f"❌ Still missing: {', '.join(missing)}.\n"
        f"   Re-run this cell to upload the missing file(s)."
    )

# Build reverse maps: index → breed name
cattle_idx_to_breed = {v: k for k, v in cattle_classes.items()}
buffalo_idx_to_breed = {v: k for k, v in buffalo_classes.items()}

print(f"\n🐄 Cattle breeds: {sorted(cattle_classes.keys())[:5]}... ({len(cattle_classes)} total)")
print(f"🐃 Buffalo breeds: {sorted(buffalo_classes.keys())[:5]}... ({len(buffalo_classes)} total)")

# %% [markdown]
# ## §3 — Load ONNX Model

# %%
# Choose provider: CUDA if available, else CPU
providers_list = []
if "CUDAExecutionProvider" in ort.get_available_providers():
    providers_list.append("CUDAExecutionProvider")
    print("🟢 Using GPU (CUDA) for inference")
else:
    print("🟡 Using CPU for inference")
providers_list.append("CPUExecutionProvider")

# Create ONNX Runtime session
session = ort.InferenceSession(onnx_filename, providers=providers_list)

# Inspect model I/O
print(f"\n📋 Model inputs:")
for inp in session.get_inputs():
    print(f"   {inp.name}: shape={inp.shape}, dtype={inp.type}")

print(f"\n📋 Model outputs:")
for out in session.get_outputs():
    print(f"   {out.name}: shape={out.shape}, dtype={out.type}")

input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
print(f"\n✅ Model loaded successfully")
print(f"   Input: '{input_name}'")
print(f"   Outputs: {output_names}")

# %% [markdown]
# ## §4 — Preprocessing & Inference Utilities
#
# These match the exact preprocessing used during training evaluation:
# ```
# Image → Resize(260) → CenterCrop(260) → ToTensor() → [3, 260, 260]
# ```

# %%
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess a PIL image for ONNX inference.

    Matches the evaluation transform used during training:
    Resize(260) → CenterCrop(260) → ToTensor()

    Returns:
        np.ndarray of shape [1, 3, 260, 260], float32, range [0, 1]
    """
    img = image.convert("RGB")

    # Resize: shortest side → IMAGE_SIZE, maintain aspect ratio
    w, h = img.size
    if w < h:
        new_w = IMAGE_SIZE
        new_h = int(h * IMAGE_SIZE / w)
    else:
        new_h = IMAGE_SIZE
        new_w = int(w * IMAGE_SIZE / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to IMAGE_SIZE × IMAGE_SIZE
    w, h = img.size
    left = (w - IMAGE_SIZE) // 2
    top = (h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    # Convert to float32 array [C, H, W] in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, C]
    arr = arr.transpose(2, 0, 1)  # [C, H, W]
    arr = arr[np.newaxis, ...]  # [1, C, H, W]

    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def predict_single(image: Image.Image, top_k: int = 5) -> dict:
    """Run full inference pipeline on a single PIL image.

    Returns:
        dict with species, breed, confidences, top-K predictions, and latency
    """
    tensor = preprocess_image(image)

    t0 = time.perf_counter()
    outputs = session.run(output_names, {input_name: tensor})
    latency_ms = (time.perf_counter() - t0) * 1000

    # Parse outputs: binary(2), cattle(57), buffalo(18)
    binary_logits = outputs[0][0]   # shape [2]
    cattle_logits = outputs[1][0]   # shape [57]
    buffalo_logits = outputs[2][0]  # shape [18]

    # Species prediction
    binary_probs = softmax(binary_logits)
    species_idx = int(np.argmax(binary_probs))
    species_name = SPECIES_LABELS[species_idx]
    species_conf = float(binary_probs[species_idx]) * 100

    # Breed prediction based on species
    if species_idx == 0:  # Cattle
        breed_probs = softmax(cattle_logits)
        idx_to_breed = cattle_idx_to_breed
    else:  # Buffalo
        breed_probs = softmax(buffalo_logits)
        idx_to_breed = buffalo_idx_to_breed

    # Top-K breed predictions
    top_k_indices = np.argsort(breed_probs)[::-1][:top_k]
    top_k_results = []
    for idx in top_k_indices:
        breed = idx_to_breed.get(int(idx), f"class_{idx}")
        display_name = breed.replace("_", " ").title()
        confidence = float(breed_probs[idx]) * 100
        top_k_results.append({"breed": display_name, "confidence": confidence})

    return {
        "species": species_name,
        "species_confidence": round(species_conf, 2),
        "top_breed": top_k_results[0]["breed"] if top_k_results else "Unknown",
        "top_breed_confidence": round(top_k_results[0]["confidence"], 2) if top_k_results else 0,
        "top_k_breeds": top_k_results,
        "latency_ms": round(latency_ms, 1),
        "binary_probs": {SPECIES_LABELS[i]: round(float(binary_probs[i]) * 100, 2) for i in range(2)},
    }


print("✅ Preprocessing & inference utilities loaded")

# %% [markdown]
# ## §5 — Single Image Prediction
#
# Upload any cattle or buffalo image and get:
# - Species (Cattle / Buffalo) with confidence
# - Top breed prediction with confidence
# - Top-5 breed bar chart

# %%
print("📸 Upload an image to classify:")
uploaded_img = files.upload()

for fname, content in uploaded_img.items():
    img = Image.open(io.BytesIO(content))
    result = predict_single(img)

    print(f"\n{'='*60}")
    print(f"  📁 File: {fname}")
    print(f"  📐 Size: {img.size[0]}×{img.size[1]}")
    print(f"{'='*60}")
    print(f"\n  🐄 Species: {result['species']} ({result['species_confidence']:.1f}%)")
    print(f"     Cattle prob:  {result['binary_probs']['Cattle']:.1f}%")
    print(f"     Buffalo prob: {result['binary_probs']['Buffalo']:.1f}%")
    print(f"\n  🏆 Top Breed: {result['top_breed']} ({result['top_breed_confidence']:.1f}%)")
    print(f"  ⏱️  Latency: {result['latency_ms']:.1f} ms")
    print(f"\n  📊 Top-5 Predictions:")
    for i, pred in enumerate(result["top_k_breeds"], 1):
        bar = "█" * int(pred["confidence"] / 2)
        print(f"     {i}. {pred['breed']:25s} {pred['confidence']:6.2f}% {bar}")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Image with prediction overlay
    axes[0].imshow(img)
    axes[0].set_title(
        f"{result['species']}: {result['top_breed']}\n"
        f"({result['top_breed_confidence']:.1f}% confidence)",
        fontsize=12, fontweight="bold", pad=10
    )
    axes[0].axis("off")

    # Right: Top-5 horizontal bar chart
    breeds = [p["breed"] for p in result["top_k_breeds"]][::-1]
    confs = [p["confidence"] for p in result["top_k_breeds"]][::-1]
    colors = ["#6366f1" if c < max(confs) else "#22c55e" for c in confs]
    axes[1].barh(breeds, confs, color=colors, height=0.6, edgecolor="none")
    axes[1].set_xlabel("Confidence (%)", fontsize=10)
    axes[1].set_title("Top-5 Breed Predictions", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlim(0, 100)
    for i, (breed, conf) in enumerate(zip(breeds, confs)):
        axes[1].text(conf + 1, i, f"{conf:.1f}%", va="center", fontsize=9,
                     fontweight="bold", color="#374151")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    print(f"\n{'='*60}\n")

# %% [markdown]
# ## §6 — Batch Evaluation (Automated)
#
# Upload a **zip file** containing test images organized as:
# ```
# test_images.zip
# └── cattle/
# │   ├── sahiwal/
# │   │   ├── img1.jpg
# │   │   └── img2.jpg
# │   └── gir/
# │       └── img1.jpg
# └── buffalo/
#     ├── murrah/
#     │   └── img1.jpg
#     └── banni/
#         └── img1.jpg
# ```
#
# The folder structure must be `{species}/{breed}/*.jpg` — matching the training data layout.

# %%
import zipfile
from collections import Counter

print("📁 Upload a zip file with test images (species/breed/image.jpg structure):")
uploaded_zip = files.upload()

zip_filename = None
for fname in uploaded_zip:
    if fname.endswith(".zip"):
        zip_filename = fname
        print(f"✅ Received: {fname}")
        break

if zip_filename is None:
    raise FileNotFoundError("❌ No .zip file uploaded. Please upload a zip with test images.")

# Extract
extract_dir = "/content/test_data"
if os.path.exists(extract_dir):
    import shutil
    shutil.rmtree(extract_dir)

with zipfile.ZipFile(zip_filename, "r") as zf:
    zf.extractall(extract_dir)
    print(f"📦 Extracted {len(zf.namelist())} entries to {extract_dir}")

# Auto-detect root: handle nested directories (e.g., zip contains a single root folder)
contents = os.listdir(extract_dir)
if len(contents) == 1 and os.path.isdir(os.path.join(extract_dir, contents[0])):
    extract_dir = os.path.join(extract_dir, contents[0])
    print(f"   → Auto-detected root: {extract_dir}")

# Discover images
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
test_images = []  # list of (species, breed, filepath)

for species in ("cattle", "buffalo"):
    species_dir = os.path.join(extract_dir, species)
    if not os.path.isdir(species_dir):
        print(f"⚠️  No '{species}/' directory found — skipping")
        continue
    for breed in sorted(os.listdir(species_dir)):
        breed_dir = os.path.join(species_dir, breed)
        if not os.path.isdir(breed_dir):
            continue
        for fname in sorted(os.listdir(breed_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTS:
                test_images.append((species, breed, os.path.join(breed_dir, fname)))

print(f"\n📊 Found {len(test_images)} test images")
species_counts = Counter(s for s, _, _ in test_images)
for sp, cnt in species_counts.items():
    breed_cnt = len(set(b for s, b, _ in test_images if s == sp))
    print(f"   {sp.title()}: {cnt} images across {breed_cnt} breeds")

# %% [markdown]
# ### §6.1 — Run Batch Inference

# %%
from tqdm.notebook import tqdm as tqdm_notebook

print(f"\n🔄 Running inference on {len(test_images)} images...\n")

results = []
t_start = time.perf_counter()

for species_true, breed_true, filepath in tqdm_notebook(test_images, desc="Evaluating"):
    try:
        img = Image.open(filepath)
        pred = predict_single(img, top_k=5)
        results.append({
            "filepath": filepath,
            "species_true": species_true,
            "breed_true": breed_true,
            "species_pred": pred["species"].lower(),
            "breed_pred": pred["top_breed"].lower().replace(" ", "_"),
            "breed_conf": pred["top_breed_confidence"],
            "species_conf": pred["species_confidence"],
            "top5_breeds": [p["breed"].lower().replace(" ", "_") for p in pred["top_k_breeds"]],
            "latency_ms": pred["latency_ms"],
        })
    except Exception as e:
        print(f"⚠️  Skipped {filepath}: {e}")

total_time = time.perf_counter() - t_start
print(f"\n✅ Evaluated {len(results)} images in {total_time:.1f}s")
print(f"   Avg latency: {np.mean([r['latency_ms'] for r in results]):.1f} ms/image")

# %% [markdown]
# ### §6.2 — Compute Metrics

# %%
# --- Binary (Species) Accuracy ---
species_correct = sum(1 for r in results if r["species_true"] == r["species_pred"])
species_acc = species_correct / len(results) * 100 if results else 0

# --- Per-species breed accuracy ---
cattle_results = [r for r in results if r["species_true"] == "cattle"]
buffalo_results = [r for r in results if r["species_true"] == "buffalo"]

cattle_breed_correct = sum(1 for r in cattle_results if r["breed_true"] == r["breed_pred"])
buffalo_breed_correct = sum(1 for r in buffalo_results if r["breed_true"] == r["breed_pred"])
cattle_breed_acc = cattle_breed_correct / len(cattle_results) * 100 if cattle_results else 0
buffalo_breed_acc = buffalo_breed_correct / len(buffalo_results) * 100 if buffalo_results else 0

# --- Combined accuracy (species + breed both correct) ---
combined_correct = sum(
    1 for r in results
    if r["species_true"] == r["species_pred"] and r["breed_true"] == r["breed_pred"]
)
combined_acc = combined_correct / len(results) * 100 if results else 0

# --- Top-3 and Top-5 accuracy ---
top3_correct = sum(1 for r in results
    if r["species_true"] == r["species_pred"] and r["breed_true"] in r["top5_breeds"][:3]
)
top5_correct = sum(1 for r in results
    if r["species_true"] == r["species_pred"] and r["breed_true"] in r["top5_breeds"]
)
top3_acc = top3_correct / len(results) * 100 if results else 0
top5_acc = top5_correct / len(results) * 100 if results else 0

# --- F1 Score (binary: buffalo as positive class) ---
tp = sum(1 for r in results if r["species_true"] == "buffalo" and r["species_pred"] == "buffalo")
fp = sum(1 for r in results if r["species_true"] == "cattle" and r["species_pred"] == "buffalo")
fn = sum(1 for r in results if r["species_true"] == "buffalo" and r["species_pred"] == "cattle")
binary_f1 = (2 * tp / (2 * tp + fp + fn) * 100) if (2 * tp + fp + fn) > 0 else 0

# --- Per-breed F1 (macro average) ---
def compute_macro_f1(result_list, class_map):
    """Compute macro-averaged F1 across all breeds."""
    breed_names = set(class_map.keys())
    f1_scores = []
    for breed in breed_names:
        tp_b = sum(1 for r in result_list if r["breed_true"] == breed and r["breed_pred"] == breed)
        fp_b = sum(1 for r in result_list if r["breed_true"] != breed and r["breed_pred"] == breed)
        fn_b = sum(1 for r in result_list if r["breed_true"] == breed and r["breed_pred"] != breed)
        if (2 * tp_b + fp_b + fn_b) > 0:
            f1_scores.append(2 * tp_b / (2 * tp_b + fp_b + fn_b))
    return np.mean(f1_scores) * 100 if f1_scores else 0

cattle_macro_f1 = compute_macro_f1(cattle_results, cattle_classes)
buffalo_macro_f1 = compute_macro_f1(buffalo_results, buffalo_classes)

# --- Print results ---
print(f"""
{'='*60}
  📊 EVALUATION RESULTS
{'='*60}

  Binary (Species) Classification
  ─────────────────────────────────
    Accuracy:     {species_acc:6.2f}%  ({species_correct}/{len(results)})
    F1 Score:     {binary_f1:6.2f}%

  Breed Classification
  ─────────────────────────────────
    Cattle Acc:   {cattle_breed_acc:6.2f}%  ({cattle_breed_correct}/{len(cattle_results)})
    Buffalo Acc:  {buffalo_breed_acc:6.2f}%  ({buffalo_breed_correct}/{len(buffalo_results)})
    Cattle F1:    {cattle_macro_f1:6.2f}%  (macro-average)
    Buffalo F1:   {buffalo_macro_f1:6.2f}%  (macro-average)

  Combined (Species + Breed)
  ─────────────────────────────────
    Top-1 Acc:    {combined_acc:6.2f}%  ({combined_correct}/{len(results)})
    Top-3 Acc:    {top3_acc:6.2f}%  ({top3_correct}/{len(results)})
    Top-5 Acc:    {top5_acc:6.2f}%  ({top5_correct}/{len(results)})

  Performance
  ─────────────────────────────────
    Total images: {len(results)}
    Total time:   {total_time:.1f}s
    Avg latency:  {np.mean([r['latency_ms'] for r in results]):.1f} ms/img
    Throughput:   {len(results) / total_time:.1f} img/s

{'='*60}
""")

# %% [markdown]
# ### §6.3 — Confusion Matrices

# %%
def plot_confusion_matrix(results_list, class_map, species_title):
    """Plot a confusion matrix heatmap for a single species."""
    # Get all breeds that appear in results OR class map
    all_breeds = sorted(set(
        list(class_map.keys()) +
        [r["breed_true"] for r in results_list] +
        [r["breed_pred"] for r in results_list]
    ))

    # Filter to only breeds that appear in results (for readability)
    active_breeds = sorted(set(
        [r["breed_true"] for r in results_list] +
        [r["breed_pred"] for r in results_list]
    ))

    n = len(active_breeds)
    if n == 0:
        print(f"⚠️  No {species_title} results to plot")
        return

    breed_to_idx = {b: i for i, b in enumerate(active_breeds)}
    cm = np.zeros((n, n), dtype=int)

    for r in results_list:
        true_idx = breed_to_idx.get(r["breed_true"])
        pred_idx = breed_to_idx.get(r["breed_pred"])
        if true_idx is not None and pred_idx is not None:
            cm[true_idx, pred_idx] += 1

    # Truncate labels for display
    display_labels = [b.replace("_", " ").title()[:15] for b in active_breeds]

    fig_size = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(cm, annot=(n <= 25), fmt="d", cmap="Blues",
                xticklabels=display_labels, yticklabels=display_labels,
                ax=ax, cbar_kws={"shrink": 0.8})

    ax.set_xlabel("Predicted Breed", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Breed", fontsize=11, fontweight="bold")
    ax.set_title(f"{species_title} Confusion Matrix\n({len(results_list)} images, {n} breeds)",
                 fontsize=13, fontweight="bold", pad=15)

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"/content/{species_title.lower()}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Saved: /content/{species_title.lower()}_confusion_matrix.png")


# Plot cattle confusion matrix
if cattle_results:
    plot_confusion_matrix(cattle_results, cattle_classes, "Cattle")

# Plot buffalo confusion matrix
if buffalo_results:
    plot_confusion_matrix(buffalo_results, buffalo_classes, "Buffalo")

# %% [markdown]
# ### §6.4 — Per-Breed Accuracy Breakdown

# %%
def per_breed_report(results_list, species_name):
    """Print and plot per-breed accuracy."""
    breed_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results_list:
        breed_stats[r["breed_true"]]["total"] += 1
        if r["breed_true"] == r["breed_pred"]:
            breed_stats[r["breed_true"]]["correct"] += 1

    breeds = sorted(breed_stats.keys())
    if not breeds:
        print(f"⚠️  No {species_name} results")
        return

    print(f"\n📋 Per-Breed Accuracy — {species_name}")
    print(f"{'─'*55}")
    print(f"  {'Breed':<25s} {'Correct':>8s} {'Total':>6s} {'Acc':>7s}")
    print(f"{'─'*55}")

    accs = []
    for breed in breeds:
        s = breed_stats[breed]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        accs.append(acc)
        display = breed.replace("_", " ").title()
        marker = "✅" if acc >= 80 else ("⚠️" if acc >= 50 else "❌")
        print(f"  {marker} {display:<23s} {s['correct']:>6d} / {s['total']:<4d}  {acc:6.1f}%")

    avg_acc = np.mean(accs)
    print(f"{'─'*55}")
    print(f"  {'Average':<25s} {'':>8s} {'':>6s} {avg_acc:6.1f}%")
    print()

    # Bar chart
    display_breeds = [b.replace("_", " ").title()[:18] for b in breeds]
    colors = ["#22c55e" if a >= 80 else ("#f59e0b" if a >= 50 else "#ef4444") for a in accs]

    fig, ax = plt.subplots(figsize=(max(10, len(breeds) * 0.4), 6))
    bars = ax.bar(range(len(breeds)), accs, color=colors, edgecolor="none", width=0.7)
    ax.set_xticks(range(len(breeds)))
    ax.set_xticklabels(display_breeds, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"{species_name} — Per-Breed Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(y=avg_acc, color="#6366f1", linestyle="--", linewidth=1.5,
               label=f"Average: {avg_acc:.1f}%")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"/content/{species_name.lower()}_per_breed_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Saved: /content/{species_name.lower()}_per_breed_accuracy.png")


per_breed_report(cattle_results, "Cattle")
per_breed_report(buffalo_results, "Buffalo")

# %% [markdown]
# ### §6.5 — Misclassified Images Gallery

# %%
def show_misclassified(results_list, species_name, max_show=16):
    """Display a gallery of misclassified images."""
    misclassified = [
        r for r in results_list
        if r["breed_true"] != r["breed_pred"]
    ]

    if not misclassified:
        print(f"🎉 No misclassifications for {species_name}!")
        return

    print(f"\n❌ {len(misclassified)} misclassified {species_name} images "
          f"(showing up to {max_show}):\n")

    show = misclassified[:max_show]
    cols = min(4, len(show))
    rows = (len(show) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, r in enumerate(show):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        try:
            img = Image.open(r["filepath"])
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Load Error", ha="center", va="center", transform=ax.transAxes)
        true_display = r["breed_true"].replace("_", " ").title()
        pred_display = r["breed_pred"].replace("_", " ").title()
        ax.set_title(f"True: {true_display}\nPred: {pred_display}\n({r['breed_conf']:.1f}%)",
                     fontsize=8, color="red", fontweight="bold")
        ax.axis("off")

    # Hide empty subplots
    for idx in range(len(show), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    plt.suptitle(f"{species_name} — Misclassified Images", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"/content/{species_name.lower()}_misclassified.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"💾 Saved: /content/{species_name.lower()}_misclassified.png")


show_misclassified(cattle_results, "Cattle")
show_misclassified(buffalo_results, "Buffalo")

# %% [markdown]
# ### §6.6 — Export Evaluation Report

# %%
report = {
    "model": onnx_filename,
    "total_images": len(results),
    "binary_accuracy": round(species_acc, 4),
    "binary_f1": round(binary_f1, 4),
    "cattle_breed_accuracy": round(cattle_breed_acc, 4),
    "buffalo_breed_accuracy": round(buffalo_breed_acc, 4),
    "cattle_macro_f1": round(cattle_macro_f1, 4),
    "buffalo_macro_f1": round(buffalo_macro_f1, 4),
    "combined_top1": round(combined_acc, 4),
    "combined_top3": round(top3_acc, 4),
    "combined_top5": round(top5_acc, 4),
    "avg_latency_ms": round(float(np.mean([r['latency_ms'] for r in results])), 1),
    "throughput_imgs_per_sec": round(len(results) / total_time, 1),
    "cattle_images": len(cattle_results),
    "buffalo_images": len(buffalo_results),
}

report_path = "/content/evaluation_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"📄 Evaluation report saved to: {report_path}")
print(json.dumps(report, indent=2))

# Download report and confusion matrices
print("\n📥 Downloading evaluation artifacts...")
for fpath in [
    report_path,
    "/content/cattle_confusion_matrix.png",
    "/content/buffalo_confusion_matrix.png",
    "/content/cattle_per_breed_accuracy.png",
    "/content/buffalo_per_breed_accuracy.png",
    "/content/cattle_misclassified.png",
    "/content/buffalo_misclassified.png",
]:
    if os.path.exists(fpath):
        try:
            files.download(fpath)
            print(f"   ✅ {os.path.basename(fpath)}")
        except Exception:
            print(f"   ⚠️  Auto-download failed for {os.path.basename(fpath)} (download manually from Files panel)")

# %% [markdown]
# ## §7 — Interactive Testing (Multiple Images)
#
# Upload multiple images at once for quick testing without the batch evaluation overhead.

# %%
print("📸 Upload one or more images for quick classification:")
uploaded_multi = files.upload()

all_results = []
for fname, content in uploaded_multi.items():
    try:
        img = Image.open(io.BytesIO(content))
        result = predict_single(img, top_k=3)
        all_results.append((fname, img, result))
    except Exception as e:
        print(f"⚠️  Skipped {fname}: {e}")

# Display results grid
if all_results:
    cols = min(3, len(all_results))
    rows = (len(all_results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (fname, img, result) in enumerate(all_results):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        ax.imshow(img)
        species_color = "#f59e0b" if result["species"] == "Cattle" else "#22c55e"
        ax.set_title(
            f"{result['species']}: {result['top_breed']}\n"
            f"({result['top_breed_confidence']:.1f}% | {result['latency_ms']:.0f}ms)",
            fontsize=10, fontweight="bold", color=species_color
        )
        ax.axis("off")

    for idx in range(len(all_results), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    plt.suptitle("Classification Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Print summary table
    print(f"\n{'File':<30s} {'Species':<10s} {'Breed':<25s} {'Conf':>6s} {'Time':>6s}")
    print("─" * 80)
    for fname, _, result in all_results:
        print(f"{fname[:29]:<30s} {result['species']:<10s} "
              f"{result['top_breed'][:24]:<25s} "
              f"{result['top_breed_confidence']:5.1f}% "
              f"{result['latency_ms']:5.1f}ms")

# %% [markdown]
# ## §8 — Export Complete Evaluation Report (HTML Document)
#
# Generates a comprehensive, self-contained HTML report combining results from:
# - §5 Single Image Prediction
# - §6 Batch Evaluation (metrics, confusion matrices, per-breed breakdown)
# - §7 Multi-Image Interactive Testing

# %%
import base64
from datetime import datetime

def img_to_base64(path):
    """Convert an image file to base64 data URI for embedding in HTML."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


# ── Collect all saved chart PNGs ──
chart_files = {
    "cattle_cm": "/content/cattle_confusion_matrix.png",
    "buffalo_cm": "/content/buffalo_confusion_matrix.png",
    "cattle_breed_acc": "/content/cattle_per_breed_accuracy.png",
    "buffalo_breed_acc": "/content/buffalo_per_breed_accuracy.png",
    "cattle_misclassified": "/content/cattle_misclassified.png",
    "buffalo_misclassified": "/content/buffalo_misclassified.png",
}
chart_b64 = {k: img_to_base64(v) for k, v in chart_files.items()}

# ── Build per-breed tables ──
def breed_table_html(results_list, species_name):
    breed_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results_list:
        breed_stats[r["breed_true"]]["total"] += 1
        if r["breed_true"] == r["breed_pred"]:
            breed_stats[r["breed_true"]]["correct"] += 1
    rows_html = ""
    for breed in sorted(breed_stats.keys()):
        s = breed_stats[breed]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        color = "#22c55e" if acc >= 80 else ("#f59e0b" if acc >= 50 else "#ef4444")
        display = breed.replace("_", " ").title()
        rows_html += f"""<tr>
            <td>{display}</td>
            <td style="text-align:center">{s['correct']}</td>
            <td style="text-align:center">{s['total']}</td>
            <td style="text-align:center;color:{color};font-weight:700">{acc:.1f}%</td>
        </tr>\n"""
    return rows_html

cattle_breed_rows = breed_table_html(cattle_results, "Cattle") if cattle_results else ""
buffalo_breed_rows = breed_table_html(buffalo_results, "Buffalo") if buffalo_results else ""

# ── Build single-image results (from §5 variable if available) ──
single_img_html = ""
try:
    if 'uploaded_img' in dir() and uploaded_img:
        for fname, content in uploaded_img.items():
            img = Image.open(io.BytesIO(content))
            res = predict_single(img)
            # Embed thumbnail
            buf = io.BytesIO()
            img_thumb = img.copy()
            img_thumb.thumbnail((300, 300))
            img_thumb.save(buf, format="JPEG", quality=85)
            thumb_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            top5_rows = ""
            for i, p in enumerate(res["top_k_breeds"], 1):
                bar_w = max(4, p["confidence"])
                top5_rows += f"""<tr>
                    <td>{i}</td><td>{p['breed']}</td>
                    <td><div style="background:linear-gradient(90deg,#6366f1 {bar_w}%,#1e293b {bar_w}%);
                         height:18px;border-radius:4px;min-width:40px"></div></td>
                    <td style="text-align:right;font-weight:600">{p['confidence']:.2f}%</td>
                </tr>\n"""

            single_img_html += f"""
            <div class="result-card">
                <h3>📸 {fname}</h3>
                <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start">
                    <img src="data:image/jpeg;base64,{thumb_b64}" style="width:250px;border-radius:8px;border:1px solid #2d3a4f">
                    <div style="flex:1;min-width:280px">
                        <div class="metric-badge" style="background:{'#f59e0b22;border-color:#f59e0b' if res['species']=='Cattle' else '#22c55e22;border-color:#22c55e'}">
                            {res['species']} — {res['species_confidence']:.1f}%
                        </div>
                        <p style="font-size:1.4em;font-weight:800;margin:8px 0">{res['top_breed']}</p>
                        <p style="color:#94a3b8">Confidence: {res['top_breed_confidence']:.1f}% · Latency: {res['latency_ms']:.1f}ms</p>
                        <table class="data-table" style="margin-top:12px">
                            <tr><th>#</th><th>Breed</th><th>Bar</th><th>Conf</th></tr>
                            {top5_rows}
                        </table>
                    </div>
                </div>
            </div>"""
except Exception:
    pass

if not single_img_html:
    single_img_html = '<p style="color:#94a3b8">No single-image results captured. Run §5 before this cell.</p>'

# ── Build multi-image results (from §7 variable if available) ──
multi_img_html = ""
try:
    if 'all_results' in dir() and all_results:
        multi_img_html += '<div style="display:flex;flex-wrap:wrap;gap:16px">'
        for fname, img, res in all_results:
            buf = io.BytesIO()
            img_thumb = img.copy()
            img_thumb.thumbnail((200, 200))
            img_thumb.save(buf, format="JPEG", quality=85)
            thumb_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            sp_color = "#f59e0b" if res["species"] == "Cattle" else "#22c55e"
            multi_img_html += f"""
            <div style="background:#111827;border:1px solid #2d3a4f;border-radius:10px;padding:12px;width:200px">
                <img src="data:image/jpeg;base64,{thumb_b64}" style="width:100%;border-radius:6px">
                <p style="margin:8px 0 2px;font-weight:700;font-size:0.85em;color:{sp_color}">{res['species']}: {res['top_breed']}</p>
                <p style="font-size:0.75em;color:#94a3b8">{res['top_breed_confidence']:.1f}% · {res['latency_ms']:.0f}ms</p>
                <p style="font-size:0.7em;color:#64748b;word-break:break-all">{fname}</p>
            </div>"""
        multi_img_html += '</div>'
except Exception:
    pass

if not multi_img_html:
    multi_img_html = '<p style="color:#94a3b8">No multi-image results captured. Run §7 before this cell.</p>'

# ── Embed chart images ──
def chart_img_tag(key, alt, width="100%"):
    b64 = chart_b64.get(key)
    if b64:
        return f'<img src="{b64}" alt="{alt}" style="width:{width};border-radius:8px;margin:8px 0">'
    return f'<p style="color:#94a3b8">⚠️ {alt} not generated — run §6 first.</p>'

# ── Assemble full HTML report ──
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Breed Classifier — Evaluation Report</title>
<style>
  :root {{ --bg:#0a0e17; --surface:#111827; --surface2:#1e293b; --border:#2d3a4f;
           --text:#e2e8f0; --muted:#94a3b8; --accent:#6366f1; --green:#22c55e; --amber:#f59e0b; --red:#ef4444; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text);
          line-height:1.6; padding:40px 20px; }}
  .container {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:2em; font-weight:800; text-align:center; margin-bottom:4px;
       background:linear-gradient(135deg,#818cf8,#22c55e); -webkit-background-clip:text;
       -webkit-text-fill-color:transparent; background-clip:text; }}
  h2 {{ font-size:1.3em; font-weight:700; margin:32px 0 16px; padding-bottom:8px; border-bottom:1px solid var(--border); }}
  h3 {{ font-size:1.05em; font-weight:600; margin:16px 0 8px; }}
  .subtitle {{ text-align:center; color:var(--muted); font-size:0.9em; margin-bottom:32px; }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; margin:16px 0; }}
  .result-card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; margin:12px 0; }}
  .metrics-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:16px 0; }}
  .metric-box {{ background:var(--surface2); border-radius:10px; padding:16px; text-align:center; border:1px solid var(--border); }}
  .metric-box .value {{ font-size:1.8em; font-weight:800; }}
  .metric-box .label {{ font-size:0.75em; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; margin-top:4px; }}
  .metric-badge {{ display:inline-block; padding:4px 14px; border-radius:16px; font-size:0.85em; font-weight:600;
                   border:1px solid; margin:4px 0; }}
  .data-table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
  .data-table th {{ background:var(--surface2); padding:10px 12px; text-align:left; font-weight:600;
                    font-size:0.8em; text-transform:uppercase; letter-spacing:0.04em; color:var(--muted); border-bottom:1px solid var(--border); }}
  .data-table td {{ padding:8px 12px; border-bottom:1px solid var(--border); }}
  .data-table tr:hover {{ background:var(--surface2); }}
  .chart-section {{ margin:20px 0; }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:768px) {{ .two-col {{ grid-template-columns:1fr; }} }}
  .footer {{ text-align:center; color:var(--muted); font-size:0.78em; margin-top:40px; padding-top:20px; border-top:1px solid var(--border); }}
  .green {{ color:var(--green); }} .amber {{ color:var(--amber); }} .red {{ color:var(--red); }} .accent {{ color:var(--accent); }}
</style>
</head>
<body>
<div class="container">

<h1>🐄 Cattle &amp; Buffalo Breed Classifier</h1>
<p class="subtitle">Evaluation Report · Generated {timestamp}</p>

<!-- MODEL INFO -->
<div class="card">
  <h2>📋 Model Information</h2>
  <table class="data-table">
    <tr><td style="font-weight:600;width:200px">ONNX Model</td><td>{onnx_filename}</td></tr>
    <tr><td style="font-weight:600">Architecture</td><td>EfficientNet-Lite + CBAM/SE + 3-Head Classifier</td></tr>
    <tr><td style="font-weight:600">Input Size</td><td>{IMAGE_SIZE}×{IMAGE_SIZE} RGB</td></tr>
    <tr><td style="font-weight:600">Output Heads</td><td>Binary (2) · Cattle ({NUM_CATTLE_BREEDS}) · Buffalo ({NUM_BUFFALO_BREEDS})</td></tr>
    <tr><td style="font-weight:600">Inference Runtime</td><td>ONNX Runtime — {', '.join(providers_list)}</td></tr>
    <tr><td style="font-weight:600">Report Date</td><td>{timestamp}</td></tr>
  </table>
</div>

<!-- §5 SINGLE IMAGE -->
<div class="card">
  <h2>📸 §5 — Single Image Prediction</h2>
  {single_img_html}
</div>

<!-- §6 BATCH EVAL SUMMARY -->
<div class="card">
  <h2>📊 §6 — Batch Evaluation Summary</h2>

  <h3>Overall Metrics</h3>
  <div class="metrics-grid">
    <div class="metric-box"><div class="value green">{species_acc:.1f}%</div><div class="label">Species Accuracy</div></div>
    <div class="metric-box"><div class="value green">{binary_f1:.1f}%</div><div class="label">Binary F1</div></div>
    <div class="metric-box"><div class="value accent">{combined_acc:.1f}%</div><div class="label">Combined Top-1</div></div>
    <div class="metric-box"><div class="value accent">{top3_acc:.1f}%</div><div class="label">Combined Top-3</div></div>
    <div class="metric-box"><div class="value accent">{top5_acc:.1f}%</div><div class="label">Combined Top-5</div></div>
    <div class="metric-box"><div class="value" style="color:var(--muted)">{len(results)}</div><div class="label">Total Images</div></div>
  </div>

  <h3>Detailed Metrics</h3>
  <table class="data-table">
    <tr><th>Metric</th><th>Value</th><th>Detail</th></tr>
    <tr><td>Binary (Species) Accuracy</td><td class="green" style="font-weight:700">{species_acc:.2f}%</td><td>{species_correct} / {len(results)}</td></tr>
    <tr><td>Binary F1 (Buffalo +ve)</td><td class="green" style="font-weight:700">{binary_f1:.2f}%</td><td>TP={tp} FP={fp} FN={fn}</td></tr>
    <tr><td>Cattle Breed Accuracy</td><td style="font-weight:700">{cattle_breed_acc:.2f}%</td><td>{cattle_breed_correct} / {len(cattle_results)}</td></tr>
    <tr><td>Buffalo Breed Accuracy</td><td style="font-weight:700">{buffalo_breed_acc:.2f}%</td><td>{buffalo_breed_correct} / {len(buffalo_results)}</td></tr>
    <tr><td>Cattle Macro F1</td><td style="font-weight:700">{cattle_macro_f1:.2f}%</td><td>Averaged over {len(cattle_classes)} breeds</td></tr>
    <tr><td>Buffalo Macro F1</td><td style="font-weight:700">{buffalo_macro_f1:.2f}%</td><td>Averaged over {len(buffalo_classes)} breeds</td></tr>
    <tr><td>Combined Top-1 Accuracy</td><td class="accent" style="font-weight:700">{combined_acc:.2f}%</td><td>{combined_correct} / {len(results)}</td></tr>
    <tr><td>Combined Top-3 Accuracy</td><td class="accent" style="font-weight:700">{top3_acc:.2f}%</td><td>{top3_correct} / {len(results)}</td></tr>
    <tr><td>Combined Top-5 Accuracy</td><td class="accent" style="font-weight:700">{top5_acc:.2f}%</td><td>{top5_correct} / {len(results)}</td></tr>
    <tr><td>Average Latency</td><td>{np.mean([r['latency_ms'] for r in results]):.1f} ms</td><td>Per image</td></tr>
    <tr><td>Throughput</td><td>{len(results)/total_time:.1f} img/s</td><td>Total: {total_time:.1f}s</td></tr>
  </table>
</div>

<!-- CONFUSION MATRICES -->
<div class="card">
  <h2>🔀 Confusion Matrices</h2>
  <div class="two-col">
    <div class="chart-section">
      <h3>Cattle ({len(cattle_results)} images)</h3>
      {chart_img_tag("cattle_cm", "Cattle Confusion Matrix")}
    </div>
    <div class="chart-section">
      <h3>Buffalo ({len(buffalo_results)} images)</h3>
      {chart_img_tag("buffalo_cm", "Buffalo Confusion Matrix")}
    </div>
  </div>
</div>

<!-- PER-BREED ACCURACY -->
<div class="card">
  <h2>📈 Per-Breed Accuracy</h2>
  <div class="two-col">
    <div class="chart-section">
      <h3>Cattle Accuracy Chart</h3>
      {chart_img_tag("cattle_breed_acc", "Cattle Per-Breed Accuracy")}
    </div>
    <div class="chart-section">
      <h3>Buffalo Accuracy Chart</h3>
      {chart_img_tag("buffalo_breed_acc", "Buffalo Per-Breed Accuracy")}
    </div>
  </div>

  <h3>Cattle — Per-Breed Detail ({len(cattle_results)} images)</h3>
  <table class="data-table">
    <tr><th>Breed</th><th style="text-align:center">Correct</th><th style="text-align:center">Total</th><th style="text-align:center">Accuracy</th></tr>
    {cattle_breed_rows}
  </table>

  <h3 style="margin-top:24px">Buffalo — Per-Breed Detail ({len(buffalo_results)} images)</h3>
  <table class="data-table">
    <tr><th>Breed</th><th style="text-align:center">Correct</th><th style="text-align:center">Total</th><th style="text-align:center">Accuracy</th></tr>
    {buffalo_breed_rows}
  </table>
</div>

<!-- MISCLASSIFIED -->
<div class="card">
  <h2>❌ Misclassified Images</h2>
  <div class="two-col">
    <div class="chart-section">
      <h3>Cattle Misclassifications</h3>
      {chart_img_tag("cattle_misclassified", "Cattle Misclassified")}
    </div>
    <div class="chart-section">
      <h3>Buffalo Misclassifications</h3>
      {chart_img_tag("buffalo_misclassified", "Buffalo Misclassified")}
    </div>
  </div>
</div>

<!-- §7 MULTI-IMAGE -->
<div class="card">
  <h2>🖼️ §7 — Multi-Image Testing Results</h2>
  {multi_img_html}
</div>

<div class="footer">
  <p>🐄 Cattle &amp; Buffalo Breed Classifier — Evaluation Report</p>
  <p>Model: {onnx_filename} · {len(results)} test images · Generated: {timestamp}</p>
</div>

</div>
</body>
</html>"""

# ── Save & download ──
report_html_path = "/content/evaluation_report.html"
with open(report_html_path, "w", encoding="utf-8") as f:
    f.write(html_report)

size_kb = os.path.getsize(report_html_path) / 1024
print(f"✅ Report saved: {report_html_path} ({size_kb:.0f} KB)")
print(f"   Contains: model info, single-image results, batch metrics,")
print(f"   confusion matrices, per-breed tables, misclassified gallery,")
print(f"   and multi-image results — all in one self-contained HTML file.")

try:
    files.download(report_html_path)
    print(f"\n📥 Download started!")
except Exception:
    print(f"\n⚠️  Auto-download failed. Download manually from Files panel → {report_html_path}")

# %% [markdown]
# ## §9 — Large-Scale Kaggle Dataset Evaluation
#
# Downloads a dataset directly from Kaggle and runs automated classification on **ALL** images.
# - Provide your Kaggle credentials (username + API key)
# - Set the `KAGGLE_DATASET_SLUG` below (default: training dataset for full-scale validation)
# - Auto-detects `species/breed/image` folder structure
# - Maps discovered breeds to your model's class maps
# - Skips breeds not present in the model
#
# > **To test on a different dataset**: change `KAGGLE_DATASET_SLUG` to any
# > Kaggle dataset with `cattle/<breed>/*.jpg` and/or `buffalo/<breed>/*.jpg` structure.

# %%
import os, zipfile, shutil, time, json, io
from collections import Counter, defaultdict
from getpass import getpass

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION — Change the dataset slug to test on a
#  different Kaggle dataset. Must have breed/image structure.
# ═══════════════════════════════════════════════════════════
KAGGLE_DATASET_SLUG = "algsoch/breed-cattle-buffalo"   # <-- change this
DOWNLOAD_DIR = "/content/kaggle_test_data"
# ═══════════════════════════════════════════════════════════

# --- Step 1: Kaggle credentials ---
print("🔑 Enter your Kaggle credentials (from kaggle.com → Account → API → Create Token):")
kaggle_user = input("   Kaggle Username: ").strip()
kaggle_key = getpass("   Kaggle API Key:  ").strip()

os.environ["KAGGLE_USERNAME"] = kaggle_user
os.environ["KAGGLE_KEY"] = kaggle_key
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
with open(kaggle_json, "w") as f:
    json.dump({"username": kaggle_user, "key": kaggle_key}, f)
os.chmod(kaggle_json, 0o600)
print(f"✅ Kaggle credentials configured for: {kaggle_user}")

# --- Step 2: Install kaggle CLI & download dataset ---
!pip install -q kaggle

if os.path.exists(DOWNLOAD_DIR):
    shutil.rmtree(DOWNLOAD_DIR)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print(f"\n📥 Downloading dataset: {KAGGLE_DATASET_SLUG}...")
ret = os.system(f"kaggle datasets download -d {KAGGLE_DATASET_SLUG} -p {DOWNLOAD_DIR} --unzip --force")
if ret != 0:
    raise RuntimeError(f"❌ Kaggle download failed (exit code {ret}). Check credentials and dataset slug.")

print(f"✅ Dataset downloaded to {DOWNLOAD_DIR}")

# --- Step 3: Auto-detect folder structure ---
def find_species_root(base_dir):
    """Walk down until we find cattle/ or buffalo/ directories."""
    for root, dirs, _files in os.walk(base_dir):
        dir_names_lower = [d.lower() for d in dirs]
        if "cattle" in dir_names_lower or "buffalo" in dir_names_lower:
            return root
    return None

data_root = find_species_root(DOWNLOAD_DIR)
if data_root is None:
    print(f"⚠️  No cattle/buffalo directories found. Contents of {DOWNLOAD_DIR}:")
    for item in sorted(os.listdir(DOWNLOAD_DIR))[:20]:
        print(f"   {item}")
    raise FileNotFoundError("❌ Could not find cattle/ or buffalo/ folder structure in downloaded dataset.")

print(f"📂 Data root: {data_root}")

# --- Step 4: Discover all images ---
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
kaggle_images = []

for species in ("cattle", "buffalo"):
    species_dir = os.path.join(data_root, species)
    if not os.path.isdir(species_dir):
        for d in os.listdir(data_root):
            if d.lower() == species:
                species_dir = os.path.join(data_root, d)
                break
    if not os.path.isdir(species_dir):
        print(f"⚠️  No '{species}/' directory in {data_root}")
        continue
    for breed in sorted(os.listdir(species_dir)):
        breed_dir = os.path.join(species_dir, breed)
        if not os.path.isdir(breed_dir):
            continue
        for fname in sorted(os.listdir(breed_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTS:
                kaggle_images.append((species, breed, os.path.join(breed_dir, fname)))

print(f"\n📊 Discovered {len(kaggle_images)} images")
species_counts = Counter(s for s, _, _ in kaggle_images)
for sp, cnt in species_counts.items():
    breed_cnt = len(set(b for s, b, _ in kaggle_images if s == sp))
    print(f"   {sp.title()}: {cnt} images across {breed_cnt} breeds")

# Check overlap with model's known classes
known_cattle = set(cattle_classes.keys()) if cattle_classes else set()
known_buffalo = set(buffalo_classes.keys()) if buffalo_classes else set()
dataset_cattle = set(b for s, b, _ in kaggle_images if s == "cattle")
dataset_buffalo = set(b for s, b, _ in kaggle_images if s == "buffalo")

matched_cattle = dataset_cattle & known_cattle
matched_buffalo = dataset_buffalo & known_buffalo
unknown_cattle = dataset_cattle - known_cattle
unknown_buffalo = dataset_buffalo - known_buffalo

print(f"\n🔗 Breed mapping:")
print(f"   Cattle:  {len(matched_cattle)}/{len(dataset_cattle)} breeds match model ({len(unknown_cattle)} unknown)")
print(f"   Buffalo: {len(matched_buffalo)}/{len(dataset_buffalo)} breeds match model ({len(unknown_buffalo)} unknown)")
if unknown_cattle:
    print(f"   ⚠️  Unknown cattle breeds (skipped): {sorted(unknown_cattle)[:10]}")
if unknown_buffalo:
    print(f"   ⚠️  Unknown buffalo breeds (skipped): {sorted(unknown_buffalo)[:10]}")

# Filter to only images with known breeds
eval_images = []
skipped_imgs = 0
for species, breed, fpath in kaggle_images:
    if species == "cattle" and breed in known_cattle:
        eval_images.append((species, breed, fpath))
    elif species == "buffalo" and breed in known_buffalo:
        eval_images.append((species, breed, fpath))
    else:
        skipped_imgs += 1

print(f"\n✅ {len(eval_images)} images ready for evaluation ({skipped_imgs} skipped — unknown breeds)")

# %% [markdown]
# ### §9.1 — Run Large-Scale Inference

# %%
from tqdm.notebook import tqdm as tqdm_notebook

print(f"\n🔄 Running inference on {len(eval_images)} images...\n")

kaggle_results = []
t_start = time.perf_counter()

for species_true, breed_true, filepath in tqdm_notebook(eval_images, desc="Large-scale eval"):
    try:
        img = Image.open(filepath)
        pred = predict_single(img, top_k=5)
        kaggle_results.append({
            "filepath": filepath,
            "species_true": species_true,
            "breed_true": breed_true,
            "species_pred": pred["species"].lower(),
            "breed_pred": pred["top_breed"].lower().replace(" ", "_"),
            "breed_conf": pred["top_breed_confidence"],
            "species_conf": pred["species_confidence"],
            "top5_breeds": [p["breed"].lower().replace(" ", "_") for p in pred["top_k_breeds"]],
            "latency_ms": pred["latency_ms"],
        })
    except Exception:
        pass

total_time_k = time.perf_counter() - t_start
print(f"\n✅ Evaluated {len(kaggle_results)} images in {total_time_k:.1f}s")
print(f"   Avg latency: {np.mean([r['latency_ms'] for r in kaggle_results]):.1f} ms/image")
print(f"   Throughput:  {len(kaggle_results)/total_time_k:.1f} img/s")

# %% [markdown]
# ### §9.2 — Full Metrics & Per-Breed Breakdown

# %%
k_species_correct = sum(1 for r in kaggle_results if r["species_true"] == r["species_pred"])
k_species_acc = k_species_correct / len(kaggle_results) * 100

k_cattle = [r for r in kaggle_results if r["species_true"] == "cattle"]
k_buffalo = [r for r in kaggle_results if r["species_true"] == "buffalo"]

k_cattle_correct = sum(1 for r in k_cattle if r["breed_true"] == r["breed_pred"])
k_buffalo_correct = sum(1 for r in k_buffalo if r["breed_true"] == r["breed_pred"])
k_cattle_acc = k_cattle_correct / len(k_cattle) * 100 if k_cattle else 0
k_buffalo_acc = k_buffalo_correct / len(k_buffalo) * 100 if k_buffalo else 0

k_combined = sum(1 for r in kaggle_results
    if r["species_true"] == r["species_pred"] and r["breed_true"] == r["breed_pred"])
k_combined_acc = k_combined / len(kaggle_results) * 100

k_top3 = sum(1 for r in kaggle_results
    if r["species_true"] == r["species_pred"] and r["breed_true"] in r["top5_breeds"][:3])
k_top5 = sum(1 for r in kaggle_results
    if r["species_true"] == r["species_pred"] and r["breed_true"] in r["top5_breeds"])
k_top3_acc = k_top3 / len(kaggle_results) * 100
k_top5_acc = k_top5 / len(kaggle_results) * 100

k_tp = sum(1 for r in kaggle_results if r["species_true"] == "buffalo" and r["species_pred"] == "buffalo")
k_fp = sum(1 for r in kaggle_results if r["species_true"] == "cattle" and r["species_pred"] == "buffalo")
k_fn = sum(1 for r in kaggle_results if r["species_true"] == "buffalo" and r["species_pred"] == "cattle")
k_binary_f1 = (2 * k_tp / (2 * k_tp + k_fp + k_fn) * 100) if (2 * k_tp + k_fp + k_fn) > 0 else 0

k_cattle_f1 = compute_macro_f1(k_cattle, cattle_classes)
k_buffalo_f1 = compute_macro_f1(k_buffalo, buffalo_classes)

def per_breed_stats(result_list):
    stats = defaultdict(lambda: {"correct": 0, "total": 0, "top3": 0, "top5": 0})
    for r in result_list:
        stats[r["breed_true"]]["total"] += 1
        if r["breed_true"] == r["breed_pred"]:
            stats[r["breed_true"]]["correct"] += 1
        if r["breed_true"] in r["top5_breeds"][:3]:
            stats[r["breed_true"]]["top3"] += 1
        if r["breed_true"] in r["top5_breeds"]:
            stats[r["breed_true"]]["top5"] += 1
    return stats

cattle_stats = per_breed_stats(k_cattle)
buffalo_stats = per_breed_stats(k_buffalo)

print(f"""
{'═'*70}
  📊 LARGE-SCALE EVALUATION — {KAGGLE_DATASET_SLUG}
{'═'*70}

  Dataset: {KAGGLE_DATASET_SLUG}
  Total images evaluated: {len(kaggle_results)}
  Cattle: {len(k_cattle)} images · Buffalo: {len(k_buffalo)} images

  ┌─────────────────────────────────────────────────────────┐
  │  BINARY (SPECIES) CLASSIFICATION                        │
  ├─────────────────────────────────────────────────────────┤
  │  Accuracy:    {k_species_acc:6.2f}%   ({k_species_correct}/{len(kaggle_results)})               │
  │  F1 Score:    {k_binary_f1:6.2f}%   (TP={k_tp} FP={k_fp} FN={k_fn})            │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  BREED CLASSIFICATION                                   │
  ├─────────────────────────────────────────────────────────┤
  │  Cattle Acc:  {k_cattle_acc:6.2f}%   ({k_cattle_correct}/{len(k_cattle)})                       │
  │  Buffalo Acc: {k_buffalo_acc:6.2f}%   ({k_buffalo_correct}/{len(k_buffalo)})                      │
  │  Cattle F1:   {k_cattle_f1:6.2f}%   (macro-avg, {len(cattle_stats)} breeds)          │
  │  Buffalo F1:  {k_buffalo_f1:6.2f}%   (macro-avg, {len(buffalo_stats)} breeds)         │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  COMBINED (SPECIES + BREED CORRECT)                     │
  ├─────────────────────────────────────────────────────────┤
  │  Top-1 Acc:   {k_combined_acc:6.2f}%   ({k_combined}/{len(kaggle_results)})                      │
  │  Top-3 Acc:   {k_top3_acc:6.2f}%   ({k_top3}/{len(kaggle_results)})                      │
  │  Top-5 Acc:   {k_top5_acc:6.2f}%   ({k_top5}/{len(kaggle_results)})                      │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  PERFORMANCE                                            │
  ├─────────────────────────────────────────────────────────┤
  │  Total time:  {total_time_k:6.1f}s                                        │
  │  Avg latency: {np.mean([r['latency_ms'] for r in kaggle_results]):6.1f} ms/img                                 │
  │  Throughput:  {len(kaggle_results)/total_time_k:6.1f} img/s                                  │
  └─────────────────────────────────────────────────────────┘
""")

for species_name, stats, n_imgs in [("CATTLE", cattle_stats, len(k_cattle)),
                                      ("BUFFALO", buffalo_stats, len(k_buffalo))]:
    if not stats:
        continue
    print(f"\n  📋 {species_name} PER-BREED BREAKDOWN ({n_imgs} images, {len(stats)} breeds)")
    print(f"  {'─'*68}")
    print(f"  {'Breed':<25s} {'Imgs':>5s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s} {'Status':>8s}")
    print(f"  {'─'*68}")
    breed_accs = []
    for breed in sorted(stats.keys()):
        s = stats[breed]
        acc = s["correct"] / s["total"] * 100
        t3 = s["top3"] / s["total"] * 100
        t5 = s["top5"] / s["total"] * 100
        breed_accs.append(acc)
        marker = "✅" if acc >= 80 else ("⚠️" if acc >= 50 else "❌")
        display = breed.replace("_", " ").title()
        print(f"  {display:<25s} {s['total']:>5d} {acc:>6.1f}% {t3:>6.1f}% {t5:>6.1f}%   {marker}")
    avg = np.mean(breed_accs)
    print(f"  {'─'*68}")
    print(f"  {'AVERAGE':<25s} {'':>5s} {avg:>6.1f}%")

# %% [markdown]
# ### §9.3 — Confusion Matrices (Kaggle Dataset)

# %%
if k_cattle:
    plot_confusion_matrix(k_cattle, cattle_classes, "Cattle (Kaggle Large-Scale)")
if k_buffalo:
    plot_confusion_matrix(k_buffalo, buffalo_classes, "Buffalo (Kaggle Large-Scale)")

# %% [markdown]
# ### §9.4 — Save Large-Scale Report

# %%
kaggle_report = {
    "dataset": KAGGLE_DATASET_SLUG,
    "total_images": len(kaggle_results),
    "cattle_images": len(k_cattle),
    "buffalo_images": len(k_buffalo),
    "binary_accuracy": round(k_species_acc, 4),
    "binary_f1": round(k_binary_f1, 4),
    "cattle_breed_accuracy": round(k_cattle_acc, 4),
    "buffalo_breed_accuracy": round(k_buffalo_acc, 4),
    "cattle_macro_f1": round(k_cattle_f1, 4),
    "buffalo_macro_f1": round(k_buffalo_f1, 4),
    "combined_top1": round(k_combined_acc, 4),
    "combined_top3": round(k_top3_acc, 4),
    "combined_top5": round(k_top5_acc, 4),
    "avg_latency_ms": round(float(np.mean([r['latency_ms'] for r in kaggle_results])), 1),
    "throughput_imgs_per_sec": round(len(kaggle_results) / total_time_k, 1),
    "per_breed_cattle": {b: {"accuracy": round(s["correct"]/s["total"]*100, 2),
                              "top3": round(s["top3"]/s["total"]*100, 2),
                              "total": s["total"]}
                          for b, s in cattle_stats.items()},
    "per_breed_buffalo": {b: {"accuracy": round(s["correct"]/s["total"]*100, 2),
                               "top3": round(s["top3"]/s["total"]*100, 2),
                               "total": s["total"]}
                           for b, s in buffalo_stats.items()},
}

report_path_k = "/content/kaggle_large_scale_report.json"
with open(report_path_k, "w") as f:
    json.dump(kaggle_report, f, indent=2)

print(f"📄 Full report saved: {report_path_k}")
print(json.dumps(kaggle_report, indent=2))
try:
    files.download(report_path_k)
    print("\n📥 Download started!")
except Exception:
    print("\n⚠️  Download manually from Files panel")

# %% [markdown]
# ---
#
# ## 📝 Notes
#
# ### Preprocessing Pipeline
# The inference preprocessing exactly matches what was used during training evaluation:
# ```
# Image → Resize(260) → CenterCrop(260) → ToTensor() → [3, 260, 260] float32
# ```
#
# ### ONNX Model Outputs
# The exported ONNX model has 3 outputs (matching the `_ExportWrapper` in `src/export.py`):
# | Output | Shape | Description |
# |---|---|---|
# | `binary` | `[B, 2]` | Species logits: index 0 = cattle, 1 = buffalo |
# | `cattle` | `[B, 57]` | Cattle breed logits |
# | `buffalo` | `[B, 18]` | Buffalo breed logits |
#
# ### Inference Flow
# ```
# binary = softmax(output[0]) → argmax → species (0=cattle, 1=buffalo)
# breed_logits = output[1] if cattle else output[2]
# breed = softmax(breed_logits) → argmax → breed name (from class map)
# ```


