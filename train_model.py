#!/usr/bin/env python3
"""
🐄 Cattle & Buffalo Breed Classifier — Unified Training CLI
===========================================================

High-level CLI entrypoint supporting the modern training pipeline:
- Dynamic head sizing (breed_dim 384 / 512, auto class mapping)
- SOTA multi-task training (Warmup -> Fine-tuning + CosineAnnealing + EMA + SupCon)
- Multi-format auto-export (Portable bundle, ONNX FP32, Float16, INT8 PTQ)
- Automatic OOD detector calibration (Energy & MSP validation profiling)
- Web Platform synchronization (registers model in outputs/app_config.json with --set-active)

Usage:
    # 1. Fast sanity check (tiny mini-dataset, 1 epoch):
    python train_model.py --smoke-test --skip-qat

    # 2. Fast local iteration (25% dataset per breed):
    python train_model.py --quarter-data --backbone lite2

    # 3. Half-dataset training with multi-format export:
    python train_model.py --half-data --backbone lite2

    # 4. Full training and automatically set as the active model in the web app:
    python train_model.py --full-data --backbone lite2 --set-active

    # 5. ArcFace cosine breed heads with light augmentations:
    python train_model.py --cosine-head --augment-preset light
"""

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional

def _find_project_root() -> str:
    """Robustly find the project root directory containing src/config.py.
    
    Walks up from:
      1. SCRIPT_DIR (the directory of the script file)
      2. Current working directory (os.getcwd())
      3. PYTHONPATH and sys.path entries
      4. Any parent directories up to filesystem root
    """
    candidates = []
    if "__file__" in globals() and __file__:
        try:
            candidates.append(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
    try:
        candidates.append(os.path.abspath(os.getcwd()))
    except Exception:
        pass
    for p in sys.path:
        if p and os.path.isdir(p):
            candidates.append(os.path.abspath(p))

    # 1. Walk up looking for src/config.py
    for start in candidates:
        cur = start
        while cur:
            if os.path.isfile(os.path.join(cur, "src", "config.py")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    # 2. Check 1-level down from candidates
    for start in candidates:
        if os.path.isdir(start):
            try:
                for entry in os.listdir(start):
                    child = os.path.join(start, entry)
                    if os.path.isdir(child) and os.path.isfile(os.path.join(child, "src", "config.py")):
                        return child
            except Exception:
                pass

    # 3. Fallback
    if "__file__" in globals() and __file__:
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


PROJECT_ROOT = _find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    os.chdir(PROJECT_ROOT)
except Exception:
    pass

from src.config import (
    BACKBONE_WEIGHTS, BATCH_SIZE, BREED_DIM, CHECKPOINT_DIR,
    EXPORT_DIR, NUM_WORKERS, OOD_DIR, OOD_PERCENTILE,
    PORTABLE_EXPORT_DIR, RAW_DATA_DIR, SEED, SPLIT_DIR
)


# ============================================================
#  Preflight & Helpers
# ============================================================

# All standalone src modules required for training.
# Note: cbam, efficientnet_lite, traits, and ood_detector are integrated into model.py.
REQUIRED_SRC_MODULES = [
    "__init__.py", "config.py", "run_utils.py", "data_pipeline.py",
    "model.py", "train.py", "metrics.py", "export.py",
    "evaluate.py", "parity_check.py", "verify.py"
]


def _banner(title: str, char: str = "="):
    line = char * 68
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def _file_size_mb(path: str) -> float:
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def _python_exe() -> str:
    """Return venv python if present, else sys.executable."""
    venv_dir = os.path.join(PROJECT_ROOT, ".venv")
    if os.path.isdir(venv_dir):
        if platform.system() == "Windows":
            exe = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            exe = os.path.join(venv_dir, "bin", "python")
        if os.path.exists(exe):
            return exe
    return sys.executable


def preflight_check():
    """Verify runtime environment and dependencies."""
    _banner("PREFLIGHT ENVIRONMENT CHECK", "-")
    
    # 1. Verify src modules
    missing = [m for m in REQUIRED_SRC_MODULES if not os.path.exists(os.path.join(PROJECT_ROOT, "src", m))]
    if missing:
        raise RuntimeError(f"Missing core src module(s): {', '.join(missing)}")
    print("  [OK] Core source modules verified.")

    # 2. Check Torch & Device
    try:
        import torch
        print(f"  [OK] PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  [OK] CUDA GPU Available: {dev_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print("  [INFO] Running on CPU (CUDA not detected)")
    except ImportError:
        raise RuntimeError("PyTorch is not installed in the active environment.")

    # 3. Check data availability
    cattle_raw = os.path.join(RAW_DATA_DIR, "cattle")
    buffalo_raw = os.path.join(RAW_DATA_DIR, "buffalo")
    if not os.path.exists(cattle_raw) and not os.path.exists(buffalo_raw):
        splits_exist = os.path.exists(os.path.join(SPLIT_DIR, "train.csv"))
        if not splits_exist:
            print("  [WARNING] Neither data/raw nor data/splits found. Ensure dataset is downloaded.")
        else:
            print("  [OK] Found data/splits/ CSV files.")
    else:
        print("  [OK] Raw image dataset located.")


# ============================================================
#  App Config DB Synchronization
# ============================================================

def sync_shared_config(model_tag: str, model_path: str, ood_profile_path: Optional[str] = None, set_active: bool = False):
    """Register newly trained model and its OOD profile in outputs/app_config.json."""
    config_file = os.path.join(PROJECT_ROOT, "outputs", "app_config.json")
    os.makedirs(os.path.dirname(config_file), exist_ok=True)

    data = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data.setdefault("version", 1)
    ood_section = data.setdefault("ood", {
        "enabled": True,
        "selected_profile": "auto",
        "energy_threshold": -5.0,
        "msp_threshold": 0.5,
        "temperature": 1.0,
    })

    # Read calibrated threshold values if available
    ood_prof_name = None
    if ood_profile_path and os.path.exists(ood_profile_path):
        try:
            with open(ood_profile_path, "r", encoding="utf-8") as f:
                ood_meta = json.load(f)
            ood_prof_name = os.path.basename(ood_profile_path)
            if set_active:
                ood_section["selected_profile"] = ood_prof_name
                if "energy_threshold" in ood_meta:
                    ood_section["energy_threshold"] = round(float(ood_meta["energy_threshold"]), 4)
                if "msp_threshold" in ood_meta:
                    ood_section["msp_threshold"] = round(float(ood_meta["msp_threshold"]), 4)
                if "temperature" in ood_meta:
                    ood_section["temperature"] = float(ood_meta["temperature"])
        except Exception as e:
            print(f"  [Warning] Could not parse OOD thresholds file: {e}")

    # Set active model if requested
    if set_active:
        data["active_model"] = model_tag
        print(f"  [Config] Set active model -> {model_tag}")
        if ood_prof_name:
            print(f"  [Config] Set active OOD profile -> {ood_prof_name}")

    # Write back atomically
    tmp = config_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, config_file)
    print(f"  [Config] Updated {os.path.relpath(config_file, PROJECT_ROOT)}")


# ============================================================
#  Post-Training Exports & Calibration
# ============================================================

def run_post_training_pipeline(args, best_ckpt: str, run_id: str):
    """Export to multiple formats, calibrate OOD, and synchronize shared app config."""
    _banner("STAGE 2: MULTI-FORMAT EXPORTS & CALIBRATION", "=")
    py_exe = _python_exe()
    
    # 1. Multi-format exports
    if not args.no_export and best_ckpt and os.path.exists(best_ckpt):
        formats = [f.strip().lower() for f in args.export_formats.split(",") if f.strip()]
        for fmt in formats:
            if fmt == "portable":
                print("\n  📦 Ensuring portable model bundle...")
                subprocess.run([
                    py_exe, "-m", "src.export",
                    "--mode", "portable",
                    "--backbone", args.backbone,
                    "--checkpoint", best_ckpt
                ], cwd=PROJECT_ROOT)
            elif fmt == "onnx":
                print("\n  📦 Exporting ONNX (FP32)...")
                subprocess.run([
                    py_exe, "-m", "src.export",
                    "--mode", "onnx",
                    "--backbone", args.backbone,
                    "--checkpoint", best_ckpt
                ], cwd=PROJECT_ROOT)
            elif fmt == "float16":
                print("\n  📦 Exporting Float16...")
                subprocess.run([
                    py_exe, "-m", "src.export",
                    "--mode", "float16",
                    "--backbone", args.backbone,
                    "--checkpoint", best_ckpt
                ], cwd=PROJECT_ROOT)
            elif fmt in ("int8", "onnx-int8"):
                print("\n  📦 Exporting INT8 (PTQ)...")
                subprocess.run([
                    py_exe, "-m", "src.export",
                    "--mode", "onnx-int8",
                    "--backbone", args.backbone,
                    "--checkpoint", best_ckpt
                ], cwd=PROJECT_ROOT)

    # 2. OOD Calibration
    calibrated_profile_path = None
    if not args.no_ood_calibrate and best_ckpt and os.path.exists(best_ckpt):
        print("\n  🛡️  Calibrating Out-Of-Distribution (OOD) safety thresholds...")
        cal_cmd = [
            py_exe, "scripts/calibrate_ood.py",
            "--backbone", args.backbone,
            "--checkpoint", best_ckpt,
            "--percentile", str(args.ood_percentile)
        ]
        res = subprocess.run(cal_cmd, cwd=PROJECT_ROOT)
        if res.returncode == 0:
            ckpt_stem = os.path.splitext(os.path.basename(best_ckpt))[0]
            candidate_json = os.path.join(OOD_DIR, f"{ckpt_stem}.json")
            if os.path.exists(candidate_json):
                calibrated_profile_path = candidate_json

    # 3. Synchronize with Web Application Shared Database
    ckpt_name = os.path.splitext(os.path.basename(best_ckpt))[0]
    sync_shared_config(
        model_tag=ckpt_name,
        model_path=best_ckpt,
        ood_profile_path=calibrated_profile_path,
        set_active=args.set_active
    )


# ============================================================
#  Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cattle & Buffalo Breed Classifier — Unified Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke-test", action="store_true",
                            help="Sanity test: mini dataset, 1 epoch, skip QAT")
    mode_group.add_argument("--quarter-data", action="store_true",
                            help="Use 25%% of images per breed (fast local iteration)")
    mode_group.add_argument("--half-data", action="store_true",
                            help="Use 50%% of images per breed")
    mode_group.add_argument("--full-data", action="store_true",
                            help="Explicitly use 100%% of images")

    # Architecture & Model
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2",
                        help="EfficientNet-Lite backbone")
    parser.add_argument("--attention", choices=["cbam", "se"], default="cbam",
                        help="Attention module")
    parser.add_argument("--breed-dim", type=int, default=BREED_DIM,
                        help="Intermediate hidden dimension in breed heads (384 or 512)")
    parser.add_argument("--cosine-head", action="store_true",
                        help="Enable ArcFace normalized scaled-cosine breed heads")
    parser.add_argument("--weights", default=None,
                        help="Pretrained backbone weights path")

    # Epochs & Hyperparameters
    parser.add_argument("--phase1-epochs", type=int, default=None,
                        help="Warmup epochs (default: 8, or 1 in smoke-test)")
    parser.add_argument("--phase2-epochs", type=int, default=None,
                        help="Fine-tune epochs (default: 80, or 1 in smoke-test)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size per step")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="DataLoader worker count")
    parser.add_argument("--device", default=None,
                        help="Force 'cuda' or 'cpu'")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--min-images-per-breed", type=int, default=15,
                        help="Minimum images required per breed (dynamic head sizing)")
    parser.add_argument("--contrastive-weight", type=float, default=0.2,
                        help="Supervised contrastive loss weight on feature embeddings")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Label smoothing factor")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Phase 2 linear warmup epochs")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="AdamW weight decay")

    # QAT
    parser.add_argument("--include-qat", action="store_true",
                        help="Enable Phase 3 QAT fine-tuning (opt-in)")
    parser.add_argument("--skip-qat", action="store_true", default=True,
                        help="Skip Phase 3 QAT (default: True, PTQ preferred)")

    # Augmentation
    parser.add_argument("--augment-preset", choices=["none", "light"], default="none",
                        help="'light' = flip + mild RRC + mild color jitter")
    parser.add_argument("--augment-all", action="store_true",
                        help="Enable all augmentations (flip, jitter, randaug, rrc, mix)")
    parser.add_argument("--mix", action="store_true", help="Enable CutMix/MixUp")
    parser.add_argument("--flip", action="store_true", help="Enable RandomHorizontalFlip")
    parser.add_argument("--color-jitter", action="store_true", help="Enable ColorJitter")
    parser.add_argument("--randaugment", action="store_true", help="Enable RandAugment")
    parser.add_argument("--rrc", action="store_true", help="Enable RandomResizedCrop")
    parser.add_argument("--pad-to-square", action="store_true",
                        help="Preserve full-body aspect ratio via square padding")

    # Exports & Calibration (The New Method)
    parser.add_argument("--export-formats", default="portable,onnx,float16",
                        help="Comma-separated export formats (portable, onnx, float16, int8)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip post-training model exports")
    parser.add_argument("--no-ood-calibrate", action="store_true",
                        help="Skip post-training OOD calibration")
    parser.add_argument("--ood-percentile", type=float, default=OOD_PERCENTILE,
                        help="Percentile of validation energy to set as OOD threshold")

    # Platform & Web App Integration
    parser.add_argument("--set-active", action="store_true",
                        help="Automatically set newly trained model as active in outputs/app_config.json")
    parser.add_argument("--run-tag", default=None,
                        help="Custom timestamp or identifier tag for model outputs")

    args = parser.parse_args()

    # Preflight
    preflight_check()

    _banner("STAGE 1: MODEL TRAINING", "=")
    print(f"  Backbone:           {args.backbone}")
    print(f"  Attention:          {args.attention}")
    print(f"  Breed Head Dim:     {args.breed_dim}")
    print(f"  Cosine Heads:       {'YES (ArcFace)' if args.cosine_head else 'NO'}")
    
    if args.smoke_test:
        mode_str = "SMOKE TEST (1 epoch, mini dataset sanity run)"
    elif args.quarter_data:
        mode_str = "QUARTER DATA (25% dataset per breed)"
    elif args.half_data:
        mode_str = "HALF DATA (50% dataset per breed)"
    else:
        mode_str = "FULL DATA (100% dataset)"
    print(f"  Dataset Mode:       {mode_str}")
    print(f"  Augmentation:       {args.augment_preset} (all={args.augment_all})")
    print(f"  Auto Multi-Export:  {'NO' if args.no_export else args.export_formats}")
    print(f"  OOD Calibration:    {'NO' if args.no_ood_calibrate else f'YES ({args.ood_percentile}%)'}")
    print(f"  Set Active Model:   {'YES' if args.set_active else 'NO'}")

    # Build src.train argument list
    train_args = [
        _python_exe(), "-m", "src.train",
        "--backbone", args.backbone,
        "--attention", args.attention,
        "--breed-dim", str(args.breed_dim),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--min-images-per-breed", str(args.min_images_per_breed),
        "--contrastive-weight", str(args.contrastive_weight),
        "--label-smoothing", str(args.label_smoothing),
        "--warmup-epochs", str(args.warmup_epochs),
        "--grad-accum", str(args.grad_accum),
        "--weight-decay", str(args.weight_decay),
    ]

    if args.smoke_test:
        train_args.extend(["--smoke-test", "--skip-qat"])
    elif args.quarter_data:
        train_args.append("--quarter-data")
    elif args.half_data:
        train_args.append("--half-data")

    if args.cosine_head:
        train_args.append("--cosine-head")
    if args.include_qat:
        train_args.append("--include-qat")
    elif args.skip_qat:
        train_args.append("--skip-qat")

    if args.phase1_epochs is not None:
        train_args.extend(["--phase1-epochs", str(args.phase1_epochs)])
    if args.phase2_epochs is not None:
        train_args.extend(["--phase2-epochs", str(args.phase2_epochs)])
    if args.device:
        train_args.extend(["--device", args.device])
    if args.weights:
        train_args.extend(["--weights", args.weights])
    if args.run_tag:
        train_args.extend(["--run-tag", args.run_tag])

    # Augmentations
    if args.augment_all:
        train_args.append("--augment-all")
    else:
        for flag, attr in (("--mix", "mix"), ("--flip", "flip"),
                           ("--color-jitter", "color_jitter"),
                           ("--randaugment", "randaugment"), ("--rrc", "rrc")):
            if getattr(args, attr, False):
                train_args.append(flag)
    if args.augment_preset and args.augment_preset != "none":
        train_args.extend(["--augment-preset", args.augment_preset])
    if args.pad_to_square:
        train_args.append("--pad-to-square")

    # Run training
    start_time = time.time()
    res = subprocess.run(train_args, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        print(f"\n❌ Training failed with exit code {res.returncode}")
        return res.returncode

    duration = time.time() - start_time
    m, s = divmod(int(duration), 60)
    print(f"\n  ✅ Training completed in {m}m {s}s")

    # Locate the best checkpoint produced
    from src.run_utils import find_latest_checkpoint
    best_ckpt = None
    for phase in ("phase2", "phase3", "phase1"):
        candidate = find_latest_checkpoint(CHECKPOINT_DIR, args.backbone, phase)
        if candidate and os.path.exists(candidate):
            best_ckpt = candidate
            break

    if not best_ckpt:
        cand_pt = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, f"{args.backbone}_*.pt")))
        if cand_pt:
            best_ckpt = cand_pt[-1]

    if best_ckpt:
        print(f"\n  🏆 Best Checkpoint: {os.path.basename(best_ckpt)} ({_file_size_mb(best_ckpt):.1f} MB)")
        run_post_training_pipeline(args, best_ckpt, run_id=os.path.splitext(os.path.basename(best_ckpt))[0])
    else:
        print("\n  ⚠️  No checkpoint file found in outputs/checkpoints/")

    # Print Final Summary Banner
    _banner("TRAINING PIPELINE COMPLETE", "=")
    print("  Artifacts Summary:")
    if best_ckpt:
        print(f"    • Checkpoint:       {os.path.relpath(best_ckpt, PROJECT_ROOT)}")
    portable_bundles = sorted(glob.glob(os.path.join(PORTABLE_EXPORT_DIR, f"{args.backbone}_*")))
    if portable_bundles:
        print(f"    • Portable Bundle:  {os.path.relpath(portable_bundles[-1], PROJECT_ROOT)}")
    onnx_files = sorted(glob.glob(os.path.join(EXPORT_DIR, f"{args.backbone}_*.onnx")))
    if onnx_files:
        print(f"    • ONNX Model:       {os.path.relpath(onnx_files[-1], PROJECT_ROOT)}")
    ood_files = sorted(glob.glob(os.path.join(OOD_DIR, "*.json")))
    if ood_files:
        print(f"    • OOD Profile:      {os.path.relpath(ood_files[-1], PROJECT_ROOT)}")

    print("\n  Next Steps:")
    print("    1. Start the Dual-Portal Web Platform (User & Dev pages):")
    print("       python run.py")
    print(f"    2. Run full evaluation metrics:")
    print(f"       python -m src.evaluate --backbone {args.backbone}")
    print("    3. Launch Visual GUI Tester:")
    print("       python test_model.py\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
