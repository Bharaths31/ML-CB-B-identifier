#!/usr/bin/env python3
"""
🐄 Cattle & Buffalo Breed Classifier — Local Training Automation
================================================================

Fully automated pipeline: prerequisites → venv → dataset → verify → train → export.

Usage:
    python local_train.py                     # Full data, skip QAT
    python local_train.py --half-data         # 50% data per breed (quick)
    python local_train.py --quarter-data      # 25% data per breed (fastest local)
    python local_train.py --smoke-test        # Tiny dataset, 1 epoch (sanity)
    python local_train.py --full-data         # Explicitly use all images
    python local_train.py --include-qat       # Include QAT phase 3
    python local_train.py --skip-download     # Skip Kaggle download
    python local_train.py --skip-setup        # Skip venv creation
    python local_train.py --backbone lite4    # Use lite4 backbone
    python local_train.py --help              # Show all options
"""

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile

# ============================================================
#  Constants
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR  # This script lives at project root

KAGGLE_DATASET = "algsoch/breed-cattle-buffalo"

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements.txt")

# Webapp-only deps to skip for training-only installs
WEBAPP_DEPS = {"fastapi", "uvicorn", "python-multipart", "mem0ai",
               "chromadb", "sentence-transformers", "litellm"}

# Export formats to produce after training
EXPORT_FORMATS = ["portable", "onnx", "int8", "float16"]


# ============================================================
#  Helpers
# ============================================================

def _banner(stage, title):
    """Print a prominent stage banner."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  §{stage} — {title}")
    print(f"{'=' * width}\n")


def _run(cmd, cwd=None, env=None, check=True, capture=False):
    """Run a subprocess with live output (unless capture=True)."""
    cwd = cwd or PROJECT_ROOT
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    kwargs = dict(cwd=cwd, env=merged_env)
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        if capture:
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def _python():
    """Return the path to the venv Python, or system Python."""
    if os.path.isdir(VENV_DIR):
        if platform.system() == "Windows":
            return os.path.join(VENV_DIR, "Scripts", "python.exe")
        return os.path.join(VENV_DIR, "bin", "python")
    return sys.executable


def _pip():
    """Return pip command as a list."""
    return [_python(), "-m", "pip"]


def _elapsed(start):
    """Format elapsed time."""
    e = time.time() - start
    return time.strftime("%H:%M:%S", time.gmtime(e))


def _file_size_mb(path):
    """Return file size in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


# ============================================================
#  Windows prerequisite helpers
# ============================================================

def _check_windows_build_tools():
    """Warn or raise if Visual C++ Build Tools are missing on Windows.

    PyTorch wheels ship pre-compiled on PyPI so MSVC is NOT needed to *install*
    torch.  However, some optional C-extension packages (e.g. sentencepiece,
    triton, or any package without a wheel) will fail to build without MSVC.
    We check for the compiler at cl.exe and for the VS Build Tools via the
    vswhere utility, and print a clear actionable message if they are absent.
    """
    import shutil
    
    # 1. Check cl.exe (MSVC compiler) is directly accessible in PATH
    cl_path = shutil.which("cl")
    if cl_path:
        print(f"  ✅ MSVC compiler found: {cl_path}")
        return

    # 2. Check vswhere (ships with VS 2017+ and Build Tools)
    vswhere = shutil.which("vswhere") or os.path.expandvars(
        r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
    
    if os.path.exists(vswhere):
        try:
            # We must use '-products *' because Build Tools is considered a different 
            # product family than Visual Studio (Community/Pro/Enterprise)
            result = subprocess.run(
                [vswhere, "-latest", "-products", "*", "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "displayName"],
                capture_output=True, text=True, timeout=10)
            
            vs_name = result.stdout.strip().split('\n')[0]  # Take first line if multiple
            if vs_name:
                print(f"  ✅ Visual Studio / Build Tools found: {vs_name}")
                print("  ℹ️  Note: If a pip install fails, try running from the 'x64 Native Tools Command Prompt'")
                return
        except Exception:
            pass

    # 3. Report missing
    print()
    print("  ⚠️  WARNING: Missing Windows build dependencies:")
    print("      - cl.exe not in PATH")
    print("      - Visual Studio / Build Tools not found via vswhere")
    print()
    print("  Some Python packages require C++ compilation and will FAIL to install.")
    print("  Fix: Install 'Microsoft C++ Build Tools' (free):")
    print("    https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("  Select workload: 'Desktop development with C++'")
    print()
    print("  PyTorch itself installs fine without MSVC (uses pre-built wheels).")
    print("  Only continue if you do NOT need packages that compile C extensions.")
    print()
    # Don't raise — PyTorch training works without MSVC on Windows.
    # The user is warned and can proceed if they only need torch+torchvision.

# ============================================================
#  §0 — Prerequisites Check
# ============================================================

def stage_prerequisites(args):
    _banner(0, "Prerequisites Check")

    # Python version
    v = sys.version_info
    print(f"  Python: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        raise RuntimeError("Python 3.9+ required")
    print("  ✅ Python version OK")

    # git
    try:
        r = _run(["git", "--version"], capture=True)
        print(f"  ✅ {r.stdout.strip()}")
    except (FileNotFoundError, RuntimeError):
        raise RuntimeError("git is required but not found in PATH")

    # pip
    if not args.skip_setup:
        try:
            r = _run([*_pip(), "--version"], capture=True)
            print(f"  ✅ pip available")
        except RuntimeError:
            raise RuntimeError("pip is required. Install with: python -m ensurepip")
    else:
        print("  ⏭️  Skipping pip check (--skip-setup)")

    # Windows-specific: check for Visual C++ / Build Tools
    if platform.system() == "Windows":
        _check_windows_build_tools()

    # Check project structure
    if not os.path.exists(os.path.join(PROJECT_ROOT, "src", "config.py")):
        raise RuntimeError(
            f"src/config.py not found. Run this script from the project root "
            f"(ML-CB-B-identifier/)")
    print("  ✅ Project structure verified")


# ============================================================
#  GPU detection & CUDA PyTorch installation
# ============================================================

# Map CUDA driver major.minor → PyTorch CUDA index tag
# PyTorch publishes wheels for specific CUDA versions; pick the closest one
# that doesn't exceed the driver's supported CUDA version.
PYTORCH_CUDA_INDEXES = [
    # (min_driver_cuda, index_tag)
    (12, 4, "cu124"),   # CUDA 12.4+
    (12, 1, "cu121"),   # CUDA 12.1–12.3
    (11, 8, "cu118"),   # CUDA 11.8
]

PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/{tag}"


def _detect_nvidia_gpu():
    """Check for NVIDIA GPU via nvidia-smi. Returns (gpu_name, cuda_version) or None."""
    import shutil
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print("  ℹ️  No NVIDIA GPU detected (nvidia-smi not found) → will use CPU")
        return None

    try:
        r = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print("  ℹ️  nvidia-smi failed → will use CPU")
            return None

        # Parse first GPU line: "NVIDIA GeForce RTX 3050, 560.35.03"
        line = r.stdout.strip().split("\n")[0]
        parts = line.split(",")
        gpu_name = parts[0].strip()

        # Get CUDA version from nvidia-smi
        r2 = subprocess.run(
            [nvidia_smi], capture_output=True, text=True, timeout=10)
        # Look for "CUDA Version: 12.6" in the output
        cuda_ver = None
        for out_line in r2.stdout.split("\n"):
            if "CUDA Version" in out_line:
                import re
                m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out_line)
                if m:
                    cuda_ver = (int(m.group(1)), int(m.group(2)))
                break

        if cuda_ver:
            print(f"  🎮 NVIDIA GPU detected: {gpu_name}")
            print(f"  🔧 CUDA driver version: {cuda_ver[0]}.{cuda_ver[1]}")
            return {"gpu": gpu_name, "cuda": cuda_ver}
        else:
            print(f"  🎮 NVIDIA GPU detected: {gpu_name}")
            print(f"  ⚠️  Could not determine CUDA version → will try CUDA 12.4")
            return {"gpu": gpu_name, "cuda": (12, 4)}

    except Exception as e:
        print(f"  ℹ️  GPU detection failed: {e} → will use CPU")
        return None


def _install_torch(torch_deps, gpu_info):
    """Install torch/torchvision with CUDA support if a GPU was detected."""
    if gpu_info is None:
        # No GPU — install CPU version from PyPI (default)
        print("  Installing PyTorch (CPU)...")
        _run([*_pip(), "install", "-q", *torch_deps])
        print("  ✅ Installed PyTorch (CPU-only)")
        return

    # Find best matching CUDA index
    cuda_major, cuda_minor = gpu_info["cuda"]
    chosen_tag = None
    for req_major, req_minor, tag in PYTORCH_CUDA_INDEXES:
        if (cuda_major, cuda_minor) >= (req_major, req_minor):
            chosen_tag = tag
            break

    if chosen_tag is None:
        print(f"  ⚠️  CUDA {cuda_major}.{cuda_minor} is too old for GPU PyTorch")
        print("  Falling back to CPU-only PyTorch...")
        _run([*_pip(), "install", "-q", *torch_deps])
        print("  ✅ Installed PyTorch (CPU-only)")
        return

    index_url = PYTORCH_INDEX_URL.format(tag=chosen_tag)
    print(f"  Installing PyTorch with CUDA ({chosen_tag}) for {gpu_info['gpu']}...")
    print(f"  Index: {index_url}")

    # Uninstall existing CPU torch first (if any) to avoid conflicts
    _run([*_pip(), "uninstall", "-y", "torch", "torchvision"],
         check=False)

    # Install CUDA version
    _run([*_pip(), "install", "-q", *torch_deps,
          "--index-url", index_url])
    print(f"  ✅ Installed PyTorch with CUDA ({chosen_tag})")


# ============================================================
#  §1 — Environment Setup
# ============================================================

def stage_setup_env():
    _banner(1, "Environment Setup (venv + dependencies)")

    if os.path.isdir(VENV_DIR):
        print(f"  ✅ Virtual environment already exists: {VENV_DIR}")
    else:
        print(f"  Creating virtual environment at {VENV_DIR}...")
        _run([sys.executable, "-m", "venv", VENV_DIR])
        print(f"  ✅ venv created")

    # Upgrade pip
    print("  Upgrading pip...")
    _run([*_pip(), "install", "--upgrade", "pip", "-q"])

    # --- GPU Detection & CUDA PyTorch ---
    # By default, `pip install torch` from PyPI installs CPU-only builds.
    # We must detect NVIDIA GPUs and install from the PyTorch CUDA index.
    gpu_detected = _detect_nvidia_gpu()

    # Install training-only dependencies (skip webapp deps)
    print("  Installing training dependencies...")
    if os.path.exists(REQUIREMENTS):
        # Read requirements and filter out webapp-only deps
        with open(REQUIREMENTS) as f:
            lines = f.readlines()
        training_deps = []
        torch_deps = []  # torch/torchvision handled separately for CUDA
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg_name = line.split(">=")[0].split("==")[0].split("[")[0].strip()
            if pkg_name.lower() in WEBAPP_DEPS:
                continue
            if pkg_name.lower() in ("torch", "torchvision"):
                torch_deps.append(line)
            else:
                training_deps.append(line)

        # Install torch/torchvision with CUDA if GPU detected
        if torch_deps:
            _install_torch(torch_deps, gpu_detected)

        # Install remaining training deps
        if training_deps:
            _run([*_pip(), "install", "-q", *training_deps])
            print(f"  ✅ Installed {len(training_deps)} other training packages")
    else:
        # Fallback: install core deps directly
        torch_deps = ["torch>=2.1.0", "torchvision>=0.16.0"]
        _install_torch(torch_deps, gpu_detected)
        _run([*_pip(), "install", "-q",
              "numpy>=1.24", "pandas>=1.5", "matplotlib>=3.7",
              "scikit-learn>=1.3", "tqdm>=4.66", "Pillow>=10.0", "onnx>=1.16"])
        print("  ✅ Installed core training packages")

    # Install kaggle CLI
    print("  Ensuring kaggle CLI is available...")
    _run([*_pip(), "install", "-q", "kaggle"])
    print("  ✅ kaggle CLI ready")

    # Verify torch import + CUDA status
    verify_script = (
        "import torch; "
        "gpu = f' GPU={torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ''; "
        "print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}{gpu}')"
    )
    r = _run([_python(), "-c", verify_script], capture=True)
    print(f"  ✅ {r.stdout.strip()}")


# ============================================================
#  §2 — Kaggle Dataset Download
# ============================================================

def stage_download_dataset():
    _banner(2, "Kaggle Dataset Download")

    # 1. Check if dataset is already downloaded
    cattle_dir = os.path.join(DATA_RAW_DIR, "cattle")
    buffalo_dir = os.path.join(DATA_RAW_DIR, "buffalo")
    if os.path.isdir(cattle_dir) and os.path.isdir(buffalo_dir):
        n_cattle = len([d for d in os.listdir(cattle_dir)
                        if os.path.isdir(os.path.join(cattle_dir, d))])
        n_buffalo = len([d for d in os.listdir(buffalo_dir)
                         if os.path.isdir(os.path.join(buffalo_dir, d))])
        if n_cattle > 0 and n_buffalo > 0:
            print(f"  ✅ Dataset already present: {n_cattle} cattle breeds, "
                  f"{n_buffalo} buffalo breeds")
            return

    # 2. Setup Kaggle credentials — check existing, or prompt user
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    try:
        os.makedirs(kaggle_dir, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create {kaggle_dir}: {e}")

    # Check if credentials already exist and are valid
    creds_ok = False
    if os.path.exists(kaggle_json):
        try:
            with open(kaggle_json) as f:
                existing = json.load(f)
            if existing.get("username") and existing.get("key"):
                print(f"  ✅ Kaggle credentials found ({existing['username']})")
                creds_ok = True
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    if not creds_ok:
        print("  ┌─────────────────────────────────────────────┐")
        print("  │  Kaggle API credentials required             │")
        print("  │  Get your key at: kaggle.com/settings → API │")
        print("  └─────────────────────────────────────────────┘")
        print()
        try:
            username = input("  Enter your Kaggle username: ").strip()
            api_key = input("  Enter your Kaggle API key:  ").strip()
        except EOFError:
            raise RuntimeError("Input stream closed. Cannot prompt for credentials.")
            
        if not username or not api_key:
            raise RuntimeError(
                "Kaggle credentials are required to download the dataset. "
                "Get your API key at https://www.kaggle.com/settings → API")
        
        creds = {"username": username, "key": api_key}
        try:
            with open(kaggle_json, "w") as f:
                json.dump(creds, f)
            os.chmod(kaggle_json, 0o600)
            print(f"  ✅ Kaggle credentials saved to {kaggle_json}")
        except OSError as e:
            raise RuntimeError(f"Failed to save credentials to {kaggle_json}: {e}")

    # Download from Kaggle
    print(f"  Downloading dataset: {KAGGLE_DATASET}...")
    print("  ⏳ This may take several minutes depending on your connection...")
    download_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(download_dir, exist_ok=True)
    _run([_python(), "-m", "kaggle", "datasets", "download",
          "-d", KAGGLE_DATASET, "-p", download_dir])
    print("  ✅ Download complete")


# ============================================================
#  §3 — Unzip & Organize
# ============================================================

def stage_unzip_organize():
    _banner(3, "Unzip & Organize Dataset")

    cattle_dir = os.path.join(DATA_RAW_DIR, "cattle")
    buffalo_dir = os.path.join(DATA_RAW_DIR, "buffalo")

    # Check if already organized
    if os.path.isdir(cattle_dir) and os.path.isdir(buffalo_dir):
        n_cattle = len([d for d in os.listdir(cattle_dir)
                        if os.path.isdir(os.path.join(cattle_dir, d))])
        n_buffalo = len([d for d in os.listdir(buffalo_dir)
                         if os.path.isdir(os.path.join(buffalo_dir, d))])
        if n_cattle > 0 and n_buffalo > 0:
            print(f"  ✅ Dataset already organized: {n_cattle} cattle, "
                  f"{n_buffalo} buffalo breeds")
            return

    # Find the zip file
    zip_candidates = glob.glob(os.path.join(PROJECT_ROOT, "data", "*.zip"))
    if not zip_candidates:
        zip_candidates = glob.glob(os.path.join(PROJECT_ROOT, "*.zip"))
        zip_candidates = [z for z in zip_candidates
                          if "training_package" not in z and "colab" not in z]
    if not zip_candidates:
        raise RuntimeError(
            "No dataset zip found. Run without --skip-download or place "
            "the zip in data/")

    zip_path = zip_candidates[0]
    print(f"  Found archive: {os.path.basename(zip_path)} "
          f"({_file_size_mb(zip_path):.0f} MB)")

    # Extract
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    print("  ⏳ Extracting (this may take a while for large datasets)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        total = len(zf.namelist())
        for i, member in enumerate(zf.namelist()):
            zf.extract(member, DATA_RAW_DIR)
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"    Extracted {i+1}/{total} files...", end="\r")
    print(f"\n  ✅ Extracted {total} files to {DATA_RAW_DIR}")

    # Handle nested directories — the zip might extract into a subfolder
    # Look for cattle/ and buffalo/ directories anywhere under DATA_RAW_DIR
    for species in ("cattle", "buffalo"):
        target = os.path.join(DATA_RAW_DIR, species)
        if os.path.isdir(target):
            continue
        # Search for it in subdirectories
        for root, dirs, _ in os.walk(DATA_RAW_DIR):
            if species in dirs:
                src = os.path.join(root, species)
                if src != target:
                    print(f"  Moving {src} → {target}")
                    shutil.move(src, target)
                break

    # Validate
    for species, expected_dir in [("cattle", cattle_dir),
                                   ("buffalo", buffalo_dir)]:
        if not os.path.isdir(expected_dir):
            raise RuntimeError(f"{species}/ directory not found after extraction")
        breeds = [d for d in os.listdir(expected_dir)
                  if os.path.isdir(os.path.join(expected_dir, d))]
        n_images = sum(
            len([f for f in os.listdir(os.path.join(expected_dir, b))
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
            for b in breeds)
        print(f"  ✅ {species}: {len(breeds)} breeds, {n_images} images")

    # Clean up zip to save disk space
    print(f"  Removing archive to save disk space...")
    os.remove(zip_path)
    print(f"  ✅ Removed {os.path.basename(zip_path)}")


# ============================================================
#  §4 — Verify Architecture
# ============================================================

def stage_verify():
    _banner(4, "Architecture Verification")

    print("  Running src.verify (backbone loading + forward pass shapes)...")
    result = _run([_python(), "-m", "src.verify"], check=False)
    if result.returncode != 0:
        print("  ⚠️  Verification failed — check backbone weight files")
        print("  Checking for pretrained weights...")
        for name in ("efficientnet_lite2.pth", "efficientnet_lite4.pth"):
            path = os.path.join(PROJECT_ROOT, name)
            if os.path.exists(path):
                print(f"    ✅ {name} ({_file_size_mb(path):.1f} MB)")
            else:
                print(f"    ❌ {name} MISSING — model will train from scratch "
                      f"(much worse accuracy)")
        print("  Continuing anyway...")
    else:
        print("  ✅ Architecture verification passed")


# ============================================================
#  §5 — Data Splits
# ============================================================

def stage_data_splits(args):
    _banner(5, "Data Splitting & Preprocessing")

    # Training does its own split, but let's validate the data first
    print("  Validating dataset structure...")
    for species in ("cattle", "buffalo"):
        species_dir = os.path.join(DATA_RAW_DIR, species)
        if not os.path.isdir(species_dir):
            raise RuntimeError(f"Missing: {species_dir}")
        breeds = sorted([d for d in os.listdir(species_dir)
                         if os.path.isdir(os.path.join(species_dir, d))])
        total_imgs = 0
        min_imgs = float("inf")
        min_breed = ""
        for b in breeds:
            bd = os.path.join(species_dir, b)
            n = len([f for f in os.listdir(bd)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            total_imgs += n
            if n < min_imgs:
                min_imgs = n
                min_breed = b
        print(f"  {species}: {len(breeds)} breeds, {total_imgs} images, "
              f"smallest={min_breed} ({min_imgs} imgs)")

    mode = "half-data" if args.half_data else (
        "smoke-test" if args.smoke_test else "full")
    print(f"\n  Training will use: {mode} mode")
    print("  ✅ Dataset validated (splits generated during training)")


# ============================================================
#  §6 — Training
# ============================================================

def stage_train(args):
    _banner(6, "Model Training")

    # Build training command
    cmd = [_python(), "-m", "src.train",
           "--backbone", args.backbone,
           "--attention", args.attention]

    if args.smoke_test:
        cmd.append("--smoke-test")
        cmd.append("--skip-qat")
        print("  Mode: SMOKE TEST (tiny dataset, 1 epoch, fast sanity check)")
    elif args.half_data:
        cmd.append("--half-data")
        if not args.include_qat:
            cmd.append("--skip-qat")
        print("  Mode: HALF-DATA (50% images/breed, faster training)")
    elif getattr(args, 'quarter_data', False):
        cmd.append("--quarter-data")
        if not args.include_qat:
            cmd.append("--skip-qat")
        print("  Mode: QUARTER-DATA (25% images/breed, fastest local training)")
    else:
        if not args.include_qat:
            cmd.append("--skip-qat")
        print("  Mode: FULL DATA (all images, all phases)")

    if args.include_qat:
        print("  QAT: ENABLED (Phase 3)")
    else:
        print("  QAT: SKIPPED (use --include-qat to enable)")

    # Pass through optional overrides
    if args.phase1_epochs is not None:
        cmd.extend(["--phase1-epochs", str(args.phase1_epochs)])
    if args.phase2_epochs is not None:
        cmd.extend(["--phase2-epochs", str(args.phase2_epochs)])
    if args.phase3_epochs is not None:
        cmd.extend(["--phase3-epochs", str(args.phase3_epochs)])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])

    print(f"  Backbone: {args.backbone}")
    print(f"  Attention: {args.attention}")
    print(f"\n  Command: {' '.join(cmd)}\n")

    start = time.time()
    _run(cmd)
    elapsed = _elapsed(start)
    print(f"\n  ✅ Training completed in {elapsed}")


# ============================================================
#  §7 — Export
# ============================================================

def stage_export(args):
    _banner(7, "Model Export")

    checkpoint_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    best_ckpt = None

    # Find the best checkpoint (prefer phase3 if QAT was done, else phase2)
    for phase in ("phase3", "phase2", "phase1"):
        candidate = os.path.join(checkpoint_dir,
                                 f"{args.backbone}_{phase}_best.pt")
        if os.path.exists(candidate):
            best_ckpt = candidate
            break

    if not best_ckpt:
        print("  ⚠️  No checkpoint found — training may have failed")
        return

    print(f"  Best checkpoint: {os.path.basename(best_ckpt)} "
          f"({_file_size_mb(best_ckpt):.1f} MB)")

    # Portable export is auto-done by training; ensure it's there
    portable_dir = os.path.join(PROJECT_ROOT, "outputs", "export", "portable")
    portable_bundles = glob.glob(os.path.join(portable_dir, f"{args.backbone}_*"))
    if portable_bundles:
        print(f"  ✅ Portable bundle: {os.path.basename(portable_bundles[0])}")
    else:
        print("  Exporting portable bundle...")
        _run([_python(), "-m", "src.export",
              "--mode", "portable", "--backbone", args.backbone,
              "--checkpoint", best_ckpt], check=False)

    # Export ONNX
    print("\n  Exporting ONNX (FP32)...")
    _run([_python(), "-m", "src.export",
          "--mode", "onnx", "--backbone", args.backbone,
          "--checkpoint", best_ckpt], check=False)

    # Export INT8
    print("\n  Exporting INT8 (quantized)...")
    _run([_python(), "-m", "src.export",
          "--mode", "int8", "--backbone", args.backbone,
          "--checkpoint", best_ckpt], check=False)

    # Export float16
    print("\n  Exporting float16...")
    _run([_python(), "-m", "src.export",
          "--mode", "float16", "--backbone", args.backbone,
          "--checkpoint", best_ckpt], check=False)


# ============================================================
#  §8 — Summary
# ============================================================

def stage_summary(args, total_start):
    _banner(8, "Training Complete — Summary")

    elapsed = _elapsed(total_start)
    print(f"  Total time: {elapsed}\n")

    # Checkpoints
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    if os.path.isdir(ckpt_dir):
        print("  📦 Checkpoints:")
        for f in sorted(os.listdir(ckpt_dir)):
            if f.endswith(".pt"):
                path = os.path.join(ckpt_dir, f)
                print(f"    {f:40s} {_file_size_mb(path):8.1f} MB")

    # Exports
    export_dir = os.path.join(PROJECT_ROOT, "outputs", "export")
    if os.path.isdir(export_dir):
        print("\n  📤 Exports:")
        for root, dirs, files in os.walk(export_dir):
            for f in sorted(files):
                if f.endswith((".pt", ".onnx", ".json")):
                    path = os.path.join(root, f)
                    rel = os.path.relpath(path, export_dir)
                    print(f"    {rel:40s} {_file_size_mb(path):8.1f} MB")

    # Portable bundles
    portable_dir = os.path.join(PROJECT_ROOT, "outputs", "export", "portable")
    if os.path.isdir(portable_dir):
        bundles = [d for d in os.listdir(portable_dir)
                   if os.path.isdir(os.path.join(portable_dir, d))]
        if bundles:
            print(f"\n  🎯 Portable model bundles (self-contained, ready to deploy):")
            for b in sorted(bundles):
                bd = os.path.join(portable_dir, b)
                total_size = sum(
                    os.path.getsize(os.path.join(bd, f))
                    for f in os.listdir(bd) if os.path.isfile(os.path.join(bd, f))
                ) / (1024 * 1024)
                print(f"    {b}/  ({total_size:.1f} MB total)")

    print(f"\n  {'=' * 50}")
    print(f"  🎉 All done! Your model is ready.")
    print(f"  {'=' * 50}")

    print(f"\n  Quick test:")
    print(f"    python -m src.evaluate --backbone {args.backbone}")
    print(f"\n  Run webapp:")
    print(f"    python webapp/server.py  →  http://localhost:8000")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="🐄 Cattle & Buffalo Breed Classifier — "
                    "Fully Automated Local Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python local_train.py                    Full training (skip QAT)
  python local_train.py --half-data        Quick train with 50%% data
  python local_train.py --smoke-test       Tiny sanity check
  python local_train.py --half-data --include-qat   Half data + QAT
  python local_train.py --skip-download    Data already downloaded
  python local_train.py --skip-setup       Venv already ready
        """)

    # Data mode (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--half-data", action="store_true",
                            help="use 50%% of images per breed (faster)")
    data_group.add_argument("--quarter-data", action="store_true",
                            help="use 25%% of images per breed (fastest local)")
    data_group.add_argument("--smoke-test", action="store_true",
                            help="tiny dataset, 1 epoch per phase (sanity)")
    data_group.add_argument("--full-data", action="store_true",
                            help="use all images (default)")

    # Model config
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2",
                        help="backbone architecture (default: lite2)")
    parser.add_argument("--attention", choices=["cbam", "se"], default="cbam",
                        help="attention module (default: cbam)")

    # Training overrides
    parser.add_argument("--include-qat", action="store_true",
                        help="include QAT phase 3 (default: skipped)")
    parser.add_argument("--phase1-epochs", type=int, default=None,
                        help="override phase 1 epoch count")
    parser.add_argument("--phase2-epochs", type=int, default=None,
                        help="override phase 2 epoch count")
    parser.add_argument("--phase3-epochs", type=int, default=None,
                        help="override phase 3 epoch count")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="dataloader worker count")

    # Skip stages
    parser.add_argument("--skip-download", action="store_true",
                        help="skip Kaggle dataset download")
    parser.add_argument("--skip-setup", action="store_true",
                        help="skip venv creation / dependency install")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip architecture verification")
    parser.add_argument("--skip-export", action="store_true",
                        help="skip multi-format export after training")

    args = parser.parse_args()

    total_start = time.time()

    print("""
╔══════════════════════════════════════════════════════════╗
║  🐄 Cattle & Buffalo Breed Classifier                    ║
║  Fully Automated Local Training Pipeline                 ║
╚══════════════════════════════════════════════════════════╝
    """)

    mode = ("quarter-data (25% images/breed)" if getattr(args, 'quarter_data', False)
            else "half-data (50% images/breed)" if args.half_data
            else "smoke-test (tiny dataset)" if args.smoke_test
            else "full data")
    print(f"  Training mode: {mode}")
    print(f"  Backbone: {args.backbone}, Attention: {args.attention}")
    print(f"  QAT: {'enabled' if args.include_qat else 'skipped'}")
    print()

    try:
        # §0 — Prerequisites
        stage_prerequisites(args)

        # §1 — Environment
        if not args.skip_setup:
            stage_setup_env()
        else:
            print("\n  ⏭️  Skipping environment setup (--skip-setup)")

        # §2 — Dataset Download
        if not args.skip_download:
            stage_download_dataset()
        else:
            print("\n  ⏭️  Skipping dataset download (--skip-download)")

        # §3 — Unzip & Organize
        if not args.skip_download:
            stage_unzip_organize()
        else:
            # Still validate the data exists
            cattle = os.path.join(DATA_RAW_DIR, "cattle")
            buffalo = os.path.join(DATA_RAW_DIR, "buffalo")
            if not (os.path.isdir(cattle) and os.path.isdir(buffalo)):
                raise RuntimeError(
                    f"Dataset not found at {DATA_RAW_DIR}. "
                    f"Remove --skip-download to auto-download from Kaggle.")
            print("\n  ✅ Dataset already present")

        # §4 — Verify
        if not args.skip_verify:
            stage_verify()
        else:
            print("\n  ⏭️  Skipping verification (--skip-verify)")

        # §5 — Data Splits (validation only; actual splits done during training)
        stage_data_splits(args)

        # §6 — Training
        stage_train(args)

        # §7 — Export
        if not args.skip_export:
            stage_export(args)
        else:
            print("\n  ⏭️  Skipping export (--skip-export)")

        # §8 — Summary
        stage_summary(args, total_start)

        return 0

    except KeyboardInterrupt:
        print("\n\n  ⚠️  Interrupted by user")
        return 130
    except RuntimeError as e:
        print(f"\n  ❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n  ❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
