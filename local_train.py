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

KAGGLE_DATASET_ALGSOCH = "algsoch/breed-cattle-buffalo"
KAGGLE_DATASET_CATTLE = "atharvadarpude/indian-cattle-image-dataset"
KAGGLE_DATASET_BUFFALO = "atharvadarpude/indian-buffalo-dataset"

# Spelling-variant merge map applied to breed folder names during merging.
# Empty by default — add entries only after confirming with audit_data.py, e.g.
#   BREED_ALIASES = {"amruthamahal": "amritmahal", "hallikaru": "hallikar",
#                    "malenadu_gidda": "malnad_gidda"}
BREED_ALIASES = {
    "holstein-friesian": "holstein_friesian",
    "luit_(swamp)": "luit"
}


DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements.txt")

# Every src module the training run imports. If any is missing the run crashes
# with "No module named 'src.X'" — usually a stale checkout that never got the
# new files (e.g. src/run_utils.py). Checked up front with a clear message.
REQUIRED_SRC_MODULES = (
    "__init__.py", "config.py", "run_utils.py", "data_pipeline.py", "model.py",
    "cbam.py", "efficientnet_lite.py", "train.py", "metrics.py", "export.py",
    "evaluate.py", "parity_check.py", "verify.py",
)

# Export formats to produce after training
EXPORT_FORMATS = ["portable", "onnx", "int8", "float16"]


# ============================================================
#  Helpers
# ============================================================

def _log_event(event, **fields):
    """Best-effort structured logging (never raises if the logger is absent)."""
    try:
        from src.run_logger import log_event
        return log_event(event, **fields)
    except Exception:
        return None


def _banner(stage, title):
    """Print a prominent stage banner."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  §{stage} — {title}")
    print(f"{'=' * width}\n")
    _log_event("stage_start", category="actions", stage=stage, title=title)


def _run(cmd, cwd=None, env=None, check=True, capture=False):
    """Run a subprocess with live output (unless capture=True)."""
    cwd = cwd or PROJECT_ROOT
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    # Give child processes the SAME execution id so their logs correlate.
    try:
        from src.run_logger import get_logger
        lg = get_logger()
        if lg is not None:
            merged_env.setdefault("RUN_EXEC_ID", lg.exec_id)
    except Exception:
        pass
    kwargs = dict(cwd=cwd, env=merged_env)
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    start = time.time()
    _log_event("command_start", category="actions", cmd=list(cmd), cwd=cwd)
    result = subprocess.run(cmd, **kwargs)
    _log_event("command_end", category="actions", cmd=list(cmd),
               returncode=result.returncode,
               duration_s=round(time.time() - start, 3))
    if check and result.returncode != 0:
        if capture:
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
        _log_event("command_failed", category="actions", level="error",
                   cmd=list(cmd), returncode=result.returncode)
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
    cpu_index_url = PYTORCH_INDEX_URL.format(tag="cpu")
    if gpu_info is None:
        # No GPU — install CPU version from PyTorch CPU index
        print("  Installing PyTorch (CPU)...")
        _run([*_pip(), "install", "-q", *torch_deps, "--index-url", cpu_index_url])
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
        _run([*_pip(), "install", "-q", *torch_deps, "--index-url", cpu_index_url])
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

    # Install training dependencies
    print("  Installing training dependencies...")
    if os.path.exists(REQUIREMENTS):
        with open(REQUIREMENTS) as f:
            lines = f.readlines()
        training_deps = []
        torch_deps = []  # torch/torchvision handled separately for CUDA
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg_name = line.split(">=")[0].split("==")[0].split("[")[0].strip()
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

def stage_download_dataset(args):
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
    download_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(download_dir, exist_ok=True)
    
    datasets_to_download = []
    if args.dataset_mode in ("algsoch", "both"):
        datasets_to_download.append(KAGGLE_DATASET_ALGSOCH)
    if args.dataset_mode in ("atharvadarpude", "both"):
        datasets_to_download.extend([KAGGLE_DATASET_CATTLE, KAGGLE_DATASET_BUFFALO])
    
    for ds in datasets_to_download:
        print(f"  Downloading dataset: {ds}...")
        _run([_python(), "-m", "kaggle", "datasets", "download", "-d", ds, "-p", download_dir])
    
    print("  ✅ Download complete")


# ============================================================
#  §3 — Unzip & Organize
# ============================================================

def normalize_breed_name(name):
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    for suffix in ("_cattle", "_buffalo", "_breed", "indian_"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
        if name.startswith(suffix):
            name = name[len(suffix):]
    return name

def merge_into_species_dir(source_base, target_species_dir, species_hint=None):
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

    breed_dirs = []
    for root, dirs, files in os.walk(scan_root):
        img_files = [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTS]
        if img_files and not dirs:
            breed_dirs.append(root)

    if not breed_dirs:
        for subdir in os.listdir(scan_root):
            subpath = os.path.join(scan_root, subdir)
            if os.path.isdir(subpath):
                for root, dirs, files in os.walk(subpath):
                    img_files = [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTS]
                    if img_files and not dirs:
                        breed_dirs.append(root)

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


INVENTORY_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_inventory")


def build_dataset_inventory(source_base, dataset_name, out_dir=None,
                            source_hint=None, include_images=True):
    """Write a JSON inventory of a dataset to ``<out_dir>/<dataset_name>.json``.

    For every breed folder it records:
      * the breed name,
      * whether it sits under a cattle or buffalo directory (``species``),
      * the number of images,
      * the resolution of EACH image (width x height),
      * a resolution histogram (e.g. ``{"640x480": 700}``).

    ``source_hint`` ("cattle"/"buffalo") is used when the source tree has no
    species sub-directory (e.g. the atharvadarpude datasets). Unreadable images
    are recorded under ``"errors"`` instead of aborting. This function only
    reads; it never modifies or deletes data. Returns the inventory dict.
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out_dir = out_dir or INVENTORY_DIR
    os.makedirs(out_dir, exist_ok=True)

    def _species_from_path(rel_parts):
        for p in rel_parts:
            pl = p.lower()
            if pl in ("cattle", "cow", "cows"):
                return "cattle"
            if pl in ("buffalo", "buff", "buffaloes"):
                return "buffalo"
        return source_hint or "unknown"

    breeds = []
    errors = []
    for root, dirs, files in os.walk(source_base):
        imgs = sorted(f for f in files
                      if os.path.splitext(f)[1].lower() in VALID_EXTS)
        if not imgs:
            continue
        rel = os.path.relpath(root, source_base)
        rel_parts = [] if rel == "." else rel.split(os.sep)
        species = _species_from_path(rel_parts)
        breed = normalize_breed_name(os.path.basename(root))
        entry = {
            "breed": breed,
            "species": species,
            "source_folder": rel.replace(os.sep, "/"),
            "count": 0,
            "resolutions": {},
            "images": [] if include_images else None,
        }
        for fname in imgs:
            fpath = os.path.join(root, fname)
            w = h = None
            if Image is not None:
                try:
                    with Image.open(fpath) as im:
                        w, h = im.size
                except Exception as exc:  # unreadable / corrupt
                    errors.append({"file": fpath, "error": str(exc)})
                    continue
            entry["count"] += 1
            key = f"{w}x{h}" if w and h else "unknown"
            entry["resolutions"][key] = entry["resolutions"].get(key, 0) + 1
            if include_images:
                entry["images"].append(
                    {"file": os.path.relpath(fpath, source_base).replace(os.sep, "/"),
                     "width": w, "height": h})
        breeds.append(entry)

    breeds.sort(key=lambda e: (e["species"], e["breed"]))
    by_species = {}
    for e in breeds:
        by_species[e["species"]] = by_species.get(e["species"], 0) + e["count"]

    inventory = {
        "dataset": dataset_name,
        "source_dir": os.path.abspath(source_base),
        "source_hint": source_hint,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "totals": {
            "breeds": len(breeds),
            "images": sum(e["count"] for e in breeds),
            "by_species": by_species,
            "unreadable": len(errors),
        },
        "breeds": breeds,
        "errors": errors,
    }
    out_path = os.path.join(out_dir, f"{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"  📋 Inventory: {dataset_name} -> {out_path} "
          f"({inventory['totals']['breeds']} breeds, "
          f"{inventory['totals']['images']} images"
          + (f", {len(errors)} unreadable" if errors else "") + ")")
    _log_event("dataset_inventory", category="data", dataset=dataset_name,
               out_path=out_path, totals=inventory["totals"],
               breeds={e["breed"]: {"species": e["species"], "count": e["count"]}
                       for e in breeds})
    return inventory


def stage_unzip_organize(args):
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
            if not getattr(args, "skip_inventory", False):
                build_dataset_inventory(DATA_RAW_DIR, "merged")
            return

    # Find the zip files
    zip_candidates = glob.glob(os.path.join(PROJECT_ROOT, "data", "*.zip"))
    if not zip_candidates:
        raise RuntimeError(
            "No dataset zip found. Run without --skip-download or place "
            "the zip in data/")
            
    tmp_dl = os.path.join(PROJECT_ROOT, "data", "tmp_extract")
    os.makedirs(tmp_dl, exist_ok=True)

    print("  ⏳ Extracting and merging datasets (this may take a while)...")
    for zip_path in zip_candidates:
        if "training_package" in zip_path or "colab" in zip_path:
            continue
        print(f"  Extracting archive: {os.path.basename(zip_path)}...")
        
        # Extract to a subfolder based on zip name
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        zip_extract_dir = os.path.join(tmp_dl, zip_name)
        os.makedirs(zip_extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_extract_dir)

        # Inventory the raw source dataset BEFORE merging, so the JSON records
        # which breeds live under cattle/ vs buffalo/ with per-image resolutions.
        if not getattr(args, "skip_inventory", False):
            if "breed-cattle-buffalo" in zip_path:
                build_dataset_inventory(zip_extract_dir, "algsoch")
            elif "indian-cattle" in zip_path:
                build_dataset_inventory(zip_extract_dir, "atharvadarpude_cattle",
                                        source_hint="cattle")
            elif "indian-buffalo" in zip_path:
                build_dataset_inventory(zip_extract_dir, "atharvadarpude_buffalo",
                                        source_hint="buffalo")
            else:
                build_dataset_inventory(zip_extract_dir, zip_name)

        # Merge logic based on filename
        if "breed-cattle-buffalo" in zip_path:
            merge_into_species_dir(zip_extract_dir, cattle_dir, species_hint="cattle")
            merge_into_species_dir(zip_extract_dir, buffalo_dir, species_hint="buffalo")
        elif "indian-cattle" in zip_path:
            merge_into_species_dir(zip_extract_dir, cattle_dir)
        elif "indian-buffalo" in zip_path:
            merge_into_species_dir(zip_extract_dir, buffalo_dir)
        else:
            # Fallback, try both
            merge_into_species_dir(zip_extract_dir, cattle_dir, species_hint="cattle")
            merge_into_species_dir(zip_extract_dir, buffalo_dir, species_hint="buffalo")

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

    # Inventory the final merged tree (data/raw/{cattle,buffalo}/<breed>/).
    if not getattr(args, "skip_inventory", False):
        build_dataset_inventory(DATA_RAW_DIR, "merged")

    # Clean up zip to save disk space
    print(f"  Removing archives and temp files to save disk space...")
    shutil.rmtree(tmp_dl)
    for zip_path in zip_candidates:
        if "training_package" not in zip_path and "colab" not in zip_path:
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

    # Preflight: make sure every src module exists before spawning training.
    missing = [m for m in REQUIRED_SRC_MODULES
               if not os.path.exists(os.path.join(PROJECT_ROOT, "src", m))]
    if missing:
        raise RuntimeError(
            "Missing src module(s): " + ", ".join(f"src/{m}" for m in missing)
            + "\n  These are new/updated files that must be synced (git pull / "
              "copy the whole src/ folder). Training cannot start without them.")

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

    # Augmentation is OFF unless explicitly requested.
    if args.augment_all:
        cmd.append("--augment-all")
    else:
        for flag, attr in (("--mix", "mix"), ("--flip", "flip"),
                           ("--color-jitter", "color_jitter"),
                           ("--randaugment", "randaugment"), ("--rrc", "rrc")):
            if getattr(args, attr, False):
                cmd.append(flag)
    
    if getattr(args, "augment_preset", None):
        cmd.extend(["--augment-preset", args.augment_preset])

    aug_on = [a for a in ("mix", "flip", "color_jitter", "randaugment", "rrc")
              if args.augment_all or getattr(args, a, False)]
    preset_str = f" (preset={args.augment_preset})" if getattr(args, "augment_preset", None) else ""
    print(f"  Augmentation: {', '.join(aug_on) if aug_on else 'NONE (default)'}{preset_str}")

    # Imbalance: sampler only unless --logit-adjust is passed.
    if args.logit_adjust:
        cmd.append("--logit-adjust")
        if args.logit_adjust_prior:
            cmd.extend(["--logit-adjust-prior", args.logit_adjust_prior])
        print("  Imbalance: sampler + logit adjustment")
    else:
        print("  Imbalance: effective-number sampler ONLY (default)")

    if args.run_tag:
        cmd.extend(["--run-tag", args.run_tag])

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

    # Find the newest checkpoint. Phase 2 (EMA) is preferred: phase-3 QAT
    # measurably degrades accuracy and is opt-in only. Checkpoints are
    # timestamped per run, so pick the latest matching each phase.
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from src.run_utils import find_latest_checkpoint
    for phase in ("phase2", "phase3", "phase1"):
        candidate = find_latest_checkpoint(checkpoint_dir, args.backbone, phase)
        if candidate:
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

    # Export INT8 via converter-side PTQ (no QAT)
    print("\n  Exporting INT8 (ONNX Runtime Mobile, PTQ)...")
    _run([_python(), "-m", "src.export",
          "--mode", "onnx-int8", "--backbone", args.backbone,
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
    print(f"\n  Test visually:")
    print(f"    python test_model.py")


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
  python local_train.py                    Full training, NO augmentation (default)
  python local_train.py --half-data        Quick train with 50%% data
  python local_train.py --smoke-test       Tiny sanity check
  python local_train.py --mix              Enable CutMix/MixUp (opt-in)
  python local_train.py --augment-all      Enable flip+color-jitter+randaugment+rrc+mix
  python local_train.py --logit-adjust     Add logit adjustment on top of the sampler
  python local_train.py --run-tag exp1     Name this run's timestamped outputs
  python local_train.py --skip-download    Data already downloaded
  python local_train.py --skip-setup       Venv already ready

Augmentation and CutMix/MixUp are OFF by default: they measurably hurt
fine-grained breed identification. Enable only the specific flags you want.
Outputs are timestamped per run and never overwrite previous results.
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

    # Augmentation (ALL OFF by default — opt-in per run)
    aug = parser.add_argument_group("augmentation (off by default)")
    aug.add_argument("--mix", action="store_true",
                     help="enable CutMix/MixUp batch mixing")
    aug.add_argument("--flip", action="store_true",
                     help="enable RandomHorizontalFlip")
    aug.add_argument("--color-jitter", action="store_true",
                     help="enable ColorJitter")
    aug.add_argument("--randaugment", action="store_true",
                     help="enable RandAugment")
    aug.add_argument("--rrc", action="store_true",
                     help="enable RandomResizedCrop")
    aug.add_argument("--augment-all", action="store_true",
                     help="enable flip + color-jitter + randaugment + rrc + mix")
    aug.add_argument("--augment-preset", choices=["none", "light"],
                     help="apply pre-defined augmentation preset (e.g. light)")

    # Imbalance mechanism (single mechanism by default)
    imb = parser.add_argument_group("imbalance")
    imb.add_argument("--logit-adjust", action="store_true",
                     help="enable logit adjustment ON TOP of the sampler "
                          "(off by default; double-corrects)")
    imb.add_argument("--logit-adjust-prior", choices=["sampled", "raw"],
                     default=None,
                     help="prior source when --logit-adjust is used")

    parser.add_argument("--run-tag", default=None,
                        help="run id for timestamped outputs "
                             "(default: current time DD-MM-YYYY-HH-MM)")
    parser.add_argument("--exec-id", default=None,
                        help="execution-log folder name under logs/ "
                             "(default: auto YYYYmmdd-HHMMSS-xxxx)")

    # Skip stages
    parser.add_argument("--skip-download", action="store_true",
                        help="skip Kaggle dataset download")
    parser.add_argument("--skip-setup", action="store_true",
                        help="skip venv creation / dependency install")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip architecture verification")
    parser.add_argument("--skip-export", action="store_true",
                        help="skip multi-format export after training")
    parser.add_argument("--skip-inventory", action="store_true",
                        help="skip writing data/dataset_inventory/*.json "
                             "(per-dataset breed/image/resolution logs)")
    
    # Dataset mode
    parser.add_argument("--dataset-mode", choices=["algsoch", "atharvadarpude", "both"], default="both",
                        help="select the dataset sources to use for training (default: both)")

    args = parser.parse_args()

    # --- execution logger (single folder logs/<exec_id>/) ---
    try:
        from src.run_logger import init_run_logger, log_event
        init_run_logger(exec_id=getattr(args, "exec_id", None), module="local_train")
        log_event("cli_args", category="actions", **vars(args))
    except Exception:
        pass

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
    aug_on = [a for a in ("mix", "flip", "color_jitter", "randaugment", "rrc")
              if args.augment_all or getattr(args, a, False)]
    print(f"  Training mode: {mode}")
    print(f"  Backbone: {args.backbone}, Attention: {args.attention}")
    print(f"  QAT: {'enabled' if args.include_qat else 'skipped'}")
    print(f"  Augmentation: {', '.join(aug_on) if aug_on else 'NONE (default)'}")
    print(f"  Imbalance: {'sampler + logit adjustment' if args.logit_adjust else 'sampler ONLY (default)'}")
    print(f"  Run id: {args.run_tag or '(auto-incremented tag)'}")
    print()

    try:
        # §0.5 — Organize old outputs
        import shutil
        import re
        output_root = os.path.join(PROJECT_ROOT, "outputs")
        for sub in ["checkpoints", "export", "metrics"]:
            cat_dir = os.path.join(output_root, sub)
            if not os.path.isdir(cat_dir): continue
            for item in os.listdir(cat_dir):
                if item == "portable" or os.path.isdir(os.path.join(cat_dir, item)): continue
                path = os.path.join(cat_dir, item)
                if os.path.isfile(path):
                    match = re.search(r'_([vV]\d+|[\d-]{10,})\.', item)
                    if match:
                        tag = match.group(1).lower()
                        target_dir = os.path.join(output_root, tag, sub)
                        os.makedirs(target_dir, exist_ok=True)
                        shutil.move(path, os.path.join(target_dir, item))

        # §0 — Prerequisites
        stage_prerequisites(args)

        # §1 — Environment
        if not args.skip_setup:
            stage_setup_env()
        else:
            print("\n  ⏭️  Skipping environment setup (--skip-setup)")

        # §2 — Dataset Download
        if not args.skip_download:
            stage_download_dataset(args)
        else:
            print("\n  ⏭️  Skipping dataset download (--skip-download)")

        # §3 — Unzip & Organize
        if not args.skip_download:
            stage_unzip_organize(args)
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
