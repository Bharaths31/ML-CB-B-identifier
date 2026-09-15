#!/usr/bin/env python3
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements.txt")

MIN_PYTHON = (3, 10)

PYTORCH_CUDA_INDEXES = [
    # (min_driver_cuda, index_tag)
    (12, 4, "cu124"),   # CUDA 12.4+
    (12, 1, "cu121"),   # CUDA 12.1–12.3
    (11, 8, "cu118"),   # CUDA 11.8
]

PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/{tag}"


def log(msg):
    print(f"[setup] {msg}", flush=True)


def check_python():
    if sys.version_info < MIN_PYTHON:
        log(f"Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required, found "
            f"{platform.python_version()}")
        return False
    log(f"Using Python {platform.python_version()} ({sys.executable})")
    return True


def venv_python():
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def create_venv():
    if os.path.exists(venv_python()):
        log(f"Virtual environment already exists at {VENV_DIR}")
        return True
    log(f"Creating virtual environment at {VENV_DIR}")
    code = subprocess.call([sys.executable, "-m", "venv", VENV_DIR])
    return code == 0


def detect_nvidia_gpu():
    """Detect NVIDIA GPU hardware and CUDA driver version.

    Returns:
        dict: {"gpu": gpu_name, "cuda": (major, minor), "tag": index_tag} or None.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        log("No NVIDIA GPU detected (nvidia-smi not found)")
        return None

    try:
        r = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            log("nvidia-smi execution failed")
            return None

        lines = r.stdout.strip().split("\n")
        if not lines or not lines[0].strip():
            log("No GPU listed by nvidia-smi")
            return None

        parts = lines[0].split(",")
        gpu_name = parts[0].strip()

        # Parse CUDA Version from full nvidia-smi output
        r2 = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=10)
        cuda_ver = None
        for out_line in r2.stdout.split("\n"):
            if "CUDA Version" in out_line:
                m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out_line)
                if m:
                    cuda_ver = (int(m.group(1)), int(m.group(2)))
                break

        cuda_ver = cuda_ver or (12, 4)
        cuda_major, cuda_minor = cuda_ver

        chosen_tag = None
        for req_major, req_minor, tag in PYTORCH_CUDA_INDEXES:
            if (cuda_major, cuda_minor) >= (req_major, req_minor):
                chosen_tag = tag
                break

        if not chosen_tag:
            log(f"NVIDIA GPU detected ({gpu_name}), but CUDA {cuda_major}.{cuda_minor} is older than supported versions.")
            return None

        log(f"NVIDIA GPU detected: {gpu_name} (CUDA {cuda_major}.{cuda_minor} → {chosen_tag})")
        return {"gpu": gpu_name, "cuda": cuda_ver, "tag": chosen_tag}

    except Exception as e:
        log(f"GPU detection failed ({e})")
        return None


def test_venv_torch():
    """Test if PyTorch is installed and functioning in the venv.

    Returns:
        tuple: (working: bool, cuda_available: bool)
    """
    py = venv_python()
    if not os.path.exists(py):
        return False, False

    cmd = [
        py, "-c",
        "import torch; print(f'OK {torch.__version__} CUDA={torch.cuda.is_available()}')"
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and "OK" in r.stdout:
            cuda_avail = "CUDA=True" in r.stdout
            return True, cuda_avail
    except Exception:
        pass

    return False, False


def clean_broken_torch():
    """Uninstall broken or conflicting PyTorch / CUDA packages from the venv."""
    py = venv_python()
    log("Cleaning up existing torch/torchvision packages to ensure clean backend installation...")
    subprocess.call(
        [py, "-m", "pip", "uninstall", "-y", "torch", "torchvision",
         "cuda-toolkit", "cuda-bindings", "nvidia-cudnn-cu13", "nvidia-cudnn-cu12"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def install_requirements(force_cpu=False, force_gpu=False, cuda_tag=None, reinstall=False):
    py = venv_python()
    log("Upgrading pip")
    if subprocess.call([py, "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        return False

    # Determine target backend
    target_tag = None
    gpu_info = None

    if force_cpu:
        log("Forcing CPU mode as requested via CLI")
        target_tag = "cpu"
    elif cuda_tag:
        log(f"Using explicitly specified CUDA tag: {cuda_tag}")
        target_tag = cuda_tag
    else:
        gpu_info = detect_nvidia_gpu()
        if gpu_info:
            target_tag = gpu_info["tag"]
        elif force_gpu:
            log("Forcing GPU mode (defaulting to cu124 tag)")
            target_tag = "cu124"
        else:
            log("Selecting CPU mode")
            target_tag = "cpu"

    is_gpu_target = target_tag != "cpu"
    index_url = PYTORCH_INDEX_URL.format(tag=target_tag)

    # Check existing PyTorch health in venv
    torch_ok, cuda_avail = test_venv_torch()
    needs_torch_reinstall = reinstall or not torch_ok

    if torch_ok and not reinstall:
        # Check backend mismatch (e.g. want GPU but venv has CPU, or vice versa)
        if is_gpu_target and not cuda_avail:
            log("Existing venv PyTorch does not have CUDA enabled. Reinstalling PyTorch with CUDA...")
            needs_torch_reinstall = True
        elif not is_gpu_target and cuda_avail:
            log("Existing venv PyTorch has CUDA enabled, but CPU mode was requested. Reinstalling PyTorch for CPU...")
            needs_torch_reinstall = True

    # Separate requirements
    torch_deps = []
    other_deps = []

    if os.path.exists(REQUIREMENTS):
        with open(REQUIREMENTS) as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                pkg_name = line_str.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
                if pkg_name in ("torch", "torchvision"):
                    torch_deps.append(line_str)
                else:
                    other_deps.append(line_str)
    else:
        torch_deps = ["torch>=2.1.0", "torchvision>=0.16.0"]
        other_deps = ["numpy>=1.24", "pandas>=1.5", "matplotlib>=3.7",
                      "scikit-learn>=1.3", "tqdm>=4.66", "Pillow>=10.0",
                      "requests>=2.31", "onnx>=1.16"]

    if needs_torch_reinstall:
        clean_broken_torch()
        log(f"Installing PyTorch packages ({target_tag}) from {index_url}...")
        cmd = [py, "-m", "pip", "install", *torch_deps, "--index-url", index_url]
        if subprocess.call(cmd) != 0:
            log("FAILED to install PyTorch packages. Try manually installing from PyTorch index.")
            return False
    else:
        log("PyTorch installation in venv is healthy and matches target mode.")

    # Install remaining non-torch requirements
    if other_deps:
        log(f"Installing non-torch requirements from {REQUIREMENTS}...")
        cmd = [py, "-m", "pip", "install", *other_deps]
        if subprocess.call(cmd) != 0:
            log("FAILED to install non-torch requirements.")
            return False

    return True


def run_verification():
    py = venv_python()
    log("Running environment + model verification")
    code = subprocess.call([py, "-m", "src.verify"], cwd=PROJECT_ROOT)
    return code == 0


def main():
    parser = argparse.ArgumentParser(
        description="Create a venv, install requirements and verify the model.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the post-install verification step")
    parser.add_argument("--cpu", "--force-cpu", dest="force_cpu", action="store_true",
                        help="Force CPU mode PyTorch installation")
    parser.add_argument("--gpu", "--force-gpu", dest="force_gpu", action="store_true",
                        help="Force GPU mode PyTorch installation")
    parser.add_argument("--cuda-tag", type=str, default=None,
                        help="Explicit PyTorch CUDA tag (e.g. cu124, cu121, cu118, cpu)")
    parser.add_argument("--reinstall", action="store_true",
                        help="Force re-installation of PyTorch and requirements")
    args = parser.parse_args()

    if not check_python():
        sys.exit(1)
    if not create_venv():
        sys.exit(1)
    if not install_requirements(
        force_cpu=args.force_cpu,
        force_gpu=args.force_gpu,
        cuda_tag=args.cuda_tag,
        reinstall=args.reinstall
    ):
        log("FAILED to install requirements. See the error above.")
        sys.exit(1)
    if not args.no_verify and not run_verification():
        log("FAILED verification. See the error above.")
        sys.exit(1)

    log("Setup complete.")
    print()
    print("Activate the environment with:")
    print(f"    source {os.path.join(VENV_DIR, 'bin', 'activate')}")
    print()
    print("Quick start:")
    print("    python -m src.verify                 # model + weights sanity check")
    print("    python -m src.train --help           # three-phase training")
    print("    python -m src.evaluate --help        # per-head evaluation")
    print("    python -m src.export --help          # quantization / export")


if __name__ == "__main__":
    main()