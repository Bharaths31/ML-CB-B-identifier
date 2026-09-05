#!/usr/bin/env python3
import argparse
import os
import platform
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements.txt")

MIN_PYTHON = (3, 10)


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


def install_requirements():
    py = venv_python()
    log("Upgrading pip")
    if subprocess.call([py, "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        return False
    log(f"Installing requirements from {REQUIREMENTS}")
    if subprocess.call([py, "-m", "pip", "install", "-r", REQUIREMENTS]) != 0:
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
    args = parser.parse_args()

    if not check_python():
        sys.exit(1)
    if not create_venv():
        sys.exit(1)
    if not install_requirements():
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