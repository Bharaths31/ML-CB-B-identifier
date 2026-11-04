#!/usr/bin/env python3
"""Smart, idempotent environment setup for the BreedID project.

The installer is intentionally safe to re-run on an existing machine:

* creates the venv only when missing;
* reuses the existing venv and its interpreter;
* inspects installed package versions and installs **only** missing/outdated
  requirements;
* re-installs PyTorch only when it is missing, broken, outdated, or the
  requested CPU/GPU backend does not match;
* handles missing pip, network failures, permissions, timeouts and platform
  differences with actionable messages instead of a traceback.

Linux:   bash setup.sh
Windows: powershell -ExecutionPolicy Bypass -File setup.ps1

The same run also installs the Node dependencies for ``breed-classifier-react``
and ``vet-portal`` and syncs the built web assets into the Capacitor Android
project, so a fresh checkout is ready for both the web and Android builds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements.txt")
STATUS_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "status_report.py")

MIN_PYTHON = (3, 10)

PYTORCH_CUDA_INDEXES = [
    # (min_driver_cuda, index_tag)
    (12, 4, "cu124"),
    (12, 1, "cu121"),
    (11, 8, "cu118"),
]
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/{tag}"

# --- Node / frontend prerequisites -------------------------------------------
# Vite 5 needs Node 18+; Capacitor 8 (@capacitor/cli 8.6) declares Node 22+.
MIN_NODE_MAJOR = 18
MIN_NODE_ANDROID_MAJOR = 22
NODE_PROJECTS = (
    ("breed-classifier-react", "farmer web app + Capacitor Android shell"),
    ("vet-portal", "vet portal web app"),
)
ANDROID_PROJECT = os.path.join(PROJECT_ROOT, "breed-classifier-react", "android")


class SetupError(RuntimeError):
    """A recoverable setup failure with a user-facing hint."""


@dataclass
class Requirement:
    raw: str
    name: str
    operator: str
    version: str
    constraints: tuple[tuple[str, str], ...] = ()

    @property
    def normalized(self) -> str:
        return normalize_name(self.name)


def log(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def normalize_name(name: str) -> str:
    """PEP 503-ish distribution-name normalization."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def venv_python() -> str:
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def venv_activate_hint() -> str:
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "Activate.ps1")
    return os.path.join(VENV_DIR, "bin", "activate")


def run_command(command: list[str], *, timeout: int | None = None,
                capture: bool = False,
                cwd: str | None = None) -> subprocess.CompletedProcess | None:
    """Run a command, converting OS/network/timeout failures into logs."""
    try:
        return subprocess.run(
            command,
            cwd=cwd or PROJECT_ROOT,
            capture_output=capture,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        log(f"Command not found: {command[0]}")
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout}s: {' '.join(command[:3])} …")
    except PermissionError:
        log(f"Permission denied while running: {' '.join(command[:3])} …")
    except OSError as exc:
        log(f"OS error while running {' '.join(command[:3])} …: {exc}")
    return None


def check_python() -> bool:
    if sys.version_info < MIN_PYTHON:
        log(f"Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required, found "
            f"{platform.python_version()} ({sys.executable})")
        return False
    log(f"Using Python {platform.python_version()} ({sys.executable})")
    return True


def create_venv() -> bool:
    py = venv_python()
    if os.path.exists(py):
        result = run_command([py, "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
                             capture=True, timeout=15)
        if result and result.returncode == 0:
            version = result.stdout.strip()
            log(f"Virtual environment already exists at {VENV_DIR} (Python {version})")
            return True
        log("Existing virtual environment could not be used. Attempting repair...")

    log(f"Creating virtual environment at {VENV_DIR}")
    result = run_command([sys.executable, "-m", "venv", VENV_DIR], timeout=180)
    if result is None or result.returncode != 0:
        log("Could not create the virtual environment. Check write permissions for the project folder.")
        return False
    return os.path.exists(venv_python())


def ensure_pip(py: str, upgrade: bool = False) -> bool:
    """Ensure pip exists; upgrade only when explicitly requested or clearly old."""
    check = run_command([py, "-m", "pip", "--version"], capture=True, timeout=30)
    if check is None or check.returncode != 0:
        log("pip is missing from the virtual environment; bootstrapping with ensurepip...")
        boot = run_command([py, "-m", "ensurepip", "--upgrade"], timeout=180)
        if boot is None or boot.returncode != 0:
            log("Could not bootstrap pip. Recreate the venv or install pip manually.")
            return False
        check = run_command([py, "-m", "pip", "--version"], capture=True, timeout=30)
        if check is None or check.returncode != 0:
            return False

    if upgrade:
        log("Upgrading pip/setuptools/wheel (requested by --reinstall)")
        result = run_command([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel",
                              "--disable-pip-version-check"], timeout=180)
        return result is not None and result.returncode == 0

    # Upgrade very old pip versions that cannot resolve modern wheels.
    match = re.search(r"pip\s+(\d+)\.(\d+)", check.stdout or "")
    if match and (int(match.group(1)), int(match.group(2))) < (23, 0):
        log("pip is old; upgrading pip/setuptools/wheel...")
        result = run_command([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel",
                              "--disable-pip-version-check"], timeout=180)
        if result is None or result.returncode != 0:
            log("pip upgrade failed; continuing with the existing pip.")
    return True


def detect_nvidia_gpu() -> dict | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        log("No NVIDIA GPU detected (nvidia-smi not found).")
        return None

    result = run_command(
        [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
        capture=True, timeout=10,
    )
    if result is None or result.returncode != 0 or not result.stdout.strip():
        log("nvidia-smi could not report a GPU; using CPU mode.")
        return None

    gpu_name = result.stdout.strip().splitlines()[0].split(",")[0].strip()
    full = run_command([nvidia_smi], capture=True, timeout=10)
    cuda_ver = (12, 4)
    if full and full.stdout:
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", full.stdout)
        if match:
            cuda_ver = (int(match.group(1)), int(match.group(2)))

    chosen_tag = None
    for req_major, req_minor, tag in PYTORCH_CUDA_INDEXES:
        if cuda_ver >= (req_major, req_minor):
            chosen_tag = tag
            break

    if not chosen_tag:
        log(f"NVIDIA GPU detected ({gpu_name}), but CUDA {cuda_ver[0]}.{cuda_ver[1]} is older than supported versions.")
        return None

    log(f"NVIDIA GPU detected: {gpu_name} (CUDA {cuda_ver[0]}.{cuda_ver[1]} → {chosen_tag})")
    return {"gpu": gpu_name, "cuda": cuda_ver, "tag": chosen_tag}


def version_key(value: str) -> tuple[int, ...]:
    value = str(value).split("+", 1)[0]
    numbers = re.findall(r"\d+", value)
    return tuple(int(part) for part in numbers) if numbers else (0,)


def satisfies(installed: str, operator: str, required: str) -> bool:
    installed_key = version_key(installed)
    required_key = version_key(required)
    if operator in ("", "==", "==="):
        return installed_key == required_key
    if operator == ">=":
        return installed_key >= required_key
    if operator == "<=":
        return installed_key <= required_key
    if operator == ">":
        return installed_key > required_key
    if operator == "<":
        return installed_key < required_key
    if operator == "!=":
        return installed_key != required_key
    if operator == "~=":
        # Compatible release: same prefix, upper bound on the second-last part.
        parts = required.split(".")
        if len(parts) < 2:
            return installed_key >= required_key
        upper = list(parts[:-2]) + [str(int(parts[-2]) + 1)]
        return installed_key >= required_key and installed_key < version_key(".".join(upper))
    return True


def parse_requirement(line: str) -> Requirement | None:
    raw = line.split("#", 1)[0].strip()
    if not raw or raw.startswith("-"):
        return None
    raw = raw.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(.*)$", raw)
    if not match:
        return None
    name, rest = match.group(1), match.group(2).strip()
    constraints = tuple(re.findall(r"(===|==|>=|<=|~=|!=|>|<)\s*([A-Za-z0-9_.+!-]+)", rest))
    if constraints:
        operator, version = constraints[0]
        return Requirement(raw=raw, name=name, operator=operator,
                           version=version, constraints=constraints)
    return Requirement(raw=raw, name=name, operator="", version="")


def read_requirements() -> list[Requirement]:
    if not os.path.exists(REQUIREMENTS):
        log(f"requirements.txt not found at {REQUIREMENTS}; using built-in defaults.")
        defaults = [
            "torch>=2.1.0", "torchvision>=0.16.0", "numpy>=1.24", "pandas>=1.5",
            "matplotlib>=3.7", "scikit-learn>=1.3", "tqdm>=4.66", "Pillow>=10.0",
            "requests>=2.31", "Flask>=3.0", "flask-cors>=4.0",
            "Flask-SocketIO>=5.3", "python-socketio>=5.10", "Werkzeug>=3.0",
            "onnx>=1.16", "onnxruntime>=1.16", "odfpy>=1.4.1", "qrcode>=8.0",
            "opencv-python-headless>=4.9,<5", "transformers>=4.40",
        ]
        return [req for req in (parse_requirement(item) for item in defaults) if req]

    requirements = []
    with open(REQUIREMENTS, encoding="utf-8") as handle:
        for line in handle:
            requirement = parse_requirement(line)
            if requirement:
                requirements.append(requirement)
    return requirements


def installed_packages(py: str) -> dict[str, str]:
    result = run_command(
        [py, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        capture=True, timeout=60,
    )
    if result is None or result.returncode != 0:
        return {}
    try:
        data = json.loads(result.stdout or "[]")
        return {normalize_name(item["name"]): str(item.get("version", "")) for item in data}
    except (ValueError, KeyError, TypeError):
        return {}


def torch_health(py: str) -> dict:
    code = (
        "import json, torch\n"
        "print(json.dumps({'ok': True, 'version': torch.__version__, "
        "'cuda': bool(torch.cuda.is_available())}))"
    )
    result = run_command([py, "-c", code], capture=True, timeout=25)
    if result is None or result.returncode != 0:
        return {"ok": False, "version": "", "cuda": False, "error": (result.stderr.strip() if result and result.stderr else "")}
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
        data["ok"] = True
        return data
    except (ValueError, IndexError):
        return {"ok": False, "version": "", "cuda": False, "error": "Could not parse torch status"}


def filter_conflicting_requirements(requirements: Iterable[Requirement],
                                    installed: dict[str, str]) -> list[Requirement]:
    """Drop requirements that would fight an equivalent installed package.

    ``opencv-python`` and ``opencv-python-headless`` ship the same ``cv2``
    module; installing one over the other can corrupt the install.
    """
    filtered = []
    for requirement in requirements:
        if (requirement.normalized == "opencv-python-headless"
                and "opencv-python" in installed):
            log("opencv-python is already installed; skipping opencv-python-headless "
                "(both provide cv2).")
            continue
        filtered.append(requirement)
    return filtered


def package_needs_update(requirement: Requirement, installed: dict[str, str]) -> bool:
    current = installed.get(requirement.normalized)
    if current is None:
        return True
    constraints = requirement.constraints or (
        ((requirement.operator, requirement.version),)
        if requirement.operator and requirement.version else ()
    )
    if not constraints:
        return False
    return not all(satisfies(current, op, ver) for op, ver in constraints)


def install_packages(py: str, requirements: Iterable[Requirement], *,
                     index_url: str | None = None) -> bool:
    packages = [req.raw for req in requirements]
    if not packages:
        return True
    command = [py, "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed"]
    if index_url:
        command += ["--index-url", index_url]
    command += ["--disable-pip-version-check", *packages]
    log(f"Installing/updating {len(packages)} package(s): {', '.join(packages)}")
    result = run_command(command, timeout=1800)
    if result is None or result.returncode != 0:
        log("Package installation failed.")
        log("Hints: check your internet/proxy, run setup again, or use --cpu for CPU-only PyTorch.")
        return False
    return True


def clean_broken_torch(py: str) -> None:
    log("Removing unusable torch/torchvision packages before reinstalling...")
    run_command([py, "-m", "pip", "uninstall", "-y", "torch", "torchvision",
                 "cuda-toolkit", "cuda-bindings", "nvidia-cudnn-cu13", "nvidia-cudnn-cu12"],
                capture=True, timeout=180)


def install_or_update(force_cpu: bool, force_gpu: bool, cuda_tag: str | None,
                      reinstall: bool, check_only: bool = False) -> bool:
    py = venv_python()
    if not os.path.exists(py):
        log("Virtual environment interpreter is missing; cannot install packages.")
        return False

    requirements = read_requirements()
    if not requirements:
        log("No requirements found.")
        return False

    torch_reqs = [req for req in requirements if req.name in ("torch", "torchvision")]
    other_reqs = [req for req in requirements if req.name not in ("torch", "torchvision")]
    installed = installed_packages(py)
    health = torch_health(py)

    # Determine desired backend.
    if force_cpu:
        target_tag = "cpu"
        log("Requested backend: CPU (--cpu)")
    elif cuda_tag:
        target_tag = cuda_tag
        log(f"Requested backend: explicit {cuda_tag}")
    else:
        gpu = detect_nvidia_gpu()
        if gpu:
            target_tag = gpu["tag"]
        elif force_gpu:
            target_tag = "cu124"
            log("Requested GPU backend; defaulting to cu124")
        else:
            target_tag = "cpu"
            log("No explicit backend requested and no usable NVIDIA CUDA GPU detected; using CPU.")

    is_gpu_target = target_tag != "cpu"

    # Torch must be installed/reinstalled when missing, broken, outdated, or
    # when the existing CPU/CUDA backend does not match the request.
    torch_outdated = any(package_needs_update(req, installed) for req in torch_reqs)
    backend_mismatch = health.get("ok") and (
        (is_gpu_target and not health.get("cuda")) or
        (not is_gpu_target and health.get("cuda"))
    )
    needs_torch = reinstall or not health.get("ok") or torch_outdated or backend_mismatch

    if check_only:
        log("Check mode only; no packages will be changed.")
        if needs_torch:
            log(f"PyTorch needs attention (installed={health.get('version') or 'missing'}, target={target_tag}).")
        else:
            log(f"PyTorch is healthy ({health.get('version')}, CUDA={health.get('cuda')}).")
        pending = [req.raw for req in filter_conflicting_requirements(other_reqs, installed)
                   if package_needs_update(req, installed)]
        log("Requirements already satisfied." if not pending else f"Missing/outdated: {', '.join(pending)}")
        return True

    if needs_torch:
        if health.get("ok") or reinstall:
            clean_broken_torch(py)
        index_url = PYTORCH_INDEX_URL.format(tag=target_tag)
        if not install_packages(py, torch_reqs, index_url=index_url):
            return False
        health = torch_health(py)
        if not health.get("ok"):
            log("PyTorch still cannot be imported after installation.")
            return False
        log(f"PyTorch ready: {health.get('version')} (CUDA={health.get('cuda')})")
    else:
        log(f"PyTorch is already healthy: {health.get('version')} (CUDA={health.get('cuda')})")

    # Refresh inventory after the possible torch install.
    installed = installed_packages(py)
    pending = [req for req in filter_conflicting_requirements(other_reqs, installed)
               if package_needs_update(req, installed)]
    if pending:
        log(f"Updating only missing/outdated packages ({len(pending)} of {len(other_reqs)}).")
        if not install_packages(py, pending):
            return False
    else:
        log("All non-PyTorch requirements are already satisfied.")

    return True


def run_verification() -> bool:
    py = venv_python()
    log("Running environment + model verification")
    result = run_command([py, "-m", "src.verify"], timeout=900)
    return result is not None and result.returncode == 0


def run_status_report() -> None:
    py = venv_python() if os.path.exists(venv_python()) else sys.executable
    if not os.path.exists(STATUS_SCRIPT):
        log(f"Status report script not found at {STATUS_SCRIPT}; skipping.")
        return
    log("Project status report:")
    run_command([py, STATUS_SCRIPT], timeout=120)


def node_install_reason(project_dir: str) -> str | None:
    """Return why a Node project needs ``npm install``, or None if current."""
    package_json = os.path.join(project_dir, "package.json")
    if not os.path.isfile(package_json):
        return "package.json missing"
    node_modules = os.path.join(project_dir, "node_modules")
    if not os.path.isdir(node_modules):
        return "node_modules missing"
    marker = os.path.join(node_modules, ".package-lock.json")
    if not os.path.exists(marker):
        return "npm install marker missing"
    newest_manifest = os.path.getmtime(package_json)
    lock = os.path.join(project_dir, "package-lock.json")
    if os.path.exists(lock):
        newest_manifest = max(newest_manifest, os.path.getmtime(lock))
    if newest_manifest > os.path.getmtime(marker):
        return "package.json/package-lock.json changed since the last install"
    return None


def install_node_project(npm: str, project_dir: str, reason: str) -> bool:
    """Install/update one Node project without touching a healthy tree."""
    has_lock = os.path.exists(os.path.join(project_dir, "package-lock.json"))
    fresh = reason == "node_modules missing" and has_lock
    command = [npm, "ci" if fresh else "install",
               "--no-audit", "--no-fund", "--loglevel=error"]
    name = os.path.basename(project_dir)
    log(f"{'Installing' if fresh else 'Updating'} Node dependencies in {name} ({reason})")
    result = run_command(command, cwd=project_dir, timeout=1800)
    if result is None or result.returncode != 0:
        log(f"npm failed in {name}; run `npm install` there manually for details.")
        return False
    return True


def build_android_apk() -> bool:
    """Compile the Capacitor Android debug APK when the SDK/JDK are present."""
    android_dir = ANDROID_PROJECT
    gradlew = os.path.join(android_dir, "gradlew.bat" if sys.platform == "win32" else "gradlew")
    if not os.path.exists(gradlew):
        log("gradlew not found; skipping the Android APK build.")
        return False
    sdk = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if not sdk or not os.environ.get("JAVA_HOME"):
        log("ANDROID_HOME/ANDROID_SDK_ROOT and JAVA_HOME (JDK 21) are required "
            "to compile the APK; web assets were synced but the APK was not built.")
        return True
    log("Building the Android debug APK (gradlew assembleDebug)...")
    command = [gradlew, "assembleDebug"] if sys.platform == "win32" else ["sh", gradlew, "assembleDebug"]
    result = run_command(command, cwd=android_dir, timeout=3600)
    if result is None or result.returncode != 0:
        log("Android build failed; open breed-classifier-react/android in Android Studio "
            "or run gradlew assembleDebug there for details.")
        return False
    apk = os.path.join(android_dir, "app", "build", "outputs", "apk", "debug", "app-debug.apk")
    if os.path.exists(apk):
        log(f"Android APK ready: {apk}")
    return True


def sync_android(npm: str, build_apk: bool, force_build: bool = False) -> bool:
    """Build the web bundle (if missing/stale) and copy it into Android."""
    project_dir = os.path.join(PROJECT_ROOT, "breed-classifier-react")
    if force_build or not os.path.isdir(os.path.join(project_dir, "dist")):
        log("Building the web app before the Capacitor sync...")
        result = run_command([npm, "run", "build"], cwd=project_dir, timeout=900)
        if result is None or result.returncode != 0:
            log("Web build failed; skipping the Capacitor sync.")
            return False
    npx = shutil.which("npx")
    if not npx:
        log("npx not found; skipping the Capacitor sync.")
        return False
    log("Syncing web assets into the Android project (npx cap sync android)...")
    result = run_command([npx, "cap", "sync", "android"], cwd=project_dir, timeout=900)
    if result is None or result.returncode != 0:
        log("Capacitor sync failed; the Android project was left intact.")
        return False
    if build_apk:
        return build_android_apk()
    log("Android web assets synced (pass --android-build to compile the debug APK).")
    return True


def setup_frontend(check_only: bool = False, skip_android: bool = False,
                   android_build: bool = False) -> bool:
    """Install Node dependencies and refresh the Capacitor Android project.

    Missing Node.js is a warning, not a failure: the Python app works without
    the web apps, and a fresh checkout may not have Node installed yet.
    """
    node = shutil.which("node")
    npm = shutil.which("npm")
    if not node or not npm:
        log("Node.js/npm not found; skipping frontend/Android dependencies.")
        log("Install Node.js 22+ (https://nodejs.org) and re-run setup to prepare the web apps.")
        return True

    node_version = command_version(node, "--version")
    npm_version = command_version(npm, "--version")
    node_major = major_version(node_version)
    log(f"Node.js {node_version or '?'} / npm {npm_version or '?'}")
    if node_major is not None and node_major < MIN_NODE_MAJOR:
        log(f"Node.js >= {MIN_NODE_MAJOR} is required by Vite {npm_version or ''}; "
            "upgrade Node.js and re-run.")
        return False

    ok = True
    changed = False
    for name, description in NODE_PROJECTS:
        project_dir = os.path.join(PROJECT_ROOT, name)
        if not os.path.isfile(os.path.join(project_dir, "package.json")):
            log(f"Skipping {name}: package.json not found ({description}).")
            continue
        reason = node_install_reason(project_dir)
        if check_only:
            log(f"{name}: {'dependencies up to date' if reason is None else reason}")
        elif reason is not None:
            changed = True
            if not install_node_project(npm, project_dir, reason):
                ok = False
                break

    if not ok or skip_android:
        return ok
    if not os.path.isdir(ANDROID_PROJECT):
        log("Capacitor Android project not found; skipping the Android sync.")
        return True
    if node_major is not None and node_major < MIN_NODE_ANDROID_MAJOR:
        log(f"Capacitor 8 needs Node.js >= {MIN_NODE_ANDROID_MAJOR}; skipping the Android "
            "sync (upgrade Node.js to build the APK).")
        return True
    if check_only:
        log("Android: Capacitor project present; run setup without --check to sync "
            "the built web assets.")
        return True
    return sync_android(npm, android_build, force_build=changed)


def command_version(command: str, *args: str) -> str | None:
    result = run_command([command, *args], capture=True, timeout=30)
    if result is None or result.returncode != 0:
        return None
    lines = (result.stdout or "").strip().splitlines()
    return lines[0] if lines else None


def major_version(version: str | None) -> int | None:
    match = re.match(r"v?(\d+)", version or "")
    return int(match.group(1)) if match else None


def print_activation_hint() -> None:
    hint = venv_activate_hint()
    print()
    print("Activate the environment with:")
    if sys.platform == "win32":
        print(f"    . \"{hint}\"")
    else:
        print(f"    source \"{hint}\"")
    print()
    print("Quick start:")
    print("    python -m src.verify                 # model + weights sanity check")
    print("    python -m src.train --help           # training")
    print("    python -m src.evaluate --help        # evaluation")
    print("    python -m src.export --help          # export")
    print("    python scripts/status_report.py      # project/git/model/venv report")
    print()
    print("Web / Android apps (dependencies are installed by this setup):")
    print("    cd breed-classifier-react && npm run dev        # farmer web app")
    print("    cd vet-portal && npm run dev                    # vet portal")
    print("    cd breed-classifier-react && npm run android:assemble")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smart venv creation/update and dependency installation.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the post-install model verification step")
    parser.add_argument("--cpu", "--force-cpu", dest="force_cpu", action="store_true",
                        help="Force CPU PyTorch installation")
    parser.add_argument("--gpu", "--force-gpu", dest="force_gpu", action="store_true",
                        help="Force GPU PyTorch installation")
    parser.add_argument("--cuda-tag", default=None,
                        help="Explicit PyTorch CUDA tag (e.g. cu124, cu121, cu118, cpu)")
    parser.add_argument("--reinstall", action="store_true",
                        help="Force reinstall PyTorch and refresh all requirements")
    parser.add_argument("--check", action="store_true",
                        help="Report missing/outdated packages without changing anything")
    parser.add_argument("--report", action="store_true",
                        help="Run the project status report after setup")
    parser.add_argument("--no-frontend", action="store_true",
                        help="Skip the Node/npm installs for breed-classifier-react and vet-portal")
    parser.add_argument("--no-android", action="store_true",
                        help="Skip `npx cap sync android` (implies leaving the APK untouched)")
    parser.add_argument("--android-build", action="store_true",
                        help="After the sync, run gradlew assembleDebug (needs JDK 21 + Android SDK)")
    args = parser.parse_args()

    if not check_python():
        return 1
    if not args.check and not create_venv():
        return 1
    if not args.check and not ensure_pip(venv_python(), upgrade=args.reinstall):
        return 1

    venv_ready = os.path.exists(venv_python())
    if args.check and not venv_ready:
        log("Virtual environment does not exist yet; run setup to create it.")

    if not args.check or venv_ready:
        try:
            if not install_or_update(
                force_cpu=args.force_cpu,
                force_gpu=args.force_gpu,
                cuda_tag=args.cuda_tag,
                reinstall=args.reinstall,
                check_only=args.check,
            ):
                return 1
        except SetupError as exc:
            log(str(exc))
            return 1
        except KeyboardInterrupt:
            log("Setup interrupted by user.")
            return 130
        except Exception as exc:  # last-resort guard: never dump a raw traceback
            log(f"Unexpected setup error: {type(exc).__name__}: {exc}")
            log("The venv was left intact. Re-run with --reinstall if the package state is corrupt.")
            return 1

    if not args.no_frontend:
        try:
            if not setup_frontend(
                check_only=args.check,
                skip_android=args.no_android,
                android_build=args.android_build,
            ):
                return 1
        except KeyboardInterrupt:
            log("Setup interrupted by user.")
            return 130
        except Exception as exc:
            log(f"Frontend setup failed: {type(exc).__name__}: {exc}")
            return 1

    if args.check:
        log("Check complete; no changes were made.")
        return 0

    if not args.no_verify:
        try:
            if not run_verification():
                log("Verification failed. The venv is intact; fix the reported issue and re-run.")
                return 1
        except Exception as exc:
            log(f"Verification could not run: {exc}")
            return 1

    log("Setup complete.")
    print_activation_hint()
    if args.report:
        run_status_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
