# Installation & Setup

Step-by-step installation guide for both **Windows** and **Linux / macOS**.

> **Quick path**: If you just want to train, skip this page and use `local_train.py` — it handles everything automatically. See [Local Training (Automated)](local-training.md).

---

## Prerequisites

| Requirement | Minimum | Recommended | Notes |
|---|---|---|---|
| **Python** | 3.9 | 3.11+ | Must be on `PATH` |
| **Git** | Any | Latest | For cloning the repository |
| **GPU (CUDA)** | Optional | RTX 3050 4 GB+ | Falls back to CPU gracefully |
| **Kaggle account** | Required | — | Free, for dataset download |
| **Disk space** | ~8 GB | ~15 GB | Dataset + checkpoints + exports |
| **RAM** | 8 GB | 16 GB+ | |

---

## Installation — Windows

Open **Command Prompt** or **PowerShell** (not Administrator).

### Step 1 — Install Python 3.11+

Download from [python.org/downloads](https://www.python.org/downloads/) and run the installer.

- ✅ Check **"Add Python to PATH"** before clicking Install.
- ✅ Check **"Install for all users"** (optional but recommended).

Verify:
```cmd
python --version
pip --version
```

### Step 2 — Install Git

Download from [git-scm.com/download/win](https://git-scm.com/download/win) and install with default settings.

Verify:
```cmd
git --version
```

### Step 3 — (Optional) Visual C++ Build Tools

Some Python packages compile C extensions. PyTorch does **not** require this (pre-built wheels). Install only if you encounter build errors with other packages.

Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/  
Select workload: **"Desktop development with C++"**

### Step 4 — Clone the repository

```cmd
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier
```

### Step 5 — Set up Kaggle credentials

Get your API key: [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**

```cmd
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

Or skip this step — `local_train.py` will prompt you interactively on first run.

### Step 6 — Run the pipeline

```cmd
python local_train.py --quarter-data
```

The script creates `.venv`, installs dependencies, downloads the dataset, trains the model, and exports it.

---

### Windows: Manual Virtual Environment (Optional)

If you prefer to set up the environment manually without `local_train.py`:

```cmd
:: Create virtual environment
python -m venv .venv

:: Activate
.venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install training dependencies (excludes webapp deps)
pip install torch>=2.1.0 torchvision>=0.16.0
pip install numpy pandas matplotlib scikit-learn tqdm Pillow requests onnx
```

After activation, your prompt shows `(.venv)`. All subsequent `python` and `pip` commands use the venv.

To deactivate:
```cmd
deactivate
```

---

## Installation — Linux / macOS

### Step 1 — Install Python 3.11+

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip -y
```

Check your default `python3` version:
```bash
python3 --version
```
If it's still an older version, use `python3.11` explicitly.

**macOS (Homebrew):**
```bash
brew install python@3.11
```

Add to PATH if needed (put in `~/.zshrc` or `~/.bashrc`):
```bash
export PATH="$(brew --prefix python@3.11)/bin:$PATH"
```

**Fedora / RHEL:**
```bash
sudo dnf install python3.11 python3.11-devel -y
```

### Step 2 — Install Git

**Ubuntu / Debian:**
```bash
sudo apt install git -y
```

**macOS:**
```bash
brew install git
```

**Fedora:**
```bash
sudo dnf install git -y
```

Verify:
```bash
git --version
```

### Step 3 — Clone the repository

```bash
git clone https://github.com/Bharaths31/ML-CB-B-identifier
cd ML-CB-B-identifier
```

### Step 4 — Set up Kaggle credentials

Get your API key: [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json   # Required — Kaggle CLI checks permissions
```

Or skip this step — `local_train.py` will prompt you interactively on first run.

### Step 5 — Run the pipeline

```bash
python3 local_train.py --quarter-data
```

---

### Linux / macOS: Manual Virtual Environment (Optional)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install training dependencies (excludes webapp deps)
pip install torch>=2.1.0 torchvision>=0.16.0
pip install numpy pandas matplotlib scikit-learn tqdm Pillow requests onnx
```

After activation, your prompt shows `(.venv)`. All subsequent `python` and `pip` commands use the venv.

To deactivate:
```bash
deactivate
```

---

## Backbone Weights

The pretrained ImageNet backbone weights are **already included in the repository**:

```
efficientnet_lite2.pth   # ~24 MB — used when --backbone lite2 (default)
efficientnet_lite4.pth   # ~50 MB — used when --backbone lite4
```

No separate download needed. They are loaded automatically by `src/train.py` at the start of Phase 1.

---

## Dependencies

### Training Dependencies (required)

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24
pandas>=1.5
matplotlib>=3.7
scikit-learn>=1.3
tqdm>=4.66
Pillow>=10.0
requests>=2.31
onnx>=1.16
```

`local_train.py` installs these automatically (skips webapp-only deps).

### Model Tester (`test_model.py`)

Uses only `torch`, `torchvision`, and `Pillow` — already included above. No extra installation required.

---

## Troubleshooting Installation

| Error | Cause | Fix |
|---|---|---|
| `'python' is not recognized` (Windows) | Python not on PATH | Re-run Python installer, check "Add to PATH" |
| `python3: command not found` (Linux) | Python not installed | `sudo apt install python3.11 -y` |
| `pip install` build fails | Missing MSVC (Windows) | Install Visual C++ Build Tools |
| `kaggle: command not found` | Kaggle CLI not installed | Not needed — `local_train.py` uses the Kaggle REST API directly |
| `Permission denied ~/.kaggle` | Wrong file permissions | `chmod 600 ~/.kaggle/kaggle.json` |
| GPU not detected | CUDA drivers missing | Install NVIDIA drivers + CUDA toolkit, or use `--device cpu` |
