#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 setup_venv.py
echo
echo "Activate the environment in your shell with:"
echo "    source .venv/bin/activate"