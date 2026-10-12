#!/usr/bin/env python3
"""Convenience CLI entrypoint for training the Cattle & Buffalo Breed Classifier.

Delegates directly to src.train.main().
Usage:
    python train_model.py --smoke-test --skip-qat
    python train_model.py --backbone lite2 --phase2-epochs 50
"""
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import main

if __name__ == "__main__":
    raise SystemExit(main())
