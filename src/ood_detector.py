"""Compatibility forwarding module for OOD detector.

All OOD detector definitions are consolidated in ``src.model``.
This module re-exports them for backwards compatibility.
"""

from .model import OODDetector

__all__ = ["OODDetector"]
