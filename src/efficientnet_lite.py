"""Compatibility forwarding module for EfficientNet-Lite backbone.

All backbone definitions are consolidated in ``src.model``.
This module re-exports them for backwards compatibility.
"""

from .model import (
    MBConvBlock,
    EfficientNetLite,
    load_backbone_weights,
)

__all__ = [
    "MBConvBlock",
    "EfficientNetLite",
    "load_backbone_weights",
]
