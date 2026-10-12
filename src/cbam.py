"""Compatibility forwarding module for CBAM and SE attention mechanisms.

All attention definitions are consolidated in ``src.model``.
This module re-exports them for backwards compatibility.
"""

from .model import (
    ChannelAttention,
    SpatialAttention,
    CBAM,
    SEBlock,
    build_attention,
)

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "SEBlock",
    "build_attention",
]
