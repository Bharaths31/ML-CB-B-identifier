"""Compatibility forwarding module for breed trait heads.

The canonical implementation lives in :mod:`src.model` (SECTION 5) together
with the rest of the network. This shim keeps older imports
(``from src.traits import TraitClassifier``) working without maintaining a
second copy of the code.
"""

from .model import (TraitClassifier, breed_trait_targets, build_trait_vocab,
                    evaluate_traits, load_trait_spec)

__all__ = [
    "TraitClassifier",
    "breed_trait_targets",
    "build_trait_vocab",
    "evaluate_traits",
    "load_trait_spec",
]
