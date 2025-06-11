"""
Prime Resonance Field (RFH3) - Adaptive Resonance Field Architecture

A paradigm shift from static resonance field mapping to dynamic, adaptive field exploration.
"""

from .core import (
    HierarchicalSearch,
    LazyResonanceIterator,
    MultiScaleResonance,
    StateManager,
)
from .learning import ResonancePatternLearner, ZonePredictor
from .rfh3 import RFH3, RFH3Config

__version__ = "3.0.0"
__all__ = [
    "RFH3",
    "RFH3Config",
    "LazyResonanceIterator",
    "MultiScaleResonance",
    "StateManager",
    "HierarchicalSearch",
    "ResonancePatternLearner",
    "ZonePredictor",
]
