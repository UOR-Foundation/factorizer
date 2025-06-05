"""
Axiom 3: Duality Principle
Wave-particle duality for number factorization through spectral analysis
"""

from .spectral_core import (
    binary_spectrum,
    modular_spectrum,
    digital_spectrum,
    harmonic_spectrum,
    spectral_vector
)

from .coherence import (
    coherence,
    CoherenceCache
)

from .fold_topology import (
    fold_energy,
    sharp_fold_candidates,
    FoldTopology
)

from .interference import (
    prime_fib_interference,
    interference_extrema,
    identify_resonance_source
)

__all__ = [
    # Spectral Core
    'binary_spectrum',
    'modular_spectrum',
    'digital_spectrum',
    'harmonic_spectrum',
    'spectral_vector',
    
    # Coherence
    'coherence',
    'CoherenceCache',
    
    # Fold Topology
    'fold_energy',
    'sharp_fold_candidates',
    'FoldTopology',
    
    # Interference
    'prime_fib_interference',
    'interference_extrema',
    'identify_resonance_source'
]
