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

from .spectral_signature_cache import SpectralSignatureCache

from .accelerated_analysis import (
    accelerated_spectral_vector,
    accelerated_coherence,
    accelerated_fold_energy,
    accelerated_interference_analysis,
    accelerated_coherence_field,
    accelerated_sharp_folds,
    accelerated_spectral_analysis,
    create_accelerated_analyzer
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
    'identify_resonance_source',
    
    # Acceleration
    'SpectralSignatureCache',
    
    # Accelerated Analysis
    'accelerated_spectral_vector',
    'accelerated_coherence',
    'accelerated_fold_energy',
    'accelerated_interference_analysis',
    'accelerated_coherence_field',
    'accelerated_sharp_folds',
    'accelerated_spectral_analysis',
    'create_accelerated_analyzer'
]
