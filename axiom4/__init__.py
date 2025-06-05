"""
Axiom 4: Observer Effect
Adaptive observation creates quantum superposition that collapses toward factors
"""

from .adaptive_observer import (
    MultiScaleObserver,
    generate_superposition,
    collapse_wavefunction
)

from .spectral_navigation import (
    coherence_gradient,
    gradient_ascent,
    multi_path_search,
    harmonic_jump
)

from .quantum_tools import (
    QuantumTunnel,
    harmonic_amplify,
    SpectralFolder
)

from .resonance_memory import (
    ResonanceMemory
)

__all__ = [
    # Adaptive Observer
    'MultiScaleObserver',
    'generate_superposition',
    'collapse_wavefunction',
    
    # Spectral Navigation
    'coherence_gradient',
    'gradient_ascent',
    'multi_path_search',
    'harmonic_jump',
    
    # Quantum Tools
    'QuantumTunnel',
    'harmonic_amplify',
    'SpectralFolder',
    
    # Resonance Memory
    'ResonanceMemory'
]
