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

# Acceleration imports
from .observer_cache import ObserverCache
from .accelerated_observer import (
    accelerated_observe,
    accelerated_gradient,
    accelerated_collapse,
    accelerated_navigation,
    accelerated_coherence_field,
    accelerated_gradient_ascent,
    accelerated_multi_path,
    create_accelerated_observer,
    benchmark_acceleration,
    get_global_cache,
    set_global_cache
)

# Integrated tools
from .integrated_tools import (
    IntegratedObserver,
    integrated_observer_search,
    integrated_axiom4_factor
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
    'ResonanceMemory',
    
    # Acceleration
    'ObserverCache',
    'accelerated_observe',
    'accelerated_gradient',
    'accelerated_collapse',
    'accelerated_navigation',
    'accelerated_coherence_field',
    'accelerated_gradient_ascent',
    'accelerated_multi_path',
    'create_accelerated_observer',
    'benchmark_acceleration',
    'get_global_cache',
    'set_global_cache',
    
    # Integrated Tools
    'IntegratedObserver',
    'integrated_observer_search',
    'integrated_axiom4_factor'
]
