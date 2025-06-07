"""
Integrated Tools - Provides versions of Axiom 4 tools that use acceleration
Ensures all modules benefit from the Observer Cache
"""

from typing import List, Tuple, Optional, Dict
import math

# Import core modules
from .adaptive_observer import MultiScaleObserver, generate_superposition
from .observer_cache import ObserverCache
from .accelerated_observer import (
    accelerated_observe,
    accelerated_gradient,
    accelerated_collapse,
    accelerated_navigation,
    accelerated_coherence_field,
    accelerated_gradient_ascent,
    accelerated_multi_path,
    create_accelerated_observer
)
from .quantum_tools import (
    QuantumTunnel,
    harmonic_amplify,
    SpectralFolder,
    quantum_superposition_collapse,
    entangle_positions
)
from .resonance_memory import ResonanceMemory
from .spectral_navigation import harmonic_jump

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom2 import PHI, fib


class IntegratedObserver:
    """
    Integrated observer that automatically uses acceleration
    """
    
    def __init__(self, n: int):
        """
        Initialize integrated observer with automatic caching
        
        Args:
            n: Number being factored
        """
        self.n = n
        self.observer, self.cache = create_accelerated_observer(n)
        
    def observe(self, x: int) -> float:
        """Observe with caching"""
        return accelerated_observe(self.observer, x, self.cache)
        
    def gradient(self, x: int, delta: int = 1) -> float:
        """Calculate gradient with caching"""
        return accelerated_gradient(self.n, x, self.observer, delta, self.cache)
        
    def coherence_field(self, positions: List[int]) -> Dict[int, float]:
        """Build coherence field with caching"""
        return accelerated_coherence_field(self.n, positions, self.observer, self.cache)
        
    def collapse_wavefunction(self, candidates: List[int], 
                            iterations: int = 5) -> List[Tuple[int, float]]:
        """Collapse wavefunction with caching"""
        return accelerated_collapse(self.n, candidates, self.observer, 
                                  iterations, self.cache)
        
    def navigate_to_factor(self, start: int, 
                          max_iterations: int = 100) -> Optional[int]:
        """Navigate to factor with caching"""
        return accelerated_navigation(self.n, start, self.observer, 
                                    max_iterations, self.cache)
        
    def gradient_ascent(self, start: int, max_steps: int = 50, 
                       tolerance: float = 1e-6) -> List[int]:
        """Gradient ascent with caching"""
        return accelerated_gradient_ascent(self.n, start, self.observer,
                                         max_steps, tolerance, self.cache)
        
    def multi_path_search(self, starts: List[int], 
                         max_paths: int = 10) -> List[Tuple[int, float]]:
        """Multi-path search with caching"""
        return accelerated_multi_path(self.n, starts, self.observer, 
                                    max_paths, self.cache)
        
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        return self.cache.get_cache_statistics()


def integrated_observer_search(n: int, memory: Optional[ResonanceMemory] = None,
                             max_iterations: int = 100) -> Optional[int]:
    """
    Integrated search using all Axiom 4 tools with acceleration
    
    Args:
        n: Number to factor
        memory: Optional resonance memory
        max_iterations: Maximum iterations
        
    Returns:
        Factor if found, None otherwise
    """
    # Create integrated observer
    observer = IntegratedObserver(n)
    
    # Generate superposition
    hints = []
    if memory:
        # Get predictions from memory
        predictions = memory.predict(n, top_k=10)
        hints = [pos for pos, _ in predictions]
    
    candidates = generate_superposition(n, hints)
    
    # Collapse wavefunction
    collapsed = observer.collapse_wavefunction(candidates[:50])
    
    # Try top candidates
    for pos, weight in collapsed[:10]:
        if n % pos == 0 and pos > 1:
            return pos
    
    # Navigate from top positions
    for pos, _ in collapsed[:5]:
        factor = observer.navigate_to_factor(pos, max_iterations=20)
        if factor:
            return factor
    
    # Try quantum tunneling if stuck
    tunnel = QuantumTunnel(n)
    for pos, _ in collapsed[:3]:
        tunnel_exits = tunnel.tunnel_sequence(pos)
        for exit_pos in tunnel_exits:
            factor = observer.navigate_to_factor(exit_pos, max_iterations=10)
            if factor:
                return factor
    
    # Try harmonic amplification
    for pos, _ in collapsed[:3]:
        harmonics = harmonic_amplify(n, pos)
        for h_pos in harmonics[:5]:
            if n % h_pos == 0 and h_pos > 1:
                return h_pos
    
    # Try spectral folding
    folder = SpectralFolder(n)
    for pos, _ in collapsed[:5]:
        fold = folder.nearest_fold(pos)
        if n % fold == 0 and fold > 1:
            return fold
        
        # Navigate from fold
        factor = observer.navigate_to_factor(fold, max_iterations=10)
        if factor:
            return factor
    
    return None


def integrated_axiom4_factor(n: int, use_memory: bool = True,
                           verbose: bool = False) -> Optional[int]:
    """
    Complete Axiom 4 factorization with full acceleration
    
    Args:
        n: Number to factor
        use_memory: Whether to use resonance memory
        verbose: Print progress information
        
    Returns:
        Factor if found, None otherwise
    """
    if verbose:
        print(f"Axiom 4: Factoring {n} with full acceleration")
    
    # Initialize components
    observer = IntegratedObserver(n)
    memory = ResonanceMemory() if use_memory else None
    
    # Phase 1: Initial exploration
    if verbose:
        print("Phase 1: Initial exploration...")
    
    # Find coherence peaks
    root = int(math.isqrt(n))
    peak_positions = []
    for i in range(10):
        pos = int(root * (i + 1) / 11)
        if 2 <= pos <= root:
            peak_positions.append(pos)
    
    # Multi-path search from peaks
    endpoints = observer.multi_path_search(peak_positions)
    
    # Check endpoints
    for pos, coh in endpoints:
        if n % pos == 0 and pos > 1:
            if verbose:
                print(f"Found factor at coherence peak: {pos}")
            return pos
    
    # Phase 2: Quantum superposition
    if verbose:
        print("Phase 2: Quantum superposition collapse...")
    
    # Generate and collapse superposition
    candidates = generate_superposition(n)
    collapsed = observer.collapse_wavefunction(candidates)
    
    # Try collapsed positions
    for pos, weight in collapsed[:20]:
        if n % pos == 0 and pos > 1:
            if verbose:
                print(f"Found factor in collapsed state: {pos}")
            if memory:
                # Record success
                p = 2  # Default prime
                f = 1  # Default fibonacci
                memory.record(p, f, n, weight, pos)
            return pos
    
    # Phase 3: Integrated search
    if verbose:
        print("Phase 3: Integrated search...")
    
    factor = integrated_observer_search(n, memory)
    if factor:
        if verbose:
            print(f"Found factor through integrated search: {factor}")
        return factor
    
    # Phase 4: Exhaustive navigation
    if verbose:
        print("Phase 4: Exhaustive navigation...")
        
    # Try more starting positions
    for i in range(20):
        start = int(root * (i + 1) / 21)
        if 2 <= start <= root:
            factor = observer.navigate_to_factor(start, max_iterations=50)
            if factor:
                if verbose:
                    print(f"Found factor through navigation: {factor}")
                return factor
    
    if verbose:
        print("No factor found")
        stats = observer.get_cache_stats()
        print(f"Cache hit rate: {stats['total_hit_rate']:.2%}")
    
    return None


# Export integrated versions
__all__ = [
    'IntegratedObserver',
    'integrated_observer_search',
    'integrated_axiom4_factor'
]
