"""
Accelerated Observer - High-level functions using ObserverCache
Provides accelerated observation, navigation, and quantum collapse
"""

import math
from typing import List, Dict, Tuple, Optional, Any
from .observer_cache import ObserverCache
from .adaptive_observer import (
    MultiScaleObserver, 
    generate_superposition,
    collapse_wavefunction as direct_collapse,
    create_coherence_gradient_field as direct_gradient_field
)
from .spectral_navigation import (
    coherence_gradient as direct_gradient,
    gradient_ascent as direct_ascent,
    multi_path_search as direct_multi_path,
    navigate_to_factor as direct_navigate
)
from .quantum_tools import (
    quantum_superposition_collapse as direct_quantum_collapse
)

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom2 import fib, PHI

# Global cache instance (can be overridden)
_global_cache: Optional[ObserverCache] = None

def get_global_cache() -> ObserverCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = ObserverCache(cache_size=10000)
    return _global_cache

def set_global_cache(cache: ObserverCache):
    """Set custom global cache"""
    global _global_cache
    _global_cache = cache

def accelerated_observe(observer: MultiScaleObserver, position: int,
                       cache: Optional[ObserverCache] = None) -> float:
    """
    Perform cached multi-scale observation
    
    Args:
        observer: MultiScaleObserver instance
        position: Position to observe
        cache: Optional cache instance
        
    Returns:
        Observation coherence value
    """
    if cache is None:
        cache = get_global_cache()
    return cache.get_observation(observer, position)

def accelerated_gradient(n: int, position: int, observer: MultiScaleObserver,
                        delta: int = 1, cache: Optional[ObserverCache] = None) -> float:
    """
    Calculate coherence gradient with caching
    
    Args:
        n: Number being factored
        position: Position to calculate gradient
        observer: MultiScaleObserver instance
        delta: Step size for finite difference
        cache: Optional cache instance
        
    Returns:
        Gradient value
    """
    if cache is None:
        cache = get_global_cache()
    return cache.get_gradient(n, position, observer, delta)

def accelerated_collapse(n: int, candidates: List[int], 
                        observer: MultiScaleObserver,
                        iterations: int = 5,
                        cache: Optional[ObserverCache] = None) -> List[Tuple[int, float]]:
    """
    Optimized wavefunction collapse with caching
    
    Args:
        n: Number being factored
        candidates: Initial superposition
        observer: Multi-scale observer
        iterations: Number of collapse iterations
        cache: Optional cache instance
        
    Returns:
        List of (position, weight) tuples sorted by weight
    """
    if cache is None:
        cache = get_global_cache()
        
    root = int(math.isqrt(n))
    
    # Check if we have a cached state we can resume from
    for i in range(iterations - 1, -1, -1):
        cached_state = cache.get_quantum_state(n, i)
        if cached_state:
            # Resume from cached state
            weighted_candidates = cached_state
            start_iteration = i + 1
            break
    else:
        # Initialize weights using cached observations
        weighted_candidates = [(x, cache.get_observation(observer, x)) for x in candidates]
        start_iteration = 0
    
    for iteration in range(start_iteration, iterations):
        new_candidates = []
        
        for x, weight in weighted_candidates:
            # Calculate gradient using cache
            gradient = cache.get_gradient(n, x, observer)
            
            # Move in gradient direction
            step_size = max(1, int(root * 0.02 / (iteration + 1)))
            if gradient > 0:
                new_x = min(root, x + step_size)
            elif gradient < 0:
                new_x = max(2, x - step_size)
            else:
                new_x = x
            
            # Calculate new weight using cached observation
            new_coherence = cache.get_observation(observer, new_x)
            new_weight = new_coherence * (1 + abs(gradient))
            
            new_candidates.append((new_x, new_weight))
        
        # Keep top candidates
        new_candidates.sort(key=lambda t: -t[1])
        weighted_candidates = new_candidates[:max(20, len(candidates) // 2)]
        
        # Add exploration positions
        if iteration < iterations - 1:
            gradient_positions = []
            for x, _ in weighted_candidates[:10]:
                for offset in [-1, 1]:
                    test_x = x + offset * step_size
                    if 2 <= test_x <= root:
                        coh = cache.get_observation(observer, test_x)
                        gradient_positions.append((test_x, coh))
            
            # Merge with existing candidates
            all_positions = weighted_candidates + gradient_positions
            seen = set()
            unique = []
            for x, w in all_positions:
                if x not in seen:
                    seen.add(x)
                    unique.append((x, w))
            weighted_candidates = sorted(unique, key=lambda t: -t[1])[:len(candidates)]
        
        # Cache the quantum state
        cache.cache_quantum_state(n, iteration, weighted_candidates)
    
    return weighted_candidates

def accelerated_navigation(n: int, start: int, observer: MultiScaleObserver,
                          max_iterations: int = 100,
                          cache: Optional[ObserverCache] = None) -> Optional[int]:
    """
    Navigate coherence field with path caching
    
    Args:
        n: Number being factored
        start: Starting position
        observer: Multi-scale observer
        max_iterations: Maximum iterations
        cache: Optional cache instance
        
    Returns:
        Factor if found, None otherwise
    """
    if cache is None:
        cache = get_global_cache()
        
    # Check if we have a cached path that might help
    root = int(math.isqrt(n))
    for end in range(2, min(root + 1, 100)):
        if n % end == 0:
            path = cache.get_navigation_path(n, start, end)
            if path:
                # Follow cached path
                for pos in path:
                    if n % pos == 0 and pos > 1:
                        return pos
    
    # No cached path, navigate with caching
    current = start
    path = [current]
    stuck_count = 0
    visited = set()
    
    for iteration in range(max_iterations):
        # Check if current is a factor
        if n % current == 0 and current > 1:
            # Store successful path
            cache.store_navigation_path(n, start, current, path)
            return current
        
        visited.add(current)
        
        # Calculate gradient using cache
        gradient = cache.get_gradient(n, current, observer)
        
        # Determine step size
        step_size = max(1, int(root * 0.02 / (iteration + 1)))
        
        # Move in gradient direction
        if abs(gradient) < 1e-6:
            # No gradient, need to jump
            stuck_count += 1
            # Use harmonic jump
            from .spectral_navigation import harmonic_jump
            new_pos = harmonic_jump(n, current, stuck_count)
        else:
            if gradient > 0:
                new_pos = min(root, current + step_size)
            else:
                new_pos = max(2, current - step_size)
                
            if new_pos == current or new_pos in visited:
                stuck_count += 1
                from .spectral_navigation import harmonic_jump
                new_pos = harmonic_jump(n, current, stuck_count)
            else:
                stuck_count = 0
        
        current = new_pos
        path.append(current)
    
    return None

def accelerated_coherence_field(n: int, positions: List[int],
                               observer: MultiScaleObserver,
                               cache: Optional[ObserverCache] = None) -> Dict[int, float]:
    """
    Build coherence field with batch caching
    
    Args:
        n: Number being factored
        positions: List of positions to evaluate
        observer: Multi-scale observer
        cache: Optional cache instance
        
    Returns:
        Dictionary mapping position to coherence
    """
    if cache is None:
        cache = get_global_cache()
        
    # Pre-compute critical positions for better cache efficiency
    if len(positions) > 10:
        cache.precompute_critical_positions(n, observer)
    
    field = {}
    for pos in positions:
        field[pos] = cache.get_observation(observer, pos)
    
    return field

def accelerated_gradient_ascent(n: int, start: int, observer: MultiScaleObserver,
                               max_steps: int = 50, tolerance: float = 1e-6,
                               cache: Optional[ObserverCache] = None) -> List[int]:
    """
    Gradient ascent with caching
    
    Args:
        n: Number being factored
        start: Starting position
        observer: Multi-scale observer
        max_steps: Maximum steps
        tolerance: Convergence tolerance
        cache: Optional cache instance
        
    Returns:
        Path of positions visited
    """
    if cache is None:
        cache = get_global_cache()
        
    root = int(math.isqrt(n))
    path = [start]
    current = start
    
    for step in range(max_steps):
        # Get cached gradient
        grad = cache.get_gradient(n, current, observer)
        
        # Check convergence
        if abs(grad) < tolerance:
            break
        
        # Adaptive step size
        step_size = max(1, int(root * 0.02 / (step + 1)))
        
        # Move in gradient direction
        if grad > 0:
            next_pos = min(root, current + step_size)
        elif grad < 0:
            next_pos = max(2, current - step_size)
        else:
            break
        
        # Check if stuck
        if next_pos == current:
            break
        
        current = next_pos
        path.append(current)
        
        # Check if we found a factor
        if n % current == 0:
            break
    
    return path

def accelerated_multi_path(n: int, starts: List[int], observer: MultiScaleObserver,
                          max_paths: int = 10,
                          cache: Optional[ObserverCache] = None) -> List[Tuple[int, float]]:
    """
    Multi-path search with caching
    
    Args:
        n: Number being factored
        starts: Starting positions
        observer: Multi-scale observer
        max_paths: Maximum paths to explore
        cache: Optional cache instance
        
    Returns:
        List of (position, coherence) for path endpoints
    """
    if cache is None:
        cache = get_global_cache()
        
    endpoints = []
    paths_to_explore = starts[:max_paths]
    
    for start in paths_to_explore:
        # Follow gradient with caching
        path = accelerated_gradient_ascent(n, start, observer, cache=cache)
        
        if path:
            endpoint = path[-1]
            coh = cache.get_observation(observer, endpoint)
            endpoints.append((endpoint, coh))
    
    # Sort by coherence
    endpoints.sort(key=lambda t: -t[1])
    
    # Remove duplicates
    seen = set()
    unique_endpoints = []
    for pos, coh in endpoints:
        if pos not in seen:
            seen.add(pos)
            unique_endpoints.append((pos, coh))
    
    return unique_endpoints

def create_accelerated_observer(n: int) -> Tuple[MultiScaleObserver, ObserverCache]:
    """
    Create an accelerated observer setup for number n
    
    Args:
        n: Number to be analyzed
        
    Returns:
        Tuple of (observer, cache) configured for n
    """
    observer = MultiScaleObserver(n)
    cache = ObserverCache.create_optimized(n)
    
    # Pre-compute critical positions
    cache.precompute_critical_positions(n, observer)
    
    # Warm up cache with high-probability positions
    cache.warm_cache(n, observer)
    
    return observer, cache

def benchmark_acceleration(n: int, iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark acceleration vs direct computation
    
    Args:
        n: Number to test
        iterations: Number of iterations
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    observer = MultiScaleObserver(n)
    cache = ObserverCache.create_optimized(n)
    
    # Test observation
    test_positions = list(range(2, min(20, int(math.isqrt(n)))))
    
    # Direct observation
    start = time.time()
    for _ in range(iterations):
        for pos in test_positions:
            _ = observer.observe(pos)
    time_direct_observe = time.time() - start
    
    # Cached observation
    cache.clear()
    start = time.time()
    for _ in range(iterations):
        for pos in test_positions:
            _ = cache.get_observation(observer, pos)
    time_cached_observe = time.time() - start
    
    # Test gradient computation
    start = time.time()
    for _ in range(iterations // 10):
        for pos in test_positions:
            _ = direct_gradient(n, pos, observer)
    time_direct_gradient = time.time() - start
    
    cache.clear()
    start = time.time()
    for _ in range(iterations // 10):
        for pos in test_positions:
            _ = cache.get_gradient(n, pos, observer)
    time_cached_gradient = time.time() - start
    
    # Calculate speedups
    speedup_observe = time_direct_observe / time_cached_observe if time_cached_observe > 0 else float('inf')
    speedup_gradient = time_direct_gradient / time_cached_gradient if time_cached_gradient > 0 else float('inf')
    
    return {
        'time_direct_observe': time_direct_observe,
        'time_cached_observe': time_cached_observe,
        'speedup_observe': speedup_observe,
        'time_direct_gradient': time_direct_gradient,
        'time_cached_gradient': time_cached_gradient,
        'speedup_gradient': speedup_gradient,
        'cache_stats': cache.get_cache_statistics()
    }
