"""
Accelerated Analysis - High-level functions using SpectralSignatureCache
Provides accelerated spectral analysis for Axiom 3
"""

import math
from typing import List, Dict, Tuple, Optional
from .spectral_signature_cache import SpectralSignatureCache
from .spectral_core import spectral_vector
from .coherence import coherence as direct_coherence
from .fold_topology import FoldTopology, fold_energy as direct_fold_energy
from .interference import prime_fib_interference as direct_interference, interference_extrema

# Global cache instance (can be overridden)
_global_cache: Optional[SpectralSignatureCache] = None

def get_global_cache() -> SpectralSignatureCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SpectralSignatureCache(cache_size=20000)
    return _global_cache

def set_global_cache(cache: SpectralSignatureCache):
    """Set custom global cache"""
    global _global_cache
    _global_cache = cache

def accelerated_spectral_vector(n: int, cache: Optional[SpectralSignatureCache] = None) -> List[float]:
    """
    Get spectral vector with acceleration
    
    Args:
        n: Number to analyze
        cache: Optional cache instance (uses global if not provided)
        
    Returns:
        Spectral vector S(n)
    """
    if cache is None:
        cache = get_global_cache()
    return cache.get_spectral_vector(n)

def accelerated_coherence(a: int, b: int, n: int, cache: Optional[SpectralSignatureCache] = None) -> float:
    """
    Calculate coherence with acceleration
    
    Args:
        a: First number
        b: Second number
        n: Target number
        cache: Optional cache instance
        
    Returns:
        Coherence value C(a,b,n)
    """
    if cache is None:
        cache = get_global_cache()
    return cache.get_coherence(a, b, n)

def accelerated_fold_energy(n: int, x: int, cache: Optional[SpectralSignatureCache] = None) -> float:
    """
    Calculate fold energy with acceleration
    
    Args:
        n: Number being factored
        x: Position to evaluate
        cache: Optional cache instance
        
    Returns:
        Fold energy E(x)
    """
    if cache is None:
        cache = get_global_cache()
    return cache.get_fold_energy(n, x)

def accelerated_interference_analysis(n: int, cache: Optional[SpectralSignatureCache] = None) -> Tuple[List[float], List[int]]:
    """
    Analyze interference pattern with acceleration
    
    Args:
        n: Number to analyze
        cache: Optional cache instance
        
    Returns:
        (interference_pattern, extrema_positions)
    """
    if cache is None:
        cache = get_global_cache()
    return cache.get_interference_pattern(n)

def accelerated_coherence_field(n: int, candidates: List[int], cache: Optional[SpectralSignatureCache] = None) -> Dict[int, float]:
    """
    Build coherence field for multiple candidates with acceleration
    
    Args:
        n: Target number
        candidates: List of candidate factors
        cache: Optional cache instance
        
    Returns:
        Dictionary mapping candidate -> coherence
    """
    if cache is None:
        cache = get_global_cache()
        
    field = {}
    for x in candidates:
        if n % x == 0:
            complement = n // x
            field[x] = cache.get_coherence(x, complement, n)
        else:
            # For non-factors, use a heuristic coherence
            field[x] = 0.0
            
    return field

def accelerated_fold_topology(n: int, cache: Optional[SpectralSignatureCache] = None) -> FoldTopology:
    """
    Build fold topology with acceleration
    
    Creates a FoldTopology instance that uses cached fold energies.
    
    Args:
        n: Number to analyze
        cache: Optional cache instance
        
    Returns:
        FoldTopology instance with accelerated computation
    """
    if cache is None:
        cache = get_global_cache()
        
    # Pre-compute some fold energies to warm up cache
    root = int(math.isqrt(n))
    for x in range(2, min(root + 1, 100)):
        cache.get_fold_energy(n, x)
        
    # Return standard topology (it will benefit from cached values)
    return FoldTopology(n)

def accelerated_sharp_folds(n: int, cache: Optional[SpectralSignatureCache] = None) -> List[int]:
    """
    Find sharp fold candidates with acceleration
    
    Args:
        n: Number to analyze
        cache: Optional cache instance
        
    Returns:
        List of sharp fold positions
    """
    if cache is None:
        cache = get_global_cache()
    return cache.get_sharp_folds(n)

def accelerated_spectral_analysis(n: int, full_analysis: bool = True) -> Dict[str, any]:
    """
    Complete spectral analysis with acceleration
    
    Performs comprehensive spectral analysis using all cached components.
    
    Args:
        n: Number to analyze
        full_analysis: Whether to include all analysis types
        
    Returns:
        Dictionary containing:
        - spectral_vector: S(n)
        - sharp_folds: Top fold candidates
        - interference_extrema: Interference peaks/valleys
        - coherence_samples: Sample coherence values
        - cache_stats: Cache performance statistics
    """
    # Create optimized cache for this n
    cache = SpectralSignatureCache.create_optimized(n)
    
    # Get spectral vector
    spec_vector = cache.get_spectral_vector(n)
    
    # Find sharp folds
    sharp_folds = cache.get_sharp_folds(n)
    
    # Get interference pattern
    pattern, extrema = cache.get_interference_pattern(n)
    
    result = {
        'spectral_vector': spec_vector,
        'sharp_folds': sharp_folds[:10],  # Top 10
        'interference_extrema': extrema[:10],  # Top 10
    }
    
    if full_analysis:
        # Sample coherence for some candidates
        coherence_samples = {}
        candidates = list(set(sharp_folds[:5] + extrema[:5]))
        
        for x in candidates:
            if n % x == 0:
                complement = n // x
                coherence_samples[x] = cache.get_coherence(x, complement, n)
                
        result['coherence_samples'] = coherence_samples
        
        # Include cache statistics
        result['cache_stats'] = cache.get_cache_statistics()
        
    return result

def benchmark_acceleration(n: int, iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark acceleration vs direct computation
    
    Args:
        n: Number to test
        iterations: Number of iterations for timing
        
    Returns:
        Dictionary with timing results and speedup factors
    """
    import time
    
    # Test spectral vector computation
    start = time.time()
    for _ in range(iterations):
        s = spectral_vector(n)
    time_direct_spectral = time.time() - start
    
    cache = SpectralSignatureCache.create_optimized(n)
    start = time.time()
    for _ in range(iterations):
        s = cache.get_spectral_vector(n)
    time_cached_spectral = time.time() - start
    
    # Test coherence computation
    test_pairs = [(3, n//3) if n % 3 == 0 else (2, n//2) if n % 2 == 0 else (5, 7)]
    a, b = test_pairs[0]
    
    start = time.time()
    for _ in range(iterations // 10):  # Fewer iterations as coherence is slower
        c = direct_coherence(a, b, n)
    time_direct_coherence = time.time() - start
    
    start = time.time()
    for _ in range(iterations // 10):
        c = cache.get_coherence(a, b, n)
    time_cached_coherence = time.time() - start
    
    # Calculate speedups
    speedup_spectral = time_direct_spectral / time_cached_spectral if time_cached_spectral > 0 else float('inf')
    speedup_coherence = time_direct_coherence / time_cached_coherence if time_cached_coherence > 0 else float('inf')
    
    return {
        'time_direct_spectral': time_direct_spectral,
        'time_cached_spectral': time_cached_spectral,
        'speedup_spectral': speedup_spectral,
        'time_direct_coherence': time_direct_coherence,
        'time_cached_coherence': time_cached_coherence,
        'speedup_coherence': speedup_coherence,
        'cache_stats': cache.get_cache_statistics()
    }

# Convenience function for creating an accelerated analyzer
def create_accelerated_analyzer(n: int) -> SpectralSignatureCache:
    """
    Create an accelerated analyzer for number n
    
    This is a convenience function that creates an optimized cache
    instance configured for analyzing number n.
    
    Args:
        n: Number to be analyzed
        
    Returns:
        Configured SpectralSignatureCache instance
    """
    return SpectralSignatureCache.create_optimized(n)
