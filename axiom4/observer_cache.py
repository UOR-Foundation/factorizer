"""
Observer Cache - Accelerates multi-scale observation through intelligent caching
Caches observation results, gradients, quantum states, and navigation paths
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom1 import primes_up_to
from axiom2 import fib, PHI, GOLDEN_ANGLE
from axiom3 import SpectralSignatureCache, accelerated_coherence


class ObserverCache:
    """
    Intelligent cache for accelerating observer operations
    
    Components:
    - Observation cache: Multi-scale observation results
    - Gradient field memory: Pre-computed gradients
    - Quantum state cache: Intermediate wavefunction states
    - Navigation path index: Successful navigation paths
    """
    
    def __init__(self, cache_size: int = 10000):
        """
        Initialize observer cache
        
        Args:
            cache_size: Maximum number of entries per cache
        """
        self.cache_size = cache_size
        
        # Core caches - using OrderedDict for LRU eviction
        self.observation_cache = OrderedDict()  # (pos, scales_key) -> coherence
        self.gradient_cache = OrderedDict()     # (n, pos, delta) -> gradient
        self.state_cache = {}                   # (n, iteration) -> quantum_state
        self.path_cache = OrderedDict()         # (n, start, end) -> path
        
        # Pre-computation flags
        self.precomputed_fibonacci = set()
        self.precomputed_primes = set()
        self.precomputed_sqrt = set()
        self.precomputed_gradients = set()  # Track gradient pre-computation
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.gradient_hits = 0
        self.gradient_misses = 0
        self.path_hits = 0
        self.path_misses = 0
        
        # Integration with Axiom 3
        self.spectral_cache = None
        
        # Batch operation support
        self.batch_mode = False
        
    def _make_scales_key(self, scales: Dict[str, int]) -> tuple:
        """Convert scales dict to hashable key"""
        return tuple(sorted(scales.items()))
        
    def _enforce_cache_limit(self, cache: OrderedDict):
        """Enforce LRU eviction when cache exceeds limit"""
        while len(cache) > self.cache_size:
            cache.popitem(last=False)  # Remove oldest
            
    def get_observation(self, observer: Any, position: int) -> float:
        """
        Get cached observation or compute and cache
        
        Args:
            observer: MultiScaleObserver instance
            position: Position to observe
            
        Returns:
            Observation coherence value
        """
        # Create cache key
        scales_key = self._make_scales_key(observer.scales)
        key = (position, scales_key)
        
        # Check cache
        if key in self.observation_cache:
            self.hits += 1
            # Move to end (most recently used)
            self.observation_cache.move_to_end(key)
            return self.observation_cache[key]
        
        # Cache miss - compute value
        self.misses += 1
        
        # Use accelerated coherence from Axiom 3 when computing
        # Store original coherence method and replace temporarily
        original_coherence = sys.modules.get('axiom3.coherence', None)
        if hasattr(original_coherence, 'coherence'):
            original_func = original_coherence.coherence
            original_coherence.coherence = accelerated_coherence
        
        try:
            value = observer.observe(position)
        finally:
            # Restore original coherence
            if hasattr(original_coherence, 'coherence'):
                original_coherence.coherence = original_func
        
        # Cache the result
        self.observation_cache[key] = value
        self._enforce_cache_limit(self.observation_cache)
        
        return value
        
    def get_gradient(self, n: int, position: int, observer: Any, delta: int = 1) -> float:
        """
        Get cached gradient or compute and cache
        
        Args:
            n: Number being factored
            position: Position to calculate gradient
            observer: MultiScaleObserver instance
            delta: Step size for finite difference
            
        Returns:
            Gradient value
        """
        key = (n, position, delta)
        
        # Check cache
        if key in self.gradient_cache:
            self.gradient_hits += 1
            self.gradient_cache.move_to_end(key)
            return self.gradient_cache[key]
        
        # Cache miss - compute gradient
        self.gradient_misses += 1
        root = int(math.isqrt(n))
        
        # Ensure position is in valid range
        if position < 2 or position > root:
            gradient = 0.0
        else:
            # Forward difference
            if position + delta <= root:
                coh_plus = self.get_observation(observer, position + delta)
            else:
                coh_plus = self.get_observation(observer, position)
            
            # Backward difference
            if position - delta >= 2:
                coh_minus = self.get_observation(observer, position - delta)
            else:
                coh_minus = self.get_observation(observer, position)
            
            # Central difference gradient
            gradient = (coh_plus - coh_minus) / (2 * delta)
        
        # Cache the result
        self.gradient_cache[key] = gradient
        self._enforce_cache_limit(self.gradient_cache)
        
        return gradient
        
    def cache_quantum_state(self, n: int, iteration: int, 
                           candidates: List[Tuple[int, float]]):
        """
        Cache intermediate quantum state during wavefunction collapse
        
        Args:
            n: Number being factored
            iteration: Iteration number
            candidates: List of (position, weight) tuples
        """
        key = (n, iteration)
        self.state_cache[key] = candidates.copy()
        
    def get_quantum_state(self, n: int, iteration: int) -> Optional[List[Tuple[int, float]]]:
        """
        Retrieve cached quantum state
        
        Args:
            n: Number being factored
            iteration: Iteration number
            
        Returns:
            Cached state or None
        """
        key = (n, iteration)
        return self.state_cache.get(key)
        
    def store_navigation_path(self, n: int, start: int, end: int, path: List[int]):
        """
        Store successful navigation path
        
        Args:
            n: Number being factored
            start: Starting position
            end: Ending position
            path: Path taken
        """
        key = (n, start, end)
        self.path_cache[key] = path.copy()
        self._enforce_cache_limit(self.path_cache)
        
    def get_navigation_path(self, n: int, start: int, end: int) -> Optional[List[int]]:
        """
        Retrieve cached navigation path
        
        Args:
            n: Number being factored
            start: Starting position
            end: Target ending position
            
        Returns:
            Cached path or None
        """
        key = (n, start, end)
        
        if key in self.path_cache:
            self.path_hits += 1
            self.path_cache.move_to_end(key)
            return self.path_cache[key].copy()
        
        self.path_misses += 1
        return None
        
    def precompute_fibonacci_positions(self, n: int, observer: Any):
        """
        Pre-compute observations at Fibonacci positions
        
        Args:
            n: Number being factored
            observer: MultiScaleObserver instance
        """
        if n in self.precomputed_fibonacci:
            return
            
        root = int(math.isqrt(n))
        k = 1
        
        while True:
            f = fib(k)
            if f > root:
                break
            if f >= 2:
                # Pre-compute observation
                self.get_observation(observer, f)
                
                # Also pre-compute golden ratio scaled positions
                golden_pos = int(f * PHI)
                if 2 <= golden_pos <= root:
                    self.get_observation(observer, golden_pos)
                    
                # Pre-compute inverse golden positions
                inv_golden = int(f / PHI)
                if 2 <= inv_golden <= root:
                    self.get_observation(observer, inv_golden)
            k += 1
            
        self.precomputed_fibonacci.add(n)
        
    def precompute_prime_positions(self, n: int, observer: Any, prime_limit: int = 100):
        """
        Pre-compute observations at prime positions
        
        Args:
            n: Number being factored
            observer: MultiScaleObserver instance
            prime_limit: Maximum prime to pre-compute
        """
        if n in self.precomputed_primes:
            return
            
        root = int(math.isqrt(n))
        primes = primes_up_to(min(prime_limit, root))
        
        for p in primes:
            if p <= root:
                self.get_observation(observer, p)
                
        self.precomputed_primes.add(n)
        
    def precompute_sqrt_neighborhood(self, n: int, observer: Any, radius: int = 50):
        """
        Pre-compute observations near square root
        
        Args:
            n: Number being factored
            observer: MultiScaleObserver instance
            radius: Neighborhood radius
        """
        if n in self.precomputed_sqrt:
            return
            
        root = int(math.isqrt(n))
        
        for offset in range(-radius, radius + 1):
            pos = root + offset
            if 2 <= pos <= root:
                self.get_observation(observer, pos)
                
        self.precomputed_sqrt.add(n)
        
    def precompute_critical_positions(self, n: int, observer: Any):
        """
        Pre-compute observations at all critical positions
        
        Combines Fibonacci, prime, and sqrt positions for optimal caching.
        
        Args:
            n: Number being factored
            observer: MultiScaleObserver instance
        """
        self.precompute_fibonacci_positions(n, observer)
        self.precompute_prime_positions(n, observer)
        self.precompute_sqrt_neighborhood(n, observer)
        
    def precompute_gradients(self, n: int, observer: Any, positions: List[int]):
        """
        Pre-compute gradients for a list of positions
        
        Args:
            n: Number being factored
            observer: MultiScaleObserver instance
            positions: List of positions to pre-compute gradients for
        """
        gradient_key = (n, tuple(sorted(positions)))
        if gradient_key in self.precomputed_gradients:
            return
            
        for pos in positions:
            self.get_gradient(n, pos, observer)
            
        self.precomputed_gradients.add(gradient_key)
        
    def batch_observe(self, observer: Any, positions: List[int]) -> Dict[int, float]:
        """
        Batch observation with optimized caching
        
        Args:
            observer: MultiScaleObserver instance
            positions: List of positions to observe
            
        Returns:
            Dictionary mapping position to coherence
        """
        self.batch_mode = True
        results = {}
        
        # Sort positions for better cache locality
        sorted_positions = sorted(positions)
        
        for pos in sorted_positions:
            results[pos] = self.get_observation(observer, pos)
            
        self.batch_mode = False
        return results
        
    def warm_cache(self, n: int, observer: Any):
        """
        Warm up cache with high-probability positions
        
        Args:
            n: Number being factored
            observer: MultiScaleObserver instance
        """
        root = int(math.isqrt(n))
        
        # Pre-compute divisors of n if n is small enough
        if n < 10000:
            for d in range(2, min(100, root + 1)):
                if n % d == 0:
                    self.get_observation(observer, d)
                    self.get_observation(observer, n // d)
                    
        # Pre-compute positions near small multiples of sqrt
        for k in [0.5, 0.618, 1.0, 1.414, 1.618, 2.0]:
            pos = int(root * k)
            if 2 <= pos <= root:
                self.get_observation(observer, pos)
                
        # Pre-compute gradients for critical positions
        critical_positions = []
        
        # Add Fibonacci positions
        k = 1
        while True:
            f = fib(k)
            if f > min(root, 100):
                break
            if f >= 2:
                critical_positions.append(f)
            k += 1
            
        # Add small primes
        critical_positions.extend(primes_up_to(min(50, root)))
        
        # Pre-compute gradients
        self.precompute_gradients(n, observer, critical_positions)
        
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_hits = self.hits + self.gradient_hits + self.path_hits
        total_misses = self.misses + self.gradient_misses + self.path_misses
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        
        return {
            'observation_cache_size': len(self.observation_cache),
            'gradient_cache_size': len(self.gradient_cache),
            'state_cache_size': len(self.state_cache),
            'path_cache_size': len(self.path_cache),
            'observation_hits': self.hits,
            'observation_misses': self.misses,
            'gradient_hits': self.gradient_hits,
            'gradient_misses': self.gradient_misses,
            'path_hits': self.path_hits,
            'path_misses': self.path_misses,
            'total_hit_rate': hit_rate,
            'precomputed_numbers': len(self.precomputed_fibonacci | self.precomputed_primes | self.precomputed_sqrt)
        }
        
    def clear(self):
        """Clear all caches"""
        self.observation_cache.clear()
        self.gradient_cache.clear()
        self.state_cache.clear()
        self.path_cache.clear()
        self.precomputed_fibonacci.clear()
        self.precomputed_primes.clear()
        self.precomputed_sqrt.clear()
        self.hits = 0
        self.misses = 0
        self.gradient_hits = 0
        self.gradient_misses = 0
        self.path_hits = 0
        self.path_misses = 0
        
    @classmethod
    def create_optimized(cls, n: int) -> 'ObserverCache':
        """
        Create an optimized cache for a specific number
        
        Args:
            n: Number to optimize for
            
        Returns:
            Configured ObserverCache instance
        """
        # Intelligent cache sizing based on problem characteristics
        root = int(math.isqrt(n))
        
        # Base size on sqrt(n) with scaling factors
        if n < 1000:
            cache_size = min(2000, root * 10)
        elif n < 100000:
            cache_size = min(5000, root * 5)
        elif n < 10000000:
            cache_size = min(10000, root * 2)
        else:
            cache_size = min(20000, root)
            
        # Ensure minimum cache size
        cache_size = max(cache_size, 1000)
            
        cache = cls(cache_size=cache_size)
        
        # Link to Axiom 3's spectral cache if available
        try:
            cache.spectral_cache = SpectralSignatureCache.create_optimized(n)
        except:
            pass
            
        return cache
