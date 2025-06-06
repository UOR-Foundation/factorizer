"""
Spectral Signature Cache - Acceleration for Duality Principle
Caches spectral vectors, coherence values, and interference patterns
"""

import math
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict

# Import spectral computation functions
from .spectral_core import spectral_vector
from .coherence import coherence, CoherenceCache
from .interference import prime_fib_interference, interference_extrema
from .fold_topology import FoldTopology

# Import axiom integration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom1 import primes_up_to, is_prime
from axiom2 import fib, PHI


class SpectralSignatureCache:
    """
    Accelerates spectral analysis through intelligent caching
    of spectral signatures and derived computations.
    """
    
    def __init__(self, cache_size: int = 10000):
        """
        Initialize cache with specified size limit
        
        Args:
            cache_size: Maximum number of entries per cache type
        """
        self.cache_size = cache_size
        
        # LRU caches using OrderedDict
        self.spectral_cache = OrderedDict()  # n -> S(n)
        self.coherence_cache = OrderedDict()  # (a,b,n) -> coherence
        self.interference_bank = OrderedDict()  # n -> (pattern, extrema)
        self.fold_map = OrderedDict()  # n -> {pos: energy}
        
        # Statistics for meta-observation
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Pre-compute priority numbers
        self._precompute_priority_numbers()
        
    def _precompute_priority_numbers(self):
        """Pre-compute spectral vectors for priority numbers"""
        # Minimal pre-computation to avoid startup overhead
        # Focus on the most reusable values only
        
        # Small Fibonacci numbers (very common)
        for k in range(1, 12):  # fib(11) = 89
            self._ensure_spectral_cached(fib(k))
            
        # Very small primes (most common factors)
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            self._ensure_spectral_cached(p)
            
        # Small powers of 2
        for i in range(10):  # Up to 512
            self._ensure_spectral_cached(2**i)
            
    def _ensure_spectral_cached(self, n: int):
        """Ensure spectral vector is cached for n"""
        if n not in self.spectral_cache:
            self.spectral_cache[n] = spectral_vector(n)
            self._enforce_cache_limit(self.spectral_cache)
            
    def _enforce_cache_limit(self, cache: OrderedDict):
        """Enforce LRU eviction when cache exceeds size limit"""
        while len(cache) > self.cache_size:
            # Remove least recently used (first item)
            cache.popitem(last=False)
            
    def get_spectral_vector(self, n: int) -> List[float]:
        """
        Get spectral vector with caching
        
        Args:
            n: Number to analyze
            
        Returns:
            Spectral vector S(n)
        """
        if n in self.spectral_cache:
            self.cache_hits += 1
            # Move to end (most recently used)
            self.spectral_cache.move_to_end(n)
            return self.spectral_cache[n]
            
        self.cache_misses += 1
        
        # Compute spectral vector
        spec_vector = spectral_vector(n)
        
        # Cache it
        self.spectral_cache[n] = spec_vector
        self._enforce_cache_limit(self.spectral_cache)
        
        return spec_vector
        
    def get_coherence(self, a: int, b: int, n: int) -> float:
        """
        Get coherence value with caching and symmetry exploitation
        
        Args:
            a, b: Numbers to check coherence
            n: Target number
            
        Returns:
            Coherence value C(a,b,n)
        """
        # Exploit symmetry: C(a,b,n) = C(b,a,n)
        key = tuple(sorted([a, b]) + [n])
        
        if key in self.coherence_cache:
            self.cache_hits += 1
            self.coherence_cache.move_to_end(key)
            return self.coherence_cache[key]
            
        self.cache_misses += 1
        
        # Get spectral vectors (may hit cache)
        s_a = self.get_spectral_vector(a)
        s_b = self.get_spectral_vector(b)
        s_n = self.get_spectral_vector(n)
        
        # Compute coherence
        # C(a,b,n) = exp(-||S(a)+S(b)-2S(n)||²)
        diff_squared = 0
        for i in range(len(s_a)):
            diff = s_a[i] + s_b[i] - 2 * s_n[i]
            diff_squared += diff * diff
            
        coherence = math.exp(-diff_squared)
        
        # Cache it
        self.coherence_cache[key] = coherence
        self._enforce_cache_limit(self.coherence_cache)
        
        return coherence
        
    def get_interference_pattern(self, n: int) -> Tuple[List[float], List[int]]:
        """
        Get interference pattern with caching
        
        Args:
            n: Number to analyze
            
        Returns:
            (interference_values, extrema_positions)
        """
        if n in self.interference_bank:
            self.cache_hits += 1
            self.interference_bank.move_to_end(n)
            return self.interference_bank[n]
            
        self.cache_misses += 1
        
        # Get the interference pattern
        pattern = prime_fib_interference(n)
        
        # Get extrema positions
        extrema = interference_extrema(n)
                
        # Cache it
        self.interference_bank[n] = (pattern, extrema)
        self._enforce_cache_limit(self.interference_bank)
        
        return pattern, extrema
        
    def get_fold_energy(self, n: int, x: int) -> float:
        """
        Get fold energy with caching
        
        Args:
            n: Number being analyzed
            x: Position to evaluate
            
        Returns:
            Fold energy E(x)
        """
        if n in self.fold_map and x in self.fold_map[n]:
            self.cache_hits += 1
            return self.fold_map[n][x]
            
        self.cache_misses += 1
        
        # Initialize fold map for n if needed
        if n not in self.fold_map:
            self.fold_map[n] = {}
            
        # Get spectral vectors (may hit cache)
        s_x = self.get_spectral_vector(x)
        s_n = self.get_spectral_vector(n)
        
        # Handle division
        if n % x == 0:
            y = n // x
        else:
            # For non-factors, use approximate complementary value
            y = n / x
            
        s_nx = self.get_spectral_vector(int(y))
        
        # E(x) = ||S(x) + S(n/x) - 2*S(n)||²
        energy = 0
        for i in range(len(s_x)):
            diff = s_x[i] + s_nx[i] - 2 * s_n[i]
            energy += diff * diff
            
        # Cache it
        self.fold_map[n][x] = energy
        
        # Limit fold map entries per n
        if len(self.fold_map[n]) > 1000:
            # Keep most important positions
            sorted_positions = sorted(self.fold_map[n].items(), key=lambda x: x[1])[:500]
            self.fold_map[n] = dict(sorted_positions)
            
        self._enforce_cache_limit(self.fold_map)
        
        return energy
        
    def get_sharp_folds(self, n: int) -> List[int]:
        """
        Get pre-identified sharp fold candidates
        
        Args:
            n: Number to analyze
            
        Returns:
            List of sharp fold positions
        """
        # Check if we have enough fold data cached
        if n not in self.fold_map or len(self.fold_map[n]) < 10:
            # Build fold map
            sqrt_n = int(math.isqrt(n))
            for x in range(2, min(sqrt_n + 1, 100)):
                self.get_fold_energy(n, x)
                
        # Find sharp folds from cached data
        if n in self.fold_map:
            energies = [(x, e) for x, e in self.fold_map[n].items() 
                       if e != float('inf')]
            energies.sort(key=lambda x: x[1])
            
            # Return positions with lowest energy
            return [x for x, _ in energies[:10]]
        
        return []
        
    def precompute_for_n(self, n: int):
        """
        Minimal pre-computation for a specific n
        
        Only pre-compute what's absolutely necessary to avoid overhead.
        Let the cache fill naturally during use.
        
        Args:
            n: Number to pre-compute for
        """
        # Only pre-compute n itself
        self.get_spectral_vector(n)
            
    def get_cache_statistics(self) -> Dict[str, any]:
        """
        Get cache performance statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'spectral_cached': len(self.spectral_cache),
            'coherence_cached': len(self.coherence_cache),
            'interference_cached': len(self.interference_bank),
            'fold_maps_cached': len(self.fold_map),
            'total_entries': (len(self.spectral_cache) + 
                            len(self.coherence_cache) +
                            len(self.interference_bank) +
                            len(self.fold_map))
        }
        
    @classmethod
    def create_optimized(cls, n: int) -> 'SpectralSignatureCache':
        """
        Create an optimized cache for analyzing number n
        
        Pure acceleration through lazy caching:
        - No unnecessary pre-computation overhead
        - Cache fills naturally with actual access patterns
        - Focuses on reusability, not prediction
        
        Args:
            n: Number to be analyzed
            
        Returns:
            Optimized SpectralSignatureCache instance
        """
        # Adaptive cache size based on n
        if n < 10000:
            cache_size = 5000
        elif n < 1000000:
            cache_size = 10000  
        else:
            cache_size = 20000
            
        cache = cls(cache_size=cache_size)
        
        # Only cache n itself - everything else fills on demand
        # This ensures pure acceleration without overhead
        cache.get_spectral_vector(n)
        
        return cache
