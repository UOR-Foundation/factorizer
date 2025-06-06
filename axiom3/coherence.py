"""
Coherence - Measure spectral alignment between numbers
Implements coherence calculation with caching for efficiency
"""

import math
from typing import Dict, Tuple, Optional, List
from .spectral_core import spectral_vector

def coherence(a: int, b: int, n: int) -> float:
    """
    Calculate coherence between numbers a, b and their product n
    
    Coherence measures how well the spectral signatures of a and b
    combine to match the spectral signature of n. High coherence
    suggests a×b = n.
    
    Formula: C(a,b,n) = exp(-||S(a)+S(b)-2S(n)||²)
    
    Args:
        a: First number
        b: Second number  
        n: Target number (ideally a×b)
        
    Returns:
        Coherence value in range [0, 1]
    """
    # Get spectral vectors
    sa = spectral_vector(a)
    sb = spectral_vector(b)
    sn = spectral_vector(n)
    
    # Calculate squared distance
    # For perfect factorization, we expect S(a) + S(b) ≈ 2*S(n)
    # So we measure ||S(a) + S(b) - 2*S(n)||²
    squared_distance = 0.0
    for i in range(len(sa)):
        diff = sa[i] + sb[i] - 2 * sn[i]
        squared_distance += diff * diff
    
    # Convert to coherence using exponential decay
    return math.exp(-squared_distance)

class CoherenceCache:
    """
    Cache coherence calculations for efficiency
    
    Since spectral vector computation can be expensive for large numbers,
    this cache stores computed values to avoid redundant calculations.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize coherence cache
        
        Args:
            max_size: Maximum number of entries to cache
        """
        self.max_size = max_size
        self.spectral_cache: Dict[int, List[float]] = {}
        self.coherence_cache: Dict[Tuple[int, int, int], float] = {}
    
    def get_spectral(self, n: int) -> List[float]:
        """
        Get spectral vector with caching
        
        Args:
            n: Number to get spectrum for
            
        Returns:
            Spectral vector
        """
        if n not in self.spectral_cache:
            # Check cache size
            if len(self.spectral_cache) >= self.max_size:
                # Simple eviction: remove first entry
                first_key = next(iter(self.spectral_cache))
                del self.spectral_cache[first_key]
            
            self.spectral_cache[n] = spectral_vector(n)
        
        return self.spectral_cache[n]
    
    def get_coherence(self, a: int, b: int, n: int) -> float:
        """
        Get coherence with caching
        
        Args:
            a: First number
            b: Second number
            n: Target number
            
        Returns:
            Coherence value
        """
        # Create canonical key (order doesn't matter for a, b)
        key = (min(a, b), max(a, b), n)
        
        if key not in self.coherence_cache:
            # Check cache size
            if len(self.coherence_cache) >= self.max_size:
                # Simple eviction: remove first entry
                first_key = next(iter(self.coherence_cache))
                del self.coherence_cache[first_key]
            
            # Get spectral vectors (with caching)
            sa = self.get_spectral(a)
            sb = self.get_spectral(b)
            sn = self.get_spectral(n)
            
            # Calculate coherence
            squared_distance = 0.0
            for i in range(len(sa)):
                diff = sa[i] + sb[i] - 2 * sn[i]
                squared_distance += diff * diff
            
            self.coherence_cache[key] = math.exp(-squared_distance)
        
        return self.coherence_cache[key]
    
    def clear(self):
        """Clear all cached values"""
        self.spectral_cache.clear()
        self.coherence_cache.clear()

def triple_coherence(p: int, q: int, r: int, n: int) -> float:
    """
    Calculate coherence for triple product p×q×r = n
    
    This extends the coherence concept to three factors,
    useful for detecting cases where n has three prime factors.
    
    Args:
        p: First factor
        q: Second factor
        r: Third factor
        n: Target number
        
    Returns:
        Triple coherence value
    """
    # Get spectral vectors
    sp = spectral_vector(p)
    sq = spectral_vector(q)
    sr = spectral_vector(r)
    sn = spectral_vector(n)
    
    # For triple product, we expect S(p) + S(q) + S(r) ≈ 3×S(n)
    squared_distance = 0.0
    for i in range(len(sp)):
        diff = sp[i] + sq[i] + sr[i] - 3 * sn[i]
        squared_distance += diff * diff
    
    return math.exp(-squared_distance)
