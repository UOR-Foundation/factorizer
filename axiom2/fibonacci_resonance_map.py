"""
Fibonacci Resonance Map - Acceleration for Fibonacci Flow
Caches Fibonacci calculations, vortex positions, and wave values
"""

import math
from typing import List, Tuple, Dict, Set
from .fibonacci_core import (
    PHI, PSI, SQRT5, GOLDEN_ANGLE,
    fib, fib_wave
)

class FibonacciResonanceMap:
    """
    Accelerates golden ratio calculations and Fibonacci pattern detection
    through intelligent caching and pre-computation.
    """
    
    def __init__(self, max_index: int = 1000):
        """
        Initialize resonance map with caching structures
        
        Args:
            max_index: Maximum Fibonacci index to pre-compute
        """
        self.max_index = max_index
        
        # Caching structures
        self.fib_cache = {}  # k -> fib(k)
        self.vortex_map = {}  # (center, radius) -> vortex_points
        self.wave_table = {}  # x -> fib_wave(x)
        self.spiral_paths = {}  # center -> spiral_points
        
        # Statistics for meta-observation
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_fibonacci(self, k: int) -> int:
        """
        Get k-th Fibonacci number with caching
        
        Args:
            k: Fibonacci index
            
        Returns:
            k-th Fibonacci number
        """
        if k in self.fib_cache:
            self.cache_hits += 1
            return self.fib_cache[k]
        
        self.cache_misses += 1
        
        # Compute using fast doubling algorithm
        fib_k = fib(k)
        
        # Cache the result
        self.fib_cache[k] = fib_k
        
        return fib_k
    
    def get_vortex_points(self, center: int, radius: float) -> Set[int]:
        """
        Get vortex points around a center with caching
        
        Generates points at Fibonacci positions scaled by φ
        
        Args:
            center: Center of vortex
            radius: Maximum radius
            
        Returns:
            Set of vortex point positions
        """
        key = (center, radius)
        if key in self.vortex_map:
            self.cache_hits += 1
            return self.vortex_map[key]
        
        self.cache_misses += 1
        
        # Generate vortex points
        points = set()
        k = 1
        
        while True:
            fib_k = self.get_fibonacci(k)
            if fib_k > radius:
                break
                
            # Basic Fibonacci position
            points.add(center + fib_k)
            points.add(center - fib_k)
            
            # φ-scaled positions
            phi_scaled = int(fib_k * PHI)
            if phi_scaled <= radius:
                points.add(center + phi_scaled)
                points.add(center - phi_scaled)
            
            # 1/φ-scaled positions
            invphi_scaled = int(fib_k / PHI)
            if invphi_scaled > 0:
                points.add(center + invphi_scaled)
                points.add(center - invphi_scaled)
                
            k += 1
        
        # Filter to valid range
        valid_points = {p for p in points if 0 <= p <= 2 * center}
        
        # Cache the result
        if len(self.vortex_map) < 10000:  # Limit cache size
            self.vortex_map[key] = valid_points
            
        return valid_points
    
    def get_wave_value(self, x: float) -> float:
        """
        Get Fibonacci wave value at position x with caching
        
        Uses exact Binet formula, no interpolation
        
        Args:
            x: Position to evaluate
            
        Returns:
            fib_wave(x) value
        """
        if x in self.wave_table:
            self.cache_hits += 1
            return self.wave_table[x]
        
        self.cache_misses += 1
        
        # Compute exact value using Binet's formula
        wave_value = fib_wave(x)
        
        # Cache the result
        if len(self.wave_table) < 100000:  # Limit cache size
            self.wave_table[x] = wave_value
            
        return wave_value
    
    def get_spiral_path(self, center: int, max_radius: float, 
                       steps: int = 100) -> List[Tuple[float, float]]:
        """
        Get golden spiral path points with caching
        
        Args:
            center: Center of spiral
            max_radius: Maximum radius
            steps: Number of points on spiral
            
        Returns:
            List of (x, y) coordinates on spiral
        """
        key = (center, max_radius, steps)
        if key in self.spiral_paths:
            self.cache_hits += 1
            return self.spiral_paths[key]
        
        self.cache_misses += 1
        
        # Generate golden spiral
        path = []
        for k in range(steps):
            angle = k * GOLDEN_ANGLE
            radius = max_radius * (k / steps)
            
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            
            path.append((x, y))
        
        # Cache the result
        if len(self.spiral_paths) < 1000:  # Limit cache size
            self.spiral_paths[key] = path
            
        return path
    
    def precompute_common_values(self):
        """
        Pre-compute commonly used Fibonacci values
        Tuned based on performance analysis
        """
        # Pre-compute Fibonacci numbers (optimal: k < 100)
        for k in range(min(100, self.max_index)):
            self.get_fibonacci(k)
            
        # Pre-compute wave values at integer positions
        # Analysis shows PHI^999 is safe, but we limit to reasonable range
        for k in range(50):  # Direct integer positions
            self.get_wave_value(float(k))
            
        # Pre-compute wave values at small Fibonacci positions
        # Only for fib(k) < 100 to ensure no overflow
        for k in range(14):  # fib(13) = 233, fib(14) = 377
            fib_k = self.get_fibonacci(k)
            if fib_k < 100:
                self.get_wave_value(float(fib_k))
            
        # Pre-compute wave values at key golden ratio positions
        for k in range(1, 20):
            self.get_wave_value(k * PHI)
            self.get_wave_value(k / PHI)
    
    def find_nearest_fibonacci(self, n: int) -> Tuple[int, int, int]:
        """
        Find nearest Fibonacci number to n
        
        Args:
            n: Target number
            
        Returns:
            (index, fibonacci_number, distance)
        """
        k = 0
        while True:
            fib_k = self.get_fibonacci(k)
            if fib_k >= n:
                # Check if previous is closer
                if k > 0:
                    fib_prev = self.get_fibonacci(k - 1)
                    if n - fib_prev < fib_k - n:
                        return (k - 1, fib_prev, n - fib_prev)
                return (k, fib_k, fib_k - n)
            k += 1
    
    def get_fibonacci_range(self, start: int, end: int) -> List[int]:
        """
        Get all Fibonacci numbers in a range
        
        Args:
            start: Start of range (inclusive)
            end: End of range (inclusive)
            
        Returns:
            List of Fibonacci numbers in range
        """
        fibs = []
        k = 0
        
        while True:
            fib_k = self.get_fibonacci(k)
            if fib_k > end:
                break
            if fib_k >= start:
                fibs.append(fib_k)
            k += 1
            
        return fibs
    
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
            'fibonacci_cached': len(self.fib_cache),
            'vortices_cached': len(self.vortex_map),
            'waves_cached': len(self.wave_table),
            'spirals_cached': len(self.spiral_paths)
        }
    
    @classmethod
    def create_optimized(cls, n: int) -> 'FibonacciResonanceMap':
        """
        Create an optimized resonance map for factoring number n
        
        Uses adaptive strategy based on performance tuning:
        - n < 10000: full pre-computation
        - n < 1000000: selective pre-computation  
        - n >= 1000000: lazy evaluation only
        
        Args:
            n: Number to be factored
            
        Returns:
            Optimized FibonacciResonanceMap instance
        """
        # Determine appropriate max index based on n
        max_index = 100
        k = 0
        while k < 200 and fib(k) < n:  # Limit search to k=200
            k += 1
        max_index = min(k + 10, 200)  # Cap at reasonable limit
        
        # Create resonance map
        resonance_map = cls(max_index=max_index)
        
        # Adaptive pre-computation strategy
        if n < 10000:
            # Full pre-computation for small numbers
            resonance_map.precompute_common_values()
        elif n < 1000000:
            # Selective pre-computation for medium numbers
            for k in range(min(50, max_index)):
                resonance_map.get_fibonacci(k)
            # Only pre-compute essential wave values
            for k in range(20):
                resonance_map.get_wave_value(float(k))
        # else: lazy evaluation only for large numbers
            
        return resonance_map
