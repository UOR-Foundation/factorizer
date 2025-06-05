"""
Prime Geodesic - Navigate through prime coordinate space
Implements prime coordinates and geodesic paths to factors
"""

import math
from typing import List
from .prime_core import primes_up_to

class PrimeGeodesic:
    """
    Navigate number space using prime coordinates
    Each number has coordinates: [n mod 2, n mod 3, n mod 5, ...]
    Geodesics follow paths of maximum prime attraction
    """
    
    def __init__(self, n: int):
        """
        Initialize geodesic system for number n
        
        Args:
            n: The number being factored
        """
        self.n = n
        # Compute prime coordinates for n
        self.coord = [n % p for p in primes_up_to(min(1000, n))[:100]]  # Limit to first 100 primes
    
    def _pull(self, x: int) -> float:
        """
        Calculate gravitational-like pull toward prime factors
        
        The pull is stronger when:
        - x shares prime factors with n (coordinate is 0)
        - x is divisible by smaller primes (1/p weighting)
        
        Args:
            x: Position to calculate pull for
            
        Returns:
            Pull strength as float
        """
        pull = 0.0
        
        # Check alignment with prime coordinates
        for i, p in enumerate(primes_up_to(min(30, x))):
            if i < len(self.coord) and self.coord[i] == 0 and x % p == 0:
                # Stronger pull for smaller primes
                pull += 1 / p
        
        return pull
    
    def walk(self, start: int, steps: int = 20) -> List[int]:
        """
        Walk along geodesic path from starting position
        
        Follows the path of steepest descent in prime space,
        moving toward positions with maximum prime attraction
        
        Args:
            start: Starting position
            steps: Maximum steps to take
            
        Returns:
            Path taken as list of positions
        """
        path = [start]
        cur = start
        root = int(math.isqrt(self.n))
        
        for _ in range(min(steps, 30)):  # Limit maximum steps
            best, best_s = cur, 0
            
            # Check neighboring positions
            for d in (-3, -2, -1, 1, 2, 3):
                cand = cur + d
                if 2 <= cand <= root:
                    s = self._pull(cand)
                    if s > best_s:
                        best, best_s = cand, s
            
            # Stop if no improvement found
            if best == cur:
                break
            
            cur = best
            path.append(cur)
        
        return path
