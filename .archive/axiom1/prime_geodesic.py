"""
Prime Geodesic - Navigate through prime coordinate space
Implements prime coordinates and geodesic paths to factors
"""

import math
from typing import List, Optional
from .prime_core import primes_up_to, is_prime
from .prime_coordinate_index import PrimeCoordinateIndex
from .prime_cascade import PrimeCascade

class PrimeGeodesic:
    """
    Navigate number space using prime coordinates
    Each number has coordinates: [n mod 2, n mod 3, n mod 5, ...]
    Geodesics follow paths of maximum prime attraction
    """
    
    def __init__(self, n: int, coordinate_index: Optional[PrimeCoordinateIndex] = None):
        """
        Initialize geodesic system for number n
        
        Args:
            n: The number being factored
            coordinate_index: Optional acceleration cache
        """
        self.n = n
        self.sqrt_n = int(math.isqrt(n))
        
        # Use coordinate index for acceleration if provided
        self.coord_index = coordinate_index
        if self.coord_index is None:
            # Create optimized index automatically
            self.coord_index = PrimeCoordinateIndex.create_optimized(n)
            
        # Get prime coordinates through index
        self.coord = self.coord_index.get_coordinates(n)
        self.primes = self.coord_index.primes
        
        # Initialize cascade for prime relationship exploration
        self.cascade = PrimeCascade(n)
    
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
        # Use accelerated lookup
        return self.coord_index.get_pull(x, self.n)
    
    def walk(self, start: int, steps: int = 50) -> List[int]:
        """
        Enhanced walk along geodesic path from starting position
        
        Uses multi-scale search, prime cascades, and momentum to find factors
        
        Args:
            start: Starting position
            steps: Maximum steps to take
            
        Returns:
            Path taken as list of positions
        """
        # Check for cached path first
        for potential_factor in [p for p in self.primes if p <= self.sqrt_n]:
            cached_path = self.coord_index.get_geodesic_path(start, potential_factor, self.n)
            if cached_path:
                # Verify the path leads to a factor
                for pos in cached_path:
                    if self.n % pos == 0 and is_prime(pos):
                        return cached_path
        
        # Get exploration suggestions from coordinate patterns
        suggestions = self.coord_index.suggest_exploration_points(self.n, start)
        
        path = [start]
        cur = start
        momentum = 0  # Track direction of improvement
        last_direction = 0
        
        for step_num in range(min(steps, 100)):  # Allow more steps for exploration
            best, best_s = cur, self._pull(cur)
            candidates = []
            
            # 1. Multi-scale search - check at different scales
            scales = [1, 2, 3, 5, 7, 11]  # Prime-based scales
            for scale in scales:
                for direction in [-1, 1]:
                    cand = cur + direction * scale
                    if 2 <= cand <= self.sqrt_n:
                        candidates.append(cand)
            
            # 2. Prime cascade exploration
            if is_prime(cur):
                cascade_primes = self.cascade.cascade(cur)
                for cp in cascade_primes:
                    if 2 <= cp <= self.sqrt_n:
                        candidates.append(cp)
            
            # 3. Add suggestions from coordinate analysis
            candidates.extend([s for s in suggestions if abs(s - cur) <= 30])
            
            # 4. Add momentum-based candidates
            if momentum > 0 and last_direction != 0:
                # Continue in successful direction with larger steps
                momentum_cand = cur + last_direction * min(momentum * 2, 20)
                if 2 <= momentum_cand <= self.sqrt_n:
                    candidates.append(momentum_cand)
            
            # Evaluate all candidates
            for cand in set(candidates):  # Remove duplicates
                s = self._pull(cand)
                if s > best_s:
                    best, best_s = cand, s
            
            # Update momentum and direction
            if best != cur:
                new_direction = 1 if best > cur else -1
                if new_direction == last_direction:
                    momentum += 1
                else:
                    momentum = 1
                last_direction = new_direction
            else:
                # No improvement found - try escape strategies
                if step_num < steps - 10:  # Still have steps left
                    # Jump to a prime multiple
                    for p in self.primes[:10]:
                        escape_pos = cur * p
                        if escape_pos <= self.sqrt_n:
                            candidates.append(escape_pos)
                        escape_pos = cur // p if cur % p == 0 else 0
                        if escape_pos >= 2:
                            candidates.append(escape_pos)
                    
                    # Re-evaluate with escape candidates
                    for cand in candidates:
                        if 2 <= cand <= self.sqrt_n:
                            s = self._pull(cand)
                            if s >= best_s * 0.8:  # Accept slight decrease
                                best = cand
                                momentum = 0
                                break
                
                if best == cur:  # Still stuck
                    break
            
            cur = best
            path.append(cur)
            
            # Check if we found a factor
            if self.n % cur == 0 and is_prime(cur):
                # Cache successful path
                self.coord_index.store_geodesic_path(start, cur, self.n, path)
                break
            
            # Periodically update suggestions based on current position
            if step_num % 10 == 0:
                suggestions = self.coord_index.suggest_exploration_points(self.n, cur)
        
        return path
