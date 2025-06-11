"""
Prime Coordinate System - Adaptive prime space mapping

Implements the foundation of Axiom 1 for the Prime Sieve.
Maps numbers into an infinite-dimensional coordinate space where
each dimension corresponds to a prime number.
"""

import math
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

# Import prime generation from axiom1
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from axiom1.prime_core import primes_up_to, is_prime


@dataclass
class CoordinateAlignment:
    """Represents alignment between two coordinate vectors"""
    position: int
    alignment_score: float
    zero_alignments: List[int]  # Prime indices where both coords are 0
    nonzero_alignments: List[int]  # Prime indices where coords match but != 0
    pull_value: float


class PrimeCoordinateSystem:
    """
    Adaptive prime coordinate system that scales with input size.
    
    Key features:
    - Dynamic prime basis generation based on bit length
    - Coordinate alignment detection
    - Enhanced pull field calculations
    - Memory-efficient sparse representations
    """
    
    def __init__(self, n: int):
        """
        Initialize coordinate system for number n.
        
        Args:
            n: Number to be factored
        """
        self.n = n
        self.bit_length = n.bit_length()
        self.sqrt_n = int(math.isqrt(n))
        
        # Generate adaptive prime basis
        self.prime_count = self._determine_prime_count()
        self.primes = self._generate_adaptive_primes()
        
        # Cache for coordinate calculations
        self.coordinate_cache: Dict[int, List[int]] = {}
        self.pull_cache: Dict[Tuple[int, int], float] = {}
        
        # Pre-compute n's coordinates
        self.n_coords = self.get_coordinates(n)
        
    def _determine_prime_count(self) -> int:
        """
        Determine optimal number of primes based on bit length.
        
        Scales logarithmically for massive numbers to maintain efficiency.
        """
        if self.bit_length < 64:
            return 50
        elif self.bit_length < 256:
            return self.bit_length
        elif self.bit_length < 1024:
            return int(self.bit_length * math.log(self.bit_length))
        else:
            # Logarithmic scaling for massive numbers
            return int(self.bit_length * math.log(math.log(self.bit_length)))
    
    def _generate_adaptive_primes(self) -> List[int]:
        """
        Generate prime basis adapted to the problem size.
        
        Returns:
            List of primes forming the coordinate basis
        """
        # Cap at 10000 primes for memory efficiency
        actual_count = min(self.prime_count, 10000)
        
        # For very large numbers, we might need primes beyond typical ranges
        if self.bit_length > 1024:
            # Use logarithmically spaced primes for massive numbers
            max_prime_needed = min(self.sqrt_n, 10**9)
            return self._generate_sparse_primes(actual_count, max_prime_needed)
        else:
            # Standard prime generation for smaller numbers
            primes = primes_up_to(min(actual_count * 20, 200000))
            return primes[:actual_count]
    
    def _generate_sparse_primes(self, count: int, max_prime: int) -> List[int]:
        """
        Generate logarithmically spaced primes for very large numbers.
        
        Args:
            count: Number of primes needed
            max_prime: Maximum prime value to consider
            
        Returns:
            List of primes with good coverage
        """
        # Start with small primes for fine-grained analysis
        primes = primes_up_to(min(1000, count))
        
        if len(primes) >= count:
            return primes[:count]
        
        # Add logarithmically spaced larger primes
        current = 1000
        while len(primes) < count and current < max_prime:
            # Find next prime after current
            while current < max_prime and not is_prime(current):
                current += 1
            
            if current < max_prime:
                primes.append(current)
                # Logarithmic spacing
                current = int(current * 1.1)
        
        return primes[:count]
    
    def get_coordinates(self, x: int) -> List[int]:
        """
        Get prime coordinates for number x.
        
        Coordinates are [x mod p1, x mod p2, x mod p3, ...]
        
        Args:
            x: Number to get coordinates for
            
        Returns:
            List of modular residues
        """
        if x in self.coordinate_cache:
            return self.coordinate_cache[x]
        
        coords = [x % p for p in self.primes]
        
        # Cache if within reasonable size
        if len(self.coordinate_cache) < 100000:
            self.coordinate_cache[x] = coords
        
        return coords
    
    def calculate_alignment(self, x: int) -> CoordinateAlignment:
        """
        Calculate alignment between x and n in coordinate space.
        
        Args:
            x: Position to check alignment
            
        Returns:
            CoordinateAlignment object with alignment metrics
        """
        x_coords = self.get_coordinates(x)
        
        zero_alignments = []
        nonzero_alignments = []
        alignment_score = 0.0
        
        for i, (xc, nc, p) in enumerate(zip(x_coords, self.n_coords, self.primes)):
            if xc == nc:
                if xc == 0:
                    zero_alignments.append(i)
                    alignment_score += 1.0 / p  # Stronger weight for zero alignment
                else:
                    nonzero_alignments.append(i)
                    alignment_score += 0.5 / p  # Weaker weight for non-zero alignment
        
        # Calculate enhanced pull
        pull = self._calculate_pull(x, zero_alignments, nonzero_alignments)
        
        return CoordinateAlignment(
            position=x,
            alignment_score=alignment_score,
            zero_alignments=zero_alignments,
            nonzero_alignments=nonzero_alignments,
            pull_value=pull
        )
    
    def _calculate_pull(self, x: int, zero_aligns: List[int], 
                       nonzero_aligns: List[int]) -> float:
        """
        Calculate gravitational pull at position x.
        
        Enhanced pull formula:
        Pull(x, n) = Σᵢ (1/pᵢ) × I[coords match at 0] + Σᵢ (0.5/pᵢ) × I[coords match != 0]
        
        Args:
            x: Position
            zero_aligns: Indices where both coordinates are 0
            nonzero_aligns: Indices where coordinates match but aren't 0
            
        Returns:
            Pull value
        """
        cache_key = (x, self.n)
        if cache_key in self.pull_cache:
            return self.pull_cache[cache_key]
        
        pull = 0.0
        
        # Strong pull from zero alignments
        for i in zero_aligns:
            pull += 1.0 / self.primes[i]
        
        # Weaker pull from non-zero alignments
        for i in nonzero_aligns:
            pull += 0.5 / self.primes[i]
        
        # Additional pull if x actually divides n
        if self.n % x == 0:
            pull += 2.0
        
        # Cache the result
        if len(self.pull_cache) < 50000:
            self.pull_cache[cache_key] = pull
        
        return pull
    
    def find_aligned_positions(self, search_range: Tuple[int, int], 
                             threshold: Optional[float] = None) -> List[CoordinateAlignment]:
        """
        Find positions with high coordinate alignment in given range.
        
        Args:
            search_range: (start, end) range to search
            threshold: Minimum alignment score (auto-determined if None)
            
        Returns:
            List of aligned positions sorted by alignment score
        """
        if threshold is None:
            # Adaptive threshold based on bit length
            threshold = self._adaptive_threshold()
        
        aligned_positions = []
        
        # For large ranges, use sparse sampling
        step = 1
        if search_range[1] - search_range[0] > 100000:
            # Golden ratio sampling for large ranges
            step = int((search_range[1] - search_range[0]) / 10000)
            step = max(1, step)
        
        # Allow searching slightly beyond sqrt_n for close factors
        max_search = int(self.sqrt_n * 1.1)
        
        for x in range(search_range[0], min(search_range[1] + 1, max_search), step):
            if x <= 1:
                continue
            
            alignment = self.calculate_alignment(x)
            
            if alignment.alignment_score >= threshold:
                aligned_positions.append(alignment)
        
        # Sort by alignment score (descending)
        aligned_positions.sort(key=lambda a: a.alignment_score, reverse=True)
        
        return aligned_positions
    
    def _adaptive_threshold(self) -> float:
        """
        Determine adaptive threshold based on number characteristics.
        
        Returns:
            Threshold value for alignment detection
        """
        # Base threshold decreases with bit length
        base = 1.0 / math.log2(self.bit_length + 2)
        
        # Adjust based on prime density
        prime_density = len(self.primes) / max(1, self.prime_count)
        
        return base * prime_density * 0.5
    
    def suggest_search_positions(self, current: int, radius: int = 1000) -> List[int]:
        """
        Suggest positions to search based on coordinate patterns.
        
        Args:
            current: Current search position
            radius: Search radius around current
            
        Returns:
            List of suggested positions
        """
        suggestions = set()
        
        # Find positions with similar coordinate patterns
        current_alignment = self.calculate_alignment(current)
        
        # Look for positions that share zero alignments
        for x in range(max(2, current - radius), min(self.sqrt_n, current + radius + 1)):
            x_alignment = self.calculate_alignment(x)
            
            # Count shared zero alignments
            shared_zeros = len(set(current_alignment.zero_alignments) & 
                             set(x_alignment.zero_alignments))
            
            if shared_zeros >= len(current_alignment.zero_alignments) * 0.5:
                suggestions.add(x)
        
        # Add positions based on small prime factors
        for i, p in enumerate(self.primes[:10]):
            if self.n_coords[i] == 0:  # n is divisible by p
                # Check multiples of p near current
                base = (current // p) * p
                for k in range(-5, 6):
                    candidate = base + k * p
                    if 2 <= candidate <= self.sqrt_n:
                        suggestions.add(candidate)
        
        return sorted(list(suggestions))
    
    def coordinate_sieve(self, candidates: Optional[Set[int]] = None) -> Set[int]:
        """
        Apply coordinate-based sieving to filter candidates.
        
        Args:
            candidates: Initial candidate set (None for full range)
            
        Returns:
            Filtered candidates passing coordinate alignment test
        """
        if candidates is None:
            # Start with positions having high alignment
            aligned = self.find_aligned_positions((2, min(10000, self.sqrt_n)))
            candidates = {a.position for a in aligned[:1000]}
        
        # Filter based on coordinate alignment
        threshold = self._adaptive_threshold()
        filtered = set()
        
        for x in candidates:
            alignment = self.calculate_alignment(x)
            if alignment.alignment_score >= threshold:
                filtered.add(x)
        
        return filtered
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the coordinate system.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            'bit_length': self.bit_length,
            'prime_count': len(self.primes),
            'max_prime': self.primes[-1] if self.primes else 0,
            'coordinate_cache_size': len(self.coordinate_cache),
            'pull_cache_size': len(self.pull_cache),
            'n_zero_coordinates': sum(1 for c in self.n_coords if c == 0),
            'adaptive_threshold': self._adaptive_threshold()
        }
