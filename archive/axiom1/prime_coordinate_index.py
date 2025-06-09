"""
Prime Coordinate Index - Acceleration for Prime Ontology
Caches prime coordinates, geodesic paths, and pull field calculations
"""

from typing import List, Tuple, Dict, Optional
import math
from .prime_core import primes_up_to, is_prime

class PrimeCoordinateIndex:
    """
    Accelerates prime space navigation through pre-computation and intelligent caching.
    Maintains invariant mathematical relationships for O(1) lookup.
    """
    
    def __init__(self, limit: int = 100000, prime_limit: int = 50):
        """
        Initialize the coordinate index with caching structures
        
        Args:
            limit: Maximum number to pre-compute coordinates for
            prime_limit: Number of primes to use in coordinate system (default: 50 for optimal performance)
        """
        self.limit = limit
        self.primes = primes_up_to(prime_limit)
        self.num_primes = len(self.primes)
        
        # Caching structures
        self.coordinates = {}  # n -> [n mod p for p in primes]
        self.geodesic_paths = {}  # (start, end) -> path
        self.pull_field = {}  # position -> pull_value
        
        # Statistics for meta-observation
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_coordinates(self, n: int) -> List[int]:
        """
        Get prime coordinates for number n with caching
        
        Args:
            n: Number to get coordinates for
            
        Returns:
            List of prime coordinates [n mod p for each prime p]
        """
        if n in self.coordinates:
            self.cache_hits += 1
            return self.coordinates[n]
        
        self.cache_misses += 1
        
        # Compute coordinates
        coords = [n % p for p in self.primes]
        
        # Cache if within limit
        if n <= self.limit:
            self.coordinates[n] = coords
            
        return coords
    
    def get_pull(self, x: int, n: int) -> float:
        """
        Get gravitational pull at position x for factoring n
        
        Enhanced to detect coordinate alignment patterns, not just divisibility
        
        Args:
            x: Position to calculate pull for
            n: Number being factored
            
        Returns:
            Gravitational pull value
        """
        # Handle invalid positions
        if x <= 0:
            return 0.0
            
        # Check cache
        cache_key = (x, n)
        if cache_key in self.pull_field:
            self.cache_hits += 1
            return self.pull_field[cache_key]
        
        self.cache_misses += 1
        
        # Calculate enhanced pull
        pull = 0.0
        coords_x = self.get_coordinates(x)
        coords_n = self.get_coordinates(n)
        
        for i, p in enumerate(self.primes):
            # Strong pull when both x and n are divisible by p
            if coords_x[i] == 0 and coords_n[i] == 0:
                pull += 1.0 / p
            # Weaker pull when coordinates match but not at zero
            elif coords_x[i] == coords_n[i]:
                pull += 0.5 / p
                
        # Additional pull for actual divisors (if found)
        if n % x == 0:
            pull += 2.0
            
        # Cache the result
        if len(self.pull_field) < self.limit:
            self.pull_field[cache_key] = pull
            
        return pull
    
    def get_geodesic_path(self, start: int, end: int, n: int) -> Optional[List[int]]:
        """
        Retrieve cached geodesic path or return None if not cached
        
        Args:
            start: Starting position
            end: Target position
            n: Number being factored (for context)
            
        Returns:
            Cached path if available, None otherwise
        """
        cache_key = (start, end, n)
        if cache_key in self.geodesic_paths:
            self.cache_hits += 1
            return self.geodesic_paths[cache_key]
        
        # Try reverse path
        reverse_key = (end, start, n)
        if reverse_key in self.geodesic_paths:
            self.cache_hits += 1
            return list(reversed(self.geodesic_paths[reverse_key]))
            
        return None
    
    def store_geodesic_path(self, start: int, end: int, n: int, path: List[int]):
        """
        Store a successful geodesic path for reuse
        
        Args:
            start: Starting position
            end: Target position
            n: Number being factored
            path: The successful path
        """
        cache_key = (start, end, n)
        
        # Limit cache size
        if len(self.geodesic_paths) < self.limit // 10:
            self.geodesic_paths[cache_key] = path
    
    def find_coordinate_matches(self, target_coords: List[int], 
                               search_range: Tuple[int, int]) -> List[int]:
        """
        Find numbers with similar coordinate patterns
        
        Args:
            target_coords: Target coordinate pattern
            search_range: (min, max) range to search
            
        Returns:
            List of numbers with matching coordinate patterns
        """
        matches = []
        
        for x in range(search_range[0], search_range[1] + 1):
            coords = self.get_coordinates(x)
            
            # Count matching coordinates
            match_count = sum(1 for i in range(self.num_primes) 
                            if coords[i] == target_coords[i])
            
            # High match threshold
            if match_count >= self.num_primes * 0.7:
                matches.append(x)
                
        return matches
    
    def precompute_common_coordinates(self):
        """
        Pre-compute coordinates for commonly accessed numbers
        Optimized to only pre-compute small numbers for faster startup
        """
        # Small numbers only (< 1000) for optimal performance
        for n in range(2, min(1000, self.limit + 1)):
            self.get_coordinates(n)
            
        # Powers of 2 up to reasonable limit
        power = 2
        while power <= min(65536, self.limit):  # 2^16
            self.get_coordinates(power)
            power *= 2
            
        # First 20 primes (most commonly needed)
        for p in self.primes[:20]:
            if p <= self.limit:
                self.get_coordinates(p)
    
    def get_zero_coordinates(self, n: int) -> List[Tuple[int, int]]:
        """
        Find which coordinates are zero for fast divisibility checks
        
        Args:
            n: Number to analyze
            
        Returns:
            List of (prime_index, prime) pairs where n mod prime = 0
        """
        coords = self.get_coordinates(n)
        zeros = []
        
        for i, coord in enumerate(coords):
            if coord == 0:
                zeros.append((i, self.primes[i]))
                
        return zeros
    
    def compute_coordinate_distance(self, x: int, y: int) -> int:
        """
        Compute distance between two numbers in prime coordinate space
        
        Args:
            x: First number
            y: Second number
            
        Returns:
            Manhattan distance in coordinate space
        """
        coords_x = self.get_coordinates(x)
        coords_y = self.get_coordinates(y)
        
        return sum(abs(cx - cy) for cx, cy in zip(coords_x, coords_y))
    
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
            'coordinates_cached': len(self.coordinates),
            'paths_cached': len(self.geodesic_paths),
            'pull_values_cached': len(self.pull_field)
        }
    
    def suggest_exploration_points(self, n: int, current: int) -> List[int]:
        """
        Suggest next exploration points based on coordinate patterns
        
        Args:
            n: Number being factored
            current: Current position
            
        Returns:
            List of suggested positions to explore
        """
        suggestions = []
        current_coords = self.get_coordinates(current)
        
        # Look for positions that share many coordinates with n
        n_coords = self.get_coordinates(n)
        
        # Search in neighborhood
        sqrt_n = int(math.isqrt(n))
        search_radius = max(10, min(sqrt_n // 2, int(math.log(n) * 10)))
        
        # Lower threshold for smaller numbers
        threshold = max(1, int(self.num_primes * 0.2))
        
        for delta in range(-search_radius, search_radius + 1):
            candidate = current + delta
            if 2 <= candidate <= sqrt_n:
                candidate_coords = self.get_coordinates(candidate)
                
                # Count coordinate alignments
                alignments = sum(1 for i in range(min(10, self.num_primes))
                               if candidate_coords[i] == n_coords[i])
                
                if alignments >= threshold:
                    suggestions.append(candidate)
        
        # If no suggestions from coordinate alignment, add some based on prime factors
        if not suggestions:
            # Add positions near small prime factors of n
            for p in self.primes[:10]:
                if n % p == 0 and p <= sqrt_n:
                    suggestions.append(p)
                    if p * 2 <= sqrt_n:
                        suggestions.append(p * 2)
                    if p * 3 <= sqrt_n:
                        suggestions.append(p * 3)
        
        # Also check cached successful positions
        for (start, end, cached_n), path in self.geodesic_paths.items():
            if cached_n > 0 and n > 0 and abs(math.log(n) - math.log(cached_n)) < 1:  # Similar magnitude
                for pos in path:
                    if cached_n > 0:
                        scaled_pos = int(pos * math.sqrt(n) / math.sqrt(cached_n))
                        if 2 <= scaled_pos <= sqrt_n and scaled_pos not in suggestions:
                            suggestions.append(scaled_pos)
        
        return sorted(set(suggestions))[:10]  # Return top 10 unique suggestions
    
    @classmethod
    def create_optimized(cls, n: int) -> 'PrimeCoordinateIndex':
        """
        Create an optimized index for factoring number n
        
        Automatically selects appropriate parameters based on n
        
        Args:
            n: Number to be factored
            
        Returns:
            Optimized PrimeCoordinateIndex instance
        """
        # Determine appropriate limit
        limit = min(100000, max(10000, n))
        
        # Use 50 primes for optimal balance
        prime_limit = 50
        
        # Create and initialize index
        index = cls(limit=limit, prime_limit=prime_limit)
        
        # Only pre-compute if n is small
        if n < 10000:
            index.precompute_common_coordinates()
            
        return index
