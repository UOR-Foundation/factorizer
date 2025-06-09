"""
Meta-Acceleration Cache - Accelerates self-referential operations
Intelligent caching of meta-level computations and pattern recognition
"""

import math
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from sortedcontainers import SortedList
import hashlib

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom2 import PHI
from axiom3 import accelerated_spectral_vector, accelerated_coherence
from axiom4 import ObserverCache

class MetaAccelerationCache:
    """
    Accelerates self-referential operations through intelligent caching
    """
    
    def __init__(self, cache_size: int = 20000):
        """
        Initialize meta-acceleration cache
        
        Args:
            cache_size: Maximum cache size
        """
        self.cache_size = cache_size
        
        # Meta-coherence cache
        self.meta_coherence_cache: Dict[Tuple[int, str], float] = {}
        
        # Observation index for O(1) lookups
        self.observation_index = {
            'by_position': defaultdict(list),
            'by_axiom': defaultdict(list),
            'by_coherence': SortedList(key=lambda x: x[1])  # (observation, coherence)
        }
        
        # Recursive coherence memory
        self.coherence_fields: Dict[Tuple[int, int], Dict[int, float]] = {}  # (n, iteration) -> field
        self.fixed_points: Dict[int, List[int]] = {}  # n -> positions
        self.attractor_cache: Dict[Tuple[int, int], List[int]] = {}  # (n, initial) -> attractors
        
        # Spectral mirror cache
        self.mirror_map: Dict[Tuple[int, int], int] = {}  # (n, pos) -> mirror_pos
        self.spectral_distances: Dict[Tuple[int, int], float] = {}  # (x, y) -> distance
        self.recursive_mirrors: Dict[Tuple[int, int, int], List[int]] = {}  # (n, start, depth) -> sequence
        
        # Pattern cache
        self.axiom_combinations: Dict[str, float] = {}  # pattern_hash -> success_rate
        self.interference_matrix: Dict[Tuple[str, str], float] = {}  # (axiom1, axiom2) -> strength
        
        # LRU tracking
        self.access_count = defaultdict(int)
        self.access_order = []
        
    def _hash_field(self, field: Dict[int, float]) -> str:
        """
        Create hash of coherence field for caching
        
        Args:
            field: Coherence field
            
        Returns:
            Hash string
        """
        # Sort by position for consistent hashing
        items = sorted(field.items())
        field_str = str([(pos, round(coh, 6)) for pos, coh in items])
        return hashlib.md5(field_str.encode()).hexdigest()
    
    def _evict_if_needed(self):
        """Evict least recently used items if cache is full"""
        if len(self.access_order) > self.cache_size:
            # Find least accessed items
            to_remove = self.access_order[:len(self.access_order) - self.cache_size]
            self.access_order = self.access_order[len(self.access_order) - self.cache_size:]
            
            # Remove from caches
            for key in to_remove:
                if isinstance(key, tuple) and len(key) == 2:
                    # Meta-coherence cache
                    self.meta_coherence_cache.pop(key, None)
                # Add other cache evictions as needed
    
    def get_meta_coherence(self, position: int, coherence_field: Dict[int, float], 
                          compute_func: Optional[callable] = None) -> float:
        """
        Get meta-coherence with caching
        
        Args:
            position: Position to evaluate
            coherence_field: Coherence field
            compute_func: Function to compute if not cached
            
        Returns:
            Meta-coherence value
        """
        field_hash = self._hash_field(coherence_field)
        key = (position, field_hash)
        
        if key in self.meta_coherence_cache:
            self.access_count[key] += 1
            return self.meta_coherence_cache[key]
        
        if compute_func:
            value = compute_func(position, coherence_field)
            self.meta_coherence_cache[key] = value
            self.access_order.append(key)
            self._evict_if_needed()
            return value
        
        return 0.0
    
    def add_observation(self, observation: Dict[str, Any]):
        """
        Add observation to indexed storage
        
        Args:
            observation: Observation dict with position, coherence, axiom, etc.
        """
        # Index by position
        if 'position' in observation:
            self.observation_index['by_position'][observation['position']].append(observation)
        
        # Index by axiom
        if 'axiom' in observation:
            self.observation_index['by_axiom'][observation['axiom']].append(observation)
        
        # Index by coherence
        if 'coherence' in observation:
            self.observation_index['by_coherence'].add((observation, observation['coherence']))
    
    def query_observations(self, position: Optional[int] = None, 
                          axiom: Optional[str] = None,
                          min_coherence: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Query observations with O(1) lookup
        
        Args:
            position: Filter by position
            axiom: Filter by axiom
            min_coherence: Minimum coherence threshold
            
        Returns:
            List of matching observations
        """
        results = []
        
        if position is not None:
            results = self.observation_index['by_position'][position]
        elif axiom is not None:
            results = self.observation_index['by_axiom'][axiom]
        elif min_coherence is not None:
            # Use sorted list for range query
            results = [obs for obs, coh in self.observation_index['by_coherence'] 
                      if coh >= min_coherence]
        else:
            # Return all
            results = [obs for obs, _ in self.observation_index['by_coherence']]
        
        return results
    
    def get_coherence_field(self, n: int, iteration: int, 
                           compute_func: Optional[callable] = None) -> Optional[Dict[int, float]]:
        """
        Get cached coherence field
        
        Args:
            n: Number being factored
            iteration: Iteration number
            compute_func: Function to compute if not cached
            
        Returns:
            Coherence field or None
        """
        key = (n, iteration)
        
        if key in self.coherence_fields:
            return self.coherence_fields[key]
        
        if compute_func:
            field = compute_func(n, iteration)
            self.coherence_fields[key] = field
            return field
        
        return None
    
    def get_fixed_points(self, n: int) -> List[int]:
        """
        Get cached fixed points
        
        Args:
            n: Number being factored
            
        Returns:
            List of fixed points
        """
        return self.fixed_points.get(n, [])
    
    def add_fixed_points(self, n: int, points: List[int]):
        """
        Cache fixed points
        
        Args:
            n: Number being factored
            points: Fixed point positions
        """
        self.fixed_points[n] = points
    
    def get_attractors(self, n: int, initial: int) -> Optional[List[int]]:
        """
        Get cached attractor positions
        
        Args:
            n: Number being factored
            initial: Initial position
            
        Returns:
            Attractor positions or None
        """
        key = (n, initial)
        return self.attractor_cache.get(key)
    
    def add_attractors(self, n: int, initial: int, attractors: List[int]):
        """
        Cache attractor positions
        
        Args:
            n: Number being factored
            initial: Initial position
            attractors: Attractor positions
        """
        key = (n, initial)
        self.attractor_cache[key] = attractors
    
    def get_mirror_position(self, n: int, pos: int, 
                           compute_func: Optional[callable] = None) -> Optional[int]:
        """
        Get cached mirror position
        
        Args:
            n: Number being factored
            pos: Position to mirror
            compute_func: Function to compute if not cached
            
        Returns:
            Mirror position or None
        """
        key = (n, pos)
        
        if key in self.mirror_map:
            return self.mirror_map[key]
        
        if compute_func:
            mirror = compute_func(n, pos)
            self.mirror_map[key] = mirror
            return mirror
        
        return None
    
    def get_spectral_distance(self, x: int, y: int, 
                             compute_func: Optional[callable] = None) -> Optional[float]:
        """
        Get cached spectral distance using accelerated functions
        
        Args:
            x: First number
            y: Second number
            compute_func: Function to compute if not cached
            
        Returns:
            Spectral distance or None
        """
        # Ensure consistent key ordering
        key = tuple(sorted([x, y]))
        
        if key in self.spectral_distances:
            return self.spectral_distances[key]
        
        if compute_func:
            # Use accelerated spectral vectors from Axiom 3
            spec_x = accelerated_spectral_vector(x)
            spec_y = accelerated_spectral_vector(y)
            
            # Euclidean distance
            distance = 0.0
            for sx, sy in zip(spec_x, spec_y):
                distance += (sx - sy) ** 2
            
            distance = math.sqrt(distance)
            self.spectral_distances[key] = distance
            return distance
        
        return None
    
    def get_recursive_mirrors(self, n: int, start: int, depth: int,
                             compute_func: Optional[callable] = None) -> Optional[List[int]]:
        """
        Get cached recursive mirror sequence
        
        Args:
            n: Number being factored
            start: Starting position
            depth: Recursion depth
            compute_func: Function to compute if not cached
            
        Returns:
            Mirror sequence or None
        """
        key = (n, start, depth)
        
        if key in self.recursive_mirrors:
            return self.recursive_mirrors[key]
        
        if compute_func:
            sequence = compute_func(n, start, depth)
            self.recursive_mirrors[key] = sequence
            return sequence
        
        return None
    
    def get_axiom_combination_score(self, pattern_hash: str) -> Optional[float]:
        """
        Get cached axiom combination success rate
        
        Args:
            pattern_hash: Hash of axiom combination pattern
            
        Returns:
            Success rate or None
        """
        return self.axiom_combinations.get(pattern_hash)
    
    def add_axiom_combination(self, pattern_hash: str, success_rate: float):
        """
        Cache axiom combination success rate
        
        Args:
            pattern_hash: Hash of axiom combination pattern
            success_rate: Success rate [0, 1]
        """
        self.axiom_combinations[pattern_hash] = success_rate
    
    def get_interference_strength(self, axiom1: str, axiom2: str) -> Optional[float]:
        """
        Get cached interference strength between axioms
        
        Args:
            axiom1: First axiom
            axiom2: Second axiom
            
        Returns:
            Interference strength or None
        """
        # Ensure consistent ordering
        key = tuple(sorted([axiom1, axiom2]))
        return self.interference_matrix.get(key)
    
    def add_interference_strength(self, axiom1: str, axiom2: str, strength: float):
        """
        Cache interference strength between axioms
        
        Args:
            axiom1: First axiom
            axiom2: Second axiom
            strength: Interference strength
        """
        key = tuple(sorted([axiom1, axiom2]))
        self.interference_matrix[key] = strength
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Dictionary of performance metrics
        """
        total_accesses = sum(self.access_count.values())
        
        stats = {
            'meta_coherence_cache_size': len(self.meta_coherence_cache),
            'observation_count': sum(len(obs) for obs in self.observation_index['by_position'].values()),
            'coherence_fields_cached': len(self.coherence_fields),
            'fixed_points_cached': len(self.fixed_points),
            'mirror_positions_cached': len(self.mirror_map),
            'spectral_distances_cached': len(self.spectral_distances),
            'total_accesses': total_accesses,
            'cache_hit_rate': len(self.access_count) / max(1, total_accesses)
        }
        
        return stats

# Global cache instance
_meta_cache = None

def get_meta_cache() -> MetaAccelerationCache:
    """Get or create global meta acceleration cache"""
    global _meta_cache
    if _meta_cache is None:
        _meta_cache = MetaAccelerationCache()
    return _meta_cache

# Convenience functions that use the global cache

def accelerated_meta_coherence(position: int, coherence_field: Dict[int, float],
                              compute_func: callable) -> float:
    """
    Get meta-coherence with acceleration
    
    Args:
        position: Position to evaluate
        coherence_field: Coherence field
        compute_func: Function to compute if not cached
        
    Returns:
        Meta-coherence value
    """
    cache = get_meta_cache()
    return cache.get_meta_coherence(position, coherence_field, compute_func)

def accelerated_spectral_distance(x: int, y: int) -> float:
    """
    Get spectral distance with acceleration
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        Spectral distance
    """
    cache = get_meta_cache()
    
    def compute():
        spec_x = accelerated_spectral_vector(x)
        spec_y = accelerated_spectral_vector(y)
        distance = sum((sx - sy) ** 2 for sx, sy in zip(spec_x, spec_y)) ** 0.5
        return distance
    
    return cache.get_spectral_distance(x, y, compute)

def accelerated_mirror_position(n: int, pos: int, mirror_func: callable) -> int:
    """
    Get mirror position with acceleration
    
    Args:
        n: Number being factored
        pos: Position to mirror
        mirror_func: Function to compute mirror
        
    Returns:
        Mirror position
    """
    cache = get_meta_cache()
    result = cache.get_mirror_position(n, pos, lambda n, p: mirror_func(p))
    return result if result is not None else pos
