"""
Lazy Resonance Iterator - Generates resonance nodes on-demand based on importance
"""

import heapq
import math
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from .multi_scale_resonance import MultiScaleResonance


class LazyResonanceIterator:
    """
    Generates resonance nodes on-demand based on importance sampling.
    Uses a priority queue to explore high-resonance regions first.
    """

    def __init__(self, n: int, analyzer: "MultiScaleResonance"):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.log_n = math.log(n)
        self.analyzer = analyzer
        self.importance_heap: List[Tuple[float, int]] = (
            []
        )  # Min heap (negative importance)
        self.visited: List[Tuple[int, float]] = []  # Store (position, importance) for inspection
        self.visited_set: Set[int] = set()  # For fast lookup
        self.expansion_radius = max(1, int(self.sqrt_n**0.01))
        self._initialize_seeds()

    def _initialize_seeds(self):
        """Initialize with high-probability seed points"""
        seeds = []

        # Small primes and their powers
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            k = 1
            while p**k <= self.sqrt_n:
                seeds.append(p**k)
                k += 1

        # Golden ratio points
        phi = (1 + math.sqrt(5)) / 2
        x = self.sqrt_n
        while x > 2:
            seeds.append(int(x))
            x = x / phi

        # Tribonacci points
        tau = 1.839286755214161
        x = self.sqrt_n
        while x > 2:
            seeds.append(int(x))
            x = x / tau

        # Perfect squares near sqrt(n)
        base = int(self.sqrt_n**0.5)
        for offset in range(-5, 6):
            if base + offset > 1:
                seeds.append((base + offset) ** 2)

        # Near sqrt(n) for balanced factors - CRITICAL for balanced semiprimes
        # Expand search radius based on n's size
        if self.n.bit_length() < 50:
            search_radius = min(1000, int(self.sqrt_n**0.1))
        else:
            search_radius = min(10000, int(self.sqrt_n**0.05))

        for offset in range(-search_radius, search_radius + 1):
            candidate = self.sqrt_n + offset
            if 2 <= candidate <= self.sqrt_n:
                seeds.append(candidate)

        # Add to heap with initial importance
        for seed in set(seeds):
            if 2 <= seed <= self.sqrt_n:
                importance = self.analyzer.compute_coarse_resonance(seed, self.n)
                heapq.heappush(self.importance_heap, (-importance, seed))
                self.visited.append((seed, importance))
                self.visited_set.add(seed)

    def __iter__(self):
        while self.importance_heap:
            neg_importance, x = heapq.heappop(self.importance_heap)
            yield x
            # Expand region around x based on its importance
            self._expand_region(x, -neg_importance)

    def _expand_region(self, x: int, importance: float):
        """Dynamically expand high-resonance regions"""
        # Adaptive radius based on importance and n's size
        if self.n.bit_length() > 70:
            # For very large numbers, limit expansion
            radius = min(10, int(self.expansion_radius * importance))
        else:
            radius = int(self.expansion_radius * (1 + importance))

        # Compute local gradient
        gradient = self._estimate_gradient(x)

        # Generate neighbors with bias toward gradient direction
        neighbors = []

        # Along gradient
        for step in [1, 2, 5, 10]:
            next_x = x + int(step * gradient * radius)
            if 2 <= next_x <= self.sqrt_n and next_x not in self.visited_set:
                neighbors.append(next_x)

        # Perpendicular to gradient (exploration)
        for offset in [-radius, -radius // 2, radius // 2, radius]:
            next_x = x + offset
            if 2 <= next_x <= self.sqrt_n and next_x not in self.visited_set:
                neighbors.append(next_x)

        # Add neighbors to heap
        for neighbor in neighbors:
            if neighbor not in self.visited_set:
                imp = self.analyzer.compute_coarse_resonance(neighbor, self.n)
                heapq.heappush(self.importance_heap, (-imp, neighbor))
                self.visited.append((neighbor, imp))
                self.visited_set.add(neighbor)

    def _estimate_gradient(self, x: int) -> float:
        """Estimate resonance gradient at x"""
        delta = max(1, int(x * 0.001))

        # Forward difference if at boundary
        if x - delta < 2:
            r1 = self.analyzer.compute_coarse_resonance(x, self.n)
            r2 = self.analyzer.compute_coarse_resonance(x + delta, self.n)
            return (r2 - r1) / delta

        # Backward difference if at boundary
        if x + delta > self.sqrt_n:
            r1 = self.analyzer.compute_coarse_resonance(x - delta, self.n)
            r2 = self.analyzer.compute_coarse_resonance(x, self.n)
            return (r2 - r1) / delta

        # Central difference
        r1 = self.analyzer.compute_coarse_resonance(x - delta, self.n)
        r2 = self.analyzer.compute_coarse_resonance(x + delta, self.n)
        return (r2 - r1) / (2 * delta)
