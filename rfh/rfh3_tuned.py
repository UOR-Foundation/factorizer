"""
Tuned RFH3 implementation that prioritizes balanced semiprime factors
"""

import heapq
import math
import time
import json
import logging
import pickle
from collections import defaultdict, deque
from functools import lru_cache
from typing import Dict, Tuple, List, Set, Optional, Any
import numpy as np

from rfh3_ultimate import RFH3Ultimate, RFH3Config, HierarchicalSearchUltimate
from rfh3 import (
    LazyResonanceIterator, MultiScaleResonance, StateManager,
    ZonePredictor, ResonancePatternLearner
)


class RFH3Tuned(RFH3Ultimate):
    """Tuned RFH3 that prioritizes balanced factors for semiprimes"""
    
    def __init__(self, config: Optional[RFH3Config] = None):
        super().__init__(config)
        self.prefer_balanced = True  # New flag
        
    def factor(self, n: int, timeout: float = 60.0) -> Tuple[int, int]:
        """Tuned factorization that checks for balanced factors first"""
        if n < 4:
            raise ValueError("n must be >= 4")
        
        start_time = time.time()
        self.logger.debug(f"Factoring {n} ({n.bit_length()} bits)")
        
        # Initialize
        self.analyzer = MultiScaleResonance()
        self.state = StateManager()
        sqrt_n = int(math.sqrt(n))
        
        # Phase 0A: Quick check for balanced factors FIRST
        phase_start = time.time()
        
        # Check near sqrt(n) for balanced semiprimes
        search_radius = max(1, int(sqrt_n ** 0.1))
        for offset in range(0, search_radius + 1):
            for sign in [1, -1]:
                if offset == 0 and sign == -1:
                    continue
                x = sqrt_n + sign * offset
                if 2 <= x <= sqrt_n and n % x == 0:
                    self._record_phase_success(0, time.time() - phase_start)
                    return (x, n // x)
        
        # Phase 0B: Small primes (only if NOT a balanced semiprime candidate)
        # Check if n looks like a balanced semiprime
        is_balanced_candidate = self._is_balanced_semiprime_candidate(n)
        
        if not is_balanced_candidate:
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                if n % p == 0:
                    # Only return if cofactor is small (not balanced)
                    cofactor = n // p
                    if cofactor < sqrt_n * 10:  # Not balanced
                        self._record_phase_success(0, time.time() - phase_start)
                        return (p, cofactor)
        
        self.stats['phase_times'][0] += time.time() - phase_start
        
        # Phase 1: Enhanced balanced search
        phase_start = time.time()
        result = self._phase1_balanced_search(n, timeout * 0.3)
        if result:
            self._record_phase_success(1, time.time() - phase_start)
            return result
        self.stats['phase_times'][1] += time.time() - phase_start
        
        # Phase 2: Hierarchical search (already has balanced bias)
        if self.config.hierarchical_search:
            phase_start = time.time()
            result = self._phase2_hierarchical(n, timeout * 0.2)
            if result:
                self._record_phase_success(2, time.time() - phase_start)
                return result
            self.stats['phase_times'][2] += time.time() - phase_start
        
        # Phase 3: Adaptive resonance search
        phase_start = time.time()
        result = self._phase3_adaptive_resonance(n, timeout * 0.3)
        if result:
            self._record_phase_success(3, time.time() - phase_start)
            return result
        self.stats['phase_times'][3] += time.time() - phase_start
        
        # Phase 4: Advanced algorithms (including small primes if missed)
        phase_start = time.time()
        result = self._phase4_advanced_with_small_primes(n, timeout * 0.2)
        if result:
            self._record_phase_success(4, time.time() - phase_start)
            return result
        self.stats['phase_times'][4] += time.time() - phase_start
        
        # Should not reach here for composite numbers
        raise ValueError(f"Failed to factor {n}")
    
    def _is_balanced_semiprime_candidate(self, n: int) -> bool:
        """Check if n is likely a balanced semiprime"""
        # Several heuristics:
        # 1. n is odd (most balanced semiprimes are)
        if n % 2 == 0:
            return False
        
        # 2. n is not divisible by small primes up to some threshold
        # but we need to be careful not to do too much work
        small_primes = [3, 5, 7]
        for p in small_primes:
            if n % p == 0:
                cofactor = n // p
                sqrt_n = int(math.sqrt(n))
                # If cofactor is close to sqrt(n), it might still be balanced
                if abs(cofactor - sqrt_n) < sqrt_n * 0.1:
                    return True
                else:
                    return False
        
        # 3. Digit sum test (many balanced semiprimes have certain patterns)
        # This is a weak heuristic but can help
        digit_sum = sum(int(d) for d in str(n))
        if digit_sum % 3 == 0:
            return False  # Often divisible by 3
        
        # 4. Check if n is close to a perfect square
        # Balanced semiprimes n = p*q where p ≈ q have n ≈ p²
        sqrt_n = int(math.sqrt(n))
        if abs(n - sqrt_n * sqrt_n) < n * 0.01:
            return True
        
        # Default: assume it could be balanced
        return True
    
    def _phase1_balanced_search(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Enhanced search focusing on balanced factors"""
        sqrt_n = int(math.sqrt(n))
        start_time = time.time()
        
        # Multiple search strategies for balanced factors
        strategies = [
            # Strategy 1: Expanding rings around sqrt(n)
            lambda: self._expanding_ring_search(n, sqrt_n, timeout / 3),
            
            # Strategy 2: Binary search guided by resonance
            lambda: self._binary_resonance_search(n, sqrt_n, timeout / 3),
            
            # Strategy 3: Adaptive step search
            lambda: self._adaptive_step_search(n, sqrt_n, timeout / 3),
        ]
        
        for strategy in strategies:
            if time.time() - start_time > timeout:
                break
            result = strategy()
            if result:
                return result
        
        return None
    
    def _expanding_ring_search(self, n: int, sqrt_n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Search in expanding rings around sqrt(n)"""
        start_time = time.time()
        max_radius = min(sqrt_n // 2, 1000000)
        
        # Adaptive radius growth
        radius = 1
        growth_factor = 1.1
        
        checked = set()
        while radius < max_radius and time.time() - start_time < timeout:
            # Check the ring at current radius
            for x in range(max(2, sqrt_n - radius), min(sqrt_n + radius + 1, sqrt_n + 1)):
                if x in checked:
                    continue
                checked.add(x)
                
                if n % x == 0:
                    return (x, n // x)
            
            # Grow radius
            radius = int(radius * growth_factor) + 1
            if radius > 100:
                growth_factor = 1.2  # Accelerate for larger radii
        
        return None
    
    def _binary_resonance_search(self, n: int, sqrt_n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Binary search guided by resonance field"""
        start_time = time.time()
        
        # Define search bounds
        low = max(2, int(sqrt_n * 0.5))
        high = min(sqrt_n, int(sqrt_n * 1.5))
        
        # Coarse resonance scan
        best_candidates = []
        step = max(1, (high - low) // 100)
        
        for x in range(low, high + 1, step):
            if time.time() - start_time > timeout / 2:
                break
            if n % x == 0:
                return (x, n // x)
            
            res = self.analyzer.compute_coarse_resonance(x, n)
            if res > 0.3:
                best_candidates.append((x, res))
        
        # Refine around best candidates
        best_candidates.sort(key=lambda x: x[1], reverse=True)
        for x, _ in best_candidates[:10]:
            for offset in range(-step, step + 1):
                candidate = x + offset
                if 2 <= candidate <= sqrt_n:
                    if n % candidate == 0:
                        return (candidate, n // candidate)
        
        return None
    
    def _adaptive_step_search(self, n: int, sqrt_n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Adaptive step size based on local resonance gradient"""
        start_time = time.time()
        
        x = sqrt_n
        step = 1
        direction = -1  # Start going down
        
        last_resonance = self.analyzer.compute_coarse_resonance(x, n)
        oscillations = 0
        
        while 2 <= x <= sqrt_n and time.time() - start_time < timeout:
            if n % x == 0:
                return (x, n // x)
            
            # Compute resonance gradient
            next_x = x + direction * step
            if 2 <= next_x <= sqrt_n:
                next_resonance = self.analyzer.compute_coarse_resonance(next_x, n)
                
                # If resonance decreased, maybe change direction
                if next_resonance < last_resonance:
                    direction *= -1
                    oscillations += 1
                    
                    # After several oscillations, increase step size
                    if oscillations > 3:
                        step = min(step * 2, 100)
                        oscillations = 0
                
                last_resonance = next_resonance
                x = next_x
            else:
                # Hit boundary, reverse direction
                direction *= -1
                x += direction * step
        
        return None
    
    def _phase4_advanced_with_small_primes(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Advanced algorithms but also check small primes comprehensively"""
        start_time = time.time()
        
        # First, comprehensively check small primes (in case we missed any)
        primes_to_check = []
        for p in range(2, min(10000, int(math.sqrt(n)) + 1)):
            if time.time() - start_time > timeout * 0.2:
                break
            # Simple primality check for p
            if p == 2 or (p > 2 and p % 2 != 0):
                is_prime = True
                for d in range(3, int(math.sqrt(p)) + 1, 2):
                    if p % d == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes_to_check.append(p)
        
        for p in primes_to_check:
            if n % p == 0:
                return (p, n // p)
        
        # Then run standard phase 4
        return super()._phase4_advanced(n, timeout * 0.8)
