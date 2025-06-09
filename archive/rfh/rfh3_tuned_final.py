"""
Final tuned RFH3 - Complete handling of all semiprime sizes
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

from rfh3_tuned_v2 import RFH3TunedV2
from rfh3_ultimate import RFH3Config
from rfh3 import MultiScaleResonance, StateManager, ResonancePatternLearner


class RFH3TunedFinal(RFH3TunedV2):
    """Final tuning with complete semiprime handling"""
    
    def factor(self, n: int, timeout: float = 60.0) -> Tuple[int, int]:
        """Ultimate factorization that always prioritizes balanced factors"""
        if n < 4:
            raise ValueError("n must be >= 4")
        
        # For ALL numbers, check if they're likely balanced semiprimes first
        if self._is_likely_balanced_semiprime(n):
            # Try balanced factorization first
            result = self._try_balanced_factorization(n, timeout * 0.7)
            if result:
                return result
        
        # Then use standard approach
        return super().factor(n, timeout * 0.3)
    
    def _is_likely_balanced_semiprime(self, n: int) -> bool:
        """Enhanced heuristic to detect balanced semiprimes"""
        # Quick checks to avoid expensive computation
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0 or n % 7 == 0:
            # Check if the cofactor is large (indicating balanced)
            for p in [2, 3, 5, 7]:
                if n % p == 0:
                    cofactor = n // p
                    sqrt_n = int(math.sqrt(n))
                    # If cofactor is within order of magnitude of sqrt(n), might be balanced
                    if cofactor > sqrt_n / 100:
                        return True
            return False
        
        # For odd numbers not divisible by small primes, likely balanced
        return True
    
    def _try_balanced_factorization(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Dedicated balanced factorization for suspected balanced semiprimes"""
        start_time = time.time()
        sqrt_n = int(math.sqrt(n))
        
        # Strategy 1: Fermat's method first (best for balanced)
        result = self._fermat_method_enhanced(n, timeout * 0.4)
        if result:
            return result
        
        # Strategy 2: Focused sqrt search
        if time.time() - start_time < timeout * 0.6:
            result = self._focused_sqrt_search(n, timeout * 0.2)
            if result:
                return result
        
        # Strategy 3: Advanced Pollard's Rho
        if time.time() - start_time < timeout * 0.8:
            result = self._pollard_rho_balanced(n, timeout * 0.2)
            if result:
                return result
        
        return None
    
    def _fermat_method_enhanced(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Enhanced Fermat's method optimized for very large balanced semiprimes"""
        start_time = time.time()
        
        # Start from ceil(sqrt(n))
        a = int(math.sqrt(n))
        if a * a < n:
            a += 1
        
        # Adaptive step size for very large numbers
        step = 1
        if n.bit_length() > 80:
            step = max(1, int(a ** 0.001))  # Larger steps for huge numbers
        
        iterations = 0
        max_iterations = min(10000000, int(n ** 0.1))  # Adjusted for large n
        
        while iterations < max_iterations and time.time() - start_time < timeout:
            b2 = a * a - n
            if b2 >= 0:
                b = int(math.sqrt(b2))
                if b * b == b2:
                    # Found factorization
                    factor1 = a - b
                    factor2 = a + b
                    if factor1 > 1 and factor2 > 1:
                        return (min(factor1, factor2), max(factor1, factor2))
            
            a += step
            iterations += 1
            
            # Adaptive step adjustment
            if iterations % 1000 == 0 and n.bit_length() > 100:
                step = min(step * 2, int(a ** 0.01))
        
        return None
    
    def _focused_sqrt_search(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Very focused search around sqrt(n) for balanced factors"""
        start_time = time.time()
        sqrt_n = int(math.sqrt(n))
        
        # Dynamic radius based on n's size
        if n.bit_length() < 50:
            max_radius = int(sqrt_n ** 0.1)
        elif n.bit_length() < 80:
            max_radius = int(sqrt_n ** 0.01)
        else:
            max_radius = int(sqrt_n ** 0.001)
        
        max_radius = min(max_radius, 1000000)  # Cap for efficiency
        
        # Check in waves
        for radius in range(1, max_radius + 1):
            if time.time() - start_time > timeout:
                break
            
            # Check both sides
            for sign in [-1, 1]:
                x = sqrt_n + sign * radius
                if 2 <= x <= sqrt_n and n % x == 0:
                    return (x, n // x)
        
        return None
    
    def _pollard_rho_balanced(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Pollard's Rho specifically tuned for balanced semiprimes"""
        start_time = time.time()
        
        # Use multiple starting points and polynomials
        starts = [2, 3, int(math.sqrt(n)) % n, int(n ** 0.25) % n]
        polynomials = [
            lambda x, n: (x * x + 1) % n,
            lambda x, n: (x * x - 1) % n,
            lambda x, n: (x * x + 2) % n,
            lambda x, n: (x * x * x + x + 1) % n,
        ]
        
        for start in starts:
            for poly in polynomials:
                if time.time() - start_time > timeout:
                    return None
                
                x = start
                y = start
                d = 1
                
                # Brent's optimization
                power = 1
                lam = 1
                
                max_steps = min(100000, int(n ** 0.1))
                steps = 0
                
                while d == 1 and steps < max_steps:
                    if power == lam:
                        x = y
                        power *= 2
                        lam = 0
                    
                    y = poly(y, n)
                    lam += 1
                    steps += 1
                    
                    # Batch GCD computation for efficiency
                    if steps % 100 == 0:
                        d = math.gcd(abs(x - y), n)
                        
                        if 1 < d < n:
                            # Check if it's a balanced factorization
                            other = n // d
                            if min(d, other) > n ** 0.3:  # Both factors are substantial
                                return (min(d, other), max(d, other))
        
        return None
    
    def _factor_large_semiprime(self, n: int, timeout: float) -> Tuple[int, int]:
        """Override to ensure balanced factorization for large numbers"""
        # Always try balanced methods first for large numbers
        result = self._try_balanced_factorization(n, timeout * 0.8)
        if result:
            return result
        
        # Only then fall back to parent method
        return super()._factor_large_semiprime(n, timeout * 0.2)
