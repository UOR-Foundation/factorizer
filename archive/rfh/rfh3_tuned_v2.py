"""
Tuned RFH3 v2 - Better handling of very large semiprimes
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

from rfh3_tuned import RFH3Tuned
from rfh3_ultimate import RFH3Config
from rfh3 import MultiScaleResonance, StateManager, ResonancePatternLearner


class RFH3TunedV2(RFH3Tuned):
    """Enhanced tuning for very large semiprimes"""
    
    def factor(self, n: int, timeout: float = 60.0) -> Tuple[int, int]:
        """Optimized factorization with special handling for large numbers"""
        if n < 4:
            raise ValueError("n must be >= 4")
        
        # For very large numbers, use specialized approach
        if n.bit_length() > 70:
            return self._factor_large_semiprime(n, timeout)
        
        # Otherwise use parent method
        return super().factor(n, timeout)
    
    def _factor_large_semiprime(self, n: int, timeout: float) -> Tuple[int, int]:
        """Specialized factorization for very large semiprimes"""
        start_time = time.time()
        sqrt_n = int(math.sqrt(n))
        
        self.logger.debug(f"Large semiprime factorization: {n} ({n.bit_length()} bits)")
        
        # Phase 0: Quick check around sqrt(n) with limited radius
        phase_start = time.time()
        max_radius = min(1000, int(sqrt_n ** 0.01))  # Very limited radius
        
        for offset in range(0, max_radius + 1):
            if time.time() - start_time > timeout * 0.1:
                break
            for sign in [1, -1]:
                if offset == 0 and sign == -1:
                    continue
                x = sqrt_n + sign * offset
                if 2 <= x <= sqrt_n and n % x == 0:
                    self._record_phase_success(0, time.time() - phase_start)
                    return (x, n // x)
        
        self.stats['phase_times'][0] += time.time() - phase_start
        
        # Phase 1: Fermat's method (very efficient for balanced semiprimes)
        phase_start = time.time()
        result = self._fermat_method_optimized(n, timeout * 0.4)
        if result:
            self._record_phase_success(1, time.time() - phase_start)
            return result
        self.stats['phase_times'][1] += time.time() - phase_start
        
        # Phase 2: Pollard's Rho with better parameters
        phase_start = time.time()
        result = self._pollard_rho_optimized(n, timeout * 0.4)
        if result:
            self._record_phase_success(2, time.time() - phase_start)
            return result
        self.stats['phase_times'][2] += time.time() - phase_start
        
        # Phase 3: ECM-style approach (simplified)
        phase_start = time.time()
        result = self._ecm_style_search(n, timeout * 0.2)
        if result:
            self._record_phase_success(3, time.time() - phase_start)
            return result
        self.stats['phase_times'][3] += time.time() - phase_start
        
        raise ValueError(f"Failed to factor large semiprime {n}")
    
    def _fermat_method_optimized(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Optimized Fermat's method for balanced semiprimes"""
        start_time = time.time()
        
        # Start from ceil(sqrt(n))
        a = int(math.sqrt(n))
        if a * a < n:
            a += 1
        
        # Limit iterations based on n's size
        max_iterations = min(1000000, int(n ** 0.25))
        
        for _ in range(max_iterations):
            if time.time() - start_time > timeout:
                break
            
            b2 = a * a - n
            if b2 < 0:
                a += 1
                continue
            
            b = int(math.sqrt(b2))
            if b * b == b2:
                # Found factorization
                factor1 = a - b
                factor2 = a + b
                if factor1 > 1 and factor2 > 1:
                    return (min(factor1, factor2), max(factor1, factor2))
            
            a += 1
        
        return None
    
    def _pollard_rho_optimized(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Optimized Pollard's Rho for large numbers"""
        start_time = time.time()
        
        # Try multiple polynomials
        polynomials = [
            lambda x, n: (x * x + 1) % n,
            lambda x, n: (x * x + 2) % n,
            lambda x, n: (x * x - 1) % n,
            lambda x, n: (x * x + x + 1) % n,
        ]
        
        for poly in polynomials:
            if time.time() - start_time > timeout:
                break
            
            # Random starting point
            x = 2 + int(time.time() * 1000) % 100
            y = x
            d = 1
            
            # Brent's improvement
            power = 1
            lam = 1
            
            while d == 1:
                if time.time() - start_time > timeout / len(polynomials):
                    break
                
                if power == lam:
                    x = y
                    power *= 2
                    lam = 0
                
                y = poly(y, n)
                lam += 1
                
                d = math.gcd(abs(x - y), n)
                
                if 1 < d < n:
                    return (min(d, n // d), max(d, n // d))
        
        return None
    
    def _ecm_style_search(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Simplified ECM-style search"""
        start_time = time.time()
        sqrt_n = int(math.sqrt(n))
        
        # Use modular arithmetic patterns
        # For balanced semiprimes p*q where p ≈ q ≈ sqrt(n)
        # We can use the fact that (p+q)² - 4pq = (p-q)²
        
        # Try to find k such that gcd(k² - n, n) gives a factor
        k_start = sqrt_n
        k_range = min(10000, int(sqrt_n ** 0.1))
        
        for offset in range(-k_range, k_range + 1):
            if time.time() - start_time > timeout:
                break
            
            k = k_start + offset
            if k < 2:
                continue
            
            # Check gcd(k² - n, n)
            k2_minus_n = k * k - n
            if k2_minus_n != 0:
                d = math.gcd(abs(k2_minus_n), n)
                if 1 < d < n:
                    return (min(d, n // d), max(d, n // d))
        
        return None
    
    def _phase1_balanced_search(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Override to limit search for very large numbers"""
        if n.bit_length() > 50:
            # For large numbers, use more limited search
            sqrt_n = int(math.sqrt(n))
            
            # Only try immediate vicinity of sqrt(n)
            max_radius = min(1000, int(sqrt_n ** 0.01))
            
            for x in range(max(2, sqrt_n - max_radius), min(sqrt_n + max_radius + 1, sqrt_n + 1)):
                if n % x == 0:
                    return (x, n // x)
            
            return None
        else:
            # Use parent method for smaller numbers
            return super()._phase1_balanced_search(n, timeout)
