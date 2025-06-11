"""
Ultimate RFH3 implementation with perfected components and hard semiprime handling
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


# ============================================================================
# IMPORT CORE COMPONENTS
# ============================================================================

from rfh3 import (
    LazyResonanceIterator, MultiScaleResonance, StateManager,
    ZonePredictor, ResonancePatternLearner, RFH3Config
)


# ============================================================================
# ENHANCED HIERARCHICAL SEARCH
# ============================================================================

class HierarchicalSearchUltimate:
    """Ultimate hierarchical search with semiprime optimizations"""
    
    def __init__(self, n: int, analyzer: MultiScaleResonance):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.analyzer = analyzer
        self.log_n = math.log(n)
        
    def search(self, max_time: float = 5.0) -> List[Tuple[int, float]]:
        """Multi-strategy hierarchical search"""
        start_time = time.time()
        candidates = []
        
        # Strategy 1: Critical points
        candidates.extend(self._check_critical_points())
        
        if time.time() - start_time > max_time / 3:
            return self._finalize_candidates(candidates)
        
        # Strategy 2: Balanced factor search
        candidates.extend(self._balanced_factor_search())
        
        if time.time() - start_time > 2 * max_time / 3:
            return self._finalize_candidates(candidates)
        
        # Strategy 3: Resonance-guided sampling
        candidates.extend(self._resonance_guided_search())
        
        return self._finalize_candidates(candidates)
    
    def _check_critical_points(self) -> List[Tuple[int, float]]:
        """Check mathematically significant points"""
        candidates = []
        
        # Small primes (extended list)
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                       53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 
                       109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 
                       173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229]
        
        for p in small_primes:
            if p > self.sqrt_n:
                break
            if self.n % p == 0:
                return [(p, 1.0)]  # Found factor!
            candidates.append((p, self.analyzer.compute_coarse_resonance(p, self.n)))
        
        # Perfect powers
        for base in range(2, min(100, int(self.log_n) + 1)):
            exp = 2
            while base ** exp <= self.sqrt_n:
                x = base ** exp
                if self.n % x == 0:
                    return [(x, 1.0)]
                res = self.analyzer.compute_coarse_resonance(x, self.n)
                if res > 0.1:
                    candidates.append((x, res))
                exp += 1
        
        # Near sqrt(n) - critical for balanced semiprimes
        sqrt_region = max(1, int(self.sqrt_n ** 0.05))
        for offset in range(-sqrt_region * 10, sqrt_region * 10 + 1):
            x = self.sqrt_n + offset
            if 2 <= x <= self.sqrt_n:
                if self.n % x == 0:
                    return [(x, 1.0)]
                res = self.analyzer.compute_resonance(x, self.n)
                if res > 0.2:
                    candidates.append((x, res))
        
        return candidates
    
    def _balanced_factor_search(self) -> List[Tuple[int, float]]:
        """Search for balanced factors (p ≈ q)"""
        candidates = []
        
        # For balanced semiprimes, factors are near sqrt(n)
        # Use multiple search radii
        radii = [
            int(self.sqrt_n ** 0.01),
            int(self.sqrt_n ** 0.02),
            int(self.sqrt_n ** 0.05),
            int(self.sqrt_n ** 0.1),
        ]
        
        checked = set()
        for radius in radii:
            for x in range(max(2, self.sqrt_n - radius), 
                          min(self.sqrt_n + radius + 1, self.sqrt_n + 1)):
                if x in checked:
                    continue
                checked.add(x)
                
                if self.n % x == 0:
                    return [(x, 1.0)]
                
                # Compute resonance for promising candidates
                if abs(x - self.sqrt_n) < radius / 2:
                    res = self.analyzer.compute_resonance(x, self.n)
                    candidates.append((x, res))
        
        return candidates
    
    def _resonance_guided_search(self) -> List[Tuple[int, float]]:
        """Use resonance field to guide search"""
        candidates = []
        
        # Adaptive sampling based on n's size
        if self.sqrt_n < 1000:
            step = 1
        elif self.sqrt_n < 10000:
            step = max(1, self.sqrt_n // 1000)
        else:
            step = max(1, self.sqrt_n // 100)
        
        # Quick coarse scan
        coarse_peaks = []
        for x in range(2, min(self.sqrt_n + 1, 10000), step):
            res = self.analyzer.compute_coarse_resonance(x, self.n)
            if res > 0.3:
                coarse_peaks.append((x, res))
        
        # Refine around peaks
        coarse_peaks.sort(key=lambda x: x[1], reverse=True)
        for peak, _ in coarse_peaks[:20]:
            window = max(1, step // 2)
            for offset in range(-window, window + 1):
                x = peak + offset
                if 2 <= x <= self.sqrt_n:
                    if self.n % x == 0:
                        return [(x, 1.0)]
                    res = self.analyzer.compute_resonance(x, self.n)
                    candidates.append((x, res))
        
        return candidates
    
    def _finalize_candidates(self, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Remove duplicates and sort candidates"""
        seen = {}
        for x, res in candidates:
            if x not in seen or res > seen[x]:
                seen[x] = res
        
        final = [(x, res) for x, res in seen.items()]
        final.sort(key=lambda x: x[1], reverse=True)
        return final[:100]


# ============================================================================
# ULTIMATE RFH3 CLASS
# ============================================================================

class RFH3Ultimate:
    """Ultimate RFH3 with perfected algorithms"""
    
    def __init__(self, config: Optional[RFH3Config] = None):
        self.config = config or RFH3Config()
        self.logger = self._setup_logging()
        
        # Components
        self.learner = ResonancePatternLearner()
        self.state = StateManager(self.config.checkpoint_interval)
        self.analyzer = None
        
        # Statistics
        self.stats = {
            'factorizations': 0,
            'successes': 0,
            'total_time': 0,
            'phase_times': defaultdict(float),
            'phase_successes': defaultdict(int)
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('RFH3Ultimate')
        logger.setLevel(self.config.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')  # Simplified
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def factor(self, n: int, timeout: float = 60.0) -> Tuple[int, int]:
        """Ultimate factorization with multi-phase approach"""
        if n < 4:
            raise ValueError("n must be >= 4")
        
        # Phase 0: Quick divisibility checks
        phase_start = time.time()
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if n % p == 0:
                self._record_phase_success(0, time.time() - phase_start)
                return (p, n // p)
        self.stats['phase_times'][0] += time.time() - phase_start
        
        start_time = time.time()
        self.logger.debug(f"Factoring {n} ({n.bit_length()} bits)")
        
        # Initialize
        self.analyzer = MultiScaleResonance()
        self.state = StateManager()
        
        # Phase 1: Pattern-based search
        if self.config.learning_enabled and self.learner.success_patterns:
            phase_start = time.time()
            result = self._phase1_pattern_search(n, timeout * 0.1)
            if result:
                self._record_phase_success(1, time.time() - phase_start)
                return result
            self.stats['phase_times'][1] += time.time() - phase_start
        
        # Phase 2: Hierarchical search
        if self.config.hierarchical_search:
            phase_start = time.time()
            result = self._phase2_hierarchical(n, timeout * 0.2)
            if result:
                self._record_phase_success(2, time.time() - phase_start)
                return result
            self.stats['phase_times'][2] += time.time() - phase_start
        
        # Phase 3: Adaptive resonance search
        phase_start = time.time()
        result = self._phase3_adaptive_resonance(n, timeout * 0.4)
        if result:
            self._record_phase_success(3, time.time() - phase_start)
            return result
        self.stats['phase_times'][3] += time.time() - phase_start
        
        # Phase 4: Advanced algorithms
        phase_start = time.time()
        result = self._phase4_advanced(n, timeout * 0.3)
        if result:
            self._record_phase_success(4, time.time() - phase_start)
            return result
        self.stats['phase_times'][4] += time.time() - phase_start
        
        # Should not reach here for composite numbers
        raise ValueError(f"Failed to factor {n}")
    
    def _phase1_pattern_search(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Use learned patterns"""
        zones = self.learner.predict_high_resonance_zones(n)
        sqrt_n = int(math.sqrt(n))
        
        start_time = time.time()
        for start, end, conf in zones[:10]:
            if time.time() - start_time > timeout:
                break
            
            # Higher density for high confidence
            step = 1 if conf > 0.8 else max(1, (end - start) // 50)
            
            for x in range(max(2, start), min(end + 1, sqrt_n + 1), step):
                if n % x == 0:
                    self.learner.record_success(n, x, {'resonance': conf})
                    return (x, n // x)
        
        return None
    
    def _phase2_hierarchical(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Hierarchical search"""
        search = HierarchicalSearchUltimate(n, self.analyzer)
        candidates = search.search(max_time=timeout)
        
        # Check candidates
        for x, resonance in candidates:
            if n % x == 0:
                self.learner.record_success(n, x, {'resonance': resonance})
                return (x, n // x)
        
        return None
    
    def _phase3_adaptive_resonance(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Adaptive resonance field navigation"""
        iterator = LazyResonanceIterator(n, self.analyzer)
        threshold = 1.0 / (math.log(n) ** 0.3)  # Aggressive threshold
        
        start_time = time.time()
        high_res_nodes = []
        
        for i, x in enumerate(iterator):
            if time.time() - start_time > timeout:
                break
            
            # Quick check
            if n % x == 0:
                resonance = self.analyzer.compute_resonance(x, n)
                self.learner.record_success(n, x, {'resonance': resonance})
                return (x, n // x)
            
            # Periodic resonance computation
            if i < 100 or i % 20 == 0:
                resonance = self.analyzer.compute_resonance(x, n)
                if resonance > threshold:
                    high_res_nodes.append((x, resonance))
            
            if i > 50000:  # Limit
                break
        
        # Check neighborhoods of high resonance nodes
        high_res_nodes.sort(key=lambda x: x[1], reverse=True)
        for x, res in high_res_nodes[:20]:
            for offset in range(-10, 11):
                candidate = x + offset
                if 2 <= candidate <= int(math.sqrt(n)) and n % candidate == 0:
                    self.learner.record_success(n, candidate, {'resonance': res})
                    return (candidate, n // candidate)
        
        return None
    
    def _phase4_advanced(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """Advanced algorithms including Pollard's Rho"""
        start_time = time.time()
        
        # Pollard's Rho with multiple polynomials
        for c in [1, 2, 3, 5, 7, 11]:
            if time.time() - start_time > timeout * 0.8:
                break
            
            x, y = 2, 2
            d = 1
            f = lambda x: (x * x + c) % n
            
            for _ in range(min(100000, n)):
                if time.time() - start_time > timeout:
                    break
                
                x = f(x)
                y = f(f(y))
                d = math.gcd(abs(x - y), n)
                
                if 1 < d < n:
                    return (min(d, n // d), max(d, n // d))
        
        # Fermat's method for numbers with close factors
        a = int(math.sqrt(n)) + 1
        limit = min(a + 100000, n)
        
        while a < limit:
            if time.time() - start_time > timeout:
                break
            
            b2 = a * a - n
            b = int(math.sqrt(b2))
            if b * b == b2:
                factor1 = a - b
                factor2 = a + b
                if factor1 > 1 and factor2 > 1:
                    return (min(factor1, factor2), max(factor1, factor2))
            a += 1
        
        # Trial division as last resort
        sqrt_n = int(math.sqrt(n))
        limit = min(1000000, sqrt_n)
        
        for i in range(101, limit, 2):
            if n % i == 0:
                return (i, n // i)
        
        return None
    
    def _record_phase_success(self, phase: int, time_taken: float):
        """Record successful phase"""
        self.stats['phase_successes'][phase] += 1
        self.stats['phase_times'][phase] += time_taken
        self.stats['successes'] += 1
        self.stats['total_time'] += time_taken
    
    def print_stats(self):
        """Print detailed statistics"""
        print("\nDETAILED STATISTICS:")
        print("-" * 40)
        
        total_attempts = sum(self.stats['phase_successes'].values())
        if total_attempts == 0:
            print("No successful factorizations yet")
            return
        
        for phase in range(5):
            successes = self.stats['phase_successes'][phase]
            time_spent = self.stats['phase_times'][phase]
            
            if successes > 0:
                avg_time = time_spent / successes
                percentage = successes / total_attempts * 100
                print(f"Phase {phase}: {successes:3d} successes ({percentage:5.1f}%), "
                      f"avg time: {avg_time:.4f}s")
        
        print(f"\nTotal: {total_attempts} factorizations in {self.stats['total_time']:.3f}s")
        print(f"Average: {self.stats['total_time']/total_attempts:.4f}s per factorization")


# ============================================================================
# ULTIMATE TEST SUITE
# ============================================================================

def test_ultimate():
    """Ultimate test with hard semiprimes"""
    
    config = RFH3Config()
    config.max_iterations = 100000
    config.hierarchical_search = True
    config.learning_enabled = True
    config.log_level = logging.WARNING  # Less verbose
    
    rfh3 = RFH3Ultimate(config)
    
    # Comprehensive test suite
    test_cases = [
        # Warm-up
        (143, 11, 13),
        (323, 17, 19),
        (1147, 31, 37),
        
        # Balanced semiprimes
        (10403, 101, 103),
        (40001, 13, 3077),  # Note: 13 × 3077
        (104729, 317, 331),
        
        # Hard balanced semiprimes
        (282797, 523, 541),
        (1299827, 1117, 1163),
        (16777259, 4093, 4099),
        
        # Very hard cases
        (1073676287, 32749, 32771),  # Twin primes
        (2147483713, 46337, 46349),  # Close primes
        
        # Special structure
        (536870923, 11, 48806447),   # Note: 11 × 48806447
        (4294967357, 43, 99883427),  # Note: 43 × 99883427
        
        # Additional hard cases
        (1000000007, 1000003, 1),    # Near billion
        (1234567891, 1, 1234567891), # Prime!
        (9999999967, 99991, 100003), # Large balanced
    ]
    
    print("\nRFH3 ULTIMATE - COMPREHENSIVE SEMIPRIME TEST")
    print("=" * 85)
    print(f"{'n':>12} | {'Bits':>4} | {'Expected':^15} | {'Found':^15} | {'Time':>8} | {'Status'}")
    print("-" * 85)
    
    results = []
    
    for test in test_cases:
        if len(test) == 3:
            n, p_true, q_true = test
        else:
            # Handle special cases
            n = test[0]
            # Find actual factors
            p_true = q_true = 0
            for i in range(2, min(1000000, int(math.sqrt(n)) + 1)):
                if n % i == 0:
                    p_true = i
                    q_true = n // i
                    break
            
            if p_true == 0:  # Likely prime
                p_true = 1
                q_true = n
        
        try:
            start = time.time()
            
            # Check if actually prime
            if p_true == 1 or q_true == 1:
                # Skip primes
                print(f"{n:12d} | {n.bit_length():4d} | {'PRIME':^15} | {'SKIPPED':^15} | "
                      f"{0:8.3f}s | -")
                continue
            
            p_found, q_found = rfh3.factor(n, timeout=30.0)
            elapsed = time.time() - start
            
            expected = f"{p_true} × {q_true}"
            found = f"{p_found} × {q_found}"
            success = {p_found, q_found} == {p_true, q_true}
            
            results.append({
                'n': n,
                'bits': n.bit_length(),
                'success': success,
                'time': elapsed
            })
            
            status = "✓" if success else "✗"
            print(f"{n:12d} | {n.bit_length():4d} | {expected:^15} | {found:^15} | "
                  f"{elapsed:8.3f}s | {status}")
            
        except Exception as e:
            print(f"{n:12d} | {n.bit_length():4d} | {p_true} × {q_true:^15} | "
                  f"{'ERROR':^15} | {0:8.3f}s | ✗")
    
    print("=" * 85)
    
    # Summary
    if results:
        successes = sum(1 for r in results if r['success'])
        total_time = sum(r['time'] for r in results)
        
        print(f"\nSUMMARY:")
        print(f"  Success Rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Average Time: {total_time/len(results):.3f}s")
        
        # Show phase statistics
        rfh3.print_stats()


if __name__ == "__main__":
    test_ultimate()
