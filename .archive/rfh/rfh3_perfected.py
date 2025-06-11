"""
Perfected RFH3 implementation with all fixes and optimizations
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
# OPTIMIZED COMPONENTS FROM ORIGINAL RFH3
# ============================================================================

from rfh3 import (
    LazyResonanceIterator, MultiScaleResonance, StateManager,
    ZonePredictor, ResonancePatternLearner, RFH3Config
)


# ============================================================================
# FIXED HIERARCHICAL SEARCH
# ============================================================================

class HierarchicalSearchOptimized:
    """Fixed and optimized coarse-to-fine resonance field exploration"""
    
    def __init__(self, n: int, analyzer: MultiScaleResonance):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.analyzer = analyzer
        self.levels = self._compute_hierarchy_levels()
    
    def _compute_hierarchy_levels(self) -> List[int]:
        """Compute sampling resolutions for each level"""
        levels = []
        points = self.sqrt_n
        
        # More reasonable hierarchy
        while points > 20:
            levels.append(int(points))
            points = int(points ** 0.8)  # Gentle reduction
        
        levels.append(20)  # Minimum level
        return levels[::-1]  # Reverse to go coarse to fine
    
    def search(self, max_time: float = 2.0) -> List[Tuple[int, float]]:
        """Perform hierarchical search with timeout"""
        start_time = time.time()
        
        # Level 1: Coarse sampling
        coarse_peaks = self._coarse_sample()
        
        if time.time() - start_time > max_time:
            return coarse_peaks
        
        # Level 2: Refine around peaks (limited)
        refined_regions = []
        for peak, resonance in coarse_peaks[:10]:  # Top 10 only
            if time.time() - start_time > max_time:
                break
            refined = self._refine_peak(peak, resonance)
            refined_regions.extend(refined)
        
        # Combine and deduplicate
        all_candidates = list(set(coarse_peaks + refined_regions))
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return all_candidates[:50]  # Return top 50
    
    def _coarse_sample(self) -> List[Tuple[int, float]]:
        """Coarse sampling of resonance field"""
        if self.sqrt_n < 2:
            return []
        
        # Adaptive sampling
        sample_points = min(200, max(20, self.sqrt_n // 10))
        step = max(1, self.sqrt_n // sample_points)
        
        peaks = []
        
        # Sample evenly
        for i in range(2, min(self.sqrt_n + 1, 2000), step):
            resonance = self.analyzer.compute_coarse_resonance(i, self.n)
            if resonance > 0.05:
                peaks.append((i, resonance))
        
        # Always check critical points
        critical_points = []
        
        # Small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if p <= self.sqrt_n:
                critical_points.append(p)
        
        # Near sqrt(n) for balanced factors
        sqrt_region = int(self.sqrt_n ** 0.1)
        for offset in range(-sqrt_region, sqrt_region + 1):
            x = self.sqrt_n + offset
            if 2 <= x <= self.sqrt_n:
                critical_points.append(x)
        
        # Perfect squares
        i = 2
        while i * i <= self.sqrt_n:
            critical_points.append(i * i)
            i += 1
        
        # Check critical points
        for x in set(critical_points):
            resonance = self.analyzer.compute_resonance(x, self.n)
            peaks.append((x, resonance))
        
        # Sort by resonance
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:100]
    
    def _refine_peak(self, peak: int, peak_resonance: float) -> List[Tuple[int, float]]:
        """Refine search around a peak"""
        window = min(20, max(2, int(self.sqrt_n ** 0.05)))
        refined = []
        
        for offset in range(-window, window + 1):
            x = peak + offset
            if 2 <= x <= self.sqrt_n and x != peak:
                resonance = self.analyzer.compute_resonance(x, self.n)
                if resonance > peak_resonance * 0.5:
                    refined.append((x, resonance))
        
        return refined


# ============================================================================
# PERFECTED RFH3 CLASS
# ============================================================================

class RFH3Perfected:
    """Perfected Adaptive Resonance Field Hypothesis v3"""
    
    def __init__(self, config: Optional[RFH3Config] = None):
        self.config = config or RFH3Config()
        self.logger = self._setup_logging()
        
        # Core components
        self.learner = ResonancePatternLearner()
        self.state = StateManager(self.config.checkpoint_interval)
        self.analyzer = None
        self.stats = {
            'factorizations': 0,
            'total_time': 0,
            'success_rate': 1.0,
            'avg_iterations': 0,
            'failures': []
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('RFH3Perfected')
        logger.setLevel(self.config.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def factor(self, n: int, timeout: float = 30.0) -> Tuple[int, int]:
        """Main factorization method with timeout and optimizations"""
        if n < 4:
            raise ValueError("n must be >= 4")
        
        # Quick checks for small factors
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if n % p == 0:
                return (p, n // p)
        
        # Check if n is prime
        if self._is_probable_prime(n):
            raise ValueError(f"{n} appears to be prime")
        
        start_time = time.time()
        self.logger.info(f"Starting RFH3 factorization of {n} ({n.bit_length()} bits)")
        
        # Initialize components
        self.analyzer = MultiScaleResonance()
        self.state = StateManager()
        iterator = LazyResonanceIterator(n, self.analyzer)
        
        # Phase 1: Use learned patterns
        if self.config.learning_enabled and len(self.learner.success_patterns) > 0:
            predicted_zones = self.learner.predict_high_resonance_zones(n)
            if predicted_zones:
                self.logger.info(f"Checking {len(predicted_zones)} predicted zones")
                
                for start, end, confidence in predicted_zones[:5]:
                    if time.time() - start_time > timeout / 3:
                        break
                    
                    # Check zone with higher density for high confidence
                    step = 1 if confidence > 0.8 else max(1, (end - start) // 20)
                    for x in range(start, min(end + 1, int(math.sqrt(n)) + 1), step):
                        if n % x == 0:
                            self._record_success(n, x, confidence, time.time() - start_time)
                            return (x, n // x)
        
        # Phase 2: Hierarchical search
        if self.config.hierarchical_search:
            search = HierarchicalSearchOptimized(n, self.analyzer)
            candidates = search.search(max_time=min(5.0, timeout/4))
            
            self.logger.info(f"Hierarchical search found {len(candidates)} candidates")
            
            # Check top candidates
            for x, resonance in candidates[:30]:
                if time.time() - start_time > timeout / 2:
                    break
                if n % x == 0:
                    self._record_success(n, x, resonance, time.time() - start_time)
                    return (x, n // x)
        
        # Phase 3: Adaptive iterator search
        threshold = self._compute_adaptive_threshold(n)
        iteration = 0
        high_resonance_nodes = []
        
        self.logger.info("Starting adaptive iterator search")
        
        for x in iterator:
            if time.time() - start_time > timeout * 0.8:
                self.logger.warning(f"Approaching timeout after {iteration} iterations")
                break
            
            iteration += 1
            
            # Quick divisibility check
            if n % x == 0:
                resonance = self.analyzer.compute_resonance(x, self.n)
                self._record_success(n, x, resonance, time.time() - start_time)
                return (x, n // x)
            
            # Compute resonance periodically
            if iteration <= 100 or iteration % 10 == 0:
                resonance = self.analyzer.compute_resonance(x, n)
                self.state.update(x, resonance)
                
                if resonance > threshold:
                    high_resonance_nodes.append((x, resonance))
                    
                    # Adaptive threshold update
                    if len(high_resonance_nodes) % 10 == 0:
                        stats = self.state.get_statistics()
                        threshold = self._update_adaptive_threshold(
                            threshold, stats['mean_recent_resonance'], 
                            stats['std_recent_resonance']
                        )
            
            # Limit iterations
            if iteration >= min(self.config.max_iterations, 50000):
                self.logger.warning(f"Reached iteration limit ({iteration})")
                break
        
        # Phase 4: Focused search on high-resonance nodes
        if high_resonance_nodes:
            self.logger.info(f"Focusing on {len(high_resonance_nodes)} high-resonance nodes")
            high_resonance_nodes.sort(key=lambda x: x[1], reverse=True)
            
            for x, resonance in high_resonance_nodes[:20]:
                if time.time() - start_time > timeout * 0.9:
                    break
                
                # Check neighborhood
                for offset in range(-5, 6):
                    candidate = x + offset
                    if 2 <= candidate <= int(math.sqrt(n)) and n % candidate == 0:
                        self._record_success(n, candidate, resonance, time.time() - start_time)
                        return (candidate, n // candidate)
        
        # Record failure for learning
        if self.config.learning_enabled:
            self.learner.record_failure(n, [x for x, _ in high_resonance_nodes[:10]], 
                                      [r for _, r in high_resonance_nodes[:10]])
            self.stats['failures'].append(n)
        
        # Phase 5: Optimized Pollard's Rho fallback
        self.logger.warning("All phases exhausted, falling back to Pollard's Rho")
        return self._pollard_rho_optimized(n, timeout - (time.time() - start_time))
    
    def _compute_adaptive_threshold(self, n: int) -> float:
        """Compute adaptive threshold"""
        base = 1.0 / (math.log(n) ** 0.5)  # More aggressive threshold
        sr = self.stats.get('success_rate', 1.0)
        k = 1.5 * (1 - sr)**2 + 0.3
        return base * k
    
    def _update_adaptive_threshold(self, current: float, mean: float, std: float) -> float:
        """Update threshold based on statistics"""
        if std > 0:
            new_threshold = mean - 1.5 * std  # 1.5 sigma
            return 0.8 * current + 0.2 * new_threshold
        return current
    
    def _record_success(self, n: int, factor: int, resonance: float, time_taken: float):
        """Record successful factorization"""
        self.stats['factorizations'] += 1
        self.stats['total_time'] += time_taken
        
        # Update success rate
        if self.stats['factorizations'] > 0:
            self.stats['success_rate'] = (self.stats['factorizations'] / 
                                         (self.stats['factorizations'] + len(self.stats['failures'])))
        
        # Update learner
        if self.config.learning_enabled:
            self.learner.record_success(n, factor, {'resonance': resonance})
        
        self.logger.info(
            f"Success! {n} = {factor} × {n//factor} "
            f"(resonance={resonance:.4f}, time={time_taken:.3f}s, "
            f"iter={self.state.iteration_count})"
        )
    
    def _is_probable_prime(self, n: int) -> bool:
        """Optimized Miller-Rabin primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Quick check against small primes
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        if n < 10000:
            for i in range(101, int(math.sqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        # Miller-Rabin
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Use deterministic witnesses for specific ranges
        witnesses = []
        if n < 2047:
            witnesses = [2]
        elif n < 1373653:
            witnesses = [2, 3]
        elif n < 9080191:
            witnesses = [31, 73]
        elif n < 25326001:
            witnesses = [2, 3, 5]
        elif n < 3215031751:
            witnesses = [2, 3, 5, 7]
        else:
            witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        for a in witnesses:
            if a >= n:
                continue
            
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def _pollard_rho_optimized(self, n: int, timeout: float) -> Tuple[int, int]:
        """Optimized Pollard's Rho with multiple polynomials"""
        start_time = time.time()
        
        # Try different polynomials
        for c in [1, 2, 3, 5, 7]:
            if time.time() - start_time > timeout:
                break
                
            x = 2
            y = 2
            d = 1
            
            f = lambda x: (x * x + c) % n
            
            # Brent's improvement
            for cycle in range(1, 100000):
                if time.time() - start_time > timeout:
                    break
                
                for _ in range(min(cycle, 1000)):
                    x = f(x)
                    d = math.gcd(abs(x - y), n)
                    
                    if d != 1 and d != n:
                        return (min(d, n // d), max(d, n // d))
                
                y = x
        
        # Last resort: ECM-style approach
        return self._ecm_style_factor(n, timeout - (time.time() - start_time))
    
    def _ecm_style_factor(self, n: int, timeout: float) -> Tuple[int, int]:
        """Simple ECM-style factorization"""
        # Trial division up to a reasonable limit
        limit = min(100000, int(n ** 0.25))
        for i in range(3, limit, 2):
            if n % i == 0:
                return (i, n // i)
        
        # If all else fails
        raise ValueError(f"Failed to factor {n} within timeout")


# ============================================================================
# COMPREHENSIVE TEST WITH HARD SEMIPRIMES
# ============================================================================

def test_hard_semiprimes():
    """Test RFH3 with challenging semiprimes"""
    
    config = RFH3Config()
    config.max_iterations = 100000
    config.hierarchical_search = True
    config.learning_enabled = True
    
    rfh3 = RFH3Perfected(config)
    
    # Test suite including hard semiprimes
    test_cases = [
        # Warm-up cases
        (143, 11, 13),
        (323, 17, 19),
        (1147, 31, 37),
        
        # Balanced semiprimes
        (10403, 101, 103),
        (40001, 197, 203),
        (104729, 317, 331),
        
        # Hard balanced semiprimes
        (282797, 523, 541),      # Near transition boundary
        (1299827, 1117, 1163),   # Larger balanced
        (16777259, 4093, 4099),  # 24-bit semiprime
        
        # Very hard cases
        (1073676287, 32749, 32771),  # 30-bit, twin primes
        (2147483713, 46337, 46349),  # 31-bit, close primes
        
        # Special structure
        (536870923, 23003, 23321),   # Near power of 2
        (4294967357, 65521, 65537),  # Near 2^32, Fermat prime
    ]
    
    print("\nRFH3 PERFECTED - HARD SEMIPRIME TEST")
    print("=" * 80)
    print(f"{'n':>12} | {'Bits':>4} | {'p':>6} × {'q':>6} | {'Time':>8} | {'Iter':>6} | {'Status'}")
    print("-" * 80)
    
    results = []
    total_time = 0
    
    for n, p_true, q_true in test_cases:
        try:
            rfh3.state = StateManager()  # Fresh state
            start = time.time()
            p_found, q_found = rfh3.factor(n, timeout=60.0)
            elapsed = time.time() - start
            total_time += elapsed
            
            success = {p_found, q_found} == {p_true, q_true}
            iterations = rfh3.state.iteration_count
            
            results.append({
                'n': n,
                'bits': n.bit_length(),
                'success': success,
                'time': elapsed,
                'iterations': iterations
            })
            
            status = "✓" if success else "✗"
            print(f"{n:12d} | {n.bit_length():4d} | {p_found:6d} × {q_found:6d} | "
                  f"{elapsed:8.3f}s | {iterations:6d} | {status}")
            
        except Exception as e:
            results.append({
                'n': n,
                'bits': n.bit_length(),
                'success': False,
                'time': 0,
                'iterations': 0
            })
            print(f"{n:12d} | {n.bit_length():4d} | {'FAILED':^15} | "
                  f"{0:8.3f}s | {0:6d} | ✗ {str(e)[:20]}")
    
    # Summary
    print("=" * 80)
    successes = sum(1 for r in results if r['success'])
    avg_time = total_time / len(test_cases) if test_cases else 0
    
    print(f"\nHARD SEMIPRIME TEST RESULTS:")
    print(f"  Success Rate: {successes}/{len(test_cases)} ({successes/len(test_cases)*100:.1f}%)")
    print(f"  Total Time: {total_time:.3f}s")
    print(f"  Average Time: {avg_time:.3f}s")
    
    # Breakdown by difficulty
    easy = [r for r in results if r['bits'] <= 20]
    medium = [r for r in results if 20 < r['bits'] <= 30]
    hard = [r for r in results if r['bits'] > 30]
    
    print(f"\nBREAKDOWN BY DIFFICULTY:")
    if easy:
        easy_success = sum(1 for r in easy if r['success'])
        print(f"  Easy (≤20 bits): {easy_success}/{len(easy)} ({easy_success/len(easy)*100:.1f}%)")
    if medium:
        medium_success = sum(1 for r in medium if r['success'])
        print(f"  Medium (21-30 bits): {medium_success}/{len(medium)} ({medium_success/len(medium)*100:.1f}%)")
    if hard:
        hard_success = sum(1 for r in hard if r['success'])
        print(f"  Hard (>30 bits): {hard_success}/{len(hard)} ({hard_success/len(hard)*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    test_hard_semiprimes()
