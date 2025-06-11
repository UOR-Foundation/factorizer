"""
Fixed RFH3 implementation with corrected hierarchical search
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
# FIXED HIERARCHICAL SEARCH
# ============================================================================

class HierarchicalSearchFixed:
    """Fixed coarse-to-fine resonance field exploration"""
    
    def __init__(self, n: int, analyzer: 'MultiScaleResonance'):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.analyzer = analyzer
        self.levels = self._compute_hierarchy_levels()
    
    def _compute_hierarchy_levels(self) -> List[int]:
        """Compute sampling resolutions for each level"""
        levels = []
        points = self.sqrt_n
        
        # More reasonable hierarchy
        while points > 10:
            levels.append(int(points))
            points = int(points ** 0.7)  # Less aggressive reduction
        
        levels.append(10)  # Minimum level
        return levels[::-1]  # Reverse to go coarse to fine
    
    def search(self, max_time: float = 1.0) -> List[Tuple[int, float]]:
        """Perform hierarchical search with timeout"""
        start_time = time.time()
        
        # Level 1: Coarse sampling
        coarse_peaks = self._coarse_sample()
        
        if time.time() - start_time > max_time:
            return coarse_peaks
        
        # Level 2: Refine around peaks
        refined_regions = []
        for peak, resonance in coarse_peaks[:5]:  # Limit refinement
            if time.time() - start_time > max_time:
                break
            refined = self._refine_peak(peak, resonance)
            refined_regions.extend(refined)
        
        # Combine all candidates
        all_candidates = coarse_peaks + refined_regions
        
        # Remove duplicates and sort
        seen = set()
        unique_candidates = []
        for x, res in all_candidates:
            if x not in seen:
                seen.add(x)
                unique_candidates.append((x, res))
        
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        return unique_candidates[:100]
    
    def _coarse_sample(self) -> List[Tuple[int, float]]:
        """Coarse sampling of resonance field"""
        if not self.levels or self.sqrt_n < 2:
            return []
        
        # More reasonable sampling
        sample_points = min(100, self.sqrt_n // 2)
        step = max(1, self.sqrt_n // sample_points)
        
        peaks = []
        for i in range(2, min(self.sqrt_n + 1, 1000), step):
            resonance = self.analyzer.compute_coarse_resonance(i, self.n)
            if resonance > 0.05:  # Lower threshold
                peaks.append((i, resonance))
        
        # Always check small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if p <= self.sqrt_n:
                resonance = self.analyzer.compute_coarse_resonance(p, self.n)
                peaks.append((p, resonance))
        
        # Sort by resonance
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:50]  # Limit results
    
    def _refine_peak(self, peak: int, peak_resonance: float) -> List[Tuple[int, float]]:
        """Refine search around a peak"""
        # Smaller window
        window = min(10, int(self.sqrt_n ** 0.05))
        refined = []
        
        # Check around peak
        for offset in range(-window, window + 1):
            x = peak + offset
            if 2 <= x <= self.sqrt_n:
                resonance = self.analyzer.compute_resonance(x, self.n)
                if resonance > peak_resonance * 0.3:
                    refined.append((x, resonance))
        
        return refined


# ============================================================================
# FIXED RFH3 CLASS
# ============================================================================

class RFH3Fixed:
    """Fixed Adaptive Resonance Field Hypothesis v3"""
    
    def __init__(self, config: Optional['RFH3Config'] = None):
        from rfh3 import RFH3Config, MultiScaleResonance, LazyResonanceIterator
        from rfh3 import StateManager, ResonancePatternLearner
        
        self.config = config or RFH3Config()
        self.logger = self._setup_logging()
        
        # Core components
        self.learner = ResonancePatternLearner()
        self.state = StateManager(self.config.checkpoint_interval)
        self.analyzer = None  # Initialized per factorization
        self.stats = {
            'factorizations': 0,
            'total_time': 0,
            'success_rate': 1.0,
            'avg_iterations': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('RFH3Fixed')
        logger.setLevel(self.config.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def factor(self, n: int, timeout: float = 10.0) -> Tuple[int, int]:
        """Main factorization method with timeout"""
        from rfh3 import MultiScaleResonance, LazyResonanceIterator
        
        if n < 4:
            raise ValueError("n must be >= 4")
        
        # Check trivial cases first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if n % p == 0:
                return (p, n // p)
        
        # Check if n is prime (quick test)
        if self._is_probable_prime(n):
            raise ValueError(f"{n} appears to be prime")
        
        start_time = time.time()
        self.logger.info(f"Starting RFH3 factorization of {n} ({n.bit_length()} bits)")
        
        # Initialize components
        self.analyzer = MultiScaleResonance()
        self.state = StateManager()  # Fresh state
        iterator = LazyResonanceIterator(n, self.analyzer)
        
        # Use learned patterns to predict high-resonance zones
        predicted_zones = []
        if self.config.learning_enabled and len(self.learner.success_patterns) > 0:
            predicted_zones = self.learner.predict_high_resonance_zones(n)
            if predicted_zones:
                self.logger.info(f"Predicted {len(predicted_zones)} high-resonance zones")
        
        # Hierarchical search if enabled
        if self.config.hierarchical_search:
            search = HierarchicalSearchFixed(n, self.analyzer)
            candidates = search.search(max_time=min(2.0, timeout/5))
            
            # Check hierarchical candidates first
            for x, resonance in candidates[:20]:  # Limit checks
                if time.time() - start_time > timeout:
                    break
                if n % x == 0:
                    factor = x
                    other = n // x
                    self._record_success(n, factor, resonance, time.time() - start_time)
                    return (min(factor, other), max(factor, other))
        
        # Adaptive threshold computation
        threshold = self._compute_adaptive_threshold(n)
        
        # Main adaptive search loop
        iteration = 0
        tested_positions = []
        resonances = []
        
        for x in iterator:
            if time.time() - start_time > timeout:
                self.logger.warning(f"Timeout after {iteration} iterations")
                break
                
            iteration += 1
            
            # Quick divisibility check first
            if n % x == 0:
                self.logger.info(f"Found factor {x} at iteration {iteration}")
                factor = x
                other = n // x
                resonance = self.analyzer.compute_resonance(x, n)
                self._record_success(n, factor, resonance, time.time() - start_time)
                return (min(factor, other), max(factor, other))
            
            # Compute resonance only if needed
            if iteration < 1000 or iteration % 10 == 0:
                resonance = self.analyzer.compute_resonance(x, n)
                self.state.update(x, resonance)
                tested_positions.append(x)
                resonances.append(resonance)
            
            # Check termination conditions
            if iteration >= min(self.config.max_iterations, 10000):
                self.logger.warning(f"Reached max iterations ({iteration})")
                break
        
        # Record failure for learning
        if self.config.learning_enabled and tested_positions:
            self.learner.record_failure(n, tested_positions[-10:], resonances[-10:])
        
        # Fallback to Pollard's Rho
        self.logger.warning("RFH3 exhausted, falling back to Pollard's Rho")
        return self._pollard_rho_fallback(n)
    
    def _compute_adaptive_threshold(self, n: int) -> float:
        """Compute initial adaptive threshold"""
        # Base threshold from theory
        base = 1.0 / math.log(n)
        
        # Adjust based on success rate
        sr = self.stats.get('success_rate', 1.0)
        k = 2.0 * (1 - sr)**2 + 0.5
        
        return base * k
    
    def _record_success(self, n: int, factor: int, resonance: float, time_taken: float):
        """Record successful factorization"""
        self.stats['factorizations'] += 1
        self.stats['total_time'] += time_taken
        self.stats['avg_iterations'] = (
            self.state.iteration_count / self.stats['factorizations']
            if self.stats['factorizations'] > 0 else 0
        )
        
        # Update learner
        if self.config.learning_enabled:
            self.learner.record_success(n, factor, {'resonance': resonance})
        
        self.logger.info(
            f"Success! {n} = {factor} × {n//factor} "
            f"(resonance={resonance:.4f}, time={time_taken:.3f}s)"
        )
    
    def _is_probable_prime(self, n: int) -> bool:
        """Quick primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Check small primes
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        # For small numbers, check all
        if n < 2500:
            for i in range(51, int(math.sqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        # Miller-Rabin for larger numbers
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Test with a few witnesses
        for a in [2, 3, 5, 7, 11]:
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
    
    def _pollard_rho_fallback(self, n: int) -> Tuple[int, int]:
        """Pollard's Rho as fallback"""
        if n == 4:
            return (2, 2)
        
        # Try Pollard's Rho
        for c in [1, 2]:
            x = 2
            y = 2
            d = 1
            
            f = lambda x: (x * x + c) % n
            
            iterations = 0
            max_iter = min(10000, n)
            
            while d == 1 and iterations < max_iter:
                x = f(x)
                y = f(f(y))
                d = math.gcd(abs(x - y), n)
                iterations += 1
                
                if d != 1 and d != n:
                    return (min(d, n // d), max(d, n // d))
        
        # Last resort: trial division
        sqrt_n = int(math.sqrt(n))
        for i in range(3, min(10000, sqrt_n + 1), 2):
            if n % i == 0:
                return (i, n // i)
        
        raise ValueError(f"Failed to factor {n}")


def test_fixed():
    """Test the fixed RFH3 implementation"""
    from rfh3 import RFH3Config
    
    config = RFH3Config()
    config.max_iterations = 5000
    config.hierarchical_search = True
    
    rfh3 = RFH3Fixed(config)
    
    test_cases = [
        (35, 5, 7),
        (143, 11, 13),
        (323, 17, 19),
        (1147, 31, 37),
        (10403, 101, 103),
    ]
    
    print("Testing Fixed RFH3 Implementation")
    print("=" * 50)
    
    successes = 0
    for n, p_true, q_true in test_cases:
        try:
            start = time.time()
            p_found, q_found = rfh3.factor(n, timeout=2.0)
            elapsed = time.time() - start
            
            if {p_found, q_found} == {p_true, q_true}:
                print(f"✓ {n:6d} = {p_found:4d} × {q_found:4d} ({elapsed:.3f}s)")
                successes += 1
            else:
                print(f"✗ {n:6d}: Expected {p_true} × {q_true}, got {p_found} × {q_found}")
        except Exception as e:
            print(f"✗ {n:6d}: FAILED - {str(e)}")
    
    print("=" * 50)
    print(f"Success rate: {successes}/{len(test_cases)} ({successes/len(test_cases)*100:.1f}%)")


if __name__ == "__main__":
    test_fixed()
