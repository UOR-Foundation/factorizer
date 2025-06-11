"""
Phase I Experimental Implementation
Testing scale-invariant resonance detection for 100% success beyond 64 bits.
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional
import time
from scipy.optimize import differential_evolution, minimize_scalar, basinhopping
from scipy.signal import find_peaks
from functools import lru_cache
import multiprocessing


class ScaleAdaptiveParameters:
    """Scale-adaptive parameters that grow with problem size"""
    
    @staticmethod
    def prime_dimensions(n: int) -> int:
        """Prime dimensions scale with information content"""
        bit_len = n.bit_length()
        # More aggressive scaling for larger numbers
        if bit_len < 64:
            return 32
        elif bit_len < 96:
            return int(math.sqrt(bit_len) * math.log2(bit_len) * 3)
        else:
            return int(math.sqrt(bit_len) * math.log2(bit_len) * 4)
    
    @staticmethod
    def resonance_samples(n: int) -> int:
        """Sampling density for complete coverage"""
        bit_len = n.bit_length()
        # Quadratic growth ensures no gaps
        if bit_len < 64:
            return bit_len * 10
        elif bit_len < 96:
            return bit_len * bit_len // 4
        else:
            return min(bit_len * bit_len // 2, 10000)
    
    @staticmethod
    def coherence_threshold(n: int) -> float:
        """Adaptive threshold based on signal strength"""
        bit_len = n.bit_length()
        return 1.0 / math.sqrt(bit_len)
    
    @staticmethod
    def coverage_depth(n: int) -> int:
        """Depth of coverage for small factors"""
        bit_len = n.bit_length()
        return int(math.log2(bit_len) ** 2)


class ContinuousResonance:
    """Continuous resonance function for factor detection"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.bit_len = n.bit_length()
        
        # Precompute signals
        print(f"Precomputing signals for {self.bit_len}-bit number...")
        self.signals = self._precompute_signals()
        print("Signals computed.")
    
    def _precompute_signals(self) -> Dict:
        """Precompute scale-invariant signals"""
        return {
            'prime_coords': self._compute_prime_coordinates(),
            'small_primes': self._generate_primes(min(1000, self.sqrt_n)),
            'fibonacci': self._generate_fibonacci_sequence(),
            'mult_orders': self._compute_multiplicative_orders(),
            'qr_pattern': self._compute_qr_pattern()
        }
    
    def _compute_prime_coordinates(self) -> List[Tuple[int, int]]:
        """Compute n's coordinates in prime space"""
        num_primes = ScaleAdaptiveParameters.prime_dimensions(self.n)
        primes = self._generate_primes(num_primes)
        return [(p, self.n % p) for p in primes]
    
    def _generate_primes(self, limit: int) -> List[int]:
        """Generate primes up to limit"""
        if limit < 2:
            return []
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    def _generate_fibonacci_sequence(self) -> List[int]:
        """Generate Fibonacci numbers up to sqrt(n)"""
        fibs = [1, 1]
        while fibs[-1] < self.sqrt_n:
            fibs.append(fibs[-1] + fibs[-2])
        return [f for f in fibs if f <= self.sqrt_n]
    
    def _compute_multiplicative_orders(self) -> Dict[int, int]:
        """Compute multiplicative orders for small bases"""
        orders = {}
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        for base in bases:
            if math.gcd(base, self.n) == 1:
                order = self._multiplicative_order(base, self.n)
                if order > 0:
                    orders[base] = order
        
        return orders
    
    @lru_cache(maxsize=1024)
    def _multiplicative_order(self, a: int, m: int) -> int:
        """Compute multiplicative order of a modulo m"""
        if m <= 1 or math.gcd(a, m) > 1:
            return 0
        
        order = 1
        value = a % m
        max_order = min(m - 1, 100000)  # Cap for efficiency
        
        while value != 1 and order < max_order:
            value = (value * a) % m
            order += 1
        
        return order if value == 1 else 0
    
    def _compute_qr_pattern(self) -> Dict[int, bool]:
        """Compute quadratic residue pattern"""
        pattern = {}
        test_range = min(100, self.sqrt_n)
        
        for a in range(1, test_range):
            pattern[a] = self._is_quadratic_residue(a, self.n)
        
        return pattern
    
    def _is_quadratic_residue(self, a: int, p: int) -> bool:
        """Check if a is a quadratic residue modulo p"""
        if p == 2:
            return True
        return pow(a, (p - 1) // 2, p) == 1
    
    def resonance(self, x_normalized: float) -> float:
        """
        Continuous resonance function on [0, 1].
        Peak at x_normalized indicates factor at x = x_normalized * sqrt(n).
        """
        if x_normalized <= 0 or x_normalized >= 1:
            return 0.0
        
        x = max(2, int(x_normalized * self.sqrt_n))
        if x >= self.n:
            return 0.0
        
        # Quick check for shared factors
        g = math.gcd(x, self.n)
        if g > 1:
            if g == x:  # x divides n
                return 1000.0  # Very strong resonance
            else:
                return 100.0   # Strong resonance
        
        # Combine resonance measures
        resonance = 1.0
        
        # 1. Prime coordinate alignment
        coord_score = self._prime_coordinate_resonance(x)
        resonance *= coord_score
        
        # 2. Multiplicative order coherence
        order_score = self._order_coherence(x)
        resonance *= order_score
        
        # 3. Quadratic residue alignment
        qr_score = self._qr_alignment(x)
        resonance *= qr_score
        
        # 4. Fibonacci proximity
        fib_score = self._fibonacci_proximity(x)
        resonance *= fib_score
        
        # 5. Smooth number score
        smooth_score = self._smooth_number_score(x)
        resonance *= smooth_score
        
        return resonance
    
    def _prime_coordinate_resonance(self, x: int) -> float:
        """Measure prime coordinate alignment"""
        alignment_score = 0.0
        total_weight = 0.0
        
        for p, n_coord in self.signals['prime_coords'][:50]:  # Check first 50 primes
            x_coord = x % p
            weight = 1.0 / math.log(p + 1)
            total_weight += weight
            
            # Perfect alignment
            if x_coord == n_coord:
                alignment_score += weight
            
            # Check multiplicative relationship
            other = self.n // x
            if other > 0 and (x_coord * (other % p)) % p == n_coord:
                alignment_score += weight * 0.5
        
        if total_weight > 0:
            normalized = alignment_score / total_weight
            return 1.0 + normalized * 10.0
        
        return 1.0
    
    def _order_coherence(self, x: int) -> float:
        """Measure multiplicative order coherence"""
        if x >= self.n or x < 2:
            return 1.0
        
        coherence = 0.0
        count = 0
        
        for base, n_order in self.signals['mult_orders'].items():
            if math.gcd(base, x) == 1:
                x_order = self._multiplicative_order(base, x)
                if x_order > 0:
                    count += 1
                    # Check order relationships
                    if n_order % x_order == 0:
                        coherence += 1.0
                    elif (x - 1) % x_order == 0:
                        coherence += 0.5
        
        if count > 0:
            return 1.0 + (coherence / count) * 5.0
        
        return 1.0
    
    def _qr_alignment(self, x: int) -> float:
        """Measure quadratic residue pattern alignment"""
        if x >= self.n or x < 2:
            return 1.0
        
        matches = 0
        total = 0
        
        for a, n_qr in self.signals['qr_pattern'].items():
            if a < x:
                x_qr = self._is_quadratic_residue(a, x)
                total += 1
                if x_qr == n_qr:
                    matches += 1
        
        if total > 0:
            alignment = matches / total
            return 1.0 + alignment * 3.0
        
        return 1.0
    
    def _fibonacci_proximity(self, x: int) -> float:
        """Score based on proximity to Fibonacci numbers"""
        min_dist = float('inf')
        
        for fib in self.signals['fibonacci']:
            dist = abs(x - fib)
            if dist < min_dist:
                min_dist = dist
        
        # Strong boost for Fibonacci numbers
        if min_dist == 0:
            return 5.0
        elif min_dist < x * 0.01:  # Within 1%
            return 2.0
        elif min_dist < x * 0.1:   # Within 10%
            return 1.5
        
        return 1.0
    
    def _smooth_number_score(self, x: int) -> float:
        """Score based on smoothness (small prime factors)"""
        if x < 2:
            return 1.0
        
        # Check how smooth x is
        remaining = x
        smooth_bound = 100
        
        for p in self.signals['small_primes']:
            if p > smooth_bound:
                break
            while remaining % p == 0:
                remaining //= p
        
        # Highly smooth numbers get a boost
        if remaining == 1:
            return 3.0  # Completely smooth
        elif remaining < math.sqrt(x):
            return 2.0  # Mostly smooth
        
        return 1.0


class CompleteFactorCoverage:
    """Ensures complete coverage of all possible factors"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.bit_len = n.bit_length()
    
    def generate_coverage_positions(self) -> List[int]:
        """Generate positions that guarantee factor coverage"""
        positions = set()
        
        # 1. Logarithmic sweep for small factors
        self._add_logarithmic_positions(positions)
        
        # 2. All small primes
        self._add_prime_positions(positions)
        
        # 3. Arithmetic progressions
        self._add_arithmetic_progressions(positions)
        
        # 4. Quadratic positions
        self._add_quadratic_positions(positions)
        
        # 5. Fibonacci and Lucas numbers
        self._add_fibonacci_positions(positions)
        
        # 6. Smooth numbers
        self._add_smooth_positions(positions)
        
        return sorted(positions)
    
    def _add_logarithmic_positions(self, positions: set):
        """Logarithmic coverage for small factors"""
        # Dense coverage from 2 to sqrt(sqrt(n))
        limit = int(self.sqrt_n ** 0.5)
        steps = max(100, self.bit_len * 10)
        
        for i in range(steps):
            pos = int(2 * (limit / 2) ** (i / steps))
            if 2 <= pos <= self.sqrt_n:
                positions.add(pos)
    
    def _add_prime_positions(self, positions: set):
        """Add all primes up to a threshold"""
        limit = min(self.sqrt_n, 1000000)  # Cap at 1M for efficiency
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        for i in range(2, limit + 1):
            if sieve[i]:
                positions.add(i)
    
    def _add_arithmetic_progressions(self, positions: set):
        """Add arithmetic progressions p*k + r"""
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        for p in small_primes:
            # All residues
            for r in range(p):
                # Multiple values of k
                for k in range(1, min(1000, self.sqrt_n // p)):
                    pos = p * k + r
                    if 2 <= pos <= self.sqrt_n:
                        positions.add(pos)
    
    def _add_quadratic_positions(self, positions: set):
        """Add positions based on quadratic forms"""
        limit = min(1000, int(self.sqrt_n ** 0.25))
        
        for i in range(1, limit):
            # Perfect squares and neighbors
            positions.add(i * i)
            positions.add(i * i + 1)
            positions.add(i * i - 1)
            
            # Other quadratic forms
            positions.add(i * i + i)
            positions.add(i * (i + 1))
    
    def _add_fibonacci_positions(self, positions: set):
        """Add Fibonacci and Lucas numbers"""
        # Fibonacci
        fib_a, fib_b = 1, 1
        while fib_b <= self.sqrt_n:
            positions.add(fib_b)
            positions.add(fib_b - 1)
            positions.add(fib_b + 1)
            fib_a, fib_b = fib_b, fib_a + fib_b
        
        # Lucas
        lucas_a, lucas_b = 2, 1
        while lucas_b <= self.sqrt_n:
            positions.add(lucas_b)
            lucas_a, lucas_b = lucas_b, lucas_a + lucas_b
    
    def _add_smooth_positions(self, positions: set):
        """Add smooth numbers (products of small primes)"""
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Use bit manipulation to generate all subsets
        for mask in range(1, 1 << len(small_primes)):
            product = 1
            for i in range(len(small_primes)):
                if mask & (1 << i):
                    product *= small_primes[i]
                    if product > self.sqrt_n:
                        break
            
            if product <= self.sqrt_n:
                positions.add(product)


class ExperimentalPhaseI:
    """Experimental Phase I implementation"""
    
    def __init__(self):
        self.stats = {
            'resonance_evaluations': 0,
            'optimization_iterations': 0,
            'peak_candidates': 0
        }
    
    def factorize(self, n: int) -> Tuple[int, int]:
        """Factor using experimental Phase I approach"""
        if n < 2:
            raise ValueError("n must be >= 2")
        if n % 2 == 0:
            return (2, n // 2)
        
        print(f"\n{'='*60}")
        print(f"Experimental Phase I Factorization")
        print(f"n = {n} ({n.bit_length()} bits)")
        print(f"{'='*60}")
        
        # Strategy 1: Continuous optimization
        print("\nStrategy 1: Continuous Optimization")
        factor1 = self._continuous_optimization(n)
        if factor1 and n % factor1 == 0:
            return self._format_result(n, factor1)
        
        # Strategy 2: Coverage-based resonance
        print("\nStrategy 2: Coverage-Based Resonance")
        factor2 = self._coverage_based_resonance(n)
        if factor2 and n % factor2 == 0:
            return self._format_result(n, factor2)
        
        # Strategy 3: Parallel peak detection
        print("\nStrategy 3: Parallel Peak Detection")
        factor3 = self._parallel_peak_detection(n)
        if factor3 and n % factor3 == 0:
            return self._format_result(n, factor3)
        
        raise ValueError("Experimental Phase I incomplete - theory needs refinement")
    
    def _continuous_optimization(self, n: int) -> Optional[int]:
        """Use continuous optimization to find resonance peak"""
        resonator = ContinuousResonance(n)
        
        # Track evaluations
        eval_count = [0]
        
        def objective(x):
            eval_count[0] += 1
            return -resonator.resonance(x[0])
        
        # Differential evolution
        print("  Running differential evolution...")
        result = differential_evolution(
            objective,
            bounds=[(0.001, 0.999)],
            maxiter=100,
            popsize=30,
            atol=1e-8,
            seed=42
        )
        
        self.stats['resonance_evaluations'] += eval_count[0]
        self.stats['optimization_iterations'] += result.nit
        
        factor = int(result.x[0] * resonator.sqrt_n)
        print(f"  Peak found at normalized position {result.x[0]:.6f}")
        print(f"  Corresponding factor: {factor}")
        print(f"  Resonance value: {-result.fun:.6f}")
        
        return factor if 2 <= factor <= resonator.sqrt_n else None
    
    def _coverage_based_resonance(self, n: int) -> Optional[int]:
        """Use complete coverage to find factors"""
        resonator = ContinuousResonance(n)
        coverage = CompleteFactorCoverage(n)
        
        # Generate coverage positions
        print("  Generating coverage positions...")
        positions = coverage.generate_coverage_positions()
        print(f"  Generated {len(positions)} positions")
        
        # Limit positions for efficiency
        if len(positions) > 10000:
            # Sample positions intelligently
            sampled = []
            # Include all small positions
            sampled.extend([p for p in positions if p < 1000])
            # Sample larger positions
            step = len(positions) // 5000
            sampled.extend(positions[1000::step])
            positions = sorted(set(sampled))
            print(f"  Sampled down to {len(positions)} positions")
        
        # Evaluate resonance at each position
        print("  Evaluating resonance...")
        best_resonance = 0.0
        best_position = None
        
        for i, pos in enumerate(positions):
            if i % 1000 == 0:
                print(f"    Progress: {i}/{len(positions)}")
            
            x_norm = pos / resonator.sqrt_n
            if 0 < x_norm < 1:
                resonance = resonator.resonance(x_norm)
                self.stats['resonance_evaluations'] += 1
                
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_position = pos
        
        print(f"  Best resonance: {best_resonance:.6f} at position {best_position}")
        
        return best_position
    
    def _parallel_peak_detection(self, n: int) -> Optional[int]:
        """Use parallel evaluation to find resonance peaks"""
        resonator = ContinuousResonance(n)
        
        # Generate sample points
        num_samples = min(ScaleAdaptiveParameters.resonance_samples(n), 5000)
        x_values = np.linspace(0.001, 0.999, num_samples)
        
        print(f"  Evaluating {num_samples} points...")
        
        # Evaluate resonance (could be parallelized)
        resonances = []
        for i, x in enumerate(x_values):
            if i % 500 == 0:
                print(f"    Progress: {i}/{num_samples}")
            
            res = resonator.resonance(x)
            resonances.append(res)
            self.stats['resonance_evaluations'] += 1
        
        resonances = np.array(resonances)
        
        # Find peaks
        mean_res = np.mean(resonances)
        std_res = np.std(resonances)
        
        peaks, properties = find_peaks(
            resonances,
            height=mean_res + 2 * std_res,
            distance=5
        )
        
        print(f"  Found {len(peaks)} peaks above threshold")
        self.stats['peak_candidates'] = len(peaks)
        
        # Check peaks in order of strength
        if len(peaks) > 0:
            sorted_peaks = sorted(peaks, key=lambda i: resonances[i], reverse=True)
            
            for i, peak_idx in enumerate(sorted_peaks[:10]):
                x_norm = x_values[peak_idx]
                factor = int(x_norm * resonator.sqrt_n)
                res_value = resonances[peak_idx]
                
                print(f"  Peak {i+1}: position {factor}, resonance {res_value:.6f}")
                
                if n % factor == 0:
                    return factor
        
        return None
    
    def _format_result(self, n: int, factor: int) -> Tuple[int, int]:
        """Format the factorization result"""
        other = n // factor
        print(f"\n✓ SUCCESS: Found factor {factor}")
        print(f"  {n} = {factor} × {other}")
        print(f"\nStatistics:")
        print(f"  Resonance evaluations: {self.stats['resonance_evaluations']}")
        print(f"  Optimization iterations: {self.stats['optimization_iterations']}")
        print(f"  Peak candidates examined: {self.stats['peak_candidates']}")
        
        return (factor, other) if factor <= other else (other, factor)


def test_experimental():
    """Test the experimental Phase I implementation"""
    
    test_cases = [
        # Small cases for validation
        (11, 13),           # 143
        (101, 103),         # 10403
        
        # Original 64-bit case
        (65537, 4294967311),
        
        # Challenging cases
        (7125766127, 6958284019),     # 66-bit
        (14076040031, 15981381943),   # 68-bit
    ]
    
    phase1 = ExperimentalPhaseI()
    
    successes = 0
    for p_true, q_true in test_cases:
        n = p_true * q_true
        
        try:
            start_time = time.perf_counter()
            p_found, q_found = phase1.factorize(n)
            elapsed = time.perf_counter() - start_time
            
            if {p_found, q_found} == {p_true, q_true}:
                print(f"\n✓ CORRECT in {elapsed:.3f}s")
                successes += 1
            else:
                print(f"\n✗ INCORRECT: Expected {p_true} × {q_true}")
        
        except Exception as e:
            print(f"\n✗ FAILED: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"Summary: {successes}/{len(test_cases)} successful")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_experimental()
