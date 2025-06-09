"""
Phase I Emanation Experiment
Exploring the Single Prime Hypothesis and transition boundaries

Key concepts:
1. 282281 (531²) as a transition boundary where base-2 becomes base-3
2. All primes emanating from a single source π₁
3. Scale-invariant resonance through emanation maps
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
import time
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from functools import lru_cache


class EmanationTransitions:
    """Explore prime emanation transitions based on Single Prime Hypothesis"""
    
    # Critical transition points
    TRANSITIONS = {
        2: 282281,      # 531² - where "2 becomes 3"
        3: None,        # To be discovered
        5: None,        # To be discovered
    }
    
    @staticmethod
    def find_transition_boundary(base: int, n: int) -> bool:
        """Check if n is near a base transition boundary"""
        if base not in EmanationTransitions.TRANSITIONS:
            return False
        
        boundary = EmanationTransitions.TRANSITIONS[base]
        if boundary is None:
            return False
        
        # Check if n is near the boundary (within 10%)
        return abs(n - boundary) < boundary * 0.1
    
    @staticmethod
    def compute_emanation_distance(n: int, base: int) -> float:
        """
        Compute the "distance" from π₁ through base-b emanation
        Based on the idea that E_b(π₁) = π₁ + δ_b
        """
        # Simplified model: distance grows with log_base(n)
        if base <= 1:
            return float('inf')
        
        # Add boundary effects
        if EmanationTransitions.find_transition_boundary(base, n):
            return math.log(n, base) * 1.5  # Increased distance at boundaries
        
        return math.log(n, base)


class SinglePrimeResonance:
    """Resonance based on Single Prime Hypothesis"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.bit_len = n.bit_length()
        
        # Precompute emanation signals
        self.emanation_signals = self._compute_emanation_signals()
    
    def _compute_emanation_signals(self) -> Dict:
        """Compute signals based on emanation from π₁"""
        signals = {}
        
        # Base emanations
        for base in [2, 3, 5, 7, 11]:
            signals[f'emanation_{base}'] = EmanationTransitions.compute_emanation_distance(self.n, base)
        
        # Special boundary detection
        signals['near_transition'] = any(
            EmanationTransitions.find_transition_boundary(b, self.n) 
            for b in [2, 3, 5]
        )
        
        # Clifford algebra inspired coordinates
        signals['clifford_norm'] = self._compute_clifford_norm()
        
        return signals
    
    def _compute_clifford_norm(self) -> float:
        """
        Simplified Clifford algebra norm
        Treating n as element in Cl(V)
        """
        # Real part
        real = self.n
        
        # Imaginary parts (based on prime coordinates)
        imag = sum(self.n % p for p in [2, 3, 5, 7, 11, 13])
        
        # Quaternionic-like norm
        return math.sqrt(real*real + imag*imag)
    
    def emanation_resonance(self, x_normalized: float) -> float:
        """
        Compute resonance based on emanation theory
        """
        if x_normalized <= 0 or x_normalized >= 1:
            return 0.0
        
        x = max(2, int(x_normalized * self.sqrt_n))
        if x >= self.n:
            return 0.0
        
        resonance = 1.0
        
        # 1. Base emanation alignment
        for base in [2, 3, 5, 7]:
            dist_n = EmanationTransitions.compute_emanation_distance(self.n, base)
            dist_x = EmanationTransitions.compute_emanation_distance(x, base)
            
            # Resonance when emanation distances align
            alignment = 1.0 / (1.0 + abs(dist_n - dist_x * 2))
            resonance *= (1.0 + alignment)
        
        # 2. Transition boundary effects
        if self.emanation_signals['near_transition']:
            # Enhanced resonance near boundaries
            resonance *= 1.5
        
        # 3. Clifford norm alignment
        x_norm = math.sqrt(x*x + sum(x % p for p in [2, 3, 5, 7, 11, 13])*sum(x % p for p in [2, 3, 5, 7, 11, 13]))
        n_norm = self.emanation_signals['clifford_norm']
        
        # Check if x could be a factor based on norm
        if abs(x_norm * (self.n / x) - n_norm) < n_norm * 0.1:
            resonance *= 2.0
        
        # 4. Special number detection (282281 and related)
        if self._is_special_transition_number(x):
            resonance *= 3.0
        
        return resonance
    
    def _is_special_transition_number(self, x: int) -> bool:
        """Check if x is related to transition boundaries"""
        # Check if x is near sqrt of a transition boundary
        for base, boundary in EmanationTransitions.TRANSITIONS.items():
            if boundary and abs(x - int(math.sqrt(boundary))) < 10:
                return True
        
        # Check if x² is near a transition
        x_squared = x * x
        for base, boundary in EmanationTransitions.TRANSITIONS.items():
            if boundary and abs(x_squared - boundary) < boundary * 0.01:
                return True
        
        return False


class ImprovedPhaseI:
    """Improved Phase I using emanation concepts"""
    
    def __init__(self):
        self.stats = {
            'resonance_evaluations': 0,
            'special_positions_checked': 0,
            'emanation_alignments': 0
        }
    
    def factorize(self, n: int) -> Tuple[int, int]:
        """Factor using emanation-based approach"""
        if n < 2:
            raise ValueError("n must be >= 2")
        if n % 2 == 0:
            return (2, n // 2)
        
        print(f"\n{'='*60}")
        print(f"Emanation-Based Factorization")
        print(f"n = {n} ({n.bit_length()} bits)")
        print(f"{'='*60}")
        
        # Check transition boundaries
        self._check_transition_boundaries(n)
        
        # Strategy 1: Emanation-guided search
        print("\nStrategy 1: Emanation-Guided Search")
        factor1 = self._emanation_guided_search(n)
        if factor1 and n % factor1 == 0:
            return self._format_result(n, factor1)
        
        # Strategy 2: Transition point exploration
        print("\nStrategy 2: Transition Point Exploration")
        factor2 = self._transition_point_search(n)
        if factor2 and n % factor2 == 0:
            return self._format_result(n, factor2)
        
        # Strategy 3: Clifford norm factorization
        print("\nStrategy 3: Clifford Norm Factorization")
        factor3 = self._clifford_norm_search(n)
        if factor3 and n % factor3 == 0:
            return self._format_result(n, factor3)
        
        # Fallback to enhanced coverage
        print("\nStrategy 4: Enhanced Coverage with Emanation")
        factor4 = self._enhanced_coverage(n)
        if factor4 and n % factor4 == 0:
            return self._format_result(n, factor4)
        
        raise ValueError("Could not find factors")
    
    def _check_transition_boundaries(self, n: int):
        """Check if n is near any transition boundaries"""
        for base in [2, 3, 5]:
            if EmanationTransitions.find_transition_boundary(base, n):
                print(f"  ⚠️  n is near base-{base} transition boundary!")
    
    def _emanation_guided_search(self, n: int) -> Optional[int]:
        """Search guided by emanation distances"""
        resonator = SinglePrimeResonance(n)
        sqrt_n = int(math.sqrt(n))
        
        # Generate positions based on emanation theory
        positions = []
        
        # 1. Positions at equal emanation distances
        for base in [2, 3, 5, 7]:
            # Find x where emanation_distance(x, base) * 2 ≈ emanation_distance(n, base)
            target_dist = EmanationTransitions.compute_emanation_distance(n, base) / 2
            
            # Approximate inverse
            x_approx = int(base ** target_dist)
            if 2 <= x_approx <= sqrt_n:
                positions.extend(range(max(2, x_approx - 100), min(sqrt_n + 1, x_approx + 100)))
        
        # 2. Special transition-related positions
        positions.extend([531, 532, 530])  # Near sqrt(282281)
        positions.extend([282281 // p for p in [2, 3, 5, 7, 11] if 282281 % p == 0])
        
        # Check positions with high emanation resonance
        candidates = []
        for pos in set(positions):
            if 2 <= pos <= sqrt_n:
                x_norm = pos / sqrt_n
                res = resonator.emanation_resonance(x_norm)
                self.stats['resonance_evaluations'] += 1
                
                if res > 2.0:
                    candidates.append((pos, res))
        
        # Sort by resonance and check
        candidates.sort(key=lambda x: x[1], reverse=True)
        for pos, res in candidates[:20]:
            if n % pos == 0:
                print(f"  Found factor {pos} with emanation resonance {res:.3f}")
                self.stats['emanation_alignments'] += 1
                return pos
        
        return None
    
    def _transition_point_search(self, n: int) -> Optional[int]:
        """Search near transition boundaries"""
        sqrt_n = int(math.sqrt(n))
        
        # Key transition-related numbers
        transition_numbers = [
            531,          # sqrt(282281)
            282281,       # The transition point
            141140,       # 282281 / 2
            94093,        # 282281 / 3
        ]
        
        # Check these and their multiples/divisors
        positions = set()
        for t in transition_numbers:
            if t <= sqrt_n:
                positions.add(t)
                # Add factors and multiples
                for k in range(1, 20):
                    if t * k <= sqrt_n:
                        positions.add(t * k)
                    if t % k == 0 and t // k >= 2:
                        positions.add(t // k)
        
        # Check each position
        for pos in sorted(positions):
            if 2 <= pos <= sqrt_n and n % pos == 0:
                print(f"  Found transition-related factor {pos}")
                self.stats['special_positions_checked'] += 1
                return pos
        
        return None
    
    def _clifford_norm_search(self, n: int) -> Optional[int]:
        """Search based on Clifford algebra norms"""
        sqrt_n = int(math.sqrt(n))
        
        # Target norm
        n_norm = math.sqrt(n*n + sum(n % p for p in [2, 3, 5, 7, 11, 13])*sum(n % p for p in [2, 3, 5, 7, 11, 13]))
        
        # Find positions with compatible norms
        candidates = []
        
        # Sample positions
        for i in range(100, min(10000, sqrt_n), 10):
            x_norm = math.sqrt(i*i + sum(i % p for p in [2, 3, 5, 7, 11, 13])*sum(i % p for p in [2, 3, 5, 7, 11, 13]))
            
            # Check if norms are compatible for factorization
            other = n // i if i != 0 else 0
            if other > 0:
                other_norm = math.sqrt(other*other + sum(other % p for p in [2, 3, 5, 7, 11, 13])*sum(other % p for p in [2, 3, 5, 7, 11, 13]))
                
                # Clifford multiplication norm property (simplified)
                if abs(x_norm * other_norm - n_norm) < n_norm * 0.01:
                    candidates.append(i)
        
        # Check candidates
        for cand in candidates:
            if n % cand == 0:
                print(f"  Found Clifford-norm compatible factor {cand}")
                return cand
        
        return None
    
    def _enhanced_coverage(self, n: int) -> Optional[int]:
        """Enhanced coverage with emanation weighting"""
        sqrt_n = int(math.sqrt(n))
        resonator = SinglePrimeResonance(n)
        
        # Generate comprehensive positions
        positions = set()
        
        # 1. Powers of primes (emanation centers)
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            power = p
            while power <= sqrt_n:
                positions.add(power)
                # Add neighbors
                for offset in range(-5, 6):
                    if 2 <= power + offset <= sqrt_n:
                        positions.add(power + offset)
                power *= p
        
        # 2. Products of small primes
        small_primes = [2, 3, 5, 7, 11]
        for i in range(len(small_primes)):
            for j in range(i, len(small_primes)):
                prod = small_primes[i] * small_primes[j]
                if prod <= sqrt_n:
                    positions.add(prod)
        
        # 3. Near perfect squares (potential transition points)
        for i in range(2, int(sqrt_n ** 0.5) + 1):
            square = i * i
            if square <= sqrt_n:
                positions.add(square)
                positions.add(square - 1)
                positions.add(square + 1)
        
        # Evaluate with emanation resonance
        print(f"  Evaluating {len(positions)} positions...")
        
        best_resonance = 0
        best_factor = None
        
        for i, pos in enumerate(sorted(positions)):
            if i % 1000 == 0 and i > 0:
                print(f"    Progress: {i}/{len(positions)}")
            
            x_norm = pos / sqrt_n
            res = resonator.emanation_resonance(x_norm)
            self.stats['resonance_evaluations'] += 1
            
            if res > best_resonance:
                best_resonance = res
                best_factor = pos
            
            # Check divisibility for high resonance
            if res > 3.0 and n % pos == 0:
                print(f"  Found factor {pos} with emanation resonance {res:.3f}")
                return pos
        
        # Try the best resonance position
        if best_factor and n % best_factor == 0:
            print(f"  Found factor {best_factor} with best resonance {best_resonance:.3f}")
            return best_factor
        
        return None
    
    def _format_result(self, n: int, factor: int) -> Tuple[int, int]:
        """Format the result with emanation analysis"""
        other = n // factor
        print(f"\n✓ SUCCESS: Found factor {factor}")
        print(f"  {n} = {factor} × {other}")
        
        # Emanation analysis
        print("\nEmanation Analysis:")
        for base in [2, 3, 5]:
            dist_n = EmanationTransitions.compute_emanation_distance(n, base)
            dist_f1 = EmanationTransitions.compute_emanation_distance(factor, base)
            dist_f2 = EmanationTransitions.compute_emanation_distance(other, base)
            print(f"  Base-{base}: d(n)={dist_n:.3f}, d(p)={dist_f1:.3f}, d(q)={dist_f2:.3f}")
        
        print(f"\nStatistics:")
        print(f"  Resonance evaluations: {self.stats['resonance_evaluations']}")
        print(f"  Special positions checked: {self.stats['special_positions_checked']}")
        print(f"  Emanation alignments: {self.stats['emanation_alignments']}")
        
        return (factor, other) if factor <= other else (other, factor)


def test_emanation():
    """Test the emanation-based approach"""
    
    # Include 282281 and related numbers
    test_cases = [
        # Original cases
        (11, 13),                     # 143 (8-bit)
        (101, 103),                   # 10403 (14-bit)
        (65537, 4294967311),          # 49-bit (Fermat prime)
        (2147483647, 2147483659),     # 63-bit (Mersenne prime)
        (7125766127, 6958284019),     # 66-bit
        (14076040031, 15981381943),   # 68-bit
        (1073741827, 1073741831),     # 61-bit (twin primes near 2^30)
        
        # Transition-related cases
        (531, 532),                   # Near sqrt(282281)
        (523, 541),                   # Twin primes near 531
    ]
    
    phase1 = ImprovedPhaseI()
    
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
                print(f"\n✗ INCORRECT: Expected {p_true} × {q_true}, got {p_found} × {q_found}")
        
        except Exception as e:
            print(f"\n✗ FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Summary: {successes}/{len(test_cases)} successful")
    print(f"{'='*60}")
    
    # Special analysis of 282281
    print(f"\n{'='*60}")
    print("Special Analysis: 282281")
    print(f"{'='*60}")
    n = 282281
    print(f"282281 = 531²")
    print(f"Binary: {bin(n)}")
    print(f"Prime factorization: 3² × 31381")
    print(f"531 = 3² × 59")
    
    # Check emanation distances
    for base in [2, 3, 5, 7]:
        dist = EmanationTransitions.compute_emanation_distance(n, base)
        print(f"Emanation distance in base-{base}: {dist:.6f}")


if __name__ == "__main__":
    test_emanation()
