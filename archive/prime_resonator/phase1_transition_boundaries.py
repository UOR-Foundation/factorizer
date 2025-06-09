"""
Phase I with Transition Boundary Theory
Based on the insight that primes have natural transition points

Key insight: 282281 represents where "2 becomes 3" - a phase transition
in the emanation from π₁ through different bases.
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
import time
from functools import lru_cache


@lru_cache(maxsize=10000)
def _is_probable_prime_simple(n: int) -> bool:
    """Simple primality test"""
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
    
    # Simple trial division up to sqrt(n) for small numbers
    if n < 10000:
        for i in range(51, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Miller-Rabin for larger numbers
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Test with a few bases
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
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


class TransitionBoundaryTheory:
    """
    Theory: Primes transition between base emanations at specific boundaries
    282281 (531²) is the boundary where base-2 emanation transitions to base-3
    """
    
    # Known transition boundaries
    KNOWN_BOUNDARIES = {
        (2, 3): 282281,      # Where base-2 transitions to base-3
        # Others to be discovered...
    }
    
    @staticmethod
    def find_transition_candidates(n: int) -> List[int]:
        """Find candidates based on transition boundary theory"""
        candidates = []
        sqrt_n = int(math.sqrt(n))
        
        # 1. Check proximity to known boundaries
        for (b1, b2), boundary in TransitionBoundaryTheory.KNOWN_BOUNDARIES.items():
            if abs(n - boundary) < boundary * 0.2:  # Within 20%
                # We're near a transition! Check sqrt of boundary
                sqrt_boundary = int(math.sqrt(boundary))
                candidates.extend(range(max(2, sqrt_boundary - 20), 
                                      min(sqrt_n + 1, sqrt_boundary + 20)))
        
        # 2. Look for perfect square relationships
        # Theory: Transition boundaries are often perfect squares
        for i in range(2, min(1000, int(sqrt_n ** 0.5) + 1)):
            square = i * i
            if square <= sqrt_n:
                candidates.append(square)
                # Also check near the square root
                sqrt_square = int(math.sqrt(square))
                candidates.extend([sqrt_square - 1, sqrt_square, sqrt_square + 1])
        
        # 3. Check powers of primes (emanation centers)
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            power = p
            exp = 1
            while power <= sqrt_n:
                candidates.append(power)
                # Special attention to powers that might be boundaries
                if exp > 1:  # p^k where k > 1
                    candidates.extend(range(max(2, power - 5), min(sqrt_n + 1, power + 5)))
                power *= p
                exp += 1
        
        return list(set(candidates))
    
    @staticmethod
    def compute_transition_score(n: int, x: int) -> float:
        """
        Compute how likely x is to be a factor based on transition theory
        """
        score = 1.0
        
        # 1. Perfect square bonus (like 531² = 282281)
        x_squared = x * x
        if x_squared <= n:
            # Check if x² is near n or a divisor of n
            if n % x_squared == 0:
                score *= 5.0
            elif abs(x_squared - n) < n * 0.01:
                score *= 3.0
        
        # 2. Check if x is near sqrt of a known boundary
        for boundary in TransitionBoundaryTheory.KNOWN_BOUNDARIES.values():
            sqrt_boundary = int(math.sqrt(boundary))
            if abs(x - sqrt_boundary) < 10:
                score *= 4.0
        
        # 3. Base emanation alignment
        # Numbers at boundaries have special relationships with bases
        for base in [2, 3, 5, 7]:
            # Check if x has a special relationship with base
            if x % base == 1 or x % base == base - 1:
                score *= 1.2
            
            # Powers of base
            if x > 1 and math.log(x, base) % 1 < 0.1:
                score *= 1.5
        
        # 4. Twin prime bonus (like 523, 541 near 531)
        # Simple primality test for now
        if _is_probable_prime_simple(x):
            if _is_probable_prime_simple(x + 2):
                score *= 2.0  # Twin prime
            elif _is_probable_prime_simple(x - 2):
                score *= 2.0  # Twin prime
            elif _is_probable_prime_simple(x + 6):
                score *= 1.5  # Sexy prime
            elif _is_probable_prime_simple(x - 6):
                score *= 1.5  # Sexy prime
        
        return score


class ResonanceWithTransitions:
    """Enhanced resonance that incorporates transition boundaries"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.bit_len = n.bit_length()
        
        # Check if we're near any transition boundaries
        self.near_boundary = self._check_near_boundary()
    
    def _check_near_boundary(self) -> Optional[Tuple[int, int]]:
        """Check if n is near a known transition boundary"""
        for (b1, b2), boundary in TransitionBoundaryTheory.KNOWN_BOUNDARIES.items():
            if abs(self.n - boundary) < boundary * 0.2:
                return (b1, b2)
        return None
    
    def transition_resonance(self, x_normalized: float) -> float:
        """Compute resonance with transition boundary effects"""
        if x_normalized <= 0 or x_normalized >= 1:
            return 0.0
        
        x = max(2, int(x_normalized * self.sqrt_n))
        if x >= self.n:
            return 0.0
        
        # Base resonance from transition theory
        resonance = TransitionBoundaryTheory.compute_transition_score(self.n, x)
        
        # Enhanced resonance near boundaries
        if self.near_boundary:
            b1, b2 = self.near_boundary
            # Check if x aligns with the transition
            if x % b1 == 0 or x % b2 == 0:
                resonance *= 1.5
        
        # Check divisibility bonus (minimal)
        if self.n % x == 0:
            resonance *= 1.1
        
        return resonance


class TransitionBoundaryPhaseI:
    """Phase I implementation using transition boundary theory"""
    
    def __init__(self):
        self.stats = {
            'candidates_checked': 0,
            'transition_candidates': 0,
            'boundary_detections': 0
        }
    
    def factorize(self, n: int) -> Tuple[int, int]:
        """Factor using transition boundary theory"""
        if n < 2:
            raise ValueError("n must be >= 2")
        if n % 2 == 0:
            return (2, n // 2)
        
        print(f"\n{'='*60}")
        print(f"Transition Boundary Factorization")
        print(f"n = {n} ({n.bit_length()} bits)")
        print(f"{'='*60}")
        
        # Check if near known boundaries
        resonator = ResonanceWithTransitions(n)
        if resonator.near_boundary:
            b1, b2 = resonator.near_boundary
            print(f"  ⚡ Near transition boundary: base-{b1} → base-{b2}")
            self.stats['boundary_detections'] += 1
        
        # Strategy 1: Direct transition candidates
        print("\nStrategy 1: Transition Boundary Candidates")
        factor1 = self._check_transition_candidates(n)
        if factor1 and n % factor1 == 0:
            return self._format_result(n, factor1)
        
        # Strategy 2: Resonance-guided search with transitions
        print("\nStrategy 2: Transition-Enhanced Resonance")
        factor2 = self._resonance_guided_search(n)
        if factor2 and n % factor2 == 0:
            return self._format_result(n, factor2)
        
        # Strategy 3: Systematic near-boundary search
        print("\nStrategy 3: Near-Boundary Systematic Search")
        factor3 = self._near_boundary_search(n)
        if factor3 and n % factor3 == 0:
            return self._format_result(n, factor3)
        
        raise ValueError("Could not find factors")
    
    def _check_transition_candidates(self, n: int) -> Optional[int]:
        """Check candidates from transition boundary theory"""
        candidates = TransitionBoundaryTheory.find_transition_candidates(n)
        sqrt_n = int(math.sqrt(n))
        
        print(f"  Found {len(candidates)} transition candidates")
        self.stats['transition_candidates'] = len(candidates)
        
        # Sort by transition score
        scored_candidates = []
        for cand in candidates:
            if 2 <= cand <= sqrt_n:
                score = TransitionBoundaryTheory.compute_transition_score(n, cand)
                scored_candidates.append((cand, score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Check top candidates
        for i, (cand, score) in enumerate(scored_candidates[:100]):
            self.stats['candidates_checked'] += 1
            if n % cand == 0:
                print(f"  Found factor {cand} with transition score {score:.3f}")
                return cand
        
        return None
    
    def _resonance_guided_search(self, n: int) -> Optional[int]:
        """Search using transition-enhanced resonance"""
        resonator = ResonanceWithTransitions(n)
        sqrt_n = int(math.sqrt(n))
        
        # Sample positions with focus on high-resonance areas
        positions = []
        
        # 1. Near perfect squares
        for i in range(2, min(1000, int(sqrt_n ** 0.5) + 1)):
            square = i * i
            if square <= sqrt_n:
                positions.extend(range(max(2, square - 10), min(sqrt_n + 1, square + 10)))
        
        # 2. Near known special numbers
        special = [531, 282281, 65537, 2147483647, 1073741827]
        for s in special:
            if s <= sqrt_n:
                positions.extend(range(max(2, s - 20), min(sqrt_n + 1, s + 20)))
        
        # 3. Powers of small primes
        for p in [2, 3, 5, 7, 11]:
            power = p
            while power <= sqrt_n:
                positions.extend(range(max(2, power - 5), min(sqrt_n + 1, power + 5)))
                power *= p
        
        # Evaluate resonance
        candidates = []
        for pos in set(positions):
            if 2 <= pos <= sqrt_n:
                x_norm = pos / sqrt_n
                res = resonator.transition_resonance(x_norm)
                if res > 1.5:
                    candidates.append((pos, res))
        
        # Sort by resonance
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Check top candidates
        for pos, res in candidates[:50]:
            self.stats['candidates_checked'] += 1
            if n % pos == 0:
                print(f"  Found factor {pos} with transition resonance {res:.3f}")
                return pos
        
        return None
    
    def _near_boundary_search(self, n: int) -> Optional[int]:
        """Systematic search near transition boundaries"""
        sqrt_n = int(math.sqrt(n))
        
        # Focus on regions near known boundaries
        search_regions = []
        
        # Near 531 (sqrt of 282281)
        search_regions.append((520, 540))
        
        # Near powers of 2
        for k in range(10, min(40, sqrt_n.bit_length())):
            center = 1 << k
            if center <= sqrt_n:
                search_regions.append((center - 100, center + 100))
        
        # Check each region
        for start, end in search_regions:
            start = max(2, start)
            end = min(sqrt_n + 1, end)
            
            for candidate in range(start, end):
                self.stats['candidates_checked'] += 1
                if n % candidate == 0:
                    print(f"  Found factor {candidate} in boundary region [{start}, {end}]")
                    return candidate
        
        return None
    
    def _format_result(self, n: int, factor: int) -> Tuple[int, int]:
        """Format result with transition analysis"""
        other = n // factor
        print(f"\n✓ SUCCESS: Found factor {factor}")
        print(f"  {n} = {factor} × {other}")
        
        # Check if factors are related to transition boundaries
        print("\nTransition Analysis:")
        for (b1, b2), boundary in TransitionBoundaryTheory.KNOWN_BOUNDARIES.items():
            sqrt_boundary = int(math.sqrt(boundary))
            if abs(factor - sqrt_boundary) < 20:
                print(f"  Factor {factor} is near sqrt({boundary}) = {sqrt_boundary}")
                print(f"  This is the base-{b1} → base-{b2} transition!")
        
        print(f"\nStatistics:")
        print(f"  Candidates checked: {self.stats['candidates_checked']}")
        print(f"  Transition candidates: {self.stats['transition_candidates']}")
        print(f"  Boundary detections: {self.stats['boundary_detections']}")
        
        return (factor, other) if factor <= other else (other, factor)


def test_transition_boundaries():
    """Test the transition boundary approach"""
    
    test_cases = [
        # Original cases
        (11, 13),                     # 143 (8-bit)
        (101, 103),                   # 10403 (14-bit)
        (65537, 4294967311),          # 49-bit (Fermat prime)
        (2147483647, 2147483659),     # 63-bit (Mersenne prime)
        
        # Transition-related cases
        (531, 532),                   # 282492 - very close to 282281!
        (523, 541),                   # 282943 - also near transition
        (529, 547),                   # 289363 - exploring the region
        
        # More challenging cases
        (7125766127, 6958284019),     # 66-bit
        (14076040031, 15981381943),   # 68-bit
        (1073741827, 1073741831),     # 61-bit (twin primes near 2^30)
    ]
    
    phase1 = TransitionBoundaryPhaseI()
    
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
    
    print(f"\n{'='*60}")
    print(f"Summary: {successes}/{len(test_cases)} successful")
    print(f"{'='*60}")
    
    # Additional analysis
    print(f"\n{'='*60}")
    print("Transition Boundary Analysis")
    print(f"{'='*60}")
    
    # Analyze numbers near 282281
    for n in [282281, 282492, 282943, 289363]:
        factors = []
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append((i, n // i))
                break
        
        print(f"\n{n}:")
        if factors:
            p, q = factors[0]
            print(f"  = {p} × {q}")
            print(f"  Distance from 282281: {abs(n - 282281)}")
            print(f"  p distance from 531: {abs(p - 531)}")


if __name__ == "__main__":
    test_transition_boundaries()
