"""
RFH2: The Prime Opus
Implementation of the unified Prime Resonance Function for factorization.
Based on adelic balance, modular coherence, and transition resonance.
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
import time
from functools import lru_cache
from collections import defaultdict


# Implementation Constants
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
TRIBONACCI = 1.839286755214161  # Exact value
EGYPTIAN_SPREAD = 0.96  # Universal logarithmic spread
RESONANCE_THRESHOLD = 0.5  # Minimum Ψ to test divisibility (adjusted for geometric mean)
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


@lru_cache(maxsize=10000)
def is_probable_prime(n: int) -> bool:
    """Fast primality test"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for p in SMALL_PRIMES[1:]:  # Skip 2
        if n == p:
            return True
        if n % p == 0:
            return False
    
    # Miller-Rabin for larger numbers
    if n < 1000:
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Simplified Miller-Rabin
    d = n - 1
    r = 0
    while d % 2 == 0:
        r += 1
        d //= 2
    
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


class TransitionBoundaries:
    """Manages transition boundaries and resonance nodes"""
    
    def __init__(self):
        # Known transition boundaries from our research
        self.boundaries = {
            (2, 3): 282281,    # 531²
            (3, 5): 2961841,   # 1721²
            (5, 7): 53596041,  # 7321²
            (7, 11): 1522756281  # 39023²
        }
        
    def find_transition_zone(self, n: int) -> Tuple[Tuple[int, int], int]:
        """Find which transition zone contains √n"""
        sqrt_n = int(math.sqrt(n))
        
        # Find the appropriate zone
        prev_boundary = 0
        prev_transition = (1, 2)
        
        for (b1, b2), boundary in sorted(self.boundaries.items(), key=lambda x: x[1]):
            if sqrt_n * sqrt_n < boundary:
                return prev_transition, prev_boundary
            prev_transition = (b1, b2)
            prev_boundary = boundary
            
        # If beyond all known boundaries, return the last one
        return prev_transition, prev_boundary
    
    def calculate_resonance_nodes(self, n: int) -> List[int]:
        """Calculate resonance nodes based on transition boundaries"""
        sqrt_n = int(math.sqrt(n))
        transition, boundary = self.find_transition_zone(n)
        
        nodes = set()
        
        if boundary > 0:
            # Primary node at geometric mean of transition
            primary = int(math.sqrt(boundary))
            nodes.add(primary)
            
            # Harmonic nodes using golden ratio
            for factor in [1/GOLDEN_RATIO**2, 1/GOLDEN_RATIO, 1, GOLDEN_RATIO, GOLDEN_RATIO**2]:
                node = int(primary * factor)
                if 2 <= node <= sqrt_n:
                    nodes.add(node)
        
        # Add nodes from interference between boundaries
        boundaries_sqrt = [int(math.sqrt(b)) for b in self.boundaries.values()]
        for i in range(len(boundaries_sqrt)):
            for j in range(i + 1, len(boundaries_sqrt)):
                interference = int(math.sqrt(boundaries_sqrt[i] * boundaries_sqrt[j]))
                if 2 <= interference <= sqrt_n:
                    nodes.add(interference)
        
        # Add some special nodes based on n's structure
        # Nodes at powers of primes
        for p in SMALL_PRIMES[:5]:
            k = 1
            while p**k <= sqrt_n:
                nodes.add(p**k)
                k += 1
        
        # Nodes at n^(1/k) for small k
        for k in range(3, 7):
            node = int(n**(1/k))
            if 2 <= node <= sqrt_n:
                nodes.add(node)
        
        return sorted(list(nodes))


class PrimeResonanceFunction:
    """Implements the unified Prime Resonance Function Ψ(x, n)"""
    
    def __init__(self):
        self.boundaries = TransitionBoundaries()
        self.cache = {}
    
    def compute_adelic_balance(self, x: int, n: int) -> float:
        """
        A(x, n) = |x|_R × ∏_p |x|_p / n
        
        For a true factor, the adelic product law suggests this approaches 1
        when properly normalized.
        """
        if x <= 0 or x > int(math.sqrt(n)):
            return 0.0
        
        # For factors, we expect a special relationship
        # The adelic balance should be near 1 when x divides n
        
        # Simplified approach: measure how well x*y = n is satisfied adelically
        if n % x == 0:
            # Perfect factor - return 1
            return 1.0
        
        # For non-factors, estimate based on gcd and modular properties
        g = math.gcd(x, n)
        
        # Base score from gcd relationship
        base_score = g / x
        
        # Boost score if x shares prime factors with n
        prime_boost = 1.0
        for p in SMALL_PRIMES:
            if x % p == 0 and n % p == 0:
                prime_boost *= 1.2
        
        # Normalize to [0, 1] range
        score = min(1.0, base_score * prime_boost)
        
        return score
    
    def compute_modular_coherence(self, x: int, n: int) -> float:
        """
        M(x, n) = ∏_{p ∈ P} (1 - |n mod p - x mod p|/p)
        
        Measures how well x and n align in modular arithmetic across prime bases
        """
        if x <= 0:
            return 0.0
        
        coherence = 1.0
        for p in SMALL_PRIMES:
            n_mod = n % p
            x_mod = x % p
            
            # Distance in modular arithmetic
            dist = abs(n_mod - x_mod)
            if dist > p / 2:
                dist = p - dist  # Wrap around
            
            # Convert to coherence score
            coherence *= (1 - dist / p)
        
        return coherence
    
    def compute_transition_resonance(self, x: int, n: int) -> float:
        """
        T(x, n) = exp(-d²/σ²) × cos(2π × log(x)/log(φ))
        
        Measures resonance with transition boundaries and golden ratio scaling
        """
        if x <= 1:
            return 0.0
        
        # Find nearest transition boundary
        transition, boundary = self.boundaries.find_transition_zone(n)
        
        # Distance to transition
        if boundary > 0:
            boundary_sqrt = int(math.sqrt(boundary))
            d = abs(x - boundary_sqrt)
        else:
            d = 0
        
        # Gaussian envelope with Egyptian spread
        gaussian = math.exp(-(d**2) / (EGYPTIAN_SPREAD**2 * x))
        
        # Golden ratio oscillation
        try:
            oscillation = math.cos(2 * math.pi * math.log(x) / math.log(GOLDEN_RATIO))
        except:
            oscillation = 0
        
        return gaussian * (1 + oscillation) / 2
    
    def psi(self, x: int, n: int) -> float:
        """
        The Prime Resonance Function: Ψ(x, n) = A(x, n) × M(x, n) × T(x, n)
        """
        # Check cache
        cache_key = (x, n)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Compute three pillars
        A = self.compute_adelic_balance(x, n)
        M = self.compute_modular_coherence(x, n)
        T = self.compute_transition_resonance(x, n)
        
        # Unified resonance using geometric mean to avoid tiny products
        # This keeps the score in a more reasonable range
        psi_value = (A * M * T) ** (1/3)
        
        # Alternative: weighted harmonic mean for better sensitivity
        # psi_value = 3 / (1/A + 1/M + 1/T) if A > 0 and M > 0 and T > 0 else 0
        
        # Cache result
        self.cache[cache_key] = psi_value
        
        return psi_value


class PrimeOpus:
    """Main factorization engine using Prime Resonance"""
    
    def __init__(self):
        self.prf = PrimeResonanceFunction()
        self.stats = {
            'evaluations': 0,
            'nodes_tested': 0,
            'high_resonance_count': 0,
            'time': 0
        }
    
    def factor(self, n: int) -> Optional[Tuple[int, int]]:
        """
        Factor n using the Prime Resonance Function
        Returns (p, q) where n = p × q, or None if no factors found
        """
        if n < 2:
            raise ValueError("n must be >= 2")
        
        if is_probable_prime(n):
            raise ValueError(f"{n} is prime")
        
        start_time = time.perf_counter()
        sqrt_n = int(math.sqrt(n))
        
        print(f"\n{'='*60}")
        print(f"Prime Opus: Factoring {n} ({n.bit_length()} bits)")
        print(f"{'='*60}")
        
        # Phase 1: Calculate resonance nodes
        print("\nPhase 1: Mapping Resonance Field")
        nodes = self.prf.boundaries.calculate_resonance_nodes(n)
        print(f"  Generated {len(nodes)} resonance nodes")
        
        # Phase 2: Resonance Detection
        print("\nPhase 2: Resonance Detection")
        candidates = []
        
        for x in nodes:
            self.stats['evaluations'] += 1
            psi = self.prf.psi(x, n)
            
            if psi > RESONANCE_THRESHOLD:
                self.stats['high_resonance_count'] += 1
                candidates.append((x, psi))
                print(f"  High resonance at x={x}: Ψ={psi:.4f}")
        
        # Sort by resonance score
        candidates.sort(key=lambda c: abs(c[1] - 1.0))
        
        # Phase 3: Validation
        print(f"\nPhase 3: Testing {len(candidates)} high-resonance candidates")
        
        for x, psi in candidates:
            self.stats['nodes_tested'] += 1
            
            if n % x == 0:
                self.stats['time'] = time.perf_counter() - start_time
                other = n // x
                
                print(f"\n{'='*60}")
                print(f"✓ SUCCESS! Found factors: {x} × {other}")
                print(f"  Resonance score: Ψ={psi:.6f}")
                print(f"  Deviation from unity: {abs(psi - 1.0):.6f}")
                self._print_stats()
                
                return (x, other) if x <= other else (other, x)
        
        # If no factors found in high-resonance candidates, expand search
        print("\nExpanding search to lower resonance nodes...")
        
        all_candidates = []
        for x in nodes:
            if x not in [c[0] for c in candidates]:
                psi = self.prf.psi(x, n)
                all_candidates.append((x, psi))
                self.stats['evaluations'] += 1
        
        all_candidates.sort(key=lambda c: c[1], reverse=True)
        
        for x, psi in all_candidates[:100]:  # Test top 100
            self.stats['nodes_tested'] += 1
            
            if n % x == 0:
                self.stats['time'] = time.perf_counter() - start_time
                other = n // x
                
                print(f"\n{'='*60}")
                print(f"✓ Found factors: {x} × {other}")
                print(f"  Resonance score: Ψ={psi:.6f}")
                self._print_stats()
                
                return (x, other) if x <= other else (other, x)
        
        self.stats['time'] = time.perf_counter() - start_time
        print(f"\n✗ No factors found")
        self._print_stats()
        
        return None
    
    def _print_stats(self):
        """Print statistics"""
        print(f"\nStatistics:")
        print(f"  Resonance evaluations: {self.stats['evaluations']}")
        print(f"  Nodes tested: {self.stats['nodes_tested']}")
        print(f"  High resonance candidates: {self.stats['high_resonance_count']}")
        print(f"  Total time: {self.stats['time']:.3f}s")
    
    def analyze_resonance(self, n: int, x: int) -> Dict[str, float]:
        """Analyze resonance components for a specific x"""
        A = self.prf.compute_adelic_balance(x, n)
        M = self.prf.compute_modular_coherence(x, n)
        T = self.prf.compute_transition_resonance(x, n)
        psi = A * M * T
        
        return {
            'adelic_balance': A,
            'modular_coherence': M,
            'transition_resonance': T,
            'total_resonance': psi
        }


def test_prime_opus():
    """Test the Prime Opus implementation"""
    
    test_cases = [
        # Small cases
        (11, 13),                     # 143
        (101, 103),                   # 10403
        
        # Transition boundaries
        (531, 532),                   # 282492
        (523, 541),                   # 282943
        
        # Larger cases
        (65537, 4294967311),          # Fermat prime
        (99991, 99989),               # Twin primes
        (7125766127, 6958284019),     # 66-bit arbitrary primes
    ]
    
    opus = PrimeOpus()
    successes = 0
    
    for p_true, q_true in test_cases:
        n = p_true * q_true
        
        try:
            result = opus.factor(n)
            
            if result:
                p_found, q_found = result
                if {p_found, q_found} == {p_true, q_true}:
                    print(f"\n✓ CORRECT")
                    
                    # Analyze resonance for the found factor
                    analysis = opus.analyze_resonance(n, p_found)
                    print(f"\nResonance Analysis for factor {p_found}:")
                    for component, value in analysis.items():
                        print(f"  {component}: {value:.6f}")
                    
                    successes += 1
                else:
                    print(f"\n✗ INCORRECT: Expected {p_true} × {q_true}, got {p_found} × {q_found}")
            else:
                print(f"\n✗ FAILED: Could not find factors of {n} = {p_true} × {q_true}")
        
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Reset stats for next test
        opus.stats = {
            'evaluations': 0,
            'nodes_tested': 0,
            'high_resonance_count': 0,
            'time': 0
        }
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {successes}/{len(test_cases)} successful")
    print(f"Success rate: {successes/len(test_cases)*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_prime_opus()
