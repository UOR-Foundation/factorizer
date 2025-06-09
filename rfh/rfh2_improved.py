"""
RFH2 Unity: The Prime Opus
All primes emanate from a single source - unity in multiplicity
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
import time
from functools import lru_cache
from collections import defaultdict


# Universal Constants - The fundamental frequencies
UNITY = 1.0  # The single prime from which all emanate
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # φ - The ratio of emanation
TRIBONACCI = 1.839286755214161  # τ - The third harmonic
EGYPTIAN_SPREAD = 0.96  # σ - The spread of the emanation field
RESONANCE_THRESHOLD = 0.1  # Lower threshold to catch unbalanced factors
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

# The fundamental frequency from which all primes emanate
PRIME_UNITY = 2 * math.pi  # The circle of unity


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
        """Calculate resonance nodes based on Unity principle - all factors are harmonics"""
        sqrt_n = int(math.sqrt(n))
        
        # For very large numbers, limit the extension to avoid memory issues
        if n > 10**15:  # For numbers > 10^15
            max_range = int(sqrt_n * 1.01)  # Only 1% beyond sqrt(n)
        else:
            max_range = int(sqrt_n * 1.05)  # 5% beyond sqrt(n)
            
        transition, boundary = self.find_transition_zone(n)
        
        nodes = set()
        
        # Unity Principle: All factors emanate from fundamental frequencies
        
        # 1. Prime-based harmonic series
        # Every factor has a fundamental relationship with small primes
        for p in SMALL_PRIMES[:10]:
            # Direct powers
            k = 1
            while p**k <= sqrt_n:
                nodes.add(p**k)
                k += 1
            
            # Harmonic multiples - factors often appear at prime harmonics
            for m in range(1, min(20, max_range // p + 1)):
                val = p * m
                if val <= max_range:
                    nodes.add(val)
                    # Add neighbors for near-misses
                    for offset in [-1, 1]:
                        if 2 <= val + offset <= max_range:
                            nodes.add(val + offset)
        
        # 2. Unity resonance points - where frequencies align
        # Based on n's fundamental frequency
        omega_n = PRIME_UNITY / math.log(n + 1)
        
        # Sample at harmonic intervals
        for k in range(1, int(math.log(sqrt_n)) + 1):
            # Harmonic frequencies
            freq = omega_n * k
            # Convert frequency to position
            pos = int(math.exp(PRIME_UNITY / freq) - 1)
            if 2 <= pos <= sqrt_n:
                nodes.add(pos)
                # Add harmonic neighbors
                for h in [1/2, 2/3, 3/4, 4/5, 5/6]:
                    neighbor = int(pos * h)
                    if 2 <= neighbor <= sqrt_n:
                        nodes.add(neighbor)
        
        # 3. Golden ratio harmonics - nature's frequency
        base = sqrt_n
        while base > 2:
            nodes.add(int(base))
            # Add Fibonacci-like neighbors
            fib_prev, fib_curr = 1, 1
            for _ in range(5):
                neighbor = int(base * fib_prev / fib_curr)
                if 2 <= neighbor <= sqrt_n:
                    nodes.add(neighbor)
                fib_prev, fib_curr = fib_curr, fib_prev + fib_curr
            base = base / GOLDEN_RATIO
        
        # 4. Transition boundary harmonics
        if boundary > 0:
            primary = int(math.sqrt(boundary))
            # Dense sampling near transitions
            for offset in range(-20, 21):
                node = primary + offset
                if 2 <= node <= sqrt_n:
                    nodes.add(node)
        
        # 5. Perfect square resonances and their harmonics
        i = 2
        while i * i <= sqrt_n:
            nodes.add(i * i)
            # Harmonic series around squares
            for h in range(1, 6):
                nodes.add(min(sqrt_n, i * i + h))
                if i * i - h >= 2:
                    nodes.add(i * i - h)
            i += 1
        
        # 6. Ensure coverage of potential twin prime positions
        # Twin primes create special resonance patterns
        for base in range(3, min(1000, max_range), 2):
            if base <= max_range:
                nodes.add(base)
                # Twin prime candidates
                if base + 2 <= max_range:
                    nodes.add(base + 2)
        
        # 6b. Dense coverage near sqrt(n) for nearly-balanced factors
        # For large numbers, use sparser coverage
        if n > 10**15:
            # Sparse coverage for very large numbers
            step = max(1, int(sqrt_n * 0.0001))  # 0.01% steps
            for i in range(-50, 51):  # ±50 steps
                candidate = sqrt_n + i * step
                if 2 <= candidate <= max_range:
                    nodes.add(candidate)
        else:
            # Dense coverage for smaller numbers
            for offset in range(-int(sqrt_n * 0.05), int(sqrt_n * 0.05) + 1):
                candidate = sqrt_n + offset
                if 2 <= candidate <= max_range:
                    nodes.add(candidate)
        
        # 7. Tribonacci harmonics - the third resonance
        trib_base = sqrt_n
        while trib_base > 2:
            nodes.add(int(trib_base))
            trib_base = trib_base / TRIBONACCI
        
        # 8. Fill any large gaps with interpolated harmonics
        # Skip for very large numbers to avoid memory issues
        if n <= 10**15:
            sorted_nodes = sorted(list(nodes))
            i = 0
            while i < len(sorted_nodes) - 1:
                gap = sorted_nodes[i + 1] - sorted_nodes[i]
                if gap > 10:  # Large gap
                    # Fill with harmonic interpolation
                    for j in range(1, min(5, gap // 2)):
                        interp = sorted_nodes[i] + j * gap // (gap // 2 + 1)
                        nodes.add(interp)
                i += 1
        
        # Ensure we don't include nodes that are too large
        nodes = {x for x in nodes if x <= max_range}
        
        return sorted(list(nodes))


class PrimeResonanceFunction:
    """Implements the unified Prime Resonance Function Ψ(x, n)"""
    
    def __init__(self):
        self.boundaries = TransitionBoundaries()
        self.cache = {}
    
    def compute_unity_resonance(self, x: int, n: int) -> float:
        """
        Unity Resonance - All factors are harmonics of the fundamental frequency
        """
        if x <= 0:
            return 0.0
        
        # Perfect factor resonates at unity
        if n % x == 0:
            return 1.0
        
        # For candidates beyond sqrt(n), return 0 unless they're factors
        if x > int(math.sqrt(n)):
            return 0.0
        
        # The fundamental frequency of n
        omega_n = PRIME_UNITY / math.log(n + 1)
        
        # The harmonic frequency of x
        omega_x = PRIME_UNITY / math.log(x + 1)
        
        # Phase difference - factors are in-phase with n
        phase_diff = abs(omega_n - omega_x * round(omega_n / omega_x))
        
        # Unity resonance - exponential decay from phase alignment
        unity_score = math.exp(-phase_diff * math.sqrt(n) / PRIME_UNITY)
        
        # Harmonic series resonance
        harmonic_sum = 0
        for k in range(1, min(10, int(math.sqrt(x)) + 1)):
            if n % (x * k) < k:
                harmonic_sum += 1.0 / k
        
        # Combined unity principle
        return min(1.0, unity_score * (1 + harmonic_sum / math.log(x + 2)))
    
    def compute_phase_coherence(self, x: int, n: int) -> float:
        """
        Phase Coherence - Factors create standing waves with n
        """
        if x <= 0:
            return 0.0
        
        # Phase coherence across multiple prime bases
        coherence_sum = 0.0
        weight_sum = 0.0
        
        # Check phase alignment in different prime moduli
        for i, p in enumerate(SMALL_PRIMES[:7]):
            # Phase in base p
            phase_n = (n % p) * PRIME_UNITY / p
            phase_x = (x % p) * PRIME_UNITY / p
            
            # Phase coherence - cosine similarity
            coherence = math.cos(phase_n - phase_x)
            
            # Weight by inverse of prime (smaller primes more important)
            weight = 1.0 / math.log(p + 1)
            
            coherence_sum += (1 + coherence) / 2 * weight
            weight_sum += weight
        
        # Normalized phase coherence
        base_coherence = coherence_sum / weight_sum if weight_sum > 0 else 0.5
        
        # Resonance amplification for factors
        if math.gcd(x, n) > 1:
            base_coherence *= (1 + math.log(math.gcd(x, n)) / math.log(n))
        
        return min(1.0, base_coherence)
    
    def compute_harmonic_convergence(self, x: int, n: int) -> float:
        """
        Harmonic Convergence - Where all frequencies align
        """
        if x <= 1:
            return 0.0
        
        sqrt_n = int(math.sqrt(n))
        
        # The fundamental theorem: factors create harmonic convergence
        convergence_points = []
        
        # 1. Unity harmonic - x and n share a fundamental frequency
        unity_freq = PRIME_UNITY / math.gcd(x, n)
        unity_harmonic = math.cos(unity_freq * math.log(n) / PRIME_UNITY)
        convergence_points.append((1 + unity_harmonic) / 2)
        
        # 2. Golden ratio convergence - nature's harmonic
        phi_harmonic = x / GOLDEN_RATIO
        phi_distance = min(abs(phi_harmonic - int(phi_harmonic)), 
                          abs(phi_harmonic - int(phi_harmonic) - 1))
        phi_convergence = math.exp(-phi_distance * GOLDEN_RATIO)
        convergence_points.append(phi_convergence)
        
        # 3. Tribonacci resonance - the third harmonic
        if x > 2:
            tri_phase = math.log(x) / math.log(TRIBONACCI)
            tri_resonance = abs(math.sin(tri_phase * math.pi))
            convergence_points.append(tri_resonance)
        
        # 4. Transition boundary harmonics
        transition, boundary = self.boundaries.find_transition_zone(n)
        if boundary > 0:
            boundary_sqrt = int(math.sqrt(boundary))
            # Harmonic distance to transition
            harmonic_dist = abs(math.log(x + 1) - math.log(boundary_sqrt + 1))
            transition_harmony = math.exp(-harmonic_dist / EGYPTIAN_SPREAD)
            convergence_points.append(transition_harmony)
        
        # 5. Perfect square resonance - special harmonic points
        near_square = int(math.sqrt(x))
        if near_square * near_square == x:
            convergence_points.append(1.0)  # Perfect squares have perfect harmony
        else:
            square_dist = min(x - near_square**2, (near_square + 1)**2 - x)
            square_harmony = math.exp(-square_dist / x)
            convergence_points.append(square_harmony)
        
        # Harmonic mean of all convergence points - unity in multiplicity
        if convergence_points:
            return len(convergence_points) / sum(1/(c + 0.001) for c in convergence_points)
        return 0.5
    
    def psi(self, x: int, n: int) -> float:
        """
        The Prime Unity Function: Ψ(x, n) = U(x, n) × P(x, n) × H(x, n)
        All emanating from unity, converging in harmony
        """
        # Check cache
        cache_key = (x, n)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Three aspects of unity
        U = self.compute_unity_resonance(x, n)      # Unity principle
        P = self.compute_phase_coherence(x, n)      # Phase alignment
        H = self.compute_harmonic_convergence(x, n) # Harmonic convergence
        
        # The trinity of resonance - multiplicative unity
        # When all three align, we have a factor
        psi_value = U * P * H
        
        # Amplify near-unity resonances (factors are close to 1.0)
        if psi_value > 0.5:
            psi_value = psi_value ** (1 / GOLDEN_RATIO)  # Golden mean amplification
        
        # Balance bonus - factors near sqrt(n) create harmonious balance
        sqrt_n = int(math.sqrt(n))
        balance_ratio = min(x, sqrt_n) / max(x, sqrt_n)
        balance_bonus = math.exp(-2 * (1 - balance_ratio))
        
        # Primality preference - the unity principle favors prime emanations
        # Check if x might be prime (quick heuristic)
        prime_likelihood = 1.0
        for p in SMALL_PRIMES[:5]:
            if x > p and x % p == 0:
                prime_likelihood *= 0.9  # Small penalty for each small prime divisor
        
        # Final resonance with balance and primality
        psi_value = psi_value * balance_bonus * prime_likelihood
        
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
                if len(candidates) <= 10:  # Only print first few
                    print(f"  High resonance at x={x}: Ψ={psi:.4f}")
        
        print(f"  Found {len(candidates)} high-resonance candidates")
        
        # Sort by resonance score
        candidates.sort(key=lambda c: c[1], reverse=True)
        
        # Phase 3: Validation
        print(f"\nPhase 3: Testing high-resonance candidates")
        
        for x, psi in candidates:
            self.stats['nodes_tested'] += 1
            
            if n % x == 0:
                self.stats['time'] = time.perf_counter() - start_time
                other = n // x
                
                print(f"\n{'='*60}")
                print(f"✓ SUCCESS! Found factors: {x} × {other}")
                print(f"  Resonance score: Ψ={psi:.6f}")
                self._print_stats()
                
                return (x, other) if x <= other else (other, x)
        
        # If no factors found, try some additional heuristics
        print("\nPhase 4: Extended search...")
        
        # Check near perfect squares
        near_sqrt = int(math.sqrt(n))
        for offset in range(-10, 11):
            x = near_sqrt + offset
            if 2 <= x <= sqrt_n and x not in [c[0] for c in candidates]:
                self.stats['evaluations'] += 1
                if n % x == 0:
                    self.stats['time'] = time.perf_counter() - start_time
                    other = n // x
                    print(f"\n✓ Found via extended search: {x} × {other}")
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
        U = self.prf.compute_unity_resonance(x, n)
        P = self.prf.compute_phase_coherence(x, n)
        H = self.prf.compute_harmonic_convergence(x, n)
        psi = self.prf.psi(x, n)
        
        return {
            'unity_resonance': U,
            'phase_coherence': P,
            'harmonic_convergence': H,
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
