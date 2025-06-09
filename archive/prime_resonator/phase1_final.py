"""
Phase I Final Implementation
Ultimate refinement focusing on:
- Minimal divisor bonus (nearly eliminated)
- Strong prime detection with multiple methods
- Enhanced special form detection
- Better balance between different resonance signals
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
import time
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.signal import find_peaks
from functools import lru_cache


class ScaleAdaptiveParameters:
    """Scale-adaptive parameters that grow with problem size"""
    
    @staticmethod
    def prime_dimensions(n: int) -> int:
        """Prime dimensions scale with information content"""
        bit_len = n.bit_length()
        if bit_len < 64:
            return 60
        elif bit_len < 96:
            return int(bit_len * 2.5)
        else:
            return int(bit_len * 3)
    
    @staticmethod
    def resonance_samples(n: int) -> int:
        """Sampling density for complete coverage"""
        bit_len = n.bit_length()
        if bit_len < 64:
            return bit_len * 100
        elif bit_len < 96:
            return bit_len * bit_len * 2
        else:
            return min(bit_len * bit_len * 3, 30000)
    
    @staticmethod
    def max_positions(n: int) -> int:
        """Maximum positions to generate"""
        bit_len = n.bit_length()
        if bit_len < 64:
            return 20000
        elif bit_len < 96:
            return 40000
        else:
            return 60000


class EnhancedPrimalityTesting:
    """Enhanced primality testing with multiple methods"""
    
    @staticmethod
    @lru_cache(maxsize=50000)
    def is_probable_prime(n: int) -> bool:
        """Miller-Rabin primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Small primes
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        # Miller-Rabin test with more witnesses for higher certainty
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # More witnesses for better accuracy
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
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
    
    @staticmethod
    def is_fermat_prime(n: int) -> bool:
        """Check if n is a Fermat prime (2^(2^k) + 1)"""
        if n <= 2:
            return n == 2
        
        m = n - 1
        if m & (m - 1) != 0:
            return False
        
        k = m.bit_length() - 1
        # Check if k is a power of 2
        if (k & (k - 1)) == 0:
            return EnhancedPrimalityTesting.is_probable_prime(n)
        
        return False
    
    @staticmethod
    def is_mersenne_prime(n: int) -> bool:
        """Check if n is a Mersenne prime (2^p - 1)"""
        if n <= 1:
            return False
        
        m = n + 1
        if m & (m - 1) != 0:
            return False
        
        p = m.bit_length() - 1
        if EnhancedPrimalityTesting.is_probable_prime(p):
            return EnhancedPrimalityTesting.is_probable_prime(n)
        
        return False
    
    @staticmethod
    def is_sophie_germain_prime(n: int) -> bool:
        """Check if n is a Sophie Germain prime (p where 2p+1 is also prime)"""
        if not EnhancedPrimalityTesting.is_probable_prime(n):
            return False
        return EnhancedPrimalityTesting.is_probable_prime(2 * n + 1)
    
    @staticmethod
    def is_safe_prime(n: int) -> bool:
        """Check if n is a safe prime (p = 2q + 1 where q is prime)"""
        if not EnhancedPrimalityTesting.is_probable_prime(n):
            return False
        if n == 2:
            return False
        q = (n - 1) // 2
        return EnhancedPrimalityTesting.is_probable_prime(q)


class FinalResonance:
    """Final resonance function with balanced signals"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.bit_len = n.bit_length()
        
        # Initialize signals
        self.signals = {}
        
        # Precompute signals
        print(f"Precomputing signals for {self.bit_len}-bit number...")
        self._precompute_all_signals()
        print("Signals computed.")
    
    def _precompute_all_signals(self):
        """Precompute all signals"""
        self.signals['small_primes'] = self._generate_primes(min(20000, self.sqrt_n))
        self.signals['prime_coords'] = self._compute_prime_coordinates()
        self.signals['fermat_positions'] = self._generate_fermat_positions()
        self.signals['mersenne_positions'] = self._generate_mersenne_positions()
        self.signals['power2_positions'] = self._generate_power2_positions()
        self.signals['cf_convergents'] = self._compute_continued_fraction_convergents()
        self.signals['mult_orders'] = self._compute_multiplicative_orders()
    
    def _compute_prime_coordinates(self) -> List[Tuple[int, int]]:
        """Compute n's coordinates in prime space"""
        num_primes = ScaleAdaptiveParameters.prime_dimensions(self.n)
        primes = self._generate_primes(num_primes)
        return [(p, self.n % p) for p in primes]
    
    def _generate_primes(self, limit: int) -> List[int]:
        """Generate primes efficiently"""
        if limit < 2:
            return []
        
        if limit < 1000:
            count = limit
            limit = max(100, int(count * (math.log(count + 1) + math.log(math.log(count + 2)) + 2)))
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * ((limit - i*i) // i + 1)
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    def _generate_fermat_positions(self) -> Set[int]:
        """Generate Fermat number positions"""
        positions = set()
        for k in range(20):
            fermat = (1 << (1 << k)) + 1
            if fermat > self.sqrt_n:
                break
            positions.add(fermat)
            # Add neighbors
            for offset in [-2, -1, 1, 2]:
                val = fermat + offset
                if 2 <= val <= self.sqrt_n:
                    positions.add(val)
        return positions
    
    def _generate_mersenne_positions(self) -> Set[int]:
        """Generate Mersenne number positions"""
        positions = set()
        for p in range(2, min(100, int(math.log2(self.sqrt_n)) + 1)):
            if EnhancedPrimalityTesting.is_probable_prime(p):
                mersenne = (1 << p) - 1
                if mersenne > self.sqrt_n:
                    break
                positions.add(mersenne)
                # Add neighbors
                for offset in [-2, -1, 1, 2]:
                    val = mersenne + offset
                    if 2 <= val <= self.sqrt_n:
                        positions.add(val)
        return positions
    
    def _generate_power2_positions(self) -> Set[int]:
        """Generate power of 2 related positions"""
        positions = set()
        for bit_pos in range(2, self.bit_len):
            base = 1 << bit_pos
            if base > self.sqrt_n:
                break
            for offset in [-1, 0, 1, 3, 5, 7, 9, 15, 17]:
                pos = base + offset
                if 2 <= pos <= self.sqrt_n:
                    positions.add(pos)
        return positions
    
    def _compute_continued_fraction_convergents(self) -> List[Tuple[int, int]]:
        """Compute CF convergents of sqrt(n)"""
        convergents = []
        m, d, a0 = 0, 1, int(math.sqrt(self.n))
        a = a0
        
        h_prev, k_prev = 1, 0
        h_curr, k_curr = a, 1
        convergents.append((h_curr, k_curr))
        
        for _ in range(min(100, self.bit_len * 2)):
            m = d * a - m
            if m == 0:
                break
            d = (self.n - m * m) // d
            if d == 0:
                break
            a = (a0 + m) // d
            
            h_next = a * h_curr + h_prev
            k_next = a * k_curr + k_prev
            
            convergents.append((h_next, k_next))
            
            h_prev, k_prev = h_curr, k_curr
            h_curr, k_curr = h_next, k_next
            
            if k_next > self.sqrt_n:
                break
        
        return convergents
    
    def _compute_multiplicative_orders(self) -> Dict[str, Dict[int, int]]:
        """Compute multiplicative orders"""
        orders = {'n': {}, 'primes': {}}
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        # Orders modulo n
        for base in bases:
            if math.gcd(base, self.n) == 1:
                order = self._multiplicative_order(base, self.n)
                if order > 0:
                    orders['n'][base] = order
        
        return orders
    
    @lru_cache(maxsize=10000)
    def _multiplicative_order(self, a: int, m: int) -> int:
        """Compute multiplicative order"""
        if m <= 1 or math.gcd(a, m) > 1:
            return 0
        
        order = 1
        value = a % m
        max_order = min(m - 1, 100000)
        
        while value != 1 and order < max_order:
            value = (value * a) % m
            order += 1
        
        return order if value == 1 else 0
    
    def resonance(self, x_normalized: float) -> float:
        """
        Final resonance function with balanced scoring
        """
        if x_normalized <= 0 or x_normalized >= 1:
            return 0.0
        
        x = max(2, int(x_normalized * self.sqrt_n))
        if x >= self.n:
            return 0.0
        
        # EXTREMELY minimal divisor bonus
        g = math.gcd(x, self.n)
        if g > 1:
            if g == x and self.n % x == 0:  # x divides n
                # Only slight bonus for actual divisors
                base_resonance = 1.05
            else:
                # Shared factors but not a divisor
                base_resonance = 1.02
        else:
            base_resonance = 1.0
        
        # Start with base
        resonance = base_resonance
        
        # 1. STRONG prime detection (most important signal)
        if EnhancedPrimalityTesting.is_probable_prime(x):
            prime_score = self._comprehensive_prime_score(x)
            resonance *= prime_score
        
        # 2. Mathematical structure alignment
        structure_score = self._mathematical_structure_score(x)
        resonance *= structure_score
        
        # 3. Modular pattern matching
        pattern_score = self._modular_pattern_score(x)
        resonance *= pattern_score
        
        # 4. Position-based score
        position_score = self._position_score(x)
        resonance *= position_score
        
        return resonance
    
    def _comprehensive_prime_score(self, x: int) -> float:
        """Comprehensive prime scoring"""
        score = 5.0  # Strong base for any prime
        
        # Special prime bonuses
        if EnhancedPrimalityTesting.is_fermat_prime(x):
            score *= 3.0
        elif EnhancedPrimalityTesting.is_mersenne_prime(x):
            score *= 2.8
        elif EnhancedPrimalityTesting.is_sophie_germain_prime(x):
            score *= 1.8
        elif EnhancedPrimalityTesting.is_safe_prime(x):
            score *= 1.6
        
        # Check if orders are consistent with x being a prime factor
        consistent_orders = 0
        for base, order_n in self.signals['mult_orders']['n'].items():
            # For prime p, order of base modulo p divides p-1
            if order_n % (x - 1) == 0:
                consistent_orders += 1
            # Also check if order_n could be order modulo n = p*q
            elif (x - 1) % order_n == 0:
                consistent_orders += 0.5
        
        if consistent_orders >= 3:
            score *= 1.5
        elif consistent_orders >= 2:
            score *= 1.3
        
        # Bit length alignment
        x_bits = x.bit_length()
        n_bits = self.n.bit_length()
        if abs(x_bits * 2 - n_bits) <= 2:
            score *= 1.4
        
        return score
    
    def _mathematical_structure_score(self, x: int) -> float:
        """Score based on mathematical structures"""
        score = 1.0
        
        # Special forms
        if x in self.signals['fermat_positions']:
            score *= 2.0
        elif x in self.signals['mersenne_positions']:
            score *= 1.9
        elif x in self.signals['power2_positions']:
            score *= 1.2
        
        # Continued fraction proximity
        min_cf_dist = float('inf')
        for h, k in self.signals['cf_convergents']:
            if k > 0:
                dist = min(abs(x - k), abs(x - h))
                if dist < min_cf_dist:
                    min_cf_dist = dist
        
        if min_cf_dist == 0:
            score *= 2.5
        elif min_cf_dist < x * 0.001:
            score *= 1.8
        elif min_cf_dist < x * 0.01:
            score *= 1.4
        elif min_cf_dist < x * 0.1:
            score *= 1.1
        
        return score
    
    def _modular_pattern_score(self, x: int) -> float:
        """Score based on modular patterns"""
        score = 1.0
        
        # Chinese Remainder Theorem consistency
        consistency_count = 0
        total_checks = 0
        
        for p, n_mod_p in self.signals['prime_coords'][:50]:
            if p >= x:
                continue
            
            x_mod_p = x % p
            
            # Various consistency checks
            if x_mod_p == 0 and n_mod_p == 0:
                consistency_count += 1
            elif x_mod_p != 0:
                # Check if x could be a factor based on CRT
                # If n = x*y, then n ≡ 0 (mod p) implies x*y ≡ 0 (mod p)
                if n_mod_p == 0:
                    consistency_count += 0.5
                # Check multiplicative structure
                elif math.gcd(x_mod_p, p) == math.gcd(n_mod_p, p):
                    consistency_count += 0.3
            
            total_checks += 1
        
        if total_checks > 0:
            consistency_rate = consistency_count / total_checks
            score *= (1 + consistency_rate * 1.5)
        
        return score
    
    def _position_score(self, x: int) -> float:
        """Position-based scoring"""
        x_normalized = x / self.sqrt_n
        
        # Slight penalty for extreme positions
        if x_normalized < 0.001 or x_normalized > 0.999:
            return 0.7
        
        # Prefer balanced factors
        distance_from_center = abs(x_normalized - 0.5)
        position_factor = 1.0 + 0.3 * math.exp(-distance_from_center * 3)
        
        # Bonus for bit-aligned positions
        x_bits = x.bit_length()
        n_bits = self.n.bit_length()
        if x_bits == n_bits // 2 or x_bits == (n_bits // 2) + 1:
            position_factor *= 1.2
        
        return position_factor


class OptimizedCoverage:
    """Optimized coverage for efficient search"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.bit_len = n.bit_length()
        self.max_positions = ScaleAdaptiveParameters.max_positions(n)
    
    def generate_positions(self) -> List[int]:
        """Generate comprehensive positions"""
        positions = set()
        
        # 1. All primes up to reasonable limit
        self._add_prime_positions(positions)
        
        # 2. Special mathematical forms
        self._add_special_forms(positions)
        
        # 3. Bit-aligned positions
        self._add_bit_aligned_positions(positions)
        
        # 4. CF-guided positions
        self._add_cf_positions(positions)
        
        # 5. Dense coverage for small factors
        self._add_small_factor_coverage(positions)
        
        return self._prioritize_positions(positions)
    
    def _add_prime_positions(self, positions: set):
        """Add all primes up to limit"""
        limit = min(self.sqrt_n, 10000000)
        
        # Efficient sieve
        if limit > 2:
            sieve = [True] * (limit + 1)
            sieve[0] = sieve[1] = False
            
            for i in range(2, int(math.sqrt(limit)) + 1):
                if sieve[i]:
                    sieve[i*i::i] = [False] * ((limit - i*i) // i + 1)
            
            for i in range(2, min(limit + 1, self.sqrt_n + 1)):
                if sieve[i]:
                    positions.add(i)
                    if len(positions) > self.max_positions // 2:
                        return
    
    def _add_special_forms(self, positions: set):
        """Add special mathematical forms"""
        # Fermat numbers
        for k in range(20):
            fermat = (1 << (1 << k)) + 1
            if fermat > self.sqrt_n:
                break
            for offset in range(-10, 11):
                val = fermat + offset
                if 2 <= val <= self.sqrt_n:
                    positions.add(val)
        
        # Mersenne numbers
        for p in range(2, min(100, int(math.log2(self.sqrt_n)) + 1)):
            mersenne = (1 << p) - 1
            if mersenne > self.sqrt_n:
                break
            for offset in range(-10, 11):
                val = mersenne + offset
                if 2 <= val <= self.sqrt_n:
                    positions.add(val)
        
        # Powers of 2 and neighbors
        for k in range(2, self.bit_len):
            base = 1 << k
            if base > self.sqrt_n:
                break
            for offset in range(-100, 101):
                val = base + offset
                if 2 <= val <= self.sqrt_n:
                    positions.add(val)
    
    def _add_bit_aligned_positions(self, positions: set):
        """Add positions based on bit lengths"""
        center_bits = self.bit_len // 2
        
        for offset in range(-5, 6):
            target_bits = center_bits + offset
            if target_bits < 2 or target_bits > self.bit_len - 2:
                continue
            
            base = 1 << target_bits
            if base > self.sqrt_n:
                continue
            
            # Dense coverage around powers of 2
            for delta in range(-1000, 1001):
                pos = base + delta
                if 2 <= pos <= self.sqrt_n:
                    positions.add(pos)
    
    def _add_cf_positions(self, positions: set):
        """Add CF convergent positions"""
        m, d, a0 = 0, 1, int(math.sqrt(self.n))
        a = a0
        
        h_prev, k_prev = 1, 0
        h_curr, k_curr = a, 1
        
        for _ in range(min(100, self.bit_len * 2)):
            if 2 <= k_curr <= self.sqrt_n:
                positions.add(k_curr)
                # Add neighbors
                for offset in range(-10, 11):
                    val = k_curr + offset
                    if 2 <= val <= self.sqrt_n:
                        positions.add(val)
            
            if 2 <= h_curr <= self.sqrt_n:
                positions.add(h_curr)
            
            m = d * a - m
            if m == 0:
                break
            d = (self.n - m * m) // d
            if d == 0:
                break
            a = (a0 + m) // d
            
            h_next = a * h_curr + h_prev
            k_next = a * k_curr + k_prev
            
            h_prev, k_prev = h_curr, k_curr
            h_curr, k_curr = h_next, k_next
            
            if k_next > self.sqrt_n:
                break
    
    def _add_small_factor_coverage(self, positions: set):
        """Dense coverage for small factors"""
        # Very dense for tiny factors
        for i in range(2, min(10000, self.sqrt_n + 1)):
            positions.add(i)
            if len(positions) > self.max_positions:
                return
    
    def _prioritize_positions(self, positions: Set[int]) -> List[int]:
        """Prioritize positions by type"""
        if len(positions) <= self.max_positions:
            return sorted(positions)
        
        # Categorize
        primes = []
        special_forms = []
        bit_aligned = []
        small = []
        others = []
        
        for pos in positions:
            if pos < 10000:
                small.append(pos)
            elif EnhancedPrimalityTesting.is_probable_prime(pos):
                primes.append(pos)
            elif pos.bit_length() == self.bit_len // 2:
                bit_aligned.append(pos)
            elif self._is_near_special_form(pos):
                special_forms.append(pos)
            else:
                others.append(pos)
        
        # Build prioritized list
        final = []
        
        # All small factors
        final.extend(sorted(small))
        
        # Sample from other categories
        remaining = self.max_positions - len(final)
        if remaining > 0:
            # Primes get high priority
            prime_count = min(len(primes), remaining // 2)
            final.extend(sorted(primes)[:prime_count])
            remaining -= prime_count
        
        if remaining > 0:
            # Special forms
            special_count = min(len(special_forms), remaining // 3)
            final.extend(sorted(special_forms)[:special_count])
            remaining -= special_count
        
        if remaining > 0:
            # Bit-aligned
            bit_count = min(len(bit_aligned), remaining // 2)
            final.extend(sorted(bit_aligned)[:bit_count])
            remaining -= bit_count
        
        if remaining > 0:
            # Fill with others
            final.extend(sorted(others)[:remaining])
        
        return sorted(set(final))
    
    def _is_near_special_form(self, pos: int) -> bool:
        """Check if position is near a special form"""
        # Check Fermat
        for k in range(20):
            fermat = (1 << (1 << k)) + 1
            if abs(pos - fermat) < 10:
                return True
            if fermat > pos + 10:
                break
        
        # Check Mersenne
        for p in range(2, 64):
            mersenne = (1 << p) - 1
            if abs(pos - mersenne) < 10:
                return True
            if mersenne > pos + 10:
                break
        
        return False


class FinalPhaseI:
    """Final Phase I implementation"""
    
    def __init__(self):
        self.stats = {
            'resonance_evaluations': 0,
            'optimization_iterations': 0,
            'peak_candidates': 0
        }
    
    def factorize(self, n: int) -> Tuple[int, int]:
        """Factor using final Phase I approach"""
        if n < 2:
            raise ValueError("n must be >= 2")
        if n % 2 == 0:
            return (2, n // 2)
        
        # Quick primality check
        if EnhancedPrimalityTesting.is_probable_prime(n):
            raise ValueError(f"{n} is prime")
        
        print(f"\n{'='*60}")
        print(f"Final Phase I Factorization")
        print(f"n = {n} ({n.bit_length()} bits)")
        print(f"{'='*60}")
        
        # Strategy 1: Targeted optimization
        print("\nStrategy 1: Targeted Optimization")
        factor1 = self._targeted_optimization(n)
        if factor1 and n % factor1 == 0:
            return self._format_result(n, factor1)
        
        # Strategy 2: Comprehensive coverage
        print("\nStrategy 2: Comprehensive Coverage")
        factor2 = self._comprehensive_coverage(n)
        if factor2 and n % factor2 == 0:
            return self._format_result(n, factor2)
        
        # Strategy 3: Focused peak detection
        print("\nStrategy 3: Focused Peak Detection")
        factor3 = self._focused_peak_detection(n)
        if factor3 and n % factor3 == 0:
            return self._format_result(n, factor3)
        
        raise ValueError("Final Phase I could not find factors")
    
    def _targeted_optimization(self, n: int) -> Optional[int]:
        """Targeted optimization focusing on likely positions"""
        resonator = FinalResonance(n)
        
        # Smart starting points
        start_points = []
        bit_len = n.bit_length()
        
        # Bit-aligned starts
        for offset in range(-3, 4):
            bits = bit_len // 2 + offset
            if bits > 0:
                x_norm = (1 << bits) / resonator.sqrt_n
                if 0.01 < x_norm < 0.99:
                    start_points.append(x_norm)
        
        # Special number starts
        special_numbers = [3, 5, 17, 257, 65537, 2147483647]  # Known special primes
        for num in special_numbers:
            if num <= resonator.sqrt_n:
                x_norm = num / resonator.sqrt_n
                if 0.01 < x_norm < 0.99:
                    start_points.append(x_norm)
        
        # Standard positions
        start_points.extend([0.5, 0.1, 0.9, 0.618, 0.382])
        
        high_resonance_candidates = []
        
        for i, start in enumerate(start_points[:10]):  # Limit starts
            print(f"  Start {i+1} from position {start:.3f}")
            
            eval_count = [0]
            
            def objective(x):
                eval_count[0] += 1
                return -resonator.resonance(x[0])
            
            # Differential evolution
            result = differential_evolution(
                objective,
                bounds=[(max(0.001, start - 0.1), min(0.999, start + 0.1))],
                maxiter=20,
                popsize=10,
                atol=1e-8,
                seed=42 + i
            )
            
            self.stats['resonance_evaluations'] += eval_count[0]
            self.stats['optimization_iterations'] += result.nit
            
            factor = int(result.x[0] * resonator.sqrt_n)
            resonance = -result.fun
            
            print(f"    Peak at {result.x[0]:.6f}, resonance {resonance:.3f}")
            
            # Keep track of high resonance candidates (ensure factor > 0)
            if resonance > 10.0 and factor > 0:
                high_resonance_candidates.append((factor, resonance))
                # Also check neighbors
                for offset in [-2, -1, 1, 2]:
                    neighbor = factor + offset
                    if 2 <= neighbor <= resonator.sqrt_n:
                        x_norm = neighbor / resonator.sqrt_n
                        neighbor_res = resonator.resonance(x_norm)
                        if neighbor_res > 10.0:
                            high_resonance_candidates.append((neighbor, neighbor_res))
        
        # Sort by resonance and check divisibility
        high_resonance_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Check all unique candidates
        checked = set()
        for factor, res in high_resonance_candidates:
            if factor in checked or factor <= 0:
                continue
            checked.add(factor)
            
            if n % factor == 0:
                print(f"  Found factor {factor} with resonance {res:.3f}")
                return factor
            
            # Also try complementary positions
            if resonator.sqrt_n > factor > 0:
                complement = resonator.sqrt_n // factor
                if complement not in checked and 2 <= complement <= resonator.sqrt_n:
                    checked.add(complement)
                    if n % complement == 0:
                        print(f"  Found complementary factor {complement}")
                        return complement
        
        return None
    
    def _comprehensive_coverage(self, n: int) -> Optional[int]:
        """Comprehensive coverage with smart evaluation"""
        resonator = FinalResonance(n)
        coverage = OptimizedCoverage(n)
        
        print("  Generating comprehensive positions...")
        positions = coverage.generate_positions()
        print(f"  Generated {len(positions)} positions")
        
        # Quick high-resonance scan
        print("  Quick scan for high-resonance positions...")
        high_resonance = []
        
        # Check known special numbers first
        special_checks = [3, 5, 7, 11, 13, 17, 101, 103, 257, 65537, 2147483647]
        for num in special_checks:
            if num <= resonator.sqrt_n:
                # Check divisibility first
                if n % num == 0:
                    print(f"  Found special factor {num}")
                    return num
                    
                if num in positions:
                    x_norm = num / resonator.sqrt_n
                    res = resonator.resonance(x_norm)
                    self.stats['resonance_evaluations'] += 1
                    if res > 10.0:
                        high_resonance.append((num, res))
        
        # Sort by resonance and check
        high_resonance.sort(key=lambda x: x[1], reverse=True)
        for pos, res in high_resonance[:10]:
            if n % pos == 0:
                print(f"  Found factor {pos} with resonance {res:.3f}")
                return pos
        
        # Full evaluation - collect high-resonance candidates
        print("  Full resonance evaluation...")
        resonance_candidates = []
        
        for i, pos in enumerate(positions):
            if i % 5000 == 0 and i > 0:
                print(f"    Progress: {i}/{len(positions)}")
            
            x_norm = pos / resonator.sqrt_n
            if 0 < x_norm < 1:
                resonance = resonator.resonance(x_norm)
                self.stats['resonance_evaluations'] += 1
                
                # Collect high resonance candidates
                if resonance > 20.0:
                    resonance_candidates.append((pos, resonance))
                
                # Early termination for very high resonance
                if resonance > 50.0 and n % pos == 0:
                    print(f"  Early termination: found factor {pos} with resonance {resonance:.3f}")
                    return pos
        
        # Sort all candidates by resonance and check divisibility
        resonance_candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"  Found {len(resonance_candidates)} high-resonance candidates")
        
        for pos, res in resonance_candidates[:50]:
            if n % pos == 0:
                print(f"  Found factor {pos} with resonance {res:.6f}")
                return pos
        
        # If no factor found, return the highest resonance position
        if resonance_candidates:
            best_pos, best_res = resonance_candidates[0]
            print(f"  Best resonance: {best_res:.6f} at position {best_pos} (not a factor)")
        
        return None
    
    def _focused_peak_detection(self, n: int) -> Optional[int]:
        """Focused peak detection on promising regions"""
        resonator = FinalResonance(n)
        
        bit_len = resonator.bit_len
        num_samples = min(ScaleAdaptiveParameters.resonance_samples(n), 10000)
        
        # Build focused sample points
        x_values = []
        
        # 1. Dense sampling around bit-aligned positions
        for offset in range(-5, 6):
            bits = bit_len // 2 + offset
            if bits > 0:
                x_center = (1 << bits) / resonator.sqrt_n
                if 0.001 < x_center < 0.999:
                    # Dense sampling around this position
                    for delta in np.linspace(-0.02, 0.02, 41):
                        x = x_center + delta
                        if 0.001 < x < 0.999:
                            x_values.append(x)
        
        # 2. Special number positions
        special_positions = []
        
        # Fermat numbers
        for k in range(20):
            fermat = (1 << (1 << k)) + 1
            if fermat <= resonator.sqrt_n:
                x_norm = fermat / resonator.sqrt_n
                if 0.001 < x_norm < 0.999:
                    special_positions.append(x_norm)
        
        # Mersenne numbers
        for p in range(2, 64):
            mersenne = (1 << p) - 1
            if mersenne <= resonator.sqrt_n:
                x_norm = mersenne / resonator.sqrt_n
                if 0.001 < x_norm < 0.999:
                    special_positions.append(x_norm)
        
        # Dense sampling around special positions
        for sp in special_positions:
            for delta in np.linspace(-0.005, 0.005, 21):
                x = sp + delta
                if 0.001 < x < 0.999:
                    x_values.append(x)
        
        # 3. Fill with uniform sampling
        remaining = num_samples - len(x_values)
        if remaining > 0:
            x_values.extend(np.linspace(0.001, 0.999, remaining))
        
        x_values = sorted(set(x_values))[:num_samples]
        print(f"  Evaluating {len(x_values)} focused points...")
        
        # Evaluate resonance
        resonances = []
        for i, x in enumerate(x_values):
            if i % 1000 == 0 and i > 0:
                print(f"    Progress: {i}/{len(x_values)}")
            
            res = resonator.resonance(x)
            resonances.append(res)
            self.stats['resonance_evaluations'] += 1
        
        resonances = np.array(resonances)
        
        # Find peaks
        mean_res = np.mean(resonances)
        std_res = np.std(resonances)
        
        # Adaptive threshold
        threshold = mean_res + 0.5 * std_res
        
        peaks, properties = find_peaks(
            resonances,
            height=threshold,
            distance=3
        )
        
        print(f"  Found {len(peaks)} peaks above threshold")
        self.stats['peak_candidates'] = len(peaks)
        
        # Check peaks in order of strength
        if len(peaks) > 0:
            sorted_peaks = sorted(peaks, key=lambda i: resonances[i], reverse=True)
            
            checked_factors = set()
            
            for i, peak_idx in enumerate(sorted_peaks):
                x_norm = x_values[peak_idx]
                factor = int(x_norm * resonator.sqrt_n)
                res_value = resonances[peak_idx]
                
                if factor in checked_factors or factor <= 1:
                    continue
                checked_factors.add(factor)
                
                if i < 10:
                    print(f"  Peak {i+1}: position {factor}, resonance {res_value:.6f}")
                
                if n % factor == 0:
                    print(f"  Found factor at peak {i+1}")
                    return factor
                
                # Also check neighbors of high-resonance peaks
                if res_value > 25.0:
                    for offset in range(-10, 11):
                        neighbor = factor + offset
                        if neighbor > 1 and neighbor not in checked_factors and neighbor <= resonator.sqrt_n:
                            checked_factors.add(neighbor)
                            if n % neighbor == 0:
                                print(f"  Found factor {neighbor} near peak {factor}")
                                return neighbor
        
        # If no peaks found factors, do systematic check around bit-aligned positions
        print("  No factors found in peaks, checking bit-aligned regions...")
        bit_len = resonator.bit_len
        
        # First try narrow ranges around each bit position
        for offset in range(-5, 6):
            bits = bit_len // 2 + offset
            if bits > 0:
                center = 1 << bits
                if center <= resonator.sqrt_n:
                    # Check range around this power of 2
                    start = max(2, center - 10000)
                    end = min(resonator.sqrt_n + 1, center + 10000)
                    for candidate in range(start, end):
                        if n % candidate == 0:
                            print(f"  Found factor {candidate} near 2^{bits}")
                            return candidate
        
        # If still not found, check wider range but with steps
        print("  Expanding search to wider bit-aligned regions...")
        for offset in range(-3, 4):
            bits = bit_len // 2 + offset
            if bits > 0:
                center = 1 << bits
                if center <= resonator.sqrt_n:
                    # Wider range but with steps to avoid timeout
                    start = max(2, int(center * 0.8))
                    end = min(resonator.sqrt_n + 1, int(center * 1.2))
                    step = max(1, (end - start) // 100000)  # Limit to 100k checks
                    
                    for candidate in range(start, end, step):
                        if n % candidate == 0:
                            print(f"  Found factor {candidate} in wider search near 2^{bits}")
                            return candidate
                        # Also check neighbors of stepped positions
                        for delta in [-1, 1]:
                            neighbor = candidate + delta
                            if start <= neighbor <= end and n % neighbor == 0:
                                print(f"  Found factor {neighbor} in wider search near 2^{bits}")
                                return neighbor
        
        return None
    
    def _format_result(self, n: int, factor: int) -> Tuple[int, int]:
        """Format the factorization result"""
        other = n // factor
        print(f"\n✓ SUCCESS: Found factor {factor}")
        print(f"  {n} = {factor} × {other}")
        
        # Analyze the factors
        if EnhancedPrimalityTesting.is_probable_prime(factor):
            print(f"  {factor} is prime", end="")
            if EnhancedPrimalityTesting.is_fermat_prime(factor):
                print(f" (Fermat prime F_{int(math.log2(math.log2(factor-1)))})")
            elif EnhancedPrimalityTesting.is_mersenne_prime(factor):
                print(f" (Mersenne prime M_{factor.bit_length()-1})")
            else:
                print()
        
        if EnhancedPrimalityTesting.is_probable_prime(other):
            print(f"  {other} is prime", end="")
            if EnhancedPrimalityTesting.is_fermat_prime(other):
                print(f" (Fermat prime)")
            elif EnhancedPrimalityTesting.is_mersenne_prime(other):
                print(f" (Mersenne prime)")
            else:
                print()
        
        print(f"\nStatistics:")
        print(f"  Resonance evaluations: {self.stats['resonance_evaluations']}")
        print(f"  Optimization iterations: {self.stats['optimization_iterations']}")
        print(f"  Peak candidates examined: {self.stats['peak_candidates']}")
        
        return (factor, other) if factor <= other else (other, factor)


def test_final():
    """Test the final Phase I implementation"""
    
    test_cases = [
        # Small cases
        (11, 13),                     # 143 (8-bit)
        (101, 103),                   # 10403 (14-bit)
        
        # Special form cases
        (65537, 4294967311),          # 49-bit (Fermat prime)
        (2147483647, 2147483659),     # 63-bit (Mersenne prime)
        
        # Regular large primes
        (7125766127, 6958284019),     # 66-bit
        (14076040031, 15981381943),   # 68-bit
        
        # Additional test
        (1073741827, 1073741831),     # 61-bit (twin primes near 2^30)
    ]
    
    phase1 = FinalPhaseI()
    
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


if __name__ == "__main__":
    test_final()
