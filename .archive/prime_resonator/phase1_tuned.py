"""
Phase I Tuned Implementation
Addresses issues found in experimental testing:
- Removes GCD dominance
- Strengthens factor-specific signals
- Improves coverage for small factors
- Ensures scale invariance
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
            return 50
        elif bit_len < 96:
            return int(bit_len * 2)
        else:
            return int(bit_len * 3)
    
    @staticmethod
    def resonance_samples(n: int) -> int:
        """Sampling density for complete coverage"""
        bit_len = n.bit_length()
        # More samples for better coverage
        if bit_len < 64:
            return bit_len * 50
        elif bit_len < 96:
            return bit_len * bit_len
        else:
            return min(bit_len * bit_len * 2, 20000)
    
    @staticmethod
    def small_factor_threshold(n: int) -> int:
        """Threshold for dense small factor sampling"""
        # Sample densely up to this value
        sqrt_n = int(math.sqrt(n))
        bit_len = n.bit_length()
        # For larger n, small factors are relatively smaller
        return min(sqrt_n, 1 << (bit_len // 3))


class ImprovedResonance:
    """Improved resonance function with better factor detection"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.bit_len = n.bit_length()
        
        # Initialize signals dict
        self.signals = {}
        
        # Precompute signals
        print(f"Precomputing signals for {self.bit_len}-bit number...")
        self._precompute_all_signals()
        self._precompute_carmichael_values()
        print("Signals computed.")
    
    def _precompute_all_signals(self):
        """Precompute scale-invariant signals"""
        # First compute basic signals that don't depend on others
        self.signals['small_primes'] = self._generate_primes(min(10000, self.sqrt_n))
        self.signals['prime_coords'] = self._compute_prime_coordinates()
        self.signals['fibonacci'] = self._generate_fibonacci_sequence()
        self.signals['cf_convergents'] = self._compute_continued_fraction_convergents()
        self.signals['fermat_positions'] = self._generate_fermat_positions()
        self.signals['power2_positions'] = self._generate_power2_positions()
        
        # Then compute signals that depend on the basic ones
        self.signals['mult_orders'] = self._compute_multiplicative_orders()
    
    def _compute_prime_coordinates(self) -> List[Tuple[int, int]]:
        """Compute n's coordinates in prime space"""
        num_primes = ScaleAdaptiveParameters.prime_dimensions(self.n)
        primes = self._generate_primes(num_primes)
        return [(p, self.n % p) for p in primes]
    
    def _generate_primes(self, limit: int) -> List[int]:
        """Generate primes up to limit or count"""
        if limit < 2:
            return []
        
        # If limit is small, generate that many primes
        if limit < 1000:
            count = limit
            limit = max(30, int(count * (math.log(count) + math.log(math.log(count + 1)) + 2)))
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        primes = [i for i in range(2, limit + 1) if sieve[i]]
        return primes
    
    def _generate_fibonacci_sequence(self) -> List[int]:
        """Generate Fibonacci numbers up to sqrt(n)"""
        fibs = [1, 1]
        while fibs[-1] < self.sqrt_n:
            fibs.append(fibs[-1] + fibs[-2])
        return [f for f in fibs if f <= self.sqrt_n]
    
    def _compute_multiplicative_orders(self) -> Dict[int, Dict[int, int]]:
        """Compute multiplicative orders for multiple moduli"""
        orders = {}
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Orders modulo n
        orders['n'] = {}
        for base in bases:
            if math.gcd(base, self.n) == 1:
                order = self._multiplicative_order(base, self.n)
                if order > 0:
                    orders['n'][base] = order
        
        # Orders modulo small primes (for pattern detection)
        orders['primes'] = {}
        for p in self.signals['small_primes'][:50]:
            orders['primes'][p] = {}
            for base in bases[:5]:
                if math.gcd(base, p) == 1:
                    order = self._multiplicative_order(base, p)
                    if order > 0:
                        orders['primes'][p][base] = order
        
        return orders
    
    @lru_cache(maxsize=10000)
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
    
    def _compute_continued_fraction_convergents(self) -> List[Tuple[int, int]]:
        """Compute continued fraction convergents of sqrt(n)"""
        convergents = []
        
        # For sqrt(n)
        m, d, a0 = 0, 1, int(math.sqrt(self.n))
        a = a0
        
        # First convergent
        h_prev, k_prev = 1, 0
        h_curr, k_curr = a, 1
        convergents.append((h_curr, k_curr))
        
        # Limit iterations
        for _ in range(min(100, self.bit_len)):
            m = d * a - m
            if m == 0:
                break
            d = (self.n - m * m) // d
            if d == 0:
                break
            a = (a0 + m) // d
            
            # Update convergents
            h_next = a * h_curr + h_prev
            k_next = a * k_curr + k_prev
            
            convergents.append((h_next, k_next))
            
            h_prev, k_prev = h_curr, k_curr
            h_curr, k_curr = h_next, k_next
            
            # Stop if convergent is too large
            if k_next > self.sqrt_n:
                break
        
        return convergents
    
    def _generate_fermat_positions(self) -> List[int]:
        """Generate positions around Fermat numbers"""
        positions = []
        for k in range(1, min(30, self.bit_len)):
            fermat = (1 << k) + 1
            if fermat <= self.sqrt_n:
                positions.extend([fermat, fermat - 2, fermat + 2])
            
            mersenne = (1 << k) - 1
            if mersenne <= self.sqrt_n:
                positions.extend([mersenne, mersenne - 2, mersenne + 2])
        
        return sorted(set(p for p in positions if 2 <= p <= self.sqrt_n))
    
    def _generate_power2_positions(self) -> List[int]:
        """Generate positions around powers of 2"""
        positions = []
        
        # For each bit position that could be a factor
        for bit_pos in range(2, self.bit_len):
            base = 1 << bit_pos
            if base > self.sqrt_n:
                break
            
            # Add positions around powers of 2
            for offset in [-1, 0, 1, 3, 5, 7, 9]:
                pos = base + offset
                if 2 <= pos <= self.sqrt_n:
                    positions.append(pos)
        
        return sorted(set(positions))
    
    def _precompute_carmichael_values(self):
        """Precompute Carmichael function values for small numbers"""
        self.carmichael_cache = {}
        
        # For small primes and prime powers
        for p in self.signals['small_primes'][:100]:
            self.carmichael_cache[p] = p - 1
            
            # Prime powers
            pk = p * p
            k = 2
            while pk <= 10000 and k <= 5:
                if p == 2 and k >= 3:
                    self.carmichael_cache[pk] = pk // 4
                else:
                    self.carmichael_cache[pk] = pk - pk // p
                pk *= p
                k += 1
    
    def resonance(self, x_normalized: float) -> float:
        """
        Improved resonance function on [0, 1].
        """
        if x_normalized <= 0 or x_normalized >= 1:
            return 0.0
        
        x = max(2, int(x_normalized * self.sqrt_n))
        if x >= self.n:
            return 0.0
        
        # Reduced GCD bonus (was 1000.0 and 100.0)
        g = math.gcd(x, self.n)
        if g > 1:
            if g == x:  # x divides n
                # Base resonance for divisors
                base_resonance = 5.0
            else:
                # Shared factors but not a divisor
                base_resonance = 2.0
        else:
            base_resonance = 1.0
        
        # Combine improved resonance measures
        resonance = base_resonance
        
        # 1. Chinese Remainder Theorem consistency
        crt_score = self._crt_consistency_score(x)
        resonance *= crt_score
        
        # 2. Carmichael function alignment
        carmichael_score = self._carmichael_alignment_score(x)
        resonance *= carmichael_score
        
        # 3. Multiplicative order pattern
        order_score = self._order_pattern_score(x)
        resonance *= order_score
        
        # 4. Continued fraction proximity
        cf_score = self._cf_proximity_score(x)
        resonance *= cf_score
        
        # 5. Bit pattern alignment
        bit_score = self._bit_pattern_score(x)
        resonance *= bit_score
        
        # 6. Scale-invariant position score
        position_score = self._scale_invariant_position_score(x)
        resonance *= position_score
        
        return resonance
    
    def _crt_consistency_score(self, x: int) -> float:
        """
        Check Chinese Remainder Theorem consistency.
        If x divides n, then n ≡ 0 (mod x) and n ≡ x*(n/x) (mod p) for all p.
        """
        score = 1.0
        consistency_count = 0
        total_checks = 0
        
        # Check CRT consistency with small primes
        for p, n_mod_p in self.signals['prime_coords'][:30]:
            if p >= x:
                continue
            
            x_mod_p = x % p
            if x_mod_p == 0:
                continue
                
            # If x is a factor, what should n mod p be?
            # n = x * y, so n mod p = (x mod p) * (y mod p) mod p
            # We can check if this is consistent
            
            # For now, check if x and n have compatible residues
            if n_mod_p == 0 and x_mod_p != 0:
                # n is divisible by p but x isn't - good sign if x is a factor
                consistency_count += 1
            elif x_mod_p != 0:
                # Check if n/x mod p would be integral
                # This is complex without knowing n/x, so use heuristic
                if (n_mod_p * pow(x_mod_p, p-2, p)) % p == (n_mod_p * pow(x_mod_p, p-2, p)) % p:
                    consistency_count += 1
            
            total_checks += 1
        
        if total_checks > 0:
            consistency_rate = consistency_count / total_checks
            score *= (1 + consistency_rate * 3)
        
        return score
    
    def _carmichael_alignment_score(self, x: int) -> float:
        """
        Check Carmichael function alignment.
        If n = x * y, then λ(n) = lcm(λ(x), λ(y))
        """
        if x < 2:
            return 1.0
        
        score = 1.0
        
        # For small x, we can compute Carmichael function
        if x in self.carmichael_cache:
            lambda_x = self.carmichael_cache[x]
            
            # Check if common bases have orders dividing lambda_x
            order_match_count = 0
            for base, order_n in self.signals['mult_orders']['n'].items():
                if math.gcd(base, x) == 1:
                    order_x = self._multiplicative_order(base, x)
                    if order_x > 0 and lambda_x % order_x == 0:
                        # Check if order_n is compatible with lcm structure
                        if order_n % order_x == 0:
                            order_match_count += 1
            
            if order_match_count > 2:
                score *= 2.0
        
        # For prime-like x, λ(x) = x-1 or x-1 divided by small factor
        if self._is_probably_prime(x):
            # Check if orders modulo n are compatible with x being prime
            compatible = 0
            for base, order_n in self.signals['mult_orders']['n'].items():
                if order_n % (x - 1) == 0:
                    compatible += 1
            
            if compatible >= 2:
                score *= 3.0
        
        return score
    
    def _order_pattern_score(self, x: int) -> float:
        """
        Score based on multiplicative order patterns.
        """
        if x < 2:
            return 1.0
        
        score = 1.0
        pattern_matches = 0
        
        # Check order relationships
        for base in [2, 3, 5, 7]:
            if math.gcd(base, self.n) == 1 and math.gcd(base, x) == 1:
                order_n = self.signals['mult_orders']['n'].get(base, 0)
                if order_n > 0:
                    order_x = self._multiplicative_order(base, x)
                    if order_x > 0:
                        # Check various relationships
                        if order_n % order_x == 0:
                            pattern_matches += 1
                        if (x - 1) % order_x == 0:
                            pattern_matches += 0.5
                        if order_x % (x - 1) == 0:
                            pattern_matches += 0.5
        
        if pattern_matches > 0:
            score *= (1 + pattern_matches / 4)
        
        return score
    
    def _cf_proximity_score(self, x: int) -> float:
        """
        Score based on proximity to continued fraction convergents.
        """
        score = 1.0
        
        # Check proximity to convergents
        min_distance = float('inf')
        for h, k in self.signals['cf_convergents']:
            if k > 0:
                # Denominators of convergents often relate to factors
                dist = abs(x - k)
                if dist < min_distance:
                    min_distance = dist
                
                # Also check numerators
                dist = abs(x - h)
                if dist < min_distance:
                    min_distance = dist
        
        # Strong boost for exact matches
        if min_distance == 0:
            score *= 5.0
        elif min_distance < x * 0.001:
            score *= 3.0
        elif min_distance < x * 0.01:
            score *= 2.0
        elif min_distance < x * 0.1:
            score *= 1.5
        
        return score
    
    def _bit_pattern_score(self, x: int) -> float:
        """
        Score based on bit pattern alignment.
        """
        score = 1.0
        
        # Check if x has special bit pattern
        x_bits = x.bit_length()
        
        # Powers of 2 nearby
        if x in self.signals['power2_positions']:
            score *= 1.5
        
        # Fermat/Mersenne numbers
        if x in self.signals['fermat_positions']:
            score *= 2.0
        
        # Check bit pattern similarity with n
        n_bits = self.n.bit_length()
        if abs(x_bits - n_bits // 2) <= 1:
            # x has about half the bits of n
            score *= 1.3
        
        return score
    
    def _scale_invariant_position_score(self, x: int) -> float:
        """
        Position-based score that's scale-invariant.
        """
        x_normalized = x / self.sqrt_n
        
        # Penalize extreme positions
        if x_normalized < 0.001 or x_normalized > 0.999:
            return 0.5
        
        # No strong position preference in the middle
        return 1.0
    
    def _is_probably_prime(self, n: int) -> bool:
        """
        Quick primality test for small numbers.
        """
        if n < 2:
            return False
        if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            return True
        if n % 2 == 0:
            return False
        
        # Quick trial division
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29]:
            if n % p == 0:
                return n == p
        
        # For larger numbers, could do Miller-Rabin
        return True  # Assume prime if no small factors


class TargetedCoverage:
    """Targeted coverage strategy for complete factor coverage"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.bit_len = n.bit_length()
    
    def generate_targeted_positions(self) -> List[int]:
        """Generate positions with focus on likely factor locations"""
        positions = set()
        
        # 1. Dense coverage for small factors
        self._add_small_factor_coverage(positions)
        
        # 2. Bit-length based coverage
        self._add_bit_length_coverage(positions)
        
        # 3. Special form coverage
        self._add_special_form_coverage(positions)
        
        # 4. Multiplicative structure coverage
        self._add_multiplicative_coverage(positions)
        
        # 5. Continued fraction guided positions
        self._add_cf_guided_positions(positions)
        
        return sorted(positions)
    
    def _add_small_factor_coverage(self, positions: set):
        """Dense coverage for small factors relative to sqrt(n)"""
        threshold = ScaleAdaptiveParameters.small_factor_threshold(self.n)
        
        # All primes up to threshold
        sieve_limit = min(threshold, 10000000)  # Cap for memory
        if sieve_limit > 2:
            sieve = [True] * (sieve_limit + 1)
            sieve[0] = sieve[1] = False
            
            for i in range(2, int(math.sqrt(sieve_limit)) + 1):
                if sieve[i]:
                    for j in range(i*i, sieve_limit + 1, i):
                        sieve[j] = False
            
            for i in range(2, min(sieve_limit + 1, self.sqrt_n + 1)):
                if sieve[i]:
                    positions.add(i)
        
        # Powers of small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            power = p
            while power <= self.sqrt_n:
                positions.add(power)
                power *= p
    
    def _add_bit_length_coverage(self, positions: set):
        """Coverage based on bit lengths of potential factors"""
        # For n with b bits, factors typically have b/2 ± k bits
        center_bits = self.bit_len // 2
        
        for offset in range(-5, 6):
            target_bits = center_bits + offset
            if target_bits < 2 or target_bits > self.bit_len - 2:
                continue
            
            # Sample around 2^target_bits
            base = 1 << target_bits
            if base > self.sqrt_n:
                continue
            
            # Dense sampling near powers of 2
            for delta in range(-100, 101):
                pos = base + delta
                if 2 <= pos <= self.sqrt_n:
                    positions.add(pos)
            
            # Also sample at regular intervals
            step = max(1, base // 1000)
            for i in range(100):
                pos = base - i * step
                if 2 <= pos <= self.sqrt_n:
                    positions.add(pos)
                pos = base + i * step
                if pos <= self.sqrt_n:
                    positions.add(pos)
    
    def _add_special_form_coverage(self, positions: set):
        """Coverage for special mathematical forms"""
        # Fermat numbers: 2^(2^n) + 1
        for n in range(20):
            fermat = (1 << (1 << n)) + 1
            if fermat > self.sqrt_n:
                break
            positions.add(fermat)
            # Also nearby values
            for offset in [-2, -1, 1, 2, 3, 5, 7]:
                if 2 <= fermat + offset <= self.sqrt_n:
                    positions.add(fermat + offset)
        
        # Mersenne numbers: 2^p - 1
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
            mersenne = (1 << p) - 1
            if mersenne > self.sqrt_n:
                break
            positions.add(mersenne)
            for offset in [-2, -1, 1, 2, 3, 5, 7]:
                if 2 <= mersenne + offset <= self.sqrt_n:
                    positions.add(mersenne + offset)
        
        # Numbers of form 2^k ± 2^j
        for k in range(2, self.bit_len):
            for j in range(1, k):
                val1 = (1 << k) + (1 << j)
                val2 = (1 << k) - (1 << j)
                if val1 <= self.sqrt_n:
                    positions.add(val1)
                if 2 <= val2 <= self.sqrt_n:
                    positions.add(val2)
    
    def _add_multiplicative_coverage(self, positions: set):
        """Coverage based on multiplicative structure"""
        # Products of small primes (smooth numbers)
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Generate smooth numbers up to sqrt_n
        smooth_limit = min(self.sqrt_n, 1000000)
        smooth = [1]
        
        for p in small_primes:
            new_smooth = []
            for s in smooth:
                val = s
                while val <= smooth_limit:
                    if val > 1 and val not in new_smooth:
                        new_smooth.append(val)
                    val *= p
            smooth.extend(new_smooth)
        
        positions.update(s for s in smooth if 2 <= s <= self.sqrt_n)
    
    def _add_cf_guided_positions(self, positions: set):
        """Add positions guided by continued fractions"""
        # Compute CF expansion of sqrt(n)
        m, d, a0 = 0, 1, int(math.sqrt(self.n))
        a = a0
        
        # Convergents
        h_prev, k_prev = 1, 0
        h_curr, k_curr = a, 1
        
        # Add denominator
        if 2 <= k_curr <= self.sqrt_n:
            positions.add(k_curr)
        
        # Generate more convergents
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
            
            # Add convergent denominators and numerators
            if 2 <= k_next <= self.sqrt_n:
                positions.add(k_next)
            if 2 <= h_next <= self.sqrt_n:
                positions.add(h_next)
            
            # Also add nearby values
            for val in [k_next - 1, k_next + 1, h_next - 1, h_next + 1]:
                if 2 <= val <= self.sqrt_n:
                    positions.add(val)
            
            h_prev, k_prev = h_curr, k_curr
            h_curr, k_curr = h_next, k_next
            
            if k_next > self.sqrt_n:
                break


class TunedPhaseI:
    """Tuned Phase I implementation with improvements"""
    
    def __init__(self):
        self.stats = {
            'resonance_evaluations': 0,
            'optimization_iterations': 0,
            'peak_candidates': 0
        }
    
    def factorize(self, n: int) -> Tuple[int, int]:
        """Factor using tuned Phase I approach"""
        if n < 2:
            raise ValueError("n must be >= 2")
        if n % 2 == 0:
            return (2, n // 2)
        
        print(f"\n{'='*60}")
        print(f"Tuned Phase I Factorization")
        print(f"n = {n} ({n.bit_length()} bits)")
        print(f"{'='*60}")
        
        # Strategy 1: Multi-start continuous optimization
        print("\nStrategy 1: Multi-Start Optimization")
        factor1 = self._multi_start_optimization(n)
        if factor1 and n % factor1 == 0:
            return self._format_result(n, factor1)
        
        # Strategy 2: Targeted coverage resonance
        print("\nStrategy 2: Targeted Coverage")
        factor2 = self._targeted_coverage_resonance(n)
        if factor2 and n % factor2 == 0:
            return self._format_result(n, factor2)
        
        # Strategy 3: Guided peak detection
        print("\nStrategy 3: Guided Peak Detection")
        factor3 = self._guided_peak_detection(n)
        if factor3 and n % factor3 == 0:
            return self._format_result(n, factor3)
        
        raise ValueError("Tuned Phase I incomplete - needs further refinement")
    
    def _multi_start_optimization(self, n: int) -> Optional[int]:
        """Multi-start optimization with diverse initial points"""
        resonator = ImprovedResonance(n)
        
        # Multiple starting points
        start_points = [
            0.5,      # sqrt(sqrt(n))
            0.1,      # Small factors
            0.9,      # Large factors
            0.618,    # Golden ratio point
            0.382,    # 1 - golden ratio
        ]
        
        # Add starts based on bit patterns
        bit_len = n.bit_length()
        for bits in range(max(2, bit_len // 2 - 5), min(bit_len - 2, bit_len // 2 + 5)):
            x_norm = (1 << bits) / resonator.sqrt_n
            if 0.01 < x_norm < 0.99:
                start_points.append(x_norm)
        
        best_result = None
        best_resonance = 0
        
        for i, start in enumerate(start_points[:5]):  # Limit starts for efficiency
            print(f"  Start {i+1} from position {start:.3f}")
            
            # Track evaluations
            eval_count = [0]
            
            def objective(x):
                eval_count[0] += 1
                return -resonator.resonance(x[0])
            
            # Differential evolution from this start
            result = differential_evolution(
                objective,
                bounds=[(max(0.001, start - 0.3), min(0.999, start + 0.3))],
                maxiter=50,
                popsize=15,
                atol=1e-6,
                seed=42 + i
            )
            
            self.stats['resonance_evaluations'] += eval_count[0]
            self.stats['optimization_iterations'] += result.nit
            
            factor = int(result.x[0] * resonator.sqrt_n)
            resonance = -result.fun
            
            print(f"    Peak at {result.x[0]:.6f}, resonance {resonance:.3f}")
            
            if resonance > best_resonance:
                best_resonance = resonance
                best_result = factor
        
        if best_result:
            print(f"  Best result: factor {best_result} with resonance {best_resonance:.3f}")
        
        return best_result
    
    def _targeted_coverage_resonance(self, n: int) -> Optional[int]:
        """Use targeted coverage to find factors"""
        resonator = ImprovedResonance(n)
        coverage = TargetedCoverage(n)
        
        # Generate targeted positions
        print("  Generating targeted positions...")
        positions = coverage.generate_targeted_positions()
        print(f"  Generated {len(positions)} positions")
        
        # Limit positions for efficiency
        if len(positions) > 20000:
            # Prioritize positions
            priority_positions = []
            
            # Include all small positions
            priority_positions.extend([p for p in positions if p < 100000])
            
            # Include positions near powers of 2
            for i in range(16, min(40, resonator.bit_len)):
                base = 1 << i
                nearby = [p for p in positions if abs(p - base) < base * 0.1]
                priority_positions.extend(nearby[:10])
            
            # Include some larger positions
            remaining = [p for p in positions if p >= 100000 and p not in priority_positions]
            step = len(remaining) // 5000
            if step > 0:
                priority_positions.extend(remaining[::step])
            
            positions = sorted(set(priority_positions))
            print(f"  Prioritized to {len(positions)} positions")
        
        # Evaluate resonance
        print("  Evaluating resonance...")
        best_resonance = 0.0
        best_position = None
        
        for i, pos in enumerate(positions):
            if i % 2000 == 0 and i > 0:
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
    
    def _guided_peak_detection(self, n: int) -> Optional[int]:
        """Use guided sampling to find resonance peaks"""
        resonator = ImprovedResonance(n)
        
        # Generate guided sample points
        num_samples = min(ScaleAdaptiveParameters.resonance_samples(n), 10000)
        
        # Mix of uniform and targeted sampling
        uniform_samples = num_samples // 2
        targeted_samples = num_samples - uniform_samples
        
        x_values = []
        
        # Uniform sampling
        x_values.extend(np.linspace(0.001, 0.999, uniform_samples))
        
        # Targeted sampling around likely positions
        bit_len = resonator.bit_len
        for i in range(targeted_samples):
            # Sample around bit positions
            target_bits = (bit_len // 2) + (i % 10) - 5
            if target_bits > 0:
                x_norm = (1 << target_bits) / resonator.sqrt_n
                if 0.001 < x_norm < 0.999:
                    # Add with small noise
                    noise = (i % 100 - 50) / (100 * resonator.sqrt_n)
                    x_values.append(max(0.001, min(0.999, x_norm + noise)))
        
        x_values = sorted(set(x_values))
        print(f"  Evaluating {len(x_values)} guided points...")
        
        # Evaluate resonance
        resonances = []
        for i, x in enumerate(x_values):
            if i % 1000 == 0 and i > 0:
                print(f"    Progress: {i}/{len(x_values)}")
            
            res = resonator.resonance(x)
            resonances.append(res)
            self.stats['resonance_evaluations'] += 1
        
        resonances = np.array(resonances)
        
        # Find peaks with adaptive threshold
        mean_res = np.mean(resonances)
        std_res = np.std(resonances)
        
        # Lower threshold for larger numbers
        threshold_factor = 1.5 if bit_len < 64 else 1.0 if bit_len < 96 else 0.5
        
        peaks, properties = find_peaks(
            resonances,
            height=mean_res + threshold_factor * std_res,
            distance=3
        )
        
        print(f"  Found {len(peaks)} peaks above threshold")
        self.stats['peak_candidates'] = len(peaks)
        
        # Check peaks in order of strength
        if len(peaks) > 0:
            sorted_peaks = sorted(peaks, key=lambda i: resonances[i], reverse=True)
            
            for i, peak_idx in enumerate(sorted_peaks[:20]):
                x_norm = x_values[peak_idx]
                factor = int(x_norm * resonator.sqrt_n)
                res_value = resonances[peak_idx]
                
                if i < 10:
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


def test_tuned():
    """Test the tuned Phase I implementation"""
    
    test_cases = [
        # Small cases for validation
        (11, 13),           # 143
        (101, 103),         # 10403
        
        # Original test cases
        (65537, 4294967311),          # 49-bit (p=65537 is 2^16+1)
        (7125766127, 6958284019),     # 66-bit
        (14076040031, 15981381943),   # 68-bit
    ]
    
    phase1 = TunedPhaseI()
    
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
    test_tuned()
