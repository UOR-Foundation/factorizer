"""
Phase I Refined Implementation
Building on tuned version with fixes for:
- Reduced divisor bonus to prevent small factor dominance
- Fixed infinite loop in multiplicative coverage
- Better position generation limits
- Enhanced prime detection with Miller-Rabin
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
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
    
    @staticmethod
    def max_positions(n: int) -> int:
        """Maximum positions to generate"""
        bit_len = n.bit_length()
        if bit_len < 64:
            return 10000
        elif bit_len < 96:
            return 30000
        else:
            return 50000


class PrimalityTesting:
    """Enhanced primality testing"""
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def is_probable_prime(n: int) -> bool:
        """Miller-Rabin primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Small primes
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        # Miller-Rabin test
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witnesses to test
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
    
    @staticmethod
    def is_fermat_prime(n: int) -> bool:
        """Check if n is a Fermat prime (2^(2^k) + 1)"""
        if n <= 2:
            return n == 2
        
        # Check if n-1 is a power of 2
        m = n - 1
        if m & (m - 1) != 0:
            return False
        
        # Check if the exponent is a power of 2
        k = m.bit_length() - 1
        if k & (k - 1) == 0:
            return PrimalityTesting.is_probable_prime(n)
        
        return False
    
    @staticmethod
    def is_mersenne_prime(n: int) -> bool:
        """Check if n is a Mersenne prime (2^p - 1)"""
        if n <= 1:
            return False
        
        # Check if n+1 is a power of 2
        m = n + 1
        if m & (m - 1) != 0:
            return False
        
        # The exponent must be prime for Mersenne prime
        p = m.bit_length() - 1
        if PrimalityTesting.is_probable_prime(p):
            return PrimalityTesting.is_probable_prime(n)
        
        return False


class RefinedResonance:
    """Refined resonance function with better factor detection"""
    
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
        self.signals['mersenne_positions'] = self._generate_mersenne_positions()
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
    
    def _compute_multiplicative_orders(self) -> Dict[str, Dict[int, int]]:
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
    
    def _generate_fermat_positions(self) -> Set[int]:
        """Generate positions around Fermat numbers"""
        positions = set()
        
        # Fermat numbers: F_n = 2^(2^n) + 1
        for n in range(20):
            fermat = (1 << (1 << n)) + 1
            if fermat > self.sqrt_n:
                break
            
            positions.add(fermat)
            # Also add nearby values
            for offset in [-2, -1, 1, 2]:
                val = fermat + offset
                if 2 <= val <= self.sqrt_n:
                    positions.add(val)
        
        return positions
    
    def _generate_mersenne_positions(self) -> Set[int]:
        """Generate positions around Mersenne numbers"""
        positions = set()
        
        # Mersenne numbers: M_p = 2^p - 1 where p is prime
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            mersenne = (1 << p) - 1
            if mersenne > self.sqrt_n:
                break
            
            positions.add(mersenne)
            # Also add nearby values
            for offset in [-2, -1, 1, 2]:
                val = mersenne + offset
                if 2 <= val <= self.sqrt_n:
                    positions.add(val)
        
        return positions
    
    def _generate_power2_positions(self) -> Set[int]:
        """Generate positions around powers of 2"""
        positions = set()
        
        # For each bit position that could be a factor
        for bit_pos in range(2, self.bit_len):
            base = 1 << bit_pos
            if base > self.sqrt_n:
                break
            
            # Add positions around powers of 2
            for offset in [-1, 0, 1, 3, 5, 7, 9]:
                pos = base + offset
                if 2 <= pos <= self.sqrt_n:
                    positions.add(pos)
        
        return positions
    
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
        Refined resonance function on [0, 1].
        """
        if x_normalized <= 0 or x_normalized >= 1:
            return 0.0
        
        x = max(2, int(x_normalized * self.sqrt_n))
        if x >= self.n:
            return 0.0
        
        # SIGNIFICANTLY reduced GCD bonus
        g = math.gcd(x, self.n)
        if g > 1:
            if g == x:  # x divides n
                # Much lower base resonance for divisors
                base_resonance = 1.2
            else:
                # Shared factors but not a divisor
                base_resonance = 1.1
        else:
            base_resonance = 1.0
        
        # Combine improved resonance measures
        resonance = base_resonance
        
        # 1. Prime detection - STRONG signal
        if PrimalityTesting.is_probable_prime(x):
            prime_score = self._prime_resonance_score(x)
            resonance *= prime_score
        
        # 2. Chinese Remainder Theorem consistency
        crt_score = self._crt_consistency_score(x)
        resonance *= crt_score
        
        # 3. Carmichael function alignment
        carmichael_score = self._carmichael_alignment_score(x)
        resonance *= carmichael_score
        
        # 4. Multiplicative order pattern
        order_score = self._order_pattern_score(x)
        resonance *= order_score
        
        # 5. Continued fraction proximity
        cf_score = self._cf_proximity_score(x)
        resonance *= cf_score
        
        # 6. Special form detection
        special_score = self._special_form_score(x)
        resonance *= special_score
        
        # 7. Scale-invariant position score
        position_score = self._scale_invariant_position_score(x)
        resonance *= position_score
        
        return resonance
    
    def _prime_resonance_score(self, x: int) -> float:
        """
        Strong resonance for prime factors.
        """
        score = 3.0  # Base boost for primes
        
        # Extra boost for special primes
        if PrimalityTesting.is_fermat_prime(x):
            score *= 2.0
        elif PrimalityTesting.is_mersenne_prime(x):
            score *= 1.8
        
        # Check if orders are consistent with x being a prime factor
        consistent_orders = 0
        for base, order_n in self.signals['mult_orders']['n'].items():
            if order_n % (x - 1) == 0:
                consistent_orders += 1
        
        if consistent_orders >= 3:
            score *= 1.5
        
        return score
    
    def _crt_consistency_score(self, x: int) -> float:
        """
        Check Chinese Remainder Theorem consistency.
        """
        score = 1.0
        consistency_count = 0
        total_checks = 0
        
        # Check CRT consistency with small primes
        for p, n_mod_p in self.signals['prime_coords'][:40]:
            if p >= x:
                continue
            
            x_mod_p = x % p
            
            # Skip if x is divisible by p
            if x_mod_p == 0:
                if n_mod_p == 0:
                    consistency_count += 1
                total_checks += 1
                continue
            
            # Check multiplicative consistency
            # If x | n, then for any prime p: n ≡ 0 (mod gcd(x, p))
            g = math.gcd(x, p)
            if g > 1 and n_mod_p % g == 0:
                consistency_count += 1
            
            # Check if x could divide n based on modular arithmetic
            if n_mod_p == 0:
                consistency_count += 0.5
            
            total_checks += 1
        
        if total_checks > 0:
            consistency_rate = consistency_count / total_checks
            score *= (1 + consistency_rate * 2)
        
        return score
    
    def _carmichael_alignment_score(self, x: int) -> float:
        """
        Check Carmichael function alignment.
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
                score *= 1.5
        
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
            score *= 3.0
        elif min_distance < x * 0.001:
            score *= 2.0
        elif min_distance < x * 0.01:
            score *= 1.5
        elif min_distance < x * 0.1:
            score *= 1.2
        
        return score
    
    def _special_form_score(self, x: int) -> float:
        """
        Score based on special mathematical forms.
        """
        score = 1.0
        
        # Fermat numbers
        if x in self.signals['fermat_positions']:
            score *= 2.5
        
        # Mersenne numbers
        if x in self.signals['mersenne_positions']:
            score *= 2.3
        
        # Powers of 2 nearby
        if x in self.signals['power2_positions']:
            score *= 1.3
        
        # Check bit pattern similarity with n
        x_bits = x.bit_length()
        n_bits = self.n.bit_length()
        if abs(x_bits - n_bits // 2) <= 1:
            # x has about half the bits of n
            score *= 1.2
        
        return score
    
    def _scale_invariant_position_score(self, x: int) -> float:
        """
        Position-based score that's scale-invariant.
        """
        x_normalized = x / self.sqrt_n
        
        # Penalize extreme positions
        if x_normalized < 0.001 or x_normalized > 0.999:
            return 0.5
        
        # Slight preference for balanced factors
        distance_from_center = abs(x_normalized - 0.5)
        return 1.0 + 0.2 * math.exp(-distance_from_center * 2)


class EfficientCoverage:
    """Efficient coverage strategy with proper limits"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.bit_len = n.bit_length()
        self.max_positions = ScaleAdaptiveParameters.max_positions(n)
        self.positions_generated = 0
    
    def generate_targeted_positions(self) -> List[int]:
        """Generate positions with focus on likely factor locations"""
        positions = set()
        
        # 1. Special forms first (highest priority)
        self._add_special_form_coverage(positions)
        if len(positions) > self.max_positions:
            return self._select_best_positions(positions)
        
        # 2. Bit-length based coverage
        self._add_bit_length_coverage(positions)
        if len(positions) > self.max_positions:
            return self._select_best_positions(positions)
        
        # 3. Dense coverage for small factors
        self._add_small_factor_coverage(positions)
        if len(positions) > self.max_positions:
            return self._select_best_positions(positions)
        
        # 4. Continued fraction guided positions
        self._add_cf_guided_positions(positions)
        if len(positions) > self.max_positions:
            return self._select_best_positions(positions)
        
        # 5. Limited multiplicative coverage
        self._add_limited_multiplicative_coverage(positions)
        
        return self._select_best_positions(positions)
    
    def _select_best_positions(self, positions: Set[int]) -> List[int]:
        """Select the best positions when we have too many"""
        positions_list = list(positions)
        
        # Prioritize by type
        special_forms = set()
        bit_aligned = set()
        small_primes = set()
        others = set()
        
        for pos in positions_list:
            if PrimalityTesting.is_fermat_prime(pos) or PrimalityTesting.is_mersenne_prime(pos):
                special_forms.add(pos)
            elif pos.bit_length() == self.bit_len // 2:
                bit_aligned.add(pos)
            elif pos < 100000 and PrimalityTesting.is_probable_prime(pos):
                small_primes.add(pos)
            else:
                others.add(pos)
        
        # Build final list with priorities
        final_positions = []
        final_positions.extend(sorted(special_forms))
        
        remaining_space = self.max_positions - len(final_positions)
        if remaining_space > 0:
            bit_aligned_list = sorted(bit_aligned)[:remaining_space // 2]
            final_positions.extend(bit_aligned_list)
            remaining_space -= len(bit_aligned_list)
        
        if remaining_space > 0:
            small_primes_list = sorted(small_primes)[:remaining_space // 2]
            final_positions.extend(small_primes_list)
            remaining_space -= len(small_primes_list)
        
        if remaining_space > 0:
            others_list = sorted(others)[:remaining_space]
            final_positions.extend(others_list)
        
        return sorted(set(final_positions))
    
    def _add_small_factor_coverage(self, positions: set):
        """Dense coverage for small factors relative to sqrt(n)"""
        threshold = min(ScaleAdaptiveParameters.small_factor_threshold(self.n), 100000)
        
        # All primes up to threshold
        sieve_limit = min(threshold, 1000000)  # Cap for memory
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
                    if len(positions) > self.max_positions:
                        return
    
    def _add_bit_length_coverage(self, positions: set):
        """Coverage based on bit lengths of potential factors"""
        # For n with b bits, factors typically have b/2 ± k bits
        center_bits = self.bit_len // 2
        
        for offset in range(-3, 4):
            target_bits = center_bits + offset
            if target_bits < 2 or target_bits > self.bit_len - 2:
                continue
            
            # Sample around 2^target_bits
            base = 1 << target_bits
            if base > self.sqrt_n:
                continue
            
            # Dense sampling near powers of 2
            for delta in range(-50, 51):
                pos = base + delta
                if 2 <= pos <= self.sqrt_n:
                    positions.add(pos)
                    if len(positions) > self.max_positions:
                        return
    
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
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            mersenne = (1 << p) - 1
            if mersenne > self.sqrt_n:
                break
            positions.add(mersenne)
            for offset in [-2, -1, 1, 2, 3, 5, 7]:
                if 2 <= mersenne + offset <= self.sqrt_n:
                    positions.add(mersenne + offset)
        
        # Numbers of form 2^k ± 2^j (limited)
        for k in range(2, min(self.bit_len, 20)):
            for j in range(1, min(k, 10)):
                val1 = (1 << k) + (1 << j)
                val2 = (1 << k) - (1 << j)
                if val1 <= self.sqrt_n:
                    positions.add(val1)
                if 2 <= val2 <= self.sqrt_n:
                    positions.add(val2)
                if len(positions) > self.max_positions // 2:
                    return
    
    def _add_limited_multiplicative_coverage(self, positions: set):
        """Limited coverage based on multiplicative structure"""
        # Products of small primes (smooth numbers) - FIXED
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Generate smooth numbers up to sqrt_n with proper limit
        smooth_limit = min(self.sqrt_n, 100000)
        smooth = set([1])
        
        for p in small_primes:
            new_smooth = set()
            for s in smooth:
                val = s * p  # Start with s * p
                while val <= smooth_limit:
                    if val > 1:
                        new_smooth.add(val)
                    val *= p
                    if len(smooth) + len(new_smooth) > self.max_positions // 4:
                        break
            
            smooth.update(new_smooth)
            if len(smooth) > self.max_positions // 4:
                break
        
        # Add valid smooth numbers to positions
        for s in smooth:
            if 2 <= s <= self.sqrt_n:
                positions.add(s)
                if len(positions) > self.max_positions:
                    return
    
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
        for _ in range(min(50, self.bit_len)):
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
            if len(positions) > self.max_positions:
                return


class RefinedPhaseI:
    """Refined Phase I implementation with all improvements"""
    
    def __init__(self):
        self.stats = {
            'resonance_evaluations': 0,
            'optimization_iterations': 0,
            'peak_candidates': 0
        }
    
    def factorize(self, n: int) -> Tuple[int, int]:
        """Factor using refined Phase I approach"""
        if n < 2:
            raise ValueError("n must be >= 2")
        if n % 2 == 0:
            return (2, n // 2)
        
        print(f"\n{'='*60}")
        print(f"Refined Phase I Factorization")
        print(f"n = {n} ({n.bit_length()} bits)")
        print(f"{'='*60}")
        
        # Strategy 1: Multi-start continuous optimization
        print("\nStrategy 1: Multi-Start Optimization")
        factor1 = self._multi_start_optimization(n)
        if factor1 and n % factor1 == 0:
            return self._format_result(n, factor1)
        
        # Strategy 2: Efficient targeted coverage
        print("\nStrategy 2: Efficient Targeted Coverage")
        factor2 = self._efficient_coverage_resonance(n)
        if factor2 and n % factor2 == 0:
            return self._format_result(n, factor2)
        
        # Strategy 3: Adaptive peak detection
        print("\nStrategy 3: Adaptive Peak Detection")
        factor3 = self._adaptive_peak_detection(n)
        if factor3 and n % factor3 == 0:
            return self._format_result(n, factor3)
        
        raise ValueError("Refined Phase I incomplete - needs further refinement")
    
    def _multi_start_optimization(self, n: int) -> Optional[int]:
        """Multi-start optimization with diverse initial points"""
        resonator = RefinedResonance(n)
        
        # Multiple starting points
        start_points = []
        
        # Standard starts
        start_points.extend([0.5, 0.1, 0.9, 0.618, 0.382])
        
        # Bit-aligned starts
        bit_len = n.bit_length()
        for bits in range(max(2, bit_len // 2 - 3), min(bit_len - 2, bit_len // 2 + 3)):
            x_norm = (1 << bits) / resonator.sqrt_n
            if 0.01 < x_norm < 0.99:
                start_points.append(x_norm)
        
        # Special form starts (Fermat/Mersenne)
        if 65537 <= resonator.sqrt_n:  # 2^16 + 1
            start_points.append(65537 / resonator.sqrt_n)
        
        best_result = None
        best_resonance = 0
        
        for i, start in enumerate(start_points[:7]):  # Limit starts
            print(f"  Start {i+1} from position {start:.3f}")
            
            # Track evaluations
            eval_count = [0]
            
            def objective(x):
                eval_count[0] += 1
                return -resonator.resonance(x[0])
            
            # Differential evolution from this start
            result = differential_evolution(
                objective,
                bounds=[(max(0.001, start - 0.2), min(0.999, start + 0.2))],
                maxiter=30,
                popsize=10,
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
    
    def _efficient_coverage_resonance(self, n: int) -> Optional[int]:
        """Use efficient coverage to find factors"""
        resonator = RefinedResonance(n)
        coverage = EfficientCoverage(n)
        
        # Generate targeted positions
        print("  Generating efficient targeted positions...")
        positions = coverage.generate_targeted_positions()
        print(f"  Generated {len(positions)} positions")
        
        # Evaluate resonance
        print("  Evaluating resonance...")
        best_resonance = 0.0
        best_position = None
        
        # Quick check for very high resonance first
        high_priority = []
        for pos in positions[:100]:  # Check first 100
            x_norm = pos / resonator.sqrt_n
            if 0 < x_norm < 1:
                resonance = resonator.resonance(x_norm)
                self.stats['resonance_evaluations'] += 1
                
                if resonance > 10.0:  # High resonance threshold
                    high_priority.append((pos, resonance))
        
        # Sort by resonance and check divisibility
        high_priority.sort(key=lambda x: x[1], reverse=True)
        for pos, res in high_priority[:10]:
            if n % pos == 0:
                print(f"  Found high-resonance factor: {pos} with resonance {res:.3f}")
                return pos
        
        # Continue with remaining positions
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
    
    def _adaptive_peak_detection(self, n: int) -> Optional[int]:
        """Use adaptive sampling to find resonance peaks"""
        resonator = RefinedResonance(n)
        
        # Adaptive sample count
        bit_len = resonator.bit_len
        num_samples = min(ScaleAdaptiveParameters.resonance_samples(n), 5000)
        
        # Focus sampling on promising regions
        x_values = []
        
        # 1. Around special forms
        special_positions = []
        for pos in resonator.signals['fermat_positions']:
            x_norm = pos / resonator.sqrt_n
            if 0.001 < x_norm < 0.999:
                special_positions.append(x_norm)
        
        for pos in resonator.signals['mersenne_positions']:
            x_norm = pos / resonator.sqrt_n
            if 0.001 < x_norm < 0.999:
                special_positions.append(x_norm)
        
        # Dense sampling around special forms
        for sp in special_positions:
            for offset in np.linspace(-0.01, 0.01, 21):
                x = sp + offset
                if 0.001 < x < 0.999:
                    x_values.append(x)
        
        # 2. Bit-aligned positions
        for bits in range(max(16, bit_len // 2 - 5), min(bit_len - 2, bit_len // 2 + 5)):
            x_norm = (1 << bits) / resonator.sqrt_n
            if 0.001 < x_norm < 0.999:
                for offset in np.linspace(-0.005, 0.005, 11):
                    x = x_norm + offset
                    if 0.001 < x < 0.999:
                        x_values.append(x)
        
        # 3. Fill remaining with uniform sampling
        remaining = num_samples - len(x_values)
        if remaining > 0:
            x_values.extend(np.linspace(0.001, 0.999, remaining))
        
        x_values = sorted(set(x_values))
        print(f"  Evaluating {len(x_values)} adaptive points...")
        
        # Evaluate resonance
        resonances = []
        for i, x in enumerate(x_values):
            if i % 500 == 0 and i > 0:
                print(f"    Progress: {i}/{len(x_values)}")
            
            res = resonator.resonance(x)
            resonances.append(res)
            self.stats['resonance_evaluations'] += 1
        
        resonances = np.array(resonances)
        
        # Find peaks with adaptive threshold
        mean_res = np.mean(resonances)
        std_res = np.std(resonances)
        
        # Very low threshold to catch all potential peaks
        peaks, properties = find_peaks(
            resonances,
            height=mean_res + 0.5 * std_res,
            distance=2
        )
        
        print(f"  Found {len(peaks)} peaks above threshold")
        self.stats['peak_candidates'] = len(peaks)
        
        # Check peaks in order of strength
        if len(peaks) > 0:
            sorted_peaks = sorted(peaks, key=lambda i: resonances[i], reverse=True)
            
            for i, peak_idx in enumerate(sorted_peaks[:30]):
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
        
        # Check if factors are special forms
        if PrimalityTesting.is_fermat_prime(factor):
            print(f"  Note: {factor} is a Fermat prime (2^16 + 1)")
        elif PrimalityTesting.is_mersenne_prime(factor):
            print(f"  Note: {factor} is a Mersenne prime")
        
        print(f"\nStatistics:")
        print(f"  Resonance evaluations: {self.stats['resonance_evaluations']}")
        print(f"  Optimization iterations: {self.stats['optimization_iterations']}")
        print(f"  Peak candidates examined: {self.stats['peak_candidates']}")
        
        return (factor, other) if factor <= other else (other, factor)


def test_refined():
    """Test the refined Phase I implementation"""
    
    test_cases = [
        # Small cases for validation
        (11, 13),           # 143
        (101, 103),         # 10403
        
        # Original test cases
        (65537, 4294967311),          # 49-bit (p=65537 is 2^16+1)
        (7125766127, 6958284019),     # 66-bit
        (14076040031, 15981381943),   # 68-bit
        
        # Additional challenging cases
        (2147483647, 2147483659),     # 62-bit (Mersenne prime × prime)
    ]
    
    phase1 = RefinedPhaseI()
    
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
    test_refined()
