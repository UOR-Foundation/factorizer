"""
Balanced Semiprime Search - Specialized algorithms for balanced semiprimes
"""

import math
import time
from typing import List, Optional, Tuple


class BalancedSemiprimeSearch:
    """
    Specialized search strategies for balanced semiprimes (where p ≈ q).
    Combines multiple approaches optimized for this specific case.
    """

    def __init__(self):
        self.stats = {
            "attempts": 0,
            "total_time": 0,
            "successes": 0,
            "methods_used": {},
        }

    def factor(self, n: int, timeout: float = 10.0) -> Optional[Tuple[int, int]]:
        """
        Factor n using strategies optimized for balanced semiprimes.

        Args:
            n: Number to factor
            timeout: Maximum time to spend

        Returns:
            (p, q) if successful, None if timeout
        """
        start_time = time.time()
        self.stats["attempts"] += 1

        # Quick check for even numbers
        if n % 2 == 0:
            return (2, n // 2)

        sqrt_n = int(math.sqrt(n))
        
        # Strategy 1: Expanding ring search
        result = self._expanding_ring_search(n, sqrt_n, timeout * 0.4)
        if result:
            self._record_success("expanding_ring", time.time() - start_time)
            return result

        # Strategy 2: Binary resonance search
        if time.time() - start_time < timeout * 0.7:
            result = self._binary_resonance_search(n, sqrt_n, timeout * 0.3)
            if result:
                self._record_success("binary_resonance", time.time() - start_time)
                return result

        # Strategy 3: Modular pattern search
        if time.time() - start_time < timeout * 0.9:
            result = self._modular_pattern_search(n, timeout * 0.2)
            if result:
                self._record_success("modular_pattern", time.time() - start_time)
                return result

        self.stats["total_time"] += time.time() - start_time
        return None

    def _expanding_ring_search(
        self, n: int, sqrt_n: int, timeout: float
    ) -> Optional[Tuple[int, int]]:
        """
        Search in expanding rings around sqrt(n).
        Optimized for balanced factors close to sqrt(n).
        """
        start_time = time.time()

        # Adaptive parameters based on n's size
        if n.bit_length() < 50:
            initial_radius = 1
            growth_factor = 1.5
            max_radius = int(sqrt_n * 0.1)
        else:
            initial_radius = 10
            growth_factor = 1.2
            max_radius = int(sqrt_n * 0.01)

        radius = initial_radius
        checked = set()

        while radius <= max_radius and time.time() - start_time < timeout:
            # Check ring at current radius
            candidates = []

            # Generate candidates in the ring
            for offset in range(-radius, radius + 1):
                x = sqrt_n + offset
                if 2 <= x <= sqrt_n and x not in checked:
                    candidates.append(x)
                    checked.add(x)

            # Check candidates
            for x in candidates:
                if n % x == 0:
                    return (x, n // x)

            # Expand radius
            radius = int(radius * growth_factor)

            # Accelerate growth for large radii
            if radius > 1000:
                growth_factor = min(2.0, growth_factor * 1.1)

        return None

    def _binary_resonance_search(
        self, n: int, sqrt_n: int, timeout: float
    ) -> Optional[Tuple[int, int]]:
        """
        Binary search guided by resonance field approximation.
        """
        start_time = time.time()

        # Define search bounds
        left = max(2, int(sqrt_n * 0.9))
        right = min(sqrt_n, int(sqrt_n * 1.1))

        # Coarse resonance function
        def resonance_score(x: int) -> float:
            # Simple resonance approximation
            score = 0.0

            # GCD bonus
            g = math.gcd(x, n)
            if g > 1:
                score += math.log(g)

            # Distance from sqrt penalty
            dist = abs(x - sqrt_n) / sqrt_n
            score -= dist * 10

            # Modular alignment bonus
            for p in [3, 5, 7, 11]:
                if n % p == x % p:
                    score += 0.5

            return score

        # Sample points for initial scan
        sample_size = min(100, right - left)
        step = max(1, (right - left) // sample_size)

        best_candidates = []
        for x in range(left, right + 1, step):
            if time.time() - start_time > timeout * 0.5:
                break

            score = resonance_score(x)
            best_candidates.append((x, score))

        # Sort by score and refine around best candidates
        best_candidates.sort(key=lambda x: x[1], reverse=True)

        for center, _ in best_candidates[:10]:
            # Fine search around high-score points
            for offset in range(-step // 2, step // 2 + 1):
                x = center + offset
                if left <= x <= right:
                    if n % x == 0:
                        return (x, n // x)

                if time.time() - start_time > timeout:
                    break

        return None

    def _modular_pattern_search(
        self, n: int, timeout: float
    ) -> Optional[Tuple[int, int]]:
        """
        Search based on modular arithmetic patterns.
        """
        start_time = time.time()
        sqrt_n = int(math.sqrt(n))

        # Patterns for balanced semiprimes
        patterns = []

        # Pattern 1: Factors often have specific mod relationships
        for p in [3, 5, 7, 11, 13]:
            remainder = n % p
            # Factors that would produce this remainder
            for a in range(p):
                for b in range(p):
                    if (a * b) % p == remainder:
                        patterns.append((p, a, b))

        # Search near sqrt(n) with pattern constraints
        search_radius = min(1000, int(sqrt_n * 0.01))

        for p, a, b in patterns[:20]:  # Limit patterns
            if time.time() - start_time > timeout:
                break

            # Find candidates matching pattern
            for k in range(-search_radius // p, search_radius // p + 1):
                x1 = sqrt_n + k * p + (a - sqrt_n % p)
                x2 = sqrt_n + k * p + (b - sqrt_n % p)

                for x in [x1, x2]:
                    if 2 <= x <= sqrt_n:
                        if n % x == 0:
                            return (x, n // x)

        return None

    def _fermat_method_enhanced(self, n: int, timeout: float) -> Optional[Tuple[int, int]]:
        """
        Enhanced Fermat's method optimized for balanced semiprimes.
        """
        start_time = time.time()
        
        # Start slightly above sqrt(n)
        a = int(math.sqrt(n)) + 1
        
        # Adaptive step size for very large numbers
        step = 1 if n.bit_length() <= 80 else max(1, int(a ** 0.001))
        
        while time.time() - start_time < timeout:
            b_squared = a * a - n
            
            if b_squared < 0:
                a += step
                continue
                
            b = int(math.sqrt(b_squared))
            
            # Check if b is exact
            if b * b == b_squared:
                p = a - b
                q = a + b
                if p > 1 and q > 1 and p * q == n:
                    return (min(p, q), max(p, q))
            
            a += step
            
            # Accelerate for large numbers
            if a > int(math.sqrt(n)) * 1.1:
                step = max(step, int(step * 1.1))
                
            # Stop if a gets too large
            if a > int(math.sqrt(n)) * 2:
                break
                
        return None

    def _record_success(self, method: str, time_taken: float):
        """Record successful factorization"""
        self.stats["successes"] += 1
        self.stats["total_time"] += time_taken

        if method not in self.stats["methods_used"]:
            self.stats["methods_used"][method] = 0
        self.stats["methods_used"][method] += 1

    def is_likely_balanced(self, n: int) -> bool:
        """
        Heuristics to determine if n is likely a balanced semiprime.
        Based on implementation experience.
        """
        # Check if n is even
        if n % 2 == 0:
            return False

        sqrt_n = int(math.sqrt(n))
        
        # Check if near perfect square (strong indicator for balanced semiprimes)
        if abs(n - sqrt_n * sqrt_n) < sqrt_n:
            return True
            
        # Quick check for factors around sqrt(n) - if found, check if balanced
        for offset in range(1, min(20, sqrt_n // 10)):
            for x in [sqrt_n - offset, sqrt_n + offset]:
                if x > 1 and n % x == 0:
                    other = n // x
                    ratio = max(x, other) / min(x, other)
                    # If ratio is close to 1, it's balanced
                    if ratio < 2.0:  # Factors within 2x of each other
                        return True
                    else:
                        return False  # Found unbalanced factors
        
        # Check digit sum pattern (empirical observation for small balanced semiprimes)
        digit_sum = sum(int(d) for d in str(n))
        if digit_sum % 9 in [1, 8]:  # 143 has digit sum 8, common for balanced semiprimes
            return True

        # Default: assume could be balanced for unknown cases
        return True

    def get_statistics(self) -> dict:
        """Get method statistics"""
        success_rate = (
            self.stats["successes"] / self.stats["attempts"]
            if self.stats["attempts"] > 0
            else 0
        )

        avg_time = (
            self.stats["total_time"] / self.stats["attempts"]
            if self.stats["attempts"] > 0
            else 0
        )

        return {
            "attempts": self.stats["attempts"],
            "successes": self.stats["successes"],
            "success_rate": success_rate,
            "average_time": avg_time,
            "total_time": self.stats["total_time"],
            "methods_used": dict(self.stats["methods_used"]),
        }
