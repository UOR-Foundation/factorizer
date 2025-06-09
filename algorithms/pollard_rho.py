"""
Optimized Pollard's Rho Algorithm - For general factorization
"""

import math
import time
from typing import List, Optional, Tuple


class PollardRhoOptimized:
    """
    Optimized version of Pollard's Rho algorithm with multiple polynomials
    and improved cycle detection.
    """

    def __init__(self):
        self.stats = {
            "attempts": 0,
            "total_time": 0,
            "successes": 0,
            "total_iterations": 0,
        }

        # Different polynomials to try
        self.polynomials = [
            lambda x, n, c: (x * x + c) % n,
            lambda x, n, c: (x * x - c) % n,
            lambda x, n, c: (x * x * x + c) % n,
        ]

    def factor(self, n: int, timeout: float = 30.0) -> Optional[Tuple[int, int]]:
        """
        Factor n using Pollard's Rho with multiple strategies.

        Args:
            n: Number to factor
            timeout: Maximum time to spend

        Returns:
            (p, q) if successful, None if timeout
        """
        if n % 2 == 0:
            return (2, n // 2)

        start_time = time.time()
        self.stats["attempts"] += 1

        # Try different c values and polynomials
        c_values = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]

        for poly_func in self.polynomials:
            for c in c_values:
                if time.time() - start_time > timeout:
                    break

                result = self._pollard_rho_single(
                    n, c, poly_func, timeout - (time.time() - start_time)
                )
                if result:
                    self.stats["successes"] += 1
                    self.stats["total_time"] += time.time() - start_time
                    return result

        self.stats["total_time"] += time.time() - start_time
        return None

    def _pollard_rho_single(
        self, n: int, c: int, poly_func, timeout: float
    ) -> Optional[Tuple[int, int]]:
        """
        Single run of Pollard's Rho with given parameters.
        """
        start_time = time.time()

        # Initialize with different starting points based on c
        x = (2 + c) % n
        y = x
        d = 1

        # Adaptive iteration limit based on n's size
        max_iterations = min(n, 1 << 20)  # Cap at 2^20
        iterations = 0

        # Brent's improvement: only compute GCD periodically
        saved = 0

        while d == 1 and iterations < max_iterations:
            if time.time() - start_time > timeout:
                break

            # Use power of 2 cycle lengths (Brent's optimization)
            if iterations > 0 and (iterations & (iterations - 1)) == 0:
                saved = x

            x = poly_func(x, n, c)
            diff = abs(x - saved)
            d = math.gcd(diff, n)

            iterations += 1
            self.stats["total_iterations"] += 1

            # Check if we found a non-trivial factor
            if 1 < d < n:
                factor1 = d
                factor2 = n // d
                return (min(factor1, factor2), max(factor1, factor2))

            # If d == n, we need to backtrack
            if d == n:
                # Try with smaller steps
                x = saved
                y = saved
                for _ in range(10):
                    x = poly_func(x, n, c)
                    y = poly_func(poly_func(y, n, c), n, c)
                    d = math.gcd(abs(x - y), n)
                    if 1 < d < n:
                        factor1 = d
                        factor2 = n // d
                        return (min(factor1, factor2), max(factor1, factor2))

        return None

    def factor_with_hint(
        self, n: int, small_factor_likely: bool = False, timeout: float = 30.0
    ) -> Optional[Tuple[int, int]]:
        """
        Factor with hint about expected factor sizes.

        Args:
            n: Number to factor
            small_factor_likely: If True, check small factors first
            timeout: Maximum time

        Returns:
            (p, q) if successful, None if timeout
        """
        if small_factor_likely:
            # Quick trial division for small factors
            small_primes = self._generate_small_primes(10000)
            for p in small_primes:
                if n % p == 0:
                    return (p, n // p)

        # Regular Pollard's Rho
        return self.factor(n, timeout)

    def _generate_small_primes(self, limit: int) -> List[int]:
        """Generate primes up to limit using sieve."""
        if limit < 2:
            return []

        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False

        return [i for i in range(2, limit + 1) if sieve[i]]

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

        avg_iterations = (
            self.stats["total_iterations"] / self.stats["attempts"]
            if self.stats["attempts"] > 0
            else 0
        )

        return {
            "attempts": self.stats["attempts"],
            "successes": self.stats["successes"],
            "success_rate": success_rate,
            "average_time": avg_time,
            "total_time": self.stats["total_time"],
            "average_iterations": avg_iterations,
            "total_iterations": self.stats["total_iterations"],
        }
