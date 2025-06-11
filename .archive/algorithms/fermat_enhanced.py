"""
Enhanced Fermat's Method - Optimized for balanced semiprimes
"""

import math
import time
from typing import Optional, Tuple


class FermatMethodEnhanced:
    """
    Enhanced version of Fermat's method specifically optimized for balanced semiprimes.
    Includes adaptive step sizes and early termination strategies.
    """

    def __init__(self):
        self.stats = {"attempts": 0, "total_time": 0, "successes": 0}

    def factor(self, n: int, timeout: float = 10.0) -> Optional[Tuple[int, int]]:
        """
        Factor n using enhanced Fermat's method.
        Best for semiprimes where p ≈ q (balanced factors).

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

        # Start from ceiling of sqrt(n)
        a = int(math.sqrt(n))
        if a * a < n:
            a += 1

        # Adaptive step size based on n's size
        if n.bit_length() <= 50:
            step = 1
            max_iterations = 100000
        elif n.bit_length() <= 80:
            # For larger numbers, use adaptive stepping
            step = 1
            max_iterations = 1000000
        else:
            # Very large numbers - start with larger steps
            step = max(1, int(a**0.001))
            max_iterations = 10000000

        iterations = 0

        # Main loop with adaptive stepping
        while time.time() - start_time < timeout and iterations < max_iterations:
            iterations += 1

            # Check if a² - n is a perfect square
            b2 = a * a - n

            if b2 >= 0:
                b = int(math.sqrt(b2))
                if b * b == b2:
                    # Found factors!
                    factor1 = a - b
                    factor2 = a + b

                    if factor1 > 1 and factor2 > 1:
                        self.stats["successes"] += 1
                        self.stats["total_time"] += time.time() - start_time
                        return (min(factor1, factor2), max(factor1, factor2))

                # Adaptive step size increase for large numbers
                if iterations > 1000 and n.bit_length() > 80:
                    # If we've been searching for a while, increase step size
                    if iterations % 10000 == 0:
                        step = min(step * 2, int(a**0.01))

            a += step

            # Progress check for very large numbers
            if iterations % 100000 == 0 and n.bit_length() > 100:
                # Check if we're making reasonable progress
                progress = (a - int(math.sqrt(n))) / int(math.sqrt(n))
                if progress > 0.1:  # More than 10% away from start
                    # Probably not balanced, abort
                    break

        self.stats["total_time"] += time.time() - start_time
        return None

    def factor_with_hint(
        self, n: int, balance_ratio: float = 0.1, timeout: float = 10.0
    ) -> Optional[Tuple[int, int]]:
        """
        Factor with hint about balance ratio.

        Args:
            n: Number to factor
            balance_ratio: Expected |p-q|/sqrt(n) ratio (0.1 = within 10%)
            timeout: Maximum time

        Returns:
            (p, q) if successful, None if timeout
        """
        if balance_ratio > 0.5:
            # Not balanced enough for Fermat's method
            return None

        # Adjust search parameters based on balance hint
        sqrt_n = int(math.sqrt(n))
        search_radius = int(sqrt_n * balance_ratio)

        # Use bounded search
        return self._bounded_fermat(n, search_radius, timeout)

    def _bounded_fermat(
        self, n: int, max_distance: int, timeout: float
    ) -> Optional[Tuple[int, int]]:
        """
        Fermat's method with bounded search distance from sqrt(n).
        """
        start_time = time.time()

        a = int(math.sqrt(n))
        if a * a < n:
            a += 1

        a_max = a + max_distance

        while a <= a_max and time.time() - start_time < timeout:
            b2 = a * a - n

            if b2 >= 0:
                b = int(math.sqrt(b2))
                if b * b == b2:
                    factor1 = a - b
                    factor2 = a + b

                    if factor1 > 1 and factor2 > 1:
                        return (min(factor1, factor2), max(factor1, factor2))

            a += 1

        return None

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
        }
