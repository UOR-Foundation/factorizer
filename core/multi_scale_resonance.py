"""
Multi-Scale Resonance Analyzer - Analyzes resonance at multiple scales simultaneously
"""

import math
from functools import lru_cache
from typing import Dict, Optional, Tuple


class MultiScaleResonance:
    """
    Analyzes resonance at multiple scales simultaneously.
    Implements scale-invariant resonance computation in log space for numerical stability.
    """

    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.tau = 1.839286755214161  # Tribonacci constant
        self.scales = [1, self.phi, self.phi**2, self.tau, self.tau**2]
        self.cache: Dict[Tuple[int, int], float] = {}
        self.small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def compute_resonance(self, x: int, n: int) -> float:
        """Compute full scale-invariant resonance in log space"""
        # Check cache
        cache_key = (x, n)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Edge cases
        if x <= 1 or x >= n:
            self.cache[cache_key] = 0.0
            return 0.0

        # Quick check for exact divisibility
        if n % x == 0:
            self.cache[cache_key] = 1.0
            return 1.0

        # Compute resonance at multiple scales
        log_resonances = []
        sqrt_n = int(math.sqrt(n))

        for scale in self.scales:
            scaled_x = int(x * scale)
            if 2 <= scaled_x <= sqrt_n:
                # Compute in log space for stability
                log_u = self._log_unity_resonance(scaled_x, n)
                log_p = self._log_phase_coherence(scaled_x, n)
                log_h = self._log_harmonic_convergence(scaled_x, n)

                # Weight by scale (closer to 1 = higher weight)
                scale_weight = 1.0 / (1 + abs(math.log(scale)))

                log_resonances.append((log_u + log_p + log_h, scale_weight))

        # Aggregate using weighted log-sum-exp for numerical stability
        if log_resonances:
            max_log = max(lr[0] for lr in log_resonances)
            weighted_sum = sum(w * math.exp(lr - max_log) for lr, w in log_resonances)
            total_weight = sum(w for _, w in log_resonances)
            result = max_log + math.log(weighted_sum / total_weight)
        else:
            result = -float("inf")

        # Convert back from log space
        resonance = math.exp(result) if result > -100 else 0.0

        # Apply nonlinearity to sharpen peaks
        if resonance > 0.5:
            resonance = resonance ** (1 / self.phi)

        # Cache only if the cache isn't too large
        if len(self.cache) < 100000:
            self.cache[cache_key] = resonance

        return resonance

    def compute_coarse_resonance(self, x: int, n: int) -> float:
        """Fast approximation for importance sampling"""
        # Quick checks
        if n % x == 0:
            return 1.0

        # Prime harmonic indicator
        score = 0.0
        for p in self.small_primes[:5]:  # Use only first 5 primes
            if x % p == 0:
                score += 0.1
            if n % p == 0 and x % p == 0:
                score += 0.2

        # GCD bonus
        g = math.gcd(x, n)
        if g > 1:
            score += math.log(g) / math.log(n)

        # Distance to perfect square
        sqrt_x = int(math.sqrt(x))
        if sqrt_x * sqrt_x == x:
            score += 0.3

        # Near sqrt(n) bonus for balanced factors - CRITICAL
        sqrt_n = int(math.sqrt(n))
        relative_distance = abs(x - sqrt_n) / sqrt_n
        if relative_distance < 0.1:
            score += 0.4 * (1 - relative_distance / 0.1)
        elif relative_distance < 0.01:
            score += 0.6 * (1 - relative_distance / 0.01)

        return min(1.0, score)

    def _log_unity_resonance(self, x: int, n: int) -> float:
        """Compute log unity resonance"""
        if n % x == 0:
            return 0.0  # log(1) = 0

        # Frequency-based resonance
        omega_n = 2 * math.pi / math.log(n + 1)
        omega_x = 2 * math.pi / math.log(x + 1)

        # Find nearest harmonic
        k = round(omega_n / omega_x) if omega_x > 0 else 1
        phase_diff = abs(omega_n - k * omega_x)

        # Gaussian in log space
        sigma_sq = math.log(n) / (2 * math.pi)
        log_gaussian = -(phase_diff**2) / (2 * sigma_sq)

        # Harmonic series contribution
        harmonic_sum = sum(
            1 / k for k in range(1, min(10, int(math.sqrt(x)) + 1)) if n % (x * k) < k
        )
        log_harmonic = math.log(1 + harmonic_sum / math.log(x + 2))

        return log_gaussian + log_harmonic

    def _log_phase_coherence(self, x: int, n: int) -> float:
        """Compute log phase coherence"""
        weighted_sum = 0.0
        total_weight = 0.0

        for p in self.small_primes[:7]:
            phase_n = 2 * math.pi * (n % p) / p
            phase_x = 2 * math.pi * (x % p) / p
            coherence = (1 + math.cos(phase_n - phase_x)) / 2
            weight = 1 / math.log(p + 1)

            weighted_sum += coherence * weight
            total_weight += weight

        base_coherence = weighted_sum / total_weight if total_weight > 0 else 0.5

        # GCD amplification
        g = math.gcd(x, n)
        if g > 1:
            amplification = 1 + math.log(g) / math.log(n)
            base_coherence = min(1.0, base_coherence * amplification)

        return math.log(base_coherence) if base_coherence > 0 else -10

    def _log_harmonic_convergence(self, x: int, n: int) -> float:
        """Compute log harmonic convergence"""
        convergence_points = []

        # Unity harmonic
        g = math.gcd(x, n)
        unity_freq = 2 * math.pi / g if g > 0 else 2 * math.pi
        unity_harmonic = (1 + math.cos(unity_freq * math.log(n) / (2 * math.pi))) / 2
        convergence_points.append(unity_harmonic)

        # Golden ratio convergence
        phi_harmonic = x / self.phi
        phi_distance = min(
            abs(phi_harmonic - int(phi_harmonic)),
            abs(phi_harmonic - int(phi_harmonic) - 1),
        )
        phi_convergence = math.exp(-phi_distance * self.phi)
        convergence_points.append(phi_convergence)

        # Tribonacci resonance
        if x > 2:
            tri_phase = math.log(x) / math.log(self.tau)
            tri_resonance = abs(math.sin(tri_phase * math.pi))
            convergence_points.append(tri_resonance)

        # Perfect square resonance
        sqrt_x = int(math.sqrt(x))
        if sqrt_x * sqrt_x == x:
            convergence_points.append(1.0)
        else:
            square_dist = min(x - sqrt_x**2, (sqrt_x + 1) ** 2 - x)
            square_harmony = math.exp(-square_dist / x)
            convergence_points.append(square_harmony)

        # Harmonic mean in log space
        if convergence_points and all(c > 0 for c in convergence_points):
            log_hm = math.log(len(convergence_points)) - math.log(
                sum(1 / (c + 0.001) for c in convergence_points)
            )
            return log_hm
        else:
            return -10

    def clear_cache(self):
        """Clear the resonance cache"""
        self.cache.clear()
