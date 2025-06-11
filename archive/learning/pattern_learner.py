"""
Resonance Pattern Learner - Learns successful resonance patterns for acceleration
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .zone_predictor import ZonePredictor


class ResonancePatternLearner:
    """
    Learns successful resonance patterns from factorization attempts.
    Maintains a database of patterns and transition boundaries to accelerate future factorizations.
    """

    def __init__(self):
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []

        # Known transition boundaries from research
        self.transition_boundaries: Dict[Tuple[int, int], int] = {
            (2, 3): 282281,
            (3, 5): 2961841,
            (5, 7): 53596041,
            (7, 11): 1522756281,
        }

        self.feature_weights = defaultdict(float)
        self.zone_predictor = ZonePredictor()

        # Statistics for adaptive learning
        self.total_factorizations = 0
        self.successful_predictions = 0

    def record_success(self, n: int, factor: int, resonance_profile: Dict[str, float]):
        """Record successful factorization patterns"""
        self.total_factorizations += 1

        pattern = self._extract_pattern(n, factor, resonance_profile)
        self.success_patterns.append(pattern)

        # Limit pattern history
        if len(self.success_patterns) > 1000:
            self.success_patterns = self.success_patterns[-1000:]

        # Update transition boundaries if near known boundary
        self._update_transition_boundaries(n, factor)

        # Update feature weights based on success
        self._update_feature_weights(pattern, success=True)

        # Train zone predictor with successful zone
        features = self._extract_features(n)
        sqrt_n = int(math.sqrt(n))

        # Was the factor in a "balanced zone"?
        is_balanced_zone = abs(factor - sqrt_n) < sqrt_n * 0.1
        self.zone_predictor.add_training_example(features, is_balanced_zone)

        # Retrain models periodically
        if len(self.success_patterns) % 10 == 0:
            self._retrain_models()

    def record_failure(
        self, n: int, tested_positions: List[int], resonances: List[float]
    ):
        """Record failed attempts for negative learning"""
        # Keep only high-resonance failures
        high_res_failures = [
            (x, res) for x, res in zip(tested_positions, resonances) if res > 0.3
        ][
            :10
        ]  # Top 10 high-resonance failures

        for x, res in high_res_failures:
            pattern = self._extract_pattern(n, x, {"resonance": res})
            self.failure_patterns.append(pattern)
            self._update_feature_weights(pattern, success=False)

        # Limit failure history
        if len(self.failure_patterns) > 500:
            self.failure_patterns = self.failure_patterns[-500:]

    def predict_high_resonance_zones(self, n: int) -> List[Tuple[int, int, float]]:
        """Predict zones likely to contain factors based on learned patterns"""
        features = self._extract_features(n)
        zones = []

        sqrt_n = int(math.sqrt(n))

        # 1. Transition boundary predictions
        for (b1, b2), boundary in self.transition_boundaries.items():
            if boundary * 0.1 <= n <= boundary * 10:
                center = int(math.sqrt(boundary))
                width = int(center * 0.1)
                confidence = self._compute_boundary_confidence(n, boundary)
                zones.append(
                    (max(2, center - width), min(sqrt_n, center + width), confidence)
                )

        # 2. Feature-based predictions from zone predictor
        feature_zones = self.zone_predictor.predict(features, sqrt_n)
        zones.extend(feature_zones)

        # 3. Pattern-based predictions from similar successful factorizations
        pattern_zones = self._apply_learned_patterns(n)
        zones.extend(pattern_zones)

        # 4. Special zones for balanced semiprimes
        if self._is_likely_balanced_semiprime(n):
            # High-priority zone right around sqrt(n)
            width = max(10, int(sqrt_n**0.01))
            zones.append(
                (
                    max(2, sqrt_n - width),
                    min(sqrt_n, sqrt_n + width),
                    0.9,  # High confidence
                )
            )

        # Merge overlapping zones and sort by confidence
        merged_zones = self._merge_zones(zones)
        merged_zones.sort(key=lambda z: z[2], reverse=True)

        return merged_zones[:10]  # Top 10 zones

    def _extract_pattern(
        self, n: int, x: int, resonance_profile: Dict[str, float]
    ) -> Dict[str, Any]:
        """Extract pattern features from a factorization attempt"""
        sqrt_n = int(math.sqrt(n))

        pattern = {
            "n_bits": n.bit_length(),
            "x_bits": x.bit_length(),
            "relative_position": x / sqrt_n if sqrt_n > 0 else 0,
            "is_prime_x": self._is_probable_prime(x),
            "gcd_score": math.gcd(x, n) / x,
            "mod_profile": [n % p for p in [2, 3, 5, 7, 11]],
            "resonance": resonance_profile.get("resonance", 0),
            "near_square": self._near_perfect_square(x),
            "near_power": self._near_prime_power(x),
            "distance_from_sqrt": abs(x - sqrt_n) / sqrt_n if sqrt_n > 0 else 1,
            "is_balanced": abs(x - sqrt_n) < sqrt_n * 0.1,
        }

        return pattern

    def _extract_features(self, n: int) -> np.ndarray:
        """Extract feature vector for n"""
        features = []

        # Basic features
        features.append(n.bit_length() / 256)  # Normalized bit length
        features.append(math.log(n) / 100)  # Log scale

        # Modular features
        for p in [2, 3, 5, 7, 11]:
            features.append((n % p) / p)

        # Digit sum features
        digit_sum = sum(int(d) for d in str(n))
        features.append(digit_sum / (9 * len(str(n))))

        # Binary pattern features
        binary = bin(n)[2:]
        features.append(binary.count("1") / len(binary))  # Bit density

        # Check if n is odd (likely balanced semiprime)
        features.append(1.0 if n % 2 == 1 else 0.0)

        # Check divisibility by small primes
        small_prime_divisors = sum(1 for p in [2, 3, 5, 7] if n % p == 0)
        features.append(small_prime_divisors / 4)

        return np.array(features)

    def _update_transition_boundaries(self, n: int, factor: int):
        """Update transition boundaries based on discovered patterns"""
        sqrt_n = int(math.sqrt(n))

        # Check if this factorization reveals a new transition pattern
        for (b1, b2), boundary in self.transition_boundaries.items():
            if abs(sqrt_n - int(math.sqrt(boundary))) < sqrt_n * 0.01:
                # Near a known transition - refine the boundary
                self.transition_boundaries[(b1, b2)] = int((boundary + n) / 2)

    def _update_feature_weights(self, pattern: Dict[str, Any], success: bool):
        """Update feature weights based on pattern success/failure"""
        weight = 1.0 if success else -0.1

        # Update weights for pattern features
        for key, value in pattern.items():
            if isinstance(value, (int, float)):
                self.feature_weights[key] += weight * value
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    self.feature_weights[f"{key}_{i}"] += weight * v

    def _retrain_models(self):
        """Retrain all models with accumulated patterns"""
        if not self.success_patterns:
            return

        # Zone predictor is already being trained incrementally
        # Here we could add additional model retraining if needed
        pass

    def _compute_boundary_confidence(self, n: int, boundary: int) -> float:
        """Compute confidence for a transition boundary"""
        distance = abs(math.log(n) - math.log(boundary))
        return math.exp(-distance)

    def _apply_learned_patterns(self, n: int) -> List[Tuple[int, int, float]]:
        """Apply learned patterns to predict zones"""
        zones = []
        sqrt_n = int(math.sqrt(n))

        # Find similar successful patterns
        n_bits = n.bit_length()
        similar_patterns = [
            p
            for p in self.success_patterns[-100:]  # Recent patterns
            if abs(p["n_bits"] - n_bits) <= 2
        ]

        # Generate zones based on similar patterns
        for pattern in similar_patterns[:5]:  # Top 5 similar
            if pattern["relative_position"] > 0:
                center = int(pattern["relative_position"] * sqrt_n)

                # Adaptive width based on pattern characteristics
                if pattern["is_balanced"]:
                    width = int(sqrt_n * 0.01)  # Narrow for balanced
                else:
                    width = int(sqrt_n * 0.02)

                confidence = pattern["resonance"] * 0.8  # Decay factor

                if 2 <= center - width and center + width <= sqrt_n:
                    zones.append((center - width, center + width, confidence))

        return zones

    def _merge_zones(
        self, zones: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
        """Merge overlapping zones"""
        if not zones:
            return []

        # Sort by start position
        zones.sort(key=lambda z: z[0])

        merged = []
        current_start, current_end, current_conf = zones[0]

        for start, end, conf in zones[1:]:
            if start <= current_end:
                # Overlapping - merge
                current_end = max(current_end, end)
                current_conf = max(current_conf, conf)
            else:
                # Non-overlapping - save current and start new
                merged.append((current_start, current_end, current_conf))
                current_start, current_end, current_conf = start, end, conf

        merged.append((current_start, current_end, current_conf))
        return merged

    def _is_probable_prime(self, n: int) -> bool:
        """Simple primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if n == p:
                return True
            if n % p == 0:
                return False

        return True  # Probably prime for small values

    def _near_perfect_square(self, x: int) -> float:
        """Score how close x is to a perfect square"""
        sqrt_x = int(math.sqrt(x))
        if sqrt_x * sqrt_x == x:
            return 1.0

        dist = min(x - sqrt_x**2, (sqrt_x + 1) ** 2 - x)
        return math.exp(-dist / max(x, 1))

    def _near_prime_power(self, x: int) -> float:
        """Score how close x is to a prime power"""
        for p in [2, 3, 5, 7, 11]:
            if x >= p:
                k = int(math.log(x) / math.log(p))
                if k >= 1:
                    dist = abs(x - p**k)
                    if dist < x * 0.01:
                        return math.exp(-dist / x)
        return 0.0

    def _is_likely_balanced_semiprime(self, n: int) -> bool:
        """Check if n is likely a balanced semiprime"""
        # Several heuristics from implementation experience

        # 1. n is odd (most balanced semiprimes are)
        if n % 2 == 0:
            return False

        # 2. Not divisible by small primes
        for p in [3, 5, 7]:
            if n % p == 0:
                cofactor = n // p
                sqrt_n = int(math.sqrt(n))
                # If cofactor is close to sqrt(n), might still be balanced
                if abs(cofactor - sqrt_n) < sqrt_n * 0.1:
                    return True
                else:
                    return False

        # 3. Close to a perfect square
        sqrt_n = int(math.sqrt(n))
        if abs(n - sqrt_n * sqrt_n) < n * 0.01:
            return True

        # Default: assume it could be balanced
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        success_rate = (
            self.successful_predictions / self.total_factorizations
            if self.total_factorizations > 0
            else 0
        )

        return {
            "total_patterns": len(self.success_patterns),
            "failure_patterns": len(self.failure_patterns),
            "transition_boundaries": len(self.transition_boundaries),
            "feature_weights": len(self.feature_weights),
            "success_rate": success_rate,
            "total_factorizations": self.total_factorizations,
        }

    def reset(self):
        """Reset all learned patterns"""
        self.success_patterns.clear()
        self.failure_patterns.clear()
        self.feature_weights.clear()
        self.zone_predictor.reset()
        self.total_factorizations = 0
        self.successful_predictions = 0
