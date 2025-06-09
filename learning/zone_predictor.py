"""
Zone Predictor - Predicts high-resonance zones based on features
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ZonePredictor:
    """
    Predicts high-resonance zones based on learned features.
    Uses simple online learning to adapt predictions based on factorization outcomes.
    """

    def __init__(self):
        self.weights = defaultdict(float)
        self.training_data: List[Tuple[np.ndarray, bool]] = []
        self.feature_stats = {"mean": None, "std": None}
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_velocities = defaultdict(float)

    def add_training_example(self, features: np.ndarray, is_factor_zone: bool):
        """Add a training example"""
        self.training_data.append((features, is_factor_zone))

        # Update feature statistics
        self._update_feature_stats()

        # Simple online learning update every 10 examples
        if len(self.training_data) % 10 == 0:
            self.train()

    def train(self):
        """Train the predictor using recent examples"""
        if not self.training_data:
            return

        # Use recent examples (last 100)
        recent_data = self.training_data[-100:]

        # Mini-batch gradient descent
        for features, is_factor in recent_data:
            # Normalize features
            normalized = self._normalize_features(features)

            # Compute prediction
            prediction = self._compute_score(normalized)

            # Compute error
            target = 1.0 if is_factor else 0.0
            error = target - prediction

            # Update weights with momentum
            for i, f in enumerate(normalized):
                gradient = (
                    error * f * prediction * (1 - prediction)
                )  # Sigmoid derivative

                # Momentum update
                self.weight_velocities[i] = (
                    self.momentum * self.weight_velocities[i]
                    + self.learning_rate * gradient
                )
                self.weights[i] += self.weight_velocities[i]

        # Regularization - decay weights slightly
        for i in self.weights:
            self.weights[i] *= 0.999

    def predict(
        self, features: np.ndarray, sqrt_n: int
    ) -> List[Tuple[int, int, float]]:
        """Predict zones likely to contain factors"""
        # Normalize features
        normalized = self._normalize_features(features)

        # Simple linear prediction
        score = self._compute_score(normalized)
        confidence = 1 / (1 + math.exp(-score))  # Sigmoid

        zones = []

        # High confidence - multiple focused zones
        if confidence > 0.7:
            # Zone 1: Near sqrt(n) (balanced factors)
            width = int(sqrt_n * 0.02 * confidence)
            zones.append(
                (max(2, sqrt_n - width), min(sqrt_n, sqrt_n + width), confidence)
            )

            # Zone 2: Golden ratio point
            phi = (1 + math.sqrt(5)) / 2
            golden_point = int(sqrt_n / phi)
            width = int(golden_point * 0.05 * confidence)
            zones.append(
                (max(2, golden_point - width), golden_point + width, confidence * 0.8)
            )

        # Medium confidence - broader search
        elif confidence > 0.5:
            # Wider zone around sqrt(n)
            width = int(sqrt_n * 0.05 * confidence)
            zones.append(
                (max(2, sqrt_n - width), min(sqrt_n, sqrt_n + width), confidence)
            )

        # Low confidence - suggest different regions
        else:
            # Small primes region
            zones.append((2, min(1000, sqrt_n), confidence + 0.2))

            # Power of 2 region
            log2_n = int(math.log2(sqrt_n))
            for i in range(max(1, log2_n - 2), log2_n + 1):
                center = 2**i
                if center <= sqrt_n:
                    zones.append(
                        (
                            max(2, center - 10),
                            min(sqrt_n, center + 10),
                            confidence + 0.1,
                        )
                    )

        return zones

    def _compute_score(self, features: np.ndarray) -> float:
        """Compute raw score from features"""
        score = 0.0
        for i, f in enumerate(features):
            score += self.weights[i] * f
        return score

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using stored statistics"""
        if self.feature_stats["mean"] is None:
            return features

        normalized = (features - self.feature_stats["mean"]) / (
            self.feature_stats["std"] + 1e-8
        )
        return normalized

    def _update_feature_stats(self):
        """Update feature statistics for normalization"""
        if len(self.training_data) < 10:
            return

        # Extract all features
        all_features = np.array([f for f, _ in self.training_data[-100:]])

        self.feature_stats["mean"] = np.mean(all_features, axis=0)
        self.feature_stats["std"] = np.std(all_features, axis=0)

    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance scores"""
        importance = {}
        for i, weight in self.weights.items():
            importance[i] = abs(weight)
        return importance

    def reset(self):
        """Reset the predictor"""
        self.weights.clear()
        self.weight_velocities.clear()
        self.training_data.clear()
        self.feature_stats = {"mean": None, "std": None}
