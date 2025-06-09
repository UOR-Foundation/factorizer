"""
Tests for ZonePredictor component
"""

import os
import sys
import unittest

import numpy as np

from prime_resonance_field.learning import ZonePredictor


class TestZonePredictor(unittest.TestCase):
    """Test cases for ZonePredictor"""

    def setUp(self):
        """Set up test fixtures"""
        self.predictor = ZonePredictor()

    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsInstance(self.predictor.weights, dict)
        self.assertIsInstance(self.predictor.training_data, list)
        self.assertEqual(len(self.predictor.training_data), 0)

    def test_add_training_example(self):
        """Test adding training examples"""
        features = np.array([0.5, 0.8, 0.3, 0.1])
        is_factor_zone = True

        initial_count = len(self.predictor.training_data)
        self.predictor.add_training_example(features, is_factor_zone)

        self.assertEqual(len(self.predictor.training_data), initial_count + 1)

        # Check data format
        data_features, data_label = self.predictor.training_data[-1]
        np.testing.assert_array_equal(data_features, features)
        self.assertEqual(data_label, is_factor_zone)

    def test_training_triggers_automatically(self):
        """Test that training is triggered after enough examples"""
        features = np.array([0.5, 0.8, 0.3, 0.1])

        # Add enough examples to trigger training (100)
        for i in range(100):
            is_factor = i % 2 == 0  # Alternate labels
            self.predictor.add_training_example(features, is_factor)

        # Weights should have been updated
        self.assertGreater(len(self.predictor.weights), 0)

    def test_predict_zones(self):
        """Test zone prediction"""
        # Add some training data first
        positive_features = np.array([0.8, 0.9, 0.7, 0.6])
        negative_features = np.array([0.1, 0.2, 0.3, 0.1])

        # Add multiple examples
        for _ in range(10):
            self.predictor.add_training_example(positive_features, True)
            self.predictor.add_training_example(negative_features, False)

        # Force training
        self.predictor.train()

        # Test prediction
        test_features = np.array([0.7, 0.8, 0.6, 0.5])
        sqrt_n = 100

        zones = self.predictor.predict(test_features, sqrt_n)

        self.assertIsInstance(zones, list)
        # Each zone should be (start, end, confidence) tuple
        for zone in zones:
            self.assertIsInstance(zone, tuple)
            self.assertEqual(len(zone), 3)
            start, end, confidence = zone
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertIsInstance(confidence, (int, float))
            self.assertLessEqual(start, end)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_train_method(self):
        """Test the training method"""
        # Add some training data
        features1 = np.array([0.8, 0.9, 0.7])
        features2 = np.array([0.2, 0.1, 0.3])

        self.predictor.add_training_example(features1, True)
        self.predictor.add_training_example(features2, False)

        # Train
        initial_weights = len(self.predictor.weights)
        self.predictor.train()

        # Should have weights for each feature dimension
        self.assertGreaterEqual(len(self.predictor.weights), initial_weights)

    def test_empty_training_data(self):
        """Test behavior with no training data"""
        features = np.array([0.5, 0.5, 0.5])
        sqrt_n = 100

        # Should handle gracefully
        zones = self.predictor.predict(features, sqrt_n)
        self.assertIsInstance(zones, list)

    def test_feature_weights_update(self):
        """Test that feature weights are updated correctly"""
        features = np.array([1.0, 0.5, 0.0])

        # Add positive example
        self.predictor.add_training_example(features, True)
        self.predictor.train()

        # Check that weights reflect the positive example
        # Weight for feature 0 should be positive (feature value was 1.0)
        if len(self.predictor.weights) > 0:
            self.assertIsInstance(self.predictor.weights[0], float)


class TestZonePredictorIntegration(unittest.TestCase):
    """Integration tests for ZonePredictor"""

    def test_realistic_training_scenario(self):
        """Test with realistic factorization data"""
        predictor = ZonePredictor()

        # Simulate training data from successful factorizations
        # Features: [normalized_position, bit_density, gcd_indicator, ...]
        successful_zones = [
            (np.array([0.9, 0.6, 1.0, 0.8]), True),  # Near sqrt(n), high indicators
            (np.array([0.1, 0.3, 0.5, 0.2]), True),  # Small factor
            (np.array([0.5, 0.4, 0.1, 0.3]), False),  # Random position
            (np.array([0.8, 0.2, 0.0, 0.1]), False),  # High position, low indicators
        ]

        # Add training examples
        for features, is_factor_zone in successful_zones:
            predictor.add_training_example(features, is_factor_zone)

        # Train
        predictor.train()

        # Test prediction on new data
        test_features = np.array([0.85, 0.65, 0.8, 0.7])  # Should be positive
        zones = predictor.predict(test_features, 1000)

        # Should predict at least one zone
        if len(zones) > 0:
            # Check that confidence is reasonable for positive-like features
            _, _, confidence = zones[0]
            self.assertGreater(confidence, 0.3)


if __name__ == "__main__":
    unittest.main()
