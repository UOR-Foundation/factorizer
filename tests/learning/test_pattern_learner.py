"""
Tests for ResonancePatternLearner component
"""

import os
import sys
import unittest

from prime_resonance_field.learning import ResonancePatternLearner


class TestResonancePatternLearner(unittest.TestCase):
    """Test cases for ResonancePatternLearner"""

    def setUp(self):
        """Set up test fixtures"""
        self.learner = ResonancePatternLearner()

    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsInstance(self.learner.success_patterns, list)
        self.assertIsInstance(self.learner.failure_patterns, list)
        self.assertIsInstance(self.learner.transition_boundaries, dict)
        self.assertGreater(len(self.learner.transition_boundaries), 0)

    def test_record_success(self):
        """Test recording successful factorizations"""
        n = 143
        factor = 11
        resonance_profile = {"resonance": 0.8}

        initial_count = len(self.learner.success_patterns)
        self.learner.record_success(n, factor, resonance_profile)

        self.assertEqual(len(self.learner.success_patterns), initial_count + 1)

        # Check pattern content
        pattern = self.learner.success_patterns[-1]
        self.assertIn("n_bits", pattern)
        self.assertIn("relative_position", pattern)
        self.assertIn("resonance", pattern)
        self.assertEqual(pattern["n_bits"], n.bit_length())

    def test_record_failure(self):
        """Test recording failed attempts"""
        n = 143
        tested_positions = [7, 9, 12]
        resonances = [0.1, 0.2, 0.15]

        initial_count = len(self.learner.failure_patterns)
        self.learner.record_failure(n, tested_positions, resonances)

        # Should record patterns for failed positions
        self.assertGreaterEqual(len(self.learner.failure_patterns), initial_count)

    def test_predict_high_resonance_zones(self):
        """Test zone prediction functionality"""
        # First record some successful patterns
        test_cases = [(143, 11), (323, 17), (1147, 31)]

        for n, factor in test_cases:
            self.learner.record_success(n, factor, {"resonance": 0.9})

        # Now predict zones for a similar number
        n_test = 10403  # 101 × 103
        zones = self.learner.predict_high_resonance_zones(n_test)

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

    def test_pattern_extraction(self):
        """Test pattern extraction from factorization"""
        n = 143
        x = 11
        resonance_profile = {"resonance": 0.8}

        pattern = self.learner._extract_pattern(n, x, resonance_profile)

        self.assertIn("n_bits", pattern)
        self.assertIn("x_bits", pattern)
        self.assertIn("relative_position", pattern)
        self.assertIn("resonance", pattern)
        self.assertIn("gcd_score", pattern)

        self.assertEqual(pattern["n_bits"], n.bit_length())
        self.assertEqual(pattern["x_bits"], x.bit_length())
        self.assertEqual(pattern["resonance"], 0.8)

    def test_feature_extraction(self):
        """Test feature extraction for learning"""
        n = 143
        features = self.learner._extract_features(n)

        self.assertIsInstance(features, object)  # numpy array
        self.assertGreater(len(features), 0)

        # Features should be normalized (between 0 and 1)
        for feature in features:
            self.assertGreaterEqual(feature, 0.0)
            self.assertLessEqual(feature, 1.0)


class TestTransitionBoundaries(unittest.TestCase):
    """Test transition boundary functionality"""

    def setUp(self):
        self.learner = ResonancePatternLearner()

    def test_initial_boundaries(self):
        """Test that initial boundaries are set"""
        boundaries = self.learner.transition_boundaries

        # Should have some predefined boundaries
        self.assertIn((2, 3), boundaries)
        self.assertIn((3, 5), boundaries)

        # Boundary values should be positive integers
        for boundary_value in boundaries.values():
            self.assertIsInstance(boundary_value, int)
            self.assertGreater(boundary_value, 0)

    def test_boundary_confidence(self):
        """Test boundary confidence computation"""
        n = 300000  # Near (2,3) boundary
        boundary = self.learner.transition_boundaries[(2, 3)]

        confidence = self.learner._compute_boundary_confidence(n, boundary)

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
