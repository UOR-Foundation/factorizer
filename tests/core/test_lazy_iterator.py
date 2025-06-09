"""
Tests for LazyResonanceIterator component
"""

import os
import sys
import unittest

from prime_resonance_field.core import LazyResonanceIterator, MultiScaleResonance


class TestLazyResonanceIterator(unittest.TestCase):
    """Test cases for LazyResonanceIterator"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = MultiScaleResonance()
        self.n = 143  # 11 × 13
        self.factor = 11
        self.iterator = LazyResonanceIterator(self.n, self.analyzer)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.iterator.n, self.n)
        self.assertEqual(self.iterator.sqrt_n, int(self.n**0.5))
        self.assertIsNotNone(self.iterator.analyzer)
        self.assertGreater(len(self.iterator.importance_heap), 0)
        self.assertGreater(len(self.iterator.visited), 0)

    def test_iterator_protocol(self):
        """Test that iterator protocol works"""
        iterator = iter(self.iterator)
        self.assertIsNotNone(iterator)

        # Should be able to get first few values
        values = []
        for i, x in enumerate(iterator):
            values.append(x)
            if i >= 10:  # Just get first 10
                break

        self.assertEqual(len(values), 11)
        self.assertTrue(all(isinstance(x, int) for x in values))
        self.assertTrue(all(2 <= x <= self.iterator.sqrt_n for x in values))

    def test_factor_discovery(self):
        """Test that the iterator eventually finds factors"""
        found_factor = False

        for i, x in enumerate(self.iterator):
            if x == self.factor or x == self.n // self.factor:
                found_factor = True
                break
            if i >= 1000:  # Limit iterations for test
                break

        self.assertTrue(found_factor, "Iterator should eventually yield the factor")

    def test_seed_initialization(self):
        """Test that seeds are properly initialized"""
        # Check that important positions are seeded
        sqrt_n = self.iterator.sqrt_n

        # Should include positions near sqrt(n)
        near_sqrt_found = False
        for x, _ in list(self.iterator.visited)[:100]:  # Check first 100 visited
            if abs(x - sqrt_n) <= 10:
                near_sqrt_found = True
                break

        self.assertTrue(near_sqrt_found, "Should seed positions near sqrt(n)")

    def test_importance_based_ordering(self):
        """Test that high-importance nodes come first"""
        first_values = []
        for i, x in enumerate(self.iterator):
            first_values.append(x)
            if i >= 20:
                break

        # Check that factors appear early (high importance)
        factor_position = None
        for i, x in enumerate(first_values):
            if x == self.factor:
                factor_position = i
                break

        if factor_position is not None:
            self.assertLess(
                factor_position,
                15,
                "Factor should appear early in iteration (high importance)",
            )


class TestGradientEstimation(unittest.TestCase):
    """Test gradient estimation functionality"""

    def setUp(self):
        self.analyzer = MultiScaleResonance()
        self.n = 143
        self.iterator = LazyResonanceIterator(self.n, self.analyzer)

    def test_gradient_estimation(self):
        """Test gradient estimation method"""
        x = 10
        gradient = self.iterator._estimate_gradient(x)

        self.assertIsInstance(gradient, float)
        self.assertFalse(abs(gradient) > 1000, "Gradient should not be extremely large")

    def test_boundary_gradient_handling(self):
        """Test gradient estimation at boundaries"""
        # Test near lower boundary
        gradient_low = self.iterator._estimate_gradient(3)
        self.assertIsInstance(gradient_low, float)

        # Test near upper boundary
        sqrt_n = int(self.n**0.5)
        gradient_high = self.iterator._estimate_gradient(sqrt_n - 1)
        self.assertIsInstance(gradient_high, float)


if __name__ == "__main__":
    unittest.main()
