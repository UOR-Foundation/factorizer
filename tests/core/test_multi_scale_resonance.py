"""
Tests for MultiScaleResonance component
"""

import os
import sys
import unittest

from prime_resonance_field.core import MultiScaleResonance


class TestMultiScaleResonance(unittest.TestCase):
    """Test cases for MultiScaleResonance analyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = MultiScaleResonance()

        # Test data: (n, factor, non_factor)
        self.test_cases = [
            (143, 11, 7),  # Small semiprime
            (323, 17, 12),  # Small semiprime
            (10403, 101, 50),  # Medium semiprime
        ]

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.analyzer.phi, (1 + 5**0.5) / 2)
        self.assertEqual(self.analyzer.tau, 1.839286755214161)
        self.assertEqual(len(self.analyzer.scales), 5)
        self.assertIsInstance(self.analyzer.cache, dict)
        self.assertTrue(len(self.analyzer.small_primes) > 0)

    def test_compute_resonance_factors(self):
        """Test that factors have high resonance"""
        for n, factor, _ in self.test_cases:
            with self.subTest(n=n, factor=factor):
                resonance = self.analyzer.compute_resonance(factor, n)
                # Factors should have perfect resonance (1.0)
                self.assertEqual(
                    resonance,
                    1.0,
                    f"Factor {factor} of {n} should have resonance 1.0, got {resonance}",
                )

    def test_compute_resonance_non_factors(self):
        """Test that non-factors have lower resonance"""
        for n, factor, non_factor in self.test_cases:
            with self.subTest(n=n, non_factor=non_factor):
                resonance = self.analyzer.compute_resonance(non_factor, n)
                # Non-factors should have lower resonance
                self.assertLess(
                    resonance,
                    1.0,
                    f"Non-factor {non_factor} of {n} should have resonance < 1.0, got {resonance}",
                )
                self.assertGreaterEqual(
                    resonance, 0.0, f"Resonance should be non-negative, got {resonance}"
                )

    def test_compute_coarse_resonance_factors(self):
        """Test coarse resonance for factors"""
        for n, factor, _ in self.test_cases:
            with self.subTest(n=n, factor=factor):
                resonance = self.analyzer.compute_coarse_resonance(factor, n)
                # Factors should have perfect coarse resonance
                self.assertEqual(
                    resonance,
                    1.0,
                    f"Factor {factor} of {n} should have coarse resonance 1.0, got {resonance}",
                )

    def test_compute_coarse_resonance_performance(self):
        """Test that coarse resonance is faster than full resonance"""
        import time

        n = 10403
        x = 50

        # Time coarse resonance
        start = time.time()
        for _ in range(100):
            self.analyzer.compute_coarse_resonance(x, n)
        coarse_time = time.time() - start

        # Time full resonance
        start = time.time()
        for _ in range(100):
            self.analyzer.compute_resonance(x, n)
        full_time = time.time() - start

        # Coarse should be faster
        self.assertLess(
            coarse_time,
            full_time,
            "Coarse resonance should be faster than full resonance",
        )

    def test_cache_functionality(self):
        """Test that caching works correctly"""
        n, factor = 143, 11

        # Clear cache
        self.analyzer.clear_cache()
        self.assertEqual(len(self.analyzer.cache), 0)

        # First computation
        res1 = self.analyzer.compute_resonance(factor, n)
        self.assertGreater(len(self.analyzer.cache), 0)

        # Second computation should use cache
        res2 = self.analyzer.compute_resonance(factor, n)
        self.assertEqual(res1, res2)

    def test_resonance_bounds(self):
        """Test that resonance values are properly bounded"""
        for n, factor, non_factor in self.test_cases:
            with self.subTest(n=n):
                # Test factor
                res_factor = self.analyzer.compute_resonance(factor, n)
                self.assertGreaterEqual(res_factor, 0.0)
                self.assertLessEqual(res_factor, 1.0)

                # Test non-factor
                res_non_factor = self.analyzer.compute_resonance(non_factor, n)
                self.assertGreaterEqual(res_non_factor, 0.0)
                self.assertLessEqual(res_non_factor, 1.0)

    def test_scale_invariance(self):
        """Test that resonance is relatively scale-invariant"""
        # Test with small and large numbers
        small_n, small_factor = 143, 11
        large_n, large_factor = 10403, 101

        small_res = self.analyzer.compute_resonance(small_factor, small_n)
        large_res = self.analyzer.compute_resonance(large_factor, large_n)

        # Both should be 1.0 (perfect resonance for factors)
        self.assertEqual(small_res, 1.0)
        self.assertEqual(large_res, 1.0)

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        n = 143

        # Test x = 1 (should return 0 or low resonance)
        res = self.analyzer.compute_resonance(1, n)
        self.assertLess(res, 0.5)

        # Test x = n (should return 0 or low resonance)
        res = self.analyzer.compute_resonance(n, n)
        self.assertLess(res, 0.5)

        # Test x > sqrt(n) (should handle gracefully)
        sqrt_n = int(n**0.5)
        res = self.analyzer.compute_resonance(sqrt_n + 1, n)
        self.assertGreaterEqual(res, 0.0)


class TestResonanceComponents(unittest.TestCase):
    """Test individual resonance components"""

    def setUp(self):
        self.analyzer = MultiScaleResonance()
        self.n = 143
        self.factor = 11

    def test_log_unity_resonance(self):
        """Test log unity resonance component"""
        # Factor should have zero log unity resonance (log(1) = 0)
        res = self.analyzer._log_unity_resonance(self.factor, self.n)
        self.assertEqual(res, 0.0)

        # Non-factor should have negative log unity resonance
        non_factor = 7
        res = self.analyzer._log_unity_resonance(non_factor, self.n)
        self.assertLess(res, 0.0)

    def test_log_phase_coherence(self):
        """Test log phase coherence component"""
        res = self.analyzer._log_phase_coherence(self.factor, self.n)
        self.assertIsInstance(res, float)
        self.assertGreater(res, -20)  # Should not be extremely negative

    def test_log_harmonic_convergence(self):
        """Test log harmonic convergence component"""
        res = self.analyzer._log_harmonic_convergence(self.factor, self.n)
        self.assertIsInstance(res, float)
        self.assertGreater(res, -20)  # Should not be extremely negative


if __name__ == "__main__":
    unittest.main()
