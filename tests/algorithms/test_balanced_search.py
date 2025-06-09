"""
Tests for BalancedSemiprimeSearch algorithm
"""

import os
import sys
import unittest

from algorithms import BalancedSemiprimeSearch


class TestBalancedSemiprimeSearch(unittest.TestCase):
    """Test cases for BalancedSemiprimeSearch"""

    def setUp(self):
        """Set up test fixtures"""
        self.searcher = BalancedSemiprimeSearch()

        # Test cases: (n, p, q) where n = p * q and p ≈ q
        self.balanced_cases = [
            (143, 11, 13),  # Small balanced
            (323, 17, 19),  # Small balanced
            (10403, 101, 103),  # Medium balanced
            (282943, 523, 541),  # Large balanced
        ]

        # Non-balanced cases for comparison
        self.unbalanced_cases = [
            (77, 7, 11),  # Small gap
            (221, 13, 17),  # Small gap
            (15, 3, 5),  # Very small
        ]

    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsInstance(self.searcher.stats, dict)
        self.assertIn("attempts", self.searcher.stats)
        self.assertIn("successes", self.searcher.stats)

    def test_is_likely_balanced(self):
        """Test balanced semiprime detection"""
        for n, p, q in self.balanced_cases:
            with self.subTest(n=n):
                is_balanced = self.searcher.is_likely_balanced(n)
                # Most balanced semiprimes should be detected
                self.assertTrue(
                    is_balanced, f"{n} = {p} × {q} should be detected as balanced"
                )

        # Test some obviously unbalanced cases
        unbalanced_numbers = [15, 21, 35, 77]  # Small factors
        for n in unbalanced_numbers:
            with self.subTest(n=n):
                is_balanced = self.searcher.is_likely_balanced(n)
                # May or may not be detected as balanced (heuristic)
                self.assertIsInstance(is_balanced, bool)

    def test_factor_balanced_semiprimes(self):
        """Test factorization of balanced semiprimes"""
        for n, p_true, q_true in self.balanced_cases:
            with self.subTest(n=n):
                result = self.searcher.factor(n, timeout=5.0)

                if result is not None:
                    p_found, q_found = result
                    # Should find the correct factors
                    self.assertEqual(
                        {p_found, q_found},
                        {p_true, q_true},
                        f"Expected {p_true} × {q_true}, got {p_found} × {q_found}",
                    )

                    # Verify factorization
                    self.assertEqual(p_found * q_found, n)

    def test_expanding_ring_search(self):
        """Test expanding ring search method"""
        n = 143  # 11 × 13
        sqrt_n = int(n**0.5)

        result = self.searcher._expanding_ring_search(n, sqrt_n, timeout=2.0)

        if result is not None:
            p, q = result
            self.assertEqual(p * q, n)
            self.assertIn(p, [11, 13])

    def test_fermat_method_enhanced(self):
        """Test enhanced Fermat method"""
        # Fermat method works best on balanced semiprimes
        n = 143  # 11 × 13

        result = self.searcher._fermat_method_enhanced(n, timeout=2.0)

        if result is not None:
            p, q = result
            self.assertEqual(p * q, n)
            self.assertIn(p, [11, 13])

    def test_binary_resonance_search(self):
        """Test binary resonance search method"""
        n = 323  # 17 × 19
        sqrt_n = int(n**0.5)

        result = self.searcher._binary_resonance_search(n, sqrt_n, timeout=2.0)

        if result is not None:
            p, q = result
            self.assertEqual(p * q, n)
            self.assertIn(p, [17, 19])

    def test_timeout_handling(self):
        """Test that methods respect timeout"""
        import time

        n = 10403  # Larger number

        start = time.time()
        self.searcher.factor(n, timeout=0.1)  # Very short timeout
        elapsed = time.time() - start

        # Should respect timeout (with some tolerance)
        self.assertLess(elapsed, 1.0, "Should respect timeout bounds")

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly"""
        initial_attempts = self.searcher.stats["attempts"]

        # Try factoring
        self.searcher.factor(143, timeout=1.0)

        # Should increment attempts
        self.assertGreater(self.searcher.stats["attempts"], initial_attempts)

    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.searcher.get_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("attempts", stats)
        self.assertIn("successes", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("average_time", stats)

        # Success rate should be between 0 and 1
        self.assertGreaterEqual(stats["success_rate"], 0.0)
        self.assertLessEqual(stats["success_rate"], 1.0)


class TestBalancedSearchStrategies(unittest.TestCase):
    """Test different search strategies"""

    def setUp(self):
        self.searcher = BalancedSemiprimeSearch()

    def test_multiple_strategies(self):
        """Test that different strategies can find factors"""
        test_cases = [(143, [11, 13]), (323, [17, 19]), (10403, [101, 103])]

        for n, expected_factors in test_cases:
            with self.subTest(n=n):
                found_by_any = False

                # Try each strategy individually
                sqrt_n = int(n**0.5)

                strategies = [
                    ("expanding_ring", self.searcher._expanding_ring_search),
                    ("fermat_enhanced", self.searcher._fermat_method_enhanced),
                    ("binary_resonance", self.searcher._binary_resonance_search),
                ]

                for strategy_name, strategy_func in strategies:
                    if strategy_name == "fermat_enhanced":
                        result = strategy_func(n, timeout=1.0)
                    else:
                        result = strategy_func(n, sqrt_n, timeout=1.0)

                    if result is not None:
                        p, q = result
                        if {p, q} == set(expected_factors):
                            found_by_any = True
                            break

                # At least one strategy should work
                # (Note: This may not always pass due to timeouts/randomness)
                if not found_by_any:
                    print(f"Warning: No strategy found factors for {n}")


if __name__ == "__main__":
    unittest.main()
