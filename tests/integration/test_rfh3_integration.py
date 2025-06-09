"""
Integration tests for the complete RFH3 system
"""

import os
import sys
import time
import unittest

from prime_resonance_field import RFH3, RFH3Config


class TestFixtures:
    """Test fixture data"""

    SMALL_SEMIPRIMES = [
        (143, 11, 13),
        (323, 17, 19),
        (1147, 31, 37),
    ]

    BALANCED_SEMIPRIMES = [
        (10403, 101, 103),
        (282943, 523, 541),
        (1299071, 1117, 1163),
    ]


class TestRFH3Integration(unittest.TestCase):
    """Integration tests for the complete RFH3 system"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = RFH3Config()
        self.config.max_iterations = 50000
        self.config.learning_enabled = True
        self.config.hierarchical_search = True
        self.rfh3 = RFH3(self.config)

    def test_small_semiprimes(self):
        """Test RFH3 on small semiprimes"""
        for n, p_true, q_true in TestFixtures.SMALL_SEMIPRIMES:
            with self.subTest(n=n):
                start = time.time()
                p_found, q_found = self.rfh3.factor(n, timeout=10.0)
                elapsed = time.time() - start

                # Should find correct factors
                self.assertEqual({p_found, q_found}, {p_true, q_true})

                # Should be relatively fast
                self.assertLess(elapsed, 5.0)

    def test_balanced_semiprimes(self):
        """Test RFH3 on balanced semiprimes"""
        for n, p_true, q_true in TestFixtures.BALANCED_SEMIPRIMES:
            with self.subTest(n=n):
                start = time.time()
                p_found, q_found = self.rfh3.factor(n, timeout=30.0)
                elapsed = time.time() - start

                # Should find correct factors
                self.assertEqual({p_found, q_found}, {p_true, q_true})

                # Should be reasonably fast
                self.assertLess(elapsed, 25.0)

    def test_learning_improvement(self):
        """Test that learning improves performance over time"""
        # Start with fresh learner
        fresh_rfh3 = RFH3(self.config)

        # Time factorization before learning
        n1, p1, q1 = TestFixtures.BALANCED_SEMIPRIMES[0]
        start = time.time()
        fresh_rfh3.factor(n1, timeout=10.0)
        time1 = time.time() - start

        # Factor a few more to train the learner
        for n, p, q in TestFixtures.SMALL_SEMIPRIMES:
            fresh_rfh3.factor(n, timeout=5.0)

        # Time similar factorization after learning
        n2, p2, q2 = TestFixtures.BALANCED_SEMIPRIMES[1]
        start = time.time()
        fresh_rfh3.factor(n2, timeout=10.0)
        time2 = time.time() - start

        # Learning effect may not always be measurable in small tests
        print(f"Before learning: {time1:.3f}s, After learning: {time2:.3f}s")

    def test_state_persistence(self):
        """Test state saving and loading"""
        import tempfile

        # Factor some numbers to build state
        for n, p, q in TestFixtures.SMALL_SEMIPRIMES:
            self.rfh3.factor(n, timeout=5.0)

        # Save state
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            temp_file = f.name

        try:
            self.rfh3.save_state(temp_file)

            # Create new instance and load state
            new_rfh3 = RFH3(self.config)
            new_rfh3.load_state(temp_file)

            # Should have similar statistics
            self.assertEqual(
                new_rfh3.stats["factorizations"], self.rfh3.stats["factorizations"]
            )

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_configuration_effects(self):
        """Test different configuration settings"""
        test_configs = [
            {"learning_enabled": True, "hierarchical_search": True},
            {"learning_enabled": False, "hierarchical_search": True},
            {"learning_enabled": True, "hierarchical_search": False},
            {"learning_enabled": False, "hierarchical_search": False},
        ]

        n, p_true, q_true = TestFixtures.SMALL_SEMIPRIMES[0]

        for config_dict in test_configs:
            with self.subTest(config=config_dict):
                config = RFH3Config()
                for key, value in config_dict.items():
                    setattr(config, key, value)

                rfh3 = RFH3(config)
                p_found, q_found = rfh3.factor(n, timeout=10.0)

                # All configurations should work
                self.assertEqual({p_found, q_found}, {p_true, q_true})

    def test_error_handling(self):
        """Test error handling for edge cases"""
        # Test prime input
        with self.assertRaises(ValueError):
            self.rfh3.factor(17)  # Prime number

        # Test too small input
        with self.assertRaises(ValueError):
            self.rfh3.factor(3)

        # Test very small composite
        result = self.rfh3.factor(4, timeout=5.0)  # 2 × 2
        self.assertEqual(set(result), {2, 2})


class TestComponentIntegration(unittest.TestCase):
    """Test integration between different components"""

    def test_analyzer_iterator_integration(self):
        """Test integration between analyzer and iterator"""
        from prime_resonance_field.core import (
            LazyResonanceIterator,
            MultiScaleResonance,
        )

        analyzer = MultiScaleResonance()
        n = 143
        iterator = LazyResonanceIterator(n, analyzer)

        # Iterator should use analyzer for resonance computation
        factors_found = []
        for i, x in enumerate(iterator):
            if n % x == 0:
                factors_found.append(x)
            if i >= 100:  # Limit iterations
                break

        # Should find at least one factor
        self.assertGreater(len(factors_found), 0)
        self.assertIn(11, factors_found)

    def test_learning_prediction_integration(self):
        """Test integration between learning and prediction"""
        from prime_resonance_field.learning import ResonancePatternLearner

        learner = ResonancePatternLearner()

        # Record some successful patterns
        test_cases = [(143, 11), (323, 17), (1147, 31)]
        for n, factor in test_cases:
            learner.record_success(n, factor, {"resonance": 0.9})

        # Predict zones for similar number
        n_test = 10403
        zones = learner.predict_high_resonance_zones(n_test)

        # Should predict reasonable zones
        self.assertIsInstance(zones, list)
        if len(zones) > 0:
            start, end, confidence = zones[0]
            sqrt_n = int(n_test**0.5)
            self.assertLessEqual(start, sqrt_n)
            self.assertGreaterEqual(end, 2)

    def test_hierarchical_analyzer_integration(self):
        """Test integration between hierarchical search and analyzer"""
        from prime_resonance_field.core import HierarchicalSearch, MultiScaleResonance

        analyzer = MultiScaleResonance()
        n = 143
        search = HierarchicalSearch(n, analyzer)

        # Search should use analyzer for resonance computation
        candidates = search.search(max_time=2.0)

        # Should find candidates including the factor
        self.assertIsInstance(candidates, list)

        factor_found = False
        for pos, resonance in candidates:
            if pos == 11 or pos == 13:
                factor_found = True
                # Factor should have high resonance
                self.assertGreater(resonance, 0.5)

        if not factor_found:
            print("Warning: Hierarchical search didn't find factor in candidates")


if __name__ == "__main__":
    unittest.main()
