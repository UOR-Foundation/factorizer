"""
Tests for HierarchicalSearch component
"""

import os
import sys
import unittest

from core import HierarchicalSearch, MultiScaleResonance


class TestHierarchicalSearch(unittest.TestCase):
    """Test cases for HierarchicalSearch"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = MultiScaleResonance()
        self.n = 143  # 11 × 13
        self.factor = 11
        self.search = HierarchicalSearch(self.n, self.analyzer)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.search.n, self.n)
        self.assertEqual(self.search.sqrt_n, int(self.n**0.5))
        self.assertIsNotNone(self.search.analyzer)
        self.assertIsInstance(self.search.levels, list)
        self.assertGreater(len(self.search.levels), 0)

    def test_hierarchy_levels_computation(self):
        """Test hierarchy levels are computed correctly"""
        levels = self.search.levels

        # Levels should be in ascending order (coarse to fine)
        for i in range(len(levels) - 1):
            self.assertLessEqual(levels[i], levels[i + 1])

        # Should include minimum level
        self.assertIn(100, levels)

    def test_search_returns_candidates(self):
        """Test that search returns candidate list"""
        candidates = self.search.search(max_time=1.0)

        self.assertIsInstance(candidates, list)
        # Each candidate should be (position, resonance) tuple
        for candidate in candidates[:10]:  # Check first 10
            self.assertIsInstance(candidate, tuple)
            self.assertEqual(len(candidate), 2)
            position, resonance = candidate
            self.assertIsInstance(position, int)
            self.assertIsInstance(resonance, (int, float))
            self.assertGreaterEqual(position, 2)
            self.assertLessEqual(position, self.search.sqrt_n)

    def test_factor_in_candidates(self):
        """Test that factors appear in candidates"""
        candidates = self.search.search(max_time=2.0)

        # Check if factor appears in candidates
        factor_found = False
        for position, resonance in candidates:
            if position == self.factor or position == self.n // self.factor:
                factor_found = True
                # Factor should have high resonance
                self.assertGreater(resonance, 0.5)
                break

        self.assertTrue(
            factor_found, "Factor should appear in hierarchical search candidates"
        )

    def test_coarse_sample(self):
        """Test coarse sampling phase"""
        peaks = self.search._coarse_sample(timeout=0.5)

        self.assertIsInstance(peaks, list)
        # Should find some peaks
        if len(peaks) > 0:
            # Peaks should be sorted by resonance (descending)
            for i in range(len(peaks) - 1):
                self.assertGreaterEqual(peaks[i][1], peaks[i + 1][1])

    def test_refine_peak(self):
        """Test peak refinement"""
        peak_position = self.factor
        peak_resonance = 1.0

        refined = self.search._refine_peak(peak_position, peak_resonance)

        self.assertIsInstance(refined, list)
        # Should include the original peak or nearby positions
        positions = [pos for pos, _ in refined]
        self.assertTrue(any(abs(pos - peak_position) <= 5 for pos in positions))


class TestHierarchicalSearchPerformance(unittest.TestCase):
    """Test performance characteristics of hierarchical search"""

    def test_search_time_bounds(self):
        """Test that search respects time bounds"""
        import time

        analyzer = MultiScaleResonance()
        n = 10403  # Larger number
        search = HierarchicalSearch(n, analyzer)

        # Test with short timeout
        start = time.time()
        search.search(max_time=0.1)
        elapsed = time.time() - start

        # Should respect timeout (with some tolerance)
        self.assertLess(elapsed, 0.5, "Search should respect timeout bounds")

    def test_search_scalability(self):
        """Test search scales reasonably with number size"""
        analyzer = MultiScaleResonance()

        # Small number
        small_n = 143
        small_search = HierarchicalSearch(small_n, analyzer)
        small_candidates = small_search.search(max_time=0.5)

        # Larger number
        large_n = 10403
        large_search = HierarchicalSearch(large_n, analyzer)
        large_candidates = large_search.search(max_time=0.5)

        # Both should return reasonable number of candidates
        self.assertGreater(len(small_candidates), 0)
        self.assertGreater(len(large_candidates), 0)


if __name__ == "__main__":
    unittest.main()
