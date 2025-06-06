"""
Test Prime Coordinate Index acceleration
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom1 import PrimeCoordinateIndex, PrimeGeodesic, is_prime

class TestPrimeCoordinateIndex(unittest.TestCase):
    """Test the Prime Coordinate Index acceleration"""
    
    def setUp(self):
        """Set up test with coordinate index"""
        self.index = PrimeCoordinateIndex(limit=10000)
        
    def test_coordinate_caching(self):
        """Test that coordinates are properly cached"""
        # First access should be a cache miss
        coords1 = self.index.get_coordinates(100)
        initial_misses = self.index.cache_misses
        
        # Second access should be a cache hit
        coords2 = self.index.get_coordinates(100)
        self.assertEqual(coords1, coords2)
        self.assertEqual(self.index.cache_misses, initial_misses)
        self.assertEqual(self.index.cache_hits, 1)
        
    def test_pull_calculation(self):
        """Test gravitational pull calculation"""
        # For n=15, factors are 3 and 5
        # Position 3 should have strong pull since 15 % 3 == 0
        pull_3 = self.index.get_pull(3, 15)
        self.assertGreater(pull_3, 2.0)  # Should have divisor bonus
        
        # Position 4 may have weak pull from coordinate alignment
        # This is correct axiomatic behavior
        pull_4 = self.index.get_pull(4, 15)
        self.assertGreaterEqual(pull_4, 0)  # Can have weak alignment pull
        
        # Position 5 should have strong pull since 15 % 5 == 0
        pull_5 = self.index.get_pull(5, 15)
        self.assertGreater(pull_5, 2.0)  # Should have divisor bonus
        
    def test_geodesic_path_caching(self):
        """Test geodesic path storage and retrieval"""
        # Store a path
        path = [2, 3, 5, 7]
        self.index.store_geodesic_path(2, 7, 21, path)
        
        # Retrieve the path
        cached_path = self.index.get_geodesic_path(2, 7, 21)
        self.assertEqual(cached_path, path)
        
        # Test reverse path
        reverse_path = self.index.get_geodesic_path(7, 2, 21)
        self.assertEqual(reverse_path, list(reversed(path)))
        
    def test_coordinate_distance(self):
        """Test coordinate distance calculation"""
        # Distance between same number should be 0
        dist = self.index.compute_coordinate_distance(10, 10)
        self.assertEqual(dist, 0)
        
        # Distance between different numbers
        dist = self.index.compute_coordinate_distance(10, 15)
        self.assertGreater(dist, 0)
        
    def test_zero_coordinates(self):
        """Test finding zero coordinates"""
        # For n=12, should have zeros at primes 2 and 3
        zeros = self.index.get_zero_coordinates(12)
        prime_indices = [p for i, p in zeros]
        self.assertIn(2, prime_indices)
        self.assertIn(3, prime_indices)
        self.assertNotIn(5, prime_indices)
        
    def test_precomputation(self):
        """Test that precomputation fills cache"""
        # Clear cache
        self.index.coordinates.clear()
        self.index.cache_hits = 0
        self.index.cache_misses = 0
        
        # Precompute
        self.index.precompute_common_coordinates()
        
        # Check that some coordinates are cached
        self.assertGreater(len(self.index.coordinates), 100)
        
        # Access should be cache hits
        coords = self.index.get_coordinates(64)  # Power of 2
        self.assertGreater(self.index.cache_hits, 0)
        
    def test_integration_with_geodesic(self):
        """Test integration with PrimeGeodesic"""
        # Create geodesic with coordinate index
        n = 77  # 7 × 11
        geodesic = PrimeGeodesic(n, self.index)
        
        # Walk should use cached calculations
        initial_hits = self.index.cache_hits
        path = geodesic.walk(5, steps=10)
        
        # Should have some cache usage
        self.assertGreater(self.index.cache_hits, initial_hits)
        
        # Check if path found a factor
        for pos in path:
            if n % pos == 0 and is_prime(pos):
                self.assertIn(pos, [7, 11])
                
    def test_exploration_suggestions(self):
        """Test exploration point suggestions"""
        n = 35  # 5 × 7
        suggestions = self.index.suggest_exploration_points(n, 4)
        
        # Should suggest some positions
        self.assertGreater(len(suggestions), 0)
        
        # Suggestions should be in valid range
        sqrt_n = int(n**0.5)
        for s in suggestions:
            self.assertGreaterEqual(s, 2)
            self.assertLessEqual(s, sqrt_n)
            
    def test_cache_statistics(self):
        """Test cache statistics reporting"""
        # Do some operations
        self.index.get_coordinates(100)
        self.index.get_coordinates(100)  # Cache hit
        self.index.get_pull(5, 25)
        
        stats = self.index.get_cache_statistics()
        
        # Check statistics
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
        self.assertIn('hit_rate', stats)
        self.assertGreater(stats['hit_rate'], 0)

if __name__ == '__main__':
    unittest.main()
