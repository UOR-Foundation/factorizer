"""
Test Meta-Acceleration Cache implementation
"""

import unittest
from time import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom5.meta_acceleration_cache import (
    MetaAccelerationCache, get_meta_cache,
    accelerated_meta_coherence, accelerated_spectral_distance,
    accelerated_mirror_position
)
from axiom3 import spectral_vector, coherence

class TestMetaAccelerationCache(unittest.TestCase):
    """Test Meta-Acceleration Cache functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Reset global cache
        import axiom5.meta_acceleration_cache
        axiom5.meta_acceleration_cache._meta_cache = None
        self.cache = MetaAccelerationCache()
        self.n = 437
        
    def test_meta_coherence_caching(self):
        """Test meta-coherence caching functionality"""
        position = 10
        coherence_field = {5: 0.5, 10: 0.8, 15: 0.3}
        
        # Compute function
        def compute(pos, field):
            return sum(field.values()) / len(field)
        
        # First call should compute
        result1 = self.cache.get_meta_coherence(position, coherence_field, compute)
        self.assertAlmostEqual(result1, 0.533, places=2)
        
        # Second call should use cache
        result2 = self.cache.get_meta_coherence(position, coherence_field)
        self.assertEqual(result1, result2)
        
    def test_observation_indexing(self):
        """Test observation indexing and queries"""
        # Add observations
        obs1 = {'position': 10, 'coherence': 0.8, 'axiom': 'axiom1', 'found': True}
        obs2 = {'position': 20, 'coherence': 0.3, 'axiom': 'axiom2', 'found': False}
        obs3 = {'position': 10, 'coherence': 0.9, 'axiom': 'axiom3', 'found': True}
        
        self.cache.add_observation(obs1)
        self.cache.add_observation(obs2)
        self.cache.add_observation(obs3)
        
        # Query by position
        pos_10_obs = self.cache.query_observations(position=10)
        self.assertEqual(len(pos_10_obs), 2)
        
        # Query by axiom
        axiom1_obs = self.cache.query_observations(axiom='axiom1')
        self.assertEqual(len(axiom1_obs), 1)
        self.assertEqual(axiom1_obs[0]['position'], 10)
        
        # Query by coherence
        high_coh_obs = self.cache.query_observations(min_coherence=0.7)
        self.assertEqual(len(high_coh_obs), 2)
        
    def test_coherence_field_caching(self):
        """Test coherence field caching"""
        n = 100
        iteration = 5
        
        def compute_field(n, iter):
            return {i: 0.5 + i/100 for i in range(2, 10)}
        
        # First call computes
        field1 = self.cache.get_coherence_field(n, iteration, compute_field)
        self.assertIsNotNone(field1)
        self.assertEqual(len(field1), 8)
        
        # Second call uses cache
        field2 = self.cache.get_coherence_field(n, iteration)
        self.assertEqual(field1, field2)
        
    def test_fixed_points_caching(self):
        """Test fixed points caching"""
        n = 437
        points = [2, 5, 7, 11]
        
        # Add fixed points
        self.cache.add_fixed_points(n, points)
        
        # Retrieve
        retrieved = self.cache.get_fixed_points(n)
        self.assertEqual(retrieved, points)
        
    def test_spectral_distance_caching(self):
        """Test spectral distance caching"""
        x, y = 10, 20
        
        def compute():
            spec_x = spectral_vector(x)
            spec_y = spectral_vector(y)
            distance = sum((sx - sy) ** 2 for sx, sy in zip(spec_x, spec_y)) ** 0.5
            return distance
        
        # First call computes
        dist1 = self.cache.get_spectral_distance(x, y, compute)
        self.assertIsNotNone(dist1)
        self.assertGreater(dist1, 0)
        
        # Second call uses cache
        dist2 = self.cache.get_spectral_distance(x, y)
        self.assertEqual(dist1, dist2)
        
        # Order shouldn't matter
        dist3 = self.cache.get_spectral_distance(y, x)
        self.assertEqual(dist1, dist3)
        
    def test_axiom_combination_caching(self):
        """Test axiom combination success rate caching"""
        pattern_hash = "test_pattern_123"
        success_rate = 0.75
        
        # Add combination
        self.cache.add_axiom_combination(pattern_hash, success_rate)
        
        # Retrieve
        retrieved = self.cache.get_axiom_combination_score(pattern_hash)
        self.assertEqual(retrieved, success_rate)
        
    def test_interference_strength_caching(self):
        """Test interference strength caching"""
        axiom1, axiom2 = "axiom1", "axiom2"
        strength = 0.6
        
        # Add interference
        self.cache.add_interference_strength(axiom1, axiom2, strength)
        
        # Retrieve (order shouldn't matter)
        retrieved1 = self.cache.get_interference_strength(axiom1, axiom2)
        retrieved2 = self.cache.get_interference_strength(axiom2, axiom1)
        
        self.assertEqual(retrieved1, strength)
        self.assertEqual(retrieved2, strength)
        
    def test_global_cache_instance(self):
        """Test global cache instance management"""
        cache1 = get_meta_cache()
        cache2 = get_meta_cache()
        
        # Should be the same instance
        self.assertIs(cache1, cache2)
        
    def test_accelerated_functions(self):
        """Test accelerated convenience functions"""
        # Test accelerated spectral distance
        dist = accelerated_spectral_distance(10, 20)
        self.assertGreater(dist, 0)
        
        # Test accelerated meta coherence
        field = {5: 0.5, 10: 0.8}
        def compute(pos, f):
            return sum(f.values()) / len(f)
        
        mc = accelerated_meta_coherence(10, field, compute)
        self.assertGreater(mc, 0)
        
    def test_performance_improvement(self):
        """Test that caching provides performance improvement"""
        # Test spectral distance caching performance
        x_vals = list(range(10, 20))
        y_vals = list(range(20, 30))
        
        # First pass - no cache
        start = time()
        for x in x_vals:
            for y in y_vals:
                dist = accelerated_spectral_distance(x, y)
        first_pass_time = time() - start
        
        # Second pass - with cache
        start = time()
        for x in x_vals:
            for y in y_vals:
                dist = accelerated_spectral_distance(x, y)
        second_pass_time = time() - start
        
        # Second pass should be significantly faster
        self.assertLess(second_pass_time, first_pass_time * 0.5)
        
    def test_cache_statistics(self):
        """Test cache performance statistics"""
        # Add some data
        self.cache.add_observation({'position': 10, 'coherence': 0.5})
        self.cache.add_fixed_points(100, [2, 3, 5])
        self.cache.get_spectral_distance(10, 20, lambda: 1.5)
        
        # Get stats
        stats = self.cache.get_performance_stats()
        
        self.assertIn('observation_count', stats)
        self.assertIn('fixed_points_cached', stats)
        self.assertIn('spectral_distances_cached', stats)
        self.assertEqual(stats['observation_count'], 1)
        self.assertEqual(stats['fixed_points_cached'], 1)
        self.assertEqual(stats['spectral_distances_cached'], 1)

if __name__ == '__main__':
    unittest.main()
