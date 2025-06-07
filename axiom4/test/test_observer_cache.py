"""
Tests for Observer Cache functionality
Validates caching correctness, performance, and determinism
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom4.observer_cache import ObserverCache
from axiom4.adaptive_observer import MultiScaleObserver
import math

def test_observer_cache_init():
    """Test ObserverCache initialization"""
    cache = ObserverCache(cache_size=1000)
    
    assert cache.cache_size == 1000
    assert len(cache.observation_cache) == 0
    assert len(cache.gradient_cache) == 0
    assert len(cache.state_cache) == 0
    assert len(cache.path_cache) == 0
    
    assert cache.hits == 0
    assert cache.misses == 0
    
    print("✓ ObserverCache initialization")

def test_observation_caching():
    """Test observation caching behavior"""
    n = 143  # 11 × 13
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # First observation should be a miss
    obs1 = cache.get_observation(observer, 7)
    assert cache.misses == 1
    assert cache.hits == 0
    
    # Second observation of same position should be a hit
    obs2 = cache.get_observation(observer, 7)
    assert cache.hits == 1
    assert cache.misses == 1
    
    # Results should be identical
    assert obs1 == obs2
    
    # Different position should be a miss
    obs3 = cache.get_observation(observer, 8)
    assert cache.misses == 2
    
    print("✓ Observation caching")

def test_gradient_caching():
    """Test gradient caching behavior"""
    n = 77  # 7 × 11
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # First gradient calculation should be a miss
    grad1 = cache.get_gradient(n, 5, observer)
    assert cache.gradient_misses == 1
    assert cache.gradient_hits == 0
    
    # Second gradient at same position should be a hit
    grad2 = cache.get_gradient(n, 5, observer)
    assert cache.gradient_hits == 1
    assert cache.gradient_misses == 1
    
    # Results should be identical
    assert grad1 == grad2
    
    # Different delta should be a miss
    grad3 = cache.get_gradient(n, 5, observer, delta=2)
    assert cache.gradient_misses == 2
    
    print("✓ Gradient caching")

def test_quantum_state_caching():
    """Test quantum state caching"""
    n = 55  # 5 × 11
    cache = ObserverCache()
    
    # Cache a quantum state
    state1 = [(3, 0.8), (5, 0.9), (7, 0.7)]
    cache.cache_quantum_state(n, iteration=0, candidates=state1)
    
    # Retrieve the state
    retrieved = cache.get_quantum_state(n, iteration=0)
    assert retrieved == state1
    
    # Non-existent state should return None
    assert cache.get_quantum_state(n, iteration=1) is None
    
    # Different n should return None
    assert cache.get_quantum_state(99, iteration=0) is None
    
    print("✓ Quantum state caching")

def test_navigation_path_caching():
    """Test navigation path caching"""
    n = 91  # 7 × 13
    cache = ObserverCache()
    
    # Store a path
    path = [2, 3, 5, 7]
    cache.store_navigation_path(n, start=2, end=7, path=path)
    
    # Retrieve the path
    retrieved = cache.get_navigation_path(n, start=2, end=7)
    assert retrieved == path
    assert cache.path_hits == 1
    
    # Non-existent path should return None
    assert cache.get_navigation_path(n, start=3, end=7) is None
    assert cache.path_misses == 1
    
    print("✓ Navigation path caching")

def test_lru_eviction():
    """Test LRU eviction policy"""
    n = 21  # 3 × 7
    observer = MultiScaleObserver(n)
    cache = ObserverCache(cache_size=3)  # Small cache
    
    # Fill cache
    cache.get_observation(observer, 2)
    cache.get_observation(observer, 3)
    cache.get_observation(observer, 4)
    
    assert len(cache.observation_cache) == 3
    
    # Add one more - should evict position 2
    cache.get_observation(observer, 5)
    assert len(cache.observation_cache) == 3
    
    # Access position 3 to make it most recent
    cache.get_observation(observer, 3)
    
    # Add another - should evict position 4
    cache.get_observation(observer, 6)
    
    # Check that 3 is still in cache (was accessed recently)
    initial_hits = cache.hits
    cache.get_observation(observer, 3)
    assert cache.hits == initial_hits + 1
    
    print("✓ LRU eviction")

def test_precompute_fibonacci():
    """Test Fibonacci position pre-computation"""
    n = 100
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Pre-compute Fibonacci positions
    cache.precompute_fibonacci_positions(n, observer)
    
    # Should have some cached observations
    assert len(cache.observation_cache) > 0
    
    # Fibonacci numbers should be cached
    initial_hits = cache.hits
    cache.get_observation(observer, 2)  # fib(3)
    cache.get_observation(observer, 3)  # fib(4)
    cache.get_observation(observer, 5)  # fib(5)
    cache.get_observation(observer, 8)  # fib(6)
    
    # Most should be hits
    assert cache.hits > initial_hits
    
    print("✓ Fibonacci pre-computation")

def test_precompute_primes():
    """Test prime position pre-computation"""
    n = 100
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Pre-compute prime positions
    cache.precompute_prime_positions(n, observer)
    
    # Primes should be cached
    initial_hits = cache.hits
    cache.get_observation(observer, 2)
    cache.get_observation(observer, 3)
    cache.get_observation(observer, 5)
    cache.get_observation(observer, 7)
    
    # All should be hits
    assert cache.hits == initial_hits + 4
    
    print("✓ Prime pre-computation")

def test_precompute_sqrt_neighborhood():
    """Test sqrt neighborhood pre-computation"""
    n = 144  # sqrt = 12
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Pre-compute sqrt neighborhood
    cache.precompute_sqrt_neighborhood(n, observer, radius=5)
    
    # Should have pre-computed observations
    initial_cache_size = len(cache.observation_cache)
    assert initial_cache_size > 0
    
    # Positions near sqrt should be cached
    initial_hits = cache.hits
    hits_before_each = []
    for offset in [-5, -3, 0, 3, 5]:
        pos = 12 + offset
        hits_before = cache.hits
        cache.get_observation(observer, pos)
        hits_after = cache.hits
        hits_before_each.append((pos, hits_after > hits_before))
    
    # Most should be hits (at least 3 out of 5)
    hit_count = sum(1 for _, was_hit in hits_before_each if was_hit)
    assert hit_count >= 3, f"Only {hit_count} hits out of 5. Details: {hits_before_each}"
    
    print("✓ Sqrt neighborhood pre-computation")

def test_cache_statistics():
    """Test cache statistics tracking"""
    n = 35  # 5 × 7
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Track observation stats
    cache.get_observation(observer, 5)  # miss
    obs_misses_1 = cache.misses
    cache.get_observation(observer, 5)  # hit
    obs_hits_1 = cache.hits
    
    # Track gradient stats (note: gradient computation may call get_observation internally)
    cache.get_gradient(n, 5, observer)  # miss
    grad_misses_1 = cache.gradient_misses
    cache.get_gradient(n, 5, observer)  # hit
    grad_hits_1 = cache.gradient_hits
    
    # Track path stats
    path = [2, 3, 5]
    cache.store_navigation_path(n, 2, 5, path)
    cache.get_navigation_path(n, 2, 5)  # hit
    path_hits_1 = cache.path_hits
    cache.get_navigation_path(n, 3, 5)  # miss
    path_misses_1 = cache.path_misses
    
    stats = cache.get_cache_statistics()
    
    # Verify the counts increased as expected
    assert obs_hits_1 == 1
    assert obs_misses_1 >= 1  # At least 1 (may be more from gradient computation)
    assert grad_hits_1 == 1
    assert grad_misses_1 == 1
    assert path_hits_1 == 1
    assert path_misses_1 == 1
    
    # Verify stats match current counts
    assert stats['observation_hits'] == cache.hits
    assert stats['observation_misses'] == cache.misses
    assert stats['gradient_hits'] == cache.gradient_hits
    assert stats['gradient_misses'] == cache.gradient_misses
    assert stats['path_hits'] == cache.path_hits
    assert stats['path_misses'] == cache.path_misses
    assert 0 <= stats['total_hit_rate'] <= 1
    
    print("✓ Cache statistics")

def test_cache_clear():
    """Test cache clearing"""
    n = 15  # 3 × 5
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Add some data
    cache.get_observation(observer, 3)
    cache.get_gradient(n, 3, observer)
    cache.cache_quantum_state(n, 0, [(3, 0.5)])
    cache.store_navigation_path(n, 2, 3, [2, 3])
    
    # Clear cache
    cache.clear()
    
    # All caches should be empty
    assert len(cache.observation_cache) == 0
    assert len(cache.gradient_cache) == 0
    assert len(cache.state_cache) == 0
    assert len(cache.path_cache) == 0
    
    # Stats should be reset
    assert cache.hits == 0
    assert cache.misses == 0
    
    print("✓ Cache clearing")

def test_create_optimized():
    """Test optimized cache creation"""
    # Small number
    cache_small = ObserverCache.create_optimized(100)
    assert cache_small.cache_size == 1000  # min(2000, 10*10) = 100, then max(100, 1000) = 1000
    
    # Medium number (n < 100000)
    cache_medium = ObserverCache.create_optimized(10000)
    assert cache_medium.cache_size == 1000  # min(5000, 100*5) = 500, then max(500, 1000) = 1000
    
    # Large number (n < 10000000)
    cache_large = ObserverCache.create_optimized(9999999)
    assert cache_large.cache_size == 6324  # min(10000, 3162*2) = 6324
    
    # Very large number (n >= 10000000)
    cache_xlarge = ObserverCache.create_optimized(10000000)
    assert cache_xlarge.cache_size == 3162  # min(20000, 3162) = 3162
    
    # Even larger number
    cache_xxlarge = ObserverCache.create_optimized(100000000)
    assert cache_xxlarge.cache_size == 10000  # min(20000, 10000) = 10000
    
    print("✓ Optimized cache creation")

def test_cache_determinism():
    """Test that caching maintains determinism"""
    n = 143  # 11 × 13
    
    # Create two separate setups
    observer1 = MultiScaleObserver(n)
    cache1 = ObserverCache()
    
    observer2 = MultiScaleObserver(n)
    cache2 = ObserverCache()
    
    # Perform same operations
    positions = [5, 7, 11, 13]
    
    results1 = []
    for pos in positions:
        obs = cache1.get_observation(observer1, pos)
        grad = cache1.get_gradient(n, pos, observer1)
        results1.append((obs, grad))
    
    results2 = []
    for pos in positions:
        obs = cache2.get_observation(observer2, pos)
        grad = cache2.get_gradient(n, pos, observer2)
        results2.append((obs, grad))
    
    # Results should be identical
    assert results1 == results2
    
    print("✓ Cache determinism")

def test_edge_cases():
    """Test edge cases"""
    n = 6
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Boundary positions
    assert cache.get_gradient(n, 1, observer) == 0.0  # Too small
    assert cache.get_gradient(n, 100, observer) == 0.0  # Too large
    
    # Empty path retrieval
    assert cache.get_navigation_path(n, 1, 2) is None
    
    # Pre-compute with no valid positions
    cache.precompute_fibonacci_positions(4, observer)  # Small n
    
    print("✓ Edge cases")

def run_all_tests():
    """Run all observer cache tests"""
    print("Testing Observer Cache (Axiom 4 Acceleration)...")
    print("-" * 50)
    
    test_observer_cache_init()
    test_observation_caching()
    test_gradient_caching()
    test_quantum_state_caching()
    test_navigation_path_caching()
    test_lru_eviction()
    test_precompute_fibonacci()
    test_precompute_primes()
    test_precompute_sqrt_neighborhood()
    test_cache_statistics()
    test_cache_clear()
    test_create_optimized()
    test_cache_determinism()
    test_edge_cases()
    
    print("-" * 50)
    print("All Observer Cache tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
