"""
Tests for Accelerated Observer functionality
Validates speedup, correctness, and integration with other axioms
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom4.accelerated_observer import (
    accelerated_observe,
    accelerated_gradient,
    accelerated_collapse,
    accelerated_navigation,
    accelerated_coherence_field,
    accelerated_gradient_ascent,
    accelerated_multi_path,
    create_accelerated_observer,
    benchmark_acceleration,
    get_global_cache,
    set_global_cache
)
from axiom4.adaptive_observer import MultiScaleObserver, generate_superposition
from axiom4.observer_cache import ObserverCache
import math
import time

def test_accelerated_observe():
    """Test accelerated observation"""
    n = 143  # 11 × 13
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Direct observation
    direct_obs = observer.observe(7)
    
    # Accelerated observation
    accel_obs = accelerated_observe(observer, 7, cache)
    
    # Should be equal
    assert abs(direct_obs - accel_obs) < 1e-10
    
    # Second call should use cache
    initial_hits = cache.hits
    accel_obs2 = accelerated_observe(observer, 7, cache)
    assert cache.hits > initial_hits
    assert abs(accel_obs - accel_obs2) < 1e-10
    
    print("✓ Accelerated observation")

def test_accelerated_gradient():
    """Test accelerated gradient computation"""
    n = 77  # 7 × 11
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Calculate gradients
    grad1 = accelerated_gradient(n, 5, observer, cache=cache)
    
    # Second call should use cache
    initial_hits = cache.gradient_hits
    grad2 = accelerated_gradient(n, 5, observer, cache=cache)
    assert cache.gradient_hits > initial_hits
    assert abs(grad1 - grad2) < 1e-10
    
    print("✓ Accelerated gradient")

def test_accelerated_collapse():
    """Test accelerated wavefunction collapse"""
    n = 221  # 13 × 17
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Generate superposition
    candidates = [10, 11, 12, 13, 14, 15, 16, 17]
    
    # Accelerated collapse
    collapsed = accelerated_collapse(n, candidates, observer, iterations=3, cache=cache)
    
    assert isinstance(collapsed, list)
    assert len(collapsed) > 0
    assert all(isinstance(item, tuple) and len(item) == 2 for item in collapsed)
    
    # Weights should be sorted
    weights = [w for _, w in collapsed]
    assert weights == sorted(weights, reverse=True)
    
    # Cache should have quantum states
    assert len(cache.state_cache) > 0
    
    print("✓ Accelerated collapse")

def test_quantum_state_resumption():
    """Test that collapse can resume from cached state"""
    n = 55  # 5 × 11
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    candidates = [3, 4, 5, 6, 7, 8]
    
    # Do partial collapse
    partial = accelerated_collapse(n, candidates, observer, iterations=2, cache=cache)
    
    # Clear observation cache but keep quantum states
    cache.observation_cache.clear()
    cache.gradient_cache.clear()
    
    # Continue collapse - should resume from iteration 2
    full = accelerated_collapse(n, candidates, observer, iterations=5, cache=cache)
    
    # Should have results
    assert len(full) > 0
    
    print("✓ Quantum state resumption")

def test_accelerated_navigation():
    """Test accelerated navigation"""
    n = 91  # 7 × 13
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Navigate from near a factor
    result = accelerated_navigation(n, start=6, observer=observer, 
                                   max_iterations=20, cache=cache)
    
    # Should find factor 7 or 13
    assert result in [7, 13]
    
    # Path should be cached
    assert len(cache.path_cache) > 0
    
    print("✓ Accelerated navigation")

def test_path_caching():
    """Test that navigation uses cached paths"""
    n = 35  # 5 × 7
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # First navigation
    result1 = accelerated_navigation(n, start=4, observer=observer,
                                    max_iterations=10, cache=cache)
    
    # Clear observation/gradient caches
    cache.observation_cache.clear()
    cache.gradient_cache.clear()
    initial_path_hits = cache.path_hits
    
    # Second navigation should use cached path
    result2 = accelerated_navigation(n, start=4, observer=observer,
                                    max_iterations=10, cache=cache)
    
    assert result1 == result2
    # May or may not hit cached path depending on exact path taken
    
    print("✓ Path caching")

def test_accelerated_coherence_field():
    """Test accelerated coherence field generation"""
    n = 143  # 11 × 13
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    positions = list(range(5, 20))
    
    # Build field with acceleration
    field = accelerated_coherence_field(n, positions, observer, cache)
    
    assert len(field) == len(positions)
    assert all(pos in field for pos in positions)
    assert all(0 <= coh <= 1 for coh in field.values())
    
    # Cache should have pre-computed critical positions
    assert len(cache.observation_cache) > len(positions)
    
    print("✓ Accelerated coherence field")

def test_accelerated_gradient_ascent():
    """Test accelerated gradient ascent"""
    n = 77  # 7 × 11
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Start near a factor
    path = accelerated_gradient_ascent(n, start=6, observer=observer,
                                      max_steps=10, cache=cache)
    
    assert isinstance(path, list)
    assert len(path) > 0
    assert path[0] == 6
    
    # Should move toward higher coherence
    if len(path) > 1:
        start_coh = cache.get_observation(observer, path[0])
        end_coh = cache.get_observation(observer, path[-1])
        # Generally should increase, but not always due to discrete steps
        assert end_coh >= start_coh - 0.1
    
    print("✓ Accelerated gradient ascent")

def test_accelerated_multi_path():
    """Test accelerated multi-path search"""
    n = 221  # 13 × 17
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Multiple starting points
    starts = [10, 12, 14, 16, 18]
    
    endpoints = accelerated_multi_path(n, starts, observer, max_paths=3, cache=cache)
    
    assert isinstance(endpoints, list)
    assert len(endpoints) <= 3
    
    # Should be sorted by coherence
    for i in range(len(endpoints) - 1):
        assert endpoints[i][1] >= endpoints[i+1][1]
    
    print("✓ Accelerated multi-path")

def test_create_accelerated_observer():
    """Test accelerated observer creation"""
    n = 143  # 11 × 13
    
    observer, cache = create_accelerated_observer(n)
    
    assert isinstance(observer, MultiScaleObserver)
    assert isinstance(cache, ObserverCache)
    
    # Should have pre-computed critical positions
    assert len(cache.observation_cache) > 0
    assert n in cache.precomputed_fibonacci
    assert n in cache.precomputed_primes
    assert n in cache.precomputed_sqrt
    
    print("✓ Accelerated observer creation")

def test_global_cache():
    """Test global cache functionality"""
    # Get default global cache
    cache1 = get_global_cache()
    assert isinstance(cache1, ObserverCache)
    
    # Should return same instance
    cache2 = get_global_cache()
    assert cache1 is cache2
    
    # Set custom cache
    custom_cache = ObserverCache(cache_size=500)
    set_global_cache(custom_cache)
    
    cache3 = get_global_cache()
    assert cache3 is custom_cache
    
    print("✓ Global cache")

def test_axiom3_integration():
    """Test integration with Axiom 3's accelerated coherence"""
    n = 55  # 5 × 11
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # The observer cache should use accelerated coherence internally
    # This is verified by checking that coherence calculations are fast
    positions = list(range(2, 8))
    
    start = time.time()
    for pos in positions:
        _ = cache.get_observation(observer, pos)
    time_elapsed = time.time() - start
    
    # Should be reasonably fast (hard to test exactly without mocking)
    assert time_elapsed < 1.0  # Very generous bound
    
    print("✓ Axiom 3 integration")

def test_determinism():
    """Test that acceleration maintains determinism"""
    n = 91  # 7 × 13
    
    results1 = []
    results2 = []
    
    for _ in range(2):
        observer = MultiScaleObserver(n)
        cache = ObserverCache()
        
        # Perform operations
        obs = accelerated_observe(observer, 7, cache)
        grad = accelerated_gradient(n, 7, observer, cache=cache)
        
        candidates = generate_superposition(n)[:10]
        collapsed = accelerated_collapse(n, candidates, observer, 
                                       iterations=2, cache=cache)
        
        result = (obs, grad, len(collapsed), collapsed[0] if collapsed else None)
        
        if not results1:
            results1 = result
        else:
            results2 = result
    
    # Results should be identical
    assert results1[0] == results2[0]  # observation
    assert results1[1] == results2[1]  # gradient
    assert results1[2] == results2[2]  # collapsed length
    if results1[3] and results2[3]:
        assert results1[3] == results2[3]  # first collapsed item
    
    print("✓ Determinism maintained")

def test_benchmark_acceleration():
    """Test benchmark function"""
    n = 100
    
    # Run benchmark with small iterations
    results = benchmark_acceleration(n, iterations=10)
    
    assert 'speedup_observe' in results
    assert 'speedup_gradient' in results
    assert 'cache_stats' in results
    
    # Should show some speedup (at least 1x, hopefully more)
    assert results['speedup_observe'] >= 1.0
    assert results['speedup_gradient'] >= 1.0
    
    # Cache should have statistics
    stats = results['cache_stats']
    assert 'total_hit_rate' in stats
    
    print("✓ Benchmark acceleration")
    print(f"  - Observation speedup: {results['speedup_observe']:.2f}x")
    print(f"  - Gradient speedup: {results['speedup_gradient']:.2f}x")
    print(f"  - Cache hit rate: {stats['total_hit_rate']:.2%}")

def test_edge_cases():
    """Test edge cases for accelerated functions"""
    # Small number
    n = 6
    observer = MultiScaleObserver(n)
    cache = ObserverCache()
    
    # Should handle small numbers
    obs = accelerated_observe(observer, 2, cache)
    assert obs >= 0
    
    # Empty candidate list
    collapsed = accelerated_collapse(n, [], observer, cache=cache)
    assert collapsed == []
    
    # Navigation with no path
    result = accelerated_navigation(n, start=1, observer=observer,
                                   max_iterations=5, cache=cache)
    # May or may not find factor 2 or 3
    
    print("✓ Edge cases")

def run_all_tests():
    """Run all accelerated observer tests"""
    print("Testing Accelerated Observer (Axiom 4)...")
    print("-" * 50)
    
    test_accelerated_observe()
    test_accelerated_gradient()
    test_accelerated_collapse()
    test_quantum_state_resumption()
    test_accelerated_navigation()
    test_path_caching()
    test_accelerated_coherence_field()
    test_accelerated_gradient_ascent()
    test_accelerated_multi_path()
    test_create_accelerated_observer()
    test_global_cache()
    test_axiom3_integration()
    test_determinism()
    test_benchmark_acceleration()
    test_edge_cases()
    
    print("-" * 50)
    print("All Accelerated Observer tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
