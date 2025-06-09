"""
Performance tuning tests for Axiom 4 acceleration
Verifies cache tuning improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom4.observer_cache import ObserverCache
from axiom4.accelerated_observer import (
    create_accelerated_observer,
    accelerated_coherence_field,
    benchmark_acceleration
)
from axiom4.adaptive_observer import MultiScaleObserver
import time

def test_batch_observe():
    """Test batch observation performance"""
    n = 1000
    observer = MultiScaleObserver(n)
    cache = ObserverCache.create_optimized(n)
    
    positions = list(range(2, 32))  # 30 positions
    
    # Time individual observations
    start = time.time()
    results1 = {}
    for pos in positions:
        results1[pos] = cache.get_observation(observer, pos)
    time_individual = time.time() - start
    
    # Clear cache for fair comparison
    cache.clear()
    
    # Time batch observation
    start = time.time()
    results2 = cache.batch_observe(observer, positions)
    time_batch = time.time() - start
    
    # Results should be identical
    assert len(results1) == len(results2)
    for pos in positions:
        assert abs(results1[pos] - results2[pos]) < 1e-10
    
    print(f"✓ Batch observation (individual: {time_individual:.4f}s, batch: {time_batch:.4f}s)")

def test_warm_cache_effectiveness():
    """Test cache warming effectiveness"""
    n = 143  # 11 × 13
    observer = MultiScaleObserver(n)
    
    # Cache without warming
    cache1 = ObserverCache.create_optimized(n)
    cache1.precompute_critical_positions(n, observer)
    
    # Cache with warming
    cache2 = ObserverCache.create_optimized(n)
    cache2.precompute_critical_positions(n, observer)
    cache2.warm_cache(n, observer)
    
    # Test positions that include factors
    test_positions = [7, 11, 13, 22, 26]
    
    hits1 = 0
    for pos in test_positions:
        initial_hits = cache1.hits
        cache1.get_observation(observer, pos)
        if cache1.hits > initial_hits:
            hits1 += 1
            
    hits2 = 0
    for pos in test_positions:
        initial_hits = cache2.hits
        cache2.get_observation(observer, pos)
        if cache2.hits > initial_hits:
            hits2 += 1
    
    # Warmed cache should have more hits
    assert hits2 >= hits1
    print(f"✓ Cache warming (cold hits: {hits1}, warm hits: {hits2})")

def test_gradient_precomputation():
    """Test gradient pre-computation effectiveness"""
    n = 221  # 13 × 17
    observer = MultiScaleObserver(n)
    cache = ObserverCache.create_optimized(n)
    
    # Pre-compute gradients for critical positions
    critical_positions = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    cache.precompute_gradients(n, observer, critical_positions)
    
    # Test gradient hits
    hits = 0
    for pos in critical_positions:
        initial_hits = cache.gradient_hits
        cache.get_gradient(n, pos, observer)
        if cache.gradient_hits > initial_hits:
            hits += 1
    
    # Should have high hit rate
    assert hits >= len(critical_positions) * 0.8
    print(f"✓ Gradient pre-computation ({hits}/{len(critical_positions)} hits)")

def test_intelligent_cache_sizing():
    """Test intelligent cache sizing"""
    # Small number
    cache_small = ObserverCache.create_optimized(100)
    
    # Medium number  
    cache_medium = ObserverCache.create_optimized(10000)
    
    # Large number
    cache_large = ObserverCache.create_optimized(1000000)
    
    # Cache sizes should be appropriate
    assert cache_small.cache_size >= 1000
    assert cache_medium.cache_size >= cache_small.cache_size
    assert cache_large.cache_size >= cache_medium.cache_size
    
    print(f"✓ Intelligent cache sizing (small: {cache_small.cache_size}, " +
          f"medium: {cache_medium.cache_size}, large: {cache_large.cache_size})")

def test_performance_improvement():
    """Test overall performance improvement with tuning"""
    test_numbers = [143, 221, 323, 437]  # Various semiprimes
    
    total_improvement = 0
    for n in test_numbers:
        # Benchmark with basic cache
        observer = MultiScaleObserver(n)
        basic_cache = ObserverCache(cache_size=1000)
        
        # Benchmark with optimized cache
        optimized_observer, optimized_cache = create_accelerated_observer(n)
        
        # Compare cache hit rates after some operations
        positions = list(range(2, min(20, int(n**0.5))))
        
        # Basic cache
        for _ in range(2):  # Two passes
            for pos in positions:
                basic_cache.get_observation(observer, pos)
        basic_stats = basic_cache.get_cache_statistics()
        
        # Optimized cache (already warmed)
        for _ in range(2):  # Two passes
            for pos in positions:
                optimized_cache.get_observation(optimized_observer, pos)
        optimized_stats = optimized_cache.get_cache_statistics()
        
        improvement = optimized_stats['total_hit_rate'] / (basic_stats['total_hit_rate'] + 0.01)
        total_improvement += improvement
    
    avg_improvement = total_improvement / len(test_numbers)
    assert avg_improvement > 1.0  # Should show improvement
    
    print(f"✓ Performance improvement (avg hit rate improvement: {avg_improvement:.2f}x)")

def test_cache_locality():
    """Test cache locality optimization"""
    n = 1000
    observer = MultiScaleObserver(n)
    cache = ObserverCache.create_optimized(n)
    
    # Unsorted positions
    positions_unsorted = [20, 5, 15, 2, 30, 10, 25]
    
    # Clear and test unsorted
    cache.clear()
    start = time.time()
    for pos in positions_unsorted:
        cache.get_observation(observer, pos)
    time_unsorted = time.time() - start
    
    # Clear and test sorted (via batch_observe)
    cache.clear()
    start = time.time()
    cache.batch_observe(observer, positions_unsorted)
    time_sorted = time.time() - start
    
    # Sorted should be similar or slightly better due to cache locality
    print(f"✓ Cache locality (unsorted: {time_unsorted:.4f}s, sorted: {time_sorted:.4f}s)")

def test_combined_benchmark():
    """Test combined benchmark with all tuning"""
    n = 10000
    
    # Run benchmark
    results = benchmark_acceleration(n, iterations=50)
    
    # Should show good speedup
    assert results['speedup_observe'] > 5.0
    assert results['speedup_gradient'] > 1.5
    
    stats = results['cache_stats']
    print(f"✓ Combined benchmark:")
    print(f"  - Observation speedup: {results['speedup_observe']:.2f}x")
    print(f"  - Gradient speedup: {results['speedup_gradient']:.2f}x")
    print(f"  - Cache hit rate: {stats['total_hit_rate']:.2%}")

def run_all_tests():
    """Run all performance tuning tests"""
    print("Testing Axiom 4 Performance Tuning...")
    print("-" * 50)
    
    test_batch_observe()
    test_warm_cache_effectiveness()
    test_gradient_precomputation()
    test_intelligent_cache_sizing()
    test_performance_improvement()
    test_cache_locality()
    test_combined_benchmark()
    
    print("-" * 50)
    print("All performance tuning tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
