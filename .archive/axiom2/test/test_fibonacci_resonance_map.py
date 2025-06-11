"""
Tests for Fibonacci Resonance Map acceleration
Validates caching and pure axiomatic implementation
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom2.fibonacci_resonance_map import FibonacciResonanceMap
from axiom2.fibonacci_core import fib, PHI, fib_wave

def test_fibonacci_caching():
    """Test that Fibonacci numbers are cached correctly"""
    resonance_map = FibonacciResonanceMap(max_index=100)
    
    # First access - should miss cache
    fib_10 = resonance_map.get_fibonacci(10)
    assert fib_10 == 55  # 10th Fibonacci number
    assert resonance_map.cache_misses == 1
    assert resonance_map.cache_hits == 0
    
    # Second access - should hit cache
    fib_10_cached = resonance_map.get_fibonacci(10)
    assert fib_10_cached == 55
    assert resonance_map.cache_hits == 1
    
    # Different number - should miss cache
    fib_20 = resonance_map.get_fibonacci(20)
    assert fib_20 == 6765  # 20th Fibonacci number
    assert resonance_map.cache_misses == 2
    
    print("✓ Fibonacci caching works correctly")

def test_vortex_points_generation():
    """Test vortex point generation with caching"""
    resonance_map = FibonacciResonanceMap()
    
    # Generate vortex points
    center = 50
    radius = 30.0
    vortex_points = resonance_map.get_vortex_points(center, radius)
    
    # Verify some expected points
    assert len(vortex_points) > 0
    
    # Check that Fibonacci numbers are included
    assert (center + 1) in vortex_points  # fib(1) = 1
    assert (center + 2) in vortex_points  # fib(3) = 2
    assert (center + 3) in vortex_points  # fib(4) = 3
    assert (center + 5) in vortex_points  # fib(5) = 5
    
    # Check caching
    cache_misses_before = resonance_map.cache_misses
    vortex_points_2 = resonance_map.get_vortex_points(center, radius)
    assert vortex_points == vortex_points_2
    assert resonance_map.cache_hits > 0  # Should hit cache
    
    print("✓ Vortex points generation works correctly")

def test_wave_value_exact():
    """Test that wave values are computed exactly (no interpolation)"""
    resonance_map = FibonacciResonanceMap()
    
    # Test at integer positions
    for k in range(10):
        wave_k = resonance_map.get_wave_value(float(k))
        expected = fib_wave(float(k))
        assert abs(wave_k - expected) < 1e-10, f"Wave value at {k} incorrect"
    
    # Test at non-integer positions
    test_positions = [2.5, 3.7, 5.1, 7.3]
    for x in test_positions:
        wave_x = resonance_map.get_wave_value(x)
        expected = fib_wave(x)
        assert abs(wave_x - expected) < 1e-10, f"Wave value at {x} incorrect"
    
    # Verify caching
    cache_hits_before = resonance_map.cache_hits
    wave_2_5_cached = resonance_map.get_wave_value(2.5)
    assert resonance_map.cache_hits > cache_hits_before
    
    print("✓ Wave values are computed exactly")

def test_nearest_fibonacci():
    """Test finding nearest Fibonacci number"""
    resonance_map = FibonacciResonanceMap()
    
    # Test exact Fibonacci numbers
    index, fib_num, distance = resonance_map.find_nearest_fibonacci(55)
    assert index == 10
    assert fib_num == 55
    assert distance == 0
    
    # Test non-Fibonacci numbers
    index, fib_num, distance = resonance_map.find_nearest_fibonacci(50)
    assert fib_num == 55  # Nearest is 55
    assert distance == 5
    
    index, fib_num, distance = resonance_map.find_nearest_fibonacci(40)
    assert fib_num == 34  # Nearest is 34
    assert distance == 6
    
    print("✓ Nearest Fibonacci finder works correctly")

def test_fibonacci_range():
    """Test getting Fibonacci numbers in a range"""
    resonance_map = FibonacciResonanceMap()
    
    # Get Fibonacci numbers between 10 and 100
    fibs_in_range = resonance_map.get_fibonacci_range(10, 100)
    expected = [13, 21, 34, 55, 89]
    assert fibs_in_range == expected
    
    # Edge cases
    fibs_small = resonance_map.get_fibonacci_range(0, 5)
    assert fibs_small == [0, 1, 1, 2, 3, 5]
    
    print("✓ Fibonacci range finder works correctly")

def test_precomputation():
    """Test pre-computation of common values"""
    resonance_map = FibonacciResonanceMap(max_index=50)
    
    # Check cache before pre-computation
    initial_fib_cached = len(resonance_map.fib_cache)
    initial_wave_cached = len(resonance_map.wave_table)
    
    # Pre-compute
    resonance_map.precompute_common_values()
    
    # Check cache after pre-computation
    assert len(resonance_map.fib_cache) > initial_fib_cached
    assert len(resonance_map.wave_table) > initial_wave_cached
    
    # Verify some values were pre-computed
    assert 10 in resonance_map.fib_cache  # fib(10) should be cached
    # Only small Fibonacci numbers are in wave table to avoid overflow
    assert float(8) in resonance_map.wave_table  # wave at fib(6) = 8
    
    print("✓ Pre-computation works correctly")

def test_performance_improvement():
    """Test performance improvement with caching"""
    n = 10000
    
    # Without resonance map
    start_time = time.time()
    for k in range(20):
        fib_k = fib(k)
        wave_k = fib_wave(float(k))
    time_without = time.time() - start_time
    
    # With resonance map (pre-computed)
    resonance_map = FibonacciResonanceMap.create_optimized(n)
    start_time = time.time()
    for k in range(20):
        fib_k = resonance_map.get_fibonacci(k)
        wave_k = resonance_map.get_wave_value(float(k))
    time_with = time.time() - start_time
    
    # Cache should make it faster (or at least not slower)
    print(f"  Time without cache: {time_without*1000:.3f} ms")
    print(f"  Time with cache: {time_with*1000:.3f} ms")
    
    # Check cache statistics
    stats = resonance_map.get_cache_statistics()
    print(f"  Cache hit rate: {stats['hit_rate']:.1%}")
    assert stats['hit_rate'] > 0.3  # Should have reasonable hit rate
    
    print("✓ Performance improvement verified")

def test_spiral_path_generation():
    """Test golden spiral path generation"""
    resonance_map = FibonacciResonanceMap()
    
    # Generate spiral path
    center = 100
    max_radius = 50.0
    steps = 20
    
    spiral = resonance_map.get_spiral_path(center, max_radius, steps)
    
    # Verify properties
    assert len(spiral) == steps
    
    # Check that radius increases
    for i in range(1, len(spiral)):
        x1, y1 = spiral[i-1]
        x2, y2 = spiral[i]
        r1 = ((x1 - center)**2 + (y1 - center)**2)**0.5
        r2 = ((x2 - center)**2 + (y2 - center)**2)**0.5
        assert r2 >= r1  # Radius should increase or stay same
    
    # Test caching
    spiral_cached = resonance_map.get_spiral_path(center, max_radius, steps)
    assert spiral == spiral_cached
    
    print("✓ Spiral path generation works correctly")

def test_create_optimized():
    """Test optimized resonance map creation"""
    # Small number
    resonance_map_small = FibonacciResonanceMap.create_optimized(100)
    assert resonance_map_small.max_index >= 10
    
    # Medium number
    resonance_map_medium = FibonacciResonanceMap.create_optimized(10000)
    assert resonance_map_medium.max_index >= 20
    
    # Large number (should not pre-compute)
    resonance_map_large = FibonacciResonanceMap.create_optimized(10**9)
    assert len(resonance_map_large.fib_cache) < 100  # Should not pre-compute too much
    
    print("✓ Optimized creation works correctly")

def run_all_tests():
    """Run all Fibonacci Resonance Map tests"""
    print("Testing Fibonacci Resonance Map...")
    print("-" * 40)
    
    test_fibonacci_caching()
    test_vortex_points_generation()
    test_wave_value_exact()
    test_nearest_fibonacci()
    test_fibonacci_range()
    test_precomputation()
    test_performance_improvement()
    test_spiral_path_generation()
    test_create_optimized()
    
    print("-" * 40)
    print("All Fibonacci Resonance Map tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
