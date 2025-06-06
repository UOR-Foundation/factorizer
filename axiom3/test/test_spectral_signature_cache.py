"""
Tests for Spectral Signature Cache acceleration
Validates caching and pure axiomatic implementation
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom3.spectral_signature_cache import SpectralSignatureCache
from axiom3.spectral_core import spectral_vector
from axiom3.coherence import coherence
from axiom3.interference import prime_fib_interference, interference_extrema
from axiom3.fold_topology import fold_energy

def test_spectral_vector_caching():
    """Test that spectral vectors are cached correctly"""
    cache = SpectralSignatureCache(cache_size=100)
    
    # First access - should miss cache
    s1 = cache.get_spectral_vector(100)
    assert cache.cache_misses > 0
    initial_misses = cache.cache_misses
    
    # Second access - should hit cache
    s2 = cache.get_spectral_vector(100)
    assert s1 == s2
    assert cache.cache_hits > 0
    assert cache.cache_misses == initial_misses  # No new miss
    
    print("✓ Spectral vector caching works correctly")

def test_coherence_symmetry():
    """Test coherence caching with symmetry exploitation"""
    cache = SpectralSignatureCache()
    
    # C(a,b,n) should equal C(b,a,n)
    c1 = cache.get_coherence(3, 5, 15)
    c2 = cache.get_coherence(5, 3, 15)
    assert c1 == c2
    
    # Both should be cache hits after first computation
    initial_hits = cache.cache_hits
    c3 = cache.get_coherence(3, 5, 15)
    assert c3 == c1
    assert cache.cache_hits > initial_hits
    
    print("✓ Coherence symmetry exploitation works")

def test_interference_pattern_caching():
    """Test interference pattern caching"""
    cache = SpectralSignatureCache()
    
    # Get interference pattern
    pattern1, extrema1 = cache.get_interference_pattern(35)
    initial_misses = cache.cache_misses
    
    # Second access should hit cache
    pattern2, extrema2 = cache.get_interference_pattern(35)
    assert pattern1 == pattern2
    assert extrema1 == extrema2
    assert cache.cache_misses == initial_misses
    
    # Verify extrema are correct
    assert all(isinstance(pos, int) for pos in extrema1)
    assert all(2 <= pos <= 5 for pos in extrema1)  # sqrt(35) ≈ 5.9
    
    print("✓ Interference pattern caching works")

def test_fold_energy_map():
    """Test fold energy caching and computation"""
    cache = SpectralSignatureCache()
    
    # Test with n=15 (factors 3, 5)
    e3 = cache.get_fold_energy(15, 3)
    assert e3 != float('inf')  # 3 divides 15
    
    e4 = cache.get_fold_energy(15, 4)
    assert e4 == float('inf')  # 4 doesn't divide 15
    
    e5 = cache.get_fold_energy(15, 5)
    assert e5 != float('inf')  # 5 divides 15
    
    # Check caching
    initial_hits = cache.cache_hits
    e3_cached = cache.get_fold_energy(15, 3)
    assert e3_cached == e3
    assert cache.cache_hits > initial_hits
    
    print("✓ Fold energy map works correctly")

def test_sharp_folds_identification():
    """Test sharp fold candidate identification"""
    cache = SpectralSignatureCache()
    
    # For n=35 (5×7), sharp folds should include factors
    sharp = cache.get_sharp_folds(35)
    
    # At least one factor should be in the sharp folds
    assert 5 in sharp or 7 in sharp, f"Expected 5 or 7 in sharp folds, got {sharp}"
    
    # Should return some candidates
    assert len(sharp) > 0
    
    # Should be sorted by energy (lowest first)
    energies = [cache.get_fold_energy(35, x) for x in sharp]
    assert energies == sorted(energies)
    
    # Check that returned positions are valid
    sqrt_35 = int(35**0.5)
    assert all(2 <= x <= sqrt_35 for x in sharp)
    
    print("✓ Sharp fold identification works")

def test_priority_precomputation():
    """Test pre-computation of priority numbers"""
    cache = SpectralSignatureCache()
    
    # Check that Fibonacci numbers are pre-cached
    # fib(6) = 8, fib(7) = 13, fib(8) = 21
    assert 8 in cache.spectral_cache
    assert 13 in cache.spectral_cache
    assert 21 in cache.spectral_cache
    
    # Check that small primes are pre-cached
    assert 2 in cache.spectral_cache
    assert 3 in cache.spectral_cache
    assert 5 in cache.spectral_cache
    assert 7 in cache.spectral_cache
    
    # Check powers of 2
    assert 4 in cache.spectral_cache
    assert 8 in cache.spectral_cache
    assert 16 in cache.spectral_cache
    
    print("✓ Priority number pre-computation works")

def test_lru_eviction():
    """Test LRU cache eviction"""
    cache = SpectralSignatureCache(cache_size=10)
    
    # Fill cache beyond limit
    for i in range(15):
        cache.get_spectral_vector(i + 100)
    
    # Cache should not exceed size limit
    assert len(cache.spectral_cache) <= 10
    
    # Early entries should be evicted
    assert 100 not in cache.spectral_cache
    
    # Recent entries should remain
    assert 114 in cache.spectral_cache
    
    print("✓ LRU eviction works correctly")

def test_exact_computation():
    """Verify exact computation (no approximation)"""
    cache = SpectralSignatureCache()
    
    # Get spectral vector through cache
    s_cached = cache.get_spectral_vector(123)
    
    # Compute directly
    s_direct = spectral_vector(123)
    
    # Should be exactly equal
    assert s_cached == s_direct
    
    # Test coherence computation
    c_cached = cache.get_coherence(3, 41, 123)
    
    # Compute directly
    s3 = spectral_vector(3)
    s41 = spectral_vector(41)
    s123 = spectral_vector(123)
    
    import math
    diff_squared = sum((s3[i] + s41[i] - 2*s123[i])**2 
                      for i in range(len(s3)))
    c_direct = math.exp(-diff_squared)
    
    assert abs(c_cached - c_direct) < 1e-10
    
    print("✓ Exact computation verified (no approximation)")

def test_precompute_for_n():
    """Test minimal pre-computation for specific n"""
    cache = SpectralSignatureCache()
    
    # Clear existing cache to test fresh (except priority numbers)
    priority_cache = dict(cache.spectral_cache)  # Save priority numbers
    cache.spectral_cache.clear()
    cache.spectral_cache.update(priority_cache)  # Restore priority numbers
    
    # Pre-compute for n=77
    cache.precompute_for_n(77)
    
    # With minimal pre-computation, only n itself should be cached
    assert 77 in cache.spectral_cache
    
    # Factors and other data should NOT be pre-cached (lazy loading)
    # This avoids overhead for pure acceleration
    
    print("✓ Minimal pre-computation for n works")

def test_performance_improvement():
    """Test performance improvement with caching - must always be positive"""
    test_sizes = [100, 1000, 10000]
    
    for n in test_sizes:
        # Without cache - compute multiple times
        start = time.time()
        for _ in range(50):
            s = spectral_vector(n)
        time_without = time.time() - start
        
        # With cache
        cache = SpectralSignatureCache.create_optimized(n)
        start = time.time()
        for _ in range(50):
            s = cache.get_spectral_vector(n)
        time_with = time.time() - start
        
        speedup = time_without / time_with
        
        print(f"\n  n={n}:")
        print(f"    Time without cache: {time_without*1000:.2f} ms")
        print(f"    Time with cache: {time_with*1000:.2f} ms")
        print(f"    Speedup: {speedup:.1f}x")
        
        # Check cache statistics
        stats = cache.get_cache_statistics()
        print(f"    Cache hit rate: {stats['hit_rate']:.1%}")
        
        # CRITICAL: Acceleration must ALWAYS be positive
        # This is a requirement of the pure UOR/Prime axiomatic approach
        assert speedup > 1.0, f"NEGATIVE ACCELERATION for n={n}! Speedup={speedup:.2f}x < 1.0x"
    
    print("\n✓ Performance improvement verified - all speedups positive")

def test_create_optimized():
    """Test optimized cache creation"""
    # Small number
    cache_small = SpectralSignatureCache.create_optimized(100)
    assert cache_small.cache_size == 5000  # n < 10000
    assert 100 in cache_small.spectral_cache
    
    # Medium number  
    cache_medium = SpectralSignatureCache.create_optimized(5000)
    assert cache_medium.cache_size == 5000  # n < 10000
    
    # Large number
    cache_large = SpectralSignatureCache.create_optimized(50000)
    assert cache_large.cache_size == 10000  # n < 1000000
    
    # Very large number
    cache_xlarge = SpectralSignatureCache.create_optimized(2000000)
    assert cache_xlarge.cache_size == 20000  # n >= 1000000
    
    print("✓ Optimized cache creation works")

def test_deterministic_results():
    """Test that cached results are deterministic"""
    cache1 = SpectralSignatureCache()
    cache2 = SpectralSignatureCache()
    
    # Same inputs should give same outputs
    for n in [10, 77, 143, 1001]:
        s1 = cache1.get_spectral_vector(n)
        s2 = cache2.get_spectral_vector(n)
        assert s1 == s2
        
        pattern1, extrema1 = cache1.get_interference_pattern(n)
        pattern2, extrema2 = cache2.get_interference_pattern(n)
        assert pattern1 == pattern2
        assert extrema1 == extrema2
    
    print("✓ Deterministic results verified")

def run_all_tests():
    """Run all Spectral Signature Cache tests"""
    print("Testing Spectral Signature Cache...")
    print("-" * 40)
    
    test_spectral_vector_caching()
    test_coherence_symmetry()
    test_interference_pattern_caching()
    test_fold_energy_map()
    test_sharp_folds_identification()
    test_priority_precomputation()
    test_lru_eviction()
    test_exact_computation()
    test_precompute_for_n()
    test_performance_improvement()
    test_create_optimized()
    test_deterministic_results()
    
    print("-" * 40)
    print("All Spectral Signature Cache tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
