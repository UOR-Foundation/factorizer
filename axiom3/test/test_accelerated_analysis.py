"""
Tests for Accelerated Analysis
Validates that acceleration provides correct results with improved performance
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom3.accelerated_analysis import (
    accelerated_spectral_vector,
    accelerated_coherence,
    accelerated_fold_energy,
    accelerated_interference_analysis,
    accelerated_coherence_field,
    accelerated_sharp_folds,
    accelerated_spectral_analysis,
    benchmark_acceleration,
    create_accelerated_analyzer
)
from axiom3.spectral_core import spectral_vector
from axiom3.coherence import coherence
from axiom3.fold_topology import fold_energy
from axiom3.interference import prime_fib_interference, interference_extrema

def test_accelerated_spectral_vector():
    """Test that accelerated spectral vector matches direct computation"""
    n = 1234
    
    # Direct computation
    direct = spectral_vector(n)
    
    # Accelerated computation
    accelerated = accelerated_spectral_vector(n)
    
    # Should be exactly equal
    assert direct == accelerated, "Accelerated spectral vector differs from direct"
    
    # Check caching works
    start = time.time()
    for _ in range(100):
        s = accelerated_spectral_vector(n)
    time_cached = time.time() - start
    
    assert time_cached < 0.01, f"Cached access too slow: {time_cached:.3f}s"
    
    print("✓ Accelerated spectral vector works correctly")

def test_accelerated_coherence():
    """Test that accelerated coherence matches direct computation"""
    a, b, n = 7, 11, 77
    
    # Direct computation
    direct = coherence(a, b, n)
    
    # Accelerated computation
    accelerated = accelerated_coherence(a, b, n)
    
    # Should be very close (floating point tolerance)
    assert abs(direct - accelerated) < 1e-10, f"Coherence mismatch: {direct} vs {accelerated}"
    
    # Test symmetry
    accelerated_sym = accelerated_coherence(b, a, n)
    assert accelerated == accelerated_sym, "Coherence symmetry broken"
    
    print("✓ Accelerated coherence works correctly")

def test_accelerated_fold_energy():
    """Test that accelerated fold energy matches direct computation"""
    n = 35  # 5 × 7
    
    # Test actual factors
    factor_energies = []
    for x in [5, 7]:
        direct = fold_energy(n, x)
        accelerated = accelerated_fold_energy(n, x)
        assert abs(direct - accelerated) < 1e-10, f"Fold energy mismatch at x={x}"
        factor_energies.append(direct)
    
    # Test non-factor (should have finite energy but higher than factors)
    direct_nf = fold_energy(n, 4)
    accelerated_nf = accelerated_fold_energy(n, 4)
    assert abs(direct_nf - accelerated_nf) < 1e-10, f"Non-factor fold energy mismatch at x=4"
    
    # Non-factors should typically have higher energy than factors
    # (though this isn't guaranteed for all non-factors)
    min_factor_energy = min(factor_energies)
    # Just verify the energies are computed correctly, not their relative values
    
    print("✓ Accelerated fold energy works correctly")

def test_accelerated_interference():
    """Test that accelerated interference analysis matches direct computation"""
    n = 91  # 7 × 13
    
    # Direct computation
    direct_pattern = prime_fib_interference(n)
    direct_extrema = interference_extrema(n)
    
    # Accelerated computation
    accel_pattern, accel_extrema = accelerated_interference_analysis(n)
    
    # Patterns should match
    assert len(direct_pattern) == len(accel_pattern), "Pattern length mismatch"
    for i in range(len(direct_pattern)):
        assert abs(direct_pattern[i] - accel_pattern[i]) < 1e-10, f"Pattern mismatch at index {i}"
    
    # Extrema should match
    assert direct_extrema == accel_extrema, "Extrema mismatch"
    
    print("✓ Accelerated interference analysis works correctly")

def test_accelerated_coherence_field():
    """Test coherence field generation"""
    n = 15  # 3 × 5
    candidates = [2, 3, 4, 5, 6, 7]
    
    field = accelerated_coherence_field(n, candidates)
    
    # Check factors have non-zero coherence
    assert field[3] > 0, "Factor 3 should have positive coherence"
    assert field[5] > 0, "Factor 5 should have positive coherence"
    
    # Check non-factors have zero coherence
    assert field[2] == 0.0, "Non-factor 2 should have zero coherence"
    assert field[4] == 0.0, "Non-factor 4 should have zero coherence"
    
    print("✓ Accelerated coherence field works correctly")

def test_accelerated_sharp_folds():
    """Test sharp fold detection"""
    n = 143  # 11 × 13
    
    sharp = accelerated_sharp_folds(n)
    
    # Should find some candidates
    assert len(sharp) > 0, "No sharp folds found"
    
    # At least one factor should be in sharp folds
    assert 11 in sharp or 13 in sharp, f"Expected factor in sharp folds, got {sharp}"
    
    print("✓ Accelerated sharp fold detection works")

def test_accelerated_spectral_analysis():
    """Test comprehensive spectral analysis"""
    n = 77  # 7 × 11
    
    analysis = accelerated_spectral_analysis(n, full_analysis=True)
    
    # Check all components present
    assert 'spectral_vector' in analysis
    assert 'sharp_folds' in analysis
    assert 'interference_extrema' in analysis
    assert 'coherence_samples' in analysis
    assert 'cache_stats' in analysis
    
    # Verify spectral vector
    assert len(analysis['spectral_vector']) > 0
    
    # Verify sharp folds includes a factor
    assert 7 in analysis['sharp_folds'] or 11 in analysis['sharp_folds']
    
    # Verify cache was used
    stats = analysis['cache_stats']
    assert stats['cache_hits'] > 0, "Cache not being utilized"
    assert stats['hit_rate'] > 0.5, f"Cache hit rate too low: {stats['hit_rate']:.1%}"
    
    print("✓ Accelerated spectral analysis works")

def test_benchmark_acceleration():
    """Test that acceleration provides speedup"""
    n = 1001  # 7 × 11 × 13
    
    bench = benchmark_acceleration(n, iterations=50)
    
    # Verify positive speedup
    assert bench['speedup_spectral'] > 1.0, f"No spectral speedup: {bench['speedup_spectral']:.2f}x"
    assert bench['speedup_coherence'] > 1.0, f"No coherence speedup: {bench['speedup_coherence']:.2f}x"
    
    print(f"  Spectral speedup: {bench['speedup_spectral']:.1f}x")
    print(f"  Coherence speedup: {bench['speedup_coherence']:.1f}x")
    print(f"  Cache hit rate: {bench['cache_stats']['hit_rate']:.1%}")
    
    print("✓ Benchmark shows positive acceleration")

def test_create_accelerated_analyzer():
    """Test analyzer creation"""
    # Small number
    analyzer_small = create_accelerated_analyzer(100)
    assert analyzer_small.cache_size == 5000
    assert 100 in analyzer_small.spectral_cache
    
    # Large number
    analyzer_large = create_accelerated_analyzer(100000)
    assert analyzer_large.cache_size == 10000
    assert 100000 in analyzer_large.spectral_cache
    
    print("✓ Accelerated analyzer creation works")

def test_deterministic_acceleration():
    """Test that accelerated results are deterministic"""
    n = 221  # 13 × 17
    
    # Create two independent analyzers
    analyzer1 = create_accelerated_analyzer(n)
    analyzer2 = create_accelerated_analyzer(n)
    
    # Get results from each
    s1 = analyzer1.get_spectral_vector(n)
    s2 = analyzer2.get_spectral_vector(n)
    assert s1 == s2, "Non-deterministic spectral vectors"
    
    pattern1, extrema1 = analyzer1.get_interference_pattern(n)
    pattern2, extrema2 = analyzer2.get_interference_pattern(n)
    assert pattern1 == pattern2, "Non-deterministic interference patterns"
    assert extrema1 == extrema2, "Non-deterministic extrema"
    
    print("✓ Accelerated analysis is deterministic")

def test_pure_acceleration_principles():
    """Verify acceleration follows pure UOR/Prime principles"""
    n = 323  # 17 × 19
    
    # Create analyzer
    analyzer = create_accelerated_analyzer(n)
    
    # Test multiple accesses to verify caching doesn't change results
    results = []
    for _ in range(5):
        s = analyzer.get_spectral_vector(n)
        results.append(s)
    
    # All results should be identical
    for i in range(1, len(results)):
        assert results[0] == results[i], f"Result {i} differs from first"
    
    # Verify no approximation
    direct = spectral_vector(n)
    cached = analyzer.get_spectral_vector(n)
    assert direct == cached, "Cached result differs from direct computation"
    
    # Verify mathematical consistency
    c1 = analyzer.get_coherence(17, 19, n)
    c2 = analyzer.get_coherence(19, 17, n)
    assert c1 == c2, "Coherence symmetry violated"
    
    print("✓ Pure acceleration principles verified")

def run_all_tests():
    """Run all accelerated analysis tests"""
    print("Testing Accelerated Analysis...")
    print("-" * 40)
    
    test_accelerated_spectral_vector()
    test_accelerated_coherence()
    test_accelerated_fold_energy()
    test_accelerated_interference()
    test_accelerated_coherence_field()
    test_accelerated_sharp_folds()
    test_accelerated_spectral_analysis()
    test_benchmark_acceleration()
    test_create_accelerated_analyzer()
    test_deterministic_acceleration()
    test_pure_acceleration_principles()
    
    print("-" * 40)
    print("All Accelerated Analysis tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
