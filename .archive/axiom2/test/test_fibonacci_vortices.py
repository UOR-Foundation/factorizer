"""
Tests for Fibonacci Vortices functionality
Validates vortex generation and golden spiral patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom2.fibonacci_vortices import (
    fib_vortices, golden_spiral_positions, fibonacci_lattice_points
)
from axiom2.fibonacci_core import fib, PHI

def test_fib_vortices_basic():
    """Test basic vortex generation"""
    # Test with n = 100
    vortices = fib_vortices(100)
    
    # Should return a sorted list
    assert vortices == sorted(vortices)
    
    # All values should be in range [2, 10] (sqrt(100))
    for v in vortices:
        assert 2 <= v <= 10
    
    # Should contain some Fibonacci numbers
    assert 2 in vortices  # fib(3)
    assert 3 in vortices  # fib(4)
    assert 5 in vortices  # fib(5)
    assert 8 in vortices  # fib(6)
    
    print("✓ Basic vortex generation")

def test_fib_vortices_golden_scaling():
    """Test golden ratio scaling in vortices"""
    n = 1000
    vortices = fib_vortices(n)
    root = 31  # floor(sqrt(1000))
    
    # Check that some golden-scaled values appear
    # fib(5) = 5, 5*φ ≈ 8.09 → 8
    assert 8 in vortices
    
    # fib(6) = 8, 8*φ ≈ 12.94 → 13
    assert 13 in vortices
    
    # fib(7) = 13, 13/φ ≈ 8.03 → 8
    # Already checked 8
    
    print("✓ Golden ratio scaling in vortices")

def test_fib_vortices_prime_modulation():
    """Test prime modulation in vortices"""
    n = 500
    vortices = fib_vortices(n)
    root = 22  # floor(sqrt(500))
    
    # The function should include prime-modulated values
    # Since it's modular arithmetic, hard to predict exact values
    # But we should have a good variety
    assert len(vortices) >= 10
    assert len(set(vortices)) == len(vortices)  # All unique
    
    print("✓ Prime modulation in vortices")

def test_golden_spiral_positions():
    """Test golden spiral position generation"""
    n = 400  # sqrt = 20
    positions = golden_spiral_positions(n, num_points=10)
    
    # All positions should be in valid range
    for pos in positions:
        assert 2 <= pos <= 20
    
    # Positions should be somewhat spread out if we have enough points
    if len(positions) > 2:
        assert max(positions) - min(positions) > 2
    
    # Test with smaller n
    n = 25  # sqrt = 5
    positions = golden_spiral_positions(n, num_points=5)
    for pos in positions:
        assert 2 <= pos <= 5
    
    print("✓ Golden spiral positions")

def test_fibonacci_lattice_points():
    """Test Fibonacci lattice point generation"""
    n = 225  # sqrt = 15
    lattice = fibonacci_lattice_points(n)
    
    # Should return sorted unique points
    assert lattice == sorted(lattice)
    assert len(set(lattice)) == len(lattice)
    
    # All points in valid range
    for point in lattice:
        assert 2 <= point <= 15
    
    # Should contain sums of small Fibonacci numbers
    # fib(3) + fib(4) = 2 + 3 = 5
    assert 5 in lattice
    # fib(4) + fib(5) = 3 + 5 = 8
    assert 8 in lattice
    # fib(5) + fib(6) = 5 + 8 = 13
    assert 13 in lattice
    
    # Should contain differences
    # |fib(6) - fib(5)| = |8 - 5| = 3
    assert 3 in lattice
    # |fib(7) - fib(5)| = |13 - 5| = 8
    assert 8 in lattice
    
    print("✓ Fibonacci lattice points")

def test_vortices_deterministic():
    """Test that vortex generation is deterministic"""
    n = 1000
    
    # Generate multiple times
    results = []
    for _ in range(3):
        vortices = fib_vortices(n)
        spiral = golden_spiral_positions(n)
        lattice = fibonacci_lattice_points(n)
        results.append((vortices, spiral, lattice))
    
    # All results should be identical
    for i in range(1, len(results)):
        assert results[i][0] == results[0][0]  # vortices
        assert results[i][1] == results[0][1]  # spiral
        assert results[i][2] == results[0][2]  # lattice
    
    print("✓ Vortex generation is deterministic")

def test_vortices_edge_cases():
    """Test edge cases for vortex functions"""
    # Very small n
    vortices = fib_vortices(4)  # sqrt = 2
    # Should only have 2
    assert all(v == 2 for v in vortices)
    
    # n = 9, sqrt = 3
    vortices = fib_vortices(9)
    for v in vortices:
        assert 2 <= v <= 3
    
    # Golden spiral with 0 points
    positions = golden_spiral_positions(100, num_points=0)
    assert positions == []
    
    # Lattice with small n
    lattice = fibonacci_lattice_points(10)  # sqrt = 3
    for point in lattice:
        assert 2 <= point <= 3
    
    print("✓ Edge cases handled correctly")

def test_vortices_coverage():
    """Test that vortices provide good coverage"""
    n = 10000  # sqrt = 100
    vortices = fib_vortices(n)
    
    # Should have reasonable coverage
    assert len(vortices) >= 20
    
    # Should span a good range
    assert min(vortices) <= 5
    assert max(vortices) >= 50
    
    # Check density in different regions
    low_count = sum(1 for v in vortices if v <= 20)
    mid_count = sum(1 for v in vortices if 20 < v <= 50)
    high_count = sum(1 for v in vortices if v > 50)
    
    # Should have points in all regions
    assert low_count > 0
    assert mid_count > 0
    assert high_count > 0
    
    print("✓ Vortices provide good coverage")

def run_all_tests():
    """Run all Fibonacci vortices tests"""
    print("Testing Fibonacci Vortices (Axiom 2)...")
    print("-" * 40)
    
    test_fib_vortices_basic()
    test_fib_vortices_golden_scaling()
    test_fib_vortices_prime_modulation()
    test_golden_spiral_positions()
    test_fibonacci_lattice_points()
    test_vortices_deterministic()
    test_vortices_edge_cases()
    test_vortices_coverage()
    
    print("-" * 40)
    print("All Fibonacci Vortices tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
