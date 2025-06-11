"""
Tests for Fibonacci Core functionality
Validates golden ratio constants and Fibonacci operations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
from axiom2.fibonacci_core import (
    SQRT5, PHI, PSI, GOLDEN_ANGLE,
    fib, fib_wave, lucas, is_fibonacci, nearest_fibonacci_index
)

def test_constants():
    """Test golden ratio constants"""
    # Check SQRT5
    assert abs(SQRT5 - math.sqrt(5)) < 1e-10
    print(f"✓ SQRT5 = {SQRT5:.10f}")
    
    # Check PHI (golden ratio)
    expected_phi = (1 + math.sqrt(5)) / 2
    assert abs(PHI - expected_phi) < 1e-10
    assert abs(PHI - 1.6180339887) < 1e-9
    print(f"✓ PHI = {PHI:.10f}")
    
    # Check PSI (conjugate)
    expected_psi = (1 - math.sqrt(5)) / 2
    assert abs(PSI - expected_psi) < 1e-10
    assert abs(PSI - (-0.6180339887)) < 1e-9
    print(f"✓ PSI = {PSI:.10f}")
    
    # Check golden angle
    expected_angle = 2 * math.pi * (PHI - 1)
    assert abs(GOLDEN_ANGLE - expected_angle) < 1e-10
    print(f"✓ GOLDEN_ANGLE = {GOLDEN_ANGLE:.10f} radians")
    
    # Verify PHI properties
    assert abs(PHI * PHI - PHI - 1) < 1e-10  # φ² = φ + 1
    assert abs(1/PHI - (PHI - 1)) < 1e-10   # 1/φ = φ - 1
    print("✓ Golden ratio properties verified")

def test_fibonacci_basic():
    """Test basic Fibonacci number generation"""
    # First 20 Fibonacci numbers
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
    
    for k, expected_val in enumerate(expected):
        assert fib(k) == expected_val, f"fib({k}) = {fib(k)}, expected {expected_val}"
    
    print("✓ First 20 Fibonacci numbers correct")

def test_fibonacci_large():
    """Test larger Fibonacci numbers"""
    # Test some known large Fibonacci numbers
    assert fib(30) == 832040
    assert fib(40) == 102334155
    assert fib(50) == 12586269025
    assert fib(60) == 1548008755920
    assert fib(70) == 190392490709135
    
    print("✓ Large Fibonacci numbers correct")

def test_fibonacci_edge_cases():
    """Test edge cases for Fibonacci"""
    # Test negative input
    try:
        fib(-1)
        assert False, "Should raise ValueError for negative input"
    except ValueError:
        pass
    
    # Test F(0) = 0, F(1) = 1
    assert fib(0) == 0
    assert fib(1) == 1
    
    print("✓ Fibonacci edge cases handled")

def test_fib_wave():
    """Test continuous Fibonacci wave function"""
    # Should match integer Fibonacci at integer points
    for k in range(20):
        wave_val = fib_wave(k)
        fib_val = fib(k)
        assert abs(wave_val - fib_val) < 1e-10, f"fib_wave({k}) = {wave_val}, fib({k}) = {fib_val}"
    
    # Test fractional values
    assert fib_wave(0.5) > 0
    # fib(1) = 1, fib(2) = 1, so test between 2 and 3
    assert fib_wave(2.5) > fib(2) and fib_wave(2.5) < fib(3)
    # Test between 3 and 4
    assert fib_wave(3.5) > fib(3) and fib_wave(3.5) < fib(4)
    
    # Test negative values
    assert abs(fib_wave(0)) < 1e-10  # Should be 0
    
    print("✓ Fibonacci wave function correct")

def test_lucas():
    """Test Lucas number generation"""
    # First 15 Lucas numbers
    expected = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843]
    
    for k, expected_val in enumerate(expected):
        assert lucas(k) == expected_val, f"lucas({k}) = {lucas(k)}, expected {expected_val}"
    
    # Verify Lucas-Fibonacci relationship: L(n) = F(n-1) + F(n+1)
    for n in range(2, 10):
        assert lucas(n) == fib(n-1) + fib(n+1)
    
    print("✓ Lucas numbers correct")

def test_is_fibonacci():
    """Test Fibonacci number detection"""
    # Test known Fibonacci numbers
    fib_numbers = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    for f in fib_numbers:
        assert is_fibonacci(f), f"{f} should be recognized as Fibonacci"
    
    # Test non-Fibonacci numbers
    non_fib = [4, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25]
    for n in non_fib:
        assert not is_fibonacci(n), f"{n} should not be Fibonacci"
    
    # Test edge cases
    assert not is_fibonacci(-1)
    assert is_fibonacci(0)
    
    print("✓ Fibonacci detection correct")

def test_nearest_fibonacci_index():
    """Test finding nearest Fibonacci index"""
    # Test exact Fibonacci numbers
    assert nearest_fibonacci_index(0) == 0
    assert nearest_fibonacci_index(1) in [1, 2]  # Could be 1 or 2 (both equal 1)
    assert nearest_fibonacci_index(5) == 5
    assert nearest_fibonacci_index(13) == 7
    assert nearest_fibonacci_index(89) == 11
    
    # Test between Fibonacci numbers
    assert nearest_fibonacci_index(4) == 5  # Closer to fib(5)=5
    assert nearest_fibonacci_index(6) == 5  # Closer to fib(5)=5
    assert nearest_fibonacci_index(7) == 6  # Closer to fib(6)=8
    assert nearest_fibonacci_index(10) == 6  # Closer to fib(6)=8
    assert nearest_fibonacci_index(11) == 7  # Closer to fib(7)=13
    
    print("✓ Nearest Fibonacci index detection correct")

def test_deterministic():
    """Test that all operations are deterministic"""
    # Run multiple times to ensure same results
    for _ in range(5):
        assert fib(20) == 6765
        assert abs(fib_wave(10.5) - 69.96570594) < 1e-6
        assert lucas(10) == 123
        assert is_fibonacci(89) == True
        assert nearest_fibonacci_index(100) == 11
    
    print("✓ All operations are deterministic")

def run_all_tests():
    """Run all Fibonacci core tests"""
    print("Testing Fibonacci Core (Axiom 2)...")
    print("-" * 40)
    
    test_constants()
    test_fibonacci_basic()
    test_fibonacci_large()
    test_fibonacci_edge_cases()
    test_fib_wave()
    test_lucas()
    test_is_fibonacci()
    test_nearest_fibonacci_index()
    test_deterministic()
    
    print("-" * 40)
    print("All Fibonacci Core tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
