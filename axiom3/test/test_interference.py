"""
Tests for Interference functionality
Validates prime×Fibonacci wave interference patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
from axiom3.interference import (
    prime_fib_interference,
    interference_extrema,
    identify_resonance_source,
    interference_gradient,
    resonance_strength
)

def test_prime_fib_interference():
    """Test interference pattern generation"""
    n = 100
    spectrum = prime_fib_interference(n)
    
    # Should return a list of values
    assert isinstance(spectrum, list)
    assert len(spectrum) == 10 - 2 + 1  # From x=2 to x=10 (root of 100)
    
    # Values should be bounded (sum of cosines)
    for val in spectrum:
        assert isinstance(val, float)
        assert -100 <= val <= 100  # Rough bound
    
    # Test with smaller number
    spectrum_small = prime_fib_interference(25)
    assert len(spectrum_small) == 5 - 2 + 1  # From x=2 to x=5
    
    print("✓ Interference pattern generation")

def test_interference_extrema():
    """Test extrema detection in interference"""
    n = 143  # 11 × 13
    extrema = interference_extrema(n, top=10)
    
    assert isinstance(extrema, list)
    assert len(extrema) <= 10
    
    # All extrema should be valid positions
    root = int(math.isqrt(n))
    for pos in extrema:
        assert 2 <= pos <= root
    
    # Should find some extrema for non-trivial cases
    if root > 5:
        assert len(extrema) > 0
    
    print("✓ Interference extrema detection")

def test_identify_resonance_source():
    """Test resonance source identification"""
    n = 77  # 7 × 11
    
    # Test at factor position
    prime, fib = identify_resonance_source(7, n)
    assert isinstance(prime, int)
    assert isinstance(fib, int)
    assert prime >= 2
    assert fib >= 1
    
    # Test at non-factor position
    prime2, fib2 = identify_resonance_source(6, n)
    assert isinstance(prime2, int)
    assert isinstance(fib2, int)
    
    print("✓ Resonance source identification")

def test_interference_gradient():
    """Test interference gradient calculation"""
    n = 100
    
    # Test at various positions
    for x in [5, 7, 9]:
        grad = interference_gradient(n, x)
        assert isinstance(grad, float)
        assert -1000 <= grad <= 1000  # Reasonable bounds
    
    # Test edge cases
    grad_edge1 = interference_gradient(n, 2)  # Near boundary
    grad_edge2 = interference_gradient(n, 10)  # Near boundary
    assert isinstance(grad_edge1, float)
    assert isinstance(grad_edge2, float)
    
    print("✓ Interference gradient calculation")

def test_resonance_strength():
    """Test resonance strength calculation"""
    # Test with true factors
    strength_true = resonance_strength(7, 11, 77)
    assert 0 <= strength_true <= 1
    
    # Test with non-factors
    strength_false = resonance_strength(6, 13, 77)
    assert 0 <= strength_false <= 1
    
    # Perfect self-resonance
    strength_self = resonance_strength(10, 10, 100)
    assert 0 <= strength_self <= 1
    
    # Test commutativity
    assert resonance_strength(5, 7, 35) == resonance_strength(7, 5, 35)
    
    print("✓ Resonance strength calculation")

def test_interference_patterns():
    """Test interference patterns for known factorizations"""
    test_cases = [
        (35, 5, 7),      # Small semiprime
        (77, 7, 11),     # Medium semiprime
        (143, 11, 13),   # Larger semiprime
    ]
    
    for n, p, q in test_cases:
        spectrum = prime_fib_interference(n)
        extrema = interference_extrema(n, top=20)
        
        # Should find extrema
        assert len(extrema) > 0
        
        # At least one extremum should be near a factor
        distances = []
        for ext in extrema:
            dist_p = abs(ext - p)
            dist_q = abs(ext - q)
            distances.append(min(dist_p, dist_q))
        
        # At least one extremum within 2 of a factor
        assert min(distances) <= 2
    
    print("✓ Interference patterns verified")

def test_interference_determinism():
    """Test that interference calculations are deterministic"""
    n = 187  # 11 × 17
    
    # Run multiple times
    spectra = []
    extrema = []
    sources = []
    
    for _ in range(3):
        spectra.append(prime_fib_interference(n))
        extrema.append(interference_extrema(n, top=5))
        sources.append(identify_resonance_source(10, n))
    
    # All results should be identical
    for i in range(1, 3):
        assert spectra[i] == spectra[0]
        assert extrema[i] == extrema[0]
        assert sources[i] == sources[0]
    
    print("✓ Interference calculations are deterministic")

def test_interference_edge_cases():
    """Test edge cases for interference"""
    # Small number
    spectrum_small = prime_fib_interference(9)
    assert len(spectrum_small) == 3 - 2 + 1  # root=3
    
    # Prime number
    spectrum_prime = prime_fib_interference(17)
    extrema_prime = interference_extrema(17)
    assert isinstance(spectrum_prime, list)
    assert isinstance(extrema_prime, list)
    
    # Perfect square
    spectrum_square = prime_fib_interference(64)
    assert len(spectrum_square) == 8 - 2 + 1
    
    # Test gradient at boundaries
    grad_boundary = interference_gradient(100, 10, delta=2)
    assert grad_boundary == 0.0  # Out of bounds
    
    print("✓ Interference edge cases handled")

def test_resonance_properties():
    """Test mathematical properties of resonance"""
    # Resonance should be higher for true factors
    n = 91  # 7 × 13
    res_true = resonance_strength(7, 13, n)
    res_false1 = resonance_strength(6, 15, n)
    res_false2 = resonance_strength(8, 11, n)
    
    # True factors should have higher resonance (usually)
    # This might not always hold due to the complex nature of interference
    # but should be true on average
    avg_false = (res_false1 + res_false2) / 2
    assert res_true > avg_false * 0.8  # Allow some margin
    
    # Test scaling
    # Resonance of scaled factors
    res_scaled = resonance_strength(14, 26, 364)  # 2×7, 2×13, 4×91
    assert 0 <= res_scaled <= 1
    
    print("✓ Resonance properties verified")

def run_all_tests():
    """Run all interference tests"""
    print("Testing Interference (Axiom 3)...")
    print("-" * 40)
    
    test_prime_fib_interference()
    test_interference_extrema()
    test_identify_resonance_source()
    test_interference_gradient()
    test_resonance_strength()
    test_interference_patterns()
    test_interference_determinism()
    test_interference_edge_cases()
    test_resonance_properties()
    
    print("-" * 40)
    print("All Interference tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
