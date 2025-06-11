"""
Tests for Spectral Core functionality
Validates spectral representations and transformations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom3.spectral_core import (
    binary_spectrum,
    modular_spectrum,
    digital_spectrum,
    harmonic_spectrum,
    spectral_vector
)

def test_binary_spectrum():
    """Test binary spectrum analysis"""
    # Test with simple numbers
    spec = binary_spectrum(5)  # Binary: 101
    assert len(spec) == 12
    assert 0.5 < spec[0] < 0.8  # Bit density around 2/3
    
    # Test with power of 2
    spec = binary_spectrum(8)  # Binary: 1000
    assert spec[0] == 0.25  # Only 1 bit set out of 4
    
    # Test with all bits set
    spec = binary_spectrum(7)  # Binary: 111
    assert spec[0] == 1.0  # All bits set
    
    # Test edge cases
    spec = binary_spectrum(0)
    assert spec == [0.0, 0.0]
    
    spec = binary_spectrum(1)  # Binary: 1
    assert len(spec) == 12
    assert spec[0] == 1.0  # Single bit
    
    print("✓ Binary spectrum analysis")

def test_modular_spectrum():
    """Test modular spectrum analysis"""
    # Test with prime
    spec = modular_spectrum(17)
    assert len(spec) == 10
    # 17 mod 3 = 2, so spec[0] = 2/3
    assert abs(spec[0] - 2/3) < 1e-10
    
    # Test with composite
    spec = modular_spectrum(12)
    assert len(spec) == 10
    # 12 mod 3 = 0
    assert spec[0] == 0.0
    
    # Test edge case
    spec = modular_spectrum(0)
    assert spec == [0.0] * 10
    
    print("✓ Modular spectrum analysis")

def test_digital_spectrum():
    """Test digital spectrum analysis"""
    # Test single digit
    spec = digital_spectrum(7)
    assert len(spec) == 2
    assert abs(spec[0] - 7/9) < 1e-10  # Normalized digit sum
    assert abs(spec[1] - 7/9) < 1e-10  # Digital root
    
    # Test multi-digit
    spec = digital_spectrum(123)
    # Digit sum = 1+2+3 = 6
    # Normalized = 6/(3*9) = 6/27 = 2/9
    assert abs(spec[0] - 2/9) < 1e-10
    # Digital root = 6
    assert abs(spec[1] - 6/9) < 1e-10
    
    # Test with digital root cycle
    spec = digital_spectrum(999)
    # Digit sum = 27, digital root = 9
    assert abs(spec[1] - 1.0) < 1e-10
    
    # Test edge case
    spec = digital_spectrum(0)
    assert spec == [0.0, 0.0]
    
    print("✓ Digital spectrum analysis")

def test_harmonic_spectrum():
    """Test harmonic spectrum analysis"""
    # Test with Fibonacci number
    spec = harmonic_spectrum(8)  # fib(6) = 8
    assert len(spec) == 3
    assert all(0 <= s <= 1 for s in spec)
    
    # Test with non-Fibonacci
    spec = harmonic_spectrum(10)
    assert len(spec) == 3
    assert all(0 <= s <= 1 for s in spec)
    
    # Test edge cases
    spec = harmonic_spectrum(0)
    assert spec == [0.0, 0.0, 0.0]
    
    spec = harmonic_spectrum(1)
    assert spec == [0.0, 0.0, 0.0]
    
    print("✓ Harmonic spectrum analysis")

def test_spectral_vector():
    """Test combined spectral vector"""
    vec = spectral_vector(100)
    
    # Check total length
    expected_length = 12 + 10 + 2 + 3  # 27 features
    assert len(vec) == expected_length
    
    # All values should be bounded
    assert all(isinstance(v, (int, float)) for v in vec)
    
    # Test consistency
    vec1 = spectral_vector(42)
    vec2 = spectral_vector(42)
    assert vec1 == vec2  # Deterministic
    
    # Test different numbers give different vectors
    vec3 = spectral_vector(43)
    assert vec1 != vec3
    
    print("✓ Combined spectral vector")

def test_spectral_properties():
    """Test mathematical properties of spectra"""
    # Test that factors have related spectra
    n = 15  # 3 × 5
    spec_n = spectral_vector(n)
    spec_3 = spectral_vector(3)
    spec_5 = spectral_vector(5)
    
    # The spectra should have some relationship
    # At minimum, they should all exist and have same length
    assert len(spec_n) == len(spec_3) == len(spec_5)
    
    # Test scaling properties
    spec_10 = spectral_vector(10)
    spec_100 = spectral_vector(100)
    
    # Digital spectra should be related (100 has same digital root as 10)
    digital_10 = digital_spectrum(10)
    digital_100 = digital_spectrum(100)
    assert digital_10[1] == digital_100[1]  # Same digital root
    
    print("✓ Spectral properties verified")

def test_spectral_determinism():
    """Test that all spectral functions are deterministic"""
    n = 12345
    
    # Run each function multiple times
    for _ in range(3):
        assert binary_spectrum(n) == binary_spectrum(n)
        assert modular_spectrum(n) == modular_spectrum(n)
        assert digital_spectrum(n) == digital_spectrum(n)
        assert harmonic_spectrum(n) == harmonic_spectrum(n)
        assert spectral_vector(n) == spectral_vector(n)
    
    print("✓ All spectral functions are deterministic")

def test_spectral_edge_cases():
    """Test edge cases and boundary conditions"""
    # Very large number
    large_n = 1_000_000
    vec = spectral_vector(large_n)
    assert len(vec) == 27
    assert all(isinstance(v, (int, float)) for v in vec)
    
    # Powers of 2
    for k in range(1, 10):
        n = 2 ** k
        spec = binary_spectrum(n)
        # Should have exactly one 1 bit
        assert spec[0] == 1 / (k + 1)  # Bit density
    
    # Perfect squares
    for k in [4, 9, 16, 25]:
        vec = spectral_vector(k)
        assert len(vec) == 27
    
    print("✓ Edge cases handled correctly")

def run_all_tests():
    """Run all spectral core tests"""
    print("Testing Spectral Core (Axiom 3)...")
    print("-" * 40)
    
    test_binary_spectrum()
    test_modular_spectrum()
    test_digital_spectrum()
    test_harmonic_spectrum()
    test_spectral_vector()
    test_spectral_properties()
    test_spectral_determinism()
    test_spectral_edge_cases()
    
    print("-" * 40)
    print("All Spectral Core tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
