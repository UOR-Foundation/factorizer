"""
Test suite for The Pattern implementation
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pattern import Pattern
from universal_basis import UniversalBasis
from factor_decoder import FactorDecoder
from advanced_pattern import AdvancedPattern


def test_basic_pattern():
    """Test basic Pattern functionality"""
    print("Testing Basic Pattern Implementation")
    print("=" * 50)
    
    pattern = Pattern()
    basis = UniversalBasis()
    decoder = FactorDecoder(basis)
    
    pattern.universal_basis = basis
    pattern.decoder = decoder
    
    # Test cases with known factors
    test_cases = [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (91, 7, 13),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
    ]
    
    successes = 0
    total_time = 0
    
    for n, expected_p, expected_q in test_cases:
        start = time.time()
        
        # Execute The Pattern
        signature = pattern.recognize(n)
        formalization = pattern.formalize(signature)
        p, q = pattern.execute(formalization)
        
        elapsed = time.time() - start
        total_time += elapsed
        
        # Verify result
        if p * q == n and {p, q} == {expected_p, expected_q}:
            successes += 1
            print(f"✓ {n} = {p} × {q} ({elapsed:.6f}s)")
        else:
            print(f"✗ {n}: expected {expected_p} × {expected_q}, got {p} × {q}")
    
    print(f"\nSuccess rate: {successes}/{len(test_cases)} ({100*successes/len(test_cases):.1f}%)")
    print(f"Average time: {total_time/len(test_cases):.6f}s")
    print()


def test_universal_basis():
    """Test Universal Basis functionality"""
    print("Testing Universal Basis")
    print("=" * 50)
    
    basis = UniversalBasis()
    
    # Test projection and reconstruction
    test_numbers = [15, 21, 35, 77, 143]
    
    for n in test_numbers:
        coords = basis.project(n)
        print(f"n={n}: coordinates = [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}, {coords[3]:.3f}]")
        
        # Test transformations
        rotated = basis.transform(coords, 'rotate')
        scaled = basis.transform(coords, 'scale')
        reflected = basis.transform(coords, 'reflect')
        
        print(f"  Rotated: [{rotated[0]:.3f}, {rotated[1]:.3f}, {rotated[2]:.3f}, {rotated[3]:.3f}]")
        
        # Test resonance points
        resonance_points = basis.find_resonance_points(n)
        if resonance_points:
            print(f"  Resonance points near {n}: {resonance_points[:5]}")
    
    print()


def test_advanced_pattern():
    """Test advanced Pattern techniques"""
    print("Testing Advanced Pattern Implementation")
    print("=" * 50)
    
    pattern = AdvancedPattern()
    basis = UniversalBasis()
    
    test_cases = [
        (77, 7, 11),
        (143, 11, 13),
        (323, 17, 19),
    ]
    
    for n, expected_p, expected_q in test_cases:
        start = time.time()
        
        # Advanced recognition and execution
        signature = pattern.recognize_advanced(n)
        formalization = pattern.formalize(signature)
        p, q = pattern.execute_advanced(formalization)
        
        elapsed = time.time() - start
        
        if p * q == n and {p, q} == {expected_p, expected_q}:
            print(f"✓ {n} = {p} × {q} (Advanced, {elapsed:.6f}s)")
        else:
            print(f"✗ {n}: expected {expected_p} × {expected_q}, got {p} × {q}")
    
    print()


def test_larger_numbers():
    """Test with larger semiprimes"""
    print("Testing Larger Numbers")
    print("=" * 50)
    
    pattern = Pattern()
    basis = UniversalBasis()
    decoder = FactorDecoder(basis)
    
    pattern.universal_basis = basis
    pattern.decoder = decoder
    
    # Larger test cases
    test_cases = [
        (1147, 31, 37),     # 31 × 37
        (1763, 41, 43),     # 41 × 43
        (2021, 43, 47),     # 43 × 47
        (10403, 101, 103),  # 101 × 103
    ]
    
    for n, expected_p, expected_q in test_cases:
        start = time.time()
        
        signature = pattern.recognize(n)
        formalization = pattern.formalize(signature)
        p, q = pattern.execute(formalization)
        
        elapsed = time.time() - start
        
        if p * q == n and {p, q} == {expected_p, expected_q}:
            print(f"✓ {n} = {p} × {q} ({elapsed:.6f}s)")
        else:
            print(f"✗ {n}: expected {expected_p} × {expected_q}, got {p} × {q}")
    
    print()


def analyze_pattern_relationships():
    """Analyze relationships in universal space"""
    print("Analyzing Pattern Relationships")
    print("=" * 50)
    
    basis = UniversalBasis()
    
    # Analyze twin primes
    twin_primes = [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43)]
    
    print("Twin Prime Analysis:")
    for p, q in twin_primes:
        n = p * q
        n_coords = basis.project(n)
        p_coords = basis.project(p)
        q_coords = basis.project(q)
        
        phi_ratio = p_coords[0] / q_coords[0]
        distance = basis.measure_distance(p_coords, q_coords)
        
        print(f"  {p} × {q} = {n}: φ-ratio = {phi_ratio:.4f}, distance = {distance:.4f}")
    
    print()
    
    # Analyze cousin primes (differ by 4)
    cousin_primes = [(3, 7), (7, 11), (13, 17), (19, 23), (37, 41), (43, 47)]
    
    print("Cousin Prime Analysis:")
    for p, q in cousin_primes:
        n = p * q
        n_coords = basis.project(n)
        p_coords = basis.project(p)
        q_coords = basis.project(q)
        
        phi_ratio = p_coords[0] / q_coords[0]
        distance = basis.measure_distance(p_coords, q_coords)
        
        print(f"  {p} × {q} = {n}: φ-ratio = {phi_ratio:.4f}, distance = {distance:.4f}")
    
    print()


def main():
    """Run all tests"""
    print("The Pattern - Test Suite")
    print("========================\n")
    
    test_basic_pattern()
    test_universal_basis()
    test_advanced_pattern()
    test_larger_numbers()
    analyze_pattern_relationships()
    
    print("\nAll tests complete!")


if __name__ == "__main__":
    main()