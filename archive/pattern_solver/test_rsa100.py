"""
Test The Pattern on RSA-100
RSA-100 is a 330-bit semiprime that was part of the RSA Factoring Challenge
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pattern import Pattern
from advanced_pattern import AdvancedPattern
from factor_decoder import FactorDecoder


def test_rsa100():
    """Test factorization of RSA-100"""
    # RSA-100 value
    n = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
    
    # Known factors for verification
    expected_p = 37975227936943673922808872755445627854565536638199
    expected_q = 40094690950920881030683735292761468389214899724061
    
    print("=" * 80)
    print("THE PATTERN vs RSA-100")
    print("=" * 80)
    print(f"\nn = {n}")
    print(f"Bit length: {n.bit_length()} bits")
    print()
    
    # Test basic Pattern
    print("Testing Basic Pattern:")
    print("-" * 40)
    pattern = Pattern()
    
    start = time.time()
    signature = pattern.recognize(n)
    formalization = pattern.formalize(signature)
    p, q = pattern.execute(formalization)
    elapsed = time.time() - start
    
    if p * q == n:
        print(f"✓ Success! Factored in {elapsed:.3f} seconds")
        print(f"  p = {p}")
        print(f"  q = {q}")
        success = (p == expected_p and q == expected_q) or (p == expected_q and q == expected_p)
        if success:
            print("  ✓ Factors match expected values!")
    else:
        print(f"✗ Failed with basic Pattern in {elapsed:.3f} seconds")
        print(f"  Result: {p} × {q} = {p*q}")
    
    # Test Advanced Pattern
    print("\nTesting Advanced Pattern:")
    print("-" * 40)
    advanced = AdvancedPattern()
    
    start = time.time()
    signature = advanced.recognize_advanced(n)
    formalization = advanced.formalize(signature)
    p, q = advanced.execute_advanced(formalization)
    elapsed = time.time() - start
    
    if p * q == n:
        print(f"✓ Success! Factored in {elapsed:.3f} seconds")
        print(f"  p = {p}")
        print(f"  q = {q}")
        success = (p == expected_p and q == expected_q) or (p == expected_q and q == expected_p)
        if success:
            print("  ✓ Factors match expected values!")
    else:
        print(f"✗ Failed with advanced Pattern in {elapsed:.3f} seconds")
        print(f"  Result: {p} × {q} = {p*q}")
    
    # Test with custom decoder
    print("\nTesting with Custom FactorDecoder:")
    print("-" * 40)
    from universal_basis import UniversalBasis
    basis = UniversalBasis()
    decoder = FactorDecoder(basis)
    
    start = time.time()
    factors = decoder.decode_comprehensive(n)
    elapsed = time.time() - start
    
    if factors and len(factors) >= 2:
        p, q = factors[0], factors[1]
        if p * q == n:
            print(f"✓ Success! Factored in {elapsed:.3f} seconds")
            print(f"  p = {p}")
            print(f"  q = {q}")
            success = (p == expected_p and q == expected_q) or (p == expected_q and q == expected_p)
            if success:
                print("  ✓ Factors match expected values!")
        else:
            print(f"✗ Incomplete factorization in {elapsed:.3f} seconds")
            print(f"  Found factors: {factors}")
    else:
        print(f"✗ Failed with decoder in {elapsed:.3f} seconds")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print("\nRSA-100 Properties:")
    print(f"  n bit length: {n.bit_length()}")
    print(f"  p bit length: {expected_p.bit_length()}")
    print(f"  q bit length: {expected_q.bit_length()}")
    print(f"  |p - q|: {abs(expected_p - expected_q)}")
    print(f"  p/q ratio: {max(expected_p, expected_q) / min(expected_p, expected_q):.6f}")
    
    # Factor relationship analysis
    sqrt_n = int(n ** 0.5)
    a = (expected_p + expected_q) // 2
    b = abs(expected_p - expected_q) // 2
    offset = a - sqrt_n
    
    print(f"\nFermat Factorization Analysis:")
    print(f"  sqrt(n) = {sqrt_n}")
    print(f"  a = (p+q)/2 = {a}")
    print(f"  b = |p-q|/2 = {b}")
    print(f"  offset = a - sqrt(n) = {offset}")
    print(f"  offset/sqrt(n) = {offset/sqrt_n:.6%}")
    
    print("\nPattern Signature Analysis:")
    signature = pattern.recognize(n)
    print(f"  φ-component: {signature.phi_component:.6f}")
    print(f"  π-component: {signature.pi_component:.6f}")
    print(f"  e-component: {signature.e_component:.6f}")
    print(f"  Unity phase: {signature.unity_phase:.6f} radians")
    print(f"  Max resonance: {max(abs(r) for r in signature.resonance_field):.6f}")


if __name__ == "__main__":
    test_rsa100()