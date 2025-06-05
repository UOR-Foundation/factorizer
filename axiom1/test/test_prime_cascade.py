"""
Tests for Prime Cascade functionality
Validates twin primes and Sophie Germain chains
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom1.prime_cascade import PrimeCascade
from axiom1.prime_core import is_prime

def test_prime_cascade_init():
    """Test PrimeCascade initialization"""
    cascade = PrimeCascade(1000)
    assert cascade.n == 1000
    print("✓ PrimeCascade initialization")

def test_twin_primes():
    """Test twin prime detection"""
    cascade = PrimeCascade(1000)
    
    # Test known twin prime pairs
    twin_prime_tests = [
        (3, [5]),       # 3+2=5 is prime
        (5, [3, 7]),    # 5-2=3 and 5+2=7 are prime
        (11, [13]),     # 11+2=13 is prime
        (17, [19]),     # 17+2=19 is prime
        (29, [31]),     # 29+2=31 is prime
        (41, [43]),     # 41+2=43 is prime
    ]
    
    for p, expected_twins in twin_prime_tests:
        result = cascade.cascade(p)
        for twin in expected_twins:
            assert twin in result, f"Expected twin {twin} for prime {p}"
    
    print("✓ Twin prime detection")

def test_sophie_germain():
    """Test Sophie Germain chain detection"""
    cascade = PrimeCascade(10000)
    
    # Test known Sophie Germain primes
    # p is Sophie Germain prime if 2p+1 is also prime
    sophie_germain_tests = [
        (2, 5),     # 2×2+1 = 5
        (3, 7),     # 2×3+1 = 7
        (5, 11),    # 2×5+1 = 11
        (11, 23),   # 2×11+1 = 23
        (23, 47),   # 2×23+1 = 47
        (29, 59),   # 2×29+1 = 59
        (41, 83),   # 2×41+1 = 83
        (53, 107),  # 2×53+1 = 107
    ]
    
    for p, expected in sophie_germain_tests:
        result = cascade.cascade(p)
        assert expected in result, f"Expected Sophie Germain prime {expected} for {p}"
    
    print("✓ Sophie Germain chain detection")

def test_cascade_chains():
    """Test longer Sophie Germain chains"""
    cascade = PrimeCascade(100000)
    
    # Starting from 2: 2 → 5 → 11 → 23 → 47
    result = cascade.cascade(2)
    assert 5 in result
    assert 11 in result
    assert 23 in result
    assert 47 in result
    
    # Starting from 89: 89 → 179 → 359 → 719 → 1439
    result = cascade.cascade(89)
    assert 179 in result
    assert 359 in result
    assert 719 in result
    assert 1439 in result
    
    print("✓ Sophie Germain chain following")

def test_cascade_limits():
    """Test cascade behavior with limits"""
    # Small n should limit cascade
    cascade = PrimeCascade(100)
    result = cascade.cascade(89)
    # 179 > 100, so it shouldn't be included
    assert 179 not in result
    
    # Test that cascade stops at length limit
    cascade = PrimeCascade(1000000)
    result = cascade.cascade(2)
    assert len(result) <= 10  # Max cascade length
    
    print("✓ Cascade limits respected")

def test_non_prime_input():
    """Test cascade with composite number input"""
    cascade = PrimeCascade(1000)
    
    # Cascade will still check relationships even for composite inputs
    result = cascade.cascade(4)
    # 4+2=6 is not prime (correctly excluded)
    # 4-2=2 is prime (correctly included since 4 > 2)
    # 2×4+1=9 is not prime (correctly excluded)
    assert 2 in result  # 4-2=2 is prime
    assert 6 not in result  # 4+2=6 is not prime
    assert 9 not in result  # 2×4+1=9 is not prime
    
    # Test with a larger composite
    result = cascade.cascade(10)
    # 10+2=12 is not prime
    # 10-2=8 is not prime
    # 2×10+1=21 is not prime
    assert len(result) == 0  # No prime relationships found
    
    print("✓ Composite input handled correctly")

def test_edge_cases():
    """Test edge cases"""
    cascade = PrimeCascade(1000)
    
    # Test with 2 (smallest prime)
    result = cascade.cascade(2)
    assert 5 in result  # 2×2+1 = 5
    # Note: 2-2=0 is not checked, 2+2=4 is not prime
    
    # Test with large prime near n
    cascade = PrimeCascade(100)
    result = cascade.cascade(97)
    # 97+2=99 is not prime
    # 2×97+1=195 > 100, so not included
    
    print("✓ Edge cases handled")

def test_all_cascade_elements_prime():
    """Verify all cascade results are prime"""
    cascade = PrimeCascade(10000)
    
    test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for p in test_primes:
        result = cascade.cascade(p)
        for r in result:
            assert is_prime(r), f"Cascade returned non-prime {r} for input {p}"
    
    print("✓ All cascade elements are prime")

def run_all_tests():
    """Run all prime cascade tests"""
    print("Testing Prime Cascade (Axiom 1)...")
    print("-" * 40)
    
    test_prime_cascade_init()
    test_twin_primes()
    test_sophie_germain()
    test_cascade_chains()
    test_cascade_limits()
    test_non_prime_input()
    test_edge_cases()
    test_all_cascade_elements_prime()
    
    print("-" * 40)
    print("All Prime Cascade tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
