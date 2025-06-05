"""
Tests for Prime Core functionality
Validates primality testing and prime generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom1.prime_core import is_prime, primes_up_to, SMALL_PRIMES

def test_small_primes():
    """Test SMALL_PRIMES constant"""
    assert SMALL_PRIMES == (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
    print("✓ SMALL_PRIMES constant correct")

def test_is_prime_basic():
    """Test basic primality testing"""
    # Test small primes
    assert is_prime(2) == True
    assert is_prime(3) == True
    assert is_prime(5) == True
    assert is_prime(7) == True
    assert is_prime(11) == True
    assert is_prime(13) == True
    
    # Test small composites
    assert is_prime(0) == False
    assert is_prime(1) == False
    assert is_prime(4) == False
    assert is_prime(6) == False
    assert is_prime(8) == False
    assert is_prime(9) == False
    assert is_prime(10) == False
    
    print("✓ Basic primality tests pass")

def test_is_prime_larger():
    """Test primality for larger numbers"""
    # Known primes
    primes = [97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
              149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
              197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
    
    for p in primes:
        assert is_prime(p) == True
    
    # Known composites
    composites = [91, 111, 119, 121, 133, 143, 161, 169, 187, 203,
                  209, 217, 221, 247, 253, 259, 287, 289, 299, 301]
    
    for c in composites:
        assert is_prime(c) == False
    
    print("✓ Larger primality tests pass")

def test_is_prime_special_cases():
    """Test special cases and edge conditions"""
    # Mersenne primes
    assert is_prime(3) == True      # 2^2 - 1
    assert is_prime(7) == True      # 2^3 - 1
    assert is_prime(31) == True     # 2^5 - 1
    assert is_prime(127) == True    # 2^7 - 1
    assert is_prime(8191) == True   # 2^13 - 1
    
    # Carmichael numbers (pseudoprimes)
    assert is_prime(561) == False   # 3 × 11 × 17
    assert is_prime(1105) == False  # 5 × 13 × 17
    assert is_prime(1729) == False  # 7 × 13 × 19
    
    # Twin primes
    assert is_prime(29) == True and is_prime(31) == True
    assert is_prime(41) == True and is_prime(43) == True
    assert is_prime(71) == True and is_prime(73) == True
    
    print("✓ Special case primality tests pass")

def test_primes_up_to():
    """Test prime generation"""
    # Test empty cases
    assert primes_up_to(0) == []
    assert primes_up_to(1) == []
    
    # Test small limits
    assert primes_up_to(2) == [2]
    assert primes_up_to(3) == [2, 3]
    assert primes_up_to(10) == [2, 3, 5, 7]
    assert primes_up_to(20) == [2, 3, 5, 7, 11, 13, 17, 19]
    
    # Test larger limit
    primes_100 = primes_up_to(100)
    assert len(primes_100) == 25
    assert primes_100[0] == 2
    assert primes_100[-1] == 97
    
    # Verify all generated numbers are prime
    for p in primes_100:
        assert is_prime(p) == True
    
    # Verify no primes are missed
    for n in range(2, 101):
        if is_prime(n):
            assert n in primes_100
    
    print("✓ Prime generation tests pass")

def test_primes_up_to_consistency():
    """Test consistency between is_prime and primes_up_to"""
    limit = 1000
    generated = primes_up_to(limit)
    
    # Check all generated are prime
    for p in generated:
        assert is_prime(p) == True
    
    # Check no primes missed
    prime_count = 0
    for n in range(2, limit + 1):
        if is_prime(n):
            prime_count += 1
            assert n in generated
    
    assert len(generated) == prime_count
    print(f"✓ Generated {prime_count} primes up to {limit} consistently")

def test_deterministic():
    """Test that prime operations are deterministic"""
    # Run multiple times to ensure deterministic results
    for _ in range(5):
        assert is_prime(97) == True
        assert is_prime(98) == False
        assert primes_up_to(50) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    print("✓ Prime operations are deterministic")

def run_all_tests():
    """Run all prime core tests"""
    print("Testing Prime Core (Axiom 1)...")
    print("-" * 40)
    
    test_small_primes()
    test_is_prime_basic()
    test_is_prime_larger()
    test_is_prime_special_cases()
    test_primes_up_to()
    test_primes_up_to_consistency()
    test_deterministic()
    
    print("-" * 40)
    print("All Prime Core tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
