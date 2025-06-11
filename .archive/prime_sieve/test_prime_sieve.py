"""
Test script for Prime Sieve implementation

Demonstrates the Prime Sieve's capabilities with various test cases.
"""

import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prime_sieve import PrimeSieve


def test_prime_sieve():
    """Run basic tests on the Prime Sieve."""
    print("=" * 60)
    print("Prime Sieve Test Suite")
    print("=" * 60)
    
    # Initialize sieve with learning enabled
    sieve = PrimeSieve(enable_learning=True)
    
    # Test cases with known factors
    test_cases = [
        # Small semiprimes
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (143, 11, 13),
        
        # Medium semiprimes
        (323, 17, 19),
        (1147, 31, 37),
        (2021, 43, 47),
        (3599, 59, 61),
        
        # Larger semiprimes (32-bit range)
        (294409, 37, 7957),  # Correct factors: 37 × 7957 = 294409
        (1299071, 1117, 1163),  # Correct: 1117 × 1163 = 1299071
        
        # Even larger (if you want to test)
        # (10967535067, 104723, 104729),  # 34-bit semiprime
    ]
    
    print("\nTesting known semiprimes:")
    print("-" * 60)
    
    for n, expected_p, expected_q in test_cases:
        print(f"\nFactoring n = {n} (expected: {expected_p} × {expected_q})")
        
        # Factor with details
        result = sieve.factor_with_details(n)
        
        # Verify result
        p, q = result.factors
        is_correct = (p * q == n) and ({p, q} == {expected_p, expected_q} or n == 1)
        
        print(f"  Result: {p} × {q} = {p * q}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        print(f"  Method: {result.method}")
        print(f"  Time: {result.time_taken:.6f}s")
        print(f"  Iterations: {result.iterations}")
        print(f"  Candidates tested: {result.candidates_tested}")
        print(f"  Peak coherence: {result.peak_coherence:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        
    # Test arbitrary precision support
    print("\n" + "=" * 60)
    print("Testing arbitrary precision support:")
    print("-" * 60)
    
    # Test increasing bit lengths
    bit_tests = [
        (8, 15),    # 4-bit
        (16, 323),  # 9-bit  
        (32, 1299071),  # 21-bit (1117 × 1163)
    ]
    
    for bits, n in bit_tests:
        print(f"\nTesting {n.bit_length()}-bit number: {n}")
        result = sieve.factor_with_details(n)
        p, q = result.factors
        print(f"  Factors: {p} × {q}")
        print(f"  Time: {result.time_taken:.6f}s")
        print(f"  Success: {'✓' if p * q == n and p > 1 else '✗'}")
    
    # Display statistics
    print("\n" + "=" * 60)
    print("Prime Sieve Statistics:")
    print("-" * 60)
    
    stats = sieve.get_statistics()
    print(f"Total factorizations: {stats['total_factorizations']}")
    print(f"Successful factorizations: {stats['successful_factorizations']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    
    if 'meta_observer' in stats:
        meta_stats = stats['meta_observer']
        print(f"\nMeta-Observer Learning:")
        print(f"  Total observations: {meta_stats['total_observations']}")
        print(f"  Success rate: {meta_stats['success_rate']:.2%}")
        print(f"  Learned strategies: {meta_stats['learned_strategies']}")


def demonstrate_scaling():
    """Demonstrate how the sieve scales with problem size."""
    print("\n" + "=" * 60)
    print("Demonstrating Scaling Behavior")
    print("=" * 60)
    
    sieve = PrimeSieve(enable_learning=False)  # Disable learning for pure timing
    
    # Test different problem sizes
    test_numbers = [
        77,       # 7-bit
        323,      # 9-bit
        3599,     # 12-bit
        294409,   # 19-bit
        1299071,  # 21-bit (1117 × 1163)
    ]
    
    print("\nbit_length | number      | time (s)   | method")
    print("-" * 50)
    
    for n in test_numbers:
        start = time.time()
        p, q = sieve.factor(n)
        elapsed = time.time() - start
        
        # Get method used
        result = sieve.factor_with_details(n)
        
        print(f"{n.bit_length():10d} | {n:11d} | {elapsed:10.6f} | {result.method}")


if __name__ == "__main__":
    # Run tests
    test_prime_sieve()
    
    # Demonstrate scaling
    demonstrate_scaling()
    
    print("\n" + "=" * 60)
    print("Prime Sieve test complete!")
    print("=" * 60)
