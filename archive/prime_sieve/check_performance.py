"""
Check current Prime Sieve performance and diagnose issues
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prime_sieve import PrimeSieve


def diagnose_performance():
    """Diagnose current Prime Sieve performance."""
    print("="*60)
    print("PRIME SIEVE PERFORMANCE DIAGNOSTIC")
    print("="*60)
    
    sieve = PrimeSieve(enable_learning=False)
    
    # Test the problematic case
    n = 294409
    expected_p, expected_q = 37, 7957
    
    print(f"\nTesting problematic case: {n} = {expected_p} × {expected_q}")
    
    # Run multiple times to check consistency
    for i in range(3):
        print(f"\nAttempt {i+1}:")
        start = time.time()
        result = sieve.factor_with_details(n)
        elapsed = time.time() - start
        
        print(f"  Found: {result.factors[0]} × {result.factors[1]}")
        print(f"  Correct: {result.factors == (expected_p, expected_q) or result.factors == (expected_q, expected_p)}")
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Method: {result.method}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Candidates tested: {result.candidates_tested}")
    
    # Test all cases from the test suite
    print("\n" + "="*60)
    print("FULL TEST SUITE PERFORMANCE")
    print("="*60)
    
    test_cases = [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (143, 11, 13),
        (323, 17, 19),
        (1147, 31, 37),
        (2021, 43, 47),
        (3599, 59, 61),
        (294409, 37, 7957),
        (1299071, 1117, 1163),
    ]
    
    total_time = 0
    successful = 0
    
    for n, expected_p, expected_q in test_cases:
        start = time.time()
        result = sieve.factor_with_details(n)
        elapsed = time.time() - start
        total_time += elapsed
        
        p, q = result.factors
        is_correct = (p == expected_p and q == expected_q) or (p == expected_q and q == expected_p)
        
        if is_correct:
            successful += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{n:8d}: {p:6d} × {q:6d} = {p*q:8d} {status} ({elapsed:.4f}s, {result.method})")
    
    print(f"\nSuccess rate: {successful}/{len(test_cases)} = {successful/len(test_cases):.0%}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time: {total_time/len(test_cases):.4f}s")
    
    # Check what's in the current implementation
    print("\n" + "="*60)
    print("CHECKING CURRENT PARAMETERS")
    print("="*60)
    
    print("\nTo optimize performance, consider adjusting these parameters")
    print("in the _generate_initial_candidates method:")
    print("- coord_candidates: Number of prime coordinate candidates")
    print("- vortex_candidates: Number of Fibonacci vortex candidates") 
    print("- sqrt_delta_range: Range around sqrt(n) to search")
    print("- sqrt_extension: Factor to extend search beyond sqrt(n)")


if __name__ == "__main__":
    diagnose_performance()
