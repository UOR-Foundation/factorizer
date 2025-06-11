"""
Test the optimized factorizer implementation
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factorizer_optimized import OptimizedFactorizer, FactorizationResult
from factorizer_64bit_benchmark_simple import BenchmarkCase


def test_optimized():
    """Test the optimized factorizer with various cases"""
    print("Testing Optimized Factorizer Implementation")
    print("=" * 60)
    
    # Initialize optimized factorizer
    factorizer = OptimizedFactorizer(learning_enabled=True)
    
    # Test cases
    test_cases = [
        # 8-bit cases
        BenchmarkCase(15, 3, 5, 8, "easy"),
        BenchmarkCase(21, 3, 7, 8, "easy"),
        BenchmarkCase(35, 5, 7, 8, "easy"),
        BenchmarkCase(77, 7, 11, 8, "easy"),
        BenchmarkCase(91, 7, 13, 8, "easy"),
        BenchmarkCase(143, 11, 13, 8, "easy"),
        
        # 16-bit cases
        BenchmarkCase(323, 17, 19, 16, "easy"),
        BenchmarkCase(391, 17, 23, 16, "easy"),
        BenchmarkCase(667, 23, 29, 16, "easy"),
        BenchmarkCase(899, 29, 31, 16, "easy"),
        BenchmarkCase(1147, 31, 37, 16, "easy"),
        BenchmarkCase(1517, 37, 41, 16, "easy"),
        BenchmarkCase(10403, 101, 103, 16, "hard"),  # Close primes
        
        # 32-bit cases
        BenchmarkCase(16843009, 257, 65537, 32, "easy"),
        BenchmarkCase(269419387, 16411, 16417, 32, "hard"),  # Close primes
        
        # Special cases
        BenchmarkCase(25, 5, 5, 8, "special"),  # Perfect square
        BenchmarkCase(6, 2, 3, 8, "fibonacci"),  # Fib primes
    ]
    
    # Run tests
    successful = 0
    total_time = 0.0
    
    for i, case in enumerate(test_cases):
        print(f"\nTest {i+1}/{len(test_cases)}: n={case.n} (p={case.p}, q={case.q})")
        
        try:
            # Run factorization
            result = factorizer.factorize_with_details(case.n)
            
            # Check success
            success = (
                result.factors[0] * result.factors[1] == case.n and
                result.factors[0] > 1 and result.factors[1] > 1
            )
            
            # Verify correct factors
            if success:
                expected = {case.p, case.q}
                actual = {result.factors[0], result.factors[1]}
                success = expected == actual
            
            if success:
                successful += 1
                status = "✓ SUCCESS"
            else:
                status = f"✗ FAILED (got {result.factors})"
            
            print(f"  {status}")
            print(f"  Time: {result.time_elapsed:.3f}s")
            print(f"  Primary axiom: {result.primary_axiom}")
            print(f"  Max coherence: {result.max_coherence:.3f}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Candidates explored: {result.candidates_explored}")
            print(f"  Method sequence: {' -> '.join(result.method_sequence)}")
            
            total_time += result.time_elapsed
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {successful}/{len(test_cases)} successful")
    print(f"Success rate: {successful/len(test_cases)*100:.1f}%")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time per case: {total_time/len(test_cases):.3f}s")


def compare_implementations():
    """Compare original vs optimized implementation"""
    print("\n\nComparing Original vs Optimized Implementation")
    print("=" * 60)
    
    from factorizer import Factorizer
    
    original = Factorizer(learning_enabled=True)
    optimized = OptimizedFactorizer(learning_enabled=True)
    
    # Test cases for comparison
    test_numbers = [77, 323, 1147, 10403, 16843009]
    
    print(f"{'Number':<10} {'Original Time':<15} {'Optimized Time':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for n in test_numbers:
        # Test original
        start = time.time()
        orig_result = original.factorize_with_details(n)
        orig_time = time.time() - start
        
        # Test optimized
        start = time.time()
        opt_result = optimized.factorize_with_details(n)
        opt_time = time.time() - start
        
        # Calculate speedup
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        
        # Verify same results
        same_result = orig_result.factors == opt_result.factors
        
        print(f"{n:<10} {orig_time:<15.3f} {opt_time:<15.3f} {speedup:<10.2f}x {'✓' if same_result else '✗'}")


if __name__ == "__main__":
    test_optimized()
    compare_implementations()
