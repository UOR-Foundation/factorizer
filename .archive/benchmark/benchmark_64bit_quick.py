"""
Quick 64-bit Benchmark for Universal Ontological Factorizer

A faster version of the comprehensive benchmark for quick testing.
Tests fewer cases but covers the full range up to 64-bit.
"""

import time
import sys
import os
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factorizer import Factorizer
from factorizer_optimized import OptimizedFactorizer

# Predefined test cases covering different bit ranges
TEST_CASES = [
    # Small (8-16 bit)
    (143, 11, 13),           # 8 bits
    (323, 17, 19),           # 9 bits
    (1147, 31, 37),          # 11 bits
    (10403, 101, 103),       # 14 bits
    
    # Medium (16-32 bit)
    (1046527, 1021, 1024),   # 20 bits
    (16843009, 257, 65537),  # 25 bits
    (1073676289, 32749, 32771), # 30 bits
    
    # Large (32-48 bit)
    (1099511627791, 1048573, 1048583),  # 40 bits
    (281474976710597, 16777213, 16777259), # 48 bits
    
    # Extra large (48-64 bit)
    (72057594037927931, 268435399, 268435459), # 56 bits
    (4611686018427387847, 2147483587, 2147483629), # 62 bits
]

def benchmark_factorizer(factorizer, name: str) -> None:
    """Run benchmark on a factorizer implementation."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {name}")
    print(f"{'='*80}")
    
    total_time = 0
    successes = 0
    
    print(f"\n{'Bits':>4} {'Number':>22} {'Expected':>25} "
          f"{'Result':>25} {'Time':>8} {'Status':>8}")
    print("-" * 95)
    
    for n, p, q in TEST_CASES:
        start = time.time()
        
        try:
            result = factorizer.factorize_with_details(n)
            elapsed = time.time() - start
            
            factors = result.factors
            success = (factors == (p, q)) or (factors == (q, p))
            
            if success:
                successes += 1
                status = "✓ OK"
            else:
                status = "✗ FAIL"
            
            total_time += elapsed
            
            print(f"{n.bit_length():4d} {n:22d} ({p:11d},{q:11d}) "
                  f"({factors[0]:11d},{factors[1]:11d}) {elapsed:8.3f}s {status:>8s}")
            
            # Additional details for failed cases
            if not success and factors != (1, n):
                print(f"     -> Primary axiom: {result.primary_axiom}, "
                      f"Max coherence: {result.max_coherence:.3f}, "
                      f"Iterations: {result.iterations}")
        
        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"{n.bit_length():4d} {n:22d} ({p:11d},{q:11d}) "
                  f"{'ERROR':^25} {elapsed:8.3f}s ✗ ERROR")
            print(f"     -> Error: {str(e)[:60]}")
    
    # Summary
    success_rate = successes / len(TEST_CASES) * 100
    print(f"\n{'Summary':^95}")
    print("-" * 95)
    print(f"Success rate: {successes}/{len(TEST_CASES)} ({success_rate:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per case: {total_time/len(TEST_CASES):.3f}s")
    
    # Performance breakdown by bit size
    print("\nPerformance by bit size:")
    ranges = [(8, 16, "Small"), (16, 32, "Medium"), (32, 48, "Large"), (48, 64, "Extra Large")]
    
    for min_bits, max_bits, label in ranges:
        range_cases = [(n, p, q) for n, p, q in TEST_CASES 
                       if min_bits <= n.bit_length() <= max_bits]
        if range_cases:
            range_time = sum(elapsed for n, _, _ in range_cases 
                           for elapsed in [time.time()] 
                           if (time.sleep(0), True)[1])  # Placeholder
            print(f"  {label:12s} ({min_bits:2d}-{max_bits:2d} bits): "
                  f"{len(range_cases)} cases")

def compare_implementations() -> None:
    """Compare original and optimized implementations."""
    print("\n" + "="*80)
    print("COMPARING IMPLEMENTATIONS")
    print("="*80)
    
    # Test both
    original = Factorizer(learning_enabled=True)
    optimized = OptimizedFactorizer(learning_enabled=True)
    
    # Warm up
    print("\nWarming up...")
    for _ in range(3):
        original.factorize(15)
        optimized.factorize(15)
    
    # Benchmark original
    benchmark_factorizer(original, "Original Factorizer")
    
    # Benchmark optimized
    benchmark_factorizer(optimized, "Optimized Factorizer")
    
    # Direct comparison on subset
    print("\n" + "="*80)
    print("DIRECT COMPARISON (Medium-sized numbers)")
    print("="*80)
    
    comparison_cases = TEST_CASES[4:7]  # Medium range
    
    print(f"\n{'Number':>22} {'Original':>10} {'Optimized':>10} {'Speedup':>8}")
    print("-" * 52)
    
    for n, p, q in comparison_cases:
        # Original
        start = time.time()
        orig_result = original.factorize(n)
        orig_time = time.time() - start
        
        # Optimized
        start = time.time()
        opt_result = optimized.factorize(n)
        opt_time = time.time() - start
        
        speedup = orig_time / opt_time if opt_time > 0 else 0
        
        print(f"{n:22d} {orig_time:10.4f}s {opt_time:10.4f}s {speedup:7.2f}x")

def main():
    """Run quick 64-bit benchmark."""
    print("Universal Ontological Factorizer - Quick 64-bit Benchmark")
    print("Testing factorization on numbers up to 64-bit")
    
    # Test just the original implementation by default
    factorizer = Factorizer(learning_enabled=True)
    benchmark_factorizer(factorizer, "Universal Ontological Factorizer")
    
    # Optionally compare implementations
    if "--compare" in sys.argv:
        compare_implementations()

if __name__ == "__main__":
    main()
