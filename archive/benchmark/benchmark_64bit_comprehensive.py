"""
Comprehensive 64-bit Benchmark for Universal Ontological Factorizer

Tests the factorizer on semiprimes up to 64-bit, leveraging all acceleration features.
Includes performance profiling and detailed metrics.
"""

import time
import random
import math
import sys
import os
from typing import List, Tuple, Dict
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factorizer import Factorizer, FactorizationResult
from factorizer_optimized import OptimizedFactorizer
from axiom1 import primes_up_to

# Test categories for 64-bit numbers
TEST_CATEGORIES = {
    "small": {
        "name": "Small (< 16-bit)",
        "bit_range": (8, 16),
        "count": 10
    },
    "medium": {
        "name": "Medium (16-32 bit)",
        "bit_range": (16, 32),
        "count": 10
    },
    "large": {
        "name": "Large (32-48 bit)",
        "bit_range": (32, 48),
        "count": 10
    },
    "xlarge": {
        "name": "Extra Large (48-64 bit)",
        "bit_range": (48, 64),
        "count": 5
    }
}

def generate_semiprime(bit_size: int) -> Tuple[int, int, int]:
    """Generate a semiprime with approximately the given bit size."""
    # Each prime should be roughly half the bit size
    prime_bits = bit_size // 2
    
    # Generate primes in the appropriate range
    min_val = 2 ** (prime_bits - 1)
    max_val = 2 ** prime_bits - 1
    
    # Get primes in range
    all_primes = primes_up_to(max_val)
    primes_in_range = [p for p in all_primes if min_val <= p <= max_val]
    
    if len(primes_in_range) < 2:
        # Fall back to smaller range
        primes_in_range = [p for p in all_primes if p > max_val // 2]
    
    # Select two random primes
    p = random.choice(primes_in_range)
    q = random.choice(primes_in_range)
    
    n = p * q
    return n, min(p, q), max(p, q)

def generate_special_semiprimes() -> List[Tuple[int, int, int]]:
    """Generate special test cases that stress different axioms."""
    special_cases = []
    
    # Twin primes (stress Axiom 1)
    twin_primes = [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), 
                   (41, 43), (59, 61), (71, 73), (101, 103)]
    for p, q in twin_primes:
        special_cases.append((p * q, p, q))
    
    # Fibonacci primes (stress Axiom 2)
    fib_primes = [2, 3, 5, 13, 89, 233, 1597, 28657, 514229]
    for i in range(len(fib_primes) - 1):
        p, q = fib_primes[i], fib_primes[i + 1]
        if p * q < 2**64:
            special_cases.append((p * q, p, q))
    
    # Sophie Germain primes (p and 2p+1 are both prime)
    sophie_germain = [(2, 5), (3, 7), (5, 11), (11, 23), (23, 47),
                      (29, 59), (41, 83), (53, 107), (83, 167)]
    for p, q in sophie_germain:
        special_cases.append((p * q, p, q))
    
    # Large gap primes (stress quantum tunneling)
    large_gap = [(887, 907), (1327, 1361), (31397, 31469),
                 (492113, 492227), (1357201, 1357333)]
    for p, q in large_gap:
        if p * q < 2**64:
            special_cases.append((p * q, p, q))
    
    return special_cases

def benchmark_single(factorizer: Factorizer, n: int, p: int, q: int, 
                    use_optimized: bool = False) -> Dict:
    """Benchmark a single factorization."""
    start_time = time.time()
    
    try:
        if use_optimized:
            result = factorizer.factorize_with_details(n)
        else:
            result = factorizer.factorize_with_details(n)
        
        elapsed = time.time() - start_time
        
        success = (result.factors == (p, q)) or (result.factors == (q, p))
        
        return {
            "n": n,
            "expected": (p, q),
            "result": result.factors,
            "success": success,
            "time": elapsed,
            "iterations": result.iterations,
            "candidates": result.candidates_explored,
            "max_coherence": result.max_coherence,
            "primary_axiom": result.primary_axiom,
            "methods": result.method_sequence,
            "learning": result.learning_applied
        }
    
    except Exception as e:
        return {
            "n": n,
            "expected": (p, q),
            "result": None,
            "success": False,
            "time": time.time() - start_time,
            "error": str(e),
            "iterations": 0,
            "candidates": 0,
            "max_coherence": 0.0,
            "primary_axiom": "error",
            "methods": [],
            "learning": False
        }

def run_category_benchmark(factorizer: Factorizer, category: str, 
                          test_cases: List[Tuple[int, int, int]],
                          use_optimized: bool = False) -> Dict:
    """Run benchmark for a category of test cases."""
    results = []
    successes = 0
    total_time = 0
    
    print(f"\n{TEST_CATEGORIES[category]['name']} Semiprimes:")
    print("-" * 80)
    
    for i, (n, p, q) in enumerate(test_cases):
        result = benchmark_single(factorizer, n, p, q, use_optimized)
        results.append(result)
        
        if result["success"]:
            successes += 1
            status = "✓"
        else:
            status = "✗"
        
        total_time += result["time"]
        
        print(f"{i+1:3d}. n={n:20d} ({n.bit_length()} bits) "
              f"[{result['time']:6.3f}s] {status} "
              f"axiom={result['primary_axiom']:6s} "
              f"coh={result['max_coherence']:.3f}")
    
    success_rate = successes / len(test_cases) * 100 if test_cases else 0
    avg_time = total_time / len(test_cases) if test_cases else 0
    
    print(f"\nCategory Summary: {successes}/{len(test_cases)} successful "
          f"({success_rate:.1f}%), Avg time: {avg_time:.3f}s")
    
    return {
        "category": category,
        "results": results,
        "successes": successes,
        "total": len(test_cases),
        "success_rate": success_rate,
        "total_time": total_time,
        "avg_time": avg_time
    }

def analyze_results(all_results: List[Dict]) -> None:
    """Analyze and print detailed statistics."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    total_tests = sum(r["total"] for r in all_results)
    total_successes = sum(r["successes"] for r in all_results)
    total_time = sum(r["total_time"] for r in all_results)
    
    print(f"\nOverall: {total_successes}/{total_tests} successful "
          f"({total_successes/total_tests*100:.1f}%)")
    print(f"Total benchmark time: {total_time:.2f}s")
    
    # Axiom usage analysis
    axiom_counts = defaultdict(int)
    axiom_successes = defaultdict(int)
    
    for category_result in all_results:
        for result in category_result["results"]:
            axiom = result["primary_axiom"]
            axiom_counts[axiom] += 1
            if result["success"]:
                axiom_successes[axiom] += 1
    
    print("\nPrimary Axiom Analysis:")
    for axiom in sorted(axiom_counts.keys()):
        count = axiom_counts[axiom]
        successes = axiom_successes[axiom]
        rate = successes / count * 100 if count > 0 else 0
        print(f"  {axiom:10s}: {successes:3d}/{count:3d} ({rate:5.1f}%)")
    
    # Performance by bit size
    print("\nPerformance by Bit Size:")
    for category_result in all_results:
        category = category_result["category"]
        cat_info = TEST_CATEGORIES[category]
        print(f"  {cat_info['name']:20s}: "
              f"{category_result['success_rate']:5.1f}% success, "
              f"{category_result['avg_time']:6.3f}s avg")
    
    # Method sequence analysis
    method_counts = defaultdict(int)
    for category_result in all_results:
        for result in category_result["results"]:
            if result["success"]:
                for method in result["methods"]:
                    method_counts[method] += 1
    
    print("\nSuccessful Method Sequences:")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {method:30s}: {count:3d} times")

def compare_implementations() -> None:
    """Compare original and optimized implementations."""
    print("\n" + "=" * 80)
    print("IMPLEMENTATION COMPARISON")
    print("=" * 80)
    
    # Generate test cases
    test_cases = []
    for _ in range(5):
        n, p, q = generate_semiprime(32)
        test_cases.append((n, p, q))
    
    # Test both implementations
    original = Factorizer(learning_enabled=True)
    optimized = OptimizedFactorizer(learning_enabled=True)
    
    print("\n{:20s} {:>12s} {:>12s} {:>8s} {:>8s}".format(
        "Number", "Original", "Optimized", "Speedup", "Both OK"))
    print("-" * 70)
    
    for n, p, q in test_cases:
        orig_result = benchmark_single(original, n, p, q, False)
        opt_result = benchmark_single(optimized, n, p, q, True)
        
        speedup = orig_result["time"] / opt_result["time"] if opt_result["time"] > 0 else 0
        both_ok = "✓" if orig_result["success"] and opt_result["success"] else "✗"
        
        print(f"{n:20d} {orig_result['time']:12.6f} {opt_result['time']:12.6f} "
              f"{speedup:8.2f}x {both_ok:>8s}")

def main():
    """Run comprehensive 64-bit benchmark."""
    print("Universal Ontological Factorizer - 64-bit Comprehensive Benchmark")
    print("=" * 80)
    
    # Initialize factorizer with learning enabled
    factorizer = Factorizer(learning_enabled=True)
    
    # Warm up the factorizer with small cases
    print("\nWarming up factorizer...")
    warmup_cases = [(15, 3, 5), (21, 3, 7), (35, 5, 7)]
    for n, p, q in warmup_cases:
        factorizer.factorize(n)
    
    all_results = []
    
    # Run benchmarks for each category
    for category in ["small", "medium", "large", "xlarge"]:
        # Generate test cases
        test_cases = []
        cat_info = TEST_CATEGORIES[category]
        
        for _ in range(cat_info["count"]):
            bit_size = random.randint(*cat_info["bit_range"])
            n, p, q = generate_semiprime(bit_size)
            test_cases.append((n, p, q))
        
        # Run benchmark
        results = run_category_benchmark(factorizer, category, test_cases)
        all_results.append(results)
    
    # Test special cases
    print("\n" + "=" * 80)
    print("SPECIAL TEST CASES")
    print("=" * 80)
    
    special_cases = generate_special_semiprimes()
    special_results = run_category_benchmark(
        factorizer, "special", special_cases[:15]
    )
    all_results.append(special_results)
    
    # Analyze results
    analyze_results(all_results)
    
    # Compare implementations
    compare_implementations()
    
    print("\n" + "=" * 80)
    print("Benchmark completed!")

if __name__ == "__main__":
    main()
