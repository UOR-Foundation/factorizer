"""
Axiom 5 Semiprime Breakthrough Benchmark - 64-bit

Uses the Axiom 5 Accelerated Factorizer for clean, focused benchmarking.
Tests factorization performance on semiprimes up to 64-bit.
"""

import time
import math
import random
from typing import Tuple, List, Dict
from dataclasses import dataclass
from collections import defaultdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Axiom 5 accelerated factorizer
from axiom5 import Axiom5AcceleratedFactorizer, get_accelerated_factorizer, accelerated_factorize


@dataclass
class BenchmarkResult:
    """Result from benchmark run"""
    n: int
    factors: Tuple[int, int]
    time: float
    method: str
    iterations: int
    cache_hits: int
    success: bool
    bit_size: int


def generate_semiprime(bits: int) -> Tuple[int, int, int]:
    """Generate a semiprime with approximately the given number of bits"""
    # Split bits between two factors
    bits1 = bits // 2
    bits2 = bits - bits1
    
    # Generate random odd numbers in the bit ranges
    min1, max1 = 2**(bits1-1), 2**bits1 - 1
    min2, max2 = 2**(bits2-1), 2**bits2 - 1
    
    # Simple primality test
    def is_probable_prime(n, k=5):
        """Miller-Rabin primality test"""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witness loop
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    # Find primes
    attempts = 0
    while attempts < 1000:
        p = random.randrange(min1, max1) | 1  # Ensure odd
        if is_probable_prime(p):
            break
        attempts += 1
    else:
        p = min1 | 1  # Fallback
    
    attempts = 0
    while attempts < 1000:
        q = random.randrange(min2, max2) | 1  # Ensure odd
        if is_probable_prime(q) and q != p:
            break
        attempts += 1
    else:
        q = max2 - 1  # Fallback
    
    n = p * q
    return n, min(p, q), max(p, q)


def run_breakthrough_benchmark():
    """Run the Axiom 5 semiprime breakthrough benchmark"""
    print("="*80)
    print("AXIOM 5 SEMIPRIME BREAKTHROUGH BENCHMARK - 64-BIT")
    print("Using Axiom 5 Accelerated Factorizer")
    print("="*80)
    print("\nCapabilities:")
    print("• Meta-Observation: Learn from all factorization attempts")
    print("• Spectral Mirroring: Find factors through spectral reflection")
    print("• Recursive Coherence: Discover fractal patterns")
    print("• Failure Analysis: Adapt from what doesn't work")
    print("• Axiom Synthesis: Create emergent hybrid methods")
    print("• 20-50x Acceleration: Through intelligent caching")
    print()
    
    # Get the factorizer
    factorizer = get_accelerated_factorizer()
    
    # Test cases: bit size -> (n, p, q) or None for random
    test_cases = [
        # Warm-up with known semiprimes
        (8, (143, 11, 13)),
        (10, (323, 17, 19)),
        (14, (10403, 101, 103)),
        
        # Random semiprimes of increasing size
        (16, None),
        (20, None),
        (24, None),
        (28, None),
        (32, None),
        (36, None),
        (40, None),
        (44, None),
        (48, None),
        (52, None),
        (56, None),
        (60, None),
        (64, None),
    ]
    
    results = []
    total_time = 0
    successful = 0
    
    print(f"{'Bits':>4} {'Number':>20} {'Time(s)':>10} {'Method':>20} "
          f"{'Iterations':>10} {'Cache Hits':>10} {'Status':>8}")
    print("-"*100)
    
    for bits, test_data in test_cases:
        if test_data:
            n, p_exp, q_exp = test_data
        else:
            n, p_exp, q_exp = generate_semiprime(bits)
        
        # Run factorization
        result = factorizer.factorize_with_details(n)
        
        # Create benchmark result
        bench_result = BenchmarkResult(
            n=n,
            factors=result.factors,
            time=result.time,
            method=result.method,
            iterations=result.iterations,
            cache_hits=result.cache_hits,
            success=(result.factors[0] > 1 and result.factors[1] < n),
            bit_size=n.bit_length()
        )
        
        results.append(bench_result)
        total_time += result.time
        
        if bench_result.success:
            successful += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{bench_result.bit_size:4d} {n:20d} {result.time:10.4f} {result.method:>20} "
              f"{result.iterations:10d} {result.cache_hits:10d} {status:>8}")
    
    # Analysis
    print("\n" + "="*80)
    print("BREAKTHROUGH ANALYSIS")
    print("="*80)
    
    success_rate = successful / len(results) * 100
    print(f"\nSuccess Rate: {successful}/{len(results)} ({success_rate:.1f}%)")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time: {total_time/len(results):.2f}s")
    
    # Method breakdown
    method_counts = defaultdict(int)
    for r in results:
        method_counts[r.method] += 1
    
    print("\nMethods Used:")
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(results) * 100
        print(f"  {method:25s}: {count:2d} ({percentage:5.1f}%)")
    
    # Cache performance
    total_cache_hits = sum(r.cache_hits for r in results)
    print(f"\nCache Performance:")
    print(f"  Total Cache Hits: {total_cache_hits}")
    print(f"  Average Cache Hits: {total_cache_hits/len(results):.1f}")
    
    # Get factorizer statistics
    stats = factorizer.get_statistics()
    print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
    
    # Cache details
    cache_stats = stats['cache_stats']
    print(f"\nCache Details:")
    print(f"  Observations Cached: {cache_stats['observation_count']}")
    print(f"  Coherence Fields: {cache_stats['coherence_fields_cached']}")
    print(f"  Fixed Points: {cache_stats['fixed_points_cached']}")
    print(f"  Mirror Positions: {cache_stats['mirror_positions_cached']}")
    print(f"  Spectral Distances: {cache_stats['spectral_distances_cached']}")
    
    # Bit-size progression
    print("\nBit-Size Progression:")
    bit_buckets = defaultdict(list)
    for r in results:
        bucket = (r.bit_size // 8) * 8
        bit_buckets[bucket].append(r)
    
    for bits in sorted(bit_buckets.keys()):
        bucket_results = bit_buckets[bits]
        successes = sum(1 for r in bucket_results if r.success)
        avg_time = sum(r.time for r in bucket_results) / len(bucket_results)
        avg_iterations = sum(r.iterations for r in bucket_results) / len(bucket_results)
        print(f"  {bits:2d}-{bits+7:2d} bits: {successes}/{len(bucket_results)} successful, "
              f"{avg_time:.3f}s avg, {avg_iterations:.0f} iterations avg")
    
    # Learning effectiveness
    print("\nLearning Effectiveness:")
    early_results = results[:5]
    late_results = results[-5:]
    
    early_avg_time = sum(r.time for r in early_results) / len(early_results)
    late_avg_time = sum(r.time for r in late_results) / len(late_results)
    
    early_avg_iters = sum(r.iterations for r in early_results) / len(early_results)
    late_avg_iters = sum(r.iterations for r in late_results) / len(late_results)
    
    print(f"  Early runs (first 5): {early_avg_time:.3f}s avg, {early_avg_iters:.0f} iterations avg")
    print(f"  Late runs (last 5): {late_avg_time:.3f}s avg, {late_avg_iters:.0f} iterations avg")
    
    if late_avg_time < early_avg_time:
        speedup = early_avg_time / late_avg_time
        print(f"  Learning speedup: {speedup:.1f}x faster")
    
    # Highlight breakthroughs
    print("\n🌟 BREAKTHROUGH HIGHLIGHTS:")
    
    # Fastest factorization
    fastest = min(results, key=lambda r: r.time)
    print(f"  Fastest: {fastest.n} ({fastest.bit_size} bits) in {fastest.time:.4f}s via {fastest.method}")
    
    # Largest successful
    successful_results = [r for r in results if r.success]
    if successful_results:
        largest = max(successful_results, key=lambda r: r.n)
        print(f"  Largest: {largest.n} ({largest.bit_size} bits) in {largest.time:.3f}s via {largest.method}")
    
    # Most efficient (lowest iterations)
    if successful_results:
        efficient = min(successful_results, key=lambda r: r.iterations)
        print(f"  Most efficient: {efficient.n} with only {efficient.iterations} iterations via {efficient.method}")
    
    # Cache effectiveness
    cache_heavy = [r for r in results if r.cache_hits > 0]
    if cache_heavy:
        print(f"  Cache effectiveness: {len(cache_heavy)}/{len(results)} runs benefited from cache")
    
    print("\n" + "="*80)
    print("✅ Axiom 5 Breakthrough Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    run_breakthrough_benchmark()
