"""
Benchmark Prime Sieve with true semiprimes and performance optimization
"""

import time
import json
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prime_sieve import PrimeSieve


def benchmark_prime_sieve():
    """Benchmark Prime Sieve with verified semiprimes."""
    print("="*60)
    print("PRIME SIEVE PERFORMANCE BENCHMARK")
    print("="*60)
    
    # True semiprimes (product of exactly two primes)
    test_cases = [
        # Small semiprimes
        (15, 3, 5),          # 4 bit
        (21, 3, 7),          # 5 bit
        (35, 5, 7),          # 6 bit
        (77, 7, 11),         # 7 bit
        (143, 11, 13),       # 8 bit
        
        # Medium semiprimes
        (323, 17, 19),       # 9 bit
        (1147, 31, 37),      # 11 bit
        (2021, 43, 47),      # 11 bit
        (3599, 59, 61),      # 12 bit
        
        # Larger semiprimes
        (5767, 71, 83),      # 13 bit
        (10403, 101, 103),   # 14 bit (twin primes)
        (20213, 139, 149),   # 15 bit
        (50851, 223, 229),   # 16 bit
        
        # Large semiprimes
        (121061, 347, 349),  # 17 bit (twin primes)
        (341887, 577, 593),  # 19 bit
        (1299071, 1117, 1163), # 21 bit
        
        # Very large semiprimes (if you want to test limits)
        # (10344599, 3217, 3221),  # 24 bit (twin primes)
        # (100140049, 9973, 10039), # 27 bit
    ]
    
    print(f"\nTesting {len(test_cases)} true semiprimes...")
    print("-" * 60)
    
    sieve = PrimeSieve(enable_learning=True)
    
    results = []
    total_time = 0
    successful = 0
    
    # Header
    print(f"{'Number':>10} | {'Expected':>15} | {'Found':>15} | {'Time':>8} | {'Method':>10} | Status")
    print("-" * 75)
    
    for n, expected_p, expected_q in test_cases:
        start = time.time()
        result = sieve.factor_with_details(n)
        elapsed = time.time() - start
        
        p, q = result.factors
        is_correct = (p == expected_p and q == expected_q) or \
                    (p == expected_q and q == expected_p)
        
        if is_correct:
            successful += 1
            status = "✓"
        else:
            status = "✗"
        
        total_time += elapsed
        
        print(f"{n:10d} | {expected_p:6d} × {expected_q:6d} | "
              f"{p:6d} × {q:6d} | {elapsed:7.4f}s | {result.method:>10s} | {status}")
        
        results.append({
            'n': n,
            'bit_length': n.bit_length(),
            'expected': (expected_p, expected_q),
            'found': (p, q),
            'correct': is_correct,
            'time': elapsed,
            'method': result.method,
            'iterations': result.iterations,
            'confidence': result.confidence
        })
    
    # Summary
    print("-" * 75)
    print(f"Success rate: {successful}/{len(test_cases)} = {successful/len(test_cases):.1%}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time: {total_time/len(test_cases):.4f}s")
    
    # Performance by bit length
    print("\n" + "="*60)
    print("PERFORMANCE BY BIT LENGTH")
    print("="*60)
    
    bit_groups = {}
    for r in results:
        bit_len = r['bit_length']
        if bit_len not in bit_groups:
            bit_groups[bit_len] = []
        bit_groups[bit_len].append(r)
    
    print(f"{'Bits':>4} | {'Count':>5} | {'Avg Time':>10} | {'Max Time':>10} | Success")
    print("-" * 45)
    
    for bits in sorted(bit_groups.keys()):
        group = bit_groups[bits]
        avg_time = sum(r['time'] for r in group) / len(group)
        max_time = max(r['time'] for r in group)
        success = sum(1 for r in group if r['correct']) / len(group)
        
        print(f"{bits:4d} | {len(group):5d} | {avg_time:9.4f}s | {max_time:9.4f}s | {success:6.1%}")
    
    # Method distribution
    print("\n" + "="*60)
    print("METHOD DISTRIBUTION")
    print("="*60)
    
    method_counts = {}
    for r in results:
        method = r['method']
        if method not in method_counts:
            method_counts[method] = 0
        method_counts[method] += 1
    
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(results) * 100
        print(f"{method:>12s}: {count:3d} ({percentage:5.1f}%)")
    
    # Save results
    output_file = "prime_sieve_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'test_cases': test_cases,
            'results': results,
            'summary': {
                'total_tests': len(test_cases),
                'successful': successful,
                'success_rate': successful / len(test_cases),
                'total_time': total_time,
                'average_time': total_time / len(test_cases)
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Get statistics
    stats = sieve.get_statistics()
    print("\n" + "="*60)
    print("PRIME SIEVE STATISTICS")
    print("="*60)
    print(f"Total factorizations: {stats['total_factorizations']}")
    print(f"Successful factorizations: {stats['successful_factorizations']}")
    print(f"Overall success rate: {stats['success_rate']:.1%}")
    
    if 'meta_observer' in stats:
        meta = stats['meta_observer']
        print(f"\nMeta-Observer Learning:")
        print(f"  Observations: {meta['total_observations']}")
        print(f"  Strategies learned: {meta['learned_strategies']}")


def compare_configurations():
    """Compare different parameter configurations."""
    print("\n" + "="*60)
    print("CONFIGURATION COMPARISON")
    print("="*60)
    
    # Test a few key semiprimes
    test_numbers = [
        (323, 17, 19),      # Small
        (10403, 101, 103),  # Medium
        (1299071, 1117, 1163), # Large
    ]
    
    configs = {
        'Current': {},  # Use current settings
        # Add other configs here if we implement parameterization
    }
    
    for config_name, config in configs.items():
        print(f"\n{config_name} configuration:")
        sieve = PrimeSieve(enable_learning=False)
        
        total_time = 0
        for n, p, q in test_numbers:
            start = time.time()
            factors = sieve.factor(n)
            elapsed = time.time() - start
            total_time += elapsed
            
            correct = (factors == (p, q) or factors == (q, p))
            print(f"  {n}: {elapsed:.4f}s {correct and '✓' or '✗'}")
        
        print(f"  Total: {total_time:.4f}s")


if __name__ == "__main__":
    # Run main benchmark
    benchmark_prime_sieve()
    
    # Compare configurations
    compare_configurations()
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)
