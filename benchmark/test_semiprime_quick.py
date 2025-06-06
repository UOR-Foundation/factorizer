#!/usr/bin/env python3
"""
Quick test of the semiprime benchmark functionality
"""

import sys
sys.path.append('/workspaces/factorizer')

from benchmark.semiprime_benchmark import SemiprimeBenchmark

def quick_test():
    """Test a few small semiprimes to verify functionality"""
    print("=== Quick Semiprime Benchmark Test ===")
    
    benchmark = SemiprimeBenchmark()
    
    # Test a few known small semiprimes
    test_cases = [
        (15, [3, 5], "small"),      # 4-bit
        (35, [5, 7], "small"),      # 6-bit  
        (77, [7, 11], "small"),     # 7-bit
        (143, [11, 13], "small"),   # 8-bit
        (323, [17, 19], "small"),   # 9-bit
    ]
    
    results = []
    for n, factors, difficulty in test_cases:
        print(f"\nTesting n={n} (factors: {factors})")
        try:
            result = benchmark.benchmark_semiprime(n, factors, difficulty)
            results.append(result)
            
            success_rate = (result.success_count / 5) * 100
            print(f"Result: {result.success_count}/5 axioms ({success_rate:.1f}%) in {result.total_time:.3f}s")
            
            # Show which axioms succeeded
            for axiom, factor_found in result.found_factors.items():
                if factor_found in factors:
                    print(f"  ✓ {axiom}: {factor_found}")
                else:
                    print(f"  ✗ {axiom}: {factor_found}")
                    
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== Quick Test Summary ===")
    if results:
        avg_success = sum(r.success_count for r in results) / len(results)
        print(f"Average axioms succeeding: {avg_success:.1f}/5")
        print(f"Total breakthrough cases (3+ axioms): {len([r for r in results if r.success_count >= 3])}")
    
    print("Quick test complete!")

if __name__ == "__main__":
    quick_test()
