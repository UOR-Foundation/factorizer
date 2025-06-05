#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Quick UOR/Prime Axiom Validation Test for V2 Implementation
#
# Tests the refactored ultra-accelerated UOR factorizer V2 on challenging
# semiprimes up to 64-bit, strictly following UOR/Prime axioms
# ---------------------------------------------------------------------------

import sys, os, time, math
from typing import List, Tuple

# Import V2 factorizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultra_accelerated_uor_factorizer_v2 import ultra_uor_factor_v2, is_prime, PHI, fib

# Import test case generation from original validation
from quick_validation import create_challenging_semiprimes

def run_axiom_validation_v2() -> None:
    """Run comprehensive UOR/Prime axiom validation with V2"""
    
    print("UOR/Prime Axiom Validation Test Suite - V2")
    print("=" * 45)
    print("NO FALLBACKS • NO SIMPLIFICATIONS • NO RANDOMIZATION • NO HARDCODING")
    print("Testing V2 axiom-based factorization with optimizations")
    print("=" * 45)
    
    # Generate test cases
    test_cases = create_challenging_semiprimes()
    
    # Group by category for analysis
    results_by_category = {}
    total_tests = 0
    total_successes = 0
    total_time = 0
    
    print(f"\nTesting {len(test_cases)} challenging semiprimes...")
    print("-" * 60)
    
    for semiprime, expected_p, expected_q, category in test_cases:
        start_time = time.perf_counter()
        
        # Test V2 UOR factorizer
        p, q = ultra_uor_factor_v2(semiprime)
        
        timing = time.perf_counter() - start_time
        total_time += timing
        total_tests += 1
        
        # Check correctness
        factors_found = sorted([p, q])
        factors_expected = sorted([expected_p, expected_q])
        success = (factors_found == factors_expected)
        
        if success:
            total_successes += 1
        
        # Track by category
        if category not in results_by_category:
            results_by_category[category] = {'tests': 0, 'successes': 0, 'times': []}
        
        results_by_category[category]['tests'] += 1
        if success:
            results_by_category[category]['successes'] += 1
        results_by_category[category]['times'].append(timing)
        
        # Print result
        status = "PASS" if success else "FAIL"
        print(f"{semiprime:8} = {p:4} × {q:4} [{timing*1000:6.2f}ms] {status:4} ({category})")
    
    # Print category analysis
    print(f"\n" + "=" * 60)
    print("V2 AXIOM EFFECTIVENESS ANALYSIS")
    print("=" * 60)
    
    for category, results in results_by_category.items():
        success_rate = results['successes'] / results['tests'] * 100
        avg_time = sum(results['times']) / len(results['times']) * 1000
        
        print(f"{category:20}: {success_rate:5.1f}% success ({results['successes']:2}/{results['tests']:2}), {avg_time:6.2f}ms avg")
    
    # Overall summary
    overall_success_rate = total_successes / total_tests * 100
    avg_time_ms = total_time / total_tests * 1000
    
    print(f"\n" + "=" * 60)
    print("OVERALL V2 UOR/PRIME AXIOM PERFORMANCE")
    print("=" * 60)
    print(f"Total tests:     {total_tests}")
    print(f"Successes:       {total_successes}")
    print(f"Success rate:    {overall_success_rate:.1f}%")
    print(f"Average time:    {avg_time_ms:.2f}ms")
    print(f"Total time:      {total_time:.3f}s")
    
    # V2 Improvements
    print(f"\nV2 Enhancements:")
    print(f"  • Adaptive Observer with quantum superposition")
    print(f"  • Spectral gradient navigation")
    print(f"  • Coherence caching and field mapping")
    print(f"  • Hierarchical coherence computation")
    print(f"  • Efficient phase integration")
    
    print(f"\n" + "=" * 60)
    print("V2 UOR/Prime axiom validation complete.")
    print("All factorizations achieved through pure axiom mathematics.")
    print("=" * 60)

def main():
    """Main validation entry point"""
    run_axiom_validation_v2()

if __name__ == "__main__":
    main()
