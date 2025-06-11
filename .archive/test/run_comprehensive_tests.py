"""
Comprehensive test runner for the entire factorizer system.

Runs all tests across all axioms and the main factorizer.
"""

import sys
import os
import unittest
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_axiom_tests():
    """Run tests for all individual axioms."""
    print("\n" + "="*70)
    print("RUNNING AXIOM TESTS")
    print("="*70)
    
    axiom_results = {}
    
    # Test each axiom
    for axiom_num in range(1, 6):
        axiom_dir = f"axiom{axiom_num}"
        test_script = os.path.join(axiom_dir, "test", "run_all_tests.py")
        
        if os.path.exists(test_script):
            print(f"\nTesting Axiom {axiom_num}...")
            print("-" * 50)
            
            start_time = time.time()
            
            # Import and run the axiom's test runner
            try:
                axiom_module = f"axiom{axiom_num}.test.run_all_tests"
                axiom_test = __import__(axiom_module, fromlist=['run_all_tests'])
                
                if hasattr(axiom_test, 'run_all_tests'):
                    result = axiom_test.run_all_tests()
                    axiom_results[f"axiom{axiom_num}"] = result
                else:
                    print(f"Warning: No run_all_tests function in {axiom_module}")
                    axiom_results[f"axiom{axiom_num}"] = False
                    
            except Exception as e:
                print(f"Error running tests for Axiom {axiom_num}: {e}")
                axiom_results[f"axiom{axiom_num}"] = False
            
            elapsed = time.time() - start_time
            print(f"Axiom {axiom_num} tests completed in {elapsed:.2f}s")
        else:
            print(f"No test runner found for Axiom {axiom_num}")
            axiom_results[f"axiom{axiom_num}"] = False
    
    return axiom_results


def run_factorizer_tests():
    """Run the main factorizer integration tests."""
    print("\n" + "="*70)
    print("RUNNING FACTORIZER INTEGRATION TESTS")
    print("="*70)
    
    # Import the factorizer test module
    from test_factorizer import run_all_tests
    
    start_time = time.time()
    result = run_all_tests()
    elapsed = time.time() - start_time
    
    print(f"\nFactorizer tests completed in {elapsed:.2f}s")
    
    return result


def run_benchmark_tests():
    """Run benchmark tests if available."""
    print("\n" + "="*70)
    print("RUNNING BENCHMARK TESTS")
    print("="*70)
    
    benchmark_script = os.path.join("benchmark", "quick_benchmark.py")
    
    if os.path.exists(benchmark_script):
        try:
            # Import and run benchmark
            import benchmark.quick_benchmark as bench
            
            print("\nRunning quick benchmark...")
            start_time = time.time()
            
            # Run a subset of benchmarks for testing
            from factorizer import Factorizer
            factorizer = Factorizer()
            
            test_numbers = [143, 667, 3233, 10403]
            successes = 0
            
            for n in test_numbers:
                try:
                    p, q = factorizer.factorize(n)
                    if p * q == n and p > 1 and q > 1:
                        successes += 1
                        print(f"✓ {n} = {p} × {q}")
                    else:
                        print(f"✗ {n} factorization failed")
                except Exception as e:
                    print(f"✗ {n} error: {e}")
            
            elapsed = time.time() - start_time
            print(f"\nBenchmark completed in {elapsed:.2f}s")
            print(f"Success rate: {successes}/{len(test_numbers)}")
            
            return successes == len(test_numbers)
            
        except Exception as e:
            print(f"Error running benchmarks: {e}")
            return False
    else:
        print("No benchmark tests found")
        return True  # Not a failure if benchmarks don't exist


def print_summary(axiom_results, factorizer_result, benchmark_result):
    """Print a summary of all test results."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    # Axiom results
    print("\nAxiom Tests:")
    for axiom, result in axiom_results.items():
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {axiom}: {status}")
    
    # Factorizer result
    factorizer_status = "PASSED" if factorizer_result else "FAILED"
    factorizer_symbol = "✓" if factorizer_result else "✗"
    print(f"\nFactorizer Integration: {factorizer_symbol} {factorizer_status}")
    
    # Benchmark result
    benchmark_status = "PASSED" if benchmark_result else "FAILED"
    benchmark_symbol = "✓" if benchmark_result else "✗"
    print(f"Benchmark Tests: {benchmark_symbol} {benchmark_status}")
    
    # Overall result
    all_axioms_passed = all(axiom_results.values())
    all_passed = all_axioms_passed and factorizer_result and benchmark_result
    
    print("\n" + "-"*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
        failed_count = (
            len([r for r in axiom_results.values() if not r]) +
            (0 if factorizer_result else 1) +
            (0 if benchmark_result else 1)
        )
        print(f"  Failed: {failed_count}")
    
    return all_passed


def main():
    """Main test runner."""
    print("UOR/Prime Axioms Factorizer - Comprehensive Test Suite")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run all test suites
    axiom_results = run_axiom_tests()
    factorizer_result = run_factorizer_tests()
    benchmark_result = run_benchmark_tests()
    
    # Print summary
    all_passed = print_summary(axiom_results, factorizer_result, benchmark_result)
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal test time: {total_elapsed:.2f}s")
    
    # Exit with appropriate code
    exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
