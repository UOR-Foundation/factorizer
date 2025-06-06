"""
Quick benchmark runner for rapid performance assessment
Provides essential metrics without full comprehensive testing
"""

import time
import sys
import os
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from the module in the same directory
from benchmark_runner import BenchmarkRunner

def quick_benchmark():
    """Run a quick benchmark with essential test cases"""
    print("UOR/Prime Axioms Factorizer - Quick Benchmark")
    print("=" * 45)
    
    try:
        # Import within the function to debug
        from benchmark_runner import BenchmarkRunner
        print("Import successful")
        
        # Reduced test set for quick evaluation
        quick_test_cases = [
            (15, [3, 5]),
            (35, [5, 7]),
            (77, [7, 11])
        ]
        
        runner = BenchmarkRunner()
        runner.test_numbers = quick_test_cases
        print("Runner initialized")
        
        start_time = time.time()
        results = runner.run_comprehensive_benchmark()
        end_time = time.time()
        
        print(f"\nQuick benchmark completed in {end_time - start_time:.2f}s")
        print("\nSUMMARY:")
        print("-" * 20)
        
        # Quick summary
        for axiom, benchmark in results.items():
            print(f"{axiom}: {benchmark.success_rate:.1%} success, {benchmark.avg_time:.4f}s avg")
        
        # Overall stats
        total_success = sum(b.successes for b in results.values())
        total_tests = sum(b.total_runs for b in results.values())
        overall_rate = total_success / total_tests if total_tests > 0 else 0
        
        print(f"\nOverall: {overall_rate:.1%} success rate ({total_success}/{total_tests})")
        
        # Quick recommendations
        print("\nQUICK ASSESSMENT:")
        if results:
            best_axiom = max(results.items(), key=lambda x: x[1].success_rate)
            fastest_axiom = min(results.items(), key=lambda x: x[1].avg_time)
            
            print(f"- Best performing: {best_axiom[0]} ({best_axiom[1].success_rate:.1%})")
            print(f"- Fastest: {fastest_axiom[0]} ({fastest_axiom[1].avg_time:.4f}s)")
            
            if overall_rate < 0.5:
                print("- WARNING: Overall success rate below 50%")
            elif overall_rate > 0.8:
                print("- EXCELLENT: High overall success rate")
            else:
                print("- GOOD: Reasonable performance levels")
        
    except Exception as e:
        print(f"Error in quick benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_benchmark()
