"""
Quick test of the 64-bit benchmark to verify it works correctly
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factorizer_64bit_benchmark import Factorizer64BitBenchmark, BenchmarkCase


def test_quick():
    """Run a quick test with just a few cases"""
    print("Running quick 64-bit benchmark test...")
    
    # Create benchmark with fixed seed
    benchmark = Factorizer64BitBenchmark(use_learning=True, seed=42)
    
    # Override with just a few test cases
    benchmark.test_cases = [
        # 8-bit
        BenchmarkCase(15, 3, 5, 8, "easy"),
        BenchmarkCase(77, 7, 11, 8, "easy"),
        
        # 16-bit  
        BenchmarkCase(323, 17, 19, 16, "easy"),
        BenchmarkCase(667, 23, 29, 16, "easy"),
        
        # 32-bit
        BenchmarkCase(1147, 31, 37, 32, "easy"),
        
        # Special cases
        BenchmarkCase(25, 5, 5, 8, "special"),  # Perfect square
        BenchmarkCase(6, 2, 3, 8, "fibonacci"),  # Fibonacci primes
    ]
    
    # Run benchmark
    benchmark.run_benchmark()
    
    # Generate report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Save results
    benchmark.save_results("quick_test_results.json")
    
    print("\nQuick test complete!")


if __name__ == "__main__":
    test_quick()
