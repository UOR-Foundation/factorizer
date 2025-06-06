#!/usr/bin/env python3
"""
Focused debug test for specific axiom issues
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_axiom3_interference():
    """Test the interference function specifically"""
    print("Testing Axiom 3 interference...")
    
    try:
        from axiom3.interference import prime_fib_interference, interference_extrema
        
        n = 15
        print(f"Testing with n={n}")
        
        # Test prime_fib_interference
        print("Calling prime_fib_interference...")
        interference = prime_fib_interference(n)
        print(f"Interference result type: {type(interference)}")
        print(f"Interference length: {len(interference) if hasattr(interference, '__len__') else 'N/A'}")
        
        # Test interference_extrema
        print("Calling interference_extrema...")
        extrema = interference_extrema(n, top=5)
        print(f"Extrema result: {extrema}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_performance_profiler_memory():
    """Test the memory measurement issue"""
    print("\nTesting memory measurement...")
    
    import psutil
    import time
    
    process = psutil.Process()
    
    # Test memory measurement
    start_memory = process.memory_info().rss
    
    # Do some work
    data = [i**2 for i in range(10000)]
    time.sleep(0.001)
    
    end_memory = process.memory_info().rss
    memory_diff = end_memory - start_memory
    
    print(f"Start memory: {start_memory} bytes")
    print(f"End memory: {end_memory} bytes") 
    print(f"Memory difference: {memory_diff} bytes ({memory_diff/1024:.2f} KB)")
    
    # Try a different approach using memory usage tracking
    try:
        import tracemalloc
        tracemalloc.start()
        
        # Do some work
        more_data = [i**3 for i in range(5000)]
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Current memory usage: {current / 1024:.2f} KB")
        print(f"Peak memory usage: {peak / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Tracemalloc error: {e}")

def test_benchmark_runner():
    """Test the benchmark runner"""
    print("\nTesting benchmark runner...")
    
    try:
        from benchmark_runner import BenchmarkRunner
        print("Successfully imported BenchmarkRunner")
        
        runner = BenchmarkRunner()
        print(f"Runner created with {len(runner.test_numbers)} test cases")
        
        # Try a simple single benchmark
        result = runner.benchmark_axiom3(15, [3, 5])
        print(f"Single axiom3 benchmark result: {result.success}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_axiom3_interference()
    test_performance_profiler_memory()
    test_benchmark_runner()
