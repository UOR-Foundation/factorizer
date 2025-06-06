"""
Performance profiler for detailed analysis of axiom performance
Includes memory usage, algorithmic complexity analysis, and optimization suggestions
"""

import time
import psutil
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom3.coherence import CoherenceCache
from axiom4.resonance_memory import ResonanceMemory
from axiom5.meta_observer import MetaObserver

@dataclass
class PerformanceMetrics:
    """Detailed performance metrics for a single operation"""
    operation_name: str
    input_size: int
    execution_time: float
    memory_used: float
    cpu_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    iterations: int = 1

class PerformanceProfiler:
    """Detailed performance profiler with system metrics"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss
        self.metrics: List[PerformanceMetrics] = []
        
    def profile_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Profile a single operation with detailed metrics"""
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Initial measurements
        start_cpu = self.process.cpu_percent()
        start_time = time.perf_counter()
        
        # Execute operation
        result = operation_func(*args, **kwargs)
        
        # Final measurements
        end_time = time.perf_counter()
        end_cpu = self.process.cpu_percent()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_used = peak  # Use peak memory from tracemalloc
        cpu_usage = (start_cpu + end_cpu) / 2
        
        # Determine input size
        input_size = 0
        if args:
            if isinstance(args[0], int):
                input_size = args[0]
            elif isinstance(args[0], (list, tuple)):
                input_size = len(args[0])
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            input_size=input_size,
            execution_time=execution_time,
            memory_used=memory_used,
            cpu_usage=cpu_usage
        )
        
        self.metrics.append(metrics)
        return result, metrics
    
    def profile_scaling_behavior(self, operation_func, sizes: List[int], iterations: int = 5):
        """Profile how operation scales with input size"""
        import tracemalloc
        scaling_results = []
        
        for size in sizes:
            times = []
            memories = []
            
            for _ in range(iterations):
                tracemalloc.start()
                start_time = time.perf_counter()
                
                # Generate appropriate input for size
                try:
                    if hasattr(operation_func, '__name__'):
                        if 'spectral' in operation_func.__name__:
                            result = operation_func(size)
                        elif 'coherence' in operation_func.__name__:
                            result = operation_func(size, size+1, size*(size+1))
                        else:
                            result = operation_func(size)
                except Exception as e:
                    print(f"Error profiling {operation_func} with size {size}: {e}")
                    continue
                
                end_time = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                times.append(end_time - start_time)
                memories.append(peak)
            
            if times:  # Only add if we have valid measurements
                avg_time = sum(times) / len(times)
                avg_memory = sum(memories) / len(memories)
                scaling_results.append((size, avg_time, avg_memory))
        
        return scaling_results
    
    def analyze_cache_performance(self, cache_obj):
        """Analyze cache hit rates and efficiency"""
        if hasattr(cache_obj, 'spectral_cache'):
            spectral_size = len(cache_obj.spectral_cache)
            coherence_size = len(cache_obj.coherence_cache)
            max_size = cache_obj.max_size
            
            return {
                'spectral_cache_size': spectral_size,
                'coherence_cache_size': coherence_size,
                'total_cache_usage': (spectral_size + coherence_size) / (2 * max_size),
                'cache_efficiency': (spectral_size + coherence_size) / max_size
            }
        return {}
    
    def generate_complexity_analysis(self) -> str:
        """Analyze algorithmic complexity from timing data"""
        report = []
        report.append("Algorithmic Complexity Analysis")
        report.append("=" * 40)
        
        # Group metrics by operation
        operations = {}
        for metric in self.metrics:
            if metric.operation_name not in operations:
                operations[metric.operation_name] = []
            operations[metric.operation_name].append(metric)
        
        for op_name, op_metrics in operations.items():
            if len(op_metrics) < 3:
                continue
                
            # Sort by input size
            op_metrics.sort(key=lambda x: x.input_size)
            
            sizes = [m.input_size for m in op_metrics]
            times = [m.execution_time for m in op_metrics]
            
            # Simple complexity estimation
            if len(sizes) >= 3:
                # Check for linear, quadratic, or logarithmic growth
                ratios = []
                for i in range(1, len(times)):
                    if times[i-1] > 0:
                        ratios.append(times[i] / times[i-1])
                
                avg_ratio = sum(ratios) / len(ratios) if ratios else 1
                
                complexity_estimate = "O(1)"
                if avg_ratio > 1.5:
                    if avg_ratio > 3:
                        complexity_estimate = "O(n²) or worse"
                    else:
                        complexity_estimate = "O(n log n)"
                elif avg_ratio > 1.1:
                    complexity_estimate = "O(n)"
                
                report.append(f"{op_name}:")
                report.append(f"  Estimated Complexity: {complexity_estimate}")
                report.append(f"  Average Time Ratio: {avg_ratio:.2f}")
                report.append(f"  Size Range: {min(sizes)} - {max(sizes)}")
                report.append("")
        
        return "\n".join(report)
    
    def generate_memory_analysis(self) -> str:
        """Analyze memory usage patterns"""
        report = []
        report.append("Memory Usage Analysis")
        report.append("=" * 30)
        
        total_memory = sum(m.memory_used for m in self.metrics)
        avg_memory = total_memory / len(self.metrics) if self.metrics else 0
        
        report.append(f"Total Memory Used: {total_memory / 1024 / 1024:.2f} MB")
        report.append(f"Average per Operation: {avg_memory / 1024:.2f} KB")
        
        # Find memory-intensive operations
        memory_intensive = sorted(self.metrics, key=lambda x: x.memory_used, reverse=True)[:5]
        
        report.append("\nMost Memory-Intensive Operations:")
        for i, metric in enumerate(memory_intensive, 1):
            report.append(f"  {i}. {metric.operation_name}: {metric.memory_used / 1024:.2f} KB")
        
        return "\n".join(report)

def profile_axiom_performance():
    """Profile performance of each axiom's core operations"""
    profiler = PerformanceProfiler()
    
    # Test sizes for scaling analysis
    test_sizes = [100, 500, 1000, 2000, 5000]
    
    print("Profiling Axiom Performance...")
    print("=" * 40)
    
    # Profile Axiom 1 operations
    print("Profiling Axiom 1...")
    from axiom1.prime_core import is_prime
    from axiom1.prime_geodesic import PrimeGeodesic
    
    for size in test_sizes[:4]:  # Limit for prime operations
        _, metrics = profiler.profile_operation(f"is_prime_{size}", is_prime, size)
        print(f"  is_prime({size}): {metrics.execution_time:.6f}s")
    
    # Profile Axiom 2 operations  
    print("Profiling Axiom 2...")
    from axiom2.fibonacci_core import fib
    from axiom2.fibonacci_vortices import fib_vortices
    
    for size in [10, 20, 30, 50, 100]:
        _, metrics = profiler.profile_operation(f"fib_{size}", fib, size)
        print(f"  fib({size}): {metrics.execution_time:.6f}s")
    
    for size in test_sizes[:4]:
        _, metrics = profiler.profile_operation(f"fib_vortices_{size}", fib_vortices, size)
        print(f"  fib_vortices({size}): {metrics.execution_time:.4f}s")
    
    # Profile Axiom 3 operations
    print("Profiling Axiom 3...")
    from axiom3.spectral_core import spectral_vector
    
    cache = CoherenceCache(max_size=1000)
    
    for size in test_sizes[:5]:
        _, metrics = profiler.profile_operation(f"spectral_vector_{size}", spectral_vector, size)
        print(f"  spectral_vector({size}): {metrics.execution_time:.4f}s")
        
        a, b = size // 3, size // 2
        if a * b <= size:
            _, metrics = profiler.profile_operation(f"coherence_{size}", cache.get_coherence, a, b, size)
            print(f"  coherence({a},{b},{size}): {metrics.execution_time:.4f}s")
    
    # Profile Axiom 4 operations
    print("Profiling Axiom 4...")
    from axiom4.adaptive_observer import MultiScaleObserver
    
    for size in test_sizes[:4]:
        observer = MultiScaleObserver(size)
        _, metrics = profiler.profile_operation(f"multi_scale_observer_{size}", observer.observe, size // 3)
        print(f"  observer.observe({size//3}) for n={size}: {metrics.execution_time:.4f}s")
    
    # Profile Axiom 5 operations
    print("Profiling Axiom 5...")
    from axiom5.spectral_mirror import SpectralMirror
    
    for size in test_sizes[:4]:
        mirror = SpectralMirror(size)
        _, metrics = profiler.profile_operation(f"spectral_mirror_{size}", mirror.find_mirror_point, size // 4)
        print(f"  mirror.find_mirror_point({size//4}) for n={size}: {metrics.execution_time:.4f}s")
    
    print("\nGenerating Analysis...")
    
    # Generate reports
    complexity_report = profiler.generate_complexity_analysis()
    memory_report = profiler.generate_memory_analysis()
    
    print("\n" + complexity_report)
    print("\n" + memory_report)
    
    # Analyze cache performance
    cache_analysis = profiler.analyze_cache_performance(cache)
    if cache_analysis:
        print("\nCache Performance:")
        for key, value in cache_analysis.items():
            print(f"  {key}: {value}")
    
    return profiler

def main():
    """Run performance profiling"""
    print("UOR/Prime Axioms Factorizer - Performance Profiler")
    print("=" * 55)
    
    profiler = profile_axiom_performance()
    
    # Save detailed results
    results_path = os.path.join(os.path.dirname(__file__), "performance_results.txt")
    with open(results_path, 'w') as f:
        f.write("Performance Profiling Results\n")
        f.write("=" * 35 + "\n\n")
        
        f.write("Individual Operation Metrics:\n")
        f.write("-" * 30 + "\n")
        for metric in profiler.metrics:
            f.write(f"{metric.operation_name}:\n")
            f.write(f"  Input Size: {metric.input_size}\n")
            f.write(f"  Execution Time: {metric.execution_time:.6f}s\n")
            f.write(f"  Memory Used: {metric.memory_used / 1024:.2f} KB\n")
            f.write(f"  CPU Usage: {metric.cpu_usage:.1f}%\n\n")
        
        f.write(profiler.generate_complexity_analysis())
        f.write("\n\n")
        f.write(profiler.generate_memory_analysis())
    
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    main()
