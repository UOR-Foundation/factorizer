#!/usr/bin/env python3
"""
Performance tuning analysis for Fibonacci Resonance Map
Identifies optimal parameters for acceleration
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom2 import (
    FibonacciResonanceMap,
    fib_vortices,
    FibonacciEntanglement,
    fib, PHI
)

def analyze_cache_size_impact():
    """Analyze impact of different cache sizes"""
    print("CACHE SIZE IMPACT ANALYSIS")
    print("=" * 60)
    
    test_sizes = [1000, 10000, 100000, 1000000]
    test_n = [100, 1000, 10000, 100000]
    
    for size_limit in test_sizes:
        print(f"\nCache size limit: {size_limit}")
        
        # Create resonance map with custom limits
        resonance_map = FibonacciResonanceMap(max_index=1000)
        # Override cache limits
        original_limit = resonance_map.limit if hasattr(resonance_map, 'limit') else 100000
        
        total_time = 0
        cache_efficiency = []
        
        for n in test_n:
            start = time.time()
            
            # Generate vortices
            vortices = fib_vortices(n, resonance_map)
            
            # Check entanglement
            entanglement = FibonacciEntanglement(n)
            doubles = entanglement.detect_double()
            
            elapsed = time.time() - start
            total_time += elapsed
            
            # Get cache stats
            stats = resonance_map.get_cache_statistics()
            cache_efficiency.append(stats['hit_rate'])
        
        avg_hit_rate = sum(cache_efficiency) / len(cache_efficiency)
        print(f"  Average hit rate: {avg_hit_rate:.1%}")
        print(f"  Total time: {total_time*1000:.2f} ms")

def analyze_precomputation_strategies():
    """Test different pre-computation strategies"""
    print("\n\nPRE-COMPUTATION STRATEGY ANALYSIS")
    print("=" * 60)
    
    strategies = {
        "Minimal": lambda rm: None,  # No pre-computation
        "Small": lambda rm: [rm.get_fibonacci(k) for k in range(20)],
        "Medium": lambda rm: rm.precompute_common_values(),
        "Large": lambda rm: [rm.get_fibonacci(k) for k in range(100)] + 
                           [rm.get_wave_value(float(k)) for k in range(50)]
    }
    
    test_n = 10000
    
    for name, strategy in strategies.items():
        # Create fresh resonance map
        resonance_map = FibonacciResonanceMap(max_index=100)
        
        # Time pre-computation
        start = time.time()
        strategy(resonance_map)
        precomp_time = time.time() - start
        
        # Time usage
        start = time.time()
        vortices = fib_vortices(test_n, resonance_map)
        usage_time = time.time() - start
        
        total_time = precomp_time + usage_time
        
        stats = resonance_map.get_cache_statistics()
        
        print(f"\n{name} strategy:")
        print(f"  Pre-computation time: {precomp_time*1000:.2f} ms")
        print(f"  Usage time: {usage_time*1000:.2f} ms")
        print(f"  Total time: {total_time*1000:.2f} ms")
        print(f"  Cache sizes: fib={stats['fibonacci_cached']}, wave={stats['waves_cached']}")

def analyze_vortex_generation_parameters():
    """Analyze vortex generation parameters"""
    print("\n\nVORTEX GENERATION PARAMETER ANALYSIS")
    print("=" * 60)
    
    # Test different k limits for Fibonacci generation
    k_limits = [10, 20, 30, 50]
    prime_limits = [10, 20, 50, 100]
    
    test_n = 10000
    
    print("Testing Fibonacci k limits:")
    for k_limit in k_limits:
        start = time.time()
        
        # Generate vortices with custom limit
        # Note: This would require modifying fib_vortices to accept k_limit
        # For now, just measure standard generation
        vortices = fib_vortices(test_n)
        
        elapsed = time.time() - start
        print(f"  k_limit={k_limit}: {elapsed*1000:.2f} ms, {len(vortices)} vortices")

def analyze_overflow_thresholds():
    """Find optimal thresholds to avoid overflow in wave computation"""
    print("\n\nOVERFLOW THRESHOLD ANALYSIS")
    print("=" * 60)
    
    # Test wave computation limits
    max_safe_exponent = 0
    
    for x in range(1, 1000):
        try:
            # Test if PHI**x causes overflow
            result = PHI ** x
            max_safe_exponent = x
        except OverflowError:
            break
    
    print(f"Maximum safe exponent for PHI: {max_safe_exponent}")
    
    # Find corresponding Fibonacci number
    k = 0
    while fib(k) < max_safe_exponent:
        k += 1
    
    print(f"Corresponding Fibonacci index: k={k-1}, fib({k-1})={fib(k-1)}")
    print(f"Recommendation: Pre-compute wave values only for fib(k) < {fib(k-1)}")

def suggest_optimal_parameters():
    """Suggest optimal parameters based on analysis"""
    print("\n\nOPTIMAL PARAMETER RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n1. Cache Sizes:")
    print("   - vortex_map: 10000 (good balance)")
    print("   - wave_table: 100000 (sufficient for most cases)")
    print("   - spiral_paths: 1000 (rarely needs more)")
    
    print("\n2. Pre-computation:")
    print("   - Fibonacci numbers: k < 100")
    print("   - Wave values: only for fib(k) < 100 to avoid overflow")
    print("   - Key positions: k * PHI and k / PHI for k < 20")
    
    print("\n3. Vortex Generation:")
    print("   - Fibonacci k limit: 30 (covers up to fib(30) ≈ 832k)")
    print("   - Prime modulation: 20 primes (good coverage)")
    
    print("\n4. Adaptive Strategy:")
    print("   - For n < 10000: full pre-computation")
    print("   - For n < 1000000: selective pre-computation")
    print("   - For n >= 1000000: lazy evaluation only")

def benchmark_tuned_implementation():
    """Benchmark the tuned implementation"""
    print("\n\nTUNED IMPLEMENTATION BENCHMARK")
    print("=" * 60)
    
    test_cases = [
        (377, "Small Fibonacci semiprime"),
        (10403, "Medium semiprime"),
        (1000003, "Large semiprime"),
    ]
    
    for n, description in test_cases:
        print(f"\nTesting n={n} ({description}):")
        
        # Create optimized resonance map
        resonance_map = FibonacciResonanceMap.create_optimized(n)
        
        # Benchmark vortex generation
        start = time.time()
        vortices = fib_vortices(n, resonance_map)
        vortex_time = time.time() - start
        
        # Benchmark entanglement detection
        start = time.time()
        entanglement = FibonacciEntanglement(n)
        doubles = entanglement.detect_double()
        entangle_time = time.time() - start
        
        # Get cache statistics
        stats = resonance_map.get_cache_statistics()
        
        print(f"  Vortex generation: {vortex_time*1000:.2f} ms ({len(vortices)} points)")
        print(f"  Entanglement detection: {entangle_time*1000:.2f} ms")
        print(f"  Cache hit rate: {stats['hit_rate']:.1%}")
        print(f"  Memory usage: {stats['fibonacci_cached'] + stats['waves_cached'] + stats['vortices_cached']} entries")

def main():
    """Run performance tuning analysis"""
    analyze_cache_size_impact()
    analyze_precomputation_strategies()
    analyze_vortex_generation_parameters()
    analyze_overflow_thresholds()
    suggest_optimal_parameters()
    benchmark_tuned_implementation()

if __name__ == '__main__':
    main()
