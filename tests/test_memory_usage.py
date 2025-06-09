"""
Memory usage testing for Prime Resonance Field (RFH3)
"""

import gc
import os
import sys
import time
from typing import Any, Dict

from prime_resonance_field import RFH3, RFH3Config


def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback using resource module (Unix only)
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return 0.0


def profile_memory_usage():
    """Profile memory usage of RFH3 components"""
    print("Memory Usage Profiling for RFH3")
    print("=" * 50)

    # Initial memory
    gc.collect()
    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.1f} MB")

    # Test cases with increasing complexity
    test_cases = [
        (143, "8-bit semiprime"),
        (10403, "14-bit semiprime"),
        (1299071, "21-bit semiprime"),
        (1073217479, "30-bit semiprime"),
    ]

    rfh3 = RFH3()

    for n, description in test_cases:
        print(f"\nTesting {description}: {n}")

        # Memory before factorization
        gc.collect()
        before_memory = get_memory_usage()

        try:
            start_time = time.time()
            result = rfh3.factor(n, timeout=30.0)
            elapsed = time.time() - start_time

            # Memory after factorization
            gc.collect()
            after_memory = get_memory_usage()

            memory_increase = after_memory - before_memory
            memory_per_bit = memory_increase / n.bit_length()

            print(f"  Result: {result}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Memory before: {before_memory:.1f} MB")
            print(f"  Memory after: {after_memory:.1f} MB")
            print(f"  Memory increase: {memory_increase:.1f} MB")
            print(f"  Memory per bit: {memory_per_bit:.3f} MB/bit")

            # Check for memory leaks
            if memory_increase > 50:  # More than 50MB increase
                print("  WARNING: Large memory increase detected!")

        except Exception as e:
            print(f"  ERROR: {e}")
            gc.collect()
            after_memory = get_memory_usage()
            memory_increase = after_memory - before_memory
            print(f"  Memory increase during error: {memory_increase:.1f} MB")

    # Final memory check
    gc.collect()
    final_memory = get_memory_usage()
    total_increase = final_memory - initial_memory

    print("\nFinal Analysis:")
    print(f"  Total memory increase: {total_increase:.1f} MB")
    print(f"  Final memory usage: {final_memory:.1f} MB")

    if total_increase > 100:
        print("  WARNING: Significant memory usage detected")
        print("  Consider optimizing caching and state management")
    else:
        print("  Memory usage appears reasonable")

    return {
        "initial_memory": initial_memory,
        "final_memory": final_memory,
        "total_increase": total_increase,
        "test_results": test_cases,
    }


def profile_component_memory():
    """Profile memory usage of individual components"""
    print("\nComponent Memory Profiling")
    print("=" * 30)

    from prime_resonance_field.core import LazyResonanceIterator, MultiScaleResonance
    from prime_resonance_field.learning import ResonancePatternLearner

    gc.collect()

    components = []

    # Test MultiScaleResonance
    gc.collect()
    before = get_memory_usage()
    analyzer = MultiScaleResonance()
    # Compute some resonances to populate cache
    for i in range(2, 100):
        analyzer.compute_resonance(i, 143)
    gc.collect()
    after = get_memory_usage()
    components.append(("MultiScaleResonance", after - before))

    # Test LazyResonanceIterator
    gc.collect()
    before = get_memory_usage()
    iterator = LazyResonanceIterator(10403, analyzer)
    # Consume some values
    for i, x in enumerate(iterator):
        if i >= 100:
            break
    gc.collect()
    after = get_memory_usage()
    components.append(("LazyResonanceIterator", after - before))

    # Test ResonancePatternLearner
    gc.collect()
    before = get_memory_usage()
    learner = ResonancePatternLearner()
    # Add some training data
    for i in range(100):
        learner.record_success(143, 11, {"resonance": 0.5})
    gc.collect()
    after = get_memory_usage()
    components.append(("ResonancePatternLearner", after - before))

    # Report results
    for component, memory_used in components:
        print(f"  {component:25s}: {memory_used:6.1f} MB")

    total_component_memory = sum(mem for _, mem in components)
    print(f"  {'Total Components':25}: {total_component_memory:6.1f} MB")

    return components


def stress_test_memory():
    """Stress test memory usage with large numbers"""
    print("\nMemory Stress Test")
    print("=" * 30)

    # Test with many small factorizations
    rfh3 = RFH3()

    gc.collect()
    start_memory = get_memory_usage()

    # Factor many small numbers
    small_semiprimes = [143, 323, 1147, 10403] * 10  # 40 factorizations

    for i, n in enumerate(small_semiprimes):
        try:
            rfh3.factor(n, timeout=5.0)

            if i % 10 == 0:
                gc.collect()
                current_memory = get_memory_usage()
                increase = current_memory - start_memory
                print(f"  After {i+1:2d} factorizations: {increase:5.1f} MB increase")

                # Check for memory leaks
                if increase > i * 2:  # More than 2MB per factorization
                    print("    WARNING: Possible memory leak detected")

        except Exception:
            pass  # Continue stress test even on failures

    gc.collect()
    final_memory = get_memory_usage()
    total_increase = final_memory - start_memory
    avg_per_factorization = total_increase / len(small_semiprimes)

    print("\nStress Test Results:")
    print(f"  Factorizations: {len(small_semiprimes)}")
    print(f"  Total memory increase: {total_increase:.1f} MB")
    print(f"  Average per factorization: {avg_per_factorization:.3f} MB")

    if avg_per_factorization > 1.0:
        print("  WARNING: High memory usage per factorization")
    else:
        print("  Memory usage per factorization is reasonable")


def main():
    """Main memory profiling entry point"""
    try:
        print("Prime Resonance Field - Memory Usage Analysis")
        print("=" * 60)

        # Basic memory profiling
        profile_memory_usage()

        # Component-specific profiling
        profile_component_memory()

        # Stress testing
        stress_test_memory()

        print("\nMemory profiling completed successfully!")

    except Exception as e:
        print(f"Memory profiling failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
