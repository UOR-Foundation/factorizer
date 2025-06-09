"""
Stress testing for Prime Resonance Field (RFH3)
"""

import multiprocessing
import os
import sys
import threading
import time
import traceback
from typing import Dict, List, Tuple

from rfh3 import RFH3, RFH3Config


class StressTestSuite:
    """Comprehensive stress testing for RFH3"""

    def __init__(self):
        self.results = []
        self.config = RFH3Config()

    def test_memory_usage(self):
        """Test memory usage under load"""
        print("Memory Usage Stress Test")
        print("=" * 40)

        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            print("psutil not available - skipping memory test")
            return

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory: {initial_memory:.1f} MB")

        rfh3 = RFH3(self.config)

        # Test with progressively larger numbers
        test_numbers = [143, 323, 1147, 10403, 282943, 1299071, 16777207]

        for n in test_numbers:
            try:
                start_memory = process.memory_info().rss / 1024 / 1024

                # Factor the number
                start_time = time.time()
                rfh3.factor(n, timeout=30.0)
                elapsed = time.time() - start_time

                end_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = end_memory - start_memory

                print(
                    f"  {n:>10} ({n.bit_length():>2} bits): "
                    f"{elapsed:>6.3f}s, memory: +{memory_increase:>5.1f} MB"
                )

                # Check for memory leaks
                if memory_increase > 100:  # More than 100MB increase
                    print(
                        f"    WARNING: Large memory increase: {memory_increase:.1f} MB"
                    )

            except Exception as e:
                print(f"  {n:>10}: ERROR - {e}")

        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        print(f"\nTotal memory increase: {total_increase:.1f} MB")

        if total_increase > 500:  # More than 500MB total
            print("WARNING: Significant memory usage detected")
        else:
            print("Memory usage appears reasonable")

    def test_concurrent_factorization(self):
        """Test concurrent factorization"""
        print("\nConcurrent Factorization Test")
        print("=" * 40)

        def factor_worker(numbers: List[int], results: List[Dict], worker_id: int):
            """Worker function for concurrent testing"""
            rfh3 = RFH3(self.config)
            worker_results = []

            for n in numbers:
                try:
                    start_time = time.time()
                    p, q = rfh3.factor(n, timeout=15.0)
                    elapsed = time.time() - start_time

                    worker_results.append(
                        {
                            "worker_id": worker_id,
                            "n": n,
                            "result": (p, q),
                            "time": elapsed,
                            "success": True,
                        }
                    )

                except Exception as e:
                    worker_results.append(
                        {
                            "worker_id": worker_id,
                            "n": n,
                            "error": str(e),
                            "success": False,
                        }
                    )

            results.extend(worker_results)

        # Test data - distribute across workers
        test_numbers = [143, 323, 1147, 10403, 282943, 1299071]
        num_workers = min(4, len(test_numbers))

        # Distribute work
        work_chunks = []
        chunk_size = len(test_numbers) // num_workers
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = (
                start_idx + chunk_size if i < num_workers - 1 else len(test_numbers)
            )
            work_chunks.append(test_numbers[start_idx:end_idx])

        # Run concurrent workers
        threads = []
        results = []

        start_time = time.time()

        for i, chunk in enumerate(work_chunks):
            thread = threading.Thread(target=factor_worker, args=(chunk, results, i))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Analyze results
        successes = sum(1 for r in results if r.get("success", False))
        total_attempts = len(results)

        print(f"Workers: {num_workers}")
        print(f"Total time: {total_time:.3f}s")
        print(
            f"Success rate: {successes}/{total_attempts} ({successes/total_attempts*100:.1f}%)"
        )

        # Check for race conditions or inconsistencies
        number_results = {}
        for result in results:
            if result.get("success"):
                n = result["n"]
                if n in number_results:
                    # Same number factored by multiple workers
                    if number_results[n] != result["result"]:
                        print(f"WARNING: Inconsistent results for {n}")
                else:
                    number_results[n] = result["result"]

        print("Concurrent test completed successfully")

    def test_large_number_handling(self):
        """Test handling of very large numbers"""
        print("\nLarge Number Handling Test")
        print("=" * 40)

        # Test progressively larger semiprimes
        large_semiprimes = [
            (2147766287, 46349, 46351),  # ~31 bits
            (68718952001, 262139, 262147),  # ~36 bits
            (274876858367, 524269, 524287),  # ~38 bits
        ]

        rfh3 = RFH3(self.config)

        for n, p_expected, q_expected in large_semiprimes:
            print(f"Testing {n} ({n.bit_length()} bits)...")

            try:
                start_time = time.time()
                p_found, q_found = rfh3.factor(n, timeout=120.0)  # Longer timeout
                elapsed = time.time() - start_time

                success = {p_found, q_found} == {p_expected, q_expected}
                status = "✓" if success else "✗"

                print(f"  {status} Result: {p_found} × {q_found}")
                print(f"  Time: {elapsed:.3f}s")

                if not success:
                    print(f"  Expected: {p_expected} × {q_expected}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")

    def test_algorithm_robustness(self):
        """Test algorithm robustness with edge cases"""
        print("\nAlgorithm Robustness Test")
        print("=" * 40)

        rfh3 = RFH3(self.config)

        # Test edge cases
        edge_cases = [
            (4, "smallest composite"),
            (9, "perfect square"),
            (15, "product of small primes"),
            (21, "product of small primes"),
            (35, "product of small primes"),
            (77, "unbalanced semiprime"),
            (91, "unbalanced semiprime"),
        ]

        for n, description in edge_cases:
            try:
                start_time = time.time()
                p, q = rfh3.factor(n, timeout=10.0)
                elapsed = time.time() - start_time

                # Verify factorization
                if p * q == n:
                    print(f"  ✓ {n:>3} ({description}): {p} × {q} in {elapsed:.3f}s")
                else:
                    print(f"  ✗ {n:>3} ({description}): Invalid result {p} × {q}")

            except Exception as e:
                print(f"  ✗ {n:>3} ({description}): ERROR - {e}")

    def test_timeout_handling(self):
        """Test timeout handling"""
        print("\nTimeout Handling Test")
        print("=" * 40)

        rfh3 = RFH3(self.config)

        # Test with very short timeouts
        test_number = 1147  # Should be solvable quickly

        timeouts = [0.001, 0.01, 0.1, 1.0, 10.0]

        for timeout in timeouts:
            try:
                start_time = time.time()
                rfh3.factor(test_number, timeout=timeout)
                elapsed = time.time() - start_time

                print(f"  Timeout {timeout:>5.3f}s: Success in {elapsed:.3f}s")

                # Check that timeout was respected (with some tolerance)
                if elapsed > timeout * 2:
                    print(f"    WARNING: Exceeded timeout by {elapsed - timeout:.3f}s")

            except Exception as e:
                elapsed = time.time() - start_time
                print(
                    f"  Timeout {timeout:>5.3f}s: Failed in {elapsed:.3f}s - {type(e).__name__}"
                )

    def run_all_tests(self):
        """Run all stress tests"""
        print("Prime Resonance Field (RFH3) - Stress Test Suite")
        print("=" * 60)

        tests = [
            self.test_memory_usage,
            self.test_concurrent_factorization,
            self.test_large_number_handling,
            self.test_algorithm_robustness,
            self.test_timeout_handling,
        ]

        for i, test in enumerate(tests, 1):
            try:
                print(f"\n[{i}/{len(tests)}] Running {test.__name__}...")
                test()
            except Exception as e:
                print(f"Test {test.__name__} failed: {e}")
                traceback.print_exc()

        print("\n" + "=" * 60)
        print("Stress testing completed!")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="RFH3 Stress Testing")
    parser.add_argument("--memory", action="store_true", help="Test memory usage only")
    parser.add_argument(
        "--concurrent", action="store_true", help="Test concurrency only"
    )
    parser.add_argument("--large", action="store_true", help="Test large numbers only")
    parser.add_argument(
        "--robustness", action="store_true", help="Test robustness only"
    )
    parser.add_argument(
        "--timeout", action="store_true", help="Test timeout handling only"
    )

    args = parser.parse_args()

    suite = StressTestSuite()

    if args.memory:
        suite.test_memory_usage()
    elif args.concurrent:
        suite.test_concurrent_factorization()
    elif args.large:
        suite.test_large_number_handling()
    elif args.robustness:
        suite.test_algorithm_robustness()
    elif args.timeout:
        suite.test_timeout_handling()
    else:
        suite.run_all_tests()


if __name__ == "__main__":
    main()
