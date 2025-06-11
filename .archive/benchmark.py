"""
Benchmarking module for Prime Resonance Field (RFH3)
"""

import argparse
import json
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import RFH3, RFH3Config
except ImportError:
    # For direct execution and testing
    from rfh3 import RFH3, RFH3Config


class BenchmarkSuite:
    """Comprehensive benchmark suite for RFH3"""

    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.results = []

    def _load_test_cases(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """Load categorized test cases"""
        return {
            "small": [
                (143, 11, 13),
                (323, 17, 19),
                (1147, 31, 37),
                (10403, 101, 103),
            ],
            "balanced": [
                (282943, 523, 541),
                (1299071, 1129, 1151),
                (16777207, 4093, 4099),
                (1073217479, 32749, 32771),
            ],
            "large": [
                (2147766287, 46349, 46351),
                (68718952001, 262139, 262147),
                (274876858367, 524269, 524287),
                (1099509530623, 1048573, 1048576),
            ],
            "rsa_style": [
                (2533375639, 50321, 50363),
                (10005200147, 99991, 100019),
                (39916801007, 199933, 199967),
                (159832009999, 399979, 399989),
            ],
        }

    def run_quick_benchmark(self) -> Dict[str, Any]:
        """Run quick benchmark on small test cases"""
        print("Running Quick Benchmark...")
        print("=" * 50)

        results = {}
        rfh3 = RFH3()

        for category in ["small"]:
            category_results = []

            print(f"\nCategory: {category.upper()}")
            print("-" * 30)

            for n, p_true, q_true in self.test_cases[category]:
                try:
                    start = time.time()
                    p_found, q_found = rfh3.factor(n, timeout=10.0)
                    elapsed = time.time() - start

                    success = {p_found, q_found} == {p_true, q_true}

                    result = {
                        "n": n,
                        "expected": (p_true, q_true),
                        "found": (p_found, q_found),
                        "time": elapsed,
                        "success": success,
                        "bits": n.bit_length(),
                    }

                    category_results.append(result)

                    status = "✓" if success else "✗"
                    print(
                        f"  {status} {n:>12} ({n.bit_length():>2} bits): {elapsed:>7.3f}s"
                    )

                except Exception as e:
                    result = {
                        "n": n,
                        "expected": (p_true, q_true),
                        "found": None,
                        "time": 10.0,
                        "success": False,
                        "error": str(e),
                        "bits": n.bit_length(),
                    }
                    category_results.append(result)
                    print(f"  ✗ {n:>12} ({n.bit_length():>2} bits): ERROR - {e}")

            results[category] = category_results

        # Summary
        self._print_summary(results)
        return results

    def run_extensive_benchmark(self) -> Dict[str, Any]:
        """Run extensive benchmark on all test cases"""
        print("Running Extensive Benchmark...")
        print("=" * 50)

        results = {}
        rfh3 = RFH3()

        timeout_map = {
            "small": 10.0,
            "balanced": 30.0,
            "large": 60.0,
            "rsa_style": 120.0,
        }

        for category in self.test_cases:
            category_results = []
            timeout = timeout_map.get(category, 60.0)

            print(f"\nCategory: {category.upper()}")
            print("-" * 30)

            for n, p_true, q_true in self.test_cases[category]:
                try:
                    start = time.time()
                    p_found, q_found = rfh3.factor(n, timeout=timeout)
                    elapsed = time.time() - start

                    success = {p_found, q_found} == {p_true, q_true}

                    result = {
                        "n": n,
                        "expected": (p_true, q_true),
                        "found": (p_found, q_found),
                        "time": elapsed,
                        "success": success,
                        "bits": n.bit_length(),
                    }

                    category_results.append(result)

                    status = "✓" if success else "✗"
                    print(
                        f"  {status} {n:>15} ({n.bit_length():>2} bits): {elapsed:>7.3f}s"
                    )

                except Exception as e:
                    result = {
                        "n": n,
                        "expected": (p_true, q_true),
                        "found": None,
                        "time": timeout,
                        "success": False,
                        "error": str(e),
                        "bits": n.bit_length(),
                    }
                    category_results.append(result)
                    print(f"  ✗ {n:>15} ({n.bit_length():>2} bits): ERROR - {e}")

            results[category] = category_results

        # Summary
        self._print_summary(results)
        return results

    def compare_algorithms(self) -> Dict[str, Any]:
        """Compare RFH3 with baseline algorithms"""
        print("Algorithm Comparison Benchmark...")
        print("=" * 50)

        # Use small test cases for comparison
        test_cases = self.test_cases["small"]

        algorithms = {
            "RFH3 (Full)": lambda n: self._run_rfh3_full(n),
            "RFH3 (No Learning)": lambda n: self._run_rfh3_no_learning(n),
            "Trial Division": lambda n: self._run_trial_division(n),
            "Pollard Rho": lambda n: self._run_pollard_rho(n),
        }

        results = {}

        for alg_name, alg_func in algorithms.items():
            print(f"\nTesting {alg_name}:")
            print("-" * 30)

            alg_results = []

            for n, p_true, q_true in test_cases:
                try:
                    start = time.time()
                    result = alg_func(n)
                    elapsed = time.time() - start

                    if result:
                        p_found, q_found = result
                        success = {p_found, q_found} == {p_true, q_true}
                    else:
                        success = False
                        p_found = q_found = None

                    alg_results.append(
                        {
                            "n": n,
                            "time": elapsed,
                            "success": success,
                            "found": (p_found, q_found) if result else None,
                        }
                    )

                    status = "✓" if success else "✗"
                    print(f"  {status} {n:>8}: {elapsed:>7.3f}s")

                except Exception as e:
                    alg_results.append(
                        {"n": n, "time": 10.0, "success": False, "error": str(e)}
                    )
                    print(f"  ✗ {n:>8}: ERROR")

            results[alg_name] = alg_results

        self._print_comparison_summary(results)
        return results

    def _run_rfh3_full(self, n: int) -> Optional[Tuple[int, int]]:
        """Run RFH3 with all features"""
        rfh3 = RFH3()
        try:
            return rfh3.factor(n, timeout=10.0)
        except Exception:
            return None

    def _run_rfh3_no_learning(self, n: int) -> Optional[Tuple[int, int]]:
        """Run RFH3 without learning"""
        config = RFH3Config()
        config.learning_enabled = False
        rfh3 = RFH3(config)
        try:
            return rfh3.factor(n, timeout=10.0)
        except Exception:
            return None

    def _run_trial_division(self, n: int) -> Optional[Tuple[int, int]]:
        """Simple trial division"""
        import math

        sqrt_n = int(math.sqrt(n))
        for i in range(2, min(sqrt_n + 1, 10000)):
            if n % i == 0:
                return (i, n // i)
        return None

    def _run_pollard_rho(self, n: int) -> Optional[Tuple[int, int]]:
        """Basic Pollard's Rho implementation"""
        import math

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def pollard_rho_single(n):
            if n % 2 == 0:
                return 2

            x = 2
            y = 2
            d = 1

            def f(x):
                return (x * x + 1) % n

            while d == 1:
                x = f(x)
                y = f(f(y))
                d = gcd(abs(x - y), n)

                if d == n:
                    return None

            return d

        factor = pollard_rho_single(n)
        if factor and factor != n:
            return (factor, n // factor)
        return None

    def _print_summary(self, results: Dict[str, List[Dict]]):
        """Print benchmark summary"""
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)

        total_cases = 0
        total_successes = 0
        total_time = 0

        for category, category_results in results.items():
            successes = sum(1 for r in category_results if r["success"])
            total_cases += len(category_results)
            total_successes += successes

            times = [r["time"] for r in category_results if r["success"]]
            if times:
                avg_time = statistics.mean(times)
                total_time += sum(times)
            else:
                avg_time = 0

            success_rate = successes / len(category_results) * 100

            print(
                f"{category.upper():>12}: {successes:>2}/{len(category_results):>2} "
                f"({success_rate:>5.1f}%) avg: {avg_time:>7.3f}s"
            )

        overall_rate = total_successes / total_cases * 100 if total_cases > 0 else 0
        avg_overall = total_time / total_successes if total_successes > 0 else 0

        print("-" * 50)
        print(
            f"{'OVERALL':>12}: {total_successes:>2}/{total_cases:>2} "
            f"({overall_rate:>5.1f}%) avg: {avg_overall:>7.3f}s"
        )

    def _print_comparison_summary(self, results: Dict[str, List[Dict]]):
        """Print algorithm comparison summary"""
        print("\n" + "=" * 60)
        print("ALGORITHM COMPARISON SUMMARY")
        print("=" * 60)

        for alg_name, alg_results in results.items():
            successes = sum(1 for r in alg_results if r["success"])
            total = len(alg_results)

            times = [r["time"] for r in alg_results if r["success"]]
            avg_time = statistics.mean(times) if times else 0

            success_rate = successes / total * 100

            print(
                f"{alg_name:>20}: {successes:>2}/{total:>2} "
                f"({success_rate:>5.1f}%) avg: {avg_time:>7.3f}s"
            )

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save benchmark results to file"""
        # Add metadata
        output = {
            "benchmark_version": "1.0",
            "timestamp": time.time(),
            "results": results,
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filename}")


def run_benchmark(config: Dict[str, Any]):
    """Main benchmark runner"""
    suite = BenchmarkSuite()

    if config.get("quick"):
        results = suite.run_quick_benchmark()
    elif config.get("extensive"):
        results = suite.run_extensive_benchmark()
    elif config.get("compare_algorithms"):
        results = suite.compare_algorithms()
    else:
        # Default: run quick benchmark
        results = suite.run_quick_benchmark()

    if config.get("save_results"):
        suite.save_results(results, config["save_results"])

    return results


def main():
    """CLI entry point for benchmark module"""
    parser = argparse.ArgumentParser(description="RFH3 Benchmark Suite")

    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument(
        "--extensive", action="store_true", help="Run extensive benchmark"
    )
    parser.add_argument("--compare", action="store_true", help="Compare algorithms")
    parser.add_argument("--save-results", metavar="FILE", help="Save results to file")
    parser.add_argument("--profile", action="store_true", help="Profile performance")

    args = parser.parse_args()

    config = {
        "quick": args.quick,
        "extensive": args.extensive,
        "compare_algorithms": args.compare,
        "save_results": args.save_results,
    }

    if args.profile:
        import cProfile
        import pstats

        pr = cProfile.Profile()
        pr.enable()

        run_benchmark(config)

        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats("cumulative")
        stats.print_stats(20)
    else:
        run_benchmark(config)


if __name__ == "__main__":
    main()
