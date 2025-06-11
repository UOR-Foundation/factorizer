"""
Command-line interface for Prime Resonance Field (RFH3)
"""

import argparse
import sys
import time
from typing import Optional

try:
    from . import RFH3, RFH3Config, __version__
except ImportError:
    # For direct execution and testing
    from rfh3 import RFH3, RFH3Config
    __version__ = "3.0.0"


def factor_command(args):
    """Handle the factor command"""
    config = RFH3Config()

    # Apply command-line overrides
    if args.timeout:
        config.max_iterations = int(args.timeout * 1000000)  # Rough estimate

    if args.verbose:
        import logging

        config.log_level = logging.INFO

    if args.no_learning:
        config.learning_enabled = False

    if args.no_hierarchical:
        config.hierarchical_search = False

    # Create RFH3 instance
    rfh3 = RFH3(config)

    print(f"Prime Resonance Field v{__version__}")
    print("Publisher: UOR Foundation (https://uor.foundation)")
    print()

    total_time = 0
    total_successes = 0

    for number in args.numbers:
        try:
            n = int(number)
            if n < 4:
                print(f"Error: {n} is too small (must be >= 4)")
                continue

            print(f"Factoring {n} ({n.bit_length()} bits)...")

            start_time = time.time()
            p, q = rfh3.factor(n, timeout=args.timeout)
            elapsed = time.time() - start_time
            total_time += elapsed
            total_successes += 1

            print(f"  {n} = {p} × {q}")
            print(f"  Time: {elapsed:.3f}s")

            if args.verify:
                if p * q != n:
                    print("  ❌ Verification failed!")
                    sys.exit(1)
                else:
                    print("  ✓ Verified")

            print()

        except ValueError as e:
            print(f"Error factoring {number}: {e}")
            if args.strict:
                sys.exit(1)
        except Exception as e:
            print(f"Unexpected error factoring {number}: {e}")
            if args.strict:
                sys.exit(1)

    # Summary
    if total_successes > 0:
        avg_time = total_time / total_successes
        print(f"Summary: {total_successes} successful factorizations")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time: {avg_time:.3f}s")

        if args.verbose:
            rfh3.print_stats()


def benchmark_command(args):
    """Handle the benchmark command"""
    try:
        from .benchmark import run_benchmark
    except ImportError:
        try:
            from benchmark import run_benchmark
        except ImportError:
            print("Error: benchmark module not found")
            sys.exit(1)

    config = {
        "quick": args.quick,
        "extensive": args.extensive,
        "save_results": args.save_results,
        "compare_algorithms": args.compare,
    }

    run_benchmark(config)


def run_tests_command(args):
    """Handle the test command"""
    import subprocess

    cmd = ["python", "-m", "pytest"]

    if args.verbose:
        cmd.append("-v")

    if args.coverage:
        cmd.extend(["--cov=prime_resonance_field", "--cov-report=term"])

    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])

    cmd.append("tests/")

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("Error: pytest not found. Install with: pip install pytest")
        sys.exit(1)


def info_command(args):
    """Handle the info command"""
    print("Prime Resonance Field (RFH3)")
    print(f"Version: {__version__}")
    print("Publisher: UOR Foundation")
    print("Homepage: https://uor.foundation")
    print("Repository: https://github.com/UOR-Foundation/factorizer")
    print()
    print("RFH3 is an adaptive resonance field architecture for prime factorization")
    print(
        "that achieves 85.2% success rate on hard semiprimes with learning capabilities."
    )
    print()
    print("Available commands:")
    print("  factor     Factor one or more numbers")
    print("  benchmark  Run performance benchmarks")
    print("  test       Run test suite")
    print("  info       Show this information")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="rfh3",
        description="Prime Resonance Field (RFH3) - Adaptive Integer Factorization",
        epilog="Published by UOR Foundation (https://uor.foundation)",
    )

    parser.add_argument("--version", action="version", version=f"RFH3 {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Factor command
    factor_parser = subparsers.add_parser("factor", help="Factor numbers")
    factor_parser.add_argument("numbers", nargs="+", help="Numbers to factor")
    factor_parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds (default: 60)",
    )
    factor_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    factor_parser.add_argument(
        "--verify", action="store_true", help="Verify factorization results"
    )
    factor_parser.add_argument(
        "--strict", action="store_true", help="Exit on first error"
    )
    factor_parser.add_argument(
        "--no-learning", action="store_true", help="Disable learning features"
    )
    factor_parser.add_argument(
        "--no-hierarchical", action="store_true", help="Disable hierarchical search"
    )
    factor_parser.set_defaults(func=factor_command)

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark"
    )
    benchmark_parser.add_argument(
        "--extensive", action="store_true", help="Run extensive benchmark"
    )
    benchmark_parser.add_argument(
        "--save-results", metavar="FILE", help="Save results to file"
    )
    benchmark_parser.add_argument(
        "--compare", action="store_true", help="Compare with other algorithms"
    )
    benchmark_parser.set_defaults(func=benchmark_command)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose test output"
    )
    test_parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage"
    )
    test_parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    test_parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    test_parser.set_defaults(func=run_tests_command)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show package information")
    info_parser.set_defaults(func=info_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        # Default behavior - show info
        info_command(args)
        return

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
