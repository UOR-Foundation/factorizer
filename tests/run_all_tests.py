"""
Test runner for the complete Prime Resonance Field test suite
"""

import os
import sys
import time
import unittest


def run_test_suite():
    """Run the complete test suite"""
    print("=" * 80)
    print("PRIME RESONANCE FIELD - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()

    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py", top_level_dir=start_dir)

    # Configure test runner
    runner = unittest.TextTestRunner(
        verbosity=2, stream=sys.stdout, buffer=True, failfast=False
    )

    print(f"Starting test discovery from: {start_dir}")
    print("Test pattern: test_*.py")
    print()

    # Run tests
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time

    # Print summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Total time: {total_time:.2f} seconds")
    print()

    if result.failures:
        print("FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        print()

    if result.errors:
        print("ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
        print()

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(
        result.testsRun, 1
    )
    print(f"Success rate: {success_rate * 100:.1f}%")

    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ Some tests failed. Check the output above for details.")

    print("=" * 80)

    return result.wasSuccessful()


def run_component_tests(component_name):
    """Run tests for a specific component"""
    print(f"Running {component_name} tests...")

    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), component_name)

    if not os.path.exists(start_dir):
        print(f"Component '{component_name}' not found!")
        return False

    suite = loader.discover(start_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_specific_test(test_path):
    """Run a specific test file"""
    print(f"Running specific test: {test_path}")

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_path)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--component":
            if len(sys.argv) > 2:
                success = run_component_tests(sys.argv[2])
            else:
                print("Usage: python run_all_tests.py --component <component_name>")
                print("Available components: core, learning, algorithms, integration")
                sys.exit(1)
        elif sys.argv[1] == "--test":
            if len(sys.argv) > 2:
                success = run_specific_test(sys.argv[2])
            else:
                print(
                    "Usage: python run_all_tests.py --test <test_module.TestClass.test_method>"
                )
                sys.exit(1)
        elif sys.argv[1] == "--help":
            print("Prime Resonance Field Test Runner")
            print()
            print("Usage:")
            print("  python run_all_tests.py                    # Run all tests")
            print(
                "  python run_all_tests.py --component core   # Run core component tests"
            )
            print("  python run_all_tests.py --test test_name   # Run specific test")
            print("  python run_all_tests.py --help            # Show this help")
            print()
            print("Available components:")
            print("  - core: MultiScaleResonance, LazyResonanceIterator, etc.")
            print("  - learning: ResonancePatternLearner, ZonePredictor")
            print("  - algorithms: BalancedSemiprimeSearch, etc.")
            print("  - integration: End-to-end system tests")
            sys.exit(0)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        # Run all tests
        success = run_test_suite()

    sys.exit(0 if success else 1)
