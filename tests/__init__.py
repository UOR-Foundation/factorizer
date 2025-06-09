"""
Test suite for the Prime Resonance Field architecture
"""

import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Test utilities
def run_all_tests():
    """Run all tests in the test suite"""
    import unittest

    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


# Common test fixtures
class TestFixtures:
    """Common test data and fixtures"""

    SMALL_SEMIPRIMES = [
        (143, 11, 13),
        (323, 17, 19),
        (1147, 31, 37),
        (10403, 101, 103),
    ]

    BALANCED_SEMIPRIMES = [
        (282943, 523, 541),
        (1299071, 1117, 1163),
        (16777207, 4099, 4093),
    ]

    LARGE_SEMIPRIMES = [
        (1073217479, 32749, 32771),
        (2147766287, 46337, 46351),
    ]

    RSA_STYLE_SEMIPRIMES = [
        (2533375639, 47947, 52837),
        (10005200147, 100003, 100049),
    ]


__all__ = ["run_all_tests", "TestFixtures"]
