"""
Comprehensive test suite for Axiom 1: Prime Ontology
Runs all tests and provides summary
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom1.test.test_prime_core import run_all_tests as test_core
from axiom1.test.test_prime_cascade import run_all_tests as test_cascade
from axiom1.test.test_prime_geodesic import run_all_tests as test_geodesic

def run_axiom1_test_suite():
    """Run complete test suite for Axiom 1"""
    print("=" * 60)
    print("AXIOM 1: PRIME ONTOLOGY - COMPLETE TEST SUITE")
    print("=" * 60)
    print()
    
    # Track overall success
    all_passed = True
    
    # Run each test module
    try:
        test_core()
        print()
    except Exception as e:
        print(f"❌ Prime Core tests failed: {e}")
        all_passed = False
        print()
    
    try:
        test_cascade()
        print()
    except Exception as e:
        print(f"❌ Prime Cascade tests failed: {e}")
        all_passed = False
        print()
    
    try:
        test_geodesic()
        print()
    except Exception as e:
        print(f"❌ Prime Geodesic tests failed: {e}")
        all_passed = False
        print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL AXIOM 1 TESTS PASSED!")
        print("Prime Ontology implementation verified.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = run_axiom1_test_suite()
    sys.exit(0 if success else 1)
