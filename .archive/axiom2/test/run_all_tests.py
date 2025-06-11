"""
Comprehensive test suite for Axiom 2: Fibonacci Flow
Runs all tests and provides summary
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom2.test.test_fibonacci_core import run_all_tests as test_core
from axiom2.test.test_fibonacci_vortices import run_all_tests as test_vortices
from axiom2.test.test_fibonacci_entanglement import run_all_tests as test_entanglement

def run_axiom2_test_suite():
    """Run complete test suite for Axiom 2"""
    print("=" * 60)
    print("AXIOM 2: FIBONACCI FLOW - COMPLETE TEST SUITE")
    print("=" * 60)
    print()
    
    # Track overall success
    all_passed = True
    
    # Run each test module
    try:
        test_core()
        print()
    except Exception as e:
        print(f"❌ Fibonacci Core tests failed: {e}")
        all_passed = False
        print()
    
    try:
        test_vortices()
        print()
    except Exception as e:
        print(f"❌ Fibonacci Vortices tests failed: {e}")
        all_passed = False
        print()
    
    try:
        test_entanglement()
        print()
    except Exception as e:
        print(f"❌ Fibonacci Entanglement tests failed: {e}")
        all_passed = False
        print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL AXIOM 2 TESTS PASSED!")
        print("Fibonacci Flow implementation verified.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = run_axiom2_test_suite()
    sys.exit(0 if success else 1)
