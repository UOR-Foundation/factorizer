"""
Run all tests for Axiom 5: Self-Reference/Reflection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom5.test.test_meta_observer import run_all_tests as test_meta_observer
from axiom5.test.test_spectral_mirror import run_all_tests as test_spectral_mirror
from axiom5.test.test_axiom_synthesis import run_all_tests as test_axiom_synthesis
from axiom5.test.test_recursive_coherence import run_all_tests as test_recursive_coherence
from axiom5.test.test_failure_analysis import run_all_tests as test_failure_analysis

def main():
    """Run all Axiom 5 tests"""
    print("=" * 60)
    print("AXIOM 5: SELF-REFERENCE/REFLECTION - COMPLETE TEST SUITE")
    print("=" * 60)
    print()
    
    # Test each component
    test_meta_observer()
    print()
    
    test_spectral_mirror()
    print()
    
    test_axiom_synthesis()
    print()
    
    test_recursive_coherence()
    print()
    
    test_failure_analysis()
    print()
    
    print("=" * 60)
    print("✅ ALL AXIOM 5 TESTS PASSED!")
    print("Self-Reference/Reflection implementation verified.")
    print("=" * 60)

if __name__ == "__main__":
    main()
