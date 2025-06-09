"""
Run all tests for Axiom 3: Duality Principle
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom3.test.test_spectral_core import run_all_tests as test_spectral
from axiom3.test.test_coherence import run_all_tests as test_coherence
from axiom3.test.test_fold_topology import run_all_tests as test_fold
from axiom3.test.test_interference import run_all_tests as test_interference
from axiom3.test.test_spectral_signature_cache import run_all_tests as test_cache

def main():
    """Run all Axiom 3 tests"""
    print("=" * 60)
    print("AXIOM 3: DUALITY PRINCIPLE - COMPLETE TEST SUITE")
    print("=" * 60)
    print()
    
    # Run each test module
    test_spectral()
    print()
    
    test_coherence()
    print()
    
    test_fold()
    print()
    
    test_interference()
    print()
    
    test_cache()
    print()
    
    print("=" * 60)
    print("✅ ALL AXIOM 3 TESTS PASSED!")
    print("Duality Principle implementation verified.")
    print("=" * 60)

if __name__ == "__main__":
    main()
