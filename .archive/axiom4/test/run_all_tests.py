"""
Run all tests for Axiom 4: Observer Effect
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom4.test.test_adaptive_observer import run_all_tests as test_observer
from axiom4.test.test_spectral_navigation import run_all_tests as test_navigation
from axiom4.test.test_quantum_tools import run_all_tests as test_quantum
from axiom4.test.test_resonance_memory import run_all_tests as test_memory
from axiom4.test.test_observer_cache import run_all_tests as test_cache
from axiom4.test.test_accelerated_observer import run_all_tests as test_accelerated
from axiom4.test.test_performance_tuning import run_all_tests as test_performance
from axiom4.test.test_integration import run_all_tests as test_integration

def main():
    """Run all Axiom 4 tests"""
    print("=" * 60)
    print("AXIOM 4: OBSERVER EFFECT - COMPLETE TEST SUITE")
    print("=" * 60)
    print()
    
    # Test core components
    test_observer()
    print()
    
    test_navigation()
    print()
    
    test_quantum()
    print()
    
    test_memory()
    print()
    
    # Test acceleration components
    print("=" * 60)
    print("AXIOM 4: ACCELERATION TESTS")
    print("=" * 60)
    print()
    
    test_cache()
    print()
    
    test_accelerated()
    print()
    
    test_performance()
    print()
    
    test_integration()
    print()
    
    print("=" * 60)
    print("✅ ALL AXIOM 4 TESTS PASSED!")
    print("Observer Effect implementation verified.")
    print("Acceleration components verified.")
    print("=" * 60)

if __name__ == "__main__":
    main()
