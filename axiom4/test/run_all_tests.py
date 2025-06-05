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

def main():
    """Run all Axiom 4 tests"""
    print("=" * 60)
    print("AXIOM 4: OBSERVER EFFECT - COMPLETE TEST SUITE")
    print("=" * 60)
    print()
    
    # Test each component
    test_observer()
    print()
    
    test_navigation()
    print()
    
    test_quantum()
    print()
    
    test_memory()
    print()
    
    print("=" * 60)
    print("✅ ALL AXIOM 4 TESTS PASSED!")
    print("Observer Effect implementation verified.")
    print("=" * 60)

if __name__ == "__main__":
    main()
