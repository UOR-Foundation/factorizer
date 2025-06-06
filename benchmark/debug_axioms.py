#!/usr/bin/env python3
"""
Debug script to test individual axioms and identify issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_axiom1(n=15):
    """Debug Axiom 1 with a simple test case"""
    print(f"=== Debugging Axiom 1 with n={n} ===")
    
    try:
        from axiom1.prime_cascade import PrimeCascade
        from axiom1.prime_geodesic import PrimeGeodesic
        
        print("Imports successful")
        
        # Test PrimeCascade
        print("Testing PrimeCascade...")
        cascade = PrimeCascade(n)
        twin_primes = cascade.find_twin_primes()
        sophie_chains = cascade.find_sophie_germain_chains()
        print(f"  Twin primes found: {len(twin_primes)}")
        print(f"  Sophie Germain chains: {len(sophie_chains)}")
        
        # Test PrimeGeodesic
        print("Testing PrimeGeodesic...")
        geodesic = PrimeGeodesic(n)
        coords = geodesic.prime_coordinates()
        print(f"  Prime coordinates: {coords}")
        
        # Test factor detection
        print("Testing factor detection...")
        root = int(n**0.5) + 1
        print(f"  Checking candidates up to {root}")
        
        factors_found = []
        for candidate in range(2, root):
            if n % candidate == 0:
                pull = geodesic.geodesic_pull(candidate)
                factors_found.append((candidate, pull))
                print(f"    Factor {candidate}: pull = {pull}")
        
        if factors_found:
            best_factor = max(factors_found, key=lambda x: x[1])
            print(f"  Best factor: {best_factor[0]} with pull {best_factor[1]}")
        else:
            print("  No factors found!")
        
    except Exception as e:
        print(f"ERROR in Axiom 1: {e}")
        import traceback
        traceback.print_exc()

def debug_axiom2(n=15):
    """Debug Axiom 2 with a simple test case"""
    print(f"\n=== Debugging Axiom 2 with n={n} ===")
    
    try:
        from axiom2.fibonacci_vortices import fib_vortices
        from axiom2.fibonacci_entanglement import FibonacciEntanglement
        
        print("Imports successful")
        
        # Test fibonacci vortices
        print("Testing fibonacci vortices...")
        vortices = fib_vortices(20)  # Get enough fibonacci numbers
        print(f"  Got {len(vortices)} vortices")
        
        # Test fibonacci entanglement
        print("Testing fibonacci entanglement...")
        entangler = FibonacciEntanglement(n)
        phi_resonance = entangler.phi_resonance()
        factors = entangler.entangled_factors()
        
        print(f"  Phi resonance: {phi_resonance}")
        print(f"  Entangled factors: {factors}")
        
    except Exception as e:
        print(f"ERROR in Axiom 2: {e}")
        import traceback
        traceback.print_exc()

def debug_axiom3(n=15):
    """Debug Axiom 3 with a simple test case"""
    print(f"\n=== Debugging Axiom 3 with n={n} ===")
    
    try:
        from axiom3.interference import prime_fib_interference, interference_extrema
        
        print("Imports successful")
        
        # Test interference
        print("Testing prime-fibonacci interference...")
        interference = prime_fib_interference(n)
        extrema = interference_extrema(interference)
        
        print(f"  Interference pattern length: {len(interference)}")
        print(f"  Extrema found: {len(extrema)}")
        
        # Simple factor detection
        factors_found = []
        for candidate in range(2, int(n**0.5) + 1):
            if n % candidate == 0:
                factors_found.append(candidate)
        
        print(f"  Actual factors: {factors_found}")
        
        # Check if extrema correspond to factors
        factor_matches = []
        for ext in extrema:
            if ext < len(factors_found) and factors_found:
                factor_matches.append(ext)
        
        print(f"  Factor matches from extrema: {factor_matches}")
        
    except Exception as e:
        print(f"ERROR in Axiom 3: {e}")
        import traceback
        traceback.print_exc()

def debug_axiom4(n=15):
    """Debug Axiom 4 with a simple test case"""
    print(f"\n=== Debugging Axiom 4 with n={n} ===")
    
    try:
        from axiom4.adaptive_observer import MultiScaleObserver
        
        print("Imports successful")
        
        # Test adaptive observer
        print("Testing adaptive observer...")
        observer = MultiScaleObserver(n)
        observation = observer.observe(n // 3)
        
        print(f"  Observation result: {observation}")
        
    except Exception as e:
        print(f"ERROR in Axiom 4: {e}")
        import traceback
        traceback.print_exc()

def debug_axiom5(n=15):
    """Debug Axiom 5 with a simple test case"""
    print(f"\n=== Debugging Axiom 5 with n={n} ===")
    
    try:
        from axiom5.spectral_mirror import SpectralMirror
        
        print("Imports successful")
        
        # Test spectral mirror
        print("Testing spectral mirror...")
        mirror = SpectralMirror(n)
        mirror_point = mirror.find_mirror_point(n // 4)
        
        print(f"  Mirror point: {mirror_point}")
        
    except Exception as e:
        print(f"ERROR in Axiom 5: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Debugging Individual Axioms")
    print("=" * 40)
    
    test_n = 15  # Simple test case: 15 = 3 × 5
    
    debug_axiom1(test_n)
    debug_axiom2(test_n)
    debug_axiom3(test_n)
    debug_axiom4(test_n)
    debug_axiom5(test_n)

if __name__ == "__main__":
    main()
