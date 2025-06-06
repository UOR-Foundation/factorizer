#!/usr/bin/env python3
"""
Debug specific failing axioms to understand why they're not working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_axiom1_failure():
    """Debug why Axiom 1 is failing"""
    print("=== Debugging Axiom 1 Failure ===")
    
    try:
        from axiom1.prime_cascade import PrimeCascade
        from axiom1.prime_geodesic import PrimeGeodesic
        from axiom1.prime_core import is_prime
        
        n = 15  # 3 × 5
        factors = [3, 5]
        
        print(f"Testing n={n}, factors={factors}")
        
        # Test prime cascade
        cascade = PrimeCascade(n)
        primes_up_to_sqrt = [p for p in range(2, int(n**0.5)+1) if is_prime(p)]
        print(f"Primes up to sqrt({n}): {primes_up_to_sqrt}")
        
        cascade_results = []
        for p in primes_up_to_sqrt[:5]:
            result = cascade.cascade(p)
            cascade_results.extend(result)
            print(f"  Cascade from {p}: {result}")
        
        print(f"All cascade results: {cascade_results}")
        
        # Test prime geodesic
        geodesic = PrimeGeodesic(n)
        coords = geodesic.prime_coordinates()
        print(f"Prime coordinates: {coords}")
        
        # Test factor detection
        print("Testing factor detection...")
        root = int(n**0.5) + 1
        for candidate in range(2, root):
            if n % candidate == 0:
                pull = geodesic.geodesic_pull(candidate)
                print(f"  Factor {candidate}: pull = {pull}")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def debug_axiom4_failure():
    """Debug why Axiom 4 is failing"""
    print("\n=== Debugging Axiom 4 Failure ===")
    
    try:
        from axiom4.adaptive_observer import MultiScaleObserver
        from axiom4.quantum_tools import QuantumTunnel
        
        n = 15  # 3 × 5
        factors = [3, 5]
        
        print(f"Testing n={n}, factors={factors}")
        
        # Test multi-scale observer
        observer = MultiScaleObserver(n)
        print("Observer created")
        
        # Test quantum tunneling
        tunnel = QuantumTunnel(n)
        print("Tunnel created")
        
        # Generate candidates
        root = int(n**0.5) + 1
        candidates = list(range(2, root))
        print(f"Candidates: {candidates}")
        
        # Test quantum tunneling
        tunnel_positions = []
        for c in candidates[:5]:
            tunnel_seq = tunnel.tunnel_sequence(c, max_tunnels=3)
            tunnel_positions.extend(tunnel_seq)
            print(f"  Tunnel from {c}: {tunnel_seq}")
        
        print(f"All tunnel positions: {tunnel_positions}")
        
        # Test coherence field
        coherence_field = observer.coherence_field(candidates)
        print(f"Coherence field: {coherence_field}")
        
        # Find best candidate
        for candidate, coh in coherence_field.items():
            if n % candidate == 0:
                print(f"  Factor {candidate}: coherence = {coh}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def debug_axiom5_failure():
    """Debug why Axiom 5 is failing"""
    print("\n=== Debugging Axiom 5 Failure ===")
    
    try:
        from axiom5.spectral_mirror import SpectralMirror
        from axiom5.recursive_coherence import RecursiveCoherence
        
        n = 15  # 3 × 5
        factors = [3, 5]
        
        print(f"Testing n={n}, factors={factors}")
        
        # Test spectral mirror
        mirror = SpectralMirror(n)
        print("Mirror created")
        
        # Test recursive coherence
        recursive_coh = RecursiveCoherence(n)
        print("Recursive coherence created")
        
        # Find mirror points
        root = int(n**0.5) + 1
        mirror_points = []
        for candidate in range(2, min(root, 10)):
            mirror_point = mirror.find_mirror_point(candidate)
            mirror_points.append((candidate, mirror_point))
            print(f"  Mirror point for {candidate}: {mirror_point}")
        
        # Apply recursive coherence
        initial_field = {i: 0.5 for i in range(2, min(root, 10))}
        print(f"Initial field: {initial_field}")
        
        final_field = recursive_coh.recursive_coherence_iteration(initial_field, depth=3)
        print(f"Final field: {final_field}")
        
        # Find best factor
        print("Scoring factors...")
        for candidate, mirror_pos in mirror_points:
            if n % candidate == 0:
                score = final_field.get(candidate, 0) + final_field.get(mirror_pos, 0)
                print(f"  Factor {candidate} (mirror {mirror_pos}): score = {score}")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_axiom1_failure()
    debug_axiom4_failure()
    debug_axiom5_failure()
