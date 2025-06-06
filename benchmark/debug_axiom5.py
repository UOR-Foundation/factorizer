#!/usr/bin/env python3
"""
Debug Axiom 5 specific failure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_axiom5_detailed():
    """Debug Axiom 5 in detail"""
    print("=== Detailed Axiom 5 Debug ===")
    
    try:
        from axiom5.spectral_mirror import SpectralMirror
        from axiom5.recursive_coherence import RecursiveCoherence
        
        n = 15  # 3 × 5
        factors = [3, 5]
        
        print(f"Testing n={n}, expected factors={factors}")
        
        # Test spectral mirror
        mirror = SpectralMirror(n)
        print("✓ Mirror created")
        
        # Test recursive coherence
        recursive_coh = RecursiveCoherence(n)
        print("✓ Recursive coherence created")
        
        # Find mirror points for small factors
        root = int(n**0.5) + 1
        print(f"Checking candidates up to {root}")
        
        mirror_points = []
        for candidate in range(2, min(root, 10)):
            mirror_point = mirror.find_mirror_point(candidate)
            mirror_points.append((candidate, mirror_point))
            is_factor = n % candidate == 0
            print(f"  Candidate {candidate}: mirror={mirror_point}, is_factor={is_factor}")
        
        # Apply recursive coherence
        initial_field = {i: 0.5 for i in range(2, min(root, 10))}
        print(f"Initial field: {initial_field}")
        
        field_evolution = recursive_coh.recursive_coherence_iteration(initial_field, depth=3)
        print(f"Field evolution has {len(field_evolution)} levels")
        
        for i, field in enumerate(field_evolution):
            print(f"  Level {i}: {field}")
        
        final_field = field_evolution[-1]
        print(f"Final field: {final_field}")
        print(f"Final field type: {type(final_field)}")
        
        # Find best factor from mirrors and recursion
        best_factor = None
        best_score = 0
        
        print("Scoring factors...")
        for candidate, mirror_pos in mirror_points:
            if n % candidate == 0:
                candidate_score = final_field.get(candidate, 0) if hasattr(final_field, 'get') else 0
                mirror_score = final_field.get(mirror_pos, 0) if hasattr(final_field, 'get') else 0
                score = candidate_score + mirror_score
                print(f"  Factor {candidate}: candidate_score={candidate_score}, mirror_score={mirror_score}, total={score}")
                
                if score > best_score:
                    best_score = score
                    best_factor = candidate
        
        print(f"Best factor found: {best_factor} with score {best_score}")
        print(f"Success: {best_factor in factors}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_axiom5_detailed()
