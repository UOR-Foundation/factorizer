"""
Example demonstrating The Pattern in action
"""

import time
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pattern import Pattern
from universal_basis import UniversalBasis
from factor_decoder import FactorDecoder


def demonstrate_pattern(n: int):
    """Demonstrate The Pattern's three stages"""
    print(f"\n{'='*60}")
    print(f"Demonstrating The Pattern for n = {n}")
    print(f"{'='*60}")
    
    # Initialize components
    pattern = Pattern()
    basis = UniversalBasis()
    decoder = FactorDecoder(basis)
    
    # Connect components
    pattern.universal_basis = basis
    pattern.decoder = decoder
    
    start_time = time.time()
    
    # Stage 1: Recognition
    print("\nStage 1: RECOGNITION")
    print("Extracting universal signature...")
    signature = pattern.recognize(n)
    print(f"  φ-component: {signature.phi_component:.6f}")
    print(f"  π-component: {signature.pi_component:.6f}")
    print(f"  e-component: {signature.e_component:.6f}")
    print(f"  Unity phase: {signature.unity_phase:.6f}")
    print(f"  Resonance field: {len(signature.resonance_field)} points")
    
    # Stage 2: Formalization
    print("\nStage 2: FORMALIZATION")
    print("Expressing in universal mathematical language...")
    formalization = pattern.formalize(signature)
    
    print(f"  Universal coordinates: {formalization['universal_coordinates']}")
    print(f"  Resonance peaks: {formalization['resonance_peaks'][:5]}...")
    print(f"  Pattern matrix shape: {formalization['pattern_matrix'].shape}")
    print(f"  Factor encoding: {formalization['factor_encoding']}")
    
    # Stage 3: Execution
    print("\nStage 3: EXECUTION")
    print("Decoding factors through pattern operations...")
    
    # Try direct pattern execution first
    p, q = pattern.execute(formalization)
    
    # Verify result
    if p * q == n:
        elapsed = time.time() - start_time
        print(f"\n✓ SUCCESS: {n} = {p} × {q}")
        print(f"  Time: {elapsed:.6f} seconds")
        
        # Show how the factors relate in universal space
        p_coords = basis.project(p)
        q_coords = basis.project(q)
        n_coords = basis.project(n)
        
        print(f"\nUniversal space analysis:")
        print(f"  p({p}) coordinates: [{p_coords[0]:.3f}, {p_coords[1]:.3f}, {p_coords[2]:.3f}, {p_coords[3]:.3f}]")
        print(f"  q({q}) coordinates: [{q_coords[0]:.3f}, {q_coords[1]:.3f}, {q_coords[2]:.3f}, {q_coords[3]:.3f}]")
        print(f"  n({n}) coordinates: [{n_coords[0]:.3f}, {n_coords[1]:.3f}, {n_coords[2]:.3f}, {n_coords[3]:.3f}]")
        
        # Check for special relationships
        phi_ratio = p_coords[0] / q_coords[0]
        print(f"\n  φ-ratio (p/q): {phi_ratio:.6f}")
        if abs(phi_ratio - basis.PHI) < 0.1:
            print(f"    → Near golden ratio!")
        
        harmonic_sum = p_coords[1] + q_coords[1]
        print(f"  π-sum (p+q): {harmonic_sum:.6f}")
        if abs(harmonic_sum - n_coords[1]) < 0.1:
            print(f"    → Harmonic conservation!")
    else:
        print(f"\n✗ Failed to factor {n}")
        print(f"  Got: {p} × {q} = {p*q}")


def main():
    """Run examples"""
    print("THE PATTERN - Universal Factorization")
    print("=====================================")
    print("\nThe Pattern operates through three stages:")
    print("1. Recognition - Extract the universal signature")
    print("2. Formalization - Express in mathematical language")
    print("3. Execution - Apply operations to reveal structure")
    
    # Test cases
    test_numbers = [
        15,      # 3 × 5
        21,      # 3 × 7
        35,      # 5 × 7
        77,      # 7 × 11
        91,      # 7 × 13
        143,     # 11 × 13
        221,     # 13 × 17
        323,     # 17 × 19
        437,     # 19 × 23
        1147,    # 31 × 37
        1763,    # 41 × 43
        10403,   # 101 × 103
    ]
    
    for n in test_numbers:
        demonstrate_pattern(n)
        print()


if __name__ == "__main__":
    main()