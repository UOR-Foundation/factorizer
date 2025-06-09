"""
Debug scale-invariant resonance to understand why it's failing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scale_invariant_resonance import ScaleInvariantResonator
import math


def debug_resonance_landscape(n, p, q):
    """Visualize the resonance landscape for a given semiprime."""
    print(f"\nDebugging n = {n} = {p} × {q}")
    print(f"sqrt(n) = {int(math.sqrt(n))}")
    
    resonator = ScaleInvariantResonator()
    
    # Sample resonance function
    num_samples = 1000
    x_normalized = np.linspace(0.001, 0.999, num_samples)
    resonance_values = []
    
    for x_norm in x_normalized:
        try:
            res = resonator._compute_scale_invariant_resonance(n, x_norm)
            resonance_values.append(res)
        except:
            resonance_values.append(0.0)
    
    resonance_values = np.array(resonance_values)
    
    # Find peaks
    max_idx = np.argmax(resonance_values)
    max_resonance = resonance_values[max_idx]
    max_x_norm = x_normalized[max_idx]
    max_x = int(max_x_norm * math.sqrt(n))
    
    print(f"\nResonance statistics:")
    print(f"  Max resonance: {max_resonance:.6f} at x_normalized={max_x_norm:.4f}")
    print(f"  Corresponding x: {max_x}")
    print(f"  Expected factor p: {p}")
    print(f"  Distance from p: {abs(max_x - p)}")
    
    # Check resonance at true factors
    p_normalized = p / math.sqrt(n)
    q_normalized = q / math.sqrt(n)
    
    if 0 < p_normalized < 1:
        p_resonance = resonator._compute_scale_invariant_resonance(n, p_normalized)
        print(f"\nResonance at p={p}: {p_resonance:.6f}")
    
    if 0 < q_normalized < 1:
        q_resonance = resonator._compute_scale_invariant_resonance(n, q_normalized)
        print(f"Resonance at q={q}: {q_resonance:.6f}")
    
    # Plot resonance landscape
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(x_normalized, resonance_values)
    plt.axvline(p_normalized, color='r', linestyle='--', label=f'p={p}')
    if q_normalized < 1:
        plt.axvline(q_normalized, color='g', linestyle='--', label=f'q={q}')
    plt.axvline(max_x_norm, color='b', linestyle=':', label=f'Max @ {max_x}')
    plt.xlabel('Normalized position (x / sqrt(n))')
    plt.ylabel('Resonance')
    plt.title(f'Resonance Landscape for n={n}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed plot near p
    plt.subplot(2, 1, 2)
    zoom_range = 0.1
    zoom_mask = np.abs(x_normalized - p_normalized) < zoom_range
    if np.any(zoom_mask):
        plt.plot(x_normalized[zoom_mask], resonance_values[zoom_mask])
        plt.axvline(p_normalized, color='r', linestyle='--', label=f'p={p}')
        plt.xlabel('Normalized position (x / sqrt(n))')
        plt.ylabel('Resonance')
        plt.title(f'Zoomed near p={p}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'scale_invariant_debug_{n}.png')
    plt.close()
    
    # Debug individual components at p
    if 0 < p_normalized < 1:
        print(f"\nDetailed resonance components at p={p}:")
        
        # Get individual scores
        order_score = resonator._multiplicative_order_coherence(n, p)
        qr_score = resonator._quadratic_residue_alignment(n, p)
        phase_score = resonator._phase_coherence(n, p)
        cf_score = resonator._continued_fraction_proximity(n, p)
        entropy_score = resonator._entropy_differential(n, p)
        norm_score = resonator._algebraic_norm_score(n, p)
        carmichael_score = resonator._carmichael_resonance(n, p)
        jacobi_score = resonator._jacobi_correlation(n, p)
        
        print(f"  Multiplicative order coherence: {order_score:.6f}")
        print(f"  Quadratic residue alignment: {qr_score:.6f}")
        print(f"  Phase coherence: {phase_score:.6f}")
        print(f"  Continued fraction proximity: {cf_score:.6f}")
        print(f"  Entropy differential: {entropy_score:.6f}")
        print(f"  Algebraic norm score: {norm_score:.6f}")
        print(f"  Carmichael resonance: {carmichael_score:.6f}")
        print(f"  Jacobi correlation: {jacobi_score:.6f}")
        
        total = 1.0
        for score in [order_score, qr_score, phase_score, cf_score, entropy_score, norm_score, carmichael_score, jacobi_score]:
            total *= (1 + score)
        print(f"  Total resonance: {total:.6f}")


def analyze_scale_invariance():
    """Analyze why scale-invariant detection is failing."""
    test_cases = [
        (11, 13),  # 143
        (17, 19),  # 323
        (101, 103),  # 10403
    ]
    
    for p, q in test_cases:
        n = p * q
        debug_resonance_landscape(n, p, q)
        print("\n" + "="*60)


def test_individual_properties():
    """Test if individual scale-invariant properties work."""
    resonator = ScaleInvariantResonator()
    
    # Test on 143 = 11 × 13
    n = 143
    p = 11
    
    print("\nTesting individual properties for 143 = 11 × 13")
    print("="*50)
    
    # 1. Multiplicative order
    print("\n1. Multiplicative Order Tests:")
    for a in [2, 3, 5, 7]:
        order_n = resonator._multiplicative_order(a, n)
        order_p = resonator._multiplicative_order(a, p)
        print(f"  ord_{n}({a}) = {order_n}")
        print(f"  ord_{p}({a}) = {order_p}")
        if order_p > 0 and order_n % order_p == 0:
            print(f"    ✓ ord_n divides ord_p")
    
    # 2. Quadratic residues
    print("\n2. Quadratic Residue Patterns:")
    qr_n = [a for a in range(1, 20) if resonator._is_quadratic_residue(a, n)]
    qr_p = [a for a in range(1, min(p, 20)) if resonator._is_quadratic_residue(a, p)]
    print(f"  QR mod {n}: {qr_n}")
    print(f"  QR mod {p}: {qr_p}")
    
    # 3. Continued fractions
    print("\n3. Continued Fraction Convergents of sqrt(143):")
    convergents = resonator._continued_fraction_convergents(n, max_terms=10)
    for i, (p_conv, q_conv) in enumerate(convergents):
        approx = p_conv / q_conv if q_conv > 0 else 0
        print(f"  {i}: {p_conv}/{q_conv} ≈ {approx:.6f}")
        if q_conv == p or q_conv == 13:
            print(f"    ✓ Denominator is a factor!")
    
    # 4. Jacobi symbols
    print("\n4. Jacobi Symbol Patterns:")
    for a in range(1, 10):
        j_n = resonator._jacobi_symbol(a, n)
        j_p = resonator._jacobi_symbol(a, p)
        print(f"  ({a}/{n}) = {j_n}, ({a}/{p}) = {j_p}")


if __name__ == "__main__":
    print("Scale-Invariant Resonance Debug\n" + "="*50)
    
    # First test individual properties
    test_individual_properties()
    
    # Then analyze full resonance
    print("\n\nFull Resonance Analysis")
    print("="*50)
    analyze_scale_invariance()
