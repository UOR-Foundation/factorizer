"""
RFH2 Unity Diagnostic Tool
Analyzes the Unity-based Prime Resonance Function
"""

import math
from rfh2_improved import PrimeResonanceFunction, TransitionBoundaries, SMALL_PRIMES, GOLDEN_RATIO, PRIME_UNITY, TRIBONACCI, EGYPTIAN_SPREAD

def diagnose_factor(n: int, factor: int):
    """Diagnose why a specific factor might be missed"""
    print(f"\n{'='*60}")
    print(f"Diagnostic Analysis: n={n}, factor={factor}")
    print(f"{'='*60}")
    
    prf = PrimeResonanceFunction()
    
    # Compute components
    U = prf.compute_unity_resonance(factor, n)
    P = prf.compute_phase_coherence(factor, n)
    H = prf.compute_harmonic_convergence(factor, n)
    psi = prf.psi(factor, n)
    
    print(f"\nResonance Components for factor {factor}:")
    print(f"  U (Unity Resonance):       {U:.6f}")
    print(f"  P (Phase Coherence):       {P:.6f}")
    print(f"  H (Harmonic Convergence):  {H:.6f}")
    print(f"  Ψ (Total Unity):           {psi:.6f}")
    print(f"  Product U×P×H:             {U*P*H:.6f}")
    
    # Check if factor is in resonance nodes
    tb = TransitionBoundaries()
    nodes = tb.calculate_resonance_nodes(n)
    
    if factor in nodes:
        print(f"\n✓ Factor {factor} IS in resonance nodes (position {nodes.index(factor)+1}/{len(nodes)})")
    else:
        print(f"\n✗ Factor {factor} is NOT in resonance nodes")
        
        # Find nearest nodes
        nearest_below = max([x for x in nodes if x < factor], default=None)
        nearest_above = min([x for x in nodes if x > factor], default=None)
        
        if nearest_below:
            print(f"  Nearest node below: {nearest_below} (distance: {factor - nearest_below})")
        if nearest_above:
            print(f"  Nearest node above: {nearest_above} (distance: {nearest_above - factor})")
    
    # Analyze why each component might be low
    print(f"\nComponent Analysis:")
    
    # Unity Resonance
    print(f"\n1. Unity Resonance (U={U:.6f}):")
    if n % factor == 0:
        print(f"   - Perfect factor, U=1.0 by design")
    else:
        omega_n = PRIME_UNITY / math.log(n + 1)
        omega_x = PRIME_UNITY / math.log(factor + 1)
        phase_diff = abs(omega_n - omega_x * round(omega_n / omega_x))
        print(f"   - Fundamental freq of n: {omega_n:.6f}")
        print(f"   - Harmonic freq of x: {omega_x:.6f}")
        print(f"   - Phase difference: {phase_diff:.6f}")
    
    # Phase Coherence
    print(f"\n2. Phase Coherence (P={P:.6f}):")
    phase_details = []
    for p in SMALL_PRIMES[:7]:
        phase_n = (n % p) * PRIME_UNITY / p
        phase_x = (factor % p) * PRIME_UNITY / p
        coherence = math.cos(phase_n - phase_x)
        phase_details.append((p, phase_n, phase_x, coherence))
    
    # Show phase alignments
    phase_details.sort(key=lambda x: x[3])
    print("   Phase alignments (worst to best):")
    for p, p_n, p_x, coh in phase_details[:5]:
        print(f"   - Prime {p}: phase_n={p_n:.3f}, phase_x={p_x:.3f}, coherence={coh:.3f}")
    
    # Harmonic Convergence
    print(f"\n3. Harmonic Convergence (H={H:.6f}):")
    
    # Unity harmonic
    unity_freq = PRIME_UNITY / math.gcd(factor, n)
    unity_harmonic = math.cos(unity_freq * math.log(n) / PRIME_UNITY)
    print(f"   - Unity harmonic: {(1 + unity_harmonic) / 2:.6f}")
    
    # Golden ratio convergence
    phi_harmonic = factor / GOLDEN_RATIO
    phi_distance = min(abs(phi_harmonic - int(phi_harmonic)), 
                      abs(phi_harmonic - int(phi_harmonic) - 1))
    phi_convergence = math.exp(-phi_distance * GOLDEN_RATIO)
    print(f"   - Golden ratio convergence: {phi_convergence:.6f}")
    
    # Tribonacci resonance
    if factor > 2:
        tri_phase = math.log(factor) / math.log(TRIBONACCI)
        tri_resonance = abs(math.sin(tri_phase * math.pi))
        print(f"   - Tribonacci resonance: {tri_resonance:.6f}")
    
    # Perfect square check
    near_square = int(math.sqrt(factor))
    if near_square * near_square == factor:
        print(f"   - Perfect square: YES (perfect harmony)")
    else:
        square_dist = min(factor - near_square**2, (near_square + 1)**2 - factor)
        square_harmony = math.exp(-square_dist / factor)
        print(f"   - Square harmony: {square_harmony:.6f} (distance: {square_dist})")


def analyze_test_cases():
    """Analyze all test cases to understand failure patterns"""
    
    test_cases = [
        (143, 11, 13),
        (10403, 101, 103),
        (282492, 531, 532),
        (282943, 523, 541),
        (281479272661007, 65537, 4294967311),
        (9998000099, 99991, 99989),
        (49583104564635624413, 7125766127, 6958284019),
    ]
    
    for n, p, q in test_cases:
        # Test smaller factor
        diagnose_factor(n, p)
        
        # Also show resonance node distribution
        tb = TransitionBoundaries()
        nodes = tb.calculate_resonance_nodes(n)
        print(f"\nResonance nodes near {p}:")
        relevant = [x for x in nodes if 0.5*p <= x <= 2*p]
        if relevant:
            print(f"  {relevant}")
        else:
            print(f"  No nodes within 50-200% of factor")


if __name__ == "__main__":
    analyze_test_cases()
