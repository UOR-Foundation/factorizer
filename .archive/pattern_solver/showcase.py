"""
Showcase of The Pattern - Demonstrating Universal Factorization
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pattern import Pattern
from universal_basis import UniversalBasis
from factor_decoder import FactorDecoder
from advanced_pattern import AdvancedPattern


def visualize_universal_space(numbers, filename='universal_space.png'):
    """Visualize numbers in universal space"""
    basis = UniversalBasis()
    
    fig = plt.figure(figsize=(12, 8))
    
    # 3D visualization of first 3 coordinates
    ax1 = fig.add_subplot(121, projection='3d')
    
    for n in numbers:
        coords = basis.project(n)
        ax1.scatter(coords[0], coords[1], coords[2], s=100, alpha=0.6)
        ax1.text(coords[0], coords[1], coords[2], str(n), fontsize=8)
    
    ax1.set_xlabel('φ-coordinate')
    ax1.set_ylabel('π-coordinate')
    ax1.set_zlabel('e-coordinate')
    ax1.set_title('Numbers in Universal Space (3D)')
    
    # 2D projection
    ax2 = fig.add_subplot(122)
    
    for n in numbers:
        coords = basis.project(n)
        ax2.scatter(coords[0], coords[1], s=100, alpha=0.6)
        ax2.annotate(str(n), (coords[0], coords[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('φ-coordinate')
    ax2.set_ylabel('π-coordinate')
    ax2.set_title('Numbers in Universal Space (φ-π projection)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Universal space visualization saved to {filename}")


def showcase_pattern_discovery():
    """Showcase The Pattern's ability to discover relationships"""
    print("\n" + "="*60)
    print("THE PATTERN - DISCOVERING UNIVERSAL RELATIONSHIPS")
    print("="*60)
    
    basis = UniversalBasis()
    
    # Analyze Fibonacci-related semiprimes
    print("\n1. Fibonacci-Related Semiprimes")
    print("-" * 40)
    
    fib_semiprimes = [
        (15, "F₅ × F₄ = 5 × 3"),
        (35, "F₅ × F₆ = 5 × 7"),  # Close to Fib
        (143, "F₇ × F₇ = 11 × 13"),  # Close to Fib
        (323, "F₈ × F₈+2 = 17 × 19"),  # Close to Fib
    ]
    
    for n, description in fib_semiprimes:
        coords = basis.project(n)
        decomp = basis.decompose_in_basis(n)
        
        print(f"\n{description} = {n}")
        print(f"  φ-component: {coords[0]:.4f} (≈ {coords[0]/np.log(basis.PHI):.2f} × ln(φ))")
        print(f"  Resonance strength: {decomp['resonance_field']:.4f}")
        print(f"  Universal phase: {decomp['universal_phase']:.4f} radians")
    
    # Analyze prime gaps
    print("\n\n2. Prime Gap Patterns")
    print("-" * 40)
    
    gap_patterns = [
        (15, 3, 5, "Gap 2"),
        (77, 7, 11, "Gap 4"),
        (221, 13, 17, "Gap 4"),
        (899, 29, 31, "Gap 2"),
    ]
    
    for n, p, q, gap_type in gap_patterns:
        p_coords = basis.project(p)
        q_coords = basis.project(q)
        
        distance = basis.measure_distance(p_coords, q_coords)
        
        print(f"\n{p} × {q} = {n} ({gap_type})")
        print(f"  Universal distance(p,q): {distance:.4f}")
        print(f"  φ-ratio(p/q): {p_coords[0]/q_coords[0]:.4f}")
        print(f"  Phase alignment: {abs(p_coords[3] - q_coords[3]):.4f}")


def benchmark_pattern_performance():
    """Benchmark Pattern performance across different number sizes"""
    print("\n" + "="*60)
    print("PATTERN PERFORMANCE BENCHMARK")
    print("="*60)
    
    pattern = Pattern()
    basis = UniversalBasis()
    decoder = FactorDecoder(basis)
    
    pattern.universal_basis = basis
    pattern.decoder = decoder
    
    # Test different bit sizes
    test_suites = [
        ("Small (< 10 bits)", [
            (15, 3, 5),
            (21, 3, 7),
            (35, 5, 7),
            (77, 7, 11),
            (143, 11, 13),
        ]),
        ("Medium (10-15 bits)", [
            (1147, 31, 37),
            (1763, 41, 43),
            (2021, 43, 47),
            (3233, 53, 61),
            (5183, 71, 73),
        ]),
        ("Large (15-20 bits)", [
            (10403, 101, 103),
            (25619, 151, 169),  # 151 × 169 (13²)
            (39203, 197, 199),
            (69169, 257, 269),
            (121103, 311, 389),
        ])
    ]
    
    for suite_name, test_cases in test_suites:
        print(f"\n{suite_name}:")
        total_time = 0
        successes = 0
        
        for n, _, _ in test_cases:
            start = time.time()
            
            signature = pattern.recognize(n)
            formalization = pattern.formalize(signature)
            p, q = pattern.execute(formalization)
            
            elapsed = time.time() - start
            total_time += elapsed
            
            if p * q == n:
                successes += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {status} {n:6d} = {p:3d} × {q:3d} ({elapsed:.6f}s)")
        
        print(f"  Success rate: {successes}/{len(test_cases)}")
        print(f"  Average time: {total_time/len(test_cases):.6f}s")


def demonstrate_advanced_techniques():
    """Demonstrate advanced Pattern techniques"""
    print("\n" + "="*60)
    print("ADVANCED PATTERN TECHNIQUES")
    print("="*60)
    
    advanced = AdvancedPattern()
    
    # Test with challenging semiprimes
    challenging_cases = [
        (1073, 29, 37, "Non-adjacent primes"),
        (2491, 47, 53, "Larger gap"),
        (4087, 61, 67, "Gap 6 primes"),
        (7663, 83, 92, "Should be 83 × 92.35..."),  # This will fail
        (9409, 97, 97, "Perfect square of prime"),
    ]
    
    print("\nChallenging Semiprimes:")
    print("-" * 40)
    
    for n, _, _, description in challenging_cases:
        start = time.time()
        
        signature = advanced.recognize_advanced(n)
        formalization = advanced.formalize(signature)
        p, q = advanced.execute_advanced(formalization)
        
        elapsed = time.time() - start
        
        if p * q == n:
            print(f"✓ {n} = {p} × {q} - {description} ({elapsed:.6f}s)")
            
            # Show advanced analysis
            lie_coords = advanced.lie_structure.embed(n)
            adelic_profile = advanced.adelic_analyzer.analyze(n)
            
            print(f"  Lie coordinates: [{lie_coords[0]:.3f}, {lie_coords[1]:.3f}, {lie_coords[2]:.3f}]")
            print(f"  Adelic profile: {dict(list(adelic_profile.items())[:3])}")
        else:
            print(f"✗ {n} - {description} (got {p} × {q} = {p*q})")


def interactive_demonstration():
    """Interactive demonstration of The Pattern"""
    print("\n" + "="*60)
    print("INTERACTIVE PATTERN DEMONSTRATION")
    print("="*60)
    
    pattern = Pattern()
    basis = UniversalBasis()
    decoder = FactorDecoder(basis)
    
    pattern.universal_basis = basis
    pattern.decoder = decoder
    
    # Example number for detailed analysis
    n = 10403  # 101 × 103
    
    print(f"\nDetailed analysis of n = {n}")
    print("-" * 40)
    
    # Stage 1: Recognition
    print("\nStage 1: RECOGNITION")
    signature = pattern.recognize(n)
    
    print(f"Universal signature extracted:")
    print(f"  φ-component: {signature.phi_component:.6f}")
    print(f"  π-component: {signature.pi_component:.6f}")
    print(f"  e-component: {signature.e_component:.6f}")
    print(f"  Unity phase: {signature.unity_phase:.6f} radians")
    
    # Show resonance field
    print(f"\nResonance field (first 20 points):")
    field_display = [f"{x:.3f}" for x in signature.resonance_field[:20]]
    print(f"  {', '.join(field_display)}")
    
    # Stage 2: Formalization
    print("\nStage 2: FORMALIZATION")
    formalization = pattern.formalize(signature)
    
    print("Mathematical formalization:")
    print(f"  Harmonic series: {formalization['harmonic_series'][:5]}")
    print(f"  Resonance peaks: {formalization['resonance_peaks']}")
    print(f"  Factor encoding: {formalization['factor_encoding']}")
    
    # Show pattern matrix
    print("\nPattern matrix:")
    matrix = formalization['pattern_matrix']
    for i in range(matrix.shape[0]):
        row_str = " ".join(f"{x:7.3f}" for x in matrix[i])
        print(f"  [{row_str}]")
    
    # Stage 3: Execution
    print("\nStage 3: EXECUTION")
    p, q = pattern.execute(formalization)
    
    print(f"\nResult: {n} = {p} × {q}")
    
    # Verify and show relationships
    if p * q == n:
        print("\n✓ Factorization successful!")
        
        # Show factor relationships
        p_coords = basis.project(p)
        q_coords = basis.project(q)
        n_coords = basis.project(n)
        
        print(f"\nFactor relationships in universal space:")
        print(f"  Distance(p,q): {basis.measure_distance(p_coords, q_coords):.4f}")
        print(f"  φ-ratio(p/q): {p_coords[0]/q_coords[0]:.4f}")
        print(f"  Phase difference: {abs(p_coords[3] - q_coords[3]):.4f}")
        
        # Check for special relationships
        if abs(p_coords[0]/q_coords[0] - basis.PHI) < 0.1:
            print("  → Factors have golden ratio relationship!")
        
        if abs((p_coords[1] + q_coords[1]) - n_coords[1]) < 0.1:
            print("  → Harmonic conservation detected!")
    
    # Visualize
    numbers_to_viz = [p, q, n, 15, 21, 35, 77, 143, 221, 323]
    visualize_universal_space(numbers_to_viz, 'pattern_universal_space.png')


def main():
    """Run the showcase"""
    print("THE PATTERN SHOWCASE")
    print("===================")
    print("\nDemonstrating universal factorization through")
    print("Recognition → Formalization → Execution")
    
    showcase_pattern_discovery()
    benchmark_pattern_performance()
    demonstrate_advanced_techniques()
    interactive_demonstration()
    
    print("\n" + "="*60)
    print("SHOWCASE COMPLETE")
    print("="*60)
    print("\nThe Pattern reveals the universal structure underlying")
    print("prime factorization, transforming search into recognition.")


if __name__ == "__main__":
    main()