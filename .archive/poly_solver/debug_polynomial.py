"""
Debug tool for polynomial construction in PPTS

Helps visualize and debug the polynomial construction process.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from .polynomial import Polynomial, construct_polynomial_system
from .harmonic import extract_harmonic_signature
from .adelic import construct_adelic_system


def analyze_polynomial(n: int) -> None:
    """
    Analyze the polynomial constructed for factoring n
    """
    print(f"\nAnalyzing polynomial for n = {n}")
    print(f"Factors: ", end="")
    
    # Find actual factors
    factors = []
    sqrt_n = int(math.sqrt(n))
    for i in range(2, sqrt_n + 1):
        if n % i == 0:
            factors.append(i)
            print(f"{i} ", end="")
    print()
    
    # Extract harmonic signature
    signature = extract_harmonic_signature(n)
    print(f"\nHarmonic signature (trace): {signature.trace():.4f}")
    
    # Construct adelic system
    adelic_system = construct_adelic_system(n, signature)
    print(f"Adelic constraints: {len(adelic_system.p_adic_constraints)} primes")
    
    # Construct polynomial system
    poly_system = construct_polynomial_system(n, adelic_system)
    poly = poly_system.polynomials[0]
    
    print(f"\nPolynomial degree: {poly.degree}")
    print(f"Coefficients: {[f'{c:.4f}' for c in poly.coefficients[:5]]}...")
    
    # Evaluate polynomial at factors
    print("\nPolynomial values at factors:")
    for factor in factors:
        value = poly.evaluate(factor)
        print(f"  P({factor}) = {value:.6f}")
    
    # Find roots
    from .polynomial import PolynomialSolver
    solver = PolynomialSolver()
    
    # Try different root finding methods
    print("\nRoot finding results:")
    
    # 1. Integer roots
    integer_roots = solver.find_integer_roots(poly, sqrt_n)
    print(f"  Integer roots: {integer_roots}")
    
    # 2. Companion matrix
    complex_roots = solver.find_roots_companion_matrix(poly)
    real_roots = [r.real for r in complex_roots if abs(r.imag) < 0.01]
    print(f"  Real roots from companion: {[f'{r:.2f}' for r in real_roots]}")
    
    # Plot polynomial
    if poly.degree <= 10:
        plot_polynomial(poly, n, factors)


def plot_polynomial(poly: Polynomial, n: int, factors: List[int]) -> None:
    """
    Plot the polynomial to visualize its behavior
    """
    sqrt_n = int(math.sqrt(n))
    
    # Generate x values
    x_min = max(2, sqrt_n - sqrt_n // 2)
    x_max = min(sqrt_n + sqrt_n // 2, n // 2)
    x_vals = np.linspace(x_min, x_max, 1000)
    
    # Evaluate polynomial
    y_vals = [poly.evaluate(x) for x in x_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'b-', label='P(x)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Mark factors
    for factor in factors:
        if x_min <= factor <= x_max:
            y_factor = poly.evaluate(factor)
            plt.plot(factor, y_factor, 'ro', markersize=10, label=f'Factor {factor}')
            plt.axvline(x=factor, color='r', linestyle=':', alpha=0.3)
    
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.title(f'Polynomial for n = {n}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(f'poly_debug_{n}.png')
    plt.close()
    print(f"\nPlot saved as poly_debug_{n}.png")


def construct_direct_polynomial(n: int, degree: int) -> Polynomial:
    """
    Construct a polynomial directly with known factors as roots
    
    This is a more direct approach for testing.
    """
    sqrt_n = int(math.sqrt(n))
    
    # Find all factors
    factors = []
    for i in range(2, min(1000, sqrt_n + 1)):
        if n % i == 0:
            factors.append(i)
    
    if not factors:
        # No small factors found, construct based on sqrt(n)
        return Polynomial([-sqrt_n, 1.0])
    
    # Start with polynomial (x - factors[0])
    coeffs = [-factors[0], 1.0]
    
    # Multiply by (x - factor) for each additional factor
    for factor in factors[1:]:
        if len(coeffs) >= degree:
            break
        
        # Multiply current polynomial by (x - factor)
        new_coeffs = [0.0] * (len(coeffs) + 1)
        for i in range(len(coeffs)):
            new_coeffs[i] -= factor * coeffs[i]
            new_coeffs[i + 1] += coeffs[i]
        coeffs = new_coeffs
    
    # Ensure degree doesn't exceed limit
    if len(coeffs) > degree + 1:
        coeffs = coeffs[:degree + 1]
    
    return Polynomial(coeffs)


if __name__ == "__main__":
    # Test on small semiprimes
    test_cases = [15, 21, 35, 77, 91, 143, 221, 323]
    
    for n in test_cases:
        analyze_polynomial(n)
        print("\n" + "="*60)
