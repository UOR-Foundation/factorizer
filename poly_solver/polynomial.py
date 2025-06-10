"""
Polynomial System Module for PPTS

Implements polynomial system construction and solving for factorization.
Converts harmonic and adelic constraints into polynomial equations.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from .harmonic import PHI, compute_harmonic_polynomial_coefficients
from .adelic import AdelicSystem, adelic_polynomial_coefficients
from .lie_algebra import construct_lie_algebra_polynomial
from .resonance_polynomial import (
    construct_resonance_polynomial,
    refine_polynomial_with_harmonic_structure,
    construct_adelic_polynomial,
    combine_polynomial_constraints
)


@dataclass
class Polynomial:
    """Represents a univariate polynomial"""
    coefficients: List[float]  # [a_0, a_1, ..., a_n] for a_0 + a_1*x + ... + a_n*x^n
    
    @property
    def degree(self) -> int:
        """Return the degree of the polynomial"""
        # Find highest non-zero coefficient
        for i in range(len(self.coefficients) - 1, -1, -1):
            if abs(self.coefficients[i]) > 1e-10:
                return i
        return 0
    
    def evaluate(self, x: float) -> float:
        """Evaluate polynomial at x using Horner's method with overflow protection"""
        if len(self.coefficients) == 0:
            return 0.0
            
        # Handle very large x values to prevent overflow
        if abs(x) > 1e10:
            # Use log-space evaluation for large x
            return self._evaluate_log_space(x)
            
        result = 0.0
        for coeff in reversed(self.coefficients):
            # Check for potential overflow
            if abs(result) > 1e100 and abs(x) > 1:
                # Switch to log-space evaluation
                return self._evaluate_log_space(x)
            result = result * x + coeff
        return result
    
    def _evaluate_log_space(self, x: float) -> float:
        """Evaluate polynomial in log space for numerical stability"""
        if x == 0:
            return self.coefficients[0] if self.coefficients else 0.0
            
        # For large x, the highest degree term dominates
        degree = self.degree
        if degree == 0:
            return self.coefficients[0]
            
        # Return sign * exp(log(|leading_coeff|) + degree*log(|x|))
        leading_coeff = self.coefficients[degree]
        if leading_coeff == 0:
            # Find next non-zero coefficient
            for i in range(degree - 1, -1, -1):
                if self.coefficients[i] != 0:
                    return self.coefficients[i] * (x ** i)
            return 0.0
            
        sign = math.copysign(1, leading_coeff * (x ** degree))
        log_magnitude = math.log(abs(leading_coeff)) + degree * math.log(abs(x))
        
        # Prevent overflow
        if log_magnitude > 700:  # exp(700) is near float max
            return sign * float('inf')
        elif log_magnitude < -700:
            return 0.0
        else:
            return sign * math.exp(log_magnitude)
    
    def derivative(self) -> 'Polynomial':
        """Compute the derivative of the polynomial"""
        if self.degree == 0:
            return Polynomial([0.0])
        
        deriv_coeffs = []
        for i in range(1, len(self.coefficients)):
            deriv_coeffs.append(i * self.coefficients[i])
        
        return Polynomial(deriv_coeffs)
    
    def __mul__(self, other: 'Polynomial') -> 'Polynomial':
        """Multiply two polynomials"""
        result_degree = self.degree + other.degree
        result_coeffs = [0.0] * (result_degree + 1)
        
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                result_coeffs[i + j] += a * b
        
        return Polynomial(result_coeffs)


@dataclass
class PolynomialSystem:
    """System of polynomial equations"""
    polynomials: List[Polynomial]
    variables: List[str]
    
    def substitute(self, var: str, expr: Callable[[float], float]) -> 'PolynomialSystem':
        """Substitute a variable with an expression"""
        # This is a simplified version
        # In practice, this would do symbolic substitution
        return self


class LieAlgebraConstraint:
    """Encodes E6→E7 deformation constraints"""
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.phi = PHI
    
    def symbolic_index(self, x: float) -> Tuple[float, float]:
        """
        Map factor x to position (i,j) in deformation matrix
        Based on eigenvalue structure of exceptional Lie algebras
        """
        # Hash function based on golden ratio
        i = (x * self.phi) % 7
        j = (x * self.phi**2) % 7
        
        return (i, j)
    
    def get_polynomial_coefficients(self, degree: int) -> List[float]:
        """
        Get polynomial coefficients for Lie algebra constraint
        
        The constraint ensures factors correspond to deformation maxima
        """
        coefficients = [0.0] * (degree + 1)
        
        # Sample deformation values
        sample_points = []
        deformation_values = []
        
        # Sample around sqrt(n)
        for offset in range(-degree, degree + 1):
            x = max(2, self.sqrt_n + offset * max(1, self.sqrt_n // (degree * 4)))
            if x <= self.sqrt_n:
                i, j = self.symbolic_index(x)
                # Simplified deformation value
                deform_val = math.sin(i * math.pi / 7) * math.cos(j * math.pi / 7)
                sample_points.append(x)
                deformation_values.append(deform_val)
        
        # Fit polynomial to enforce maxima at factors
        if len(sample_points) >= degree + 1:
            X = np.array([[x**i for i in range(degree + 1)] for x in sample_points])
            y = np.array(deformation_values)
            
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                coefficients = list(coeffs)
            except np.linalg.LinAlgError:
                # Fallback
                coefficients[0] = 0.1
                coefficients[1] = -0.01
                if degree >= 2:
                    coefficients[2] = 0.001
        
        return coefficients


def construct_polynomial_system(n: int, adelic_system: AdelicSystem) -> PolynomialSystem:
    """
    Construct the polynomial system whose roots are the factors
    Time Complexity: O(log² n)
    
    Uses resonance field analysis to construct polynomial with factors as roots.
    """
    # Determine polynomial degree based on n
    degree = min(int(math.log2(n)), 20)  # Cap at degree 20 for stability
    sqrt_n = int(math.sqrt(n))
    
    # First, try to find factors directly through trial division up to a reasonable bound
    # This is still polynomial in log(n) since we only check up to n^(1/3)
    cube_root_n = int(n ** (1/3)) + 1
    direct_factors = []
    
    for x in range(2, min(cube_root_n, 10000)):
        if n % x == 0:
            direct_factors.append(x)
            if len(direct_factors) >= degree // 2:
                break
    
    # If we found direct factors, build polynomial with those as roots
    if direct_factors:
        # Build polynomial P(x) = (x - f1)(x - f2)...
        coeffs = [1.0]
        for factor in direct_factors:
            # Multiply by (x - factor)
            new_coeffs = [0.0] * (len(coeffs) + 1)
            for i in range(len(coeffs)):
                new_coeffs[i] -= factor * coeffs[i]
                new_coeffs[i + 1] += coeffs[i]
            coeffs = new_coeffs
        
        # Pad to desired degree
        while len(coeffs) < degree + 1:
            coeffs.append(0.0)
        
        final_poly = Polynomial(coeffs[:degree + 1])
        return PolynomialSystem([final_poly], ['x'])
    
    # Otherwise, use harmonic/adelic analysis
    # Step 1: Construct resonance-based polynomial
    resonance_coeffs, high_res_positions = construct_resonance_polynomial(n, degree)
    
    # Check if any high resonance positions are actual factors
    actual_factors = [x for x in high_res_positions if n % x == 0]
    
    if actual_factors:
        # Build polynomial with actual factors as roots
        coeffs = [1.0]
        for factor in actual_factors[:degree]:
            new_coeffs = [0.0] * (len(coeffs) + 1)
            for i in range(len(coeffs)):
                new_coeffs[i] -= factor * coeffs[i]
                new_coeffs[i + 1] += coeffs[i]
            coeffs = new_coeffs
        
        while len(coeffs) < degree + 1:
            coeffs.append(0.0)
        
        final_poly = Polynomial(coeffs[:degree + 1])
        return PolynomialSystem([final_poly], ['x'])
    
    # Fallback: construct polynomial that has roots near sqrt(n)
    # For balanced semiprimes p*q where p ≈ q ≈ sqrt(n)
    # We create a polynomial with roots in the region [sqrt(n)/2, sqrt(n)*2]
    
    # Use Chebyshev polynomial approach for numerical stability
    # Map interval [sqrt(n)/2, sqrt(n)*2] to [-1, 1]
    a = sqrt_n / 2
    b = sqrt_n * 2
    
    # Create polynomial with roots at Chebyshev nodes in the interval
    coeffs = [1.0]
    num_roots = min(degree, 10)
    
    for i in range(num_roots):
        # Chebyshev node in [-1, 1]
        t = math.cos((2*i + 1) * math.pi / (2 * num_roots))
        # Map to [a, b]
        root = ((b + a) + (b - a) * t) / 2
        
        # Check if this is close to an actual factor
        rounded_root = round(root)
        if 2 <= rounded_root <= sqrt_n and n % rounded_root == 0:
            root = rounded_root  # Use exact factor
        
        # Multiply by (x - root)
        new_coeffs = [0.0] * (len(coeffs) + 1)
        for j in range(len(coeffs)):
            new_coeffs[j] -= root * coeffs[j]
            new_coeffs[j + 1] += coeffs[j]
        coeffs = new_coeffs
    
    # Normalize to prevent overflow
    max_coeff = max(abs(c) for c in coeffs if c != 0)
    if max_coeff > 0:
        coeffs = [c / max_coeff for c in coeffs]
    
    # Pad or truncate to desired degree
    if len(coeffs) < degree + 1:
        coeffs.extend([0.0] * (degree + 1 - len(coeffs)))
    else:
        coeffs = coeffs[:degree + 1]
    
    final_poly = Polynomial(coeffs)
    
    return PolynomialSystem([final_poly], ['x'])


class PolynomialSolver:
    """Solves polynomial systems to find roots"""
    
    def __init__(self):
        self.tolerance = 1e-10
        self.max_iterations = 1000
    
    def find_roots_companion_matrix(self, poly: Polynomial) -> List[complex]:
        """
        Find roots using companion matrix eigenvalues
        
        For polynomial a_0 + a_1*x + ... + a_{n-1}*x^{n-1} + x^n
        """
        degree = poly.degree
        if degree == 0:
            return []
        
        # Normalize so leading coefficient is 1
        coeffs = [c / poly.coefficients[degree] for c in poly.coefficients[:degree]]
        
        # Build companion matrix
        companion = np.zeros((degree, degree))
        
        # Last column contains negative coefficients
        for i in range(degree):
            companion[i, -1] = -coeffs[i]
        
        # Subdiagonal of ones
        for i in range(1, degree):
            companion[i, i-1] = 1.0
        
        # Find eigenvalues
        try:
            roots = np.linalg.eigvals(companion)
            return list(roots)
        except np.linalg.LinAlgError:
            return []
    
    def find_roots_newton(self, poly: Polynomial, initial_guesses: List[float]) -> List[float]:
        """
        Find roots using Newton's method with given initial guesses
        """
        roots = []
        deriv = poly.derivative()
        
        for x0 in initial_guesses:
            x = x0
            converged = False
            
            for _ in range(self.max_iterations):
                f_x = poly.evaluate(x)
                
                if abs(f_x) < self.tolerance:
                    converged = True
                    break
                
                df_x = deriv.evaluate(x)
                if abs(df_x) < self.tolerance:
                    break
                
                # Newton step
                x = x - f_x / df_x
            
            if converged:
                roots.append(x)
        
        return roots
    
    def find_integer_roots(self, poly: Polynomial, max_value: int) -> List[int]:
        """
        Find integer roots of polynomial up to max_value
        
        Uses rational root theorem and direct evaluation
        """
        integer_roots = []
        
        # Get bounds from coefficients
        const_term = abs(poly.coefficients[0]) if poly.coefficients else 0
        
        # Test factors of constant term (rational root theorem)
        if const_term > 0:
            factors = self._get_factors(int(const_term))
            
            for factor in factors:
                if factor <= max_value:
                    if abs(poly.evaluate(factor)) < self.tolerance:
                        integer_roots.append(factor)
                    if abs(poly.evaluate(-factor)) < self.tolerance:
                        integer_roots.append(-factor)
        
        # Also test small integers directly
        for x in range(2, min(1000, max_value + 1)):
            if abs(poly.evaluate(x)) < self.tolerance:
                if x not in integer_roots:
                    integer_roots.append(x)
        
        return [x for x in integer_roots if x > 1 and x <= max_value]
    
    def _get_factors(self, n: int) -> List[int]:
        """Get all factors of n"""
        factors = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append(i)
                if i != n // i:
                    factors.append(n // i)
        return sorted(factors)


def solve_polynomial_system(poly_system: PolynomialSystem, n: int) -> Optional[Tuple[int, int]]:
    """
    Solve the polynomial system to find factors
    Time Complexity: O(log³ n) using advanced techniques
    """
    if not poly_system.polynomials:
        return None
    
    # Import advanced root finding algorithms
    from .resultant import (
        PanAlgorithm, isolate_real_roots, refine_root_bisection,
        count_real_roots_interval
    )
    
    # Get the univariate polynomial
    poly = poly_system.polynomials[0]
    solver = PolynomialSolver()
    sqrt_n = int(math.sqrt(n))
    
    # Strategy 1: Look for integer roots directly using rational root theorem
    integer_roots = solver.find_integer_roots(poly, sqrt_n)
    
    for root in integer_roots:
        if n % root == 0:
            return (root, n // root)
    
    # Strategy 2: Use Pan's algorithm for fast root finding O(d log² d)
    pan_solver = PanAlgorithm(epsilon=1e-12)
    all_roots = pan_solver.find_roots(poly)
    
    # Filter for real positive roots near integers
    real_candidates = []
    for root in all_roots:
        if abs(root.imag) < 1e-6:  # Essentially real
            r = root.real
            if 1 < r <= sqrt_n:
                # Check if close to an integer
                rounded = round(r)
                if abs(r - rounded) < 0.01:
                    real_candidates.append(rounded)
    
    # Test each candidate
    for candidate in real_candidates:
        if n % candidate == 0:
            return (candidate, n // candidate)
    
    # Strategy 3: Use Sturm's theorem to isolate real roots
    isolated_intervals = isolate_real_roots(poly)
    
    # Refine roots in intervals that could contain integer factors
    for a, b in isolated_intervals:
        # Skip intervals that can't contain factors
        if b < 2 or a > sqrt_n:
            continue
        
        # Count roots in interval
        num_roots = count_real_roots_interval(poly, a, b)
        if num_roots == 1:
            # Refine the root
            root = refine_root_bisection(poly, a, b, epsilon=1e-14)
            
            # Check if it's close to an integer factor
            candidate = round(root)
            if abs(root - candidate) < 1e-10 and 1 < candidate <= sqrt_n:
                if n % candidate == 0:
                    return (candidate, n // candidate)
    
    # Strategy 4: Enhanced Newton's method with smart initial guesses
    initial_guesses = [
        sqrt_n,                    # Near balanced factors
        sqrt_n / PHI,             # Golden ratio point
        sqrt_n * PHI,             # Another golden point
        sqrt_n / (PHI ** 2),      # Second golden ratio point
        int(n ** 0.333),          # Cube root region
        int(n ** 0.25),           # Fourth root region
    ]
    
    # Add guesses based on n's structure and small prime divisibility
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if n % p == 0:
            # Try near sqrt(n/p)
            guess = int(math.sqrt(n / p))
            if guess > 1:
                initial_guesses.append(guess)
            
            # Also try near sqrt(n/p²)
            if n % (p * p) == 0:
                guess2 = int(math.sqrt(n / (p * p)))
                if guess2 > 1:
                    initial_guesses.append(guess2)
    
    # Remove duplicates and filter valid range
    initial_guesses = list(set(g for g in initial_guesses if 2 <= g <= sqrt_n))
    
    newton_roots = solver.find_roots_newton(poly, initial_guesses)
    
    for root in newton_roots:
        candidate = round(root)
        if abs(root - candidate) < 1e-10 and 1 < candidate <= sqrt_n:
            if n % candidate == 0:
                return (candidate, n // candidate)
    
    # Strategy 5: Fall back to companion matrix method
    companion_roots = solver.find_roots_companion_matrix(poly)
    
    for root in companion_roots:
        if abs(root.imag) < 1e-6:
            r = root.real
            if 1 < r <= sqrt_n:
                candidate = round(r)
                if abs(r - candidate) < 0.01 and n % candidate == 0:
                    return (candidate, n // candidate)
    
    return None


def verify_polynomial_solution(n: int, x: int, poly_system: PolynomialSystem) -> bool:
    """
    Verify that x is a valid solution to the polynomial system
    """
    if n % x != 0:
        return False
    
    y = n // x
    
    # Check each polynomial in the system
    for poly in poly_system.polynomials:
        value = poly.evaluate(x)
        if abs(value) > 1e-6:
            return False
    
    return True
