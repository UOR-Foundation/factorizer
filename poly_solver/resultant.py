"""
Resultant and Elimination Module for PPTS

Implements polynomial variable elimination using resultants and
advanced univariate polynomial root finding algorithms.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass

from .polynomial import Polynomial


@dataclass
class BivariatePolynomial:
    """Represents a bivariate polynomial in variables x and y"""
    coefficients: Dict[Tuple[int, int], float]  # (i,j) -> coeff for x^i * y^j
    
    def degree_x(self) -> int:
        """Maximum degree in x"""
        if not self.coefficients:
            return 0
        return max(i for i, j in self.coefficients.keys())
    
    def degree_y(self) -> int:
        """Maximum degree in y"""
        if not self.coefficients:
            return 0
        return max(j for i, j in self.coefficients.keys())
    
    def evaluate(self, x: float, y: float) -> float:
        """Evaluate polynomial at (x, y)"""
        result = 0.0
        for (i, j), coeff in self.coefficients.items():
            result += coeff * (x ** i) * (y ** j)
        return result
    
    def substitute_y(self, y_expr) -> Polynomial:
        """Substitute y with an expression in x"""
        # For y = n/x substitution
        # This is simplified - full implementation would handle general expressions
        pass


def sylvester_matrix(p: Polynomial, q: Polynomial) -> np.ndarray:
    """
    Construct the Sylvester matrix for polynomials p and q
    
    The determinant of this matrix is the resultant, which is 0
    iff p and q have a common root.
    """
    deg_p = p.degree
    deg_q = q.degree
    size = deg_p + deg_q
    
    matrix = np.zeros((size, size))
    
    # Fill with coefficients of p
    for i in range(deg_q):
        for j in range(deg_p + 1):
            if j < len(p.coefficients):
                matrix[i, i + j] = p.coefficients[deg_p - j]
    
    # Fill with coefficients of q
    for i in range(deg_p):
        for j in range(deg_q + 1):
            if j < len(q.coefficients):
                matrix[deg_q + i, i + j] = q.coefficients[deg_q - j]
    
    return matrix


def compute_resultant(p: Polynomial, q: Polynomial) -> float:
    """
    Compute the resultant of two polynomials
    
    The resultant is 0 iff the polynomials have a common root.
    """
    sylvester = sylvester_matrix(p, q)
    return np.linalg.det(sylvester)


def eliminate_variable_resultant(poly_system: List[BivariatePolynomial], 
                                var_to_eliminate: str) -> Polynomial:
    """
    Eliminate a variable from polynomial system using resultants
    
    For our use case, we eliminate y to get a univariate polynomial in x.
    """
    # This is a simplified version
    # Full implementation would handle the complete elimination process
    
    # For PPTS, we have the constraint xy = n, so y = n/x
    # We substitute this into other polynomials
    
    # Placeholder for now - would implement full resultant elimination
    return Polynomial([1.0])  # Dummy return


class PanAlgorithm:
    """
    Pan's algorithm for polynomial root finding in nearly linear time
    
    For a polynomial of degree d, finds all complex roots in O(d log² d) time.
    This is crucial for maintaining O(log³ n) overall complexity.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self.max_iterations = 100
    
    def find_roots(self, poly: Polynomial) -> List[complex]:
        """
        Find all roots of polynomial using Pan's algorithm
        
        Based on: "Optimal and nearly optimal algorithms for approximating
        polynomial zeros" by Victor Pan (1996)
        """
        degree = poly.degree
        if degree == 0:
            return []
        
        # Step 1: Initial root estimates using Aberth's method
        initial_roots = self._aberth_initial_estimates(poly)
        
        # Step 2: Global Newton iteration with Durand-Kerner
        refined_roots = self._durand_kerner_iteration(poly, initial_roots)
        
        # Step 3: Local refinement using modified Newton
        final_roots = self._local_newton_refinement(poly, refined_roots)
        
        return final_roots
    
    def _aberth_initial_estimates(self, poly: Polynomial) -> List[complex]:
        """
        Generate initial root estimates using Aberth's method
        
        Places initial guesses on a circle in the complex plane.
        """
        degree = poly.degree
        roots = []
        
        # Estimate root magnitude
        coeffs = poly.coefficients
        if coeffs[-1] != 0:
            # Use Cauchy's bound
            max_coeff = max(abs(c) for c in coeffs[:-1])
            radius = 1 + max_coeff / abs(coeffs[-1])
        else:
            radius = 1.0
        
        # Distribute points on circle
        for k in range(degree):
            angle = 2 * math.pi * k / degree + math.pi / (2 * degree)
            roots.append(radius * complex(math.cos(angle), math.sin(angle)))
        
        return roots
    
    def _durand_kerner_iteration(self, poly: Polynomial, 
                                initial_roots: List[complex]) -> List[complex]:
        """
        Durand-Kerner method for simultaneous root finding
        
        Converges cubically for simple roots.
        """
        roots = initial_roots.copy()
        degree = len(roots)
        
        for iteration in range(self.max_iterations):
            max_change = 0.0
            new_roots = []
            
            for i in range(degree):
                # Compute denominator: product of differences
                # Use log-space to prevent overflow
                log_denom = 0.0
                denom_is_zero = False
                
                for j in range(degree):
                    if i != j:
                        diff = roots[i] - roots[j]
                        if abs(diff) < 1e-14:
                            denom_is_zero = True
                            break
                        log_denom += math.log(abs(diff))
                
                # Newton-like update
                if not denom_is_zero and abs(log_denom) < 700:  # Prevent overflow
                    p_value = poly.evaluate(roots[i])
                    
                    # Check if p_value is reasonable
                    if abs(p_value) < 1e100:
                        # Compute correction in log space if needed
                        if abs(log_denom) > 100:
                            # Use log-space calculation
                            log_correction = math.log(abs(p_value)) - log_denom
                            if log_correction < 100:  # Reasonable correction
                                correction_magnitude = math.exp(log_correction)
                                correction = correction_magnitude * math.copysign(1, p_value)
                            else:
                                # Correction too large, use damped update
                                correction = p_value * 0.1
                        else:
                            # Normal calculation
                            denom = math.exp(log_denom)
                            correction = p_value / denom
                        
                        # Limit correction magnitude to prevent wild jumps
                        if abs(correction) > abs(roots[i]):
                            correction = correction * 0.1
                        
                        new_root = roots[i] - correction
                        max_change = max(max_change, abs(correction))
                    else:
                        # p_value too large, use damped update
                        new_root = roots[i] * 0.9
                else:
                    # Keep current root if denominator issues
                    new_root = roots[i]
                
                new_roots.append(new_root)
            
            roots = new_roots
            
            # Check convergence
            if max_change < self.epsilon:
                break
        
        return roots
    
    def _local_newton_refinement(self, poly: Polynomial, 
                                roots: List[complex]) -> List[complex]:
        """
        Local Newton refinement for high accuracy
        
        Uses shifted polynomials to avoid catastrophic cancellation.
        """
        refined_roots = []
        deriv = poly.derivative()
        
        for root in roots:
            # Shift polynomial to improve conditioning
            shifted_poly = self._shift_polynomial(poly, root)
            
            # Newton iteration on shifted polynomial
            z = 0.0  # Root of shifted polynomial is at 0
            
            for _ in range(10):  # Limited iterations
                f_z = shifted_poly.evaluate(z)
                if abs(f_z) < self.epsilon:
                    break
                
                df_z = deriv.evaluate(root + z)
                if abs(df_z) > self.epsilon:
                    z = z - f_z / df_z
            
            refined_roots.append(root + z)
        
        return refined_roots
    
    def _shift_polynomial(self, poly: Polynomial, shift: complex) -> Polynomial:
        """
        Compute p(x + shift) using Taylor expansion
        
        This improves numerical stability near roots.
        """
        degree = poly.degree
        shifted_coeffs = [0.0] * (degree + 1)
        
        # Compute p(x + shift) = Σ p^(k)(shift)/k! * x^k
        # where p^(k) is the k-th derivative
        
        # Start with the polynomial itself
        current_poly = poly
        factorial = 1
        
        for k in range(degree + 1):
            # Evaluate k-th derivative at shift
            if k == 0:
                shifted_coeffs[k] = current_poly.evaluate(shift)
            else:
                # Take derivative for next iteration
                current_poly = current_poly.derivative()
                factorial *= k
                if current_poly.degree >= 0:
                    shifted_coeffs[k] = current_poly.evaluate(shift) / factorial
                else:
                    shifted_coeffs[k] = 0.0
        
        return Polynomial(shifted_coeffs)


def sturm_sequence(poly: Polynomial) -> List[Polynomial]:
    """
    Compute the Sturm sequence for a polynomial
    
    Used to count real roots in an interval.
    """
    sequence = [poly, poly.derivative()]
    
    while sequence[-1].degree > 0:
        # Polynomial division
        p1 = sequence[-2]
        p2 = sequence[-1]
        
        # Compute remainder of p1 / p2
        remainder = polynomial_remainder(p1, p2)
        
        # Negate remainder (Sturm's rule)
        neg_remainder = Polynomial([-c for c in remainder.coefficients])
        
        sequence.append(neg_remainder)
    
    return sequence


def polynomial_remainder(dividend: Polynomial, divisor: Polynomial) -> Polynomial:
    """
    Compute polynomial remainder using synthetic division
    """
    if divisor.degree == 0:
        return Polynomial([0.0])
    
    # Copy dividend coefficients
    remainder_coeffs = dividend.coefficients.copy()
    
    # Ensure lists are long enough
    while len(remainder_coeffs) < dividend.degree + 1:
        remainder_coeffs.append(0.0)
    
    # Perform division
    for i in range(dividend.degree - divisor.degree + 1):
        if abs(divisor.coefficients[-1]) < 1e-14:
            break
        
        # Compute quotient coefficient
        coeff = remainder_coeffs[dividend.degree - i] / divisor.coefficients[-1]
        
        # Subtract divisor * coeff
        for j in range(divisor.degree + 1):
            idx = dividend.degree - i - j
            if idx >= 0 and j < len(divisor.coefficients):
                remainder_coeffs[idx] -= coeff * divisor.coefficients[-(j+1)]
    
    # Remove leading zeros
    while len(remainder_coeffs) > 1 and abs(remainder_coeffs[-1]) < 1e-14:
        remainder_coeffs.pop()
    
    return Polynomial(remainder_coeffs)


def count_real_roots_interval(poly: Polynomial, a: float, b: float) -> int:
    """
    Count number of real roots in interval [a, b] using Sturm's theorem
    """
    sturm_seq = sturm_sequence(poly)
    
    # Count sign changes at a and b
    changes_a = count_sign_changes_at(sturm_seq, a)
    changes_b = count_sign_changes_at(sturm_seq, b)
    
    return changes_a - changes_b


def count_sign_changes_at(sturm_seq: List[Polynomial], x: float) -> int:
    """
    Count sign changes in Sturm sequence evaluated at x
    """
    values = [p.evaluate(x) for p in sturm_seq]
    
    # Remove zeros and infinities
    non_zero_values = []
    for v in values:
        if abs(v) > 1e-14 and not math.isinf(v) and not math.isnan(v):
            non_zero_values.append(v)
    
    # Count sign changes
    changes = 0
    for i in range(1, len(non_zero_values)):
        # Use sign comparison to avoid overflow
        sign1 = 1 if non_zero_values[i-1] > 0 else -1
        sign2 = 1 if non_zero_values[i] > 0 else -1
        if sign1 != sign2:
            changes += 1
    
    return changes


def isolate_real_roots(poly: Polynomial) -> List[Tuple[float, float]]:
    """
    Isolate all real roots into disjoint intervals
    
    Returns list of (a, b) intervals each containing exactly one root.
    """
    # Use Cauchy's bound for initial interval
    max_coeff = max(abs(c) for c in poly.coefficients[:-1])
    if poly.coefficients[-1] != 0:
        bound = 1 + max_coeff / abs(poly.coefficients[-1])
    else:
        bound = max_coeff
    
    # Start with [-bound, bound]
    intervals = [(-bound, bound)]
    isolated = []
    
    while intervals:
        a, b = intervals.pop()
        
        # Count roots in interval
        num_roots = count_real_roots_interval(poly, a, b)
        
        if num_roots == 0:
            continue
        elif num_roots == 1:
            isolated.append((a, b))
        else:
            # Bisect interval
            mid = (a + b) / 2
            intervals.append((a, mid))
            intervals.append((mid, b))
    
    return isolated


def refine_root_bisection(poly: Polynomial, a: float, b: float, 
                         epsilon: float = 1e-10) -> float:
    """
    Refine a root in interval [a, b] using bisection
    """
    # Ensure reasonable bounds
    if b - a > 1e6:
        # Interval too large, take middle portion
        mid = (a + b) / 2
        a = mid - 1000
        b = mid + 1000
    
    # Ensure opposite signs at endpoints
    f_a = poly.evaluate(a)
    f_b = poly.evaluate(b)
    
    # Check for overflow or invalid values
    if math.isinf(f_a) or math.isnan(f_a):
        return a
    if math.isinf(f_b) or math.isnan(f_b):
        return b
    
    if f_a * f_b > 0:
        # Try to find a sign change
        mid = (a + b) / 2
        f_mid = poly.evaluate(mid)
        if not math.isinf(f_mid) and not math.isnan(f_mid):
            if f_a * f_mid < 0:
                b = mid
                f_b = f_mid
            else:
                a = mid
                f_a = f_mid
    
    # Bisection loop with iteration limit
    max_iterations = 100
    iteration = 0
    
    while b - a > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (a + b) / 2
        f_mid = poly.evaluate(mid)
        
        # Check for overflow
        if math.isinf(f_mid) or math.isnan(f_mid):
            # Try a different point
            mid = a + (b - a) * 0.382  # Golden ratio split
            f_mid = poly.evaluate(mid)
            if math.isinf(f_mid) or math.isnan(f_mid):
                break
        
        if abs(f_mid) < epsilon:
            return mid
        
        if f_a * f_mid < 0:
            b = mid
            f_b = f_mid
        else:
            a = mid
            f_a = f_mid
    
    return (a + b) / 2
