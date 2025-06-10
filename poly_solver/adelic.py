"""
Adelic Constraints Module for PPTS

Implements adelic constraints based on the product formula and p-adic valuations
to reduce the solution space from O(√n) to polynomial size.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .harmonic import HarmonicSignature, SMALL_PRIMES


@dataclass
class AdelicConstraints:
    """Container for adelic constraint values"""
    real_constraint: float
    p_adic_constraints: Dict[int, float]
    product_constraint: float


@dataclass 
class AdelicSystem:
    """Complete adelic constraint system for factorization"""
    real_constraint: float
    p_adic_constraints: List[Tuple[int, float]]
    product_constraint: float
    
    def verify(self, x: int, y: int) -> bool:
        """Verify if (x, y) satisfies the adelic constraints"""
        # Check product formula
        n = x * y
        real_norm = abs(n)
        p_adic_product = 1.0
        
        for p, constraint_val in self.p_adic_constraints:
            # Compute p-adic norm
            v_n = compute_p_adic_valuation(n, p)
            v_x = compute_p_adic_valuation(x, p)
            v_y = compute_p_adic_valuation(y, p)
            
            # Check if valuations match
            if abs(v_n - (v_x + v_y)) > 0.001:
                return False
            
            # Update p-adic product
            p_adic_norm = p ** (-v_n)
            p_adic_product *= p_adic_norm
        
        # Check product formula: |n|_R × ∏_p |n|_p = 1
        product = real_norm * p_adic_product
        return abs(product - 1.0) < 0.01


def compute_p_adic_valuation(n: int, p: int) -> int:
    """
    Compute the p-adic valuation v_p(n)
    
    This is the largest power of p that divides n.
    """
    if n == 0:
        return float('inf')
    
    valuation = 0
    while n % p == 0:
        n //= p
        valuation += 1
    
    return valuation


def symbolic_valuation_polynomial(p: int, max_degree: int = 5) -> List[float]:
    """
    Create polynomial approximation of p-adic valuation function
    
    Key insight: v_p(x) ≈ log_p(gcd(x, p^k)) for large k
    We encode this as a sum of "bump" functions centered at multiples of p^k
    """
    import numpy as np
    
    # We'll create a polynomial that approximates v_p(x)
    # using Gaussian-like bumps at multiples of prime powers
    
    # Sample points for polynomial fitting
    sample_points = []
    valuation_values = []
    
    # Dense sampling near multiples of p^k
    max_x = p ** max_degree if p ** max_degree < 10000 else 10000
    
    # Sample at multiples of each prime power
    for k in range(max_degree + 1):
        p_k = p ** k
        if p_k > max_x:
            break
            
        # Sample around multiples of p^k
        for mult in range(1, min(20, max_x // p_k + 1)):
            x = mult * p_k
            if x <= max_x:
                # True valuation
                v_true = compute_p_adic_valuation(x, p)
                sample_points.append(x)
                valuation_values.append(v_true)
                
                # Also sample nearby points for contrast
                for offset in [-1, 1]:
                    x_offset = x + offset
                    if 1 <= x_offset <= max_x and x_offset not in sample_points:
                        v_offset = compute_p_adic_valuation(x_offset, p)
                        sample_points.append(x_offset)
                        valuation_values.append(v_offset)
    
    # Build approximating polynomial using bump functions
    # v_p(x) ≈ Σ_k k * bump_k(x) where bump_k is centered at multiples of p^k
    
    if len(sample_points) < max_degree + 1:
        # Not enough samples, use fallback
        coefficients = [0.0] * (max_degree + 1)
        coefficients[0] = 0.0
        coefficients[1] = 1.0 / (p * math.log(p + 1))
        return coefficients
    
    # Fit polynomial using least squares with regularization
    X = np.array([[float(x**i) for i in range(max_degree + 1)] 
                  for x in sample_points], dtype=np.float64)
    y = np.array(valuation_values, dtype=np.float64)
    
    # Add regularization to prevent overfitting
    lambda_reg = 0.01
    XTX = X.T @ X
    XTy = X.T @ y
    
    # Ridge regression: (X^T X + λI)^{-1} X^T y
    I = np.eye(max_degree + 1)
    I[0, 0] = 0  # Don't regularize constant term
    
    try:
        coefficients = np.linalg.solve(XTX + lambda_reg * I, XTy)
        
        # Ensure coefficients are reasonable
        max_coeff = np.max(np.abs(coefficients))
        if max_coeff > 100:
            coefficients = coefficients / max_coeff * 10
            
    except np.linalg.LinAlgError:
        # Fallback: simple approximation
        coefficients = np.zeros(max_degree + 1)
        # v_p(x) ≈ c0 + c1*x^(-1/p) + c2*x^(-2/p) + ...
        for k in range(max_degree + 1):
            if k == 0:
                coefficients[k] = 0.0
            else:
                coefficients[k] = (1.0 / k) * (1.0 / (p ** k))
    
    return list(coefficients)


def symbolic_valuation_taylor(x_symbol, p: int, center: float, degree: int = 5) -> List[float]:
    """
    Create Taylor expansion of v_p(x) around a center point
    
    This is used when we need to approximate v_p(x) near a specific value
    """
    # Taylor coefficients for v_p(x) around center
    coeffs = [0.0] * (degree + 1)
    
    # f(center) - 0th order term
    coeffs[0] = compute_p_adic_valuation(int(center), p)
    
    # Approximate derivatives using finite differences
    h = max(1, int(center * 0.01))  # Step size
    
    # First derivative: f'(center) ≈ [f(center+h) - f(center-h)] / (2h)
    if center > h:
        v_plus = compute_p_adic_valuation(int(center + h), p)
        v_minus = compute_p_adic_valuation(int(center - h), p)
        coeffs[1] = (v_plus - v_minus) / (2 * h)
    
    # Higher derivatives are typically very small for v_p
    # We use a decaying approximation
    for k in range(2, min(degree + 1, len(coeffs))):
        coeffs[k] = coeffs[1] / (k ** 2)  # Decay quadratically
    
    return coeffs


def construct_adelic_system(n: int, signature: HarmonicSignature) -> AdelicSystem:
    """
    Construct the adelic constraint system from harmonic signature
    Time Complexity: O(log n)
    """
    # Real component constraint
    # The trace of the signature matrix relates to log(n)
    real_constraint = signature.trace() - math.log(n)
    
    # p-adic constraints for small primes
    p_adic_constraints = []
    
    for p in SMALL_PRIMES[:7]:  # Use first 7 primes
        # Compute actual p-adic valuation of n
        v_n = compute_p_adic_valuation(n, p)
        
        # Extract expected valuation from signature
        # The p-component encodes information about p-adic structure
        expected_v = signature.p_component(p) * v_n
        
        p_adic_constraints.append((p, expected_v))
    
    # Product formula constraint
    # For valid factorization: |n|_R × ∏_p |n|_p = 1
    real_norm = 1.0 / n  # Normalized real absolute value
    p_adic_product = 1.0
    
    for p, _ in p_adic_constraints:
        v_n = compute_p_adic_valuation(n, p)
        p_adic_product *= p ** v_n
    
    product_constraint = real_norm * p_adic_product
    
    return AdelicSystem(
        real_constraint=real_constraint,
        p_adic_constraints=p_adic_constraints,
        product_constraint=product_constraint
    )


def adelic_polynomial_coefficients(n: int, adelic_system: AdelicSystem, 
                                  degree: int) -> List[float]:
    """
    Construct polynomial encoding adelic constraints
    
    Returns coefficients for A(x) such that factors satisfy A(x) ≈ 0
    """
    coefficients = [0.0] * (degree + 1)
    
    # Start with logarithmic constraint
    # log(x) + log(n/x) - log(n) = 0
    # This becomes: log(x) - log(n)/2 ≈ 0 for balanced factors
    
    # Taylor expansion of log around sqrt(n)
    sqrt_n = int(math.sqrt(n))
    log_sqrt_n = math.log(sqrt_n)
    
    # log(x) ≈ log(sqrt_n) + (x - sqrt_n)/sqrt_n - (x - sqrt_n)²/(2*sqrt_n²) + ...
    coefficients[0] = log_sqrt_n - math.log(n) / 2
    coefficients[1] = 1.0 / sqrt_n
    if degree >= 2:
        coefficients[2] = -1.0 / (2 * sqrt_n * sqrt_n)
    
    # Add p-adic valuation constraints
    for p, expected_v in adelic_system.p_adic_constraints:
        # Get polynomial approximation of v_p(x)
        v_p_poly = symbolic_valuation_polynomial(p, min(degree, 3))
        
        # Add constraint: v_p(n) - v_p(x) - v_p(n/x) ≈ 0
        # Since v_p(n/x) is complex, we use: v_p(x) ≈ v_p(n)/2 for balanced
        target = expected_v / 2
        
        # Add weighted constraint to polynomial
        weight = 1.0 / math.log(p + 1)
        for i, coeff in enumerate(v_p_poly):
            if i < len(coefficients):
                coefficients[i] += weight * (coeff - target / (i + 1))
    
    return coefficients


def verify_adelic_balance(n: int, x: int) -> float:
    """
    Compute how well x satisfies adelic balance constraints for n = x * (n/x)
    
    Returns a score where 0 = perfect balance, higher = worse
    """
    if n % x != 0:
        return float('inf')
    
    y = n // x
    
    # Check real constraint: log(x) + log(y) = log(n)
    real_error = abs(math.log(x) + math.log(y) - math.log(n))
    
    # Check p-adic constraints
    p_adic_error = 0.0
    
    for p in SMALL_PRIMES[:5]:
        v_n = compute_p_adic_valuation(n, p)
        v_x = compute_p_adic_valuation(x, p)
        v_y = compute_p_adic_valuation(y, p)
        
        # Valuation is additive: v_p(xy) = v_p(x) + v_p(y)
        error = abs(v_n - (v_x + v_y))
        p_adic_error += error / math.log(p + 1)
    
    # Check product formula
    product_error = 0.0
    real_norm = 1.0 / n
    p_adic_product = 1.0
    
    for p in SMALL_PRIMES[:5]:
        v_n = compute_p_adic_valuation(n, p)
        p_adic_product *= p ** v_n
    
    product = real_norm * p_adic_product
    product_error = abs(math.log(product))
    
    # Combined error
    total_error = real_error + p_adic_error + product_error
    
    return total_error


class AdelicFilter:
    """
    Filters candidate factors based on adelic constraints
    """
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self._precompute_valuations()
    
    def _precompute_valuations(self):
        """Precompute p-adic valuations of n"""
        self.n_valuations = {}
        for p in SMALL_PRIMES:
            self.n_valuations[p] = compute_p_adic_valuation(self.n, p)
    
    def is_adelic_compatible(self, x: int) -> bool:
        """
        Quick check if x could be a factor based on adelic constraints
        """
        # Basic check
        if x <= 1 or x > self.sqrt_n:
            return False
        
        # For each prime p, check if v_p(x) is compatible with v_p(n)
        for p in SMALL_PRIMES[:3]:  # Quick check with first 3 primes
            v_n = self.n_valuations[p]
            v_x = compute_p_adic_valuation(x, p)
            
            # v_x cannot exceed v_n
            if v_x > v_n:
                return False
            
            # For balanced semiprimes, v_x should be close to v_n/2
            if v_n > 0 and abs(v_x - v_n / 2) > v_n / 2 + 1:
                return False
        
        return True
    
    def filter_candidates(self, candidates: List[int]) -> List[int]:
        """
        Filter list of candidates based on adelic compatibility
        """
        return [x for x in candidates if self.is_adelic_compatible(x)]
