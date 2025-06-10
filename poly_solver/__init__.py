"""
Prime Polynomial-Time Solver (PPTS)

A polynomial-time factorization algorithm based on harmonic analysis,
adelic constraints, and Lie algebra deformations.
"""

from .ppts import PPTS, factor_polynomial_time
from .harmonic import HarmonicSignature, MultiScaleResonance, extract_harmonic_signature
from .adelic import (
    AdelicSystem, AdelicConstraints, AdelicFilter,
    construct_adelic_system, verify_adelic_balance,
    compute_p_adic_valuation, symbolic_valuation_polynomial
)
from .polynomial import (
    PolynomialSystem, PolynomialSolver, Polynomial,
    construct_polynomial_system, solve_polynomial_system
)
from .lie_algebra import (
    E6E7Deformation, DeformationMatrix,
    construct_lie_algebra_polynomial, verify_lie_algebra_constraint
)
from .resultant import (
    PanAlgorithm, BivariatePolynomial,
    isolate_real_roots, refine_root_bisection,
    sturm_sequence, count_real_roots_interval
)

__all__ = [
    # Main API
    'PPTS',
    'factor_polynomial_time',
    
    # Harmonic Analysis
    'HarmonicSignature',
    'MultiScaleResonance',
    'extract_harmonic_signature',
    
    # Adelic Constraints
    'AdelicSystem',
    'AdelicConstraints',
    'AdelicFilter',
    'construct_adelic_system',
    'verify_adelic_balance',
    'compute_p_adic_valuation',
    'symbolic_valuation_polynomial',
    
    # Polynomial System
    'PolynomialSystem',
    'PolynomialSolver',
    'Polynomial',
    'construct_polynomial_system',
    'solve_polynomial_system',
    
    # Lie Algebra
    'E6E7Deformation',
    'DeformationMatrix',
    'construct_lie_algebra_polynomial',
    'verify_lie_algebra_constraint',
    
    # Advanced Root Finding
    'PanAlgorithm',
    'BivariatePolynomial',
    'isolate_real_roots',
    'refine_root_bisection',
    'sturm_sequence',
    'count_real_roots_interval'
]

__version__ = '0.2.0'
