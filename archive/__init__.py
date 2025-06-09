"""
UOR/Prime Axioms Factorizer Package

A pure mathematical implementation of integer factorization based on the Universal Object Reference (UOR)
and Prime Model axioms.
"""

from .factorizer import Factorizer, FactorizationResult, factorize, get_factorizer

__version__ = "1.0.0"
__all__ = ["Factorizer", "FactorizationResult", "factorize", "get_factorizer"]
