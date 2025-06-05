"""
Axiom 1: Prime Ontology
Pure prime mathematics for UOR/Prime factorization
"""

from .prime_core import (
    SMALL_PRIMES,
    is_prime,
    primes_up_to
)

from .prime_cascade import PrimeCascade

from .prime_geodesic import PrimeGeodesic

__all__ = [
    'SMALL_PRIMES',
    'is_prime', 
    'primes_up_to',
    'PrimeCascade',
    'PrimeGeodesic'
]
