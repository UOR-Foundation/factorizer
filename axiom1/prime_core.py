"""
Prime Core - Basic prime operations for Axiom 1
Implements primality testing and prime generation
"""

from typing import List

# Constants
SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)

def is_prime(n: int) -> bool:
    """
    Deterministic Miller-Rabin primality test
    Pure axiom implementation - no randomization
    """
    if n < 2:
        return False
    
    # Quick check against small primes
    for p in SMALL_PRIMES:
        if n % p == 0:
            return n == p
    
    # Miller-Rabin with deterministic witnesses
    d, s = n - 1, 0
    while d & 1 == 0:
        d //= 2
        s += 1
    
    # These witnesses guarantee deterministic results for all 64-bit integers
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        a %= n
        if a == 0:
            continue
        
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    
    return True

def primes_up_to(limit: int) -> List[int]:
    """
    Generate all primes up to limit using Sieve of Eratosthenes
    Pure mathematical generation - no hardcoding
    """
    if limit < 2:
        return []
    
    # Initialize sieve
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    
    # Sieve process
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = b"\x00" * len(range(p * p, limit + 1, p))
    
    # Extract primes
    return [i for i, f in enumerate(sieve) if f]
