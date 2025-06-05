"""
Prime Cascade - Following prime relationships through number space
Implements twin primes, Sophie Germain chains, and cascading search
"""

from typing import List
from .prime_core import is_prime

class PrimeCascade:
    """
    Follows prime relationships through cascading patterns
    - Twin primes: p±2
    - Sophie Germain chains: 2p+1
    """
    
    def __init__(self, n: int):
        """
        Initialize cascade with target number n
        
        Args:
            n: The number being factored
        """
        self.n = n
    
    def cascade(self, p: int) -> List[int]:
        """
        Generate cascade of related primes from starting prime p
        
        Follows several prime relationships:
        1. Twin primes: p+2 and p-2
        2. Sophie Germain chain: p → 2p+1 → 2(2p+1)+1 → ...
        
        Args:
            p: Starting prime
            
        Returns:
            List of related primes in the cascade
        """
        out = []
        
        # Twin prime detection
        if is_prime(p + 2):
            out.append(p + 2)
        if p > 2 and is_prime(p - 2):
            out.append(p - 2)
        
        # Sophie Germain chain
        sophie = 2 * p + 1
        if sophie < self.n and is_prime(sophie):
            out.append(sophie)
            
            # Follow the chain
            q = sophie
            while q < self.n and len(out) < 10:  # Limit cascade length
                next_q = 2 * q + 1
                if next_q < self.n and is_prime(next_q):
                    out.append(next_q)
                    q = next_q
                else:
                    break
        
        return out
