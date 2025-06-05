"""
Fibonacci Entanglement - Detect when both factors are near Fibonacci numbers
Implements entanglement strength based on Fibonacci proximity
"""

import math
from typing import List, Tuple
from .fibonacci_core import fib, PHI

class FibonacciEntanglement:
    """
    Detect and measure entanglement when both factors of a semiprime
    are close to Fibonacci numbers
    """
    
    def __init__(self, n: int):
        """
        Initialize entanglement detector for number n
        
        Args:
            n: The number being factored
        """
        self.n = n
        self.root = int(math.isqrt(n))
        
    def detect_double(self) -> List[Tuple[int, int, float]]:
        """
        Detect cases where both p and q are near Fibonacci numbers
        
        Returns:
            List of (p, q, strength) tuples where:
            - p, q are potential factors
            - strength is the entanglement strength (0 to 1)
        """
        candidates = []
        k = 1
        
        # Check Fibonacci numbers up to sqrt(n)
        while k < 30 and fib(k) < self.root:
            base_fib = fib(k)
            
            # Check small deviations from Fibonacci number
            for delta in range(-5, 6):
                p = base_fib + delta
                
                # Check if p is a valid candidate
                if p > 1 and self.n % p == 0:
                    q = self.n // p
                    
                    # Measure how close q is to a Fibonacci number
                    min_distance = self._min_fibonacci_distance(q)
                    
                    # Calculate entanglement strength
                    # Strength decreases with distance from Fibonacci
                    if min_distance < 0.1 * q:  # Within 10% of q
                        strength = 1 / (1 + min_distance / q)
                        candidates.append((p, q, strength))
            
            k += 1
        
        # Sort by strength (highest first) and limit results
        candidates.sort(key=lambda x: -x[2])
        return candidates[:10]
    
    def _min_fibonacci_distance(self, n: int) -> int:
        """
        Find minimum distance from n to any Fibonacci number
        
        Args:
            n: Number to check
            
        Returns:
            Minimum absolute distance to a Fibonacci number
        """
        if n <= 0:
            return n
        
        # Use logarithmic search to find nearby Fibonacci numbers
        k_estimate = int(math.log(n * math.sqrt(5)) / math.log(PHI))
        
        min_dist = n  # Initialize with maximum possible
        
        # Check a range around the estimate
        for k in range(max(1, k_estimate - 2), min(100, k_estimate + 5)):
            f = fib(k)
            dist = abs(n - f)
            if dist < min_dist:
                min_dist = dist
            
            # If we've passed n, no need to check further
            if f > n + min_dist:
                break
        
        return min_dist
    
    def fibonacci_alignment_score(self, p: int, q: int) -> float:
        """
        Calculate how well aligned p and q are with Fibonacci sequence
        
        Score is higher when both numbers are close to Fibonacci numbers
        
        Args:
            p: First factor
            q: Second factor
            
        Returns:
            Alignment score (0 to 1)
        """
        dist_p = self._min_fibonacci_distance(p)
        dist_q = self._min_fibonacci_distance(q)
        
        # Calculate individual proximity scores
        score_p = 1 / (1 + dist_p / max(p, 1))
        score_q = 1 / (1 + dist_q / max(q, 1))
        
        # Combined score (geometric mean for balance)
        return math.sqrt(score_p * score_q)
    
    def golden_ratio_alignment(self, p: int, q: int) -> float:
        """
        Check if p/q or q/p is close to golden ratio
        
        Args:
            p: First factor
            q: Second factor
            
        Returns:
            Alignment score based on golden ratio proximity
        """
        if p == 0 or q == 0:
            return 0
        
        ratio1 = p / q
        ratio2 = q / p
        
        # Check proximity to φ or 1/φ
        dist1 = min(abs(ratio1 - PHI), abs(ratio1 - 1/PHI))
        dist2 = min(abs(ratio2 - PHI), abs(ratio2 - 1/PHI))
        
        min_dist = min(dist1, dist2)
        
        # Score decreases with distance from golden ratio
        return math.exp(-min_dist)
    
    def lucas_fibonacci_relation(self, p: int, q: int) -> bool:
        """
        Check if p and q follow Lucas-Fibonacci relationships
        
        Some semiprimes have factors that follow:
        - p = F(k), q = L(k) (Fibonacci and Lucas numbers)
        - p = F(k), q = F(k+1) (consecutive Fibonacci)
        
        Args:
            p: First factor
            q: Second factor
            
        Returns:
            True if a special relationship is detected
        """
        # Import lucas function
        from .fibonacci_core import lucas
        
        # Check consecutive Fibonacci
        k = 1
        while k < 50:
            f_k = fib(k)
            f_k_plus = fib(k + 1)
            
            if (p == f_k and q == f_k_plus) or (p == f_k_plus and q == f_k):
                return True
            
            # Check Fibonacci-Lucas pairs
            l_k = lucas(k)
            if (p == f_k and q == l_k) or (p == l_k and q == f_k):
                return True
            
            if f_k > max(p, q):
                break
            
            k += 1
        
        return False
