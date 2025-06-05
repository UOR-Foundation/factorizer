"""
Fibonacci Core - Constants and fundamental Fibonacci operations
Implements golden ratio mathematics and Fibonacci generation
"""

import math
from typing import Tuple

# Golden ratio constants
SQRT5 = math.sqrt(5)
PHI = (1 + SQRT5) / 2  # Golden ratio φ ≈ 1.618...
PSI = (1 - SQRT5) / 2  # Conjugate ψ = 1 - φ ≈ -0.618...
GOLDEN_ANGLE = 2 * math.pi * (PHI - 1)  # ≈ 2.399... radians

def fib(k: int) -> int:
    """
    Generate kth Fibonacci number using fast doubling algorithm
    
    This is a deterministic O(log k) algorithm that uses the identities:
    F(2k) = F(k) * (2*F(k+1) - F(k))
    F(2k+1) = F(k)² + F(k+1)²
    
    Args:
        k: Index of Fibonacci number (0-indexed)
        
    Returns:
        kth Fibonacci number
        
    Raises:
        ValueError: If k is negative
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    
    def _double(i: int) -> Tuple[int, int]:
        """
        Helper function that returns (F(i), F(i+1))
        Uses fast doubling to achieve O(log k) complexity
        """
        if i == 0:
            return (0, 1)
        
        # Get F(i//2) and F(i//2 + 1)
        a, b = _double(i >> 1)
        
        # Calculate F(i) and F(i+1) based on whether i is even or odd
        c = a * (2 * b - a)  # F(2k)
        d = a * a + b * b    # F(2k+1)
        
        if i & 1:  # i is odd
            return (d, c + d)
        else:      # i is even
            return (c, d)
    
    return _double(k)[0]

def fib_wave(x: float) -> float:
    """
    Continuous extension of Fibonacci sequence using Binet's formula
    
    Uses the closed-form expression:
    fib_wave(x) = (φ^x - ψ^x) / √5
    
    This extends Fibonacci numbers to real values, creating a smooth wave
    that passes through all integer Fibonacci values.
    
    Args:
        x: Real-valued position
        
    Returns:
        Continuous Fibonacci value at position x
    """
    # Binet's formula for continuous Fibonacci
    phi_term = PHI ** x
    psi_term = PSI ** x
    
    # For non-integer x, psi_term might be complex
    # Take the real part of the result
    result = (phi_term - psi_term) / SQRT5
    
    # If result is complex, return its real part
    if isinstance(result, complex):
        return result.real
    return result

def lucas(k: int) -> int:
    """
    Generate kth Lucas number (related to Fibonacci)
    
    Lucas numbers follow the same recurrence as Fibonacci
    but with different initial values: L(0)=2, L(1)=1
    
    Args:
        k: Index of Lucas number
        
    Returns:
        kth Lucas number
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    
    if k == 0:
        return 2
    elif k == 1:
        return 1
    
    # Use the relationship: L(n) = F(n-1) + F(n+1)
    return fib(k - 1) + fib(k + 1)

def is_fibonacci(n: int) -> bool:
    """
    Check if a number is a Fibonacci number
    
    Uses the property that n is Fibonacci if and only if
    one of 5n² + 4 or 5n² - 4 is a perfect square
    
    Args:
        n: Number to check
        
    Returns:
        True if n is a Fibonacci number
    """
    if n < 0:
        return False
    
    # Check if 5n² + 4 or 5n² - 4 is a perfect square
    discriminant1 = 5 * n * n + 4
    discriminant2 = 5 * n * n - 4
    
    def is_perfect_square(x: int) -> bool:
        """Check if x is a perfect square"""
        if x < 0:
            return False
        root = int(math.isqrt(x))
        return root * root == x
    
    return is_perfect_square(discriminant1) or is_perfect_square(discriminant2)

def nearest_fibonacci_index(n: int) -> int:
    """
    Find the index k such that fib(k) is closest to n
    
    Uses the inverse Binet formula to estimate the index
    
    Args:
        n: Target number
        
    Returns:
        Index of nearest Fibonacci number
    """
    if n <= 0:
        return 0
    
    # Use inverse Binet formula: k ≈ log_φ(n * √5)
    k_estimate = math.log(n * SQRT5) / math.log(PHI)
    k = round(k_estimate)
    
    # Fine-tune by checking neighbors
    fib_k = fib(k)
    if fib_k > n and k > 0:
        fib_k_minus = fib(k - 1)
        if n - fib_k_minus < fib_k - n:
            k -= 1
    elif fib_k < n:
        fib_k_plus = fib(k + 1)
        if fib_k_plus - n < n - fib_k:
            k += 1
    
    return k
