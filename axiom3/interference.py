"""
Interference - Prime×Fibonacci wave interference patterns
Analyzes interference between prime and Fibonacci waves
"""

import math
from typing import List, Tuple

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom1 import primes_up_to
from axiom2 import fib, PHI

def prime_fib_interference(n: int) -> List[float]:
    """
    Generate interference pattern from prime and Fibonacci waves
    
    Creates an interference pattern by multiplying:
    - Prime wave: sum of cosine waves at prime frequencies
    - Fibonacci wave: sum of cosine waves at Fibonacci frequencies
    
    Args:
        n: Number being analyzed
        
    Returns:
        List of interference values across the search space
    """
    root = int(math.isqrt(n))
    
    # Get primes and Fibonacci numbers for waves
    primes = primes_up_to(min(root, 100))[:10]  # Use first 10 primes
    
    # Get Fibonacci numbers up to root
    fibs = []
    k = 2
    while True:
        f = fib(k)
        if f > root:
            break
        fibs.append(f)
        k += 1
    
    # Limit to first 10 Fibonacci numbers
    fibs = fibs[:10]
    
    # Generate interference pattern
    spectrum = []
    
    for x in range(2, root + 1):
        # Prime wave component
        prime_amp = sum(math.cos(2 * math.pi * p * x / n) for p in primes)
        
        # Fibonacci wave component (scaled by golden ratio)
        fib_amp = sum(math.cos(2 * math.pi * f * x / (n * PHI)) for f in fibs)
        
        # Interference is the product
        interference = prime_amp * fib_amp
        spectrum.append(interference)
    
    return spectrum

def interference_extrema(n: int, top: int = 30) -> List[int]:
    """
    Find extrema (peaks and valleys) in the interference pattern
    
    Extrema in the interference pattern often correspond to
    factor positions.
    
    Args:
        n: Number being analyzed
        top: Number of extrema to return
        
    Returns:
        List of positions with strongest extrema
    """
    spectrum = prime_fib_interference(n)
    
    if len(spectrum) < 3:
        return []
    
    # Find all extrema (local maxima and minima)
    extrema = []
    
    for i in range(1, len(spectrum) - 1):
        # Check if local maximum or minimum
        if ((spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]) or
            (spectrum[i] < spectrum[i-1] and spectrum[i] < spectrum[i+1])):
            
            # Position is i+2 because spectrum starts at x=2
            position = i + 2
            # Weight by absolute value of extremum
            weight = abs(spectrum[i])
            extrema.append((weight, position))
    
    # Sort by weight (strongest extrema first)
    extrema.sort(reverse=True)
    
    # Return positions only
    return [pos for _, pos in extrema[:top]]

def identify_resonance_source(x: int, n: int) -> Tuple[int, int]:
    """
    Identify which prime and Fibonacci pair creates strongest resonance at position x
    
    This helps understand why a particular position has high interference.
    
    Args:
        x: Position to analyze
        n: Number being factored
        
    Returns:
        Tuple of (prime, fibonacci) that create strongest resonance
    """
    root = int(math.isqrt(n))
    
    # Get candidate primes
    primes = [p for p in primes_up_to(min(root, 100)) if x % p != 0]
    if not primes:
        primes = [2]  # Fallback
    
    # Get candidate Fibonacci numbers
    fibs = []
    k = 2
    while k < 20 and fib(k) <= root:
        fibs.append(fib(k))
        k += 1
    if not fibs:
        fibs = [2]  # Fallback
    
    # Find strongest resonance
    best_resonance = 0
    best_prime = primes[0]
    best_fib = fibs[0]
    
    for p in primes:
        for f in fibs:
            # Calculate resonance strength
            prime_component = abs(math.cos(2 * math.pi * p * x / n))
            fib_component = abs(math.cos(2 * math.pi * f * x / (n * PHI)))
            resonance = prime_component * fib_component
            
            if resonance > best_resonance:
                best_resonance = resonance
                best_prime = p
                best_fib = f
    
    return (best_prime, best_fib)

def interference_gradient(n: int, x: int, delta: int = 1) -> float:
    """
    Calculate gradient of interference pattern at position x
    
    The gradient indicates how quickly the interference is changing,
    which can help identify sharp features.
    
    Args:
        n: Number being analyzed
        x: Position to calculate gradient
        delta: Step size for finite difference
        
    Returns:
        Gradient value
    """
    root = int(math.isqrt(n))
    
    # Ensure x is in valid range
    if x - delta < 2 or x + delta > root:
        return 0.0
    
    # Get interference values
    spectrum = prime_fib_interference(n)
    
    # Map x to spectrum index (spectrum starts at x=2)
    idx = x - 2
    idx_minus = (x - delta) - 2
    idx_plus = (x + delta) - 2
    
    # Bounds check
    if idx_minus < 0 or idx_plus >= len(spectrum):
        return 0.0
    
    # Finite difference gradient
    gradient = (spectrum[idx_plus] - spectrum[idx_minus]) / (2 * delta)
    
    return gradient

def resonance_strength(p: int, q: int, n: int) -> float:
    """
    Calculate resonance strength between factors p, q and number n
    
    Measures how well the interference patterns of p and q
    combine to create the pattern of n.
    
    Args:
        p: First factor
        q: Second factor
        n: Target number
        
    Returns:
        Resonance strength (0 to 1)
    """
    # Get some primes and Fibonacci numbers
    primes = primes_up_to(30)[:5]
    fibs = [fib(k) for k in range(2, 7)]
    
    strength = 0.0
    count = 0
    
    for prime in primes:
        for f in fibs:
            # Phase for each number
            phase_p = (2 * math.pi * prime * p / n) % (2 * math.pi)
            phase_q = (2 * math.pi * prime * q / n) % (2 * math.pi)
            phase_n = (2 * math.pi * prime * n / n) % (2 * math.pi)
            
            # Coherence in prime dimension
            prime_coherence = abs(math.cos((phase_p + phase_q) / 2 - phase_n))
            
            # Phase for Fibonacci dimension
            fib_phase_p = (2 * math.pi * f * p / (n * PHI)) % (2 * math.pi)
            fib_phase_q = (2 * math.pi * f * q / (n * PHI)) % (2 * math.pi)
            fib_phase_n = (2 * math.pi * f * n / (n * PHI)) % (2 * math.pi)
            
            # Coherence in Fibonacci dimension
            fib_coherence = abs(math.cos((fib_phase_p + fib_phase_q) / 2 - fib_phase_n))
            
            # Combined resonance
            strength += prime_coherence * fib_coherence
            count += 1
    
    return strength / count if count > 0 else 0.0
