"""
Spectral Core - Multiple spectral representations of numbers
Implements binary, modular, digital, and harmonic spectra
"""

import math
import itertools
from typing import List

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom1 import primes_up_to
from axiom2 import PHI, GOLDEN_ANGLE, fib_wave

def binary_spectrum(n: int) -> List[float]:
    """
    Analyze binary representation patterns
    
    Computes:
    - Bit density (ratio of 1s to total bits)
    - Autocorrelation of bit sequence
    - Run lengths of consecutive bits
    
    Args:
        n: Number to analyze
        
    Returns:
        List of spectral features from binary representation
    """
    if n <= 0:
        return [0.0, 0.0]
    
    # Get binary representation
    bits = bin(n)[2:]  # Remove '0b' prefix
    length = len(bits)
    
    # Bit density
    ones_count = bits.count('1')
    density = ones_count / length
    
    # Convert to +1/-1 sequence for autocorrelation
    sequence = [1 if b == '1' else -1 for b in bits]
    mean = sum(sequence) / length
    
    # Compute autocorrelation at lag 1
    if length > 1:
        autocorr = sum((sequence[i] - mean) * (sequence[i + 1] - mean) 
                      for i in range(length - 1)) / (length - 1)
        # Normalize
        variance = sum((x - mean) ** 2 for x in sequence) / length
        autocorr = autocorr / variance if variance > 0 else 0
    else:
        autocorr = 0
    
    # Run lengths (normalized by total length)
    runs = []
    for bit, group in itertools.groupby(bits):
        run_length = len(list(group)) / length
        runs.append(run_length)
    
    # Return density, autocorrelation, and first 10 run lengths
    spectrum = [density, autocorr] + runs[:10]
    
    # Pad with zeros if needed
    while len(spectrum) < 12:
        spectrum.append(0.0)
    
    return spectrum

def modular_spectrum(n: int, k: int = 10) -> List[float]:
    """
    Analyze residue patterns across prime moduli
    
    Computes n mod p for the first k primes (excluding 2)
    and normalizes by the prime value
    
    Args:
        n: Number to analyze
        k: Number of primes to use
        
    Returns:
        List of normalized modular residues
    """
    if n <= 0:
        return [0.0] * k
    
    # Get enough primes, skipping 2 for better patterns
    all_primes = primes_up_to(max(50, 3 * k))
    primes = all_primes[1:]  # Skip 2
    
    # Compute normalized residues
    spectrum = []
    for i, p in enumerate(primes):
        if i >= k:
            break
        residue = n % p
        normalized = residue / p  # Range [0, 1)
        spectrum.append(normalized)
    
    # Ensure we have exactly k values
    while len(spectrum) < k:
        spectrum.append(0.0)
    
    return spectrum

def digital_spectrum(n: int) -> List[float]:
    """
    Analyze decimal digit patterns
    
    Computes:
    - Normalized digit sum
    - Digital root (iterative digit sum until single digit)
    
    Args:
        n: Number to analyze
        
    Returns:
        List containing normalized digit sum and digital root
    """
    if n <= 0:
        return [0.0, 0.0]
    
    # Get digits
    digits = [int(d) for d in str(n)]
    
    # Digit sum normalized by number of digits × 9
    digit_sum = sum(digits)
    normalized_sum = digit_sum / (len(digits) * 9)
    
    # Digital root
    root = digit_sum
    while root >= 10:
        root = sum(int(d) for d in str(root))
    
    # Special case: digital root 0 only for n=0
    if root == 0 and n != 0:
        root = 9
    
    normalized_root = root / 9
    
    return [normalized_sum, normalized_root]

def harmonic_spectrum(n: int) -> List[float]:
    """
    Analyze golden ratio phase relationships
    
    Computes:
    - Fibonacci wave phase at log_φ(n)
    - Distance to nearest Fibonacci number (normalized)
    - Golden angle offset
    
    Args:
        n: Number to analyze
        
    Returns:
        List of harmonic spectral features
    """
    if n <= 1:
        return [0.0, 0.0, 0.0]
    
    # Position in Fibonacci space
    x = math.log(max(n, 1)) / math.log(PHI)
    
    # Fibonacci wave value at this position
    wave_value = fib_wave(x)
    
    # Phase (normalized to [0, 1])
    # Since fib_wave can be negative, we normalize differently
    phase = abs(wave_value) % 1 if not isinstance(wave_value, complex) else abs(wave_value.real) % 1
    
    # Find nearest Fibonacci number
    from axiom2 import fib
    k = round(x)
    nearest_fib = fib(max(0, k))
    
    # Normalized distance to nearest Fibonacci
    distance = abs(n - nearest_fib)
    normalized_distance = distance / (n + 1)  # Avoid division by zero
    
    # Ratio to nearest Fibonacci
    ratio = math.log(nearest_fib + 1) / math.log(n + 1) if nearest_fib > 0 else 0
    
    # Golden angle offset
    offset = (n * GOLDEN_ANGLE / (2 * math.pi)) % 1
    
    return [phase, ratio, offset]

def spectral_vector(n: int) -> List[float]:
    """
    Combine all spectral representations into a single vector
    
    This creates a comprehensive spectral signature by concatenating:
    - Binary spectrum (12 features)
    - Modular spectrum (10 features)
    - Digital spectrum (2 features)
    - Harmonic spectrum (3 features)
    
    Args:
        n: Number to analyze
        
    Returns:
        Combined spectral vector
    """
    return (binary_spectrum(n) + 
            modular_spectrum(n) + 
            digital_spectrum(n) + 
            harmonic_spectrum(n))
