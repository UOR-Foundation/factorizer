"""
Fibonacci Vortices - Golden spiral patterns in number space
Implements vortex generation with prime modulation
"""

import math
from typing import List, Set, Optional
from .fibonacci_core import fib, PHI
from .fibonacci_resonance_map import FibonacciResonanceMap

# Import prime generation from axiom1
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom1 import primes_up_to

def fib_vortices(n: int, resonance_map: Optional[FibonacciResonanceMap] = None) -> List[int]:
    """
    Generate Fibonacci vortex points for factorization
    
    Creates a set of candidate positions based on:
    1. Fibonacci numbers up to sqrt(n)
    2. Golden ratio scaling: f×φ and f/φ
    3. Prime modulation: (f×p) mod root
    
    Args:
        n: The number being factored
        resonance_map: Optional acceleration cache
        
    Returns:
        Sorted list of vortex positions in range [2, sqrt(n)]
    """
    root = int(math.isqrt(n))
    
    # Use resonance map if provided for acceleration
    if resonance_map:
        # Get pre-computed vortex points
        vortex_points = resonance_map.get_vortex_points(root // 2, root // 2)
        
        # Also get Fibonacci numbers in range
        fibs_in_range = resonance_map.get_fibonacci_range(2, root)
        vortex_points.update(fibs_in_range)
        
        # Add prime modulated points
        for f in fibs_in_range[:10]:  # Limit for efficiency
            for p in primes_up_to(min(100, f))[:20]:
                modulated = (f * p) % root
                if modulated == 0:
                    modulated = p
                if 2 <= modulated <= root:
                    vortex_points.add(modulated)
        
        return sorted(list(vortex_points))
    
    # Original implementation without resonance map
    vortex_points: Set[int] = set()
    k = 1
    
    # Generate vortex points from Fibonacci numbers
    while k < 30:  # Limit iterations for efficiency
        f = fib(k)
        if f >= root:
            break
            
        # Add Fibonacci number itself
        if 2 <= f <= root:
            vortex_points.add(f)
        
        # Add golden ratio scaled positions
        phi_scaled = int(f * PHI)
        if 2 <= phi_scaled <= root:
            vortex_points.add(phi_scaled)
            
        psi_scaled = int(f / PHI)
        if 2 <= psi_scaled <= root:
            vortex_points.add(psi_scaled)
        
        # Prime modulation: multiply by small primes
        for p in primes_up_to(min(100, f))[:20]:  # Limit primes
            modulated = (f * p) % root
            if modulated == 0:
                modulated = p  # Avoid zero
            if 2 <= modulated <= root:
                vortex_points.add(modulated)
        
        k += 1
    
    return sorted(vortex_points)

def golden_spiral_positions(n: int, num_points: int = 20) -> List[int]:
    """
    Generate positions along a golden spiral
    
    Uses the golden angle to create evenly distributed points
    on a spiral pattern emanating from the center (sqrt(n)/2)
    
    Args:
        n: The number being factored
        num_points: Number of spiral points to generate
        
    Returns:
        List of positions along the golden spiral
    """
    root = int(math.isqrt(n))
    center = root // 2
    positions = []
    
    # Import GOLDEN_ANGLE from fibonacci_core
    from .fibonacci_core import GOLDEN_ANGLE
    
    angle = 0
    for i in range(min(num_points, root // 10)):
        # Radius increases linearly
        r = int(root * (i + 1) / num_points)
        
        # Convert polar to Cartesian
        x = int(r * math.cos(angle)) + center
        
        if 2 <= x <= root:
            positions.append(x)
        
        # Increment by golden angle for optimal spacing
        angle += GOLDEN_ANGLE
    
    return positions

def fibonacci_lattice_points(n: int) -> List[int]:
    """
    Generate lattice points based on Fibonacci relationships
    
    Creates a lattice where points are at intersections of
    Fibonacci-based grid lines
    
    Args:
        n: The number being factored
        
    Returns:
        List of Fibonacci lattice intersection points
    """
    root = int(math.isqrt(n))
    lattice_points: Set[int] = set()
    
    # Generate Fibonacci grid lines
    k = 1
    fib_lines = []
    while True:
        f = fib(k)
        if f > root:
            break
        fib_lines.append(f)
        k += 1
    
    # Find intersections and combinations
    for i, f1 in enumerate(fib_lines):
        for f2 in fib_lines[i:]:
            # Sum intersection
            point = f1 + f2
            if 2 <= point <= root:
                lattice_points.add(point)
            
            # Difference intersection
            point = abs(f1 - f2)
            if 2 <= point <= root:
                lattice_points.add(point)
            
            # Product modulo root
            point = (f1 * f2) % root
            if point == 0:
                point = min(f1, f2)
            if 2 <= point <= root:
                lattice_points.add(point)
    
    return sorted(lattice_points)
