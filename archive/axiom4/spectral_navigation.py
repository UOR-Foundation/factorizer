"""
Spectral Navigation - Navigate coherence fields using gradients
Implements gradient ascent, multi-path search, and harmonic jumps
"""

import math
from typing import List, Tuple, Optional

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom2 import PHI, fib
from axiom3 import coherence
from .adaptive_observer import MultiScaleObserver

def coherence_gradient(n: int, x: int, observer: MultiScaleObserver, 
                      delta: int = 1) -> float:
    """
    Calculate coherence gradient at position x
    
    Args:
        n: Number being factored
        x: Position to calculate gradient
        observer: Multi-scale observer
        delta: Step size for finite difference
        
    Returns:
        Gradient value
    """
    root = int(math.isqrt(n))
    
    # Ensure x is in valid range
    if x < 2 or x > root:
        return 0.0
    
    # Forward difference
    if x + delta <= root:
        coh_plus = observer.observe(x + delta)
    else:
        coh_plus = observer.observe(x)
    
    # Backward difference
    if x - delta >= 2:
        coh_minus = observer.observe(x - delta)
    else:
        coh_minus = observer.observe(x)
    
    # Central difference gradient
    gradient = (coh_plus - coh_minus) / (2 * delta)
    
    return gradient

def gradient_ascent(n: int, start: int, observer: MultiScaleObserver,
                   max_steps: int = 50, tolerance: float = 1e-6) -> List[int]:
    """
    Follow coherence gradient to find local maximum
    
    Args:
        n: Number being factored
        start: Starting position
        observer: Multi-scale observer
        max_steps: Maximum steps to take
        tolerance: Convergence tolerance
        
    Returns:
        Path of positions visited
    """
    root = int(math.isqrt(n))
    path = [start]
    current = start
    
    for step in range(max_steps):
        # Calculate gradient
        grad = coherence_gradient(n, current, observer)
        
        # Check convergence
        if abs(grad) < tolerance:
            break
        
        # Adaptive step size (decreases over time)
        step_size = max(1, int(root * 0.02 / (step + 1)))
        
        # Move in gradient direction
        if grad > 0:
            next_pos = min(root, current + step_size)
        elif grad < 0:
            next_pos = max(2, current - step_size)
        else:
            break  # No gradient
        
        # Check if we're stuck
        if next_pos == current:
            break
        
        current = next_pos
        path.append(current)
        
        # Check if we found a factor
        if n % current == 0:
            break
    
    return path

def multi_path_search(n: int, starts: List[int], observer: MultiScaleObserver,
                     max_paths: int = 10) -> List[Tuple[int, float]]:
    """
    Explore multiple gradient paths in parallel
    
    Args:
        n: Number being factored
        starts: Starting positions
        observer: Multi-scale observer
        max_paths: Maximum number of paths to explore
        
    Returns:
        List of (position, coherence) for path endpoints
    """
    endpoints = []
    
    # Limit number of paths
    paths_to_explore = starts[:max_paths]
    
    for start in paths_to_explore:
        # Follow gradient from this start
        path = gradient_ascent(n, start, observer)
        
        if path:
            # Get endpoint and its coherence
            endpoint = path[-1]
            coh = observer.observe(endpoint)
            endpoints.append((endpoint, coh))
    
    # Sort by coherence
    endpoints.sort(key=lambda t: -t[1])
    
    # Remove duplicates
    seen = set()
    unique_endpoints = []
    for pos, coh in endpoints:
        if pos not in seen:
            seen.add(pos)
            unique_endpoints.append((pos, coh))
    
    return unique_endpoints

def harmonic_jump(n: int, current: int, stuck_count: int) -> int:
    """
    Make harmonic jump to escape local minima
    
    Uses golden ratio and Fibonacci numbers for jumps
    
    Args:
        n: Number being factored
        current: Current position
        stuck_count: How many times we've been stuck
        
    Returns:
        New position after jump
    """
    root = int(math.isqrt(n))
    
    # Different jump strategies based on stuck count
    jump_type = stuck_count % 4
    
    if jump_type == 0:
        # Golden ratio jump
        new_pos = int(current * PHI)
    elif jump_type == 1:
        # Inverse golden ratio jump
        new_pos = int(current / PHI)
    elif jump_type == 2:
        # Fibonacci jump
        fib_index = min(stuck_count + 3, 20)
        new_pos = current + fib(fib_index)
    else:
        # Golden angle jump
        from axiom2 import GOLDEN_ANGLE
        angle = (stuck_count * GOLDEN_ANGLE) % (2 * math.pi)
        radius = min(root // 4, stuck_count * 10)
        new_pos = int(current + radius * math.cos(angle))
    
    # Ensure within bounds
    new_pos = max(2, min(root, new_pos))
    
    return new_pos

def find_coherence_peaks(n: int, observer: MultiScaleObserver,
                        resolution: int = 100) -> List[int]:
    """
    Find peaks in the coherence landscape
    
    Args:
        n: Number being factored
        observer: Multi-scale observer
        resolution: Number of samples
        
    Returns:
        List of peak positions
    """
    root = int(math.isqrt(n))
    step = max(1, root // resolution)
    
    peaks = []
    prev_coh = 0.0
    current_coh = observer.observe(2)
    
    for x in range(3, root + 1, step):
        next_coh = observer.observe(x)
        
        # Check for local maximum
        if current_coh > prev_coh and current_coh > next_coh:
            # Refine peak position
            refined = x - step
            for offset in range(-step, step + 1):
                test_x = refined + offset
                if 2 <= test_x <= root:
                    if observer.observe(test_x) > observer.observe(refined):
                        refined = test_x
            peaks.append(refined)
        
        prev_coh = current_coh
        current_coh = next_coh
    
    return peaks

def navigate_to_factor(n: int, start: int, observer: MultiScaleObserver,
                      max_iterations: int = 100) -> Optional[int]:
    """
    Navigate coherence field to find a factor
    
    Combines gradient ascent with harmonic jumps
    
    Args:
        n: Number being factored
        start: Starting position  
        observer: Multi-scale observer
        max_iterations: Maximum iterations
        
    Returns:
        Factor if found, None otherwise
    """
    current = start
    stuck_count = 0
    visited = set()
    
    for iteration in range(max_iterations):
        # Check if current is a factor
        if n % current == 0 and current > 1:
            return current
        
        # Mark as visited
        visited.add(current)
        
        # Try gradient ascent
        path = gradient_ascent(n, current, observer, max_steps=10)
        
        if path:
            new_pos = path[-1]
            
            # Check if we found a factor
            if n % new_pos == 0 and new_pos > 1:
                return new_pos
            
            # Check if we're stuck
            if new_pos == current or new_pos in visited:
                stuck_count += 1
                # Make harmonic jump
                new_pos = harmonic_jump(n, current, stuck_count)
            else:
                stuck_count = 0
            
            current = new_pos
        else:
            # No gradient, make jump
            stuck_count += 1
            current = harmonic_jump(n, current, stuck_count)
    
    return None

def coherence_flow_lines(n: int, observer: MultiScaleObserver,
                        num_lines: int = 20) -> List[List[int]]:
    """
    Generate flow lines following coherence gradient
    
    Args:
        n: Number being factored
        observer: Multi-scale observer
        num_lines: Number of flow lines to generate
        
    Returns:
        List of flow line paths
    """
    root = int(math.isqrt(n))
    flow_lines = []
    
    # Generate starting points using golden angle
    from axiom2 import GOLDEN_ANGLE
    angle = 0
    
    for i in range(num_lines):
        # Starting point on a spiral
        radius = root * (i + 1) / (num_lines + 1)
        x = int(root // 2 + radius * math.cos(angle))
        y = int(root // 2 + radius * math.sin(angle))
        
        # Map to 1D position
        start = max(2, min(root, x))
        
        # Follow gradient from this point
        path = gradient_ascent(n, start, observer)
        if path and len(path) > 1:
            flow_lines.append(path)
        
        angle += GOLDEN_ANGLE
    
    return flow_lines
