"""
Adaptive Observer - Multi-scale observation with quantum superposition
Implements observer effect through coherence-driven measurement
"""

import math
from typing import List, Dict, Tuple, Set

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom1 import primes_up_to
from axiom2 import fib, PHI, GOLDEN_ANGLE
from axiom3 import coherence, accelerated_coherence

class MultiScaleObserver:
    """
    Observes at multiple scales to create coherence fields
    Scales: μ (micro), m (meso), M (macro), Ω (omega)
    """
    
    def __init__(self, n: int):
        """
        Initialize multi-scale observer for number n
        
        Args:
            n: The number being factored
        """
        self.n = n
        self.root = int(math.isqrt(n))
        
        # Define observation scales based on axioms
        self.scales = {
            "μ": 1,  # Micro scale
            "m": max(1, int(math.log(self.root, PHI))),  # Meso scale  
            "M": max(1, int(self.root / PHI)),  # Macro scale
            "Ω": max(1, fib(max(1, int(math.log2(self.root)))))  # Omega scale
        }
        
    def coherence_at_scale(self, x: int, scale: int) -> float:
        """
        Calculate coherence at position x using given scale
        
        Args:
            x: Position to observe
            scale: Observation scale
            
        Returns:
            Average coherence in scale window
        """
        if x < 2 or x > self.root:
            return 0.0
            
        # Window size based on scale
        window = max(1, scale // 5)
        coherence_sum = 0.0
        count = 0
        
        # Sample coherence in window around x
        for offset in range(-scale, scale + 1, window):
            pos = x + offset
            if 2 <= pos <= self.root:
                # Check if pos divides n
                if self.n % pos == 0:
                    coh = accelerated_coherence(pos, self.n // pos, self.n)
                else:
                    # Use pos as potential factor
                    coh = accelerated_coherence(pos, pos, self.n)
                coherence_sum += coh
                count += 1
        
        return coherence_sum / count if count > 0 else 0.0
    
    def observe(self, x: int) -> float:
        """
        Perform multi-scale observation at position x
        
        Args:
            x: Position to observe
            
        Returns:
            Weighted coherence across all scales
        """
        total_coherence = 0.0
        
        for scale_name, scale_value in self.scales.items():
            # Calculate coherence at this scale
            scale_coherence = self.coherence_at_scale(x, scale_value)
            
            # Weight by inverse log of scale
            weight = 1 / (1 + math.log(max(1, scale_value)))
            total_coherence += weight * scale_coherence
        
        return total_coherence
    
    def coherence_field(self, positions: List[int]) -> Dict[int, float]:
        """
        Create coherence field for given positions
        
        Args:
            positions: List of positions to observe
            
        Returns:
            Dictionary mapping position to coherence
        """
        field = {}
        for pos in positions:
            field[pos] = self.observe(pos)
        return field

def generate_superposition(n: int, hints: List[int] = None) -> List[int]:
    """
    Generate quantum superposition of candidate positions
    
    Combines multiple sources:
    - Hints (if provided)
    - Fibonacci positions
    - Square root neighborhood
    - Golden spiral positions
    
    Args:
        n: Number being factored
        hints: Optional hint positions
        
    Returns:
        List of superposition positions
    """
    root = int(math.isqrt(n))
    superposition = set()
    
    # Add hints if provided
    if hints:
        superposition.update(h for h in hints if 2 <= h <= root)
    
    # Add Fibonacci positions
    k = 1
    while True:
        f = fib(k)
        if f > root:
            break
        if f >= 2:
            superposition.add(f)
            # Also add golden ratio scaled positions
            golden_pos = int(f * PHI)
            if 2 <= golden_pos <= root:
                superposition.add(golden_pos)
        k += 1
    
    # Add sqrt neighborhood
    sqrt_range = max(10, int(root * 0.1))
    for offset in range(-sqrt_range, sqrt_range + 1):
        pos = root + offset
        if 2 <= pos <= root:
            superposition.add(pos)
    
    # Add golden spiral positions
    angle = 0
    for i in range(1, min(50, root // 5)):
        radius = int(root * i / 50)
        x_pos = int(root // 2 + radius * math.cos(angle))
        if 2 <= x_pos <= root:
            superposition.add(x_pos)
        angle += GOLDEN_ANGLE
    
    return sorted(list(superposition))

def collapse_wavefunction(n: int, candidates: List[int], 
                         observer: MultiScaleObserver, 
                         iterations: int = 5) -> List[Tuple[int, float]]:
    """
    Collapse quantum superposition through iterative observation
    
    Uses coherence gradients to refine candidates
    
    Args:
        n: Number being factored
        candidates: Initial superposition
        observer: Multi-scale observer
        iterations: Number of collapse iterations
        
    Returns:
        List of (position, weight) tuples sorted by weight
    """
    root = int(math.isqrt(n))
    
    # Initialize weights
    weighted_candidates = [(x, observer.observe(x)) for x in candidates]
    
    for iteration in range(iterations):
        new_candidates = []
        
        for x, weight in weighted_candidates:
            # Calculate coherence gradient
            delta = max(1, int(root * 0.01))
            
            # Forward difference
            if x + delta <= root:
                coh_plus = observer.observe(x + delta)
            else:
                coh_plus = weight
            
            # Backward difference
            if x - delta >= 2:
                coh_minus = observer.observe(x - delta)
            else:
                coh_minus = weight
            
            # Gradient
            gradient = (coh_plus - coh_minus) / (2 * delta)
            
            # Move in gradient direction
            step_size = max(1, int(root * 0.02 / (iteration + 1)))
            if gradient > 0:
                new_x = min(root, x + step_size)
            elif gradient < 0:
                new_x = max(2, x - step_size)
            else:
                new_x = x
            
            # Calculate new weight
            new_coherence = observer.observe(new_x)
            new_weight = new_coherence * (1 + abs(gradient))
            
            new_candidates.append((new_x, new_weight))
        
        # Keep top candidates
        new_candidates.sort(key=lambda t: -t[1])
        weighted_candidates = new_candidates[:max(20, len(candidates) // 2)]
        
        # Add some exploration
        if iteration < iterations - 1:
            # Add positions with high gradients
            gradient_positions = []
            for x, _ in weighted_candidates[:10]:
                for offset in [-1, 1]:
                    test_x = x + offset * step_size
                    if 2 <= test_x <= root:
                        gradient_positions.append((test_x, observer.observe(test_x)))
            
            # Merge with existing candidates
            all_positions = weighted_candidates + gradient_positions
            # Remove duplicates
            seen = set()
            unique = []
            for x, w in all_positions:
                if x not in seen:
                    seen.add(x)
                    unique.append((x, w))
            weighted_candidates = sorted(unique, key=lambda t: -t[1])[:len(candidates)]
    
    return weighted_candidates

def create_coherence_gradient_field(n: int, observer: MultiScaleObserver,
                                   center: int, radius: int) -> Dict[int, float]:
    """
    Create gradient field around a center position
    
    Args:
        n: Number being factored
        observer: Multi-scale observer
        center: Center position
        radius: Field radius
        
    Returns:
        Dictionary mapping position to gradient
    """
    root = int(math.isqrt(n))
    field = {}
    
    for offset in range(-radius, radius + 1):
        x = center + offset
        if 2 <= x <= root:
            # Calculate gradient
            delta = 1
            
            if x + delta <= root:
                coh_plus = observer.observe(x + delta)
            else:
                coh_plus = observer.observe(x)
                
            if x - delta >= 2:
                coh_minus = observer.observe(x - delta)
            else:
                coh_minus = observer.observe(x)
            
            gradient = (coh_plus - coh_minus) / (2 * delta)
            field[x] = gradient
    
    return field
