"""
Quantum Tools - Quantum tunneling, harmonic amplification, and spectral folding
Tools for escaping local minima and generating new candidates
"""

import math
from typing import List, Set

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom1 import primes_up_to
from axiom2 import fib, PHI

class QuantumTunnel:
    """
    Quantum tunneling to escape blocked positions
    Uses Fibonacci and prime jumps
    """
    
    def __init__(self, n: int):
        """
        Initialize quantum tunnel for number n
        
        Args:
            n: Number being factored
        """
        self.n = n
        self.root = int(math.isqrt(n))
    
    def exit(self, blocked: int, width: int = 60) -> int:
        """
        Find exit point from blocked position
        
        Uses Fibonacci and prime positions beyond blocked region
        
        Args:
            blocked: Blocked position
            width: Tunnel width
            
        Returns:
            Exit position
        """
        target = min(self.root, blocked + width)
        candidates = []
        
        # Find Fibonacci numbers beyond target
        k = 1
        while k < 30:
            f = fib(k)
            if f > target and f <= self.root:
                candidates.append(f)
            k += 1
            if f > self.root:
                break
        
        # Find primes beyond target
        primes = primes_up_to(min(target + 100, self.root))
        for p in primes:
            if p > target:
                candidates.append(p)
        
        # Return closest candidate
        if candidates:
            return min(candidates)
        else:
            # No good exit, jump by width
            return min(self.root, blocked + width)
    
    def tunnel_sequence(self, start: int, max_tunnels: int = 5) -> List[int]:
        """
        Generate sequence of tunnel exits
        
        Args:
            start: Starting position
            max_tunnels: Maximum tunnels to generate
            
        Returns:
            List of tunnel exit positions
        """
        sequence = []
        current = start
        
        for i in range(max_tunnels):
            # Increasing tunnel width
            width = 30 * (i + 1)
            exit_pos = self.exit(current, width)
            
            if exit_pos == current:
                # Can't tunnel further
                break
            
            sequence.append(exit_pos)
            current = exit_pos
            
            # Stop if we've covered enough ground
            if current > self.root * 0.9:
                break
        
        return sequence

def harmonic_amplify(n: int, x: int) -> List[int]:
    """
    Generate harmonic positions from base position
    
    Creates harmonics at:
    - Integer multiples: k×x
    - Golden ratio scaling: φ×x
    - Prime scaling: p×x
    
    Args:
        n: Number being factored
        x: Base position
        
    Returns:
        List of harmonic positions
    """
    root = int(math.isqrt(n))
    harmonics = set()
    
    # Integer harmonics (k×x)
    for k in range(2, min(10, root // x + 1)):
        hx = k * x
        if hx <= root:
            harmonics.add(hx)
        else:
            break
    
    # Modular harmonics
    for k in range(2, min(10, root // x + 1)):
        hx = (k * x) % root
        if hx == 0:
            hx = k
        if 2 <= hx <= root:
            harmonics.add(hx)
    
    # Golden ratio harmonic
    phi_harmonic = int(x * PHI)
    if 2 <= phi_harmonic <= root:
        harmonics.add(phi_harmonic)
    
    # Inverse golden harmonic
    psi_harmonic = int(x / PHI)
    if 2 <= psi_harmonic <= root:
        harmonics.add(psi_harmonic)
    
    # Prime harmonics
    primes = primes_up_to(min(20, root // x))
    for p in primes:
        px = x * p
        if px <= root:
            harmonics.add(px)
        
        # Also modular version
        px_mod = (x * p) % root
        if px_mod == 0:
            px_mod = p
        if 2 <= px_mod <= root:
            harmonics.add(px_mod)
    
    return sorted(list(harmonics))

class SpectralFolder:
    """
    Generate folding points from binary and periodic patterns
    """
    
    def __init__(self, n: int):
        """
        Initialize spectral folder for number n
        
        Args:
            n: Number being factored
        """
        self.n = n
        self.root = int(math.isqrt(n))
        self.points = self._build_folding_points()
    
    def _build_folding_points(self) -> List[int]:
        """
        Build folding points from various patterns
        
        Returns:
            List of folding points
        """
        folds = set()
        
        # Binary folding points (powers of 2)
        folds.add(2)
        power = 2
        for i in range(1, self.n.bit_length()):
            power = 2 ** i
            if power <= self.root:
                folds.add(power)
            
            # Also add n/power
            if power < self.n:
                complement = self.n // power
                if 2 <= complement <= self.root:
                    folds.add(complement)
        
        # Periodic folding (based on digital cycles)
        for p in primes_up_to(min(100, self.root)):
            # Find period of 10^k mod p
            period = 1
            remainder = 10 % p
            start = remainder
            
            while period < p:
                remainder = (remainder * 10) % p
                if remainder == start:
                    break
                period += 1
            
            # Add folding points at period intervals
            if period < p and period > 1:
                for k in range(period, self.root, period):
                    if 2 <= k <= self.root:
                        folds.add(k)
        
        # Fibonacci folding points
        k = 1
        while True:
            f = fib(k)
            if f > self.root:
                break
            if f >= 2:
                folds.add(f)
            k += 1
        
        return sorted(list(folds))
    
    def next_fold(self, current: int) -> int:
        """
        Find next folding point after current position
        
        Args:
            current: Current position
            
        Returns:
            Next folding point
        """
        for fold in self.points:
            if fold > current:
                return fold
        
        # No more folds, return a default jump
        return min(self.root, current + max(1, self.root // 20))
    
    def nearest_fold(self, position: int) -> int:
        """
        Find nearest folding point to position
        
        Args:
            position: Target position
            
        Returns:
            Nearest folding point
        """
        if not self.points:
            return position
        
        min_distance = float('inf')
        nearest = self.points[0]
        
        for fold in self.points:
            distance = abs(fold - position)
            if distance < min_distance:
                min_distance = distance
                nearest = fold
        
        return nearest

def quantum_superposition_collapse(n: int, positions: List[int], 
                                 weights: List[float], 
                                 collapse_factor: float = 0.5) -> List[int]:
    """
    Collapse quantum superposition based on weights
    
    Args:
        n: Number being factored
        positions: Superposition positions
        weights: Position weights
        collapse_factor: How much to collapse (0=none, 1=complete)
        
    Returns:
        Collapsed positions
    """
    if not positions or not weights:
        return positions
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return positions
    
    normalized = [w / total_weight for w in weights]
    
    # Sort by weight
    sorted_pairs = sorted(zip(positions, normalized), key=lambda x: -x[1])
    
    # Determine how many to keep
    keep_count = max(1, int(len(positions) * (1 - collapse_factor)))
    
    # Keep top weighted positions
    collapsed = [pos for pos, _ in sorted_pairs[:keep_count]]
    
    return collapsed

def entangle_positions(n: int, pos1: int, pos2: int) -> List[int]:
    """
    Create entangled positions from two base positions
    
    Args:
        n: Number being factored
        pos1: First position
        pos2: Second position
        
    Returns:
        List of entangled positions
    """
    root = int(math.isqrt(n))
    entangled = set()
    
    # Direct combinations
    if pos1 + pos2 <= root:
        entangled.add(pos1 + pos2)
    
    if abs(pos1 - pos2) >= 2:
        entangled.add(abs(pos1 - pos2))
    
    # Geometric mean
    geom = int(math.sqrt(pos1 * pos2))
    if 2 <= geom <= root:
        entangled.add(geom)
    
    # Harmonic mean
    if pos1 + pos2 > 0:
        harm = int(2 * pos1 * pos2 / (pos1 + pos2))
        if 2 <= harm <= root:
            entangled.add(harm)
    
    # Golden ratio entanglement
    golden1 = int(pos1 * PHI + pos2 / PHI)
    golden2 = int(pos1 / PHI + pos2 * PHI)
    
    for g in [golden1, golden2]:
        if 2 <= g <= root:
            entangled.add(g)
    
    # Modular entanglement
    mod_pos = (pos1 * pos2) % root
    if mod_pos == 0:
        mod_pos = min(pos1, pos2)
    if 2 <= mod_pos <= root:
        entangled.add(mod_pos)
    
    return sorted(list(entangled))
