"""
Fibonacci Vortex Engine - Golden spiral flows and vortex interference

Implements the foundation of Axiom 2 for the Prime Sieve.
Creates natural factor attractors through Fibonacci vortices
and golden spiral navigation.
"""

import math
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass

# Import Fibonacci tools from axiom2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from axiom2.fibonacci_core import fib, PHI, PSI, GOLDEN_ANGLE, fib_wave
from axiom2.fibonacci_vortices import fib_vortices, golden_spiral_positions
from axiom2.fibonacci_entanglement import FibonacciEntanglement


@dataclass
class VortexPoint:
    """Represents a point in a Fibonacci vortex"""
    position: int
    vortex_type: str  # 'center', 'phi_scaled', 'inv_phi_scaled'
    source_fib: int   # Source Fibonacci number
    pull_strength: float
    entanglement_score: float


class FibonacciVortex:
    """
    Fibonacci vortex generator with golden spiral dynamics.
    
    Key features:
    - Dynamic Fibonacci generation scaled to problem size
    - Golden spiral navigation and vortex creation
    - Entanglement detection between factor pairs
    - Adaptive vortex density for large numbers
    """
    
    def __init__(self, n: int):
        """
        Initialize vortex engine for number n.
        
        Args:
            n: Number to be factored
        """
        self.n = n
        self.bit_length = n.bit_length()
        self.sqrt_n = int(math.isqrt(n))
        
        # Generate Fibonacci numbers up to sqrt(n)
        self.fibonacci_numbers = self._generate_fibonacci_basis()
        
        # Entanglement detector
        self.entanglement = FibonacciEntanglement(n)
        
        # Cache for vortex calculations
        self.vortex_cache: Dict[int, float] = {}
        
    def _generate_fibonacci_basis(self) -> List[int]:
        """
        Generate Fibonacci numbers appropriate for n's scale.
        
        Returns:
            List of Fibonacci numbers up to sqrt(n)
        """
        fibs = []
        k = 2
        
        # For very large numbers, use sparse Fibonacci sampling
        if self.bit_length > 512:
            # Logarithmic sampling
            max_k = int(math.log(self.sqrt_n) / math.log(PHI))
            step = max(1, max_k // 100)
            
            while k <= max_k:
                f = fib(k)
                if f > self.sqrt_n:
                    break
                fibs.append(f)
                k += step
        else:
            # Standard generation for smaller numbers
            while True:
                f = fib(k)
                if f > self.sqrt_n:
                    break
                fibs.append(f)
                k += 1
        
        return fibs
    
    def generate_vortex_centers(self) -> List[VortexPoint]:
        """
        Generate vortex center positions using Fibonacci numbers.
        
        Returns:
            List of vortex center points
        """
        vortex_points = []
        
        for i, f in enumerate(self.fibonacci_numbers):
            if f < 2 or f > self.sqrt_n:
                continue
            
            # Basic vortex at Fibonacci position
            pull = self._calculate_vortex_pull(f)
            entanglement = self._calculate_entanglement(f)
            
            vortex_points.append(VortexPoint(
                position=f,
                vortex_type='center',
                source_fib=f,
                pull_strength=pull,
                entanglement_score=entanglement
            ))
            
            # φ-scaled vortex
            phi_pos = int(f * PHI)
            if phi_pos <= self.sqrt_n:
                pull = self._calculate_vortex_pull(phi_pos)
                entanglement = self._calculate_entanglement(phi_pos)
                
                vortex_points.append(VortexPoint(
                    position=phi_pos,
                    vortex_type='phi_scaled',
                    source_fib=f,
                    pull_strength=pull,
                    entanglement_score=entanglement
                ))
            
            # 1/φ-scaled vortex
            inv_phi_pos = int(f / PHI)
            if inv_phi_pos >= 2:
                pull = self._calculate_vortex_pull(inv_phi_pos)
                entanglement = self._calculate_entanglement(inv_phi_pos)
                
                vortex_points.append(VortexPoint(
                    position=inv_phi_pos,
                    vortex_type='inv_phi_scaled',
                    source_fib=f,
                    pull_strength=pull,
                    entanglement_score=entanglement
                ))
        
        return vortex_points
    
    def _calculate_vortex_pull(self, x: int) -> float:
        """
        Calculate gravitational pull of vortex at position x.
        
        Args:
            x: Position to evaluate
            
        Returns:
            Pull strength value
        """
        if x in self.vortex_cache:
            return self.vortex_cache[x]
        
        # Base pull from proximity to Fibonacci numbers
        min_dist = float('inf')
        for f in self.fibonacci_numbers:
            dist = abs(x - f)
            if dist < min_dist:
                min_dist = dist
        
        # Pull decreases with distance
        pull = 1.0 / (1.0 + min_dist / math.sqrt(self.sqrt_n))
        
        # Bonus if x divides n
        if self.n % x == 0:
            pull += 1.0
        
        # Cache result
        if len(self.vortex_cache) < 10000:
            self.vortex_cache[x] = pull
        
        return pull
    
    def _calculate_entanglement(self, x: int) -> float:
        """
        Calculate Fibonacci entanglement strength at position x.
        
        Args:
            x: Position to evaluate
            
        Returns:
            Entanglement score
        """
        # Check if x and n/x (if factor) are both near Fibonacci numbers
        if self.n % x == 0:
            other = self.n // x
            
            # Find minimum distances to Fibonacci numbers
            x_dist = min(abs(x - f) for f in self.fibonacci_numbers) if self.fibonacci_numbers else float('inf')
            other_dist = min(abs(other - f) for f in self.fibonacci_numbers) if self.fibonacci_numbers else float('inf')
            
            # Entanglement strength
            combined_dist = x_dist + other_dist
            return 1.0 / (1.0 + combined_dist / self.sqrt_n)
        
        return 0.0
    
    def golden_spiral_search(self, center: int, max_radius: Optional[int] = None) -> List[int]:
        """
        Generate positions along a golden spiral from center.
        
        Args:
            center: Center of spiral
            max_radius: Maximum radius (auto-determined if None)
            
        Returns:
            List of positions along golden spiral
        """
        if max_radius is None:
            max_radius = min(1000, self.sqrt_n // 10)
        
        positions = []
        
        # Number of spiral turns based on problem size
        turns = min(100, self.bit_length * 2)
        
        for k in range(turns):
            # Golden spiral equation
            angle = k * GOLDEN_ANGLE
            radius = max_radius * (k / turns) ** PHI
            
            # Convert to position
            dx = int(radius * math.cos(angle))
            dy = int(radius * math.sin(angle))
            
            # Map 2D spiral to 1D number line
            pos = center + dx
            
            if 2 <= pos <= self.sqrt_n:
                positions.append(pos)
        
        return list(set(positions))
    
    def generate_interference_pattern(self, vortex_centers: List[VortexPoint]) -> Dict[int, float]:
        """
        Generate interference pattern from multiple vortices.
        
        Args:
            vortex_centers: List of vortex center points
            
        Returns:
            Dictionary mapping positions to interference strength
        """
        interference = {}
        
        # Sample positions based on problem size
        if self.bit_length < 128:
            # Dense sampling for small numbers
            sample_positions = range(2, min(10000, self.sqrt_n + 1))
        else:
            # Sparse sampling for large numbers
            sample_positions = []
            for v in vortex_centers[:20]:  # Limit vortices for efficiency
                # Sample around each vortex
                for delta in range(-50, 51, 5):
                    pos = v.position + delta
                    if 2 <= pos <= self.sqrt_n:
                        sample_positions.append(pos)
        
        for pos in sample_positions:
            # Calculate interference from all vortices
            total_interference = 0.0
            
            for vortex in vortex_centers:
                # Distance from vortex center
                dist = abs(pos - vortex.position)
                
                # Wave amplitude decreases with distance
                amplitude = vortex.pull_strength / (1.0 + dist / 100.0)
                
                # Phase based on golden angle
                phase = (dist * GOLDEN_ANGLE) % (2 * math.pi)
                
                # Add wave contribution
                total_interference += amplitude * math.cos(phase)
            
            interference[pos] = total_interference
        
        return interference
    
    def filter_by_entanglement(self, positions: List[int]) -> List[int]:
        """
        Filter positions by Fibonacci entanglement strength.
        
        Args:
            positions: Candidate positions
            
        Returns:
            Positions with high entanglement
        """
        entangled = []
        
        for pos in positions:
            score = self._calculate_entanglement(pos)
            if score > 0.1:  # Threshold for significant entanglement
                entangled.append(pos)
        
        # Sort by entanglement strength
        entangled.sort(key=lambda x: self._calculate_entanglement(x), reverse=True)
        
        return entangled
    
    def fibonacci_vortex_sieve(self, candidates: Optional[Set[int]] = None) -> Set[int]:
        """
        Apply Fibonacci vortex sieving to filter candidates.
        
        Args:
            candidates: Initial candidate set (None to generate)
            
        Returns:
            Filtered candidates passing vortex tests
        """
        # Generate vortex centers
        vortex_centers = self.generate_vortex_centers()
        
        if candidates is None:
            # Start with vortex positions
            candidates = {v.position for v in vortex_centers}
            
            # Add golden spiral positions from top vortices
            for v in sorted(vortex_centers, key=lambda x: x.pull_strength, reverse=True)[:10]:
                spiral_positions = self.golden_spiral_search(v.position)
                candidates.update(spiral_positions[:50])  # Limit per spiral
        
        # Generate interference pattern
        interference = self.generate_interference_pattern(vortex_centers)
        
        # Filter based on interference strength
        filtered = set()
        
        # Find interference peaks
        if interference:
            mean_interference = sum(interference.values()) / len(interference)
            std_interference = math.sqrt(sum((v - mean_interference)**2 for v in interference.values()) / len(interference))
            threshold = mean_interference + std_interference  # One std above mean
            
            for pos in candidates:
                if pos in interference and interference[pos] >= threshold:
                    filtered.add(pos)
                elif pos not in interference:
                    # Check entanglement for positions not in interference map
                    if self._calculate_entanglement(pos) > 0.1:
                        filtered.add(pos)
        else:
            # Fallback to entanglement filtering
            filtered = set(self.filter_by_entanglement(list(candidates)))
        
        return filtered
    
    def detect_double_fibonacci(self) -> List[Tuple[int, int]]:
        """
        Detect potential factors that are both near Fibonacci numbers.
        
        Returns:
            List of (position, score) tuples
        """
        detections = self.entanglement.detect_double()
        
        # Convert to our format
        results = []
        for det in detections[:20]:  # Limit results
            if len(det) >= 2:
                pos = det[0]
                score = det[1] if len(det) > 1 else 1.0
                results.append((pos, score))
        
        return results
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the vortex engine.
        
        Returns:
            Dictionary with engine statistics
        """
        vortex_centers = self.generate_vortex_centers()
        
        return {
            'bit_length': self.bit_length,
            'fibonacci_count': len(self.fibonacci_numbers),
            'max_fibonacci': self.fibonacci_numbers[-1] if self.fibonacci_numbers else 0,
            'vortex_count': len(vortex_centers),
            'vortex_cache_size': len(self.vortex_cache)
        }
