"""
Fold Topology - Energy landscape and topological navigation
Analyzes fold patterns in spectral space
"""

import math
from typing import List, Dict, Tuple, Set
from .spectral_core import spectral_vector

def fold_energy(n: int, x: int) -> float:
    """
    Calculate fold energy at position x for number n
    
    Fold energy measures the spectral distance between:
    - x (potential factor)
    - n/x (complementary factor)
    - n (target number)
    
    Low energy indicates x is likely a factor of n.
    
    Args:
        n: Number being factored
        x: Position to evaluate
        
    Returns:
        Fold energy (lower is better)
    """
    if x <= 0 or x > n:
        return float('inf')
    
    if n % x != 0:
        # For non-factors, use approximate complementary value
        y = n / x
    else:
        y = n // x
    
    # Get spectral vectors
    sx = spectral_vector(x)
    sy = spectral_vector(int(y))
    sn = spectral_vector(n)
    
    # Calculate energy as squared distance
    energy = 0.0
    for i in range(len(sx)):
        # Expected: sx + sy ≈ 2×sn for factors
        diff = sx[i] + sy[i] - sn[i]
        energy += diff * diff
    
    return energy

def sharp_fold_candidates(n: int, span: int = 25) -> List[int]:
    """
    Find sharp folds (local minima) in the energy landscape
    
    Searches around sqrt(n) for positions where the fold energy
    has high negative curvature, indicating potential factors.
    
    Args:
        n: Number being factored
        span: Search radius around sqrt(n)
        
    Returns:
        List of candidate positions sorted by curvature
    """
    root = int(math.isqrt(n))
    
    # Define search window
    start = max(2, root - span)
    end = min(root + span, n // 2) + 1
    window = list(range(start, end))
    
    # Calculate energies
    energies = [fold_energy(n, x) for x in window]
    
    # Calculate curvatures (second derivative)
    curvatures = []
    for i in range(1, len(window) - 1):
        # Negative curvature indicates local minimum
        curvature = energies[i - 1] - 2 * energies[i] + energies[i + 1]
        curvatures.append((curvature, window[i]))
    
    # Sort by curvature (most negative first)
    curvatures.sort()
    
    # Return top candidates
    return [x for _, x in curvatures[:10]]

class FoldTopology:
    """
    Navigate the fold energy landscape using topological structure
    
    This class builds a topology of the energy landscape, identifying
    connected components and paths between low-energy regions.
    """
    
    def __init__(self, n: int):
        """
        Initialize fold topology for number n
        
        Args:
            n: Number to analyze
        """
        self.n = n
        self.root = int(math.isqrt(n))
        self.points: List[int] = []
        self.connections: Dict[int, List[Tuple[int, float]]] = {}
        self._build_topology()
    
    def _build_topology(self):
        """Build the topological structure"""
        # Find all local minima
        energies = {}
        for x in range(2, self.root + 1):
            energies[x] = fold_energy(self.n, x)
        
        # Identify local minima
        for x in range(3, self.root):
            if energies[x] < energies[x - 1] and energies[x] < energies[x + 1]:
                self.points.append(x)
        
        # Build connections between points
        for p1 in self.points:
            connections = []
            
            for p2 in self.points:
                if p2 == p1:
                    continue
                
                # Sample energy along path between p1 and p2
                mid_points = [int(p1 + t * (p2 - p1)) for t in [0.25, 0.5, 0.75]]
                path_energies = [energies.get(m, fold_energy(self.n, m)) for m in mid_points]
                
                # Average energy along path
                avg_path_energy = sum(path_energies) / len(path_energies)
                
                # Energy at endpoints
                endpoint_energy = (energies[p1] + energies[p2]) / 2
                
                # Connect if path doesn't go too high above endpoints
                if avg_path_energy < endpoint_energy * 1.5:
                    # Weight by inverse of average energy
                    weight = 1 / (1 + avg_path_energy)
                    connections.append((p2, weight))
            
            self.connections[p1] = connections
    
    def components(self) -> List[List[int]]:
        """
        Find connected components in the energy landscape
        
        Returns:
            List of components, each containing connected points
        """
        components = []
        unvisited = set(self.points)
        
        while unvisited:
            # Start new component
            component = []
            stack = [unvisited.pop()]
            
            while stack:
                current = stack.pop()
                component.append(current)
                
                # Add connected neighbors
                for neighbor, _ in self.connections.get(current, []):
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        stack.append(neighbor)
            
            components.append(sorted(component))
        
        return components
    
    def traverse(self) -> List[int]:
        """
        Traverse the topology following strongest connections
        
        Returns:
            Path through low-energy regions
        """
        if not self.points:
            return []
        
        # Start from lowest energy point
        current = min(self.points, key=lambda x: fold_energy(self.n, x))
        visited = set()
        path = []
        
        while current not in visited:
            visited.add(current)
            path.append(current)
            
            # Check if current is a factor
            if self.n % current == 0:
                return path
            
            # Find best neighbor
            neighbors = self.connections.get(current, [])
            if neighbors:
                # Choose neighbor with highest weight (lowest energy path)
                current = max(neighbors, key=lambda t: t[1])[0]
            else:
                break
        
        return path

def find_energy_valleys(n: int, resolution: int = 100) -> List[int]:
    """
    Find valleys (local minima) in the energy landscape
    
    Scans the energy landscape at given resolution to find
    all local minima.
    
    Args:
        n: Number being factored
        resolution: Number of points to sample
        
    Returns:
        List of valley positions
    """
    root = int(math.isqrt(n))
    step = max(1, root // resolution)
    
    valleys = []
    prev_energy = float('inf')
    current_energy = fold_energy(n, 2)
    
    for x in range(3, root + 1, step):
        next_energy = fold_energy(n, x)
        
        # Check for local minimum
        if current_energy < prev_energy and current_energy < next_energy:
            valleys.append(x - step)
        
        prev_energy = current_energy
        current_energy = next_energy
    
    return valleys
