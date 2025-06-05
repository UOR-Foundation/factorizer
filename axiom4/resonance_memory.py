"""
Resonance Memory - Remember and predict successful resonance patterns
Scales past successes to new problems
"""

import math
from typing import List, Tuple, Dict, Optional

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom2 import PHI

class ResonanceMemory:
    """
    Records successful resonance patterns and predicts future positions
    """
    
    def __init__(self, memory_size: int = 100):
        """
        Initialize resonance memory
        
        Args:
            memory_size: Maximum number of patterns to remember
        """
        self.memory_size = memory_size
        
        # Graph of (p,f) -> strength mappings
        self.resonance_graph: Dict[Tuple[int, int], float] = {}
        
        # List of successful factorizations
        # Format: (prime, fibonacci, n, factor)
        self.successes: List[Tuple[int, int, int, int]] = []
        
        # Pattern strength decay factor
        self.decay_factor = 0.7
        
    def record(self, p: int, f: int, n: int, strength: float, 
               factor: Optional[int] = None):
        """
        Record a resonance pattern
        
        Args:
            p: Prime component
            f: Fibonacci component  
            n: Number being factored
            strength: Resonance strength
            factor: Factor found (if successful)
        """
        key = (p, f)
        
        # Update resonance graph with exponential moving average
        old_strength = self.resonance_graph.get(key, 0.0)
        new_strength = self.decay_factor * old_strength + (1 - self.decay_factor) * strength
        self.resonance_graph[key] = new_strength
        
        # Record success if factor found
        if factor and factor > 1:
            self.successes.append((p, f, n, factor))
            
            # Limit memory size
            if len(self.successes) > self.memory_size:
                self.successes = self.successes[-self.memory_size:]
    
    def predict(self, n: int, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Predict likely factor positions based on past successes
        
        Args:
            n: Number to factor
            top_k: Number of predictions to return
            
        Returns:
            List of (position, weight) predictions
        """
        root = int(math.isqrt(n))
        predictions = {}
        
        # Scale past successes to current problem
        for p, f, prev_n, factor in self.successes:
            # Scale factor
            scale = n / prev_n
            
            # Direct scaling
            pos1 = int(factor * scale)
            if 2 <= pos1 <= root:
                weight = 0.8 / (1 + abs(math.log(scale)))
                predictions[pos1] = max(predictions.get(pos1, 0), weight)
            
            # Golden ratio scaling
            pos2 = int(factor * scale * PHI)
            if 2 <= pos2 <= root:
                weight = 0.6 / (1 + abs(math.log(scale * PHI)))
                predictions[pos2] = max(predictions.get(pos2, 0), weight)
            
            # Inverse golden scaling
            pos3 = int(factor * scale / PHI)
            if 2 <= pos3 <= root:
                weight = 0.6 / (1 + abs(math.log(scale / PHI)))
                predictions[pos3] = max(predictions.get(pos3, 0), weight)
        
        # Use resonance graph patterns
        for (p2, f2), strength in self.resonance_graph.items():
            # Look for similar patterns
            for p, f, _, _ in self.successes:
                if abs(p2 - p) <= 2 and abs(f2 - f) <= 1:
                    # Similar pattern found
                    pos = (p2 * f2) % root
                    if pos == 0:
                        pos = p2
                    if 2 <= pos <= root:
                        weight = strength * 0.5
                        predictions[pos] = max(predictions.get(pos, 0), weight)
        
        # Sort by weight and return top k
        sorted_predictions = sorted(predictions.items(), key=lambda x: -x[1])
        return sorted_predictions[:top_k]
    
    def get_pattern_strength(self, p: int, f: int) -> float:
        """
        Get strength of a specific resonance pattern
        
        Args:
            p: Prime component
            f: Fibonacci component
            
        Returns:
            Pattern strength (0 if not found)
        """
        return self.resonance_graph.get((p, f), 0.0)
    
    def find_similar_patterns(self, p: int, f: int, 
                            tolerance: int = 3) -> List[Tuple[Tuple[int, int], float]]:
        """
        Find patterns similar to given (p,f)
        
        Args:
            p: Prime component
            f: Fibonacci component
            tolerance: How close patterns need to be
            
        Returns:
            List of ((p,f), strength) for similar patterns
        """
        similar = []
        
        for (p2, f2), strength in self.resonance_graph.items():
            if abs(p2 - p) <= tolerance and abs(f2 - f) <= tolerance:
                if (p2, f2) != (p, f):  # Exclude exact match
                    similar.append(((p2, f2), strength))
        
        # Sort by strength
        similar.sort(key=lambda x: -x[1])
        
        return similar
    
    def success_rate(self) -> float:
        """
        Calculate success rate based on recorded patterns
        
        Returns:
            Success rate (0 to 1)
        """
        if not self.resonance_graph:
            return 0.0
        
        # Count strong patterns
        strong_patterns = sum(1 for s in self.resonance_graph.values() if s > 0.5)
        
        return strong_patterns / len(self.resonance_graph)
    
    def clear(self):
        """
        Clear all memory
        """
        self.resonance_graph.clear()
        self.successes.clear()
    
    def merge(self, other: 'ResonanceMemory'):
        """
        Merge another memory into this one
        
        Args:
            other: Other resonance memory to merge
        """
        # Merge resonance graph
        for key, strength in other.resonance_graph.items():
            old_strength = self.resonance_graph.get(key, 0.0)
            # Use maximum strength
            self.resonance_graph[key] = max(old_strength, strength)
        
        # Merge successes
        self.successes.extend(other.successes)
        
        # Limit size
        if len(self.successes) > self.memory_size:
            self.successes = self.successes[-self.memory_size:]

def analyze_resonance_landscape(n: int, memory: ResonanceMemory,
                              resolution: int = 50) -> Dict[int, float]:
    """
    Create resonance landscape based on memory
    
    Args:
        n: Number being factored
        memory: Resonance memory
        resolution: Landscape resolution
        
    Returns:
        Dictionary mapping position to resonance strength
    """
    root = int(math.isqrt(n))
    landscape = {}
    
    # Get predictions
    predictions = memory.predict(n, top_k=resolution)
    
    # Create landscape
    for pos, weight in predictions:
        landscape[pos] = weight
        
        # Add some spread around strong positions
        if weight > 0.5:
            spread = max(1, int(root * 0.02))
            for offset in range(-spread, spread + 1):
                neighbor = pos + offset
                if 2 <= neighbor <= root and neighbor not in landscape:
                    # Gaussian-like falloff
                    neighbor_weight = weight * math.exp(-(offset**2) / (2 * spread**2))
                    landscape[neighbor] = neighbor_weight
    
    return landscape

def resonance_guided_search(n: int, memory: ResonanceMemory,
                          max_attempts: int = 20) -> Optional[int]:
    """
    Search for factors using resonance memory guidance
    
    Args:
        n: Number to factor
        memory: Resonance memory
        max_attempts: Maximum positions to try
        
    Returns:
        Factor if found, None otherwise
    """
    # Get predictions
    predictions = memory.predict(n, top_k=max_attempts)
    
    # Try each prediction
    for pos, weight in predictions:
        if n % pos == 0 and pos > 1:
            return pos
    
    # Also try positions near high-weight predictions
    root = int(math.isqrt(n))
    for pos, weight in predictions[:5]:  # Top 5 predictions
        if weight > 0.5:
            # Search neighborhood
            search_radius = max(1, int(weight * 10))
            for offset in range(-search_radius, search_radius + 1):
                test_pos = pos + offset
                if 2 <= test_pos <= root and n % test_pos == 0:
                    return test_pos
    
    return None
