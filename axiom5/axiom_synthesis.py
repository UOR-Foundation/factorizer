"""
Axiom Synthesis - Combine patterns from different axioms
Creates hybrid methods and emergent algorithms
"""

import math
from typing import List, Dict, Tuple, Callable, Optional
from collections import defaultdict

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom1 import primes_up_to, is_prime
from axiom2 import fib, PHI
from axiom3 import coherence
from axiom4 import MultiScaleObserver

class AxiomSynthesizer:
    """
    Synthesizes new factorization methods from axiom combinations
    """
    
    def __init__(self, n: int):
        """
        Initialize axiom synthesizer
        
        Args:
            n: Number to factor
        """
        self.n = n
        self.root = int(math.isqrt(n))
        self.success_patterns = []
        self.hybrid_methods = []
        
    def record_success(self, axioms_used: List[str], position: int, 
                      method_description: str = ""):
        """
        Record a successful pattern
        
        Args:
            axioms_used: List of axioms that contributed
            position: Position where factor was found
            method_description: Optional description
        """
        pattern = {
            'axioms': axioms_used,
            'position': position,
            'description': method_description,
            'weight': 1.0 / len(axioms_used)  # Share credit
        }
        self.success_patterns.append(pattern)
    
    def synthesize_method(self, axiom_weights: Dict[str, float]) -> Callable:
        """
        Create hybrid method from weighted axiom combination
        
        Args:
            axiom_weights: Weights for each axiom
            
        Returns:
            Hybrid method function
        """
        def hybrid_method(x: int) -> float:
            """Combined axiom evaluation"""
            score = 0.0
            
            # Axiom 1: Prime-based score
            if 'axiom1' in axiom_weights:
                # Check prime proximity
                prime_score = 0.0
                for p in primes_up_to(min(20, x)):
                    if x % p == 0:
                        prime_score += 1.0 / p
                    elif abs(x - p) <= 2:
                        prime_score += 0.5 / (1 + abs(x - p))
                score += axiom_weights['axiom1'] * prime_score
            
            # Axiom 2: Fibonacci-based score
            if 'axiom2' in axiom_weights:
                # Check Fibonacci proximity
                fib_score = 0.0
                k = 1
                while fib(k) < x + 5:
                    f = fib(k)
                    if abs(x - f) <= 2:
                        fib_score += 1.0 / (1 + abs(x - f))
                    k += 1
                score += axiom_weights['axiom2'] * fib_score
            
            # Axiom 3: Coherence-based score
            if 'axiom3' in axiom_weights:
                # Use coherence if x divides n
                if self.n % x == 0:
                    coh = coherence(x, self.n // x, self.n)
                else:
                    coh = coherence(x, x, self.n)
                score += axiom_weights['axiom3'] * coh
            
            # Axiom 4: Observer-based score
            if 'axiom4' in axiom_weights:
                # Multi-scale observation
                observer = MultiScaleObserver(self.n)
                obs_score = observer.observe(x)
                score += axiom_weights['axiom4'] * obs_score
            
            return score
        
        self.hybrid_methods.append(hybrid_method)
        return hybrid_method
    
    def learn_weights(self) -> Dict[str, float]:
        """
        Learn optimal axiom weights from success patterns
        
        Returns:
            Optimal weights for each axiom
        """
        axiom_scores = defaultdict(float)
        axiom_counts = defaultdict(int)
        
        for pattern in self.success_patterns:
            for axiom in pattern['axioms']:
                axiom_scores[axiom] += pattern['weight']
                axiom_counts[axiom] += 1
        
        # Normalize to get weights
        weights = {}
        total_score = sum(axiom_scores.values())
        
        if total_score > 0:
            for axiom in axiom_scores:
                weights[axiom] = axiom_scores[axiom] / total_score
        else:
            # Default equal weights
            axioms = ['axiom1', 'axiom2', 'axiom3', 'axiom4']
            for axiom in axioms:
                weights[axiom] = 0.25
        
        return weights

def pattern_fusion(patterns: List[Dict], n: int) -> List[int]:
    """
    Fuse successful patterns to generate new candidates
    
    Args:
        patterns: List of success patterns
        n: Number being factored
        
    Returns:
        Fused candidate positions
    """
    root = int(math.isqrt(n))
    candidates = set()
    
    # Extract positions from patterns
    positions = [p['position'] for p in patterns]
    
    # Direct positions
    candidates.update(positions)
    
    # Pairwise combinations
    for i, pos1 in enumerate(positions):
        for pos2 in positions[i+1:]:
            # Arithmetic mean
            mean = (pos1 + pos2) // 2
            if 2 <= mean <= root:
                candidates.add(mean)
            
            # Geometric mean
            geom = int(math.sqrt(pos1 * pos2))
            if 2 <= geom <= root:
                candidates.add(geom)
            
            # Golden ratio combination
            golden = int(pos1 * PHI + pos2 / PHI)
            if 2 <= golden <= root:
                candidates.add(golden)
    
    return sorted(list(candidates))

def create_hybrid_method(axiom_combinations: List[Tuple[str, str]], 
                        n: int) -> Callable:
    """
    Create hybrid method from axiom combinations
    
    Args:
        axiom_combinations: List of (axiom1, axiom2) pairs
        n: Number being factored
        
    Returns:
        Hybrid evaluation function
    """
    synthesizer = AxiomSynthesizer(n)
    
    # Count axiom appearances
    axiom_counts = defaultdict(int)
    for ax1, ax2 in axiom_combinations:
        axiom_counts[ax1] += 1
        axiom_counts[ax2] += 1
    
    # Convert to weights
    total = sum(axiom_counts.values())
    weights = {ax: count/total for ax, count in axiom_counts.items()}
    
    return synthesizer.synthesize_method(weights)

def cross_axiom_resonance(n: int, axiom_results: Dict[str, List[int]]) -> List[int]:
    """
    Find positions where multiple axioms resonate
    
    Args:
        n: Number being factored
        axiom_results: Dictionary of axiom -> candidate positions
        
    Returns:
        Resonant positions
    """
    root = int(math.isqrt(n))
    position_scores = defaultdict(float)
    
    # Score each position by axiom overlap
    for axiom, positions in axiom_results.items():
        for pos in positions:
            position_scores[pos] += 1.0
            
            # Add nearby positions with decay
            for offset in range(1, 3):
                if 2 <= pos - offset <= root:
                    position_scores[pos - offset] += 0.5 / offset
                if 2 <= pos + offset <= root:
                    position_scores[pos + offset] += 0.5 / offset
    
    # Find high-resonance positions
    threshold = len(axiom_results) * 0.6  # 60% axiom agreement
    resonant = [pos for pos, score in position_scores.items() 
                if score >= threshold]
    
    # Sort by score
    resonant.sort(key=lambda p: -position_scores[p])
    
    return resonant

def emergent_pattern_detection(n: int, observation_history: List[Dict]) -> List[int]:
    """
    Detect emergent patterns from observation history
    
    Args:
        n: Number being factored
        observation_history: History of observations
        
    Returns:
        Emergent candidate positions
    """
    root = int(math.isqrt(n))
    
    # Group by axiom
    axiom_positions = defaultdict(list)
    for obs in observation_history:
        if 'axiom' in obs and 'position' in obs:
            axiom_positions[obs['axiom']].append(obs['position'])
    
    # Find cross-axiom patterns
    emergent = set()
    
    # Pattern 1: Fibonacci-Prime intersections
    if 'axiom1' in axiom_positions and 'axiom2' in axiom_positions:
        primes = set(axiom_positions['axiom1'])
        fibs = set(axiom_positions['axiom2'])
        
        # Positions that are both near primes and Fibonacci
        for p in primes:
            for f in fibs:
                if abs(p - f) <= 2:
                    emergent.add((p + f) // 2)
    
    # Pattern 2: Coherence peaks at golden ratio intervals
    if 'axiom3' in axiom_positions:
        coherence_peaks = sorted(axiom_positions['axiom3'])
        
        for i in range(len(coherence_peaks) - 1):
            gap = coherence_peaks[i+1] - coherence_peaks[i]
            golden_gap = int(gap * PHI)
            
            next_pos = coherence_peaks[i] + golden_gap
            if 2 <= next_pos <= root:
                emergent.add(next_pos)
    
    # Pattern 3: Observer fixed points
    if 'axiom4' in axiom_positions:
        observer_positions = axiom_positions['axiom4']
        
        # Find positions that appear multiple times
        pos_counts = defaultdict(int)
        for pos in observer_positions:
            pos_counts[pos] += 1
        
        # Fixed points appear frequently
        for pos, count in pos_counts.items():
            if count >= 3:
                emergent.add(pos)
    
    return sorted(list(emergent))

def synthesize_from_failures(n: int, failed_positions: List[int]) -> List[int]:
    """
    Synthesize new positions from failure patterns
    
    Learn what doesn't work to find what might
    
    Args:
        n: Number being factored
        failed_positions: Positions that didn't yield factors
        
    Returns:
        New candidate positions
    """
    root = int(math.isqrt(n))
    candidates = set()
    
    if not failed_positions:
        return []
    
    # Find gaps in failed positions
    failed_set = set(failed_positions)
    
    # Large gaps might hide factors
    sorted_failed = sorted(failed_positions)
    for i in range(len(sorted_failed) - 1):
        gap = sorted_failed[i+1] - sorted_failed[i]
        
        if gap > root // 20:  # Significant gap
            # Try middle of gap
            mid = (sorted_failed[i] + sorted_failed[i+1]) // 2
            if mid not in failed_set and 2 <= mid <= root:
                candidates.add(mid)
    
    # Invert the failure pattern
    # If many failures are even, try odd positions
    even_failures = sum(1 for p in failed_positions if p % 2 == 0)
    odd_failures = len(failed_positions) - even_failures
    
    if even_failures > odd_failures * 2:
        # Many even failures, focus on odd
        for x in range(3, min(root, 100), 2):
            if x not in failed_set:
                candidates.add(x)
    
    return sorted(list(candidates))[:20]  # Top 20
