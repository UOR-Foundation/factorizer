"""
Meta-Observer - Observation of observers
Tracks patterns across all axiom applications and identifies blind spots
"""

import math
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom2 import PHI
from axiom3 import coherence, spectral_vector, accelerated_coherence
from .meta_acceleration_cache import get_meta_cache, accelerated_meta_coherence

class AxiomPerformanceProfile:
    """
    Tracks performance of each axiom for different number types
    """
    
    def __init__(self):
        """Initialize performance tracking"""
        # Track successes by axiom and number characteristics
        self.success_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.failure_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.timing_data: Dict[str, List[float]] = defaultdict(list)
        
    def record_attempt(self, axiom: str, n: int, success: bool, 
                      time_taken: float = 0.0, factor_found: Optional[int] = None):
        """
        Record an axiom application attempt
        
        Args:
            axiom: Name of axiom used
            n: Number being factored
            success: Whether factor was found
            time_taken: Time taken in seconds
            factor_found: Factor if found
        """
        # Characterize the number
        characteristics = self._characterize_number(n)
        
        # Record success or failure
        if success:
            for char in characteristics:
                self.success_patterns[axiom][char] += 1
        else:
            for char in characteristics:
                self.failure_patterns[axiom][char] += 1
        
        # Record timing
        if time_taken > 0:
            self.timing_data[axiom].append(time_taken)
    
    def _characterize_number(self, n: int) -> Set[str]:
        """
        Extract characteristics of a number
        
        Args:
            n: Number to characterize
            
        Returns:
            Set of characteristics
        """
        chars = set()
        
        # Size categories
        if n < 100:
            chars.add("small")
        elif n < 10000:
            chars.add("medium")
        else:
            chars.add("large")
        
        # Binary properties
        bit_count = n.bit_length()
        chars.add(f"bits_{bit_count//10}0s")  # 10s, 20s, etc.
        
        # Modular properties
        chars.add(f"mod3_{n % 3}")
        chars.add(f"mod5_{n % 5}")
        chars.add(f"mod7_{n % 7}")
        
        # Digital root
        dr = 1 + (n - 1) % 9 if n > 0 else 0
        chars.add(f"droot_{dr}")
        
        # Spectral properties (simplified)
        spec = spectral_vector(n)
        if spec[0] > 0.5:  # High bit density
            chars.add("high_density")
        else:
            chars.add("low_density")
        
        return chars
    
    def get_best_axiom(self, n: int) -> str:
        """
        Predict best axiom for a given number
        
        Args:
            n: Number to factor
            
        Returns:
            Best axiom name
        """
        chars = self._characterize_number(n)
        scores = {}
        
        for axiom in self.success_patterns:
            score = 0
            total = 0
            
            for char in chars:
                successes = self.success_patterns[axiom].get(char, 0)
                failures = self.failure_patterns[axiom].get(char, 0)
                total_attempts = successes + failures
                
                if total_attempts > 0:
                    score += successes / total_attempts
                    total += 1
            
            if total > 0:
                scores[axiom] = score / total
        
        # Return axiom with highest score, or default
        if scores:
            return max(scores, key=scores.get)
        else:
            return "axiom1"  # Default to prime-based

class MetaObserver:
    """
    Observes the observation process itself
    Creates meta-coherence fields and detects blind spots
    """
    
    def __init__(self, n: int):
        """
        Initialize meta-observer for number n
        
        Args:
            n: Number being factored
        """
        self.n = n
        self.root = int(math.isqrt(n))
        self.observation_history = []
        self.coherence_history = []
        self.blind_spots = set()
        self.cache = get_meta_cache()
        
    def observe_observation(self, position: int, coherence_value: float, 
                          axiom_used: str, found_factor: bool = False):
        """
        Record an observation event
        
        Args:
            position: Position observed
            coherence_value: Coherence at position
            axiom_used: Which axiom was used
            found_factor: Whether a factor was found
        """
        event = {
            'position': position,
            'coherence': coherence_value,
            'axiom': axiom_used,
            'found': found_factor,
            'timestamp': len(self.observation_history)
        }
        self.observation_history.append(event)
        self.coherence_history.append(coherence_value)
        
        # Add to accelerated cache for O(1) lookups
        self.cache.add_observation(event)
        
        # Update blind spots
        if not found_factor and coherence_value < 0.1:
            self.blind_spots.add(position)
    
    def get_meta_coherence(self, position: int) -> float:
        """
        Calculate meta-coherence at a position
        
        Meta-coherence is coherence of the coherence patterns
        
        Args:
            position: Position to evaluate
            
        Returns:
            Meta-coherence value
        """
        # Create coherence field for caching
        coherence_field = {}
        if self.coherence_history:
            # Sample coherence values from observation history
            for event in self.observation_history:
                coherence_field[event['position']] = event['coherence']
        
        def compute_meta_coherence(pos, field):
            if not self.coherence_history:
                return 0.0
            
            # Use cached observations for faster lookup
            window_size = max(1, self.root // 20)
            nearby_obs = self.cache.query_observations(position=position)
            
            if not nearby_obs:
                # Fall back to computing from observation history
                nearby_coherences = []
                for event in self.observation_history:
                    if abs(event['position'] - pos) <= window_size:
                        nearby_coherences.append(event['coherence'])
            else:
                # Use cached observations
                nearby_coherences = [obs['coherence'] for obs in nearby_obs 
                                   if abs(obs['position'] - pos) <= window_size]
            
            if len(nearby_coherences) < 2:
                return 0.0
            
            # Calculate coherence of coherences
            mean_coh = sum(nearby_coherences) / len(nearby_coherences)
            variance = sum((c - mean_coh) ** 2 for c in nearby_coherences) / len(nearby_coherences)
            
            # Low variance in coherence = high meta-coherence
            meta_coh = math.exp(-variance)
            
            # Boost if we're near successful observations
            success_boost = 0
            success_obs = [e for e in self.observation_history if e['found']]
            for event in success_obs:
                if abs(event['position'] - pos) <= 2:
                    success_boost += 0.5
            
            return min(1.0, meta_coh + success_boost)
        
        # Use accelerated meta-coherence
        return accelerated_meta_coherence(position, coherence_field, compute_meta_coherence)
    
    def detect_observation_patterns(self) -> Dict[str, List[int]]:
        """
        Detect patterns in observation history
        
        Returns:
            Dictionary of pattern types to positions
        """
        patterns = {
            'high_activity': [],
            'coherence_peaks': [],
            'axiom_boundaries': [],
            'repetition_cycles': []
        }
        
        if len(self.observation_history) < 10:
            return patterns
        
        # High activity regions
        position_counts = defaultdict(int)
        for event in self.observation_history:
            position_counts[event['position']] += 1
        
        avg_count = sum(position_counts.values()) / len(position_counts)
        for pos, count in position_counts.items():
            if count > 2 * avg_count:
                patterns['high_activity'].append(pos)
        
        # Coherence peaks
        for i in range(1, len(self.coherence_history) - 1):
            if (self.coherence_history[i] > self.coherence_history[i-1] and
                self.coherence_history[i] > self.coherence_history[i+1]):
                patterns['coherence_peaks'].append(self.observation_history[i]['position'])
        
        # Axiom transition boundaries
        for i in range(1, len(self.observation_history)):
            if self.observation_history[i]['axiom'] != self.observation_history[i-1]['axiom']:
                patterns['axiom_boundaries'].append(self.observation_history[i]['position'])
        
        # Repetition cycles
        seen_positions = {}
        for i, event in enumerate(self.observation_history):
            pos = event['position']
            if pos in seen_positions:
                cycle_length = i - seen_positions[pos]
                if cycle_length > 2:
                    patterns['repetition_cycles'].append(pos)
            seen_positions[pos] = i
        
        return patterns

def detect_blind_spots(observer: MetaObserver, coverage_threshold: float = 0.8) -> List[int]:
    """
    Detect blind spots in observation coverage
    
    Args:
        observer: Meta-observer instance
        coverage_threshold: Minimum coverage ratio
        
    Returns:
        List of blind spot positions
    """
    root = observer.root
    
    # Track observed positions
    observed = set()
    for event in observer.observation_history:
        observed.add(event['position'])
    
    # Find gaps in coverage
    blind_spots = []
    
    # Check coverage in windows
    window_size = max(1, root // 20)
    step_size = max(1, window_size // 2)
    for start in range(2, root - window_size + 1, step_size):
        window_observed = sum(1 for p in range(start, start + window_size) if p in observed)
        coverage = window_observed / window_size
        
        if coverage < coverage_threshold:
            # This window is a blind spot
            center = start + window_size // 2
            if center not in observer.blind_spots:
                blind_spots.append(center)
    
    # Add positions with consistently low coherence
    blind_spots.extend(list(observer.blind_spots))
    
    return sorted(list(set(blind_spots)))

def create_meta_coherence_field(n: int, observers: List[MetaObserver]) -> Dict[int, float]:
    """
    Create meta-coherence field from multiple observers
    
    Args:
        n: Number being factored
        observers: List of meta-observers
        
    Returns:
        Dictionary mapping position to meta-coherence
    """
    root = int(math.isqrt(n))
    field = {}
    
    # Sample positions
    for x in range(2, root + 1):
        meta_coherences = []
        
        for observer in observers:
            mc = observer.get_meta_coherence(x)
            if mc > 0:
                meta_coherences.append(mc)
        
        if meta_coherences:
            # Combine meta-coherences
            field[x] = sum(meta_coherences) / len(meta_coherences)
        else:
            field[x] = 0.0
    
    return field

def analyze_axiom_interference(observers: List[MetaObserver]) -> Dict[Tuple[str, str], float]:
    """
    Analyze interference patterns between axioms
    
    Args:
        observers: List of meta-observers
        
    Returns:
        Dictionary of (axiom1, axiom2) -> interference strength
    """
    interference = defaultdict(float)
    
    for observer in observers:
        # Look for axiom transitions
        for i in range(1, len(observer.observation_history)):
            prev = observer.observation_history[i-1]
            curr = observer.observation_history[i]
            
            if prev['axiom'] != curr['axiom']:
                # Measure coherence change at transition
                coh_change = abs(curr['coherence'] - prev['coherence'])
                
                # High change = high interference
                pair = tuple(sorted([prev['axiom'], curr['axiom']]))
                interference[pair] += coh_change
    
    # Normalize by count
    counts = defaultdict(int)
    for observer in observers:
        for i in range(1, len(observer.observation_history)):
            prev = observer.observation_history[i-1]
            curr = observer.observation_history[i]
            if prev['axiom'] != curr['axiom']:
                pair = tuple(sorted([prev['axiom'], curr['axiom']]))
                counts[pair] += 1
    
    for pair in interference:
        if counts[pair] > 0:
            interference[pair] /= counts[pair]
    
    return dict(interference)
