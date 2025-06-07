"""
Failure Analysis - Learn from what doesn't work
Tracks failure patterns and adapts strategies accordingly
"""

import math
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

# Import dependencies from other axioms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axiom2 import PHI
from axiom3 import spectral_vector, coherence, accelerated_spectral_vector, accelerated_coherence
from .meta_acceleration_cache import get_meta_cache

class FailureMemory:
    """
    Tracks and analyzes failure patterns
    """
    
    def __init__(self, memory_size: int = 100):
        """
        Initialize failure memory
        
        Args:
            memory_size: Maximum patterns to remember
        """
        self.memory_size = memory_size
        
        # Track failed positions and their characteristics
        self.failed_positions: Dict[int, Dict[str, float]] = {}
        self.failure_patterns: List[Dict] = []
        self.dead_end_regions: List[Tuple[int, int]] = []
        self.spectral_nulls: List[int] = []
        self.cache = get_meta_cache()
        
    def record_failure(self, n: int, position: int, method: str, 
                      coherence_value: float = 0.0):
        """
        Record a failure event
        
        Args:
            n: Number being factored
            position: Position that failed
            method: Method/axiom that failed
            coherence_value: Coherence at failure
        """
        # Characterize the failure
        failure = {
            'n': n,
            'position': position,
            'method': method,
            'coherence': coherence_value,
            'spectrum': accelerated_spectral_vector(position),  # Use accelerated version
            'characteristics': self._analyze_position(position, n),
            'axiom': method,  # Add axiom field for cache compatibility
            'found': False,  # Failure
            'timestamp': len(self.failure_patterns)
        }
        
        self.failure_patterns.append(failure)
        
        # Add to cache for fast queries
        self.cache.add_observation(failure)
        
        # Limit memory size
        if len(self.failure_patterns) > self.memory_size:
            self.failure_patterns = self.failure_patterns[-self.memory_size:]
        
        # Update failed positions
        if position not in self.failed_positions:
            self.failed_positions[position] = {}
        
        self.failed_positions[position][method] = coherence_value
        
        # Check for spectral null
        if coherence_value < 0.1:
            if position not in self.spectral_nulls:
                self.spectral_nulls.append(position)
    
    def _analyze_position(self, position: int, n: int) -> Dict[str, bool]:
        """
        Analyze characteristics of a position
        
        Args:
            position: Position to analyze
            n: Number being factored
            
        Returns:
            Position characteristics
        """
        chars = {
            'even': position % 2 == 0,
            'divisible_by_3': position % 3 == 0,
            'divisible_by_5': position % 5 == 0,
            'near_sqrt': abs(position - int(math.isqrt(n))) <= 5,
            'power_of_2': (position & (position - 1)) == 0 and position > 0,
            'high_bit_count': bin(position).count('1') > position.bit_length() // 2
        }
        return chars
    
    def identify_dead_ends(self, threshold: int = 5) -> List[Tuple[int, int]]:
        """
        Identify regions with many failures
        
        Args:
            threshold: Minimum failures to classify as dead end
            
        Returns:
            List of (start, end) dead end regions
        """
        if not self.failed_positions:
            return []
        
        # Sort failed positions
        sorted_fails = sorted(self.failed_positions.keys())
        
        # Find dense failure regions
        dead_ends = []
        i = 0
        
        while i < len(sorted_fails):
            # Count failures in window
            start = sorted_fails[i]
            count = 1
            j = i + 1
            
            while j < len(sorted_fails) and sorted_fails[j] - start <= 10:
                count += 1
                j += 1
            
            if count >= threshold:
                end = sorted_fails[j-1]
                dead_ends.append((start, end))
            
            i = j if count >= threshold else i + 1
        
        self.dead_end_regions = dead_ends
        return dead_ends
    
    def get_failure_probability(self, position: int) -> float:
        """
        Estimate failure probability at a position
        
        Args:
            position: Position to evaluate
            
        Returns:
            Failure probability [0, 1]
        """
        # Direct failure
        if position in self.failed_positions:
            return 0.9  # High probability
        
        # In dead end region
        for start, end in self.dead_end_regions:
            if start <= position <= end:
                return 0.8
        
        # Near failures
        min_distance = float('inf')
        for failed_pos in self.failed_positions:
            distance = abs(position - failed_pos)
            if distance < min_distance:
                min_distance = distance
        
        if min_distance < 3:
            return 0.6 / (1 + min_distance)
        
        return 0.1  # Low baseline

def analyze_failure_patterns(memory: FailureMemory) -> Dict[str, float]:
    """
    Analyze patterns in failures
    
    Args:
        memory: Failure memory
        
    Returns:
        Pattern statistics
    """
    if not memory.failure_patterns:
        return {}
    
    # Count method failures
    method_failures = defaultdict(int)
    for failure in memory.failure_patterns:
        method_failures[failure['method']] += 1
    
    # Analyze characteristics
    char_failures = defaultdict(int)
    total_failures = len(memory.failure_patterns)
    
    for failure in memory.failure_patterns:
        for char, value in failure['characteristics'].items():
            if value:
                char_failures[char] += 1
    
    # Calculate rates
    patterns = {}
    
    # Method failure rates
    for method, count in method_failures.items():
        patterns[f"method_{method}_fail_rate"] = count / total_failures
    
    # Characteristic failure rates
    for char, count in char_failures.items():
        patterns[f"char_{char}_fail_rate"] = count / total_failures
    
    # Average coherence at failure
    avg_coherence = sum(f['coherence'] for f in memory.failure_patterns) / total_failures
    patterns['avg_failure_coherence'] = avg_coherence
    
    return patterns

def detect_spectral_nulls(n: int, memory: FailureMemory, 
                         coherence_threshold: float = 0.1) -> List[int]:
    """
    Detect positions with persistently low coherence
    
    Args:
        n: Number being factored
        memory: Failure memory
        coherence_threshold: Threshold for null
        
    Returns:
        List of spectral null positions
    """
    nulls = []
    
    # Use cache to query low coherence failures
    low_coh_obs = memory.cache.query_observations(min_coherence=0.0)
    low_coh_failures = [obs for obs in low_coh_obs 
                       if obs.get('coherence', 1.0) < coherence_threshold 
                       and not obs.get('found', True)]
    
    # Get positions from low coherence failures
    for failure in low_coh_failures:
        if 'position' in failure:
            pos = failure['position']
            if pos not in nulls:
                nulls.append(pos)
    
    # Also check memory patterns if cache is empty
    if not nulls:
        for failure in memory.failure_patterns:
            if failure['coherence'] < coherence_threshold:
                pos = failure['position']
                if pos not in nulls:
                    nulls.append(pos)
    
    # Expand to nearby positions
    expanded_nulls = set(nulls)
    for null_pos in nulls:
        # Check neighbors
        for offset in [-1, 1]:
            neighbor = null_pos + offset
            if neighbor >= 2:
                # Estimate coherence using accelerated version
                if n % neighbor == 0:
                    coh = accelerated_coherence(neighbor, n // neighbor, n)
                else:
                    coh = accelerated_coherence(neighbor, neighbor, n)
                
                if coh < coherence_threshold:
                    expanded_nulls.add(neighbor)
    
    return sorted(list(expanded_nulls))

def adaptive_strategy(n: int, memory: FailureMemory,
                     base_candidates: List[int]) -> List[int]:
    """
    Adapt strategy based on failure patterns
    
    Args:
        n: Number being factored
        memory: Failure memory
        base_candidates: Initial candidates
        
    Returns:
        Adapted candidate list
    """
    root = int(math.isqrt(n))
    
    # Analyze failure patterns
    patterns = analyze_failure_patterns(memory)
    
    # Filter out high-failure positions
    adapted = []
    for candidate in base_candidates:
        fail_prob = memory.get_failure_probability(candidate)
        
        # Keep if low failure probability
        if fail_prob < 0.5:
            adapted.append(candidate)
    
    # If too many filtered out, add alternatives
    if len(adapted) < len(base_candidates) // 2:
        # Identify what characteristics tend to fail
        high_fail_chars = []
        for char, rate_key in [('even', 'char_even_fail_rate'),
                               ('divisible_by_3', 'char_divisible_by_3_fail_rate'),
                               ('divisible_by_5', 'char_divisible_by_5_fail_rate')]:
            if patterns.get(rate_key, 0) > 0.6:
                high_fail_chars.append(char)
        
        # Generate alternatives avoiding high-fail characteristics
        alternatives = []
        for x in range(2, min(root + 1, 100)):
            # Check if avoids high-fail characteristics
            avoid = True
            
            if 'even' in high_fail_chars and x % 2 == 0:
                avoid = False
            if 'divisible_by_3' in high_fail_chars and x % 3 == 0:
                avoid = False
            if 'divisible_by_5' in high_fail_chars and x % 5 == 0:
                avoid = False
            
            if avoid and x not in memory.failed_positions:
                alternatives.append(x)
        
        # Add best alternatives
        adapted.extend(alternatives[:max(5, len(base_candidates) // 4)])
    
    return sorted(list(set(adapted)))

def inverse_failure_search(n: int, memory: FailureMemory) -> List[int]:
    """
    Search positions that are opposite of failure patterns
    
    Args:
        n: Number being factored
        memory: Failure memory
        
    Returns:
        Anti-failure positions
    """
    root = int(math.isqrt(n))
    anti_positions = []
    
    # Analyze what tends to fail
    patterns = analyze_failure_patterns(memory)
    
    # Generate positions with opposite characteristics
    for x in range(2, min(root + 1, 200)):
        # Skip if already failed
        if x in memory.failed_positions:
            continue
        
        # Calculate anti-failure score
        score = 0.0
        
        # Opposite of high-failure characteristics
        chars = memory._analyze_position(x, n)
        
        for char, value in chars.items():
            fail_rate = patterns.get(f"char_{char}_fail_rate", 0.5)
            
            # If this characteristic has high failure, reward opposite
            if fail_rate > 0.6 and not value:
                score += 1.0
            elif fail_rate < 0.4 and value:
                score += 0.5
        
        # Avoid dead end regions
        in_dead_end = False
        for start, end in memory.dead_end_regions:
            if start <= x <= end:
                in_dead_end = True
                break
        
        if not in_dead_end:
            score += 2.0
        
        # Keep high-scoring positions
        if score > 2.0:
            anti_positions.append((x, score))
    
    # Sort by score
    anti_positions.sort(key=lambda t: -t[1])
    
    return [x for x, _ in anti_positions[:20]]

def failure_gradient(n: int, memory: FailureMemory) -> Dict[int, float]:
    """
    Create gradient field based on failure density
    
    Args:
        n: Number being factored
        memory: Failure memory
        
    Returns:
        Gradient field (high = away from failures)
    """
    root = int(math.isqrt(n))
    gradient = {}
    
    # Sample positions
    for x in range(2, root + 1):
        # Distance to nearest failure
        min_dist = root
        for failed in memory.failed_positions:
            dist = abs(x - failed)
            if dist < min_dist:
                min_dist = dist
        
        # Gradient is high when far from failures
        gradient[x] = min_dist / root
        
        # Reduce if in dead end
        for start, end in memory.dead_end_regions:
            if start <= x <= end:
                gradient[x] *= 0.1
                break
    
    return gradient
