"""
Meta Observer - Self-referential learning and strategy synthesis

Implements Axiom 5 for the Prime Sieve. Observes the observation process
itself to identify patterns, learn from successes/failures, and synthesize
emergent strategies.
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

# Import dependencies
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from axiom5.meta_observer import MetaObserver as BaseMetaObserver
from axiom5.axiom_synthesis import AxiomSynthesizer
from axiom5.failure_analysis import FailureMemory


@dataclass
class ObservationEvent:
    """Record of an observation during factorization"""
    position: int
    method: str  # Which sieve/axiom was used
    score: float  # Success metric
    found_factor: bool
    bit_length: int
    timestamp: float


@dataclass
class SieveStrategy:
    """Represents a sieving strategy configuration"""
    name: str
    weights: Dict[str, float]  # Weights for each sieve dimension
    success_rate: float
    applicable_range: Tuple[int, int]  # Bit length range
    
    def apply(self, candidates: Set[int]) -> Set[int]:
        """Apply strategy weights to filter candidates"""
        # This will be implemented in the main PrimeSieve class
        return candidates


class MetaObserver:
    """
    Self-referential observation and learning system.
    
    Key features:
    - Pattern recognition across factorization attempts
    - Success/failure analysis and learning
    - Emergent strategy synthesis
    - Adaptive parameter tuning
    """
    
    def __init__(self):
        """Initialize meta-observer with learning components."""
        # Observation history
        self.observations: List[ObservationEvent] = []
        
        # Pattern database
        self.successful_patterns: Dict[str, List[ObservationEvent]] = defaultdict(list)
        self.failure_patterns: Dict[str, List[ObservationEvent]] = defaultdict(list)
        
        # Strategy synthesizer
        self.synthesizer = AxiomSynthesizer(1)  # Dummy n, will be updated
        
        # Failure memory
        self.failure_memory = FailureMemory(memory_size=1000)
        
        # Learned strategies
        self.strategies: List[SieveStrategy] = []
        
        # Performance metrics by bit length
        self.performance_by_bits: Dict[int, Dict[str, float]] = defaultdict(dict)
        
    def observe(self, event: ObservationEvent):
        """
        Record an observation event.
        
        Args:
            event: Observation to record
        """
        self.observations.append(event)
        
        # Categorize by success/failure
        if event.found_factor:
            self.successful_patterns[event.method].append(event)
            
            # Update synthesizer
            self.synthesizer.record_success([event.method], event.position)
        else:
            self.failure_patterns[event.method].append(event)
            
            # Record failure
            self.failure_memory.record_failure(
                event.position, event.position, event.method, event.score
            )
        
        # Update performance metrics
        self._update_performance_metrics(event)
    
    def _update_performance_metrics(self, event: ObservationEvent):
        """Update performance tracking for bit length."""
        bit_key = event.bit_length
        method = event.method
        
        if method not in self.performance_by_bits[bit_key]:
            self.performance_by_bits[bit_key][method] = 0.0
        
        # Exponential moving average
        alpha = 0.1
        old_score = self.performance_by_bits[bit_key][method]
        new_score = 1.0 if event.found_factor else 0.0
        self.performance_by_bits[bit_key][method] = (
            alpha * new_score + (1 - alpha) * old_score
        )
    
    def analyze_patterns(self, n: int) -> Dict[str, Any]:
        """
        Analyze observation patterns for number n.
        
        Args:
            n: Number being factored
            
        Returns:
            Pattern analysis results
        """
        bit_length = n.bit_length()
        
        # Find similar bit length observations
        similar_obs = [o for o in self.observations 
                      if abs(o.bit_length - bit_length) <= 10]
        
        if not similar_obs:
            return self._default_analysis()
        
        # Analyze success rates by method
        method_success = defaultdict(lambda: {'success': 0, 'total': 0})
        
        for obs in similar_obs:
            method_success[obs.method]['total'] += 1
            if obs.found_factor:
                method_success[obs.method]['success'] += 1
        
        # Calculate success rates
        success_rates = {}
        for method, stats in method_success.items():
            if stats['total'] > 0:
                success_rates[method] = stats['success'] / stats['total']
            else:
                success_rates[method] = 0.0
        
        # Find common successful positions (relative to sqrt)
        successful_positions = []
        for obs in similar_obs:
            if obs.found_factor:
                # Normalize position relative to problem size
                sqrt_n = int(math.isqrt(n))
                relative_pos = obs.position / sqrt_n if sqrt_n > 0 else 0
                successful_positions.append(relative_pos)
        
        return {
            'success_rates': success_rates,
            'best_method': max(success_rates.items(), key=lambda x: x[1])[0] if success_rates else None,
            'successful_positions': successful_positions,
            'total_observations': len(similar_obs)
        }
    
    def _default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when no patterns available."""
        return {
            'success_rates': {
                'coordinate': 0.5,
                'coherence': 0.5,
                'vortex': 0.5,
                'interference': 0.5,
                'quantum': 0.5
            },
            'best_method': 'coordinate',
            'successful_positions': [],
            'total_observations': 0
        }
    
    def synthesize_strategy(self, n: int) -> SieveStrategy:
        """
        Synthesize optimal strategy for number n.
        
        Args:
            n: Number to factor
            
        Returns:
            Synthesized sieving strategy
        """
        bit_length = n.bit_length()
        
        # Analyze patterns
        analysis = self.analyze_patterns(n)
        
        # Learn weights from synthesizer
        weights = self.synthesizer.learn_weights()
        
        # Adjust weights based on bit length performance
        if bit_length in self.performance_by_bits:
            perf = self.performance_by_bits[bit_length]
            
            # Normalize weights based on performance
            total_perf = sum(perf.values())
            if total_perf > 0:
                for method, score in perf.items():
                    if method in weights:
                        weights[method] = (weights[method] + score) / 2
        
        # Create strategy
        strategy = SieveStrategy(
            name=f"adaptive_{bit_length}bit",
            weights=self._normalize_weights(weights),
            success_rate=sum(analysis['success_rates'].values()) / len(analysis['success_rates']) if analysis['success_rates'] else 0.5,
            applicable_range=(max(1, bit_length - 10), bit_length + 10)
        )
        
        # Store strategy for reuse
        self.strategies.append(strategy)
        
        return strategy
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        if total > 0:
            return {k: v/total for k, v in weights.items()}
        else:
            # Equal weights as fallback
            num_methods = 5
            return {
                'coordinate': 1.0 / num_methods,
                'coherence': 1.0 / num_methods,
                'vortex': 1.0 / num_methods,
                'interference': 1.0 / num_methods,
                'quantum': 1.0 / num_methods
            }
    
    def suggest_exploration_order(self, n: int, candidates: Set[int]) -> List[int]:
        """
        Suggest order to explore candidates based on patterns.
        
        Args:
            n: Number being factored
            candidates: Candidate positions
            
        Returns:
            Ordered list of candidates
        """
        analysis = self.analyze_patterns(n)
        sqrt_n = int(math.isqrt(n))
        
        # Score each candidate based on similarity to successful patterns
        scored_candidates = []
        
        for candidate in candidates:
            score = 0.0
            
            # Check proximity to successful relative positions
            if sqrt_n > 0:
                relative_pos = candidate / sqrt_n
                for success_pos in analysis['successful_positions']:
                    distance = abs(relative_pos - success_pos)
                    score += 1.0 / (1.0 + distance)
            
            scored_candidates.append((candidate, score))
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [c[0] for c in scored_candidates]
    
    def learn_from_failure(self, n: int, tried_positions: List[int]):
        """
        Learn from failed factorization attempt.
        
        Args:
            n: Number that failed to factor
            tried_positions: Positions that were tried
        """
        bit_length = n.bit_length()
        
        # Record all failed positions
        for pos in tried_positions:
            event = ObservationEvent(
                position=pos,
                method='unknown',
                score=0.0,
                found_factor=False,
                bit_length=bit_length,
                timestamp=0  # Simplified timestamp
            )
            self.observe(event)
        
        # Analyze failure patterns
        if len(self.failure_patterns) > 100:
            # Identify common failure characteristics
            self._analyze_failure_patterns(n)
    
    def _analyze_failure_patterns(self, n: int):
        """Analyze patterns in failures to avoid them."""
        # Group failures by characteristics
        failure_groups = defaultdict(list)
        
        for method, failures in self.failure_patterns.items():
            for failure in failures:
                # Group by relative position
                sqrt_n = int(math.isqrt(n))
                if sqrt_n > 0:
                    rel_pos = failure.position / sqrt_n
                    bucket = int(rel_pos * 10)  # 10 buckets
                    failure_groups[bucket].append(failure)
        
        # Find positions to avoid
        self.positions_to_avoid = set()
        for bucket, failures in failure_groups.items():
            if len(failures) > 10:  # Consistent failure pattern
                # Avoid this relative position range
                for f in failures:
                    self.positions_to_avoid.add(f.position)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get meta-observer statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_obs = len(self.observations)
        successful_obs = sum(1 for o in self.observations if o.found_factor)
        
        return {
            'total_observations': total_obs,
            'successful_observations': successful_obs,
            'success_rate': successful_obs / total_obs if total_obs > 0 else 0,
            'learned_strategies': len(self.strategies),
            'failure_memory_size': len(self.failure_memory.failure_patterns),
            'performance_by_bits': dict(self.performance_by_bits)
        }
    
    def save_state(self, filepath: str):
        """Save meta-observer state to file."""
        state = {
            'observations': [(o.position, o.method, o.score, o.found_factor, o.bit_length) 
                           for o in self.observations],
            'strategies': [(s.name, s.weights, s.success_rate, s.applicable_range) 
                         for s in self.strategies],
            'performance_by_bits': dict(self.performance_by_bits)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load meta-observer state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore observations
            self.observations = []
            for pos, method, score, found, bits in state.get('observations', []):
                self.observations.append(ObservationEvent(
                    position=pos,
                    method=method,
                    score=score,
                    found_factor=found,
                    bit_length=bits,
                    timestamp=0
                ))
            
            # Restore strategies
            self.strategies = []
            for name, weights, success_rate, range_tuple in state.get('strategies', []):
                self.strategies.append(SieveStrategy(
                    name=name,
                    weights=weights,
                    success_rate=success_rate,
                    applicable_range=tuple(range_tuple)
                ))
            
            # Restore performance metrics
            self.performance_by_bits = defaultdict(dict)
            for bit_str, perf in state.get('performance_by_bits', {}).items():
                self.performance_by_bits[int(bit_str)] = perf
                
        except Exception as e:
            print(f"Failed to load meta-observer state: {e}")
