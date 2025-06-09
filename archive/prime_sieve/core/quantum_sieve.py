"""
Quantum Sieve - Superposition management and wavefunction collapse

Implements quantum-inspired superposition and measurement for the Prime Sieve.
Manages candidate positions in superposition until coherence-based collapse.
"""

import math
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

# Import dependencies
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from axiom4.quantum_tools import QuantumTunnel, harmonic_amplify
from axiom4.adaptive_observer import MultiScaleObserver
from axiom3.coherence import coherence as calculate_coherence


@dataclass
class QuantumState:
    """Represents a quantum state in superposition"""
    position: int
    amplitude: complex
    probability: float
    coherence: float
    collapsed: bool = False


class QuantumSieve:
    """
    Quantum-inspired superposition and collapse mechanism.
    
    Key features:
    - Superposition creation from multiple sources
    - Coherence-weighted wavefunction collapse
    - Multi-scale observation and measurement
    - Gradient-guided state refinement
    """
    
    def __init__(self, n: int):
        """
        Initialize quantum sieve for number n.
        
        Args:
            n: Number to be factored
        """
        self.n = n
        self.bit_length = n.bit_length()
        self.sqrt_n = int(math.isqrt(n))
        
        # Multi-scale observer
        self.observer = MultiScaleObserver(n)
        
        # Quantum tunnel for escaping local minima
        self.quantum_tunnel = QuantumTunnel(n)
        
        # Cache for coherence calculations
        self.coherence_cache: Dict[int, float] = {}
        
    def create_superposition(self, candidates: Set[int]) -> Dict[int, QuantumState]:
        """
        Create quantum superposition from candidate positions.
        
        Args:
            candidates: Set of candidate positions
            
        Returns:
            Dictionary mapping positions to quantum states
        """
        superposition = {}
        
        # Calculate initial amplitudes based on position characteristics
        total_weight = 0.0
        
        for pos in candidates:
            # Initial amplitude based on proximity to sqrt(n)
            distance_factor = 1.0 / (1.0 + abs(pos - int(self.sqrt_n ** 0.5)))
            
            # Coherence factor
            coh = self._get_coherence(pos)
            
            # Combined amplitude
            amplitude = complex(math.sqrt(distance_factor * coh), 0)
            probability = abs(amplitude) ** 2
            
            total_weight += probability
            
            superposition[pos] = QuantumState(
                position=pos,
                amplitude=amplitude,
                probability=probability,
                coherence=coh,
                collapsed=False
            )
        
        # Normalize probabilities
        if total_weight > 0:
            for state in superposition.values():
                state.probability /= total_weight
                state.amplitude /= math.sqrt(total_weight)
        
        return superposition
    
    def _get_coherence(self, x: int) -> float:
        """
        Get coherence value for position x with caching.
        
        Args:
            x: Position to evaluate
            
        Returns:
            Coherence value
        """
        if x in self.coherence_cache:
            return self.coherence_cache[x]
        
        # Calculate coherence assuming x is a factor
        if self.n % x == 0:
            other = self.n // x
            coh = calculate_coherence(x, other, self.n)
        else:
            # Approximate coherence
            coh = calculate_coherence(x, x, self.n) * 0.5
        
        # Cache result
        if len(self.coherence_cache) < 10000:
            self.coherence_cache[x] = coh
        
        return coh
    
    def measure_state(self, state: QuantumState) -> float:
        """
        Perform measurement on a quantum state.
        
        Args:
            state: Quantum state to measure
            
        Returns:
            Measurement result (coherence-weighted probability)
        """
        # Multi-scale observation
        observation = self.observer.observe(state.position)
        
        # Combine with state properties
        measurement = state.probability * state.coherence * observation
        
        return measurement
    
    def collapse_step(self, superposition: Dict[int, QuantumState], 
                     temperature: float = 1.0) -> Dict[int, QuantumState]:
        """
        Perform one step of wavefunction collapse.
        
        Args:
            superposition: Current superposition state
            temperature: Collapse temperature (lower = more selective)
            
        Returns:
            Updated superposition after collapse
        """
        # Measure all states
        measurements = {}
        for pos, state in superposition.items():
            if not state.collapsed:
                measurements[pos] = self.measure_state(state)
        
        if not measurements:
            return superposition
        
        # Calculate new amplitudes based on measurements
        total_measurement = sum(measurements.values())
        
        if total_measurement > 0:
            for pos, measurement in measurements.items():
                # Update amplitude with temperature-controlled collapse
                factor = math.exp(measurement / temperature)
                new_amplitude = superposition[pos].amplitude * factor
                
                # Update state
                superposition[pos].amplitude = new_amplitude
                superposition[pos].probability = abs(new_amplitude) ** 2
                
                # Mark as collapsed if probability too low
                if superposition[pos].probability < 1e-6:
                    superposition[pos].collapsed = True
        
        # Renormalize
        self._normalize_superposition(superposition)
        
        return superposition
    
    def _normalize_superposition(self, superposition: Dict[int, QuantumState]):
        """
        Normalize superposition to ensure total probability = 1.
        
        Args:
            superposition: Superposition to normalize
        """
        total_prob = sum(s.probability for s in superposition.values() 
                        if not s.collapsed)
        
        if total_prob > 0:
            for state in superposition.values():
                if not state.collapsed:
                    state.probability /= total_prob
                    state.amplitude /= math.sqrt(total_prob)
    
    def quantum_collapse(self, candidates: Set[int], 
                        max_iterations: Optional[int] = None) -> List[int]:
        """
        Perform full quantum collapse process.
        
        Args:
            candidates: Initial candidate positions
            max_iterations: Maximum collapse iterations (adaptive if None)
            
        Returns:
            Final collapsed positions sorted by probability
        """
        # Create initial superposition
        superposition = self.create_superposition(candidates)
        
        # Determine iterations based on problem size
        if max_iterations is None:
            if self.bit_length < 128:
                max_iterations = 10
            elif self.bit_length < 512:
                max_iterations = 5
            else:
                max_iterations = 3
        
        # Perform iterative collapse
        temperature = 1.0
        for i in range(max_iterations):
            # Decrease temperature (more selective collapse)
            temperature = 1.0 / (i + 1)
            
            # Collapse step
            superposition = self.collapse_step(superposition, temperature)
            
            # Check for convergence
            active_states = [s for s in superposition.values() if not s.collapsed]
            if len(active_states) <= 10:
                break
        
        # Extract final positions
        final_positions = []
        for state in sorted(superposition.values(), 
                          key=lambda s: s.probability, 
                          reverse=True):
            if not state.collapsed:
                final_positions.append(state.position)
        
        return final_positions
    
    def quantum_tunnel_escape(self, stuck_position: int) -> List[int]:
        """
        Use quantum tunneling to escape local minima.
        
        Args:
            stuck_position: Position where search is stuck
            
        Returns:
            New positions to explore
        """
        # Generate tunnel sequence
        tunnel_positions = self.quantum_tunnel.tunnel_sequence(
            stuck_position, 
            max_tunnels=min(20, self.bit_length)
        )
        
        # Filter valid positions
        valid_positions = [p for p in tunnel_positions 
                          if 2 <= p <= self.sqrt_n]
        
        return valid_positions
    
    def harmonic_expansion(self, position: int) -> List[int]:
        """
        Generate harmonic positions from a base position.
        
        Args:
            position: Base position
            
        Returns:
            List of harmonic positions
        """
        harmonics = harmonic_amplify(self.n, position)
        
        # Filter and limit
        valid_harmonics = [h for h in harmonics 
                          if 2 <= h <= self.sqrt_n][:10]
        
        return valid_harmonics
    
    def gradient_refinement(self, position: int, 
                          coherence_gradient_fn, 
                          max_steps: int = 5) -> int:
        """
        Refine position using coherence gradient.
        
        Args:
            position: Starting position
            coherence_gradient_fn: Function to calculate gradient
            max_steps: Maximum refinement steps
            
        Returns:
            Refined position
        """
        current = position
        step_size = 1
        
        for _ in range(max_steps):
            # Calculate gradient
            grad = coherence_gradient_fn(current)
            
            if abs(grad) < 0.001:  # Converged
                break
            
            # Update position
            next_pos = current + int(step_size * grad)
            
            # Ensure valid
            if 2 <= next_pos <= self.sqrt_n:
                current = next_pos
            else:
                break
            
            # Adaptive step size
            step_size *= 0.9
        
        return current
    
    def quantum_sieve(self, candidates: Set[int]) -> Set[int]:
        """
        Apply quantum sieving to filter candidates.
        
        Args:
            candidates: Initial candidate set
            
        Returns:
            Filtered candidates after quantum collapse
        """
        # Perform quantum collapse
        collapsed_positions = self.quantum_collapse(candidates)
        
        # Add harmonics of top positions
        enhanced_positions = set(collapsed_positions)
        
        for pos in collapsed_positions[:5]:  # Top 5 positions
            harmonics = self.harmonic_expansion(pos)
            enhanced_positions.update(harmonics)
        
        # Check for stuck positions and tunnel if needed
        if len(collapsed_positions) > 0:
            top_position = collapsed_positions[0]
            tunnel_positions = self.quantum_tunnel_escape(top_position)
            enhanced_positions.update(tunnel_positions[:10])
        
        return enhanced_positions
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the quantum sieve.
        
        Returns:
            Dictionary with sieve statistics
        """
        return {
            'bit_length': self.bit_length,
            'coherence_cache_size': len(self.coherence_cache),
            'observer_scales': len(self.observer.scales),
            'quantum_tunnel_available': True
        }
