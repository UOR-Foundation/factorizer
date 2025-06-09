"""
Interference Analyzer - Prime×Fibonacci wave interference detection

Implements interference pattern analysis for the Prime Sieve.
Finds extrema in wave interference that indicate factor positions.
"""

import math
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass

# Import dependencies
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from axiom3.interference import prime_fib_interference, interference_extrema
from axiom3.interference import identify_resonance_source, interference_gradient
from axiom2.fibonacci_core import PHI, fib
from axiom1.prime_core import primes_up_to


@dataclass
class InterferencePoint:
    """Represents a point in the interference pattern"""
    position: int
    amplitude: float
    is_extremum: bool
    extremum_type: str  # 'maximum', 'minimum', or 'none'
    resonance_source: Tuple[int, int]  # (prime, fibonacci) causing resonance


class InterferenceAnalyzer:
    """
    Wave interference pattern analyzer for factor detection.
    
    Key features:
    - Prime×Fibonacci wave generation and combination
    - Extrema detection with adaptive algorithms
    - Resonance source identification
    - Gradient-based refinement
    """
    
    def __init__(self, n: int):
        """
        Initialize interference analyzer for number n.
        
        Args:
            n: Number to be factored
        """
        self.n = n
        self.bit_length = n.bit_length()
        self.sqrt_n = int(math.isqrt(n))
        
        # Generate wave components
        self.primes = self._generate_wave_primes()
        self.fibonacci_numbers = self._generate_wave_fibonacci()
        
        # Cache for interference calculations
        self.interference_cache: Dict[int, float] = {}
        self.extrema_cache: Optional[List[InterferencePoint]] = None
        
    def _generate_wave_primes(self) -> List[int]:
        """
        Generate primes for wave interference based on problem size.
        
        Returns:
            List of primes for wave generation
        """
        if self.bit_length < 128:
            # Use first 10 primes for small numbers
            return primes_up_to(30)[:10]
        elif self.bit_length < 512:
            # Use fewer primes for medium numbers
            return primes_up_to(20)[:7]
        else:
            # Minimal primes for large numbers
            return [2, 3, 5, 7, 11]
    
    def _generate_wave_fibonacci(self) -> List[int]:
        """
        Generate Fibonacci numbers for wave interference.
        
        Returns:
            List of Fibonacci numbers for wave generation
        """
        fibs = []
        k = 2
        
        # Limit based on problem size
        max_fibs = 10 if self.bit_length < 128 else 5
        
        while len(fibs) < max_fibs:
            f = fib(k)
            if f > self.sqrt_n:
                break
            fibs.append(f)
            k += 1
        
        return fibs
    
    def calculate_interference(self, x: int) -> float:
        """
        Calculate interference amplitude at position x.
        
        Args:
            x: Position to evaluate
            
        Returns:
            Interference amplitude
        """
        if x in self.interference_cache:
            return self.interference_cache[x]
        
        # Prime wave component
        prime_amp = 0.0
        for p in self.primes:
            prime_amp += math.cos(2 * math.pi * p * x / self.n)
        
        # Fibonacci wave component (scaled by golden ratio)
        fib_amp = 0.0
        for f in self.fibonacci_numbers:
            fib_amp += math.cos(2 * math.pi * f * x / (self.n * PHI))
        
        # Interference is the product
        interference = prime_amp * fib_amp
        
        # Cache result
        if len(self.interference_cache) < 50000:
            self.interference_cache[x] = interference
        
        return interference
    
    def generate_interference_pattern(self, search_range: Optional[Tuple[int, int]] = None) -> Dict[int, float]:
        """
        Generate full interference pattern over search range.
        
        Args:
            search_range: (start, end) range (None for adaptive range)
            
        Returns:
            Dictionary mapping positions to interference values
        """
        if search_range is None:
            # Adaptive range based on problem size
            if self.bit_length < 128:
                search_range = (2, min(10000, self.sqrt_n))
            else:
                # Sparse sampling for large numbers
                search_range = (2, min(1000, self.sqrt_n))
        
        pattern = {}
        
        # Adaptive step size
        step = 1
        if search_range[1] - search_range[0] > 10000:
            step = (search_range[1] - search_range[0]) // 10000
        
        for x in range(search_range[0], search_range[1] + 1, step):
            pattern[x] = self.calculate_interference(x)
        
        return pattern
    
    def find_extrema(self, pattern: Optional[Dict[int, float]] = None) -> List[InterferencePoint]:
        """
        Find extrema (peaks and valleys) in interference pattern.
        
        Args:
            pattern: Pre-computed pattern (None to generate)
            
        Returns:
            List of extrema points
        """
        if self.extrema_cache is not None:
            return self.extrema_cache
        
        if pattern is None:
            pattern = self.generate_interference_pattern()
        
        extrema = []
        sorted_positions = sorted(pattern.keys())
        
        for i in range(1, len(sorted_positions) - 1):
            prev_pos = sorted_positions[i - 1]
            curr_pos = sorted_positions[i]
            next_pos = sorted_positions[i + 1]
            
            prev_val = pattern[prev_pos]
            curr_val = pattern[curr_pos]
            next_val = pattern[next_pos]
            
            # Check for local maximum
            if curr_val > prev_val and curr_val > next_val:
                # Find resonance source
                res_source = self._identify_resonance_source(curr_pos)
                
                extrema.append(InterferencePoint(
                    position=curr_pos,
                    amplitude=curr_val,
                    is_extremum=True,
                    extremum_type='maximum',
                    resonance_source=res_source
                ))
            
            # Check for local minimum
            elif curr_val < prev_val and curr_val < next_val:
                res_source = self._identify_resonance_source(curr_pos)
                
                extrema.append(InterferencePoint(
                    position=curr_pos,
                    amplitude=curr_val,
                    is_extremum=True,
                    extremum_type='minimum',
                    resonance_source=res_source
                ))
        
        # Sort by absolute amplitude (strongest extrema first)
        extrema.sort(key=lambda x: abs(x.amplitude), reverse=True)
        
        # Cache results
        self.extrema_cache = extrema
        
        return extrema
    
    def _identify_resonance_source(self, x: int) -> Tuple[int, int]:
        """
        Identify which prime and Fibonacci pair creates strongest resonance.
        
        Args:
            x: Position to analyze
            
        Returns:
            Tuple of (prime, fibonacci) creating resonance
        """
        best_resonance = 0.0
        best_prime = self.primes[0] if self.primes else 2
        best_fib = self.fibonacci_numbers[0] if self.fibonacci_numbers else 2
        
        for p in self.primes:
            for f in self.fibonacci_numbers:
                # Calculate resonance strength
                prime_phase = (2 * math.pi * p * x / self.n) % (2 * math.pi)
                fib_phase = (2 * math.pi * f * x / (self.n * PHI)) % (2 * math.pi)
                
                # Resonance when phases align
                phase_diff = abs(prime_phase - fib_phase)
                resonance = math.cos(phase_diff)
                
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_prime = p
                    best_fib = f
        
        return (best_prime, best_fib)
    
    def calculate_gradient(self, x: int, delta: int = 1) -> float:
        """
        Calculate interference gradient at position x.
        
        Args:
            x: Position to evaluate
            delta: Step size for gradient
            
        Returns:
            Gradient value
        """
        if x - delta < 2 or x + delta > self.sqrt_n:
            return 0.0
        
        # Use cached or calculate interference values
        val_minus = self.calculate_interference(x - delta)
        val_plus = self.calculate_interference(x + delta)
        
        return (val_plus - val_minus) / (2 * delta)
    
    def find_gradient_peaks(self, threshold: float = 0.5) -> List[int]:
        """
        Find positions with high gradient magnitude.
        
        Args:
            threshold: Minimum gradient magnitude
            
        Returns:
            List of positions with strong gradients
        """
        peaks = []
        
        # Sample positions based on problem size
        if self.bit_length < 128:
            sample_step = 1
        else:
            sample_step = max(1, self.sqrt_n // 1000)
        
        for x in range(2, self.sqrt_n + 1, sample_step):
            grad = abs(self.calculate_gradient(x))
            if grad >= threshold:
                peaks.append(x)
        
        return peaks
    
    def interference_sieve(self, candidates: Optional[Set[int]] = None,
                          extrema_count: int = 50) -> Set[int]:
        """
        Apply interference-based sieving to filter candidates.
        
        Args:
            candidates: Initial candidate set (None to generate)
            extrema_count: Number of top extrema to consider
            
        Returns:
            Filtered candidates passing interference tests
        """
        # Find extrema
        extrema = self.find_extrema()
        
        if candidates is None:
            # Start with extrema positions
            candidates = {e.position for e in extrema[:extrema_count]}
            
            # Add gradient peaks
            gradient_peaks = self.find_gradient_peaks()
            candidates.update(gradient_peaks[:20])
        
        # Filter based on interference characteristics
        filtered = set()
        
        # Calculate mean and std of interference
        pattern = {x: self.calculate_interference(x) for x in candidates}
        if pattern:
            values = list(pattern.values())
            mean_val = sum(values) / len(values)
            std_val = math.sqrt(sum((v - mean_val)**2 for v in values) / len(values))
            
            # Keep positions with significant interference
            threshold = mean_val + 0.5 * std_val
            
            for x in candidates:
                if abs(pattern[x]) >= threshold:
                    filtered.add(x)
        
        return filtered
    
    def refine_position(self, x: int, max_steps: int = 10) -> int:
        """
        Refine position using gradient ascent on interference.
        
        Args:
            x: Starting position
            max_steps: Maximum refinement steps
            
        Returns:
            Refined position
        """
        current = x
        
        for _ in range(max_steps):
            grad = self.calculate_gradient(current)
            
            if abs(grad) < 0.01:  # Converged
                break
            
            # Move in gradient direction
            step = 1 if grad > 0 else -1
            next_pos = current + step
            
            # Ensure valid position
            if 2 <= next_pos <= self.sqrt_n:
                current = next_pos
            else:
                break
        
        return current
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the interference analyzer.
        
        Returns:
            Dictionary with analyzer statistics
        """
        extrema = self.find_extrema()
        
        return {
            'bit_length': self.bit_length,
            'prime_count': len(self.primes),
            'fibonacci_count': len(self.fibonacci_numbers),
            'interference_cache_size': len(self.interference_cache),
            'extrema_count': len(extrema),
            'max_amplitude': max(abs(e.amplitude) for e in extrema) if extrema else 0
        }
