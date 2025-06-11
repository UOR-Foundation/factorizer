"""
Prime Sieve - Main implementation integrating all dimensional sieves

The Prime Sieve systematically filters number space through multiple
mathematical dimensions to reveal factors at their intersection.
"""

import math
import time
from typing import Tuple, Set, List, Dict, Optional, Union
from dataclasses import dataclass

# Import all sieve components
from .prime_coordinate_system import PrimeCoordinateSystem
from .coherence_engine import CoherenceEngine
from .fibonacci_vortex import FibonacciVortex
from .interference_analyzer import InterferenceAnalyzer
from .quantum_sieve import QuantumSieve
from .meta_observer import MetaObserver, ObservationEvent, SieveStrategy


@dataclass
class PrimeSieveResult:
    """Detailed result from Prime Sieve factorization"""
    n: int
    factors: Tuple[int, int]
    time_taken: float
    method: str
    iterations: int
    candidates_tested: int
    peak_coherence: float
    dimensions_used: List[str]
    confidence: float


class PrimeSieve:
    """
    Pure combinatorial factorization engine using multi-dimensional sieving.
    
    Integrates all five axiom-based sieves:
    1. Prime Coordinate Sieve (Axiom 1)
    2. Coherence Field Sieve (Axiom 3)
    3. Fibonacci Vortex Sieve (Axiom 2)
    4. Interference Pattern Sieve (Axiom 3)
    5. Quantum Collapse Sieve (Axiom 4)
    
    With meta-observation and learning (Axiom 5)
    """
    
    def __init__(self, enable_learning: bool = True):
        """
        Initialize Prime Sieve.
        
        Args:
            enable_learning: Whether to enable meta-observation learning
        """
        self.enable_learning = enable_learning
        
        # Global meta-observer (persists across factorizations)
        self.meta_observer = MetaObserver() if enable_learning else None
        
        # Statistics
        self.total_factorizations = 0
        self.successful_factorizations = 0
        
    def factor(self, n: int) -> Tuple[int, int]:
        """
        Factor number n using Prime Sieve.
        
        Args:
            n: Number to factor (arbitrary bit length)
            
        Returns:
            Tuple of (p, q) where p * q = n
        """
        result = self.factor_with_details(n)
        return result.factors
    
    def factor_with_details(self, n: int) -> PrimeSieveResult:
        """
        Factor with detailed result information.
        
        Args:
            n: Number to factor
            
        Returns:
            PrimeSieveResult with factors and metadata
        """
        start_time = time.time()
        self.total_factorizations += 1
        
        # Quick checks
        if n <= 1:
            return self._trivial_result(n, start_time)
        
        if n % 2 == 0:
            return self._even_result(n, start_time)
        
        # Check if prime
        if self._is_prime_quick(n):
            return self._prime_result(n, start_time)
        
        # Initialize all dimensional sieves
        coord_system = PrimeCoordinateSystem(n)
        coherence_engine = CoherenceEngine(n)
        vortex_engine = FibonacciVortex(n)
        interference_analyzer = InterferenceAnalyzer(n)
        quantum_sieve = QuantumSieve(n)
        
        # Get strategy from meta-observer
        strategy = None
        if self.enable_learning and self.meta_observer:
            strategy = self.meta_observer.synthesize_strategy(n)
        
        # Phase 1: Generate initial candidates from each dimension
        candidates = self._generate_initial_candidates(
            coord_system, coherence_engine, vortex_engine, 
            interference_analyzer, n
        )
        
        # Phase 2: Apply dimensional filters with strategy weights
        filtered_candidates = self._apply_dimensional_sieves(
            candidates, coord_system, coherence_engine, 
            vortex_engine, interference_analyzer, strategy
        )
        
        # Phase 3: Quantum collapse refinement
        refined_candidates = quantum_sieve.quantum_collapse(filtered_candidates)
        
        # Phase 4: Check candidates for factors
        result = self._check_candidates(
            refined_candidates, n, coherence_engine, start_time
        )
        
        if result.factors != (1, n):
            self.successful_factorizations += 1
            
            # Record success
            if self.enable_learning and self.meta_observer:
                event = ObservationEvent(
                    position=result.factors[0],
                    method=result.method,
                    score=result.confidence,
                    found_factor=True,
                    bit_length=n.bit_length(),
                    timestamp=time.time()
                )
                self.meta_observer.observe(event)
        else:
            # Learn from failure
            if self.enable_learning and self.meta_observer:
                self.meta_observer.learn_from_failure(n, list(refined_candidates)[:20])
        
        return result
    
    def _generate_initial_candidates(self, coord_system: PrimeCoordinateSystem,
                                   coherence_engine: CoherenceEngine,
                                   vortex_engine: FibonacciVortex,
                                   interference_analyzer: InterferenceAnalyzer,
                                   n: int) -> Set[int]:
        """Generate initial candidates from all dimensions."""
        candidates = set()
        sqrt_n = int(math.isqrt(n))
        
        # 1. Prime coordinate candidates - expand search range
        search_limit = min(sqrt_n, 100000)  # Much larger range
        coord_aligned = coord_system.find_aligned_positions(
            (2, search_limit)
        )
        candidates.update(a.position for a in coord_aligned[:500])  # Take more candidates
        
        # 2. Coherence field peaks - lower threshold
        coherence_field = coherence_engine.generate_coherence_field()
        candidates.update(coherence_field.get_peaks(threshold=0.3))  # Lower threshold
        
        # 3. Fibonacci vortex centers - take more
        vortex_centers = vortex_engine.generate_vortex_centers()
        candidates.update(v.position for v in vortex_centers[:200])  # More vortices
        
        # 4. Interference extrema - take more
        extrema = interference_analyzer.find_extrema()
        candidates.update(e.position for e in extrema[:200])  # More extrema
        
        # 5. Special positions based on n's characteristics
        # Near sqrt(n) - CRITICAL for close factors
        sqrt_region = int(sqrt_n)
        # Add positions around sqrt(n), including slightly above
        for delta in range(-100, 101):  # Much wider range around sqrt
            pos = sqrt_region + delta
            if 2 <= pos <= int(sqrt_n * 1.1):  # Allow 10% above sqrt(n)
                candidates.add(pos)
        
        # Also check near the square root of sqrt(n)
        sqrt_sqrt_region = int(sqrt_n ** 0.5)
        for delta in range(-50, 51):
            pos = sqrt_sqrt_region + delta
            if 2 <= pos <= sqrt_n:
                candidates.add(pos)
        
        # Golden ratio positions
        golden_pos = int(sqrt_n / coherence_engine.n_spectrum[0] if coherence_engine.n_spectrum else 1.618)
        if 2 <= golden_pos <= sqrt_n:
            candidates.add(golden_pos)
        
        # 6. Add systematic sampling for coverage
        # Sample positions logarithmically
        if sqrt_n > 1000:
            log_base = 1.1
            pos = 2
            while pos <= sqrt_n:
                candidates.add(int(pos))
                pos *= log_base
        
        # 7. Add positions based on small prime factors
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            if n % p == 0:
                # Add multiples of this prime
                for k in range(1, min(100, sqrt_n // p)):
                    candidates.add(p * k)
        
        # 8. Add positions near factors of numbers close to n
        for offset in [-2, -1, 1, 2]:
            nearby = n + offset
            for p in small_primes:
                if nearby % p == 0 and p <= sqrt_n:
                    candidates.add(p)
                    candidates.add(nearby // p if nearby // p <= sqrt_n else 0)
        
        # Remove invalid candidates
        candidates.discard(0)
        candidates.discard(1)
        # Allow candidates slightly above sqrt(n) for close factor pairs
        candidates = {c for c in candidates if c <= int(sqrt_n * 1.1)}
        
        return candidates
    
    def _apply_dimensional_sieves(self, candidates: Set[int],
                                coord_system: PrimeCoordinateSystem,
                                coherence_engine: CoherenceEngine,
                                vortex_engine: FibonacciVortex,
                                interference_analyzer: InterferenceAnalyzer,
                                strategy: Optional[SieveStrategy]) -> Set[int]:
        """Apply all dimensional sieves with optional strategy weights."""
        # Default weights if no strategy
        if strategy is None:
            weights = {
                'coordinate': 0.2,
                'coherence': 0.2,
                'vortex': 0.2,
                'interference': 0.2,
                'quantum': 0.2
            }
        else:
            weights = strategy.weights
        
        # Apply each sieve and collect results
        sieve_results = {}
        
        # Coordinate sieve
        coord_filtered = coord_system.coordinate_sieve(candidates)
        sieve_results['coordinate'] = coord_filtered
        
        # Coherence sieve
        coherence_filtered = coherence_engine.coherence_sieve(candidates)
        sieve_results['coherence'] = coherence_filtered
        
        # Vortex sieve
        vortex_filtered = vortex_engine.fibonacci_vortex_sieve(candidates)
        sieve_results['vortex'] = vortex_filtered
        
        # Interference sieve
        interference_filtered = interference_analyzer.interference_sieve(candidates)
        sieve_results['interference'] = interference_filtered
        
        # Combine results with weights
        scored_candidates = {}
        
        for sieve_name, filtered_set in sieve_results.items():
            weight = weights.get(sieve_name, 0.2)
            for candidate in filtered_set:
                if candidate not in scored_candidates:
                    scored_candidates[candidate] = 0.0
                scored_candidates[candidate] += weight
        
        # Select candidates above threshold - much lower threshold
        threshold = 0.1  # Candidate must pass at least 10% weighted sieves
        filtered = {c for c, score in scored_candidates.items() if score >= threshold}
        
        # Ensure we have enough candidates
        min_candidates = min(100, len(candidates))  # Keep more candidates
        if len(filtered) < min_candidates:
            # Add top scored candidates
            sorted_candidates = sorted(scored_candidates.items(), 
                                     key=lambda x: x[1], reverse=True)
            for candidate, _ in sorted_candidates[:min_candidates]:
                filtered.add(candidate)
        
        return filtered
    
    def _check_candidates(self, candidates: List[int], n: int,
                        coherence_engine: CoherenceEngine,
                        start_time: float) -> PrimeSieveResult:
        """Check candidates for actual factors."""
        iterations = 0
        candidates_tested = 0
        peak_coherence = 0.0
        
        # Order candidates by meta-observer suggestion if available
        if self.enable_learning and self.meta_observer:
            ordered_candidates = self.meta_observer.suggest_exploration_order(
                n, set(candidates)
            )
        else:
            # Default ordering - prioritize near sqrt(n) then by coherence
            sqrt_n = int(math.isqrt(n))
            
            # Calculate distance from sqrt(n) and coherence for each candidate
            def candidate_score(x):
                # Distance factor - closer to sqrt(n) is better
                distance = abs(x - sqrt_n) / sqrt_n
                distance_score = 1.0 / (1.0 + distance)
                
                # Coherence factor
                if n % x == 0:
                    coherence_score = coherence_engine.calculate_coherence(x, n // x)
                else:
                    coherence_score = coherence_engine.calculate_coherence(x, x) * 0.5
                
                # Combined score - weight distance heavily for balanced factors
                return distance_score * 2.0 + coherence_score
            
            ordered_candidates = sorted(
                candidates,
                key=candidate_score,
                reverse=True
            )
        
        # Check each candidate
        for candidate in ordered_candidates:
            iterations += 1
            candidates_tested += 1
            
            if candidate <= 1:
                continue
            
            # Check if it's a factor
            if n % candidate == 0:
                other = n // candidate
                
                # Ensure we have valid factors
                if other >= 1 and candidate * other == n:
                    # Calculate coherence
                    coherence = coherence_engine.calculate_coherence(candidate, other)
                    peak_coherence = max(peak_coherence, coherence)
                    
                    # Determine which method found it
                    method = self._determine_method(candidate, n)
                    
                    return PrimeSieveResult(
                        n=n,
                        factors=(min(candidate, other), max(candidate, other)),
                        time_taken=time.time() - start_time,
                        method=method,
                        iterations=iterations,
                        candidates_tested=candidates_tested,
                        peak_coherence=peak_coherence,
                        dimensions_used=['coordinate', 'coherence', 'vortex', 
                                       'interference', 'quantum'],
                        confidence=coherence
                    )
        
        # No factor found
        return PrimeSieveResult(
            n=n,
            factors=(1, n),
            time_taken=time.time() - start_time,
            method='none',
            iterations=iterations,
            candidates_tested=candidates_tested,
            peak_coherence=peak_coherence,
            dimensions_used=['coordinate', 'coherence', 'vortex', 
                           'interference', 'quantum'],
            confidence=0.0
        )
    
    def _determine_method(self, factor: int, n: int) -> str:
        """Determine which sieve method likely found the factor."""
        # Simple heuristic based on factor characteristics
        sqrt_n = int(math.isqrt(n))
        
        # Check if near Fibonacci
        fib_dist = self._min_fibonacci_distance(factor)
        if fib_dist < factor * 0.1:
            return 'vortex'
        
        # Check if high coordinate alignment
        if factor in [2, 3, 5, 7, 11, 13]:
            return 'coordinate'
        
        # Check if near sqrt
        if abs(factor - int(sqrt_n ** 0.5)) < 10:
            return 'coherence'
        
        # Default
        return 'quantum'
    
    def _min_fibonacci_distance(self, x: int) -> int:
        """Find minimum distance to a Fibonacci number."""
        a, b = 1, 1
        min_dist = abs(x - 1)
        
        while b < x * 2:
            a, b = b, a + b
            min_dist = min(min_dist, abs(x - b))
        
        return min_dist
    
    def _is_prime_quick(self, n: int) -> bool:
        """Quick primality check for small numbers."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Check small primes
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        # For larger numbers, assume composite (will be factored)
        return False
    
    def _trivial_result(self, n: int, start_time: float) -> PrimeSieveResult:
        """Return result for trivial cases."""
        return PrimeSieveResult(
            n=n,
            factors=(1, n),
            time_taken=time.time() - start_time,
            method='trivial',
            iterations=0,
            candidates_tested=0,
            peak_coherence=1.0,
            dimensions_used=[],
            confidence=1.0
        )
    
    def _even_result(self, n: int, start_time: float) -> PrimeSieveResult:
        """Return result for even numbers."""
        return PrimeSieveResult(
            n=n,
            factors=(2, n // 2),
            time_taken=time.time() - start_time,
            method='even',
            iterations=0,
            candidates_tested=0,
            peak_coherence=1.0,
            dimensions_used=[],
            confidence=1.0
        )
    
    def _prime_result(self, n: int, start_time: float) -> PrimeSieveResult:
        """Return result for prime numbers."""
        return PrimeSieveResult(
            n=n,
            factors=(1, n),
            time_taken=time.time() - start_time,
            method='prime',
            iterations=0,
            candidates_tested=0,
            peak_coherence=1.0,
            dimensions_used=[],
            confidence=1.0
        )
    
    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Get Prime Sieve statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {
            'total_factorizations': self.total_factorizations,
            'successful_factorizations': self.successful_factorizations,
            'success_rate': self.successful_factorizations / max(1, self.total_factorizations)
        }
        
        if self.enable_learning and self.meta_observer:
            stats['meta_observer'] = self.meta_observer.get_statistics()
        
        return stats
    
    def save_learning_state(self, filepath: str):
        """Save meta-observer learning state."""
        if self.enable_learning and self.meta_observer:
            self.meta_observer.save_state(filepath)
    
    def load_learning_state(self, filepath: str):
        """Load meta-observer learning state."""
        if self.enable_learning and self.meta_observer:
            self.meta_observer.load_state(filepath)
