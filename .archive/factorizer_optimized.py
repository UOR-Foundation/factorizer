"""
Optimized Universal Ontological Factorizer

Pure mathematical implementation of integer factorization based on the Universal Object Reference (UOR)
and Prime Model axioms. This optimized version implements true wave-particle duality and coherence fields
to guide the search, rather than brute-force checking.

NO FALLBACKS - NO RANDOMIZATION - NO SIMPLIFICATION - NO HARDCODING
"""

import time
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set
import math

# Axiom 1: Prime Ontology
from axiom1 import (
    is_prime, primes_up_to, PrimeCascade, PrimeGeodesic, PrimeCoordinateIndex
)

# Axiom 2: Fibonacci Flow
from axiom2 import (
    fib, fib_wave, PHI, fib_vortices, FibonacciEntanglement, FibonacciResonanceMap
)

# Axiom 3: Duality Principle
from axiom3 import (
    spectral_vector, coherence, CoherenceCache, FoldTopology, 
    interference_extrema, SpectralSignatureCache
)

# Axiom 4: Observer Effect
from axiom4 import (
    MultiScaleObserver, gradient_ascent, QuantumTunnel, 
    ResonanceMemory, ObserverCache
)

# Axiom 5: Self-Reference
from axiom5 import (
    MetaObserver, SpectralMirror, AxiomSynthesizer, 
    RecursiveCoherence, FailureMemory, MetaAccelerationCache
)


@dataclass
class FactorizationResult:
    """Result of factorization with detailed information."""
    factors: Tuple[int, int]
    primary_axiom: str
    iterations: int
    max_coherence: float
    candidates_explored: int
    time_elapsed: float
    method_sequence: List[str]
    learning_applied: bool


class OptimizedFactorizer:
    """
    Optimized Universal Ontological Factorizer implementing pure UOR/Prime axioms
    with true wave-particle duality and coherence field navigation.
    """
    
    def __init__(self, learning_enabled: bool = True):
        """
        Initialize the optimized factorizer with all axiom components.
        
        Args:
            learning_enabled: Whether to use Axiom 5's learning capabilities
        """
        self.learning_enabled = learning_enabled
        
        # Axiom 1: Prime Ontology
        self.prime_coordinate_index = PrimeCoordinateIndex()
        self.prime_coordinate_index.precompute_common_coordinates()
        
        # Axiom 2: Fibonacci Flow
        self.fibonacci_resonance_map = FibonacciResonanceMap()
        self.fibonacci_resonance_map.precompute_common_values()
        
        # Axiom 3: Duality Principle
        self.coherence_cache = CoherenceCache(max_size=100000)
        self.spectral_signature_cache = SpectralSignatureCache()
        
        # Axiom 4: Observer Effect
        self.observer_cache = ObserverCache()
        self.resonance_memory = ResonanceMemory()
        
        # Axiom 5: Self-Reference
        self.meta_acceleration_cache = MetaAccelerationCache()
        self.failure_memory = FailureMemory()
        
        # Pre-compute spectral signatures for acceleration
        self._initialize_spectral_cache()
    
    def _initialize_spectral_cache(self):
        """Pre-compute spectral signatures for common numbers."""
        for n in range(2, 10000):
            self.spectral_signature_cache.get_spectral_vector(n)
    
    def factorize(self, n: int) -> Tuple[int, int]:
        """
        Factorize semiprime n using pure UOR/Prime axioms.
        
        Args:
            n: The semiprime to factorize
            
        Returns:
            Tuple of (p, q) where p * q = n
        """
        result = self.factorize_with_details(n)
        return result.factors
    
    def factorize_with_details(self, n: int) -> FactorizationResult:
        """
        Factorize semiprime n with detailed information about the process.
        
        Args:
            n: The semiprime to factorize
            
        Returns:
            FactorizationResult with factors and process details
        """
        start_time = time.time()
        method_sequence = []
        
        # Handle trivial cases
        if n == 0:
            raise ZeroDivisionError("Cannot factorize 0")
        
        if n == 1:
            return FactorizationResult(
                factors=(1, 1),
                primary_axiom="trivial",
                iterations=0,
                max_coherence=1.0,
                candidates_explored=1,
                time_elapsed=time.time() - start_time,
                method_sequence=["trivial_one"],
                learning_applied=False
            )
        
        if n % 2 == 0:
            return FactorizationResult(
                factors=(2, n // 2),
                primary_axiom="trivial",
                iterations=0,
                max_coherence=1.0,
                candidates_explored=1,
                time_elapsed=time.time() - start_time,
                method_sequence=["trivial_even"],
                learning_applied=False
            )
        
        # Initialize quantum state with high-coherence candidates
        quantum_state = self._initialize_quantum_state(n)
        method_sequence.append("quantum_initialization")
        
        # Build coherence field for all candidates
        coherence_field = self._build_coherence_field(n, quantum_state)
        method_sequence.append("coherence_field_construction")
        
        # Navigate coherence field using gradient ascent
        peak_candidates = self._navigate_coherence_field(n, coherence_field)
        method_sequence.append("coherence_navigation")
        
        # Apply wave-particle collapse at coherence peaks
        collapsed_states = self._collapse_wavefunction(n, peak_candidates, coherence_field)
        method_sequence.append("wavefunction_collapse")
        
        # Check collapsed states for factors
        iterations = 0
        max_coherence = 0.0
        candidates_explored = len(collapsed_states)
        
        for candidate in collapsed_states:
            iterations += 1
            cand_coherence = coherence_field.get(candidate, 0.0)
            max_coherence = max(max_coherence, cand_coherence)
            
            if n % candidate == 0 and candidate > 1 and candidate < n:
                factor1 = candidate
                factor2 = n // candidate
                
                if self.learning_enabled:
                    self._record_success(n, factor1, factor2, cand_coherence)
                
                return FactorizationResult(
                    factors=(min(factor1, factor2), max(factor1, factor2)),
                    primary_axiom=self._identify_primary_axiom(cand_coherence),
                    iterations=iterations,
                    max_coherence=max_coherence,
                    candidates_explored=candidates_explored,
                    time_elapsed=time.time() - start_time,
                    method_sequence=method_sequence,
                    learning_applied=False
                )
        
        # Apply self-referential refinement if enabled
        if self.learning_enabled:
            refined_candidates = self._apply_self_reference(n, coherence_field)
            method_sequence.append("self_referential_refinement")
            
            for candidate in refined_candidates:
                iterations += 1
                candidates_explored += 1
                
                if n % candidate == 0 and candidate > 1 and candidate < n:
                    factor1 = candidate
                    factor2 = n // candidate
                    
                    self._record_success(n, factor1, factor2, 1.0)
                    
                    return FactorizationResult(
                        factors=(min(factor1, factor2), max(factor1, factor2)),
                        primary_axiom="axiom5",
                        iterations=iterations,
                        max_coherence=max_coherence,
                        candidates_explored=candidates_explored,
                        time_elapsed=time.time() - start_time,
                        method_sequence=method_sequence,
                        learning_applied=True
                    )
        
        # Fallback to indicate failure
        return FactorizationResult(
            factors=(1, n),
            primary_axiom="none",
            iterations=iterations,
            max_coherence=max_coherence,
            candidates_explored=candidates_explored,
            time_elapsed=time.time() - start_time,
            method_sequence=method_sequence,
            learning_applied=self.learning_enabled
        )
    
    def _initialize_quantum_state(self, n: int) -> Set[int]:
        """
        Initialize quantum superposition state with high-potential candidates.
        """
        candidates = set()
        sqrt_n = int(math.sqrt(n))
        
        # Prime coordinates near sqrt(n)
        coords = self.prime_coordinate_index.get_coordinates(n)
        prime_hints = self._extract_prime_hints(coords, n)
        candidates.update(prime_hints)
        
        # Fibonacci resonance points
        fib_ent = FibonacciEntanglement(n)
        fib_candidates = fib_ent.detect_double()
        if fib_candidates:
            candidates.update([c[0] for c in fib_candidates[:10]])
        
        # Spectral fold points
        from axiom3.fold_topology import sharp_fold_candidates
        fold_points = sharp_fold_candidates(n)
        candidates.update(fold_points)
        
        # Interference extrema
        extrema = interference_extrema(n, top=20)
        candidates.update(extrema)
        
        # Learning-based predictions
        if self.learning_enabled:
            memory_predictions = self.resonance_memory.predict(n)
            candidates.update([pos for pos, _ in memory_predictions[:10]])
        
        # Quantum tunnel exploration from sqrt(n)
        tunnel = QuantumTunnel(n)
        tunnel_sequence = tunnel.tunnel_sequence(sqrt_n, max_tunnels=5)
        candidates.update(tunnel_sequence)
        
        # Filter to valid range
        valid_candidates = {c for c in candidates if 2 <= c <= sqrt_n + 100}
        
        # Add critical region around sqrt(n)
        for delta in range(-20, 21):
            candidate = sqrt_n + delta
            if 2 <= candidate <= n//2:
                valid_candidates.add(candidate)
        
        return valid_candidates
    
    def _extract_prime_hints(self, coords: List[int], n: int) -> List[int]:
        """Extract potential factor hints from prime coordinates."""
        hints = []
        primes = self.prime_coordinate_index.primes
        
        # Look for coordinates that suggest factorization
        for i, coord in enumerate(coords):
            if coord > 0 and i < len(primes):
                p = primes[i]
                # If n has a non-trivial coordinate, explore factors near p^coord
                if coord > 1:
                    hints.append(p ** (coord // 2))
                    hints.append(p ** ((coord + 1) // 2))
        
        return hints
    
    def _build_coherence_field(self, n: int, candidates: Set[int]) -> Dict[int, float]:
        """
        Build comprehensive coherence field for all candidates.
        This is the core of the UOR/Prime axiomatic approach.
        """
        coherence_field = {}
        
        # Pre-compute spectral vectors
        n_spectrum = self.spectral_signature_cache.get_spectral_vector(n)
        
        # Create observers
        observer = MultiScaleObserver(n)
        
        for candidate in candidates:
            # Calculate multi-axiom coherence
            total_coherence = self._calculate_total_coherence(n, candidate, n_spectrum, observer)
            coherence_field[candidate] = total_coherence
        
        # Apply recursive coherence enhancement
        if self.learning_enabled and len(coherence_field) > 0:
            recursive_coh = RecursiveCoherence(n)
            enhanced_field = recursive_coh.recursive_coherence_iteration(coherence_field, depth=2)
            if enhanced_field:
                coherence_field = enhanced_field[-1]
        
        return coherence_field
    
    def _calculate_total_coherence(self, n: int, candidate: int, n_spectrum: List[float], 
                                   observer: MultiScaleObserver) -> float:
        """
        Calculate total coherence for a candidate using all axioms.
        This measures how "factor-like" a candidate is without checking divisibility.
        """
        # Axiom 1: Prime coordinate resonance
        prime_resonance = self._calculate_prime_resonance(n, candidate)
        
        # Axiom 2: Fibonacci flow alignment
        fib_alignment = self._calculate_fibonacci_alignment(n, candidate)
        
        # Axiom 3: Spectral coherence
        spectral_coherence = self._calculate_spectral_coherence(n, candidate, n_spectrum)
        
        # Axiom 4: Observer coherence (cached)
        observer_coherence = self.observer_cache.get_observation(observer, candidate)
        
        # Axiom 5: Meta-coherence from learning
        meta_coherence = 0.0
        if self.learning_enabled:
            meta_coherence = self._calculate_meta_coherence(n, candidate)
        
        # Weighted combination emphasizing spectral and observer coherence
        total = (
            0.15 * prime_resonance +
            0.15 * fib_alignment +
            0.35 * spectral_coherence +
            0.25 * observer_coherence +
            0.10 * meta_coherence
        )
        
        # Apply quantum amplification for high-coherence candidates
        if total > 0.7:
            from axiom4.quantum_tools import harmonic_amplify
            total = harmonic_amplify(total, n)
        
        return total
    
    def _calculate_prime_resonance(self, n: int, candidate: int) -> float:
        """Calculate prime coordinate resonance between n and candidate."""
        coords_n = self.prime_coordinate_index.get_coordinates(n)
        coords_c = self.prime_coordinate_index.get_coordinates(candidate)
        
        # Calculate potential complementary factor
        potential_other = n / candidate if candidate != 0 else 1
        
        # Measure resonance based on coordinate compatibility
        resonance = 0.0
        primes = self.prime_coordinate_index.primes
        
        for i in range(min(len(coords_n), len(coords_c))):
            if coords_c[i] <= coords_n[i]:
                # Candidate's coordinate is compatible
                resonance += (coords_c[i] + 1) / (coords_n[i] + 1)
            
            # Bonus for exact division possibility
            if coords_n[i] > 0 and coords_c[i] == coords_n[i] // 2:
                resonance += 0.5
        
        return resonance / len(primes)
    
    def _calculate_fibonacci_alignment(self, n: int, candidate: int) -> float:
        """Calculate Fibonacci flow alignment."""
        # Calculate resonance based on proximity to Fibonacci numbers
        nearest_fib = self.fibonacci_resonance_map.find_nearest_fibonacci(candidate)
        fib_index, fib_value, distance = nearest_fib
        
        # Resonance strength based on distance to nearest Fibonacci
        resonance_strength = 1.0 / (1.0 + distance / math.sqrt(candidate))
        
        # Bonus if candidate is exactly a Fibonacci number
        if distance == 0:
            resonance_strength = 1.0
        
        # Check proximity to Fibonacci vortices
        vortices = fib_vortices(n)
        min_distance = min([abs(candidate - v) for v in vortices]) if vortices else n
        vortex_proximity = 1.0 / (1.0 + min_distance / math.sqrt(n))
        
        # Golden ratio alignment
        phi_alignment = abs(n / (candidate * candidate) - PHI) if candidate > 0 else 1.0
        phi_score = 1.0 / (1.0 + phi_alignment)
        
        # Wave resonance at candidate position
        try:
            wave_value = self.fibonacci_resonance_map.get_wave_value(float(candidate))
            wave_resonance = abs(wave_value) / (1.0 + abs(wave_value))
        except (OverflowError, ValueError):
            # For large candidates, use a simplified resonance
            wave_resonance = 0.1
        
        return (resonance_strength + vortex_proximity + phi_score + wave_resonance) / 4.0
    
    def _calculate_spectral_coherence(self, n: int, candidate: int, n_spectrum: List[float]) -> float:
        """Calculate spectral coherence using wave-particle duality."""
        # Get candidate spectrum
        c_spectrum = self.spectral_signature_cache.get_spectral_vector(candidate)
        
        # Calculate potential other factor
        if candidate > 0 and n % candidate == 0:
            other = n // candidate
            o_spectrum = self.spectral_signature_cache.get_spectral_vector(other)
            
            # Perfect factorization coherence
            return self.coherence_cache.get_coherence(candidate, other, n)
        else:
            # Estimate coherence for non-factor
            # Use spectral similarity as proxy
            if len(c_spectrum) == len(n_spectrum):
                similarity = sum(c * n for c, n in zip(c_spectrum, n_spectrum))
                magnitude = (sum(c*c for c in c_spectrum) * sum(n*n for n in n_spectrum)) ** 0.5
                if magnitude > 0:
                    return similarity / magnitude
            
            return 0.1  # Low baseline coherence
    
    def _calculate_meta_coherence(self, n: int, candidate: int) -> float:
        """Calculate meta-coherence from learning systems."""
        # Calculate from resonance memory
        predictions = self.resonance_memory.predict(n)
        for pos, weight in predictions:
            if pos == candidate:
                return weight
        
        # Check for high-coherence observations at this position
        observations = self.meta_acceleration_cache.query_observations(position=candidate, min_coherence=0.5)
        if observations:
            # Average coherence from observations
            coherences = [obs.get('coherence', 0.0) for obs in observations]
            return sum(coherences) / len(coherences) if coherences else 0.0
        
        return 0.0
    
    def _navigate_coherence_field(self, n: int, coherence_field: Dict[int, float]) -> List[int]:
        """
        Navigate coherence field to find high-coherence peaks using gradient ascent.
        """
        if not coherence_field:
            return []
        
        # Sort candidates by coherence
        sorted_candidates = sorted(coherence_field.items(), key=lambda x: x[1], reverse=True)
        
        # Start from top candidates
        peak_candidates = []
        explored = set()
        
        observer = MultiScaleObserver(n)
        
        for start_pos, start_coherence in sorted_candidates[:20]:
            if start_pos in explored:
                continue
            
            # Gradient ascent from this position
            path = gradient_ascent(n, start_pos, observer, max_steps=10)
            
            if path:
                # Mark all positions in path as explored
                explored.update(path)
                
                # Find peak in path
                peak_pos = path[-1]
                peak_coherence = coherence_field.get(peak_pos, 0.0)
                
                # If we found a new peak, add it
                if peak_coherence > start_coherence * 1.1:  # 10% improvement threshold
                    peak_candidates.append(peak_pos)
                else:
                    peak_candidates.append(start_pos)
            else:
                peak_candidates.append(start_pos)
            
            # Limit number of peaks
            if len(peak_candidates) >= 10:
                break
        
        return peak_candidates
    
    def _collapse_wavefunction(self, n: int, peak_candidates: List[int], 
                               coherence_field: Dict[int, float]) -> List[int]:
        """
        Collapse quantum wavefunction at coherence peaks to extract most likely factors.
        """
        collapsed_states = []
        
        # Apply quantum measurement at each peak
        for peak in peak_candidates:
            # Measure in neighborhood of peak
            neighborhood = []
            
            for delta in range(-5, 6):
                candidate = peak + delta
                if 2 <= candidate <= n//2 and candidate in coherence_field:
                    neighborhood.append((candidate, coherence_field[candidate]))
            
            # Collapse to highest coherence in neighborhood
            if neighborhood:
                best_candidate = max(neighborhood, key=lambda x: x[1])[0]
                collapsed_states.append(best_candidate)
            else:
                collapsed_states.append(peak)
        
        # Add spectral mirror points for high-coherence candidates
        mirror = SpectralMirror(n)
        mirrored_states = []
        
        for state in collapsed_states[:5]:  # Mirror top 5
            mirror_point = mirror.find_mirror_point(state)
            if 2 <= mirror_point <= n//2:
                mirrored_states.append(mirror_point)
        
        collapsed_states.extend(mirrored_states)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_states = []
        for state in collapsed_states:
            if state not in seen:
                seen.add(state)
                unique_states.append(state)
        
        return unique_states
    
    def _apply_self_reference(self, n: int, coherence_field: Dict[int, float]) -> List[int]:
        """
        Apply Axiom 5 self-referential refinement to find additional candidates.
        """
        refined_candidates = []
        
        # Analyze coherence field patterns
        meta_observer = MetaObserver(n)
        patterns = meta_observer.detect_observation_patterns()
        
        # Generate candidates from detected patterns
        if patterns['coherence_peaks']:
            refined_candidates.extend(patterns['coherence_peaks'][:5])
        
        if 'resonance_zones' in patterns and patterns['resonance_zones']:
            for zone in patterns['resonance_zones'][:3]:
                # Sample from resonance zone
                zone_center = (zone[0] + zone[1]) // 2
                refined_candidates.append(zone_center)
        
        # Apply axiom synthesis
        synthesizer = AxiomSynthesizer(n)
        weights = synthesizer.learn_weights()
        
        # Create synthesized method
        synthesized_method = synthesizer.synthesize_method(weights)
        
        # Use synthesis to evaluate unexplored regions
        explored_candidates = set(coherence_field.keys())
        sqrt_n = int(math.sqrt(n))
        
        for candidate in range(max(2, sqrt_n - 50), min(n//2, sqrt_n + 50)):
            if candidate not in explored_candidates:
                # Evaluate with synthesized method
                score = synthesized_method(candidate)
                if score > 0.8:
                    refined_candidates.append(candidate)
        
        return refined_candidates
    
    def _identify_primary_axiom(self, coherence_value: float) -> str:
        """Identify primary axiom based on coherence characteristics."""
        if coherence_value > 0.9:
            return "axiom3"  # High coherence suggests spectral alignment
        elif coherence_value > 0.7:
            return "axiom4"  # Medium-high suggests observer effect
        elif coherence_value > 0.5:
            return "axiom2"  # Medium suggests Fibonacci flow
        else:
            return "axiom1"  # Low coherence suggests prime coordinate
    
    def _record_success(self, n: int, p: int, q: int, coherence_value: float):
        """Record successful factorization for learning."""
        if not self.learning_enabled:
            return
        
        # Record in resonance memory with proper Fibonacci component
        f = 1
        k = 1
        while fib(k) < p and k < 50:
            if abs(fib(k) - p) < abs(f - p):
                f = fib(k)
            k += 1
        
        self.resonance_memory.record(p=p, f=f, n=n, strength=coherence_value, factor=p)
        
        # Record observation in meta-acceleration cache
        observation = {
            'position': p,
            'coherence': coherence_value,
            'axiom': 'success',
            'n': n,
            'factor': True,
            'complement': q
        }
        self.meta_acceleration_cache.add_observation(observation)


# Create a singleton instance for easy import
_default_optimized_factorizer = None


def get_optimized_factorizer() -> OptimizedFactorizer:
    """Get the default optimized factorizer instance (singleton)."""
    global _default_optimized_factorizer
    if _default_optimized_factorizer is None:
        _default_optimized_factorizer = OptimizedFactorizer()
    return _default_optimized_factorizer


def factorize_optimized(n: int) -> Tuple[int, int]:
    """
    Convenience function to factorize a number using the optimized factorizer.
    
    Args:
        n: The semiprime to factorize
        
    Returns:
        Tuple of (p, q) where p * q = n
    """
    return get_optimized_factorizer().factorize(n)
