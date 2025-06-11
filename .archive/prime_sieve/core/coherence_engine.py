"""
Coherence Engine - Scalable spectral analysis and coherence fields

Implements the foundation of Axiom 3 for the Prime Sieve.
Generates multi-resolution coherence fields that identify factor
relationships through spectral signature alignment.
"""

import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.interpolate import interp1d

# Import spectral analysis from axiom3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from axiom3.spectral_core import spectral_vector, binary_spectrum, modular_spectrum
from axiom3.spectral_core import digital_spectrum, harmonic_spectrum
from axiom3.coherence import coherence
from axiom2.fibonacci_core import PHI


@dataclass
class SparseField:
    """Represents a sparse coherence field with interpolation"""
    positions: Dict[int, float]  # position -> coherence value
    interpolation: str = 'cubic'
    _interpolator: Optional[object] = None
    
    def get_value(self, x: int) -> float:
        """Get coherence value at position x with interpolation"""
        if x in self.positions:
            return self.positions[x]
        
        # Build interpolator if needed
        if self._interpolator is None and len(self.positions) > 3:
            sorted_pos = sorted(self.positions.items())
            xs = [p[0] for p in sorted_pos]
            ys = [p[1] for p in sorted_pos]
            
            try:
                self._interpolator = interp1d(xs, ys, kind=self.interpolation, 
                                            bounds_error=False, fill_value=0.0)
            except:
                # Fallback to linear if cubic fails
                self._interpolator = interp1d(xs, ys, kind='linear', 
                                            bounds_error=False, fill_value=0.0)
        
        # Use interpolation
        if self._interpolator is not None:
            return float(self._interpolator(x))
        
        # Fallback: return nearest neighbor
        if self.positions:
            nearest = min(self.positions.keys(), key=lambda p: abs(p - x))
            return self.positions[nearest]
        
        return 0.0
    
    def get_peaks(self, threshold: float = 0.7) -> List[int]:
        """Find positions with coherence above threshold"""
        return [pos for pos, val in self.positions.items() if val >= threshold]


class CoherenceEngine:
    """
    Coherence field generator with adaptive scaling.
    
    Key features:
    - Sparse coherence fields for large numbers
    - Adaptive spectral feature selection
    - Multi-resolution coherence maps
    - Memory-efficient computation
    """
    
    def __init__(self, n: int):
        """
        Initialize coherence engine for number n.
        
        Args:
            n: Number to be factored
        """
        self.n = n
        self.bit_length = n.bit_length()
        self.sqrt_n = int(math.isqrt(n))
        
        # Caches
        self.spectral_cache: Dict[int, List[float]] = {}
        self.coherence_cache: Dict[Tuple[int, int], float] = {}
        
        # Pre-compute n's spectral signature
        self.n_spectrum = self._get_adaptive_spectrum(n)
        
    def _get_adaptive_spectrum(self, x: int) -> List[float]:
        """
        Get spectral signature adapted to bit length.
        
        For very large numbers, uses selective features.
        
        Args:
            x: Number to analyze
            
        Returns:
            Spectral vector
        """
        if x in self.spectral_cache:
            return self.spectral_cache[x]
        
        if self.bit_length < 128:
            # Full spectrum for smaller numbers
            spectrum = spectral_vector(x)
        elif self.bit_length < 512:
            # Selective features for medium numbers
            spectrum = self._medium_spectrum(x)
        else:
            # Minimal features for large numbers
            spectrum = self._sparse_spectrum(x)
        
        # Cache if space allows
        if len(self.spectral_cache) < 10000:
            self.spectral_cache[x] = spectrum
        
        return spectrum
    
    def _medium_spectrum(self, x: int) -> List[float]:
        """
        Medium-resolution spectrum for 128-512 bit numbers.
        
        Args:
            x: Number to analyze
            
        Returns:
            Reduced spectral vector
        """
        # Binary spectrum (first 6 features only)
        bin_spec = binary_spectrum(x)[:6]
        
        # Modular spectrum (5 primes)
        mod_spec = modular_spectrum(x, k=5)
        
        # Digital spectrum (full)
        dig_spec = digital_spectrum(x)
        
        # Harmonic spectrum (full)
        harm_spec = harmonic_spectrum(x)
        
        return bin_spec + mod_spec + dig_spec + harm_spec
    
    def _sparse_spectrum(self, x: int) -> List[float]:
        """
        Sparse spectrum for very large numbers (>512 bits).
        
        Args:
            x: Number to analyze
            
        Returns:
            Minimal spectral vector
        """
        # Only core features
        features = []
        
        # Binary density
        bits = bin(x)[2:]
        density = bits.count('1') / len(bits)
        features.append(density)
        
        # First few modular residues
        primes = [3, 5, 7, 11, 13]
        for p in primes:
            features.append((x % p) / p)
        
        # Digital root
        dig_sum = sum(int(d) for d in str(x))
        while dig_sum >= 10:
            dig_sum = sum(int(d) for d in str(dig_sum))
        features.append(dig_sum / 9)
        
        # Golden ratio phase
        log_phi = math.log(max(x, 1)) / math.log(PHI)
        phase = (log_phi % 1)
        features.append(phase)
        
        return features
    
    def calculate_coherence(self, a: int, b: int) -> float:
        """
        Calculate coherence between potential factors a and b.
        
        High coherence indicates a × b ≈ n.
        
        Args:
            a: First potential factor
            b: Second potential factor
            
        Returns:
            Coherence value in [0, 1]
        """
        # Check cache
        cache_key = (min(a, b), max(a, b))
        if cache_key in self.coherence_cache:
            return self.coherence_cache[cache_key]
        
        # Special case: if a*b == n, perfect coherence
        if a * b == self.n:
            coh = 1.0
            if len(self.coherence_cache) < 50000:
                self.coherence_cache[cache_key] = coh
            return coh
        
        # Get adaptive spectra
        sa = self._get_adaptive_spectrum(a)
        sb = self._get_adaptive_spectrum(b)
        
        # For product approximation, get spectrum of a*b
        product = a * b
        if abs(product - self.n) < self.n * 0.1:  # Within 10% of n
            sp = self._get_adaptive_spectrum(product)
        else:
            # For distant products, interpolate
            sp = [(sa[i] + sb[i]) / 2 for i in range(len(sa))]
        
        sn = self.n_spectrum
        
        # Ensure same length
        min_len = min(len(sp), len(sn))
        sp = sp[:min_len]
        sn = sn[:min_len]
        
        # Calculate coherence: exp(-||S(a*b) - S(n)||²)
        squared_distance = 0.0
        for i in range(min_len):
            diff = sp[i] - sn[i]
            squared_distance += diff * diff
        
        # Scale by proximity of product to n
        proximity = 1.0 / (1.0 + abs(product - self.n) / self.n)
        
        coh = math.exp(-squared_distance) * proximity
        
        # Cache result
        if len(self.coherence_cache) < 50000:
            self.coherence_cache[cache_key] = coh
        
        return coh
    
    def generate_dense_field(self, search_range: Tuple[int, int]) -> SparseField:
        """
        Generate dense coherence field for small numbers.
        
        Args:
            search_range: (start, end) range to analyze
            
        Returns:
            SparseField with coherence values
        """
        positions = {}
        
        for x in range(search_range[0], search_range[1] + 1):
            if x <= 1 or x > self.sqrt_n:
                continue
            
            # Calculate coherence assuming x is a factor
            if self.n % x == 0:
                other = self.n // x
                coh = self.calculate_coherence(x, other)
            else:
                # Approximate coherence
                coh = self.calculate_coherence(x, x)
            
            positions[x] = coh
        
        return SparseField(positions, interpolation='cubic')
    
    def generate_sparse_field(self, resolution: Optional[float] = None) -> SparseField:
        """
        Generate sparse coherence field for large numbers.
        
        Args:
            resolution: Sampling resolution (auto-determined if None)
            
        Returns:
            SparseField with sampled coherence values
        """
        if resolution is None:
            resolution = self._adaptive_resolution()
        
        # Golden ratio sampling positions
        positions = self._golden_ratio_sample(resolution)
        
        field_positions = {}
        for x in positions:
            if x <= 1 or x > self.sqrt_n:
                continue
            
            # Calculate coherence
            if self.n % x == 0:
                other = self.n // x
                coh = self.calculate_coherence(x, other)
            else:
                # Approximate coherence
                coh = self.calculate_coherence(x, x)
            
            field_positions[x] = coh
        
        return SparseField(field_positions, interpolation='cubic')
    
    def _adaptive_resolution(self) -> float:
        """
        Determine adaptive sampling resolution based on bit length.
        
        Returns:
            Sampling rate between 0 and 1
        """
        if self.bit_length < 128:
            return 1.0  # Full sampling
        elif self.bit_length < 512:
            return 128 / self.bit_length
        else:
            return math.log(128) / math.log(self.bit_length)
    
    def _golden_ratio_sample(self, sample_rate: float) -> List[int]:
        """
        Generate sampling positions using golden ratio spacing.
        
        Args:
            sample_rate: Fraction of positions to sample
            
        Returns:
            List of sample positions
        """
        num_samples = int(self.sqrt_n * sample_rate)
        num_samples = max(100, min(num_samples, 10000))  # Bounded
        
        positions = []
        
        # Golden ratio sequence
        for i in range(num_samples):
            # Use golden ratio to distribute points
            pos = int(2 + (self.sqrt_n - 2) * ((i * PHI) % 1))
            positions.append(pos)
        
        # Add some positions near sqrt(n)
        sqrt_region = int(self.sqrt_n ** 0.5)
        for delta in range(-10, 11):
            pos = sqrt_region + delta
            if 2 <= pos <= self.sqrt_n:
                positions.append(pos)
        
        return sorted(list(set(positions)))
    
    def generate_coherence_field(self) -> SparseField:
        """
        Generate appropriate coherence field based on n's size.
        
        Returns:
            SparseField optimized for the problem size
        """
        if self.bit_length < 128:
            # Dense field for small numbers
            return self.generate_dense_field((2, min(10000, self.sqrt_n)))
        else:
            # Sparse field for large numbers
            return self.generate_sparse_field()
    
    def coherence_sieve(self, candidates: Optional[Set[int]] = None,
                       threshold: float = 0.1) -> Set[int]:
        """
        Apply coherence-based sieving to filter candidates.
        
        Args:
            candidates: Initial candidate set (None to generate)
            threshold: Minimum coherence value (lowered for better coverage)
            
        Returns:
            Filtered candidates passing coherence test
        """
        # Generate coherence field
        field = self.generate_coherence_field()
        
        if candidates is None:
            # Start with high-coherence positions - lower threshold
            candidates = set(field.get_peaks(0.2))
        
        # Filter based on coherence
        filtered = set()
        
        for x in candidates:
            coh = field.get_value(x)
            if coh >= threshold:
                filtered.add(x)
        
        # Ensure we keep enough candidates
        if len(filtered) < 50 and candidates:
            # Add top coherence candidates
            sorted_candidates = sorted(
                candidates, 
                key=lambda x: field.get_value(x), 
                reverse=True
            )
            filtered.update(sorted_candidates[:50])
        
        return filtered
    
    def find_coherence_gradient(self, x: int, delta: int = 1) -> float:
        """
        Calculate coherence gradient at position x.
        
        Args:
            x: Position to evaluate
            delta: Step size for gradient
            
        Returns:
            Gradient value
        """
        if x - delta < 2 or x + delta > self.sqrt_n:
            return 0.0
        
        # Calculate coherence at neighboring positions
        coh_minus = self.calculate_coherence(x - delta, self.n // (x - delta) 
                                           if self.n % (x - delta) == 0 else x - delta)
        coh_plus = self.calculate_coherence(x + delta, self.n // (x + delta)
                                          if self.n % (x + delta) == 0 else x + delta)
        
        return (coh_plus - coh_minus) / (2 * delta)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the coherence engine.
        
        Returns:
            Dictionary with engine statistics
        """
        return {
            'bit_length': self.bit_length,
            'spectrum_length': len(self.n_spectrum),
            'spectral_cache_size': len(self.spectral_cache),
            'coherence_cache_size': len(self.coherence_cache),
            'adaptive_resolution': self._adaptive_resolution()
        }
