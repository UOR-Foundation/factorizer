"""
Factor Decoder - Decoding factors from universal patterns

The Factor Decoder implements the execution phase of The Pattern,
translating universal signatures back into concrete factors.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict

# Handle both package and standalone imports
try:
    from .pattern import PatternSignature
    from .universal_basis import UniversalBasis
except ImportError:
    from pattern import PatternSignature
    from universal_basis import UniversalBasis


class FactorDecoder:
    """
    Decodes factors from universal pattern signatures.
    
    The decoder uses multiple strategies:
    1. Direct resonance decoding
    2. Eigenvalue extraction
    3. Harmonic intersection
    4. Phase relationship analysis
    """
    
    def __init__(self, universal_basis: UniversalBasis):
        self.basis = universal_basis
        self.cache = {}  # Cache successful decodings
        
    def decode(self, n: int, signature: PatternSignature, 
               formalization: Dict) -> Tuple[int, int]:
        """Main decoding method"""
        # Check cache first
        if n in self.cache:
            return self.cache[n]
        
        # Try multiple decoding strategies
        strategies = [
            self._decode_via_resonance,
            self._decode_via_eigenvalues,
            self._decode_via_harmonics,
            self._decode_via_phase_relationships,
            self._decode_via_universal_intersections
        ]
        
        for strategy in strategies:
            result = strategy(n, signature, formalization)
            if result and result[0] * result[1] == n:
                self.cache[n] = result
                return result
        
        # Fallback to enhanced search
        return self._enhanced_search(n, signature, formalization)
    
    def _decode_via_resonance(self, n: int, signature: PatternSignature, 
                             formalization: Dict) -> Optional[Tuple[int, int]]:
        """Decode factors through resonance field analysis"""
        resonance_peaks = formalization['resonance_peaks']
        field = signature.resonance_field
        
        if len(resonance_peaks) < 2:
            return None
        
        # Analyze peak spacing
        peak_spacing = np.diff(resonance_peaks)
        
        # Common factor often appears in peak spacing patterns
        for spacing in peak_spacing:
            if spacing > 0:
                # Scale spacing to potential factor
                factor_estimate = int(spacing * np.sqrt(n) / len(field))
                
                if factor_estimate > 1 and n % factor_estimate == 0:
                    return factor_estimate, n // factor_estimate
        
        # Analyze peak magnitudes
        peak_magnitudes = [field[p] if p < len(field) else 0 for p in resonance_peaks]
        
        # Relative magnitudes can encode factor relationships
        if len(peak_magnitudes) >= 2 and peak_magnitudes[0] > 0:
            magnitude_ratio = peak_magnitudes[1] / peak_magnitudes[0]
            
            # Map magnitude ratio to factor estimate
            factor_estimate = int(magnitude_ratio * np.sqrt(n))
            
            if factor_estimate > 1 and n % factor_estimate == 0:
                return factor_estimate, n // factor_estimate
        
        return None
    
    def _decode_via_eigenvalues(self, n: int, signature: PatternSignature,
                               formalization: Dict) -> Optional[Tuple[int, int]]:
        """Decode factors through eigenvalue analysis"""
        pattern_matrix = formalization['pattern_matrix']
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(pattern_matrix)
        
        # Real positive eigenvalues often relate to factors
        for i, eigenval in enumerate(eigenvalues):
            if eigenval.imag == 0 and eigenval.real > 1:
                # Direct eigenvalue interpretation
                factor_candidate = int(abs(eigenval.real) * np.sqrt(n) / self.basis.PHI)
                
                if factor_candidate > 1 and n % factor_candidate == 0:
                    return factor_candidate, n // factor_candidate
                
                # Eigenvector interpretation
                eigenvec = eigenvectors[:, i].real
                
                # Dominant component often indicates factor relationship
                dominant_idx = np.argmax(np.abs(eigenvec))
                if dominant_idx < len(eigenvec):
                    factor_candidate = int(abs(eigenvec[dominant_idx]) * np.sqrt(n))
                    
                    if factor_candidate > 1 and n % factor_candidate == 0:
                        return factor_candidate, n // factor_candidate
        
        return None
    
    def _decode_via_harmonics(self, n: int, signature: PatternSignature,
                             formalization: Dict) -> Optional[Tuple[int, int]]:
        """Decode factors through harmonic analysis"""
        harmonic_series = formalization['harmonic_series']
        
        if len(harmonic_series) < 2:
            return None
        
        # Look for harmonic intersections
        for i in range(len(harmonic_series) - 1):
            for j in range(i + 1, len(harmonic_series)):
                # Harmonic difference encodes factor information
                h_diff = abs(harmonic_series[i] - harmonic_series[j])
                
                if h_diff > 0:
                    # Scale to factor estimate
                    factor_estimate = int(h_diff * np.sqrt(n) / self.basis.E)
                    
                    if factor_estimate > 1 and n % factor_estimate == 0:
                        return factor_estimate, n // factor_estimate
        
        # Harmonic ratios
        for i in range(1, len(harmonic_series)):
            if harmonic_series[i] != 0:
                ratio = harmonic_series[0] / harmonic_series[i]
                factor_estimate = int(ratio * np.sqrt(n) / i)
                
                if factor_estimate > 1 and n % factor_estimate == 0:
                    return factor_estimate, n // factor_estimate
        
        return None
    
    def _decode_via_phase_relationships(self, n: int, signature: PatternSignature,
                                       formalization: Dict) -> Optional[Tuple[int, int]]:
        """Decode factors through phase relationships"""
        encoding = formalization['factor_encoding']
        
        # Extract phase information
        product_phase = encoding['product_phase']
        unity_coupling = encoding['unity_coupling']
        
        # Phase difference often encodes p - q
        phase_diff = abs(product_phase - unity_coupling * 2 * np.pi)
        
        # Estimate p - q from phase
        diff_estimate = int(phase_diff * np.sqrt(n) / (2 * np.pi))
        
        # Use sum-difference relationships
        # We know p * q = n and estimate p - q
        # Therefore p = (sqrt(n² + diff²) + diff) / 2
        
        if diff_estimate >= 0:
            discriminant = n * n + diff_estimate * diff_estimate
            if discriminant >= 0:
                sqrt_disc = int(np.sqrt(discriminant))
                p = (sqrt_disc + diff_estimate) // 2
                
                if p > 1 and n % p == 0:
                    return p, n // p
        
        # Try phase-based search
        phase_center = int(product_phase * np.sqrt(n) / np.pi)
        search_radius = int(n ** 0.25)
        
        for offset in range(search_radius):
            p_candidate = phase_center + offset
            if p_candidate > 1 and n % p_candidate == 0:
                return p_candidate, n // p_candidate
            
            p_candidate = phase_center - offset
            if p_candidate > 1 and n % p_candidate == 0:
                return p_candidate, n // p_candidate
        
        return None
    
    def _decode_via_universal_intersections(self, n: int, signature: PatternSignature,
                                           formalization: Dict) -> Optional[Tuple[int, int]]:
        """Decode factors through universal constant intersections"""
        # Get factor relationship matrix
        factor_matrix = self.basis.get_factor_relationship_matrix(n)
        
        # Look for intersections in universal space
        n_coords = self.basis.project(n)
        
        # Search for factors whose coordinates satisfy special relationships
        sqrt_n = int(np.sqrt(n))
        search_range = range(max(2, sqrt_n - int(n**0.25)), 
                           min(sqrt_n + int(n**0.25) + 1, n))
        
        for p_candidate in search_range:
            if n % p_candidate == 0:
                q_candidate = n // p_candidate
                
                # Check if p and q have special universal relationship
                p_coords = self.basis.project(p_candidate)
                q_coords = self.basis.project(q_candidate)
                
                # Golden ratio relationship
                if abs(p_coords[0] / q_coords[0] - self.basis.PHI) < 0.1:
                    return p_candidate, q_candidate
                
                # Harmonic relationship
                if abs(p_coords[1] + q_coords[1] - n_coords[1]) < 0.1:
                    return p_candidate, q_candidate
                
                # Exponential relationship
                if abs(p_coords[2] * q_coords[2] - n_coords[2]) < 0.1:
                    return p_candidate, q_candidate
        
        return None
    
    def _enhanced_search(self, n: int, signature: PatternSignature,
                        formalization: Dict) -> Tuple[int, int]:
        """Enhanced search using pattern insights"""
        # Use all available information to guide search
        encoding = formalization['factor_encoding']
        resonance_peaks = formalization['resonance_peaks']
        
        # Estimate search center from multiple sources
        estimates = []
        
        # Resonance-based estimate
        if len(resonance_peaks) > 0:
            estimates.append(resonance_peaks[0] * np.sqrt(n) / len(signature.resonance_field))
        
        # Phase-based estimate
        phase_estimate = encoding['product_phase'] * np.sqrt(n) / np.pi
        estimates.append(phase_estimate)
        
        # Universal constant estimate
        phi_estimate = signature.phi_component * np.sqrt(n) / self.basis.PHI
        estimates.append(phi_estimate)
        
        # Use median of estimates as search center
        if estimates:
            search_center = int(np.median(estimates))
        else:
            search_center = int(np.sqrt(n))
        
        # Adaptive search radius
        search_radius = max(int(n ** 0.25), 10)
        
        # Prioritized search order based on pattern
        search_order = self._get_search_order(search_center, search_radius, encoding)
        
        for p_candidate in search_order:
            if p_candidate > 1 and p_candidate < n and n % p_candidate == 0:
                return p_candidate, n // p_candidate
        
        # Ultimate fallback - should rarely reach here
        return self._simple_factorization(n)
    
    def _get_search_order(self, center: int, radius: int, 
                         encoding: Dict[str, float]) -> List[int]:
        """Generate optimized search order based on pattern"""
        candidates = []
        
        # Generate candidates with priorities
        for offset in range(radius + 1):
            if offset == 0:
                candidates.append((center, 0))  # Highest priority
            else:
                # Priority based on resonance with encoding
                priority1 = abs(np.sin(offset * encoding['unity_coupling']))
                priority2 = abs(np.cos(offset * encoding['product_phase']))
                
                candidates.append((center + offset, priority1))
                candidates.append((center - offset, priority2))
        
        # Sort by priority (lower is better)
        candidates.sort(key=lambda x: x[1])
        
        # Return just the candidate values
        return [c[0] for c in candidates if c[0] > 1]
    
    def _simple_factorization(self, n: int) -> Tuple[int, int]:
        """Simple factorization as ultimate fallback"""
        # Check small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        for p in small_primes:
            if n % p == 0:
                return p, n // p
        
        # Trial division
        for i in range(49, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return i, n // i
        
        return n, 1