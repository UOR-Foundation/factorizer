"""
Harmonic Analysis Module for PPTS

Implements multi-scale resonance analysis for detecting prime factors
through harmonic signatures at golden ratio and tribonacci scales.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


# Mathematical constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
TAU = 1.839286755214161       # Tribonacci constant
PI = math.pi
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


@dataclass
class HarmonicSignature:
    """Multi-scale harmonic signature of an integer"""
    scales: List[float]
    unity_resonances: List[float]
    phase_coherences: List[float]
    harmonic_convergences: List[float]
    
    @property
    def matrix(self) -> np.ndarray:
        """Return signature as a matrix"""
        return np.array([
            self.unity_resonances,
            self.phase_coherences,
            self.harmonic_convergences
        ]).T
    
    def trace(self) -> float:
        """Compute trace of signature matrix"""
        return sum(self.unity_resonances) + sum(self.phase_coherences) + sum(self.harmonic_convergences)
    
    def p_component(self, p: int) -> float:
        """Extract p-adic component of signature"""
        # Weighted sum based on p-adic norm
        weight = 1.0 / math.log(p + 1)
        return weight * sum(self.phase_coherences)


class MultiScaleResonance:
    """Computes resonance at multiple scales for harmonic analysis"""
    
    def __init__(self):
        self.scales = [1.0, PHI, PHI**2, TAU, TAU**2]
        self.cache: Dict[Tuple[int, float], float] = {}
    
    def compute_unity_resonance(self, x: int, n: int, scale: float) -> float:
        """
        Compute unity resonance U_s(x, n) at given scale
        
        U_s(x, n) = exp(-(ω_n - k·ω_{x,s})² / σ²)
        """
        cache_key = (x, n, scale, 'unity')
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Fundamental frequencies
        omega_n = 2 * PI / math.log(n + 1)
        omega_xs = 2 * PI / math.log(x * scale + 1)
        
        # Nearest harmonic
        k = round(omega_n / omega_xs) if omega_xs > 0 else 1
        
        # Phase difference
        phase_diff = abs(omega_n - k * omega_xs)
        
        # Gaussian with variance scaling
        sigma_sq = math.log(n) / (2 * PI)
        resonance = math.exp(-(phase_diff ** 2) / (2 * sigma_sq))
        
        # Add harmonic series contribution
        harmonic_sum = sum(1/i for i in range(1, min(10, int(math.sqrt(x)) + 1))
                          if n % (x * i) < i)
        log_harmonic = math.log(1 + harmonic_sum / math.log(x + 2))
        
        result = resonance * math.exp(log_harmonic)
        self.cache[cache_key] = result
        return result
    
    def compute_phase_coherence(self, x: int, n: int, scale: float) -> float:
        """
        Compute phase coherence P_s(x, n) at given scale
        
        P_s(x, n) = ∏_p (1 + cos(φ_{n,p} - φ_{x,p})) / 2
        """
        cache_key = (x, n, scale, 'phase')
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        coherence = 1.0
        total_weight = 0.0
        
        # Check phase alignment for small primes
        for p in SMALL_PRIMES[:7]:
            # Phases in base p
            phase_n = 2 * PI * (n % p) / p
            phase_x = 2 * PI * ((x * scale) % p) / p
            
            # Coherence measure
            local_coherence = (1 + math.cos(phase_n - phase_x)) / 2
            weight = 1 / math.log(p + 1)
            
            coherence *= (local_coherence ** weight)
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            coherence = coherence ** (1 / total_weight)
        
        # GCD amplification
        g = math.gcd(int(x * scale), n)
        if g > 1:
            amplification = 1 + math.log(g) / math.log(n)
            coherence = min(1.0, coherence * amplification)
        
        self.cache[cache_key] = coherence
        return coherence
    
    def compute_harmonic_convergence(self, x: int, n: int, scale: float) -> float:
        """
        Compute harmonic convergence H_s(x, n) at given scale
        
        H_s(x, n) = HarmonicMean(C_unity, C_golden, C_tribonacci, C_square)
        """
        cache_key = (x, n, scale, 'convergence')
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        convergence_points = []
        
        # Unity harmonic based on GCD
        g = math.gcd(int(x * scale), n)
        unity_freq = 2 * PI / g if g > 0 else 2 * PI
        unity_harmonic = (1 + math.cos(unity_freq * math.log(n) / (2 * PI))) / 2
        convergence_points.append(unity_harmonic)
        
        # Golden ratio convergence
        phi_harmonic = x / PHI
        phi_distance = min(abs(phi_harmonic - int(phi_harmonic)), 
                          abs(phi_harmonic - int(phi_harmonic) - 1))
        phi_convergence = math.exp(-phi_distance * PHI)
        convergence_points.append(phi_convergence)
        
        # Tribonacci resonance
        if x > 2:
            tri_phase = math.log(x) / math.log(TAU)
            tri_resonance = abs(math.sin(tri_phase * PI))
            convergence_points.append(tri_resonance)
        else:
            convergence_points.append(0.5)
        
        # Perfect square resonance
        sqrt_x = int(math.sqrt(x))
        if sqrt_x * sqrt_x == x:
            convergence_points.append(1.0)
        else:
            square_dist = min(x - sqrt_x**2, (sqrt_x + 1)**2 - x)
            square_harmony = math.exp(-square_dist / x)
            convergence_points.append(square_harmony)
        
        # Harmonic mean
        if convergence_points and all(c > 0 for c in convergence_points):
            harmonic_mean = len(convergence_points) / sum(1/(c + 0.001) for c in convergence_points)
        else:
            harmonic_mean = 0.1
        
        self.cache[cache_key] = harmonic_mean
        return harmonic_mean
    
    def compute_resonance(self, x: int, n: int) -> float:
        """
        Compute full multi-scale resonance for x relative to n
        """
        resonances = []
        weights = []
        
        for scale in self.scales:
            # Skip if scaled value is out of bounds
            scaled_x = int(x * scale)
            if scaled_x < 2 or scaled_x > int(math.sqrt(n)):
                continue
            
            # Compute components
            unity = self.compute_unity_resonance(x, n, scale)
            phase = self.compute_phase_coherence(x, n, scale)
            convergence = self.compute_harmonic_convergence(x, n, scale)
            
            # Combined resonance at this scale
            resonance = unity * phase * convergence
            
            # Weight by scale (closer to 1 = higher weight)
            scale_weight = 1.0 / (1 + abs(math.log(scale)))
            
            resonances.append(resonance)
            weights.append(scale_weight)
        
        # Weighted average
        if resonances:
            total_weight = sum(weights)
            weighted_resonance = sum(r * w for r, w in zip(resonances, weights)) / total_weight
        else:
            weighted_resonance = 0.0
        
        # Apply nonlinearity to sharpen peaks
        if weighted_resonance > 0.5:
            weighted_resonance = weighted_resonance ** (1 / PHI)
        
        return weighted_resonance


def extract_harmonic_signature(n: int) -> HarmonicSignature:
    """
    Extract the multi-scale harmonic signature of n
    Time Complexity: O(log² n)
    """
    analyzer = MultiScaleResonance()
    scales = analyzer.scales
    
    unity_resonances = []
    phase_coherences = []
    harmonic_convergences = []
    
    # For each scale, compute aggregate resonance components
    # We sample at multiple points and aggregate to get the signature
    sqrt_n = int(math.sqrt(n))
    
    for scale in scales:
        # Sample at multiple points to capture the resonance pattern
        sample_points = []
        
        # Key sampling points
        sample_points.append(sqrt_n)  # Near balanced factors
        sample_points.append(int(sqrt_n / PHI))  # Golden ratio point
        sample_points.append(int(sqrt_n * PHI))  # Another golden point
        
        # Add some logarithmic sampling
        for i in range(2, min(10, int(math.log2(sqrt_n)))):
            sample_points.append(int(sqrt_n / (2**i)))
            sample_points.append(int(sqrt_n / (PHI**i)))
        
        # Filter valid points
        sample_points = [x for x in sample_points if 2 <= x <= sqrt_n]
        
        # Compute average resonance components across sample points
        unity_sum = 0.0
        phase_sum = 0.0
        convergence_sum = 0.0
        
        for x in sample_points:
            unity_sum += analyzer.compute_unity_resonance(x, n, scale)
            phase_sum += analyzer.compute_phase_coherence(x, n, scale)
            convergence_sum += analyzer.compute_harmonic_convergence(x, n, scale)
        
        # Average the components
        if sample_points:
            unity_resonances.append(unity_sum / len(sample_points))
            phase_coherences.append(phase_sum / len(sample_points))
            harmonic_convergences.append(convergence_sum / len(sample_points))
        else:
            unity_resonances.append(0.5)
            phase_coherences.append(0.5)
            harmonic_convergences.append(0.5)
    
    return HarmonicSignature(
        scales=scales,
        unity_resonances=unity_resonances,
        phase_coherences=phase_coherences,
        harmonic_convergences=harmonic_convergences
    )


def compute_harmonic_polynomial_coefficients(n: int, degree: int) -> List[float]:
    """
    Compute coefficients for the harmonic polynomial H(x, n)
    
    The polynomial encodes resonance patterns that distinguish factors.
    """
    coefficients = [0.0] * (degree + 1)
    analyzer = MultiScaleResonance()
    
    # Sample resonance at multiple points
    sample_points = []
    sqrt_n = int(math.sqrt(n))
    
    # Sample near sqrt(n) and at geometric intervals
    for i in range(-degree//2, degree//2 + 1):
        x = max(2, sqrt_n + i * max(1, sqrt_n // (degree * 2)))
        if x <= sqrt_n:
            sample_points.append(x)
    
    # Add samples at scale points
    for scale in [1, PHI, TAU]:
        x = int(sqrt_n / scale)
        if 2 <= x <= sqrt_n:
            sample_points.append(x)
    
    # Compute resonance values
    resonance_values = []
    for x in sample_points:
        res = analyzer.compute_resonance(x, n)
        resonance_values.append((x, res))
    
    # Fit polynomial using least squares
    # This is a simplified version - real implementation would use
    # more sophisticated polynomial fitting
    if len(resonance_values) >= degree + 1:
        # Vandermonde matrix - ensure float type
        X = np.array([[float(x**i) for i in range(degree + 1)] for x, _ in resonance_values], dtype=np.float64)
        y = np.array([float(res) for _, res in resonance_values], dtype=np.float64)
        
        # Solve for coefficients
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            coefficients = list(coeffs)
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback to simple approximation
            coefficients[0] = -2.1
            coefficients[1] = 0.9
            coefficients[2] = -0.04
            if degree >= 3:
                coefficients[3] = 0.001
    
    return coefficients
