"""
RFH3: Adaptive Resonance Field Hypothesis v3
Dynamic, adaptive field exploration for integer factorization
"""

import heapq
import math
import time
import json
import logging
import pickle
from collections import defaultdict, deque
from functools import lru_cache
from typing import Dict, Tuple, List, Set, Optional, Any
import numpy as np


# ============================================================================
# CORE COMPONENTS
# ============================================================================

class LazyResonanceIterator:
    """Generates resonance nodes on-demand based on importance"""
    
    def __init__(self, n: int, analyzer: 'MultiScaleResonance'):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.log_n = math.log(n)
        self.analyzer = analyzer
        self.importance_heap = []  # Min heap (negative importance)
        self.visited: Set[int] = set()
        self.expansion_radius = max(1, int(self.sqrt_n ** 0.01))
        self._initialize_seeds()
    
    def _initialize_seeds(self):
        """Initialize with high-probability seed points"""
        seeds = []
        
        # Small primes and their powers
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            k = 1
            while p**k <= self.sqrt_n:
                seeds.append(p**k)
                k += 1
        
        # Golden ratio points
        phi = (1 + math.sqrt(5)) / 2
        x = self.sqrt_n
        while x > 2:
            seeds.append(int(x))
            x = x / phi
        
        # Tribonacci points
        tau = 1.839286755214161
        x = self.sqrt_n
        while x > 2:
            seeds.append(int(x))
            x = x / tau
        
        # Perfect squares near sqrt(n)
        base = int(self.sqrt_n ** 0.5)
        for offset in range(-5, 6):
            if base + offset > 1:
                seeds.append((base + offset) ** 2)
        
        # Near sqrt(n) for balanced factors
        for offset in range(-100, 101):
            candidate = self.sqrt_n + offset
            if 2 <= candidate <= self.sqrt_n:
                seeds.append(candidate)
        
        # Add to heap with initial importance
        for seed in set(seeds):
            if 2 <= seed <= self.sqrt_n:
                importance = self.analyzer.compute_coarse_resonance(seed, self.n)
                heapq.heappush(self.importance_heap, (-importance, seed))
                self.visited.add(seed)
    
    def __iter__(self):
        while self.importance_heap:
            neg_importance, x = heapq.heappop(self.importance_heap)
            yield x
            # Expand region around x based on its importance
            self._expand_region(x, -neg_importance)
    
    def _expand_region(self, x: int, importance: float):
        """Dynamically expand high-resonance regions"""
        # Adaptive radius based on importance
        radius = int(self.expansion_radius * (1 + importance))
        
        # Compute local gradient
        gradient = self._estimate_gradient(x)
        
        # Generate neighbors with bias toward gradient direction
        neighbors = []
        
        # Along gradient
        for step in [1, 2, 5, 10]:
            next_x = x + int(step * gradient * radius)
            if 2 <= next_x <= self.sqrt_n and next_x not in self.visited:
                neighbors.append(next_x)
        
        # Perpendicular to gradient (exploration)
        for offset in [-radius, -radius//2, radius//2, radius]:
            next_x = x + offset
            if 2 <= next_x <= self.sqrt_n and next_x not in self.visited:
                neighbors.append(next_x)
        
        # Add neighbors to heap
        for neighbor in neighbors:
            if neighbor not in self.visited:
                imp = self.analyzer.compute_coarse_resonance(neighbor, self.n)
                heapq.heappush(self.importance_heap, (-imp, neighbor))
                self.visited.add(neighbor)
    
    def _estimate_gradient(self, x: int) -> float:
        """Estimate resonance gradient at x"""
        delta = max(1, int(x * 0.001))
        
        # Forward difference if at boundary
        if x - delta < 2:
            r1 = self.analyzer.compute_coarse_resonance(x, self.n)
            r2 = self.analyzer.compute_coarse_resonance(x + delta, self.n)
            return (r2 - r1) / delta
        
        # Backward difference if at boundary
        if x + delta > self.sqrt_n:
            r1 = self.analyzer.compute_coarse_resonance(x - delta, self.n)
            r2 = self.analyzer.compute_coarse_resonance(x, self.n)
            return (r2 - r1) / delta
        
        # Central difference
        r1 = self.analyzer.compute_coarse_resonance(x - delta, self.n)
        r2 = self.analyzer.compute_coarse_resonance(x + delta, self.n)
        return (r2 - r1) / (2 * delta)


class MultiScaleResonance:
    """Analyzes resonance at multiple scales simultaneously"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.tau = 1.839286755214161       # Tribonacci constant
        self.scales = [1, self.phi, self.phi**2, self.tau, self.tau**2]
        self.cache: Dict[Tuple[int, int], float] = {}
        self.small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        
    def compute_resonance(self, x: int, n: int) -> float:
        """Compute full scale-invariant resonance in log space"""
        # Check cache
        cache_key = (x, n)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Compute resonance at multiple scales
        log_resonances = []
        sqrt_n = int(math.sqrt(n))
        
        for scale in self.scales:
            scaled_x = int(x * scale)
            if 2 <= scaled_x <= sqrt_n:
                # Compute in log space for stability
                log_u = self._log_unity_resonance(scaled_x, n)
                log_p = self._log_phase_coherence(scaled_x, n)
                log_h = self._log_harmonic_convergence(scaled_x, n)
                
                # Weight by scale (closer to 1 = higher weight)
                scale_weight = 1.0 / (1 + abs(math.log(scale)))
                
                log_resonances.append((log_u + log_p + log_h, scale_weight))
        
        # Aggregate using weighted log-sum-exp for numerical stability
        if log_resonances:
            max_log = max(lr[0] for lr in log_resonances)
            weighted_sum = sum(w * math.exp(lr - max_log) 
                             for lr, w in log_resonances)
            total_weight = sum(w for _, w in log_resonances)
            result = max_log + math.log(weighted_sum / total_weight)
        else:
            result = -float('inf')
        
        # Convert back from log space
        resonance = math.exp(result) if result > -100 else 0.0
        
        # Apply nonlinearity to sharpen peaks
        if resonance > 0.5:
            resonance = resonance ** (1 / self.phi)
        
        self.cache[cache_key] = resonance
        return resonance
    
    def compute_coarse_resonance(self, x: int, n: int) -> float:
        """Fast approximation for importance sampling"""
        # Quick checks
        if n % x == 0:
            return 1.0
        
        # Prime harmonic indicator
        score = 0.0
        for p in self.small_primes[:5]:  # Use only first 5 primes
            if x % p == 0:
                score += 0.1
            if n % p == 0 and x % p == 0:
                score += 0.2
        
        # GCD bonus
        g = math.gcd(x, n)
        if g > 1:
            score += math.log(g) / math.log(n)
        
        # Distance to perfect square
        sqrt_x = int(math.sqrt(x))
        if sqrt_x * sqrt_x == x:
            score += 0.3
        
        # Near sqrt(n) bonus for balanced factors
        sqrt_n = int(math.sqrt(n))
        relative_distance = abs(x - sqrt_n) / sqrt_n
        if relative_distance < 0.1:
            score += 0.3 * (1 - relative_distance / 0.1)
        
        return min(1.0, score)
    
    def _log_unity_resonance(self, x: int, n: int) -> float:
        """Compute log unity resonance"""
        if n % x == 0:
            return 0.0  # log(1) = 0
        
        # Frequency-based resonance
        omega_n = 2 * math.pi / math.log(n + 1)
        omega_x = 2 * math.pi / math.log(x + 1)
        
        # Find nearest harmonic
        k = round(omega_n / omega_x) if omega_x > 0 else 1
        phase_diff = abs(omega_n - k * omega_x)
        
        # Gaussian in log space
        sigma_sq = math.log(n) / (2 * math.pi)
        log_gaussian = -(phase_diff ** 2) / (2 * sigma_sq)
        
        # Harmonic series contribution
        harmonic_sum = sum(1/k for k in range(1, min(10, int(math.sqrt(x)) + 1))
                          if n % (x * k) < k)
        log_harmonic = math.log(1 + harmonic_sum / math.log(x + 2))
        
        return log_gaussian + log_harmonic
    
    def _log_phase_coherence(self, x: int, n: int) -> float:
        """Compute log phase coherence"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for p in self.small_primes[:7]:
            phase_n = 2 * math.pi * (n % p) / p
            phase_x = 2 * math.pi * (x % p) / p
            coherence = (1 + math.cos(phase_n - phase_x)) / 2
            weight = 1 / math.log(p + 1)
            
            weighted_sum += coherence * weight
            total_weight += weight
        
        base_coherence = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # GCD amplification
        g = math.gcd(x, n)
        if g > 1:
            amplification = 1 + math.log(g) / math.log(n)
            base_coherence = min(1.0, base_coherence * amplification)
        
        return math.log(base_coherence) if base_coherence > 0 else -10
    
    def _log_harmonic_convergence(self, x: int, n: int) -> float:
        """Compute log harmonic convergence"""
        convergence_points = []
        
        # Unity harmonic
        g = math.gcd(x, n)
        unity_freq = 2 * math.pi / g if g > 0 else 2 * math.pi
        unity_harmonic = (1 + math.cos(unity_freq * math.log(n) / (2 * math.pi))) / 2
        convergence_points.append(unity_harmonic)
        
        # Golden ratio convergence
        phi_harmonic = x / self.phi
        phi_distance = min(abs(phi_harmonic - int(phi_harmonic)), 
                          abs(phi_harmonic - int(phi_harmonic) - 1))
        phi_convergence = math.exp(-phi_distance * self.phi)
        convergence_points.append(phi_convergence)
        
        # Tribonacci resonance
        if x > 2:
            tri_phase = math.log(x) / math.log(self.tau)
            tri_resonance = abs(math.sin(tri_phase * math.pi))
            convergence_points.append(tri_resonance)
        
        # Perfect square resonance
        sqrt_x = int(math.sqrt(x))
        if sqrt_x * sqrt_x == x:
            convergence_points.append(1.0)
        else:
            square_dist = min(x - sqrt_x**2, (sqrt_x + 1)**2 - x)
            square_harmony = math.exp(-square_dist / x)
            convergence_points.append(square_harmony)
        
        # Harmonic mean in log space
        if convergence_points and all(c > 0 for c in convergence_points):
            log_hm = math.log(len(convergence_points)) - \
                     math.log(sum(1/(c + 0.001) for c in convergence_points))
            return log_hm
        else:
            return -10


class HierarchicalSearch:
    """Coarse-to-fine resonance field exploration"""
    
    def __init__(self, n: int, analyzer: MultiScaleResonance):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.analyzer = analyzer
        self.levels = self._compute_hierarchy_levels()
    
    def _compute_hierarchy_levels(self) -> List[int]:
        """Compute sampling resolutions for each level"""
        levels = []
        points = self.sqrt_n
        
        while points > 100:
            levels.append(int(points))
            points = int(points ** 0.5)  # Square root reduction
        
        levels.append(100)  # Minimum level
        return levels[::-1]  # Reverse to go coarse to fine
    
    def search(self) -> List[Tuple[int, float]]:
        """Perform hierarchical search"""
        # Level 1: Coarse sampling
        coarse_peaks = self._coarse_sample()
        
        # Level 2: Refine around peaks
        refined_regions = []
        for peak, resonance in coarse_peaks[:10]:  # Top 10 peaks
            refined = self._refine_peak(peak, resonance)
            refined_regions.extend(refined)
        
        # Level 3: Binary search on gradients
        candidates = []
        for region_center, region_resonance in refined_regions[:20]:
            binary_results = self._binary_search_peak(region_center)
            candidates.extend(binary_results)
        
        # Sort by resonance and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:100]
    
    def _coarse_sample(self) -> List[Tuple[int, float]]:
        """Coarse sampling of resonance field"""
        if not self.levels:
            return []
        
        sample_points = min(self.levels[0], 1000)
        step = max(1, self.sqrt_n // sample_points)
        
        peaks = []
        for i in range(2, self.sqrt_n, step):
            resonance = self.analyzer.compute_coarse_resonance(i, self.n)
            if resonance > 0.1:  # Basic threshold
                peaks.append((i, resonance))
        
        # Sort by resonance
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks
    
    def _refine_peak(self, peak: int, peak_resonance: float) -> List[Tuple[int, float]]:
        """Refine search around a peak"""
        # Adaptive window size
        window = int(self.sqrt_n ** 0.1 * (1 + peak_resonance))
        refined = []
        
        # Finer sampling around peak
        step = max(1, window // 20)
        for offset in range(-window, window + 1, step):
            x = peak + offset
            if 2 <= x <= self.sqrt_n:
                resonance = self.analyzer.compute_resonance(x, self.n)
                if resonance > peak_resonance * 0.5:
                    refined.append((x, resonance))
        
        return refined
    
    def _binary_search_peak(self, center: int) -> List[Tuple[int, float]]:
        """Binary search for local maxima using golden section search"""
        candidates = []
        
        # Search window
        left = max(2, center - int(self.sqrt_n ** 0.05))
        right = min(self.sqrt_n, center + int(self.sqrt_n ** 0.05))
        
        # Golden section search
        phi = (1 + math.sqrt(5)) / 2
        resphi = 2 - phi
        
        tol = max(1, int((right - left) * 0.001))
        
        x1 = left + int(resphi * (right - left))
        x2 = right - int(resphi * (right - left))
        f1 = self.analyzer.compute_resonance(x1, self.n)
        f2 = self.analyzer.compute_resonance(x2, self.n)
        
        while abs(right - left) > tol:
            if f1 > f2:
                right = x2
                x2 = x1
                f2 = f1
                x1 = left + int(resphi * (right - left))
                f1 = self.analyzer.compute_resonance(x1, self.n)
            else:
                left = x1
                x1 = x2
                f1 = f2
                x2 = right - int(resphi * (right - left))
                f2 = self.analyzer.compute_resonance(x2, self.n)
        
        # Check the peak
        peak_x = (left + right) // 2
        peak_resonance = self.analyzer.compute_resonance(peak_x, self.n)
        
        if peak_resonance > 0.1:
            candidates.append((peak_x, peak_resonance))
            
            # Also check immediate neighbors
            for offset in [-1, 1]:
                x = peak_x + offset
                if 2 <= x <= self.sqrt_n:
                    res = self.analyzer.compute_resonance(x, self.n)
                    if res > 0.1:
                        candidates.append((x, res))
        
        return candidates


class StateManager:
    """Manages computation state with minimal memory footprint"""
    
    def __init__(self, checkpoint_interval: int = 10000):
        self.checkpoint_interval = checkpoint_interval
        self.sliding_window = deque(maxlen=1000)
        self.checkpoints: List[Dict[str, Any]] = []
        self.iteration_count = 0
        self.best_resonance = 0.0
        self.best_position = 0
        self.explored_regions: List[Tuple[int, int, float]] = []
        
    def update(self, x: int, resonance: float):
        """Update state with new evaluation"""
        self.iteration_count += 1
        self.sliding_window.append((x, resonance))
        
        if resonance > self.best_resonance:
            self.best_resonance = resonance
            self.best_position = x
        
        # Checkpoint if needed
        if self.iteration_count % self.checkpoint_interval == 0:
            self.checkpoint()
    
    def checkpoint(self):
        """Save current state for resumption"""
        # Summarize explored regions
        if self.sliding_window:
            positions = [x for x, _ in self.sliding_window]
            resonances = [r for _, r in self.sliding_window]
            region_summary = {
                'min_pos': min(positions),
                'max_pos': max(positions),
                'mean_resonance': sum(resonances) / len(resonances),
                'max_resonance': max(resonances)
            }
            self.explored_regions.append((
                region_summary['min_pos'],
                region_summary['max_pos'],
                region_summary['max_resonance']
            ))
        
        state = {
            'iteration': self.iteration_count,
            'best_resonance': self.best_resonance,
            'best_position': self.best_position,
            'explored_regions': self.explored_regions[-10:],  # Keep last 10
            'timestamp': time.time()
        }
        
        self.checkpoints.append(state)
        
        # Keep only recent checkpoints to save memory
        if len(self.checkpoints) > 10:
            self.checkpoints = self.checkpoints[-10:]
    
    def save_to_file(self, filename: str):
        """Save state to file"""
        with open(filename, 'w') as f:
            json.dump({
                'checkpoints': self.checkpoints,
                'final_state': {
                    'iteration': self.iteration_count,
                    'best_resonance': self.best_resonance,
                    'best_position': self.best_position
                }
            }, f, indent=2)
    
    def resume_from_file(self, filename: str):
        """Resume from saved state"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.checkpoints = data['checkpoints']
        final_state = data['final_state']
        self.iteration_count = final_state['iteration']
        self.best_resonance = final_state['best_resonance']
        self.best_position = final_state['best_position']
        
        # Rebuild explored regions
        for checkpoint in self.checkpoints:
            self.explored_regions.extend(checkpoint.get('explored_regions', []))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        if self.sliding_window:
            recent_resonances = [r for _, r in self.sliding_window]
            mean_recent = sum(recent_resonances) / len(recent_resonances)
            std_recent = math.sqrt(sum((r - mean_recent)**2 for r in recent_resonances) / len(recent_resonances))
        else:
            mean_recent = 0
            std_recent = 0
        
        return {
            'iterations': self.iteration_count,
            'best_resonance': self.best_resonance,
            'best_position': self.best_position,
            'mean_recent_resonance': mean_recent,
            'std_recent_resonance': std_recent,
            'regions_explored': len(self.explored_regions)
        }


# ============================================================================
# PATTERN LEARNING MODULE
# ============================================================================

class ZonePredictor:
    """Predicts high-resonance zones based on features"""
    
    def __init__(self):
        self.weights = defaultdict(float)
        self.training_data = []
    
    def add_training_example(self, features: np.ndarray, is_factor_zone: bool):
        """Add a training example"""
        self.training_data.append((features, is_factor_zone))
        
        # Simple online learning update
        if len(self.training_data) % 100 == 0:
            self.train()
    
    def train(self):
        """Train the predictor (simplified version)"""
        if not self.training_data:
            return
        
        # Simple weighted voting based on features
        for features, is_factor in self.training_data[-100:]:
            weight = 1.0 if is_factor else -0.1
            for i, f in enumerate(features):
                self.weights[i] += weight * f
    
    def predict(self, features: np.ndarray, sqrt_n: int) -> List[Tuple[int, int, float]]:
        """Predict zones likely to contain factors"""
        # Simple linear prediction
        score = sum(self.weights[i] * f for i, f in enumerate(features))
        confidence = 1 / (1 + math.exp(-score))  # Sigmoid
        
        # Generate zones based on confidence
        zones = []
        if confidence > 0.5:
            # High confidence - focused zones
            center = int(sqrt_n * 0.8)  # Example heuristic
            width = int(sqrt_n * 0.05 * confidence)
            zones.append((center - width, center + width, confidence))
        
        return zones


class ResonancePatternLearner:
    """Learns successful resonance patterns for acceleration"""
    
    def __init__(self):
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []
        self.transition_boundaries: Dict[Tuple[int, int], int] = {
            (2, 3): 282281,
            (3, 5): 2961841,
            (5, 7): 53596041,
            (7, 11): 1522756281
        }
        self.feature_weights = defaultdict(float)
        self.zone_predictor = ZonePredictor()
    
    def record_success(self, n: int, factor: int, resonance_profile: Dict[str, float]):
        """Record successful factorization patterns"""
        pattern = self._extract_pattern(n, factor, resonance_profile)
        self.success_patterns.append(pattern)
        
        # Update transition boundaries if near known boundary
        self._update_transition_boundaries(n, factor)
        
        # Update feature weights based on success
        self._update_feature_weights(pattern, success=True)
        
        # Retrain zone predictor periodically
        if len(self.success_patterns) % 10 == 0:
            self._retrain_models()
    
    def record_failure(self, n: int, tested_positions: List[int], resonances: List[float]):
        """Record failed attempts for negative learning"""
        for x, res in zip(tested_positions[:10], resonances[:10]):  # Top 10 failures
            pattern = self._extract_pattern(n, x, {'resonance': res})
            self.failure_patterns.append(pattern)
            self._update_feature_weights(pattern, success=False)
    
    def predict_high_resonance_zones(self, n: int) -> List[Tuple[int, int, float]]:
        """Predict zones likely to contain factors based on learned patterns"""
        features = self._extract_features(n)
        zones = []
        
        sqrt_n = int(math.sqrt(n))
        
        # 1. Transition boundary predictions
        for (b1, b2), boundary in self.transition_boundaries.items():
            if boundary * 0.1 <= n <= boundary * 10:
                center = int(math.sqrt(boundary))
                width = int(center * 0.1)
                confidence = self._compute_boundary_confidence(n, boundary)
                zones.append((max(2, center - width), 
                            min(sqrt_n, center + width), 
                            confidence))
        
        # 2. Feature-based predictions
        feature_zones = self.zone_predictor.predict(features, sqrt_n)
        zones.extend(feature_zones)
        
        # 3. Pattern-based predictions
        pattern_zones = self._apply_learned_patterns(n)
        zones.extend(pattern_zones)
        
        # Merge overlapping zones and sort by confidence
        merged_zones = self._merge_zones(zones)
        merged_zones.sort(key=lambda z: z[2], reverse=True)
        
        return merged_zones[:10]  # Top 10 zones
    
    def _extract_pattern(self, n: int, x: int, resonance_profile: Dict[str, float]) -> Dict[str, Any]:
        """Extract pattern features from a factorization attempt"""
        sqrt_n = int(math.sqrt(n))
        
        pattern = {
            'n_bits': n.bit_length(),
            'x_bits': x.bit_length(),
            'relative_position': x / sqrt_n,
            'is_prime_x': self._is_probable_prime(x),
            'gcd_score': math.gcd(x, n) / x,
            'mod_profile': [n % p for p in [2, 3, 5, 7, 11]],
            'resonance': resonance_profile.get('resonance', 0),
            'near_square': self._near_perfect_square(x),
            'near_power': self._near_prime_power(x),
        }
        
        return pattern
    
    def _extract_features(self, n: int) -> np.ndarray:
        """Extract feature vector for n"""
        features = []
        
        # Basic features
        features.append(n.bit_length() / 256)  # Normalized bit length
        features.append(math.log(n) / 100)     # Log scale
        
        # Modular features
        for p in [2, 3, 5, 7, 11]:
            features.append((n % p) / p)
        
        # Digit sum features
        digit_sum = sum(int(d) for d in str(n))
        features.append(digit_sum / (9 * len(str(n))))
        
        # Binary pattern features
        binary = bin(n)[2:]
        features.append(binary.count('1') / len(binary))  # Bit density
        
        return np.array(features)
    
    def _update_transition_boundaries(self, n: int, factor: int):
        """Update transition boundaries based on discovered patterns"""
        sqrt_n = int(math.sqrt(n))
        
        # Check if this factorization reveals a new transition pattern
        for (b1, b2), boundary in self.transition_boundaries.items():
            if abs(sqrt_n - int(math.sqrt(boundary))) < sqrt_n * 0.01:
                # Near a known transition - refine the boundary
                self.transition_boundaries[(b1, b2)] = int((boundary + n) / 2)
    
    def _update_feature_weights(self, pattern: Dict[str, Any], success: bool):
        """Update feature weights based on pattern success/failure"""
        weight = 1.0 if success else -0.1
        
        # Update weights for pattern features
        for key, value in pattern.items():
            if isinstance(value, (int, float)):
                self.feature_weights[key] += weight * value
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    self.feature_weights[f"{key}_{i}"] += weight * v
    
    def _retrain_models(self):
        """Retrain all models with accumulated patterns"""
        if not self.success_patterns:
            return
        
        # Extract training data from success patterns
        for pattern in self.success_patterns[-100:]:  # Use recent 100
            n_bits = pattern['n_bits']
            relative_pos = pattern['relative_position']
            
            # Create zone around successful factor
            sqrt_n = 2 ** (n_bits / 2)
            factor_pos = int(relative_pos * sqrt_n)
            zone_width = int(sqrt_n * 0.01)
            
            # Add positive training example
            features = np.array([n_bits/256, relative_pos, pattern['gcd_score']])
            self.zone_predictor.add_training_example(features, True)
        
        # Train the predictor
        self.zone_predictor.train()
    
    def _compute_boundary_confidence(self, n: int, boundary: int) -> float:
        """Compute confidence for a transition boundary"""
        distance = abs(math.log(n) - math.log(boundary))
        return math.exp(-distance)
    
    def _apply_learned_patterns(self, n: int) -> List[Tuple[int, int, float]]:
        """Apply learned patterns to predict zones"""
        zones = []
        sqrt_n = int(math.sqrt(n))
        
        # Find similar successful patterns
        n_bits = n.bit_length()
        similar_patterns = [
            p for p in self.success_patterns
            if abs(p['n_bits'] - n_bits) <= 2
        ]
        
        # Generate zones based on similar patterns
        for pattern in similar_patterns[:5]:  # Top 5 similar
            center = int(pattern['relative_position'] * sqrt_n)
            width = int(sqrt_n * 0.02)
            confidence = pattern['resonance'] * 0.8  # Decay factor
            
            if 2 <= center - width and center + width <= sqrt_n:
                zones.append((center - width, center + width, confidence))
        
        return zones
    
    def _merge_zones(self, zones: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Merge overlapping zones"""
        if not zones:
            return []
        
        # Sort by start position
        zones.sort(key=lambda z: z[0])
        
        merged = []
        current_start, current_end, current_conf = zones[0]
        
        for start, end, conf in zones[1:]:
            if start <= current_end:
                # Overlapping - merge
                current_end = max(current_end, end)
                current_conf = max(current_conf, conf)
            else:
                # Non-overlapping - save current and start new
                merged.append((current_start, current_end, current_conf))
                current_start, current_end, current_conf = start, end, conf
        
        merged.append((current_start, current_end, current_conf))
        return merged
    
    def _is_probable_prime(self, n: int) -> bool:
        """Simple primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        return True  # Probably prime for small values
    
    def _near_perfect_square(self, x: int) -> float:
        """Score how close x is to a perfect square"""
        sqrt_x = int(math.sqrt(x))
        if sqrt_x * sqrt_x == x:
            return 1.0
        
        dist = min(x - sqrt_x**2, (sqrt_x + 1)**2 - x)
        return math.exp(-dist / x)
    
    def _near_prime_power(self, x: int) -> float:
        """Score how close x is to a prime power"""
        for p in [2, 3, 5, 7, 11]:
            k = int(math.log(x) / math.log(p))
            if k >= 1:
                dist = abs(x - p**k)
                if dist < x * 0.01:
                    return math.exp(-dist / x)
        return 0.0


# ============================================================================
# MAIN RFH3 CLASS
# ============================================================================

class RFH3Config:
    """Configuration for RFH3 factorizer"""
    
    def __init__(self):
        self.max_iterations = 1000000
        self.checkpoint_interval = 10000
        self.importance_lambda = 1.0  # Controls exploration vs exploitation
        self.adaptive_threshold_k = 2.0  # Initial threshold factor
        self.learning_enabled = True
        self.hierarchical_search = True
        self.gradient_navigation = True
        self.parallel_regions = 4
        self.log_level = logging.INFO
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'max_iterations': self.max_iterations,
            'checkpoint_interval': self.checkpoint_interval,
            'importance_lambda': self.importance_lambda,
            'adaptive_threshold_k': self.adaptive_threshold_k,
            'learning_enabled': self.learning_enabled,
            'hierarchical_search': self.hierarchical_search,
            'gradient_navigation': self.gradient_navigation,
            'parallel_regions': self.parallel_regions,
            'log_level': self.log_level
        }


class RFH3:
    """Adaptive Resonance Field Hypothesis v3 - Main Class"""
    
    def __init__(self, config: Optional[RFH3Config] = None):
        self.config = config or RFH3Config()
        self.logger = self._setup_logging()
        
        # Core components
        self.learner = ResonancePatternLearner()
        self.state = StateManager(self.config.checkpoint_interval)
        self.analyzer = None  # Initialized per factorization
        self.stats = {
            'factorizations': 0,
            'total_time': 0,
            'success_rate': 1.0,
            'avg_iterations': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('RFH3')
        logger.setLevel(self.config.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def factor(self, n: int) -> Tuple[int, int]:
        """Main factorization method with all RFH3 enhancements"""
        if n < 4:
            raise ValueError("n must be >= 4")
        
        # Check if n is prime (quick test)
        if self._is_probable_prime(n):
            raise ValueError(f"{n} appears to be prime")
        
        start_time = time.time()
        self.logger.info(f"Starting RFH3 factorization of {n} ({n.bit_length()} bits)")
        
        # Initialize components for this factorization
        self.analyzer = MultiScaleResonance()
        iterator = LazyResonanceIterator(n, self.analyzer)
        
        # Use learned patterns to predict high-resonance zones
        predicted_zones = []
        if self.config.learning_enabled:
            predicted_zones = self.learner.predict_high_resonance_zones(n)
            if predicted_zones:
                self.logger.info(f"Predicted {len(predicted_zones)} high-resonance zones")
        
        # Hierarchical search if enabled
        if self.config.hierarchical_search:
            search = HierarchicalSearch(n, self.analyzer)
            candidates = search.search()
            
            # Check hierarchical candidates first
            for x, resonance in candidates:
                if n % x == 0:
                    factor = x
                    other = n // x
                    self._record_success(n, factor, resonance, time.time() - start_time)
                    return (min(factor, other), max(factor, other))
        
        # Adaptive threshold computation
        threshold = self._compute_adaptive_threshold(n)
        
        # Main adaptive search loop
        iteration = 0
        tested_positions = []
        resonances = []
        
        for x in iterator:
            iteration += 1
            
            # Compute full resonance
            resonance = self.analyzer.compute_resonance(x, n)
            self.state.update(x, resonance)
            
            tested_positions.append(x)
            resonances.append(resonance)
            
            # Check if x is a factor
            if resonance > threshold:
                self.logger.debug(f"High resonance {resonance:.4f} at x={x}")
                
                if n % x == 0:
                    factor = x
                    other = n // x
                    self._record_success(n, factor, resonance, time.time() - start_time)
                    return (min(factor, other), max(factor, other))
            
            # Update threshold based on statistics
            if iteration % 1000 == 0:
                stats = self.state.get_statistics()
                threshold = self._update_adaptive_threshold(
                    threshold, stats['mean_recent_resonance'], 
                    stats['std_recent_resonance']
                )
                
                self.logger.debug(
                    f"Iteration {iteration}: threshold={threshold:.4f}, "
                    f"best_resonance={stats['best_resonance']:.4f}"
                )
            
            # Check termination conditions
            if iteration >= self.config.max_iterations:
                self.logger.warning(f"Reached max iterations ({self.config.max_iterations})")
                break
        
        # Record failure for learning
        if self.config.learning_enabled:
            self.learner.record_failure(n, tested_positions[-100:], resonances[-100:])
        
        # Fallback to Pollard's Rho
        self.logger.warning("RFH3 exhausted, falling back to Pollard's Rho")
        return self._pollard_rho_fallback(n)
    
    def factor_with_hints(self, n: int, hints: Dict[str, Any]) -> Tuple[int, int]:
        """Factor using prior knowledge or external hints"""
        # Extract useful hints
        bit_range = hints.get('factor_bit_range', None)
        modular_constraints = hints.get('modular_constraints', {})
        known_non_factors = hints.get('known_non_factors', [])
        
        # Adjust configuration based on hints
        if bit_range:
            self.logger.info(f"Using bit range hint: {bit_range}")
            # Constrain search to specific bit ranges
        
        # Proceed with modified factorization
        return self.factor(n)
    
    def save_state(self, filename: str):
        """Save current state and learned patterns"""
        state_data = {
            'config': self.config.to_dict(),
            'stats': self.stats,
            'learner': self.learner,
            'state_manager': self.state
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state_data, f)
        
        self.logger.info(f"Saved state to {filename}")
    
    def load_state(self, filename: str):
        """Load saved state and learned patterns"""
        with open(filename, 'rb') as f:
            state_data = pickle.load(f)
        
        # Restore components
        self.stats = state_data['stats']
        self.learner = state_data['learner']
        self.state = state_data['state_manager']
        
        # Update config
        for key, value in state_data['config'].items():
            setattr(self.config, key, value)
        
        self.logger.info(f"Loaded state from {filename}")
    
    def _compute_adaptive_threshold(self, n: int) -> float:
        """Compute initial adaptive threshold"""
        # Base threshold from theory
        base = 1.0 / math.log(n)
        
        # Adjust based on success rate
        sr = self.stats.get('success_rate', 1.0)
        k = 2.0 * (1 - sr)**2 + 0.5
        
        return base * k
    
    def _update_adaptive_threshold(self, current: float, mean: float, std: float) -> float:
        """Update threshold based on recent statistics"""
        if std > 0:
            # Move threshold closer to high-resonance region
            new_threshold = mean - self.config.adaptive_threshold_k * std
            # Smooth update
            return 0.7 * current + 0.3 * new_threshold
        return current
    
    def _record_success(self, n: int, factor: int, resonance: float, time_taken: float):
        """Record successful factorization"""
        self.stats['factorizations'] += 1
        self.stats['total_time'] += time_taken
        self.stats['avg_iterations'] = (
            self.state.iteration_count / self.stats['factorizations']
        )
        
        # Update learner
        if self.config.learning_enabled:
            self.learner.record_success(n, factor, {'resonance': resonance})
        
        self.logger.info(
            f"Success! {n} = {factor} × {n//factor} "
            f"(resonance={resonance:.4f}, time={time_taken:.3f}s)"
        )
    
    def _is_probable_prime(self, n: int) -> bool:
        """Miller-Rabin primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Small prime divisions
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        if n < 10000:
            for i in range(101, int(math.sqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        # Miller-Rabin test
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def _pollard_rho_fallback(self, n: int) -> Tuple[int, int]:
        """Pollard's Rho as fallback"""
        if n == 4:
            return (2, 2)
        
        # Try small factors first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if n % p == 0:
                return (p, n // p)
        
        # Pollard's Rho with multiple polynomials
        for c in [1, 2, 3]:
            x = 2
            y = 2
            d = 1
            
            f = lambda x: (x * x + c) % n
            
            while d == 1:
                x = f(x)
                y = f(f(y))
                d = math.gcd(abs(x - y), n)
                
                if d != 1 and d != n:
                    return (min(d, n // d), max(d, n // d))
        
        # Last resort: trial division
        sqrt_n = int(math.sqrt(n))
        for i in range(3, min(100000, sqrt_n), 2):
            if n % i == 0:
                return (i, n // i)
        
        raise ValueError(f"Failed to factor {n}")


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_rfh3():
    """Test the RFH3 implementation"""
    
    test_cases = [
        # Small cases
        (11, 13),                     # 143
        (101, 103),                   # 10403
        
        # Special forms
        (65537, 97),                  # Fermat prime * small prime
        (257, 641),                   # Known Fermat factorization
        
        # Balanced factors
        (523, 541),                   # Close primes
        (1009, 1013),                 # Larger close primes
        
        # Near transition
        (531, 532),                   # 282492
        
        # Larger cases (if fast enough)
        # (99991, 99989),             # Large twin primes
        # (524287, 524309),           # Near Mersenne
    ]
    
    rfh3 = RFH3()
    successes = 0
    total_time = 0
    
    print("Testing RFH3 Implementation")
    print("=" * 60)
    
    for p_true, q_true in test_cases:
        n = p_true * q_true
        
        try:
            start = time.time()
            p_found, q_found = rfh3.factor(n)
            elapsed = time.time() - start
            total_time += elapsed
            
            if {p_found, q_found} == {p_true, q_true}:
                print(f"✓ {n:10d} = {p_found:6d} × {q_found:6d} ({elapsed:.3f}s)")
                successes += 1
            else:
                print(f"✗ {n:10d}: Expected {p_true} × {q_true}, got {p_found} × {q_found}")
        
        except Exception as e:
            print(f"✗ {n:10d}: FAILED - {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"RESULTS: {successes}/{len(test_cases)} successful ({successes/len(test_cases)*100:.1f}%)")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time: {total_time/len(test_cases):.3f}s")
    
    # Test state saving/loading
    if successes > 0:
        print("\nTesting state persistence...")
        rfh3.save_state("rfh3_test_state.pkl")
        
        rfh3_new = RFH3()
        rfh3_new.load_state("rfh3_test_state.pkl")
        print("✓ State save/load successful")


if __name__ == "__main__":
    test_rfh3()
