"""
RFH1: Pure Resonance Field Research Tool
Discovers how to effectively leverage prime resonance fields for factorization.
No fallback methods - pure resonance detection only.
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Set, Any
import time
from functools import lru_cache
from collections import defaultdict
import json


@lru_cache(maxsize=50000)
def is_probable_prime(n: int) -> bool:
    """Fast primality test"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
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
    
    # Miller-Rabin
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


class ResonanceMetrics:
    """Collection of different resonance detection methods"""
    
    @staticmethod
    def phase_coherence(n: int, x: int, well_center: int, well_width: int) -> float:
        """Original phase coherence metric"""
        if x <= 1 or x > int(math.sqrt(n)):
            return 0.0
        
        coherence = 1.0
        
        # Well alignment
        dist_from_center = abs(x - well_center) / max(well_center, 1)
        coherence *= math.exp(-dist_from_center)
        
        # Phase matching
        phase_x = (x * 2 * math.pi) / int(math.sqrt(n))
        phase_n = (n % (x * x) if x * x <= n else n % x) * 2 * math.pi / x
        phase_match = abs(math.cos(phase_x - phase_n))
        coherence *= (1 + phase_match)
        
        return coherence
    
    @staticmethod
    def standing_wave_nodes(n: int, x: int) -> float:
        """Detect standing wave nodes where factors emerge"""
        sqrt_n = int(math.sqrt(n))
        if x <= 1 or x > sqrt_n:
            return 0.0
        
        # Model n as a standing wave with wavelength related to its factors
        # Nodes occur at integer multiples of wavelength/2
        wavelength = 2 * sqrt_n / math.pi
        
        # Check if x aligns with a node
        node_positions = []
        for k in range(1, 20):
            node_pos = k * wavelength / 2
            if node_pos <= sqrt_n:
                node_positions.append(node_pos)
        
        # Score based on proximity to nodes
        min_dist = min(abs(x - node) for node in node_positions) if node_positions else sqrt_n
        score = math.exp(-min_dist / (wavelength / 10))
        
        # Amplitude modulation based on n's structure
        amp_mod = abs(math.sin(x * math.pi / sqrt_n))
        score *= (1 + amp_mod)
        
        return score
    
    @staticmethod
    def modular_resonance(n: int, x: int) -> float:
        """Resonance in modular arithmetic space"""
        if x <= 1 or x > int(math.sqrt(n)):
            return 0.0
        
        score = 0.0
        
        # Check resonance with small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        resonances = []
        
        for p in small_primes:
            if p >= x:
                break
            
            # Modular harmony: when n mod p and x mod p have special relationships
            n_mod = n % p
            x_mod = x % p
            
            # Perfect resonance when x_mod divides n_mod
            if x_mod != 0 and n_mod % x_mod == 0:
                resonances.append(2.0)
            # Harmonic resonance
            elif x_mod != 0:
                harmony = 1.0 / (1 + abs(n_mod - x_mod * (n_mod // x_mod)))
                resonances.append(harmony)
        
        if resonances:
            score = np.mean(resonances) * math.sqrt(len(resonances))
        
        return score
    
    @staticmethod
    def bit_pattern_resonance(n: int, x: int) -> float:
        """Resonance in binary representation"""
        if x <= 1 or x > int(math.sqrt(n)):
            return 0.0
        
        n_bits = bin(n)[2:]
        x_bits = bin(x)[2:]
        
        # Pad to same length
        max_len = max(len(n_bits), len(x_bits))
        n_bits = n_bits.zfill(max_len)
        x_bits = x_bits.zfill(max_len)
        
        score = 0.0
        
        # Pattern matching at different scales
        for shift in range(max_len // 2):
            if shift < len(x_bits):
                # Check if x pattern appears in n when shifted
                x_shifted = '0' * shift + x_bits[:-shift] if shift > 0 else x_bits
                
                matches = sum(1 for i in range(len(x_shifted)) 
                            if i < len(n_bits) and n_bits[i] == x_shifted[i])
                
                score += matches / max_len
        
        # Bit density similarity
        n_density = n_bits.count('1') / len(n_bits)
        x_density = x_bits.count('1') / len(x_bits)
        density_sim = 1 - abs(n_density - x_density)
        
        score *= density_sim
        
        return score
    
    @staticmethod
    def fibonacci_resonance(n: int, x: int) -> float:
        """Resonance with Fibonacci-like sequences"""
        if x <= 1 or x > int(math.sqrt(n)):
            return 0.0
        
        # Generate Fibonacci sequence mod n
        fib_mod = [0, 1]
        for _ in range(20):
            fib_mod.append((fib_mod[-1] + fib_mod[-2]) % n)
        
        # Check if x appears in or divides Fibonacci residues
        score = 0.0
        for fib_val in fib_mod[2:]:  # Skip 0, 1
            if fib_val % x == 0:
                score += 1.0
            elif x % fib_val == 0 and fib_val > 1:
                score += 0.5
        
        # Normalize
        score = score / len(fib_mod)
        
        # Golden ratio alignment
        phi = (1 + math.sqrt(5)) / 2
        x_phi = x * phi
        n_phi = n / phi
        
        # Check if x aligns with golden ratio divisions of n
        for k in range(1, 10):
            golden_point = n / (phi ** k)
            if abs(x - golden_point) < x * 0.1:
                score *= 2.0
                break
        
        return score
    
    @staticmethod
    def prime_cascade_resonance(n: int, x: int) -> float:
        """Resonance with prime cascade patterns (Axiom 1)"""
        if x <= 1 or x > int(math.sqrt(n)):
            return 0.0
        
        # Prime emanation pattern
        base_primes = [2, 3, 5, 7, 11]
        score = 0.0
        
        for p in base_primes:
            # Check if x is in the emanation field of p relative to n
            emanation_points = []
            for k in range(1, 10):
                point = int(n ** (1.0 / (p * k)))
                if point > 1:
                    emanation_points.append(point)
            
            # Score based on proximity to emanation points
            if emanation_points:
                min_dist = min(abs(x - point) for point in emanation_points)
                score += math.exp(-min_dist / (x * 0.1))
        
        return score / len(base_primes)


class TransitionBoundaryAnalyzer:
    """Advanced transition boundary detection and analysis"""
    
    def __init__(self):
        self.known_transitions = {
            (2, 3): 282281,  # 531²
            (3, 5): 2961841,  # Hypothesis
            (5, 7): 53596041,  # Hypothesis
        }
        self.transition_patterns = defaultdict(list)
    
    def analyze_transition_signature(self, n: int) -> Dict[str, Any]:
        """Analyze n's relationship to transition boundaries"""
        sqrt_n = int(math.sqrt(n))
        analysis = {
            'nearest_transition': None,
            'distance_ratio': float('inf'),
            'resonance_echo': 0.0,
            'boundary_harmonics': []
        }
        
        # Find nearest transition
        for (b1, b2), boundary in self.known_transitions.items():
            sqrt_boundary = int(math.sqrt(boundary))
            distance = abs(sqrt_n - sqrt_boundary)
            ratio = distance / sqrt_boundary if sqrt_boundary > 0 else float('inf')
            
            if ratio < analysis['distance_ratio']:
                analysis['distance_ratio'] = ratio
                analysis['nearest_transition'] = (b1, b2, boundary)
            
            # Harmonic analysis
            for harmonic in [1, 2, 3, 5, 8]:  # Fibonacci-like
                harmonic_point = sqrt_boundary * harmonic
                if abs(sqrt_n - harmonic_point) < sqrt_n * 0.1:
                    analysis['boundary_harmonics'].append({
                        'transition': (b1, b2),
                        'harmonic': harmonic,
                        'point': harmonic_point,
                        'distance': abs(sqrt_n - harmonic_point)
                    })
        
        # Echo detection
        if analysis['nearest_transition']:
            b1, b2, boundary = analysis['nearest_transition']
            # Model echo as decaying wave
            echo_strength = math.exp(-analysis['distance_ratio'])
            analysis['resonance_echo'] = echo_strength
        
        return analysis
    
    def predict_factor_zones(self, n: int) -> List[Tuple[int, int, float]]:
        """Predict zones where factors are likely based on transitions"""
        sqrt_n = int(math.sqrt(n))
        zones = []
        
        analysis = self.analyze_transition_signature(n)
        
        # Zone 1: Near transition echoes
        if analysis['nearest_transition']:
            b1, b2, boundary = analysis['nearest_transition']
            sqrt_boundary = int(math.sqrt(boundary))
            
            # Primary echo zone
            zone_center = sqrt_boundary
            zone_width = int(sqrt_boundary * 0.1)
            zones.append((max(2, zone_center - zone_width), 
                         min(sqrt_n, zone_center + zone_width),
                         analysis['resonance_echo']))
        
        # Zone 2: Harmonic zones
        for harmonic in analysis['boundary_harmonics']:
            center = int(harmonic['point'])
            width = int(center * 0.05)
            confidence = 1.0 / (1 + harmonic['distance'] / center)
            zones.append((max(2, center - width),
                         min(sqrt_n, center + width),
                         confidence))
        
        # Zone 3: Interference zones between transitions
        transitions = list(self.known_transitions.values())
        for i in range(len(transitions)):
            for j in range(i + 1, len(transitions)):
                t1, t2 = int(math.sqrt(transitions[i])), int(math.sqrt(transitions[j]))
                interference_point = int(math.sqrt(t1 * t2))
                if 2 < interference_point < sqrt_n:
                    width = int(interference_point * 0.02)
                    zones.append((max(2, interference_point - width),
                                 min(sqrt_n, interference_point + width),
                                 0.5))
        
        return sorted(zones, key=lambda z: z[2], reverse=True)


class ResonanceFieldExplorer:
    """Main research tool for discovering resonance patterns"""
    
    def __init__(self):
        self.metrics = ResonanceMetrics()
        self.boundary_analyzer = TransitionBoundaryAnalyzer()
        self.discovery_log = []
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
    
    def explore(self, n: int, max_candidates: int = 10000) -> Optional[Tuple[int, int]]:
        """
        Explore resonance field to find factors.
        Limited candidates to force reliance on resonance detection.
        """
        if n < 2:
            raise ValueError("n must be >= 2")
        
        if is_probable_prime(n):
            raise ValueError(f"{n} is prime")
        
        sqrt_n = int(math.sqrt(n))
        start_time = time.perf_counter()
        
        print(f"\n{'='*70}")
        print(f"RFH1: Pure Resonance Field Exploration")
        print(f"n = {n} ({n.bit_length()} bits)")
        print(f"sqrt(n) ≈ {sqrt_n}")
        print(f"{'='*70}\n")
        
        # Analyze transition signature
        print("Analyzing transition boundary signature...")
        transition_analysis = self.boundary_analyzer.analyze_transition_signature(n)
        self._print_transition_analysis(transition_analysis)
        
        # Get predicted factor zones
        zones = self.boundary_analyzer.predict_factor_zones(n)
        print(f"\nIdentified {len(zones)} high-probability zones")
        
        # Comprehensive resonance scoring
        candidates = self._generate_candidates(n, zones, max_candidates)
        print(f"\nEvaluating {len(candidates)} resonance candidates...")
        
        scored_candidates = []
        evaluation_count = 0
        
        for x in candidates:
            # Compute all resonance metrics
            scores = {
                'phase_coherence': self.metrics.phase_coherence(n, x, sqrt_n//2, sqrt_n),
                'standing_wave': self.metrics.standing_wave_nodes(n, x),
                'modular': self.metrics.modular_resonance(n, x),
                'bit_pattern': self.metrics.bit_pattern_resonance(n, x),
                'fibonacci': self.metrics.fibonacci_resonance(n, x),
                'prime_cascade': self.metrics.prime_cascade_resonance(n, x)
            }
            
            # Weighted combination
            total_score = (
                scores['phase_coherence'] * 2.0 +
                scores['standing_wave'] * 1.5 +
                scores['modular'] * 1.8 +
                scores['bit_pattern'] * 1.2 +
                scores['fibonacci'] * 1.3 +
                scores['prime_cascade'] * 2.0
            )
            
            scored_candidates.append({
                'x': x,
                'total_score': total_score,
                'scores': scores
            })
            
            evaluation_count += 1
            if evaluation_count % 1000 == 0:
                print(f"  Evaluated {evaluation_count} candidates...")
        
        # Sort by total score
        scored_candidates.sort(key=lambda c: c['total_score'], reverse=True)
        
        # Test top candidates
        print(f"\nTesting top {min(100, len(scored_candidates))} candidates by resonance score...")
        
        tested = 0
        for candidate in scored_candidates[:100]:
            x = candidate['x']
            if n % x == 0:
                elapsed = time.perf_counter() - start_time
                other = n // x
                
                print(f"\n{'='*70}")
                print(f"✓ SUCCESS! Found factor: {x}")
                print(f"  {n} = {x} × {other}")
                print(f"  Time: {elapsed:.3f}s")
                print(f"  Candidates evaluated: {evaluation_count}")
                print(f"  Candidates tested: {tested + 1}")
                print(f"\nResonance scores for {x}:")
                for metric, score in candidate['scores'].items():
                    print(f"  {metric:15s}: {score:.4f}")
                print(f"  Total score: {candidate['total_score']:.4f}")
                
                # Log discovery
                self._log_discovery(n, x, candidate, transition_analysis)
                
                return (x, other) if x <= other else (other, x)
            
            tested += 1
        
        # Failure analysis
        elapsed = time.perf_counter() - start_time
        print(f"\n{'='*70}")
        print(f"✗ EXPLORATION INCOMPLETE")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Candidates evaluated: {evaluation_count}")
        print(f"  Candidates tested: {tested}")
        print(f"\nTop 5 candidates that were tested:")
        for i, candidate in enumerate(scored_candidates[:5]):
            print(f"  {i+1}. x={candidate['x']}, score={candidate['total_score']:.4f}")
        
        # Log failure for analysis
        self._log_failure(n, scored_candidates[:10], transition_analysis)
        
        return None
    
    def _generate_candidates(self, n: int, zones: List[Tuple[int, int, float]], 
                           max_candidates: int) -> List[int]:
        """Generate candidates focusing on high-resonance zones"""
        sqrt_n = int(math.sqrt(n))
        candidates = set()
        
        # Priority 1: Zone candidates
        zone_budget = max_candidates * 0.6
        for start, end, confidence in zones:
            zone_size = end - start + 1
            if zone_size <= zone_budget / len(zones):
                # Can test entire zone
                candidates.update(range(start, end + 1))
            else:
                # Sample proportionally to confidence
                sample_size = int(zone_budget * confidence / (len(zones) * 2))
                if sample_size > 0:
                    step = max(1, zone_size // sample_size)
                    candidates.update(range(start, end + 1, step))
        
        # Priority 2: Special mathematical forms
        special_forms = []
        
        # Powers of small primes
        for p in [2, 3, 5, 7, 11]:
            k = 1
            while p**k <= sqrt_n:
                special_forms.append(p**k)
                k += 1
        
        # Mersenne-like numbers
        for i in range(2, int(math.log2(sqrt_n)) + 1):
            special_forms.extend([2**i - 1, 2**i + 1])
        
        # Products of small primes
        small_primes = [p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] if p <= sqrt_n]
        for i, p1 in enumerate(small_primes):
            for p2 in small_primes[i:]:
                if p1 * p2 <= sqrt_n:
                    special_forms.append(p1 * p2)
        
        candidates.update(f for f in special_forms if 2 <= f <= sqrt_n)
        
        # Priority 3: Resonance field samples
        remaining_budget = max_candidates - len(candidates)
        if remaining_budget > 0:
            # Logarithmic sampling for scale invariance
            log_samples = np.logspace(math.log10(2), math.log10(sqrt_n), 
                                    num=remaining_budget, dtype=int)
            candidates.update(int(x) for x in log_samples if 2 <= x <= sqrt_n)
        
        return sorted(list(candidates))[:max_candidates]
    
    def _print_transition_analysis(self, analysis: Dict[str, Any]):
        """Pretty print transition analysis"""
        if analysis['nearest_transition']:
            b1, b2, boundary = analysis['nearest_transition']
            print(f"  Nearest transition: {b1}→{b2} at {boundary}")
            print(f"  Distance ratio: {analysis['distance_ratio']:.4f}")
            print(f"  Resonance echo strength: {analysis['resonance_echo']:.4f}")
        
        if analysis['boundary_harmonics']:
            print(f"  Detected {len(analysis['boundary_harmonics'])} boundary harmonics:")
            for h in analysis['boundary_harmonics'][:3]:
                print(f"    - {h['transition']} harmonic {h['harmonic']} at {h['point']}")
    
    def _log_discovery(self, n: int, factor: int, candidate: Dict, 
                      transition_analysis: Dict):
        """Log successful discovery for pattern analysis"""
        log_entry = {
            'n': n,
            'factor': factor,
            'bit_length': n.bit_length(),
            'scores': candidate['scores'],
            'total_score': candidate['total_score'],
            'transition_analysis': transition_analysis,
            'timestamp': time.time()
        }
        self.discovery_log.append(log_entry)
        
        # Extract patterns
        for metric, score in candidate['scores'].items():
            if score > 0.5:  # Significant score
                self.success_patterns[metric].append({
                    'n': n,
                    'factor': factor,
                    'score': score
                })
    
    def _log_failure(self, n: int, top_candidates: List[Dict], 
                    transition_analysis: Dict):
        """Log failure for pattern analysis"""
        self.failure_patterns['high_scoring_misses'].append({
            'n': n,
            'top_candidates': [(c['x'], c['total_score']) for c in top_candidates],
            'transition_analysis': transition_analysis
        })
    
    def save_discoveries(self, filename: str = "rfh1_discoveries.json"):
        """Save discovery log for analysis"""
        data = {
            'discoveries': self.discovery_log,
            'success_patterns': dict(self.success_patterns),
            'failure_patterns': dict(self.failure_patterns),
            'summary': {
                'total_attempts': len(self.discovery_log) + len(self.failure_patterns.get('high_scoring_misses', [])),
                'successes': len(self.discovery_log),
                'success_rate': len(self.discovery_log) / max(1, len(self.discovery_log) + len(self.failure_patterns.get('high_scoring_misses', [])))
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nDiscovery log saved to {filename}")


def test_rfh1():
    """Test the pure resonance field approach"""
    
    test_cases = [
        # Start with cases that worked in Phase 1
        (11, 13),                     # 143 - small primes
        (523, 541),                   # Near transition
        (65537, 4294967311),          # Fermat prime
        
        # Then challenging cases
        (101, 103),                   # 10403
        (531, 532),                   # 282492 - at transition
        
        # The hard ones that needed Phase 2
        (7125766127, 6958284019),     # 66-bit arbitrary primes
        (99991, 99989),               # Twin primes
    ]
    
    explorer = ResonanceFieldExplorer()
    
    for p_true, q_true in test_cases:
        n = p_true * q_true
        
        try:
            result = explorer.explore(n, max_candidates=10000)
            
            if result:
                p_found, q_found = result
                if {p_found, q_found} == {p_true, q_true}:
                    print(f"\n✓ CORRECT")
                else:
                    print(f"\n✗ INCORRECT: Expected {p_true} × {q_true}, got {p_found} × {q_found}")
            else:
                print(f"\n✗ FAILED: Could not find factors of {n} = {p_true} × {q_true}")
        
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save discoveries for analysis
    explorer.save_discoveries("rfh1_discoveries.json")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    success_count = len(explorer.discovery_log)
    total_count = success_count + len(explorer.failure_patterns.get('high_scoring_misses', []))
    success_rate = success_count / max(1, total_count) * 100
    print(f"Success rate: {success_rate:.1f}%")
    print(f"See rfh1_discoveries.json for detailed patterns")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_rfh1()
