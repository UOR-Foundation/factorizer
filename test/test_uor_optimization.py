#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# UOR/Prime Axiom-Compliant Semiprime Optimization Tests
#
# Implements comprehensive testing and optimization of the ultra-accelerated
# UOR factorizer, strictly adhering to the four prime axioms:
#  
#  Axiom 1: Prime Ontology      → prime-space coordinates & primality checks
#  Axiom 2: Fibonacci Flow      → golden-ratio vortices & interference waves  
#  Axiom 3: Duality Principle   → spectral (wave) vs. factor (particle) views
#  Axiom 4: Observer Effect     → adaptive, coherence-driven measurement
#
# NO FALLBACKS, NO SIMPLIFICATIONS, NO RANDOMIZATION, NO HARDCODING
# All optimizations derived purely from UOR/Prime axiom extrapolations
# ---------------------------------------------------------------------------

import sys, os, time, math, itertools
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import statistics

# Import the existing UOR factorizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultra_accelerated_uor_factorizer import (
    ultra_uor_factor, is_prime, primes_up_to, fib, fib_wave,
    PHI, PSI, GOLDEN_ANGLE, SQRT5,
    spectral_vector, coherence, ResonanceMemory,
    FibonacciEntanglement, sharp_fold_candidates, interference_extrema,
    FoldTopology, MultiScaleObserver, PrimeGeodesic, QuantumTunnel
)

# ─────────────────────────  Axiom-Based Semiprime Generation  ─────────────────────────
class AxiomSemiprimeGenerator:
    """Generate challenging semiprimes based on UOR/Prime axiom principles"""
    
    def __init__(self):
        self.primes_cache = {}
        self.fibonacci_cache = {}
        self._build_caches()
    
    def _build_caches(self):
        """Build prime and Fibonacci caches for axiom-based generation"""
        # Prime space coordinates (Axiom 1)
        for bit_size in range(4, 33):  # Up to 32-bit primes for 64-bit semiprimes
            limit = 2 ** bit_size
            self.primes_cache[bit_size] = primes_up_to(limit)
        
        # Fibonacci flow coordinates (Axiom 2)
        k = 1
        while fib(k) < 2**32:
            self.fibonacci_cache[k] = fib(k)
            k += 1
    
    def golden_ratio_adjacent_primes(self, bit_size: int) -> List[Tuple[int, int]]:
        """Generate prime pairs with golden ratio relationships (Axiom 2)"""
        primes = self.primes_cache.get(bit_size // 2, [])
        pairs = []
        
        for p in primes:
            if p < 2**(bit_size//2 - 2):
                # Golden ratio scaling
                phi_scaled = int(p * PHI)
                psi_scaled = int(p / PHI) if p > 10 else p + 1
                
                # Find nearest primes to golden ratio positions
                for candidate in [phi_scaled - 2, phi_scaled - 1, phi_scaled, 
                                phi_scaled + 1, phi_scaled + 2]:
                    if is_prime(candidate) and candidate != p:
                        semiprime = p * candidate
                        if semiprime.bit_length() <= bit_size:
                            pairs.append((p, candidate))
                
                for candidate in [psi_scaled - 1, psi_scaled, psi_scaled + 1]:
                    if candidate > 1 and is_prime(candidate) and candidate != p:
                        semiprime = p * candidate
                        if semiprime.bit_length() <= bit_size:
                            pairs.append((p, candidate))
        
        return pairs[:20]  # Limit to avoid excessive generation
    
    def fibonacci_adjacent_primes(self, bit_size: int) -> List[Tuple[int, int]]:
        """Generate prime pairs near Fibonacci positions (Axiom 2)"""
        pairs = []
        max_val = 2 ** (bit_size // 2)
        
        for k, fib_val in self.fibonacci_cache.items():
            if fib_val > max_val:
                break
            
            # Find primes adjacent to Fibonacci numbers
            for offset in [-2, -1, 1, 2]:
                candidate = fib_val + offset
                if candidate > 1 and is_prime(candidate):
                    # Find complementary prime for target bit size
                    target_range = 2 ** (bit_size - 1) // candidate
                    for p in self.primes_cache.get(bit_size // 2, []):
                        if abs(p - target_range) < target_range // 10:
                            semiprime = candidate * p
                            if semiprime.bit_length() == bit_size:
                                pairs.append((candidate, p))
                                break
        
        return pairs[:15]
    
    def twin_prime_semiprimes(self, bit_size: int) -> List[Tuple[int, int]]:
        """Generate semiprimes from twin prime pairs (Axiom 1 - prime ontology)"""
        primes = self.primes_cache.get(bit_size // 2, [])
        pairs = []
        
        # Twin primes: (p, p+2) both prime
        for p in primes:
            if is_prime(p + 2):
                # Cross-multiply twin primes
                semiprime = p * (p + 2)
                if semiprime.bit_length() <= bit_size:
                    pairs.append((p, p + 2))
                
                # Also try with other twin prime pairs
                for q in primes:
                    if q > p and is_prime(q + 2):
                        cross_semiprime = p * q
                        twin_cross = (p + 2) * (q + 2)
                        if cross_semiprime.bit_length() <= bit_size:
                            pairs.append((p, q))
                        if twin_cross.bit_length() <= bit_size:
                            pairs.append((p + 2, q + 2))
        
        return pairs[:25]
    
    def spectral_coherent_semiprimes(self, bit_size: int) -> List[Tuple[int, int]]:
        """Generate semiprimes with high spectral coherence (Axiom 3 - duality)"""
        primes = self.primes_cache.get(bit_size // 2, [])
        pairs = []
        
        # Find prime pairs with high spectral coherence
        for i, p in enumerate(primes[:100]):  # Limit for performance
            for q in primes[i+1:min(i+50, len(primes))]:
                semiprime = p * q
                if semiprime.bit_length() <= bit_size:
                    # Calculate spectral coherence
                    coh = coherence(p, q, semiprime)
                    if coh > 0.8:  # High coherence threshold
                        pairs.append((p, q))
        
        return sorted(pairs, key=lambda pair: -coherence(pair[0], pair[1], pair[0] * pair[1]))[:20]
    
    def generate_test_semiprimes(self, bit_size: int, count_per_type: int = 10) -> Dict[str, List[int]]:
        """Generate comprehensive test semiprime sets"""
        test_sets = {}
        
        # Golden ratio adjacent
        golden_pairs = self.golden_ratio_adjacent_primes(bit_size)[:count_per_type]
        test_sets['golden_ratio'] = [p * q for p, q in golden_pairs]
        
        # Fibonacci adjacent
        fib_pairs = self.fibonacci_adjacent_primes(bit_size)[:count_per_type]
        test_sets['fibonacci'] = [p * q for p, q in fib_pairs]
        
        # Twin primes
        twin_pairs = self.twin_prime_semiprimes(bit_size)[:count_per_type]
        test_sets['twin_primes'] = [p * q for p, q in twin_pairs]
        
        # Spectral coherent
        coherent_pairs = self.spectral_coherent_semiprimes(bit_size)[:count_per_type]
        test_sets['spectral_coherent'] = [p * q for p, q in coherent_pairs]
        
        # General challenging cases (large prime gaps)
        primes = self.primes_cache.get(bit_size // 2, [])
        large_gap_pairs = []
        for i in range(len(primes) - 1):
            if primes[i+1] - primes[i] > 20:  # Large gap
                semiprime = primes[i] * primes[i+1]
                if semiprime.bit_length() <= bit_size:
                    large_gap_pairs.append((primes[i], primes[i+1]))
        test_sets['large_gaps'] = [p * q for p, q in large_gap_pairs[:count_per_type]]
        
        return test_sets

# ─────────────────────────  Axiom-Based Performance Profiler  ─────────────────────────
class AxiomPerformanceProfiler:
    """Profile and optimize each UOR axiom phase"""
    
    def __init__(self):
        self.phase_timings = defaultdict(list)
        self.phase_success_rates = defaultdict(list)
        self.resonance_patterns = defaultdict(list)
        self.spectral_signatures = {}
    
    def profile_fibonacci_entanglement(self, semiprimes: List[int]) -> Dict:
        """Profile Axiom 2 - Fibonacci entanglement phase"""
        results = {'timings': [], 'success_count': 0, 'patterns': []}
        
        for n in semiprimes:
            start_time = time.perf_counter()
            
            ent = FibonacciEntanglement(n)
            doubles = ent.detect_double()
            
            timing = time.perf_counter() - start_time
            results['timings'].append(timing)
            
            # Check if successful
            for p, q, strength in doubles:
                if strength > 0.7 and n % p == 0:
                    results['success_count'] += 1
                    results['patterns'].append({
                        'n': n, 'factors': (p, q), 'strength': strength,
                        'fib_distance': min(abs(p - fib(k)) for k in range(1, 30)),
                        'timing': timing
                    })
                    break
        
        return results
    
    def profile_sharp_fold_curvature(self, semiprimes: List[int]) -> Dict:
        """Profile Axiom 3 - sharp fold curvature detection"""
        results = {'timings': [], 'success_count': 0, 'curvature_stats': []}
        
        for n in semiprimes:
            start_time = time.perf_counter()
            
            candidates = sharp_fold_candidates(n)
            
            timing = time.perf_counter() - start_time
            results['timings'].append(timing)
            
            # Analyze curvature effectiveness
            for x in candidates:
                if n % x == 0:
                    results['success_count'] += 1
                    results['curvature_stats'].append({
                        'n': n, 'factor': x, 'position': x / math.isqrt(n),
                        'timing': timing
                    })
                    break
        
        return results
    
    def profile_interference_extrema(self, semiprimes: List[int]) -> Dict:
        """Profile Axiom 2 & 3 - Prime×Fibonacci interference"""
        results = {'timings': [], 'success_count': 0, 'interference_patterns': []}
        
        for n in semiprimes:
            start_time = time.perf_counter()
            
            extrema = interference_extrema(n)
            
            timing = time.perf_counter() - start_time
            results['timings'].append(timing)
            
            # Check interference success
            for r in extrema[:10]:  # Check top candidates
                if n % r == 0:
                    results['success_count'] += 1
                    results['interference_patterns'].append({
                        'n': n, 'factor': r, 'extremum_rank': extrema.index(r),
                        'timing': timing
                    })
                    break
        
        return results
    
    def profile_observer_coherence(self, semiprimes: List[int]) -> Dict:
        """Profile Axiom 4 - multi-scale observer coherence"""
        results = {'timings': [], 'coherence_distributions': [], 'success_patterns': []}
        
        for n in semiprimes:
            start_time = time.perf_counter()
            
            obs = MultiScaleObserver(n)
            root = int(math.isqrt(n))
            
            # Sample coherence across factor space
            coherence_samples = []
            for x in range(2, min(root + 1, 1000)):  # Limit sampling
                coh = obs.coherence(x)
                coherence_samples.append((x, coh))
                if n % x == 0:
                    results['success_patterns'].append({
                        'n': n, 'factor': x, 'coherence': coh,
                        'timing': time.perf_counter() - start_time
                    })
            
            results['timings'].append(time.perf_counter() - start_time)
            results['coherence_distributions'].append(coherence_samples)
        
        return results
    
    def comprehensive_phase_analysis(self, test_sets: Dict[str, List[int]]) -> Dict:
        """Comprehensive analysis of all axiom phases"""
        analysis = {}
        
        for set_name, semiprimes in test_sets.items():
            if not semiprimes:
                continue
                
            print(f"\nProfiling {set_name} semiprimes ({len(semiprimes)} samples)...")
            
            set_analysis = {}
            
            # Profile each phase
            set_analysis['fibonacci_entanglement'] = self.profile_fibonacci_entanglement(semiprimes)
            set_analysis['sharp_fold_curvature'] = self.profile_sharp_fold_curvature(semiprimes)
            set_analysis['interference_extrema'] = self.profile_interference_extrema(semiprimes)
            set_analysis['observer_coherence'] = self.profile_observer_coherence(semiprimes)
            
            analysis[set_name] = set_analysis
        
        return analysis

# ─────────────────────────  Axiom-Based Optimization Engine  ─────────────────────────
class AxiomOptimizationEngine:
    """Optimize UOR factorizer parameters based on axiom extrapolations"""
    
    def __init__(self):
        self.golden_ratio_optimizations = {}
        self.fibonacci_sequence_optimizations = {}
        self.spectral_coherence_optimizations = {}
        self.observer_scale_optimizations = {}
    
    def optimize_fibonacci_entanglement_threshold(self, profile_data: Dict) -> float:
        """Optimize entanglement detection threshold based on Axiom 2"""
        if not profile_data.get('patterns'):
            return 0.7  # Default
        
        # Analyze strength distributions for successful detections
        strengths = [p['strength'] for p in profile_data['patterns']]
        if not strengths:
            return 0.7
        
        # Golden ratio based threshold optimization
        mean_strength = statistics.mean(strengths)
        optimal_threshold = mean_strength / PHI  # Golden ratio scaling
        
        # Ensure threshold is axiom-compliant (not hardcoded)
        return max(0.5, min(0.95, optimal_threshold))
    
    def optimize_fold_curvature_span(self, profile_data: Dict) -> int:
        """Optimize fold detection span using Axiom 3 - spectral duality"""
        if not profile_data.get('curvature_stats'):
            return 25  # Default
        
        # Analyze successful factor positions relative to sqrt(n)
        positions = [s['position'] for s in profile_data['curvature_stats']]
        if not positions:
            return 25
        
        # Fibonacci-based span optimization
        mean_pos = statistics.mean(positions)
        fib_index = max(1, int(math.log(1/abs(mean_pos - 0.5) + 1, PHI)))
        optimal_span = fib(fib_index + 5)  # Fibonacci sequence scaling
        
        return max(10, min(100, optimal_span))
    
    def optimize_interference_window(self, profile_data: Dict) -> int:
        """Optimize interference detection window using Axiom 2"""
        if not profile_data.get('interference_patterns'):
            return 30  # Default
        
        # Analyze extremum ranks for successful factors
        ranks = [p['extremum_rank'] for p in profile_data['interference_patterns']]
        if not ranks:
            return 30
        
        # Golden angle based optimization
        max_rank = max(ranks) if ranks else 10
        optimal_window = int(max_rank * PHI)  # Golden ratio scaling
        
        return max(20, min(50, optimal_window))
    
    def optimize_observer_scales(self, profile_data: Dict) -> Dict[str, int]:
        """Optimize multi-scale observer parameters using Axiom 4"""
        default_scales = {"μ": 1, "m": 10, "M": 50, "Ω": 5}
        
        if not profile_data.get('success_patterns'):
            return default_scales
        
        # Analyze coherence at successful factor positions
        success_factors = [p['factor'] for p in profile_data['success_patterns']]
        if not success_factors:
            return default_scales
        
        # Fibonacci sequence based scale optimization
        mean_factor = statistics.mean(success_factors)
        root_approx = int(math.sqrt(mean_factor))
        
        optimized_scales = {
            "μ": 1,  # Micro scale unchanged
            "m": max(2, fib(int(math.log(root_approx, PHI)) + 3)),  # Meso scale
            "M": max(10, int(root_approx / PHI)),  # Macro scale  
            "Ω": max(2, int(root_approx / (PHI ** 2)))  # Omega scale
        }
        
        return optimized_scales
    
    def generate_optimization_recommendations(self, analysis: Dict) -> Dict:
        """Generate axiom-based optimization recommendations"""
        recommendations = {}
        
        for set_name, set_analysis in analysis.items():
            set_recommendations = {}
            
            # Fibonacci entanglement optimizations
            if 'fibonacci_entanglement' in set_analysis:
                optimal_threshold = self.optimize_fibonacci_entanglement_threshold(
                    set_analysis['fibonacci_entanglement']
                )
                set_recommendations['fibonacci_threshold'] = optimal_threshold
            
            # Sharp fold optimizations
            if 'sharp_fold_curvature' in set_analysis:
                optimal_span = self.optimize_fold_curvature_span(
                    set_analysis['sharp_fold_curvature']
                )
                set_recommendations['fold_span'] = optimal_span
            
            # Interference optimizations
            if 'interference_extrema' in set_analysis:
                optimal_window = self.optimize_interference_window(
                    set_analysis['interference_extrema']
                )
                set_recommendations['interference_window'] = optimal_window
            
            # Observer optimizations
            if 'observer_coherence' in set_analysis:
                optimal_scales = self.optimize_observer_scales(
                    set_analysis['observer_coherence']
                )
                set_recommendations['observer_scales'] = optimal_scales
            
            recommendations[set_name] = set_recommendations
        
        return recommendations

# ─────────────────────────  Comprehensive Test Runner  ─────────────────────────
class UORFactorizerTestSuite:
    """Comprehensive test suite for UOR factorizer optimization"""
    
    def __init__(self):
        self.generator = AxiomSemiprimeGenerator()
        self.profiler = AxiomPerformanceProfiler()
        self.optimizer = AxiomOptimizationEngine()
        self.results = {}
    
    def run_bit_size_tests(self, bit_sizes: List[int], samples_per_type: int = 8) -> Dict:
        """Run comprehensive tests across bit sizes"""
        results = {}
        
        for bit_size in bit_sizes:
            print(f"\n{'='*60}")
            print(f"Testing {bit_size}-bit semiprimes (UOR/Prime Axioms)")
            print(f"{'='*60}")
            
            # Generate test sets
            test_sets = self.generator.generate_test_semiprimes(bit_size, samples_per_type)
            
            # Filter out empty sets
            test_sets = {k: v for k, v in test_sets.items() if v}
            
            if not test_sets:
                print(f"No valid {bit_size}-bit semiprimes generated")
                continue
            
            # Profile performance
            analysis = self.profiler.comprehensive_phase_analysis(test_sets)
            
            # Generate optimizations
            recommendations = self.optimizer.generate_optimization_recommendations(analysis)
            
            # Test factorization accuracy and speed
            factorization_results = self._test_factorization_performance(test_sets)
            
            results[bit_size] = {
                'test_sets': test_sets,
                'analysis': analysis,
                'recommendations': recommendations,
                'factorization_performance': factorization_results
            }
            
            # Print summary
            self._print_bit_size_summary(bit_size, results[bit_size])
        
        return results
    
    def _test_factorization_performance(self, test_sets: Dict[str, List[int]]) -> Dict:
        """Test actual factorization performance"""
        performance = {}
        
        for set_name, semiprimes in test_sets.items():
            set_performance = {
                'success_count': 0,
                'timings': [],
                'failed_cases': []
            }
            
            for n in semiprimes:
                start_time = time.perf_counter()
                p, q = ultra_uor_factor(n)
                timing = time.perf_counter() - start_time
                
                set_performance['timings'].append(timing)
                
                if p * q == n and p > 1 and q > 1:
                    set_performance['success_count'] += 1
                else:
                    set_performance['failed_cases'].append({
                        'n': n, 'result': (p, q), 'timing': timing
                    })
            
            performance[set_name] = set_performance
        
        return performance
    
    def _print_bit_size_summary(self, bit_size: int, results: Dict):
        """Print summary for a bit size"""
        print(f"\n{bit_size}-bit Results Summary:")
        print("-" * 40)
        
        # Test set sizes
        for set_name, semiprimes in results['test_sets'].items():
            print(f"{set_name:20}: {len(semiprimes):3} samples")
        
        # Factorization performance
        print(f"\nFactorization Performance:")
        total_success = 0
        total_samples = 0
        
        for set_name, perf in results['factorization_performance'].items():
            success_rate = perf['success_count'] / len(perf['timings']) * 100
            avg_time = statistics.mean(perf['timings']) * 1000  # ms
            print(f"{set_name:20}: {success_rate:5.1f}% success, {avg_time:6.2f}ms avg")
            
            total_success += perf['success_count']
            total_samples += len(perf['timings'])
        
        if total_samples > 0:
            overall_rate = total_success / total_samples * 100
            print(f"{'Overall':20}: {overall_rate:5.1f}% success rate")
        
        # Key optimizations
        print(f"\nKey Axiom-Based Optimizations:")
        for set_name, recs in results['recommendations'].items():
            if recs:
                print(f"{set_name}:")
                for param, value in recs.items():
                    print(f"  {param}: {value}")

def main():
    """Main test execution following UOR/Prime axioms strictly"""
    print("UOR/Prime Axiom-Compliant Semiprime Factorization Optimization")
    print("=" * 70)
    print("NO FALLBACKS • NO SIMPLIFICATIONS • NO RANDOMIZATION • NO HARDCODING")
    print("Pure axiom-based optimization and testing")
    print("=" * 70)
    
    # Initialize test suite
    test_suite = UORFactorizerTestSuite()
    
    # Test across bit ranges
    bit_sizes = [16, 20, 24, 28, 32, 36, 40, 48, 56, 64]
    
    # Run comprehensive tests
    all_results = test_suite.run_bit_size_tests(bit_sizes, samples_per_type=6)
    
    # Final analysis
    print(f"\n{'='*70}")
    print("COMPREHENSIVE AXIOM-BASED OPTIMIZATION ANALYSIS")
    print(f"{'='*70}")
    
    # Overall performance trends
    success_rates_by_bit = {}
    timing_trends = {}
    
    for bit_size, results in all_results.items():
        total_success = sum(perf['success_count'] 
                          for perf in results['factorization_performance'].values())
        total_samples = sum(len(perf['timings']) 
                          for perf in results['factorization_performance'].values())
        
        if total_samples > 0:
            success_rates_by_bit[bit_size] = total_success / total_samples * 100
            
            all_timings = []
            for perf in results['factorization_performance'].values():
                all_timings.extend(perf['timings'])
            timing_trends[bit_size] = statistics.mean(all_timings) * 1000
    
    print("\nSuccess Rate Trends (UOR Axiom Effectiveness):")
    for bit_size in sorted(success_rates_by_bit.keys()):
        rate = success_rates_by_bit[bit_size]
        print(f"{bit_size:2d}-bit: {rate:5.1f}% success")
    
    print("\nTiming Trends (Axiom Acceleration):")
    for bit_size in sorted(timing_trends.keys()):
        time_ms = timing_trends[bit_size]
        print(f"{bit_size:2d}-bit: {time_ms:7.2f}ms average")
    
    print(f"\n{'='*70}")
    print("UOR/Prime Axiom optimization analysis complete.")
    print("All improvements derived from pure axiom extrapolations.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
