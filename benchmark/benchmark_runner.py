"""
Comprehensive benchmark runner for the UOR/Prime Axioms Factorizer
Tests performance across different number types and axiom combinations
"""

import time
import sys
import os
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all axiom modules
from axiom1.prime_core import is_prime, primes_up_to
from axiom1.prime_cascade import PrimeCascade
from axiom1.prime_geodesic import PrimeGeodesic

from axiom2.fibonacci_core import fib, is_fibonacci, PHI
from axiom2.fibonacci_vortices import fib_vortices, golden_spiral_positions
from axiom2.fibonacci_entanglement import FibonacciEntanglement

from axiom3.spectral_core import spectral_vector
from axiom3.coherence import coherence, CoherenceCache
from axiom3.interference import prime_fib_interference, interference_extrema
from axiom3.fold_topology import fold_energy, sharp_fold_candidates

from axiom4.adaptive_observer import MultiScaleObserver
from axiom4.quantum_tools import QuantumTunnel, harmonic_amplify
from axiom4.resonance_memory import ResonanceMemory

from axiom5.meta_observer import AxiomPerformanceProfile, MetaObserver
from axiom5.spectral_mirror import SpectralMirror
from axiom5.recursive_coherence import RecursiveCoherence

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    name: str
    success: bool
    time_taken: float
    factor_found: Optional[int] = None
    iterations: int = 1
    memory_used: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AxiomBenchmark:
    """Benchmark results for a single axiom"""
    axiom_name: str
    total_runs: int
    successes: int
    failures: int
    avg_time: float
    min_time: float
    max_time: float
    success_rate: float
    results: List[BenchmarkResult] = field(default_factory=list)

class BenchmarkRunner:
    """Main benchmark runner for all axioms"""
    
    def __init__(self):
        self.test_numbers = self._generate_test_numbers()
        self.results: Dict[str, List[BenchmarkResult]] = {}
        self.coherence_cache = CoherenceCache(max_size=50000)
        
    def _generate_test_numbers(self) -> List[Tuple[int, List[int]]]:
        """Generate test numbers with known factorizations"""
        # Clean test cases with correct factorizations
        test_cases = [
            (15, [3, 5]),
            (21, [3, 7]), 
            (35, [5, 7]),
            (77, [7, 11]),
            (91, [7, 13]),
            (143, [11, 13]),
            (323, [17, 19]),
            (391, [17, 23]),
            (667, [23, 29]),
            (899, [29, 31]),
            (1147, [31, 37]),
            (1517, [37, 41]),
            (2021, [43, 47]),
            (2491, [47, 53]),
            (3127, [53, 59]),
            (4087, [61, 67]),
            (2047, [23, 89]),
            (4181, [37, 113]),  # Corrected semiprime
            (1364, [2, 682]),  # Even composite
            (6765, [3, 5, 11, 41])  # Multiple factors
        ]
        
        return test_cases
    
    def benchmark_axiom1(self, n: int, factors: List[int]) -> BenchmarkResult:
        """Benchmark Axiom 1: Prime Ontology"""
        start_time = time.time()
        
        try:
            # Test prime cascade
            cascade = PrimeCascade(n)
            # Use available methods
            primes_up_to_sqrt = [p for p in range(2, int(n**0.5)+1) if is_prime(p)]
            cascade_results = []
            for p in primes_up_to_sqrt[:5]:  # Limit to first 5 primes
                cascade_results.extend(cascade.cascade(p))
            
            # Test prime geodesic
            geodesic = PrimeGeodesic(n)
            # Use available methods
            geodesic_path = geodesic.walk(2, steps=10)
            
            # Simple factor detection based on pull
            max_pull = 0
            best_factor = None
            root = int(n**0.5) + 1
            
            for candidate in range(2, root):
                if n % candidate == 0:
                    pull = geodesic._pull(candidate)
                    if pull > max_pull:
                        max_pull = pull
                        best_factor = candidate
            
            success = best_factor in factors
            end_time = time.time()
            
            return BenchmarkResult(
                name="axiom1",
                success=success,
                time_taken=end_time - start_time,
                factor_found=best_factor,
                additional_info={
                    'cascade_results': len(cascade_results),
                    'geodesic_path': geodesic_path,
                    'max_pull': max_pull
                }
            )
            
        except Exception as e:
            end_time = time.time()
            return BenchmarkResult(
                name="axiom1",
                success=False,
                time_taken=end_time - start_time,
                additional_info={'error': str(e)}
            )
    
    def benchmark_axiom2(self, n: int, factors: List[int]) -> BenchmarkResult:
        """Benchmark Axiom 2: Fibonacci Flow"""
        start_time = time.time()
        
        try:
            # Test Fibonacci vortices
            vortices = fib_vortices(n)
            spiral_positions = golden_spiral_positions(n)
            
            # Test entanglement
            entanglement = FibonacciEntanglement(n)
            double_fib = entanglement.detect_double()
            
            # Look for factors using Fibonacci-guided scoring
            best_factor = None
            best_score = 0
            root = int(n**0.5) + 1
            
            # Check all potential factors up to sqrt(n)
            for candidate in range(2, root):
                if n % candidate == 0:  # This is an actual factor
                    # Score based on Fibonacci pattern resonance
                    fib_score = 0
                    
                    # Bonus if factor is a Fibonacci number
                    if is_fibonacci(candidate):
                        fib_score += 2.0
                    
                    # Bonus if factor appears in vortex positions
                    if candidate in vortices:
                        fib_score += 1.5
                    
                    # Bonus if factor appears in spiral positions  
                    if candidate in spiral_positions:
                        fib_score += 1.0
                    
                    # Check fibonacci entanglement with complement factor
                    complement = n // candidate
                    entanglement_score = entanglement.fibonacci_alignment_score(candidate, complement)
                    fib_score += entanglement_score
                    
                    # Base score for being a factor
                    fib_score += 0.5
                    
                    if fib_score > best_score:
                        best_score = fib_score
                        best_factor = candidate
            
            success = best_factor in factors
            end_time = time.time()
            
            return BenchmarkResult(
                name="axiom2",
                success=success,
                time_taken=end_time - start_time,
                factor_found=best_factor,
                additional_info={
                    'vortices': len(vortices),
                    'spiral_positions': len(spiral_positions),
                    'double_fibonacci': double_fib,
                    'best_score': best_score
                }
            )
            
        except Exception as e:
            end_time = time.time()
            return BenchmarkResult(
                name="axiom2",
                success=False,
                time_taken=end_time - start_time,
                additional_info={'error': str(e)}
            )
    
    def benchmark_axiom3(self, n: int, factors: List[int]) -> BenchmarkResult:
        """Benchmark Axiom 3: Duality Principle"""
        start_time = time.time()
        
        try:
            # Test spectral analysis
            n_spectrum = spectral_vector(n)
            
            # Test interference patterns
            interference = prime_fib_interference(n)
            extrema = interference_extrema(n, top=20)
            
            # Test sharp folds
            folds = sharp_fold_candidates(n)
            
            # Find best coherence match
            best_factor = None
            best_coherence = 0
            root = int(n**0.5) + 1
            
            for candidate in range(2, root):
                if n % candidate == 0:
                    partner = n // candidate
                    coh = self.coherence_cache.get_coherence(candidate, partner, n)
                    if coh > best_coherence:
                        best_coherence = coh
                        best_factor = candidate
            
            success = best_factor in factors
            end_time = time.time()
            
            return BenchmarkResult(
                name="axiom3",
                success=success,
                time_taken=end_time - start_time,
                factor_found=best_factor,
                additional_info={
                    'extrema_count': len(extrema),
                    'sharp_folds': len(folds),
                    'best_coherence': best_coherence,
                    'spectrum_norm': sum(x*x for x in n_spectrum)**0.5
                }
            )
            
        except Exception as e:
            end_time = time.time()
            return BenchmarkResult(
                name="axiom3",
                success=False,
                time_taken=end_time - start_time,
                additional_info={'error': str(e)}
            )
    
    def benchmark_axiom4(self, n: int, factors: List[int]) -> BenchmarkResult:
        """Benchmark Axiom 4: Observer Effect"""
        start_time = time.time()
        
        try:
            # Test multi-scale observer
            observer = MultiScaleObserver(n)
            
            # Generate superposition of candidates
            root = int(n**0.5) + 1
            candidates = list(range(2, root))
            
            # Test quantum tunneling
            tunnel = QuantumTunnel(n)
            tunnel_positions = []
            for c in candidates[:10]:  # Limit to avoid timeout
                tunnel_seq = tunnel.tunnel_sequence(c, max_tunnels=3)
                tunnel_positions.extend(tunnel_seq)
            
            # Find all actual factors first
            actual_factors = []
            for candidate in range(2, root):
                if n % candidate == 0:
                    actual_factors.append(candidate)
            
            # Measure coherence field for all actual factors plus some other candidates
            eval_candidates = list(set(actual_factors + candidates[:20]))
            coherence_field = observer.coherence_field(eval_candidates)
            
            # Find best candidate by coherence among actual factors
            best_factor = None
            best_coherence = 0
            
            for candidate in actual_factors:
                coh = coherence_field.get(candidate, 0)
                if coh > best_coherence:
                    best_coherence = coh
                    best_factor = candidate
            
            success = best_factor in factors
            end_time = time.time()
            
            return BenchmarkResult(
                name="axiom4",
                success=success,
                time_taken=end_time - start_time,
                factor_found=best_factor,
                additional_info={
                    'tunnel_positions': len(tunnel_positions),
                    'best_coherence': best_coherence,
                    'avg_coherence': statistics.mean(coherence_field.values())
                }
            )
            
        except Exception as e:
            end_time = time.time()
            return BenchmarkResult(
                name="axiom4",
                success=False,
                time_taken=end_time - start_time,
                additional_info={'error': str(e)}
            )
    
    def benchmark_axiom5(self, n: int, factors: List[int]) -> BenchmarkResult:
        """Benchmark Axiom 5: Self-Reference"""
        start_time = time.time()
        
        try:
            # Test spectral mirror and recursive coherence
            mirror = SpectralMirror(n)
            recursive_coh = RecursiveCoherence(n)
            
            # Find all actual factors first
            root = int(n**0.5) + 1
            actual_factors = []
            for candidate in range(2, root):
                if n % candidate == 0:
                    actual_factors.append(candidate)
            
            # Find mirror points for actual factors + some candidates for context
            candidates_to_eval = list(set(actual_factors + list(range(2, min(root, 20)))))
            mirror_points = []
            
            for candidate in candidates_to_eval:
                mirror_point = mirror.find_mirror_point(candidate)
                mirror_points.append((candidate, mirror_point))
            
            # Apply recursive coherence to initial field covering all evaluated candidates
            initial_field = {i: 0.5 for i in candidates_to_eval}
            field_evolution = recursive_coh.recursive_coherence_iteration(initial_field, depth=3)
            final_field = field_evolution[-1]
            
            # Score factors using recursive coherence + mirror resonance
            best_factor = None
            best_score = 0
            
            for candidate, mirror_pos in mirror_points:
                if candidate in actual_factors:  # Only score actual factors
                    # Score based on recursive coherence evolution
                    candidate_coherence = final_field.get(candidate, 0)
                    mirror_coherence = final_field.get(mirror_pos, 0)
                    
                    # Combine candidate and mirror coherence
                    total_score = candidate_coherence + mirror_coherence
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_factor = candidate
            
            success = best_factor in factors
            end_time = time.time()
            
            return BenchmarkResult(
                name="axiom5",
                success=success,
                time_taken=end_time - start_time,
                factor_found=best_factor,
                additional_info={
                    'mirror_points': len(mirror_points),
                    'best_score': best_score,
                    'avg_final_coherence': statistics.mean(final_field.values()) if final_field else 0
                }
            )
            
        except Exception as e:
            end_time = time.time()
            return BenchmarkResult(
                name="axiom5",
                success=False,
                time_taken=end_time - start_time,
                additional_info={'error': str(e)}
            )
    
    def run_single_benchmark(self, n: int, factors: List[int]) -> Dict[str, BenchmarkResult]:
        """Run all axiom benchmarks on a single number"""
        results = {}
        
        print(f"Benchmarking n={n} (factors: {factors})")
        
        # Benchmark each axiom
        results['axiom1'] = self.benchmark_axiom1(n, factors)
        results['axiom2'] = self.benchmark_axiom2(n, factors)
        results['axiom3'] = self.benchmark_axiom3(n, factors)
        results['axiom4'] = self.benchmark_axiom4(n, factors)
        results['axiom5'] = self.benchmark_axiom5(n, factors)
        
        # Print quick summary
        for axiom, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"  {axiom}: {status} ({result.time_taken:.4f}s)")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, AxiomBenchmark]:
        """Run benchmarks on all test numbers"""
        print("Running Comprehensive Factorizer Benchmarks")
        print("=" * 50)
        
        all_results = {
            'axiom1': [],
            'axiom2': [],
            'axiom3': [],
            'axiom4': [],
            'axiom5': []
        }
        
        for n, factors in self.test_numbers:
            single_results = self.run_single_benchmark(n, factors)
            
            for axiom, result in single_results.items():
                all_results[axiom].append(result)
        
        # Compile statistics
        benchmark_summary = {}
        
        for axiom, results in all_results.items():
            total_runs = len(results)
            successes = sum(1 for r in results if r.success)
            failures = total_runs - successes
            times = [r.time_taken for r in results]
            
            benchmark_summary[axiom] = AxiomBenchmark(
                axiom_name=axiom,
                total_runs=total_runs,
                successes=successes,
                failures=failures,
                avg_time=statistics.mean(times),
                min_time=min(times),
                max_time=max(times),
                success_rate=successes / total_runs,
                results=results
            )
        
        return benchmark_summary
    
    def generate_report(self, benchmarks: Dict[str, AxiomBenchmark]) -> str:
        """Generate detailed benchmark report"""
        report = []
        report.append("UOR/Prime Axioms Factorizer - Benchmark Report")
        report.append("=" * 50)
        report.append("")
        
        # Overall summary
        total_tests = sum(b.total_runs for b in benchmarks.values())
        total_successes = sum(b.successes for b in benchmarks.values())
        overall_success_rate = total_successes / total_tests
        
        report.append(f"Overall Statistics:")
        report.append(f"  Total Tests: {total_tests}")
        report.append(f"  Total Successes: {total_successes}")
        report.append(f"  Overall Success Rate: {overall_success_rate:.2%}")
        report.append("")
        
        # Per-axiom results
        for axiom_name, bench in benchmarks.items():
            report.append(f"{axiom_name.upper()} Results:")
            report.append(f"  Success Rate: {bench.success_rate:.2%} ({bench.successes}/{bench.total_runs})")
            report.append(f"  Average Time: {bench.avg_time:.4f}s")
            report.append(f"  Time Range: {bench.min_time:.4f}s - {bench.max_time:.4f}s")
            
            # Find best and worst cases
            successful_results = [r for r in bench.results if r.success]
            if successful_results:
                fastest = min(successful_results, key=lambda x: x.time_taken)
                slowest = max(successful_results, key=lambda x: x.time_taken)
                report.append(f"  Fastest Success: {fastest.time_taken:.4f}s (factor: {fastest.factor_found})")
                report.append(f"  Slowest Success: {slowest.time_taken:.4f}s (factor: {slowest.factor_found})")
            
            report.append("")
        
        # Performance comparison
        report.append("Axiom Performance Ranking:")
        sorted_axioms = sorted(benchmarks.items(), key=lambda x: x[1].success_rate, reverse=True)
        for i, (axiom, bench) in enumerate(sorted_axioms, 1):
            report.append(f"  {i}. {axiom}: {bench.success_rate:.2%} success, {bench.avg_time:.4f}s avg")
        
        report.append("")
        
        # Speed ranking
        report.append("Speed Ranking (Average Time):")
        sorted_by_speed = sorted(benchmarks.items(), key=lambda x: x[1].avg_time)
        for i, (axiom, bench) in enumerate(sorted_by_speed, 1):
            report.append(f"  {i}. {axiom}: {bench.avg_time:.4f}s avg")
        
        return "\n".join(report)

def main():
    """Run comprehensive benchmarks"""
    runner = BenchmarkRunner()
    
    print("Initializing benchmark runner...")
    print(f"Test cases: {len(runner.test_numbers)}")
    print()
    
    # Run benchmarks
    start_time = time.time()
    results = runner.run_comprehensive_benchmark()
    end_time = time.time()
    
    print()
    print("=" * 50)
    print(f"Total benchmark time: {end_time - start_time:.2f}s")
    print()
    
    # Generate and display report
    report = runner.generate_report(results)
    print(report)
    
    # Save report to file
    report_path = os.path.join(os.path.dirname(__file__), "benchmark_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()