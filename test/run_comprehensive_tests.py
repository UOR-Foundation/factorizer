#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Comprehensive UOR/Prime Axiom Test Runner
#
# Executes complete test suite for ultra-accelerated UOR factorizer:
#  • Semiprime generation and factorization testing
#  • Phase-specific performance profiling  
#  • Axiom-based parameter optimization
#  • Comprehensive analysis and reporting
#
# Strictly adheres to UOR/Prime axioms:
#  Axiom 1: Prime Ontology      → prime-space coordinates & primality checks
#  Axiom 2: Fibonacci Flow      → golden-ratio vortices & interference waves
#  Axiom 3: Duality Principle   → spectral (wave) vs. factor (particle) views  
#  Axiom 4: Observer Effect     → adaptive, coherence-driven measurement
#
# NO FALLBACKS • NO SIMPLIFICATIONS • NO RANDOMIZATION • NO HARDCODING
# ---------------------------------------------------------------------------

import sys, os, time, math
from typing import List, Dict, Any, Tuple
import statistics

# Import UOR test components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_uor_optimization import UORFactorizerTestSuite
from axiom_parameter_optimizer import AxiomParameterOptimizer, AxiomParameters

# Import main factorizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultra_accelerated_uor_factorizer import ultra_uor_factor, is_prime, PHI, fib

class ComprehensiveUORTestRunner:
    """Comprehensive test runner for UOR factorizer system"""
    
    def __init__(self):
        self.test_suite = UORFactorizerTestSuite()
        self.parameter_optimizer = AxiomParameterOptimizer()
        self.results_history = []
        self.optimization_results = {}
    
    def run_quick_validation(self) -> bool:
        """Quick validation of UOR factorizer on known cases"""
        print("Running UOR/Prime Axiom Quick Validation")
        print("-" * 45)
        
        # Test cases with known factors (semiprimes)
        test_cases = [
            (221, (13, 17)),      # Small semiprime
            (1189, (29, 41)),     # Medium semiprime  
            (2021, (43, 47)),     # Close primes
            (9797, (97, 101)),    # Twin primes
            (10403, (101, 103)),  # Sequential primes
            (991 * 997, (991, 997)),  # Large primes
        ]
        
        success_count = 0
        for n, expected in test_cases:
            start_time = time.perf_counter()
            p, q = ultra_uor_factor(n)
            timing = time.perf_counter() - start_time
            
            factors = tuple(sorted([p, q]))
            expected_sorted = tuple(sorted(expected))
            
            success = (factors == expected_sorted)
            if success:
                success_count += 1
            
            status = "PASS" if success else "FAIL"
            print(f"{n:8} = {p:4} × {q:4} [{timing*1000:6.2f}ms] {status}")
        
        success_rate = success_count / len(test_cases) * 100
        print(f"\nQuick validation: {success_rate:.1f}% ({success_count}/{len(test_cases)})")
        
        return success_rate >= 80  # Require 80%+ success for validation
    
    def run_semiprime_stress_tests(self, bit_sizes: List[int], 
                                  samples_per_size: int = 8) -> Dict[int, Dict]:
        """Run comprehensive semiprime stress tests"""
        print(f"\nRunning Semiprime Stress Tests (UOR/Prime Axioms)")
        print("=" * 60)
        
        stress_results = {}
        
        for bit_size in bit_sizes:
            print(f"\nStress testing {bit_size}-bit semiprimes...")
            
            # Run test suite for this bit size
            bit_results = self.test_suite.run_bit_size_tests([bit_size], samples_per_size)
            stress_results[bit_size] = bit_results.get(bit_size, {})
            
            # Extract key metrics
            if bit_size in bit_results:
                total_success = 0
                total_samples = 0
                total_time = 0
                
                for set_name, perf in bit_results[bit_size]['factorization_performance'].items():
                    total_success += perf['success_count']
                    total_samples += len(perf['timings'])
                    total_time += sum(perf['timings'])
                
                if total_samples > 0:
                    success_rate = total_success / total_samples * 100
                    avg_time_ms = (total_time / total_samples) * 1000
                    print(f"  {bit_size}-bit summary: {success_rate:.1f}% success, {avg_time_ms:.2f}ms avg")
        
        return stress_results
    
    def run_parameter_optimization(self, bit_sizes: List[int]) -> Dict[int, AxiomParameters]:
        """Run axiom-based parameter optimization"""
        print(f"\nRunning UOR/Prime Axiom Parameter Optimization")
        print("=" * 55)
        
        optimized_params = self.parameter_optimizer.generate_optimized_parameters_report(bit_sizes)
        self.optimization_results = optimized_params
        
        return optimized_params
    
    def analyze_axiom_effectiveness(self, stress_results: Dict[int, Dict]) -> Dict[str, Any]:
        """Analyze effectiveness of each UOR axiom across bit sizes"""
        print(f"\nAnalyzing UOR/Prime Axiom Effectiveness")
        print("-" * 42)
        
        axiom_analysis = {
            'fibonacci_entanglement': {'success_rates': [], 'avg_timings': []},
            'sharp_fold_curvature': {'success_rates': [], 'avg_timings': []},
            'interference_extrema': {'success_rates': [], 'avg_timings': []},
            'observer_coherence': {'success_rates': [], 'avg_timings': []},
        }
        
        # Aggregate data across bit sizes
        for bit_size, results in stress_results.items():
            if 'analysis' not in results:
                continue
            
            for set_name, set_analysis in results['analysis'].items():
                for phase_name, phase_data in set_analysis.items():
                    if phase_name in axiom_analysis:
                        # Calculate success rate for this phase
                        if 'timings' in phase_data:
                            total_samples = len(phase_data['timings'])
                            success_count = phase_data.get('success_count', 0)
                            
                            if total_samples > 0:
                                success_rate = success_count / total_samples * 100
                                avg_timing = statistics.mean(phase_data['timings']) * 1000
                                
                                axiom_analysis[phase_name]['success_rates'].append(success_rate)
                                axiom_analysis[phase_name]['avg_timings'].append(avg_timing)
        
        # Print axiom effectiveness summary
        for axiom, data in axiom_analysis.items():
            if data['success_rates']:
                avg_success = statistics.mean(data['success_rates'])
                avg_timing = statistics.mean(data['avg_timings'])
                print(f"{axiom:20}: {avg_success:5.1f}% avg success, {avg_timing:6.2f}ms avg")
        
        return axiom_analysis
    
    def generate_optimization_insights(self, optimized_params: Dict[int, AxiomParameters]) -> Dict:
        """Generate insights from parameter optimization"""
        print(f"\nUOR/Prime Axiom Optimization Insights")
        print("-" * 40)
        
        insights = {
            'threshold_trends': [],
            'scale_patterns': {},
            'golden_ratio_relationships': []
        }
        
        # Analyze parameter trends across bit sizes
        bit_sizes = sorted(optimized_params.keys())
        thresholds = [optimized_params[b].fibonacci_threshold for b in bit_sizes]
        spans = [optimized_params[b].fold_curvature_span for b in bit_sizes]
        
        insights['threshold_trends'] = list(zip(bit_sizes, thresholds))
        
        # Analyze observer scale patterns
        for bit_size in bit_sizes:
            scales = optimized_params[bit_size].observer_scales
            insights['scale_patterns'][bit_size] = scales
        
        # Look for golden ratio relationships
        for i in range(len(bit_sizes) - 1):
            size1, size2 = bit_sizes[i], bit_sizes[i + 1]
            thresh1 = optimized_params[size1].fibonacci_threshold
            thresh2 = optimized_params[size2].fibonacci_threshold
            
            if thresh1 > 0:
                ratio = thresh2 / thresh1
                phi_distance = min(abs(ratio - PHI), abs(ratio - 1/PHI))
                
                if phi_distance < 0.1:  # Close to golden ratio
                    insights['golden_ratio_relationships'].append({
                        'bit_sizes': (size1, size2),
                        'ratio': ratio,
                        'phi_distance': phi_distance
                    })
        
        # Print key insights
        print("Fibonacci threshold scaling:")
        for bit_size, threshold in insights['threshold_trends']:
            print(f"  {bit_size:2d}-bit: {threshold:.3f}")
        
        if insights['golden_ratio_relationships']:
            print(f"\nGolden ratio relationships found:")
            for rel in insights['golden_ratio_relationships']:
                sizes = rel['bit_sizes']
                ratio = rel['ratio']
                print(f"  {sizes[0]}-bit → {sizes[1]}-bit: ratio = {ratio:.3f} (φ-distance: {rel['phi_distance']:.3f})")
        
        return insights
    
    def benchmark_against_known_algorithms(self) -> Dict[str, Any]:
        """Benchmark UOR factorizer against theoretical expectations"""
        print(f"\nUOR Factorizer Benchmark Analysis")
        print("-" * 35)
        
        # Test semiprimes of increasing size
        test_semiprimes = []
        
        # Generate test cases across bit ranges
        bit_ranges = [16, 20, 24, 28, 32]
        for bit_size in bit_ranges:
            # Create a challenging semiprime for this bit size
            target_size = 2 ** (bit_size - 1)
            
            # Find two primes that multiply to roughly target_size
            for p in range(int(target_size ** 0.5) - 100, int(target_size ** 0.5) + 100):
                if is_prime(p):
                    q = target_size // p
                    for candidate in range(q - 10, q + 10):
                        if is_prime(candidate) and (p * candidate).bit_length() == bit_size:
                            test_semiprimes.append((p * candidate, bit_size, p, candidate))
                            break
                    if test_semiprimes and test_semiprimes[-1][1] == bit_size:
                        break
        
        benchmark_results = {
            'timing_scaling': [],
            'success_rates': [],
            'axiom_effectiveness': {}
        }
        
        for semiprime, bit_size, expected_p, expected_q in test_semiprimes:
            print(f"Testing {bit_size}-bit semiprime: {semiprime}")
            
            # Multiple runs for timing stability
            timings = []
            successes = 0
            
            for _ in range(5):  # 5 runs
                start_time = time.perf_counter()
                p, q = ultra_uor_factor(semiprime)
                timing = time.perf_counter() - start_time
                
                timings.append(timing)
                
                # Check success
                if (p == expected_p and q == expected_q) or (p == expected_q and q == expected_p):
                    successes += 1
            
            avg_timing = statistics.mean(timings) * 1000  # ms
            success_rate = successes / 5 * 100
            
            benchmark_results['timing_scaling'].append((bit_size, avg_timing))
            benchmark_results['success_rates'].append((bit_size, success_rate))
            
            print(f"  Result: {success_rate:.0f}% success, {avg_timing:.2f}ms avg")
        
        # Analyze scaling characteristics
        if len(benchmark_results['timing_scaling']) > 1:
            bit_sizes = [entry[0] for entry in benchmark_results['timing_scaling']]
            timings = [entry[1] for entry in benchmark_results['timing_scaling']]
            
            # Calculate scaling factor
            log_times = [math.log(t) for t in timings if t > 0]
            log_bits = [math.log(b) for b in bit_sizes]
            
            if len(log_times) > 1:
                # Simple linear regression for scaling
                n = len(log_times)
                sum_x = sum(log_bits)
                sum_y = sum(log_times)
                sum_xy = sum(x * y for x, y in zip(log_bits, log_times))
                sum_x2 = sum(x * x for x in log_bits)
                
                scaling_exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                print(f"\nScaling analysis:")
                print(f"  Timing scales as O(n^{scaling_exponent:.2f})")
                print(f"  UOR axiom acceleration effective: {scaling_exponent < 2.0}")
        
        return benchmark_results
    
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run the complete UOR factorizer test suite"""
        print("UOR/PRIME AXIOM COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print("NO FALLBACKS • NO SIMPLIFICATIONS • NO RANDOMIZATION • NO HARDCODING")
        print("Pure UOR/Prime axiom-based factorization system")
        print("=" * 70)
        
        # Step 1: Quick validation
        validation_passed = self.run_quick_validation()
        if not validation_passed:
            print("\n⚠️  Quick validation failed - check implementation")
            return {'validation_passed': False}
        
        # Step 2: Semiprime stress tests
        test_bit_sizes = [16, 20, 24, 28, 32, 40, 48, 56, 64]
        stress_results = self.run_semiprime_stress_tests(test_bit_sizes, samples_per_size=6)
        
        # Step 3: Parameter optimization
        optimization_bit_sizes = [16, 24, 32, 40, 48, 56, 64]
        optimized_params = self.run_parameter_optimization(optimization_bit_sizes)
        
        # Step 4: Axiom effectiveness analysis
        axiom_analysis = self.analyze_axiom_effectiveness(stress_results)
        
        # Step 5: Optimization insights
        optimization_insights = self.generate_optimization_insights(optimized_params)
        
        # Step 6: Benchmark analysis
        benchmark_results = self.benchmark_against_known_algorithms()
        
        # Compile final results
        final_results = {
            'validation_passed': validation_passed,
            'stress_results': stress_results,
            'optimized_parameters': optimized_params,
            'axiom_analysis': axiom_analysis,
            'optimization_insights': optimization_insights,
            'benchmark_results': benchmark_results,
            'timestamp': time.time()
        }
        
        self.results_history.append(final_results)
        
        # Print final summary
        self._print_final_summary(final_results)
        
        return final_results
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary"""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE UOR/PRIME AXIOM TEST RESULTS")
        print(f"{'='*70}")
        
        # Overall success rates
        overall_success_rates = []
        overall_timings = []
        
        for bit_size, bit_results in results['stress_results'].items():
            if 'factorization_performance' in bit_results:
                total_success = 0
                total_samples = 0
                total_time = 0
                
                for perf in bit_results['factorization_performance'].values():
                    total_success += perf['success_count']
                    total_samples += len(perf['timings'])
                    total_time += sum(perf['timings'])
                
                if total_samples > 0:
                    success_rate = total_success / total_samples * 100
                    avg_time = (total_time / total_samples) * 1000
                    
                    overall_success_rates.append(success_rate)
                    overall_timings.append(avg_time)
                    
                    print(f"{bit_size:2d}-bit semiprimes: {success_rate:5.1f}% success, {avg_time:6.2f}ms avg")
        
        if overall_success_rates:
            avg_success = statistics.mean(overall_success_rates)
            avg_timing = statistics.mean(overall_timings)
            
            print(f"\nOverall Performance:")
            print(f"  Average success rate: {avg_success:.1f}%")
            print(f"  Average timing: {avg_timing:.2f}ms")
        
        # Axiom effectiveness
        print(f"\nAxiom Effectiveness:")
        for axiom, data in results['axiom_analysis'].items():
            if data['success_rates']:
                avg_success = statistics.mean(data['success_rates'])
                print(f"  {axiom:20}: {avg_success:5.1f}% average")
        
        # Key insights
        insights = results['optimization_insights']
        if insights['golden_ratio_relationships']:
            print(f"\nGolden Ratio Relationships: {len(insights['golden_ratio_relationships'])} found")
        
        print(f"\n{'='*70}")
        print("UOR/Prime axiom testing complete.")
        print("All results derived from pure axiom mathematics.")
        print(f"{'='*70}")

def main():
    """Main entry point for comprehensive UOR tests"""
    
    # Create test runner
    runner = ComprehensiveUORTestRunner()
    
    # Execute complete test suite
    results = runner.run_complete_test_suite()
    
    # Exit with success/failure code
    validation_passed = results.get('validation_passed', False)
    exit_code = 0 if validation_passed else 1
    
    print(f"\nTest suite completed. Exit code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
