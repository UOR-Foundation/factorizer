#!/usr/bin/env python3
"""
Practical Prime Resonator Tuning Script
=======================================

This script finds optimal parameter values for the Prime Resonator
by testing variations and measuring performance on diverse test cases.
"""

import time
import math
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# We'll test by modifying prime_resonator.py directly
# First, let's create a baseline measurement


@dataclass
class TuningResult:
    """Result of testing a parameter configuration"""
    param_name: str
    param_value: Any
    bit_length: int
    avg_time: float
    success_rate: float
    phase1_hit_rate: float


class PracticalTuner:
    """Practical tuning for Prime Resonator"""
    
    def __init__(self):
        self.results = []
        self.test_cases = {}
        
    def generate_test_cases(self, bit_sizes: List[int], cases_per_size: int = 5):
        """Generate test cases using the prime_resonator module"""
        import prime_resonator as pr
        
        for bits in bit_sizes:
            self.test_cases[bits] = []
            
            for i in range(cases_per_size):
                # Generate random semiprime
                n, p, q = pr._rand_semiprime(bits)
                self.test_cases[bits].append({
                    'n': n,
                    'p': p,
                    'q': q,
                    'balanced': abs(p.bit_length() - q.bit_length()) <= 1
                })
                
        print(f"Generated {sum(len(v) for v in self.test_cases.values())} test cases")
    
    def measure_baseline(self) -> Dict[int, float]:
        """Measure baseline performance"""
        import prime_resonator as pr
        
        baseline_times = {}
        
        for bit_length, cases in self.test_cases.items():
            times = []
            for case in cases:
                start = time.perf_counter()
                try:
                    p, q = pr.prime_resonate(case['n'])
                    elapsed = time.perf_counter() - start
                    if p * q == case['n']:
                        times.append(elapsed)
                except:
                    pass
            
            if times:
                baseline_times[bit_length] = sum(times) / len(times)
            else:
                baseline_times[bit_length] = float('inf')
                
        return baseline_times
    
    def test_parameter_variation(self, param_name: str, variations: List[Tuple[str, str]]):
        """Test variations of a parameter by modifying prime_resonator.py"""
        
        # Read original file
        with open('prime_resonator.py', 'r') as f:
            original_content = f.read()
            
        results = []
        
        for old_value, new_value in variations:
            # Modify the file
            modified_content = original_content.replace(old_value, new_value)
            
            with open('prime_resonator.py', 'w') as f:
                f.write(modified_content)
            
            # Reload the module
            import importlib
            import prime_resonator
            importlib.reload(prime_resonator)
            
            # Test performance
            for bit_length, cases in self.test_cases.items():
                times = []
                phase1_hits = 0
                
                for case in cases:
                    start = time.perf_counter()
                    try:
                        p, q = prime_resonator.prime_resonate(case['n'])
                        elapsed = time.perf_counter() - start
                        
                        if p * q == case['n']:
                            times.append(elapsed)
                            # Assume Phase I succeeded if very fast
                            if elapsed < 0.5:
                                phase1_hits += 1
                    except:
                        pass
                
                if times:
                    result = TuningResult(
                        param_name=param_name,
                        param_value=new_value,
                        bit_length=bit_length,
                        avg_time=sum(times) / len(times),
                        success_rate=len(times) / len(cases),
                        phase1_hit_rate=phase1_hits / len(cases)
                    )
                    results.append(result)
                    
        # Restore original file
        with open('prime_resonator.py', 'w') as f:
            f.write(original_content)
            
        return results
    
    def find_optimal_parameters(self):
        """Find optimal parameter values through systematic testing"""
        
        print("\n=== Testing Parameter Variations ===\n")
        
        # Test 1: Prime dimension scaling
        print("1. Testing prime dimension scaling...")
        prime_variations = [
            ("return 32", "return 32"),  # baseline
            ("return 32", "return 48"),  # more primes for small
            ("return 32", "return 24"),  # fewer primes
            ("32 + bit_len // 4", "32 + bit_len // 3"),  # faster growth
            ("32 + bit_len // 4", "32 + bit_len // 6"),  # slower growth
        ]
        
        prime_results = self.test_parameter_variation(
            "prime_dimensions", 
            prime_variations
        )
        self.results.extend(prime_results)
        
        # Test 2: Candidate limits
        print("\n2. Testing Phase I candidate limits...")
        candidate_variations = [
            ("1000 if bit_len < 80 else 500", "1000 if bit_len < 80 else 500"),  # baseline
            ("1000 if bit_len < 80 else 500", "2000 if bit_len < 80 else 1000"),  # more candidates
            ("1000 if bit_len < 80 else 500", "500 if bit_len < 80 else 250"),   # fewer candidates
            ("500 if bit_len < 128 else 200", "800 if bit_len < 128 else 400"),  # more for medium
            ("500 if bit_len < 128 else 200", "300 if bit_len < 128 else 150"),  # fewer for medium
        ]
        
        candidate_results = self.test_parameter_variation(
            "phase1_candidates",
            candidate_variations
        )
        self.results.extend(candidate_results)
        
        # Test 3: Scoring thresholds
        print("\n3. Testing scoring thresholds...")
        threshold_variations = [
            ("score > 0.5", "score > 0.5"),   # baseline
            ("score > 0.5", "score > 0.3"),   # lower threshold
            ("score > 0.5", "score > 0.7"),   # higher threshold
            ("score > 0.5", "score > 0.4"),   # slightly lower
            ("score > 0.5", "score > 0.6"),   # slightly higher
        ]
        
        threshold_results = self.test_parameter_variation(
            "score_threshold",
            threshold_variations
        )
        self.results.extend(threshold_results)
        
        # Test 4: Harmonic weights
        print("\n4. Testing harmonic resonance weights...")
        harmonic_variations = [
            ("harmonic *= (1 + 1.0 / p)", "harmonic *= (1 + 1.0 / p)"),      # baseline
            ("harmonic *= (1 + 1.0 / p)", "harmonic *= (1 + 2.0 / p)"),      # stronger bonus
            ("harmonic *= (1 + 1.0 / p)", "harmonic *= (1 + 0.5 / p)"),      # weaker bonus
            ("harmonic *= (1 - 0.05 / p)", "harmonic *= (1 - 0.1 / p)"),     # stronger penalty
            ("harmonic *= (1 - 0.05 / p)", "harmonic *= (1 - 0.02 / p)"),    # weaker penalty
        ]
        
        harmonic_results = self.test_parameter_variation(
            "harmonic_weights",
            harmonic_variations
        )
        self.results.extend(harmonic_results)
        
        # Test 5: Golden spiral samples
        print("\n5. Testing golden spiral density...")
        golden_variations = [
            ("50 if bit_len < 96 else 20", "50 if bit_len < 96 else 20"),    # baseline
            ("50 if bit_len < 96 else 20", "100 if bit_len < 96 else 40"),   # more samples
            ("50 if bit_len < 96 else 20", "25 if bit_len < 96 else 10"),    # fewer samples
            ("50 if bit_len < 96 else 20", "75 if bit_len < 96 else 30"),    # moderate increase
        ]
        
        golden_results = self.test_parameter_variation(
            "golden_samples",
            golden_variations
        )
        self.results.extend(golden_results)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze tuning results to find optimal values"""
        
        analysis = {
            'best_by_param': {},
            'best_by_bitsize': {},
            'recommendations': []
        }
        
        # Group results by parameter
        param_groups = {}
        for result in self.results:
            if result.param_name not in param_groups:
                param_groups[result.param_name] = []
            param_groups[result.param_name].append(result)
        
        # Find best value for each parameter
        for param_name, results in param_groups.items():
            # Group by bit length
            by_bitlen = {}
            for r in results:
                if r.bit_length not in by_bitlen:
                    by_bitlen[r.bit_length] = []
                by_bitlen[r.bit_length].append(r)
            
            # Find best for each bit length
            best_values = {}
            for bit_len, bit_results in by_bitlen.items():
                # Sort by time (lower is better)
                sorted_results = sorted(bit_results, key=lambda x: x.avg_time)
                if sorted_results:
                    best = sorted_results[0]
                    best_values[bit_len] = {
                        'value': best.param_value,
                        'time': best.avg_time,
                        'phase1_rate': best.phase1_hit_rate
                    }
            
            analysis['best_by_param'][param_name] = best_values
        
        # Generate recommendations
        self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]):
        """Generate specific recommendations based on results"""
        
        recommendations = []
        
        # Analyze prime dimensions
        if 'prime_dimensions' in analysis['best_by_param']:
            prime_data = analysis['best_by_param']['prime_dimensions']
            if 96 in prime_data:
                best_96 = prime_data[96]['value']
                recommendations.append(
                    f"For 96-bit numbers, use prime dimension formula: {best_96}"
                )
        
        # Analyze candidate limits
        if 'phase1_candidates' in analysis['best_by_param']:
            cand_data = analysis['best_by_param']['phase1_candidates']
            
            # Check if more candidates help
            baseline_found = False
            more_is_better = False
            
            for bit_len, data in cand_data.items():
                if "1000" in data['value'] and "500" in data['value']:
                    baseline_found = True
                    baseline_time = data['time']
                    
                    # Check if higher values are better
                    for other_data in cand_data.values():
                        if "2000" in other_data['value'] or "1000" in other_data['value']:
                            if other_data['time'] < baseline_time * 0.9:
                                more_is_better = True
                                
            if more_is_better:
                recommendations.append(
                    "Increase Phase I candidate limits - more candidates improve performance"
                )
            else:
                recommendations.append(
                    "Reduce Phase I candidate limits - fewer, better candidates are more efficient"
                )
        
        # Analyze scoring threshold
        if 'score_threshold' in analysis['best_by_param']:
            threshold_data = analysis['best_by_param']['score_threshold']
            
            # Find trend
            lower_better = 0
            higher_better = 0
            
            for bit_len, data in threshold_data.items():
                if "0.3" in data['value'] or "0.4" in data['value']:
                    lower_better += 1
                elif "0.6" in data['value'] or "0.7" in data['value']:
                    higher_better += 1
                    
            if lower_better > higher_better:
                recommendations.append(
                    "Lower the score threshold to 0.3-0.4 for better candidate inclusion"
                )
            elif higher_better > lower_better:
                recommendations.append(
                    "Raise the score threshold to 0.6-0.7 for more selective candidates"
                )
        
        analysis['recommendations'] = recommendations
    
    def save_results(self, filename: str = "practical_tuning_results.json"):
        """Save results to file"""
        
        # Convert results to dict format
        results_dict = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'test_cases': {
                str(k): len(v) for k, v in self.test_cases.items()
            },
            'raw_results': [
                {
                    'param_name': r.param_name,
                    'param_value': r.param_value,
                    'bit_length': r.bit_length,
                    'avg_time': r.avg_time,
                    'success_rate': r.success_rate,
                    'phase1_hit_rate': r.phase1_hit_rate
                }
                for r in self.results
            ],
            'analysis': self.analyze_results()
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        print(f"\nResults saved to {filename}")
    
    def print_summary(self):
        """Print a summary of findings"""
        
        analysis = self.analyze_results()
        
        print("\n=== TUNING SUMMARY ===\n")
        
        print("Best Parameters by Bit Size:")
        print("-" * 60)
        
        for param_name, bit_data in analysis['best_by_param'].items():
            print(f"\n{param_name}:")
            for bit_len in sorted(bit_data.keys()):
                data = bit_data[bit_len]
                print(f"  {bit_len}-bit: {data['value']}")
                print(f"    Average time: {data['time']:.3f}s")
                print(f"    Phase I success: {data['phase1_rate']*100:.1f}%")
        
        print("\n\nRecommendations:")
        print("-" * 60)
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "=" * 60)


def main():
    """Run practical tuning"""
    
    print("=== Practical Prime Resonator Tuning ===\n")
    
    # Change to prime_resonator directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tuner = PracticalTuner()
    
    # Generate test cases
    print("Generating test cases...")
    tuner.generate_test_cases(bit_sizes=[64, 80, 96], cases_per_size=3)
    
    # Measure baseline
    print("\nMeasuring baseline performance...")
    baseline = tuner.measure_baseline()
    print("Baseline times:")
    for bits, time_val in sorted(baseline.items()):
        print(f"  {bits}-bit: {time_val:.3f}s average")
    
    # Test parameter variations
    tuner.find_optimal_parameters()
    
    # Print summary
    tuner.print_summary()
    
    # Save results
    tuner.save_results()
    
    print("\nTuning complete!")


if __name__ == "__main__":
    main()
