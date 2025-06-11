"""
Prime Sieve Performance Tuning Script

Tests various parameter configurations to find optimal settings
for the Prime Sieve implementation.
"""

import time
import json
from typing import Dict, List, Tuple, Any
import itertools
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prime_sieve import PrimeSieve
from prime_sieve.core.prime_sieve import PrimeSieveResult


class PrimeSieveTuner:
    """
    Tunes Prime Sieve parameters for optimal performance.
    """
    
    def __init__(self):
        """Initialize tuner with test cases."""
        # Test cases of varying difficulty
        self.test_cases = [
            # Small (fast baseline)
            (143, 11, 13),      # 8-bit
            (323, 17, 19),      # 9-bit
            
            # Medium (good for parameter sensitivity)
            (1147, 31, 37),     # 11-bit
            (3599, 59, 61),     # 12-bit
            (9409, 97, 97),     # 14-bit (perfect square)
            
            # Large (stress test)
            (294409, 37, 7957), # 19-bit
            (1299071, 1117, 1163), # 21-bit (close factors)
        ]
        
        # Parameters to tune
        self.tuning_params = {
            # Candidate generation limits
            'coord_candidates': [100, 200, 500, 1000],
            'coherence_threshold': [0.1, 0.2, 0.3, 0.5],
            'vortex_candidates': [50, 100, 200, 400],
            'interference_candidates': [50, 100, 200, 400],
            
            # Search range extension
            'sqrt_extension': [1.0, 1.05, 1.1, 1.15],
            'sqrt_delta_range': [50, 100, 150, 200],
            
            # Filtering thresholds
            'sieve_threshold': [0.05, 0.1, 0.15, 0.2],
            'min_candidates': [50, 100, 150, 200],
            
            # Scoring weights
            'distance_weight': [1.0, 1.5, 2.0, 2.5],
            'coherence_weight': [0.5, 1.0, 1.5, 2.0],
        }
        
        # Results storage
        self.results = []
        
    def test_configuration(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Test a specific configuration.
        
        Args:
            config: Parameter configuration to test
            
        Returns:
            Performance metrics
        """
        # Apply configuration by monkey-patching
        # (In production, these would be proper parameters)
        original_factor_with_details = PrimeSieve.factor_with_details
        
        def patched_factor_with_details(self, n):
            # Save original method references
            orig_generate = self._generate_initial_candidates
            orig_apply = self._apply_dimensional_sieves
            orig_check = self._check_candidates
            
            # Patch methods with configuration
            def patched_generate(coord_system, coherence_engine, vortex_engine, 
                               interference_analyzer, n):
                # Apply tuned parameters
                candidates = set()
                sqrt_n = int(n**0.5)
                
                # Use tuned limits
                search_limit = min(int(sqrt_n * config['sqrt_extension']), 100000)
                coord_aligned = coord_system.find_aligned_positions((2, search_limit))
                candidates.update(a.position for a in coord_aligned[:config['coord_candidates']])
                
                coherence_field = coherence_engine.generate_coherence_field()
                candidates.update(coherence_field.get_peaks(threshold=config['coherence_threshold']))
                
                vortex_centers = vortex_engine.generate_vortex_centers()
                candidates.update(v.position for v in vortex_centers[:config['vortex_candidates']])
                
                extrema = interference_analyzer.find_extrema()
                candidates.update(e.position for e in extrema[:config['interference_candidates']])
                
                # Near sqrt(n) with tuned range
                sqrt_region = int(sqrt_n)
                for delta in range(-config['sqrt_delta_range'], config['sqrt_delta_range'] + 1):
                    pos = sqrt_region + delta
                    if 2 <= pos <= int(sqrt_n * config['sqrt_extension']):
                        candidates.add(pos)
                
                # Remove invalid
                candidates.discard(0)
                candidates.discard(1)
                candidates = {c for c in candidates if c <= int(sqrt_n * config['sqrt_extension'])}
                
                return candidates
            
            def patched_apply(candidates, coord_system, coherence_engine,
                            vortex_engine, interference_analyzer, strategy):
                # Apply dimensional sieves with tuned threshold
                result = orig_apply(candidates, coord_system, coherence_engine,
                                  vortex_engine, interference_analyzer, strategy)
                
                # Apply tuned filtering
                if len(result) < config['min_candidates']:
                    # Add more candidates if too few
                    remaining = candidates - result
                    if remaining:
                        sorted_remaining = sorted(remaining, 
                                                key=lambda x: coherence_engine.calculate_coherence(x, x),
                                                reverse=True)
                        result.update(sorted_remaining[:config['min_candidates'] - len(result)])
                
                return result
            
            def patched_check(candidates, n, coherence_engine, start_time):
                # Use tuned scoring weights
                sqrt_n = int(n**0.5)
                
                def candidate_score(x):
                    distance = abs(x - sqrt_n) / sqrt_n
                    distance_score = 1.0 / (1.0 + distance)
                    
                    if n % x == 0:
                        coherence_score = coherence_engine.calculate_coherence(x, n // x)
                    else:
                        coherence_score = coherence_engine.calculate_coherence(x, x) * 0.5
                    
                    return distance_score * config['distance_weight'] + coherence_score * config['coherence_weight']
                
                ordered = sorted(candidates, key=candidate_score, reverse=True)
                
                # Check candidates
                iterations = 0
                peak_coherence = 0.0
                
                for candidate in ordered:
                    iterations += 1
                    if candidate <= 1:
                        continue
                    
                    if n % candidate == 0:
                        other = n // candidate
                        if other >= 1 and candidate * other == n:
                            coherence = coherence_engine.calculate_coherence(candidate, other)
                            peak_coherence = max(peak_coherence, coherence)
                            
                            return PrimeSieveResult(
                                n=n,
                                factors=(min(candidate, other), max(candidate, other)),
                                time_taken=time.time() - start_time,
                                method='tuned',
                                iterations=iterations,
                                candidates_tested=iterations,
                                peak_coherence=peak_coherence,
                                dimensions_used=['all'],
                                confidence=coherence
                            )
                
                return PrimeSieveResult(
                    n=n, factors=(1, n), time_taken=time.time() - start_time,
                    method='none', iterations=iterations, candidates_tested=iterations,
                    peak_coherence=peak_coherence, dimensions_used=['all'], confidence=0.0
                )
            
            # Apply patches
            self._generate_initial_candidates = patched_generate
            self._apply_dimensional_sieves = patched_apply
            self._check_candidates = patched_check
            
            try:
                result = original_factor_with_details(self, n)
            finally:
                # Restore original methods
                self._generate_initial_candidates = orig_generate
                self._apply_dimensional_sieves = orig_apply
                self._check_candidates = orig_check
            
            return result
        
        # Test with patched version
        PrimeSieve.factor_with_details = patched_factor_with_details
        
        try:
            # Run tests
            sieve = PrimeSieve(enable_learning=False)
            
            total_time = 0
            successful = 0
            total_iterations = 0
            
            for n, expected_p, expected_q in self.test_cases:
                start = time.time()
                result = sieve.factor_with_details(n)
                elapsed = time.time() - start
                
                total_time += elapsed
                total_iterations += result.iterations
                
                if result.factors == (expected_p, expected_q) or \
                   result.factors == (expected_q, expected_p):
                    successful += 1
            
            metrics = {
                'total_time': total_time,
                'avg_time': total_time / len(self.test_cases),
                'success_rate': successful / len(self.test_cases),
                'avg_iterations': total_iterations / len(self.test_cases),
                'score': (successful / len(self.test_cases)) / (1 + total_time)  # Balance success and speed
            }
            
            return metrics
            
        finally:
            # Restore original
            PrimeSieve.factor_with_details = original_factor_with_details
    
    def grid_search(self, param_subset: List[str] = None):
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_subset: Subset of parameters to tune (None for all)
        """
        if param_subset is None:
            param_subset = list(self.tuning_params.keys())
        
        # Create parameter grid
        param_values = [self.tuning_params[p] for p in param_subset]
        param_combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(param_combinations)} configurations...")
        print(f"Parameters: {param_subset}")
        print("-" * 60)
        
        best_config = None
        best_score = -1
        
        for i, values in enumerate(param_combinations):
            # Create configuration
            config = dict(zip(param_subset, values))
            
            # Set defaults for non-tuned parameters
            for param in self.tuning_params:
                if param not in config:
                    config[param] = self.tuning_params[param][1]  # Use second value as default
            
            # Test configuration
            metrics = self.test_configuration(config)
            
            # Store result
            result = {
                'config': config,
                'metrics': metrics
            }
            self.results.append(result)
            
            # Update best - prioritize success rate, then score
            if (metrics['success_rate'] > 0.8 and metrics['score'] > best_score) or \
               (best_config is None and metrics['success_rate'] > 0):
                best_score = metrics['score']
                best_config = config
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(param_combinations)} configurations tested")
                if best_config:
                    print(f"  Best score so far: {best_score:.4f}")
        
        return best_config, best_score
    
    def adaptive_tuning(self):
        """
        Perform adaptive tuning by focusing on most impactful parameters.
        """
        print("\n" + "="*60)
        print("ADAPTIVE PARAMETER TUNING")
        print("="*60)
        
        # Phase 1: Test impact of each parameter individually
        print("\nPhase 1: Individual parameter impact")
        print("-" * 40)
        
        parameter_impacts = {}
        
        for param in self.tuning_params:
            print(f"\nTesting parameter: {param}")
            
            scores = []
            for value in self.tuning_params[param]:
                config = {p: self.tuning_params[p][1] for p in self.tuning_params}  # Defaults
                config[param] = value
                
                metrics = self.test_configuration(config)
                scores.append(metrics['score'])
            
            # Calculate impact (variance in scores)
            impact = max(scores) - min(scores)
            parameter_impacts[param] = impact
            print(f"  Impact score: {impact:.4f}")
        
        # Phase 2: Tune high-impact parameters
        print("\nPhase 2: Tuning high-impact parameters")
        print("-" * 40)
        
        # Sort by impact
        sorted_params = sorted(parameter_impacts.items(), key=lambda x: x[1], reverse=True)
        high_impact_params = [p[0] for p in sorted_params[:5]]  # Top 5 parameters
        
        print(f"High-impact parameters: {high_impact_params}")
        
        # Grid search on high-impact parameters
        best_config, best_score = self.grid_search(high_impact_params)
        
        return best_config, best_score
    
    def benchmark_configurations(self):
        """
        Benchmark specific configurations.
        """
        print("\n" + "="*60)
        print("CONFIGURATION BENCHMARKS")
        print("="*60)
        
        configs = {
            'default': {
                'coord_candidates': 500,
                'coherence_threshold': 0.3,
                'vortex_candidates': 200,
                'interference_candidates': 200,
                'sqrt_extension': 1.1,
                'sqrt_delta_range': 100,
                'sieve_threshold': 0.1,
                'min_candidates': 100,
                'distance_weight': 2.0,
                'coherence_weight': 1.0,
            },
            'aggressive': {
                'coord_candidates': 1000,
                'coherence_threshold': 0.1,
                'vortex_candidates': 400,
                'interference_candidates': 400,
                'sqrt_extension': 1.15,
                'sqrt_delta_range': 200,
                'sieve_threshold': 0.05,
                'min_candidates': 200,
                'distance_weight': 2.5,
                'coherence_weight': 1.5,
            },
            'balanced': {
                'coord_candidates': 500,
                'coherence_threshold': 0.2,
                'vortex_candidates': 200,
                'interference_candidates': 200,
                'sqrt_extension': 1.1,
                'sqrt_delta_range': 150,
                'sieve_threshold': 0.1,
                'min_candidates': 100,
                'distance_weight': 1.5,
                'coherence_weight': 1.5,
            },
            'fast': {
                'coord_candidates': 200,
                'coherence_threshold': 0.3,
                'vortex_candidates': 100,
                'interference_candidates': 100,
                'sqrt_extension': 1.05,
                'sqrt_delta_range': 50,
                'sieve_threshold': 0.15,
                'min_candidates': 50,
                'distance_weight': 2.0,
                'coherence_weight': 0.5,
            }
        }
        
        results = {}
        
        for name, config in configs.items():
            print(f"\nTesting {name} configuration...")
            metrics = self.test_configuration(config)
            results[name] = metrics
            
            print(f"  Success rate: {metrics['success_rate']:.0%}")
            print(f"  Average time: {metrics['avg_time']:.4f}s")
            print(f"  Average iterations: {metrics['avg_iterations']:.1f}")
            print(f"  Overall score: {metrics['score']:.4f}")
        
        # Find best
        best_name = max(results.keys(), key=lambda k: results[k]['score'])
        print(f"\nBest configuration: {best_name}")
        
        return configs[best_name], results
    
    def save_results(self, filename: str = "tuning_results.json"):
        """Save tuning results to file."""
        with open(filename, 'w') as f:
            json.dump({
                'results': self.results,
                'test_cases': self.test_cases,
                'parameters': self.tuning_params
            }, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def generate_optimized_config(self) -> Dict[str, Any]:
        """
        Generate optimized configuration file.
        """
        # Run adaptive tuning
        best_config, best_score = self.adaptive_tuning()
        
        print("\n" + "="*60)
        print("OPTIMAL CONFIGURATION FOUND")
        print("="*60)
        print(f"Score: {best_score:.4f}")
        print("\nParameters:")
        for param, value in best_config.items():
            print(f"  {param}: {value}")
        
        # Save configuration
        config_file = "prime_sieve_optimal_config.json"
        with open(config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\nOptimal configuration saved to {config_file}")
        
        return best_config


def main():
    """Run Prime Sieve tuning."""
    tuner = PrimeSieveTuner()
    
    # Benchmark standard configurations
    print("Starting Prime Sieve performance tuning...")
    best_config, benchmark_results = tuner.benchmark_configurations()
    
    # Run adaptive tuning for further optimization
    optimal_config = tuner.generate_optimized_config()
    
    # Save all results
    tuner.save_results()
    
    print("\n" + "="*60)
    print("TUNING COMPLETE!")
    print("="*60)
    print("\nTo apply optimal configuration, update the Prime Sieve")
    print("implementation with the values in prime_sieve_optimal_config.json")


if __name__ == "__main__":
    main()
