#!/usr/bin/env python3
"""
Prime Resonator Tuning Script
=============================

This script tunes the Prime Resonator parameters to achieve optimal performance,
particularly for 96-bit semiprimes and beyond. It uses the concept of resonance
to find parameter combinations that harmonize with semiprime mathematical structure.
"""

import time
import math
import json
import itertools
import statistics
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from functools import lru_cache
import secrets

# Import the prime resonator for testing
import prime_resonator as pr


@dataclass
class TestCase:
    """A semiprime test case with metadata"""
    n: int
    p: int
    q: int
    bit_length: int
    balance_ratio: float  # p/q ratio
    near_square: bool
    near_fibonacci: bool
    
    @property
    def category(self) -> str:
        """Categorize the semiprime type"""
        if self.near_square:
            return "near_square"
        elif self.near_fibonacci:
            return "near_fibonacci"
        elif self.balance_ratio > 0.8:
            return "balanced"
        elif self.balance_ratio > 0.5:
            return "semi_balanced"
        else:
            return "unbalanced"


@dataclass
class ParameterConfig:
    """Configuration parameters for the Prime Resonator"""
    # Prime dimensions
    prime_count_base: int = 32
    prime_count_scale: float = 0.25  # Additional primes per bit
    
    # Candidate generation
    crt_pairs_small: int = 5
    crt_pairs_large: int = 3
    search_width_base: int = 3
    search_width_scale: float = 1.0
    focus_primes_small: int = 10
    focus_primes_large: int = 5
    golden_samples_small: int = 50
    golden_samples_large: int = 20
    
    # Resonance scoring
    harmonic_match_bonus: float = 1.0  # Multiplier for 1/p
    harmonic_mismatch_penalty: float = 0.05  # Multiplier for penalty
    quick_score_threshold: float = 0.5
    tier2_prime_check: int = 10
    full_score_primes: int = 20
    
    # Phase I limits
    phase1_candidates_small: int = 1000
    phase1_candidates_medium: int = 500
    phase1_candidates_large: int = 200
    top_scoring_ratio: float = 0.1  # Fraction to fully score
    check_limit_ratio: float = 0.1  # Fraction to check for factors
    
    # Phase II parameters
    base_steps_scale: float = 1000.0  # Multiplied by sqrt(bit_len)
    steps_growth_factor: float = 2.0  # Growth per bit range
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ParameterConfig':
        return cls(**d)


@dataclass
class PerformanceMetrics:
    """Metrics for evaluating a configuration"""
    time_taken: float
    phase1_success: bool
    candidates_generated: int
    candidates_scored: int
    candidates_checked: int
    true_factor_score: float
    true_factor_rank: int


class ResonanceTuner:
    """Tunes Prime Resonator parameters for optimal performance"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_suite: List[TestCase] = []
        self.results: Dict[str, Any] = {}
        self.best_configs: Dict[int, ParameterConfig] = {}
        
    def log(self, msg: str):
        """Print if verbose mode is on"""
        if self.verbose:
            print(msg)
            
    def generate_test_suite(self, bit_sizes: List[int] = None) -> List[TestCase]:
        """Generate diverse test cases for tuning"""
        if bit_sizes is None:
            bit_sizes = [64, 80, 96, 112, 128]
            
        test_cases = []
        
        for bits in bit_sizes:
            # Balanced semiprime
            p1, q1 = self._generate_balanced_semiprime(bits)
            test_cases.append(self._create_test_case(p1, q1))
            
            # Unbalanced semiprime
            p2, q2 = self._generate_unbalanced_semiprime(bits)
            test_cases.append(self._create_test_case(p2, q2))
            
            # Near-square semiprime
            p3, q3 = self._generate_near_square_semiprime(bits)
            test_cases.append(self._create_test_case(p3, q3))
            
            # Near-Fibonacci semiprime (if possible)
            try:
                p4, q4 = self._generate_near_fibonacci_semiprime(bits)
                test_cases.append(self._create_test_case(p4, q4))
            except:
                pass  # Not all bit sizes have good Fibonacci primes
                
        self.test_suite = test_cases
        self.log(f"Generated {len(test_cases)} test cases")
        return test_cases
    
    def _generate_balanced_semiprime(self, bits: int) -> Tuple[int, int]:
        """Generate a semiprime with p ≈ q"""
        half_bits = bits // 2
        p = self._generate_prime(half_bits)
        q = self._generate_prime(bits - half_bits)
        return min(p, q), max(p, q)
    
    def _generate_unbalanced_semiprime(self, bits: int) -> Tuple[int, int]:
        """Generate a semiprime with p << q"""
        p_bits = bits // 3
        q_bits = bits - p_bits - 1  # Ensure product has correct bit length
        p = self._generate_prime(p_bits)
        q = self._generate_prime(q_bits)
        
        # Adjust q to ensure correct bit length
        while (p * q).bit_length() != bits:
            if (p * q).bit_length() < bits:
                q = self._next_prime(q)
            else:
                q = self._prev_prime(q)
                
        return p, q
    
    def _generate_near_square_semiprime(self, bits: int) -> Tuple[int, int]:
        """Generate n = p*q where p and q are close"""
        target = 1 << (bits - 1)  # 2^(bits-1)
        sqrt_target = int(math.isqrt(target))
        
        # Find two primes near sqrt(target)
        p = self._next_prime(sqrt_target - 1000)
        q = self._next_prime(sqrt_target + 1000)
        
        # Adjust to get correct bit length
        while (p * q).bit_length() != bits:
            if (p * q).bit_length() < bits:
                q = self._next_prime(q)
            else:
                p = self._prev_prime(p)
                
        return p, q
    
    def _generate_near_fibonacci_semiprime(self, bits: int) -> Tuple[int, int]:
        """Generate semiprime with factors near Fibonacci numbers"""
        # Generate Fibonacci numbers in the right range
        fibs = [1, 1]
        while fibs[-1].bit_length() < bits // 2:
            fibs.append(fibs[-1] + fibs[-2])
            
        # Find suitable Fibonacci numbers for the bit range
        suitable_fibs = [f for f in fibs if bits//3 <= f.bit_length() <= bits//2]
        
        if len(suitable_fibs) < 2:
            raise ValueError("No suitable Fibonacci numbers for this bit range")
            
        # Find primes near Fibonacci numbers
        f1 = suitable_fibs[-2]
        f2 = suitable_fibs[-1]
        
        p = self._next_prime(f1)
        q = self._next_prime(f2)
        
        # Adjust for correct bit length
        while (p * q).bit_length() != bits:
            if (p * q).bit_length() < bits:
                q = self._next_prime(q)
            else:
                break
                
        return p, q
    
    def _generate_prime(self, bits: int) -> int:
        """Generate a random prime of specified bit length"""
        return pr._rand_prime(bits)
    
    def _next_prime(self, n: int) -> int:
        """Find the next prime after n"""
        if n % 2 == 0:
            n += 1
        else:
            n += 2
        while not pr._is_probable_prime(n):
            n += 2
        return n
    
    def _prev_prime(self, n: int) -> int:
        """Find the previous prime before n"""
        if n % 2 == 0:
            n -= 1
        else:
            n -= 2
        while n > 2 and not pr._is_probable_prime(n):
            n -= 2
        return n
    
    def _create_test_case(self, p: int, q: int) -> TestCase:
        """Create a test case from prime factors"""
        n = p * q
        balance_ratio = min(p, q) / max(p, q)
        
        # Check if near square
        sqrt_n = int(math.isqrt(n))
        near_square = abs(p - q) < sqrt_n * 0.01
        
        # Check if near Fibonacci
        near_fibonacci = self._is_near_fibonacci(p) or self._is_near_fibonacci(q)
        
        return TestCase(
            n=n,
            p=min(p, q),
            q=max(p, q),
            bit_length=n.bit_length(),
            balance_ratio=balance_ratio,
            near_square=near_square,
            near_fibonacci=near_fibonacci
        )
    
    def _is_near_fibonacci(self, x: int) -> bool:
        """Check if x is near a Fibonacci number"""
        a, b = 1, 1
        while b < x * 0.8:
            a, b = b, a + b
        
        # Check if within 5% of a Fibonacci number
        while b < x * 1.2:
            if abs(b - x) < x * 0.05:
                return True
            a, b = b, a + b
            
        return False
    
    def measure_performance(self, config: ParameterConfig, test_case: TestCase) -> PerformanceMetrics:
        """Measure performance of a configuration on a test case"""
        # Monkey-patch the configuration into the prime_resonator module
        original_functions = self._apply_config(config, test_case.bit_length)
        
        try:
            start_time = time.perf_counter()
            
            # Track Phase I performance
            phase1_success = False
            candidates_generated = 0
            candidates_scored = 0
            candidates_checked = 0
            true_factor_score = 0.0
            true_factor_rank = -1
            
            # Run with instrumentation
            p_found, q_found = pr.prime_resonate(test_case.n)
            
            elapsed = time.perf_counter() - start_time
            
            # Verify correctness
            assert p_found * q_found == test_case.n
            
            # Get performance data from instrumentation (would need to modify prime_resonator.py)
            # For now, we'll estimate based on the time taken
            phase1_success = elapsed < 1.0  # Assume Phase I succeeded if fast
            
            metrics = PerformanceMetrics(
                time_taken=elapsed,
                phase1_success=phase1_success,
                candidates_generated=candidates_generated,
                candidates_scored=candidates_scored,
                candidates_checked=candidates_checked,
                true_factor_score=true_factor_score,
                true_factor_rank=true_factor_rank
            )
            
        finally:
            # Restore original functions
            self._restore_functions(original_functions)
            
        return metrics
    
    def _apply_config(self, config: ParameterConfig, bit_length: int) -> Dict:
        """Apply configuration parameters to prime_resonator module"""
        # Store original functions
        originals = {
            '_adaptive_prime_count': pr._adaptive_prime_count,
            '_coordinate_convergence': pr._coordinate_convergence,
            '_golden_positions': pr._golden_positions,
            '_tiered_resonance_score': pr._tiered_resonance_score,
            '_multiplicative_resonance_score': pr._multiplicative_resonance_score,
        }
        
        # Create modified functions with config parameters
        def adaptive_prime_count(n: int) -> int:
            bit_len = n.bit_length()
            return int(config.prime_count_base + bit_len * config.prime_count_scale)
        
        # Apply modifications
        pr._adaptive_prime_count = adaptive_prime_count
        
        # TODO: Modify other functions based on config
        # This would require more sophisticated monkey-patching or
        # modifying prime_resonator.py to accept configuration
        
        return originals
    
    def _restore_functions(self, originals: Dict):
        """Restore original functions"""
        for name, func in originals.items():
            setattr(pr, name, func)
    
    def grid_search(self, param_ranges: Dict[str, List[Any]]) -> Dict[int, ParameterConfig]:
        """Perform grid search over parameter space"""
        self.log("Starting grid search...")
        
        # Group test cases by bit length
        bit_groups = {}
        for tc in self.test_suite:
            if tc.bit_length not in bit_groups:
                bit_groups[tc.bit_length] = []
            bit_groups[tc.bit_length].append(tc)
        
        best_configs = {}
        
        for bit_length, test_cases in bit_groups.items():
            self.log(f"\nTuning for {bit_length}-bit numbers...")
            
            best_time = float('inf')
            best_config = ParameterConfig()
            
            # Generate parameter combinations
            param_names = list(param_ranges.keys())
            param_values = [param_ranges[name] for name in param_names]
            
            for values in itertools.product(*param_values):
                # Create configuration
                config_dict = dict(zip(param_names, values))
                config = ParameterConfig(**config_dict)
                
                # Test on all cases for this bit length
                total_time = 0
                success_count = 0
                
                for tc in test_cases:
                    try:
                        metrics = self.measure_performance(config, tc)
                        total_time += metrics.time_taken
                        if metrics.phase1_success:
                            success_count += 1
                    except Exception as e:
                        self.log(f"Error testing config: {e}")
                        total_time = float('inf')
                        break
                
                avg_time = total_time / len(test_cases)
                
                if avg_time < best_time:
                    best_time = avg_time
                    best_config = config
                    self.log(f"  New best: {avg_time:.3f}s average")
            
            best_configs[bit_length] = best_config
            self.log(f"Best config for {bit_length}-bit: {best_time:.3f}s average")
        
        self.best_configs = best_configs
        return best_configs
    
    def analyze_resonance_patterns(self) -> Dict[str, Any]:
        """Analyze which parameters have the most impact"""
        analysis = {
            'parameter_sensitivity': {},
            'optimal_scaling': {},
            'resonance_patterns': {}
        }
        
        # Analyze how parameters scale with bit length
        if len(self.best_configs) >= 2:
            bit_lengths = sorted(self.best_configs.keys())
            
            # Track parameter progression
            param_progressions = {}
            
            for param_name in asdict(ParameterConfig()).keys():
                values = []
                for bl in bit_lengths:
                    config = self.best_configs[bl]
                    values.append(getattr(config, param_name))
                
                param_progressions[param_name] = {
                    'bit_lengths': bit_lengths,
                    'values': values,
                    'trend': self._analyze_trend(bit_lengths, values)
                }
            
            analysis['optimal_scaling'] = param_progressions
        
        # Identify resonance patterns
        patterns = []
        
        # Pattern 1: Prime dimension resonance
        if 'prime_count_scale' in analysis['optimal_scaling']:
            scale_trend = analysis['optimal_scaling']['prime_count_scale']['trend']
            if scale_trend == 'increasing':
                patterns.append("Prime dimensions should grow with bit length")
            elif scale_trend == 'decreasing':
                patterns.append("Fewer prime dimensions work better for large numbers")
        
        # Pattern 2: Candidate generation patterns
        if 'phase1_candidates_large' in analysis['optimal_scaling']:
            cand_trend = analysis['optimal_scaling']['phase1_candidates_large']['trend']
            if cand_trend == 'decreasing':
                patterns.append("Fewer, better candidates beat many mediocre ones")
        
        analysis['resonance_patterns'] = patterns
        
        return analysis
    
    def _analyze_trend(self, x: List[int], y: List[Any]) -> str:
        """Analyze trend in data"""
        if len(x) < 2:
            return "insufficient_data"
        
        # Simple trend analysis
        try:
            y_numeric = [float(val) for val in y]
            
            # Calculate correlation
            if len(set(y_numeric)) == 1:
                return "constant"
            
            # Simple slope check
            first_half_avg = sum(y_numeric[:len(y)//2]) / (len(y)//2)
            second_half_avg = sum(y_numeric[len(y)//2:]) / (len(y) - len(y)//2)
            
            if second_half_avg > first_half_avg * 1.1:
                return "increasing"
            elif second_half_avg < first_half_avg * 0.9:
                return "decreasing"
            else:
                return "stable"
                
        except:
            return "non_numeric"
    
    def save_results(self, filename: str = "resonance_tuning_results.json"):
        """Save tuning results to file"""
        results = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'test_cases': [asdict(tc) for tc in self.test_suite],
            'best_configs': {
                str(bl): config.to_dict() 
                for bl, config in self.best_configs.items()
            },
            'analysis': self.analyze_resonance_patterns()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.log(f"Results saved to {filename}")
    
    def generate_tuned_code(self, output_file: str = "prime_resonator_tuned.py"):
        """Generate optimized version with tuned parameters"""
        # This would generate a modified version of prime_resonator.py
        # with the optimal parameters hard-coded
        self.log(f"Tuned code generation not yet implemented")


def main():
    """Run the tuning process"""
    print("=== Prime Resonator Tuning ===")
    print("Finding optimal resonance frequencies...\n")
    
    tuner = ResonanceTuner(verbose=True)
    
    # Generate test suite
    print("1. Generating test suite...")
    tuner.generate_test_suite(bit_sizes=[64, 80, 96])
    
    # Define parameter search space
    print("\n2. Defining parameter space...")
    param_ranges = {
        'prime_count_scale': [0.1, 0.25, 0.5],
        'crt_pairs_large': [2, 3, 5],
        'harmonic_match_bonus': [0.5, 1.0, 2.0],
        'quick_score_threshold': [0.3, 0.5, 0.7],
        'phase1_candidates_large': [100, 200, 500],
        'base_steps_scale': [500, 1000, 2000]
    }
    
    # Perform grid search
    print("\n3. Performing grid search...")
    best_configs = tuner.grid_search(param_ranges)
    
    # Analyze results
    print("\n4. Analyzing resonance patterns...")
    analysis = tuner.analyze_resonance_patterns()
    
    print("\n=== Resonance Patterns Discovered ===")
    for pattern in analysis.get('resonance_patterns', []):
        print(f"  • {pattern}")
    
    # Save results
    print("\n5. Saving results...")
    tuner.save_results()
    
    print("\n=== Tuning Complete ===")
    print("Optimal configurations found for each bit range.")
    print("Results saved to resonance_tuning_results.json")


if __name__ == "__main__":
    main()
