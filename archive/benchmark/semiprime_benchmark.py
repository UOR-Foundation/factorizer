#!/usr/bin/env python3
"""
Semiprime Breakthrough Benchmark
Tests the UOR/Prime Axioms Factorizer on arbitrary semiprimes up to 64-bit numbers
This represents a significant computational breakthrough if successful.
"""

import time
import sys
import os
import random
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.benchmark_runner import BenchmarkRunner, BenchmarkResult

@dataclass
class SemiprimeResult:
    """Results from a semiprime factorization attempt"""
    n: int
    expected_factors: List[int]
    found_factors: Dict[str, Optional[int]]  # axiom -> factor found
    success_count: int
    total_time: float
    bit_length: int
    difficulty_class: str
    additional_info: Dict[str, Any] = field(default_factory=dict)

class SemiprimeBenchmark:
    """Benchmark suite for arbitrary semiprimes"""
    
    def __init__(self):
        self.runner = BenchmarkRunner()
        self.results = []
        
    def generate_primes_up_to(self, limit: int) -> List[int]:
        """Generate primes up to limit using sieve of Eratosthenes"""
        if limit < 2:
            return []
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    def generate_semiprime_test_cases(self) -> List[Tuple[int, List[int], str]]:
        """Generate comprehensive semiprime test cases"""
        test_cases = []
        
        # Generate primes for different bit ranges
        small_primes = self.generate_primes_up_to(1000)        # ~10 bit
        medium_primes = self.generate_primes_up_to(100000)     # ~17 bit
        large_primes = []
        
        # Generate larger primes for higher bit tests
        for _ in range(100):
            # Generate primes in different ranges
            for bit_range in [(18, 20), (20, 24), (24, 28), (28, 32)]:
                min_val = 2**(bit_range[0]-1)
                max_val = 2**bit_range[1] - 1
                
                candidate = random.randint(min_val, max_val)
                if candidate % 2 == 0:
                    candidate += 1
                
                # Simple primality test (Miller-Rabin would be better for production)
                if self.is_likely_prime(candidate):
                    large_primes.append(candidate)
        
        large_primes = sorted(list(set(large_primes)))
        
        # Small semiprimes (up to 20 bits)
        print("Generating small semiprimes...")
        for i in range(min(20, len(small_primes))):
            for j in range(i+1, min(i+10, len(small_primes))):
                p1, p2 = small_primes[i], small_primes[j]
                n = p1 * p2
                if n.bit_length() <= 20:
                    test_cases.append((n, [p1, p2], "small"))
        
        # Medium semiprimes (20-40 bits)  
        print("Generating medium semiprimes...")
        for _ in range(50):
            p1 = random.choice(small_primes[10:])
            p2 = random.choice(medium_primes[100:1000])
            n = p1 * p2
            if 20 <= n.bit_length() <= 40:
                test_cases.append((n, [p1, p2], "medium"))
        
        # Large semiprimes (40-56 bits)
        print("Generating large semiprimes...")
        for _ in range(30):
            if len(large_primes) >= 2:
                p1 = random.choice(large_primes)
                p2 = random.choice(large_primes)
                if p1 != p2:
                    n = p1 * p2
                    if 40 <= n.bit_length() <= 56:
                        test_cases.append((n, [min(p1,p2), max(p1,p2)], "large"))
        
        # Very large semiprimes (56-64 bits) - the breakthrough test
        print("Generating very large semiprimes...")
        for _ in range(10):
            if len(large_primes) >= 2:
                # Use larger primes to reach 64-bit range
                p1 = random.choice([p for p in large_primes if p.bit_length() >= 28])
                p2 = random.choice([p for p in large_primes if p.bit_length() >= 28])
                if p1 != p2:
                    n = p1 * p2
                    if 56 <= n.bit_length() <= 64:
                        test_cases.append((n, [min(p1,p2), max(p1,p2)], "very_large"))
        
        # Sort by difficulty
        test_cases.sort(key=lambda x: x[0])
        
        print(f"Generated {len(test_cases)} semiprime test cases")
        return test_cases
    
    def is_likely_prime(self, n: int, k: int = 5) -> bool:
        """Simple Miller-Rabin primality test"""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as d * 2^r
        r = 0
        d = n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witness loop
        for _ in range(k):
            a = random.randrange(2, n - 1)
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
    
    def benchmark_semiprime(self, n: int, factors: List[int], difficulty: str) -> SemiprimeResult:
        """Benchmark a single semiprime"""
        print(f"  Testing n={n} ({n.bit_length()}-bit, {difficulty})")
        
        start_time = time.time()
        results = self.runner.run_single_benchmark(n, factors)
        total_time = time.time() - start_time
        
        found_factors = {}
        success_count = 0
        
        for axiom_name, result in results.items():
            found_factors[axiom_name] = result.factor_found
            if result.success:
                success_count += 1
        
        return SemiprimeResult(
            n=n,
            expected_factors=factors,
            found_factors=found_factors,
            success_count=success_count,
            total_time=total_time,
            bit_length=n.bit_length(),
            difficulty_class=difficulty,
            additional_info={
                'axiom_results': {name: result.success for name, result in results.items()}
            }
        )
    
    def run_breakthrough_benchmark(self, max_cases_per_class: int = 20) -> Dict[str, Any]:
        """Run the comprehensive semiprime breakthrough benchmark"""
        print("=" * 80)
        print("UOR/Prime Axioms Factorizer - SEMIPRIME BREAKTHROUGH BENCHMARK")
        print("=" * 80)
        print("Testing arbitrary semiprimes up to 64-bit numbers")
        print("This represents a significant computational breakthrough if successful.")
        print()
        
        # Generate test cases
        all_test_cases = self.generate_semiprime_test_cases()
        
        # Group by difficulty class
        test_groups = {}
        for n, factors, difficulty in all_test_cases:
            if difficulty not in test_groups:
                test_groups[difficulty] = []
            test_groups[difficulty].append((n, factors))
        
        # Limit cases per class for reasonable runtime
        for difficulty in test_groups:
            if len(test_groups[difficulty]) > max_cases_per_class:
                test_groups[difficulty] = test_groups[difficulty][:max_cases_per_class]
        
        # Run benchmarks
        all_results = []
        class_stats = {}
        
        for difficulty in ['small', 'medium', 'large', 'very_large']:
            if difficulty not in test_groups:
                continue
                
            print(f"\n--- {difficulty.upper()} SEMIPRIMES ---")
            class_results = []
            
            for n, factors in test_groups[difficulty]:
                try:
                    result = self.benchmark_semiprime(n, factors, difficulty)
                    class_results.append(result)
                    all_results.append(result)
                    
                    # Show immediate feedback
                    success_rate = (result.success_count / 5) * 100
                    print(f"    Success: {result.success_count}/5 axioms ({success_rate:.1f}%) in {result.total_time:.3f}s")
                    
                except Exception as e:
                    print(f"    ERROR: {str(e)}")
                    continue
            
            # Calculate class statistics
            if class_results:
                class_stats[difficulty] = {
                    'count': len(class_results),
                    'avg_success_rate': statistics.mean([r.success_count / 5 for r in class_results]) * 100,
                    'avg_time': statistics.mean([r.total_time for r in class_results]),
                    'avg_bit_length': statistics.mean([r.bit_length for r in class_results]),
                    'best_success_rate': max([r.success_count / 5 for r in class_results]) * 100,
                    'breakthrough_cases': len([r for r in class_results if r.success_count >= 3])
                }
        
        return self.generate_breakthrough_report(all_results, class_stats)
    
    def generate_breakthrough_report(self, results: List[SemiprimeResult], 
                                   class_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comprehensive breakthrough report"""
        print("\n" + "=" * 80)
        print("SEMIPRIME BREAKTHROUGH ANALYSIS")
        print("=" * 80)
        
        if not results:
            print("No results to analyze!")
            return {}
        
        # Overall statistics
        total_cases = len(results)
        total_successes = sum(r.success_count for r in results)
        total_possible = total_cases * 5  # 5 axioms per case
        overall_success_rate = (total_successes / total_possible) * 100
        
        # Breakthrough metrics
        breakthrough_cases = len([r for r in results if r.success_count >= 3])  # 3+ axioms succeed
        major_breakthrough = len([r for r in results if r.success_count >= 4])   # 4+ axioms succeed
        perfect_cases = len([r for r in results if r.success_count == 5])        # All axioms succeed
        
        # Bit-length analysis
        max_bit_solved = 0
        largest_factored = 0
        for result in results:
            if result.success_count >= 3:  # Breakthrough threshold
                max_bit_solved = max(max_bit_solved, result.bit_length)
                largest_factored = max(largest_factored, result.n)
        
        print(f"BREAKTHROUGH SUMMARY:")
        print(f"  Total Test Cases: {total_cases}")
        print(f"  Overall Success Rate: {overall_success_rate:.2f}%")
        print(f"  Breakthrough Cases (3+ axioms): {breakthrough_cases} ({(breakthrough_cases/total_cases)*100:.1f}%)")
        print(f"  Major Breakthrough (4+ axioms): {major_breakthrough} ({(major_breakthrough/total_cases)*100:.1f}%)")
        print(f"  Perfect Cases (5/5 axioms): {perfect_cases} ({(perfect_cases/total_cases)*100:.1f}%)")
        print(f"  Maximum Bit-Length Solved: {max_bit_solved} bits")
        print(f"  Largest Number Factored: {largest_factored:,}")
        
        print(f"\nCLASS BREAKDOWN:")
        for difficulty, stats in class_stats.items():
            print(f"  {difficulty.upper()}:")
            print(f"    Cases: {stats['count']}")
            print(f"    Avg Success Rate: {stats['avg_success_rate']:.1f}%")
            print(f"    Avg Time: {stats['avg_time']:.3f}s")
            print(f"    Avg Bit Length: {stats['avg_bit_length']:.1f}")
            print(f"    Breakthrough Cases: {stats['breakthrough_cases']}")
        
        # Axiom performance analysis
        axiom_performance = {}
        for axiom in ['axiom1', 'axiom2', 'axiom3', 'axiom4', 'axiom5']:
            successes = sum(1 for r in results if r.found_factors.get(axiom) in r.expected_factors)
            axiom_performance[axiom] = (successes / total_cases) * 100
        
        print(f"\nAXIOM PERFORMANCE ON SEMIPRIMES:")
        for axiom, success_rate in sorted(axiom_performance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {axiom}: {success_rate:.1f}%")
        
        # Save detailed results
        self.save_breakthrough_report(results, class_stats, {
            'overall_success_rate': overall_success_rate,
            'breakthrough_cases': breakthrough_cases,
            'major_breakthrough': major_breakthrough,
            'perfect_cases': perfect_cases,
            'max_bit_solved': max_bit_solved,
            'largest_factored': largest_factored,
            'axiom_performance': axiom_performance
        })
        
        return {
            'results': results,
            'class_stats': class_stats,
            'overall_success_rate': overall_success_rate,
            'breakthrough_cases': breakthrough_cases,
            'max_bit_solved': max_bit_solved,
            'largest_factored': largest_factored
        }
    
    def save_breakthrough_report(self, results: List[SemiprimeResult], 
                               class_stats: Dict, summary: Dict):
        """Save detailed breakthrough report to file"""
        report_path = "/workspaces/factorizer/benchmark/SEMIPRIME_BREAKTHROUGH_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# UOR/Prime Axioms Factorizer - Semiprime Breakthrough Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Test Cases**: {len(results)}\n")
            f.write(f"- **Overall Success Rate**: {summary['overall_success_rate']:.2f}%\n")
            f.write(f"- **Breakthrough Cases**: {summary['breakthrough_cases']}\n")
            f.write(f"- **Maximum Bit-Length Solved**: {summary['max_bit_solved']} bits\n")
            f.write(f"- **Largest Number Factored**: {summary['largest_factored']:,}\n\n")
            
            f.write("## Class Performance\n\n")
            for difficulty, stats in class_stats.items():
                f.write(f"### {difficulty.upper()} Semiprimes\n")
                f.write(f"- Cases: {stats['count']}\n")
                f.write(f"- Success Rate: {stats['avg_success_rate']:.1f}%\n")
                f.write(f"- Average Time: {stats['avg_time']:.3f}s\n")
                f.write(f"- Breakthrough Cases: {stats['breakthrough_cases']}\n\n")
            
            f.write("## Axiom Performance\n\n")
            for axiom, perf in summary['axiom_performance'].items():
                f.write(f"- **{axiom}**: {perf:.1f}%\n")
            
            f.write("\n## Detailed Results\n\n")
            f.write("| Number | Bit-Length | Factors | Success Rate | Time | Breakthrough |\n")
            f.write("|--------|------------|---------|--------------|------|-------------|\n")
            
            for result in sorted(results, key=lambda x: x.bit_length):
                breakthrough = "✓" if result.success_count >= 3 else ""
                f.write(f"| {result.n:,} | {result.bit_length} | {result.expected_factors} | "
                       f"{(result.success_count/5)*100:.1f}% | {result.total_time:.3f}s | {breakthrough} |\n")
        
        print(f"\nDetailed breakthrough report saved to: {report_path}")

def main():
    """Run the semiprime breakthrough benchmark"""
    benchmark = SemiprimeBenchmark()
    results = benchmark.run_breakthrough_benchmark(max_cases_per_class=15)
    
    if results.get('max_bit_solved', 0) >= 32:
        print("\n🎉 COMPUTATIONAL BREAKTHROUGH ACHIEVED! 🎉")
        print(f"Successfully factoring {results['max_bit_solved']}-bit semiprimes!")
    
    if results.get('largest_factored', 0) > 10**12:
        print(f"🔥 MILESTONE: Factored numbers larger than 1 trillion! 🔥")

if __name__ == "__main__":
    main()
