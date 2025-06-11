"""
Comprehensive 64-bit Benchmark for the UOR/Prime Axioms Factorizer

This benchmark tests the factorizer performance across different bit ranges
up to 64-bit, leveraging all acceleration features including:
- Prime coordinate pre-computation
- Fibonacci resonance maps
- Spectral signature caching
- Coherence caching
- Observer caching
- Meta-acceleration caching
- Learning/memory systems
"""

import time
import sys
import os
import random
import json
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factorizer import Factorizer, FactorizationResult
from axiom1.prime_core import is_prime, primes_up_to


@dataclass
class BenchmarkCase:
    """Single benchmark test case"""
    n: int
    p: int
    q: int
    bit_size: int
    difficulty: str  # easy, medium, hard


@dataclass
class BenchmarkRun:
    """Result of a single factorization run"""
    case: BenchmarkCase
    result: FactorizationResult
    success: bool
    error: Optional[str] = None


@dataclass
class BitRangeStats:
    """Statistics for a specific bit range"""
    bit_size: int
    total_cases: int
    successful: int
    failed: int
    avg_time: float
    min_time: float
    max_time: float
    avg_iterations: float
    avg_candidates: float
    avg_coherence: float
    primary_axioms: Dict[str, int]
    learning_used: int


class Factorizer64BitBenchmark:
    """Comprehensive benchmark for factorizer up to 64-bit numbers"""
    
    def __init__(self, use_learning: bool = True, seed: Optional[int] = None):
        """
        Initialize benchmark
        
        Args:
            use_learning: Whether to use Axiom 5 learning capabilities
            seed: Random seed for reproducible results
        """
        self.use_learning = use_learning
        if seed is not None:
            random.seed(seed)
        
        # Initialize factorizer with acceleration
        print(f"Initializing factorizer (learning={'enabled' if use_learning else 'disabled'})...")
        self.factorizer = Factorizer(learning_enabled=use_learning)
        
        # Generate test cases
        self.test_cases = self._generate_test_cases()
        
        # Results storage
        self.results: List[BenchmarkRun] = []
    
    def _generate_test_cases(self) -> List[BenchmarkCase]:
        """Generate comprehensive test cases across bit ranges"""
        cases = []
        
        # Pre-generate primes for each bit range
        primes_8bit = [p for p in primes_up_to(256) if p >= 16]
        primes_16bit = [p for p in primes_up_to(65536) if p >= 256]
        primes_32bit = []
        primes_64bit = []
        
        # Generate some 32-bit primes
        for _ in range(20):
            candidate = random.randint(65536, 2**31 - 1)
            while not is_prime(candidate):
                candidate += 2
                if candidate >= 2**31:
                    candidate = random.randint(65536, 2**31 - 1)
            primes_32bit.append(candidate)
        
        # Generate some 64-bit primes
        for _ in range(10):
            candidate = random.randint(2**31, 2**63 - 1)
            attempts = 0
            while not is_prime(candidate) and attempts < 1000:
                candidate += 2
                attempts += 1
                if candidate >= 2**63:
                    candidate = random.randint(2**31, 2**63 - 1)
            if is_prime(candidate):
                primes_64bit.append(candidate)
        
        # 8-bit range (up to 255)
        print("Generating 8-bit test cases...")
        for i in range(10):
            p = random.choice(primes_8bit)
            q = random.choice(primes_8bit)
            n = p * q
            if n < 256:
                cases.append(BenchmarkCase(n, min(p, q), max(p, q), 8, "easy"))
        
        # 16-bit range (256 to 65,535)
        print("Generating 16-bit test cases...")
        for i in range(10):
            p = random.choice(primes_16bit)
            q = random.choice(primes_16bit)
            n = p * q
            if 256 <= n < 65536:
                cases.append(BenchmarkCase(n, min(p, q), max(p, q), 16, "easy"))
        
        # Add some harder 16-bit cases (closer primes)
        for i in range(5):
            p = random.choice(primes_16bit[len(primes_16bit)//2:])
            # Find a prime close to p
            q = p + 2
            while not is_prime(q) and q < 65536:
                q += 2
            if is_prime(q):
                n = p * q
                if n < 65536:
                    cases.append(BenchmarkCase(n, min(p, q), max(p, q), 16, "medium"))
        
        # 32-bit range (65,536 to 4,294,967,295)
        print("Generating 32-bit test cases...")
        for i in range(10):
            if i < 5 and primes_32bit:
                # Use generated 32-bit primes
                p = random.choice(primes_32bit)
                q = random.choice(primes_32bit)
            else:
                # Mix 16-bit and 32-bit primes
                p = random.choice(primes_16bit[-20:])
                q = random.choice(primes_32bit) if primes_32bit else random.choice(primes_16bit[-20:])
            
            n = p * q
            if 65536 <= n < 2**32:
                difficulty = "medium" if abs(p - q) < max(p, q) * 0.1 else "easy"
                cases.append(BenchmarkCase(n, min(p, q), max(p, q), 32, difficulty))
        
        # 64-bit range (4,294,967,296 to 2^64-1)
        print("Generating 64-bit test cases...")
        
        # Easy 64-bit cases (one small prime)
        for i in range(5):
            p = random.choice(primes_16bit[-50:])
            if primes_64bit:
                q = random.choice(primes_64bit)
            else:
                # Generate a large prime-like number
                q = random.randint(2**32, 2**48)
                while not is_prime(q):
                    q += 2
            
            n = p * q
            if 2**32 <= n < 2**64:
                cases.append(BenchmarkCase(n, min(p, q), max(p, q), 64, "easy"))
        
        # Medium 64-bit cases
        if len(primes_32bit) >= 2:
            for i in range(5):
                p = random.choice(primes_32bit)
                q = random.choice(primes_32bit)
                n = p * q
                if 2**32 <= n < 2**64:
                    cases.append(BenchmarkCase(n, min(p, q), max(p, q), 64, "medium"))
        
        # Hard 64-bit cases (close primes)
        if primes_64bit:
            for i in range(min(3, len(primes_64bit))):
                p = primes_64bit[i]
                # Try to find a close prime
                q = p + 2
                attempts = 0
                while not is_prime(q) and attempts < 100:
                    q += 2
                    attempts += 1
                if is_prime(q):
                    n = p * q
                    if n < 2**64:
                        cases.append(BenchmarkCase(n, min(p, q), max(p, q), 64, "hard"))
        
        # Add some special cases
        # Perfect squares
        special_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        for p in special_primes[:5]:
            cases.append(BenchmarkCase(p*p, p, p, 8 if p*p < 256 else 16, "special"))
        
        # Fibonacci-related semiprimes
        fib_primes = [2, 3, 5, 13, 89, 233, 1597]
        for i in range(len(fib_primes)-1):
            n = fib_primes[i] * fib_primes[i+1]
            bit_size = n.bit_length()
            if bit_size <= 64:
                cases.append(BenchmarkCase(
                    n, 
                    min(fib_primes[i], fib_primes[i+1]), 
                    max(fib_primes[i], fib_primes[i+1]),
                    8 if bit_size <= 8 else 16 if bit_size <= 16 else 32 if bit_size <= 32 else 64,
                    "fibonacci"
                ))
        
        print(f"Generated {len(cases)} test cases")
        return cases
    
    def run_single_case(self, case: BenchmarkCase) -> BenchmarkRun:
        """Run factorization on a single test case"""
        try:
            # Run factorization with detailed results
            result = self.factorizer.factorize_with_details(case.n)
            
            # Check success
            success = (
                result.factors[0] * result.factors[1] == case.n and
                result.factors[0] > 1 and result.factors[1] > 1
            )
            
            # Verify against expected factors
            if success:
                expected = {case.p, case.q}
                actual = {result.factors[0], result.factors[1]}
                success = expected == actual
            
            return BenchmarkRun(
                case=case,
                result=result,
                success=success,
                error=None if success else "Incorrect factors"
            )
            
        except Exception as e:
            # Create a dummy result for failures
            dummy_result = FactorizationResult(
                factors=(1, case.n),
                primary_axiom="error",
                iterations=0,
                max_coherence=0.0,
                candidates_explored=0,
                time_elapsed=0.0,
                method_sequence=[],
                learning_applied=False
            )
            
            return BenchmarkRun(
                case=case,
                result=dummy_result,
                success=False,
                error=str(e)
            )
    
    def run_warmup(self):
        """Run warmup phase to initialize caches"""
        print("Running warmup phase...")
        warmup_cases = [
            BenchmarkCase(15, 3, 5, 8, "warmup"),
            BenchmarkCase(77, 7, 11, 8, "warmup"),
            BenchmarkCase(323, 17, 19, 16, "warmup"),
            BenchmarkCase(667, 23, 29, 16, "warmup"),
        ]
        
        for case in warmup_cases:
            _ = self.run_single_case(case)
        
        print("Warmup complete - caches initialized")
    
    def run_benchmark(self, max_cases: Optional[int] = None) -> None:
        """Run the complete benchmark"""
        print(f"\nStarting 64-bit Factorizer Benchmark")
        print(f"Total test cases: {len(self.test_cases)}")
        print(f"Learning: {'Enabled' if self.use_learning else 'Disabled'}")
        print("=" * 80)
        
        # Run warmup
        self.run_warmup()
        
        # Clear results
        self.results = []
        
        # Determine cases to run
        cases_to_run = self.test_cases[:max_cases] if max_cases else self.test_cases
        
        # Group by bit size for progress reporting
        cases_by_bit = {}
        for case in cases_to_run:
            if case.bit_size not in cases_by_bit:
                cases_by_bit[case.bit_size] = []
            cases_by_bit[case.bit_size].append(case)
        
        # Run benchmarks by bit size
        for bit_size in sorted(cases_by_bit.keys()):
            print(f"\nTesting {bit_size}-bit numbers ({len(cases_by_bit[bit_size])} cases)...")
            
            for i, case in enumerate(cases_by_bit[bit_size]):
                print(f"  Case {i+1}/{len(cases_by_bit[bit_size])}: "
                      f"n={case.n} ({case.difficulty})", end='', flush=True)
                
                # Run the case
                run = self.run_single_case(case)
                self.results.append(run)
                
                # Report result
                if run.success:
                    print(f" ✓ {run.result.time_elapsed:.3f}s "
                          f"({run.result.iterations} iter, "
                          f"{run.result.candidates_explored} candidates)")
                else:
                    print(f" ✗ Failed: {run.error}")
    
    def analyze_results(self) -> Dict[int, BitRangeStats]:
        """Analyze benchmark results by bit range"""
        stats_by_bit = {}
        
        # Group results by bit size
        results_by_bit = {}
        for run in self.results:
            bit_size = run.case.bit_size
            if bit_size not in results_by_bit:
                results_by_bit[bit_size] = []
            results_by_bit[bit_size].append(run)
        
        # Calculate statistics for each bit range
        for bit_size, runs in results_by_bit.items():
            successful_runs = [r for r in runs if r.success]
            
            # Time statistics
            times = [r.result.time_elapsed for r in successful_runs]
            avg_time = statistics.mean(times) if times else 0
            min_time = min(times) if times else 0
            max_time = max(times) if times else 0
            
            # Iteration and candidate statistics
            iterations = [r.result.iterations for r in successful_runs]
            candidates = [r.result.candidates_explored for r in successful_runs]
            coherences = [r.result.max_coherence for r in successful_runs]
            
            # Axiom usage
            axiom_counts = {}
            learning_count = 0
            for r in successful_runs:
                axiom = r.result.primary_axiom
                axiom_counts[axiom] = axiom_counts.get(axiom, 0) + 1
                if r.result.learning_applied:
                    learning_count += 1
            
            stats_by_bit[bit_size] = BitRangeStats(
                bit_size=bit_size,
                total_cases=len(runs),
                successful=len(successful_runs),
                failed=len(runs) - len(successful_runs),
                avg_time=avg_time,
                min_time=min_time,
                max_time=max_time,
                avg_iterations=statistics.mean(iterations) if iterations else 0,
                avg_candidates=statistics.mean(candidates) if candidates else 0,
                avg_coherence=statistics.mean(coherences) if coherences else 0,
                primary_axioms=axiom_counts,
                learning_used=learning_count
            )
        
        return stats_by_bit
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report"""
        stats = self.analyze_results()
        
        report = []
        report.append("=" * 80)
        report.append("UOR/Prime Axioms Factorizer - 64-bit Benchmark Report")
        report.append("=" * 80)
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Learning: {'Enabled' if self.use_learning else 'Disabled'}")
        report.append(f"Total test cases: {len(self.results)}")
        
        # Overall statistics
        total_success = sum(s.successful for s in stats.values())
        total_cases = sum(s.total_cases for s in stats.values())
        overall_success_rate = total_success / total_cases if total_cases > 0 else 0
        
        report.append(f"\nOverall Success Rate: {overall_success_rate:.1%} ({total_success}/{total_cases})")
        
        # Statistics by bit range
        report.append("\n" + "=" * 80)
        report.append("Performance by Bit Range:")
        report.append("=" * 80)
        
        for bit_size in sorted(stats.keys()):
            s = stats[bit_size]
            success_rate = s.successful / s.total_cases if s.total_cases > 0 else 0
            
            report.append(f"\n{bit_size}-bit Numbers:")
            report.append(f"  Success Rate: {success_rate:.1%} ({s.successful}/{s.total_cases})")
            report.append(f"  Time: avg={s.avg_time:.3f}s, min={s.min_time:.3f}s, max={s.max_time:.3f}s")
            report.append(f"  Iterations: avg={s.avg_iterations:.1f}")
            report.append(f"  Candidates Explored: avg={s.avg_candidates:.1f}")
            report.append(f"  Max Coherence: avg={s.avg_coherence:.3f}")
            
            if s.primary_axioms:
                report.append("  Primary Axioms Used:")
                for axiom, count in sorted(s.primary_axioms.items(), key=lambda x: x[1], reverse=True):
                    report.append(f"    - {axiom}: {count} times")
            
            if self.use_learning and s.learning_used > 0:
                report.append(f"  Learning Applied: {s.learning_used} times ({s.learning_used/s.successful*100:.1f}% of successes)")
        
        # Difficulty analysis
        report.append("\n" + "=" * 80)
        report.append("Performance by Difficulty:")
        report.append("=" * 80)
        
        difficulty_stats = {}
        for run in self.results:
            diff = run.case.difficulty
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {"total": 0, "success": 0, "times": []}
            difficulty_stats[diff]["total"] += 1
            if run.success:
                difficulty_stats[diff]["success"] += 1
                difficulty_stats[diff]["times"].append(run.result.time_elapsed)
        
        for diff in ["easy", "medium", "hard", "special", "fibonacci"]:
            if diff in difficulty_stats:
                stats = difficulty_stats[diff]
                success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
                avg_time = statistics.mean(stats["times"]) if stats["times"] else 0
                report.append(f"\n{diff.capitalize()}:")
                report.append(f"  Success Rate: {success_rate:.1%} ({stats['success']}/{stats['total']})")
                if stats["times"]:
                    report.append(f"  Average Time: {avg_time:.3f}s")
        
        # Method sequence analysis
        report.append("\n" + "=" * 80)
        report.append("Method Sequence Analysis:")
        report.append("=" * 80)
        
        method_counts = {}
        for run in self.results:
            if run.success:
                for method in run.result.method_sequence:
                    method_counts[method] = method_counts.get(method, 0) + 1
        
        report.append("\nMost Used Methods:")
        for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"  - {method}: {count} times")
        
        # Acceleration impact
        if self.use_learning:
            report.append("\n" + "=" * 80)
            report.append("Acceleration Impact:")
            report.append("=" * 80)
            
            # Compare cases where learning was applied vs not
            learning_times = []
            no_learning_times = []
            
            for run in self.results:
                if run.success:
                    if run.result.learning_applied:
                        learning_times.append(run.result.time_elapsed)
                    else:
                        no_learning_times.append(run.result.time_elapsed)
            
            if learning_times and no_learning_times:
                avg_with_learning = statistics.mean(learning_times)
                avg_without_learning = statistics.mean(no_learning_times)
                speedup = avg_without_learning / avg_with_learning if avg_with_learning > 0 else 1
                
                report.append(f"\nCases with learning applied: {len(learning_times)}")
                report.append(f"Average time with learning: {avg_with_learning:.3f}s")
                report.append(f"Average time without learning: {avg_without_learning:.3f}s")
                report.append(f"Learning speedup factor: {speedup:.2f}x")
        
        return "\n".join(report)
    
    def save_results(self, filename: str):
        """Save detailed results to JSON file"""
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "learning_enabled": self.use_learning,
                "total_cases": len(self.results),
                "successful": sum(1 for r in self.results if r.success)
            },
            "results": []
        }
        
        for run in self.results:
            results_data["results"].append({
                "case": {
                    "n": run.case.n,
                    "p": run.case.p,
                    "q": run.case.q,
                    "bit_size": run.case.bit_size,
                    "difficulty": run.case.difficulty
                },
                "result": {
                    "success": run.success,
                    "factors": run.result.factors,
                    "time": run.result.time_elapsed,
                    "iterations": run.result.iterations,
                    "candidates_explored": run.result.candidates_explored,
                    "max_coherence": run.result.max_coherence,
                    "primary_axiom": run.result.primary_axiom,
                    "method_sequence": run.result.method_sequence,
                    "learning_applied": run.result.learning_applied
                },
                "error": run.error
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")


def main():
    """Run the 64-bit factorizer benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description="64-bit Factorizer Benchmark")
    parser.add_argument("--no-learning", action="store_true", help="Disable learning/acceleration")
    parser.add_argument("--max-cases", type=int, help="Maximum number of cases to run")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output", default="factorizer_64bit_results.json", help="Output filename for results")
    
    args = parser.parse_args()
    
    # Create and run benchmark
    benchmark = Factorizer64BitBenchmark(
        use_learning=not args.no_learning,
        seed=args.seed
    )
    
    # Run benchmark
    benchmark.run_benchmark(max_cases=args.max_cases)
    
    # Generate and display report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Save results
    benchmark.save_results(args.output)
    
    # Save report
    report_filename = args.output.replace('.json', '_report.txt')
    with open(report_filename, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_filename}")


if __name__ == "__main__":
    main()
