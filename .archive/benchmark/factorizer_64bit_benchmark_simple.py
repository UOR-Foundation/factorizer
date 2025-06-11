"""
Simplified 64-bit Benchmark for the UOR/Prime Axioms Factorizer

This version uses pre-selected test cases to avoid hanging during prime generation.
"""

import time
import sys
import os
import json
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factorizer import Factorizer, FactorizationResult


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


class Factorizer64BitBenchmarkSimple:
    """Simplified benchmark for factorizer up to 64-bit numbers"""
    
    def __init__(self, use_learning: bool = True):
        """
        Initialize benchmark
        
        Args:
            use_learning: Whether to use Axiom 5 learning capabilities
        """
        self.use_learning = use_learning
        
        # Initialize factorizer with acceleration
        print(f"Initializing factorizer (learning={'enabled' if use_learning else 'disabled'})...")
        self.factorizer = Factorizer(learning_enabled=use_learning)
        
        # Use pre-selected test cases
        self.test_cases = self._get_preselected_cases()
        
        # Results storage
        self.results: List[BenchmarkRun] = []
    
    def _get_preselected_cases(self) -> List[BenchmarkCase]:
        """Get pre-selected test cases to avoid hanging during generation"""
        cases = []
        
        # 8-bit cases
        cases.extend([
            BenchmarkCase(15, 3, 5, 8, "easy"),
            BenchmarkCase(21, 3, 7, 8, "easy"),
            BenchmarkCase(35, 5, 7, 8, "easy"),
            BenchmarkCase(77, 7, 11, 8, "easy"),
            BenchmarkCase(91, 7, 13, 8, "easy"),
            BenchmarkCase(143, 11, 13, 8, "easy"),
            BenchmarkCase(187, 11, 17, 8, "medium"),
            BenchmarkCase(221, 13, 17, 8, "medium"),
            BenchmarkCase(247, 13, 19, 8, "medium"),
            # Special cases
            BenchmarkCase(9, 3, 3, 8, "special"),   # Perfect square
            BenchmarkCase(25, 5, 5, 8, "special"),  # Perfect square
            BenchmarkCase(6, 2, 3, 8, "fibonacci"), # Fib primes
        ])
        
        # 16-bit cases
        cases.extend([
            BenchmarkCase(323, 17, 19, 16, "easy"),
            BenchmarkCase(391, 17, 23, 16, "easy"),
            BenchmarkCase(667, 23, 29, 16, "easy"),
            BenchmarkCase(899, 29, 31, 16, "easy"),
            BenchmarkCase(1147, 31, 37, 16, "easy"),
            BenchmarkCase(1517, 37, 41, 16, "easy"),
            BenchmarkCase(2021, 43, 47, 16, "easy"),
            BenchmarkCase(2491, 47, 53, 16, "easy"),
            BenchmarkCase(3127, 53, 59, 16, "medium"),
            BenchmarkCase(4087, 61, 67, 16, "medium"),
            BenchmarkCase(4757, 67, 71, 16, "medium"),
            BenchmarkCase(5767, 73, 79, 16, "medium"),
            BenchmarkCase(6557, 79, 83, 16, "medium"),
            BenchmarkCase(8633, 89, 97, 16, "medium"),
            BenchmarkCase(10403, 101, 103, 16, "hard"),  # Close primes
            BenchmarkCase(11021, 103, 107, 16, "hard"),  # Close primes
            BenchmarkCase(11663, 107, 109, 16, "hard"),  # Close primes
            BenchmarkCase(15251, 113, 135, 16, "medium"), # Note: 135 = 3³×5
            # Special cases
            BenchmarkCase(169, 13, 13, 16, "special"),   # Perfect square
            BenchmarkCase(377, 13, 29, 16, "fibonacci"), # 13 is Fib
            BenchmarkCase(1597, 37, 43, 16, "medium"),   # 1597 is Fib
        ])
        
        # 32-bit cases  
        cases.extend([
            BenchmarkCase(65537 * 257, 257, 65537, 32, "easy"),        # 16,843,009
            BenchmarkCase(65537 * 1021, 1021, 65537, 32, "easy"),      # 66,912,877
            BenchmarkCase(65537 * 2053, 2053, 65537, 32, "easy"),      # 134,545,861
            BenchmarkCase(65537 * 4099, 4099, 65537, 32, "easy"),      # 268,665,763
            BenchmarkCase(65537 * 8209, 8209, 65537, 32, "easy"),      # 537,980,833
            BenchmarkCase(16411 * 16417, 16411, 16417, 32, "hard"),    # 269,363,387 (close primes)
            BenchmarkCase(32771 * 32779, 32771, 32779, 32, "hard"),    # 1,074,266,209 (close primes)
            BenchmarkCase(65521 * 65537, 65521, 65537, 32, "hard"),    # 4,293,918,577 (close primes)
        ])
        
        # 64-bit cases (using smaller factors to keep computation reasonable)
        cases.extend([
            # Easy cases with one small prime
            BenchmarkCase(65537 * 4294967291, 65537, 4294967291, 64, "easy"),  # 281,470,681,677,567
            BenchmarkCase(131071 * 2147483647, 131071, 2147483647, 64, "easy"), # 281,474,959,933,537
            # Medium cases  
            BenchmarkCase(1073741827 * 1073741831, 1073741827, 1073741831, 64, "medium"), # 1,152,921,510,754,820,637
            # Special Fibonacci-related
            BenchmarkCase(2971215073, 59393, 50021, 64, "fibonacci"),  # 2^32 - 1 is close
        ])
        
        print(f"Loaded {len(cases)} pre-selected test cases")
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
    
    def run_benchmark(self, max_cases: Optional[int] = None, bit_range: Optional[int] = None) -> None:
        """
        Run the benchmark
        
        Args:
            max_cases: Maximum number of cases to run
            bit_range: If specified, only run cases up to this bit size
        """
        print(f"\nStarting 64-bit Factorizer Benchmark (Simplified)")
        print(f"Total test cases: {len(self.test_cases)}")
        print(f"Learning: {'Enabled' if self.use_learning else 'Disabled'}")
        print("=" * 80)
        
        # Run warmup
        self.run_warmup()
        
        # Clear results
        self.results = []
        
        # Filter cases by bit range if specified
        cases_to_run = self.test_cases
        if bit_range:
            cases_to_run = [c for c in cases_to_run if c.bit_size <= bit_range]
        
        # Limit cases if specified
        if max_cases:
            cases_to_run = cases_to_run[:max_cases]
        
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
        report.append("UOR/Prime Axioms Factorizer - 64-bit Benchmark Report (Simplified)")
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
    """Run the simplified 64-bit factorizer benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified 64-bit Factorizer Benchmark")
    parser.add_argument("--no-learning", action="store_true", help="Disable learning/acceleration")
    parser.add_argument("--max-cases", type=int, help="Maximum number of cases to run")
    parser.add_argument("--bit-range", type=int, choices=[8, 16, 32, 64], help="Maximum bit size to test")
    parser.add_argument("--output", default="factorizer_64bit_simple_results.json", help="Output filename for results")
    
    args = parser.parse_args()
    
    # Create and run benchmark
    benchmark = Factorizer64BitBenchmarkSimple(
        use_learning=not args.no_learning
    )
    
    # Run benchmark
    benchmark.run_benchmark(max_cases=args.max_cases, bit_range=args.bit_range)
    
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
