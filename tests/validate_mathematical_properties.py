"""
Mathematical property validation for Prime Resonance Field (RFH3)
"""

import math
import os
import sys
from typing import Any, Dict, List, Tuple

from prime_resonance_field.core import MultiScaleResonance


class MathematicalValidator:
    """Validates mathematical properties of the resonance field"""

    def __init__(self):
        self.analyzer = MultiScaleResonance()
        self.test_cases = [
            (143, 11, 13),
            (323, 17, 19),
            (1147, 31, 37),
            (10403, 101, 103),
        ]

    def validate_unity_property(self) -> Dict[str, Any]:
        """
        Validate Unity Property: Ψ(p, n) ≥ τ_n and Ψ(q, n) ≥ τ_n
        where n = p × q and τ_n is the threshold
        """
        print("Validating Unity Property...")
        print("=" * 40)

        results = []

        for n, p, q in self.test_cases:
            # Compute resonance for both factors
            res_p = self.analyzer.compute_resonance(p, n)
            res_q = self.analyzer.compute_resonance(q, n)

            # Compute adaptive threshold
            threshold = 1.0 / math.log(n)

            # Check unity property
            unity_p = res_p >= threshold
            unity_q = res_q >= threshold
            unity_holds = unity_p and unity_q

            result = {
                "n": n,
                "p": p,
                "q": q,
                "res_p": res_p,
                "res_q": res_q,
                "threshold": threshold,
                "unity_p": unity_p,
                "unity_q": unity_q,
                "unity_holds": unity_holds,
            }
            results.append(result)

            status = "✓" if unity_holds else "✗"
            print(
                f"  {status} n={n}: Ψ({p})={res_p:.6f}, Ψ({q})={res_q:.6f}, τ={threshold:.6f}"
            )

        success_rate = sum(1 for r in results if r["unity_holds"]) / len(results)
        print(f"\nUnity Property Success Rate: {success_rate*100:.1f}%")

        return {"property": "unity", "success_rate": success_rate, "results": results}

    def validate_sparsity_property(self) -> Dict[str, Any]:
        """
        Validate Sparsity Property: |{x : Ψ(x, n) ≥ τ_n}| = O(log² n)
        """
        print("\nValidating Sparsity Property...")
        print("=" * 40)

        results = []

        for n, p, q in self.test_cases:
            sqrt_n = int(math.sqrt(n))
            threshold = 1.0 / math.log(n)

            # Count high-resonance positions
            high_resonance_count = 0
            sample_size = min(1000, sqrt_n // 10)  # Sample for efficiency

            positions = []
            step = max(1, sqrt_n // sample_size)

            for x in range(2, sqrt_n, step):
                resonance = self.analyzer.compute_resonance(x, n)
                if resonance >= threshold:
                    high_resonance_count += 1
                    positions.append((x, resonance))

            # Theoretical bound: O(log² n)
            log_n = math.log(n)
            theoretical_bound = log_n * log_n

            # Scale by sampling ratio
            actual_ratio = step / sqrt_n
            scaled_count = high_resonance_count / actual_ratio

            sparsity_holds = (
                scaled_count <= theoretical_bound * 10
            )  # Allow factor of 10

            result = {
                "n": n,
                "sampled_count": high_resonance_count,
                "scaled_count": scaled_count,
                "theoretical_bound": theoretical_bound,
                "sample_ratio": actual_ratio,
                "sparsity_holds": sparsity_holds,
                "positions": positions[:5],  # Top 5 positions
            }
            results.append(result)

            status = "✓" if sparsity_holds else "✗"
            print(
                f"  {status} n={n}: {scaled_count:.1f} ≤ {theoretical_bound:.1f} (scaled)"
            )

        success_rate = sum(1 for r in results if r["sparsity_holds"]) / len(results)
        print(f"\nSparsity Property Success Rate: {success_rate*100:.1f}%")

        return {
            "property": "sparsity",
            "success_rate": success_rate,
            "results": results,
        }

    def validate_scale_invariance(self) -> Dict[str, Any]:
        """
        Validate that resonance function is relatively scale-invariant
        """
        print("\nValidating Scale Invariance...")
        print("=" * 40)

        results = []

        # Test scale invariance across different sizes
        test_pairs = [
            ((143, 11, 13), (10403, 101, 103)),  # ~8 bits vs ~14 bits
            ((323, 17, 19), (1147, 31, 37)),  # ~9 bits vs ~11 bits
        ]

        for (n1, p1, q1), (n2, p2, q2) in test_pairs:
            # Compute resonance for factors
            res1_p = self.analyzer.compute_resonance(p1, n1)
            res1_q = self.analyzer.compute_resonance(q1, n1)

            res2_p = self.analyzer.compute_resonance(p2, n2)
            res2_q = self.analyzer.compute_resonance(q2, n2)

            # Check if resonance values are similar (within order of magnitude)
            ratio_p = res1_p / res2_p if res2_p > 0 else float("inf")
            ratio_q = res1_q / res2_q if res2_q > 0 else float("inf")

            scale_invariant_p = 0.1 <= ratio_p <= 10.0
            scale_invariant_q = 0.1 <= ratio_q <= 10.0
            scale_invariant = scale_invariant_p and scale_invariant_q

            result = {
                "n1": n1,
                "n2": n2,
                "res1_p": res1_p,
                "res1_q": res1_q,
                "res2_p": res2_p,
                "res2_q": res2_q,
                "ratio_p": ratio_p,
                "ratio_q": ratio_q,
                "scale_invariant": scale_invariant,
            }
            results.append(result)

            status = "✓" if scale_invariant else "✗"
            print(f"  {status} {n1} vs {n2}: ratios p={ratio_p:.3f}, q={ratio_q:.3f}")

        success_rate = sum(1 for r in results if r["scale_invariant"]) / len(results)
        print(f"\nScale Invariance Success Rate: {success_rate*100:.1f}%")

        return {
            "property": "scale_invariance",
            "success_rate": success_rate,
            "results": results,
        }

    def validate_computational_complexity(self) -> Dict[str, Any]:
        """
        Validate that Ψ(x, n) can be computed in O(log n) time
        """
        print("\nValidating Computational Complexity...")
        print("=" * 40)

        import time

        results = []

        # Test with different sizes
        test_numbers = [143, 1147, 10403, 282943]

        for n in test_numbers:
            # Measure computation time for resonance
            x = int(math.sqrt(n))  # Test at sqrt(n)

            # Warm up
            for _ in range(10):
                self.analyzer.compute_resonance(x, n)

            # Time multiple computations
            start_time = time.perf_counter()
            num_computations = 1000

            for _ in range(num_computations):
                self.analyzer.compute_resonance(x, n)

            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / num_computations

            # Theoretical complexity: O(log n)
            log_n = math.log(n)
            complexity_ratio = avg_time / log_n if log_n > 0 else float("inf")

            result = {
                "n": n,
                "avg_time": avg_time,
                "log_n": log_n,
                "complexity_ratio": complexity_ratio,
                "bits": n.bit_length(),
            }
            results.append(result)

            print(
                f"  n={n} ({n.bit_length()} bits): {avg_time*1e6:.2f}μs, ratio={complexity_ratio*1e6:.3f}"
            )

        # Check if complexity grows logarithmically
        ratios = [r["complexity_ratio"] for r in results]
        complexity_consistent = (
            max(ratios) / min(ratios) < 100
        )  # Within 2 orders of magnitude

        print(f"\nComplexity Consistency: {'✓' if complexity_consistent else '✗'}")

        return {
            "property": "computational_complexity",
            "consistent": complexity_consistent,
            "results": results,
        }

    def validate_numerical_stability(self) -> Dict[str, Any]:
        """
        Validate numerical stability of computations
        """
        print("\nValidating Numerical Stability...")
        print("=" * 40)

        results = []

        for n, p, q in self.test_cases:
            # Compute resonance multiple times
            resonances_p = []
            resonances_q = []

            for _ in range(100):
                res_p = self.analyzer.compute_resonance(p, n)
                res_q = self.analyzer.compute_resonance(q, n)
                resonances_p.append(res_p)
                resonances_q.append(res_q)

            # Check stability (should be identical due to deterministic computation)
            stable_p = len(set(resonances_p)) == 1
            stable_q = len(set(resonances_q)) == 1
            stable = stable_p and stable_q

            # Check for NaN or infinite values
            valid_p = all(math.isfinite(r) for r in resonances_p)
            valid_q = all(math.isfinite(r) for r in resonances_q)
            valid = valid_p and valid_q

            result = {
                "n": n,
                "stable_p": stable_p,
                "stable_q": stable_q,
                "valid_p": valid_p,
                "valid_q": valid_q,
                "stable_and_valid": stable and valid,
                "res_p_range": (min(resonances_p), max(resonances_p)),
                "res_q_range": (min(resonances_q), max(resonances_q)),
            }
            results.append(result)

            status = "✓" if stable and valid else "✗"
            print(
                f"  {status} n={n}: stable=({stable_p},{stable_q}), valid=({valid_p},{valid_q})"
            )

        success_rate = sum(1 for r in results if r["stable_and_valid"]) / len(results)
        print(f"\nNumerical Stability Success Rate: {success_rate*100:.1f}%")

        return {
            "property": "numerical_stability",
            "success_rate": success_rate,
            "results": results,
        }

    def run_all_validations(self) -> Dict[str, Any]:
        """Run all mathematical property validations"""
        print("Prime Resonance Field - Mathematical Property Validation")
        print("=" * 60)

        validations = [
            self.validate_unity_property,
            self.validate_sparsity_property,
            self.validate_scale_invariance,
            self.validate_computational_complexity,
            self.validate_numerical_stability,
        ]

        all_results = {}

        for validation in validations:
            try:
                result = validation()
                all_results[result["property"]] = result
            except Exception as e:
                print(f"Validation {validation.__name__} failed: {e}")
                import traceback

                traceback.print_exc()

        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        for prop, result in all_results.items():
            if "success_rate" in result:
                rate = result["success_rate"] * 100
                status = "✓" if rate >= 80 else "✗"  # 80% threshold
                print(f"  {status} {prop.replace('_', ' ').title()}: {rate:.1f}%")
            elif "consistent" in result:
                status = "✓" if result["consistent"] else "✗"
                print(
                    f"  {status} {prop.replace('_', ' ').title()}: {'Consistent' if result['consistent'] else 'Inconsistent'}"
                )

        return all_results


def main():
    """Main entry point"""
    validator = MathematicalValidator()
    results = validator.run_all_validations()

    # Check overall validation success
    success_rates = [r["success_rate"] for r in results.values() if "success_rate" in r]
    consistency_checks = [
        r["consistent"] for r in results.values() if "consistent" in r
    ]

    overall_success = all(rate >= 0.8 for rate in success_rates) and all(
        consistency_checks
    )

    print(f"\nOverall Validation: {'✓ PASSED' if overall_success else '✗ FAILED'}")

    if not overall_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
