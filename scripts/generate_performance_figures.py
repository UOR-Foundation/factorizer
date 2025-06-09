"""
Generate performance figures for Prime Resonance Field (RFH3) research
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prime_resonance_field import RFH3, RFH3Config


class PerformanceFigureGenerator:
    """Generates performance analysis figures for research publication"""

    def __init__(self):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            self.plt = plt
            self.sns = sns
            self.plotting_available = True

            # Set up plotting style
            plt.style.use("seaborn-v0_8")
            sns.set_palette("husl")

        except ImportError:
            self.plotting_available = False
            print(
                "Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn"
            )

    def generate_success_rate_by_bits(
        self, save_path: str = "success_rate_by_bits.png"
    ):
        """Generate success rate vs bit size figure"""
        if not self.plotting_available:
            print("Plotting not available - skipping figure generation")
            return

        print("Generating success rate by bit size figure...")

        # Test data by bit ranges
        test_data = [
            (8, 143, 11, 13),
            (9, 323, 17, 19),
            (11, 1147, 31, 37),
            (14, 10403, 101, 103),
            (19, 282943, 523, 541),
            (21, 1299071, 1129, 1151),
            (24, 16777207, 4093, 4099),
            (27, 68718952001, 262139, 262147),
        ]

        rfh3 = RFH3()

        bits = []
        success_rates = []
        times = []

        for bit_size, n, p_true, q_true in test_data:
            print(f"  Testing {bit_size}-bit number: {n}")

            successes = 0
            total_time = 0
            trials = 3  # Multiple trials for statistics

            for trial in range(trials):
                try:
                    start = time.time()
                    p_found, q_found = rfh3.factor(n, timeout=60.0)
                    elapsed = time.time() - start

                    if {p_found, q_found} == {p_true, q_true}:
                        successes += 1

                    total_time += elapsed

                except Exception:
                    total_time += 60.0  # Timeout time

            success_rate = successes / trials * 100
            avg_time = total_time / trials

            bits.append(bit_size)
            success_rates.append(success_rate)
            times.append(avg_time)

            print(
                f"    {bit_size} bits: {success_rate:.1f}% success, {avg_time:.3f}s avg"
            )

        # Create figure
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(12, 5))

        # Success rate plot
        ax1.plot(bits, success_rates, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("Number Size (bits)")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("RFH3 Success Rate vs Bit Size")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        # Add theoretical performance boundary
        theoretical_bits = np.linspace(8, max(bits), 100)
        theoretical_rate = 100 * np.exp(
            -(theoretical_bits - 8) * 0.1
        )  # Exponential decay
        ax1.plot(
            theoretical_bits,
            theoretical_rate,
            "--",
            alpha=0.7,
            label="Theoretical Limit",
        )
        ax1.legend()

        # Time complexity plot
        ax2.semilogy(bits, times, "o-", linewidth=2, markersize=8, color="orange")
        ax2.set_xlabel("Number Size (bits)")
        ax2.set_ylabel("Average Time (seconds)")
        ax2.set_title("RFH3 Time Complexity")
        ax2.grid(True, alpha=0.3)

        # Add polynomial fit
        if len(bits) > 2:
            poly_coeffs = np.polyfit(bits, np.log(times), 1)
            poly_fit = np.exp(np.polyval(poly_coeffs, bits))
            ax2.plot(
                bits,
                poly_fit,
                "--",
                alpha=0.7,
                label=f"Exponential Fit (slope={poly_coeffs[0]:.2f})",
            )
            ax2.legend()

        self.plt.tight_layout()
        self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

        return {"bits": bits, "success_rates": success_rates, "times": times}

    def generate_phase_effectiveness_comparison(
        self, save_path: str = "phase_effectiveness.png"
    ):
        """Generate phase effectiveness comparison figure"""
        if not self.plotting_available:
            return

        print("Generating phase effectiveness comparison...")

        # Different configurations
        configs = {
            "Full RFH3": RFH3Config(),
            "No Learning": self._create_config(learning_enabled=False),
            "No Hierarchical": self._create_config(hierarchical_search=False),
            "Basic Only": self._create_config(
                learning_enabled=False, hierarchical_search=False
            ),
        }

        test_cases = [
            (143, 11, 13),
            (323, 17, 19),
            (1147, 31, 37),
            (10403, 101, 103),
        ]

        config_names = []
        success_rates = []
        avg_times = []

        for config_name, config in configs.items():
            print(f"  Testing {config_name}...")

            rfh3 = RFH3(config)
            successes = 0
            total_time = 0

            for n, p_true, q_true in test_cases:
                try:
                    start = time.time()
                    p_found, q_found = rfh3.factor(n, timeout=30.0)
                    elapsed = time.time() - start

                    if {p_found, q_found} == {p_true, q_true}:
                        successes += 1

                    total_time += elapsed

                except Exception:
                    total_time += 30.0

            success_rate = successes / len(test_cases) * 100
            avg_time = total_time / len(test_cases)

            config_names.append(config_name)
            success_rates.append(success_rate)
            avg_times.append(avg_time)

            print(f"    {success_rate:.1f}% success, {avg_time:.3f}s avg")

        # Create figure
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(12, 5))

        # Success rate comparison
        bars1 = ax1.bar(config_names, success_rates, alpha=0.7)
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("Phase Effectiveness: Success Rate")
        ax1.set_ylim(0, 105)

        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
            )

        # Time comparison
        bars2 = ax2.bar(config_names, avg_times, alpha=0.7, color="orange")
        ax2.set_ylabel("Average Time (seconds)")
        ax2.set_title("Phase Effectiveness: Time Performance")

        # Add value labels on bars
        for bar, time_val in zip(bars2, avg_times):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{time_val:.2f}s",
                ha="center",
                va="bottom",
            )

        # Rotate x-axis labels
        for ax in [ax1, ax2]:
            ax.tick_params(axis="x", rotation=45)

        self.plt.tight_layout()
        self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

        return {
            "configs": config_names,
            "success_rates": success_rates,
            "avg_times": avg_times,
        }

    def generate_resonance_field_visualization(
        self, n: int = 143, save_path: str = "resonance_field_143.png"
    ):
        """Generate resonance field visualization"""
        if not self.plotting_available:
            return

        print(f"Generating resonance field visualization for n={n}...")

        from prime_resonance_field.core.multi_scale_resonance import MultiScaleResonance

        analyzer = MultiScaleResonance()
        sqrt_n = int(np.sqrt(n))

        # Compute resonance for range of x values
        x_values = np.arange(2, sqrt_n + 1)
        resonances = []

        for x in x_values:
            resonance = analyzer.compute_resonance(x, n)
            resonances.append(resonance)

        # Find actual factors
        factors = []
        for x in x_values:
            if n % x == 0:
                factors.append((x, analyzer.compute_resonance(x, n)))

        # Create figure
        fig, ax = self.plt.subplots(figsize=(10, 6))

        # Plot resonance field
        ax.plot(
            x_values, resonances, "b-", alpha=0.7, linewidth=1, label="Resonance Field"
        )

        # Highlight factors
        if factors:
            factor_x = [f[0] for f in factors]
            factor_res = [f[1] for f in factors]
            ax.scatter(
                factor_x,
                factor_res,
                color="red",
                s=100,
                zorder=5,
                label=f"Factors: {factor_x}",
            )

        # Add threshold line
        threshold = 1.0 / np.log(n)
        ax.axhline(
            y=threshold,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold: {threshold:.4f}",
        )

        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Resonance Ψ(x, n)")
        ax.set_title(f"Resonance Field for n = {n}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add annotations for factors
        for x, res in factors:
            ax.annotate(
                f"x={x}",
                (x, res),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        self.plt.tight_layout()
        self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

        return {
            "x_values": x_values.tolist(),
            "resonances": resonances,
            "factors": factors,
            "threshold": threshold,
        }

    def generate_algorithm_comparison_chart(
        self, save_path: str = "algorithm_comparison.png"
    ):
        """Generate algorithm comparison chart"""
        if not self.plotting_available:
            return

        print("Generating algorithm comparison chart...")

        from prime_resonance_field.benchmark import BenchmarkSuite

        suite = BenchmarkSuite()
        results = suite.compare_algorithms()

        # Extract data for plotting
        algorithms = list(results.keys())
        success_rates = []
        avg_times = []

        for alg_name in algorithms:
            alg_results = results[alg_name]
            successes = sum(1 for r in alg_results if r.get("success", False))
            total = len(alg_results)

            successful_times = [
                r["time"] for r in alg_results if r.get("success", False)
            ]
            avg_time = np.mean(successful_times) if successful_times else 0

            success_rates.append(successes / total * 100)
            avg_times.append(avg_time)

        # Create figure
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(12, 5))

        # Success rate comparison
        bars1 = ax1.bar(algorithms, success_rates, alpha=0.7)
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("Algorithm Comparison: Success Rate")
        ax1.set_ylim(0, 105)

        # Add value labels
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Time comparison (only for successful cases)
        bars2 = ax2.bar(algorithms, avg_times, alpha=0.7, color="orange")
        ax2.set_ylabel("Average Time (seconds)")
        ax2.set_title("Algorithm Comparison: Performance")

        # Add value labels
        for bar, time_val in zip(bars2, avg_times):
            height = bar.get_height()
            if height > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.001,
                    f"{time_val:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Rotate labels
        for ax in [ax1, ax2]:
            ax.tick_params(axis="x", rotation=45)

        self.plt.tight_layout()
        self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

        return results

    def _create_config(self, **kwargs) -> RFH3Config:
        """Create configuration with overrides"""
        config = RFH3Config()
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config

    def generate_all_figures(self, output_dir: str = "figures"):
        """Generate all performance figures"""
        if not self.plotting_available:
            print("Plotting libraries not available. Install with:")
            print("pip install matplotlib seaborn")
            return

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("Prime Resonance Field (RFH3) - Performance Figure Generation")
        print("=" * 60)

        figures = [
            (
                "Success Rate by Bit Size",
                lambda: self.generate_success_rate_by_bits(
                    os.path.join(output_dir, "success_rate_by_bits.png")
                ),
            ),
            (
                "Phase Effectiveness",
                lambda: self.generate_phase_effectiveness_comparison(
                    os.path.join(output_dir, "phase_effectiveness.png")
                ),
            ),
            (
                "Resonance Field Visualization",
                lambda: self.generate_resonance_field_visualization(
                    143, os.path.join(output_dir, "resonance_field_143.png")
                ),
            ),
            (
                "Algorithm Comparison",
                lambda: self.generate_algorithm_comparison_chart(
                    os.path.join(output_dir, "algorithm_comparison.png")
                ),
            ),
        ]

        results = {}

        for fig_name, fig_func in figures:
            try:
                print(f"\nGenerating {fig_name}...")
                result = fig_func()
                results[fig_name] = result
                print(f"✓ {fig_name} completed")
            except Exception as e:
                print(f"✗ {fig_name} failed: {e}")
                import traceback

                traceback.print_exc()

        print(f"\nFigure generation completed. Files saved to {output_dir}/")
        return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate RFH3 Performance Figures")
    parser.add_argument(
        "--output-dir", default="figures", help="Output directory for figures"
    )
    parser.add_argument(
        "--figure",
        choices=[
            "success_rate",
            "phase_effectiveness",
            "resonance_field",
            "algorithm_comparison",
        ],
        help="Generate specific figure only",
    )
    parser.add_argument(
        "--resonance-n",
        type=int,
        default=143,
        help="Number to use for resonance field visualization",
    )

    args = parser.parse_args()

    generator = PerformanceFigureGenerator()

    if not generator.plotting_available:
        print("Error: Plotting libraries not available")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.figure:
        # Generate specific figure
        figure_map = {
            "success_rate": lambda: generator.generate_success_rate_by_bits(
                os.path.join(args.output_dir, "success_rate_by_bits.png")
            ),
            "phase_effectiveness": lambda: generator.generate_phase_effectiveness_comparison(
                os.path.join(args.output_dir, "phase_effectiveness.png")
            ),
            "resonance_field": lambda: generator.generate_resonance_field_visualization(
                args.resonance_n,
                os.path.join(
                    args.output_dir, f"resonance_field_{args.resonance_n}.png"
                ),
            ),
            "algorithm_comparison": lambda: generator.generate_algorithm_comparison_chart(
                os.path.join(args.output_dir, "algorithm_comparison.png")
            ),
        }

        if args.figure in figure_map:
            figure_map[args.figure]()
        else:
            print(f"Unknown figure: {args.figure}")
    else:
        # Generate all figures
        generator.generate_all_figures(args.output_dir)


if __name__ == "__main__":
    main()
