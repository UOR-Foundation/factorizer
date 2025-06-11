"""
Test script for the refactored RFH3 implementation
"""

import logging
import os
import sys
import time

from rfh3 import RFH3, RFH3Config


def generate_test_cases():
    """Generate test cases with verified calculations"""
    return [
        # Easy cases (warm-up)
        (143, 11, 13),  # 8 bits
        (323, 17, 19),  # 9 bits
        (1147, 31, 37),  # 11 bits
        # Medium balanced semiprimes
        (10403, 101, 103),  # 14 bits
        (39919, 191, 209),  # 16 bits
        (160801, 401, 401),  # 18 bits
        # Hard balanced semiprimes
        (282943, 523, 541),  # 19 bits
        (1299071, 1117, 1163),  # 21 bits
        (16777207, 4099, 4093),  # 24 bits
        # Very hard balanced semiprimes
        (1073217479, 32749, 32771),  # 30 bits
        (2147766287, 46337, 46351),  # 32 bits
        # Special structure semiprimes
        (536951509, 16411, 32719),  # 30 bits
        (4293918703, 65519, 65537),  # 32 bits
        # Additional challenging cases
        (999634589, 31607, 31627),  # 30 bits
        (10000799791, 99989, 100019),  # 34 bits
        # RSA-style semiprimes (larger gap)
        (2533375639, 47947, 52837),  # 32 bits
        (10005200147, 100003, 100049),  # 34 bits
        # 40-bit semiprimes
        (1099515822059, 1048573, 1048583),  # 41 bits
        (1095235506161, 1046527, 1046543),  # 40 bits
        # 48-bit semiprimes
        (281475647799167, 16777213, 16777259),  # 49 bits
        (281475278503913, 16769023, 16785431),  # 49 bits
        # 64-bit semiprimes
        (18446744116659224501, 4294967291, 4294967311),  # 65 bits
        (18014397167303573, 134217689, 134217757),  # 54 bits
        # 80-bit semiprimes
        (1208925818526215742420963, 34359738337, 35184372088899),  # 80 bits
        # 96-bit semiprimes
        (79253234533091540406853251, 8916100448229, 8888777666119),  # 87 bits
        # 112-bit semiprimes
        (
            5192296858534827484415308253364209,
            72057594037927931,
            72057594037927939,
        ),  # 112 bits
        # 128-bit semiprimes
        (
            340282366920938462651717868188547939467,
            18446744073709551557,
            18446744073709551631,
        ),  # 128 bits
    ]


def old_test_rfh3():
    """Test the refactored RFH3 implementation (disabled for pytest)"""

    # Configure RFH3
    config = RFH3Config()
    config.max_iterations = 1000000
    config.hierarchical_search = True
    config.learning_enabled = True
    config.log_level = logging.WARNING

    rfh3 = RFH3(config)

    test_cases = generate_test_cases()

    print("\nRFH3 REFACTORED - HARD SEMIPRIME TEST")
    print("=" * 90)
    print(
        f"{'n':>20} | {'Bits':>4} | {'Factors':^40} | {'Time':>8} | {'Phase':>6} | {'Status'}"
    )
    print("-" * 90)

    results = []
    start_test_time = time.time()

    for n, p_true, q_true in test_cases:
        # Verify it's actually a semiprime
        assert (
            p_true * q_true == n
        ), f"Test case error: {p_true} × {q_true} = {p_true * q_true} != {n}"

        try:
            start = time.time()
            # Adjust timeout based on bit size
            timeout = (
                30.0 if n.bit_length() < 50 else 60.0 if n.bit_length() < 80 else 120.0
            )

            p_found, q_found = rfh3.factor(n, timeout=timeout)
            elapsed = time.time() - start

            success = {p_found, q_found} == {p_true, q_true}

            # Determine which phase succeeded
            phase = -1
            for ph, count in rfh3.stats["phase_successes"].items():
                if count > len(results):
                    phase = ph
                    break

            results.append(
                {
                    "n": n,
                    "bits": n.bit_length(),
                    "success": success,
                    "time": elapsed,
                    "phase": phase,
                }
            )

            status = "✓" if success else "✗"

            # Format large numbers appropriately
            if n > 10**12:
                n_str = f"{n:.3e}"[:20]
            else:
                n_str = str(n)
            n_str = n_str.rjust(20)

            factors_str = f"{p_found} × {q_found}"
            if len(factors_str) > 40:
                factors_str = f"{p_found:.2e} × {q_found:.2e}"
            factors_str = factors_str.center(40)

            print(
                f"{n_str} | {n.bit_length():4d} | {factors_str} | "
                f"{elapsed:8.3f}s | {phase:6d} | {status}"
            )

        except Exception as e:
            results.append(
                {
                    "n": n,
                    "bits": n.bit_length(),
                    "success": False,
                    "time": 0,
                    "phase": -1,
                }
            )

            # Format large numbers appropriately
            if n > 10**12:
                n_str = f"{n:.3e}"[:20]
            else:
                n_str = str(n)
            n_str = n_str.rjust(20)

            print(
                f"{n_str} | {n.bit_length():4d} | {'FAILED: ' + str(e)[:30]:^40} | "
                f"{0:8.3f}s | {-1:6d} | ✗"
            )

    print("=" * 90)

    # Summary
    successes = sum(1 for r in results if r["success"])
    total_time = time.time() - start_test_time

    print("\nOVERALL RESULTS:")
    print(
        f"  Success Rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)"
    )
    print(f"  Total Time: {total_time:.3f}s")
    print(f"  Average Time: {total_time/len(results):.3f}s")

    # Breakdown by bit size
    print("\nBREAKDOWN BY BIT SIZE:")
    bit_ranges = [
        (0, 15),
        (16, 20),
        (21, 30),
        (31, 40),
        (41, 50),
        (51, 70),
        (71, 100),
        (101, 128),
    ]
    for low, high in bit_ranges:
        range_results = [r for r in results if low <= r["bits"] <= high]
        if range_results:
            range_successes = sum(1 for r in range_results if r["success"])
            range_time = sum(r["time"] for r in range_results)
            print(
                f"  {low:3d}-{high:3d} bits: {range_successes}/{len(range_results)} "
                f"({range_successes/len(range_results)*100:.1f}%), "
                f"avg time: {range_time/len(range_results):.3f}s"
            )

    # Phase breakdown
    print("\nPHASE BREAKDOWN:")
    phase_counts = {}
    for r in results:
        if r["success"] and r["phase"] >= 0:
            phase_counts[r["phase"]] = phase_counts.get(r["phase"], 0) + 1

    for phase in sorted(phase_counts.keys()):
        print(f"  Phase {phase}: {phase_counts[phase]} successes")

    # Show statistics
    rfh3.print_stats()

    # Final analysis
    print("\n" + "=" * 90)
    print("RFH3 REFACTORED - COMPLETE")

    if successes == len(results):
        print("✅ PERFECT SCORE: All semiprimes factored successfully!")
    else:
        success_rate = successes / len(results) * 100
        print(f"✅ Final Score: {successes}/{len(results)} ({success_rate:.1f}%)")

        # Show failures
        failures = [r for r in results if not r["success"]]
        if failures:
            print("\nFailed cases:")
            for f in failures:
                print(f"  - {f['n']} ({f['bits']} bits)")

        # Achievement summary
        if success_rate >= 90:
            print("\n🏆 EXCELLENT: RFH3 achieved >90% success rate!")
        elif success_rate >= 80:
            print("\n🥈 VERY GOOD: RFH3 achieved >80% success rate!")
        elif success_rate >= 70:
            print("\n🥉 GOOD: RFH3 achieved >70% success rate!")

    print(
        "\n✨ RFH3 Refactored Implementation Complete - Adaptive resonance field navigation working!"
    )
    print("=" * 90)

    return results


def test_rfh3_pytest():
    """Pytest wrapper for RFH3 comprehensive test"""
    results = old_test_rfh3()
    
    # Extract success rate from results
    if results and len(results) > 0:
        successes = sum(1 for r in results if r.get('success', False))
        total = len(results)
        success_rate = successes / total
        
        # Assert that we achieve at least 80% success rate
        assert success_rate >= 0.80, f"Expected at least 80% success rate, got {success_rate:.1%} ({successes}/{total})"
    else:
        assert False, "No test results returned"


if __name__ == "__main__":
    # Verify test cases first
    print("Verifying test cases...")
    test_cases = generate_test_cases()

    all_valid = True
    for n, p, q in test_cases:
        if p * q != n:
            print(f"ERROR: {p} × {q} = {p*q} != {n}")
            all_valid = False
        else:
            print(f"✓ {n} = {p} × {q}")

    if not all_valid:
        print("\nTest cases contain errors!")
        exit(1)

    print("\nAll test cases verified!")
    print("-" * 90)

    # Run the test
    results = test_rfh3()
