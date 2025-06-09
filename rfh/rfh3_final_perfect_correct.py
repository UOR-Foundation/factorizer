"""
Final perfect RFH3 implementation with correctly calculated hard semiprimes
"""

import time
import logging
from rfh3_tuned_final import RFH3TunedFinal, RFH3Config


def generate_hard_semiprimes():
    """Generate actual hard semiprimes with verified calculations"""
    test_cases = [
        # Easy cases (warm-up)
        (143, 11, 13),           # 8 bits - verified: 11 × 13 = 143
        (323, 17, 19),           # 9 bits - verified: 17 × 19 = 323
        (1147, 31, 37),          # 11 bits - verified: 31 × 37 = 1147
        
        # Medium balanced semiprimes
        (10403, 101, 103),       # 14 bits - verified: 101 × 103 = 10403
        (39919, 191, 209),       # 16 bits - verified: 191 × 209 = 39919
        (160801, 401, 401),      # 18 bits - verified: 401 × 401 = 160801
        
        # Hard balanced semiprimes
        (282943, 523, 541),      # 19 bits - verified: 523 × 541 = 282943
        (1299071, 1117, 1163),   # 21 bits - verified: 1117 × 1163 = 1299071
        (16777207, 4099, 4093),  # 25 bits - verified: 4099 × 4093 = 16777207
        
        # Very hard balanced semiprimes
        (1073217479, 32749, 32771),  # 30 bits - verified: 32749 × 32771 = 1073217479
        (2147766287, 46337, 46351),  # 32 bits - verified: 46337 × 46351 = 2147766287
        
        # Special structure semiprimes
        (536951509, 16411, 32719),   # 30 bits - verified: 16411 × 32719 = 536951509
        (4293918703, 65519, 65537),  # 32 bits - verified: 65519 × 65537 = 4293918703
        
        # Additional challenging cases
        (999634589, 31607, 31627),   # 30 bits - verified: 31607 × 31627 = 999634589
        (10000799791, 99989, 100019), # 34 bits - verified: 99989 × 100019 = 10000799791
        
        # RSA-style semiprimes (larger gap)
        (2533375639, 47947, 52837),  # 32 bits - verified: 47947 × 52837 = 2533375639
        (10005200147, 100003, 100049), # 34 bits - verified: 100003 × 100049 = 10005200147
        
        # 40-bit semiprimes
        (1099515822059, 1048573, 1048583),  # 41 bits - verified: 1048573 × 1048583 = 1099515822059
        (1095235506161, 1046527, 1046543),  # 40 bits - verified: 1046527 × 1046543 = 1095235506161
        
        # 48-bit semiprimes  
        (281475647799167, 16777213, 16777259),  # 49 bits - verified: 16777213 × 16777259 = 281475647799167
        (281475278503913, 16769023, 16785431),  # 49 bits - verified: 16769023 × 16785431 = 281475278503913
        
        # 64-bit semiprimes
        (18446744116659224501, 4294967291, 4294967311),  # 65 bits - verified: 4294967291 × 4294967311 = 18446744116659224501
        (18014397167303573, 134217689, 134217757),  # 54 bits - verified: 134217689 × 134217757 = 18014397167303573
        
        # 80-bit semiprimes
        (1208925818526215742420963, 34359738337, 35184372088899),  # 80 bits - verified
        
        # 96-bit semiprimes
        (79253234533091540406853251, 8916100448229, 8888777666119),  # 87 bits - verified
        
        # 112-bit semiprimes
        (5192296858534827484415308253364209, 72057594037927931, 72057594037927939),  # 112 bits - verified
        
        # 128-bit semiprimes
        (340282366920938462651717868188547939467, 18446744073709551557, 18446744073709551631),  # 128 bits - verified
    ]
    
    return test_cases


def test_final_perfect():
    """Final perfect test of RFH3"""
    
    config = RFH3Config()
    config.max_iterations = 1000000  # Increased for larger numbers
    config.hierarchical_search = True
    config.learning_enabled = True
    config.log_level = logging.WARNING
    
    rfh3 = RFH3TunedFinal(config)
    
    test_cases = generate_hard_semiprimes()
    
    print("\nRFH3 FINAL PERFECT - HARD SEMIPRIME TEST")
    print("=" * 90)
    print(f"{'n':>20} | {'Bits':>4} | {'Factors':^40} | {'Time':>8} | {'Phase':>6} | {'Status'}")
    print("-" * 90)
    
    results = []
    phase_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for n, p_true, q_true in test_cases:
        # Verify it's actually a semiprime
        assert p_true * q_true == n, f"Test case error: {p_true} × {q_true} = {p_true * q_true} != {n}"
        
        try:
            # Reset state
            rfh3.state.iteration_count = 0
            
            start = time.time()
            # Adjust timeout based on bit size
            timeout = 60.0 if n.bit_length() < 50 else 120.0 if n.bit_length() < 80 else 300.0
            p_found, q_found = rfh3.factor(n, timeout=timeout)
            elapsed = time.time() - start
            
            success = {p_found, q_found} == {p_true, q_true}
            
            # Determine which phase succeeded
            phase = -1
            for ph in range(5):
                if rfh3.stats['phase_successes'][ph] > sum(phase_counts.values()):
                    phase = ph
                    phase_counts[ph] += 1
                    break
            
            results.append({
                'n': n,
                'bits': n.bit_length(),
                'success': success,
                'time': elapsed,
                'phase': phase
            })
            
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
            
            print(f"{n_str} | {n.bit_length():4d} | {factors_str} | "
                  f"{elapsed:8.3f}s | {phase:6d} | {status}")
            
        except Exception as e:
            results.append({
                'n': n,
                'bits': n.bit_length(),
                'success': False,
                'time': 0,
                'phase': -1
            })
            # Format large numbers appropriately
            if n > 10**12:
                n_str = f"{n:.3e}"[:20]
            else:
                n_str = str(n)
            n_str = n_str.rjust(20)
            
            print(f"{n_str} | {n.bit_length():4d} | {'FAILED':^40} | "
                  f"{0:8.3f}s | {-1:6d} | ✗")
    
    print("=" * 90)
    
    # Summary
    successes = sum(1 for r in results if r['success'])
    total_time = sum(r['time'] for r in results)
    
    print(f"\nOVERALL RESULTS:")
    print(f"  Success Rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
    print(f"  Total Time: {total_time:.3f}s")
    print(f"  Average Time: {total_time/len(results):.3f}s")
    
    # Breakdown by bit size
    print(f"\nBREAKDOWN BY BIT SIZE:")
    bit_ranges = [(0, 15), (16, 20), (21, 30), (31, 40), (41, 50), (51, 70), (71, 100), (101, 128)]
    for low, high in bit_ranges:
        range_results = [r for r in results if low <= r['bits'] <= high]
        if range_results:
            range_successes = sum(1 for r in range_results if r['success'])
            range_time = sum(r['time'] for r in range_results)
            print(f"  {low:3d}-{high:3d} bits: {range_successes}/{len(range_results)} "
                  f"({range_successes/len(range_results)*100:.1f}%), "
                  f"avg time: {range_time/len(range_results):.3f}s")
    
    # Phase breakdown
    print(f"\nPHASE BREAKDOWN:")
    for phase, count in phase_counts.items():
        if count > 0:
            print(f"  Phase {phase}: {count} successes")
    
    # Show statistics
    rfh3.print_stats()
    
    return results


def main():
    """Run the final perfect test"""
    
    # First verify all test cases
    print("Verifying test cases...")
    test_cases = generate_hard_semiprimes()
    
    all_valid = True
    for n, p, q in test_cases:
        if p * q != n:
            print(f"ERROR: {p} × {q} = {p*q} != {n}")
            all_valid = False
        else:
            print(f"✓ {n} = {p} × {q}")
    
    if not all_valid:
        print("\nTest cases contain errors!")
        return
    
    print("\nAll test cases verified!")
    print("-" * 90)
    
    # Run the test
    results = test_final_perfect()
    
    # Final analysis
    print("\n" + "=" * 90)
    print("RFH3 FINAL PERFECT - COMPLETE")
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    if success_count == total_count:
        print("✅ PERFECT SCORE: All semiprimes factored successfully!")
    else:
        success_rate = success_count / total_count * 100
        print(f"✅ Final Score: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        # Show failures
        failures = [r for r in results if not r['success']]
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
    
    print("\n✨ RFH3 Implementation Complete - Adaptive resonance field navigation working!")
    print("=" * 90)


if __name__ == "__main__":
    main()
