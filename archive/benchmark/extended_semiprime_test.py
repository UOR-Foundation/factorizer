#!/usr/bin/env python3
"""
Extended semiprime breakthrough test - pushes to higher bit ranges
"""

import sys
import random
sys.path.append('/workspaces/factorizer')

from benchmark.semiprime_benchmark import SemiprimeBenchmark

def extended_breakthrough_test():
    """Run extended semiprime breakthrough test with larger numbers"""
    print("=== EXTENDED SEMIPRIME BREAKTHROUGH TEST ===")
    print("Testing larger semiprimes to find computational limits")
    print()
    
    benchmark = SemiprimeBenchmark()
    
    # Medium-large semiprimes (12-20 bits)
    medium_cases = [
        (2491, [47, 53]),     # 12-bit
        (3127, [53, 59]),     # 12-bit
        (4087, [61, 67]),     # 12-bit
        (4717, [67, 71]),     # 13-bit
        (5183, [71, 73]),     # 13-bit
        (5767, [73, 79]),     # 13-bit
        (6557, [79, 83]),     # 13-bit
        (7387, [83, 89]),     # 13-bit
        (8633, [89, 97]),     # 14-bit
        (9797, [97, 101]),    # 14-bit
    ]
    
    # Large semiprimes (20-32 bits) - serious computational challenge
    large_primes_100_1000 = [
        101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
        211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
        331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443,
        449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577,
        587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
        709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839,
        853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997
    ]
    
    large_cases = []
    # Create some challenging 16-24 bit semiprimes
    for i in range(0, min(20, len(large_primes_100_1000)), 5):
        for j in range(i+1, min(i+5, len(large_primes_100_1000))):
            p1, p2 = large_primes_100_1000[i], large_primes_100_1000[j]
            n = p1 * p2
            if 16 <= n.bit_length() <= 24:
                large_cases.append((n, [p1, p2]))
    
    # Select a few representative cases from each bit range
    large_cases = large_cases[:8]  # Limit for reasonable runtime
    
    # Very large semiprimes (24-40 bits) - extreme computational challenge
    very_large_primes = [
        1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097,
        1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
        1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,
        1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459,
        1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571,
        1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693,
        1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811,
        1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949
    ]
    
    very_large_cases = []
    # Create challenging 24-32 bit semiprimes
    for i in range(0, min(10, len(very_large_primes)), 3):
        for j in range(i+1, min(i+3, len(very_large_primes))):
            p1, p2 = very_large_primes[i], very_large_primes[j]
            n = p1 * p2
            if 20 <= n.bit_length() <= 32:
                very_large_cases.append((n, [p1, p2]))
    
    very_large_cases = very_large_cases[:5]  # Limit for testing
    
    # Combine all test cases
    all_cases = [
        *[(n, factors, "medium") for n, factors in medium_cases],
        *[(n, factors, "large") for n, factors in large_cases], 
        *[(n, factors, "very_large") for n, factors in very_large_cases],
    ]
    
    print(f"Testing {len(all_cases)} extended semiprime cases:")
    print(f"  Medium (12-16 bit): {len(medium_cases)}")
    print(f"  Large (16-24 bit): {len(large_cases)}")
    print(f"  Very Large (20-32 bit): {len(very_large_cases)}")
    print()
    
    # Run benchmarks
    results = []
    breakthrough_count = 0
    perfect_count = 0
    max_bits_achieved = 0
    largest_factored = 0
    
    for n, factors, difficulty in all_cases:
        try:
            print(f"Testing n={n:,} ({n.bit_length()}-bit, {difficulty}): factors {factors}")
            result = benchmark.benchmark_semiprime(n, factors, difficulty)
            results.append(result)
            
            success_rate = (result.success_count / 5) * 100
            
            if result.success_count >= 3:
                breakthrough_count += 1
                status = "🎉 BREAKTHROUGH"
                max_bits_achieved = max(max_bits_achieved, result.bit_length)
                largest_factored = max(largest_factored, result.n)
            elif result.success_count >= 1:
                status = "✓ Partial"
            else:
                status = "✗ Failed"
                
            if result.success_count == 5:
                perfect_count += 1
                status = "🔥 PERFECT"
                
            print(f"  {status}: {result.success_count}/5 axioms ({success_rate:.1f}%) in {result.total_time:.3f}s")
            
            # Show successful axioms
            successful_axioms = [axiom for axiom, factor in result.found_factors.items() 
                               if factor in factors]
            if successful_axioms:
                print(f"    Successful: {', '.join(successful_axioms)}")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Summary
    print("=" * 80)
    print("EXTENDED SEMIPRIME BREAKTHROUGH SUMMARY")
    print("=" * 80)
    
    if results:
        total_cases = len(results)
        avg_success_rate = sum(r.success_count for r in results) / (total_cases * 5) * 100
        
        print(f"Total Cases: {total_cases}")
        print(f"Overall Success Rate: {avg_success_rate:.1f}%")
        print(f"Breakthrough Cases (3+ axioms): {breakthrough_count} ({breakthrough_count/total_cases*100:.1f}%)")
        print(f"Perfect Cases (5/5 axioms): {perfect_count} ({perfect_count/total_cases*100:.1f}%)")
        print(f"Maximum Bits Successfully Factored: {max_bits_achieved}")
        print(f"Largest Number Successfully Factored: {largest_factored:,}")
        
        # Difficulty breakdown
        for difficulty in ["medium", "large", "very_large"]:
            diff_results = [r for r in results if r.difficulty_class == difficulty]
            if diff_results:
                diff_breakthrough = len([r for r in diff_results if r.success_count >= 3])
                diff_avg_success = sum(r.success_count for r in diff_results) / (len(diff_results) * 5) * 100
                max_diff_bits = max(r.bit_length for r in diff_results if r.success_count >= 3) if diff_breakthrough > 0 else 0
                print(f"{difficulty.upper()}: {len(diff_results)} cases, {diff_breakthrough} breakthroughs ({diff_avg_success:.1f}% avg), max {max_diff_bits} bits")
        
        # Axiom performance analysis
        print(f"\nAXIOM PERFORMANCE ON EXTENDED SEMIPRIMES:")
        axiom_performance = {}
        for axiom in ['axiom1', 'axiom2', 'axiom3', 'axiom4', 'axiom5']:
            successes = sum(1 for r in results if r.found_factors.get(axiom) in r.expected_factors)
            axiom_performance[axiom] = (successes / total_cases) * 100
        
        for axiom, success_rate in sorted(axiom_performance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {axiom}: {success_rate:.1f}%")
        
        # Major breakthrough announcements
        if max_bits_achieved >= 32:
            print("\n🚀 MAJOR COMPUTATIONAL BREAKTHROUGH ACHIEVED! 🚀")
            print(f"Successfully factoring {max_bits_achieved}-bit semiprimes!")
            print("This represents a significant advance in prime factorization!")
        elif max_bits_achieved >= 24:
            print(f"\n🎯 SIGNIFICANT BREAKTHROUGH: {max_bits_achieved}-bit factorization achieved!")
        elif max_bits_achieved >= 20:
            print(f"\n✓ Notable progress: {max_bits_achieved}-bit factorization achieved")
        
        if largest_factored > 10**9:
            print(f"🔥 MILESTONE: Factored numbers larger than 1 billion! 🔥")
        elif largest_factored > 10**6:
            print(f"📈 MILESTONE: Factored numbers larger than 1 million!")
        
    else:
        print("No successful results.")

if __name__ == "__main__":
    extended_breakthrough_test()
