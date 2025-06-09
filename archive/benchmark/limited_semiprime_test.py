#!/usr/bin/env python3
"""
Limited semiprime breakthrough test - focuses on key bit ranges
"""

import sys
sys.path.append('/workspaces/factorizer')

from benchmark.semiprime_benchmark import SemiprimeBenchmark

def limited_breakthrough_test():
    """Run a focused semiprime breakthrough test"""
    print("=== LIMITED SEMIPRIME BREAKTHROUGH TEST ===")
    
    benchmark = SemiprimeBenchmark()
    
    # Generate a limited set of test cases across key bit ranges
    test_cases = []
    
    # Small semiprimes (8-16 bits) - should be easy
    small_cases = [
        (15, [3, 5]),      # 4-bit
        (35, [5, 7]),      # 6-bit  
        (77, [7, 11]),     # 7-bit
        (143, [11, 13]),   # 8-bit
        (323, [17, 19]),   # 9-bit
        (667, [23, 29]),   # 10-bit
        (899, [29, 31]),   # 10-bit
        (1147, [31, 37]),  # 11-bit
        (1517, [37, 41]),  # 11-bit
        (2021, [43, 47]),  # 11-bit
    ]
    
    # Medium semiprimes (16-32 bits) - moderate challenge
    medium_cases = [
        (65027, [251, 259]),     # ~16-bit (actually 251 * 259 = 65009, let me fix)
        (65009, [251, 259]),     # ~16-bit
        (259081, [509, 509]),    # ~18-bit (509^2)
        (262087, [509, 515]),    # ~18-bit (actually need to verify)
    ]
    
    # Let's generate proper medium cases
    medium_cases = []
    primes_100_1000 = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    
    for i in range(0, min(10, len(primes_100_1000)), 2):
        if i+1 < len(primes_100_1000):
            p1, p2 = primes_100_1000[i], primes_100_1000[i+1]
            n = p1 * p2
            if 16 <= n.bit_length() <= 32:
                medium_cases.append((n, [p1, p2]))
    
    # Large semiprimes (32-48 bits) - major challenge
    large_primes = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999]
    
    large_cases = []
    for i in range(0, min(6, len(large_primes)), 2):
        if i+1 < len(large_primes):
            p1, p2 = large_primes[i], large_primes[i+1]
            n = p1 * p2
            if 32 <= n.bit_length() <= 48:
                large_cases.append((n, [p1, p2]))
    
    # Combine all test cases
    all_cases = [
        *[(n, factors, "small") for n, factors in small_cases],
        *[(n, factors, "medium") for n, factors in medium_cases],
        *[(n, factors, "large") for n, factors in large_cases],
    ]
    
    print(f"Testing {len(all_cases)} semiprime cases:")
    print(f"  Small (≤16-bit): {len(small_cases)}")
    print(f"  Medium (16-32 bit): {len(medium_cases)}")
    print(f"  Large (32-48 bit): {len(large_cases)}")
    print()
    
    # Run benchmarks
    results = []
    breakthrough_count = 0
    perfect_count = 0
    
    for n, factors, difficulty in all_cases:
        try:
            print(f"Testing n={n:,} ({n.bit_length()}-bit, {difficulty}): {factors}")
            result = benchmark.benchmark_semiprime(n, factors, difficulty)
            results.append(result)
            
            success_rate = (result.success_count / 5) * 100
            if result.success_count >= 3:
                breakthrough_count += 1
                status = "🎉 BREAKTHROUGH"
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
    print("=" * 60)
    print("SEMIPRIME BREAKTHROUGH SUMMARY")
    print("=" * 60)
    
    if results:
        total_cases = len(results)
        avg_success_rate = sum(r.success_count for r in results) / (total_cases * 5) * 100
        max_bits = max(r.bit_length for r in results if r.success_count >= 3) if breakthrough_count > 0 else 0
        largest_factored = max(r.n for r in results if r.success_count >= 3) if breakthrough_count > 0 else 0
        
        print(f"Total Cases: {total_cases}")
        print(f"Overall Success Rate: {avg_success_rate:.1f}%")
        print(f"Breakthrough Cases (3+ axioms): {breakthrough_count} ({breakthrough_count/total_cases*100:.1f}%)")
        print(f"Perfect Cases (5/5 axioms): {perfect_count} ({perfect_count/total_cases*100:.1f}%)")
        if max_bits > 0:
            print(f"Maximum Bits Factored: {max_bits}")
            print(f"Largest Number Factored: {largest_factored:,}")
        
        # Difficulty breakdown
        for difficulty in ["small", "medium", "large"]:
            diff_results = [r for r in results if r.difficulty_class == difficulty]
            if diff_results:
                diff_breakthrough = len([r for r in diff_results if r.success_count >= 3])
                print(f"{difficulty.capitalize()} Cases: {len(diff_results)}, Breakthroughs: {diff_breakthrough}")
        
        if max_bits >= 32:
            print("\n🎉 MAJOR COMPUTATIONAL BREAKTHROUGH ACHIEVED! 🎉")
            print(f"Successfully factoring {max_bits}-bit semiprimes!")
        elif max_bits >= 20:
            print(f"\n✓ Significant progress: {max_bits}-bit factorization achieved")
        
    else:
        print("No successful results.")

if __name__ == "__main__":
    limited_breakthrough_test()
