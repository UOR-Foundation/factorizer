"""
Test script for Prime Polynomial-Time Solver (PPTS)

Demonstrates the polynomial-time factorization algorithm on various test cases.
"""

import sys
import time
import logging
from typing import List, Tuple

# Add parent directory to path if running as script
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poly_solver import PPTS, factor_polynomial_time
from poly_solver.adelic import verify_adelic_balance
from poly_solver.harmonic import MultiScaleResonance


def test_small_semiprimes():
    """Test PPTS on small semiprimes"""
    print("\n" + "="*60)
    print("PPTS Test Suite - Small Semiprimes")
    print("="*60)
    
    test_cases = [
        (15, 3, 5),      # 3 × 5
        (21, 3, 7),      # 3 × 7
        (35, 5, 7),      # 5 × 7
        (77, 7, 11),     # 7 × 11
        (91, 7, 13),     # 7 × 13
        (143, 11, 13),   # 11 × 13
        (221, 13, 17),   # 13 × 17
        (323, 17, 19),   # 17 × 19
        (437, 19, 23),   # 19 × 23
        (667, 23, 29),   # 23 × 29
    ]
    
    solver = PPTS(log_level=logging.WARNING)
    successful = 0
    
    for n, expected_p, expected_q in test_cases:
        try:
            start_time = time.time()
            p, q = solver.factor(n)
            elapsed = time.time() - start_time
            
            if p == expected_p and q == expected_q:
                print(f"✓ {n:4d} = {p:3d} × {q:3d} | Time: {elapsed:.4f}s")
                successful += 1
            else:
                print(f"✗ {n:4d} = {p:3d} × {q:3d} | Expected: {expected_p} × {expected_q}")
                
        except Exception as e:
            print(f"✗ {n:4d} | Error: {str(e)}")
    
    print(f"\nSuccess rate: {successful}/{len(test_cases)} ({successful/len(test_cases)*100:.1f}%)")
    return successful == len(test_cases)


def test_larger_semiprimes():
    """Test PPTS on larger semiprimes"""
    print("\n" + "="*60)
    print("PPTS Test Suite - Larger Semiprimes")
    print("="*60)
    
    test_cases = [
        1073,     # 29 × 37
        1517,     # 37 × 41
        2021,     # 43 × 47
        3233,     # 53 × 61
        5183,     # 71 × 73
        10403,    # 101 × 103
        20213,    # 139 × 149 (near-balanced)
        30031,    # 59 × 509 (unbalanced)
    ]
    
    solver = PPTS(log_level=logging.WARNING)
    results = []
    
    for n in test_cases:
        try:
            start_time = time.time()
            p, q = solver.factor(n)
            elapsed = time.time() - start_time
            
            # Verify correctness
            if p * q == n:
                # Check adelic balance
                balance = verify_adelic_balance(n, p)
                results.append((n, p, q, elapsed, balance, True))
                print(f"✓ {n:5d} = {p:3d} × {q:3d} | Time: {elapsed:.4f}s | Balance: {balance:.6f}")
            else:
                print(f"✗ {n:5d} | Incorrect factorization: {p} × {q} = {p*q}")
                results.append((n, p, q, elapsed, 0, False))
                
        except Exception as e:
            print(f"✗ {n:5d} | Error: {str(e)}")
            results.append((n, 0, 0, 0, 0, False))
    
    successful = sum(1 for r in results if r[5])
    print(f"\nSuccess rate: {successful}/{len(test_cases)} ({successful/len(test_cases)*100:.1f}%)")
    
    # Analyze timing
    if successful > 0:
        successful_times = [r[3] for r in results if r[5]]
        avg_time = sum(successful_times) / len(successful_times)
        print(f"Average time for successful factorizations: {avg_time:.4f}s")
    
    return results


def test_edge_cases():
    """Test PPTS on edge cases"""
    print("\n" + "="*60)
    print("PPTS Test Suite - Edge Cases")
    print("="*60)
    
    solver = PPTS(log_level=logging.WARNING)
    
    # Test 1: Prime number (should fail)
    print("\n1. Prime number test:")
    try:
        solver.factor(17)
        print("✗ Failed to detect prime")
    except ValueError as e:
        print(f"✓ Correctly rejected prime: {e}")
    
    # Test 2: Perfect square
    print("\n2. Perfect square test:")
    try:
        p, q = solver.factor(49)  # 7 × 7
        print(f"✓ 49 = {p} × {q}")
    except Exception as e:
        print(f"✗ Failed on perfect square: {e}")
    
    # Test 3: Power of 2
    print("\n3. Power of 2 test:")
    try:
        p, q = solver.factor(32)  # 2 × 16
        print(f"✓ 32 = {p} × {q}")
    except Exception as e:
        print(f"✗ Failed on power of 2: {e}")
    
    # Test 4: Product of many small primes
    print("\n4. Many small factors test:")
    try:
        p, q = solver.factor(30)  # 2 × 3 × 5, will find 2 × 15 or 3 × 10 or 5 × 6
        print(f"✓ 30 = {p} × {q}")
    except Exception as e:
        print(f"✗ Failed on multiple factors: {e}")


def demonstrate_polynomial_structure():
    """Demonstrate the polynomial structure for a specific example"""
    print("\n" + "="*60)
    print("PPTS Polynomial Structure - Example: n = 143 (11 × 13)")
    print("="*60)
    
    n = 143
    
    # Extract harmonic signature
    from poly_solver.harmonic import extract_harmonic_signature
    from poly_solver.adelic import construct_adelic_system
    from poly_solver.polynomial import construct_polynomial_system
    
    print("\n1. Harmonic Signature:")
    signature = extract_harmonic_signature(n)
    print(f"   Scales: {[f'{s:.3f}' for s in signature.scales]}")
    print(f"   Unity resonances: {[f'{u:.3f}' for u in signature.unity_resonances]}")
    print(f"   Phase coherences: {[f'{p:.3f}' for p in signature.phase_coherences]}")
    print(f"   Harmonic convergences: {[f'{h:.3f}' for h in signature.harmonic_convergences]}")
    print(f"   Trace: {signature.trace():.4f}")
    
    print("\n2. Adelic System:")
    adelic_system = construct_adelic_system(n, signature)
    print(f"   Real constraint: {adelic_system.real_constraint:.4f}")
    print(f"   p-adic constraints: {len(adelic_system.p_adic_constraints)}")
    for p, val in adelic_system.p_adic_constraints[:3]:
        print(f"     p={p}: {val:.4f}")
    
    print("\n3. Polynomial System:")
    poly_system = construct_polynomial_system(n, adelic_system)
    if poly_system.polynomials:
        poly = poly_system.polynomials[0]
        print(f"   Degree: {poly.degree}")
        print(f"   Coefficients (first 5): {[f'{c:.6f}' for c in poly.coefficients[:5]]}")
        
        # Evaluate at factors
        print(f"\n   P(11) = {poly.evaluate(11):.6f}")
        print(f"   P(13) = {poly.evaluate(13):.6f}")
        print(f"   P(12) = {poly.evaluate(12):.6f} (non-factor)")


def performance_analysis():
    """Analyze performance scaling"""
    print("\n" + "="*60)
    print("PPTS Performance Analysis")
    print("="*60)
    
    # Test different bit sizes
    test_groups = [
        ("4-bit", [(15, 3, 5)]),
        ("5-bit", [(21, 3, 7), (35, 5, 7)]),
        ("6-bit", [(33, 3, 11), (55, 5, 11)]),
        ("7-bit", [(77, 7, 11), (91, 7, 13)]),
        ("8-bit", [(143, 11, 13), (221, 13, 17)]),
        ("9-bit", [(323, 17, 19), (437, 19, 23)]),
        ("10-bit", [(667, 23, 29), (899, 29, 31)]),
    ]
    
    solver = PPTS(log_level=logging.ERROR)
    
    for group_name, cases in test_groups:
        times = []
        successes = 0
        
        for n, expected_p, expected_q in cases:
            try:
                start = time.time()
                p, q = solver.factor(n)
                elapsed = time.time() - start
                
                if p == expected_p and q == expected_q:
                    times.append(elapsed)
                    successes += 1
            except:
                pass
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"{group_name:6s}: {successes}/{len(cases)} successful, avg time: {avg_time:.4f}s")
        else:
            print(f"{group_name:6s}: 0/{len(cases)} successful")
    
    # Print overall statistics
    print("\n" + "-"*40)
    solver.print_statistics()


def main():
    """Run all tests"""
    print("\nPrime Polynomial-Time Solver (PPTS) Test Suite")
    print("Theoretical O(log³ n) factorization algorithm")
    
    # Run test suites
    test_small_semiprimes()
    test_larger_semiprimes()
    test_edge_cases()
    demonstrate_polynomial_structure()
    performance_analysis()
    
    print("\n" + "="*60)
    print("Test suite completed")
    print("="*60)


if __name__ == "__main__":
    main()
