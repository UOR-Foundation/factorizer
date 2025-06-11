"""
Specific debug for 1299709 = 1117 × 1163
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prime_sieve import PrimeSieve
from prime_sieve.core.coherence_engine import CoherenceEngine


def debug_specific_case():
    n = 1299709
    p, q = 1117, 1163
    
    print(f"Debugging {n} = {p} × {q}")
    print(f"sqrt({n}) = {n**0.5:.2f}")
    print(f"p = {p}, q = {q}")
    print()
    
    # Initialize coherence engine
    coherence_engine = CoherenceEngine(n)
    
    # Test direct division
    print("Direct division test:")
    print(f"  {n} % {p} = {n % p} (should be 0)")
    print(f"  {n} / {p} = {n // p} (should be {q})")
    print(f"  {p} * {q} = {p * q} (should be {n})")
    print(f"  Verification: {p * q == n}")
    print()
    
    # Test coherence calculation
    print("Coherence calculation:")
    coh_pq = coherence_engine.calculate_coherence(p, q)
    print(f"  coherence({p}, {q}) = {coh_pq:.6f}")
    
    # Test candidate scoring
    sqrt_n = int(n**0.5)
    print(f"\nCandidate scoring (sqrt_n = {sqrt_n}):")
    
    # Score for p
    distance_p = abs(p - sqrt_n) / sqrt_n
    distance_score_p = 1.0 / (1.0 + distance_p)
    coherence_score_p = coherence_engine.calculate_coherence(p, n // p) if n % p == 0 else 0
    combined_score_p = distance_score_p * 2.0 + coherence_score_p
    
    print(f"  Factor {p}:")
    print(f"    Distance from sqrt: {abs(p - sqrt_n)}")
    print(f"    Distance score: {distance_score_p:.6f}")
    print(f"    Coherence score: {coherence_score_p:.6f}")
    print(f"    Combined score: {combined_score_p:.6f}")
    
    # Score for some other candidates
    print("\n  Other candidates for comparison:")
    for x in [37, 100, 500, 1000, 1100, 1140]:
        distance_x = abs(x - sqrt_n) / sqrt_n
        distance_score_x = 1.0 / (1.0 + distance_x)
        if n % x == 0:
            coherence_score_x = coherence_engine.calculate_coherence(x, n // x)
        else:
            coherence_score_x = coherence_engine.calculate_coherence(x, x) * 0.5
        combined_score_x = distance_score_x * 2.0 + coherence_score_x
        
        print(f"    x = {x}: distance_score={distance_score_x:.3f}, "
              f"coherence={coherence_score_x:.3f}, combined={combined_score_x:.3f}")
    
    # Try factoring with the sieve
    print("\nTrying Prime Sieve factorization:")
    sieve = PrimeSieve(enable_learning=False)
    result = sieve.factor_with_details(n)
    
    print(f"  Result: {result.factors[0]} × {result.factors[1]}")
    print(f"  Method: {result.method}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Candidates tested: {result.candidates_tested}")
    
    # Check if the issue is in the checking phase
    print("\nManual check of division:")
    for test_val in [1117, 1116, 1118]:
        print(f"  {n} % {test_val} = {n % test_val}")
        if n % test_val == 0:
            print(f"    -> Factor found! {test_val} × {n // test_val} = {n}")


if __name__ == "__main__":
    debug_specific_case()
