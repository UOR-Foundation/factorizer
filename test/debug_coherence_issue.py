#!/usr/bin/env python3
# Debug script to identify the coherence computation issue

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultra_accelerated_uor_factorizer import MultiScaleObserver, is_prime

# Test the semiprime that likely caused the hang
test_semiprime = 1189  # 29 × 41

print(f"Testing MultiScaleObserver coherence for {test_semiprime}")
print(f"Factors: 29 × 41")
print(f"Square root: {int(test_semiprime**0.5)}")

# Create observer
obs = MultiScaleObserver(test_semiprime)

print(f"\nObserver scales: {obs.scales}")
print(f"Root: {obs.root}")

# Test coherence computation at various positions
test_positions = [2, 5, 10, 15, 20, 25, 29, 30]

for x in test_positions:
    print(f"\nTesting coherence at position {x}...")
    start_time = time.perf_counter()
    
    try:
        coherence_value = obs.coherence(x)
        elapsed = time.perf_counter() - start_time
        print(f"  Coherence({x}) = {coherence_value:.6f} [{elapsed*1000:.2f}ms]")
    except Exception as e:
        print(f"  ERROR at position {x}: {e}")
        import traceback
        traceback.print_exc()
        break

# Also test larger semiprimes that follow
larger_tests = [
    (991 * 997, "991 × 997"),
    (1009 * 1013, "1009 × 1013")
]

for semiprime, desc in larger_tests:
    print(f"\n\nTesting {desc} = {semiprime}")
    obs = MultiScaleObserver(semiprime)
    print(f"Scales: {obs.scales}")
    
    # Test just one position
    x = 2
    start_time = time.perf_counter()
    try:
        coherence_value = obs.coherence(x)
        elapsed = time.perf_counter() - start_time
        print(f"Coherence(2) = {coherence_value:.6f} [{elapsed*1000:.2f}ms]")
    except Exception as e:
        print(f"ERROR: {e}")
