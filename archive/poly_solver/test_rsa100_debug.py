#!/usr/bin/env python3
"""Debug RSA-100 factorization"""

from pattern_generator_imports import PatternFactorizer, FactorPattern
import time

n = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139

# Patch the materialization to add debug output
original_materialize = FactorPattern._materialize_balanced_factors

def debug_materialize(self):
    print(f"\n[DEBUG] Starting materialization for n={self.n}")
    n = self.n
    
    pattern_positions = sorted(set(pos for pos in self.factor_positions if pos > 0 and pos < n))
    print(f"[DEBUG] Found {len(pattern_positions)} pattern positions")
    
    if not pattern_positions:
        return []
    
    sqrt_n = int(n ** 0.5)
    
    # Check positions suggested by The Pattern
    for i, pos in enumerate(pattern_positions[:3]):
        if abs(pos - sqrt_n) < sqrt_n * 0.1:
            print(f"\n[DEBUG] Checking position {i+1}: {pos}")
            b_squared = pos * pos - n
            if b_squared > 0:
                b = int(b_squared ** 0.5)
                if b * b == b_squared:
                    print(f"[DEBUG] Found perfect square at first check!")
                    p = pos + b
                    q = pos - b
                    if p > 1 and q > 1 and p * q == n:
                        return [q, p] if q < p else [p, q]
    
    # Prioritize positions
    expected_offset_min = int(sqrt_n * 0.0001)
    expected_offset_max = int(sqrt_n * 0.01)
    
    prioritized_positions = []
    for pos in pattern_positions:
        offset = pos - sqrt_n
        if expected_offset_min <= offset <= expected_offset_max:
            prioritized_positions.append((abs(offset - sqrt_n * 0.0004), pos))
    
    prioritized_positions.sort()
    print(f"\n[DEBUG] Prioritized {len(prioritized_positions)} positions")
    
    # Try top position
    if prioritized_positions:
        _, base_pos = prioritized_positions[0]
        offset = base_pos - sqrt_n
        error_estimate = int(offset * 0.0000367)
        center = base_pos - error_estimate
        radius = max(1000, error_estimate // 2)
        
        print(f"\n[DEBUG] Best position:")
        print(f"  Base pos offset: {offset}")
        print(f"  Error estimate: {error_estimate}")
        print(f"  Center: {center}")
        print(f"  Radius: {radius}")
        print(f"  Search range: [{center - radius}, {center + radius}]")
        
        if n.bit_length() > 200:
            print(f"  Using binary search (n has {n.bit_length()} bits)")
        
    return original_materialize(self)

# Monkey patch
FactorPattern._materialize_balanced_factors = debug_materialize

# Now test
print("Testing RSA-100 with debug output")
factorizer = PatternFactorizer()

start = time.time()
factors = factorizer.factor(n)
elapsed = time.time() - start

print(f"\n[RESULT] Completed in {elapsed:.3f} seconds")
print(f"[RESULT] Factors: {factors}")

if len(factors) == 2:
    print(f"\nSUCCESS!")
    print(f"p = {factors[0]}")
    print(f"q = {factors[1]}")
    print(f"p * q = {factors[0] * factors[1]}")
    print(f"Correct: {factors[0] * factors[1] == n}")