#!/usr/bin/env python3
"""Debug materialization for RSA-100"""

from pattern_generator_imports import PatternFactorizer

n = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
actual_a = 39034959443932277476746304024103548121890218181130  # Known value

factorizer = PatternFactorizer()

# Get pattern
sig = factorizer.engine.extract_signature(n)
pattern = factorizer.engine.synthesize_pattern(sig)

print(f"Pattern type: {pattern.pattern_type}")
print(f"Number of positions: {len(pattern.factor_positions)}")

# Check what would happen in materialization
sqrt_n = factorizer.engine._integer_sqrt(n)
print(f"\nsqrt(n) = {sqrt_n}")
print(f"Actual a = {actual_a}")
print(f"Actual offset = {actual_a - sqrt_n}")

# Check prioritized positions
expected_offset_min = int(sqrt_n * 0.0001)
expected_offset_max = int(sqrt_n * 0.01)

print(f"\nExpected offset range: {expected_offset_min} to {expected_offset_max}")

prioritized = []
for pos in pattern.factor_positions[:20]:
    offset = pos - sqrt_n
    if expected_offset_min <= offset <= expected_offset_max:
        prioritized.append((abs(offset - sqrt_n * 0.0004), pos, offset))

prioritized.sort()

print(f"\nPrioritized positions:")
for i, (score, pos, offset) in enumerate(prioritized[:5]):
    print(f"{i+1}. offset={offset}, score={score}")
    
    # What would materialization do?
    error_estimate = int(offset * 0.0000367)
    center = pos - error_estimate
    radius = max(100, error_estimate // 10)
    
    print(f"   Would search: center={center - sqrt_n}, radius={radius}")
    print(f"   Actual is at: {actual_a - sqrt_n}")
    print(f"   Distance: {abs(center - actual_a)}")
    print()