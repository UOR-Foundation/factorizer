#!/usr/bin/env python3
"""Debug RSA-100 pattern decoding"""

from pattern_generator_imports import PatternFactorizer

n = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
actual_offset = 14387588531011868280533721986312584688823042570

factorizer = PatternFactorizer()
sig = factorizer.engine.extract_signature(n)
sqrt_n = factorizer.engine._integer_sqrt(n)

print(f"RSA-100 Pattern Decoding Debug")
print(f"n = {n}")
print(f"sqrt_n = {sqrt_n}")
print(f"Actual offset = {actual_offset}")
print(f"Relative offset = {actual_offset / sqrt_n:.10f}")

# Check harmonic encoding
print(f"\nHarmonic nodes: {sig.harmonic_nodes[:5]}")

# Method 2a: Phase difference
if len(sig.harmonic_nodes) >= 2:
    phase_diff = sig.harmonic_nodes[0] - sig.harmonic_nodes[1]
    offset_1 = int(abs(phase_diff) * sqrt_n * factorizer.engine.constants.EPSILON / 100)
    print(f"\nMethod 2a (phase diff):")
    print(f"  phase_diff = {phase_diff}")
    print(f"  offset_1 = {offset_1}")
    print(f"  ratio to actual = {offset_1 / actual_offset:.6f}")

# Method 2b: Modular DNA bits
offset_bits = 0
for i in range(min(20, len(sig.modular_dna))):
    if sig.modular_dna[i] > factorizer.engine.primes[i] // 2:
        offset_bits |= (1 << i)

offset_2 = offset_bits * int(sqrt_n ** 0.25)
print(f"\nMethod 2b (modular bits):")
print(f"  offset_bits = {offset_bits}")
print(f"  offset_2 = {offset_2}")
print(f"  ratio to actual = {offset_2 / actual_offset:.6f}")

# Check if we can find a better encoding
print(f"\nPattern analysis:")
print(f"  Modular DNA sum = {sum(sig.modular_dna[:20])}")
print(f"  QR sum = {sum(sig.quadratic_character[:20])}")

# The offset might be encoded as a fraction of sqrt(n)
offset_fraction = actual_offset / sqrt_n
print(f"\nOffset as fraction of sqrt(n): {offset_fraction:.10f}")
print(f"This is approximately 1/{1/offset_fraction:.2f}")