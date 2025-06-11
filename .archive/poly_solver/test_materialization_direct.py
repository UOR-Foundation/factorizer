#!/usr/bin/env python3
"""Direct test of materialization with correct radius"""

def integer_sqrt(n):
    if n < 2:
        return n
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

n = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
sqrt_n = integer_sqrt(n)

# The pattern position we identified
pattern_offset = 14388116465855924756984835811968781080802099200
pattern_a = sqrt_n + pattern_offset

# Error correction
error_estimate = int(pattern_offset * 0.0000367)
center = pattern_a - error_estimate
radius = error_estimate  # Use FULL error estimate

print(f"Testing materialization directly:")
print(f"Pattern offset: {pattern_offset}")
print(f"Error estimate: {error_estimate}")
print(f"Center: {center}")
print(f"Radius: {radius}")
print(f"Search range: [{center - radius}, {center + radius}]")

# The actual 'a' we're looking for
actual_a = 39034959443932277476746304024103548121890218181130
print(f"\nActual a: {actual_a}")
print(f"Is in range? {center - radius <= actual_a <= center + radius}")

if center - radius <= actual_a <= center + radius:
    print("\n✓ The actual 'a' IS in our search range!")
    print("Binary search would find it.")
    
    # Let's verify
    b_squared = actual_a * actual_a - n
    b = integer_sqrt(b_squared)
    if b * b == b_squared:
        p = actual_a + b
        q = actual_a - b
        print(f"\nFactors:")
        print(f"p = {p}")
        print(f"q = {q}")
        print(f"p × q = {p * q}")
        print(f"Equals n: {p * q == n}")
else:
    print("\n✗ The actual 'a' is NOT in our search range")
    print(f"Distance: {min(abs(actual_a - (center - radius)), abs(actual_a - (center + radius)))}")