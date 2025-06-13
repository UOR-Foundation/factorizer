import math

# Analyze why 80-bit number fails
n = 1208925819614629174706189
p = 1099511627791
q = 1099511627773

print(f"n = {n}")
print(f"p = {p}, q = {q}")
print(f"Verification: {p} × {q} = {p * q}")
print(f"Correct: {p * q == n}")

# Phi calculations
phi = 1.618033988749895
n_phi = math.log(n) / math.log(phi)
target_phi = n_phi / 2.0
center_estimate = phi ** target_phi

print(f"\nn_φ = {n_phi:.6f}")
print(f"Target φ (n_φ/2) = {target_phi:.6f}")
print(f"φ^(n_φ/2) = {center_estimate:.2f}")
print(f"sqrt(n) = {math.sqrt(n):.2f}")

# Check distance from center
center = int(center_estimate)
print(f"\nSearch center: {center}")
print(f"Distance from p: {abs(p - center)}")
print(f"Distance from q: {abs(q - center)}")

# Current search radius calculation
bit_length = 81
if bit_length > 60:
    search_radius = 1_000_000
elif bit_length > 40:
    search_radius = 100_000
elif bit_length > 30:
    search_radius = 10_000
else:
    search_radius = max(int(n ** 0.3), 100)

print(f"\nCurrent search radius: {search_radius:,}")
print(f"Minimum radius needed: {max(abs(p - center), abs(q - center)):,}")

# The factors are extremely close for this balanced semiprime
print(f"\np - q = {p - q}")
print(f"Factor balance: p/q = {p/q:.10f}")

# For very balanced semiprimes, we should search very close to sqrt(n)
sqrt_n = int(math.sqrt(n))
print(f"\nsqrt(n) = {sqrt_n}")
print(f"Distance from p to sqrt(n): {abs(p - sqrt_n)}")
print(f"Distance from q to sqrt(n): {abs(q - sqrt_n)}")