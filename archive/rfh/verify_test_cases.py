"""
Verify the actual factors of test cases
"""

import math

def find_factors(n):
    """Find the factors of n"""
    # Check small primes first
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if n % p == 0:
            return p, n // p
    
    # Check up to sqrt(n)
    sqrt_n = int(math.sqrt(n))
    for i in range(101, min(1000000, sqrt_n + 1), 2):
        if n % i == 0:
            return i, n // i
    
    # Likely prime
    return 1, n

# Test cases to verify
test_numbers = [
    104729,
    282797, 
    1299827,
    16777259,
    1073676287,
    2147483713,
    536870923,
    4294967357,
    1000000007,
    1234567891,
    9999999967
]

print("Verifying test case factors:")
print("-" * 50)

for n in test_numbers:
    p, q = find_factors(n)
    if p == 1:
        print(f"{n}: PRIME")
    else:
        print(f"{n}: {p} × {q}")
        # Verify
        assert p * q == n, f"Error: {p} × {q} != {n}"
