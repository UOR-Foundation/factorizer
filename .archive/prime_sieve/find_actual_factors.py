"""
Find the actual factors of the test numbers
"""

def find_factors(n):
    """Find factors of n using trial division"""
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i, n // i
    return 1, n


# Check the test cases
test_cases = [
    (294409, 541, 544),
    (1299709, 1117, 1163)
]

for n, expected_p, expected_q in test_cases:
    actual_p, actual_q = find_factors(n)
    expected_product = expected_p * expected_q
    
    print(f"\nNumber: {n}")
    print(f"Expected: {expected_p} × {expected_q} = {expected_product}")
    print(f"Actual: {actual_p} × {actual_q} = {actual_p * actual_q}")
    print(f"Match: {n == expected_product}")
    
    if n != expected_product:
        print(f"ERROR: Expected factors don't multiply to n!")
        print(f"Difference: {n} - {expected_product} = {n - expected_product}")
