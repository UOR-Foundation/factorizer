"""
Calculate correct semiprimes for test cases
"""

test_primes = [
    # 40-bit range
    (1048573, 1048583),
    (1046527, 1046543),
    
    # 48-bit range
    (16777213, 16777259),
    (16769023, 16785431),
    
    # 64-bit range
    (4294967291, 4294967311),
    (134217689, 134217757),
    
    # 80-bit range
    (34359738337, 35184372088899),
    
    # 96-bit range
    (8916100448229, 8888777666119),
    
    # 112-bit range
    (72057594037927931, 72057594037927939),
    
    # 128-bit range
    (18446744073709551557, 18446744073709551631),
]

print("Calculating correct semiprimes:")
print("-" * 80)

for p, q in test_primes:
    n = p * q
    print(f"({n}, {p}, {q}),  # {n.bit_length()} bits - verified: {p} × {q} = {n}")
