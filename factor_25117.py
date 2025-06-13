def factorize(n):
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i, n // i
    return n, 1

p, q = factorize(25117)
print(f"25117 = {p} × {q}")
print(f"Verification: {p} × {q} = {p * q}")