import math

print(f'151 × 167 = {151 * 167}')

# Check if 25117 is prime
is_prime = True
for i in range(2, int(math.sqrt(25117)) + 1):
    if 25117 % i == 0:
        is_prime = False
        print(f'25117 = {i} × {25117 // i}')
        break

if is_prime:
    print('25117 is prime!')