#!/usr/bin/env python3
# Identify which test case was hanging

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quick_validation import create_challenging_semiprimes

# Get all test cases
test_cases = create_challenging_semiprimes()

# Find cases after 851 (the last successful one)
print("Test cases after 851:")
print("-" * 60)

found_851 = False
for i, (semiprime, p, q, category) in enumerate(test_cases):
    if semiprime == 851:
        found_851 = True
        print(f"[{i}] {semiprime:8} = {p:4} × {q:4} ({category}) - LAST SUCCESS")
        continue
    
    if found_851 and i < len(test_cases) - 1:
        print(f"[{i}] {semiprime:8} = {p:4} × {q:4} ({category})")
        if i - test_cases.index((851, 23, 37, "golden_ratio")) > 5:
            break

# Also check the structure of the test
print(f"\nTotal test cases: {len(test_cases)}")
print(f"\nCategory distribution:")
categories = {}
for _, _, _, cat in test_cases:
    categories[cat] = categories.get(cat, 0) + 1

for cat, count in sorted(categories.items()):
    print(f"  {cat:20}: {count:3} cases")
