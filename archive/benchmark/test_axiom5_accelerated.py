"""
Test the Axiom 5 Accelerated Factorizer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom5 import Axiom5AcceleratedFactorizer, get_accelerated_factorizer, accelerated_factorize

def test_accelerated_factorizer():
    """Test basic functionality of the accelerated factorizer"""
    print("Testing Axiom 5 Accelerated Factorizer...")
    print("-" * 50)
    
    # Test cases
    test_cases = [
        (15, (3, 5)),        # 3 × 5
        (35, (5, 7)),        # 5 × 7
        (77, (7, 11)),       # 7 × 11
        (143, (11, 13)),     # 11 × 13
        (323, (17, 19)),     # 17 × 19
        (437, (19, 23)),     # 19 × 23
        (1073, (29, 37)),    # 29 × 37
        (2021, (43, 47)),    # 43 × 47
    ]
    
    factorizer = get_accelerated_factorizer()
    
    for n, expected in test_cases:
        result = factorizer.factorize_with_details(n)
        
        print(f"\nFactoring {n}:")
        print(f"  Expected: {expected}")
        print(f"  Got: {result.factors}")
        print(f"  Method: {result.method}")
        print(f"  Time: {result.time:.4f}s")
        print(f"  Iterations: {result.iterations}")
        print(f"  Cache hits: {result.cache_hits}")
        print(f"  Learning applied: {result.learning_applied}")
        print(f"  Confidence: {result.confidence}")
        
        assert result.factors == expected, f"Failed for {n}: got {result.factors}, expected {expected}"
    
    # Test convenience function
    print("\n" + "-" * 50)
    print("Testing convenience function...")
    factors = accelerated_factorize(437)
    print(f"accelerated_factorize(437) = {factors}")
    assert factors == (19, 23)
    
    # Show statistics
    print("\n" + "-" * 50)
    print("Factorizer Statistics:")
    stats = factorizer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_accelerated_factorizer()
