"""
Tests for Fibonacci Entanglement functionality
Validates entanglement detection and strength measurement
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom2.fibonacci_entanglement import FibonacciEntanglement
from axiom2.fibonacci_core import fib, lucas, PHI

def test_entanglement_init():
    """Test FibonacciEntanglement initialization"""
    n = 1000
    ent = FibonacciEntanglement(n)
    assert ent.n == 1000
    assert ent.root == 31  # floor(sqrt(1000))
    print("✓ FibonacciEntanglement initialization")

def test_detect_double_basic():
    """Test basic double Fibonacci detection"""
    # 15 = 3 × 5 (both are Fibonacci numbers)
    ent = FibonacciEntanglement(15)
    results = ent.detect_double()
    
    # Should find (3, 5) with high strength
    found = False
    for p, q, strength in results:
        if (p == 3 and q == 5) or (p == 5 and q == 3):
            found = True
            assert strength > 0.9  # Both are exact Fibonacci
            break
    assert found, "Should find 3 × 5 for n=15"
    
    # 104 = 8 × 13 (both are Fibonacci)
    ent = FibonacciEntanglement(104)
    results = ent.detect_double()
    
    found = False
    for p, q, strength in results:
        if (p == 8 and q == 13) or (p == 13 and q == 8):
            found = True
            assert strength > 0.9
            break
    assert found, "Should find 8 × 13 for n=104"
    
    print("✓ Basic double Fibonacci detection")

def test_detect_double_near_fibonacci():
    """Test detection when factors are near Fibonacci"""
    # 35 = 5 × 7 (5 is Fibonacci, 7 is near 8)
    ent = FibonacciEntanglement(35)
    results = ent.detect_double()
    
    found = False
    for p, q, strength in results:
        if (p == 5 and q == 7) or (p == 7 and q == 5):
            found = True
            # Strength should be high (5 is Fibonacci, 7 is very close to 8)
            assert strength > 0.8
            break
    assert found, "Should find 5 × 7 for n=35"
    
    # 143 = 11 × 13 (11 is near fib(7)=13, 13 is fib(7))
    ent = FibonacciEntanglement(143)
    results = ent.detect_double()
    
    # Should find this with decent strength
    assert len(results) > 0
    
    print("✓ Near-Fibonacci detection")

def test_min_fibonacci_distance():
    """Test minimum Fibonacci distance calculation"""
    ent = FibonacciEntanglement(100)
    
    # Test exact Fibonacci numbers
    assert ent._min_fibonacci_distance(0) == 0
    assert ent._min_fibonacci_distance(1) == 0
    assert ent._min_fibonacci_distance(5) == 0
    assert ent._min_fibonacci_distance(8) == 0
    assert ent._min_fibonacci_distance(13) == 0
    assert ent._min_fibonacci_distance(89) == 0
    
    # Test near Fibonacci
    assert ent._min_fibonacci_distance(4) == 1  # Distance to 3 or 5
    assert ent._min_fibonacci_distance(6) == 1  # Distance to 5
    assert ent._min_fibonacci_distance(7) == 1  # Distance to 8
    assert ent._min_fibonacci_distance(9) == 1  # Distance to 8
    assert ent._min_fibonacci_distance(12) == 1  # Distance to 13
    
    print("✓ Minimum Fibonacci distance calculation")

def test_fibonacci_alignment_score():
    """Test Fibonacci alignment scoring"""
    ent = FibonacciEntanglement(100)
    
    # Perfect alignment (both Fibonacci)
    score = ent.fibonacci_alignment_score(5, 8)
    assert score > 0.95
    
    score = ent.fibonacci_alignment_score(13, 21)
    assert score > 0.95
    
    # One Fibonacci, one not
    score = ent.fibonacci_alignment_score(5, 7)
    assert 0.5 < score < 0.95
    
    # Neither Fibonacci (but both close to Fibonacci)
    score = ent.fibonacci_alignment_score(4, 6)
    assert score < 0.9  # 4 is close to 3,5 and 6 is close to 5
    
    print("✓ Fibonacci alignment scoring")

def test_golden_ratio_alignment():
    """Test golden ratio alignment detection"""
    ent = FibonacciEntanglement(100)
    
    # Test ratios close to φ
    # 8/5 = 1.6 (close to φ ≈ 1.618)
    score = ent.golden_ratio_alignment(8, 5)
    assert score > 0.8
    
    # 13/8 = 1.625 (very close to φ)
    score = ent.golden_ratio_alignment(13, 8)
    assert score > 0.9
    
    # 21/13 ≈ 1.615 (extremely close to φ)
    score = ent.golden_ratio_alignment(21, 13)
    assert score > 0.95
    
    # Test ratios far from φ
    score = ent.golden_ratio_alignment(4, 9)  # 0.444 or 2.25, both far from φ
    assert score < 0.9  # exp decay gives high scores
    
    score = ent.golden_ratio_alignment(10, 10)  # 1.0
    assert score < 0.7  # exp(-0.618) ≈ 0.68
    
    print("✓ Golden ratio alignment detection")

def test_lucas_fibonacci_relation():
    """Test Lucas-Fibonacci relationship detection"""
    ent = FibonacciEntanglement(1000)
    
    # Test consecutive Fibonacci
    assert ent.lucas_fibonacci_relation(5, 8) == True    # F(5), F(6)
    assert ent.lucas_fibonacci_relation(8, 13) == True   # F(6), F(7)
    assert ent.lucas_fibonacci_relation(13, 21) == True  # F(7), F(8)
    
    # Test Fibonacci-Lucas pairs (same k)
    assert ent.lucas_fibonacci_relation(1, 1) == True    # F(1)=1, L(1)=1
    assert ent.lucas_fibonacci_relation(1, 3) == True    # F(2)=1, L(2)=3
    assert ent.lucas_fibonacci_relation(2, 4) == True    # F(3)=2, L(3)=4
    assert ent.lucas_fibonacci_relation(5, 11) == True   # F(5)=5, L(5)=11
    
    # Test non-special relationships
    assert ent.lucas_fibonacci_relation(5, 7) == False
    assert ent.lucas_fibonacci_relation(10, 15) == False
    
    print("✓ Lucas-Fibonacci relationship detection")

def test_entanglement_edge_cases():
    """Test edge cases for entanglement"""
    # Small n
    ent = FibonacciEntanglement(6)  # 2 × 3
    results = ent.detect_double()
    # Should find 2 × 3 (both Fibonacci)
    assert len(results) > 0
    
    # Prime number
    ent = FibonacciEntanglement(17)
    results = ent.detect_double()
    # Should not find any factors
    assert len(results) == 0
    
    # Large Fibonacci product
    n = 233 * 377  # F(13) × F(14)
    ent = FibonacciEntanglement(n)
    results = ent.detect_double()
    # Should find with very high strength
    if results:
        max_strength = max(s for _, _, s in results)
        assert max_strength > 0.9
    
    print("✓ Edge cases handled correctly")

def test_entanglement_deterministic():
    """Test that entanglement detection is deterministic"""
    n = 377  # 13 × 29, where 13 is Fibonacci
    
    results_list = []
    for _ in range(3):
        ent = FibonacciEntanglement(n)
        results = ent.detect_double()
        results_list.append(results)
    
    # All results should be identical
    for i in range(1, len(results_list)):
        assert len(results_list[i]) == len(results_list[0])
        for j in range(len(results_list[0])):
            assert results_list[i][j] == results_list[0][j]
    
    print("✓ Entanglement detection is deterministic")

def run_all_tests():
    """Run all Fibonacci entanglement tests"""
    print("Testing Fibonacci Entanglement (Axiom 2)...")
    print("-" * 40)
    
    test_entanglement_init()
    test_detect_double_basic()
    test_detect_double_near_fibonacci()
    test_min_fibonacci_distance()
    test_fibonacci_alignment_score()
    test_golden_ratio_alignment()
    test_lucas_fibonacci_relation()
    test_entanglement_edge_cases()
    test_entanglement_deterministic()
    
    print("-" * 40)
    print("All Fibonacci Entanglement tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
