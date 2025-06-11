"""
Tests for Recursive Coherence functionality
Validates meta-coherence and fractal patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom5.recursive_coherence import (
    RecursiveCoherence,
    meta_coherence,
    find_coherence_attractors,
    golden_ratio_recursion,
    fractal_coherence_pattern,
    recursive_fixed_point_search
)

def test_recursive_coherence_init():
    """Test RecursiveCoherence initialization"""
    n = 143  # 11 × 13
    rec_coh = RecursiveCoherence(n)
    
    assert rec_coh.n == 143
    assert rec_coh.root == 11
    assert len(rec_coh.coherence_history) == 0
    assert len(rec_coh.fixed_points) == 0
    
    print("✓ RecursiveCoherence initialization")

def test_apply_coherence_to_field():
    """Test applying coherence to coherence field"""
    n = 77
    rec_coh = RecursiveCoherence(n)
    
    # Create initial field
    field = {
        2: 0.5,
        3: 0.6,
        4: 0.4,
        5: 0.7,
        6: 0.3,
        7: 0.8
    }
    
    # Apply coherence
    meta_field = rec_coh.apply_coherence_to_field(field)
    
    assert isinstance(meta_field, dict)
    assert len(meta_field) == len(field)
    
    # All values should be valid
    for pos, mc in meta_field.items():
        assert pos in field
        assert 0 <= mc <= 1
    
    print("✓ Apply coherence to field")

def test_recursive_iteration():
    """Test recursive coherence iteration"""
    n = 55  # 5 × 11
    rec_coh = RecursiveCoherence(n)
    
    # Initial field
    initial = {2: 0.3, 3: 0.5, 4: 0.4, 5: 0.9, 6: 0.2}
    
    # Apply recursion
    fields = rec_coh.recursive_coherence_iteration(initial, depth=3)
    
    assert isinstance(fields, list)
    assert len(fields) >= 2  # At least initial + one iteration
    assert fields[0] == initial  # First should be initial
    
    # Check history was saved
    assert len(rec_coh.coherence_history) == len(fields)
    
    print("✓ Recursive coherence iteration")

def test_find_fixed_points():
    """Test fixed point finding"""
    n = 91
    rec_coh = RecursiveCoherence(n)
    
    # Create converging field
    initial = {3: 0.5, 5: 0.8, 7: 0.9, 9: 0.2}
    
    # Apply recursion to create history
    rec_coh.recursive_coherence_iteration(initial, depth=5)
    
    # Find fixed points
    fixed = rec_coh.find_fixed_points()
    
    assert isinstance(fixed, list)
    # Positions with stable values should be fixed points
    
    print("✓ Fixed point finding")

def test_meta_coherence_func():
    """Test meta-coherence calculation"""
    n = 35  # 5 × 7
    
    # Calculate meta-coherence
    mc = meta_coherence(n, 5, 7)
    
    assert isinstance(mc, float)
    assert 0 <= mc <= 1
    
    # With pre-computed field
    field = {3: 0.4, 4: 0.5, 5: 0.8, 6: 0.6, 7: 0.9}
    mc_with_field = meta_coherence(n, 5, 7, field)
    
    assert 0 <= mc_with_field <= 1
    
    print("✓ Meta-coherence calculation")

def test_find_coherence_attractors():
    """Test coherence attractor finding"""
    n = 143
    initial_positions = [5, 7, 9, 11, 13]
    
    attractors = find_coherence_attractors(n, initial_positions, max_iterations=5)
    
    assert isinstance(attractors, list)
    
    # All attractors should be from initial positions
    for attr in attractors:
        assert attr in initial_positions
    
    print("✓ Coherence attractor finding")

def test_golden_ratio_recursion():
    """Test golden ratio recursion"""
    n = 77
    start = 5
    
    sequence = golden_ratio_recursion(n, start)
    
    assert isinstance(sequence, list)
    assert sequence[0] == start
    assert len(sequence) >= 2
    
    # All should be valid positions
    root = 8  # floor(sqrt(77))
    for pos in sequence:
        assert 2 <= pos <= root
    
    print("✓ Golden ratio recursion")

def test_fractal_coherence_pattern():
    """Test fractal coherence pattern generation"""
    n = 91
    
    pattern = fractal_coherence_pattern(n, base_size=5)
    
    assert isinstance(pattern, dict)
    assert len(pattern) > 0
    
    # All values should be valid
    root = 9
    for pos, coh in pattern.items():
        assert 2 <= pos <= root
        assert 0 <= coh <= 1
    
    print("✓ Fractal coherence pattern")

def test_recursive_fixed_point_search():
    """Test recursive fixed point search"""
    n = 55
    
    fixed_points = recursive_fixed_point_search(n, tolerance=0.05)
    
    assert isinstance(fixed_points, list)
    
    # All should be valid positions
    root = 7
    for fp in fixed_points:
        assert 2 <= fp <= root
    
    print("✓ Recursive fixed point search")

def test_recursive_determinism():
    """Test that recursive operations are deterministic"""
    n = 143
    
    # Create two instances
    rec1 = RecursiveCoherence(n)
    rec2 = RecursiveCoherence(n)
    
    # Same field
    field = {5: 0.5, 7: 0.7, 11: 0.9, 13: 0.3}
    
    # Apply same operations
    meta1 = rec1.apply_coherence_to_field(field)
    meta2 = rec2.apply_coherence_to_field(field)
    
    # Should be identical
    assert meta1 == meta2
    
    # Golden ratio recursion should be deterministic
    seq1 = golden_ratio_recursion(n, 7)
    seq2 = golden_ratio_recursion(n, 7)
    assert seq1 == seq2
    
    print("✓ Recursive determinism")

def run_all_tests():
    """Run all recursive coherence tests"""
    print("Testing Recursive Coherence (Axiom 5)...")
    print("-" * 40)
    
    test_recursive_coherence_init()
    test_apply_coherence_to_field()
    test_recursive_iteration()
    test_find_fixed_points()
    test_meta_coherence_func()
    test_find_coherence_attractors()
    test_golden_ratio_recursion()
    test_fractal_coherence_pattern()
    test_recursive_fixed_point_search()
    test_recursive_determinism()
    
    print("-" * 40)
    print("All Recursive Coherence tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
