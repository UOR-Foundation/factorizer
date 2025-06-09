"""
Tests for Adaptive Observer functionality
Validates multi-scale observation and wavefunction collapse
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom4.adaptive_observer import (
    MultiScaleObserver,
    generate_superposition,
    collapse_wavefunction,
    create_coherence_gradient_field
)
import math

def test_multi_scale_observer_init():
    """Test MultiScaleObserver initialization"""
    n = 143  # 11 × 13
    observer = MultiScaleObserver(n)
    
    assert observer.n == 143
    assert observer.root == 11
    
    # Check scales are properly initialized
    assert "μ" in observer.scales
    assert "m" in observer.scales
    assert "M" in observer.scales
    assert "Ω" in observer.scales
    
    # All scales should be positive
    for scale in observer.scales.values():
        assert scale > 0
    
    print("✓ MultiScaleObserver initialization")

def test_coherence_at_scale():
    """Test coherence calculation at different scales"""
    n = 35  # 5 × 7
    observer = MultiScaleObserver(n)
    
    # Test at different positions
    coh_5 = observer.coherence_at_scale(5, scale=1)  # True factor
    coh_6 = observer.coherence_at_scale(6, scale=1)  # Non-factor
    
    assert 0 <= coh_5 <= 1
    assert 0 <= coh_6 <= 1
    
    # Test with larger scale
    coh_large = observer.coherence_at_scale(5, scale=3)
    assert 0 <= coh_large <= 1
    
    print("✓ Coherence at scale calculation")

def test_multi_scale_observe():
    """Test multi-scale observation"""
    n = 77  # 7 × 11
    observer = MultiScaleObserver(n)
    
    # Observe at true factor
    coh_7 = observer.observe(7)
    assert coh_7 > 0
    
    # Observe at non-factor
    coh_5 = observer.observe(5)
    assert coh_5 >= 0
    
    # Edge cases
    assert observer.observe(1) == 0  # Too small
    assert observer.observe(100) == 0  # Too large
    
    print("✓ Multi-scale observation")

def test_coherence_field():
    """Test coherence field generation"""
    n = 21  # 3 × 7
    observer = MultiScaleObserver(n)
    
    positions = [2, 3, 4, 5, 6, 7]
    field = observer.coherence_field(positions)
    
    assert isinstance(field, dict)
    assert len(field) == len(positions)
    
    # All coherences should be valid
    for pos, coh in field.items():
        assert pos in positions
        assert 0 <= coh <= 1
    
    print("✓ Coherence field generation")

def test_generate_superposition():
    """Test quantum superposition generation"""
    n = 100
    
    # Without hints
    superposition = generate_superposition(n)
    assert isinstance(superposition, list)
    assert len(superposition) > 0
    assert all(2 <= x <= 10 for x in superposition)
    
    # With hints
    hints = [5, 7, 9]
    superposition_with_hints = generate_superposition(n, hints)
    assert all(h in superposition_with_hints for h in hints if h <= 10)
    
    # Should include Fibonacci numbers
    assert 2 in superposition or 3 in superposition
    assert 5 in superposition or 8 in superposition
    
    print("✓ Quantum superposition generation")

def test_collapse_wavefunction():
    """Test wavefunction collapse"""
    n = 143  # 11 × 13
    observer = MultiScaleObserver(n)
    
    # Initial superposition
    candidates = [5, 7, 9, 11, 13, 15]
    
    # Collapse
    collapsed = collapse_wavefunction(n, candidates, observer, iterations=3)
    
    assert isinstance(collapsed, list)
    assert len(collapsed) > 0
    assert all(isinstance(item, tuple) and len(item) == 2 for item in collapsed)
    
    # Should be sorted by weight
    weights = [w for _, w in collapsed]
    assert weights == sorted(weights, reverse=True)
    
    print("✓ Wavefunction collapse")

def test_gradient_field():
    """Test coherence gradient field creation"""
    n = 55  # 5 × 11
    observer = MultiScaleObserver(n)
    
    field = create_coherence_gradient_field(n, observer, center=7, radius=3)
    
    assert isinstance(field, dict)
    assert len(field) > 0
    
    # Check positions are in expected range
    for pos in field.keys():
        assert 4 <= pos <= 10  # center ± radius
    
    print("✓ Coherence gradient field")

def test_observer_determinism():
    """Test that observer is deterministic"""
    n = 91  # 7 × 13
    observer = MultiScaleObserver(n)
    
    # Multiple observations should give same result
    results = []
    for _ in range(3):
        coh = observer.observe(7)
        results.append(coh)
    
    assert all(r == results[0] for r in results)
    
    # Superposition should be deterministic
    super1 = generate_superposition(n, hints=[7, 13])
    super2 = generate_superposition(n, hints=[7, 13])
    assert super1 == super2
    
    print("✓ Observer determinism")

def test_edge_cases():
    """Test edge cases for observer"""
    # Small number
    n_small = 6
    observer_small = MultiScaleObserver(n_small)
    assert observer_small.observe(2) >= 0
    
    # Prime number
    n_prime = 17
    observer_prime = MultiScaleObserver(n_prime)
    field = observer_prime.coherence_field([2, 3, 4])
    assert all(0 <= c <= 1 for c in field.values())
    
    # Perfect square
    n_square = 49  # 7 × 7
    observer_square = MultiScaleObserver(n_square)
    coh_7 = observer_square.observe(7)
    assert coh_7 > 0
    
    print("✓ Observer edge cases")

def test_collapse_convergence():
    """Test that wavefunction collapse converges"""
    n = 221  # 13 × 17
    observer = MultiScaleObserver(n)
    
    # Large initial superposition
    initial = list(range(2, 16))
    
    # Collapse with different iterations
    collapsed_1 = collapse_wavefunction(n, initial, observer, iterations=1)
    collapsed_5 = collapse_wavefunction(n, initial, observer, iterations=5)
    
    # More iterations should reduce candidates
    assert len(collapsed_5) <= len(collapsed_1)
    
    # Should maintain some candidates
    assert len(collapsed_5) > 0
    
    # All weights should be valid
    for _, weight in collapsed_5:
        assert weight >= 0
    
    print("✓ Wavefunction collapse convergence")

def run_all_tests():
    """Run all adaptive observer tests"""
    print("Testing Adaptive Observer (Axiom 4)...")
    print("-" * 40)
    
    test_multi_scale_observer_init()
    test_coherence_at_scale()
    test_multi_scale_observe()
    test_coherence_field()
    test_generate_superposition()
    test_collapse_wavefunction()
    test_gradient_field()
    test_observer_determinism()
    test_edge_cases()
    test_collapse_convergence()
    
    print("-" * 40)
    print("All Adaptive Observer tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
